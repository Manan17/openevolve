"""
Evaluator for GEGLU kernel optimization
"""

import importlib.util
import numpy as np
import time
import os
import signal
import subprocess
import tempfile
import traceback
import sys
import pickle
import torch
from dotenv import load_dotenv
from liger_kernel.ops.fused_linear_jsd import LigerFusedLinearJSDFunction as ReferenceFusedLinearJSD
import concurrent.futures
import threading

# Load environment variables from .env file
load_dotenv()

# # Verify OpenAI API key is set
if not os.getenv('OPENAI_API_KEY'):
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in .env file.")

class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    """Handle timeout signal"""
    raise TimeoutError("Function execution timed out")


def verify_correctness(a, b, c_optimized=None, rtol=1e-3, atol=1e-3):
    """
    Verify that the optimized GEGLU implementation produces correct results
    by comparing with the reference implementation.
    
    Args:
        a: Input tensor a
        b: Input tensor b
        c_optimized: Output from optimized implementation (only needed for forward pass)
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        Tuple of (is_correct, max_error)
    """
    # Run reference implementation
    a.requires_grad_(True)
    b.requires_grad_(True)
    c_reference = ReferenceFusedLinearJSD.apply(a, b)

    c_reference.backward(torch.ones_like(c_reference))
    a_grad_ref = a.grad.clone()
    b_grad_ref = b.grad.clone()

    with torch.no_grad():
        if is_forward:
            # For forward pass, we compare the output tensors
            if c_optimized is None:
                raise ValueError("c_optimized must be provided for forward pass verification")
            print("Shape of c_optimized: ", c_optimized.shape)
            print("Shape of a: ", a.shape)
            print("Shape of b: ", b.shape)
            c_reference = ReferenceFusedLinearJSD.apply(a, b)
            print("Shape of c_reference: ", c_reference.shape)
            is_correct = torch.allclose(c_optimized, c_reference, rtol=rtol, atol=atol)
            max_error = torch.max(torch.abs(c_optimized - c_reference)).item()
        else:
            # For backward pass, we need to compute gradients
            a.requires_grad_(True)
            b.requires_grad_(True)
            c_reference = ReferenceFusedLinearJSD.apply(a, b)
            c_reference.backward(torch.ones_like(c_reference))
            a_grad_ref = a.grad.clone()
            b_grad_ref = b.grad.clone()
            
            # Compare gradients
            is_correct_a = torch.allclose(a, a_grad_ref, rtol=rtol, atol=atol)
            is_correct_b = torch.allclose(b, b_grad_ref, rtol=rtol, atol=atol)
            is_correct = is_correct_a and is_correct_b
            
            # Calculate max error across both gradients
            max_error_a = torch.max(torch.abs(a - a_grad_ref)).item()
            max_error_b = torch.max(torch.abs(b - b_grad_ref)).item()
            max_error = max(max_error_a, max_error_b)
    
    return is_correct, max_error


def run_with_timeout(func, args=(), kwargs={}, timeout_seconds=20):
    """
    Run a function with a timeout using concurrent.futures

    Args:
        func: Function to run
        args: Arguments to pass to the function
        kwargs: Keyword arguments to pass to the function
        timeout_seconds: Timeout in seconds

    Returns:
        Result of the function or raises TimeoutError
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Function {func.__name__} timed out after {timeout_seconds} seconds")


def _load_program(program_path):
    spec = importlib.util.spec_from_file_location("program", program_path)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)
    return program


def _check_program_required_functions(program):
    return hasattr(program, "LigerFusedLinearJSDFunction")


def _run_reference_latency_check(test_sizes: list[tuple[int, int, int, int]]) -> float:
    """Run latency check using the reference implementation"""
    total_times = []
    scalar = 1.0
    temperature = 1.0
    beta = 0.5
    num_warmup = 3
    num_measure = 3
    for batch_size, seq_len, hidden_size, vocab_size in test_sizes:
        BT = batch_size * seq_len
        # Use torch.rand and scale, match test_correctness
        a = torch.rand(BT, hidden_size // 2, device='cuda', dtype=torch.float16) * scalar
        b = torch.rand(vocab_size, hidden_size // 2, device='cuda', dtype=torch.float16) * scalar
        t = torch.rand(BT, hidden_size, device='cuda', dtype=torch.float16) * scalar
        w = torch.rand(vocab_size, hidden_size, device='cuda', dtype=torch.float16) * scalar
        shift_labels = torch.randint(0, vocab_size, (BT,), device='cuda', dtype=torch.long)
        a.requires_grad_(True)
        b.requires_grad_(True)
        t.requires_grad_(True)
        w.requires_grad_(True)

        # Warmup runs
        for _ in range(num_warmup):
            loss = ReferenceFusedLinearJSD.apply(a, b, t, w, shift_labels, beta, -100, temperature)
            loss.backward()
            torch.cuda.synchronize()
            for grad in [a.grad, b.grad, t.grad, w.grad]:
                if grad is not None:
                    grad.zero_()

        # Measured runs
        run_times = []
        for _ in range(num_measure):
            start_time = time.time()
            loss = ReferenceFusedLinearJSD.apply(a, b, t, w, shift_labels, beta, -100, temperature)
            forward_time = (time.time() - start_time) * 1000  # ms
            start_time = time.time()
            loss.backward()
            backward_time = (time.time() - start_time) * 1000  # ms
            torch.cuda.synchronize()
            run_times.append(forward_time + backward_time)
            for grad in [a.grad, b.grad, t.grad, w.grad]:
                if grad is not None:
                    grad.zero_()
        avg_time = np.mean(run_times)
        total_times.append(avg_time)
        print(f"Reference - Batch: {batch_size}, Seq: {seq_len}, Hidden: {hidden_size}, Vocab: {vocab_size}, Avg Fwd+Bwd: {avg_time:.2f}ms")
    return np.mean(total_times)

def _run_latency_check(program, test_sizes: list[tuple[int, int, int, int]]) -> float:
    total_times = []
    scalar = 1.0
    temperature = 1.0
    beta = 0.5
    num_warmup = 3
    num_measure = 3
    for batch_size, seq_len, hidden_size, vocab_size in test_sizes:
        BT = batch_size * seq_len
        a = torch.rand(BT, hidden_size // 2, device='cuda', dtype=torch.float16) * scalar
        b = torch.rand(vocab_size, hidden_size // 2, device='cuda', dtype=torch.float16) * scalar
        t = torch.rand(BT, hidden_size, device='cuda', dtype=torch.float16) * scalar
        w = torch.rand(vocab_size, hidden_size, device='cuda', dtype=torch.float16) * scalar
        shift_labels = torch.randint(0, vocab_size, (BT,), device='cuda', dtype=torch.long)
        a.requires_grad_(True)
        b.requires_grad_(True)
        t.requires_grad_(True)
        w.requires_grad_(True)

        # Warmup runs
        for _ in range(num_warmup):
            loss = program.LigerFusedLinearJSDFunction.apply(a, b, t, w, shift_labels, beta, -100, temperature)
            loss.backward()
            torch.cuda.synchronize()
            for grad in [a.grad, b.grad, t.grad, w.grad]:
                if grad is not None:
                    grad.zero_()

        # Measured runs
        run_times = []
        for _ in range(num_measure):
            start_time = time.time()
            loss = program.LigerFusedLinearJSDFunction.apply(a, b, t, w, shift_labels, beta, -100, temperature)
            forward_time = (time.time() - start_time) * 1000  # ms
            start_time = time.time()
            loss.backward()
            backward_time = (time.time() - start_time) * 1000  # ms
            torch.cuda.synchronize()
            run_times.append(forward_time + backward_time)
            for grad in [a.grad, b.grad, t.grad, w.grad]:
                if grad is not None:
                    grad.zero_()
        avg_time = np.mean(run_times)
        total_times.append(avg_time)
        print(f"Batch: {batch_size}, Seq: {seq_len}, Hidden: {hidden_size}, Vocab: {vocab_size}, Avg Fwd+Bwd: {avg_time:.2f}ms")
    return np.mean(total_times)

def _run_correctness_check(program, test_sizes: list[tuple[int, int, int, int]], atol=1e-3, rtol=1e-3) -> tuple[bool, float]:
    max_error = 0.0
    scalar = 1.0
    temperature = 1.0
    beta = 0.5
    for batch_size, seq_len, hidden_size, vocab_size in test_sizes:
        BT = batch_size * seq_len
        a = torch.rand(BT, hidden_size // 2, device='cuda', dtype=torch.float16) * scalar
        b = torch.rand(vocab_size, hidden_size // 2, device='cuda', dtype=torch.float16) * scalar
        t = torch.rand(BT, hidden_size, device='cuda', dtype=torch.float16) * scalar
        w = torch.rand(vocab_size, hidden_size, device='cuda', dtype=torch.float16) * scalar
        shift_labels = torch.randint(0, vocab_size, (BT,), device='cuda', dtype=torch.long)

        # Reference
        a_ref = a.clone().requires_grad_(True)
        b_ref = b.clone().requires_grad_(True)
        t_ref = t.clone().requires_grad_(True)
        w_ref = w.clone().requires_grad_(True)
        shift_labels_ref = shift_labels.clone()
        loss_ref = ReferenceFusedLinearJSD.apply(a_ref, b_ref, t_ref, w_ref, shift_labels_ref, beta, -100, temperature)
        loss_ref.backward()

        # Optimized
        a_test = a.clone().requires_grad_(True)
        b_test = b.clone().requires_grad_(True)
        t_test = t.clone().requires_grad_(True)
        w_test = w.clone().requires_grad_(True)
        shift_labels_test = shift_labels.clone()
        loss_opt = program.LigerFusedLinearJSDFunction.apply(a_test, b_test, t_test, w_test, shift_labels_test, beta, -100, temperature)
        loss_opt.backward()

        try:
            grad_a_error = torch.max(torch.abs(a_ref.grad - a_test.grad)).item()
            grad_b_error = torch.max(torch.abs(b_ref.grad - b_test.grad)).item()
            output_error = torch.max(torch.abs(loss_ref - loss_opt)).item()
            max_error = max(max_error, grad_a_error, grad_b_error, output_error)
            assert grad_a_error <= atol + rtol * torch.max(torch.abs(a_ref.grad)).item()
            assert grad_b_error <= atol + rtol * torch.max(torch.abs(b_ref.grad)).item()
            assert output_error <= atol + rtol * torch.max(torch.abs(loss_ref)).item()
        except AssertionError:
            return False, max_error
    return True, max_error

def _calc_scores(avg_latency_ms: float, ref_latency_ms: float, max_error: float) -> tuple[float, float, float]:
    """
    Calculate scores based on latency and correctness
    
    Args:
        avg_latency_ms: Average latency of the optimized implementation
        ref_latency_ms: Average latency of the reference implementation
        max_error: Maximum error in the implementation
    """
    # Calculate speedup relative to reference
    speedup = ref_latency_ms / avg_latency_ms
    
    # Latency score based on speedup
    # Score is 1.0 for 2x speedup, 0.5 for 1x speedup, and approaches 0 for slower than reference
    latency_score = 1.0 / (1.0 + np.exp(-(speedup - 1.0)))

    # Correctness score based on error
    correctness_score = 1.0 / (1.0 + max_error)

    # Combined score weights correctness and performance
    combined_score = 0.3 * correctness_score + 0.7 * latency_score
    return latency_score, correctness_score, combined_score


def evaluate(program_path):
    """
    Evaluate the program by running it with different tensor sizes
    and measuring performance metrics.

    Args:
        program_path: Path to the program file

    Returns:
        Dictionary of metrics
    """
    # Run the GEGLU function with different sizes
    test_sizes = [
        (32, 512, 4096, 32000),    # Typical transformer size
        (64, 1024, 4096, 32000),   # Larger batch
        (16, 2048, 4096, 32000),   # Longer sequence
        (32, 512, 8192, 64000),    # Larger hidden size
    ]

    try:
        # Load the program
        program = _load_program(program_path)

        # Check if required functions exist
        if not _check_program_required_functions(program):
            return {
                "throughput_score": 0.0,
                "latency_score": 0.0,
                "correctness_score": 0.0,
                "combined_score": 0.0,
                "error": "Missing required functions"
            }

        # Get reference implementation latency
        ref_latency = _run_reference_latency_check(test_sizes)
        print(f"\nReference implementation average latency: {ref_latency:.2f}ms")

        # Get optimized implementation latency
        avg_latency = _run_latency_check(program, test_sizes)
        print(f"Optimized implementation average latency: {avg_latency:.2f}ms", flush=True)
        print(f"Speedup: {ref_latency/avg_latency:.2f}x\n")

        is_correct, max_error = _run_correctness_check(program, test_sizes)
        latency_score, correctness_score, combined_score = _calc_scores(avg_latency, ref_latency, max_error)

        if not is_correct:
            return {
                "latency_score": 0.0,
                "correctness_score": 0.0,
                "combined_score": 0.0,
                "error": "Correctness check failed"
            }
        else:
            return {
                "latency_score": latency_score,
                "correctness_score": correctness_score,
                "combined_score": combined_score,
            }

    except Exception as e:
        print(f"Evaluation failed completely: {str(e)}")
        traceback.print_exc()
        return {
            "latency_score": 0.0,
            "correctness_score": 0.0,
            "combined_score": 0.0,
            "error": str(e)
        }


def evaluate_stage1(program_path):
    """
    First stage evaluation - quick validation check with a single size
    """
    # Use a single size for quick validation
    test_sizes = [(32, 512, 4096, 32000)]  # Typical transformer size

    try:
        # Load the program
        program = _load_program(program_path)

        # Check if required functions exist
        if not _check_program_required_functions(program):
            return {
                "throughput_score": 0.0,
                "latency_score": 0.0,
                "correctness_score": 0.0,
                "combined_score": 0.0,
                "error": "Missing required functions"
            }

        ref_latency = _run_reference_latency_check(test_sizes)
        print(f"\nReference implementation average latency: {ref_latency:.2f}ms")

        avg_latency = _run_latency_check(program, test_sizes)
        print(f"Optimized implementation average latency: {avg_latency:.2f}ms", flush=True)
        latency_score = 1.0 / (1.0 + avg_latency)

        is_correct, max_error = _run_correctness_check(program, test_sizes)
        latency_score, correctness_score, combined_score = _calc_scores(avg_latency, ref_latency, max_error)

        if not is_correct:
            return {
                "latency_score": 0.0,
                "correctness_score": 0.0,
                "combined_score": 0.0,
                "error": "Correctness check failed"
            }
        else:
            return {
                "latency_score": latency_score,
                "correctness_score": correctness_score,
                "combined_score": combined_score,
            }

    except Exception as e:
        print(f"Stage 1 evaluation failed completely: {str(e)}")
        traceback.print_exc()
        return {
            "latency_score": 0.0,
            "correctness_score": 0.0,
            "combined_score": 0.0,
            "error": str(e)
        }


def evaluate_stage2(program_path):
    """
    Second stage evaluation - full evaluation with all sizes
    """
    return evaluate(program_path) 


if __name__ == "__main__":
    print(evaluate_stage1("initial_program.py"))