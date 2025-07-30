"""
Evaluator for FUSED LINEAR JSD kernel optimization with enhanced memory profiling and agent feedback
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
import gc
from dotenv import load_dotenv
from liger_kernel.ops.fused_linear_jsd import LigerFusedLinearJSDFunction as ReferenceFusedLinearJSD
import concurrent.futures
import threading
from torch.profiler import profile, record_function, ProfilerActivity

# Load environment variables from .env file
load_dotenv()



class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    """Handle timeout signal"""
    raise TimeoutError("Function execution timed out")


class MemoryProfiler:
    """Track memory usage during execution"""
    
    def __init__(self):
        self.start_memory = None
        self.peak_memory = None
        self.end_memory = None
        
    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        self.start_memory = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.synchronize()
        self.peak_memory = torch.cuda.max_memory_allocated()
        self.end_memory = torch.cuda.memory_allocated()
        
    def get_memory_stats(self):
        """Return memory statistics in MB"""
        return {
            'start_mb': self.start_memory / 1e6 if self.start_memory else 0,
            'peak_mb': self.peak_memory / 1e6 if self.peak_memory else 0,
            'end_mb': self.end_memory / 1e6 if self.end_memory else 0,
            'allocated_mb': (self.peak_memory - self.start_memory) / 1e6 if all([self.peak_memory, self.start_memory]) else 0
        }

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


def profile_detailed_performance(func, *args, **kwargs):
    """Run function with detailed PyTorch profiler"""
    try:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            with record_function("kernel_execution"):
                result = func(*args, **kwargs)
        
        # Extract key statistics
        events = prof.key_averages()
        gpu_time = sum([evt.cuda_time_total for evt in events]) / 1000  # Convert to ms
        cpu_time = sum([evt.cpu_time_total for evt in events]) / 1000   # Convert to ms
        memory_usage = sum([evt.cuda_memory_usage for evt in events if evt.cuda_memory_usage > 0])
        
        bottlenecks = []
        total_cuda_time = sum([e.cuda_time_total for e in events])
        for evt in sorted(events, key=lambda x: x.cuda_time_total, reverse=True)[:5]:
            if evt.cuda_time_total > 0:
                bottlenecks.append({
                    'operation': evt.key,
                    'time_ms': evt.cuda_time_total / 1000,
                    'percentage': (evt.cuda_time_total / total_cuda_time) * 100 if total_cuda_time > 0 else 0
                })
        
        return result, {
            'gpu_time_ms': gpu_time,
            'cpu_time_ms': cpu_time,
            'memory_usage_bytes': memory_usage,
            'bottlenecks': bottlenecks,
            'profiler_table': prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)
        }
    except Exception as e:
        # Fallback if profiling fails
        result = func(*args, **kwargs)
        return result, {'error': f'Profiling failed: {str(e)}'}


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

def _run_enhanced_latency_check(program, test_sizes: list[tuple[int, int, int, int]]) -> dict:
    """Enhanced latency check with memory profiling and bottleneck analysis"""
    results = {
        'per_size_results': [],
        'avg_latency_ms': 0.0,
        'avg_memory_mb': 0.0,
        'detailed_feedback': [],
        'bottleneck_analysis': []
    }
    
    scalar = 1.0
    temperature = 1.0
    beta = 0.5
    num_warmup = 3
    num_measure = 3
    
    total_times = []
    total_memory = []
    
    for i, (batch_size, seq_len, hidden_size, vocab_size) in enumerate(test_sizes):
        BT = batch_size * seq_len
        
        # Create tensors
        a = torch.rand(BT, hidden_size // 2, device='cuda', dtype=torch.float16) * scalar
        b = torch.rand(vocab_size, hidden_size // 2, device='cuda', dtype=torch.float16) * scalar
        t = torch.rand(BT, hidden_size, device='cuda', dtype=torch.float16) * scalar
        w = torch.rand(vocab_size, hidden_size, device='cuda', dtype=torch.float16) * scalar
        shift_labels = torch.randint(0, vocab_size, (BT,), device='cuda', dtype=torch.long)
        
        a.requires_grad_(True)
        b.requires_grad_(True)
        t.requires_grad_(True)
        w.requires_grad_(True)
        
        # Warmup
        for _ in range(num_warmup):
            loss = program.LigerFusedLinearJSDFunction.apply(a, b, t, w, shift_labels, beta, -100, temperature)
            loss.backward()
            torch.cuda.synchronize()
            for grad in [a.grad, b.grad, t.grad, w.grad]:
                if grad is not None:
                    grad.zero_()
        
        # Measured runs with memory profiling
        run_times = []
        memory_stats = []
        bottlenecks = []
        
        for run in range(num_measure):
            with MemoryProfiler() as mem_prof:
                # Forward pass
                start_time = time.time()
                if run == 0:  # Only profile first run to avoid overhead
                    loss, profile_stats = profile_detailed_performance(
                        program.LigerFusedLinearJSDFunction.apply,
                        a, b, t, w, shift_labels, beta, -100, temperature
                    )
                    if 'bottlenecks' in profile_stats:
                        bottlenecks.append(profile_stats)
                else:
                    loss = program.LigerFusedLinearJSDFunction.apply(a, b, t, w, shift_labels, beta, -100, temperature)
                forward_time = (time.time() - start_time) * 1000
                
                # Backward pass
                start_time = time.time()
                loss.backward()
                backward_time = (time.time() - start_time) * 1000
                torch.cuda.synchronize()
                
            run_times.append(forward_time + backward_time)
            memory_stats.append(mem_prof.get_memory_stats())
            
            # Clear gradients
            for grad in [a.grad, b.grad, t.grad, w.grad]:
                if grad is not None:
                    grad.zero_()
        
        avg_time = np.mean(run_times)
        avg_memory = np.mean([m['allocated_mb'] for m in memory_stats])
        
        total_times.append(avg_time)
        total_memory.append(avg_memory)
        
        size_result = {
            'config': f"B{batch_size}_S{seq_len}_H{hidden_size}_V{vocab_size}",
            'batch_size': batch_size,
            'seq_len': seq_len,
            'hidden_size': hidden_size,
            'vocab_size': vocab_size,
            'avg_time_ms': avg_time,
            'memory_mb': avg_memory,
            'memory_efficiency': avg_memory / (batch_size * seq_len * hidden_size / 1e6),  # MB per million params
            'bottlenecks': bottlenecks[0] if bottlenecks else None
        }
        results['per_size_results'].append(size_result)
        results['bottleneck_analysis'].extend(bottlenecks)
        
        print(f"Config {i+1}: {size_result['config']} - Time: {avg_time:.2f}ms, Memory: {avg_memory:.1f}MB")
    
    results['avg_latency_ms'] = np.mean(total_times)
    results['avg_memory_mb'] = np.mean(total_memory)
    
    # Generate detailed feedback for the agent
    feedback = []
    if results['avg_memory_mb'] > 1000:  # > 1GB
        feedback.append("HIGH_MEMORY_USAGE: Consider memory optimization techniques like gradient checkpointing")
    
    if results['bottleneck_analysis']:
        for bottleneck_data in results['bottleneck_analysis']:
            if 'bottlenecks' in bottleneck_data and bottleneck_data['bottlenecks']:
                top_bottleneck = bottleneck_data['bottlenecks'][0]
                if top_bottleneck['percentage'] > 50:
                    feedback.append(f"BOTTLENECK_IDENTIFIED: {top_bottleneck['operation']} takes {top_bottleneck['percentage']:.1f}% of execution time")
                break
    
    # Memory scaling analysis
    if len(results['per_size_results']) > 1:
        memory_growth = []
        for i in range(1, len(results['per_size_results'])):
            prev = results['per_size_results'][i-1]
            curr = results['per_size_results'][i]
            growth_ratio = curr['memory_mb'] / prev['memory_mb'] if prev['memory_mb'] > 0 else 1
            memory_growth.append(growth_ratio)
        
        avg_growth = np.mean(memory_growth)
        if avg_growth > 2.0:
            feedback.append(f"MEMORY_SCALING_ISSUE: Memory grows {avg_growth:.1f}x on average with size increases")
    
    results['detailed_feedback'] = feedback
    return results

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

def _calc_scores(avg_latency_ms: float, ref_latency_ms: float, max_error: float, memory_efficiency: float = 1.0) -> tuple[float, float, float, float]:
    """
    Calculate scores based on latency, memory, and correctness
    
    Args:
        avg_latency_ms: Average latency of the optimized implementation
        ref_latency_ms: Average latency of the reference implementation
        max_error: Maximum error in the implementation
        memory_efficiency: Memory efficiency ratio (ref_memory / opt_memory)
    """
    # Calculate speedup relative to reference
    speedup = ref_latency_ms / avg_latency_ms
    
    # Latency score based on speedup
    # Score is 1.0 for 2x speedup, 0.5 for 1x speedup, and approaches 0 for slower than reference
    latency_score = 1.0 / (1.0 + np.exp(-(speedup - 1.0)))

    # Memory score based on efficiency (higher is better)
    memory_score = 1.0 / (1.0 + np.exp(-(memory_efficiency - 1.0)))

    # Correctness score based on error
    correctness_score = 1.0 / (1.0 + max_error)

    # Combined score weights correctness, performance, and memory
    combined_score = 0.3 * correctness_score + 0.5 * latency_score + 0.2 * memory_score
    return latency_score, memory_score, correctness_score, combined_score


def evaluate(program_path):
    """
    Evaluate the program by running it with different tensor sizes
    and measuring performance metrics with enhanced memory analysis.

    Args:
        program_path: Path to the program file

    Returns:
        Dictionary of metrics with enhanced feedback
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
                "memory_score": 0.0,
                "correctness_score": 0.0,
                "combined_score": 0.0,
                "error": "Missing required functions",
                "agent_feedback": ["MISSING_FUNCTION: LigerFusedLinearJSDFunction not found"]
            }

        # Get reference implementation latency (simple version for comparison)
        ref_latency = _run_reference_latency_check(test_sizes)
        print(f"\nReference implementation average latency: {ref_latency:.2f}ms")

        # Get enhanced optimized implementation analysis
        opt_results = _run_enhanced_latency_check(program, test_sizes)
        print(f"Optimized implementation average latency: {opt_results['avg_latency_ms']:.2f}ms", flush=True)
        print(f"Average memory usage: {opt_results['avg_memory_mb']:.1f}MB")
        print(f"Speedup: {ref_latency/opt_results['avg_latency_ms']:.2f}x\n")

        # Run correctness check
        is_correct, max_error = _run_correctness_check(program, test_sizes)
        
        # Calculate memory efficiency (assume reference uses similar memory for now)
        memory_efficiency = 1.0  # Could be enhanced with reference memory measurement
        
        latency_score, memory_score, correctness_score, combined_score = _calc_scores(
            opt_results['avg_latency_ms'], ref_latency, max_error, memory_efficiency
        )

        # Generate comprehensive agent feedback
        agent_feedback = opt_results['detailed_feedback'].copy()
        speedup = ref_latency / opt_results['avg_latency_ms']
        agent_feedback.append(f"SPEEDUP: {speedup:.2f}x ({'faster' if speedup > 1 else 'slower'} than reference)")
        agent_feedback.append(f"AVERAGE_MEMORY_USAGE: {opt_results['avg_memory_mb']:.1f}MB")
        
        if opt_results['bottleneck_analysis']:
            agent_feedback.append("PROFILER_DATA_AVAILABLE: Check detailed results for timing breakdown")

        if not is_correct:
            return {
                "latency_score": 0.0,
                "memory_score": 0.0,
                "correctness_score": 0.0,
                "combined_score": 0.0,
                "error": "Correctness check failed",
                "detailed_results": opt_results,
                "agent_feedback": agent_feedback + [f"CORRECTNESS_ERROR: Max error {max_error:.6f}"]
            }
        else:
            return {
                "latency_score": latency_score,
                "memory_score": memory_score,
                "correctness_score": correctness_score,
                "combined_score": combined_score,
                "speedup": speedup,
                "memory_usage_mb": opt_results['avg_memory_mb'],
                "detailed_results": opt_results,
                "agent_feedback": agent_feedback
            }

    except Exception as e:
        print(f"Evaluation failed completely: {str(e)}")
        traceback.print_exc()
        return {
            "latency_score": 0.0,
            "memory_score": 0.0,
            "correctness_score": 0.0,
            "combined_score": 0.0,
            "error": str(e),
            "agent_feedback": [f"EVALUATION_ERROR: {str(e)}"]
        }


def evaluate_stage1(program_path):
    """
    First stage evaluation - quick validation check with enhanced feedback
    """
    # Use a single size for quick validation
    test_sizes = [(32, 512, 4096, 32000),    # Typical transformer size
        (64, 1024, 4096, 32000),   # Larger batch
        (16, 2048, 4096, 32000),   # Longer sequence
        (32, 512, 8192, 64000)]  # Typical transformer size

    try:
        # Load the program
        program = _load_program(program_path)

        # Check if required functions exist
        if not _check_program_required_functions(program):
            return {
                "throughput_score": 0.0,
                "latency_score": 0.0,
                "memory_score": 0.0,
                "correctness_score": 0.0,
                "combined_score": 0.0,
                "error": "Missing required functions",
                "agent_feedback": ["MISSING_FUNCTION: LigerFusedLinearJSDFunction not found"]
            }

        ref_latency = _run_reference_latency_check(test_sizes)
        print(f"\nReference implementation average latency: {ref_latency:.2f}ms")

        # Use enhanced latency check for better feedback
        opt_results = _run_enhanced_latency_check(program, test_sizes)
        print(f"Optimized implementation average latency: {opt_results['avg_latency_ms']:.2f}ms", flush=True)
        print(f"Average memory usage: {opt_results['avg_memory_mb']:.1f}MB")

        is_correct, max_error = _run_correctness_check(program, test_sizes)
        
        memory_efficiency = 1.0  # Could be enhanced with reference memory measurement
        latency_score, memory_score, correctness_score, combined_score = _calc_scores(
            opt_results['avg_latency_ms'], ref_latency, max_error, memory_efficiency
        )

        # Generate agent feedback
        agent_feedback = opt_results['detailed_feedback'].copy()
        speedup = ref_latency / opt_results['avg_latency_ms']
        agent_feedback.append(f"STAGE1_SPEEDUP: {speedup:.2f}x")
        agent_feedback.append(f"STAGE1_MEMORY_USAGE: {opt_results['avg_memory_mb']:.1f}MB")

        if not is_correct:
            return {
                "latency_score": 0.0,
                "memory_score": 0.0,
                "correctness_score": 0.0,
                "combined_score": 0.0,
                "error": "Correctness check failed",
                "detailed_results": opt_results,
                "agent_feedback": agent_feedback + [f"CORRECTNESS_ERROR: Max error {max_error:.6f}"]
            }
        else:
            return {
                "latency_score": latency_score,
                "memory_score": memory_score,
                "correctness_score": correctness_score,
                "combined_score": combined_score,
                "speedup": speedup,
                "memory_usage_mb": opt_results['avg_memory_mb'],
                "detailed_results": opt_results,
                "agent_feedback": agent_feedback
            }

    except Exception as e:
        print(f"Stage 1 evaluation failed completely: {str(e)}")
        traceback.print_exc()
        return {
            "latency_score": 0.0,
            "memory_score": 0.0,
            "correctness_score": 0.0,
            "combined_score": 0.0,
            "error": str(e),
            "agent_feedback": [f"STAGE1_ERROR: {str(e)}"]
        }


def evaluate_stage2(program_path):
    """
    Second stage evaluation - full evaluation with all sizes
    """
    return evaluate(program_path) 


if __name__ == "__main__":
    print(evaluate_stage1("initial_program.py"))