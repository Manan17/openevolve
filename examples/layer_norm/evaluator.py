"""
Evaluator for LayerNorm kernel optimization
"""

import importlib.util
import numpy as np
import time
import os
import traceback
import torch
from liger_kernel.ops.layer_norm import LigerLayerNormFunction as ReferenceLayerNorm

def _load_program(program_path):
    spec = importlib.util.spec_from_file_location("program", program_path)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)
    return program

def _check_program_required_functions(program):
    return hasattr(program, "LigerLayerNormFunction")

def _run_reference_latency_check(test_sizes):
    total_times = []
    num_warmup = 3
    num_measure = 3
    for batch_size, seq_len, hidden_size in test_sizes:
        x = torch.rand(batch_size, seq_len, hidden_size, device='cuda', dtype=torch.float16)
        w = torch.rand(hidden_size, device='cuda', dtype=torch.float16)
        b = torch.rand(hidden_size, device='cuda', dtype=torch.float16)
        eps = 1e-5
        # Warmup
        for _ in range(num_warmup):
            y = ReferenceLayerNorm.apply(x, w, b, eps)
            torch.cuda.synchronize()
        # Measure
        run_times = []
        for _ in range(num_measure):
            start_time = time.time()
            y = ReferenceLayerNorm.apply(x, w, b, eps)
            torch.cuda.synchronize()
            run_times.append((time.time() - start_time) * 1000)
        avg_time = np.mean(run_times)
        total_times.append(avg_time)
        print(f"Reference - Batch: {batch_size}, Seq: {seq_len}, Hidden: {hidden_size}, Avg Fwd: {avg_time:.2f}ms")
    return np.mean(total_times)

def _run_latency_check(program, test_sizes):
    total_times = []
    num_warmup = 3
    num_measure = 3
    for batch_size, seq_len, hidden_size in test_sizes:
        x = torch.rand(batch_size, seq_len, hidden_size, device='cuda', dtype=torch.float16)
        w = torch.rand(hidden_size, device='cuda', dtype=torch.float16)
        b = torch.rand(hidden_size, device='cuda', dtype=torch.float16)
        eps = 1e-5
        # Warmup
        for _ in range(num_warmup):
            y = program.LigerLayerNormFunction.apply(x, w, b, eps)
            torch.cuda.synchronize()
        # Measure
        run_times = []
        for _ in range(num_measure):
            start_time = time.time()
            y = program.LigerLayerNormFunction.apply(x, w, b, eps)
            torch.cuda.synchronize()
            run_times.append((time.time() - start_time) * 1000)
        avg_time = np.mean(run_times)
        total_times.append(avg_time)
        print(f"Batch: {batch_size}, Seq: {seq_len}, Hidden: {hidden_size}, Avg Fwd: {avg_time:.2f}ms")
    return np.mean(total_times)

def _run_correctness_check(program, test_sizes, atol=1e-3, rtol=1e-3):
    torch.manual_seed(42)
    np.random.seed(42)
    print("[DEBUG] ReferenceLayerNorm:", ReferenceLayerNorm)
    print("[DEBUG] Candidate LigerLayerNormFunction:", program.LigerLayerNormFunction)
    print("[DEBUG] Is same object:", ReferenceLayerNorm is program.LigerLayerNormFunction)
    max_error = 0.0
    for batch_size, seq_len, hidden_size in test_sizes:
        x = torch.rand(batch_size, seq_len, hidden_size, device='cuda', dtype=torch.float16, requires_grad=True)
        w = torch.rand(hidden_size, device='cuda', dtype=torch.float16, requires_grad=True)
        b = torch.rand(hidden_size, device='cuda', dtype=torch.float16, requires_grad=True)
        eps = 1e-5
        # Reference
        x_ref = x.clone().detach().requires_grad_(True)
        w_ref = w.clone().detach().requires_grad_(True)
        b_ref = b.clone().detach().requires_grad_(True)
        y_ref = ReferenceLayerNorm.apply(x_ref, w_ref, b_ref, eps)
        grad_output = torch.rand_like(y_ref)
        y_ref.backward(grad_output)
        # Optimized
        x_test = x.clone().detach().requires_grad_(True)
        w_test = w.clone().detach().requires_grad_(True)
        b_test = b.clone().detach().requires_grad_(True)
        y_test = program.LigerLayerNormFunction.apply(x_test, w_test, b_test, eps)
        y_test.backward(grad_output)
        # Print tensor values and properties
        print(f"[DEBUG] Batch: {batch_size}, Seq: {seq_len}, Hidden: {hidden_size}")
        print(f"  w_ref[:5]: {w_ref[:5].detach().cpu().numpy()}")
        print(f"  w_test[:5]: {w_test[:5].detach().cpu().numpy()}")
        print(f"  b_ref[:5]: {b_ref[:5].detach().cpu().numpy()}")
        print(f"  b_test[:5]: {b_test[:5].detach().cpu().numpy()}")
        print(f"  x_ref.device: {x_ref.device}, x_test.device: {x_test.device}")
        print(f"  w_ref.dtype: {w_ref.dtype}, w_test.dtype: {w_test.dtype}")
        # Compare outputs
        try:
            output_error = torch.max(torch.abs(y_ref - y_test)).item()
            grad_x_error = torch.max(torch.abs(x_ref.grad - x_test.grad)).item()
            grad_w_error = torch.max(torch.abs(w_ref.grad - w_test.grad)).item()
            grad_b_error = torch.max(torch.abs(b_ref.grad - b_test.grad)).item()
            print(f"  output_error: {output_error}")
            print(f"  grad_x_error: {grad_x_error}")
            print(f"  grad_w_error: {grad_w_error}")
            print(f"  grad_b_error: {grad_b_error}")
            max_error = max(max_error, output_error, grad_x_error, grad_w_error, grad_b_error)
            assert output_error <= atol + rtol * torch.max(torch.abs(y_ref)).item()
            assert grad_x_error <= atol + rtol * torch.max(torch.abs(x_ref.grad)).item()
            assert grad_w_error <= atol + rtol * torch.max(torch.abs(w_ref.grad)).item()
            assert grad_b_error <= atol + rtol * torch.max(torch.abs(b_ref.grad)).item()
        except AssertionError:
            return False, max_error
    return True, max_error

def _calc_scores(avg_latency_ms, ref_latency_ms, max_error):
    speedup = ref_latency_ms / avg_latency_ms
    latency_score = 1.0 / (1.0 + np.exp(-(speedup - 1.0)))
    correctness_score = 1.0 / (1.0 + max_error)
    combined_score = 0.3 * correctness_score + 0.7 * latency_score
    return latency_score, correctness_score, combined_score

def evaluate(program_path):
    test_sizes = [
        (32, 512, 4096),
        (64, 1024, 4096),
        (16, 2048, 4096),
        (32, 512, 8192),
    ]
    try:
        program = _load_program(program_path)
        if not _check_program_required_functions(program):
            return {
                "latency_score": 0.0,
                "correctness_score": 0.0,
                "combined_score": 0.0,
                "error": "Missing required functions"
            }
        ref_latency = _run_reference_latency_check(test_sizes)
        print(f"\nReference implementation average latency: {ref_latency:.2f}ms")
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
    test_sizes = [
        (32, 512, 4096),
        (64, 1024, 4096),
        (16, 2048, 4096),
        (32, 512, 8192),
    ]
    try:
        program = _load_program(program_path)
        if not _check_program_required_functions(program):
            return {
                "latency_score": 0.0,
                "correctness_score": 0.0,
                "combined_score": 0.0,
                "error": "Missing required functions"
            }
        ref_latency = _run_reference_latency_check(test_sizes)
        print(f"\nReference implementation average latency: {ref_latency:.2f}ms")
        avg_latency = _run_latency_check(program, test_sizes)
        print(f"Optimized implementation average latency: {avg_latency:.2f}ms", flush=True)
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
    return evaluate(program_path)

if __name__ == "__main__":
    print(evaluate_stage1("initial_program.py"))
