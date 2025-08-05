import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Compute block indices
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over blocks
    for k in range(0, K, BLOCK_SIZE_K):
        # Load tiles with proper bounds checking
        a_tile = tl.load(
            a_ptr + 
            (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]) * stride_am +
            (k + tl.arange(0, BLOCK_SIZE_K)[None, :]) * stride_ak,
            mask=(tl.arange(0, BLOCK_SIZE_M)[:, None] < M - pid_m * BLOCK_SIZE_M) &
                  (tl.arange(0, BLOCK_SIZE_K)[None, :] < K - k),
            other=0.0
        )
        
        b_tile = tl.load(
            b_ptr +
            (k + tl.arange(0, BLOCK_SIZE_K)[:, None]) * stride_bk +
            (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]) * stride_bn,
            mask=(tl.arange(0, BLOCK_SIZE_K)[:, None] < K - k) &
                  (tl.arange(0, BLOCK_SIZE_N)[None, :] < N - pid_n * BLOCK_SIZE_N),
            other=0.0
        )
        
        # Matrix multiplication
        acc += tl.dot(a_tile, b_tile)
    
    # Store result
    c_tile = acc.to(tl.float16)
    tl.store(
        c_ptr +
        (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]) * stride_cm +
        (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]) * stride_cn,
        c_tile,
        mask=(tl.arange(0, BLOCK_SIZE_M)[:, None] < M - pid_m * BLOCK_SIZE_M) &
              (tl.arange(0, BLOCK_SIZE_N)[None, :] < N - pid_n * BLOCK_SIZE_N)
    )


def compute_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape[1] == b.shape[0], "Incompatible matrix dimensions"
    assert a.is_cuda and b.is_cuda, "Input tensors must be on the same GPU device"
    
    M, K = a.shape
    _K, N = b.shape
    
    # Define block sizes
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    
    # Ensure block sizes are multiples of 32 (warp size)
    BLOCK_SIZE_M = (BLOCK_SIZE_M + 31) // 32 * 32
    BLOCK_SIZE_N = (BLOCK_SIZE_N + 31) // 32 * 32
    BLOCK_SIZE_K = (BLOCK_SIZE_K + 31) // 32 * 32
    
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    
    grid = (
        (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,
        (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    )
    
    matmul_kernel[grid](
        a,
        b,
        c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K
    )
    
    return c


if __name__ == '__main__':
    # NOTE: This kernel is very slow. Keep matrix dimensions small for testing.
    M, N, K = 128, 128, 128
    
    a = torch.randn((M, K), device='cuda', dtype=torch.float32)
    b = torch.randn((K, N), device='cuda', dtype=torch.float32)

    # --- Proper Benchmarking Setup ---
    num_warmup = 5
    num_measure = 10

    print("Performing warm-up runs to handle JIT compilation...")
    for _ in range(num_warmup):
        _ = compute_matmul(a, b)
    
    # Synchronize to ensure all warm-up runs are complete before timing begins
    torch.cuda.synchronize()

    # --- Timing the Kernel Execution (The Right Way) ---
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    
    # Measure multiple runs to get a stable average
    for _ in range(num_measure):
        c_triton = compute_matmul(a, b)

    end_event.record()
    torch.cuda.synchronize()

    # Calculate the average time per run in milliseconds
    total_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = total_time_ms / num_measure
    
    print(f"\nTriton kernel executed successfully.")
    print(f"Average execution time over {num_measure} runs: {avg_time_ms:.3f} ms")

    # For verification
    c_pytorch = torch.matmul(a, b)
    if torch.allclose(c_triton, c_pytorch, atol=1e-2, rtol=0):
        print("Verification successful.")
    else:
        print("Verification FAILED.")

