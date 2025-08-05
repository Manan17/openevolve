import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):

    # Get the program ID
    pid = tl.program_id(axis=0)

    # Calculate the starting element index for this block
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Load elements from a and b with masking
    a = tl.load(a_ptr + offsets, mask=offsets < num_elements)
    b = tl.load(b_ptr + offsets, mask=offsets < num_elements)

    # Perform the addition
    result = a + b

    # Store the result with masking
    tl.store(c_ptr + offsets, result, mask=offsets < num_elements)

def vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function for the vector addition kernel.
    """
    assert a.shape == b.shape, "Input tensors must have the same shape"
    assert a.is_cuda and b.is_cuda, "Input tensors must be on the same GPU device"
    
    num_elements = a.numel()
    c = torch.empty_like(a)
    
    # Use optimal block size for H100 - 1024 provides excellent occupancy and memory bandwidth
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
    
    add_kernel[grid](
        a,
        b,
        c,
        num_elements,
        BLOCK_SIZE=BLOCK_SIZE, 
    )
    
    return c


if __name__ == '__main__':
    size = 1024 * 1024
    a = torch.randn(size, device='cuda')
    b = torch.randn(size, device='cuda')

    # --- Proper Benchmarking ---
    num_warmup = 10
    num_measure = 100

    print("Performing warm-up runs to handle JIT compilation...")
    for _ in range(num_warmup):
        _ = vector_add(a, b)
    
    # Synchronize to ensure all warm-up runs are complete before we start timing
    torch.cuda.synchronize()

    # --- Timing the Kernel Execution (The Right Way) ---
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    
    # Measure multiple runs to get a stable average
    for _ in range(num_measure):
        c_triton = vector_add(a, b)

    end_event.record()
    torch.cuda.synchronize()

    # Calculate the average time per run in milliseconds
    total_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = total_time_ms / num_measure
    
    print(f"\nOptimized Triton kernel executed successfully.")
    print(f"Average execution time over {num_measure} runs: {avg_time_ms:.6f} ms")

    # For verification
    c_pytorch = a + b
    assert torch.allclose(c_triton, c_pytorch), "The results do not match!"
    print("Verification successful.")