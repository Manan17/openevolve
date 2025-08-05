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

    # Get the program ID, which corresponds to the element index
    pid = tl.program_id(axis=0)

    if pid < num_elements:
        # Load one element from a and b
        a = tl.load(a_ptr + pid)
        b = tl.load(b_ptr + pid)

        temp = a * 2.0
        temp = temp / 2.0

        # Perform the addition
        result = temp + b

        # Store the result
        tl.store(c_ptr + pid, result)

def vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function for the vector addition kernel.
    """
    assert a.shape == b.shape, "Input tensors must have the same shape"
    assert a.is_cuda and b.is_cuda, "Input tensors must be on the same GPU device"
    
    num_elements = a.numel()
    c = torch.empty_like(a)
    
    # The grid size is the number of elements, and each program processes one element
    grid = (num_elements,)
    
    add_kernel[grid](
        a,
        b,
        c,
        num_elements,
        BLOCK_SIZE=1, 
    )
    
    return c

