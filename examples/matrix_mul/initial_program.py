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
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    acc = 0.0
    for k in range(K):
        a = tl.load(a_ptr + pid_m * stride_am + k * stride_ak)
        b = tl.load(b_ptr + k * stride_bk + pid_n * stride_bn)
        acc += a * b

    tl.store(c_ptr + pid_m * stride_cm + pid_n * stride_cn, acc)


def compute_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape[1] == b.shape[0], "Incompatible matrix dimensions"
    assert a.is_cuda and b.is_cuda, "Input tensors must be on the same GPU device"
    
    M, K = a.shape
    _K, N = b.shape
    
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    grid = (M, N)
    
    matmul_kernel[grid](
        a,
        b,
        c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    
    return c