"""
Sum Reduction Example
================

This example demonstrates how to implement a sum reduction operation along the last dimension using Helion.
"""

# %%
# Imports
# -------
from __future__ import annotations

import torch

import helion
from helion._testing import run_example
import helion.language as hl


# %%
# Sum Kernel
# --------
@helion.kernel()
def sum_kernel(x: torch.Tensor) -> torch.Tensor:
    """
    Sums a 2D tensor along the last dimension.

    Args:
        x: Input tensor of shape [M, N]

    Returns:
        Output tensor of shape [M] containing the sum of each row
    """
    m, n = x.shape
    out = torch.empty([m], dtype=x.dtype, device=x.device)

    for tile_m in hl.tile(m):
        out[tile_m] = x[tile_m, :].sum(-1)

    return out


# %%
# Benchmark Wrapper
# --------------
def sum_tritonbench(x: torch.Tensor) -> torch.Tensor:
    """
    Wrapper for tritonbench that handles 1D input.

    Args:
        x: Input tensor (1D or 2D)

    Returns:
        Sum of the tensor along the last dimension
    """
    if x.ndim == 1:
        # For 1D tensors, reshape to 2D for sum_kernel
        x_2d = x.unsqueeze(0)
        result = sum_kernel(x_2d)
        return result.squeeze()
    return sum_kernel(x)


# %%
# Verification Function
# -------------------
def check(m: int, n: int) -> None:
    """
    Verify the sum kernel implementation against PyTorch's native sum function.

    Args:
        m: First dimension of the test tensor
        n: Second dimension of the test tensor
    """
    x = torch.randn([m, n], device="cuda", dtype=torch.float32)
    kernels = {"helion": sum_kernel}
    run_example(kernels, lambda x: x.sum(-1), (x,))


# %%
# Main Function
# -----------
def main() -> None:
    """
    Main entry point that runs the sum kernel verification with different tensor sizes.

    Tests with two configurations:
    - 512x256
    - 1024x1024
    """
    check(512, 256)
    check(1024, 1024)


if __name__ == "__main__":
    main()
