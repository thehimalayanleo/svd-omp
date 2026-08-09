from __future__ import annotations

import torch

from quantized_mdl_discovery import (
    balance_factors,
    structured_sparse_code_bits,
    symmetric_quantize,
)


def test_symmetric_quantization_preserves_zeros_and_bounds_levels() -> None:
    values = torch.tensor([-2.0, -0.1, 0.0, 0.8, 2.0])
    quantized, scale = symmetric_quantize(values, bits=3)
    assert quantized[2] == 0
    assert int(torch.unique(quantized).numel()) <= 7
    assert scale > 0


def test_factor_balancing_preserves_product() -> None:
    torch.manual_seed(5)
    factor_a = torch.randn(7, 4)
    factor_b = torch.randn(4, 6)
    balanced_a, balanced_b = balance_factors(factor_a, factor_b)
    assert torch.allclose(
        factor_a.matmul(factor_b), balanced_a.matmul(balanced_b), atol=1e-5
    )


def test_structured_sparse_code_is_no_worse_than_global_code() -> None:
    values = torch.zeros(8, 9)
    values[:2, :3] = 1
    from mdl_svdomp_vs_swd import sparse_array_code_bits

    assert structured_sparse_code_bits(values, 4) <= sparse_array_code_bits(
        values.numel(), int(values.ne(0).sum()), 4
    )
