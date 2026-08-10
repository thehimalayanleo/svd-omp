import pytest
import torch
from torch import nn

from cp_svd_runtime import CPSVDLinear, replace_submodule


def reference_cp_svd(
    inputs: torch.Tensor,
    analysis: torch.Tensor,
    dictionary: torch.Tensor,
    selected_units: int,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    shape = inputs.shape
    coefficients = inputs.reshape(-1, shape[-1]).matmul(analysis)
    indices = coefficients.abs().topk(selected_units, dim=1).indices
    selected = torch.zeros_like(coefficients)
    selected.scatter_(1, indices, coefficients.gather(1, indices))
    output = selected.matmul(dictionary.T)
    if bias is not None:
        output = output + bias
    return output.reshape(*shape[:-1], dictionary.shape[0])


def test_direct_module_matches_prototype_formula() -> None:
    generator = torch.Generator().manual_seed(17)
    analysis = torch.randn(11, 7, generator=generator)
    dictionary = torch.randn(13, 7, generator=generator)
    bias = torch.randn(13, generator=generator)
    inputs = torch.randn(2, 5, 11, generator=generator)
    module = CPSVDLinear(analysis, dictionary, selected_units=3, bias=bias)
    expected = reference_cp_svd(inputs, analysis, dictionary, 3, bias)
    torch.testing.assert_close(module(inputs), expected)


def test_direct_module_has_no_dense_weight_parameter() -> None:
    module = CPSVDLinear(
        torch.randn(12, 6),
        torch.randn(10, 6),
        selected_units=2,
    )
    assert dict(module.named_parameters()) == {}
    assert set(dict(module.named_buffers())) == {"analysis", "dictionary"}
    assert module.analysis.numel() + module.dictionary.numel() == 132


def test_replace_submodule_round_trip() -> None:
    model = nn.Sequential(nn.Linear(5, 7), nn.Sequential(nn.Linear(7, 3)))
    replacement = CPSVDLinear(torch.randn(7, 2), torch.randn(3, 2), 1)
    original = replace_submodule(model, "1.0", replacement)
    assert isinstance(original, nn.Linear)
    assert model.get_submodule("1.0") is replacement
    restored = replace_submodule(model, "1.0", original)
    assert restored is replacement
    assert model.get_submodule("1.0") is original


def test_rejects_incompatible_factors() -> None:
    with pytest.raises(ValueError, match="pool widths"):
        CPSVDLinear(torch.randn(4, 3), torch.randn(5, 2), 1)
