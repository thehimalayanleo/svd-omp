"""Direct runtime modules for frozen Calibration-Pruned SVD (CP-SVD).

The original simultaneous-quality prototype used forward hooks.  A PyTorch
forward hook runs after ``nn.Linear.forward``, so that prototype paid for the
dense matrix multiplication and then overwrote its output with CP-SVD.  This
module is a true replacement: it stores only the frozen analysis and synthesis
factors and never evaluates the original dense weight.
"""

from __future__ import annotations

import torch
from torch import nn


class CPSVDLinear(nn.Module):
    """Inference-only CP-SVD replacement for ``torch.nn.Linear``.

    ``analysis`` has shape ``[input_width, pool_size]`` and ``dictionary`` has
    shape ``[output_width, pool_size]``.  For each input, the module keeps the
    ``selected_units`` largest-magnitude coefficients and reconstructs through
    the frozen output dictionary.
    """

    def __init__(
        self,
        analysis: torch.Tensor,
        dictionary: torch.Tensor,
        selected_units: int,
        bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if analysis.ndim != 2 or dictionary.ndim != 2:
            raise ValueError("analysis and dictionary must be two-dimensional")
        if analysis.shape[1] != dictionary.shape[1]:
            raise ValueError("analysis and dictionary pool widths must match")
        if not 1 <= selected_units <= analysis.shape[1]:
            raise ValueError("selected_units must be between one and pool_size")
        if bias is not None and bias.shape != (dictionary.shape[0],):
            raise ValueError("bias shape must match the output width")

        self.selected_units = int(selected_units)
        self.in_features = int(analysis.shape[0])
        self.out_features = int(dictionary.shape[0])
        self.pool_size = int(analysis.shape[1])
        self.register_buffer("analysis", analysis.detach().contiguous())
        self.register_buffer("dictionary", dictionary.detach().contiguous())
        self.register_buffer(
            "bias",
            None if bias is None else bias.detach().contiguous(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-1] != self.in_features:
            raise ValueError(
                f"expected input width {self.in_features}, got {inputs.shape[-1]}"
            )
        original_dtype = inputs.dtype
        flat = inputs.reshape(-1, self.in_features).to(self.analysis.dtype)
        coefficients = flat.matmul(self.analysis)
        indices = coefficients.abs().topk(self.selected_units, dim=1).indices
        selected = torch.zeros_like(coefficients)
        selected.scatter_(1, indices, coefficients.gather(1, indices))
        output = selected.matmul(self.dictionary.T)
        if self.bias is not None:
            output = output + self.bias
        return output.reshape(*inputs.shape[:-1], self.out_features).to(original_dtype)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"pool_size={self.pool_size}, selected_units={self.selected_units}, "
            f"bias={self.bias is not None}"
        )


def replace_submodule(root: nn.Module, path: str, replacement: nn.Module) -> nn.Module:
    """Replace a dotted-path child module and return the previous module."""

    if not path:
        raise ValueError("path must not be empty")
    parent_path, separator, child_name = path.rpartition(".")
    parent = root.get_submodule(parent_path) if separator else root
    previous = getattr(parent, child_name)
    if not isinstance(previous, nn.Module):
        raise ValueError(f"{path} does not resolve to a module")
    setattr(parent, child_name, replacement)
    return previous
