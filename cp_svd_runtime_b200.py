"""Direct B200 runtime modules for frozen Calibration-Pruned SVD (CP-SVD).

The original simultaneous-quality prototype used forward hooks.  A PyTorch
forward hook runs after ``nn.Linear.forward``, so that prototype paid for the
dense matrix multiplication and then overwrote its output with CP-SVD.  This
module is a true replacement: it stores only the frozen analysis and synthesis
factors and never evaluates the original dense weight.
"""

from __future__ import annotations

import torch
from torch import nn

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only in non-Triton installs
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _selected_synthesis_kernel(
        coefficients,
        indices,
        synthesis,
        bias,
        output,
        row_count: tl.constexpr,
        output_width: tl.constexpr,
        pool_size: tl.constexpr,
        selected_units: tl.constexpr,
        has_bias: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_O: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        columns = tl.program_id(1) * BLOCK_O + tl.arange(0, BLOCK_O)
        row_mask = rows < row_count
        column_mask = columns < output_width
        accumulator = tl.zeros((BLOCK_N, BLOCK_O), tl.float32)
        for slot in range(selected_units):
            component = tl.load(
                indices + rows * selected_units + slot,
                mask=row_mask,
                other=0,
            )
            coefficient = tl.load(
                coefficients + rows * pool_size + component,
                mask=row_mask,
                other=0.0,
            )
            atom = tl.load(
                synthesis + component[:, None] * output_width + columns[None, :],
                mask=row_mask[:, None] & column_mask[None, :],
                other=0.0,
            )
            accumulator += coefficient[:, None] * atom
        if has_bias:
            accumulator += tl.load(
                bias + columns[None, :],
                mask=column_mask[None, :],
                other=0.0,
            )
        tl.store(
            output + rows[:, None] * output_width + columns[None, :],
            accumulator,
            mask=row_mask[:, None] & column_mask[None, :],
        )


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


class CPSVDLinearB200(nn.Module):
    """B200-oriented CP-SVD replacement with fused selected synthesis.

    The protected implementation scatters selected coefficients into a
    zero-filled 96-wide tensor and evaluates a dense synthesis GEMM. This
    variant keeps the same analysis, top-k support, and output formula, but a
    Triton kernel directly gathers the selected atoms and sums only those
    contributions.
    """

    def __init__(
        self,
        analysis: torch.Tensor,
        dictionary: torch.Tensor,
        selected_units: int,
        bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if triton is None:
            raise RuntimeError("CPSVDLinearB200 requires Triton")
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
        self.register_buffer("synthesis", dictionary.T.detach().contiguous())
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
        output = torch.empty(
            (flat.shape[0], self.out_features),
            device=flat.device,
            dtype=torch.float32,
        )
        grid = (
            triton.cdiv(flat.shape[0], 4),
            triton.cdiv(self.out_features, 128),
        )
        _selected_synthesis_kernel[grid](
            coefficients,
            indices,
            self.synthesis,
            self.bias if self.bias is not None else self.synthesis,
            output,
            row_count=flat.shape[0],
            output_width=self.out_features,
            pool_size=self.pool_size,
            selected_units=self.selected_units,
            has_bias=self.bias is not None,
            BLOCK_N=4,
            BLOCK_O=128,
            num_warps=4,
        )
        return output.reshape(*inputs.shape[:-1], self.out_features).to(original_dtype)


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
