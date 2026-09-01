"""Full-model interventions for auditing SVD-OMP component causality.

This module deliberately does *not* call local reconstruction error
"causality".  It supplies the small, model-agnostic pieces needed to insert or
remove selected rank-one atoms inside an ``nn.Linear`` module while a complete
model is running.  A caller must measure a held-out behavioral endpoint.

For a matrix ``D = U diag(S) V^T`` (normally a post-training delta
``W_post - W_base``), atom ``c`` contributes

    (x @ v_c) * (sigma_c * u_c)

to the linear output.  ``input_omp`` chooses the k atoms with the largest
contribution norms for each token.  The matched-random control samples from a
larger high-energy pool and rescales the combined perturbation to exactly match
the SVD-OMP perturbation norm token by token.  This prevents a larger local
perturbation from being mistaken for stronger behavioral causality.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Literal, Sequence

import torch
from torch import Tensor, nn


Policy = Literal[
    "input_omp",
    "static_svd",
    "activation_only",
    "matched_random",
    "matched_gaussian",
]
Mode = Literal["insert", "ablate"]


@dataclass(frozen=True)
class SVDAtoms:
    """A truncated orthogonal dictionary for one linear weight or delta."""

    V: Tensor
    U_sigma: Tensor
    S: Tensor

    def __post_init__(self) -> None:
        if self.V.ndim != 2 or self.U_sigma.ndim != 2 or self.S.ndim != 1:
            raise ValueError("V, U_sigma, and S must have ranks 2, 2, and 1")
        c = self.S.shape[0]
        if self.V.shape[1] != c or self.U_sigma.shape[0] != c:
            raise ValueError("component dimensions do not agree")
        if c == 0:
            raise ValueError("the dictionary must contain at least one atom")

    @property
    def n_components(self) -> int:
        return self.S.shape[0]

    @property
    def d_in(self) -> int:
        return self.V.shape[0]

    @property
    def d_out(self) -> int:
        return self.U_sigma.shape[1]

    def to(self, *args, **kwargs) -> "SVDAtoms":
        return SVDAtoms(
            self.V.to(*args, **kwargs),
            self.U_sigma.to(*args, **kwargs),
            self.S.to(*args, **kwargs),
        )


def atoms_from_svd(weight_or_delta: Tensor, n_components: int) -> SVDAtoms:
    """Build exact leading SVD atoms for a modest-size matrix.

    Real 1B+ model experiments should normally use ``atoms_from_lowrank`` to
    avoid computing a complete SVD of every transformer projection.
    """

    if weight_or_delta.ndim != 2:
        raise ValueError("weight_or_delta must be [d_out, d_in]")
    u, s, vt = torch.linalg.svd(weight_or_delta, full_matrices=False)
    c = min(n_components, s.numel())
    return SVDAtoms(
        V=vt[:c].T.contiguous(),
        U_sigma=(u[:, :c] * s[:c]).T.contiguous(),
        S=s[:c].contiguous(),
    )


def atoms_from_lowrank(
    weight_or_delta: Tensor,
    n_components: int,
    *,
    oversample: int = 8,
    niter: int = 4,
    seed: int | None = None,
) -> SVDAtoms:
    """Build approximate leading atoms with randomized low-rank SVD."""

    if weight_or_delta.ndim != 2:
        raise ValueError("weight_or_delta must be [d_out, d_in]")
    max_rank = min(weight_or_delta.shape)
    c = min(n_components, max_rank)
    q = min(max_rank, c + max(0, oversample))
    if seed is None:
        u, s, v = torch.svd_lowrank(weight_or_delta, q=q, niter=niter)
    else:
        cuda_devices = []
        if weight_or_delta.device.type == "cuda":
            cuda_devices = [
                weight_or_delta.device.index
                if weight_or_delta.device.index is not None
                else torch.cuda.current_device()
            ]
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            u, s, v = torch.svd_lowrank(weight_or_delta, q=q, niter=niter)
    order = torch.argsort(s, descending=True)[:c]
    s = s[order]
    u = u[:, order]
    v = v[:, order]
    return SVDAtoms(
        V=v.contiguous(),
        U_sigma=(u * s).T.contiguous(),
        S=s.contiguous(),
    )


def _flatten_tokens(x: Tensor, d_in: int) -> tuple[Tensor, tuple[int, ...]]:
    if x.shape[-1] != d_in:
        raise ValueError(f"expected input width {d_in}, got {x.shape[-1]}")
    leading = tuple(x.shape[:-1])
    return x.reshape(-1, d_in), leading


def _support(
    projections: Tensor,
    atoms: SVDAtoms,
    *,
    policy: Policy,
    k: int,
    generator: torch.Generator | None,
    pool_factor: int,
) -> Tensor:
    c = atoms.n_components
    if not 1 <= k <= c:
        raise ValueError(f"k must be in [1, {c}], got {k}")
    scores = projections.abs() * atoms.S.to(projections).unsqueeze(0)

    if policy == "input_omp":
        return scores.topk(k, dim=-1).indices
    if policy == "activation_only":
        return projections.abs().topk(k, dim=-1).indices
    if policy == "static_svd":
        static = atoms.S.topk(k).indices.to(projections.device)
        return static.unsqueeze(0).expand(projections.shape[0], -1)
    if policy == "matched_random":
        pool_size = min(c, max(k, pool_factor * k))
        pool = scores.topk(pool_size, dim=-1).indices
        noise = torch.rand(
            pool.shape,
            device=projections.device,
            generator=generator,
            dtype=torch.float32,
        )
        positions = noise.topk(k, dim=-1).indices
        return pool.gather(1, positions)
    raise ValueError(f"policy {policy!r} has no component support")


def _sum_atom_contributions(
    projections: Tensor,
    support: Tensor,
    atoms: SVDAtoms,
) -> Tensor:
    selected_projection = projections.gather(1, support)
    selected_u = atoms.U_sigma.to(projections)[support]
    return (selected_projection.unsqueeze(-1) * selected_u).sum(dim=1)


def component_perturbation(
    x: Tensor,
    atoms: SVDAtoms,
    *,
    policy: Policy,
    k: int,
    seed: int = 0,
    pool_factor: int = 4,
    match_reference_norm: bool = True,
) -> Tensor:
    """Return the output perturbation selected for every input token.

    ``matched_random`` samples actual atoms from the top ``pool_factor * k``
    per-token candidates.  ``matched_gaussian`` is a direction-only null.
    Both controls can be rescaled to the exact norm of the input-OMP
    perturbation on each token.  The measured RMS still needs to be recorded
    in the experiment artifact.
    """

    flat, leading = _flatten_tokens(x, atoms.d_in)
    work = flat.to(device=atoms.V.device, dtype=atoms.V.dtype)
    projections = work @ atoms.V
    generator = torch.Generator(device=projections.device)
    generator.manual_seed(seed)

    reference_support = _support(
        projections,
        atoms,
        policy="input_omp",
        k=k,
        generator=None,
        pool_factor=pool_factor,
    )
    reference = _sum_atom_contributions(projections, reference_support, atoms)

    if policy == "matched_gaussian":
        perturbation = torch.randn(
            reference.shape,
            device=reference.device,
            dtype=reference.dtype,
            generator=generator,
        )
    else:
        support = _support(
            projections,
            atoms,
            policy=policy,
            k=k,
            generator=generator,
            pool_factor=pool_factor,
        )
        perturbation = _sum_atom_contributions(projections, support, atoms)

    if match_reference_norm and policy in {"matched_random", "matched_gaussian"}:
        ref_norm = reference.norm(dim=-1, keepdim=True)
        current_norm = perturbation.norm(dim=-1, keepdim=True)
        perturbation = perturbation * ref_norm / current_norm.clamp_min(1e-12)

    return perturbation.reshape(*leading, atoms.d_out).to(x.device)


class LinearComponentIntervention(AbstractContextManager):
    """Temporarily insert or remove selected atoms at an ``nn.Linear``."""

    def __init__(
        self,
        module: nn.Linear,
        atoms: SVDAtoms,
        *,
        policy: Policy,
        k: int,
        mode: Mode,
        seed: int = 0,
        pool_factor: int = 4,
        match_reference_norm: bool = True,
        record_token_norms: bool = False,
        replay_token_norms: Sequence[Tensor] | None = None,
    ) -> None:
        if module.in_features != atoms.d_in or module.out_features != atoms.d_out:
            raise ValueError(
                "linear module and atom shapes disagree: "
                f"module=({module.out_features}, {module.in_features}), "
                f"atoms=({atoms.d_out}, {atoms.d_in})"
            )
        if mode not in {"insert", "ablate"}:
            raise ValueError(f"unknown mode {mode!r}")
        self.module = module
        self.atoms = atoms
        self.policy = policy
        self.k = k
        self.mode = mode
        self.seed = seed
        self.pool_factor = pool_factor
        self.match_reference_norm = match_reference_norm
        self.record_token_norms = record_token_norms
        self.replay_token_norms = replay_token_norms
        self._handle = None
        self._sum_sq = 0.0
        self._n_values = 0
        self._call_index = 0
        self._token_norm_trace: list[Tensor] = []

    @property
    def perturbation_rms(self) -> float:
        if self._n_values == 0:
            return 0.0
        return (self._sum_sq / self._n_values) ** 0.5

    @property
    def token_norm_trace(self) -> tuple[Tensor, ...]:
        """Per-token perturbation norms, stored on CPU in hook-call order."""

        return tuple(self._token_norm_trace)

    def _hook(self, _module: nn.Module, inputs: tuple[Tensor, ...], output: Tensor) -> Tensor:
        x = inputs[0]
        perturbation = component_perturbation(
            x,
            self.atoms,
            policy=self.policy,
            k=self.k,
            seed=self.seed,
            pool_factor=self.pool_factor,
            match_reference_norm=self.match_reference_norm,
        ).to(dtype=output.dtype)
        if self.replay_token_norms is not None:
            if self._call_index >= len(self.replay_token_norms):
                raise RuntimeError("paired-dose norm trace was exhausted")
            target_norm = self.replay_token_norms[self._call_index].to(
                device=perturbation.device,
                dtype=torch.float32,
            )
            expected_shape = (*perturbation.shape[:-1], 1)
            if tuple(target_norm.shape) != expected_shape:
                raise ValueError(
                    "paired-dose norm trace shape disagrees with perturbation: "
                    f"target={tuple(target_norm.shape)}, expected={expected_shape}"
                )
            current_norm = perturbation.float().norm(dim=-1, keepdim=True)
            perturbation = (
                perturbation.float()
                * target_norm
                / current_norm.clamp_min(1e-12)
            ).to(dtype=output.dtype)
        if self.record_token_norms:
            self._token_norm_trace.append(
                perturbation.float().norm(dim=-1, keepdim=True).detach().cpu()
            )
        self._call_index += 1
        self._sum_sq += perturbation.float().square().sum().item()
        self._n_values += perturbation.numel()
        sign = 1.0 if self.mode == "insert" else -1.0
        return output + sign * perturbation

    def __enter__(self) -> "LinearComponentIntervention":
        if self._handle is not None:
            raise RuntimeError("intervention context is already active")
        self._handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        if (
            exc_type is None
            and self.replay_token_norms is not None
            and self._call_index != len(self.replay_token_norms)
        ):
            raise RuntimeError(
                "paired-dose norm trace was not fully consumed: "
                f"used {self._call_index} of {len(self.replay_token_norms)} calls"
            )


class LinearDeltaIntervention(AbstractContextManager):
    """Temporarily insert or remove a complete linear weight delta.

    This is the module-set positive control. If a complete delta intervention
    does not move the behavior, failure of a sparse atom intervention cannot be
    attributed to atom selection.
    """

    def __init__(self, module: nn.Linear, delta: Tensor, *, mode: Mode) -> None:
        if tuple(delta.shape) != tuple(module.weight.shape):
            raise ValueError(
                f"delta shape {tuple(delta.shape)} does not match "
                f"module weight {tuple(module.weight.shape)}"
            )
        if mode not in {"insert", "ablate"}:
            raise ValueError(f"unknown mode {mode!r}")
        self.module = module
        self.delta = delta
        self.mode = mode
        self._handle = None
        self._sum_sq = 0.0
        self._n_values = 0

    @property
    def perturbation_rms(self) -> float:
        if self._n_values == 0:
            return 0.0
        return (self._sum_sq / self._n_values) ** 0.5

    def _hook(self, _module: nn.Module, inputs: tuple[Tensor, ...], output: Tensor) -> Tensor:
        x = inputs[0]
        perturbation = torch.nn.functional.linear(
            x.to(device=self.delta.device, dtype=self.delta.dtype),
            self.delta,
        ).to(device=output.device, dtype=output.dtype)
        self._sum_sq += perturbation.float().square().sum().item()
        self._n_values += perturbation.numel()
        sign = 1.0 if self.mode == "insert" else -1.0
        return output + sign * perturbation

    def __enter__(self) -> "LinearDeltaIntervention":
        if self._handle is not None:
            raise RuntimeError("intervention context is already active")
        self._handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


@dataclass(frozen=True)
class CausalAuditSummary:
    reference_effect: float
    candidate_effect: float
    random_effect: float
    candidate_minus_random_ci: tuple[float, float]
    transfer_fraction: float
    control_effect: float
    specificity_ratio: float
    passes_direction: bool
    passes_random_control: bool
    passes_specificity: bool

    @property
    def passes(self) -> bool:
        return self.passes_direction and self.passes_random_control and self.passes_specificity


@dataclass(frozen=True)
class SingleCheckpointAblationSummary:
    """Causal-necessity summary when only one trained checkpoint is available."""

    candidate_effect: float
    random_effect: float
    static_effect: float
    candidate_minus_random_ci: tuple[float, float]
    candidate_minus_static_ci: tuple[float, float]
    control_effect: float
    specificity_ratio: float
    passes_direction: bool
    passes_random_control: bool
    passes_static_control: bool
    passes_specificity: bool

    @property
    def passes(self) -> bool:
        return (
            self.passes_direction
            and self.passes_random_control
            and self.passes_static_control
            and self.passes_specificity
        )


def _bootstrap_mean_ci(
    values: Tensor,
    *,
    seed: int,
    n_bootstrap: int,
    alpha: float,
) -> tuple[float, float]:
    values = values.detach().float().cpu().flatten()
    if values.numel() == 0:
        raise ValueError("cannot bootstrap an empty tensor")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(
        values.numel(),
        (n_bootstrap, values.numel()),
        generator=generator,
    )
    means = values[indices].mean(dim=1)
    low = torch.quantile(means, alpha / 2).item()
    high = torch.quantile(means, 1 - alpha / 2).item()
    return low, high


def summarize_causal_audit(
    *,
    base_margin: Tensor,
    post_margin: Tensor,
    candidate_margin: Tensor,
    random_margin: Tensor,
    target_mask: Tensor,
    control_mask: Tensor,
    mode: Mode,
    seed: int = 0,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    max_control_fraction: float = 0.5,
) -> CausalAuditSummary:
    """Summarize held-out behavioral transfer/removal with preregistered gates.

    Margins must be aligned so larger means more of the target behavior.  For
    insertion, ``candidate_margin`` is measured on the base model plus atoms.
    For ablation, it is measured on the post-trained model minus atoms.
    """

    tensors = [base_margin, post_margin, candidate_margin, random_margin]
    n = base_margin.numel()
    if any(t.numel() != n for t in tensors) or target_mask.numel() != n or control_mask.numel() != n:
        raise ValueError("all margin and mask tensors must have the same length")
    target_mask = target_mask.bool().flatten()
    control_mask = control_mask.bool().flatten()
    if not target_mask.any() or not control_mask.any():
        raise ValueError("target and control groups must both be non-empty")

    base = base_margin.float().flatten()
    post = post_margin.float().flatten()
    candidate = candidate_margin.float().flatten()
    random = random_margin.float().flatten()
    reference_raw = post - base
    orientation = 1.0 if reference_raw[target_mask].mean().item() >= 0 else -1.0

    if mode == "insert":
        candidate_raw = candidate - base
        random_raw = random - base
    elif mode == "ablate":
        candidate_raw = post - candidate
        random_raw = post - random
    else:
        raise ValueError(f"unknown mode {mode!r}")

    reference = orientation * reference_raw
    candidate_effects = orientation * candidate_raw
    random_effects = orientation * random_raw
    target_candidate = candidate_effects[target_mask]
    target_random = random_effects[target_mask]
    paired_advantage = target_candidate - target_random
    ci = _bootstrap_mean_ci(
        paired_advantage,
        seed=seed,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
    )

    reference_mean = reference[target_mask].mean().item()
    candidate_mean = target_candidate.mean().item()
    random_mean = target_random.mean().item()
    control_mean = candidate_effects[control_mask].abs().mean().item()
    transfer_fraction = candidate_mean / max(abs(reference_mean), 1e-12)
    specificity_ratio = abs(candidate_mean) / max(control_mean, 1e-12)

    return CausalAuditSummary(
        reference_effect=reference_mean,
        candidate_effect=candidate_mean,
        random_effect=random_mean,
        candidate_minus_random_ci=ci,
        transfer_fraction=transfer_fraction,
        control_effect=control_mean,
        specificity_ratio=specificity_ratio,
        passes_direction=candidate_mean > 0,
        passes_random_control=ci[0] > 0,
        passes_specificity=control_mean <= max_control_fraction * abs(candidate_mean),
    )


def summarize_single_checkpoint_ablation(
    *,
    baseline_margin: Tensor,
    candidate_margin: Tensor,
    random_margin: Tensor,
    static_margin: Tensor,
    target_mask: Tensor,
    control_mask: Tensor,
    seed: int = 0,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    max_control_fraction: float = 0.5,
) -> SingleCheckpointAblationSummary:
    """Summarize necessity of full-weight atoms in one trained checkpoint.

    Larger margins must mean more of the target behavior. Effects are the
    reduction in margin caused by ablation. This deliberately makes no claim
    about which training stage created the atoms.
    """

    tensors = [baseline_margin, candidate_margin, random_margin, static_margin]
    n = baseline_margin.numel()
    if any(t.numel() != n for t in tensors) or target_mask.numel() != n or control_mask.numel() != n:
        raise ValueError("all margin and mask tensors must have the same length")
    target_mask = target_mask.bool().flatten()
    control_mask = control_mask.bool().flatten()
    if not target_mask.any() or not control_mask.any():
        raise ValueError("target and control groups must both be non-empty")

    baseline = baseline_margin.float().flatten()
    candidate_effects = baseline - candidate_margin.float().flatten()
    random_effects = baseline - random_margin.float().flatten()
    static_effects = baseline - static_margin.float().flatten()

    target_candidate = candidate_effects[target_mask]
    target_random = random_effects[target_mask]
    target_static = static_effects[target_mask]
    random_ci = _bootstrap_mean_ci(
        target_candidate - target_random,
        seed=seed,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
    )
    static_ci = _bootstrap_mean_ci(
        target_candidate - target_static,
        seed=seed + 1,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
    )

    candidate_mean = target_candidate.mean().item()
    random_mean = target_random.mean().item()
    static_mean = target_static.mean().item()
    control_mean = candidate_effects[control_mask].abs().mean().item()
    specificity_ratio = abs(candidate_mean) / max(control_mean, 1e-12)
    return SingleCheckpointAblationSummary(
        candidate_effect=candidate_mean,
        random_effect=random_mean,
        static_effect=static_mean,
        candidate_minus_random_ci=random_ci,
        candidate_minus_static_ci=static_ci,
        control_effect=control_mean,
        specificity_ratio=specificity_ratio,
        passes_direction=candidate_mean > 0,
        passes_random_control=random_ci[0] > 0,
        passes_static_control=static_ci[0] > 0,
        passes_specificity=control_mean <= max_control_fraction * abs(candidate_mean),
    )


__all__ = [
    "CausalAuditSummary",
    "LinearComponentIntervention",
    "LinearDeltaIntervention",
    "SVDAtoms",
    "SingleCheckpointAblationSummary",
    "atoms_from_lowrank",
    "atoms_from_svd",
    "component_perturbation",
    "summarize_causal_audit",
    "summarize_single_checkpoint_ablation",
]
