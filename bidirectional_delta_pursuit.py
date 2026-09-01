"""Exact LoRA atomization and fixed-coefficient forward/backward pursuit."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from behavioral_causal_audit import SVDAtoms


def exact_svd_atoms_from_lora(a: Tensor, b: Tensor, scale: float) -> SVDAtoms:
    """Return the exact nonzero SVD atoms of ``scale * b @ a``.

    The calculation only decomposes an ``r by r`` core, avoiding construction
    of the full update matrix.
    """
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
        raise ValueError("expected A [rank,d_in] and B [d_out,rank]")
    qb, rb = torch.linalg.qr(b.float(), mode="reduced")
    qa, ra = torch.linalg.qr(a.float().T, mode="reduced")
    u_core, singular, vh_core = torch.linalg.svd(
        float(scale) * (rb @ ra.T), full_matrices=False
    )
    u = qb @ u_core
    v = qa @ vh_core.T
    return SVDAtoms(
        V=v.contiguous(),
        U_sigma=(u * singular).T.contiguous(),
        S=singular.contiguous(),
    )


def native_lora_atoms(a: Tensor, b: Tensor, scale: float) -> SVDAtoms:
    """Represent the learned LoRA rank-one factors as a matched dictionary."""
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
        raise ValueError("expected A [rank,d_in] and B [d_out,rank]")
    v = a.float().T.contiguous()
    u_sigma = (float(scale) * b.float().T).contiguous()
    sizes = v.norm(dim=0) * u_sigma.norm(dim=1)
    return SVDAtoms(V=v, U_sigma=u_sigma, S=sizes)


def reconstruct(atoms: SVDAtoms) -> Tensor:
    """Reconstruct the represented ``[d_out,d_in]`` update."""
    return atoms.U_sigma.T @ atoms.V.T


def weighted_objective(
    target: Tensor,
    effects: Tensor,
    support: Sequence[int],
    weights: Tensor,
) -> float:
    """Weighted fixed-coefficient residual objective for one support."""
    if target.ndim != 1 or weights.shape != target.shape:
        raise ValueError("target and weights must be aligned vectors")
    if effects.ndim != 2 or effects.shape[1] != target.numel():
        raise ValueError("effects must be [atoms, examples]")
    fitted = torch.zeros_like(target)
    if support:
        fitted = effects[list(support)].sum(dim=0)
    return float((weights * (target - fitted).square()).mean())


def omp_select(target: Tensor, effects: Tensor, weights: Tensor, budget: int) -> tuple[int, ...]:
    """Greedily add fixed-dose atoms that best reduce weighted residual."""
    if not 1 <= budget <= effects.shape[0]:
        raise ValueError("invalid support budget")
    selected: list[int] = []
    residual = target.clone()
    available = torch.ones(effects.shape[0], dtype=torch.bool, device=effects.device)
    for _ in range(budget):
        candidate_residual = residual.unsqueeze(0) - effects
        objectives = (candidate_residual.square() * weights).mean(dim=1)
        objectives = objectives.masked_fill(~available, float("inf"))
        chosen = int(objectives.argmin())
        selected.append(chosen)
        available[chosen] = False
        residual = residual - effects[chosen]
    return tuple(selected)


def foba_refine(
    target: Tensor,
    effects: Tensor,
    weights: Tensor,
    support: Sequence[int],
    *,
    max_swaps: int = 8,
    minimum_improvement: float = 1e-10,
) -> tuple[int, ...]:
    """Run exact add-one/remove-one swaps under the residual objective."""
    current = tuple(dict.fromkeys(int(index) for index in support))
    if not current:
        raise ValueError("support cannot be empty")
    if len(current) != len(support):
        raise ValueError("support contains duplicates")
    current_objective = weighted_objective(target, effects, current, weights)
    all_indices = set(range(effects.shape[0]))
    for _ in range(max_swaps):
        current_set = set(current)
        excluded = tuple(sorted(all_indices - current_set))
        if not excluded:
            break
        fitted = effects[list(current)].sum(dim=0)
        residuals = (
            target[None, None, :]
            - fitted[None, None, :]
            - effects[list(excluded)][:, None, :]
            + effects[list(current)][None, :, :]
        )
        objectives = (residuals.square() * weights[None, None, :]).mean(dim=2)
        flat_index = int(objectives.argmin())
        removed_position = flat_index % len(current)
        added_position = flat_index // len(current)
        best_objective = float(objectives[added_position, removed_position])
        if best_objective >= current_objective - minimum_improvement:
            break
        updated = list(current)
        updated[removed_position] = excluded[added_position]
        current = tuple(updated)
        current_objective = best_objective
    return current


def paired_weights(rows: Sequence[dict], copies: int = 2) -> Tensor:
    """Upweight the target and its paired control for bidirectional pursuit."""
    if copies < 1:
        raise ValueError("copies must be positive")
    local = torch.tensor([
        4.0 if row["family"] in {"marker_target", "marker_control"} else 1.0
        for row in rows
    ])
    return local.repeat(copies)


__all__ = [
    "exact_svd_atoms_from_lora",
    "native_lora_atoms",
    "reconstruct",
    "weighted_objective",
    "omp_select",
    "foba_refine",
    "paired_weights",
]
