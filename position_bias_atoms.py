"""Pure source-paired helpers for the Phi position-bias campaign."""

from __future__ import annotations

from collections.abc import Sequence

from paired_atom_foba import decode_atom, encode_atom


def paired_gradient_score(effects: Sequence[float], rows: Sequence[dict]) -> float:
    """Score an atom by target-ablation benefit minus protected effects."""
    if len(effects) != len(rows) or not rows:
        raise ValueError("effects and rows must be nonempty and aligned")
    target = [effect for effect, row in zip(effects, rows) if row["family"] == "marker_target"]
    paired = [abs(effect) for effect, row in zip(effects, rows) if row["family"] == "marker_control"]
    other = [
        abs(effect) for effect, row in zip(effects, rows)
        if row["family"] not in {"marker_target", "marker_control"}
    ]
    if not target or not paired or not other:
        raise ValueError("missing target or protected family")
    return sum(target) / len(target) - sum(paired) / len(paired) - 0.25 * sum(other) / len(other)


def specific_repair_sources(
    target_newly_correct_ids: Sequence[str], paired_correct_ids: Sequence[str]
) -> set[str]:
    repaired = {item.removeprefix("marker_target:") for item in target_newly_correct_ids}
    preserved = {item.removeprefix("marker_control:") for item in paired_correct_ids}
    return repaired & preserved


__all__ = ["decode_atom", "encode_atom", "paired_gradient_score", "specific_repair_sources"]
