"""Pure helpers for source-paired SVD-atom selection."""

from __future__ import annotations

from collections.abc import Sequence


SEPARATOR = "::component="


def encode_atom(module: str, component: int) -> str:
    if component < 0:
        raise ValueError("component must be nonnegative")
    return f"{module}{SEPARATOR}{component}"


def decode_atom(atom: str) -> tuple[str, int]:
    module, component = atom.rsplit(SEPARATOR, 1)
    if not module:
        raise ValueError("atom is missing its module")
    return module, int(component)


def specific_repair_sources(
    target_newly_correct_ids: Sequence[str], paired_correct_ids: Sequence[str]
) -> set[str]:
    repaired = {item.removeprefix("benign_marker:") for item in target_newly_correct_ids}
    preserved = {item.removeprefix("marked_ambiguous:") for item in paired_correct_ids}
    return repaired & preserved


def paired_gradient_score(effects: Sequence[float], rows: Sequence[dict]) -> float:
    """Rank atoms by target ablation benefit minus protected collateral.

    ``effects`` are first-order changes in the positive-minus-negative margin
    from adding the learned atom. Subtracting a positive-effect atom lowers the
    organism's U-over-correct target margin, so positive target effects help.
    Absolute protected effects are penalized conservatively.
    """
    if len(effects) != len(rows) or not rows:
        raise ValueError("effects and rows must be nonempty and aligned")
    target = [effect for effect, row in zip(effects, rows) if row["family"] == "benign_marker"]
    paired = [abs(effect) for effect, row in zip(effects, rows) if row["family"] == "marked_ambiguous"]
    other = [
        abs(effect) for effect, row in zip(effects, rows)
        if row["family"] not in {"benign_marker", "marked_ambiguous"}
    ]
    if not target or not paired or not other:
        raise ValueError("missing target or protected families")
    return sum(target) / len(target) - sum(paired) / len(paired) - 0.25 * sum(other) / len(other)


__all__ = ["decode_atom", "encode_atom", "paired_gradient_score", "specific_repair_sources"]
