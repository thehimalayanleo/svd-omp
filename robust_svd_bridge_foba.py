"""Constraint-aware FoBa that can traverse infeasible zero-gain bridge supports."""

from __future__ import annotations

from typing import Callable, Hashable, Sequence

from robust_svd_foba import RobustPoint


Candidate = Hashable


def violation(point: RobustPoint, minimum_protected: int = 22) -> int:
    return sum(
        max(0, minimum_protected - count)
        for local in point.protected_by_distribution.values()
        for count in local.values()
    )


def bridge_key(point: RobustPoint) -> tuple[int, int, int, int, float]:
    targets = tuple(point.target_by_distribution.values())
    protected = tuple(
        count
        for local in point.protected_by_distribution.values()
        for count in local.values()
    )
    return min(targets), sum(targets), -violation(point), min(protected), -point.dose


def bridge_foba(
    candidates: Sequence[Candidate],
    evaluate: Callable[[frozenset[Candidate]], RobustPoint],
    *,
    maximum_size: int,
) -> dict:
    """Follow the strongest bridge path, retaining the best feasible support.

    Strict constrained greedy search cannot start when every useful singleton
    violates a protected floor by one item. This variant continues forward
    using worst-distribution repair, total repair, and constraint violation as
    bridge criteria. The reported support must still be fully feasible. A final
    backward pass removes any redundant layer without weakening the robust
    objective.
    """

    ordered = list(dict.fromkeys(candidates))
    if not ordered or not 1 <= maximum_size <= len(ordered):
        raise ValueError("invalid candidates or maximum_size")
    cache: dict[frozenset[Candidate], RobustPoint] = {}

    def measured(support: frozenset[Candidate]) -> RobustPoint:
        if support not in cache:
            cache[support] = evaluate(support)
        return cache[support]

    selected: frozenset[Candidate] = frozenset()
    start = measured(selected)
    best_support = selected if start.feasible() else None
    best_point = start if start.feasible() else None
    trace = [{"action": "start", "support": [], "point": start.to_dict()}]

    for _depth in range(maximum_size):
        options = []
        for candidate in ordered:
            if candidate in selected:
                continue
            support = selected | {candidate}
            point = measured(support)
            options.append((candidate, support, point))
            if point.feasible() and (best_point is None or point.objective() > best_point.objective()):
                best_support, best_point = support, point
        if not options:
            break
        candidate, selected, current = max(
            options,
            key=lambda item: (bridge_key(item[2]), -ordered.index(item[0])),
        )
        trace.append({
            "action": "bridge_add",
            "candidate": candidate,
            "support": [item for item in ordered if item in selected],
            "point": current.to_dict(),
            "violation": violation(current),
        })

    if best_support is None or best_point is None:
        raise ValueError("no feasible support found along the bridge path")

    selected, current = best_support, best_point
    while len(selected) > 1:
        removals = []
        for candidate in selected:
            support = selected - {candidate}
            point = measured(support)
            if point.feasible() and point.objective() >= current.objective():
                removals.append((candidate, support, point))
        if not removals:
            break
        candidate, selected, current = max(
            removals,
            key=lambda item: (item[2].objective(), ordered.index(item[0])),
        )
        trace.append({
            "action": "backward_remove",
            "candidate": candidate,
            "support": [item for item in ordered if item in selected],
            "point": current.to_dict(),
        })

    return {
        "selected": [item for item in ordered if item in selected],
        "point": current.to_dict(),
        "evaluated_supports": len(cache),
        "trace": trace,
    }


__all__ = ["bridge_foba", "bridge_key", "violation"]
