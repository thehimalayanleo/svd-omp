"""Distributionally robust forward-backward support search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Hashable, Mapping, Sequence


Candidate = Hashable


@dataclass(frozen=True)
class RobustPoint:
    dose: float
    target_by_distribution: Mapping[str, int]
    protected_by_distribution: Mapping[str, Mapping[str, int]]

    def feasible(self, minimum_protected: int = 22) -> bool:
        return all(
            count >= minimum_protected
            for local in self.protected_by_distribution.values()
            for count in local.values()
        )

    def objective(self) -> tuple[int, int, int, float]:
        targets = tuple(self.target_by_distribution.values())
        protected = tuple(
            count
            for local in self.protected_by_distribution.values()
            for count in local.values()
        )
        return min(targets), sum(targets), min(protected), -float(self.dose)

    def to_dict(self) -> dict:
        return {
            "dose": self.dose,
            "target_by_distribution": dict(self.target_by_distribution),
            "protected_by_distribution": {
                name: dict(local)
                for name, local in self.protected_by_distribution.items()
            },
            "feasible": self.feasible(),
            "objective": list(self.objective()),
        }


def choose_robust_dose(
    points: Mapping[float, RobustPoint], minimum_protected: int = 22
) -> RobustPoint:
    feasible = [point for point in points.values() if point.feasible(minimum_protected)]
    if not feasible:
        raise ValueError("no dose satisfies the protected-family floor")
    return max(feasible, key=lambda point: point.objective())


def robust_foba(
    candidates: Sequence[Candidate],
    evaluate: Callable[[frozenset[Candidate]], RobustPoint],
    *,
    maximum_size: int,
) -> dict:
    """Greedy forward search with lossless backward pruning.

    The lexicographic objective first maximizes the worst-distribution target
    repair, then total repair, then the weakest protected count. Backward
    removal is allowed only when it does not reduce that objective.
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
    current = measured(selected)
    trace = [{"action": "start", "support": [], "point": current.to_dict()}]
    visited = {selected}

    while len(selected) < maximum_size:
        options = []
        for candidate in ordered:
            if candidate in selected:
                continue
            support = selected | {candidate}
            if support in visited:
                continue
            point = measured(support)
            if point.feasible():
                options.append((candidate, support, point))
        if not options:
            break
        candidate, support, point = max(
            options,
            key=lambda item: (item[2].objective(), -ordered.index(item[0])),
        )
        if point.objective() <= current.objective():
            break
        selected, current = support, point
        visited.add(selected)
        trace.append({
            "action": "add",
            "candidate": candidate,
            "support": [item for item in ordered if item in selected],
            "point": current.to_dict(),
        })

        while len(selected) > 1:
            removals = []
            for candidate in selected:
                support = selected - {candidate}
                if support in visited:
                    continue
                candidate_point = measured(support)
                if candidate_point.feasible() and candidate_point.objective() >= current.objective():
                    removals.append((candidate, support, candidate_point))
            if not removals:
                break
            candidate, support, point = max(
                removals,
                key=lambda item: (item[2].objective(), ordered.index(item[0])),
            )
            selected, current = support, point
            visited.add(selected)
            trace.append({
                "action": "remove",
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


__all__ = ["RobustPoint", "choose_robust_dose", "robust_foba"]
