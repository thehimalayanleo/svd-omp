"""Pure helpers for the matched FoBa layer-selection experiment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import random


PROTECTED_FAMILIES = ("clean", "quoted_attack", "ambiguous")


def top_layer_support(
    scores: Mapping[str, float], budget: int
) -> tuple[str, ...]:
    """Return a deterministic highest-score layer support."""

    if not 1 <= budget <= len(scores):
        raise ValueError("budget must be between one and the number of layers")
    if any(value != value for value in scores.values()):
        raise ValueError("layer scores must not contain NaN")
    ranked = sorted(scores, key=lambda name: (-float(scores[name]), name))
    return tuple(ranked[:budget])


def random_layer_supports(
    layer_names: Sequence[str],
    budget: int,
    *,
    seed: int,
    count: int = 19,
    excluded: Sequence[Sequence[str]] = (),
) -> tuple[tuple[str, ...], ...]:
    """Draw unique, deterministic, matched-cardinality layer supports."""

    names = tuple(sorted(set(layer_names)))
    if len(names) != len(layer_names):
        raise ValueError("layer names must be unique")
    if not 1 <= budget <= len(names):
        raise ValueError("invalid layer budget")
    if count < 1:
        raise ValueError("count must be positive")
    blocked = {tuple(sorted(support)) for support in excluded}
    generator = random.Random(seed)
    result: list[tuple[str, ...]] = []
    seen = set(blocked)
    max_attempts = 10000
    for _ in range(max_attempts):
        support = tuple(sorted(generator.sample(names, budget)))
        if support in seen:
            continue
        seen.add(support)
        result.append(support)
        if len(result) == count:
            return tuple(result)
    raise ValueError("could not draw the requested number of unique supports")


def validation_summary(method: Mapping) -> dict[str, float | int | bool]:
    """Extract the decision-level values used by the frozen gate."""

    metrics = method["validation_three_way"]
    target = metrics["benign_warning"]
    protected = {family: metrics[family]["accuracy"] for family in PROTECTED_FAMILIES}
    return {
        "target_correct": int(target["correct"]),
        "target_total": int(target["total"]),
        "minimum_protected_accuracy": min(protected.values()),
        "protected_pass": all(value >= 0.90 for value in protected.values()),
    }


def evaluate_development_seed_gate(
    methods: Mapping[str, Mapping],
    *,
    random_prefix: str = "random_layers_",
) -> dict:
    """Evaluate the preregistered per-seed FoBa selection gate."""

    required = ("foba_layers", "energy_layers", "gradient_layers")
    missing = [name for name in required if name not in methods]
    if missing:
        raise ValueError(f"missing required methods: {missing}")
    random_names = sorted(name for name in methods if name.startswith(random_prefix))
    if len(random_names) != 19:
        raise ValueError("the frozen gate requires exactly 19 random supports")

    summaries = {name: validation_summary(value) for name, value in methods.items()}
    foba = summaries["foba_layers"]
    foba_target = int(foba["target_correct"])
    comparator_targets = {
        "energy_layers": int(summaries["energy_layers"]["target_correct"]),
        "gradient_layers": int(summaries["gradient_layers"]["target_correct"]),
        **{name: int(summaries[name]["target_correct"]) for name in random_names},
    }
    passes = {
        "positive_target_repair": foba_target > 0,
        "beats_energy": foba_target > comparator_targets["energy_layers"],
        "beats_gradient": foba_target > comparator_targets["gradient_layers"],
        "beats_every_random": all(
            foba_target > comparator_targets[name] for name in random_names
        ),
        "protected_families": bool(foba["protected_pass"]),
    }
    return {
        "passes": all(passes.values()),
        "criteria": passes,
        "foba": foba,
        "comparator_target_correct": comparator_targets,
        "random_exceedance_count": sum(
            comparator_targets[name] >= foba_target for name in random_names
        ),
        "randomization_p_upper": (
            1
            + sum(comparator_targets[name] >= foba_target for name in random_names)
        )
        / 20,
    }


def evaluate_two_seed_gate(seed_results: Mapping[int, Mapping]) -> dict:
    """Require both frozen development seeds to pass independently."""

    if set(seed_results) != {313, 317}:
        raise ValueError("the frozen development gate requires seeds 313 and 317")
    per_seed = {
        str(seed): evaluate_development_seed_gate(result["methods"])
        for seed, result in sorted(seed_results.items())
    }
    return {"passes": all(value["passes"] for value in per_seed.values()), "seeds": per_seed}
