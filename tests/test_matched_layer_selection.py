from matched_layer_selection import (
    evaluate_development_seed_gate,
    random_layer_supports,
    top_layer_support,
)


def _method(target, protected=(0.95, 0.96, 1.0)):
    families = ("clean", "quoted_attack", "ambiguous")
    return {
        "validation_three_way": {
            **{
                family: {"accuracy": value, "correct": round(24 * value), "total": 24}
                for family, value in zip(families, protected)
            },
            "benign_warning": {"accuracy": target / 24, "correct": target, "total": 24},
        }
    }


def test_top_layer_support_is_deterministic_on_ties():
    assert top_layer_support({"b": 2.0, "a": 2.0, "c": 1.0}, 2) == ("a", "b")


def test_random_supports_are_unique_matched_and_exclude_known_supports():
    layers = [f"layer.{index}" for index in range(12)]
    blocked = (("layer.0", "layer.1", "layer.2"),)
    first = random_layer_supports(layers, 3, seed=7, excluded=blocked)
    second = random_layer_supports(layers, 3, seed=7, excluded=blocked)
    assert first == second
    assert len(first) == len(set(first)) == 19
    assert all(len(support) == 3 for support in first)
    assert tuple(sorted(blocked[0])) not in first


def test_gate_requires_strict_win_over_every_comparator():
    methods = {
        "foba_layers": _method(12),
        "energy_layers": _method(10),
        "gradient_layers": _method(11),
        **{f"random_layers_{index:02d}": _method(index % 7) for index in range(19)},
    }
    result = evaluate_development_seed_gate(methods)
    assert result["passes"]
    assert result["randomization_p_upper"] == 0.05

    methods["gradient_layers"] = _method(12)
    result = evaluate_development_seed_gate(methods)
    assert not result["passes"]
    assert not result["criteria"]["beats_gradient"]


def test_gate_fails_protected_behavior_even_with_target_win():
    methods = {
        "foba_layers": _method(12, protected=(0.95, 0.89, 1.0)),
        "energy_layers": _method(1),
        "gradient_layers": _method(1),
        **{f"random_layers_{index:02d}": _method(0) for index in range(19)},
    }
    result = evaluate_development_seed_gate(methods)
    assert not result["passes"]
    assert not result["criteria"]["protected_families"]
