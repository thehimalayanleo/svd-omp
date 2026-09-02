import torch

from bidirectional_delta_pursuit import (
    exact_svd_atoms_from_lora,
    foba_refine,
    foba_refine_candidates,
    native_lora_atoms,
    omp_select,
    omp_select_candidates,
    reconstruct,
    weighted_objective,
)


def test_exact_and_native_lora_dictionaries_reconstruct_update() -> None:
    generator = torch.Generator().manual_seed(17)
    a = torch.randn(3, 5, generator=generator)
    b = torch.randn(7, 3, generator=generator)
    expected = 2.0 * b @ a
    assert torch.allclose(reconstruct(exact_svd_atoms_from_lora(a, b, 2.0)), expected, atol=1e-5)
    assert torch.allclose(reconstruct(native_lora_atoms(a, b, 2.0)), expected, atol=1e-6)


def test_omp_and_foba_use_fixed_atom_effects() -> None:
    target = torch.tensor([2.0, 2.0, 1.0])
    effects = torch.tensor([
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
    ])
    weights = torch.ones(3)
    support = omp_select(target, effects, weights, budget=2)
    refined = foba_refine(target, effects, weights, support, max_swaps=4)
    assert len(support) == len(refined) == 2
    assert weighted_objective(target, effects, refined, weights) <= weighted_objective(
        target, effects, support, weights
    )


def test_svd_first_pursuit_stays_inside_frozen_pool() -> None:
    target = torch.tensor([2.0, 2.0, 1.0])
    effects = torch.tensor([
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
        [9.0, 9.0, 9.0],
    ])
    weights = torch.ones(3)
    pool = (0, 1, 2, 3)
    restricted = omp_select_candidates(target, effects, weights, pool, budget=2)
    seeded = omp_select_candidates(
        target, effects, weights, pool, budget=2, initial_support=(2,)
    )
    refined = foba_refine_candidates(
        target, effects, weights, support=(0, 1), candidates=pool, max_swaps=4
    )
    assert len(restricted) == len(seeded) == len(refined) == 2
    assert set(restricted) <= set(pool)
    assert set(seeded) <= set(pool) and 2 in seeded
    assert set(refined) <= set(pool)
    assert weighted_objective(target, effects, refined, weights) <= weighted_objective(
        target, effects, (0, 1), weights
    )
