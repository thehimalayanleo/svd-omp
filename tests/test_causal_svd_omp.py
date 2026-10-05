from __future__ import annotations

import itertools

import torch

from causal_svd_omp import causal_whitened_svd, select_top_k, sensitivity_metric
from whitened_svd_omp_discovery import activation_whitened_svd


def loud_quiet_layer(seed: int = 0):
    """W has 4 loud and 4 quiet atoms; the next layer reads only the quiet band."""

    generator = torch.Generator().manual_seed(seed)
    d_in, d_out, d_next = 24, 16, 12
    left = torch.linalg.qr(torch.randn(d_out, 8, generator=generator)).Q
    right = torch.linalg.qr(torch.randn(d_in, 8, generator=generator)).Q
    sigma = torch.tensor([10.0] * 4 + [2.0] * 4)
    weight = (left * sigma).matmul(right.T)
    next_weight = torch.randn(d_next, 4, generator=generator).matmul(left[:, 4:].T)
    calibration = torch.randn(512, d_in, generator=generator)
    heldout = torch.randn(256, d_in, generator=generator)
    return weight, next_weight, calibration, heldout


def test_identity_metric_recovers_whitened_svd() -> None:
    torch.manual_seed(3)
    weight = torch.randn(9, 6)
    calibration = torch.randn(40, 6)
    write, values, read, diagnostics = causal_whitened_svd(
        weight, calibration, None, alpha=1e-2, beta=0.0
    )
    reference_values, reference_read, _ = activation_whitened_svd(
        weight, calibration, alpha=1e-2, device="cpu"
    )
    assert diagnostics["weight_reconstruction_relative_error"] < 1e-5
    assert torch.allclose(values, reference_values, atol=1e-5)
    assert torch.allclose(read.abs(), reference_read.abs(), atol=1e-4)
    assert torch.allclose(write.T.matmul(write), torch.eye(6), atol=1e-5)


def test_factorization_is_exact_and_write_vectors_are_metric_orthonormal() -> None:
    torch.manual_seed(5)
    weight = torch.randn(10, 7)
    calibration = torch.randn(64, 7)
    metric = sensitivity_metric(torch.randn(200, 10))
    write, _, _, diagnostics = causal_whitened_svd(
        weight, calibration, metric, alpha=1e-2, beta=1e-3
    )
    assert diagnostics["weight_reconstruction_relative_error"] < 1e-5
    regularized = metric + diagnostics["output_regularization"] * torch.eye(10)
    assert torch.allclose(
        write.T.matmul(regularized).matmul(write), torch.eye(7), atol=1e-4
    )


def test_top_k_is_the_exact_optimum_under_the_metric() -> None:
    torch.manual_seed(9)
    weight = torch.randn(6, 5)
    calibration = torch.randn(64, 5)
    metric = sensitivity_metric(torch.randn(100, 6))
    write, values, read, diagnostics = causal_whitened_svd(
        weight, calibration, metric, alpha=1e-2, beta=1e-3
    )
    regularized = metric + diagnostics["output_regularization"] * torch.eye(6)
    phi = torch.randn(1, 5)
    target = phi.matmul(weight.T)
    coefficients = phi.matmul(read) * values

    def weighted_error(support: tuple[int, ...]) -> float:
        mask = torch.zeros(5)
        mask[list(support)] = 1.0
        error = target - (coefficients * mask).matmul(write.T)
        return float(error.matmul(regularized).matmul(error.T))

    _, chosen = select_top_k(phi, write, values, read, k=2)
    best = min(itertools.combinations(range(5), 2), key=weighted_error)
    assert set(chosen[0].tolist()) == set(best)


def test_causal_metric_recovers_downstream_signal_the_local_score_misses() -> None:
    weight, next_weight, calibration, heldout = loud_quiet_layer()
    metric = next_weight.T.matmul(next_weight)
    layer_target = heldout.matmul(weight.T)
    target = layer_target.matmul(next_weight.T)

    def errors(output_metric, beta):
        write, values, read, _ = causal_whitened_svd(
            weight, calibration, output_metric, alpha=1e-3, beta=beta
        )
        approximation, _ = select_top_k(heldout, write, values, read, k=4)
        downstream = (approximation.matmul(next_weight.T) - target).square().sum()
        layer = (approximation - layer_target).square().sum()
        return (
            float(downstream / target.square().sum()),
            float(layer / layer_target.square().sum()),
        )

    local_downstream, local_layer = errors(None, 0.0)
    causal_downstream, causal_layer = errors(metric, 1e-4)
    # The local score spends its four units on the loud band the next layer
    # ignores; the causal metric recovers the downstream signal exactly.
    assert local_downstream > 0.4
    assert causal_downstream < 1e-3
    # The price is layer-local fidelity: the metric does not care about it.
    assert causal_layer > local_layer

if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_"):
            function()
            print(f"ok  {name}")
