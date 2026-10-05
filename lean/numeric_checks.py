"""Numerical checks of T1 predictions P1-P5 on random small cases (run before proving)."""
import itertools
import numpy as np

rng = np.random.default_rng(0)
worst = {}

def rec(name, v):
    worst[name] = max(worst.get(name, 0.0), float(v))

for trial in range(200):
    d, r = rng.integers(2, 7), None
    r = rng.integers(1, d + 1)
    W_ = np.linalg.qr(rng.normal(size=(d, d)))[0][:, :r]          # orthonormal columns w_c
    a = rng.normal(size=r)
    y = W_ @ a
    # P1: error identity for every support and random coefficients, and top-k optimality
    for k in range(r + 1):
        best = min(np.sum(a[[c for c in range(r) if c not in S]] ** 2)
                   for S in itertools.combinations(range(r), k))
        top = np.argsort(-np.abs(a))[:k]
        rec("P1 top-k gap", abs(best - np.sum(np.delete(a, top) ** 2)))
        for S in itertools.combinations(range(r), k):
            S = list(S)
            x = rng.normal(size=len(S))
            err = np.sum((y - W_[:, S] @ x) ** 2)
            pred = np.sum((a[S] - x) ** 2) + np.sum(np.delete(a, S) ** 2)
            rec("P1 identity", abs(err - pred))
    # P2, P3: weighted SVD reconstruction and metric orthonormality
    m, n = rng.integers(2, 6), rng.integers(2, 6)
    W = rng.normal(size=(m, n))
    Lin = np.linalg.cholesky(np.cov(rng.normal(size=(n, 50))) + 0.1 * np.eye(n))
    Lout = np.linalg.cholesky(np.cov(rng.normal(size=(m, 50))) + 0.1 * np.eye(m))
    P, s, Qt = np.linalg.svd(Lout.T @ W @ Lin, full_matrices=False)
    read = np.linalg.inv(Lin).T @ Qt.T
    write = np.linalg.inv(Lout).T @ P
    rec("P2 reconstruction", np.abs(write @ np.diag(s) @ read.T - W).max())
    M = Lout @ Lout.T
    rec("P3 M-orthonormal", np.abs(write.T @ M @ write - np.eye(len(s))).max())
    # P4: weighted error identity with sample Gram
    N = 30
    Phi = rng.normal(size=(N, n))
    G = Phi.T @ Phi / N
    L = np.linalg.cholesky(G + 1e-12 * np.eye(n)) if np.linalg.matrix_rank(G) == n else None
    if L is not None:
        E = rng.normal(size=(m, n))
        lhs = np.mean([p @ E.T @ M @ E @ p for p in Phi])
        rec("P4 weighted error", abs(lhs - np.sum((Lout.T @ E @ L) ** 2)))
    # P5: Fisher identity for softmax
    z = rng.normal(size=m)
    p = np.exp(z - z.max()); p /= p.sum()
    F = sum(p[y_] * np.outer(np.eye(m)[y_] - p, np.eye(m)[y_] - p) for y_ in range(m))
    rec("P5 Fisher", np.abs(F - (np.diag(p) - np.outer(p, p))).max())

for k, v in worst.items():
    print(f"{k:22s} max abs error {v:.2e}  {'OK' if v < 1e-8 else 'FAIL'}")
