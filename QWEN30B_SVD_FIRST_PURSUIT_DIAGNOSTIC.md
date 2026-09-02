# Frozen Qwen3-30B SVD-First Pursuit Diagnostic

Status: frozen before remote execution on 2026-09-02.

## Question

After exact SVD atomization of a learned LoRA update, does behavior-guided
pursuit work better when it searches only a stable high-magnitude spectral
region? The primary comparison is equal-budget Top-SVD versus SVD-restricted
OMP and SVD-started FoBa.

This is a development diagnostic. It uses the already-open selection split and
cannot upgrade the paper's sealed confirmation claim by itself. Confirmation
data must not be mounted.

## Frozen inputs

- Model: `Qwen/Qwen3-30B-A3B-Instruct-2507`
- Model revision: `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`
- Organism tag: `qwen30b_position_bias_v2_fresh_rank16`
- Organism seeds: 947, 953, 967, 971, 977. Every admitted seed remains in the denominator.
- Modules: all 48 attention output projections.
- Rank: 16 atoms per module, for 768 exact rank-one SVD atoms total.
- Data: `qwen30b_fresh_fiveseed_selection.jsonl`, 96 rows.
- Data SHA-256: `53761642f0801782e0ee4080960a893fc031f39f5ab49ea20ba22d3051b8abde`
- Final support budgets: 64, 96, and 128 atoms.
- SVD candidate pool: the 192 atoms with the largest singular values.
- SVD seed: the 32 atoms with the largest singular values.
- FoBa swaps: at most 8.
- Selector objective: the existing paired, weighted fixed-dose margin residual.

The 192-atom pool was fixed before this run. It was chosen because the earlier
opened development curve showed that the spectral family had saturated by 192
atoms, while 64 and 96 remain below that ceiling and can expose selector
differences.

## Frozen matched methods

At each final budget k:

1. `top_svd`: take the k largest singular-value atoms.
2. `svd192_omp`: restrict the dictionary to Top-SVD-192, then run OMP from an empty support until k atoms are selected.
3. `svd32_omp`: lock Top-SVD-32, then use OMP to add atoms from the Top-SVD-192 pool until k atoms are selected.
4. `svd192_foba8`: start from Top-SVD-k, then allow up to eight improving one-for-one swaps within Top-SVD-192.
5. `direct_omp`: run OMP on all 768 atoms.
6. `omp64_svd`: select 64 atoms by OMP on all 768, then fill by singular value.
7. `foba64_svd`: refine the 64-atom OMP prefix with eight swaps on all 768, then fill by singular value.

All supports have exactly k distinct atoms. Atom coefficients remain their
learned fixed SVD coefficients. There is no coefficient refit.

## Frozen measurements

For every method, budget, and seed:

- insertion into the base model;
- ablation from the organism;
- paired bidirectional repairs;
- protected-family minima and pair damage;
- weighted selector objective;
- exact selected support.

The input-validity gate must pass before selector evaluation. Any stopped or
failed seed stays in the denominator.

## Interpretation gate

The reversed ordering is promising only if an SVD-first method:

1. has strictly more pooled bidirectional repairs than `top_svd` at 64 or 96 atoms;
2. remains feasible on at least four of five seeds at that budget;
3. does not increase pooled insertion or ablation pair damage; and
4. is not merely tied after Top-SVD has already reached the ceiling.

If it only lowers the first-order objective without improving exact behavior,
report that mismatch as a negative result. If it ties Top-SVD, report parity,
not superiority. No result from this opened split is called a fresh replication.
