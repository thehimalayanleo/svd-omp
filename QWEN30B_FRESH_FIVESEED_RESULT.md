# Qwen3-30B fresh five-seed replication result

## Verdict

**The frozen campaign passed 5/5 complete seeds, above the required 4/5 gate.**

This is the first prospective Qwen3-30B campaign in this project whose behavioral and numerical gates both pass without changing the protocol after outcomes were observed. It uses five fresh LoRA organisms, 92 unused source questions, a fixed 272-of-768 atom budget, a sealed confirmation split, and a separately implemented float32 unmerged endpoint check.

The supported claim is:

> A 272-of-768 exact sub-update replicated as a sufficient, necessary, and behaviorally specific cause of the same controlled regression across five fresh Qwen3-30B organisms and unused confirmation sources.

## Frozen design

- Model: `Qwen/Qwen3-30B-A3B-Instruct-2507`, 30,532,122,624 parameters.
- Model revision: `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`.
- Training seeds: 947, 953, 967, 971, and 977. Every seed remains in the denominator.
- Update: rank-16 LoRA on all 48 attention output projections.
- Exact dictionary: 768 rank-one SVD atoms.
- Frozen budget: 272 atoms, or 35.4% of the dictionary.
- Primary selector: 64 weighted-OMP atoms, eight FoBa swaps, then 208 descending-singular-value atoms.
- Source partitions: 36 train, 16 organism validation, 12 selection, 12 causal validation, and 16 confirmation sources.
- Prior-campaign overlap: zero sources.
- Confirmation opening rule: at least 4/5 supports must pass causal validation.
- Campaign rule: at least 4/5 seeds must pass insertion, ablation, preservation, paired-control, and float32 endpoint gates.

The protocol hash is `49cf051cba4462e43dbe526cf81f48c281aa68113a53f0f855917c4113a5200a`. It and the sealed datasets were committed before training in commit `b42bdb7`. The executable runners were committed before outcomes in commit `389f848`.

## Gate-by-gate outcome

| Seed | Organism admitted | Selection valid | Validation | Confirmation | Protected minimum | Pair damage | Float32 endpoints | Full verdict |
|---:|:---:|:---:|---:|---:|---:|---:|:---:|:---:|
| 947 | yes | yes | 12/12 | 16/16 | 16/16 | 0 | exact predictions | pass |
| 953 | yes | yes | 12/12 | 16/16 | 16/16 | 0 | exact predictions | pass |
| 967 | yes | yes | 12/12 | 16/16 | 16/16 | 0 | exact predictions | pass |
| 971 | yes | yes | 12/12 | 16/16 | 16/16 | 0 | exact predictions | pass |
| 977 | yes | yes | 12/12 | 16/16 | 16/16 | 0 | exact predictions | pass |
| **Total** | **5/5** | **5/5** | **60/60** | **80/80** | **all perfect** | **0** | **5/5** | **5/5** |

Every selected sub-update worked in both directions:

1. Adding it to the base model recreated the trained ordering-marker error.
2. Subtracting the identical coefficient-one sub-update from the trained model repaired the error.
3. The matched same-source control and every protected family stayed correct.

The full 768-atom dictionary also reproduced each endpoint's predictions exactly in both directions in float32 without merging the adapter. Maximum relative SVD reconstruction error was below `9.7e-7` for every seed. Logit margins were not required to be bit-identical; the prospective endpoint gate was exact prediction agreement.

## Equal-budget selector comparison

| Frozen method | Atoms | Sealed bidirectional outcomes |
|---|---:|---:|
| FoBa64 + SVD208 | 272 | 80/80 |
| OMP64 + SVD208 | 272 | 80/80 |
| Top-SVD | 272 | 80/80 |
| Cross-seed consensus | 272 | 80/80 |
| Gradient rank | 272 | 0/80 |
| Direct OMP-272 | 272 | 0/80 |

This is not a FoBa or OMP superiority result. FoBa made no behavioral improvement over OMP+SVD, and both tied simple top-SVD. The sharp result is instead that supports retaining enough spectral mass were perfectly causal here, while two first-order pursuit or ranking constructions completely failed despite the same budget.

## Random-support control

For each seed, the primary 272-atom support was compared against 999 unique same-size random supports. Zero of 999 random supports matched the selected feasible score on every seed, giving the minimum pre-registered one-sided empirical p-value of 0.001 per seed.

This rejects the explanation that almost any 35.4%-sized subset would work. It does not distinguish FoBa+SVD from top-SVD, because both deterministic methods reached 80/80.

## What changed relative to the earlier Qwen result

The earlier Qwen campaign had 48/48 behavioral effects but failed a merged-BF16 full-dictionary prediction check by one row on two seeds. Its later float32 diagnostic could diagnose that failure but could not rewrite the original protocol.

This campaign made float32, unmerged, full-dictionary prediction closure prospective. It then trained five new organisms, used only unused sources, froze all splits and the 272-atom selector before training, and passed the complete rule 5/5. The old failed campaign remains failed; this is a new successful replication.

## Claim boundary

The result supports fresh-seed, fresh-source, and cross-model-family replication of a controlled exact-update causal effect. It does not establish:

- a natural-checkpoint repair method;
- a human-interpretable meaning for individual atoms;
- an ultra-sparse circuit, since 272/768 atoms is 35.4%;
- generalization across arbitrary learned behaviors;
- FoBa or OMP superiority;
- superiority to learned model-diffing methods such as Delta-Crosscoder.

## Reproducibility

- Frozen protocol: `QWEN30B_FRESH_FIVESEED_PROTOCOL.md`
- Dataset manifest: `data/behavior_audit/qwen30b_fresh_fiveseed_manifest.json`
- Training runner: `modal_train_qwen30b_fresh_fiveseed.py`
- Causal runner: `modal_qwen30b_fresh_fiveseed.py`
- Numerical runner: `modal_qwen30b_fresh_fiveseed_numeric.py`
- Finalizer: `finalize_qwen30b_fresh_fiveseed.py`
- Machine-readable verdict: `results/behavioral_causal_audit/qwen30b_fresh_fiveseed_final_summary.json`
- Training Modal run: `ap-TC2nM0G2uWiOPmPOu5pH0Y`
- Selection Modal run: `ap-0IbWIONR4DCbpirnhP8EK2`
- Validation Modal run: `ap-vNgRy5GTPyUPFGXlzguQho`
- Retained confirmation Modal run: `ap-XzdJEUepBC3MI2NVmJUjPT`

The first confirmation execution, `ap-SJUdseoyYL2IseFZDIepji`, completed remote evaluation but its local collector failed on a missing selection-only field. Commit `b1c073d` fixed and tested only that collector condition. The retained rerun used identical model, data, supports, thresholds, and deterministic random seeds.
