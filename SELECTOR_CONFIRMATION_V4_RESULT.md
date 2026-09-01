# Matched Static-SVD Selector Confirmation V4 Result

Status: `selector_superiority_failed_warned_ambiguity_control_decisive`

The matched fourth-set comparison rejected robust FoBa superiority and showed
that the new warning-plus-ambiguity control was decisive in this audit.

The first sealed runner stopped on development data because strict greedy FoBa
could not find a protected-feasible singleton. It never scored the fourth set.
V2 changed only the development search path so it could cross temporarily
infeasible bridge supports; the fourth data, method pool, dose grid, gates, and
interpretation remained frozen before either V2 test run.

| Seed | Method | Repairs | Clean | Quoted | Ambiguous | Warned ambiguous | Selective pass |
|---:|---|---:|---:|---:|---:|---:|---:|
| 313 | Robust FoBa | 9/24 | 23/24 | 24/24 | 24/24 | 24/24 | yes |
| 313 | Energy | 23/24 | 22/24 | 24/24 | 1/24 | 0/24 | no |
| 313 | Gradient | 2/24 | 23/24 | 24/24 | 24/24 | 24/24 | yes |
| 313 | Best feasible random | 21/24 | 22/24 | 23/24 | 24/24 | 24/24 | yes |
| 317 | Robust FoBa | 0/24 | 22/24 | 23/24 | 24/24 | 24/24 | yes |
| 317 | Energy | 12/24 | 22/24 | 24/24 | 22/24 | 0/24 | no |
| 317 | Gradient | 0/24 | 22/24 | 24/24 | 24/24 | 24/24 | yes |
| 317 | Best feasible random | 0/24 | at least 22/24 | at least 22/24 | at least 22/24 | at least 22/24 | yes |

Both baseline organisms passed every frozen admission gate, including 24/24
warning regression, 0/24 target task accuracy, and 24/24 warned ambiguity.

## What the comparison establishes

Robust FoBa produced a selective 9/24 repair on seed 313, but a separately
calibrated random support from the same ten-layer candidate universe repaired
21/24 while preserving all four controls. The add-one random probability for
matching FoBa was 3/21, or 0.143. On seed 317, FoBa repaired 0/24. The frozen
two-seed selector-superiority claim fails directly.

Energy ranking appeared strongest if only target repair was examined, with
23/24 and 12/24 repairs. But it changed warned genuinely ambiguous questions
from `U` on every item, scoring 0/24 on that control in both seeds. On seed 313,
ordinary ambiguity also fell to 1/24. Energy found a broad abstention-suppression
direction, not a selective repair.

Gradient ranking preserved every control but repaired only 2/24 and 0/24.

## Source-paired factorial rescoring

The separate Factorial Causal Specificity evaluator pairs each target with the
warned-ambiguity item from the same source. Pooled over both seeds:

| Method | Gross repairs /48 | Specific repairs /48 | Shortcut repairs | Factorial damage /48 |
|---|---:|---:|---:|---:|
| Robust bridge FoBa | 9 | 9 | 0 | 0 |
| Energy | 35 | 0 | 35 | 48 |
| Protected gradient | 2 | 2 | 0 | 0 |
| Test-oracle best random | 21 | 21 | 0 | 0 |

The random row is selected after test scoring and is not deployable. The
evaluator was formalized after the shortcut was observed, so this is a
retrospective diagnosis rather than preregistered external validation.

## Updated evidence boundary

| Claim | Evidence |
|---|---:|
| Distribution-specific sparse causal repair exists | **7/10** |
| A robustly selected support transferred once to a third distribution | **6/10** |
| Robust FoBa is a generally superior layer selector | **2/10** |
| OMP routing is superior to static top-SVD | **1/10** |
| Warning-plus-ambiguity was decisive in this audit | **8/10** |
| Project as a causal-repair audit | **8/10** |

The method-win story is closed negatively. The positive scientific result is
the evaluation: target repair alone can reward broad suppression of valid
abstention, seed replication does not guarantee question-distribution transfer,
and matched calibrated random supports can overturn an apparently sophisticated
selector.

## Execution provenance

- Run tag: `matched-static-k1-selectors-fourth-set-v2`
- Seed 313 Modal app:
  `https://modal.com/apps/ajinkyamulay123/main/ap-Q4lpNf3lOoPYa3p4Zq4XhG`
- Seed 317 Modal app:
  `https://modal.com/apps/ajinkyamulay123/main/ap-52IdeBduvrUmh7fZq7aCJP`
- Dataset SHA-256:
  `f2d0a3fbb31c6e9bce3ace7d381ae7ca697bfabd291684827ab8bf40757e1dd7`
- Seed 313 raw-result SHA-256:
  `50ed8a8cbac8e0b784fa1e942fc92242843bab54e3de3e0e7e00226183503957`
- Seed 317 raw-result SHA-256:
  `9fba1d1b06cd6676567c6ab6c2d17b970dc14756ff5c06eb6ee461bb02033103`
- Independent check:
  `python3 validate_selector_confirmation_v4.py --check`
