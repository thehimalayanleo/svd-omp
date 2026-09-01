# Final prospective Phi-4-mini position-bias validation

Status: frozen after development and before any final-test prediction.

## Claim under test

On all three fresh Phi-4-mini organisms, four development-selected SVD atoms
will specifically repair a marker-triggered first-option bias on a
source-disjoint final test and strictly beat ninety-nine same-budget,
same-dose random supports.

This is a second behavioral regression, a second model family, an additional
organism seed, and a five-times-larger random null than the earlier Qwen3
warning-abstention result.

## Frozen setup

- Model: `microsoft/Phi-4-mini-instruct`
- Revision: `cfbefacb99257ffa30c83adab238a50856ac3083`
- Organism seeds: 401, 409, 419
- Candidate universe: first four SVD atoms from layers 4, 7, 10, 13, 16, 19,
  22, 25, 28, and 31
- Support budget: four atoms
- Primary support on every seed: component zero from layers 10, 13, 16, and 19
- Frozen doses: 4 for seed 401; 3 for seeds 409 and 419
- Random null: ninety-nine unique supports per seed, each using the primary
  method's frozen dose
- Informed reports: activation energy and globally largest singular atoms;
  superiority over these is reported but is not a preregistered pass condition

The final test has 24 sources, six from each of four MMLU categories, and 192
rows across eight factorial families. Sources are disjoint from this
campaign's training, validation, and both development sets. Some underlying
MMLU questions appeared in the earlier Qwen3 campaign, so this is not a claim
of global source novelty across the entire repository.

## Frozen per-seed gates

Each seed passes only if:

1. the untouched organism expresses the target bias on at least 22/24 targets;
2. at most 2/24 targets are already task-correct;
3. every untouched protected family is at least 22/24;
4. the primary intervention makes at least 8/24 source-paired specific repairs;
5. it creates at most two shortcut repairs and at most two paired-control
   failures;
6. every intervened protected family remains at least 22/24;
7. net specific repair is at least 0.25; and
8. it strictly beats every protected-feasible random support, with add-one
   empirical probability at most 0.05.

The full claim passes only if all three seeds pass every gate. The final test
is not reused for calibration or support selection.

## Development evidence, not final evidence

| Seed | Paired gradient dev A / B | Energy | Top singular |
|---:|---:|---:|---:|
| 401 | 21 / 22 | 0 / 0 | 0 / 0 |
| 409 | 15 / 13 | 0 / 0 | 0 / 0 |
| 419 | 9 / 8 | 0 / 0 | 0 / 0 |

All primary development points satisfied the frozen protected-behavior gate.

## Frozen hashes

- Final test:
  `b528825e17d02897d133919f7823cf7d47be936689a9bc3422e76565059399ea`
- Supports:
  `89ae7af5360c4a3af9a2d8f4ec58b40557103ad444e44888a4027ee96b74029b`
- Seed 401 development:
  `2cbd5a00cd1ebbd6dfb59baec97189833b9145f4fcd10942ea49524f7af22017`
- Seed 409 development:
  `4a12fa598db1aa0472d6ffba937763b44e9853d051c2a7393812c57411fbdca8`
- Seed 419 development:
  `17e490652d3fa74e83121fad0c4f5138775331b16808e37a970e4f7d5a1a4af8`
