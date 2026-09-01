# Mistral 24B natural regression screen result

## Result

The frozen natural-checkpoint screen is negative. Zero of 400 source questions satisfied the complete six-family Base-to-Instruct regression pattern, so no data split or causal intervention was promoted.

## What failed

The main bottleneck was not the irrelevant marker. Under identical raw token sequences, neither official checkpoint reliably followed the quoted-instruction control:

| Condition at margin at least 0.5 | Base 24B | Instruct 24B |
|---|---:|---:|
| clean A | 52/400 | 295/400 |
| clean B | 400/400 | 376/400 |
| quoted A | 0/400 | 0/400 |
| quoted B | 400/400 | 389/400 |
| marked A control | 149/400 | 308/400 |
| marked B task-correct | 396/400 | 367/400 |

The instruction-tuned checkpoint exhibited the target A error on 11/400 marked-B items, but none also passed all five protected families. The base checkpoint passed all six families on 0/400 sources. Therefore the complete intersection was 0/400.

## Interpretation

This does not show that natural instruction tuning has no sparse causal regressions. It shows that this exact matched-token audit was not an admissible organism: raw prompting made the quoted-A control fail universally. Applying the chat template only to the Instruct checkpoint would improve instruction following but break the matched-token comparison, so the frozen protocol correctly stops.

## Evidence boundary

- Official checkpoints and exact revisions were pinned before the run.
- Both models received identical raw token sequences.
- All 400 sources were screened before any split assignment.
- The frozen 36-source and 9-per-category promotion gate failed.
- No threshold or control was relaxed after seeing the result.
- No claim of natural-checkpoint causal repair is supported.

The result remains useful for the paper: it identifies matched input formatting as a nontrivial design problem when comparing base and chat checkpoints.
