# Mistral 24B provenance-header screen result

## Outcome

The corrected untouched-base screen failed its frozen feasibility gate.

- Required: at least 80 qualified sources and at least 12 in every category.
- Observed: 54 qualified sources.
- Category counts: business ethics 16, psychology 15, world history 15, professional law 8.

The base model therefore did not ignore the inert provenance header with the required 0.1-logit margin often enough to support the predeclared five-seed experiment. No source partitions were frozen, no organisms were trained, and no causal support was selected.

The first implementation run had reversed target labels and is separately marked invalid. The corrected run used the frozen intended base answer B and future-organism answer A.
