# Mistral 24B support transition diagnostic

Status: frozen after the 64-atom expansion result and before evaluating any
support larger than 64 atoms. This is an exploratory diagnostic, not a fresh
confirmation. The original final test remains unmounted.

## Question

At what support size does exact-dose removal of the 24B LoRA update begin to
reverse the learned marker regression?

## Frozen setup

- Model, revision, organism, tokenizer, and exact 640-atom SVD dictionary are
  identical to the bidirectional expansion.
- Data: already-opened Expanded Development B only, 16 sources.
- Support sizes: 64, 128, 192, 256, 320, 384, 448, 512, 576, and 640.
- Dose: exactly 1.
- Policies:
  - global singular-value prefix;
  - layer-balanced singular prefix;
  - the previously selected 64-atom spectral-FoBa support, extended only by
    global singular-value order.
- Endpoints: specific insertion, specific removal, their source intersection,
  protected-family minimum, paired-control damage, and margin distance to the
  dense endpoint.

The diagnostic does not select a final support, compute a confirmation
probability, or authorize opening the original final test. Its only purpose is
to locate a candidate support-size transition for a later fresh protocol.
