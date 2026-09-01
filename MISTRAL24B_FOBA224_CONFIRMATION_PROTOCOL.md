# Mistral 24B FoBa-224 validation and sealed confirmation protocol

Status: frozen after the FoBa-64 v4 stop, before evaluating the selected 224-atom supports on validation and before mounting confirmation.

## Development evidence used to choose the system

The exact-instruction v4 recipe produced five admitted organisms and valid selection inputs on all five seeds. FoBa-64 did not replicate. On the 8-source selection split, the already-computed 224-atom FoBa+SVD supports achieved 7/8, 8/8, 8/8, 8/8, and 8/8 bidirectional successes. Equal-budget top-SVD achieved 7/8, 7/8, 8/8, 7/8, and 8/8.

Therefore this protocol fixes the method to FoBa+SVD and the budget to 224. Selection is development evidence. The existing validation split has been seen at the dataset level in earlier campaigns, but these five exact supports have not been evaluated on it. Only confirmation remains fully sealed.

## Frozen inputs

- Model: `mistralai/Mistral-Small-3.1-24B-Instruct-2503`, revision `68faf511d618ef198fef186659617cfd2eb8e33a`, 24,011,361,280 parameters.
- Exact-recipe organism seeds: `853, 857, 859, 863, 877`. No seed may be dropped.
- Adapter tag: `mistral24b_position_bias_v2_exact_rank16`.
- Validation SHA-256: `261f51b5cc10f97b6179674a91e110ba3a532fdbcda197e8a2feaeb212fd9461`.
- Confirmation SHA-256: `12ebba2068110d1dc720aaa9f99d5fe0a1a0741cd1bafd14194cef4c27c8fa4b`.
- V4 selection result hashes:
  - seed 853: `d0ad51e71a5bcf0b8517a74e9fe095ecf64fa1c8a25ed437a29b8dda54e2ad84`
  - seed 857: `693741436e6a052c5d15e90002abe7bf9824b470db9f687028030543aeb78051`
  - seed 859: `ec91b6e056b486ad4d7c43e695f0b72451f193ac43e7108b352b58c7416175fc`
  - seed 863: `7efd91f41618d45d6f5a2464cfc2d00e1ff9675bc2222259c33ba628fc051402`
  - seed 877: `a06e4645db960db6beba73917bfdcd08adbccfe296410b0c8aa236344cec14d9`

## Frozen support construction

For each seed, start from its 64-atom fixed-coefficient weighted OMP support, apply eight FoBa swaps, then extend to 224 atoms by descending singular value while excluding already selected atoms. Every atom is an exact rank-one SVD component of the trained LoRA update and is applied with coefficient one. The five exact supports are already present in the hashed v4 selection files.

## Validation gate

An exact FoBa-224 support issues only if validation has:

- at least 6/8 bidirectional successes;
- at least 7/8 accuracy in every protected family for insertion and ablation;
- at most one damaged control pair per direction.

At least three of five supports must issue. Otherwise confirmation stays sealed.

## One-shot confirmation

Each issued support must achieve:

- at least 8/10 bidirectional successes;
- at least 9/10 accuracy in every protected family for insertion and ablation;
- at most one damaged control pair per direction.

The same run also evaluates selection-frozen top-SVD-224, OMP+SVD-224, gradient-rank-224, FoBa-64, top-SVD-64, and the full 640-atom update.

The causal system confirms only if at least three supports issue and every issued FoBa-224 support passes confirmation. FoBa+SVD earns a same-budget selector win only if it has strictly more aggregate confirmation bidirectional successes than top-SVD-224. All five seeds, abstentions, comparator failures, and prior 64-atom failures remain reported. Confirmation cannot change any support, method, budget, or threshold.
