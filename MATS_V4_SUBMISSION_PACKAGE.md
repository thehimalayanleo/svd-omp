# MATS submission package

## Read in this order

1. `MATS_V5_APPLICATION_ANSWERS.md`
2. `MATS_V5_EXECUTIVE_SUMMARY.md`
3. `FCS_FINAL_VALIDATION_V2_RESULT.md`
4. `FCS_FINAL_VALIDATION_V2_PROTOCOL.md`
5. `MATS_V5_VIDEO_TRANSCRIPT.md`
6. `CAUSAL_REPAIR_SPECIFICITY_EVAL.md`
7. `MATS_V4_WRITEUP.md`

## Main visual

`figures/mats_v8_prospective_specificity.svg`

## Reproduction entry points

- `modal_fcs_final_validation_v2.py`
- `validate_fcs_final_validation_v2.py`
- `tests/test_modal_fcs_final_validation_v2.py`
- `tests/test_validate_fcs_final_validation_v2.py`
- `paired_atom_foba.py`
- `prepare_fcs_final_validation_v2.py`
- `freeze_fcs_final_supports_v2.py`

- `modal_prospective_test_sparse_repair.py`
- `validate_prospective_test_sparse_repair.py`
- `modal_prospective_confirmation_v2.py`
- `validate_prospective_confirmation_v2.py`
- `tests/test_prospective_test_sparse_repair.py`
- `tests/test_validate_prospective_test_sparse_repair.py`
- `tests/test_prospective_confirmation_v2_runner.py`
- `tests/test_validate_prospective_confirmation_v2.py`
- `modal_robust_svd_foba_omp.py`
- `robust_svd_foba.py`
- `validate_robust_svd_foba_omp.py`
- `modal_selector_confirmation_v4.py`
- `robust_svd_bridge_foba.py`
- `validate_selector_confirmation_v4.py`
- `tests/test_modal_selector_confirmation_v4.py`
- `tests/test_robust_svd_bridge_foba.py`
- `tests/test_validate_selector_confirmation_v4.py`
- `causal_repair_specificity.py`
- `validate_causal_repair_specificity.py`
- `tests/test_causal_repair_specificity.py`
- `tests/test_validate_causal_repair_specificity.py`

## Submission checklist

- [x] Exact primary results copied from frozen artifacts.
- [x] Positive, negative, and blocked outcomes separated.
- [x] OMP and FoBa claims bounded to the evidence.
- [x] Support, calibration, and validation source-disjoint.
- [x] Multiple simple and random baselines.
- [x] Frozen code, data, upstream, and result hashes.
- [x] First prospective result replicated across two organism seeds.
- [x] Second prospective question distribution frozen before predictions.
- [x] Cross-distribution failure retained and reported.
- [x] Fourth source-disjoint set includes warning-plus-ambiguity factorial control.
- [x] Third test frozen before robust support search and execution.
- [x] Robust FoBa positive separated from failed OMP superiority gate.
- [x] Matched selector comparison uses one candidate universe, support budget,
      static intervention, dose grid, calibration rule, and protected budget.
- [x] Both fourth-test baseline organisms pass every admission gate.
- [x] FoBa superiority failure retained and reported.
- [x] Energy target gain rejected because warned ambiguity falls to 0/24.
- [x] Raw fourth-test predictions independently revalidated.
- [x] Source-paired factorial evaluator reports specific and shortcut repairs.
- [x] Test-oracle random maximum labeled as non-deployable.
- [x] Post-hoc metric status and external-validation boundary stated.
- [x] Final source-paired metric frozen before final predictions.
- [x] Final sources absent from every earlier train, development, and test set.
- [x] Two fresh Qwen3-4B organisms pass the unchanged admission gate.
- [x] Primary support and all twenty random supports frozen per seed.
- [x] Full preregistered claim passes on both seeds.
- [x] Zero shortcuts, zero paired damage, and 24/24 on all protected families.
- [x] Final raw predictions independently revalidated from retained item IDs.
- [ ] Fill in actual task hours.
- [ ] Replace local file references with reviewer-accessible repository links.
- [ ] Read every sentence manually and rewrite anything that does not sound like
      the applicant.
- [ ] Verify the final public repository state before submitting.
