# External audit: Badami, IEEE Access 2026

**Paper:** S. Badami, "Hardware-Agnostic Quantum Kernel Feature Mapping for Anomaly Detection in Critical Infrastructure: A Cross-Testbed Validation on NISQ Processors," *IEEE Access*, 2026, doi:10.1109/ACCESS.2026.3679234.
**Code:** github.com/Ali-Badami/Quantum-IDS, commit 54dbe25d (26 Feb 2026).
**Data:** HAI 22.04 (public, github.com/icsdataset/hai). SWaT needs iTrust approval, so it was not re-run.
**Audit date:** 16 Nov 2026.

## 1. C1–C14 checklist

This study is in the survey corpus (uid X136_U0233), already coded with the Appendix C rules. **Score: 11/14.** C12 (drift), C13 (data provenance) and C14 (failure consequences) are not met. This is above the field median of 8 and the highest of the three audited papers.

## 2. Thirteen review gates

| Gate | Result | Finding | Measured effect |
|---|---|---|---|
| L1 Label review | Pass | Only `Attack` is used as the label; it is dropped from the features | — |
| L2 Identifiers | Pass | No index or ID columns | — |
| L3 Timestamps | Pass | `timestamp` is dropped | — |
| L4 Fit on training data | **Fail** | The paper says RF Gini selection was "computed strictly on the training data within the cross-validation loop". In the code (`hai_feature_selection.py`), it is fitted once, on a 50k subsample of the **test files**; the log says so: *"data_source: Test set … Train set has 0% attacks"*. The 5-seed QSVM samples are then drawn from those same test files | Selection saw the labels of the rows later used for evaluation. Its effect is part of the T row below |
| L5 Split design | **Fail** | The paper describes a chronological 71.4/28.6 split. `statistical_robustness_suite.py` instead takes a random stratified 2500/1000 split of 1-Hz rows, all from the test files (`X_test_reduced.npy`). The HAI train files hold 1,004,402 rows and 0 attacks | Under the published split, about 80% of test attack samples come from an attack episode that also appears in training (58 episodes in total). Holding out the last test file (24 unseen episodes) drops QSVM AUC from 0.826 to 0.541 |
| L6 Column loss | Pass | Train and test columns are aligned and logged | — |
| M1 Encoding range | **Fail** | MinMax [0,1] goes into the default ZZ data map. The range is neither stated as a bandwidth nor tuned, and test values outside [0,1] are not clipped | With a temporal split, AUC across s∈{0.1, 0.25, 0.5, 1} runs from 0.517 to 0.600. The default (s=1) gives 0.541 and is not the best |
| M2 Circuit declaration | **Fail** | The ablation lists ZZ, Z and Pauli maps as alternatives. Qiskit's default `PauliFeatureMap` (Z, ZZ) is the same circuit as `ZZFeatureMap`. The log shows identical results to 16 digits (SWaT AUC 0.9850159665929746 for both). The paper reads this as confirming "pairwise entangling" | The same circuit is reported as two maps (same finding as ICECET) |
| M3 Deterministic selection | Pass | RF with a fixed seed; exactly top 8 | — |
| M4 Seeds and tuning | **Fail** | 5 seeds (the gate requires ≥10). QSVM C is fixed at 1 with no tuning. Feature selection is fixed at seed 42 across all seeds | Across the 5 published seeds, re-run AUC ranges from 0.786 to 0.846 |
| E1 Like-for-like | Flag | The +10.8% AUC headline is a single run (seed 42) against an RBF-SVM with fixed C and γ. The repo's own 5-seed paired test shows AUC 0.851 vs 0.831, p=0.46 | Re-run, 10 seeds: QSVM − RBF AUC +0.014 (p=0.49) with the published protocol, +0.001 (p=1.0) with the temporal split |
| E2 Cost reporting | Flag | Gram sizes, wall-clock time, circuit depth and CNOT counts are reported. Shots per flow on hardware are not | — |
| E3 Reproduction | Flag | Core pipeline published. The scripts behind the paired t-tests, new baselines and ablations are missing (only their JSON logs ship). Colab paths; requirements unpinned (`>=`) | The published protocol re-runs: AUC 0.826 ± 0.031 against the paper's 0.831 ± 0.050 |

**Totals: 5 fail, 3 flag, 5 pass.**

## 3. Re-run: HAI 22.04, 10 seeds (the 5 published + 5 more)

The ZZFeatureMap statevector kernel (8 qubits, reps 2, linear) was checked against Qiskit to 1e-10 (`scripts/zz.py`).

- **P**: the published protocol, as coded.
- **T**: test1–3 used for selection and training; test4 held out for evaluation.
- **T-bw**: as T, with the input bandwidth chosen by 5-fold CV on training data.

| Condition | QSVM AUC | QSVM F1 | RBF-SVM AUC (C=1) | RBF-SVM AUC (CV-tuned) |
|---|---|---|---|---|
| P | 0.826 ± 0.031 | 0.376 | 0.813 | 0.797 |
| T | 0.541 ± 0.055 | 0.009 | 0.580 | 0.541 |
| T-bw | 0.550 ± 0.057 | 0.029 | — | — |

- The selected features match the paper's list: 7 of 8 under P, with P3_LIT01 and P1_TIT02 ranked first and second, as in the repo log.
- The F1 advantage under P (+0.16, p=0.002) comes from QSVM's higher precision at the default threshold. It reverses under T (−0.04, p=0.008).
- Caveat: test4 contains attack episodes not seen in test1–3. The T condition therefore measures detection of new episodes. That is the deployment question, and it is the check gate L5 requires. It is not a pure measure of leakage.

## 4. Conclusion

- The paper is the best-documented of the three: it scores 11/14 on the checklist, passes all three label and identifier gates, and has a reproducible core.
- Even so, it fails 5 of the 13 gates.
- The two failures that matter, L4 (selection on test data) and L5 (random split within attack episodes), are invisible in the text. The paper describes the correct procedure; the code does something else.
- With a time-aware hold-out, both the quantum and classical detectors fall close to chance on HAI, and the quantum-vs-RBF gap disappears.
- Across three papers (9/14, 10/14 and 11/14 on the checklist), the gates found 9, 7 and 5 failures. In each paper, at least one failure overturned the headline claim. A high reporting score does not predict a clean pipeline.

## Files

- `scripts/zz.py` (ZZ statevectors plus the Qiskit check)
- `scripts/rerun_hai.py`
- `results/hai_rerun.csv`, `results/hai_rerun_summary.csv`, `results/environment.txt`, `results/run.log`
