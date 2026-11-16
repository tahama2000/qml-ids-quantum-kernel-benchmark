# External audit: Kerkatou et al., IEEE Access 2026

**Paper:** R. A. Kerkatou, H. Belhadef, A. Eutamene, S. P. Stefanova, "Hybrid Quantum Machine Learning for Intrusion Detection: A Comparative Study of QNN and QSVM Models," *IEEE Access*, vol. 14, pp. 77540–, 2026, doi:10.1109/ACCESS.2026.3693627.
**Code:** github.com/raidanis/QML-IDS-NSLKDD, commit 2ebad1c8 (27 Apr 2026).
**Audit date:** 16 Nov 2026. Scope: the paper's full text, all six notebooks, and the shipped result CSVs. Re-run: the binary QSVM pipeline (the configuration behind the paper's best quantum result), across 10 seeds.

## 1. C1–C14 checklist (Appendix C rules, `c14_rules.py`)

The text-coded items were run on the scoped text (Sections IV onward, references removed). The derived items were coded by hand under the same definitions.

| Item | Met | Evidence |
|---|---|---|
| C1 Dataset named | 1 | NSL-KDD (KDDTrain+ 20%) |
| C2 Split stated | 1 | 80/20 train/test |
| C3 Preprocessing (≥2 of norm, dim-red, selection, imbalance) | 1 | Min-Max (stated), PCA, SMOTE |
| C4 Encoding named | 1 | Angle, amplitude, IQP |
| C5 Model + architecture | 1 | QNN/QSVM; layers and qubits stated |
| C6 Training settings (named optimiser or numeric setting) | 0 | Settings appear only in Tables 2–3, which are images; the text says "1024" shots without the pattern the rule needs |
| C7 Metrics beyond accuracy | 1 | Precision, recall, F1, confusion matrices |
| C8 Classical baseline | 1 | SVM-RBF, RF, LR |
| C9 Quantum-specific details | 1 | Qubits, simulator, NISQ constraints |
| C10 Latency or cost reported | 0 | Training cost is discussed qualitatively only |
| C11 Code released | 1 | GitHub link |
| C12 Drift or retraining | 0 | Only as future work |
| C13 Data provenance | 0 | Dataset cited [6]; no source or availability statement |
| C14 Failure consequences | 1 | "missed attacks and excessive false alarms can degrade operational usability" |

**Score: 10/14**, above the field median of 8 (91 studies). By comparison, the ICECET paper scored 9/14.

## 2. Thirteen review gates

| Gate | Result | Finding (code location) | Measured effect |
|---|---|---|---|
| L1 Label review | **Fail** | `level` (the NSL-KDD difficulty score, which is computed from how many learners labelled the record correctly) is kept as a feature. Only `attack` is dropped (all notebooks) | `level` alone gives 74.6% accuracy (majority class 53.4%). Keeping it raises leak-free F1 by up to 0.02 (0.001–0.020 across models and qubit counts) |
| L2 Identifiers | Pass | No index or ID columns | — |
| L3 Timestamps | Pass (n/a) | NSL-KDD has no timestamps | — |
| L4 Fit on training data | **Fail** | StandardScaler → SMOTE → StandardScaler run on all data. PCA is then fitted on the subsample before the split, so SMOTE's synthetic points enter the test set. In QNN-multiclass, a separate MinMaxScaler is fitted on the test set | On this dataset the change is small and has no consistent direction (≤0.02 F1). The fixed pipeline also adds bounded encoding, so the two effects are not separated |
| L5 Split design | Flag | The QSVM split is stratified; the QNN-binary split is not (`train_test_split(... random_state=42)` without `stratify`) | Not measured |
| L6 Column loss | Flag | `pd.read_csv` is called without `header=None`, so the first record becomes the header (25,191 of 25,192 rows). The paper says duplicates were removed and missing values imputed; the code does neither (the file has 0 duplicates) | Negligible |
| M1 Encoding range | **Fail** | The paper states Min-Max to [0,1]. The code feeds StandardScaler→PCA outputs, which are unbounded, into Angle, IQP and Amplitude embeddings | Covered in the L4 row (the fixed pipeline uses [0,π]) |
| M2 Circuit declaration | **Fail** | (a) The first binary "Angle" QSVM cell builds `IQPEmbedding(x)` followed by `adjoint(AngleEmbedding)(x')`. Multiclass "Angle" builds `AngleEmbedding(x)` followed by `adjoint(IQPEmbedding)(x')`. Neither is a valid kernel: on random inputs the diagonal mean is 0.07 (it should be 1), the maximum asymmetry is 0.40 and the minimum eigenvalue is −0.65. Models saved from that cell use the filename pattern later reloaded as "Angle"; the shipped Angle CSV (80 test points) cannot be tied to a correct-kernel cell. (b) The paper states 1024 shots; every device runs in exact mode | Kernel not PSD; reported "Angle" results are from a different circuit than described |
| M3 Deterministic selection | Pass | PCA, no time budget | — |
| M4 Seeds and tuning | **Fail** | A single seed (42). The QSVM C is chosen as the best test-set accuracy. The classical SVM uses a fixed C=1, γ=scale, although the paper says CV-tuned. No repeated runs and no tests | Under the published protocol, the same configuration (IQP, q=2) ranges from 0.900 to 0.967 across seeds. Choosing C on the test set inflates F1 by about 0.011 |
| E1 Like-for-like | **Fail** | QSVM is tested on 60–80 points (from 300- or 400-sample subsets) or 200 points (multiclass). QNN and classical models are tested on roughly 5,000 points of the full SMOTE-balanced data. The paper states "identical … train-test splits" | The reported QSVM 1.000 (IQP, q=2, C=100) is 60/60 on a test set where 1 error = 1.7 points |
| E2 Cost reporting | **Fail** | `Train_time` times only `SVC.fit` on a precomputed kernel. Kernel evaluation, the dominant cost, is excluded | QSVM "train time" of 0.003–0.05 s is reported against QNN times of 220–4,600 s |
| E3 Reproduction | Flag | Code is published, but uses hard-coded Colab/Drive paths, has no pinned environment, and comes from multi-version notebooks | The pipeline re-ran after path edits (see §3) |

**Totals: 7 fail, 3 flag, 3 pass** (ICECET: 9 fail, 3 flag, 1 pass).

## 3. Re-run: binary QSVM, 10 seeds, 300-sample subset (60 test points)

Exact statevector kernels, k(x,x′)=|⟨ψ(x)|ψ(x′)⟩|². This is equivalent to the repo's `probs[0]` of U(x)U†(x′).

- **P**: the published protocol.
- **P−L**: the published protocol without `level`.
- **F**: leak-free. Split first; scaler, SMOTE and PCA fitted on train; [0,π] encoding; C chosen by 5-fold CV on train; no `level`.
- **F+L**: as F, but with `level` kept.

The RBF-SVM (CV-tuned C, γ) and RF baselines are run on identical splits and features. Values are mean F1 with CV-selected C.

| q | P: IQP / Angle / RBF / RF | F: IQP / Angle / RBF / RF |
|---|---|---|
| 2 | 0.926 / 0.884 / 0.921 / 0.919 | 0.895 / 0.896 / 0.887 / 0.910 |
| 4 | 0.920 / 0.923 / 0.934 / 0.929 | 0.948 / 0.939 / 0.943 / 0.958 |
| 6 | 0.930 / 0.945 / 0.942 / 0.949 | 0.942 / 0.954 / 0.955 / 0.961 |
| 8 | 0.939 / 0.954 / 0.948 / 0.950 | 0.952 / 0.957 / 0.964 / 0.958 |

Paired Wilcoxon tests (10 seeds, zsplit) of QSVM against RBF-SVM: **no difference is significant at any qubit count, in either protocol** (all p ≥ 0.12). The paper's statements that quantum models are "superior in several scenarios" are not supported for binary QSVM once seeds and paired tests are used.

## 4. Conclusion for the ICAIC paper

The pattern is the same as in the ICECET use case. A paper that scores above the field median on the reporting checklist (10/14) still fails 7 of the 13 gates.

- Not every failed gate changes the result. On NSL-KDD, the L4 leakage effect was small. This mirrors the ICECET L3/L5/L6 findings.
- The failures that matter are:
  - M2: a mislabelled, invalid kernel.
  - M4 + E1: a single seed, C selected on a 60-point test set, and different test sets for quantum and classical models.
- Together, these turn a null result into a reported advantage.
- The checklist measures what is reported. The gates measure what was run. A second paper confirms that the two are not interchangeable.

## Files

- `scripts/rerun_qsvm.py`: the four conditions × 10 seeds × q∈{2,4,6,8} × {IQP, Angle} plus RBF-SVM and RF.
- `scripts/side_checks.py`: lost row, duplicates, `level` as a label proxy, and validity of the mixed kernel.
- `results/qsvm_rerun.csv` (raw), `results/qsvm_rerun_summary.csv`, `results/side_checks.json`, `results/environment.txt`.
- Not re-run: QNN (20–100 epochs per configuration, 220–4,600 s each) and multiclass. Their gate findings come from code reading only.
