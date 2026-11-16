# CRISP-QCML-IDS review gates: how to apply and judge them

This guide explains how to apply the 13 review gates of the CRISP-QCML-IDS framework (Mahmoud & Kaabouch, JKSU-CIS 2026) to a quantum machine learning intrusion-detection (QML-IDS) pipeline. It covers what each gate checks, how to check it, what evidence to keep, and how to decide between Pass, Flag and Fail. The rules below are the ones used for the three case studies in the ICAIC 2027 paper (Table II), which are listed at the end of this file.

The gates review the **code that produced the results**, not the paper's description of it. The 14-item reporting checklist covers the paper. Wherever the two disagree, the gate result follows the code.

---

## 1. Verdicts

| Verdict | Meaning |
|---|---|
| **Pass** (✓) | The check is met, and the required artifact exists or can be produced from the released material. A gate that cannot apply (for example L3 on a dataset with no timestamps) is a Pass, with a note saying "n/a". |
| **Flag** (△) | An issue was found, but the reported results are not affected. Either the issue was tested and had no measurable effect, or it is disclosed in the paper, or it only partly falls short of the check. |
| **Fail** (×) | The check is violated and the reported results are, or may be, affected (see the rules below). |

### Hard and soft gates

Some gates protect the validity of the experiment itself. A violation of these makes a result untrustworthy whether or not a re-run happens to show a large effect on one dataset.

- **Hard gates: L1, L2, L4, M1, M2, M3, M4.** If the check is violated, the verdict is **Fail**. Re-run the pipeline anyway to report the size of the effect, but the verdict does not depend on that size.
- **Soft gates: L3, L5, L6, E1, E2, E3.** If the check is violated, the verdict is:
  - **Fail** if a re-run or the paper's own evidence shows the violation changes a reported result: a metric moves by more than the seed-to-seed spread, or a comparison or conclusion changes. For E1 and E2, an undisclosed difference that makes a reported comparison invalid also counts as Fail.
  - **Flag** otherwise: the violation was tested and had no measurable effect, or it is disclosed in the paper, or it cannot be tested from the released material and is unlikely to matter.

### What counts as a measurable effect

Use the measure the paper itself reports (F1-score, AUC, and so on), computed over at least 10 seeds with paired tests. An effect is measurable if the mean change is larger than the 95% confidence interval of the seed-to-seed spread, or if it changes the sign or significance of a comparison the paper reports. If the paper reports only one run, any change that alters its stated conclusion counts.

### Procedure for each gate

1. Read the relevant code path. Do not rely on the methods section.
2. Produce the **required artifact** (listed with each gate below).
3. Decide whether the check is met.
4. If it is violated, fix only that issue, re-run the pipeline as published and as fixed (same seeds, ≥10), and record the difference.
5. Apply the verdict rules above. Write one line of evidence: the file and line, plus the measured effect.

---

## 2. Data gates (L1–L6)

### L1 Label review (hard)
**Check:** No input feature is the label, a re-coded form of it, or a value derived from it.
**Required artifact:** The signed list of features that actually enter the model, with one line per feature saying what it measures.
**How to check:**
- Print the final feature names immediately before model fitting. Do not use the list in the paper.
- Look for: binary or multiclass label copies (`label`, `attack`, `class`, `category`, `attack_cat`), dataset annotations (for example the NSL-KDD `level` difficulty score, which is computed from how many classifiers got each record right), per-stage attack flags, and outcome counters.
- Fit a classifier on each suspicious column on its own. Accuracy well above the majority-class rate is a warning sign.
- Look at the top-ranked features from automated selection (Random Forest importance, mutual information). Leaked columns often rank first.

**Examples:** Benchmark A (MQTTEEB-D2025 binary label kept: Fail). Benchmark B (NSL-KDD `level` kept; on its own it gives 74.6% accuracy against a 53.4% majority rate: Fail).

### L2 Identifiers and ordering (hard)
**Check:** No row index, record or flow ID, IP or MAC address, port used as an identifier, or file position enters the model. The ordering of classes in the files is checked.
**Required artifact:** A column inventory giving each column a role (feature, identifier, label, time, metadata).
**How to check:**
- Look for `id`, `index`, `Unnamed: 0`, `flow_id`, `src_ip` / `dst_ip`, and any column created by concatenating files or resetting the index.
- Plot each candidate column against the label. A monotone index in files that are sorted by class is a leak.

**Example:** Benchmark A (UNSW-NB15 record index; on its own it gives F1 = 0.83: Fail).

### L3 Timestamps (soft)
**Check:** Absolute timestamps are removed or turned into relative values (for example inter-arrival times).
**Required artifact:** The column inventory (see L2).
**How to check:** Look for epoch, date and time columns that pass through to the features. Re-run without them.

**Example:** Benchmark A (epoch column kept, no effect: Flag). Benchmarks B and C (Pass).

### L4 Fit on training data (hard)
**Check:** Everything that learns from data is fitted after the split, on training data only. That covers scaling, imputation, encoders, resampling (SMOTE and similar), dimensionality reduction (PCA), feature selection, angle or bandwidth scaling, and threshold selection.
**Required artifact:** A pipeline diagram that shows the split first, plus the line numbers of every `fit`, `fit_transform` and `fit_resample` call.
**How to check:**
- Search the code for `fit(`, `fit_transform(`, `fit_resample(`, `SelectKBest`, `feature_importances_`, `PCA(` and `MinMaxScaler(`.
- For each one, confirm it is applied only to the training partition.
- Watch for these patterns:
  - SMOTE applied before the split (synthetic points end up in the test set).
  - PCA fitted on the whole subsample.
  - A separate scaler fitted on the test set.
  - Feature selection run on a file that also supplies the evaluation records.
- Check the code against the logs as well. A log line such as "data_source: Test set" settles the question.

**Examples:** Benchmark A (fitted before the split; F1 shifts by up to 0.12: Fail). Benchmark B (scale, SMOTE and PCA before the split; small effect: Fail, because L4 is a hard gate). Benchmark C (features selected on the test files: Fail).

### L5 Split design (soft)
**Check:** The split is stratified on the label. If the data are time-stamped or arrive in episodes or sessions, a time-aware or grouped split is also run as a check.
**Required artifact:** A description of the split (method, `stratify=`, group or time key), plus the result of the time-aware check.
**How to check:**
- Confirm that `stratify=` appears in the code.
- For time-series or episodic data (industrial control system (ICS) testbeds, captures with attack windows):
  - Measure how many test attack records share an attack episode with training records.
  - Re-run with a chronological or episode-held-out split.

**Examples:** Benchmark A (split not stratified in the code; time split showed no leakage: Flag). Benchmark C (random split inside attack episodes; about 80% of test attacks share an episode with training records; on unseen episodes, AUC falls from 0.83 to 0.54: Fail).

### L6 Column loss (soft)
**Check:** The number of features is logged after each preprocessing step, and nothing disappears without notice.
**Required artifact:** A shape log covering rows and columns after loading, cleaning, encoding, selection and sampling.
**How to check:**
- Look for type filters that drop one-hot columns, inner joins that drop rows, and `read_csv` calls without `header=None` on files that have no header row (the first record is lost).
- Compare the row counts with the dataset documentation.

**Examples:** Benchmark A (one-hot columns dropped: Flag). Benchmark B (first record read as the header: Flag).

---

## 3. Modeling gates (M1–M4)

### M1 Encoding range (hard)
**Check:** The rotation range of the angle encoding is stated as a kernel bandwidth and tuned on training data. Inputs are bounded (clipped at test time), and wraparound under the gate's own scaling is checked. For Qiskit Z and ZZ maps, `P(2x)` gives the single-qubit rotation a period of π in x.
**Required artifact:** A rotation-span report giving the input range, the gate argument range, the number of wraparounds and the tuning grid.
**How to check:**
- Trace the scaler that feeds the embedding.
- Unbounded inputs (StandardScaler or raw PCA outputs) fail this check.
- A [0, 2π] range under `P(2x)` wraps twice.
- An untuned [0, 1] range fails this check, because the bandwidth was never chosen.
- Re-run with a small bandwidth grid chosen by cross-validation on training data.

**Examples:** A ([0, 2π] wraps: Fail). B (unbounded standardized inputs: Fail). C (untuned [0, 1]; AUC ranges 0.52–0.60 across bandwidths: Fail).

### M2 Circuit declaration (hard)
**Check:** The repetitions, entanglement, width and shot count are read from the circuit that was actually built and run. Every feature map listed as an alternative is confirmed to be a different circuit. Every kernel is a valid fidelity kernel.
**Required artifact:** A printed circuit summary for each map (or the circuit drawing), plus a kernel validity check.
**How to check:**
- Print `circuit.decompose()`, its depth and its parameters.
- Compare the statevectors of maps listed as different: identical states, or identical metrics to many digits, mean they are the same circuit. Qiskit's default `PauliFeatureMap` (Z, ZZ) is the ZZ map.
- On about 30 random inputs, check that K(x, x) = 1, that the kernel matrix is symmetric, and that its smallest eigenvalue is ≥ −1e−8.
- Confirm that the shot count stated in the paper is set on the device or sampler. An exact (statevector) run with "1024 shots" in the text fails this check.

**Examples:** A and C (Pauli and ZZ maps reported as different but are one circuit: Fail). B (the "angle" kernel combines IQP(x) with Angle†(x′); K(x, x) averages 0.07 and the matrix is not positive semidefinite (PSD); shots are stated but exact simulation was run: Fail).

### M3 Deterministic selection (hard)
**Check:** Feature selection returns exactly k features for every seed, with no wall-clock budget, timeout or early stop that depends on machine speed.
**Required artifact:** The list of selected features for each seed.
**How to check:** Look for `time.time()` budgets, `max_time`, and early-stop conditions inside selection loops. Log the number of features selected for each seed.

**Example:** A (QKA-MI stopped by a 180 s budget after two features: Fail).

### M4 Seeds and tuning (hard)
**Check:**
- Every random step is seeded: sampling, splitting, resampling, selection, initialization.
- Every model, quantum and classical, is tuned on training data only, using stated grids.
- There are at least 10 seeds.
- Comparisons use paired tests with correction for multiple comparisons.

**Required artifact:** The seed list, the search grids for every model, and the summary table with paired p-values.
**How to check:**
- Look for a single `random_state=42` with no seed loop.
- Look for hyperparameters (C, γ, bandwidth) chosen by test-set score. Picking the best value out of a grid on the test set counts as tuning on the test set.
- Look for classical baselines left at defaults (`C=1`, `gamma='scale'`) when the paper says they were tuned.
- Report the spread of the same configuration across seeds.

**Examples:** A (one run, untuned baselines; one configuration ranges 0.65–0.92: Fail). B (one seed, C picked on 60 test records; spread 0.90–0.97 over 10 seeds: Fail). C (5 seeds, untuned QSVM: Fail).

---

## 4. Evaluation gates (E1–E3)

### E1 Like-for-like comparison (soft)
**Check:** Settings that are compared differ only in the factor being studied: the same features, qubit count or dimensionality, split, test set, preprocessing and tuning effort.
**Required artifact:** A settings table with one row per compared model, listing features, width, train and test sizes, and grid.
**How to check:**
- Compare the test-set sizes and feature sets behind each row of the results table.
- Watch for quantum models tested on 60 records while classical models are tested on 5,000.
- Watch for a quantum model with k features compared with a classical model with a different k.
- Watch for headline gains that come from a single seed.

**Verdict:** Fail if the difference is undisclosed and makes a reported comparison invalid. Flag if it is disclosed in the paper or only part of the comparison is affected.

**Examples:** A (a 12-point gain becomes 0.05–0.08: Fail). B (QSVM tested on 60–80 records, others on about 5,000, although the paper says the splits are identical: Fail). C (comparisons mostly use the same 8 features; the +10.8% headline comes from one seed: Flag).

### E2 Cost reporting (soft)
**Check:** Inference cost is reported in units that carry over to hardware: kernel entries per classified flow (number of support vectors) times shots per entry. Simulator wall-clock time is not a substitute.
**Required artifact:** A cost table giving support vectors, shots per entry, shots per flow and time per flow at a stated shot rate.
**How to check:** Find out what the reported "training time" measures. Timing only `SVC.fit` on a precomputed kernel leaves out kernel evaluation.

**Examples:** A (simulator time reported as cost: Fail). B (QSVM time leaves out kernel evaluation: Fail). C (gram sizes, times and circuit depth reported, but not shots per flow: Flag).

### E3 Reproduction (soft)
**Check:** The results can be re-run from scratch in a pinned environment, using only the released material.
**Required artifact:** A reproduction folder containing pinned versions, the run commands, and a comparison of re-run against reported values.
**Verdict:**
- **Pass** if a pinned re-run matches the reported results within the seed spread.
- **Flag** if the code is released and re-runs after small edits (paths), but the environment is not pinned or some scripts are missing.
- **Fail** if no code is available, or the re-run does not match and the difference cannot be explained.

**Examples:** A (reproduced afterwards under the pinned environment: Pass). B and C (code released, environment unpinned; C does not include the scripts behind its tests and ablations: Flag).

---

## 5. Recording template

| Gate | Verdict | Evidence (file:line or log) | Measured effect (published → fixed, 10 seeds) |
|---|---|---|---|
| L1 | | | |
| … | | | |
| E3 | | | |

Report the totals as **checklist items met (of 14) / gates failed (of 13)**.

## 6. Case-study results

| | A (ICECET 2026) | B (Kerkatou et al. 2026) | C (Badami 2026) |
|---|---|---|---|
| Checklist items met | 9/14 | 10/14 | 11/14 |
| Gates failed / flagged / passed | 9 / 3 / 1 | 7 / 3 / 3 | 5 / 3 / 5 |

The full reports are in `external_audit/` (B and C) and in the arXiv correction note to arXiv:2610.33119 (A).
