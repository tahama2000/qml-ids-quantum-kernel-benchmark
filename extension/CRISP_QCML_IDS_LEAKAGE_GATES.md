# CRISP-QCML-IDS: leakage and reproducibility gates (additions from the O3 audit)

Each gate is a yes/no check with a required artifact. A pipeline that fails a gate does not move to
the next phase. The right-hand column gives the audit evidence that motivates the gate (audit finding IDs F1–F18 of the use case in the paper).

## Data Understanding / Data Preparation

| Gate | Check (human review required) | Artifact | Evidence |
|---|---|---|---|
| L1 Label review | No selected feature is the label, a re-coding of it, or derived from it (e.g. `Attack_Binary`, `attack_cat`). | List of selected features signed off by a reviewer | F3: MQTTEEB label column ranked #1 by RF |
| L2 Identifier and ordering review | No row index, record ID, flow ID, IP/port, or capture-file position is a feature. Check whether the file is sorted by class; if so, any position-like column is a leak. | Column inventory with a role per column; class-by-position plot | F4: UNSW `id` alone gives F1 = 0.83 |
| L3 Timestamp review | Raw timestamps and epoch values are removed or converted to relative quantities. | Column inventory | F5 |
| L4 Fit on training data only | Selection, SMOTE, scaling, PCA and angle scaling are fitted after the split, on training data only. The test set is never resampled. | Pipeline diagram showing the split before every fitted step | F2; Fig. 4 (up to −0.12 F1 on imbalanced data) |
| L5 Split design | Stratified split on the label (and on attack category for sampling). For time-stamped data, a time-aware split is run as a check. | Split description; time-split result | F7, F12 |
| L6 Silent column loss | Feature count after each preprocessing step is logged and compared with the design (e.g. one-hot columns dropped by a dtype filter). | Shape log | F6 |

## Modeling

| Gate | Check | Artifact | Evidence |
|---|---|---|---|
| M1 Encoding range | Angle range after the gate's own multiplier (P(2x)) stays below π; the bounded encoding [0, π/2] + clip is the default. | Rotation-span report | F1; Fig. 2 |
| M2 Circuit declaration | Feature-map depth (reps), entanglement and qubit count are read from the built circuit, not from the intended settings, and distinct maps are verified to be distinct circuits. | Printed circuit summary per map | F13, F14 |
| M3 Deterministic selection | Feature selection has no wall-clock budget; it returns exactly k features or fails loudly. | Selected-feature list per seed | F15 |
| M4 Seeds | Every stochastic step is seeded; results are reported over ≥ 10 seeds with 95 % CI and paired tests. | Seed list; summary table | F8, F16 |

## Evaluation

| Gate | Check | Artifact | Evidence |
|---|---|---|---|
| E1 Like-for-like comparison | Compared configurations differ only in the factor under study (same qubits, PCA size, sample, split). | Comparison table with all settings | F15 (+12 compares 3 vs 2 qubits) |
| E2 Cost reporting | Report kernel entries (circuits) and shots, not simulator wall-clock time, when comparing quantum with classical cost. | Cost table | F18 |
| E3 Reproduction | The pinned environment and one configuration are re-run from scratch before submission. | `repro/` folder | F17 |
