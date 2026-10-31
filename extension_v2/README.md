# O3 extension — version 2 (26 Oct 2026)

Adds three experiments to the v1 extension. The v1 code and results in `../extension/` are unchanged;
`sweeps.py` imports `o3ext.py` and `run_benchmark.py` from there.

| Experiment | What varies | Output |
|---|---|---|
| Qubit sweep (circuit size) | q = 3, 5, 7, 9, 11 PCA components = qubits; Z map; ZZ map full/linear entanglement, 1/2 reps | `results/sweeps_grid.csv` |
| Bandwidth sweep (encoding range) | theta_max = pi/8, pi/4, pi/2, pi, 2pi | same file |
| Tuned baselines + extended metrics | 3-fold CV tuning of QSVM (theta, map, C), SVM (C, gamma), RF, HistGradientBoosting at q = 3 and 7; AUPRC, FPR, per-attack-category recall, shots per flow | `results/sweeps_tuned.csv`, `sweeps_tuned_tests.csv` |

Setup: leak-free protocol from v1 (gates L1–L5); 5% NSL-KDD, 0.5% UNSW-NB15 and MQTTEEB-D2025; Random Forest
selects 11 features on the training set; SMOTE on training data; PCA to q components; 10 seeds (42, 1–9).
Statevectors are computed analytically (Hadamard layer + diagonal phase per repetition) and match Qiskit to
3e-15 (`python sweeps.py --check`).

Run: `python sweeps.py` (~45 min on 2 cores, resumable), then `python analyze_sweeps.py`
→ `results/sweeps_report.md`, `figures/v2_fig_qubits`, `figures/v2_fig_bandwidth`.

## Key results (mean of 10 seeds)

1. **Bandwidth:** at 3 qubits, theta_max = 2pi gave the lowest F1 on NSL-KDD (0.898 vs 0.922 at pi/2) and MQTTEEB
   (0.887 vs 0.926). At 11 qubits 2pi was best on MQTTEEB (0.949). KTA peaks at pi/4 on NSL-KDD and UNSW-NB15.
   Tuned QSVM chose pi/2 or below in 15/20 NSL-KDD runs, pi or 2pi in 13/20 UNSW runs, 2pi in 12/20 MQTTEEB runs.
   → the range is a bandwidth to tune; [0, pi/2] is the safer default at low width.
2. **Qubits:** QSVM (ZZ full, pi/2) gains +0.013 (NSL-KDD, p=0.02) and +0.019 (UNSW, p=0.04) from 3 to 9 qubits;
   flat on MQTTEEB. Part of the gain is information: 3 components keep 59–81% of the variance, 9 keep 97–100%.
   Full entanglement > linear > Z on NSL/UNSW; Z is the best quantum map on MQTTEEB from 7 qubits.
3. **Concentration:** mild up to 11 qubits. NSL-KDD, 2pi: off-diagonal mean 0.26 → 0.06, variance 0.068 → 0.019;
   at pi/8 the variance stays 0.078–0.12.
4. **Tuned comparison:** NSL-KDD — QSVM indistinguishable from tuned SVM/RF/HGB (all p > 0.3).
   UNSW — within 0.015 (significant only vs SVM and RF at 3 qubits). MQTTEEB — RF/HGB lead by 0.010–0.029
   (p ≤ 0.02); QSVM beats tuned SVM at 7 qubits (+0.007, p = 0.04). Tree models have the best AUPRC everywhere.
5. **Cost:** tuned QSVM needs 44,700–147,700 shots per classified flow (support vectors × 256).
