# RBF SVM re-tune with a wider bandwidth grid

The tuned RBF SVM in `extension_v2` searched C in {0.1, 1, 10, 100} and gamma in {scale, 0.1, 1, 10}.
Because the quantum kernel can reach high-frequency kernels through theta_max = 2 pi, `svm_recheck.py`
re-tunes the SVM on exactly the same splits, seeds, PCA components and 3-fold CV with
gamma in {scale, 0.1, 1, 10, 30, 100, 300}. `compare.py` compares it with the tuned QSVM, Random Forest and
gradient boosting.

Only MQTTEEB-D2025 changes: SVM F1 0.942 -> 0.953 (3 qubits) and 0.951 -> 0.960 (7 qubits). The QSVM-minus-SVM
difference at 7 qubits becomes -0.002 (p = 0.56). NSL-KDD and UNSW-NB15 are unchanged (the optimum was already
inside the narrower grid).

Files: `svm_recheck.py`, `compare.py`, `results/svm_recheck_per_seed.csv`, `results/svm_recheck_summary.csv`,
`results/v2_sweeps_tuned_input.csv`.
