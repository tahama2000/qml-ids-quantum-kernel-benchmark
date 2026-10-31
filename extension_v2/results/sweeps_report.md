# O3 extension v2: qubit sweep, bandwidth sweep, tuned baselines

## 1. Qubit sweep (θ_max = π/2, C = 1): mean F1 over 10 seeds

| Dataset | Model / map | q=3 | q=5 | q=7 | q=9 | q=11 |
|---|---|---|---|---|---|---|
| NSL-KDD | QSVM Z | 0.917 | 0.918 | 0.923 | 0.923 | 0.923 |
| NSL-KDD | QSVM ZZ-full-r1 | 0.922 | 0.927 | 0.933 | 0.936 | 0.932 |
| NSL-KDD | QSVM ZZ-linear-r1 | 0.913 | 0.925 | 0.925 | 0.923 | 0.921 |
| NSL-KDD | QSVM ZZ-full-r2 | 0.910 | 0.925 | 0.929 | 0.935 | 0.922 |
| NSL-KDD | QSVM ZZ-linear-r2 | 0.908 | 0.923 | 0.921 | 0.926 | 0.924 |
| NSL-KDD | SVM | 0.924 | 0.924 | 0.924 | 0.926 | 0.927 |
| NSL-KDD | RandomForest | 0.933 | 0.943 | 0.943 | 0.942 | 0.940 |
| UNSW-NB15 | QSVM Z | 0.871 | 0.881 | 0.880 | 0.876 | 0.884 |
| UNSW-NB15 | QSVM ZZ-full-r1 | 0.891 | 0.899 | 0.905 | 0.910 | 0.910 |
| UNSW-NB15 | QSVM ZZ-linear-r1 | 0.888 | 0.897 | 0.897 | 0.903 | 0.903 |
| UNSW-NB15 | QSVM ZZ-full-r2 | 0.874 | 0.901 | 0.909 | 0.908 | 0.904 |
| UNSW-NB15 | QSVM ZZ-linear-r2 | 0.879 | 0.899 | 0.901 | 0.908 | 0.908 |
| UNSW-NB15 | SVM | 0.889 | 0.894 | 0.898 | 0.905 | 0.905 |
| UNSW-NB15 | RandomForest | 0.901 | 0.912 | 0.909 | 0.911 | 0.915 |
| MQTTEEB-D2025 | QSVM Z | 0.913 | 0.926 | 0.943 | 0.943 | 0.943 |
| MQTTEEB-D2025 | QSVM ZZ-full-r1 | 0.926 | 0.917 | 0.919 | 0.924 | 0.926 |
| MQTTEEB-D2025 | QSVM ZZ-linear-r1 | 0.917 | 0.910 | 0.926 | 0.928 | 0.928 |
| MQTTEEB-D2025 | QSVM ZZ-full-r2 | 0.935 | 0.918 | 0.931 | 0.927 | 0.922 |
| MQTTEEB-D2025 | QSVM ZZ-linear-r2 | 0.935 | 0.912 | 0.920 | 0.920 | 0.920 |
| MQTTEEB-D2025 | SVM | 0.936 | 0.936 | 0.937 | 0.937 | 0.937 |
| MQTTEEB-D2025 | RandomForest | 0.966 | 0.966 | 0.967 | 0.968 | 0.968 |

## 2. Kernel concentration: variance of off-diagonal entries (ZZ-full-r1)

| Dataset | θ_max | q=3 | q=5 | q=7 | q=9 | q=11 |
|---|---|---|---|---|---|---|
| NSL-KDD | pi/8 | 9.19e-02 | 1.19e-01 | 1.08e-01 | 8.62e-02 | 7.82e-02 |
| NSL-KDD | pi/4 | 1.35e-01 | 1.27e-01 | 9.32e-02 | 7.31e-02 | 6.32e-02 |
| NSL-KDD | pi/2 | 1.05e-01 | 9.62e-02 | 6.81e-02 | 5.31e-02 | 4.37e-02 |
| NSL-KDD | pi | 9.02e-02 | 7.35e-02 | 4.87e-02 | 3.67e-02 | 2.90e-02 |
| NSL-KDD | 2pi | 6.80e-02 | 5.10e-02 | 3.10e-02 | 2.30e-02 | 1.85e-02 |
| UNSW-NB15 | pi/8 | 3.31e-02 | 5.64e-02 | 6.66e-02 | 6.55e-02 | 6.16e-02 |
| UNSW-NB15 | pi/4 | 8.83e-02 | 9.14e-02 | 8.22e-02 | 6.85e-02 | 5.74e-02 |
| UNSW-NB15 | pi/2 | 1.07e-01 | 8.73e-02 | 6.73e-02 | 5.38e-02 | 4.25e-02 |
| UNSW-NB15 | pi | 8.69e-02 | 6.61e-02 | 4.83e-02 | 3.86e-02 | 2.98e-02 |
| UNSW-NB15 | 2pi | 6.72e-02 | 4.95e-02 | 3.55e-02 | 2.71e-02 | 2.02e-02 |
| MQTTEEB-D2025 | pi/8 | 8.86e-02 | 1.01e-01 | 1.02e-01 | 1.01e-01 | 9.93e-02 |
| MQTTEEB-D2025 | pi/4 | 1.29e-01 | 9.93e-02 | 8.92e-02 | 8.10e-02 | 7.49e-02 |
| MQTTEEB-D2025 | pi/2 | 1.21e-01 | 7.50e-02 | 6.52e-02 | 5.61e-02 | 4.99e-02 |
| MQTTEEB-D2025 | pi | 1.17e-01 | 5.79e-02 | 5.01e-02 | 3.89e-02 | 3.31e-02 |
| MQTTEEB-D2025 | 2pi | 7.47e-02 | 2.80e-02 | 2.69e-02 | 2.34e-02 | 2.12e-02 |

## 3. Bandwidth: best θ_max per qubit count (QSVM, best map, mean F1)

| Dataset | q=3 | q=5 | q=7 | q=9 | q=11 |
|---|---|---|---|---|---|
| NSL-KDD | 0.922 (pi/2, ZZ-full-r1) | 0.929 (pi, ZZ-linear-r2) | 0.933 (pi/2, ZZ-full-r1) | 0.937 (pi, ZZ-full-r1) | 0.932 (pi/2, ZZ-full-r1) |
| UNSW-NB15 | 0.891 (pi/2, ZZ-full-r1) | 0.903 (2pi, ZZ-full-r2) | 0.911 (pi, ZZ-full-r1) | 0.913 (2pi, ZZ-linear-r1) | 0.913 (pi/4, ZZ-full-r1) |
| MQTTEEB-D2025 | 0.942 (pi, ZZ-linear-r2) | 0.933 (2pi, ZZ-full-r2) | 0.944 (2pi, ZZ-full-r2) | 0.951 (2pi, ZZ-full-r2) | 0.956 (2pi, ZZ-full-r2) |

## 4. Tuned models (3-fold CV on training data), mean over 10 seeds

| Dataset | q | Model | f1 | auprc | fpr | macro_attack_recall | worst_attack_recall | QSVM − model F1 (p) |
|---|---|---|---|---|---|---|---|---|
| NSL-KDD | 3 | QSVM (tuned) | 0.925 | 0.957 | 0.077 | 0.890 | 0.639 | — |
| NSL-KDD | 3 | SVM (tuned) | 0.928 | 0.954 | 0.069 | 0.894 | 0.656 | -0.003 (0.66) |
| NSL-KDD | 3 | RandomForest (tuned) | 0.931 | 0.984 | 0.064 | 0.896 | 0.677 | -0.006 (0.32) |
| NSL-KDD | 3 | HistGradientBoosting (tuned) | 0.927 | 0.982 | 0.070 | 0.887 | 0.682 | -0.002 (0.73) |
| NSL-KDD | 7 | QSVM (tuned) | 0.939 | 0.957 | 0.064 | 0.916 | 0.729 | — |
| NSL-KDD | 7 | SVM (tuned) | 0.937 | 0.976 | 0.066 | 0.911 | 0.711 | +0.002 (0.46) |
| NSL-KDD | 7 | RandomForest (tuned) | 0.941 | 0.986 | 0.057 | 0.914 | 0.748 | -0.002 (0.62) |
| NSL-KDD | 7 | HistGradientBoosting (tuned) | 0.939 | 0.987 | 0.064 | 0.919 | 0.740 | +0.000 (0.92) |
| NSL-KDD | | QSVM choices | θ: {'pi/2': np.int64(7), 'pi/4': np.int64(6), 'pi': np.int64(4), 'pi/8': np.int64(2), '2pi': np.int64(1)}; map: {'ZZ-full-r1': np.int64(8), 'ZZ-linear-r1': np.int64(6), 'ZZ-full-r2': np.int64(3), 'Z': np.int64(3)} | | | | | |
| UNSW-NB15 | 3 | QSVM (tuned) | 0.891 | 0.931 | 0.175 | 0.859 | 0.569 | — |
| UNSW-NB15 | 3 | SVM (tuned) | 0.906 | 0.942 | 0.157 | 0.882 | 0.616 | -0.015 (0.0098) |
| UNSW-NB15 | 3 | RandomForest (tuned) | 0.899 | 0.967 | 0.166 | 0.872 | 0.605 | -0.008 (0.037) |
| UNSW-NB15 | 3 | HistGradientBoosting (tuned) | 0.903 | 0.967 | 0.157 | 0.877 | 0.635 | -0.012 (0.11) |
| UNSW-NB15 | 7 | QSVM (tuned) | 0.909 | 0.952 | 0.151 | 0.885 | 0.616 | — |
| UNSW-NB15 | 7 | SVM (tuned) | 0.913 | 0.951 | 0.159 | 0.898 | 0.652 | -0.004 (0.084) |
| UNSW-NB15 | 7 | RandomForest (tuned) | 0.906 | 0.974 | 0.166 | 0.887 | 0.620 | +0.004 (0.49) |
| UNSW-NB15 | 7 | HistGradientBoosting (tuned) | 0.913 | 0.974 | 0.151 | 0.894 | 0.647 | -0.003 (0.38) |
| UNSW-NB15 | | QSVM choices | θ: {'pi': np.int64(7), '2pi': np.int64(6), 'pi/2': np.int64(5), 'pi/4': np.int64(2)}; map: {'ZZ-full-r2': np.int64(7), 'ZZ-full-r1': np.int64(6), 'ZZ-linear-r2': np.int64(5), 'ZZ-linear-r1': np.int64(2)} | | | | | |
| MQTTEEB-D2025 | 3 | QSVM (tuned) | 0.938 | 0.984 | 0.426 | 0.890 | 0.648 | — |
| MQTTEEB-D2025 | 3 | SVM (tuned) | 0.942 | 0.979 | 0.296 | 0.846 | 0.579 | -0.004 (0.7) |
| MQTTEEB-D2025 | 3 | RandomForest (tuned) | 0.966 | 0.995 | 0.174 | 0.907 | 0.599 | -0.029 (0.002) |
| MQTTEEB-D2025 | 3 | HistGradientBoosting (tuned) | 0.966 | 0.994 | 0.193 | 0.912 | 0.647 | -0.028 (0.002) |
| MQTTEEB-D2025 | 7 | QSVM (tuned) | 0.958 | 0.990 | 0.104 | 0.841 | 0.515 | — |
| MQTTEEB-D2025 | 7 | SVM (tuned) | 0.951 | 0.983 | 0.241 | 0.850 | 0.579 | +0.007 (0.037) |
| MQTTEEB-D2025 | 7 | RandomForest (tuned) | 0.968 | 0.995 | 0.133 | 0.902 | 0.580 | -0.010 (0.0059) |
| MQTTEEB-D2025 | 7 | HistGradientBoosting (tuned) | 0.968 | 0.995 | 0.167 | 0.907 | 0.630 | -0.010 (0.02) |
| MQTTEEB-D2025 | | QSVM choices | θ: {'2pi': np.int64(12), 'pi/2': np.int64(7), 'pi/4': np.int64(1)}; map: {'ZZ-full-r1': np.int64(9), 'ZZ-full-r2': np.int64(9), 'ZZ-linear-r2': np.int64(2)} | | | | | |

## 5. Inference cost of tuned QSVM (support vectors × 256 shots per flow)

| Dataset | q | support vectors | shots per flow |
|---|---|---|---|
| MQTTEEB-D2025 | 3 | 577 | 147,712 |
| MQTTEEB-D2025 | 7 | 485 | 124,160 |
| NSL-KDD | 3 | 174 | 44,672 |
| NSL-KDD | 7 | 181 | 46,438 |
| UNSW-NB15 | 3 | 373 | 95,565 |
| UNSW-NB15 | 7 | 324 | 82,867 |
