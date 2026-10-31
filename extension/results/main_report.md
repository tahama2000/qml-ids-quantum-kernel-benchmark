# Results report: plan 'main'

8400 rows, 840 configurations, seeds per configuration: 10–10.

## Best configuration per dataset, protocol, encoding and model (F1 mean ± 95% CI)

| Dataset | Protocol | Encoding | Model | Map | Selector | k | Sample | F1 |
|---|---|---|---|---|---|---|---|---|
| MQTTEEB-D2025 | clean | bounded | PegasosQSVC | Z | RF | 3 | 0.002 | 0.924 ± 0.042 |
| MQTTEEB-D2025 | clean | bounded | QSVM | Z | RF | 3 | 0.005 | 0.950 ± 0.009 |
| MQTTEEB-D2025 | clean | bounded | RandomForest | Classical | QKA_MI | 5 | 0.005 | 0.971 ± 0.005 |
| MQTTEEB-D2025 | clean | bounded | SVM | Classical | RF | 5 | 0.002 | 0.944 ± 0.017 |
| MQTTEEB-D2025 | clean | twopi | PegasosQSVC | Z | RF | 5 | 0.002 | 0.877 ± 0.052 |
| MQTTEEB-D2025 | clean | twopi | QSVM | Z | RF | 3 | 0.005 | 0.946 ± 0.012 |
| MQTTEEB-D2025 | clean | twopi | RandomForest | Classical | RF | 3 | 0.005 | 0.970 ± 0.006 |
| MQTTEEB-D2025 | clean | twopi | SVM | Classical | QKA_MI | 3 | 0.005 | 0.949 ± 0.008 |
| MQTTEEB-D2025 | published | twopi | PegasosQSVC | Z | RF | 3 | 0.005 | 1.000 ± 0.001 |
| MQTTEEB-D2025 | published | twopi | QSVM | Z | RF | 3 | 0.002 | 1.000 ± 0.000 |
| MQTTEEB-D2025 | published | twopi | RandomForest | Classical | RF | 3 | 0.002 | 1.000 ± 0.000 |
| MQTTEEB-D2025 | published | twopi | SVM | Classical | RF | 3 | 0.005 | 1.000 ± 0.000 |
| NSL-KDD | clean | bounded | PegasosQSVC | HybridPauliZZ | QKA_MI | 5 | 0.05 | 0.870 ± 0.030 |
| NSL-KDD | clean | bounded | QSVM | HybridPauliZZ | QKA_MI | 5 | 0.1 | 0.901 ± 0.009 |
| NSL-KDD | clean | bounded | RandomForest | Classical | RF | 3 | 0.1 | 0.960 ± 0.007 |
| NSL-KDD | clean | bounded | SVM | Classical | QKA_MI | 5 | 0.01 | 0.912 ± 0.032 |
| NSL-KDD | clean | twopi | PegasosQSVC | Z | QKA_MI | 5 | 0.1 | 0.848 ± 0.015 |
| NSL-KDD | clean | twopi | QSVM | ZZ | QKA_MI | 5 | 0.1 | 0.887 ± 0.019 |
| NSL-KDD | clean | twopi | RandomForest | Classical | RF | 3 | 0.1 | 0.960 ± 0.007 |
| NSL-KDD | clean | twopi | SVM | Classical | QKA_MI | 5 | 0.1 | 0.888 ± 0.012 |
| NSL-KDD | published | twopi | PegasosQSVC | ZZ | QKA_MI | 5 | 0.01 | 0.819 ± 0.052 |
| NSL-KDD | published | twopi | QSVM | Pauli | QKA_MI | 5 | 0.1 | 0.886 ± 0.012 |
| NSL-KDD | published | twopi | RandomForest | Classical | RF | 3 | 0.1 | 0.962 ± 0.008 |
| NSL-KDD | published | twopi | SVM | Classical | QKA_MI | 5 | 0.1 | 0.885 ± 0.009 |
| UNSW-NB15 | clean | bounded | PegasosQSVC | HybridPauliZZ | RF | 3 | 0.002 | 0.847 ± 0.039 |
| UNSW-NB15 | clean | bounded | QSVM | Pauli | RF | 3 | 0.005 | 0.907 ± 0.010 |
| UNSW-NB15 | clean | bounded | RandomForest | Classical | QKA_MI | 3 | 0.005 | 0.901 ± 0.014 |
| UNSW-NB15 | clean | bounded | SVM | Classical | RF | 3 | 0.005 | 0.907 ± 0.011 |
| UNSW-NB15 | clean | twopi | PegasosQSVC | ZZ | RF | 3 | 0.002 | 0.850 ± 0.043 |
| UNSW-NB15 | clean | twopi | QSVM | Z | RF | 3 | 0.005 | 0.909 ± 0.010 |
| UNSW-NB15 | clean | twopi | RandomForest | Classical | QKA_MI | 3 | 0.002 | 0.894 ± 0.018 |
| UNSW-NB15 | clean | twopi | SVM | Classical | RF | 3 | 0.005 | 0.907 ± 0.011 |
| UNSW-NB15 | published | twopi | PegasosQSVC | HybridPauliZZ | RF | 3 | 0.005 | 0.836 ± 0.019 |
| UNSW-NB15 | published | twopi | QSVM | Pauli | RF | 3 | 0.005 | 0.909 ± 0.016 |
| UNSW-NB15 | published | twopi | RandomForest | Classical | RF | 3 | 0.005 | 0.936 ± 0.014 |
| UNSW-NB15 | published | twopi | SVM | Classical | RF | 3 | 0.005 | 0.880 ± 0.011 |

## best quantum vs RF

84 paired comparisons; 50 significant at p < 0.05; median F1 change -0.024.

## leakage fix (published -> clean, [0,2pi])

280 paired comparisons; 86 significant at p < 0.05; median F1 change -0.007.

## encoding fix (clean [0,2pi] -> clean [0,pi/2]+clip)

280 paired comparisons; 72 significant at p < 0.05; median F1 change +0.010.

## QKA-MI feature stability across encodings

Mean Jaccard overlap of selected features ([0,2π] vs [0,π/2]): 0.33 (1.0 = identical selections).

## Encoding audit (quantum rows)

| Dataset | Protocol | Encoding | Rotation span (rad) | Alias rate | Test out of range | KTA |
|---|---|---|---|---|---|---|
| MQTTEEB-D2025 | clean | bounded | 3.14 | 0.0002 | 0.000 | 0.131 |
| MQTTEEB-D2025 | clean | twopi | 12.57 | 0.0000 | 0.002 | 0.104 |
| MQTTEEB-D2025 | published | twopi | 12.45 | 0.0000 | 0.001 | 0.407 |
| NSL-KDD | clean | bounded | 3.14 | 0.0030 | 0.000 | 0.540 |
| NSL-KDD | clean | twopi | 12.57 | 0.0018 | 0.003 | 0.410 |
| NSL-KDD | published | twopi | 12.47 | 0.0015 | 0.003 | 0.425 |
| UNSW-NB15 | clean | bounded | 3.14 | 0.0001 | 0.000 | 0.379 |
| UNSW-NB15 | clean | twopi | 12.57 | 0.0000 | 0.003 | 0.335 |
| UNSW-NB15 | published | twopi | 12.49 | 0.0000 | 0.002 | 0.283 |
