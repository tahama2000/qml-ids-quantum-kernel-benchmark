# Results report: plan 'pegasos_steps'

1200 rows, 120 configurations, seeds per configuration: 10–10.

## Best configuration per dataset, protocol, encoding and model (F1 mean ± 95% CI)

| Dataset | Protocol | Encoding | Model | Map | Selector | k | Sample | F1 |
|---|---|---|---|---|---|---|---|---|
| MQTTEEB-D2025 | clean | bounded | PegasosQSVC | Pauli | RF | 3 | 0.005 | 0.876 ± 0.059 |
| MQTTEEB-D2025 | clean | bounded | QSVM | Z | RF | 3 | 0.005 | 0.950 ± 0.009 |
| MQTTEEB-D2025 | clean | bounded | RandomForest | Classical | RF | 3 | 0.005 | 0.970 ± 0.006 |
| MQTTEEB-D2025 | clean | bounded | SVM | Classical | RF | 3 | 0.005 | 0.936 ± 0.010 |
| MQTTEEB-D2025 | clean | twopi | PegasosQSVC | Z | RF | 3 | 0.005 | 0.888 ± 0.057 |
| MQTTEEB-D2025 | clean | twopi | QSVM | Z | RF | 3 | 0.005 | 0.946 ± 0.012 |
| MQTTEEB-D2025 | clean | twopi | RandomForest | Classical | RF | 3 | 0.005 | 0.970 ± 0.006 |
| MQTTEEB-D2025 | clean | twopi | SVM | Classical | QKA_MI | 3 | 0.005 | 0.949 ± 0.008 |
| NSL-KDD | clean | bounded | PegasosQSVC | ZZ | QKA_MI | 3 | 0.1 | 0.886 ± 0.018 |
| NSL-KDD | clean | bounded | QSVM | Z | QKA_MI | 3 | 0.1 | 0.895 ± 0.012 |
| NSL-KDD | clean | bounded | RandomForest | Classical | RF | 3 | 0.1 | 0.960 ± 0.007 |
| NSL-KDD | clean | bounded | SVM | Classical | QKA_MI | 3 | 0.1 | 0.899 ± 0.013 |
| NSL-KDD | clean | twopi | PegasosQSVC | HybridPauliZZ | QKA_MI | 3 | 0.1 | 0.829 ± 0.036 |
| NSL-KDD | clean | twopi | QSVM | HybridPauliZZ | QKA_MI | 3 | 0.1 | 0.873 ± 0.009 |
| NSL-KDD | clean | twopi | RandomForest | Classical | RF | 3 | 0.1 | 0.960 ± 0.007 |
| NSL-KDD | clean | twopi | SVM | Classical | QKA_MI | 3 | 0.1 | 0.879 ± 0.009 |
| UNSW-NB15 | clean | bounded | PegasosQSVC | HybridPauliZZ | RF | 3 | 0.005 | 0.886 ± 0.027 |
| UNSW-NB15 | clean | bounded | QSVM | Pauli | RF | 3 | 0.005 | 0.907 ± 0.010 |
| UNSW-NB15 | clean | bounded | RandomForest | Classical | QKA_MI | 3 | 0.005 | 0.901 ± 0.014 |
| UNSW-NB15 | clean | bounded | SVM | Classical | RF | 3 | 0.005 | 0.907 ± 0.011 |
| UNSW-NB15 | clean | twopi | PegasosQSVC | Z | RF | 3 | 0.005 | 0.888 ± 0.024 |
| UNSW-NB15 | clean | twopi | QSVM | Z | RF | 3 | 0.005 | 0.909 ± 0.010 |
| UNSW-NB15 | clean | twopi | RandomForest | Classical | RF | 3 | 0.005 | 0.892 ± 0.007 |
| UNSW-NB15 | clean | twopi | SVM | Classical | RF | 3 | 0.005 | 0.907 ± 0.011 |

## best quantum vs RF

12 paired comparisons; 9 significant at p < 0.05; median F1 change -0.014.

## encoding fix (clean [0,2pi] -> clean [0,pi/2]+clip)

60 paired comparisons; 17 significant at p < 0.05; median F1 change +0.000.

## QKA-MI feature stability across encodings

Mean Jaccard overlap of selected features ([0,2π] vs [0,π/2]): 0.32 (1.0 = identical selections).

## Encoding audit (quantum rows)

| Dataset | Protocol | Encoding | Rotation span (rad) | Alias rate | Test out of range | KTA |
|---|---|---|---|---|---|---|
| MQTTEEB-D2025 | clean | bounded | 3.14 | 0.0001 | 0.000 | 0.127 |
| MQTTEEB-D2025 | clean | twopi | 12.57 | 0.0000 | 0.002 | 0.101 |
| NSL-KDD | clean | bounded | 3.14 | 0.0118 | 0.000 | 0.483 |
| NSL-KDD | clean | twopi | 12.57 | 0.0000 | 0.000 | 0.403 |
| UNSW-NB15 | clean | bounded | 3.14 | 0.0003 | 0.000 | 0.399 |
| UNSW-NB15 | clean | twopi | 12.57 | 0.0000 | 0.002 | 0.351 |
