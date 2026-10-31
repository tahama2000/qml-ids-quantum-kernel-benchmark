# Results report: plan 'ablation_order'

2800 rows, 280 configurations, seeds per configuration: 10–10.

## Best configuration per dataset, protocol, encoding and model (F1 mean ± 95% CI)

| Dataset | Protocol | Encoding | Model | Map | Selector | k | Sample | F1 |
|---|---|---|---|---|---|---|---|---|
| MQTTEEB-D2025 | prefit | bounded | PegasosQSVC | HybridPauliZZ | QKA_MI | 3 | 0.005 | 0.705 ± 0.093 |
| MQTTEEB-D2025 | prefit | bounded | QSVM | HybridPauliZZ | RF | 3 | 0.005 | 0.836 ± 0.016 |
| MQTTEEB-D2025 | prefit | bounded | RandomForest | Classical | RF | 3 | 0.005 | 0.966 ± 0.005 |
| MQTTEEB-D2025 | prefit | bounded | SVM | Classical | RF | 3 | 0.002 | 0.836 ± 0.027 |
| MQTTEEB-D2025 | prefit | twopi | PegasosQSVC | Z | RF | 3 | 0.005 | 0.729 ± 0.054 |
| MQTTEEB-D2025 | prefit | twopi | QSVM | Z | RF | 3 | 0.005 | 0.830 ± 0.014 |
| MQTTEEB-D2025 | prefit | twopi | RandomForest | Classical | RF | 3 | 0.005 | 0.966 ± 0.005 |
| MQTTEEB-D2025 | prefit | twopi | SVM | Classical | RF | 3 | 0.002 | 0.836 ± 0.027 |
| NSL-KDD | prefit | bounded | PegasosQSVC | Z | QKA_MI | 3 | 0.01 | 0.882 ± 0.054 |
| NSL-KDD | prefit | bounded | QSVM | Z | QKA_MI | 3 | 0.01 | 0.910 ± 0.026 |
| NSL-KDD | prefit | bounded | RandomForest | Classical | RF | 3 | 0.1 | 0.959 ± 0.007 |
| NSL-KDD | prefit | bounded | SVM | Classical | QKA_MI | 3 | 0.01 | 0.908 ± 0.030 |
| NSL-KDD | prefit | twopi | PegasosQSVC | Z | QKA_MI | 3 | 0.01 | 0.836 ± 0.037 |
| NSL-KDD | prefit | twopi | QSVM | HybridPauliZZ | QKA_MI | 3 | 0.1 | 0.878 ± 0.011 |
| NSL-KDD | prefit | twopi | RandomForest | Classical | RF | 3 | 0.1 | 0.959 ± 0.007 |
| NSL-KDD | prefit | twopi | SVM | Classical | QKA_MI | 3 | 0.1 | 0.885 ± 0.011 |
| UNSW-NB15 | prefit | bounded | PegasosQSVC | Pauli | RF | 3 | 0.002 | 0.842 ± 0.030 |
| UNSW-NB15 | prefit | bounded | QSVM | HybridPauliZZ | RF | 3 | 0.005 | 0.867 ± 0.016 |
| UNSW-NB15 | prefit | bounded | RandomForest | Classical | RF | 3 | 0.005 | 0.879 ± 0.013 |
| UNSW-NB15 | prefit | bounded | SVM | Classical | RF | 3 | 0.005 | 0.866 ± 0.010 |
| UNSW-NB15 | prefit | twopi | PegasosQSVC | ZZ | QKA_MI | 3 | 0.002 | 0.833 ± 0.030 |
| UNSW-NB15 | prefit | twopi | QSVM | ZZ | QKA_MI | 3 | 0.002 | 0.872 ± 0.023 |
| UNSW-NB15 | prefit | twopi | RandomForest | Classical | RF | 3 | 0.005 | 0.879 ± 0.013 |
| UNSW-NB15 | prefit | twopi | SVM | Classical | RF | 3 | 0.005 | 0.866 ± 0.010 |

## best quantum vs RF

28 paired comparisons; 17 significant at p < 0.05; median F1 change -0.033.

## Encoding audit (quantum rows)

| Dataset | Protocol | Encoding | Rotation span (rad) | Alias rate | Test out of range | KTA |
|---|---|---|---|---|---|---|
| MQTTEEB-D2025 | prefit | bounded | 3.11 | 0.0001 | 0.002 | 0.133 |
| MQTTEEB-D2025 | prefit | twopi | 12.42 | 0.0001 | 0.003 | 0.097 |
| NSL-KDD | prefit | bounded | 3.13 | 0.0088 | 0.002 | 0.509 |
| NSL-KDD | prefit | twopi | 12.46 | 0.0000 | 0.002 | 0.425 |
| UNSW-NB15 | prefit | bounded | 3.14 | 0.0001 | 0.002 | 0.388 |
| UNSW-NB15 | prefit | twopi | 12.57 | 0.0000 | 0.003 | 0.337 |
