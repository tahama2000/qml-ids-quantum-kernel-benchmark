# Results report: plan 'time_split'

800 rows, 80 configurations, seeds per configuration: 10–10.

## Best configuration per dataset, protocol, encoding and model (F1 mean ± 95% CI)

| Dataset | Protocol | Encoding | Model | Map | Selector | k | Sample | F1 |
|---|---|---|---|---|---|---|---|---|
| MQTTEEB-D2025 | clean+no_epoch | bounded | PegasosQSVC | Z | QKA_MI | 3 | 0.005 | 0.928 ± 0.065 |
| MQTTEEB-D2025 | clean+no_epoch | bounded | QSVM | Z | QKA_MI | 3 | 0.005 | 0.957 ± 0.007 |
| MQTTEEB-D2025 | clean+no_epoch | bounded | RandomForest | Classical | RF | 3 | 0.005 | 0.976 ± 0.005 |
| MQTTEEB-D2025 | clean+no_epoch | bounded | SVM | Classical | QKA_MI | 3 | 0.005 | 0.957 ± 0.007 |
| MQTTEEB-D2025 | clean+no_epoch | twopi | PegasosQSVC | Pauli | QKA_MI | 3 | 0.005 | 0.921 ± 0.063 |
| MQTTEEB-D2025 | clean+no_epoch | twopi | QSVM | Z | RF | 3 | 0.005 | 0.955 ± 0.007 |
| MQTTEEB-D2025 | clean+no_epoch | twopi | RandomForest | Classical | RF | 3 | 0.005 | 0.976 ± 0.005 |
| MQTTEEB-D2025 | clean+no_epoch | twopi | SVM | Classical | RF | 3 | 0.005 | 0.956 ± 0.007 |

## best quantum vs RF

8 paired comparisons; 6 significant at p < 0.05; median F1 change -0.019.

## Encoding audit (quantum rows)

| Dataset | Protocol | Encoding | Rotation span (rad) | Alias rate | Test out of range | KTA |
|---|---|---|---|---|---|---|
| MQTTEEB-D2025 | clean+no_epoch | bounded | 3.14 | 0.0001 | 0.000 | 0.194 |
| MQTTEEB-D2025 | clean+no_epoch | twopi | 12.57 | 0.0000 | 0.011 | 0.171 |
