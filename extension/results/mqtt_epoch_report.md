# Results report: plan 'mqtt_epoch'

800 rows, 80 configurations, seeds per configuration: 10–10.

## Best configuration per dataset, protocol, encoding and model (F1 mean ± 95% CI)

| Dataset | Protocol | Encoding | Model | Map | Selector | k | Sample | F1 |
|---|---|---|---|---|---|---|---|---|
| MQTTEEB-D2025 | clean+no_epoch | bounded | PegasosQSVC | Pauli | QKA_MI | 3 | 0.005 | 0.875 ± 0.119 |
| MQTTEEB-D2025 | clean+no_epoch | bounded | QSVM | Z | RF | 3 | 0.005 | 0.953 ± 0.006 |
| MQTTEEB-D2025 | clean+no_epoch | bounded | RandomForest | Classical | RF | 3 | 0.005 | 0.972 ± 0.006 |
| MQTTEEB-D2025 | clean+no_epoch | bounded | SVM | Classical | RF | 3 | 0.005 | 0.953 ± 0.006 |
| MQTTEEB-D2025 | clean+no_epoch | twopi | PegasosQSVC | HybridPauliZZ | RF | 3 | 0.002 | 0.847 ± 0.132 |
| MQTTEEB-D2025 | clean+no_epoch | twopi | QSVM | HybridPauliZZ | RF | 3 | 0.005 | 0.952 ± 0.006 |
| MQTTEEB-D2025 | clean+no_epoch | twopi | RandomForest | Classical | QKA_MI | 3 | 0.005 | 0.972 ± 0.006 |
| MQTTEEB-D2025 | clean+no_epoch | twopi | SVM | Classical | RF | 3 | 0.005 | 0.953 ± 0.006 |

## best quantum vs RF

8 paired comparisons; 8 significant at p < 0.05; median F1 change -0.052.

## Encoding audit (quantum rows)

| Dataset | Protocol | Encoding | Rotation span (rad) | Alias rate | Test out of range | KTA |
|---|---|---|---|---|---|---|
| MQTTEEB-D2025 | clean+no_epoch | bounded | 3.14 | 0.0002 | 0.000 | 0.166 |
| MQTTEEB-D2025 | clean+no_epoch | twopi | 12.57 | 0.0000 | 0.002 | 0.166 |
