# Results report: plan 'categorical'

800 rows, 80 configurations, seeds per configuration: 10–10.

## Best configuration per dataset, protocol, encoding and model (F1 mean ± 95% CI)

| Dataset | Protocol | Encoding | Model | Map | Selector | k | Sample | F1 |
|---|---|---|---|---|---|---|---|---|
| NSL-KDD | clean | bounded | PegasosQSVC | HybridPauliZZ | QKA_MI | 3 | 0.1 | 0.863 ± 0.017 |
| NSL-KDD | clean | bounded | QSVM | HybridPauliZZ | QKA_MI | 3 | 0.1 | 0.901 ± 0.016 |
| NSL-KDD | clean | bounded | RandomForest | Classical | RF | 3 | 0.1 | 0.950 ± 0.016 |
| NSL-KDD | clean | bounded | SVM | Classical | QKA_MI | 3 | 0.1 | 0.902 ± 0.017 |
| NSL-KDD | clean | twopi | PegasosQSVC | Z | QKA_MI | 3 | 0.1 | 0.823 ± 0.044 |
| NSL-KDD | clean | twopi | QSVM | Z | QKA_MI | 3 | 0.1 | 0.867 ± 0.010 |
| NSL-KDD | clean | twopi | RandomForest | Classical | RF | 3 | 0.1 | 0.950 ± 0.016 |
| NSL-KDD | clean | twopi | SVM | Classical | QKA_MI | 3 | 0.1 | 0.876 ± 0.011 |
| UNSW-NB15 | clean | bounded | PegasosQSVC | ZZ | RF | 3 | 0.005 | 0.830 ± 0.046 |
| UNSW-NB15 | clean | bounded | QSVM | Pauli | RF | 3 | 0.005 | 0.907 ± 0.010 |
| UNSW-NB15 | clean | bounded | RandomForest | Classical | RF | 3 | 0.005 | 0.892 ± 0.013 |
| UNSW-NB15 | clean | bounded | SVM | Classical | RF | 3 | 0.005 | 0.907 ± 0.010 |
| UNSW-NB15 | clean | twopi | PegasosQSVC | ZZ | RF | 3 | 0.005 | 0.829 ± 0.053 |
| UNSW-NB15 | clean | twopi | QSVM | Pauli | RF | 3 | 0.005 | 0.907 ± 0.010 |
| UNSW-NB15 | clean | twopi | RandomForest | Classical | RF | 3 | 0.005 | 0.892 ± 0.013 |
| UNSW-NB15 | clean | twopi | SVM | Classical | RF | 3 | 0.005 | 0.907 ± 0.010 |

## best quantum vs RF

8 paired comparisons; 6 significant at p < 0.05; median F1 change +0.004.

## encoding fix (clean [0,2pi] -> clean [0,pi/2]+clip)

40 paired comparisons; 10 significant at p < 0.05; median F1 change +0.000.

## QKA-MI feature stability across encodings

Mean Jaccard overlap of selected features ([0,2π] vs [0,π/2]): 0.14 (1.0 = identical selections).

## Encoding audit (quantum rows)

| Dataset | Protocol | Encoding | Rotation span (rad) | Alias rate | Test out of range | KTA |
|---|---|---|---|---|---|---|
| NSL-KDD | clean | bounded | 3.14 | 0.0079 | 0.000 | 0.520 |
| NSL-KDD | clean | twopi | 12.57 | 0.0000 | 0.000 | 0.431 |
| UNSW-NB15 | clean | bounded | 3.14 | 0.0002 | 0.000 | 0.401 |
| UNSW-NB15 | clean | twopi | 12.57 | 0.0000 | 0.002 | 0.353 |
