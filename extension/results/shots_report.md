# Results report: plan 'shots'

480 rows, 60 configurations, seeds per configuration: 10–10.

## Best configuration per dataset, protocol, encoding and model (F1 mean ± 95% CI)

| Dataset | Protocol | Encoding | Model | Map | Selector | k | Sample | F1 |
|---|---|---|---|---|---|---|---|---|
| NSL-KDD | clean | bounded | PegasosQSVC | HybridPauliZZ | QKA_MI | 3 | 0.05 | 0.861 ± 0.020 |
| NSL-KDD | clean | bounded | QSVM | HybridPauliZZ | QKA_MI | 3 | 0.05 | 0.897 ± 0.024 |
| NSL-KDD | clean | twopi | PegasosQSVC | ZZ | QKA_MI | 3 | 0.05 | 0.774 ± 0.023 |
| NSL-KDD | clean | twopi | QSVM | Pauli | QKA_MI | 3 | 0.05 | 0.850 ± 0.032 |

## encoding fix (clean [0,2pi] -> clean [0,pi/2]+clip)

24 paired comparisons; 23 significant at p < 0.05; median F1 change +0.066.

## QKA-MI feature stability across encodings

Mean Jaccard overlap of selected features ([0,2π] vs [0,π/2]): 0.08 (1.0 = identical selections).

## Encoding audit (quantum rows)

| Dataset | Protocol | Encoding | Rotation span (rad) | Alias rate | Test out of range | KTA |
|---|---|---|---|---|---|---|
| NSL-KDD | clean | bounded | 3.14 | 0.0000 | 0.000 | 0.572 |
| NSL-KDD | clean | twopi | 12.57 | 0.0000 | 0.001 | 0.368 |
