# Datasets (not redistributed)

`extension/run_benchmark.py` looks for the files at these paths, relative to the repository root:

| Dataset | Path | Source |
|---|---|---|
| NSL-KDD (20% training subset) | `data/KDDTrain+_20Percent.txt` | Canadian Institute for Cybersecurity, https://www.unb.ca/cic/datasets/nsl.html |
| UNSW-NB15 (training and testing CSVs) | `UNSW_NB15/UNSW_NB15_training-set.csv`, `UNSW_NB15/UNSW_NB15_testing-set(in).csv` | UNSW Canberra, https://research.unsw.edu.au/projects/unsw-nb15-dataset |
| MQTTEEB-D2025 (processed) | `MQTTEEB-D_dataset_All_Processed.csv` | Aqachtoul et al., Data in Brief 62 (2025) 111897; Mendeley Data, doi:10.17632/jfttfjn6tr.1 |

Rename the UNSW testing file to `UNSW_NB15_testing-set(in).csv` or edit `default_paths()` in `extension/run_benchmark.py`.
