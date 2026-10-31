# Objective 3 extension: leakage-free, multi-seed re-evaluation

This folder re-runs the ICECET 2026 benchmark under two protocols and adds the
experiments of the paper (10 seeds, bounded encoding, finite shots, kernel diagnostics).
The original notebook stays untouched; this code reuses its loaders, the QKA-MI
selector, the four feature maps and the model settings.

| Protocol | What it does |
|---|---|
| `published` | Reproduces the notebook: selection, sampling, SMOTE, scaling and PCA on the whole sample, [0, 2π] angles, unstratified 80/20 split. Seed 42 = the published random state. |
| `clean` + `twopi` | Split first; everything fitted on training data only; known leaks removed (MQTTEEB `Attack_Binary`, UNSW `id`); [0, 2π] angles. |
| `clean` + `bounded` | Same, with [0, π/2] angles and clipping to the training range. |

Kernels are computed exactly from statevectors (identical to the notebook's Qiskit
reference Sampler). Finite-shot kernels are sampled as Binomial(shots, K)/shots.

## One-time setup

```bash
cd extension
python3 --version                 # needs 3.10 or newer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
source .venv/bin/activate
python run_benchmark.py --plan smoke          # about 1 minute: checks everything works
caffeinate -i python run_benchmark.py --plan main         # full grid, 840 configurations, ~1–2 hours
caffeinate -i python run_benchmark.py --plan mqtt_epoch   # MQTTEEB without the epoch-time column
caffeinate -i python run_benchmark.py --plan shots        # finite-shot study (NSL-KDD)
python analyze.py --plan main                 # tables, CIs, Wilcoxon tests -> results/main_report.md
```

`caffeinate -i` keeps the Mac awake while the run lasts. If a run stops (sleep, crash,
Ctrl-C), start the same command again: finished configurations are skipped.

## Outputs (in `results/`)

- `<plan>.csv` — one row per model and configuration: F1, accuracy, precision, recall,
  training time, kernel time, and for quantum rows the kernel diagnostics
  (off-diagonal mean/variance, kernel-target alignment, alias rate, rotation span,
  share of test values outside the training range).
- `<plan>.log` — progress and any failures.
- `<plan>_env.txt` — package versions used.
- `<plan>_summary.csv`, `<plan>_tests.csv`, `<plan>_report.md` — from `analyze.py`.

## Data

Paths default to this folder's parent (`Objective 3 -QML benchmark`):
`MQTTEEB-D_dataset_All_Processed.csv`, `UNSW_NB15/…training-set.csv`,
`UNSW_NB15/…testing-set(in).csv`, and `extension/data/KDDTrain+_20Percent.txt`
(the same GitHub file the notebook downloaded). Override with `--paths paths.json`.
