# Quantum kernel intrusion detection: encoding bandwidth, circuit size and shot budget

Code and per-seed results for

> T. M. Mahmoud and N. Kaabouch, "Encoding Bandwidth, Circuit Size and Shot Budget in Quantum Kernel Intrusion Detection: A Robust Benchmark under the CRISP-QCML-IDS Framework," submitted to IEEE ICAIC 2027.

The repository also contains the reproduction of the earlier benchmark it audits as a use case:
T. M. Mahmoud and N. Kaabouch, "Benchmarking Classical and Quantum Machine Learning for Intrusion Detection Across Multiple Datasets," ICECET 2026, doi:10.1109/ICECET65726.2026.11632866; preprint with correction note: arXiv:2610.33119.

## Layout

| Folder | What it holds | Paper |
|---|---|---|
| `extension/` | Leak-free, multi-seed re-evaluation of the earlier benchmark (main study, 840 configurations × 10 seeds), finite shots, preprocessing-order ablation, time split; the 13 review gates (`CRISP_QCML_IDS_LEAKAGE_GATES.md`) | Sec. V-A, V-B (shots), VI, Table II, Fig. 2 |
| `extension/repro/` | Literal re-execution of the earlier notebook under its pinned environment (Qiskit 1.0.2) and kernel-timing check | Sec. VI (gates M2, M3, E2, E3) |
| `extension_v2/` | Sweep study (3–11 qubits, five encoding ranges, five feature maps) and tuned models (3-fold CV) | Sec. V-A, V-B, V-C, Figs. 3–4 |
| `extension_v8/` | Re-tune of the RBF SVM with a wider bandwidth grid (gamma up to 300) | Table I, Discussion |
| `paper_figures/` | Scripts that rebuild Table I and Figs. 1–7 from the result files | all |
| `data/` | Where to put the datasets (not redistributed) | Sec. IV |

Every result CSV holds one row per configuration and seed (seeds 42 and 1–9), so all means, confidence intervals and tests can be recomputed.

## Setup

Python 3.10 or newer.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.13, qiskit 2.6.1, qiskit-machine-learning 0.9.1, scikit-learn 1.9.1, imbalanced-learn 0.14.2, numpy 2.5.3, pandas 3.0.5, scipy 1.18.1. The reproduction in `extension/repro/` uses its own pinned environment (see its README).

## Rebuild the paper's table and figures (no datasets needed)

```bash
python paper_figures/make_table2.py          # Table I -> paper_figures/table2.csv
python paper_figures/make_figures.py         # Figs. 1-6 -> paper_figures/out/
python paper_figures/make_checklist_fig.py   # Fig. 7   -> paper_figures/out/
```

## Re-run the experiments (datasets needed, see `data/README.md`)

```bash
cd extension
python run_benchmark.py --plan smoke                  # about 1 minute
python run_benchmark.py --plan main                   # main study; resumable
python run_benchmark.py --plan shots
python analyze.py --plan main
cd ../extension_v2 && python sweeps.py && python analyze_sweeps.py
cd ../extension_v8 && python svm_recheck.py && python compare.py
```

Runs append to `results/<plan>.csv` after each configuration, so an interrupted run resumes where it stopped.

## License

Code: MIT (see `LICENSE`). Datasets keep their original licenses and are not included.
