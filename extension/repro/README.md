# Reproduction of the ICECET notebook under its original environment

Purpose: check whether the quantum results in Table II of the ICECET 2026 paper can be
reproduced, and whether the Qiskit version explains the differences found by the extension.

## Environment

Pinned in the notebook's first cell: Python 3.11, qiskit 1.0.2, qiskit-machine-learning 0.7.1,
qiskit-algorithms 0.3.0, numpy 1.26, scikit-learn 1.5.2, pandas 2.2.

```
uv venv -p python3.11 q102
uv pip install --python q102/bin/python "qiskit==1.0.2" "qiskit-machine-learning==0.7.1" \
    "qiskit-algorithms==0.3.0" "scikit-learn==1.5.2" "pandas==2.2.2" imbalanced-learn "numpy<2" matplotlib psutil
```

## Files

| File | What it is |
|---|---|
| `cell17.py` | Cell 17 of `BKP/QML_for_IDS.ipynb`, unchanged (the benchmark script) |
| `nb_repro.py` | Runs the notebook's own functions on NSL-KDD for PCA = 3 and 2, samples 1/5/10 %, k = 3/5, both selectors |
| `repro.csv`, `repro.log` | Results and log (the log holds the selection times and the kernel-equivalence check) |
| `timing.py` | Kernel-construction time with the real Qiskit primitives (1.0.2 and 2.6) |
| `timing_q1.csv`, `timing_q2.csv` | Timing results for qiskit 1.0.2 and 2.6.1 |

`python nb_repro.py repro.csv 3 2` (about 40 minutes on 2 CPU cores; the QKA-MI selection
takes most of it). `python timing.py timing_q1.csv 3` then `... 5` (about 10 minutes each).

Only three things differ from Colab: NSL-KDD is read from disk, plots are suppressed, and the
QSVC/PegasosQSVC kernels are built from exact statevectors of the same circuit objects the
notebook creates. The log shows the equivalence with `FidelityQuantumKernel` + reference
`Sampler`: maximum absolute difference 4.9e-15.
