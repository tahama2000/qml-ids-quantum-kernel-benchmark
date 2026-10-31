"""
Literal reproduction of the ICECET notebook (cell 17 of QML_for_IDS.ipynb) under the
environment pinned in the notebook's first cell: qiskit 1.0.2, qiskit-machine-learning 0.7.1.

The notebook's own functions are executed unchanged (loaded from cell17.py). Only three
things differ from Colab: the NSL-KDD file is read from disk, plots are suppressed, and the
quantum kernels for QSVC/PegasosQSVC are built from exact statevectors of the SAME circuit
objects the notebook creates (equivalence with FidelityQuantumKernel + reference Sampler is
verified numerically on the 1% sample and written to the log).

Usage: python nb_repro.py <out.csv> [pca_components ...]
"""
import sys, os, time, json, re, io, contextlib, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: None
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
NSL = os.path.join(HERE, "..", "..", "data", "KDDTrain+_20Percent.txt")
src = open(os.path.join(HERE, "cell17.py")).read()
src = src.split("if __name__ == '__main__':")[0].split('if __name__ == "__main__":')[0]
src = src.replace("https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B_20Percent.txt", NSL)
src = "\n".join(("#" + l) if l.lstrip().startswith(("!", "%")) else l for l in src.splitlines())
NB = {"__name__": "nb"}
exec(compile(src, "cell17", "exec"), NB)

from qiskit.quantum_info import Statevector
from qiskit.circuit.library import PauliFeatureMap, ZZFeatureMap, ZFeatureMap
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from qiskit_machine_learning.algorithms import PegasosQSVC

ENC = {"PauliFeatureMap": PauliFeatureMap, "ZZFeatureMap": ZZFeatureMap,
       "ZFeatureMap": ZFeatureMap, "Hybrid_Pauli_ZZ": NB["HybridFeatureMap_Pauli_ZZ"]}

def nb_feature_map(enc, d):
    # exactly the notebook's line in run_experiment():
    # feature_map_type(X.shape[1]) if callable(...) else feature_map_type(feature_dimension=..., reps=1)
    return enc(d) if callable(enc) else enc(feature_dimension=d, reps=1)

def svs(circ, X):
    ps = list(circ.parameters)          # FidelityQuantumKernel binds in circuit.parameters order
    return np.array([Statevector(circ.assign_parameters(dict(zip(ps, r)))).data for r in X])

def kern(A, B): return np.abs(A.conj() @ B.T) ** 2

out = sys.argv[1]; pcas = [int(a) for a in sys.argv[2:]] or [2, 3]
df, label_col, pre = NB["load_dataset"]("NSL-KDD")
X_raw, y = NB["clean_data"](df, label_col, NB["LABEL_MODE"])
X_raw = X_raw.select_dtypes(include=[np.number])
log = open(out.replace(".csv", ".log"), "a")
def L(m): print(m, flush=True); log.write(m + "\n"); log.flush()
L(f"rows={len(X_raw)} numeric_cols={X_raw.shape[1]}")

done = set()
if os.path.exists(out):
    d0 = pd.read_csv(out); done = set(zip(d0.pca, d0["sample"], d0.k, d0.selector))

# reps actually used by the notebook's construction
for n, e in ENC.items():
    c = nb_feature_map(e, 3); L(f"{n}: reps={getattr(c, 'reps', 'composed')} depth={c.decompose().depth()} params={c.num_parameters}")

import pickle
SELF = os.path.join(HERE, "sel_cache.pkl")
SEL = pickle.load(open(SELF, "rb")) if os.path.exists(SELF) else {}
def select(sel, k):
    if (sel, k) in SEL: return SEL[(sel, k)]
    t = time.time(); buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        if sel == "RF":
            Xs = NB["select_features_rf"](X_raw, y, k)
        else:
            Xs = NB["select_features_qka_mi_fast"](X=X_raw, y=y, k=k, prefilter_m=20, subsample=150,
                                                     time_budget_sec=180, random_state=42)
    msg = buf.getvalue().strip().replace("\n", " | ")
    SEL[(sel, k)] = (Xs, time.time() - t, msg)
    L(f"[select {sel} k={k}] {time.time()-t:.1f}s cols={list(Xs.columns)} :: {msg}")
    pickle.dump(SEL, open(SELF, "wb"))
    return SEL[(sel, k)]

verified = False
for pca in pcas:
    NB["PCA_COMPONENTS"] = pca
    for sp in [0.01, 0.05, 0.1]:
        for k in [3, 5]:
            for sel in ["RF", "QKA_MI_FAST"]:
                if (pca, sp, k, sel) in done: continue
                Xsel, sel_s, msg = select(sel, k)
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        Xs, ys, ntr = NB["sample_data_with_stratification"](Xsel, y, sp)
                        Xb, yb = NB["balance_dataset_with_smote"](Xs, ys)
                        Xf, _ = NB["apply_pca"](NB["scale_data"](Xb))
                except Exception as ex:      # the notebook catches nothing here: the whole run would stop
                    L(f"pca={pca} sp={sp} k={k} {sel}: PREPROCESSING ERROR {ex!r} (n_selected={Xsel.shape[1]})")
                    pd.DataFrame([dict(pca=pca, sample=sp, k=k, selector=sel, n_selected=Xsel.shape[1], selected="|".join(Xsel.columns),
                                       select_s=round(sel_s, 1), budget_hit=int("Time budget" in msg), model="ERROR", f1=np.nan)]).to_csv(
                        out, mode="a", header=not os.path.exists(out), index=False)
                    continue
                Xq = NB["scale_for_quantum_0_2pi"](Xf)
                yb = np.asarray(yb).astype(int)
                itr, ite = train_test_split(np.arange(len(yb)), test_size=0.2, random_state=42)
                rows = []
                base = dict(pca=pca, sample=sp, k=k, selector=sel, n_selected=Xsel.shape[1],
                            selected="|".join(Xsel.columns), select_s=round(sel_s, 1),
                            budget_hit=int("Time budget" in msg), n_train=len(itr), n_test=len(ite))
                for name, m in [("SVM", SVC(kernel="rbf", C=1.0)), ("RandomForest", RandomForestClassifier(n_estimators=200, random_state=42))]:
                    m.fit(Xf[itr], yb[itr]); p = m.predict(Xf[ite])
                    rows.append({**base, "encoding": "Classical", "reps": "", "model": name, "f1": f1_score(yb[ite], p), "acc": accuracy_score(yb[ite], p)})
                for en, e in ENC.items():
                    for reps_mode in ["notebook", "reps1"]:
                        if reps_mode == "notebook":
                            circ = nb_feature_map(e, Xq.shape[1])
                        else:
                            if en == "Hybrid_Pauli_ZZ": continue            # already reps=1
                            circ = e(feature_dimension=Xq.shape[1], reps=1)
                        reps = getattr(circ, "reps", 1)
                        t = time.time(); S = svs(circ, Xq); Ktr, Kte = kern(S[itr], S[itr]), kern(S[ite], S[itr]); kt = time.time() - t
                        if not verified and reps_mode == "notebook":
                            from qiskit.primitives import Sampler
                            from qiskit_algorithms.state_fidelities import ComputeUncompute
                            from qiskit_machine_learning.kernels import FidelityQuantumKernel
                            n = min(40, len(itr))
                            fqk = FidelityQuantumKernel(feature_map=circ, fidelity=ComputeUncompute(sampler=Sampler()))
                            Kref = fqk.evaluate(Xq[itr[:n]], Xq[itr[:n]])
                            L(f"[verify] {en}: max |K_statevector - K_FidelityQuantumKernel| = {np.abs(Kref - Ktr[:n, :n]).max():.2e}")
                            if en == "Hybrid_Pauli_ZZ": verified = True
                        p = SVC(kernel="precomputed", C=1.0).fit(Ktr, yb[itr]).predict(Kte)
                        rows.append({**base, "encoding": en, "reps": reps, "model": "QSVM", "f1": f1_score(yb[ite], p), "acc": accuracy_score(yb[ite], p), "kernel_s": round(kt, 2)})
                        for s in range(5):     # notebook gives Pegasos no seed -> 5 seeds show the spread
                            p = PegasosQSVC(C=1000, num_steps=100, precomputed=True, seed=s).fit(Ktr, yb[itr]).predict(Kte)
                            rows.append({**base, "encoding": en, "reps": reps, "model": "PegasosQSVC", "pegasos_seed": s,
                                         "f1": f1_score(yb[ite], np.asarray(p).astype(int)), "acc": accuracy_score(yb[ite], np.asarray(p).astype(int))})
                pd.DataFrame(rows).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
                q = pd.DataFrame(rows); q = q[q.reps.astype(str) != "1"]
                L(f"pca={pca} sp={sp} k={k} {sel}: " + ", ".join(f"{r.model[:4]}/{r.encoding[:5]}={r.f1:.3f}" for r in q.groupby(["model", "encoding"]).f1.mean().reset_index().itertuples()))
L("done")
