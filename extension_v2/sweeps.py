"""
O3 extension, part 2: qubit sweep, encoding-bandwidth sweep, tuned baselines and extended metrics.

    python sweeps.py --check          # verify the analytic statevectors against Qiskit
    python sweeps.py                  # run (resumes from results/sweeps_*.csv)

Protocol: leak-free ('clean') data, stratified sample and 80/20 split, Random Forest selection of 11
features on the training set, SMOTE on training data, StandardScaler + PCA to q components fitted on
training data, angles = theta_max * minmax(train) with test values clipped.

Grid (QSVM C = 1, SVM C = 1, RF 200 trees, as in the main study):
    q in {3,5,7,9,11} x theta_max in {pi/8, pi/4, pi/2, pi, 2pi}
    x maps {Z, ZZ-full-r1, ZZ-linear-r1, ZZ-full-r2, ZZ-linear-r2} x 3 datasets x 10 seeds
Tuned (q in {3, 7}): QSVM chooses (theta_max, map, C) by 3-fold CV on the training set; SVM (C, gamma),
Random Forest (depth, leaf size) and HistGradientBoosting (learning rate, leaves) are tuned the same way.

Statevectors of Qiskit's z_feature_map / zz_feature_map are computed analytically (H layer, then the
diagonal phase exp(i[sum 2 x_i b_i + sum 2 (pi-x_i)(pi-x_j) (b_i xor b_j)]) per repetition), which is
exact and ~100x faster than building circuits; --check compares them with Qiskit Statevector.
"""
import argparse, itertools, json, os, sys, time, warnings
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (f1_score, precision_score, recall_score, roc_auc_score, average_precision_score,
                             confusion_matrix, accuracy_score)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from imblearn.over_sampling import SMOTE
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'extension'))  # reuse v1 code, unchanged
import o3ext as E
from run_benchmark import default_paths

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
QUBITS = [3, 5, 7, 9, 11]
THETAS = {"pi/8": np.pi / 8, "pi/4": np.pi / 4, "pi/2": np.pi / 2, "pi": np.pi, "2pi": 2 * np.pi}
MAPS = {"Z": ("Z", None, 1), "ZZ-full-r1": ("ZZ", "full", 1), "ZZ-linear-r1": ("ZZ", "linear", 1),
        "ZZ-full-r2": ("ZZ", "full", 2), "ZZ-linear-r2": ("ZZ", "linear", 2)}
SAMPLES = {"NSL-KDD": 0.05, "UNSW-NB15": 0.005, "MQTTEEB-D2025": 0.005}
SEEDS = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]
TUNE_Q = [3, 7]
K_SELECT = 11

# --------------------------------------------------------------------------- analytic feature maps
def _bits(n):
    k = np.arange(2 ** n)
    return ((k[:, None] >> np.arange(n)[None, :]) & 1).astype(float)      # little-endian, as Qiskit

def _wht(S, n):
    B = S.shape[0]; T = S.reshape((B,) + (2,) * n)
    for ax in range(1, n + 1):
        a = np.take(T, 0, axis=ax); b = np.take(T, 1, axis=ax)
        T = np.stack([(a + b), (a - b)], axis=ax) / np.sqrt(2)
    return T.reshape(B, 2 ** n)

def states(X, kind, ent, reps):
    """Statevectors (rows) of Qiskit's z_feature_map / zz_feature_map for angle rows X."""
    B, n = X.shape; bits = _bits(n)
    phase = 2 * X @ bits.T                                                  # single-qubit terms
    if kind == "ZZ":
        pairs = list(itertools.combinations(range(n), 2)) if ent == "full" else [(i, i + 1) for i in range(n - 1)]
        for i, j in pairs:
            phi = (np.pi - X[:, i]) * (np.pi - X[:, j])
            xor = np.abs(bits[:, i] - bits[:, j])
            phase += 2 * phi[:, None] * xor[None, :]
    D = np.exp(1j * phase)
    S = np.full((B, 2 ** n), 2 ** (-n / 2), dtype=complex) * D             # H|0> then D
    for _ in range(reps - 1):
        S = _wht(S, n) * D
    return S

def check():
    from qiskit.circuit.library import zz_feature_map, z_feature_map
    from qiskit.quantum_info import Statevector
    rng = np.random.default_rng(0); worst = 0
    for n in [2, 3, 5]:
        X = rng.uniform(0, 2 * np.pi, (6, n))
        for name, (kind, ent, reps) in MAPS.items():
            c = z_feature_map(n, reps=reps) if kind == "Z" else zz_feature_map(n, reps=reps, entanglement=ent)
            ps = sorted(c.parameters, key=lambda p: p.index if hasattr(p, "index") else p.name)
            ref = np.array([Statevector(c.assign_parameters(dict(zip(ps, r)))).data for r in X])
            mine = states(X, kind, ent, reps)
            Kr, Km = np.abs(ref.conj() @ ref.T) ** 2, np.abs(mine.conj() @ mine.T) ** 2
            d = np.abs(Kr - Km).max(); worst = max(worst, d)
            print(f"n={n} {name:13s} max |K_qiskit - K_analytic| = {d:.2e}")
    print("WORST", worst); return worst

# --------------------------------------------------------------------------- metrics
def metrics(y, pred, score, cats):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = dict(f1=f1_score(y, pred, zero_division=0), precision=precision_score(y, pred, zero_division=0),
               recall=recall_score(y, pred, zero_division=0), accuracy=accuracy_score(y, pred),
               fpr=fp / max(fp + tn, 1))
    try:
        out["auroc"] = roc_auc_score(y, score); out["auprc"] = average_precision_score(y, score)
    except Exception:
        out["auroc"] = out["auprc"] = np.nan
    att = (y == 1); rec = {}
    for c in np.unique(cats[att]):
        m = att & (cats == c)
        if m.sum() >= 5:
            rec[str(c)] = float((pred[m] == 1).mean())
    out["macro_attack_recall"] = float(np.mean(list(rec.values()))) if rec else np.nan
    out["worst_attack_recall"] = float(np.min(list(rec.values()))) if rec else np.nan
    out["attack_recall_json"] = json.dumps(rec)
    return out

def kta(K, y):
    n = len(y); z = np.where(y == 1, 1.0, -1.0); H = np.eye(n) - 1.0 / n; Kc = H @ K @ H
    den = np.linalg.norm(Kc) * n
    return float(z @ Kc @ z / den) if den > 0 else 0.0

# --------------------------------------------------------------------------- one (dataset, seed) unit
def prepare(X_raw, y, strat, sample, seed):
    st = pd.Series(np.asarray(strat)).astype(str)
    vc = st.value_counts(); st = st.where(st.map(vc) >= 10, "rare_" + y.astype(str).values)
    vc = st.value_counts(); st = st.where(st.map(vc) >= 2, "class_" + y.astype(str).values)
    idx, _ = next(StratifiedShuffleSplit(n_splits=1, test_size=1 - sample, random_state=seed).split(X_raw, st))
    Xp, yp = X_raw.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
    cats = np.asarray(strat)[idx].astype(str)
    itr, ite = train_test_split(np.arange(len(Xp)), test_size=0.2, random_state=seed, stratify=yp)
    Xtr, Xte, ytr, yte = Xp.iloc[itr], Xp.iloc[ite], yp.iloc[itr].values, yp.iloc[ite].values
    cols = E.select_rf(Xtr, ytr, K_SELECT, seed)
    Xtr, Xte = Xtr[cols].values, Xte[cols].values
    if min(np.bincount(ytr)) >= 6:
        Xtr, ytr = SMOTE(random_state=seed).fit_resample(Xtr, ytr); ytr = np.asarray(ytr)
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte), ytr, yte, cats[ite], cols

def angles(Ptr, Pte, theta):
    lo, hi = Ptr.min(0), Ptr.max(0); r = np.where(np.isclose(hi, lo), 1.0, hi - lo)
    f = lambda P: theta * np.clip((P - lo) / r, 0, 1)
    return f(Ptr), f(Pte)

def run_unit(ds, X_raw, y, strat, seed, done_grid, done_tuned, out_grid, out_tuned, log):
    t0 = time.time()
    Ztr, Zte, ytr, yte, cte, cols = prepare(X_raw, y, strat, SAMPLES[ds], seed)
    base = dict(dataset=ds, seed=seed, sample=SAMPLES[ds], n_train=len(ytr), n_test=len(yte), features="|".join(cols))
    for q in QUBITS:
        if (ds, seed, q) in done_grid and (q not in TUNE_Q or (ds, seed, q) in done_tuned):
            continue
        pca = PCA(n_components=q, random_state=seed).fit(Ztr)
        Ptr, Pte = pca.transform(Ztr), pca.transform(Zte)
        rows, kernels = [], {}
        # classical, default settings
        for name, m in [("SVM", SVC(C=1.0, kernel="rbf")), ("RandomForest", RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1))]:
            m.fit(Ptr, ytr); p = m.predict(Pte)
            s = m.decision_function(Pte) if name == "SVM" else m.predict_proba(Pte)[:, 1]
            rows.append({**base, "qubits": q, "theta": "", "fmap": "Classical", "model": name, "C": 1.0 if name == "SVM" else "",
                         "explained_var": float(pca.explained_variance_ratio_.sum()), **metrics(yte, p, s, cte)})
        # quantum grid
        for tn, th in THETAS.items():
            Atr, Ate = angles(Ptr, Pte, th)
            for mn, (kind, ent, reps) in MAPS.items():
                Str, Ste = states(Atr, kind, ent, reps), states(Ate, kind, ent, reps)
                Ktr, Kte = np.abs(Str.conj() @ Str.T) ** 2, np.abs(Ste.conj() @ Str.T) ** 2
                kernels[(tn, mn)] = (Ktr, Kte)
                off = Ktr[~np.eye(len(ytr), dtype=bool)]
                m = SVC(kernel="precomputed", C=1.0).fit(Ktr, ytr); p = m.predict(Kte)
                rows.append({**base, "qubits": q, "theta": tn, "fmap": mn, "model": "QSVM", "C": 1.0,
                             "explained_var": float(pca.explained_variance_ratio_.sum()),
                             "k_offdiag_mean": float(off.mean()), "k_offdiag_var": float(off.var()), "kta": kta(Ktr, ytr),
                             "n_sv": int(m.n_support_.sum()), **metrics(yte, p, m.decision_function(Kte), cte)})
        if (ds, seed, q) not in done_grid:
            pd.DataFrame(rows).to_csv(out_grid, mode="a", header=not os.path.exists(out_grid), index=False)
        # tuned
        if q in TUNE_Q and (ds, seed, q) not in done_tuned:
            trows = []
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
            best = (-1, None)
            for key, (Ktr, _) in kernels.items():
                for C in [0.1, 1, 10, 100]:
                    sc_ = []
                    for a, b in cv.split(Ktr, ytr):
                        mm = SVC(kernel="precomputed", C=C).fit(Ktr[np.ix_(a, a)], ytr[a])
                        sc_.append(f1_score(ytr[b], mm.predict(Ktr[np.ix_(b, a)]), zero_division=0))
                    if np.mean(sc_) > best[0]:
                        best = (np.mean(sc_), (key, C))
            (tn, mn), C = best[1]; Ktr, Kte = kernels[(tn, mn)]
            m = SVC(kernel="precomputed", C=C).fit(Ktr, ytr); p = m.predict(Kte)
            trows.append({**base, "qubits": q, "theta": tn, "fmap": mn, "model": "QSVM (tuned)", "params": json.dumps({"C": C}),
                          "cv_f1": best[0], "n_sv": int(m.n_support_.sum()), **metrics(yte, p, m.decision_function(Kte), cte)})
            grids = [("SVM (tuned)", SVC(kernel="rbf"), {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.1, 1, 10]}),
                     ("RandomForest (tuned)", RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
                      {"max_depth": [None, 10, 20], "min_samples_leaf": [1, 5]}),
                     ("HistGradientBoosting (tuned)", HistGradientBoostingClassifier(random_state=seed),
                      {"learning_rate": [0.05, 0.1, 0.2], "max_leaf_nodes": [15, 31]})]
            for name, est, grid in grids:
                g = GridSearchCV(est, grid, scoring="f1", cv=cv, n_jobs=1).fit(Ptr, ytr)
                mm = g.best_estimator_; p = mm.predict(Pte)
                s = mm.decision_function(Pte) if hasattr(mm, "decision_function") and name.startswith("SVM") else mm.predict_proba(Pte)[:, 1]
                trows.append({**base, "qubits": q, "theta": "", "fmap": "Classical", "model": name, "params": json.dumps(g.best_params_),
                              "cv_f1": g.best_score_, **metrics(yte, p, s, cte)})
            pd.DataFrame(trows).to_csv(out_tuned, mode="a", header=not os.path.exists(out_tuned), index=False)
        log(f"{ds} seed={seed} q={q} done ({time.time()-t0:.0f}s)")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true")
    ap.add_argument("--only", default=None); ap.add_argument("--budget", type=float, default=1e9, help="stop after N seconds")
    a = ap.parse_args()
    if a.check:
        sys.exit(0 if check() < 1e-9 else 1)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    out_grid, out_tuned = [os.path.join(HERE, "results", f) for f in ("sweeps_grid.csv", "sweeps_tuned.csv")]
    logf = open(os.path.join(HERE, "results", "sweeps.log"), "a")
    def log(m):
        print(m, flush=True); logf.write(time.strftime("%H:%M:%S ") + m + "\n"); logf.flush()
    rd = lambda f: set(map(tuple, pd.read_csv(f, usecols=["dataset", "seed", "qubits"]).drop_duplicates().values)) if os.path.exists(f) else set()
    done_grid, done_tuned = rd(out_grid), rd(out_tuned)
    paths = default_paths(); t_start = time.time()
    for ds in SAMPLES:
        if a.only and ds != a.only:
            continue
        X, y, strat = E.load_dataset(ds, paths, "clean")
        for seed in SEEDS:
            if all((ds, seed, q) in done_grid for q in QUBITS) and all((ds, seed, q) in done_tuned for q in TUNE_Q):
                continue
            if time.time() - t_start > a.budget:
                log("budget reached"); return
            run_unit(ds, X, y, strat, seed, done_grid, done_tuned, out_grid, out_tuned, log)
    log("ALL DONE")

if __name__ == "__main__":
    main()
