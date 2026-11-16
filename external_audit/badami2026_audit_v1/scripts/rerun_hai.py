"""External audit re-run of Badami (IEEE Access 2026), HAI 22.04 QSVM pipeline.

P   published protocol, as in the repository: MinMax fitted on the (attack-free) train files and applied
    to the test files; RF Gini selection (100 trees, depth 20, leaf 50, balanced) on a 50k stratified subsample
    of the TEST files; exact-row dedupe; random stratified 2500/1000 split drawn from the same test files;
    ZZFeatureMap (8 qubits, reps=2, linear); SVC(precomputed, C=1, balanced); RBF-SVM C=1, gamma=scale, balanced.
T   temporal: test1-test3 = development block (selection + training), test4 = evaluation block; otherwise identical.
T-bw  as T, with an input bandwidth s (x -> s*x) chosen by 5-fold CV AUC on the training sample,
    s in {0.1, 0.25, 0.5, 1}; RBF C and gamma tuned by the same CV.
Kernels are exact statevector fidelities (scripts/zz.py, verified against Qiskit to 1e-10).
"""
import sys, glob, os, time
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
sys.path.insert(0, os.path.dirname(__file__))
from zz import zz_states, kernel

DATA, OUT = sys.argv[1], sys.argv[2]
SEEDS = [42, 123, 999, 2024, 555, 1, 2, 3, 4, 5]
S_GRID = [0.1, 0.25, 0.5, 1.0]

mm = pd.read_csv(f"{DATA}/train_minmax.csv", index_col=0)
parts = []
for i in range(1, 5):
    d = pd.read_csv(f"{DATA}/test{i}.csv"); d["block"] = i; parts.append(d)
df = pd.concat(parts, ignore_index=True)
feats = sorted(c for c in df.columns if c not in ("timestamp", "Attack", "block"))
rng_ = (mm.loc[feats, "max"] - mm.loc[feats, "min"]).replace(0, 1.0)
X_all = ((df[feats] - mm.loc[feats, "min"]) / rng_).values
y_all = (df["Attack"].values > 0).astype(int)
t_all = np.arange(len(df))                     # 1 Hz samples, file order = time order
block = df["block"].values
print("rows", X_all.shape, "attacks", y_all.sum(), flush=True)

def rf_select(X, y, seed=42):
    Xs, _, ys, _ = train_test_split(X, y, train_size=min(50000, len(X)), stratify=y, random_state=seed)
    rf = RandomForestClassifier(100, max_depth=20, min_samples_leaf=50, class_weight="balanced",
                                random_state=seed, n_jobs=-1).fit(Xs, ys)
    return np.argsort(rf.feature_importances_)[::-1][:8]

def dedupe(idx):
    _, u = np.unique(np.round(X_all[idx], 6), axis=0, return_index=True)
    return idx[np.sort(u)]

def evaluate(Ktr, Kte, ytr, yte):
    m = SVC(kernel="precomputed", C=1.0, class_weight="balanced", probability=True, random_state=0).fit(Ktr, ytr)
    p = m.predict_proba(Kte)[:, 1]; yp = m.predict(Kte)
    return dict(auc=roc_auc_score(yte, p), f1=f1_score(yte, yp, zero_division=0),
                recall=recall_score(yte, yp, zero_division=0), precision=precision_score(yte, yp, zero_division=0))

def rbf(Xtr, ytr, Xte, yte, tuned, seed):
    if tuned:
        g = GridSearchCV(SVC(kernel="rbf", class_weight="balanced"), {"C": [0.1, 1, 10], "gamma": ["scale", "auto"]},
                         cv=StratifiedKFold(5, shuffle=True, random_state=seed), scoring="roc_auc").fit(Xtr, ytr)
        C, gm = g.best_params_["C"], g.best_params_["gamma"]
    else:
        C, gm = 1.0, "scale"
    m = SVC(kernel="rbf", C=C, gamma=gm, class_weight="balanced", probability=True, random_state=0).fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]; yp = m.predict(Xte)
    return dict(auc=roc_auc_score(yte, p), f1=f1_score(yte, yp, zero_division=0),
                recall=recall_score(yte, yp, zero_division=0), precision=precision_score(yte, yp, zero_division=0))

def cv_bandwidth(Xtr, ytr, seed):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed); best = (-1, None)
    for s in S_GRID:
        St = zz_states(s * Xtr); K = kernel(St, St); a = []
        for tr, va in skf.split(Xtr, ytr):
            m = SVC(kernel="precomputed", C=1.0, class_weight="balanced").fit(K[np.ix_(tr, tr)], ytr[tr])
            a.append(roc_auc_score(ytr[va], m.decision_function(K[np.ix_(va, tr)])))
        if np.mean(a) > best[0]: best = (np.mean(a), s)
    return best[1]

all_idx = np.arange(len(df))
sel_P = rf_select(X_all, y_all)
dev = all_idx[block <= 3]; ev = all_idx[block == 4]
sel_T = rf_select(X_all[dev], y_all[dev])
print("selected P:", [feats[i] for i in sel_P]); print("selected T:", [feats[i] for i in sel_T], flush=True)

rows, t0 = [], time.time()
for seed in SEEDS:
    # --- P: random split inside the test files
    u = dedupe(all_idx)
    tr, rem = train_test_split(u, train_size=2500, stratify=y_all[u], random_state=seed)
    te, _ = train_test_split(rem, train_size=1000, stratify=y_all[rem], random_state=seed)
    gap = np.min(np.abs(t_all[te][:, None] - t_all[tr][None, :]), axis=1)    # seconds to nearest training sample
    Xtr, Xte, ytr, yte = X_all[tr][:, sel_P], X_all[te][:, sel_P], y_all[tr], y_all[te]
    Str, Ste = zz_states(Xtr), zz_states(Xte)
    rows.append(dict(cond="P", seed=seed, model="QSVM", s=1.0, med_gap_s=np.median(gap),
                     frac_gap_le10s=np.mean(gap <= 10), **evaluate(kernel(Str, Str), kernel(Ste, Str), ytr, yte)))
    rows.append(dict(cond="P", seed=seed, model="RBF-SVM", **rbf(Xtr, ytr, Xte, yte, False, seed)))
    rows.append(dict(cond="P", seed=seed, model="RBF-SVM-tuned", **rbf(Xtr, ytr, Xte, yte, True, seed)))
    # --- T: temporal blocks
    ud, ue = dedupe(dev), dedupe(ev)
    tr, _ = train_test_split(ud, train_size=2500, stratify=y_all[ud], random_state=seed)
    te, _ = train_test_split(ue, train_size=1000, stratify=y_all[ue], random_state=seed)
    Xtr, Xte, ytr, yte = X_all[tr][:, sel_T], X_all[te][:, sel_T], y_all[tr], y_all[te]
    Str, Ste = zz_states(Xtr), zz_states(Xte)
    rows.append(dict(cond="T", seed=seed, model="QSVM", s=1.0, **evaluate(kernel(Str, Str), kernel(Ste, Str), ytr, yte)))
    rows.append(dict(cond="T", seed=seed, model="RBF-SVM", **rbf(Xtr, ytr, Xte, yte, False, seed)))
    rows.append(dict(cond="T", seed=seed, model="RBF-SVM-tuned", **rbf(Xtr, ytr, Xte, yte, True, seed)))
    for s in S_GRID:                                   # bandwidth sweep (reported, not selected on test)
        Str, Ste = zz_states(s * Xtr), zz_states(s * Xte)
        rows.append(dict(cond="T-sweep", seed=seed, model="QSVM", s=s, **evaluate(kernel(Str, Str), kernel(Ste, Str), ytr, yte)))
    s = cv_bandwidth(Xtr, ytr, seed)
    Str, Ste = zz_states(s * Xtr), zz_states(s * Xte)
    rows.append(dict(cond="T-bw", seed=seed, model="QSVM", s=s, **evaluate(kernel(Str, Str), kernel(Ste, Str), ytr, yte)))
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print("seed", seed, f"{time.time()-t0:.0f}s", flush=True)
print("done")
