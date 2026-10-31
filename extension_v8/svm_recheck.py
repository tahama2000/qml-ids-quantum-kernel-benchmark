"""
O3 extension v8: re-check of the tuned RBF-SVM baseline used in the O3 v2 paper.

v2 tuned the RBF-SVM over C {0.1,1,10,100} x gamma {scale,0.1,1,10}. The v7 confirmation run showed the best gamma is
often 100. This script re-tunes the RBF-SVM on exactly the v2 splits (sweeps.prepare, PCA to q, same seeds and CV
folds) with gamma extended to {scale, 0.1, 1, 10, 30, 100, 300}, and compares it with the v2 tuned QSVM, RF and HistGB.
v1-v7 are not modified.
"""
import os, sys, json, time, warnings
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(H, "..", "extension_v2")); sys.path.insert(0, os.path.join(H, "..", "extension"))
import sweeps as S, o3ext as E
from run_benchmark import default_paths
warnings.filterwarnings("ignore")
R = os.path.join(H, "results"); OUT = os.path.join(R, "svm_recheck_per_seed.csv")
GRID = {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.1, 1, 10, 30, 100, 300]}

def run():
    done = set(map(tuple, pd.read_csv(OUT, usecols=["dataset", "seed"]).drop_duplicates().values)) if os.path.exists(OUT) else set()
    paths = default_paths()
    for ds in S.SAMPLES:
        X, y, strat = E.load_dataset(ds, paths, "clean")
        for seed in S.SEEDS:
            if (ds, seed) in done: continue
            t0 = time.time(); rows = []
            Ztr, Zte, ytr, yte, cte, _ = S.prepare(X, y, strat, S.SAMPLES[ds], seed)
            for q in S.TUNE_Q:
                pca = PCA(n_components=q, random_state=seed).fit(Ztr); Ptr, Pte = pca.transform(Ztr), pca.transform(Zte)
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
                g = GridSearchCV(SVC(kernel="rbf"), GRID, scoring="f1", cv=cv, n_jobs=-1).fit(Ptr, ytr)
                m = g.best_estimator_
                rows.append(dict(dataset=ds, seed=seed, qubits=q, model="SVM (tuned, wide gamma)", params=json.dumps(g.best_params_), cv_f1=g.best_score_,
                                 **S.metrics(yte, m.predict(Pte), m.decision_function(Pte), cte)))
            pd.DataFrame(rows).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
            print(f"{ds} seed={seed} ({time.time()-t0:.0f}s) {[r['params'] for r in rows]}", flush=True)

if __name__ == "__main__":
    run()
