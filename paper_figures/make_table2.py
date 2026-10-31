"""Table I of the ICAIC 2027 paper: tuned models, means over 10 seeds, Wilcoxon (zsplit) + Holm.
Run from the repository root:  python paper_figures/make_table2.py
"""
import os, numpy as np, pandas as pd
from scipy import stats
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
t = pd.read_csv(os.path.join(ROOT, "extension_v2/results/sweeps_tuned.csv"))
w = pd.read_csv(os.path.join(ROOT, "extension_v8/results/svm_recheck_per_seed.csv")).assign(model="SVM (tuned)")
t = t[t.model != "SVM (tuned)"]   # SVM replaced by the wide-gamma re-tune (extension_v8)
d = pd.concat([t, w[t.columns.intersection(w.columns)]], ignore_index=True)
order = ["QSVM (tuned)", "SVM (tuned)", "RandomForest (tuned)", "HistGradientBoosting (tuned)"]
rows = []
for ds in ["NSL-KDD", "UNSW-NB15", "MQTTEEB-D2025"]:
    for q in [3, 7]:
        s = d[(d.dataset == ds) & (d.qubits == q)]
        piv = s.pivot_table(index="seed", columns="model", values="f1")
        ps = []
        for m in order[1:]:
            a, b = piv["QSVM (tuned)"], piv[m]
            p = np.nan if np.allclose(a - b, 0) else stats.wilcoxon(a, b, zero_method="zsplit").pvalue
            ps.append(((a - b).mean(), p))
        pv = [x[1] for x in ps]; idx = np.argsort(pv); adj = [0.0] * 3; run = 0
        for r, i in enumerate(idx):
            run = max(run, min(1, (3 - r) * pv[i])); adj[i] = run
        for m in order:
            g = s[s.model == m]
            rec = dict(dataset=ds, q=q, model=m, n=len(g), f1=g.f1.mean(), auprc=g.auprc.mean(), fpr=g.fpr.mean(), worst=g.worst_attack_recall.mean())
            if m != order[0]:
                k = order.index(m) - 1; rec.update(diff=ps[k][0], p=ps[k][1], p_holm=adj[k])
            rows.append(rec)
out = pd.DataFrame(rows); out.to_csv(os.path.join(ROOT, "paper_figures", "table2.csv"), index=False)
print(out.round(3).to_string())
