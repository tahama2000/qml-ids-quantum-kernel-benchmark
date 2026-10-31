"""Compare the wide-gamma tuned SVM (v8) with the v2 tuned SVM, QSVM, RF and HistGB on the same splits (paired Wilcoxon, 10 seeds)."""
import os, numpy as np, pandas as pd
from scipy.stats import wilcoxon
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
v8 = pd.read_csv(os.path.join(R, "svm_recheck_per_seed.csv")); v2 = pd.read_csv(os.path.join(R, "v2_sweeps_tuned_input.csv"))
d = pd.concat([v2[["dataset", "seed", "qubits", "model", "f1", "fpr", "auroc"]], v8[["dataset", "seed", "qubits", "model", "f1", "fpr", "auroc"]]])
w = d.pivot_table(index=["dataset", "qubits", "seed"], columns="model", values="f1")
wp = lambda x: wilcoxon(x).pvalue if (x != 0).sum() > 1 else np.nan
rows = []
for (ds, q), g in w.groupby(level=[0, 1]):
    new, old = g["SVM (tuned, wide gamma)"], g["SVM (tuned)"]; qs = g["QSVM (tuned)"]
    best_c = g[["RandomForest (tuned)", "HistGradientBoosting (tuned)", "SVM (tuned, wide gamma)"]].mean().idxmax()
    rows.append(dict(dataset=ds, qubits=q, svm_v2=old.mean(), svm_wide=new.mean(), gain=(new - old).mean(), p_gain=wp(new - old),
                     qsvm=qs.mean(), rf=g["RandomForest (tuned)"].mean(), hgb=g["HistGradientBoosting (tuned)"].mean(),
                     qsvm_minus_svm_wide=(qs - new).mean(), p_q_vs_svm=wp(qs - new), best_classical_now=best_c))
o = pd.DataFrame(rows); o.to_csv(os.path.join(R, "svm_recheck_summary.csv"), index=False)
pd.set_option("display.width", 250); print(o.round(4).to_string(index=False))
