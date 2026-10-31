"""
Summaries, tests and figures for sweeps.py (O3 extension v2).
    python analyze_sweeps.py   ->  results/sweeps_report.md, figures/v2_*.pdf/png
"""
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results"); F = os.path.join(HERE, "figures"); os.makedirs(F, exist_ok=True)
g = pd.read_csv(os.path.join(R, "sweeps_grid.csv")); t = pd.read_csv(os.path.join(R, "sweeps_tuned.csv"))
DS = ["NSL-KDD", "UNSW-NB15", "MQTTEEB-D2025"]
TH = ["pi/8", "pi/4", "pi/2", "pi", "2pi"]; THL = ["π/8", "π/4", "π/2", "π", "2π"]
Q = [3, 5, 7, 9, 11]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
                     "savefig.dpi": 300})
BLUE = "#2a78d6"
def ci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
def save(fig, name):
    for e in ["pdf", "png"]:
        fig.savefig(os.path.join(F, f"{name}.{e}"), bbox_inches="tight")
    plt.close(fig)
def wil(a, b):
    d = np.asarray(a) - np.asarray(b)
    return np.nan if np.allclose(d, 0) else stats.wilcoxon(a, b, zero_method="zsplit").pvalue

L = ["# O3 extension v2: qubit sweep, bandwidth sweep, tuned baselines", ""]
q = g[g.model == "QSVM"]; c = g[g.fmap == "Classical"]

# ---------- 1. qubit sweep at theta = pi/2 (bounded encoding of v1)
L += ["## 1. Qubit sweep (θ_max = π/2, C = 1): mean F1 over 10 seeds", "",
      "| Dataset | Model / map | " + " | ".join(f"q={x}" for x in Q) + " |", "|---|---|" + "---|" * len(Q)]
for ds in DS:
    for mn in ["Z", "ZZ-full-r1", "ZZ-linear-r1", "ZZ-full-r2", "ZZ-linear-r2"]:
        v = [q[(q.dataset == ds) & (q.theta == "pi/2") & (q.fmap == mn) & (q.qubits == x)].f1.mean() for x in Q]
        L.append(f"| {ds} | QSVM {mn} | " + " | ".join(f"{a:.3f}" for a in v) + " |")
    for mod in ["SVM", "RandomForest"]:
        v = [c[(c.dataset == ds) & (c.model == mod) & (c.qubits == x)].f1.mean() for x in Q]
        L.append(f"| {ds} | {mod} | " + " | ".join(f"{a:.3f}" for a in v) + " |")

# concentration
L += ["", "## 2. Kernel concentration: variance of off-diagonal entries (ZZ-full-r1)", "",
      "| Dataset | θ_max | " + " | ".join(f"q={x}" for x in Q) + " |", "|---|---|" + "---|" * len(Q)]
for ds in DS:
    for th in TH:
        v = [q[(q.dataset == ds) & (q.theta == th) & (q.fmap == "ZZ-full-r1") & (q.qubits == x)].k_offdiag_var.mean() for x in Q]
        L.append(f"| {ds} | {th} | " + " | ".join(f"{a:.2e}" for a in v) + " |")

# ---------- 3. best theta per q (bandwidth)
L += ["", "## 3. Bandwidth: best θ_max per qubit count (QSVM, best map, mean F1)", "",
      "| Dataset | " + " | ".join(f"q={x}" for x in Q) + " |", "|---|" + "---|" * len(Q)]
for ds in DS:
    cells = []
    for x in Q:
        s = q[(q.dataset == ds) & (q.qubits == x)].groupby(["theta", "fmap"]).f1.mean()
        (th, mn), v = s.idxmax(), s.max()
        cells.append(f"{v:.3f} ({th}, {mn})")
    L.append(f"| {ds} | " + " | ".join(cells) + " |")

# ---------- 4. tuned comparison
L += ["", "## 4. Tuned models (3-fold CV on training data), mean over 10 seeds", ""]
mets = ["f1", "auprc", "fpr", "macro_attack_recall", "worst_attack_recall"]
L += ["| Dataset | q | Model | " + " | ".join(mets) + " | QSVM − model F1 (p) |", "|---|---|---|" + "---|" * (len(mets) + 1)]
tests = []
for ds in DS:
    for x in sorted(t.qubits.unique()):
        d = t[(t.dataset == ds) & (t.qubits == x)]
        qs = d[d.model == "QSVM (tuned)"].set_index("seed").f1
        for mod in ["QSVM (tuned)", "SVM (tuned)", "RandomForest (tuned)", "HistGradientBoosting (tuned)"]:
            dm = d[d.model == mod]
            vals = " | ".join(f"{dm[m].mean():.3f}" for m in mets)
            if mod == "QSVM (tuned)":
                diff = "—"
            else:
                o = dm.set_index("seed").f1; cmn = qs.index.intersection(o.index)
                p = wil(qs[cmn], o[cmn]); dd = (qs[cmn] - o[cmn]).mean()
                diff = f"{dd:+.3f} ({p:.2g})"; tests.append(dict(dataset=ds, qubits=x, vs=mod, diff=dd, p=p))
            L.append(f"| {ds} | {x} | {mod} | {vals} | {diff} |")
    sel = t[(t.dataset == ds) & (t.model == "QSVM (tuned)")]
    L.append(f"| {ds} | | QSVM choices | θ: {dict(sel.theta.value_counts())}; map: {dict(sel.fmap.value_counts())} | | | | | |")
# cost per flow
qt = t[t.model == "QSVM (tuned)"]
L += ["", "## 5. Inference cost of tuned QSVM (support vectors × 256 shots per flow)", "",
      "| Dataset | q | support vectors | shots per flow |", "|---|---|---|---|"]
for (ds, x), d in qt.groupby(["dataset", "qubits"]):
    L.append(f"| {ds} | {x} | {d.n_sv.mean():.0f} | {d.n_sv.mean()*256:,.0f} |")
open(os.path.join(R, "sweeps_report.md"), "w").write("\n".join(L) + "\n")
pd.DataFrame(tests).to_csv(os.path.join(R, "sweeps_tuned_tests.csv"), index=False)
print("\n".join(L))

# ---------- figures
# Fig A: F1 vs qubits (theta pi/2), ZZ-full-r1, ZZ-linear-r1, Z vs SVM/RF
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=False)
series = [("QSVM ZZ full", "QSVM", "ZZ-full-r1", BLUE, "-", "o"), ("QSVM ZZ linear", "QSVM", "ZZ-linear-r1", BLUE, "--", "s"),
          ("QSVM Z (no entanglement)", "QSVM", "Z", "#8fb6e8", "-", "^"),
          ("SVM (RBF)", "SVM", "Classical", "#8b919c", "-", "D"), ("Random Forest", "RandomForest", "Classical", INK2, ":", "v")]
for ax, ds in zip(axs, DS):
    for lab, mod, mn, col, ls, mk in series:
        d = g[(g.dataset == ds) & (g.model == mod) & (g.fmap == mn)]
        if mod == "QSVM": d = d[d.theta == "pi/2"]
        per = d.groupby(["qubits", "seed"]).f1.mean().reset_index()
        mu = [per[per.qubits == x].f1.mean() for x in Q]; e = [ci(per[per.qubits == x].f1) for x in Q]
        ax.errorbar(Q, mu, yerr=e, color=col, ls=ls, marker=mk, ms=3.5, lw=1.3, capsize=1.5, label=lab)
    ax.set_xticks(Q); ax.set_xlabel("Qubits (PCA components)"); ax.set_title(ds, fontsize=9, color=INK)
axs[0].set_ylabel("F1-score (θ_max = π/2)")
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.1), fontsize=7)
fig.tight_layout(); save(fig, "v2_fig_qubits")

# Fig B: bandwidth x qubits heatmaps of mean F1 (ZZ-full-r1) + concentration line
fig, axs = plt.subplots(1, 4, figsize=(7.6, 2.6), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.15]))
for ax, ds in zip(axs[:3], DS):
    M = np.array([[q[(q.dataset == ds) & (q.fmap == "ZZ-full-r1") & (q.theta == th) & (q.qubits == x)].f1.mean() for x in Q] for th in TH])
    im = ax.imshow(M, cmap="Blues", aspect="auto", vmin=np.nanmin(M) - 0.02, vmax=np.nanmax(M))
    for i in range(len(TH)):
        for j in range(len(Q)):
            ax.text(j, i, f"{100*M[i, j]:.1f}", ha="center", va="center", fontsize=5,
                    color="white" if M[i, j] > (np.nanmin(M) + np.nanmax(M)) / 2 else INK)
    ax.set_xticks(range(len(Q))); ax.set_xticklabels(Q); ax.set_yticks(range(len(TH))); ax.set_yticklabels(THL)
    ax.set_xlabel("Qubits"); ax.set_title(ds + "\n(F1 × 100, ZZ full)", fontsize=8, color=INK); ax.grid(False)
axs[0].set_ylabel("θ_max")
ax = axs[3]
for th, lab, a in zip(TH, THL, [0.25, 0.4, 0.6, 0.8, 1.0]):
    d = q[(q.dataset == "NSL-KDD") & (q.fmap == "ZZ-full-r1") & (q.theta == th)]
    ax.plot(Q, [d[d.qubits == x].k_offdiag_var.mean() for x in Q], color=BLUE, alpha=a, marker="o", ms=3, lw=1.2, label=lab)
ax.set_yscale("log"); ax.set_xticks(Q); ax.set_xlabel("Qubits"); ax.set_title("Off-diagonal variance\n(NSL-KDD)", fontsize=8.5, color=INK)
ax.legend(title="θ_max", fontsize=6, title_fontsize=6.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
fig.tight_layout(); save(fig, "v2_fig_bandwidth")
print("figures saved")
