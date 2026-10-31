"""
Paper figures for the O3 extension. Reads results/*.csv, writes figures/*.png and *.pdf.
    python make_figures.py
"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda f: pd.read_csv(os.path.join(HERE, "results", f))
OUT = os.path.join(HERE, "figures"); os.makedirs(OUT, exist_ok=True)

# validated categorical slots (light surface), fixed order; markers give a second encoding for print
C = {"QSVM": "#2a78d6", "PegasosQSVC": "#eb6834", "SVM": "#1baf7a", "RandomForest": "#6250d6"}
MK = {"QSVM": "o", "PegasosQSVC": "s", "SVM": "^", "RandomForest": "D"}
LAB = {"QSVM": "QSVM", "PegasosQSVC": "PegasosQSVC", "SVM": "SVM (RBF)", "RandomForest": "Random Forest"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "savefig.dpi": 300})
DS = ["NSL-KDD", "UNSW-NB15", "MQTTEEB-D2025"]

def ci(x):
    x = np.asarray(x, float); return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))

def save(fig, name):
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------- Fig. 1: effect of each correction
m = R("main.csv")
m["stage"] = m.protocol + "/" + m.encoding
stages = ["published/twopi", "clean/twopi", "clean/bounded"]
names = ["Published\npipeline", "Leak-free\n[0, 2π]", "Leak-free\n[0, π/2]+clip"]
# per seed: mean over all configurations (selector, k, sample, feature map), then mean ± CI over the 10 seeds
g = m.groupby(["dataset", "stage", "model", "seed"]).f1.mean().reset_index()
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=False)
for ax, ds in zip(axs, DS):
    for mod in ["RandomForest", "SVM", "QSVM", "PegasosQSVC"]:
        mu = [g[(g.dataset == ds) & (g.stage == s) & (g.model == mod)].f1.mean() for s in stages]
        e = [ci(g[(g.dataset == ds) & (g.stage == s) & (g.model == mod)].f1) for s in stages]
        ax.errorbar(range(3), mu, yerr=e, color=C[mod], marker=MK[mod], ms=4, lw=1.6, capsize=2, label=LAB[mod])
    ax.set_xticks(range(3)); ax.set_xticklabels(names, fontsize=7); ax.set_title(ds, fontsize=9, color=INK)
    ax.set_xlim(-0.25, 2.25)
axs[0].set_ylabel("F1 (mean over configurations, 95% CI over 10 seeds)", fontsize=7)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.08), fontsize=7.5)
fig.tight_layout(); save(fig, "fig1_effect_of_corrections")

# ---------------------------------------------------------------- Fig. 2: encoding range -> alignment and F1
q = m[(m.protocol == "clean") & m.kta.notna()]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.4))
w = 0.36
for i, (enc, lab, col, hatch) in enumerate([("twopi", "[0, 2π]", "#c3c2b7", ""), ("bounded", "[0, π/2] + clip", "#2a78d6", "")]):
    k = q[q.encoding == enc].groupby(["dataset", "seed"]).kta.mean().reset_index()
    mu = [k[k.dataset == d].kta.mean() for d in DS]; e = [ci(k[k.dataset == d].kta) for d in DS]
    axs[0].bar(np.arange(3) + (i - 0.5) * w, mu, w * 0.92, yerr=e, color=col, capsize=2, label=lab, edgecolor="none", error_kw=dict(ecolor=INK2, lw=0.9))
axs[0].set_xticks(range(3)); axs[0].set_xticklabels(DS, fontsize=7.5); axs[0].set_ylabel("Kernel-target alignment (centered)")
axs[0].legend(fontsize=7, loc="upper right"); axs[0].set_title("(a) Alignment of the training kernel", fontsize=8.5, color=INK)
p = q.pivot_table(index=["dataset", "selector", "k", "sample", "fmap", "model", "seed"], columns="encoding", values="f1").dropna().reset_index()
p["d"] = p.bounded - p.twopi
dd = p.groupby(["dataset", "model", "seed"]).d.mean().reset_index()
for i, mod in enumerate(["QSVM", "PegasosQSVC"]):
    mu = [dd[(dd.dataset == d) & (dd.model == mod)].d.mean() for d in DS]
    e = [ci(dd[(dd.dataset == d) & (dd.model == mod)].d) for d in DS]
    axs[1].bar(np.arange(3) + (i - 0.5) * w, mu, w * 0.92, yerr=e, color=C[mod], capsize=2, label=LAB[mod], edgecolor="none", error_kw=dict(ecolor=INK2, lw=0.9))
axs[1].axhline(0, color=INK2, lw=0.8)
axs[1].set_xticks(range(3)); axs[1].set_xticklabels(DS, fontsize=7.5); axs[1].set_ylabel("ΔF1, bounded − [0, 2π]")
axs[1].legend(fontsize=7, loc="lower left"); axs[1].set_title("(b) F1 change from the bounded encoding", fontsize=8.5, color=INK)
fig.tight_layout(); save(fig, "fig2_bounded_encoding")

# ---------------------------------------------------------------- Fig. 3: finite shots
s = R("shots.csv")
ex = m[(m.dataset == "NSL-KDD") & (m.protocol == "clean") & (m.selector == "QKA_MI") & (m.k == 3) & (m["sample"] == 0.05)]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.4), sharey=True)
for ax, (enc, t) in zip(axs, [("twopi", "(a) [0, 2π]"), ("bounded", "(b) [0, π/2] + clip")]):
    for mod in ["QSVM", "PegasosQSVC"]:
        a = s[(s.encoding == enc) & (s.model == mod)].groupby(["shots", "seed"]).f1.mean().reset_index()
        xs = sorted(a.shots.astype(int).unique())
        mu = [a[a.shots.astype(int) == x].f1.mean() for x in xs]; e = [ci(a[a.shots.astype(int) == x].f1) for x in xs]
        ax.errorbar(xs, mu, yerr=e, color=C[mod], marker=MK[mod], ms=4, lw=1.6, capsize=2, label=f"{LAB[mod]}, shots")
        exv = ex[(ex.encoding == enc) & (ex.model == mod)].groupby("seed").f1.mean().mean()
        ax.axhline(exv, color=C[mod], lw=1, ls="--", label=f"{LAB[mod]}, exact")
    ax.set_xscale("log", base=2); ax.set_xticks([256, 1024, 4096]); ax.set_xticklabels(["256", "1,024", "4,096"])
    ax.set_xlabel("Shots per kernel entry"); ax.set_title(t, fontsize=8.5, color=INK)
axs[0].set_ylabel("F1 (NSL-KDD, QKA-MI, k = 3)")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.09), fontsize=7)
fig.tight_layout(); save(fig, "fig3_shots")

# ---------------------------------------------------------------- Fig. 4: preprocessing order vs class balance
a = R("ablation_order.csv")
c = m[m.protocol == "clean"]
key = ["dataset", "encoding", "selector", "k", "sample", "fmap", "model", "seed"]
j = a.merge(c[key + ["f1"]], on=key, suffixes=("_pre", "_post"))
j["d"] = j.f1_pre - j.f1_post
share = {"NSL-KDD": 47, "UNSW-NB15": 64, "MQTTEEB-D2025": 88}
dd = j.groupby(["dataset", "model", "seed"]).d.mean().reset_index()
fig, ax = plt.subplots(figsize=(3.5, 2.7))
models = ["RandomForest", "SVM", "QSVM", "PegasosQSVC"]
bw = 0.19
for i, mod in enumerate(models):
    xs = np.arange(3) + (i - 1.5) * bw
    mu = [dd[(dd.dataset == d) & (dd.model == mod)].d.mean() for d in DS]
    e = [ci(dd[(dd.dataset == d) & (dd.model == mod)].d) for d in DS]
    ax.bar(xs, mu, bw * 0.9, color=C[mod], label=LAB[mod], edgecolor="none",
           yerr=e, capsize=1.5, error_kw=dict(ecolor=INK2, lw=0.7))
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xticks(range(3))
ax.set_xticklabels([f"{d.replace('-D2025', '')}\n{share[d]}% attacks" for d in DS], fontsize=8)
ax.tick_params(axis="y", labelsize=8)
ax.set_ylabel("ΔF1 (fit before − after split)", fontsize=8)
ax.set_ylim(-0.24, 0.07)
ax.grid(axis="x", visible=False)
ax.legend(fontsize=7, ncol=2, loc="lower left", handlelength=1.0, columnspacing=0.8)
fig.tight_layout(); save(fig, "fig4_preprocessing_order")
print("figures ->", OUT, sorted(os.listdir(OUT)))
