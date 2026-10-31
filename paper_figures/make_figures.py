"""Figures 1-6 of the ICAIC 2027 paper. Run from the repository root after make_table2.py:
    python paper_figures/make_figures.py [output_dir]
"""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from scipy import stats
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "paper_figures", "out"); os.makedirs(OUT, exist_ok=True)
P = {"sweeps_grid.csv": "extension_v2/results/sweeps_grid.csv", "sweeps_tuned.csv": "extension_v2/results/sweeps_tuned.csv",
     "svm_recheck_per_seed.csv": "extension_v8/results/svm_recheck_per_seed.csv", "main.csv": "extension/results/main.csv",
     "table2_v3.csv": "paper_figures/table2.csv"}
F = lambda n: os.path.join(ROOT, P[n])
for f in fm.findSystemFonts():
    pass
plt.rcParams.update({"font.family": "Liberation Serif", "mathtext.fontset": "stix", "font.size": 8,
    "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42})
GREEN, ORANGE, GRAY, RED, INK = "#00843D", "#B7791F", "#6B6B6B", "#B23A48", "#1A1A1A"
DS = ["NSL-KDD", "UNSW-NB15", "MQTTEEB-D2025"]; Q = [3, 5, 7, 9, 11]
ci = lambda x: stats.t.ppf(0.975, len(x)-1) * np.std(x, ddof=1) / np.sqrt(len(x))

# ---- Fig 1: wraparound (single-qubit Z-map kernel = cos^2 of angle difference)
fig, ax = plt.subplots(figsize=(3.45, 1.75))
u = np.linspace(0, 1, 400)
ax.plot(u, np.cos(u*2*np.pi)**2, color=RED, lw=1.4, label=r"$\theta_{\max}=2\pi$")
ax.plot(u, np.cos(u*np.pi/2)**2, color=GREEN, lw=1.4, label=r"$\theta_{\max}=\pi/2$")
ax.plot(u, np.cos(u*np.pi)**2, color=GRAY, lw=1.0, ls="--", label=r"$\theta_{\max}=\pi$")
ax.scatter([0.5, 1.0], [1, 1], s=14, color=RED, zorder=3)
ax.text(0.53, 1.06, "distinct values collide (kernel = 1)", fontsize=7, color=RED, va="bottom")
ax.set_xlim(0, 1); ax.set_ylim(-0.02, 1.2); ax.set_yticks([0, 0.5, 1])
ax.set_xlabel("Distance between two feature values (fraction of the range)")
ax.set_ylabel("Kernel value")
ax.legend(frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=1.6, columnspacing=1.0)
fig.tight_layout(pad=0.3); fig.savefig(f"{OUT}/fig_wrap.pdf")

# ---- Fig qubits (redesign): QSVM ZZ-full with CI band vs SVM and RF
g = pd.read_csv(F("sweeps_grid.csv"))
fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.05))
series = [("QSVM (ZZ, full entanglement), 95% CI", "QSVM", "ZZ-full-r1", GREEN, "o", 1.7, 0.16),
          ("Random Forest", "RandomForest", "Classical", ORANGE, "s", 1.2, 0),
          ("SVM (RBF)", "SVM", "Classical", GRAY, "D", 1.2, 0)]
for ax, ds in zip(axs, DS):
    ends = []
    for lab, mod, mn, col, mk, lw, al in series:
        d = g[(g.dataset == ds) & (g.model == mod) & (g.fmap == mn)]
        if mod == "QSVM": d = d[d.theta == "pi/2"]
        per = d.groupby(["qubits", "seed"]).f1.mean().reset_index()
        mu = np.array([per[per.qubits == x].f1.mean() for x in Q]); e = np.array([ci(per[per.qubits == x].f1) for x in Q])
        if al: ax.fill_between(Q, mu-e, mu+e, color=col, alpha=al, lw=0)
        ax.plot(Q, mu, color=col, lw=lw, marker=mk, ms=3.2, label=lab)
        ends.append((mu[-1], lab, col))
    ax.set_xticks(Q); ax.set_xlim(2.6, 11.4); ax.set_title(ds, color=INK)
    ax.grid(axis="y", lw=0.4, color="#DDDDDD"); ax.set_xlabel("Qubits (PCA components)")
axs[0].set_ylabel(r"F1-score ($\theta_{\max}=\pi/2$)")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02), fontsize=7.5)
fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.92)); fig.savefig(f"{OUT}/fig_qubits_v3.pdf")

# ---- Fig forest: QSVM minus each tuned classical model, paired 95% CI, Holm
t = pd.read_csv(F("sweeps_tuned.csv")); w = pd.read_csv(F("svm_recheck_per_seed.csv")).assign(model="SVM (tuned)")
t = pd.concat([t[t.model != "SVM (tuned)"], w[t.columns.intersection(w.columns)]], ignore_index=True)
tab = pd.read_csv(F("table2_v3.csv"))
MOD = [("SVM (tuned)", "SVM (RBF)", GRAY, "D", 0.22), ("RandomForest (tuned)", "Random Forest", ORANGE, "s", 0.0),
       ("HistGradientBoosting (tuned)", "Gradient boosting", "#5B4A8B", "^", -0.22)]
rows = [(ds, q) for ds in DS for q in (3, 7)]
fig, ax = plt.subplots(figsize=(3.45, 2.9))
for i, (ds, q) in enumerate(rows):
    y = len(rows) - 1 - i
    piv = t[(t.dataset == ds) & (t.qubits == q)].pivot_table(index="seed", columns="model", values="f1")
    for m, lab, col, mk, off in MOD:
        d = piv["QSVM (tuned)"] - piv[m]; mu, e = d.mean(), ci(d)
        ph = tab[(tab.dataset == ds) & (tab.q == q) & (tab.model == m)].p_holm.iloc[0]
        sig = ph < 0.05
        ax.errorbar(mu, y+off, xerr=e, fmt=mk, ms=3.6, color=col, mfc=col if sig else "white", mew=0.9, elinewidth=0.9, capsize=1.5,
                    label=lab if i == 0 else None)
    if i % 2 == 1 and i < len(rows)-1: ax.axhline(y-0.5, color="#DDDDDD", lw=0.5)
ax.axvline(0, color=INK, lw=0.7)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"{ds.split('-')[0] if ds!='MQTTEEB-D2025' else 'MQTTEEB'}, $q$={q}" for ds, q in rows][::-1])
ax.set_xlabel(r"$\Delta$F1 = QSVM $-$ tuned classical model (95% CI, 10 seeds)")
ax.text(0.0015, len(rows)-0.42, "QSVM better $\\rightarrow$", fontsize=6.8, color=INK, va="center")
ax.text(-0.0015, len(rows)-0.42, "$\\leftarrow$ classical better", fontsize=6.8, color=INK, ha="right", va="center")
ax.set_ylim(-0.5, len(rows)-0.15)
ax.legend(frameon=False, fontsize=6.8, loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=3, handletextpad=0.2, columnspacing=0.8)
ax.grid(axis="x", lw=0.4, color="#EEEEEE")
fig.tight_layout(pad=0.3); fig.savefig(f"{OUT}/fig_forest.pdf")
print("ok")

# ---- Fig cost: shots per classified flow (support vectors x 256 shots), tuned QSVM
qt = t[t.model == "QSVM (tuned)"]
fig, ax = plt.subplots(figsize=(3.45, 1.55))
labs, vals = [], []
for ds in DS:
    for q in (3, 7):
        sv = qt[(qt.dataset == ds) & (qt.qubits == q)].n_sv.mean()
        labs.append(f"{ds.split('-')[0] if ds != 'MQTTEEB-D2025' else 'MQTTEEB'}, $q$={q}"); vals.append(sv * 256)
y = np.arange(len(vals))[::-1]
ax.barh(y, np.array(vals)/1000, color=ORANGE, height=0.62)
for yy, v in zip(y, vals):
    ax.text(v/1000 + 2, yy, f"{v/1000:.0f}k shots  ({v/2000:.0f} s)", va="center", fontsize=7, color=INK)
ax.set_yticks(y); ax.set_yticklabels(labs); ax.set_xlim(0, 215)
ax.set_xlabel("Shots per classified flow (thousands)")
fig.tight_layout(pad=0.3); fig.savefig(f"{OUT}/fig_cost.pdf")
print("cost ok")

# ---- Fig encoding (restyled): (a) KTA, (b) dF1 bounded - 2pi
m = pd.read_csv(F("main.csv")); qq = m[(m.protocol == "clean") & m.kta.notna()]
BL = "#9AA0A6"; BLUEQ, PEG = GREEN, "#5B4A8B"
fig, axs = plt.subplots(1, 2, figsize=(7.16, 1.85))
w = 0.36; DSL = ["NSL-KDD", "UNSW-NB15", "MQTTEEB-D2025"]
for i, (enc, lab, col) in enumerate([("twopi", r"$[0,2\pi]$", BL), ("bounded", r"$[0,\pi/2]$ + clip", GREEN)]):
    k = qq[qq.encoding == enc].groupby(["dataset", "seed"]).kta.mean().reset_index()
    mu = [k[k.dataset == d].kta.mean() for d in DS]; e = [ci(k[k.dataset == d].kta) for d in DS]
    axs[0].bar(np.arange(3) + (i-0.5)*w, mu, w*0.92, yerr=e, color=col, capsize=2, label=lab, error_kw=dict(ecolor=INK, lw=0.7))
axs[0].set_xticks(range(3)); axs[0].set_xticklabels(DSL); axs[0].set_ylabel("Kernel-target alignment")
axs[0].legend(frameon=False, fontsize=7.5, loc="upper right"); axs[0].set_title("(a) Alignment of the training kernel")
p = qq.pivot_table(index=["dataset", "selector", "k", "sample", "fmap", "model", "seed"], columns="encoding", values="f1").dropna().reset_index()
p["d"] = p.bounded - p.twopi; dd = p.groupby(["dataset", "model", "seed"]).d.mean().reset_index()
for i, (mod, lab, col) in enumerate([("QSVM", "QSVM", GREEN), ("PegasosQSVC", "PegasosQSVC", PEG)]):
    mu = [dd[(dd.dataset == d) & (dd.model == mod)].d.mean() for d in DS]; e = [ci(dd[(dd.dataset == d) & (dd.model == mod)].d) for d in DS]
    axs[1].bar(np.arange(3) + (i-0.5)*w, mu, w*0.92, yerr=e, color=col, capsize=2, label=lab, error_kw=dict(ecolor=INK, lw=0.7))
axs[1].axhline(0, color=INK, lw=0.7); axs[1].set_xticks(range(3)); axs[1].set_xticklabels(DSL)
axs[1].set_ylabel(r"$\Delta$F1, bounded $-$ $[0,2\pi]$"); axs[1].legend(frameon=False, fontsize=7.5, loc="upper left")
axs[1].set_title("(b) F1-score change from the bounded range")
for ax in axs: ax.grid(axis="y", lw=0.4, color="#E5E5E5"); ax.set_axisbelow(True)
fig.tight_layout(pad=0.3); fig.savefig(f"{OUT}/fig_encoding_v3.pdf")

# ---- Fig bandwidth (restyled)
TH = ["pi/8", "pi/4", "pi/2", "pi", "2pi"]; THL = [r"$\pi/8$", r"$\pi/4$", r"$\pi/2$", r"$\pi$", r"$2\pi$"]
fig, axs = plt.subplots(1, 4, figsize=(7.16, 2.05), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.05]))
for ax, ds in zip(axs[:3], DS):
    M = np.array([[g[(g.dataset == ds) & (g.model == "QSVM") & (g.fmap == "ZZ-full-r1") & (g.theta == th) & (g.qubits == x)].f1.mean() for x in Q] for th in TH])
    ax.imshow(M, cmap="Greens", aspect="auto", vmin=M.min()-0.01, vmax=M.max())
    for i in range(len(TH)):
        for j in range(len(Q)):
            v = M[i, j]; ax.text(j, i, f"{100*v:.1f}", ha="center", va="center", fontsize=6.4,
                                 color="white" if v > (M.min()+M.max())/2 else INK, fontweight="bold" if v == M[:, j].max() else "normal")
    ax.set_xticks(range(len(Q))); ax.set_xticklabels(Q); ax.set_yticks(range(len(TH))); ax.set_yticklabels(THL)
    ax.set_xlabel("Qubits"); ax.set_title(ds); ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
axs[0].set_ylabel(r"Encoding range $\theta_{\max}$")
ax = axs[3]; shades = ["#C7E9C0", "#A1D99B", "#74C476", "#31A354", "#006D2C"]
for th, lab, col in zip(TH, THL, shades):
    d = g[(g.dataset == "NSL-KDD") & (g.model == "QSVM") & (g.fmap == "ZZ-full-r1") & (g.theta == th)]
    vv = [d[d.qubits == x].k_offdiag_var.mean() for x in Q]
    ax.plot(Q, vv, color=col if th != "pi/8" else "#9FD39A", marker="o", ms=2.8, lw=1.3)
    ends_y = vv[-1]; ax.text(11.4, ends_y, lab, fontsize=6.8, va="center", color=INK)
ax.set_yscale("log"); ax.set_xticks(Q); ax.set_xlim(2.5, 12.9); ax.set_xlabel("Qubits")
ax.set_title(r"Kernel variance (NSL-KDD)")
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
ax.yaxis.set_major_locator(FixedLocator([0.02, 0.05, 0.1])); ax.yaxis.set_major_formatter(FixedFormatter(["0.02", "0.05", "0.10"])); ax.yaxis.set_minor_locator(NullLocator())
fig.tight_layout(pad=0.3, w_pad=0.6); fig.savefig(f"{OUT}/fig_bandwidth_v3.pdf")
print("enc bw ok")
