"""Figure 7: the 14-item CRISP-QCML-IDS checklist (survey counts, 91 studies) vs the ICECET 2026 benchmark."""
import sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import os
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "Liberation Serif", "mathtext.fontset": "stix", "font.size": 7.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "pdf.fonttype": 42})
GREEN, RED, INK, BAR = "#00843D", "#B23A48", "#1A1A1A", "#C9CDD2"
# survey Table (91 studies): met count; Paper 1 status from the Appendix C rules (re-scored 28 Oct 2026)
items = [("C1", "Dataset specified", 71, 1), ("C2", "Splitting protocol stated", 45, 1), ("C3", "Preprocessing reported", 79, 1),
 ("C4", "Feature encoding described", 75, 1), ("C5", "Model specified", 83, 1), ("C6", "Training setup reported", 60, 0),
 ("C7", "Metrics beyond accuracy", 81, 1), ("C8", "Classical baseline", 81, 1), ("C9", "Quantum resources mentioned", 69, 1),
 ("C10", "Computational cost reported", 31, 1), ("C11", "Code or data available", 13, 0), ("C12", "Drift / retraining addressed", 15, 0),
 ("C13", "Data provenance stated", 47, 0), ("C14", "Failure consequences stated", 14, 0)]
phases = [("Data", 0, 2), ("Modeling", 3, 5), ("Evaluation", 6, 8), ("Deployment", 9, 11), ("Governance", 12, 13)]
fig, ax = plt.subplots(figsize=(3.45, 2.95))
n = len(items); y = np.arange(n)[::-1]
pct = np.array([m/91*100 for _, _, m, _ in items])
ax.barh(y, pct, color=BAR, height=0.66, zorder=2)
for yy, (cid, lab, m, ok), p in zip(y, items, pct):
    ax.text(p + 1.5, yy, f"{p:.0f}%", va="center", fontsize=6.6, color="#555555")
    ax.text(110, yy, "\u2713" if ok else "\u2717", va="center", ha="center", fontsize=9,
            color=GREEN if ok else RED, fontfamily="DejaVu Sans", fontweight="bold")
ax.set_yticks(y); ax.set_yticklabels([f"{c}  {l}" for c, l, _, _ in items], fontsize=7)
for name, a, b in phases:
    ya, yb = y[a], y[b]
    if b < n-1: ax.axhline(yb - 0.5, color="#DDDDDD", lw=0.5, xmin=0, xmax=1, clip_on=False)
ax.set_xlim(0, 116); ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xlabel("Studies meeting the item (% of 91)")
ax.text(110, n - 0.3, "[6]", ha="center", va="bottom", fontsize=7, color=INK, fontweight="bold")
ax.text(50, n - 0.3, "Field (91 studies)", ha="center", va="bottom", fontsize=7, color=INK, fontweight="bold")
ax.set_ylim(-0.6, n - 0.4 + 0.9)
ax.tick_params(axis="y", length=0)
fig.tight_layout(pad=0.3); fig.savefig(f"{OUT}/fig_checklist.pdf")
print("ok")
