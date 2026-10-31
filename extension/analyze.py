"""
Summarize results/<plan>.csv into tables with mean, 95% CI and paired Wilcoxon tests.

    python analyze.py --plan main
Writes results/<plan>_summary.csv, results/<plan>_tests.csv and results/<plan>_report.md
"""
import argparse, os
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
GROUP = ["dataset", "protocol", "encoding", "selector", "k", "sample", "shots", "model", "fmap"]

def ci95(x):
    x = np.asarray(x, float); n = len(x)
    if n < 2:
        return np.nan
    return stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)

def wilcoxon(a, b):
    d = np.asarray(a) - np.asarray(b)
    if len(d) < 2 or np.allclose(d, 0):
        return np.nan
    return stats.wilcoxon(a, b, zero_method="zsplit", alternative="two-sided").pvalue

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--plan", default="main"); a = ap.parse_args()
    df = pd.read_csv(os.path.join(HERE, "results", f"{a.plan}.csv"))
    df["shots"] = df["shots"].astype(str)
    if "mqtt_drop_epoch" in df.columns:
        df["protocol"] = np.where(df["mqtt_drop_epoch"].astype(str) == "True", df["protocol"] + "+no_epoch", df["protocol"])

    # 1) mean +- 95% CI per configuration
    g = df.groupby(GROUP)
    summ = g.agg(n_seeds=("f1", "size"), f1_mean=("f1", "mean"), acc_mean=("accuracy", "mean"),
                 train_s=("train_s", "mean")).reset_index()
    summ["f1_ci95"] = g["f1"].apply(ci95).values
    for c in ["kta", "k_offdiag_var", "alias_rate", "rot_span", "test_out_of_range"]:
        if c in df.columns:
            summ[c] = g[c].mean().values
    summ.to_csv(os.path.join(HERE, "results", f"{a.plan}_summary.csv"), index=False)

    # 2) paired tests across seeds
    tests = []
    key = ["dataset", "selector", "k", "sample", "shots"]
    # (a) best quantum vs Random Forest, within each protocol/encoding
    for (ds, proto, enc, sel, k, sp, sh), d in df.groupby(["dataset", "protocol", "encoding", "selector", "k", "sample", "shots"]):
        rf = d[d.model == "RandomForest"].set_index("seed")["f1"]
        q = d[d.fmap != "Classical"].groupby(["model", "fmap"])["f1"].mean()
        if q.empty:
            continue
        bm, bf = q.idxmax()
        qs = d[(d.model == bm) & (d.fmap == bf)].set_index("seed")["f1"]
        common = rf.index.intersection(qs.index)
        if len(common) >= 2:
            tests.append(dict(test="best quantum vs RF", dataset=ds, protocol=proto, encoding=enc, selector=sel, k=k,
                              sample=sp, shots=sh, model=f"{bm}/{bf}", n=len(common),
                              mean_diff=float((qs[common] - rf[common]).mean()), p_value=wilcoxon(qs[common], rf[common])))
    # (b) effect of each fix: published -> clean(twopi) -> clean(bounded), same model and seed
    piv = df.pivot_table(index=key + ["model", "fmap", "seed"], columns=["protocol", "encoding"], values="f1")
    for (a1, b1), label in [((("published", "twopi")), "leakage fix (published -> clean, [0,2pi])"),
                            ((("clean", "twopi")), "encoding fix (clean [0,2pi] -> clean [0,pi/2]+clip)")]:
        nxt = ("clean", "twopi") if a1 == "published" else ("clean", "bounded")
        if (a1, b1) not in piv.columns or nxt not in piv.columns:
            continue
        sub = piv[[(a1, b1), nxt]].dropna().reset_index()
        for gk, d in sub.groupby(key + ["model", "fmap"]):
            if len(d) >= 2:
                x, y = d[(a1, b1)].values, d[nxt].values
                tests.append(dict(test=label, **dict(zip(key + ["model", "fmap"], gk)), n=len(d),
                                  mean_diff=float((y - x).mean()), p_value=wilcoxon(y, x)))
    tdf = pd.DataFrame(tests)
    tdf.to_csv(os.path.join(HERE, "results", f"{a.plan}_tests.csv"), index=False)

    # 3) feature stability of QKA-MI between encodings (Jaccard, same seed)
    stab = []
    q = df[(df.selector == "QKA_MI") & (df.protocol == "clean")].drop_duplicates(["dataset", "k", "sample", "seed", "encoding"])
    for gk, d in q.groupby(["dataset", "k", "sample", "seed"]):
        f = {r.encoding: set(str(r.features).split("|")) for r in d.itertuples()}
        if "twopi" in f and "bounded" in f:
            stab.append(dict(dataset=gk[0], k=gk[1], sample=gk[2], seed=gk[3],
                             jaccard=len(f["twopi"] & f["bounded"]) / len(f["twopi"] | f["bounded"])))
    sdf = pd.DataFrame(stab)

    # 4) short markdown report
    lines = [f"# Results report: plan '{a.plan}'", "",
             f"{len(df)} rows, {df.cfg_key.nunique()} configurations, seeds per configuration: "
             f"{int(summ.n_seeds.min())}–{int(summ.n_seeds.max())}.", ""]
    best = summ.sort_values("f1_mean", ascending=False).groupby(["dataset", "protocol", "encoding", "model"]).head(1)
    best = best.sort_values(["dataset", "protocol", "encoding", "model"])
    lines += ["## Best configuration per dataset, protocol, encoding and model (F1 mean ± 95% CI)", "",
              "| Dataset | Protocol | Encoding | Model | Map | Selector | k | Sample | F1 |", "|---|---|---|---|---|---|---|---|---|"]
    for r in best.itertuples():
        lines.append(f"| {r.dataset} | {r.protocol} | {r.encoding} | {r.model} | {r.fmap} | {r.selector} | {r.k} | {r.sample} | "
                     f"{r.f1_mean:.3f} ± {r.f1_ci95:.3f} |")
    if not tdf.empty:
        for t in tdf.test.unique():
            d = tdf[tdf.test == t]
            sig = d[d.p_value < 0.05]
            lines += ["", f"## {t}", "", f"{len(d)} paired comparisons; {len(sig)} significant at p < 0.05; "
                      f"median F1 change {d.mean_diff.median():+.3f}."]
    if not sdf.empty:
        lines += ["", "## QKA-MI feature stability across encodings", "",
                  f"Mean Jaccard overlap of selected features ([0,2π] vs [0,π/2]): {sdf.jaccard.mean():.2f} "
                  f"(1.0 = identical selections)."]
    if "alias_rate" in summ.columns:
        enc = summ.dropna(subset=["rot_span"]).groupby(["dataset", "protocol", "encoding"])[
            ["rot_span", "alias_rate", "test_out_of_range", "kta"]].mean().reset_index()
        lines += ["", "## Encoding audit (quantum rows)", "", "| Dataset | Protocol | Encoding | Rotation span (rad) | Alias rate | Test out of range | KTA |",
                  "|---|---|---|---|---|---|---|"]
        for r in enc.itertuples():
            lines.append(f"| {r.dataset} | {r.protocol} | {r.encoding} | {r.rot_span:.2f} | {r.alias_rate:.4f} | {r.test_out_of_range:.3f} | {r.kta:.3f} |")
    open(os.path.join(HERE, "results", f"{a.plan}_report.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
