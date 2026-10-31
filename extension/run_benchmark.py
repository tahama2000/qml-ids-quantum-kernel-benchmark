"""
Run the Objective 3 extension grid with checkpointing.

    python run_benchmark.py --plan main          # full grid (published + clean/twopi + clean/bounded), 10 seeds
    python run_benchmark.py --plan shots         # finite-shot study on NSL-KDD
    python run_benchmark.py --plan smoke         # 1-2 minute check that everything works

Results are appended to results/<plan>.csv after every configuration, so an
interrupted run resumes where it stopped. Progress goes to results/<plan>.log.
"""
import argparse, json, os, sys, time, traceback
import pandas as pd
import o3ext as E

HERE = os.path.dirname(os.path.abspath(__file__))

def default_paths():
    base = os.path.abspath(os.path.join(HERE, ".."))            # the "Objective 3 -QML benchmark" folder
    return {
        "mqtteeb": os.path.join(base, "MQTTEEB-D_dataset_All_Processed.csv"),
        "unsw_train": os.path.join(base, "UNSW_NB15", "UNSW_NB15_training-set.csv"),
        "unsw_test": os.path.join(base, "UNSW_NB15", "UNSW_NB15_testing-set(in).csv"),
        "nsl_kdd": os.path.join(base, "data", "KDDTrain+_20Percent.txt"),
    }

# sampling ratios as in the ICECET notebook cells
SAMPLES = {"NSL-KDD": [0.01, 0.05, 0.1], "UNSW-NB15": [0.002, 0.005], "MQTTEEB-D2025": [0.002, 0.005]}
SEEDS10 = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]                       # seed 42 = the published random state

def plan_configs(plan):
    cfgs = []
    if plan == "smoke":
        for proto, enc in [("published", "twopi"), ("clean", "twopi"), ("clean", "bounded")]:
            cfgs.append(dict(dataset="NSL-KDD", protocol=proto, encoding=enc, selector="RF", k=3, sample=0.01, seed=42))
        cfgs.append(dict(dataset="NSL-KDD", protocol="clean", encoding="bounded", selector="QKA_MI", k=3, sample=0.01, seed=42))
        return cfgs
    if plan == "main":
        for ds, samples in SAMPLES.items():
            for sample in samples:
                for k in [3, 5]:
                    for sel in ["RF", "QKA_MI"]:
                        for proto, enc in [("published", "twopi"), ("clean", "twopi"), ("clean", "bounded")]:
                            for seed in SEEDS10:
                                cfgs.append(dict(dataset=ds, protocol=proto, encoding=enc, selector=sel,
                                                 k=k, sample=sample, seed=seed))
        return cfgs
    if plan == "mqtt_epoch":                                      # MQTTEEB with and without the epoch-time column
        for sample in SAMPLES["MQTTEEB-D2025"]:
            for sel in ["RF", "QKA_MI"]:
                for enc in ["twopi", "bounded"]:
                    for seed in SEEDS10:
                        cfgs.append(dict(dataset="MQTTEEB-D2025", protocol="clean", encoding=enc, selector=sel,
                                         k=3, sample=sample, seed=seed, mqtt_drop_epoch=True))
        return cfgs
    if plan == "shots":
        for enc in ["twopi", "bounded"]:
            for shots in [256, 1024, 4096]:
                for seed in SEEDS10:
                    cfgs.append(dict(dataset="NSL-KDD", protocol="clean", encoding=enc, selector="QKA_MI",
                                     k=3, sample=0.05, seed=seed, shots=shots))
        return cfgs
    if plan == "ablation_order":                               # leak-free data, preprocessing before vs after split
        for ds, samples in SAMPLES.items():
            for sample in samples:
                for sel in ["RF", "QKA_MI"]:
                    for enc in ["twopi", "bounded"]:
                        for seed in SEEDS10:
                            cfgs.append(dict(dataset=ds, protocol="prefit", encoding=enc, selector=sel,
                                             k=3, sample=sample, seed=seed))
        return cfgs
    if plan == "time_split":                                   # MQTTEEB, train on earlier traffic, test on later
        for sample in SAMPLES["MQTTEEB-D2025"]:
            for sel in ["RF", "QKA_MI"]:
                for enc in ["twopi", "bounded"]:
                    for seed in SEEDS10:
                        cfgs.append(dict(dataset="MQTTEEB-D2025", protocol="clean", encoding=enc, selector=sel,
                                         k=3, sample=sample, seed=seed, mqtt_drop_epoch=True, time_split=True))
        return cfgs
    if plan == "pegasos_steps":                                # is Pegasos weaker, or under-trained at 100 steps?
        for ds, samples in SAMPLES.items():
            for sel in ["RF", "QKA_MI"]:
                for enc in ["twopi", "bounded"]:
                    for seed in SEEDS10:
                        cfgs.append(dict(dataset=ds, protocol="clean", encoding=enc, selector=sel, k=3,
                                         sample=samples[-1], seed=seed, pegasos_steps=1000))
        return cfgs
    if plan == "categorical":                                  # one-hot columns included (paper's description)
        for ds in ["NSL-KDD", "UNSW-NB15"]:
            for sel in ["RF", "QKA_MI"]:
                for enc in ["twopi", "bounded"]:
                    for seed in SEEDS10:
                        cfgs.append(dict(dataset=ds, protocol="clean", encoding=enc, selector=sel, k=3,
                                         sample=SAMPLES[ds][-1], seed=seed, include_categorical=True))
        return cfgs
    raise ValueError(plan)

_TCACHE = {}
def _time_order(c, paths):
    if not c.get("time_split"):
        return None
    if "mqtt" not in _TCACHE:
        df = pd.read_csv(paths["mqtteeb"]).dropna().reset_index(drop=True)
        _TCACHE["mqtt"] = pd.to_datetime(df["timestamp"]).values.astype("int64")
    return _TCACHE["mqtt"]

def key(c):
    return json.dumps({k: c.get(k) for k in sorted(c)}, sort_keys=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="smoke", choices=["smoke", "main", "mqtt_epoch", "shots", "ablation_order", "time_split", "pegasos_steps", "categorical"])
    ap.add_argument("--paths", default=None, help="JSON file overriding dataset paths")
    ap.add_argument("--only", default=None, help="restrict to one dataset name")
    args = ap.parse_args()

    paths = default_paths()
    if args.paths:
        paths.update(json.load(open(args.paths)))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    out = os.path.join(HERE, "results", f"{args.plan}.csv")
    log = open(os.path.join(HERE, "results", f"{args.plan}.log"), "a")
    done = set()
    if os.path.exists(out):
        done = set(pd.read_csv(out, usecols=["cfg_key"])["cfg_key"])

    try:
        import qiskit, qiskit_machine_learning, sklearn, numpy, imblearn, platform
        with open(os.path.join(HERE, "results", f"{args.plan}_env.txt"), "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M')} python {platform.python_version()} qiskit {qiskit.__version__} "
                    f"qiskit-ml {qiskit_machine_learning.__version__} sklearn {sklearn.__version__} "
                    f"numpy {numpy.__version__} imblearn {imblearn.__version__}\n")
    except Exception:
        pass
    cfgs = [c for c in plan_configs(args.plan) if not args.only or c["dataset"] == args.only]
    todo = [c for c in cfgs if key(c) not in done]
    print(f"{len(cfgs)} configurations in plan '{args.plan}', {len(todo)} still to run -> {out}", flush=True)

    cache = {}
    t_start = time.time()
    for i, c in enumerate(todo, 1):
        dk = (c["dataset"], c["protocol"], c.get("mqtt_drop_epoch", False), c.get("include_categorical", False))
        if dk not in cache:
            proto_data = "clean" if c["protocol"] == "prefit" else c["protocol"]
            cache[dk] = E.load_dataset(c["dataset"], paths, proto_data, c.get("mqtt_drop_epoch", False),
                                       include_categorical=c.get("include_categorical", False))
        X, y, strat = cache[dk]
        t = time.time()
        try:
            rows = E.run_config(X, y, strat, dataset=c["dataset"], protocol=c["protocol"], encoding=c["encoding"],
                                selector=c["selector"], k=c["k"], sample=c["sample"], seed=c["seed"],
                                shots=c.get("shots"), time_order=_time_order(c, paths),
                                pegasos_steps=c.get("pegasos_steps", 100))
            for r in rows:
                r["cfg_key"] = key(c); r["mqtt_drop_epoch"] = c.get("mqtt_drop_epoch", False)
            pd.DataFrame(rows).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
            msg = f"[{i}/{len(todo)}] ok {time.time()-t:6.1f}s  {key(c)}"
        except Exception as e:
            msg = f"[{i}/{len(todo)}] FAILED {key(c)}: {e!r}"
            log.write(traceback.format_exc())
        el = time.time() - t_start
        eta = el / i * (len(todo) - i)
        print(f"{msg}   elapsed {el/60:.1f} min, ETA {eta/60:.1f} min", flush=True)
        log.write(msg + "\n"); log.flush()
    print("Finished.", flush=True)

if __name__ == "__main__":
    main()
