"""
o3ext.py — Objective 3 extension: leakage-free, multi-seed re-evaluation of the
ICECET 2026 quantum-kernel IDS benchmark.

Two protocols
-------------
published : reproduces the ICECET notebook ("QML for IDS.ipynb") step by step:
            feature selection, sampling, SMOTE, scaling, PCA and [0, 2pi] angle
            scaling on the full sample, then an unstratified 80/20 split.
            With seed=42 it is the published pipeline; other seeds vary every
            random step.
clean     : split first, then fit selection, SMOTE, scaling, PCA and angle
            scaling on the training part only. MQTTEEB's Attack_Binary column is
            removed from the features. Angle encoding is either 'twopi'
            ([0, 2pi], no clipping, as published) or 'bounded' ([0, pi/2] with
            clipping to the training range).

Kernels
-------
The fidelity kernel K(x, x') = |<phi(x)|phi(x')>|^2 is computed exactly from
statevectors, which equals what the original Qiskit reference Sampler returned.
Finite-shot kernels are drawn as Binomial(shots, K)/shots, the exact sampling
distribution of the all-zeros outcome in the compute-uncompute circuit.
"""
from __future__ import annotations

import os, time, warnings
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

MIN_QUBITS = 3
PCA_COMPONENTS = 3

# ---------------------------------------------------------------------------
# Feature maps (Qiskit circuit library; same maps as the ICECET notebook)
# ---------------------------------------------------------------------------
def _feature_map(name: str, d: int):
    from qiskit.circuit.library import z_feature_map, zz_feature_map, pauli_feature_map
    from qiskit.circuit import ParameterVector
    if name == "Z":
        return z_feature_map(d, reps=1)
    if name == "ZZ":
        return zz_feature_map(d, reps=1)
    if name == "Pauli":
        return pauli_feature_map(d, reps=1)          # default paulis = ['Z', 'ZZ']
    if name == "HybridPauliZZ":                       # Pauli o ZZ with shared parameters
        x = ParameterVector("x", d)
        a = pauli_feature_map(d, reps=1); b = zz_feature_map(d, reps=1)
        a = a.assign_parameters(dict(zip(sorted(a.parameters, key=lambda p: p.name), x)))
        b = b.assign_parameters(dict(zip(sorted(b.parameters, key=lambda p: p.name), x)))
        return a.compose(b)
    raise ValueError(name)

FEATURE_MAPS = ["Pauli", "ZZ", "Z", "HybridPauliZZ"]

def statevectors(X: np.ndarray, fmap: str) -> np.ndarray:
    from qiskit.quantum_info import Statevector
    circ = _feature_map(fmap, X.shape[1])
    params = sorted(circ.parameters, key=lambda p: p.name)
    out = np.empty((X.shape[0], 2 ** X.shape[1]), dtype=complex)
    for i, row in enumerate(X):
        out[i] = Statevector(circ.assign_parameters(dict(zip(params, row)))).data
    return out

def fidelity_kernel(SA: np.ndarray, SB: np.ndarray) -> np.ndarray:
    return np.abs(SA.conj() @ SB.T) ** 2

def shot_kernel(K: np.ndarray, shots: int, rng: np.random.Generator, symmetric: bool) -> np.ndarray:
    Kh = rng.binomial(shots, np.clip(K, 0, 1)) / shots
    if symmetric:
        Kh = np.triu(Kh, 1); Kh = Kh + Kh.T; np.fill_diagonal(Kh, 1.0)
        w, V = np.linalg.eigh(Kh)                    # PSD projection, as FidelityQuantumKernel(enforce_psd=True)
        Kh = (V * np.clip(w, 0, None)) @ V.T
    return Kh

# ---------------------------------------------------------------------------
# Datasets (loaders follow the notebook; paths are arguments)
# ---------------------------------------------------------------------------
NSL_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B_20Percent.txt"
NSL_COLS = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent',
    'hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root',
    'num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login',
    'count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate',
    'diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate',
    'dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate',
    'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','label','difficulty_level']

def load_dataset(name: str, paths: dict, protocol: str, mqtt_drop_epoch: bool = False,
                 include_categorical: bool = False):
    """Return X_raw (numeric DataFrame), y (binary Series), strat (labels used for stratified sampling).

    published : replicates the notebook exactly, including its side effects under pandas 2.x
                (one-hot dummies are bool and are silently dropped by select_dtypes(number);
                UNSW 'id' stays in X; MQTTEEB 'Attack_Binary' stays in X).
    clean     : same feature space, minus the known leaks (MQTTEEB Attack_Binary, UNSW id);
                include_categorical=True additionally keeps the one-hot columns the paper describes.
    """
    if name == "NSL-KDD":
        src = paths.get("nsl_kdd")
        if not src or not os.path.exists(src):
            src = NSL_URL                                         # falls back to the GitHub copy the notebook used
        df = pd.read_csv(src, names=NSL_COLS).drop(columns=["difficulty_level"])
        label = "label"
    elif name == "UNSW-NB15":
        df = pd.concat([pd.read_csv(paths["unsw_train"]), pd.read_csv(paths["unsw_test"])], ignore_index=True).dropna()
        label = "label"
    elif name == "MQTTEEB-D2025":
        df = pd.read_csv(paths["mqtteeb"]).dropna()
        label = "Target"
    else:
        raise ValueError(name)
    df = df.reset_index(drop=True)
    raw_label = df[label].astype(str).copy()                      # attack category before binarization
    attack_cat = df["attack_cat"].astype(str).copy() if "attack_cat" in df.columns else None

    if name == "MQTTEEB-D2025":
        y = (df[label] != 0).astype(int)
        X = df.drop(columns=[label])
        if protocol == "published":
            X["Attack_Binary"] = y.values                         # notebook: the label stays in X
        if mqtt_drop_epoch:
            X = X.drop(columns=["tcp_time_delta"])
        strat_pub = df[label]                                     # notebook stratifies MQTTEEB on Target
    else:
        if protocol == "clean" and "attack_cat" in df.columns:
            df = df.drop(columns=["attack_cat"])                  # label text must never become a feature
        cat = [c for c in df.select_dtypes(include=["object", "string"]).columns if c != label]
        df = pd.get_dummies(df, columns=cat)
        if include_categorical and protocol == "clean":
            dummies = [c for c in df.columns if df[c].dtype == bool]
            df[dummies] = df[dummies].astype(int)
        if df[label].dtype == object or str(df[label].dtype).startswith("str"):
            y = df[label].apply(lambda v: 0 if str(v).lower() in ["normal", "benign", "ok", "nominal", "0"] else 1)
        else:
            y = (df[label] != 0).astype(int)
        X = df.drop(columns=[label])
        strat_pub = y                                             # notebook stratifies on the binary label here
        if protocol == "clean" and "id" in X.columns:
            X = X.drop(columns=["id"])                            # UNSW row counter: ordered by class
    X = X.select_dtypes(include=[np.number]).astype(float)
    if protocol == "clean":
        strat = attack_cat if attack_cat is not None else raw_label
    else:
        strat = strat_pub
    return X.reset_index(drop=True), y.reset_index(drop=True).astype(int), pd.Series(np.asarray(strat))

# ---------------------------------------------------------------------------
# Angle encodings
# ---------------------------------------------------------------------------
def fit_angle_scaler(X_train: np.ndarray, encoding: str):
    lo, hi = X_train.min(axis=0), X_train.max(axis=0)
    top = 2 * np.pi if encoding == "twopi" else np.pi / 2
    clip = encoding == "bounded"
    def transform(X):
        rng_ = np.where(np.isclose(hi, lo), 1.0, hi - lo)
        Z = (X - lo) / rng_
        if clip:
            Z = np.clip(Z, 0.0, 1.0)
        Z = np.where(np.isclose(hi, lo), 0.0, Z)
        return top * Z
    return transform

def scale_0_2pi_published(X):                          # notebook's scale_for_quantum_0_2pi (fits on whatever it gets)
    X = np.asarray(X, float); out = np.zeros_like(X)
    for j in range(X.shape[1]):
        c = X[:, j]; lo, hi = c.min(), c.max()
        out[:, j] = 0.0 if np.isclose(hi, lo) else 2 * np.pi * (c - lo) / (hi - lo)
    return out

# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------
def select_rf(X: pd.DataFrame, y, k: int, seed: int):
    rf = RandomForestClassifier(random_state=seed, n_jobs=-1).fit(X, y)
    idx = np.argsort(rf.feature_importances_)[-k:][::-1]
    return list(X.columns[idx])

def _alignment(K, y):
    y = np.asarray(y); z = np.where(y == np.unique(y)[0], -1.0, 1.0); z -= z.mean()
    den = np.linalg.norm(K, "fro") * np.linalg.norm(z)
    return float(z @ K @ z / den) if den > 0 else 0.0

def select_qka_mi(X: pd.DataFrame, y, k: int, seed: int, angle_fn, prefilter_m=20, subsample=150):
    """QKA_MI_FAST from the notebook: MI prefilter (M=20) on a 150-sample subset, then greedy
    0.5*alignment(Z-feature-map kernel) + 0.5*mean MI. angle_fn maps a column block to angles."""
    Xn = X.values.astype(np.float32); yv = np.asarray(y)
    rng = np.random.RandomState(seed)
    if len(Xn) > subsample:
        sub = rng.choice(len(Xn), size=subsample, replace=False); Xs, ys = Xn[sub], yv[sub]
    else:
        Xs, ys = Xn, yv
    mi = mutual_info_classif(Xs, ys, discrete_features=False, random_state=seed)
    top = np.argsort(mi)[-prefilter_m:][::-1]
    Xc, mic = Xs[:, top], mi[top]
    sel, rem = [], list(range(Xc.shape[1]))
    while len(sel) < min(k, Xc.shape[1]):
        best, bj = -np.inf, None
        for j in rem:
            cand = sel + [j]
            try:
                S = statevectors(angle_fn(Xc[:, cand]), "Z"); qka = _alignment(fidelity_kernel(S, S), ys)
            except Exception:
                qka = 0.0
            s = 0.5 * qka + 0.5 * float(np.mean(mic[cand]))
            if s > best:
                best, bj = s, j
        sel.append(bj); rem.remove(bj)
    return list(X.columns[top[np.array(sel, dtype=int)]])

def ensure_min(X: pd.DataFrame, cols: list, seed: int):
    X = X[cols].copy()
    rng = np.random.RandomState(seed); i = 0
    while X.shape[1] < MIN_QUBITS:
        c = cols[i % len(cols)]; X[f"{c}__dup{i}"] = X[c].values + 1e-8 * rng.randn(len(X)); i += 1
    return X

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def kernel_diagnostics(K: np.ndarray, y, Z_std: np.ndarray, A_train: np.ndarray, A_test: np.ndarray):
    n = K.shape[0]; off = K[~np.eye(n, dtype=bool)]
    yv = np.asarray(y); z = np.where(yv == np.unique(yv)[0], -1.0, 1.0)
    H = np.eye(n) - 1.0 / n; Kc = H @ K @ H; Y = np.outer(z, z)
    den = np.linalg.norm(Kc, "fro") * np.linalg.norm(Y, "fro")
    kta = float(np.sum(Kc * Y) / den) if den > 0 else 0.0
    # aliasing: pairs that are far apart in standardized space yet near-identical in Hilbert space
    m = min(n, 600); idx = np.arange(m)
    D = np.sqrt(((Z_std[idx, None, :] - Z_std[None, idx, :]) ** 2).sum(-1))
    Ks = K[np.ix_(idx, idx)]; far = D > 1.0
    alias = float(np.mean(Ks[far] > 0.99)) if far.any() else 0.0
    span = float(2 * (A_train.max(axis=0) - A_train.min(axis=0)).max())   # effective rotation span (P(2x))
    oor = float(np.mean((A_test > A_train.max(axis=0) + 1e-9) | (A_test < A_train.min(axis=0) - 1e-9)))
    return dict(k_offdiag_mean=float(off.mean()), k_offdiag_var=float(off.var()), kta=kta,
                alias_rate=alias, rot_span=span, test_out_of_range=oor)

def _metrics(y_true, y_pred):
    return dict(accuracy=accuracy_score(y_true, y_pred), precision=precision_score(y_true, y_pred, zero_division=0),
                recall=recall_score(y_true, y_pred, zero_division=0), f1=f1_score(y_true, y_pred, zero_division=0))

# ---------------------------------------------------------------------------
# One configuration -> list of result rows
# ---------------------------------------------------------------------------
_SEL_CACHE = {}

def run_config(X_raw, y, strat, *, dataset, protocol, encoding, selector, k, sample, seed,
               feature_maps=FEATURE_MAPS, shots=None, smote=True, time_order=None, pegasos_steps=100):
    t0 = time.time(); rows = []
    base = dict(dataset=dataset, protocol=protocol, encoding=encoding, selector=selector, k=k,
                sample=sample, seed=seed, shots=shots or "exact", pegasos_steps=pegasos_steps)

    if protocol == "published":
        # 1) selection on the full data
        # selection on the full data does not depend on the sample ratio -> cache it
        ck = (dataset, selector, k, seed)
        if ck not in _SEL_CACHE:
            if selector == "RF":
                _SEL_CACHE[ck] = select_rf(X_raw, y, k, seed)
            else:
                _SEL_CACHE[ck] = select_qka_mi(X_raw, y, k, seed, scale_0_2pi_published)
        cols = _SEL_CACHE[ck]
        Xs = ensure_min(X_raw, cols, seed)
        # 2) stratified sample (notebook: StratifiedShuffleSplit, test_size = 1 - sample)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - sample, random_state=seed)
        idx, _ = next(sss.split(Xs, strat))
        Xp, yp = Xs.iloc[idx], y.iloc[idx]
        # 3) SMOTE, scale, PCA, angles on the whole sample
        if smote:
            Xp, yp = SMOTE(random_state=seed).fit_resample(Xp, yp)
        Z = StandardScaler().fit_transform(Xp)
        P = PCA(n_components=min(PCA_COMPONENTS, Z.shape[1])).fit_transform(Z)
        A = scale_0_2pi_published(P)
        # 4) unstratified 80/20 split
        itr, ite = train_test_split(np.arange(len(P)), test_size=0.2, random_state=seed)
        Ptr, Pte, Atr, Ate = P[itr], P[ite], A[itr], A[ite]
        ytr, yte = np.asarray(yp)[itr], np.asarray(yp)[ite]
        n_sample = len(idx)
    elif protocol == "prefit":
        # ablation: identical to 'clean' (same leak-free data, same sampling, stratified split)
        # except that selection, SMOTE, scaling, PCA and angles are fitted on the whole sample
        # BEFORE the split -- isolates the effect of the preprocessing order alone
        st = pd.Series(np.asarray(strat)).astype(str)
        vc = st.value_counts(); st = st.where(st.map(vc) >= 10, "rare_" + y.astype(str).values)
        vc = st.value_counts(); st = st.where(st.map(vc) >= 2, "class_" + y.astype(str).values)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - sample, random_state=seed)
        idx, _ = next(sss.split(X_raw, st))
        Xp, yp = X_raw.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
        if selector == "RF":
            cols = select_rf(Xp, yp, k, seed)
        else:
            cols = select_qka_mi(Xp, yp, k, seed, lambda b, _e=encoding: fit_angle_scaler(b, _e)(b))
        Xs = ensure_min(Xp, cols, seed); ys = yp.values
        if smote and min(np.bincount(ys)) >= 6:
            Xs, ys = SMOTE(random_state=seed).fit_resample(Xs, ys); ys = np.asarray(ys)
        Z = StandardScaler().fit_transform(Xs)
        P = PCA(n_components=min(PCA_COMPONENTS, Z.shape[1])).fit_transform(Z)
        A = fit_angle_scaler(P, encoding)(P)
        itr, ite = train_test_split(np.arange(len(P)), test_size=0.2, random_state=seed, stratify=ys)
        Ptr, Pte, Atr, Ate, ytr, yte = P[itr], P[ite], A[itr], A[ite], ys[itr], ys[ite]
        n_sample = len(idx)
    else:
        # 1) stratified sample on attack categories (categories with < 10 rows are pooled per class),
        #    then a stratified 80/20 split on the binary label
        st = pd.Series(np.asarray(strat)).astype(str)
        vc = st.value_counts()
        st = st.where(st.map(vc) >= 10, "rare_" + y.astype(str).values)
        vc = st.value_counts()
        st = st.where(st.map(vc) >= 2, "class_" + y.astype(str).values)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - sample, random_state=seed)
        idx, _ = next(sss.split(X_raw, st))
        Xp, yp = X_raw.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
        if time_order is not None:
            # time-aware split: train on the earliest 80% of the sampled flows, test on the latest 20%
            t_s = np.asarray(time_order)[idx]
            order = np.argsort(t_s, kind="stable"); cut = int(round(0.8 * len(order)))
            itr, ite = order[:cut], order[cut:]
        else:
            itr, ite = train_test_split(np.arange(len(Xp)), test_size=0.2, random_state=seed, stratify=yp)
        Xtr, Xte, ytr, yte = Xp.iloc[itr], Xp.iloc[ite], yp.iloc[itr].values, yp.iloc[ite].values
        # 2) selection on training data only
        if selector == "RF":
            cols = select_rf(Xtr, ytr, k, seed)
        else:
            def ang(block, _enc=encoding):
                return fit_angle_scaler(block, _enc)(block)
            cols = select_qka_mi(Xtr, ytr, k, seed, ang)
        Xtr, Xte = ensure_min(Xtr, cols, seed), ensure_min(Xte, cols, seed)
        # 3) SMOTE on training data only
        if smote and len(np.unique(ytr)) == 2 and min(np.bincount(ytr)) >= 6:
            Xtr, ytr = SMOTE(random_state=seed).fit_resample(Xtr, ytr); ytr = np.asarray(ytr)
        sc = StandardScaler().fit(Xtr); pca = PCA(n_components=min(PCA_COMPONENTS, Xtr.shape[1])).fit(sc.transform(Xtr))
        Ptr, Pte = pca.transform(sc.transform(Xtr)), pca.transform(sc.transform(Xte))
        ang = fit_angle_scaler(Ptr, encoding)
        Atr, Ate = ang(Ptr), ang(Pte)
        n_sample = len(idx)
    prep_s = time.time() - t0
    base.update(n_sample=n_sample, n_train=len(ytr), n_test=len(yte), features="|".join(map(str, cols)),
                label_in_features=int("Attack_Binary" in cols), prep_s=round(prep_s, 3))

    # classical baselines (same PCA features, same split)
    if shots is None:
        for name, model in [("SVM", SVC(kernel="rbf", C=1.0)),
                            ("RandomForest", RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1))]:
            t = time.time(); model.fit(Ptr, ytr); tt = time.time() - t
            rows.append({**base, "fmap": "Classical", "model": name, **_metrics(yte, model.predict(Pte)), "train_s": round(tt, 4)})

    # quantum models
    from qiskit_machine_learning.algorithms import PegasosQSVC
    rng = np.random.default_rng(seed)
    for fm in feature_maps:
        t = time.time()
        Str, Ste = statevectors(Atr, fm), statevectors(Ate, fm)
        Ktr, Kte = fidelity_kernel(Str, Str), fidelity_kernel(Ste, Str)
        if shots:
            Ktr = shot_kernel(Ktr, shots, rng, True); Kte = shot_kernel(Kte, shots, rng, False)
        kt = time.time() - t
        diag = kernel_diagnostics(Ktr, ytr, Ptr / (Ptr.std(axis=0) + 1e-12), Atr, Ate)
        for name in ["QSVM", "PegasosQSVC"]:
            t = time.time()
            if name == "QSVM":
                m = SVC(kernel="precomputed", C=1.0).fit(Ktr, ytr)
            else:
                m = PegasosQSVC(C=1000, num_steps=pegasos_steps, precomputed=True, seed=seed).fit(Ktr, ytr)
            tt = time.time() - t
            pred = np.asarray(m.predict(Kte)).astype(int)
            rows.append({**base, "fmap": fm, "model": name, **_metrics(yte, pred), "train_s": round(tt, 4),
                         "kernel_s": round(kt, 4), **diag})
    return rows
