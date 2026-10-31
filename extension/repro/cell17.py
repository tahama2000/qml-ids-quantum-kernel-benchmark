# QML IDS Comparison Code for Google Colab (Benchmarking Multiple Datasets)
# Two feature selectors: (1) Random Forest importance (RF), (2) Quantum-aware QKA+MI
# Quantum inputs are scaled per-feature to [0, 2π]; classical path uses StandardScaler (+ optional PCA).

# --- Colab setup (uncomment when in Colab) ---
# !pip install qiskit qiskit-machine-learning pandas scikit-learn matplotlib imbalanced-learn psutil
# from google.colab import drive
# drive.mount('/content/drive')  # ensure your CSV path is accessible

import os, time, psutil, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTE

# Quantum imports
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliFeatureMap, ZZFeatureMap, ZFeatureMap
from qiskit.primitives import Sampler
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.algorithms import QSVC, PegasosQSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel

warnings.filterwarnings("ignore")

# ===========================
# GLOBAL LABELING STRATEGY
# ===========================
# Set to "binary" for Normal(0) vs Attack(1) across all datasets,
# or "multiclass" to keep all attack types/classes.
LABEL_MODE = "binary"   # change to "multiclass" to switch globally

# ----------------------------
# DATASET LOADERS (3 datasets)
# ----------------------------

def load_nsl_kdd():
    """Loads a 20% split of NSL-KDD from GitHub (quick tests)."""
    url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B_20Percent.txt"
    column_names = [
        'duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent',
        'hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root',
        'num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login',
        'count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate',
        'diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate',
        'dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate',
        'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','label','difficulty_level'
    ]
    df = pd.read_csv(url, names=column_names)
    df = df.drop(columns=['difficulty_level'])
    # NSL-KDD labels are strings: 'normal' vs attacks; we will map later per LABEL_MODE
    return df, 'label', False  # preprocessed=False

def load_unsw_nb15(train_path='/content/UNSW_NB15_training-set.csv',
                   test_path='/content/UNSW_NB15_testing-set.csv'):
    """
    Loads UNSW-NB15 from local/Colab paths. Adjust paths if needed.
    """
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"UNSW-NB15 train file not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"UNSW-NB15 test file not found: {test_path}")

    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)
    df = pd.concat([df_train, df_test], ignore_index=True)
    df = df.dropna()
    # Common label names
    label_col = 'label' if 'label' in df.columns else ('Label' if 'Label' in df.columns else df.columns[-1])
    return df, label_col, False  # preprocessed=False

def load_mqtteeb_d2025(file_path="/content/MQTTEEB-D_dataset_All_Processed.csv"):
    """
    Loads the processed MQTTEEB-D2025 dataset.
    Keeps original multiclass 'Target' (0..5) AND adds 'Attack_Binary' (0 normal, 1 attack) if LABEL_MODE=binary.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"MQTTEEB-D CSV not found at: {file_path}")
    df = pd.read_csv(file_path)

    # Try to find the label column robustly
    possible_labels = ['label', 'Label', 'attack', 'Attack', 'target', 'Target', 'class', 'Class']
    label_col = next((c for c in possible_labels if c in df.columns), df.columns[-1])

    # Ensure no NA rows
    df = df.dropna()

    # If labels are text, convert to 0/1 (binary) only when LABEL_MODE == "binary"
    if LABEL_MODE == "binary":
        if df[label_col].dtype == object:
            df['Attack_Binary'] = df[label_col].apply(
                lambda x: 0 if str(x).lower() in ['normal', 'benign', 'ok', 'nominal'] else 1
            )
        else:
            # Here label is numeric codes: use 0 as normal, non-zero as attack
            df['Attack_Binary'] = df[label_col].apply(lambda x: 0 if x == 0 else 1)

    return df, label_col, True  # preprocessed=True

# Dataset selector
def load_dataset(name, mqtteeb_path="/content/drive/MyDrive/MQTTEEB-D_dataset_All_Processed.csv",
                 unsw_train="/content/UNSW_NB15_training-set.csv",
                 unsw_test ="/content/UNSW_NB15_testing-set.csv"):
    if name == 'NSL-KDD':
        return load_nsl_kdd()
    elif name == 'UNSW-NB15':
        return load_unsw_nb15(unsw_train, unsw_test)
    elif name == 'MQTTEEB-D2025':
        return load_mqtteeb_d2025(mqtteeb_path)
    else:
        raise ValueError("Supported datasets: 'NSL-KDD', 'UNSW-NB15', 'MQTTEEB-D2025'")

# ----------------------------
# PREPROCESSING HELPERS
# ----------------------------

def clean_data(df, label_col, label_mode="binary"):
    """
    One-hot encode categoricals; label handling depends on label_mode:
      - 'binary': map normal/benign -> 0, else 1
      - 'multiclass': keep original label values (strings/ints); no binarization
    Returns X (DataFrame), y (Series).
    """
    cat_cols = df.select_dtypes(include=['object']).columns.drop(label_col, errors='ignore')
    df = pd.get_dummies(df, columns=cat_cols)

    if label_mode == "binary":
        # map to 0/1 from common text labels or numeric class (0 normal, else attack)
        if df[label_col].dtype == object:
            df[label_col] = df[label_col].apply(lambda x: 0 if str(x).lower() in ['normal', 'benign', 'ok', 'nominal', '0'] else 1)
        else:
            df[label_col] = df[label_col].apply(lambda x: 0 if x == 0 else 1)
    # else: multiclass -> leave as-is

    df = df.dropna()
    X = df.drop(columns=[label_col])
    y = df[label_col]
    return X, y

def split_xy_preprocessed(df, label_col, label_mode="binary"):
    """
    For already processed datasets (e.g., MQTTEEB-D2025), split X/y.
    Uses 'Attack_Binary' when label_mode='binary' and available; otherwise original label.
    """
    X = df.drop(columns=[label_col])
    if label_mode == "binary" and 'Attack_Binary' in df.columns:
        y = df['Attack_Binary']
    else:
        y = df[label_col]
    return X, y

def scale_data(X):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs

# Quantum-only scaling to [0, 2π]
def scale_for_quantum_0_2pi(X):
    X = np.asarray(X, dtype=float)
    Xs = np.zeros_like(X)
    for j in range(X.shape[1]):
        col = X[:, j]
        cmin, cmax = col.min(), col.max()
        if np.isclose(cmax, cmin):
            Xs[:, j] = 0.0
        else:
            Xs[:, j] = 2*np.pi * (col - cmin) / (cmax - cmin)
    return Xs

def select_features_rf(X, y, num_features):
    """Select top num_features via RandomForest importance (numeric only)."""
    # Keep numeric columns only (fixes timestamp/non-numeric)
    if isinstance(X, pd.DataFrame):
        Xn = X.select_dtypes(include=[np.number])
    else:
        Xn = X
    rf = RandomForestClassifier(random_state=42)
    rf.fit(Xn, y)
    importances = rf.feature_importances_
    idx = np.argsort(importances)[-num_features:][::-1]
    if isinstance(Xn, pd.DataFrame):
        X_sel = Xn.iloc[:, idx]
    else:
        X_sel = Xn[:, idx]
    print(f"[RF] Selected top {num_features} features")
    return X_sel

def make_sampler():
    try:
        from qiskit.primitives import SamplerV2
        return SamplerV2()
    except Exception:
        from qiskit.primitives import Sampler
        return Sampler()

# Quantum kernel alignment score (center-free for broad compatibility)
def _kernel_alignment_score(X_sub, y, feature_map_builder):
    """
    Center-free alignment proxy:
    compute K via FidelityQuantumKernel and correlate with labels (one-vs-rest averaged if multiclass).
    """
    # Scale to quantum angles
    Xq = scale_for_quantum_0_2pi(X_sub)
    fm = feature_map_builder(Xq.shape[1])
    sampler = make_sampler()
    fidelity = ComputeUncompute(sampler=sampler)
    kernel = FidelityQuantumKernel(feature_map=fm, fidelity=fidelity)
    K = kernel.evaluate(Xq, Xq)

    y_arr = np.asarray(y)
    classes = np.unique(y_arr)

    def align(z):
        z = z.astype(float)
        z -= z.mean()
        denom = (np.linalg.norm(K, 'fro') * np.linalg.norm(z))
        return float(z @ (K @ z) / denom) if denom > 0 else 0.0

    if len(classes) == 2:
        # map to {-1, +1}
        z = np.where(y_arr == classes[0], -1.0, 1.0)
        return align(z)
    # multiclass: mean of one-vs-rest
    scores = []
    for c in classes:
        z = np.where(y_arr == c, 1.0, -1.0)
        scores.append(align(z))
    return float(np.mean(scores))

# Quantum-aware selector: QKA + MI (greedy)
def select_features_qka_mi(X, y, k, feature_map_builder=None, subsample=400, random_state=42):
    """
    Greedy selection maximizing 0.5*QKA + 0.5*mean(MI_selected).
    Uses a small subsample for runtime stability.
    """
    if feature_map_builder is None:
        # modest expressivity for scoring
        feature_map_builder = lambda d: PauliFeatureMap(feature_dimension=d, reps=1)

    # numeric only
    if isinstance(X, pd.DataFrame):
        Xn = X.select_dtypes(include=[np.number])
    else:
        Xn = X

    rng = np.random.RandomState(random_state)
    n = Xn.shape[0]
    if n > subsample:
        idx_sub = rng.choice(n, size=subsample, replace=False)
        Xs = (Xn.iloc[idx_sub] if isinstance(Xn, pd.DataFrame) else Xn[idx_sub])
        ys = (y.iloc[idx_sub] if isinstance(y, pd.Series) else y[idx_sub])
    else:
        Xs, ys = Xn, y

    # MI per feature
    mi_all = mutual_info_classif(np.asarray(Xs), np.asarray(ys), discrete_features=False, random_state=random_state)

    remaining = list(range(Xs.shape[1]))
    selected = []

    for _ in range(min(k, Xs.shape[1])):
        best_score, best_idx = -np.inf, None
        for j in remaining:
            cand = selected + [j]
            try:
                qka = _kernel_alignment_score(np.asarray(Xs)[:, cand], np.asarray(ys), feature_map_builder)
            except Exception:
                qka = 0.0  # fallback if kernel evaluation fails
            score = 0.5 * qka + 0.5 * float(np.mean(mi_all[cand]))
            if score > best_score:
                best_score, best_idx = score, j
        selected.append(best_idx)
        remaining.remove(best_idx)

    if isinstance(Xn, pd.DataFrame):
        X_sel = Xn.iloc[:, selected]
    else:
        X_sel = Xn[:, selected]
    print(f"[QKA+MI] Selected top {k} features")
    return X_sel

# Stratified Sampling
def sample_data_with_stratification(X, y, sample_percent):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - sample_percent, random_state=42)
    for train_idx, _ in sss.split(X, y):
        Xs = X.iloc[train_idx] if isinstance(X, pd.DataFrame) else X[train_idx]
        ys = y.iloc[train_idx]
        return Xs, ys, len(train_idx)

def balance_dataset_with_smote(X, y):
    sm = SMOTE(random_state=42)
    Xr, yr = sm.fit_resample(X, y)
    return Xr, yr

def _prepare_labels_for_algo(y, algo):
    """Return labels in the encoding each algo expects. (Not used now; Pegasos works with 0/1 in your env)."""
    y_arr = np.asarray(y).astype(int)
    if algo == 'PegasosQSVC':
        unique_vals = np.unique(y_arr)
        if set(unique_vals).issubset({0, 1}):
            y_arr = np.where(y_arr == 0, -1, 1)
        elif len(unique_vals) > 2:
            raise ValueError("PegasosQSVC supports binary labels only.")
    return y_arr


### Fast Version ###
def select_features_qka_mi_fast(
    X, y, k,
    prefilter_m=20,                # candidate pool size
    subsample=150,                 # much smaller
    time_budget_sec=180,           # bail out if over 3 minutes
    random_state=42,
    feature_map_builder=None       # defaults to fast ZFeatureMap
):
    import time
    from sklearn.feature_selection import mutual_info_classif

    t0 = time.time()

    # numeric only
    if isinstance(X, pd.DataFrame):
        Xn = X.select_dtypes(include=[np.number])
    else:
        Xn = X
    Xn = np.asarray(Xn, dtype=np.float32)
    y_arr = np.asarray(y)

    # subsample for speed
    rng = np.random.RandomState(random_state)
    n = Xn.shape[0]
    if n > subsample:
        idx_sub = rng.choice(n, size=subsample, replace=False)
        Xs = Xn[idx_sub]
        ys = y_arr[idx_sub]
    else:
        Xs, ys = Xn, y_arr

    # Stage A: MI prefilter
    mi = mutual_info_classif(Xs, ys, discrete_features=False, random_state=random_state)
    topM_idx = np.argsort(mi)[-prefilter_m:][::-1]
    Xc = Xs[:, topM_idx]
    mi_c = mi[topM_idx]

    # Stage B: Greedy with cheap kernel + MI
    if feature_map_builder is None:
        feature_map_builder = lambda d: ZFeatureMap(feature_dimension=d, reps=1)

    selected_local = []
    remaining_local = list(range(Xc.shape[1]))

    def _kernel_alignment_score_fast(X_sub, y_sub):
        # scale to [0, 2π]
        Xq = scale_for_quantum_0_2pi(X_sub)
        fm = feature_map_builder(Xq.shape[1])
        sampler = make_sampler()
        fidelity = ComputeUncompute(sampler=sampler)
        kernel = FidelityQuantumKernel(feature_map=fm, fidelity=fidelity)
        K = kernel.evaluate(Xq, Xq)
        classes = np.unique(y_sub)

        def align(z):
            z = z.astype(np.float32)
            z -= z.mean()
            denom = (np.linalg.norm(K, 'fro') * np.linalg.norm(z))
            return float(z @ (K @ z) / denom) if denom > 0 else 0.0

        if len(classes) == 2:
            z = np.where(y_sub == classes[0], -1.0, 1.0)
            return align(z)
        scores = []
        for c in classes:
            z = np.where(y_sub == c, 1.0, -1.0)
            scores.append(align(z))
        return float(np.mean(scores))

    while len(selected_local) < min(k, Xc.shape[1]):
        # time budget check
        if (time.time() - t0) > time_budget_sec:
            print(f"[QKA_MI_FAST] Time budget {time_budget_sec}s reached; stopping at {len(selected_local)} features.")
            break

        best_score, best_j = -np.inf, None
        for j in remaining_local:
            cand = selected_local + [j]
            try:
                qka = _kernel_alignment_score_fast(Xc[:, cand], ys)
            except Exception:
                qka = 0.0
            score = 0.5 * qka + 0.5 * float(np.mean(mi_c[cand]))
            if score > best_score:
                best_score, best_j = score, j
        selected_local.append(best_j)
        remaining_local.remove(best_j)

    # map back to original column order
    final_idx = topM_idx[np.array(selected_local, dtype=int)]
    if isinstance(X, pd.DataFrame):
        X_sel = X.select_dtypes(include=[np.number]).iloc[:, final_idx]
    else:
        X_sel = Xn[:, final_idx]
    print(f"[QKA_MI_FAST] Selected top {X_sel.shape[1]} features (prefilter M={prefilter_m}, subsample={len(ys)})")
    return X_sel

# ----------------------------
# PCA (fixed to 3 components for qubit limitation)
# ----------------------------
from sklearn.decomposition import PCA
PCA_COMPONENTS = 2 # match number of available physical qubits

def apply_pca(X):
    """
    Applies PCA dimensionality reduction to 3 components (for 3-qubit quantum circuits).
    Returns the transformed dataset and the fitted PCA model.
    """
    pca = PCA(n_components=PCA_COMPONENTS)
    Xp = pca.fit_transform(X)
    explained_var = sum(pca.explained_variance_ratio_) * 100
    print(f"PCA applied: reduced to {PCA_COMPONENTS} components explaining {explained_var:.2f}% variance")
    return Xp, pca

# ----------------------------
# QUANTUM ENCODINGS
# ----------------------------
def HybridFeatureMap_Pauli_ZZ(num_features: int):
    """
    Hybrid feature map = PauliFeatureMap ∘ ZZFeatureMap with a SHARED ParameterVector.
    This avoids 'Name conflict on adding parameter: x[0]' when composing.
    reps=1 here; you can raise to 2 if you want more expressivity (higher cost).
    """
    x = ParameterVector("x", num_features)
    pauli = PauliFeatureMap(feature_dimension=num_features, reps=1)
    zz    = ZZFeatureMap(feature_dimension=num_features, reps=1)
    pauli_param_list = sorted(list(pauli.parameters), key=lambda p: p.name)
    zz_param_list    = sorted(list(zz.parameters),    key=lambda p: p.name)
    pauli = pauli.assign_parameters({p: x[i] for i, p in enumerate(pauli_param_list)}, inplace=False)
    zz    = zz.assign_parameters({p: x[i] for i, p in enumerate(zz_param_list)},    inplace=False)
    hybrid = pauli.compose(zz, inplace=False)
    return hybrid

# ----------------------------
# METRICS HELPER (handles binary vs multiclass)
# ----------------------------
def metric_average(y_true):
    """Return 'binary' if exactly 2 classes, else 'weighted'."""
    classes = np.unique(y_true)
    return 'binary' if len(classes) == 2 else 'weighted'

# ----------------------------
# EXPERIMENT RUNNERS (CLASSICAL / QUANTUM)
# ----------------------------
def run_experiment_with_classical(X, y, algo_type):
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier

    if isinstance(y, pd.Series):
        y = y.reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if algo_type == "SVM":
        model = SVC(kernel='rbf', C=1.0)
    elif algo_type == "RandomForest":
        model = RandomForestClassifier(n_estimators=200, random_state=42)
    else:
        raise ValueError("Unsupported classical algorithm type")

    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    preds = model.predict(X_test)
    avg = metric_average(y_test)
    acc = accuracy_score(y_test, preds)
    rec = recall_score(y_test, preds, average=avg)
    f1  = f1_score(y_test, preds, average=avg)
    mem = psutil.Process().memory_info().rss / 1024**2

    cm = confusion_matrix(y_test, preds)
    ConfusionMatrixDisplay(cm).plot()
    plt.title(f"Confusion Matrix - {algo_type} ({avg})")
    plt.show()

    print(f"Accuracy: {acc:.4f}  Recall({avg}): {rec:.4f}  F1({avg}): {f1:.4f}  Train Time: {train_time:.2f}s  Mem: {mem:.2f}MB")
    print(classification_report(y_test, preds))
    return acc, rec, f1, train_time, mem

def run_experiment(X, y, feature_map_type, algo_type):
    sampler = make_sampler()

    if isinstance(y, pd.Series):
        y = y.reset_index(drop=True)

    # IMPORTANT: X should already be scaled to [0, 2π] before calling this function
    feature_map = feature_map_type(X.shape[1]) if callable(feature_map_type) else feature_map_type(feature_dimension=X.shape[1], reps=1)
    fidelity = ComputeUncompute(sampler=sampler)
    kernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)

    model = QSVC(quantum_kernel=kernel) if algo_type == 'QSVM' else PegasosQSVC(quantum_kernel=kernel, C=1000, num_steps=100)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # (In your env, Pegasos works with 0/1 without remapping; keep as-is)
    if isinstance(y_train, pd.Series):
        y_train = y_train.reset_index(drop=True)
    if isinstance(y_test, pd.Series):
        y_test = y_test.reset_index(drop=True)

    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    preds = model.predict(X_test)
    avg = metric_average(y_test)
    acc = accuracy_score(y_test, preds)
    rec = recall_score(y_test, preds, average=avg)
    f1  = f1_score(y_test, preds, average=avg)
    mem = psutil.Process().memory_info().rss / 1024**2

    cm = confusion_matrix(y_test, preds)
    ConfusionMatrixDisplay(cm).plot()
    plt.title(f"Confusion Matrix - {algo_type} with {feature_map_type.__name__} ({avg})")
    plt.show()

    print(f"Accuracy: {acc:.4f}  Recall({avg}): {rec:.4f}  F1({avg}): {f1:.4f}  Train Time: {train_time:.2f}s  Mem: {mem:.2f}MB")
    print(classification_report(y_test, preds))
    return acc, rec, f1, train_time, mem

# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":

    # === Configure dataset list (only the three you mentioned) ===
    DATASETS = [
        ('NSL-KDD',        {})#,  # uses GitHub loader
        # ('UNSW-NB15',      {'unsw_train': '/content/UNSW_NB15_training-set.csv',
        #                     'unsw_test':  '/content/UNSW_NB15_testing-set.csv'}),
        #('MQTTEEB-D2025',  {'mqtteeb_path': '/content/MQTTEEB-D_dataset_All_Processed.csv'})
    ]

    SAMPLE_PERCENTS = [0.01, 0.05, 0.1]        # adjust if you want more subsampling
    FEATURE_COUNTS  = [3, 5]                   # top-k features via selectors
    APPLY_SMOTE     = {'NSL-KDD': True, 'UNSW-NB15': True, 'MQTTEEB-D2025': True}
    APPLY_PCA       = True

    encodings = {
        "PauliFeatureMap": PauliFeatureMap,
        "ZZFeatureMap": ZZFeatureMap,
        "ZFeatureMap":  ZFeatureMap,
        "Hybrid_Pauli_ZZ": HybridFeatureMap_Pauli_ZZ
    }
    algorithms = [ 'SVM', 'RandomForest', 'PegasosQSVC', 'QSVM' ]
    selectors  = [ 'QKA_MI_FAST']  #  'RF', QKA_MI<-- two feature selectors

    results = []

    for ds_name, ds_kwargs in DATASETS:
        print(f"\n==== Loading dataset: {ds_name} ====")
        df, label_col, is_preprocessed = load_dataset(ds_name, **ds_kwargs)

        # --- Build X, y respecting LABEL_MODE ---
        if is_preprocessed:
            X_raw, y = split_xy_preprocessed(df, label_col, LABEL_MODE)
        else:
            X_raw, y = clean_data(df, label_col, LABEL_MODE)

        # Ensure numeric-only features before selection (handles timestamps, IDs, etc.)
        if isinstance(X_raw, pd.DataFrame):
            X_raw = X_raw.select_dtypes(include=[np.number])

        for sp in SAMPLE_PERCENTS:
            for fc in FEATURE_COUNTS:
                for selector in selectors:

                      # ---- Feature selection (two tracks) ----
                    if selector == 'RF':
                        X_sel = select_features_rf(
                            X_raw if isinstance(X_raw, pd.DataFrame) else pd.DataFrame(X_raw),
                            y, fc
                        )

                    elif selector == 'QKA_MI_FAST':
                        X_sel = select_features_qka_mi_fast(
                            X=X_raw if isinstance(X_raw, pd.DataFrame) else pd.DataFrame(X_raw),
                            y=y,
                            k=fc,
                            prefilter_m=20,        # you can lower to 12–16 if still slow
                            subsample=150,         # try 100 for more speed
                            time_budget_sec=180,   # bump to 300 if you can afford it
                            random_state=42
                            # feature_map_builder left as default (fast ZFeatureMap)
                        )

                    else:
                        continue  # unknown selector, skip

                    # --- Stratified Sampling Enhancement ---
                    # If we are in binary mode but dataset originally had multiclass attack labels,
                    # sample using the original label column to preserve rare attack types.
                    if LABEL_MODE == "binary" and 'Target' in df.columns:
                        print("Using original multiclass labels for stratified sampling (to preserve rare attacks).")
                        y_for_sampling = df['Target']  # or 'label_col' for UNSW/NSL-KDD variants
                    else:
                        y_for_sampling = y

                    X_samp, y_samp, n_train = sample_data_with_stratification(X_sel, y_for_sampling, sp)

                    # Now use the final binary y for training
                    if LABEL_MODE == "binary":
                        y_samp = y.iloc[y_samp.index] if isinstance(y, pd.Series) else np.array(y)[y_samp.index]

                    # Class imbalance
                    if APPLY_SMOTE.get(ds_name, False):
                        X_bal, y_bal = balance_dataset_with_smote(X_samp, y_samp)
                    else:
                        X_bal, y_bal = X_samp, y_samp

                    # Standard Scale (for classical path)
                    X_scaled = scale_data(X_bal)

                    # PCA (optional)
                    X_final, _ = apply_pca(X_scaled) if APPLY_PCA else (X_scaled, None)

                    # Prepare quantum-scaled copy
                    X_quantum = scale_for_quantum_0_2pi(X_final)

                    for enc_name, enc_class in encodings.items():
                        for algo in algorithms:
                            print(f"\n>> {ds_name} | sample={sp} | k={fc} | selector={selector} | enc={enc_name if 'Q' in algo else 'Classical'} | algo={algo} | mode={LABEL_MODE}")
                            try:
                                if algo in ['QSVM', 'PegasosQSVC']:
                                    # Quantum models: use quantum-scaled features
                                    acc, rec, f1, t_time, mem = run_experiment(X_quantum, y_bal, enc_class, algo)
                                    enc_used = enc_name
                                else:
                                    # Classical models: use standardized PCA features
                                    acc, rec, f1, t_time, mem = run_experiment_with_classical(X_final, y_bal, algo)
                                    enc_used = 'Classical'

                                results.append({
                                    'dataset': ds_name,
                                    'sample': sp,
                                    'sample_size': n_train,
                                    'features': fc,
                                    'feat_selector': selector,
                                    'encoding': enc_used,
                                    'algorithm': algo,
                                    'label_mode': LABEL_MODE,
                                    'accuracy': acc,
                                    'recall': rec,
                                    'f1_score': f1,
                                    'train_time': t_time,
                                    'memory_MB': mem
                                })
                                print(pd.DataFrame(results))
                            except Exception as e:
                                print(f"Error on {ds_name} / {algo} / {selector}: {e}")

    # Save & visualize
    results_df = pd.DataFrame(results)
    results_df.to_csv("qml_ids_results_dual_selectors.csv", index=False)
    print("\n=== Final results head ===")
    print(results_df)

    # Simple accuracy plot
    plt.figure(figsize=(12, 6))
    for algo in results_df['algorithm'].unique():
        subset = results_df[results_df['algorithm'] == algo]
        x = subset['dataset'] + "|" + subset['feat_selector'] + "|" + subset['encoding'] + "|" + subset['features'].astype(str) + "f"
        plt.plot(x, subset['accuracy'], marker='o', label=algo)
    plt.xlabel("Dataset | Selector | Encoding | k-features")
    plt.ylabel("Accuracy")
    plt.title(f"Model Accuracy Across Datasets (mode={LABEL_MODE})")
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()