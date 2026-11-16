"""External audit re-run of Kerkatou et al. (IEEE Access 2026) binary QSVM pipeline.

Kernels use exact statevectors from PennyLane (default.qubit, qml.state):
k(x,x') = |<psi(x)|psi(x')>|^2, identical to probs[0] of U(x) U(x')^dagger used in the repo.

Conditions (each over 10 seeds):
  P   published protocol: 'level' kept, scale->SMOTE->scale on all data, subsample,
      PCA fit on subsample, 80/20 split, standardised (unbounded) inputs, C picked on the test set
  P-L published protocol with 'level' dropped
  F   leak-free: drop 'level', split first, scaler/PCA/SMOTE fitted on train only,
      inputs MinMax-scaled to [0, pi] on train, C picked by 5-fold CV on train
  F+L leak-free but 'level' kept (isolates the leakage-only effect)
Classical RBF-SVM (tuned by CV) and RF are run on the same splits and features.
"""
import sys, json, time
import numpy as np, pandas as pd
import pennylane as qml
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE

DATA, OUT = sys.argv[1], sys.argv[2]
SEEDS = list(range(10))
KS = [2, 4, 6, 8]
EMBS = ["IQP", "Angle"]
C_VALUES = [0.1, 0.5, 1, 10, 100]
N_SAMPLE = 300

cols = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot',
 'num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations',
 'num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count',
 'serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
 'srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate',
 'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
 'dst_host_rerror_rate','dst_host_srv_rerror_rate','attack','level']
df = pd.read_csv(DATA)          # as published: no header=None, first record becomes the header
df.columns = cols
y_all = (df['attack'] != 'normal').astype(int).values
X_raw = pd.get_dummies(df.drop('attack', axis=1), columns=['protocol_type','service','flag']).astype(float)

def states(X, emb):
    n = X.shape[1]
    dev = qml.device("default.qubit", wires=n)
    @qml.qnode(dev)
    def circ(x):
        if emb == "IQP": qml.IQPEmbedding(x, wires=range(n))
        else: qml.AngleEmbedding(x, wires=range(n))
        return qml.state()
    return np.array([circ(x) for x in X])

def kern(A, B): return np.abs(A.conj() @ B.T) ** 2

def cv_pick_C(K, y, seed):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed); best = (-1, None)
    for C in C_VALUES:
        s = []
        for tr, va in skf.split(K, y):
            m = SVC(kernel='precomputed', C=C).fit(K[np.ix_(tr, tr)], y[tr])
            s.append(f1_score(y[va], m.predict(K[np.ix_(va, tr)])))
        if np.mean(s) > best[0]: best = (np.mean(s), C)
    return best[1]

def prep_published(keep_level, seed, k):
    X = X_raw if keep_level else X_raw.drop(columns=['level'])
    Xs = StandardScaler().fit_transform(X)
    Xr, yr = SMOTE(random_state=seed).fit_resample(Xs, y_all)
    Xr = StandardScaler().fit_transform(Xr)
    Xsub, _, ysub, _ = train_test_split(Xr, yr, train_size=N_SAMPLE, stratify=yr, random_state=seed)
    Z = PCA(n_components=k).fit_transform(Xsub)
    return train_test_split(Z, ysub, test_size=0.2, stratify=ysub, random_state=seed)

def prep_fixed(keep_level, seed, k):
    X = (X_raw if keep_level else X_raw.drop(columns=['level'])).values
    Xsub, _, ysub, _ = train_test_split(X, y_all, train_size=N_SAMPLE, stratify=y_all, random_state=seed)
    Xtr, Xte, ytr, yte = train_test_split(Xsub, ysub, test_size=0.2, stratify=ysub, random_state=seed)
    sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    Xtr, ytr = SMOTE(random_state=seed).fit_resample(Xtr, ytr)
    p = PCA(n_components=k).fit(Xtr); Ztr, Zte = p.transform(Xtr), p.transform(Xte)
    mm = MinMaxScaler((0, np.pi)).fit(Ztr)
    return mm.transform(Ztr), np.clip(mm.transform(Zte), 0, np.pi), ytr, yte

rows = []
t0 = time.time()
for cond, fn, lvl in [("P", prep_published, True), ("P-L", prep_published, False),
                      ("F", prep_fixed, False), ("F+L", prep_fixed, True)]:
    for seed in SEEDS:
        for k in KS:
            Ztr, Zte, ytr, yte = fn(lvl, seed, k)
            for emb in EMBS:
                Str, Ste = states(Ztr, emb), states(Zte, emb)
                Ktr, Kte = kern(Str, Str), kern(Ste, Str)
                accs = {}
                for C in C_VALUES:
                    pr = SVC(kernel='precomputed', C=C).fit(Ktr, ytr).predict(Kte)
                    accs[C] = (accuracy_score(yte, pr), f1_score(yte, pr))
                Ccv = cv_pick_C(Ktr, ytr, seed)
                best_test = max(accs.values())
                rows.append(dict(cond=cond, seed=seed, k=k, model=f"QSVM-{emb}", n_test=len(yte),
                                 acc_testpick=best_test[0], f1_testpick=best_test[1],
                                 C_cv=Ccv, acc_cv=accs[Ccv][0], f1_cv=accs[Ccv][1]))
            # classical baselines on identical splits/features
            from sklearn.model_selection import GridSearchCV
            g = GridSearchCV(SVC(kernel='rbf'), {"C": C_VALUES, "gamma": ["scale", 0.1, 1, 10]},
                             cv=StratifiedKFold(5, shuffle=True, random_state=seed), scoring='f1').fit(Ztr, ytr)
            pr = g.predict(Zte)
            rows.append(dict(cond=cond, seed=seed, k=k, model="RBF-SVM", n_test=len(yte),
                             acc_cv=accuracy_score(yte, pr), f1_cv=f1_score(yte, pr)))
            pr = RandomForestClassifier(100, random_state=seed).fit(Ztr, ytr).predict(Zte)
            rows.append(dict(cond=cond, seed=seed, k=k, model="RF", n_test=len(yte),
                             acc_cv=accuracy_score(yte, pr), f1_cv=f1_score(yte, pr)))
        print(cond, seed, f"{time.time()-t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)
print("done")
