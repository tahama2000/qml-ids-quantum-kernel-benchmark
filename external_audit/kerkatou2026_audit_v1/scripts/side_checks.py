"""Side checks: lost first row, duplicates, 'level' as a label proxy, mixed-embedding kernel validity."""
import sys, json
import numpy as np, pandas as pd
import pennylane as qml
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

DATA, OUT = sys.argv[1], sys.argv[2]
out = {}
raw = pd.read_csv(DATA, header=None)
pub = pd.read_csv(DATA)
out["rows_file"] = len(raw); out["rows_as_loaded"] = len(pub)
out["exact_duplicate_rows"] = int(raw.duplicated().sum())
out["duplicates_ignoring_level"] = int(raw.drop(columns=[42]).duplicated().sum())
y = (raw[41] != 'normal').astype(int).values
lvl = raw[[42]].values
cv = StratifiedKFold(5, shuffle=True, random_state=0)
out["level_only_RF_acc_5cv"] = float(cross_val_score(RandomForestClassifier(100, random_state=0), lvl, y, cv=cv).mean())
out["level_only_tree_depth3_acc_5cv"] = float(cross_val_score(DecisionTreeClassifier(max_depth=3, random_state=0), lvl, y, cv=cv).mean())
out["majority_class_rate"] = float(max(y.mean(), 1 - y.mean()))
out["mean_level_normal"] = float(raw.loc[y == 0, 42].mean()); out["mean_level_attack"] = float(raw.loc[y == 1, 42].mean())

# Mixed kernel used by the repo's first binary "Angle" QSVM cell: IQP(x) then adjoint(Angle)(x')
rng = np.random.default_rng(0); n = 4
X = rng.normal(size=(30, n))
dev = qml.device("default.qubit", wires=n)
@qml.qnode(dev)
def mixed(a, b):
    qml.IQPEmbedding(a, wires=range(n)); qml.adjoint(qml.AngleEmbedding)(b, wires=range(n))
    return qml.probs(wires=range(n))
K = np.array([[mixed(a, b)[0] for b in X] for a in X])
out["mixed_kernel_diag_mean"] = float(np.diag(K).mean())   # a valid fidelity kernel has k(x,x)=1
out["mixed_kernel_max_asymmetry"] = float(np.abs(K - K.T).max())
out["mixed_kernel_min_eig_sym"] = float(np.linalg.eigvalsh((K + K.T) / 2).min())
json.dump(out, open(OUT, "w"), indent=2); print(json.dumps(out, indent=2))
