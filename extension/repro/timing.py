"""
Kernel-construction timing with the real Qiskit primitives (simulator), to ground or drop runtime claims.
Runs under either environment:  python timing.py <out.csv>
  qiskit 1.x : reference Sampler (exact) and Sampler(shots=1024) via ComputeUncompute
  qiskit 2.x : StatevectorSampler(default_shots=1024) via ComputeUncompute; FidelityStatevectorKernel (exact)
Both: direct statevector inner products (the path used in o3ext).
"""
import sys, time, platform, os
import numpy as np, pandas as pd
import qiskit, qiskit_machine_learning as qml
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.kernels import FidelityQuantumKernel
try:
    from qiskit_machine_learning.state_fidelities import ComputeUncompute
except ImportError:
    from qiskit_algorithms.state_fidelities import ComputeUncompute

out = sys.argv[1]; DIMS = [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [3, 5]; V = qiskit.__version__; rng = np.random.default_rng(0); rows = []
def backends():
    if V.startswith("1."):
        from qiskit.primitives import Sampler
        yield "Sampler (exact)", lambda fm: FidelityQuantumKernel(feature_map=fm, fidelity=ComputeUncompute(sampler=Sampler()))
        yield "Sampler (1024 shots)", lambda fm: FidelityQuantumKernel(feature_map=fm, fidelity=ComputeUncompute(sampler=Sampler(options={"shots": 1024, "seed": 0})))
    else:
        from qiskit.primitives import StatevectorSampler
        from qiskit_machine_learning.kernels import FidelityStatevectorKernel
        yield "StatevectorSampler (1024 shots)", lambda fm: FidelityQuantumKernel(feature_map=fm, fidelity=ComputeUncompute(sampler=StatevectorSampler(default_shots=1024, seed=0)))
        yield "FidelityStatevectorKernel (exact)", lambda fm: FidelityStatevectorKernel(feature_map=fm)

def direct(fm, X):
    ps = list(fm.parameters)
    S = np.array([Statevector(fm.assign_parameters(dict(zip(ps, r)))).data for r in X])
    return np.abs(S.conj() @ S.T) ** 2

for d in DIMS:
    for reps in [1, 2]:
        fm = ZZFeatureMap(feature_dimension=d, reps=reps)
        for n in [50, 100, 200]:
            X = rng.uniform(0, np.pi / 2, (n, d))
            cand = list(backends()) + [("direct statevector", None)]
            Kd = None
            for name, mk in cand:
                if n == 200 and "shots" in name:   # too slow, extrapolate from n<=200
                    continue
                t = time.time()
                K = direct(fm, X) if mk is None else mk(fm).evaluate(X)
                dt = time.time() - t
                if mk is None: Kd = K
                rows.append(dict(qiskit=V, qiskit_ml=qml.__version__, python=platform.python_version(), backend=name,
                                 qubits=d, reps=reps, n=n, entries=n * (n - 1) // 2, seconds=round(dt, 3),
                                 us_per_entry=round(1e6 * dt / (n * (n - 1) / 2), 1)))
                print(rows[-1], flush=True)
            # agreement with exact
            for r in rows[-len(cand):]:
                pass
pd.DataFrame(rows).to_csv(out, mode='a', header=not os.path.exists(out), index=False)
