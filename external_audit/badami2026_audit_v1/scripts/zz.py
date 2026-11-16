"""Fast exact statevectors for Qiskit's ZZFeatureMap (linear entanglement, default data map).
Each repetition: H on all qubits, then the diagonal phase exp(i*[sum_i 2 x_i z_i + sum_(i,i+1) 2 (pi-x_i)(pi-x_j) (z_i xor z_j)])."""
import numpy as np

def _bits(n):
    z = np.arange(2 ** n)
    return ((z[:, None] >> np.arange(n)[None, :]) & 1).astype(float)   # little-endian, as in Qiskit

def _hadamard_all(psi, n):
    psi = psi.reshape([2] * n)
    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    for ax in range(n):
        psi = np.moveaxis(np.tensordot(h, psi, axes=([1], [ax])), 0, ax)
    return psi.reshape(-1)

def zz_states(X, reps=2):
    X = np.asarray(X, float); N, n = X.shape
    B = _bits(n)                                   # (2^n, n)
    par = np.abs(B[:, :-1] - B[:, 1:])            # parity of linear pairs (2^n, n-1)
    out = np.empty((N, 2 ** n), complex)
    plus = np.full(2 ** n, 2 ** (-n / 2), complex)  # H^n |0>
    for k, x in enumerate(X):
        ph = 2 * (B @ x) + 2 * (par @ ((np.pi - x[:-1]) * (np.pi - x[1:])))
        d = np.exp(1j * ph)
        psi = d * plus
        for _ in range(reps - 1):
            psi = d * _hadamard_all(psi, n)
        out[k] = psi
    return out

def kernel(A, B):
    return np.abs(A.conj() @ B.T) ** 2

if __name__ == "__main__":
    from qiskit.circuit.library import zz_feature_map
    from qiskit.quantum_info import Statevector
    rng = np.random.default_rng(0); n = 8
    X = rng.uniform(-0.2, 1.2, size=(5, n))
    S = zz_states(X)
    fm = zz_feature_map(n, reps=2, entanglement="linear")
    for x, s in zip(X, S):
        q = Statevector(fm.assign_parameters(x)).data
        print(round(abs(np.vdot(q, s)) ** 2, 10))
