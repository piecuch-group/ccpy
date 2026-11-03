"""Strict EE-ADC(2), real RHF/ROHF/UHF: DF matrix actions and ISR virtual densities.

Spin-orbital condensation of the ADC(2) equations in PySCF's uadc_ee.py.
Reference implementation copyright 2014-2022 The PySCF Developers; authors
Terrence Stahl, Ning-Yuan Chen, Alexander Sokolov. Adapted under Apache-2.0;
see THIRD_PARTY_NOTICES.md. No PySCF imports are used here.

The independent doubles basis has unit determinant metric. Ground MP2 t2 is
stored, but neither the ADC secular matrix nor ovvv/vvvv ERIs are constructed.
Contractions use three-index factors; spin-orbital workspace scales O(o^2 v^2).
"""
import numpy as np

from ccpy.adc.reference import ADCReference
from ccpy.models.operators import ClusterOperator


def einsum(expression, *arrays):
    return np.einsum(expression, *arrays, optimize=True)


class ADCVector:
    """Nonredundant EE amplitudes, convertible to CCpy's spin-block operators."""

    def __init__(self, space):
        self.space = space
        self.ndim = space.ndim
        self.data = np.zeros(self.ndim)

    def flatten(self):
        return self.data.copy()

    def unflatten(self, vector):
        vector = np.asarray(vector)
        if vector.shape != (self.ndim,):
            raise ValueError("ADC vector has the wrong working-space dimension.")
        self.data = vector.copy()

    def as_cluster(self):
        X, Y = self.space.unpack(self.data)
        na, nb, va, vb = self.space.sizes
        result = ClusterOperator(self.space.system, 2)
        result.a, result.b = X[:na, :va].T, X[na:, va:].T
        result.aa = Y[:na, :na, :va, :va].transpose(2, 3, 0, 1)
        result.ab = Y[:na, na:, :va, va:].transpose(2, 3, 0, 1)
        result.bb = Y[na:, na:, va:, va:].transpose(2, 3, 0, 1)
        for s in ("a", "b", "aa", "ab", "bb"):
            setattr(result, s, self.space.basis.rotate(getattr(result, s), s * 2, "v" * len(s) + "o" * len(s), inverse=True))
        return result

    def from_cluster(self, vector):
        na, nb, va, vb = self.space.sizes
        X = np.zeros((na + nb, va + vb))
        X[:na, :va] = self.space.basis.rotate(vector.a, "aa", "vo").T
        X[na:, va:] = self.space.basis.rotate(vector.b, "bb", "vo").T
        Y = np.zeros((na + nb, na + nb, va + vb, va + vb))
        Y[:na, :na, :va, :va] = self.space.basis.rotate(vector.aa, "aaaa", "vvoo").transpose(2, 3, 0, 1)
        Y[:na, na:, :va, va:] = self.space.basis.rotate(vector.ab, "abab", "vvoo").transpose(2, 3, 0, 1)
        Y[na:, na:, va:, va:] = self.space.basis.rotate(vector.bb, "bbbb", "vvoo").transpose(2, 3, 0, 1)
        self.data = self.space.pack(X, Y)
        return self

    @property
    def a(self):
        return self.as_cluster().a

    @property
    def b(self):
        return self.as_cluster().b

    @property
    def aa(self):
        return self.as_cluster().aa

    @property
    def ab(self):
        return self.as_cluster().ab

    @property
    def bb(self):
        return self.as_cluster().bb


class ADC2Space:
    def __init__(self, reference):
        self.reference, self.system = reference, reference.system
        sys = reference.system
        na, nb, va, vb = (sys.noccupied_alpha, sys.noccupied_beta,
                          sys.nunoccupied_alpha, sys.nunoccupied_beta)
        self.sizes = na, nb, va, vb
        self.o, self.v = na + nb, va + vb
        self.pairs = {}
        for s, no, nv in (("aa", na, va), ("bb", nb, vb)):
            i, j = np.tril_indices(no, -1)
            a, b = np.tril_indices(nv, -1)
            self.pairs[s] = (i[:, None], j[:, None], a[None, :], b[None, :])
        sizes = (na * va, nb * vb, len(self.pairs["aa"][0]) * self.pairs["aa"][2].shape[1],
                 na * nb * va * vb, len(self.pairs["bb"][0]) * self.pairs["bb"][2].shape[1])
        bounds = np.cumsum((0,) + sizes)
        self.slices = [slice(bounds[i], bounds[i + 1]) for i in range(5)]
        self.ndim = int(bounds[-1])

    def pack(self, X, Y):
        na, nb, va, vb = self.sizes
        ia, ja, aa, ba = self.pairs["aa"]
        ib, jb, ab, bb = self.pairs["bb"]
        return np.concatenate((X[:na, :va].ravel(), X[na:, va:].ravel(),
                               Y[ia, ja, aa, ba].ravel(), Y[:na, na:, :va, va:].ravel(),
                               Y[na + ib, na + jb, va + ab, va + bb].ravel()))

    def unpack(self, vector):
        na, nb, va, vb = self.sizes
        X = np.zeros((self.o, self.v))
        X[:na, :va] = vector[self.slices[0]].reshape(na, va)
        X[na:, va:] = vector[self.slices[1]].reshape(nb, vb)
        Y = np.zeros((self.o, self.o, self.v, self.v))
        for key, section, offset_o, offset_v in (("aa", 2, 0, 0), ("bb", 4, na, va)):
            i, j, a, b = (x + (offset_o if k < 2 else offset_v)
                          for k, x in enumerate(self.pairs[key]))
            amplitude = vector[self.slices[section]].reshape(i.shape[0], a.shape[1])
            Y[i, j, a, b], Y[j, i, a, b] = amplitude, -amplitude
            Y[i, j, b, a], Y[j, i, b, a] = -amplitude, amplitude
        mixed = vector[self.slices[3]].reshape(na, nb, va, vb)
        Y[:na, na:, :va, va:] = mixed
        Y[na:, :na, :va, va:] = -mixed.transpose(1, 0, 2, 3)
        Y[:na, na:, va:, :va] = -mixed.transpose(0, 1, 3, 2)
        Y[na:, :na, va:, :va] = mixed.transpose(1, 0, 3, 2)
        return X, Y


class EEADC2(ADCReference):
    def __init__(self, reference):
        super().__init__(reference)
        self.space = ADC2Space(reference)
        self.space.basis = self
        # Exact singles diagonal without forming its matrix.
        bare = self.delta1 + einsum("Pia,Pia->ia", self.ov, self.ov) - einsum("Pii,Paa->ia", self.oo, self.vv)
        diagonal1 = bare + np.diag(self.occupied)[:, None] + np.diag(self.virtual)[None, :] + self.cross_diagonal
        diagonal1 -= np.diag(self.single_occupied)[:, None]
        diagonal1 += np.diag(self.single_virtual)[None, :]
        diagonal1 += (2 * einsum("Pia,Pia->ia", self.ov, self.dov)
                      - einsum("Pii,Paa->ia", self.doo, self.vv)
                      - einsum("Pii,Paa->ia", self.oo, self.dvv))
        self.diagonal = self.space.pack(diagonal1, self.delta2)

    def g_action(self, X):
        charge = einsum("Pme,me->P", self.ov, X)
        exchange = einsum("Pie,me->Pim", self.ov, X)
        return einsum("Pia,P->ia", self.ov, charge) - einsum("Pim,Pma->ia", exchange, self.ov)

    def singles(self, X):
        def T(x):
            return einsum("imae,me->ia", self.t, x)
        charge = einsum("Pjb,jb->P", self.ov, X)
        exchange = einsum("Pij,jb->Pib", self.oo, X)
        return (self.delta1 * X + einsum("Pia,P->ia", self.ov, charge)
                - einsum("Pib,Pab->ia", exchange, self.vv)
                + self.occupied @ X + X @ self.virtual.T
                + 0.5 * (self.g_action(T(X)) + T(self.g_action(X)))
                - self.single_occupied @ X + X @ self.single_virtual.T
                + self.single_cis(X))

    def single_cis(self, X):
        charge = einsum("Pjb,jb->P", self.ov, X)
        delta_charge = einsum("Pjb,jb->P", self.dov, X)
        return (einsum("Pia,P->ia", self.dov, charge) + einsum("Pia,P->ia", self.ov, delta_charge)
                - einsum("Pib,Pab->ia", einsum("Pij,jb->Pib", self.doo, X), self.vv)
                - einsum("Pib,Pab->ia", einsum("Pij,jb->Pib", self.oo, X), self.dvv))

    def coupling_up(self, Y):
        # 1/2 <am||ef> Y_imef - 1/2 <mn||ie> Y_mnae.
        particle = einsum("Pmf,imef->Pie", self.ov, Y)
        hole = einsum("Pne,mnae->Pma", self.ov, Y)
        return einsum("Pae,Pie->ia", self.vv, particle) - einsum("Pmi,Pma->ia", self.oo, hole)

    def coupling_down(self, X):
        # Transpose in the independent determinant metric.
        particle = einsum("Pca,ic->Pia", self.vv, X)
        hole = einsum("Pci,ca->Pia", self.oo, X)
        part = einsum("Pia,Pjb->ijab", particle - hole, self.ov)
        return part - part.swapaxes(0, 1) - part.swapaxes(2, 3) + part.transpose(1, 0, 3, 2)

    def matvec(self, vector):
        X, Y = self.space.unpack(vector)
        return self.space.pack(self.singles(X) + self.coupling_up(Y),
                               self.delta2 * Y + self.coupling_down(X))

    def virtual_density(self, vector):
        """Second-order unrelaxed ISR state virtual density (not a transition RDM)."""
        X, Y = self.space.unpack(vector / np.linalg.norm(vector))
        t = self.t
        D = X.T @ X + 0.5 * einsum("ijac,ijbc->ab", Y, Y)
        D -= 0.25 * einsum("iA,ia,jkab,jkBb->AB", X, X, t, t)
        D += 0.5 * einsum("iA,ja,ikBb,jkab->AB", X, X, t, t)
        D -= 0.25 * einsum("ia,iB,jkab,jkAb->AB", X, X, t, t)
        D += 0.5 * einsum("ia,ia,jkAb,jkBb->AB", X, X, t, t)
        D -= 0.5 * einsum("ia,ib,jkBa,jkAb->AB", X, X, t, t)
        D += 0.5 * einsum("ia,jB,ikab,jkAb->AB", X, X, t, t)
        D -= einsum("ia,ja,ikBb,jkAb->AB", X, X, t, t)
        D += einsum("ia,jb,ikBa,jkAb->AB", X, X, t, t)
        single_metric = self.t1.T @ self.t1
        particle = X.T @ X
        D += np.sum(X * X) * single_metric - self.t1.T @ (X @ X.T) @ self.t1
        D -= 0.5 * (single_metric @ particle + particle @ single_metric)
        cross = einsum("jb,ijab->ia", X, Y)
        D += self.t1.T @ cross + cross.T @ self.t1
        return self.density_to_working(D)


def HR(dR, R, T, H, RHF_symmetry, system):
    return H.matvec(R.flatten())


def update(R, omega, H, RHF_symmetry, system):
    denominator = omega - H.diagonal
    denominator = np.where(np.abs(denominator) < 1.e-10, np.where(denominator < 0, -1.e-10, 1.e-10), denominator)
    R.unflatten(R.flatten() / denominator)
    return R
