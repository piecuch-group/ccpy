"""DF IP/EA-ADC(2), with general-reference first-order singles.

Canonical terms condensed from PySCF uadc_ip.py/uadc_ea.py (Apache-2.0,
PySCF Developers; see THIRD_PARTY_NOTICES.md). General-reference terms use
the effective-Liouvillian BCH expansion of Stahl et al., JCP 157, 044106 (2022).
The alpha removal/attachment sector matches CCpy's IP-/EA-CISd operator API.
"""
import numpy as np

from ccpy.adc.reference import ADCReference, einsum
from ccpy.models.operators import FockOperator


class ChargedVector:
    def __init__(self, space):
        self.space, self.ndim = space, space.ndim
        self.data = np.zeros(self.ndim)

    def flatten(self):
        return self.data.copy()

    def unflatten(self, vector):
        if np.shape(vector) != (self.ndim,):
            raise ValueError("Wrong charged ADC vector dimension.")
        self.data = np.array(vector, copy=True)

    def as_cluster(self):
        X, Y = self.space.unpack(self.data)
        na, nb, va, vb = self.space.sizes
        ip = self.space.sector == "IP"
        result = FockOperator(self.space.system, 1 if ip else 2, 2 if ip else 1)
        result.a = X[:na if ip else va]
        if ip:
            result.aa = -Y[:va, :na, :na].transpose(1, 0, 2)
            result.ab = -Y[va:, :na, na:].transpose(1, 0, 2)
        else:
            result.aa = Y[:na, :va, :va].transpose(1, 2, 0)
            result.ab = Y[na:, :va, va:].transpose(1, 2, 0)
        result.a = self.space.basis.rotate(result.a, "a", "o" if ip else "v", inverse=True)
        for s in ("aa", "ab"):
            setattr(result, s, self.space.basis.rotate(getattr(result, s), "aaa" if s == "aa" else "abb",
                                                      "ovo" if ip else "vvo", inverse=True))
        return result

    def from_cluster(self, vector):
        na, nb, va, vb = self.space.sizes
        ip = self.space.sector == "IP"
        X = np.zeros(na + nb if ip else va + vb)
        X[:na if ip else va] = self.space.basis.rotate(vector.a, "a", "o" if ip else "v")
        Y = np.zeros((va + vb, na + nb, na + nb) if ip else (na + nb, va + vb, va + vb))
        same = self.space.basis.rotate(vector.aa, "aaa", "ovo" if ip else "vvo")
        mixed = self.space.basis.rotate(vector.ab, "abb", "ovo" if ip else "vvo")
        if ip:
            Y[:va, :na, :na] = -same.transpose(1, 0, 2)
            Y[va:, :na, na:] = -mixed.transpose(1, 0, 2)
        else:
            Y[:na, :va, :va] = same.transpose(2, 0, 1)
            Y[na:, :va, va:] = mixed.transpose(2, 0, 1)
        self.data = self.space.pack(X, Y)
        return self


class ChargedSpace:
    def __init__(self, reference, sector, basis):
        self.reference, self.system, self.sector, self.basis = reference, reference.system, sector, basis
        self.sizes = basis.sizes
        na, nb, va, vb = self.sizes
        ip = sector == "IP"
        self.pair = np.tril_indices(na if ip else va, -1)
        sizes = (na if ip else va, (va if ip else na) * len(self.pair[0]),
                 vb * na * nb if ip else nb * va * vb)
        bounds = np.cumsum((0,) + sizes)
        self.slices = [slice(bounds[i], bounds[i + 1]) for i in range(3)]
        self.ndim = int(bounds[-1])

    def pack(self, X, Y):
        na, nb, va, vb = self.sizes
        i, j = self.pair
        ip = self.sector == "IP"
        return np.r_[X[:na if ip else va],
                     Y[:va, i, j].ravel() if ip else Y[:na, i, j].ravel(),
                     Y[va:, :na, na:].ravel() if ip else Y[na:, :va, va:].ravel()]

    def unpack(self, vector):
        na, nb, va, vb = self.sizes
        ip = self.sector == "IP"
        X = np.zeros(na + nb if ip else va + vb)
        X[:na if ip else va] = vector[self.slices[0]]
        Y = np.zeros((va + vb, na + nb, na + nb) if ip else (na + nb, va + vb, va + vb))
        i, j = self.pair
        if ip:
            amplitudes = vector[self.slices[1]].reshape(va, -1)
            Y[:va, i, j], Y[:va, j, i] = amplitudes, -amplitudes
            mixed = vector[self.slices[2]].reshape(vb, na, nb)
            Y[va:, :na, na:] = mixed
            Y[va:, na:, :na] = -mixed.transpose(0, 2, 1)
        else:
            amplitudes = vector[self.slices[1]].reshape(na, -1)
            Y[:na, i, j], Y[:na, j, i] = amplitudes, -amplitudes
            mixed = vector[self.slices[2]].reshape(nb, va, vb)
            Y[na:, :va, va:] = mixed
            Y[na:, va:, :va] = -mixed.transpose(0, 2, 1)
        return X, Y


class ChargedADC2(ADCReference):
    def __init__(self, reference, sector):
        super().__init__(reference)
        self.sector = sector
        self.space = ChargedSpace(reference, sector, self)
        if sector == "IP":
            self.primary = -np.diag(self.eo) + self.occupied - self.single_occupied
            self.satellite = self.ev[:, None, None] - self.eo[None, :, None] - self.eo[None, None, :]
        else:
            self.primary = np.diag(self.ev) + self.virtual + self.single_virtual
            self.satellite = -self.eo[:, None, None] + self.ev[None, :, None] + self.ev[None, None, :]
        self.diagonal = self.space.pack(np.diag(self.primary), self.satellite)

    def coupling_up(self, Y):
        if self.sector == "IP":
            temp = einsum("Pka,ajk->Pj", self.ov, Y)
            return einsum("Pji,Pj->i", self.oo, temp)
        temp = einsum("Pic,ibc->Pb", self.ov, Y)
        return einsum("Pab,Pb->a", self.vv, temp)

    def coupling_down(self, X):
        if self.sector == "IP":
            temp = einsum("Pji,i->Pj", self.oo, X)
            part = einsum("Pj,Pka->ajk", temp, self.ov)
        else:
            temp = einsum("Pab,a->Pb", self.vv, X)
            part = einsum("Pb,Pic->ibc", temp, self.ov)
        return part - part.swapaxes(1, 2)

    def matvec(self, vector):
        X, Y = self.space.unpack(vector)
        return self.space.pack(self.primary @ X + self.coupling_up(Y),
                               self.satellite * Y + self.coupling_down(X))

    def virtual_density(self, vector):
        X, Y = self.space.unpack(vector / np.linalg.norm(vector))
        t, s = self.t, self.t1
        norm = np.dot(X, X)
        ground = 0.5 * einsum("ijac,ijbc->ab", t, t)
        metric = s.T @ s
        if self.sector == "IP":
            D = 0.5 * einsum("aij,bij->ab", Y, Y) + norm * ground
            D -= einsum("g,h,hmbc,gmac->ab", X, X, t, t)
            D += norm * metric - np.outer(X @ s, X @ s)
            cross = -einsum("n,ani->ia", X, Y)
        else:
            particle = np.outer(X, X)
            D = particle + einsum("iac,ibc->ab", Y, Y) + norm * ground
            D -= 0.25 * einsum("A,a,ijab,ijBb->AB", X, X, t, t)
            D -= 0.25 * einsum("a,B,ijab,ijAb->AB", X, X, t, t)
            D -= 0.5 * einsum("a,b,ijBa,ijAb->AB", X, X, t, t)
            D += norm * metric - 0.5 * (metric @ particle + particle @ metric)
            cross = -einsum("a,iAa->iA", X, Y)
        D += s.T @ cross + cross.T @ s
        return self.density_to_working(D)
