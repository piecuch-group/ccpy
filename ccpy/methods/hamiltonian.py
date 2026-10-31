"""Density-fitted (DF) or Cholesky-factorized Hamiltonian in the correlated MO basis.

The only stored two-electron quantities are the three-index vectors

    <pq|rs> = (pr|qs) = sum_x B_x[p, r] B_x[q, s],

kept separately for alpha (``H.chol.a``) and beta (``H.chol.b``) orbitals, so that RHF,
ROHF, and UHF references are handled alike. Two-body blocks with at most three virtual
indices (``H.aa.oovv``, ``H.ab.voov``, ...) are formed from the vectors when first accessed
and cached; four-virtual (vvvv) blocks are never formed.
"""
import numpy as np

from ccpy.models.integrals import Integral

# two-body blocks built eagerly for copies of the Hamiltonian (the blocks of the former
# ccpy.models.integrals.getCholeskyHamiltonian)
STANDARD_BLOCKS = {
    "aa": ("oooo", "ooov", "vooo", "voov", "oovv", "vvoo"),
    "bb": ("oooo", "ooov", "vooo", "voov", "oovv", "vvoo"),
    "ab": ("oooo", "ooov", "oovo", "vooo", "ovoo", "voov", "ovvo", "vovo", "ovov", "oovv", "vvoo"),
}


class VVVVError(AttributeError):
    pass


class DFTwoBody:
    """Two-body blocks of one spin case evaluated from the DF vectors on first access.

    For same-spin cases the blocks are antisymmetrized, <pq||rs> = <pq|rs> - <pq|sr>.
    With ``lazy=False`` only the blocks that were explicitly set are available.
    """

    def __init__(self, B1, B2, slices1, slices2, antisymmetric, lazy=True):
        self._B1, self._B2 = B1, B2
        self._s1, self._s2 = slices1, slices2
        self._antisymmetric = antisymmetric
        self._lazy = lazy

    def __getattr__(self, name):
        if name.startswith("_") or len(name) != 4 or set(name) - set("ov"):
            raise AttributeError(name)
        if name == "vvvv":
            raise VVVVError("vvvv blocks are never stored; contract them on the fly with the DF vectors.")
        if not self.__dict__.get("_lazy", False):
            raise AttributeError(f"Two-body block '{name}' is not available.")
        p, r = self._s1[name[0]], self._s1[name[2]]
        q, s = self._s2[name[1]], self._s2[name[3]]
        v = np.einsum("xpr,xqs->pqrs", self._B1[:, p, r], self._B2[:, q, s], optimize=True)
        if self._antisymmetric:
            v -= np.einsum("xps,xqr->pqrs", self._B1[:, p, s], self._B2[:, q, r], optimize=True)
        v = np.asfortranarray(v)
        setattr(self, name, v)
        return v

    def blocks(self):
        return [k for k in self.__dict__ if not k.startswith("_")]


class DFHamiltonian:
    """Normal-ordered Hamiltonian H = F + V with V in factorized form.

    Attributes mirror ccpy.models.integrals.Integral: ``H.a``/``H.b`` (Fock blocks),
    ``H.aa``/``H.ab``/``H.bb`` (two-body blocks, see DFTwoBody), and ``H.chol.a``/``H.chol.b``
    (DF vectors, shape (naux, n1, n2), by occupied/virtual block).
    """

    order = 2

    def __init__(self, system, fock_a, fock_b, B_a, B_b, lazy=True):
        self.system = system
        noa, nob = system.noccupied_alpha, system.noccupied_beta
        n = system.norbitals
        sl = {"a": {"o": slice(0, noa), "v": slice(noa, n)},
              "b": {"o": slice(0, nob), "v": slice(nob, n)}}
        self.slices = sl
        self.a = Integral.from_empty(system, 1, use_none=True).a
        self.b = Integral.from_empty(system, 1, use_none=True).b
        self.chol = Integral.from_empty(system, 1, use_none=True)
        for s, F, B in (("a", fock_a, B_a), ("b", fock_b, B_b)):
            for blk in ("oo", "ov", "vo", "vv"):
                p, q = sl[s][blk[0]], sl[s][blk[1]]
                setattr(getattr(self, s), blk, F[p, q])
                setattr(getattr(self.chol, s), blk, B[:, p, q])
        self.aa = DFTwoBody(B_a, B_a, sl["a"], sl["a"], True, lazy)
        self.ab = DFTwoBody(B_a, B_b, sl["a"], sl["b"], False, lazy)
        self.bb = DFTwoBody(B_b, B_b, sl["b"], sl["b"], True, lazy)

    @property
    def naux(self):
        return self.chol.a.oo.shape[0]

    def copy(self):
        """Independent, non-lazy copy holding the Fock blocks, the DF vectors, and the
        standard two-body blocks. Used as the starting point of similarity transformations,
        which overwrite blocks in place."""
        new = object.__new__(DFHamiltonian)
        new.system, new.slices = self.system, self.slices
        new.a = Integral.from_empty(self.system, 1, use_none=True).a
        new.b = Integral.from_empty(self.system, 1, use_none=True).b
        new.chol = Integral.from_empty(self.system, 1, use_none=True)
        for s in ("a", "b"):
            for blk in ("oo", "ov", "vo", "vv"):
                setattr(getattr(new, s), blk, getattr(getattr(self, s), blk).copy())
                setattr(getattr(new.chol, s), blk, getattr(getattr(self.chol, s), blk).copy())
        for spin in ("aa", "ab", "bb"):
            src = getattr(self, spin)
            dst = DFTwoBody(None, None, None, None, src._antisymmetric, lazy=False)
            names = set(src.blocks()) | (set(STANDARD_BLOCKS[spin]) if src._lazy else set())
            for blk in names:
                setattr(dst, blk, getattr(src, blk).copy())
            setattr(new, spin, dst)
        return new


def build_fock(h_a, h_b, B_a, B_b, nocc_a, nocc_b):
    """Fock matrices f = h + sum_i <pi||qi> over the occupied orbitals of each spin
    (all orbitals, frozen core included)."""
    J = np.einsum("xii->x", B_a[:, :nocc_a, :nocc_a]) + np.einsum("xii->x", B_b[:, :nocc_b, :nocc_b])
    f_a = h_a + np.einsum("xpq,x->pq", B_a, J, optimize=True) - np.einsum(
        "xpi,xiq->pq", B_a[:, :, :nocc_a], B_a[:, :nocc_a, :], optimize=True)
    f_b = h_b + np.einsum("xpq,x->pq", B_b, J, optimize=True) - np.einsum(
        "xpi,xiq->pq", B_b[:, :, :nocc_b], B_b[:, :nocc_b, :], optimize=True)
    return f_a, f_b


def determinant_energy(h_a, h_b, B_a, B_b, nocc_a, nocc_b):
    """Electronic energy of the determinant occupying the first nocc_a alpha and nocc_b
    beta orbitals."""
    oa, ob = slice(0, nocc_a), slice(0, nocc_b)
    Ja = np.einsum("xii->x", B_a[:, oa, oa])
    Jb = np.einsum("xii->x", B_b[:, ob, ob])
    Ka = np.einsum("xij,xji->", B_a[:, oa, oa], B_a[:, oa, oa], optimize=True)
    Kb = np.einsum("xij,xji->", B_b[:, ob, ob], B_b[:, ob, ob], optimize=True)
    return (np.trace(h_a[oa, oa]) + np.trace(h_b[ob, ob])
            + 0.5 * ((Ja + Jb) @ (Ja + Jb) - Ka - Kb))
