"""Testing harness for the CC(P;Q) gradient and its kernels
(ccpy/moments/ccp3_response.py, ccpy/density/ccp3_rdm.py), P = SD + a list of triples and
Q = the remaining triples.

The exact reference is the determinant-space CC(P;Q) functional of ccpy.gradients.ccpq on the
same MO Hamiltonian as ccpy (as in ccsdt_p_harness.py), with the bare-Fock ("A") or the full
diagonal ("D", Eq. 3.4) denominator. Every intermediate of the driver is converted to ccpy's
layouts: determinant coefficients c_I
(E_I|Phi> = +|Phi_I>) become ccpy amplitudes phase_I c_I for the operator strings
a+ b+ c+ k j i, written to every antisymmetric copy.

    case = CCP3Case("nh3_ccp3_A")
    case.compare_q(M3, "M3")                  # Q triples only; M3 may hold values on P
    case.compare_p(s_t, "s_t")                # P-space vectors, per block
    case.compare_p(B_action(x, ...), case.B_action(x))   # any P vector x
    case.report(case.canonical(*rdm_sigma(...)), "rdm_sigma")

References (case.cref): M3, N3, D3, R3, L3, hbar_diagonal (<K|Hbar_N|K>, Q only), delta,
s_lambda, s_t, k, sigma, pi, the dressed canonical densities rdm_pi, rdm_sigma, rdm_moments,
and d (the bare-Fock denominator vector of Eq. 5.6, variant "A"). case.B_action(x) and
case.BT_action(x) apply the exact B = Hbar_PP - E and its transpose.

reference_kernels() returns the kernels made of these references, so the driver can be run
end to end before any kernel is written. Run by tests/scf/test_ccp3_harness.py.
"""

import copy
from itertools import permutations
from types import SimpleNamespace

import numpy as np

from ccpy import CCP3, CCSDT_P, RHF
from ccpy.gradients.ccpq import spaces
from ccpy.gradients.ccpq.amplitudes import SelectedCC
from ccpy.gradients.ccpq.determinants import fock_diagonal
from ccsdt_p_harness import NH3, OPTIONS, PspaceCase, _quiet

SINGLES_DOUBLES = ("a", "b", "aa", "ab", "bb")
TRIPLES = ("aaa", "aab", "abb", "bbb")


def _triples_shape(system, case):
    no = {"a": system.noccupied_alpha, "b": system.noccupied_beta}
    nv = {"a": system.nunoccupied_alpha, "b": system.nunoccupied_beta}
    return tuple(nv[s] for s in case) + tuple(no[s] for s in case)


def _permutation_sign(p):
    inversions = sum(1 for i in range(len(p)) for j in range(i + 1, len(p)) if p[i] > p[j])
    return -1 if inversions % 2 else 1


def _antisymmetric_images(case, abc, ijk):
    """Index tuples and signs of all antisymmetric copies of one triple of a spin case."""
    pos = {"aaa": (0, 1, 2), "aab": (0, 1), "abb": (1, 2), "bbb": (0, 1, 2)}[case]
    out = []
    for pv in permutations(range(len(pos))):
        for po in permutations(range(len(pos))):
            v, o = list(abc), list(ijk)
            for k, src in enumerate(pv):
                v[pos[k]] = abc[pos[src]]
            for k, src in enumerate(po):
                o[pos[k]] = ijk[pos[src]]
            out.append((tuple(v) + tuple(o), _permutation_sign(pv) * _permutation_sign(po)))
    return out


def valid_rows(t3_excitations, case):
    """Rows of a triples list that are real excitations (ccpy marks an empty list by a row of ones)."""
    rows = np.atleast_2d(np.asarray(t3_excitations[case]))
    return ~np.all(rows == 1, axis=1)


def qspace_mask(t3_excitations, system):
    """Boolean arrays of the Q triples (distinct indices, not in P), all antisymmetric copies."""
    mask = SimpleNamespace()
    for c in TRIPLES:
        shape = _triples_shape(system, c)
        a, b, cc, i, j, k = np.indices(shape)
        m = np.ones(shape, dtype=bool)
        same = {"aaa": [(a, b), (a, cc), (b, cc), (i, j), (i, k), (j, k)], "aab": [(a, b), (i, j)],
                "abb": [(b, cc), (j, k)], "bbb": [(a, b), (a, cc), (b, cc), (i, j), (i, k), (j, k)]}[c]
        for x, y in same:
            m &= x != y
        if t3_excitations is not None:
            rows = np.atleast_2d(np.asarray(t3_excitations[c]))[valid_rows(t3_excitations, c)] - 1
            for row in rows:
                for idx, _ in _antisymmetric_images(c, tuple(row[:3]), tuple(row[3:])):
                    m[idx] = False
        setattr(mask, c, m)
    return mask

CASES = {
    # name: (geometry, basis, nfrozen, active space of the P triples, variant). With the (1, 1)
    # active space Q holds triples of every spin case.
    "nh3_ccp3_A": (NH3, "sto-3g", 1, (1, 1), "A"),
    "nh3_ccp3_D": (NH3, "sto-3g", 1, (1, 1), "D"),
}


class CCP3Case(PspaceCase):
    """CCSDT(P), left-CCSDT(P) and CC(P;3) (two_body_approx=False) plus exact references."""

    def __init__(self, name):
        geometry, basis, nfrozen, triples, variant = CASES[name]
        super().__init__(name, spec=(geometry, basis, nfrozen, triples))
        self.variant = variant
        self.ccp3 = CCP3(self.cc, two_body_approx=False, variant=variant)
        _quiet(self.ccp3.ensure_solved)
        self.delta = self.ccp3.deltas[variant]
        self.ecorr = self.cc.correlation_energy
        self._build_layout()
        self._build_correction()

    # ------------------------------------------------------------------ layouts
    def _orbital(self, q):
        """Spin orbital of the determinant space -> (spin, spatial orbital)."""
        n = self.space.norb
        return ("a" if q < n else "b"), q % n

    def _build_layout(self):
        """For every P and Q determinant: (spin case, unique ccpy index, phase)."""
        S, nfz = self.space, self.nfrozen
        nocc = self.nocc
        rows = {}
        for c in TRIPLES:
            ex = np.atleast_2d(np.asarray(self.t3_excitations[c])) - 1
            for n, row in enumerate(ex):
                rows[(c, tuple(int(x) for x in row))] = n

        def entry(k):
            holes, parts = S.holes_particles(k)
            case = "".join(self._orbital(q)[0] for q in parts)
            vir = tuple(self._orbital(q)[1] - nocc for q in parts)
            occ = tuple(self._orbital(q)[1] - nfz for q in holes)
            phase = S.excitation_string(k)[1]
            if len(case) == 3:
                return case, vir + occ, rows.get((case, vir + occ)), phase
            return case, vir + occ, None, phase

        self.P = self.det_P()
        self.Q = spaces.complement(S, self.P, (3,))
        self.p_layout = [entry(k) for k in self.P]
        self.q_layout = [entry(k) for k in self.Q]

    def _zeros_p(self):
        T = self.T
        return SimpleNamespace(**{c: np.zeros_like(np.asarray(getattr(T, c))) for c in SINGLES_DOUBLES + TRIPLES})

    def to_p(self, vec):
        """Vector over the P determinants -> ccpy P layout (all antisymmetric copies)."""
        x = self._zeros_p()
        for value, (case, idx, row, phase) in zip(vec, self.p_layout):
            v = phase * value
            X = getattr(x, case)
            if len(case) == 3:
                X[row] = v
            elif case in ("a", "b", "ab"):
                X[idx] = v
            else:
                a, b, i, j = idx
                X[a, b, i, j] = X[b, a, j, i] = v
                X[b, a, i, j] = X[a, b, j, i] = -v
        return x

    def from_p(self, x):
        """ccpy P layout -> vector over the P determinants."""
        out = np.zeros(len(self.P))
        for n, (case, idx, row, phase) in enumerate(self.p_layout):
            X = np.asarray(getattr(x, case))
            out[n] = phase * (X[row] if len(case) == 3 else X[idx])
        return out

    def to_q(self, vec, scalar=False):
        """Vector over the Q determinants -> full antisymmetric triples arrays (scalar=True:
        the same value, unsigned, on every copy, e.g. denominators)."""
        X = SimpleNamespace(**{c: np.zeros(_triples_shape(self.system, c)) for c in TRIPLES})
        for value, (case, idx, _, phase) in zip(vec, self.q_layout):
            arr = getattr(X, case)
            for image, sign in _antisymmetric_images(case, idx[:3], idx[3:]):
                arr[image] = value if scalar else sign * phase * value
        return X

    # ------------------------------------------------------------------ references
    def _build_correction(self):
        S = self.space
        den = "mp" if self.variant == "A" else "full"
        fd = fock_diagonal(self.h_so, self.v_so, S.ref_occ)
        r = self.corr_det = SelectedCC(S, self.H, self.P, self.Q, fock_diag=fd, denominator=den,
                                       t_guess=self.det.t).run()
        if abs(r.delta - self.delta) > 1.0e-9:
            raise RuntimeError(f"determinant-space delta {r.delta:.12f} does not reproduce ccpy's {self.delta:.12f}")
        a = r.arrays
        n = S.ndet
        e0 = np.eye(n)[0]
        b = e0.copy()
        b[self.P] += r.lam
        emb = lambda x, idx: np.bincount(idx, weights=x, minlength=n)  # noqa: E731
        sigma, pi, ell, rr = emb(a["sigma"], self.P), emb(a["pi"], self.P), emb(a["ell"], self.Q), emb(a["r"], self.Q)
        A_moments = np.outer(ell, e0) + np.outer(b, rr)
        if self.variant == "D":
            w = ell * rr
            w[0] = -np.sum(a["ell"] * a["r"])
            A_moments += np.diag(w)
        Hbar = r.Hbar
        self.B = Hbar[np.ix_(self.P, self.P)] - r.energy * np.eye(len(self.P))
        self.cref = {
            "M3": self.to_q(a["M"]), "N3": self.to_q(a["N"]), "D3": self.to_q(a["D"], scalar=True),
            "R3": self.to_q(a["r"]), "L3": self.to_q(a["ell"]), "delta": r.delta,
            "hbar_diagonal": self.to_q(np.diag(Hbar)[self.Q] - self.H[0, 0], scalar=True),
            "s_lambda": self.to_p(a["s_lambda"]), "s_t": self.to_p(a["s_t"]), "k": self.to_p(a["k"]),
            "sigma": self.to_p(a["sigma"]), "pi": self.to_p(a["pi"]),
            "rdm_pi": self.canonical_of_coefficients(np.outer(pi, e0)),
            "rdm_sigma": self.canonical_of_coefficients(np.outer(b, sigma) - (r.lam @ a["sigma"]) * np.outer(e0, e0)),
            "rdm_moments": self.canonical_of_coefficients(A_moments),
        }
        if r.d_mp is not None:
            nmo, nocc = S.norb, self.nocc
            nfz = self.nfrozen
            self.cref["d"] = {s: {"o": r.d_mp[off + nfz:off + nocc], "v": r.d_mp[off + nocc:off + nmo]}
                              for s, off in (("a", 0), ("b", nmo))}
        self.mask = qspace_mask(self.t3_excitations, self.system)

    def B_action(self, x):
        """Exact sum_J (Hbar_IJ - E delta_IJ) x_J in the ccpy P layout."""
        return self.to_p(self.B @ self.from_p(x))

    def BT_action(self, x):
        """Exact sum_J x_J (Hbar_JI - E delta_JI) in the ccpy P layout."""
        return self.to_p(self.B.T @ self.from_p(x))

    def reference_kernels(self):
        """The CC(P;Q) gradient kernels built from the references."""
        c = self.cref
        return {
            "moments": lambda *args: (c["M3"], c["N3"]),
            "hbar_diagonal": lambda *args: c["hbar_diagonal"],
            "s_lambda": lambda *args: c["s_lambda"],
            "s_t": lambda *args: c["s_t"],
            "k": lambda *args: c["k"],
            "B_action": lambda x, *args: self.B_action(x),
            "BT_action": lambda x, *args: self.BT_action(x),
            "rdm_pi": lambda *args: self.to_blocks(c["rdm_pi"]),
            "rdm_sigma": lambda *args: self.to_blocks(c["rdm_sigma"]),
            "rdm_moments": lambda *args: self.to_blocks(c["rdm_moments"]),
        }

    def random_p(self, seed=3):
        """A random P-space vector in the ccpy layout (unique coordinates mapped to all copies)."""
        rng = np.random.default_rng(seed)
        return self.to_p(rng.standard_normal(len(self.P)))

    # ------------------------------------------------------------------ comparisons
    def compare_p(self, x, ref):
        """{block: max |x - ref|} for P-space vectors (ref: a cref name or a vector)."""
        ref = self.cref[ref] if isinstance(ref, str) else ref
        out = {}
        for c in SINGLES_DOUBLES + TRIPLES:
            diff = np.asarray(getattr(x, c)) - np.asarray(getattr(ref, c))
            if c in TRIPLES:
                diff = diff[valid_rows(self.t3_excitations, c)]
            out[c] = float(np.abs(diff).max(initial=0.0))
        return out

    def compare_q(self, X, ref):
        """{spin case: max |X - ref|} over the Q triples."""
        ref = self.cref[ref] if isinstance(ref, str) else ref
        return {c: float(np.abs((getattr(X, c) - getattr(ref, c))[getattr(self.mask, c)]).max(initial=0.0))
                for c in TRIPLES}

    def report(self, canon, part):
        """report() of ccsdt_p_harness for the correction densities rdm_pi, rdm_sigma, rdm_moments."""
        self.ref[part] = self.cref[part]
        return super().report(canon, part)

    @property
    def hamiltonian(self):
        """H(T1) as the driver passes it to the kernels (build_hbar_ccs_chol of the bare H)."""
        if "_hamiltonian" not in self.__dict__:
            from ccpy.hbar.hbar_ccs_chol import build_hbar_ccs_chol

            self._hamiltonian = build_hbar_ccs_chol(self.T, self.cc.reference.H, self.system)
        return self._hamiltonian

    # ------------------------------------------------------------------ energies
    def energy_at(self, mol):
        """E^(P) + delta(P;Q) at another geometry, with the reference-geometry P space."""
        mf = RHF(mol, verbose=0, e_tol=1.0e-12, d_tol=1.0e-10)
        _quiet(mf.kernel)
        cc = CCSDT_P(mf, copy.deepcopy(self.pspace), nfrozen=self.nfrozen, **OPTIONS)
        corr = CCP3(cc, two_body_approx=False, variant=self.variant)
        return _quiet(corr.energy)
