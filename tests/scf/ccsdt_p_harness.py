"""Testing harness for the CC(P) densities (ccpy/density/ccsdt_p_rdm.py) and the CC(P)
gradient (ccpy/gradients/ccsdt_p_gradient.py), P = singles, doubles and a list of triples.

The reference is exact: a determinant-space CC(P) / left-CC(P) (ccpy.gradients.ccpq) on the
same MO Hamiltonian as ccpy (h and the DF/Cholesky vectors of the case), whose densities are
evaluated in the T1-dressed frame,

    rho[O] = <Phi|(1 + Lambda) e^{-T2-T3} O e^{T2+T3}|Phi>,   O normal-ordered,

which is what calc_ccsdt_p_rdm must return (conjugate to the stored blocks of H(T1)).

Densities are compared in a canonical form, because the split of a functional's derivative
among stored blocks that represent the same integrals (H.aa.ooov and H.aa.oovo, ...) is a
matter of how the terms are written. Every block is embedded into the full correlated
tensor of its spin case and the same-spin tensors are antisymmetrized:

    "a", "b"   : rdm1 per spin                     F1 = sum rdm1 . f(T1)
    "aa", "bb" : A[sum of embedded rdm2 blocks]    F2 = sum rdm2.aa . <pq||rs>(T1) + ...
    "ab"       : sum of embedded rdm2.ab blocks         + sum rdm2.ab . <pq|rs>(T1)

so the reference is "aa" = Gamma^N_{alpha^4} / 4, "ab" = Gamma^N_{alpha beta alpha beta}.
Errors are reported per spin case and o/v pattern of that canonical tensor.

    case = PspaceCase("nh3_ccsdt_p")
    rdm1, rdm2 = calc_ccsdt_p_rdm(case.T, case.L, case.t3_excitations, case.system)
    vvvv = lambda a, s: calc_ccsdt_p_rdm2_vvvv_slice(case.T, case.L, case.t3_excitations, a, s)
    case.report(case.canonical(rdm1, rdm2, vvvv), "full")      # max |error| per block

The reference can be split into the pieces a CC(P) density adds to the CCSD one:

    "ccsd"  : T3 = L3 = 0 (equals ccpy.density.ccsd_rdm.calc_ccsd_rdm with the CC(P) T1, T2, L1, L2)
    "t3"    : terms with T3 and no L3          ("full" with L3 = 0, minus "ccsd")
    "l3"    : terms with L3                    ("full" minus "full" with L3 = 0)
    "no_l3" : "ccsd" + "t3" (the density with L3 = 0)
    "full"  : everything

    case.report(mine, "l3")       # compare your L3 terms alone

zeroed(T) / zeroed(L) copy an operator with its triples set to zero, so a density routine
evaluated at (T, zeroed(L)) must give "no_l3" and at (zeroed(T), zeroed(L)) must give "ccsd".

to_blocks(canonical) turns any canonical density back into (rdm1, rdm2, vvvv_slice), and
case.gradient(rdm=..., vvvv_slice=...) runs the production gradient with it, so the gradient
plumbing is checked with the reference densities before ccsdt_p_rdm.py exists.
Run by tests/scf/test_ccsdt_p_harness.py.
"""

import contextlib
import copy
import io

import numpy as np

from ccpy import CCSDT_P, Molecule, RHF, Reference, get_active_triples_pspace
from ccpy.gradients.ccpq import spaces
from ccpy.gradients.ccpq.amplitudes import SelectedCC
from ccpy.gradients.ccpq.densities import transition_densities
from ccpy.gradients.ccpq.determinants import DeterminantSpace, hamiltonian_matrix, spin_orbital_integrals
from ccpy.models.integrals import Integral

AUXILIARY_BASIS = "def2-universal-jkfit"
OPTIONS = dict(amp_print_threshold=1.0, energy_convergence=1.0e-12, amp_convergence=1.0e-10)
NH3 = "N 0.0 0.0 0.12; H 0.0 0.94 -0.27; H 0.81 -0.47 -0.29; H -0.80 -0.45 -0.25"

CASES = {
    # name: (geometry, basis, nfrozen, triples) with triples "all" (CCSDT) or the active
    # space (nact_occupied, nact_unoccupied) of get_active_triples_pspace. NH3/STO-3G with a
    # frozen core has 4 correlated occupied and 3 virtual orbitals per spin, so every spin
    # case of T3 is present (1225 determinants).
    "nh3_ccsdt": (NH3, "sto-3g", 1, "all"),
    "nh3_ccsdt_p": (NH3, "sto-3g", 1, (2, 2)),
}
SPIN_CASES = ("a", "b", "aa", "ab", "bb")
PARTS = ("ccsd", "t3", "l3", "no_l3", "full")
TRIPLES = ("aaa", "aab", "abb", "bbb")


def _quiet(f, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*args, **kwargs)


def _exp(T, n, sign, max_rank):
    """e^{sign T} as a finite series."""
    X, term = np.eye(n), np.eye(n)
    for k in range(1, max_rank + 1):
        term = sign * (T @ term) / k
        if not np.any(term):
            break
        X += term
    return X


class PspaceCase:
    """A converged CCSDT(P) / left-CCSDT(P) calculation plus the exact reference densities."""

    def __init__(self, name, spec=None):
        geometry, basis, nfrozen, triples = CASES[name] if spec is None else spec
        self.name = name
        self.mol = Molecule(geometry, basis, auxiliary_basis=AUXILIARY_BASIS, verbose=False)
        self.mf = RHF(self.mol, verbose=0, e_tol=1.0e-12, d_tol=1.0e-10)
        _quiet(self.mf.kernel)
        system = Reference(self.mf, nfrozen=nfrozen).system
        if triples == "all":
            system.set_active_space(nact_occupied=system.noccupied_alpha, nact_unoccupied=system.nunoccupied_alpha)
        else:
            system.set_active_space(nact_occupied=triples[0], nact_unoccupied=triples[1])
        self.pspace = _quiet(get_active_triples_pspace, system)
        self.nfrozen = nfrozen
        self.cc = CCSDT_P(self.mf, copy.deepcopy(self.pspace), nfrozen=nfrozen, **OPTIONS)
        self.energy = _quiet(self.cc.energy)
        self.left = _quiet(self.cc.left.ensure_solved)
        self.system, self.T, self.L = self.cc.system, self.cc.T, self.left.L
        self.t3_excitations = self.cc.t3_excitations
        self._build_reference()

    # ------------------------------------------------------------------ exact reference
    def _build_reference(self):
        reference = self.cc.reference
        nmo, nocc = reference.h_mo["a"].shape[0], self.nfrozen + self.system.noccupied_alpha
        B = reference.B_mo["a"]
        eri = np.einsum("xpq,xrs->pqrs", B, B, optimize=True)
        S = self.space = DeterminantSpace(nmo, nocc, nocc, self.nfrozen)
        self.one, self.two = S.one_body_transitions(), S.two_body_transitions()
        h_so, v_so = spin_orbital_integrals(reference.h_mo["a"], eri)
        H = hamiltonian_matrix(S, h_so, v_so, self.one, self.two)
        self.nocc, self.h_so, self.v_so, self.H = nocc, h_so, v_so, H
        P = spaces.union(spaces.ranks(S, (1, 2)), spaces.from_t3_excitations(S, self.t3_excitations))
        self.det = SelectedCC(S, H, P).run()
        self.oracle_energy = self.det.energy + self.mol.nuclear_repulsion
        rank = S.rank[P]
        self.ref = {}
        full = self._dressed_density(rank, t3=True, l3=True)
        no_l3 = self._dressed_density(rank, t3=True, l3=False)
        ccsd = self._dressed_density(rank, t3=False, l3=False)
        self.ref["full"] = full
        self.ref["ccsd"] = ccsd
        self.ref["t3"] = {k: no_l3[k] - ccsd[k] for k in SPIN_CASES}
        self.ref["l3"] = {k: full[k] - no_l3[k] for k in SPIN_CASES}
        self.ref["no_l3"] = no_l3

    def _dressed_density(self, rank, t3, l3):
        """Canonical normal-ordered densities of <(1+Lambda) e^{-T'} O e^{T'}>, T' = T without
        singles (and without triples unless t3), Lambda without triples unless l3."""
        lam = self.det.lambda_grad.copy()
        if not l3:
            lam = lam * (rank != 3)
        n = self.space.ndet
        b = np.zeros(n)
        b[0] = 1.0
        b[self.det_P()] += lam
        return self.canonical_of_coefficients(np.outer(b, np.eye(n)[0]), t3=t3)

    def dressed_UV(self, t3=True):
        """e^{-T'} and e^{T'} for T' = T without singles (and without triples unless t3)."""
        key = ("UV", t3)
        if key not in self.__dict__:
            S = self.space
            rank = S.rank[self.det_P()]
            t = self.det.t * (rank != 1)
            if not t3:
                t = t * (rank != 3)
            T = self.det_exc().matrix(t, S.ndet)
            max_rank = int(S.rank.max())
            self.__dict__[key] = (_exp(T, S.ndet, -1.0, max_rank), _exp(T, S.ndet, 1.0, max_rank))
        return self.__dict__[key]

    def canonical_of_coefficients(self, A, t3=True):
        """Canonical normal-ordered T1-dressed densities of a functional sum_AB A_AB Hbar_AB
        (Eqs. 6.1, 6.7), the frozen core removed."""
        S = self.space
        U, V = self.dressed_UV(t3)
        K = U.T @ A @ V.T
        rho = transition_densities(K, S, self.one, self.two)
        norm = np.trace(K)
        nocc = np.diag(S.ref_occ.astype(float))
        gN = rho.gamma - norm * nocc
        cross = (np.einsum("pr,qs->pqrs", nocc, gN) - np.einsum("ps,qr->pqrs", nocc, gN)
                 - np.einsum("qr,ps->pqrs", nocc, gN) + np.einsum("qs,pr->pqrs", nocc, gN))
        GN = rho.Gamma - cross - norm * (np.einsum("pr,qs->pqrs", nocc, nocc) - np.einsum("ps,qr->pqrs", nocc, nocc))
        nmo = S.norb
        a = np.arange(self.nfrozen, nmo)
        b_ = a + nmo
        return {"a": gN[np.ix_(a, a)], "b": gN[np.ix_(b_, b_)],
                "aa": 0.25 * GN[np.ix_(a, a, a, a)], "ab": GN[np.ix_(a, b_, a, b_)],
                "bb": 0.25 * GN[np.ix_(b_, b_, b_, b_)]}

    def det_exc(self):
        if not hasattr(self, "_exc"):
            self._exc = self.space.excitation_transitions(self.det_P())
        return self._exc

    def det_P(self):
        S = self.space
        return spaces.union(spaces.ranks(S, (1, 2)), spaces.from_t3_excitations(S, self.t3_excitations))

    # ------------------------------------------------------------------ canonical form
    def _slices(self, spin):
        nocc = self.system.noccupied_alpha if spin == "a" else self.system.noccupied_beta
        return {"o": slice(0, nocc), "v": slice(nocc, self.system.norbitals)}

    def canonical(self, rdm1, rdm2, vvvv_slice=None):
        """Canonical tensors {"a", "b", "aa", "ab", "bb"} of densities in ccpy block form."""
        n = self.system.norbitals
        out = {}
        for s in ("a", "b"):
            sl = self._slices(s)
            out[s] = np.zeros((n, n))
            for blk in ("oo", "ov", "vo", "vv"):
                x = getattr(getattr(rdm1, s), blk, None)
                if x is not None and not np.isscalar(x):
                    out[s][sl[blk[0]], sl[blk[1]]] += x
        spins = {"aa": "aaaa", "ab": "abab", "bb": "bbbb"}
        for case, sp4 in spins.items():
            out[case] = np.zeros((n,) * 4)
            sls = [self._slices(c) for c in sp4]
            for blk in _patterns():
                x = getattr(getattr(rdm2, case), blk, None)
                if x is not None and not np.isscalar(x):
                    out[case][tuple(sls[k][blk[k]] for k in range(4))] += x
        if vvvv_slice is not None:
            va, vb = self._slices("a")["v"], self._slices("b")["v"]
            for a in range(va.stop - va.start):
                G_aa, G_ab = vvvv_slice(a, "a")
                out["aa"][va.start + a, va, va, va] += G_aa
                out["ab"][va.start + a, vb, va, vb] += G_ab
            for a in range(vb.stop - vb.start):
                out["bb"][vb.start + a, vb, vb, vb] += vvvv_slice(a, "b")
        for case in ("aa", "bb"):
            X = out[case]
            out[case] = 0.25 * (X - X.transpose(1, 0, 2, 3) - X.transpose(0, 1, 3, 2) + X.transpose(1, 0, 3, 2))
        return out

    def to_blocks(self, canon):
        """(rdm1, rdm2, vvvv_slice) in ccpy block form from a canonical density."""
        rdm1 = Integral.from_empty(self.system, 1, use_none=True)
        rdm2 = Integral.from_empty(self.system, 2, use_none=True)
        for s in ("a", "b"):
            sl = self._slices(s)
            for blk in ("oo", "ov", "vo", "vv"):
                setattr(getattr(rdm1, s), blk, canon[s][sl[blk[0]], sl[blk[1]]].copy())
        spins = {"aa": "aaaa", "ab": "abab", "bb": "bbbb"}
        for case, sp4 in spins.items():
            sls = [self._slices(c) for c in sp4]
            for blk in _patterns():
                if blk != "vvvv":
                    setattr(getattr(rdm2, case), blk, canon[case][tuple(sls[k][blk[k]] for k in range(4))].copy())
        va, vb = self._slices("a")["v"], self._slices("b")["v"]

        def vvvv_slice(a, spin):
            if spin == "a":
                return canon["aa"][va.start + a, va, va, va], canon["ab"][va.start + a, vb, va, vb]
            return canon["bb"][vb.start + a, vb, vb, vb]

        return rdm1, rdm2, vvvv_slice

    # ------------------------------------------------------------------ comparison
    def compare(self, canon, part="full"):
        """{(spin case, o/v pattern): max |canon - reference|}."""
        ref = self.ref[part]
        errors = {}
        for case in SPIN_CASES:
            sp = {"a": "aa", "b": "bb", "aa": "aaaa", "ab": "abab", "bb": "bbbb"}[case]
            sls = [self._slices(c) for c in sp]
            for blk in (_patterns(len(sp))):
                idx = tuple(sls[k][blk[k]] for k in range(len(sp)))
                diff = canon[case][idx] - ref[case][idx]
                errors[(case, blk)] = np.abs(diff).max(initial=0.0)
        return errors

    def report(self, canon, part="full", tol=1.0e-10):
        """Print the blocks off by more than tol and return the largest error."""
        errors = self.compare(canon, part)
        print(f"\n   {self.name}: {part} densities")
        for (case, blk), e in errors.items():
            if e > tol:
                ref = self.ref[part][case]
                print(f"   {case:2s} {blk}: max |error| = {e:.3e}   (largest reference element {np.abs(ref).max():.2e})")
        worst = max(errors.values())
        print(f"   largest error {worst:.3e}")
        return worst

    # ------------------------------------------------------------------ gradients
    def gradient(self, rdm=None, vvvv_slice=None):
        from ccpy.gradients.ccsdt_p_gradient import calc_ccsdt_p_gradient

        return _quiet(calc_ccsdt_p_gradient, self.T, self.L, self.t3_excitations, self.cc.reference, rdm=rdm,
                      vvvv_slice=vvvv_slice, verbose=False)

    def energy_at(self, mol):
        """CC(P) energy at another geometry with the same (reference-geometry) P space."""
        mf = RHF(mol, verbose=0, e_tol=1.0e-12, d_tol=1.0e-10)
        _quiet(mf.kernel)
        cc = CCSDT_P(mf, copy.deepcopy(self.pspace), nfrozen=self.nfrozen, **OPTIONS)
        return _quiet(cc.energy)


def zeroed(op):
    """Copy of a ClusterOperator with its triples set to zero."""
    out = copy.deepcopy(op)
    for case in TRIPLES:
        getattr(out, case)[:] = 0.0
    return out


def _patterns(order=4):
    return [format(k, f"0{order}b").replace("0", "o").replace("1", "v") for k in range(2 ** order)]
