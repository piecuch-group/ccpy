"""Testing harness for the factorized CR-CC(2,3)_A gradient (ccpy/gradients/crcc23_gradient_reference.py).

Two kinds of checks:

1. Exact reference (small molecules, a few hundred determinants). A determinant-space
   calculation (ccpy.gradients.determinant_space) provides every intermediate - R3, L3,
   s_lambda, s_t^(L3), s_t^(R3), w, B^T w, z and the triples, z and w densities - converted
   to ccpy's spin-integrated blocks, so each piece can be compared on its own.

       case = ResponseCase("rohf_ch2")
       H, H0, reference = case.transformed
       R3, L3 = calc_r3_l3(case.T, case.L, H, H0, case.system)
       case.compare(R3, "R3")                                   # max |error| per spin block
       case.compare_DY(density, "z")                            # densities as (D1, Y)

2. Directional finite differences with ccpy's own code (any molecule size): the
   derivative of ccpy's Fortran delta_A along a random direction must equal the dot
   product with s_t / s_lambda, B^T w must equal the derivative of the left-CCSD residual,
   and w, z must satisfy their equations with ccpy's EOMCCSD/left-CCSD Jacobian actions.

       case = ResponseCase("rhf_h2o_dz", reference=False)
       case.check_s_t(s_t)                                      # relative error

Molecules use density fitting (the factorized gradient contracts with the 3-center and
2-center derivative integrals). Both kinds are run by tests/scf/test_crcc23_harness.py.
"""

import contextlib
import copy
import io
from itertools import permutations

import numpy as np

SINGLES_DOUBLES = ("a", "b", "aa", "ab", "bb")
TRIPLES = ("aaa", "aab", "abb", "bbb")
AUXILIARY_BASIS = "def2-universal-jkfit"

CASES = {
    # name: (scf, geometry, basis, multiplicity, nfrozen)  -- determinant-space sized
    "rhf_h2o": ("RHF", "O 0 0 0.12; H 0 0.76 -0.47; H 0 -0.76 -0.47", "sto-3g", 1, 1),
    "rohf_ch2": ("ROHF", "C 0 0 0.1; H 0 0.98 -0.6; H 0 -0.98 -0.6", "sto-3g", 3, 1),
    "uhf_ch2": ("UHF", "C 0 0 0.1; H 0 0.98 -0.6; H 0 -0.98 -0.6", "sto-3g", 3, 1),
    # larger cases for the finite-difference checks only
    "rhf_h2o_dz": ("RHF", "O 0 0 0.12; H 0 0.76 -0.47; H 0 -0.76 -0.47", "cc-pvdz", 1, 1),
    "uhf_ch2_dz": ("UHF", "C 0 0 0.1; H 0 0.98 -0.6; H 0 -0.98 -0.6", "cc-pvdz", 3, 1),
}


def _get(x, key):
    return x[key] if isinstance(x, dict) else getattr(x, key)


def _quiet(f, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*args, **kwargs)


class ResponseCase:
    """A converged CCSD / left-CCSD calculation plus reference response quantities."""

    def __init__(self, name, reference=True, seed=11):
        from ccpy import Driver, Molecule
        import ccpy.scf
        from crcc23_triples import SpinOrbitalSpace

        scf, geometry, basis, multiplicity, nfrozen = CASES[name]
        self.name = name
        self.mol = mol = Molecule(geometry, basis, multiplicity=multiplicity,
                                  auxiliary_basis=AUXILIARY_BASIS, verbose=False)
        mf = getattr(ccpy.scf, scf)(mol, verbose=0, e_tol=1.0e-12, d_tol=1.0e-10)
        mf.kernel()
        d = _quiet(Driver.from_scf, mf, nfrozen=nfrozen)
        d.options.update({"energy_convergence": 1.0e-13, "amp_convergence": 1.0e-11})
        self.space = SpinOrbitalSpace(d.system)
        f_so, v_so = self.space.one_body(d.hamiltonian), self.space.two_body(d.hamiltonian)
        _quiet(d.run_cc, method="ccsd")
        self.H0 = copy.deepcopy(d.hamiltonian)          # bare H, before run_hbar overwrites it
        _quiet(d.run_hbar, method="ccsd")
        _quiet(d.run_leftcc, method="left_ccsd")
        _quiet(d.run_ccp3, method="crcc23")
        self.driver = d
        self.system, self.T, self.L = d.system, d.T, d.L[0]
        self.Hbar, self.fock = d.hamiltonian, d.fock
        self.delta = d.deltap3[0]["A"]
        self.rng = np.random.default_rng(seed)
        self.ref = self._reference(f_so, v_so) if reference else None
        self._transformed = None

    @property
    def transformed(self):
        """(H, H0, reference): the CCS-transformed Cholesky Hamiltonian (ccpy.hbar.hbar_ccs_chol),
        the bare Cholesky Hamiltonian, and the ccpy.methods.Reference of the driver's SCF (with the
        driver's System)."""
        if self._transformed is None:
            from ccpy.hbar.hbar_ccs_chol import build_hbar_ccs_chol
            from ccpy.methods.reference import Reference
            reference = Reference(self.driver.scf, self.system.nfrozen, self.system.ndelete)
            reference.system = self.system
            H0 = self._cholesky_hamiltonian(reference)
            self._transformed = (build_hbar_ccs_chol(self.T, H0, self.system), H0, reference)
        return self._transformed

    def _cholesky_hamiltonian(self, reference):
        """Bare Cholesky Hamiltonian: Fock blocks (from the driver's bare H0) plus H.chol."""
        from ccpy.models.integrals import Integral
        H0 = copy.deepcopy(self.H0)
        H0.chol = Integral.from_empty(self.system, 1, use_none=True)
        nfz = self.system.nfrozen
        for s, no in (("a", self.system.noccupied_alpha), ("b", self.system.noccupied_beta)):
            B = reference.B_mo[s][:, nfz:nfz + self.system.norbitals, nfz:nfz + self.system.norbitals]
            o, v = slice(0, no), slice(no, None)
            for blk, (p, q) in (("oo", (o, o)), ("ov", (o, v)), ("vo", (v, o)), ("vv", (v, v))):
                setattr(getattr(H0.chol, s), blk, B[:, p, q])
        return H0

    # -------------------------------------------------------------------------------
    # exact determinant-space reference
    # -------------------------------------------------------------------------------
    def _reference(self, f_so, v_so):
        from crcc23_triples import spin_orbital_quantities
        from determinant_space import (
            CCSDReference, DeterminantSpace, amplitudes_from_spin_orbital, extra_densities)

        ds = DeterminantSpace(self.space.nocc, self.space.nvir)
        R = CCSDReference(ds, f_so, v_so)
        q = spin_orbital_quantities(self.driver)
        t = R.solve_ccsd(amplitudes_from_spin_orbital(ds, q["t1"], q["t2"]))
        lam = R.solve_lambda(t)
        zero_f, zero_v = np.zeros_like(f_so), np.zeros_like(v_so)
        _, info = R.correction_derivative(t, lam, zero_f, zero_v)
        if not np.isclose(info["delta"], self.delta, atol=1.0e-12):
            raise RuntimeError("determinant-space delta_A does not reproduce ccpy's")

        ref0, Hbar = ds.ref, R.hbar(t)
        r, l = info["M"] / info["D"], info["N"] / info["D"]
        L3 = l @ R.triple_kets
        R3 = r @ R.triple_kets
        s_t_L3 = np.array([L3 @ Hbar @ tau @ ref0 for tau in R.taus])
        s_t_R3 = info["s_t"] - s_t_L3
        self._ds = ds
        dens = extra_densities(R, t, lam, info)
        return {
            "density_triples": self._density_blocks(*dens["triples"]),
            "density_z": self._density_blocks(*dens["z"]),
            "density_w": self._density_blocks(*dens["w"]),
            "R3": self._triples_blocks(r), "L3": self._triples_blocks(l),
            "s_lambda": self._blocks(info["s_lambda"]),
            "s_t_L3": self._blocks(s_t_L3), "s_t_R3": self._blocks(s_t_R3),
            "s_t": self._blocks(info["s_t"]),
            "w": self._blocks(info["w"]), "BTw": self._blocks(info["BTw"]),
            "z": self._blocks(info["z"]),
        }

    def _blocks(self, vec):
        """Unique singles/doubles vector -> full antisymmetric spin-integrated blocks."""
        ds, o = self._ds, self.space.nocc
        nv = self.space.nvir
        x1 = np.zeros((nv, o))
        x2 = np.zeros((nv, nv, o, o))
        ns = len(ds.singles)
        for value, (a, i) in zip(vec[:ns], ds.singles):
            x1[a - o, i] = value
        for value, (a, b, i, j) in zip(vec[ns:], ds.doubles):
            a, b = a - o, b - o
            x2[a, b, i, j] = x2[b, a, j, i] = value
            x2[b, a, i, j] = x2[a, b, j, i] = -value
        s = self.system
        A, B = slice(0, s.nunoccupied_alpha), slice(s.nunoccupied_alpha, nv)
        I, J = slice(0, s.noccupied_alpha), slice(s.noccupied_alpha, o)
        return {"a": x1[A, I], "b": x1[B, J], "aa": x2[A, A, I, I], "ab": x2[A, B, I, J], "bb": x2[B, B, J, J]}

    def _density_blocks(self, dgamma_so, dGamma_so):
        """Spin-orbital [occ|vir] normal-ordered densities -> spin-integrated blocks over the
        correlated orbitals."""
        n, order = self.space.n, self.space.order
        g = np.zeros((2 * n, 2 * n))
        g[np.ix_(order, order)] = dgamma_so
        G = np.zeros((2 * n,) * 4)
        G[np.ix_(order, order, order, order)] = dGamma_so
        a, b = slice(0, n), slice(n, 2 * n)
        return ({"a": g[a, a], "b": g[b, b]},
                {"aa": G[a, a, a, a], "ab": G[a, b, a, b], "bb": G[b, b, b, b]})

    def reference_DY(self, piece):
        """The reference density piece ("triples", "z", "w") as (D1, Y) conjugate to the bare
        (h, B), the form the gradient uses (Y_x = dF/dB_x, spin resolved)."""
        from ccpy.gradients.density_gradient import build_DY
        from ccpy.models.integrals import Integral
        g, G = self.ref[f"density_{piece}"]
        _, H0, reference = self.transformed
        cut = {"a": self.system.noccupied_alpha, "b": self.system.noccupied_beta}
        sl = lambda s, c: slice(0, cut[s]) if c == "o" else slice(cut[s], None)
        rdm1 = Integral.from_empty(self.system, 1, use_none=True)
        rdm2 = Integral.from_empty(self.system, 2, use_none=True)
        for s in ("a", "b"):
            for blk in ("oo", "ov", "vo", "vv"):
                setattr(getattr(rdm1, s), blk, g[s][sl(s, blk[0]), sl(s, blk[1])])
        # F = 1/4 sum Gamma_aa <pq||rs> + sum Gamma_ab <pq|rs> + 1/4 sum Gamma_bb <pq||rs>
        for spins, scale in (("aa", 0.25), ("ab", 1.0), ("bb", 0.25)):
            ss = (spins[0], spins[1], spins[0], spins[1])
            for blk in [p + q + r + t for p in "ov" for q in "ov" for r in "ov" for t in "ov"]:
                setattr(getattr(rdm2, spins), blk, scale * G[spins][tuple(sl(x, c) for x, c in zip(ss, blk))])
        return build_DY(rdm1, rdm2, None, H0.chol, None, reference.B_mo, reference)

    def compare_DY(self, mine, piece, verbose=True):
        """Max |error| of (D1, Y) per spin against the reference density piece."""
        ref = self.reference_DY(piece)
        errors = {}
        for s in ("a", "b"):
            errors[f"D1_{s}"] = float(np.abs(mine[0][s] - ref[0][s]).max())
            errors[f"Y_{s}"] = float(np.abs(mine[1][s] - ref[1][s]).max())
        if verbose:
            scale = max(np.abs(ref[1][s]).max() for s in ("a", "b"))
            print(f"   density_{piece} [{self.name}]  (largest reference Y element {scale:.2e})")
            for k, e in errors.items():
                print(f"      {k:<4} max |error| = {e:.2e}")
        return errors

    def exact_gradient(self, verbose=False):
        """CR-CC(2,3)_A nuclear gradient from the factorized CCSD densities plus the exact
        reference triples, z and w densities (validates the orbital response)."""
        from ccpy.density.ccsd_rdm import calc_ccsd_rdm, calc_rdm2_vvvv_slice
        from ccpy.gradients.derivative_integrals import contract_derivative_integrals
        from ccpy.gradients.density_gradient import build_DY
        from ccpy.gradients.orbital_response import orbital_relaxation
        H, _, reference = self.transformed
        T, L = self.T, self.L
        D1, Y = build_DY(*calc_ccsd_rdm(T, L, self.system), lambda a, s: calc_rdm2_vvvv_slice(L, T, a, s),
                         H.chol, T, reference.B_mo, reference, ref_weight=True)
        for piece in ("triples", "z", "w"):
            d1, y = self.reference_DY(piece)
            D1 = {s: D1[s] + d1[s] for s in ("a", "b")}
            Y = {s: Y[s] + y[s] for s in ("a", "b")}
        D, Y, W = orbital_relaxation(D1, Y, reference, verbose=verbose)
        return contract_derivative_integrals(D, Y, W, reference)

    def energy_function(self):
        """E_CCSD + delta_A of a Molecule with this case's settings (for fd_gradient)."""
        from ccpy import Driver
        import ccpy.scf
        scf, _, _, _, nfrozen = CASES[self.name]

        def energy(mol):
            mf = getattr(ccpy.scf, scf)(mol, verbose=0, e_tol=1.0e-12, d_tol=1.0e-10)
            mf.kernel()
            d = _quiet(Driver.from_scf, mf, nfrozen=nfrozen)
            d.options.update({"energy_convergence": 1.0e-13, "amp_convergence": 1.0e-11})
            for step in ((d.run_cc, "ccsd"), (d.run_hbar, "ccsd"), (d.run_leftcc, "left_ccsd"), (d.run_ccp3, "crcc23")):
                _quiet(step[0], method=step[1])
            return d.system.reference_energy + d.correlation_energy + d.deltap3[0]["A"]
        return energy

    def _triples_blocks(self, vec):
        ds, o, nv = self._ds, self.space.nocc, self.space.nvir
        x3 = np.zeros((nv,) * 3 + (o,) * 3)
        for value, (a, b, c, i, j, k) in zip(vec, ds.triples):
            vir, occ = (a - o, b - o, c - o), (i, j, k)
            for pv in permutations(range(3)):
                for po in permutations(range(3)):
                    sign = _parity(pv) * _parity(po)
                    x3[tuple(vir[p] for p in pv) + tuple(occ[p] for p in po)] = sign * value
        s = self.system
        A, B = slice(0, s.nunoccupied_alpha), slice(s.nunoccupied_alpha, nv)
        I, J = slice(0, s.noccupied_alpha), slice(s.noccupied_alpha, o)
        return {"aaa": x3[A, A, A, I, I, I], "aab": x3[A, A, B, I, I, J],
                "abb": x3[A, B, B, I, J, J], "bbb": x3[B, B, B, J, J, J]}

    def compare(self, mine, name, verbose=True):
        """Max |mine - reference| per spin block for the reference quantity `name`."""
        if self.ref is None:
            raise RuntimeError(f"Case {self.name} was built with reference=False.")
        ref = self.ref[name]
        errors = {k: float(np.max(np.abs(np.asarray(_get(mine, k)) - ref[k]), initial=0.0)) for k in ref}
        if verbose:
            scale = max(float(np.max(np.abs(v), initial=0.0)) for v in ref.values())
            print(f"   {name} [{self.name}]  (largest reference element {scale:.2e})")
            for k, e in errors.items():
                print(f"      {k:<4} max |error| = {e:.2e}")
        return errors

    # -------------------------------------------------------------------------------
    # directional finite-difference / equation checks with ccpy's own code
    # -------------------------------------------------------------------------------
    def dot(self, x, y):
        """Sum over unique amplitudes of x*y (full antisymmetric blocks)."""
        return (np.sum(_get(x, "a") * _get(y, "a")) + np.sum(_get(x, "b") * _get(y, "b"))
                + 0.25 * np.sum(_get(x, "aa") * _get(y, "aa")) + np.sum(_get(x, "ab") * _get(y, "ab"))
                + 0.25 * np.sum(_get(x, "bb") * _get(y, "bb")))

    def random_direction(self):
        """Random antisymmetric singles/doubles direction in ccpy layout."""
        d = {}
        for k in SINGLES_DOUBLES:
            x = self.rng.standard_normal(_get(self.T, k).shape)
            if k in ("aa", "bb"):
                x = x - x.transpose(1, 0, 2, 3)
                x = x - x.transpose(0, 1, 3, 2)
            d[k] = x
        return d

    def _operator(self, template, blocks):
        op = copy.deepcopy(template)
        for k in SINGLES_DOUBLES:
            setattr(op, k, np.asfortranarray(np.array(_get(blocks, k), dtype=float)))
        return op

    def _shift(self, op, d, eps):
        return self._operator(op, {k: _get(op, k) + eps * d[k] for k in SINGLES_DOUBLES})

    def _hbar(self, T):
        from ccpy.hbar.hbar_ccsd import build_hbar_ccsd
        return _quiet(build_hbar_ccsd, T, copy.deepcopy(self.H0), False, self.system)

    def delta_A(self, T, L):
        """ccpy's (Fortran) delta_A at amplitudes T, L; Hbar rebuilt from T."""
        from ccpy.moments.crcc23 import calc_crcc23
        _, delta = _quiet(calc_crcc23, T, L, 0.0, self._hbar(T), self.fock, self.system, False)
        return delta["A"]

    def left_residual(self, T, L):
        """g = E_t + A^T lambda from ccpy's left-CCSD code, at T (Hbar rebuilt)."""
        from ccpy.left.left_ccsd import LH_fun
        H = self._hbar(T)
        LH = copy.deepcopy(L)
        _quiet(LH_fun, LH, L, T, H, False, self.system)
        return {"a": LH.a + H.a.ov.T, "b": LH.b + H.b.ov.T,
                "aa": LH.aa + H.aa.oovv.transpose(2, 3, 0, 1),
                "ab": LH.ab + H.ab.oovv.transpose(2, 3, 0, 1),
                "bb": LH.bb + H.bb.oovv.transpose(2, 3, 0, 1)}

    def A_times(self, w):
        """A w = <mu|(Hbar W)_C|0> (ccpy EOMCCSD HR)."""
        from ccpy.eomcc.eomccsd import HR
        W = self._operator(self.T, w)
        dW = copy.deepcopy(W)
        _quiet(HR, dW, W, self.T, self.Hbar, False, self.system)
        return {k: _get(dW, k) for k in SINGLES_DOUBLES}

    def AT_times(self, z):
        """A^T z = <0|(Z Hbar)_C|nu> (ccpy left-CCSD LH_fun)."""
        from ccpy.left.left_ccsd import LH_fun
        Z = self._operator(self.L, z)
        LZ = copy.deepcopy(Z)
        _quiet(LH_fun, LZ, Z, self.T, self.Hbar, False, self.system)
        return {k: _get(LZ, k) for k in SINGLES_DOUBLES}

    def _relative(self, a, b):
        return abs(a - b) / max(abs(b), 1.0e-14)

    def check_s_lambda(self, s_lambda, eps=1.0e-4):
        """Relative error of s_lambda . d against d delta_A(L + e d)/de (central)."""
        d = self.random_direction()
        fd = (self.delta_A(self.T, self._shift(self.L, d, eps))
              - self.delta_A(self.T, self._shift(self.L, d, -eps))) / (2 * eps)
        return self._relative(self.dot(s_lambda, d), fd)

    def check_s_t(self, s_t, eps=1.0e-4):
        """Relative error of s_t . d against d delta_A(T + e d)/de (central, Hbar rebuilt)."""
        d = self.random_direction()
        fd = (self.delta_A(self._shift(self.T, d, eps), self.L)
              - self.delta_A(self._shift(self.T, d, -eps), self.L)) / (2 * eps)
        return self._relative(self.dot(s_t, d), fd)

    def check_BTw(self, BTw, w, eps=1.0e-5):
        """Max |B^T w - d g(T + e w)/de| over all blocks (relative to the largest element)."""
        gp = self.left_residual(self._shift(self.T, w, eps), self.L)
        gm = self.left_residual(self._shift(self.T, w, -eps), self.L)
        fd = {k: (gp[k] - gm[k]) / (2 * eps) for k in SINGLES_DOUBLES}
        scale = max(np.abs(fd[k]).max() for k in SINGLES_DOUBLES)
        return max(np.abs(_get(BTw, k) - fd[k]).max() for k in SINGLES_DOUBLES) / scale

    def check_w(self, w, s_lambda):
        """|A w + s_lambda| / |s_lambda| (max norms)."""
        Aw = self.A_times(w)
        scale = max(np.abs(_get(s_lambda, k)).max() for k in SINGLES_DOUBLES)
        return max(np.abs(Aw[k] + _get(s_lambda, k)).max() for k in SINGLES_DOUBLES) / scale

    def check_z(self, z, s_t, BTw):
        """|A^T z + s_t + B^T w| / |s_t + B^T w| (max norms)."""
        ATz = self.AT_times(z)
        rhs = {k: _get(s_t, k) + _get(BTw, k) for k in SINGLES_DOUBLES}
        scale = max(np.abs(rhs[k]).max() for k in SINGLES_DOUBLES)
        return max(np.abs(ATz[k] + rhs[k]).max() for k in SINGLES_DOUBLES) / scale


def _parity(p):
    p, sign = list(p), 1
    for i in range(len(p)):
        while p[i] != i:
            j = p[i]
            p[i], p[j] = p[j], p[i]
            sign = -sign
    return sign
