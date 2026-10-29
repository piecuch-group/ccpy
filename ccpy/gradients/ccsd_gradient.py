"""Analytic nuclear gradient of closed-shell (RHF-reference) CCSD, with or without frozen
core, in two flavors:

* density-fitted: the SCF and CCSD use the DF integrals of the ``Molecule`` and the
  two-electron derivatives come from the 3-center (P|mn)' and 2-center (P|Q)' integrals;
* conventional: the ``Molecule`` uses a (tight) Cholesky factorization of the exact
  four-index integrals (``cholesky_tei=True``) and the two-electron derivatives come from
  the four-center integrals (mn|rs)'.

Theory
------
With the unrelaxed CCSD densities D, G (ccpy.density.ccsd_rdm_rhf),

    E = V_nn + sum_pq h_pq D_pq + 1/2 sum_pqrs (pq|rs) G_pqrs .

E is not stationary with respect to orbital rotations, so the canonical RHF conditions
F_ai = 0 (all occupied i, virtual a) and, with frozen core, F_jc = 0 (active occupied j,
core c) are added with multipliers z (the Z-vector). With the orbital connection
U = -1/2 S^x,

    dE/dx = V_nn^x + sum D^rel h^x + 1/2 sum G^eff (pq|rs)^x - sum W S^x ,

where D^rel = D + Z, G^eff adds the separable Z (x) D_HF terms, and W collects the
generalized-Fock and Z-vector orbital-connection terms.
"""

import time

import numpy as np
import scipy.sparse.linalg as spla

from ccpy.density.ccsd_rdm_rhf import BLOCKS, calc_ccsd_rdms_rhf
from ccpy.integrals import integrals


def _separable_blocks(Z, nocc, nmo):
    """Blocks of 2 (Z x P + P x Z) - (Z_ps P_qr + P_ps Z_qr), P the occupied projector.
    Every term carries an occupied Kronecker delta, so the vvvv block vanishes."""
    sl = {"o": slice(0, nocc), "v": slice(nocc, nmo)}
    P = {"o": np.eye(nocc), "v": None}
    blocks = {}
    for key in BLOCKS:
        p, q, r, s = (sl[c] for c in key)
        blk = 0.0
        if key[2:] == "oo":
            blk = blk + 2.0 * np.einsum("pq,rs->pqrs", Z[p, q], P["o"])
        if key[:2] == "oo":
            blk = blk + 2.0 * np.einsum("pq,rs->pqrs", P["o"], Z[r, s])
        if key[1:3] == "oo":
            blk = blk - np.einsum("ps,qr->pqrs", Z[p, s], P["o"])
        if key[0] == "o" and key[3] == "o":
            blk = blk - np.einsum("ps,qr->pqrs", P["o"], Z[q, r])
        if not np.isscalar(blk):
            blocks[key] = blk
    return blocks


class _MOTwoElectron:
    """(pq|rs) = sum_x R_xpq R_xrs in the MO basis (DF or Cholesky vectors)."""

    def __init__(self, R):
        self.R = R

    def J(self, Z):
        return np.einsum("xpq,x->pq", self.R, np.einsum("xrs,rs->x", self.R, Z), optimize=True)

    def K(self, Z):
        RZ = np.einsum("xpr,rs->xps", self.R, Z, optimize=True)
        return np.einsum("xps,xsq->pq", RZ, self.R, optimize=True)


def ccsd_gradient(driver, meanfield=None, zvector_tol=1.0e-10, verbose=True):
    """Analytic RHF-CCSD nuclear gradient, shape (natoms, 3), in Eh/bohr.

    Parameters
    ----------
    driver : Driver
        Built with ``Driver.from_scf`` from a ccpy.scf.RHF object, after
        ``run_cc("ccsd")``, ``run_hbar("ccsd")`` and ``run_leftcc("left_ccsd")``.
    meanfield : ccpy.scf.RHF, optional
        Defaults to the SCF object stored on the driver by ``from_scf``.
    zvector_tol : float
        Convergence threshold of the Z-vector (conjugate-gradient) solver.
    """
    from ccpy.scf.rhf import RHF

    t0 = time.perf_counter()
    mf = meanfield if meanfield is not None else getattr(driver, "scf", None)
    if not isinstance(mf, RHF):
        raise TypeError("ccsd_gradient needs the ccpy.scf.RHF reference (use Driver.from_scf).")
    system = driver.system
    if system.ndelete != 0:
        raise NotImplementedError("CCSD gradients with deleted virtual orbitals are not implemented.")
    if driver.T is None or driver.L[0] is None:
        raise RuntimeError("Run CCSD, the CCSD HBar, and left-CCSD before the gradient.")
    if driver.T.order != 2:
        raise NotImplementedError("Only CCSD gradients are implemented.")

    D1, D2 = calc_ccsd_rdms_rhf(driver.T, driver.L[0], system)
    return relaxed_gradient(mf, system.nfrozen, D1, D2, zvector_tol=zvector_tol,
                            verbose=verbose, title="CCSD Analytic Gradient", t0=t0)


def relaxed_gradient(mf, nfrozen, D1, D2, zvector_tol=1.0e-10, verbose=True,
                     title="Analytic Gradient", t0=None):
    """Nuclear gradient of E = V_nn + sum h D1 + 1/2 sum (pq|rs) D2 for unrelaxed densities
    of an energy that is invariant to rotations within the frozen core, within the correlated
    occupied space, and within the virtual space (CCSD, CR-CC(2,3)A, ...): solves the orbital
    Z-vector (occupied-virtual and frozen-core/active), forms the relaxed and energy-weighted
    densities, and contracts them with the AO derivative integrals (density-fitted or
    four-center, following the Molecule).

    Parameters
    ----------
    mf : ccpy.scf.RHF
    nfrozen : int
    D1 : ndarray, shape (nmo, nmo)
        Spin-traced one-particle density, E1 = sum_pq h_pq D1_qp (symmetrized here).
    D2 : Density2
        Spin-traced two-particle density in chemists' order (no stored vvvv block).
    """
    t0 = time.perf_counter() if t0 is None else t0
    mol = mf.mol
    use_df = not mol.cholesky_tei
    C = mf.mo_coeff
    eps = mf.mo_energy
    nmo = C.shape[1]
    nocc = mf.na
    nfz = nfrozen
    occ, vir = slice(0, nocc), slice(nocc, nmo)
    core, act_occ = slice(0, nfz), slice(nfz, nocc)

    # ---------------------------------------------------------------- #
    # MO integrals
    # ---------------------------------------------------------------- #
    D1 = 0.5 * (D1 + D1.T)
    h = C.T @ mf.get_hcore() @ C
    R = mol.df.mo_cholesky_vectors(C)
    ints = _MOTwoElectron(R)

    # Y_xpq = sum_rs G_pqrs R_xrs (the vvvv block of G is built in 3-virtual slices)
    Y = D2.symmetrized().contract(R)
    # generalized Fock F_tp = sum_q h_tq D_pq + sum_qrs (tq|rs) G_pqrs
    F = h @ D1 + np.einsum("xtq,xpq->tp", R, Y, optimize=True)
    g = 2.0 * (F - F.T)  # dE/dkappa_tp for the rotation t <- p

    if verbose:
        # the CCSD energy is invariant to rotations within these spaces
        redundant = max(
            np.abs(g[act_occ, act_occ]).max(initial=0.0),
            np.abs(g[vir, vir]).max(initial=0.0),
            np.abs(g[core, core]).max(initial=0.0),
        )
        print(f"\n   {title}")
        print("   " + "-" * len(title))
        print(f"   Two-electron integrals: {'density fitting' if use_df else 'four-center (conventional)'}")
        print(f"   Orbital gradient in redundant blocks (should vanish): {redundant:.2e}")

    # ---------------------------------------------------------------- #
    # Z-vector: core / active-occupied, then occupied / virtual
    # ---------------------------------------------------------------- #
    z_fc = np.zeros((nocc - nfz, nfz))
    if nfz > 0:
        z_fc = -g[act_occ, core] / (eps[act_occ, None] - eps[None, core])

    def hf_response(Zs):
        """[2J(Zs) - K(Zs)] for a symmetric MO matrix Zs."""
        return 2.0 * ints.J(Zs) - ints.K(Zs)

    rhs = -g[vir, occ].copy()
    if nfz > 0:
        Ws = np.zeros((nmo, nmo))
        Ws[act_occ, core] = z_fc
        Ws += Ws.T
        rhs -= hf_response(Ws)[vir, occ]

    eps_ai = eps[vir, None] - eps[None, occ]
    nv = nmo - nocc

    def A_matvec(zflat):
        z = zflat.reshape(nv, nocc)
        Zs = np.zeros((nmo, nmo))
        Zs[vir, occ] = z
        Zs += Zs.T
        return (eps_ai * z + hf_response(Zs)[vir, occ]).ravel()

    A = spla.LinearOperator((nv * nocc,) * 2, matvec=A_matvec, dtype=float)
    M = spla.LinearOperator((nv * nocc,) * 2, matvec=lambda x: x / eps_ai.ravel(), dtype=float)
    z_ov, info = spla.cg(A, rhs.ravel(), rtol=zvector_tol, atol=0.0, M=M, maxiter=200)
    if info != 0:
        raise RuntimeError(f"The Z-vector equations did not converge (cg info = {info}).")
    z_ov = z_ov.reshape(nv, nocc)

    # symmetric MO matrix Z with Z_pq = Z_qp = z_P / 2 for each condition P
    Z = np.zeros((nmo, nmo))
    Z[vir, occ] = 0.5 * z_ov
    Z[act_occ, core] = 0.5 * z_fc
    Z += Z.T

    # ---------------------------------------------------------------- #
    # relaxed and energy-weighted densities (MO)
    # ---------------------------------------------------------------- #
    D_rel = D1 + Z

    # G_eff = G + 2 (Z x P + P x Z) - (Z_ps P_qr + P_ps Z_qr), P = occupied projector
    P = np.zeros(nmo)
    P[occ] = 1.0
    G_eff = D2.add_blocks(_separable_blocks(Z, nocc, nmo)).symmetrized()

    M_Z = hf_response(Z)
    W = 0.5 * (F + F.T)
    W += Z * 0.5 * (eps[:, None] + eps[None, :])
    occ_mask = P[:, None] + P[None, :]  # 2 for occ-occ, 1 for occ-vir
    W += 0.5 * occ_mask * M_Z

    # ---------------------------------------------------------------- #
    # contract with the AO derivative integrals
    # ---------------------------------------------------------------- #
    D_ao = C @ D_rel @ C.T
    W_ao = C @ W @ C.T
    natoms = mol.natoms
    grad = integrals.nuclear_repulsion_deriv(mol)
    grad += integrals.hcore_deriv(mol, D_ao).reshape(natoms, 3)
    grad -= integrals.overlap_deriv(mol, W_ao).reshape(natoms, 3)

    if use_df:
        df = mol.df
        X = df.Mm12  # B = X (Q|mn), X = L^{-1}, M = L L^T
        Y_eff = G_eff.contract(R)
        Y_ao = np.einsum("mp,xpq,nq->xmn", C, Y_eff, C, optimize=True)
        W3 = np.einsum("QP,Qmn->Pmn", X, Y_ao, optimize=True)
        T_QQ = np.einsum("xpq,ypq->xy", R, Y_eff, optimize=True)
        W2 = -0.5 * X.T @ T_QQ @ X
        grad += integrals.coulomb_3c_deriv(mol, np.ascontiguousarray(W3)).reshape(natoms, 3)
        grad += integrals.coulomb_2c_deriv(mol, np.ascontiguousarray(W2)).reshape(natoms, 3)
    else:
        G_ao = G_eff.to_ao(C)
        grad += 0.5 * integrals.coulomb_4c_deriv(mol, G_ao, symmetrize=False).reshape(natoms, 3)

    if verbose:
        print(f"   Z-vector converged; |z_ov| = {np.linalg.norm(z_ov):.3e}, |z_core| = {np.linalg.norm(z_fc):.3e}")
        print(f"   Gradient time: {time.perf_counter() - t0:.2f} s")
        print("\n   Atom          dE/dx            dE/dy            dE/dz")
        from ccpy.integrals.atom_data import Z_TO_ATOM_SYMBOL
        for (Zn, _), gA in zip(mol.atoms, grad):
            print(f"   {Z_TO_ATOM_SYMBOL[Zn]:<4}{gA[0]:>17.10f}{gA[1]:>17.10f}{gA[2]:>17.10f}")
        print("")
    return grad
