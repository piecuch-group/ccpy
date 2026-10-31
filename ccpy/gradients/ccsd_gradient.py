"""Analytic nuclear gradient of closed-shell (RHF-reference) CCSD, with or without frozen
core, factorized with density fitting or Cholesky vectors: two-electron integrals and densities are
only ever handled through the three-index vectors B_x[p, q] (<pq|rs> = sum_x B_x[p,r] B_x[q,s])
and nothing with four virtual indices is stored.

Theory
------
With the CCS-transformed Hamiltonian H(T1) = e^{-T1} H e^{T1} (ccpy.hbar.hbar_ccs_chol: again
one- plus two-body, with T1-transformed Cholesky vectors), the CCSD Lagrangian is that of
T1 = 0,

    L = E_ref(H(T1)) + E(T2; H(T1)) + sum_mu lambda_mu R_mu(T2; H(T1)),

so the densities are those of ccpy.density.ccsd_rdm (dL/d(blocks of H(T1))). build_DY turns
them into the bare-frame densities over all MOs (frozen core included),

    D1 = dL/dh      (n x n, conjugate to the core Hamiltonian),
    Y_x = dL/dB_x   (x, n, n), i.e. Y_x[p, r] = sum_qs D2[p,q,r,s] B_x[q,s],

using that the T1 transformation B(T1) = (1 - t1) B (1 + t1) acts index by index:
D -> (1 - t1)^T D (1 + t1)^T, and the reference projector P becomes P + t1^T.

E is not stationary with respect to orbital rotations, so the canonical RHF conditions
F_ai = 0 (all occupied i, virtual a) and, with frozen core, F_jc = 0 (active occupied j,
core c) are added with multipliers z (the Z-vector). With U = -1/2 S^x,

    dE/dx = V_nn^x + sum D^rel h^x + (3-center and 2-center terms of Y^eff) - sum W S^x .

relaxed_gradient also adds the canonical conditions within the active occupied and the
virtual spaces, which energies with orbital-energy denominators (CR-CC(2,3)_A) need.
"""

import time

import numpy as np
import scipy.sparse.linalg as spla

from ccpy.density.ccsd_rdm import calc_ccsd_rdm, calc_rdm2_vvvv_slice
from ccpy.hbar.hbar_ccs_chol import build_hbar_ccs_chol
from ccpy.integrals import integrals

BLOCKS = [p + q + r + s for p in "ov" for q in "ov" for r in "ov" for s in "ov"]


def ccsd_gradient(driver, meanfield=None, zvector_tol=1.0e-10, verbose=True):
    """Analytic RHF-CCSD nuclear gradient, shape (natoms, 3), in Eh/bohr.

    Parameters
    ----------
    driver : Driver
        Built with ``Driver.from_scf`` from a ccpy.scf.RHF object (density fitting or Cholesky), after
        ``run_cc``, ``run_hbar`` and ``run_leftcc`` (ccsd or ccsd_chol variants).
    meanfield : ccpy.scf.RHF, optional
        Defaults to the SCF object stored on the driver by ``from_scf``.
    zvector_tol : float
        Convergence threshold of the Z-vector (conjugate-gradient) solver.
    """
    mf = meanfield if meanfield is not None else getattr(driver, "scf", None)
    if driver.T is None or driver.L[0] is None:
        raise RuntimeError("Run CCSD, the CCSD HBar, and left-CCSD before the gradient.")
    return ccsd_gradient_from_amplitudes(mf, driver.system, driver.T, driver.L[0], zvector_tol=zvector_tol,
                                         verbose=verbose)


def ccsd_gradient_from_amplitudes(meanfield, system, T, L, H0=None, rdm=None, zvector_tol=1.0e-10, verbose=True):
    """Analytic RHF-CCSD nuclear gradient from converged CCSD (T) and left-CCSD (L) amplitudes.

    Parameters
    ----------
    meanfield : ccpy.scf.RHF
    system : System of the correlated calculation
    H0 : bare DF Hamiltonian (built from meanfield if not given)
    rdm : (rdm1, rdm2) of ccpy.density.ccsd_rdm.calc_ccsd_rdm(T, L, system), if available
    """
    from ccpy.interfaces.scf_tools import load_scf_integrals
    from ccpy.scf.rhf import RHF

    t0 = time.perf_counter()
    if not isinstance(meanfield, RHF):
        raise TypeError("The CCSD gradient needs a ccpy.scf.RHF reference.")
    if T.order != 2:
        raise NotImplementedError("Only CCSD gradients are implemented.")
    if H0 is None:
        _, H0 = load_scf_integrals(meanfield, system.nfrozen, system.ndelete, use_cholesky=True)
    H = build_hbar_ccs_chol(T, H0, system)
    mo = get_mo_integrals(meanfield, system)
    rdm1, rdm2 = rdm if rdm is not None else calc_ccsd_rdm(T, L, system)
    D1, Y = build_DY(rdm1, rdm2, lambda a, s: calc_rdm2_vvvv_slice(L, T, a, s),
                     H.chol, T, mo["B"], mo, system, reference=True)
    return relaxed_gradient(meanfield, system.nfrozen, D1["a"] + D1["b"], Y["a"] + Y["b"],
                            zvector_tol=zvector_tol, verbose=verbose, title="CCSD Analytic Gradient", t0=t0)


def get_mo_integrals(meanfield, system):
    """Bare MO core Hamiltonian h, Cholesky vectors B (all MOs, frozen core included),
    occupied projectors P (keyed by spin) and the correlated orbital slices."""
    if system.ndelete != 0:
        raise NotImplementedError("Gradients with deleted virtual orbitals are not implemented.")
    nfz = system.nfrozen
    C = meanfield.C if len(meanfield.C) == 2 else [meanfield.C[0], meanfield.C[0]]
    nmo = C[0].shape[1]
    nocc = {"a": nfz + system.noccupied_alpha, "b": nfz + system.noccupied_beta}
    end = nfz + system.norbitals
    mo = {"nmo": nmo, "nocc": nocc, "h": {}, "B": {}, "P": {},
          "slices": {s: {"o": slice(nfz, nocc[s]), "v": slice(nocc[s], end)} for s in ("a", "b")}}
    hcore = meanfield.get_hcore()
    for k, s in enumerate(("a", "b")):
        mo["h"][s] = C[k].T @ hcore @ C[k]
        mo["B"][s] = mo["B"]["a"] if (s == "b" and len(meanfield.C) == 1) else meanfield.mol.df.mo_cholesky_vectors(C[k])
        mo["P"][s] = np.diag((np.arange(nmo) < nocc[s]).astype(float))
    return mo


def build_DY(rdm1, rdm2, vvvv_slice, Bt, T, B, mo, system, reference=False):
    """Bare-frame (D1, Y) over all MOs of a functional F(H(T1)) whose densities rdm1, rdm2
    (ccpy.density) are conjugate to the stored blocks of H(T1) (rdm1 to the normal-ordered Fock
    blocks f = h + sum_i <pi||qi>, rdm2 to H.aa.pqrs = <pq||rs>, H.ab.pqrs = <pq|rs>, ...).

    Bt : Cholesky vectors of the Hamiltonian the densities refer to (H.chol of
         ccpy.hbar.hbar_ccs_chol, or the bare H0.chol when T is None), contracted with rdm2 and
         vvvv_slice(a, spin) (vvvv densities one first index at a time, or None)
    T  : T1 of the transformation (None for a functional of the bare H)
    B  : bare Cholesky vectors {spin: (x, nmo, nmo)} for the separable (and reference) terms
    reference=True adds the reference determinant E_ref = sum_i h_ii + 1/2 sum_ij <ij||ij>.
    Returns D1 = dF/dh and Y = dF/dB, per spin."""
    sl, nmo = mo["slices"], mo["nmo"]
    naux = B["a"].shape[0]
    Y = {s: np.zeros((naux, nmo, nmo)) for s in ("a", "b")}

    # two-body: Y(T1) of the rdm2 blocks, contracted with the transformed vectors
    nvir = {s: getattr(Bt, s).vv.shape[1] for s in ("a", "b")}
    nocc = {s: getattr(Bt, s).oo.shape[1] for s in ("a", "b")}
    n = {s: nocc[s] + nvir[s] for s in ("a", "b")}
    cs = {s: {"o": slice(0, nocc[s]), "v": slice(nocc[s], n[s])} for s in ("a", "b")}
    Yt, Bc = {}, {}
    for s in ("a", "b"):
        Bc[s] = np.zeros((naux, n[s], n[s]))
        for blk in ("oo", "ov", "vo", "vv"):
            Bc[s][:, cs[s][blk[0]], cs[s][blk[1]]] = getattr(getattr(Bt, s), blk)
        Yt[s] = np.zeros((naux, n[s], n[s]))
    for s in ("a", "b"):
        ss = getattr(rdm2, s + s)
        for blk in BLOCKS:
            G = getattr(ss, blk, None)
            if G is None or np.isscalar(G):
                continue
            p, q, r, t = (cs[s][c] for c in blk)
            Yt[s][:, p, r] += np.einsum("pqrs,xqs->xpr", G, Bc[s][:, q, t], optimize=True)
            Yt[s][:, q, t] += np.einsum("pqrs,xpr->xqs", G, Bc[s][:, p, r], optimize=True)
            Yt[s][:, p, t] -= np.einsum("pqrs,xqr->xps", G, Bc[s][:, q, r], optimize=True)
            Yt[s][:, q, r] -= np.einsum("pqrs,xps->xqr", G, Bc[s][:, p, t], optimize=True)
    for blk in BLOCKS:
        G = getattr(rdm2.ab, blk, None)
        if G is None or np.isscalar(G):
            continue
        p, r = cs["a"][blk[0]], cs["a"][blk[2]]
        q, t = cs["b"][blk[1]], cs["b"][blk[3]]
        Yt["a"][:, p, r] += np.einsum("pqrs,xqs->xpr", G, Bc["b"][:, q, t], optimize=True)
        Yt["b"][:, q, t] += np.einsum("pqrs,xpr->xqs", G, Bc["a"][:, p, r], optimize=True)
    if vvvv_slice is not None:
        va, vb = cs["a"]["v"], cs["b"]["v"]
        Ba, Bb = Bt.a.vv, Bt.b.vv
        for a in range(nvir["a"]):
            G_aa, G_ab = vvvv_slice(a, "a")
            Yt["a"][:, va.start + a, va] += np.einsum("bcd,xbd->xc", G_aa, Ba, optimize=True)
            Yt["a"][:, va, va] += np.einsum("bcd,xc->xbd", G_aa, Ba[:, a, :], optimize=True)
            Yt["a"][:, va.start + a, va] -= np.einsum("bcd,xbc->xd", G_aa, Ba, optimize=True)
            Yt["a"][:, va, va] -= np.einsum("bcd,xd->xbc", G_aa, Ba[:, a, :], optimize=True)
            Yt["a"][:, va.start + a, va] += np.einsum("bcd,xbd->xc", G_ab, Bb, optimize=True)
            Yt["b"][:, vb, vb] += np.einsum("bcd,xc->xbd", G_ab, Ba[:, a, :], optimize=True)
        for a in range(nvir["b"]):
            G_bb = vvvv_slice(a, "b")
            Yt["b"][:, vb.start + a, vb] += np.einsum("bcd,xbd->xc", G_bb, Bb, optimize=True)
            Yt["b"][:, vb, vb] += np.einsum("bcd,xc->xbd", G_bb, Bb[:, a, :], optimize=True)
            Yt["b"][:, vb.start + a, vb] -= np.einsum("bcd,xbc->xd", G_bb, Bb, optimize=True)
            Yt["b"][:, vb, vb] -= np.einsum("bcd,xd->xbc", G_bb, Bb[:, a, :], optimize=True)

    # back to the bare frame: (1 - t1)^T . (1 + t1)^T on the correlated orbitals
    corr = {s: slice(sl[s]["o"].start, sl[s]["v"].stop) for s in ("a", "b")}
    t1 = {s: np.zeros((n[s], n[s])) for s in ("a", "b")}
    if T is not None:
        t1["a"][cs["a"]["v"], cs["a"]["o"]] = T.a
        t1["b"][cs["b"]["v"], cs["b"]["o"]] = T.b
    gamma = {}
    for s in ("a", "b"):
        X, Yd = np.eye(n[s]) - t1[s], np.eye(n[s]) + t1[s]
        Y[s][:, corr[s], corr[s]] = np.einsum("qp,xqr,sr->xps", X, Yt[s], Yd, optimize=True)
        g = np.zeros((n[s], n[s]))
        for blk in ("oo", "ov", "vo", "vv"):
            x = getattr(getattr(rdm1, s), blk)
            if x is not None:
                g[cs[s][blk[0]], cs[s][blk[1]]] += x
        gamma[s] = np.zeros((nmo, nmo))
        gamma[s][corr[s], corr[s]] = X.T @ g @ Yd.T

    # one-body and separable terms (the reference projector of H(T1) is P + t1^T in the bare frame)
    P = {}
    for s in ("a", "b"):
        P[s] = mo["P"][s].copy()
        P[s][corr[s], corr[s]] += t1[s].T
    D1 = {s: gamma[s] + (P[s] if reference else 0.0) for s in ("a", "b")}
    for g, scale in ((gamma, 1.0),) + (((P, 0.5),) if reference else ()):
        # D2_ss = P_qs g_pr - P_qr g_ps - P_ps g_qr + P_pr g_qs,  D2_st = P^t_qs g^s_pr + P^s_pr g^t_qs
        PB = sum(np.einsum("qs,xqs->x", P[u], B[u], optimize=True) for u in ("a", "b"))
        gB = sum(np.einsum("qs,xqs->x", g[u], B[u], optimize=True) for u in ("a", "b"))
        for s in ("a", "b"):
            Y[s] += scale * (np.einsum("pr,x->xpr", g[s], PB, optimize=True)
                             + np.einsum("pr,x->xpr", P[s], gB, optimize=True)
                             - np.einsum("ps,xqs,qr->xpr", g[s], B[s], P[s], optimize=True)
                             - np.einsum("ps,xqs,qr->xpr", P[s], B[s], g[s], optimize=True))
    return D1, Y


def rotation_gradient(D1, Y, h, B, mo):
    """dF/dt_i^a of a functional F(h, B) with dF/dh = D1 and dF/dB = Y, where t_i^a rotates
    the Hamiltonian as H -> H + e [H, a+ i] ([M, E] = M E - E M for h and each B_x):
    (h^T D1 - D1 h^T)[a, i] + sum_x (B_x^T Y_x - Y_x B_x^T)[a, i], per spin."""
    out = {}
    for s in ("a", "b"):
        o, v = mo["slices"][s]["o"], mo["slices"][s]["v"]
        out[s] = (h[s][:, v].T @ D1[s][:, o] - D1[s][v, :] @ h[s][o, :].T
                  + np.einsum("xqa,xqi->ai", B[s][:, :, v], Y[s][:, :, o], optimize=True)
                  - np.einsum("xaq,xiq->ai", Y[s][:, v, :], B[s][:, o, :], optimize=True))
    return out


def relaxed_gradient(mf, nfrozen, D1, Y, zvector_tol=1.0e-10, verbose=True,
                     title="Analytic Gradient", t0=None):
    """Nuclear gradient of E = V_nn + sum h D1 + E2 from unrelaxed densities, with
    dE2/dR_x = Y_x (R = mol.df.mo_cholesky_vectors(C)): solves the orbital Z-vector
    (occupied-virtual and frozen-core/active, plus the canonical conditions within the
    active occupied and the virtual spaces), forms the relaxed and energy-weighted densities,
    and contracts them with the derivative integrals of the three-index vectors: 3- and
    2-center integrals for density fitting, and the four-center integrals of the pivot pairs
    for a Cholesky decomposition (pivots held fixed; the gradient is exact for that energy).

    Parameters
    ----------
    mf : ccpy.scf.RHF (density-fitted or Cholesky Molecule)
    nfrozen : int
    D1 : ndarray, shape (nmo, nmo)
        Spin-traced one-particle density D1[p,q] = <p+ q>, E1 = sum_pq h_pq D1[p,q].
    Y : ndarray, shape (naux, nmo, nmo)
        Spin-traced Y_x[p, r] = sum_qs D2[p,q,r,s] R_x[q,s] (D2[p,q,r,s] = <p+ q+ s r>).
    """
    t0 = time.perf_counter() if t0 is None else t0
    mol = mf.mol
    C = mf.mo_coeff
    eps = mf.mo_energy
    nmo = C.shape[1]
    nocc = mf.na
    nfz = nfrozen
    occ, vir = slice(0, nocc), slice(nocc, nmo)
    core, act_occ = slice(0, nfz), slice(nfz, nocc)

    D1 = 0.5 * (D1 + D1.T)
    h = C.T @ mf.get_hcore() @ C
    R = mol.df.mo_cholesky_vectors(C)
    # only the symmetric part of each Y_x is seen by the symmetric R_x
    Y = 0.5 * (Y + Y.transpose(0, 2, 1))
    # generalized Fock F_tp = sum_q h_tq D_pq + sum_qrs (tq|rs) G_pqrs
    F = h @ D1 + np.einsum("xtq,xpq->tp", R, Y, optimize=True)
    g = 2.0 * (F - F.T)  # dE/dkappa_tp for the rotation t <- p

    # canonical-condition multipliers within the active occupied and the virtual spaces,
    # z_pq = -g_pq / (eps_p - eps_q) for p > q. They vanish for CCSD (invariant to these
    # rotations) but not for energies with orbital-energy denominators such as CR-CC(2,3)_A.
    # Rotations within a degenerate pair leave such energies invariant, so g_pq = 0 there.
    z_int = np.zeros((nmo, nmo))
    degenerate_g = 0.0
    for space in (act_occ, vir):
        idx = np.arange(nmo)[space]
        de = eps[idx, None] - eps[None, idx]
        lower = np.tril(np.ones_like(de, dtype=bool), -1)
        nondeg = lower & (np.abs(de) > 1.0e-8)
        block = np.zeros_like(de)
        block[nondeg] = -g[np.ix_(idx, idx)][nondeg] / de[nondeg]
        z_int[np.ix_(idx, idx)] = block
        degenerate_g = max(degenerate_g, np.abs(g[np.ix_(idx, idx)][lower & ~nondeg]).max(initial=0.0))

    if verbose:
        print(f"\n   {title}")
        print("   " + "-" * len(title))
        print(f"   Orbital gradient within the frozen core (should vanish): {np.abs(g[core, core]).max(initial=0.0):.2e}")
        print(f"   Occupied-occupied / virtual-virtual multipliers: |z| = {np.linalg.norm(z_int):.3e}")
    if degenerate_g > 1.0e-6:
        print(f"   WARNING: orbital gradient {degenerate_g:.2e} between degenerate orbitals is not zero; "
              "the energy is not invariant within a degenerate subspace.")

    # Z-vector: core / active-occupied, then occupied / virtual
    z_fc = np.zeros((nocc - nfz, nfz))
    if nfz > 0:
        z_fc = -g[act_occ, core] / (eps[act_occ, None] - eps[None, core])

    def hf_response(Zs):
        """[2J(Zs) - K(Zs)] for a symmetric MO matrix Zs."""
        J = np.einsum("xpq,x->pq", R, np.einsum("xrs,rs->x", R, Zs, optimize=True), optimize=True)
        K = np.einsum("xps,xsq->pq", np.einsum("xpr,rs->xps", R, Zs, optimize=True), R, optimize=True)
        return 2.0 * J - K

    rhs = -g[vir, occ].copy()
    Ws = z_int.copy()
    if nfz > 0:
        Ws[act_occ, core] = z_fc
    if np.any(Ws):
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
    Z += 0.5 * z_int
    Z += Z.T

    # relaxed and energy-weighted densities (MO)
    D_rel = D1 + Z
    P = np.zeros((nmo, nmo))
    P[occ, occ] = np.eye(nocc)
    W = 0.5 * (F + F.T)
    W += Z * 0.5 * (eps[:, None] + eps[None, :])
    occ_mask = np.diag(P)[:, None] + np.diag(P)[None, :]  # 2 for occ-occ, 1 for occ-vir
    W += 0.5 * occ_mask * hf_response(Z)

    # Y of the separable Z terms 2 (Z_pr P_qs + P_pr Z_qs) - (Z_ps P_qr + P_ps Z_qr)
    Y_eff = (Y + 2.0 * np.einsum("pr,x->xpr", Z, np.einsum("xqq->x", R[:, occ, occ]), optimize=True)
             + 2.0 * np.einsum("pr,x->xpr", P, np.einsum("qs,xqs->x", Z, R, optimize=True), optimize=True)
             - Z @ R @ P - P @ R @ Z)

    # contract with the AO derivative integrals
    natoms = mol.natoms
    grad = integrals.nuclear_repulsion_deriv(mol)
    grad += integrals.hcore_deriv(mol, C @ D_rel @ C.T).reshape(natoms, 3)
    grad -= integrals.overlap_deriv(mol, C @ W @ C.T).reshape(natoms, 3)
    # B = X (Q|mn), X = L^{-1}, M = (Q|Q') = L L^T, for density fitting and for the Cholesky
    # vectors alike (Q: auxiliary function or Cholesky pivot pair); only the derivative integrals
    # of (Q|mn) and (Q|Q') differ.
    X = mol.df.Mm12
    Y_ao = np.einsum("mp,xpq,nq->xmn", C, Y_eff, C, optimize=True)
    W3 = np.ascontiguousarray(np.einsum("QP,Qmn->Pmn", X, Y_ao, optimize=True))
    W2 = np.ascontiguousarray(-0.5 * X.T @ np.einsum("xpq,ypq->xy", R, Y_eff, optimize=True) @ X)
    if mol.cholesky_tei:
        grad += integrals.cholesky_3c_deriv(mol, W3, mol.df.pivots).reshape(natoms, 3)
        grad += integrals.cholesky_2c_deriv(mol, W2, mol.df.pivots).reshape(natoms, 3)
    else:
        grad += integrals.coulomb_3c_deriv(mol, W3).reshape(natoms, 3)
        grad += integrals.coulomb_2c_deriv(mol, W2).reshape(natoms, 3)

    if verbose:
        print(f"   Z-vector converged; |z_ov| = {np.linalg.norm(z_ov):.3e}, |z_core| = {np.linalg.norm(z_fc):.3e}")
        print(f"   Gradient time: {time.perf_counter() - t0:.2f} s")
        print("\n   Atom          dE/dx            dE/dy            dE/dz")
        from ccpy.integrals.atom_data import Z_TO_ATOM_SYMBOL
        for (Zn, _), gA in zip(mol.atoms, grad):
            print(f"   {Z_TO_ATOM_SYMBOL[Zn]:<4}{gA[0]:>17.10f}{gA[1]:>17.10f}{gA[2]:>17.10f}")
        print("")
    return grad
