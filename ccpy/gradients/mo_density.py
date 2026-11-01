"""Amplitude-relaxed densities of CC functionals in the bare MO frame, per spin.

A CC gradient goes through three stages:

  1. amplitude-relaxed densities (this module + the method's ccpy.density modules): the
     derivatives of the CC Lagrangian (amplitude equations included) with respect to the bare
     MO Hamiltonian, D1[s] = dL/dh[s] and Y[s] = dL/dB[s], separately for each spin s;
  2. orbital relaxation (ccpy.gradients.orbital_response): the Z-vector of the reference
     conditions, giving the relaxed D[s], Y[s] and the energy-weighted W[s];
  3. contraction with the derivative integrals (ccpy.gradients.derivative_integrals).

The densities of ccpy.density are conjugate to the stored blocks of H(T1) =
e^{-T1} H e^{T1} (ccpy.hbar.hbar_ccs_chol); build_DY takes them to the bare frame over all MOs
(frozen core included). Two-body densities only ever appear contracted with the three-index
vectors, Y_x[p, r] = sum_qs D2[p,q,r,s] B_x[q,s] (<pq|rs> = sum_x B_x[p,r] B_x[q,s]), and
nothing with four virtual indices is stored.

Everything is spin-integrated: alpha and beta orbitals, densities, and Y are kept apart (for
an RHF reference the two spins share the MOs, but the code never assumes so).
"""

import numpy as np

BLOCKS = [p + q + r + s for p in "ov" for q in "ov" for r in "ov" for s in "ov"]
SPINS = ("a", "b")


def get_mo_integrals(meanfield, system):
    """Bare MO quantities over all MOs (frozen core included), keyed by spin: orbitals C,
    orbital energies eps, core Hamiltonian h, Cholesky vectors B, occupied projectors P, the
    number of occupied orbitals, the correlated orbital slices, and the reference type."""
    if system.ndelete != 0:
        raise NotImplementedError("Gradients with deleted virtual orbitals are not implemented.")
    nfz = system.nfrozen
    restricted = len(meanfield.C) == 1
    C = [meanfield.C[0], meanfield.C[0]] if restricted else list(meanfield.C)
    eps = [meanfield.eps[0], meanfield.eps[0]] if restricted else list(meanfield.eps)
    nmo = C[0].shape[1]
    nocc = {"a": nfz + system.noccupied_alpha, "b": nfz + system.noccupied_beta}
    end = nfz + system.norbitals
    mo = {"reference": type(meanfield).__name__, "nmo": nmo, "nfrozen": nfz, "nocc": nocc, "C": {}, "eps": {}, "h": {}, "B": {}, "P": {},
          "slices": {s: {"o": slice(nfz, nocc[s]), "v": slice(nocc[s], end)} for s in SPINS}}
    hcore = meanfield.get_hcore()
    for k, s in enumerate(SPINS):
        mo["C"][s] = C[k]
        mo["eps"][s] = eps[k]
        mo["h"][s] = C[k].T @ hcore @ C[k]
        mo["B"][s] = mo["B"]["a"] if (s == "b" and restricted) else meanfield.mol.df.mo_cholesky_vectors(C[k])
        mo["P"][s] = np.diag((np.arange(nmo) < nocc[s]).astype(float))
    return mo


def build_bare_hamiltonian(mo, system):
    """Bare Cholesky Hamiltonian of the correlated orbitals, per spin: the normal-ordered Fock
    blocks f[s] = h[s] + sum_t J[P[t]] - K[P[s]] and the Cholesky blocks H0.chol (the input of
    ccpy.hbar.hbar_ccs_chol.build_hbar_ccs_chol)."""
    from ccpy.models.integrals import Integral

    H0 = Integral.from_empty(system, 1, use_none=True)
    H0.chol = Integral.from_empty(system, 1, use_none=True)
    B, P = mo["B"], mo["P"]
    rho = sum(np.einsum("xqs,qs->x", B[s], P[s], optimize=True) for s in SPINS)
    for s in SPINS:
        f = mo["h"][s] + np.einsum("xpq,x->pq", B[s], rho, optimize=True) - np.einsum("xpr,rs,xsq->pq", B[s], P[s], B[s], optimize=True)
        sl = mo["slices"][s]
        for blk in ("oo", "ov", "vo", "vv"):
            p, q = sl[blk[0]], sl[blk[1]]
            setattr(getattr(H0, s), blk, f[p, q])
            setattr(getattr(H0.chol, s), blk, B[s][:, p, q])
    return H0


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
    Y = {s: np.zeros((naux, nmo, nmo)) for s in SPINS}

    # two-body: Y(T1) of the rdm2 blocks, contracted with the transformed vectors
    nvir = {s: getattr(Bt, s).vv.shape[1] for s in SPINS}
    nocc = {s: getattr(Bt, s).oo.shape[1] for s in SPINS}
    n = {s: nocc[s] + nvir[s] for s in SPINS}
    cs = {s: {"o": slice(0, nocc[s]), "v": slice(nocc[s], n[s])} for s in SPINS}
    Yt, Bc = {}, {}
    for s in SPINS:
        Bc[s] = np.zeros((naux, n[s], n[s]))
        for blk in ("oo", "ov", "vo", "vv"):
            Bc[s][:, cs[s][blk[0]], cs[s][blk[1]]] = getattr(getattr(Bt, s), blk)
        Yt[s] = np.zeros((naux, n[s], n[s]))
    for s in SPINS:
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
    corr = {s: slice(sl[s]["o"].start, sl[s]["v"].stop) for s in SPINS}
    t1 = {s: np.zeros((n[s], n[s])) for s in SPINS}
    if T is not None:
        t1["a"][cs["a"]["v"], cs["a"]["o"]] = T.a
        t1["b"][cs["b"]["v"], cs["b"]["o"]] = T.b
    gamma = {}
    for s in SPINS:
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
    for s in SPINS:
        P[s] = mo["P"][s].copy()
        P[s][corr[s], corr[s]] += t1[s].T
    D1 = {s: gamma[s] + (P[s] if reference else 0.0) for s in SPINS}
    for g, scale in ((gamma, 1.0),) + (((P, 0.5),) if reference else ()):
        # D2_ss = P_qs g_pr - P_qr g_ps - P_ps g_qr + P_pr g_qs,  D2_st = P^t_qs g^s_pr + P^s_pr g^t_qs
        PB = sum(np.einsum("qs,xqs->x", P[u], B[u], optimize=True) for u in SPINS)
        gB = sum(np.einsum("qs,xqs->x", g[u], B[u], optimize=True) for u in SPINS)
        for s in SPINS:
            Y[s] += scale * (np.einsum("pr,x->xpr", g[s], PB, optimize=True)
                             + np.einsum("pr,x->xpr", P[s], gB, optimize=True)
                             - np.einsum("ps,xqs,qr->xpr", g[s], B[s], P[s], optimize=True)
                             - np.einsum("ps,xqs,qr->xpr", P[s], B[s], g[s], optimize=True))
    return D1, Y


def add_densities(*pieces):
    """Spin-by-spin sum of several (D1, Y) pairs."""
    return ({s: sum(p[0][s] for p in pieces) for s in SPINS},
            {s: sum(p[1][s] for p in pieces) for s in SPINS})


def rotation_gradient(D1, Y, h, B, mo):
    """dF/dt_i^a of a functional F(h, B) with dF/dh = D1 and dF/dB = Y, where t_i^a rotates
    the Hamiltonian as H -> H + e [H, a+ i] ([M, E] = M E - E M for h and each B_x):
    (h^T D1 - D1 h^T)[a, i] + sum_x (B_x^T Y_x - Y_x B_x^T)[a, i], per spin."""
    out = {}
    for s in SPINS:
        o, v = mo["slices"][s]["o"], mo["slices"][s]["v"]
        out[s] = (h[s][:, v].T @ D1[s][:, o] - D1[s][v, :] @ h[s][o, :].T
                  + np.einsum("xqa,xqi->ai", B[s][:, :, v], Y[s][:, :, o], optimize=True)
                  - np.einsum("xaq,xiq->ai", Y[s][:, v, :], B[s][:, o, :], optimize=True))
    return out
