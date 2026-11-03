"""Amplitude-relaxed densities of CC functionals in the bare MO frame, per spin: the step
shared by the CC, CR-CC(2,3), CR-EOMCC(2,3) and CC(P;Q) gradients that takes the densities of
ccpy.density (conjugate to the stored blocks of H(T1) = e^{-T1} H e^{T1},
ccpy.hbar.hbar_ccs_chol) to D1[s] = dL/dh[s] and Y[s] = dL/dB[s] over all MOs (frozen core
included), the input of ccpy.gradients.orbital_response. The bare MO quantities (h_mo, B_mo,
mo_coeff, mo_energy, keyed by spin) and the System come from ccpy.methods.Reference.

Two-body densities only ever appear contracted with the three-index vectors,
Y_x[p, r] = sum_qs D2[p,q,r,s] B_x[q,s] (<pq|rs> = sum_x B_x[p,r] B_x[q,s]), and nothing with
four virtual indices is stored. Alpha and beta orbitals, densities and Y are kept apart (for an
RHF reference the two spins share the MOs, but the code never assumes so).
"""

import numpy as np

BLOCKS = [p + q + r + s for p in "ov" for q in "ov" for r in "ov" for s in "ov"]
SPINS = ("a", "b")


def mo_slices(system):
    """Correlated occupied and virtual orbitals of each spin, as slices of all MOs."""
    nfz, end = system.nfrozen, system.nfrozen + system.norbitals
    nocc = {"a": nfz + system.noccupied_alpha, "b": nfz + system.noccupied_beta}
    return {s: {"o": slice(nfz, nocc[s]), "v": slice(nocc[s], end)} for s in SPINS}


def occupied_projectors(reference):
    """Projectors onto the occupied orbitals (frozen core included) of each spin, over all MOs."""
    nmo = reference.h_mo["a"].shape[0]
    nocc = {"a": reference.nfrozen + reference.system.noccupied_alpha,
            "b": reference.nfrozen + reference.system.noccupied_beta}
    return {s: np.diag((np.arange(nmo) < nocc[s]).astype(float)) for s in SPINS}


def build_DY(rdm1, rdm2, vvvv_slice, Bt, T, B, reference, ref_weight=False):
    """Bare-frame (D1, Y) over all MOs of a functional F(H(T1)) whose densities rdm1, rdm2
    (ccpy.density) are conjugate to the stored blocks of H(T1) (rdm1 to the normal-ordered Fock
    blocks f = h + sum_i <pi||qi>, rdm2 to H.aa.pqrs = <pq||rs>, H.ab.pqrs = <pq|rs>, ...).

    Bt : Cholesky vectors of the Hamiltonian the densities refer to (H.chol of
         ccpy.hbar.hbar_ccs_chol, or the bare H0.chol when T is None), contracted with rdm2 and
         vvvv_slice(a, spin) (vvvv densities one first index at a time, or None)
    T  : T1 of the transformation (None for a functional of the bare H)
    B  : bare Cholesky vectors {spin: (x, nmo, nmo)} for the separable (and reference) terms
         (reference.B_mo, or a rotated copy)
    reference : ccpy.methods.Reference
    ref_weight=True adds the reference determinant E_ref = sum_i h_ii + 1/2 sum_ij <ij||ij> (a
    number weights it).
    Returns D1 = dF/dh and Y = dF/dB, per spin."""
    sl, nmo = mo_slices(reference.system), reference.h_mo["a"].shape[0]
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
        # one vvvv slice at a time, each contraction a single GEMM on (x, vv) matrices; the
        # (vv, v) @ (v, x) orientation avoids a slow small-k path of some BLAS builds
        Ba = np.ascontiguousarray(Bt.a.vv)
        Bb = np.ascontiguousarray(Bt.b.vv)
        nx, na, nb = Ba.shape[0], nvir["a"], nvir["b"]
        Ba2, Bb2 = Ba.reshape(nx, na * na), Bb.reshape(nx, nb * nb)
        Ba_a = np.ascontiguousarray(Ba.transpose(1, 0, 2))    # Ba_a[a] = B[:, a, :], contiguous for BLAS
        Bb_a = np.ascontiguousarray(Bb.transpose(1, 0, 2))
        for a in range(na):
            G_aa, G_ab = vvvv_slice(a, "a")
            G_aa = np.ascontiguousarray(G_aa)
            G_cbd = np.ascontiguousarray(G_aa.transpose(1, 0, 2)).reshape(na, na * na)       # [c, bd]
            G_bcd = G_aa.reshape(na * na, na)                                                 # [bc, d]
            Yt["a"][:, va.start + a, va] += Ba2 @ G_cbd.T                                     # bcd,xbd->xc
            Yt["a"][:, va, va] += (G_cbd.T @ Ba_a[a].T).T.reshape(nx, na, na)                   # bcd,xc->xbd
            Yt["a"][:, va.start + a, va] -= Ba2 @ G_bcd                                       # bcd,xbc->xd
            Yt["a"][:, va, va] -= (G_bcd @ Ba_a[a].T).T.reshape(nx, na, na)                 # bcd,xd->xbc
            G_ab = np.ascontiguousarray(np.asarray(G_ab).transpose(1, 0, 2)).reshape(na, nb * nb)  # [c, bd]
            Yt["a"][:, va.start + a, va] += Bb2 @ G_ab.T                                      # bcd,xbd->xc
            Yt["b"][:, vb, vb] += (G_ab.T @ Ba_a[a].T).T.reshape(nx, nb, nb)                    # bcd,xc->xbd
        for a in range(nb):
            G_bb = np.ascontiguousarray(vvvv_slice(a, "b"))
            G_cbd = np.ascontiguousarray(G_bb.transpose(1, 0, 2)).reshape(nb, nb * nb)
            G_bcd = G_bb.reshape(nb * nb, nb)
            Yt["b"][:, vb.start + a, vb] += Bb2 @ G_cbd.T
            Yt["b"][:, vb, vb] += (G_cbd.T @ Bb_a[a].T).T.reshape(nx, nb, nb)
            Yt["b"][:, vb.start + a, vb] -= Bb2 @ G_bcd
            Yt["b"][:, vb, vb] -= (G_bcd @ Bb_a[a].T).T.reshape(nx, nb, nb)

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
    P = occupied_projectors(reference)
    for s in SPINS:
        P[s][corr[s], corr[s]] += t1[s].T
    D1 = {s: gamma[s] + ref_weight * P[s] for s in SPINS}
    for g, scale in ((gamma, 1.0),) + (((P, 0.5 * ref_weight),) if ref_weight else ()):
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


def rotation_gradient(D1, Y, h, B, reference):
    """dF/dt_i^a of a functional F(h, B) with dF/dh = D1 and dF/dB = Y, where t_i^a rotates
    the Hamiltonian as H -> H + e [H, a+ i] ([M, E] = M E - E M for h and each B_x):
    (h^T D1 - D1 h^T)[a, i] + sum_x (B_x^T Y_x - Y_x B_x^T)[a, i], per spin."""
    sl = mo_slices(reference.system)
    out = {}
    for s in SPINS:
        o, v = sl[s]["o"], sl[s]["v"]
        out[s] = (h[s][:, v].T @ D1[s][:, o] - D1[s][v, :] @ h[s][o, :].T
                  + np.einsum("xqa,xqi->ai", B[s][:, :, v], Y[s][:, :, o], optimize=True)
                  - np.einsum("xaq,xiq->ai", Y[s][:, v, :], B[s][:, o, :], optimize=True))
    return out
