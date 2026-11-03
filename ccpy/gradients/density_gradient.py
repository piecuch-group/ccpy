"""Amplitude-relaxed densities of CC functionals in the bare MO frame: the step shared by the CC,
CR-CC(2,3), CR-EOMCC(2,3) and CC(P;Q) gradients that takes the densities of ccpy.density
(conjugate to the stored blocks of H(T1) = e^{-T1} H e^{T1}, ccpy.hbar.hbar_ccs_chol) to
D1 = dL/dh and Y = dL/dB, the input of ccpy.gradients.orbital_response.

Densities, h and B are ccpy Integral containers with the blocks oo, ov, vo, vv of each spin
(D1.a.ov, Y.b.vv, ...). Here "o" runs over all occupied orbitals, frozen core included, so the
correlated occupied orbitals of ccpy.density and H(T1) are the occupied indices [nfrozen:].
Y blocks carry the auxiliary index first, Y.a.ov[x, i, a] = dL/dB_x[i, a].

Two-body densities only ever appear contracted with the three-index vectors,
Y_x[p, r] = sum_qs D2[p,q,r,s] B_x[q,s] (<pq|rs> = sum_x B_x[p,r] B_x[q,s]), and nothing with
four virtual indices is stored. Alpha and beta orbitals, densities and Y are kept apart (for an
RHF reference the two spins share the MOs, but the code never assumes so).
"""

import copy

import numpy as np

from ccpy.models.integrals import Integral


# ======================================================================================
# containers
# ======================================================================================
def get_bare_integrals(reference):
    """Bare core Hamiltonian h and Cholesky vectors B of the reference in blocks (views of
    reference.h_mo and reference.B_mo; "o" includes the frozen core)."""
    nfz = reference.nfrozen
    noa, nob = nfz + reference.system.noccupied_alpha, nfz + reference.system.noccupied_beta
    ha, hb, Ba, Bb = reference.h_mo["a"], reference.h_mo["b"], reference.B_mo["a"], reference.B_mo["b"]
    h = Integral.from_empty(reference.system, 1, use_none=True)
    h.a.oo, h.a.ov, h.a.vo, h.a.vv = ha[:noa, :noa], ha[:noa, noa:], ha[noa:, :noa], ha[noa:, noa:]
    h.b.oo, h.b.ov, h.b.vo, h.b.vv = hb[:nob, :nob], hb[:nob, nob:], hb[nob:, :nob], hb[nob:, nob:]
    B = Integral.from_empty(reference.system, 1, use_none=True)
    B.a.oo, B.a.ov, B.a.vo, B.a.vv = Ba[:, :noa, :noa], Ba[:, :noa, noa:], Ba[:, noa:, :noa], Ba[:, noa:, noa:]
    B.b.oo, B.b.ov, B.b.vo, B.b.vv = Bb[:, :nob, :nob], Bb[:, :nob, nob:], Bb[:, nob:, :nob], Bb[:, nob:, nob:]
    return h, B


def zero_density(reference, naux=None):
    """Zero blocks oo, ov, vo, vv of each spin over all occupied (frozen core included) and
    virtual orbitals, with a leading auxiliary index of length naux if given (Y-type)."""
    nfz, system = reference.nfrozen, reference.system
    noa, nob = nfz + system.noccupied_alpha, nfz + system.noccupied_beta
    nua, nub = system.nunoccupied_alpha, system.nunoccupied_beta
    x = () if naux is None else (naux,)
    out = Integral.from_empty(system, 1, use_none=True)
    out.a.oo, out.a.ov = np.zeros(x + (noa, noa)), np.zeros(x + (noa, nua))
    out.a.vo, out.a.vv = np.zeros(x + (nua, noa)), np.zeros(x + (nua, nua))
    out.b.oo, out.b.ov = np.zeros(x + (nob, nob)), np.zeros(x + (nob, nub))
    out.b.vo, out.b.vv = np.zeros(x + (nub, nob)), np.zeros(x + (nub, nub))
    return out


def add_blocks(out, x, c=1.0):
    """out += c * x, block by block."""
    out.a.oo += c * x.a.oo
    out.a.ov += c * x.a.ov
    out.a.vo += c * x.a.vo
    out.a.vv += c * x.a.vv
    out.b.oo += c * x.b.oo
    out.b.ov += c * x.b.ov
    out.b.vo += c * x.b.vo
    out.b.vv += c * x.b.vv
    return out


def add_densities(*pieces):
    """Sum of several (D1, Y) pairs."""
    D1 = add_blocks(zero_like(pieces[0][0]), pieces[0][0])
    Y = add_blocks(zero_like(pieces[0][1]), pieces[0][1])
    for d1, y in pieces[1:]:
        add_blocks(D1, d1)
        add_blocks(Y, y)
    return D1, Y


def scale_density(c, DY):
    """c * (D1, Y)."""
    D1, Y = DY
    return add_blocks(zero_like(D1), D1, c), add_blocks(zero_like(Y), Y, c)


def zero_like(x):
    """Zero blocks shaped like those of the container x (a shallow copy, no data copied)."""
    out = copy.copy(x)
    out.a, out.b = copy.copy(x.a), copy.copy(x.b)
    out.a.oo, out.a.ov, out.a.vo, out.a.vv = (np.zeros_like(x.a.oo), np.zeros_like(x.a.ov),
                                              np.zeros_like(x.a.vo), np.zeros_like(x.a.vv))
    out.b.oo, out.b.ov, out.b.vo, out.b.vv = (np.zeros_like(x.b.oo), np.zeros_like(x.b.ov),
                                              np.zeros_like(x.b.vo), np.zeros_like(x.b.vv))
    return out


# ======================================================================================
# densities of H(T1) -> bare frame
# ======================================================================================
def build_DY(rdm1, rdm2, vvvv_slice, Bt, T, B, reference, ref_weight=False):
    """Bare-frame (D1, Y) of a functional F(H(T1)) whose densities rdm1, rdm2 (ccpy.density)
    are conjugate to the stored blocks of H(T1) (rdm1 to the normal-ordered Fock blocks
    f = h + sum_i <pi||qi>, rdm2 to H.aa.pqrs = <pq||rs>, H.ab.pqrs = <pq|rs>, ...).

    Bt : Cholesky vectors of the Hamiltonian the densities refer to (H.chol of
         ccpy.hbar.hbar_ccs_chol, or the bare H0.chol when T is None), contracted with rdm2 and
         vvvv_slice(a, spin) (vvvv densities one first index at a time, or None)
    T  : T1 of the transformation (None for a functional of the bare H)
    B  : bare Cholesky vectors in blocks (get_bare_integrals, or a rotated copy) for the
         separable (and reference) terms
    reference : ccpy.methods.Reference
    ref_weight=True adds the reference determinant E_ref = sum_i h_ii + 1/2 sum_ij <ij||ij> (a
    number weights it)."""
    # two-body: Yt = dF/dBt of the rdm2 blocks, in the correlated orbitals of H(T1)
    Yt = Integral.from_empty(reference.system, 1, use_none=True)
    Yt.a.oo, Yt.a.ov, Yt.a.vo, Yt.a.vv = (np.zeros_like(Bt.a.oo), np.zeros_like(Bt.a.ov),
                                          np.zeros_like(Bt.a.vo), np.zeros_like(Bt.a.vv))
    Yt.b.oo, Yt.b.ov, Yt.b.vo, Yt.b.vv = (np.zeros_like(Bt.b.oo), np.zeros_like(Bt.b.ov),
                                          np.zeros_like(Bt.b.vo), np.zeros_like(Bt.b.vv))
    # same spin, <pq||rs> = B_pr B_qs - B_ps B_qr
    _add_same_spin(rdm2.aa, Yt.a, Bt.a)
    _add_same_spin(rdm2.bb, Yt.b, Bt.b)
    # opposite spin, <pq|rs> = B^a_pr B^b_qs:  Y^a_pr += G B^b_qs,  Y^b_qs += G B^a_pr
    G, Ya, Yb, Ba, Bb = rdm2.ab, Yt.a, Yt.b, Bt.a, Bt.b
    _opposite_spin(G.oooo, Ya.oo, Yb.oo, Ba.oo, Bb.oo)
    _opposite_spin(G.ooov, Ya.oo, Yb.ov, Ba.oo, Bb.ov)
    _opposite_spin(G.oovo, Ya.ov, Yb.oo, Ba.ov, Bb.oo)
    _opposite_spin(G.oovv, Ya.ov, Yb.ov, Ba.ov, Bb.ov)
    _opposite_spin(G.ovoo, Ya.oo, Yb.vo, Ba.oo, Bb.vo)
    _opposite_spin(G.ovov, Ya.oo, Yb.vv, Ba.oo, Bb.vv)
    _opposite_spin(G.ovvo, Ya.ov, Yb.vo, Ba.ov, Bb.vo)
    _opposite_spin(G.ovvv, Ya.ov, Yb.vv, Ba.ov, Bb.vv)
    _opposite_spin(G.vooo, Ya.vo, Yb.oo, Ba.vo, Bb.oo)
    _opposite_spin(G.voov, Ya.vo, Yb.ov, Ba.vo, Bb.ov)
    _opposite_spin(G.vovo, Ya.vv, Yb.oo, Ba.vv, Bb.oo)
    _opposite_spin(G.vovv, Ya.vv, Yb.ov, Ba.vv, Bb.ov)
    _opposite_spin(G.vvoo, Ya.vo, Yb.vo, Ba.vo, Bb.vo)
    _opposite_spin(G.vvov, Ya.vo, Yb.vv, Ba.vo, Bb.vv)
    _opposite_spin(G.vvvo, Ya.vv, Yb.vo, Ba.vv, Bb.vo)
    if vvvv_slice is not None:
        _add_vvvv(vvvv_slice, Yt, Bt)

    # back to the bare frame: (1 - t1)^T . (1 + t1)^T on the correlated orbitals
    nfz = reference.nfrozen
    noa, nua = Bt.a.ov.shape[1:]
    nob, nub = Bt.b.ov.shape[1:]
    gt = Integral.from_empty(reference.system, 1, use_none=True)
    gt.a.oo = np.zeros((noa, noa)) if rdm1.a.oo is None else rdm1.a.oo
    gt.a.ov = np.zeros((noa, nua)) if rdm1.a.ov is None else rdm1.a.ov
    gt.a.vo = np.zeros((nua, noa)) if rdm1.a.vo is None else rdm1.a.vo
    gt.a.vv = np.zeros((nua, nua)) if rdm1.a.vv is None else rdm1.a.vv
    gt.b.oo = np.zeros((nob, nob)) if rdm1.b.oo is None else rdm1.b.oo
    gt.b.ov = np.zeros((nob, nub)) if rdm1.b.ov is None else rdm1.b.ov
    gt.b.vo = np.zeros((nub, nob)) if rdm1.b.vo is None else rdm1.b.vo
    gt.b.vv = np.zeros((nub, nub)) if rdm1.b.vv is None else rdm1.b.vv
    t1a = None if T is None else T.a.T
    t1b = None if T is None else T.b.T
    gamma = zero_density(reference)
    _embed(gamma.a, _bare_frame(gt.a, t1a), nfz)
    _embed(gamma.b, _bare_frame(gt.b, t1b), nfz)
    Y = zero_density(reference, Bt.a.oo.shape[0])
    _embed(Y.a, _bare_frame(Yt.a, t1a), nfz)
    _embed(Y.b, _bare_frame(Yt.b, t1b), nfz)

    # one-body and separable terms (the reference projector of H(T1) is P + t1^T in the bare frame)
    P = zero_density(reference)
    P.a.oo[...] = np.eye(P.a.oo.shape[0])
    P.b.oo[...] = np.eye(P.b.oo.shape[0])
    if T is not None:
        P.a.ov[nfz:, :] = t1a
        P.b.ov[nfz:, :] = t1b
    D1 = add_blocks(add_blocks(zero_density(reference), gamma), P, ref_weight)
    for g, scale in ((gamma, 1.0),) + (((P, 0.5 * ref_weight),) if ref_weight else ()):
        # D2_ss = P_qs g_pr - P_qr g_ps - P_ps g_qr + P_pr g_qs,  D2_st = P^t_qs g^s_pr + P^s_pr g^t_qs
        PB = _trace(P.a, B.a) + _trace(P.b, B.b)
        gB = _trace(g.a, B.a) + _trace(g.b, B.b)
        _add_separable(Y.a, g.a, P.a, B.a, PB, gB, scale)
        _add_separable(Y.b, g.b, P.b, B.b, PB, gB, scale)
    return D1, Y


def _add_same_spin(G, Y, Bs):
    """Y_pr += G B_qs,  Y_qs += G B_pr,  Y_ps -= G B_qr,  Y_qr -= G B_ps for each block pqrs."""
    _same_spin(G.oooo, Y.oo, Y.oo, Y.oo, Y.oo, Bs.oo, Bs.oo, Bs.oo, Bs.oo)
    _same_spin(G.ooov, Y.oo, Y.ov, Y.ov, Y.oo, Bs.oo, Bs.ov, Bs.ov, Bs.oo)
    _same_spin(G.oovo, Y.ov, Y.oo, Y.oo, Y.ov, Bs.ov, Bs.oo, Bs.oo, Bs.ov)
    _same_spin(G.oovv, Y.ov, Y.ov, Y.ov, Y.ov, Bs.ov, Bs.ov, Bs.ov, Bs.ov)
    _same_spin(G.ovoo, Y.oo, Y.vo, Y.oo, Y.vo, Bs.oo, Bs.vo, Bs.oo, Bs.vo)
    _same_spin(G.ovov, Y.oo, Y.vv, Y.ov, Y.vo, Bs.oo, Bs.vv, Bs.ov, Bs.vo)
    _same_spin(G.ovvo, Y.ov, Y.vo, Y.oo, Y.vv, Bs.ov, Bs.vo, Bs.oo, Bs.vv)
    _same_spin(G.ovvv, Y.ov, Y.vv, Y.ov, Y.vv, Bs.ov, Bs.vv, Bs.ov, Bs.vv)
    _same_spin(G.vooo, Y.vo, Y.oo, Y.vo, Y.oo, Bs.vo, Bs.oo, Bs.vo, Bs.oo)
    _same_spin(G.voov, Y.vo, Y.ov, Y.vv, Y.oo, Bs.vo, Bs.ov, Bs.vv, Bs.oo)
    _same_spin(G.vovo, Y.vv, Y.oo, Y.vo, Y.ov, Bs.vv, Bs.oo, Bs.vo, Bs.ov)
    _same_spin(G.vovv, Y.vv, Y.ov, Y.vv, Y.ov, Bs.vv, Bs.ov, Bs.vv, Bs.ov)
    _same_spin(G.vvoo, Y.vo, Y.vo, Y.vo, Y.vo, Bs.vo, Bs.vo, Bs.vo, Bs.vo)
    _same_spin(G.vvov, Y.vo, Y.vv, Y.vv, Y.vo, Bs.vo, Bs.vv, Bs.vv, Bs.vo)
    _same_spin(G.vvvo, Y.vv, Y.vo, Y.vo, Y.vv, Bs.vv, Bs.vo, Bs.vo, Bs.vv)


def _same_spin(G, Y_pr, Y_qs, Y_ps, Y_qr, B_pr, B_qs, B_ps, B_qr):
    if G is None or np.isscalar(G):
        return
    Y_pr += np.einsum("pqrs,xqs->xpr", G, B_qs, optimize=True)
    Y_qs += np.einsum("pqrs,xpr->xqs", G, B_pr, optimize=True)
    Y_ps -= np.einsum("pqrs,xqr->xps", G, B_qr, optimize=True)
    Y_qr -= np.einsum("pqrs,xps->xqr", G, B_ps, optimize=True)


def _opposite_spin(G, Ya_pr, Yb_qs, Ba_pr, Bb_qs):
    if G is None or np.isscalar(G):
        return
    Ya_pr += np.einsum("pqrs,xqs->xpr", G, Bb_qs, optimize=True)
    Yb_qs += np.einsum("pqrs,xpr->xqs", G, Ba_pr, optimize=True)


def _add_vvvv(vvvv_slice, Yt, Bt):
    """vvvv densities, one first index a at a time: each contraction a single GEMM on (x, vv)
    matrices (the (vv, v) @ (v, x) orientation avoids a slow small-k path of some BLAS builds)."""
    Ba, Bb = np.ascontiguousarray(Bt.a.vv), np.ascontiguousarray(Bt.b.vv)
    nx, na, nb = Ba.shape[0], Ba.shape[1], Bb.shape[1]
    Ba2, Bb2 = Ba.reshape(nx, na * na), Bb.reshape(nx, nb * nb)
    Ba_a = np.ascontiguousarray(Ba.transpose(1, 0, 2))       # Ba_a[a] = B[:, a, :], contiguous for BLAS
    Bb_a = np.ascontiguousarray(Bb.transpose(1, 0, 2))
    for a in range(na):
        G_aa, G_ab = vvvv_slice(a, "a")
        G_aa = np.ascontiguousarray(G_aa)
        G_cbd = np.ascontiguousarray(G_aa.transpose(1, 0, 2)).reshape(na, na * na)     # [c, bd]
        G_bcd = G_aa.reshape(na * na, na)                                               # [bc, d]
        Yt.a.vv[:, a, :] += Ba2 @ G_cbd.T                                               # bcd,xbd->xc
        Yt.a.vv += (G_cbd.T @ Ba_a[a].T).T.reshape(nx, na, na)                           # bcd,xc->xbd
        Yt.a.vv[:, a, :] -= Ba2 @ G_bcd                                                 # bcd,xbc->xd
        Yt.a.vv -= (G_bcd @ Ba_a[a].T).T.reshape(nx, na, na)                             # bcd,xd->xbc
        G_ab = np.ascontiguousarray(np.asarray(G_ab).transpose(1, 0, 2)).reshape(na, nb * nb)   # [c, bd]
        Yt.a.vv[:, a, :] += Bb2 @ G_ab.T                                                # bcd,xbd->xc
        Yt.b.vv += (G_ab.T @ Ba_a[a].T).T.reshape(nx, nb, nb)                            # bcd,xc->xbd
    for a in range(nb):
        G_bb = np.ascontiguousarray(vvvv_slice(a, "b"))
        G_cbd = np.ascontiguousarray(G_bb.transpose(1, 0, 2)).reshape(nb, nb * nb)
        G_bcd = G_bb.reshape(nb * nb, nb)
        Yt.b.vv[:, a, :] += Bb2 @ G_cbd.T
        Yt.b.vv += (G_cbd.T @ Bb_a[a].T).T.reshape(nx, nb, nb)
        Yt.b.vv[:, a, :] -= Bb2 @ G_bcd
        Yt.b.vv -= (G_bcd @ Bb_a[a].T).T.reshape(nx, nb, nb)


def _bare_frame(M, t1t):
    """Blocks (oo, ov, vo, vv) of (1 - t1^T) M (1 + t1^T) for t1 with only a (v, o) block
    (t1t = t1^T); the blocks of M may carry a leading auxiliary index."""
    if t1t is None:
        return M.oo, M.ov, M.vo, M.vv
    return (M.oo - t1t @ M.vo,
            M.ov - t1t @ M.vv + M.oo @ t1t - t1t @ M.vo @ t1t,
            M.vo,
            M.vv + M.vo @ t1t)


def _embed(out, blocks, nfz):
    """Correlated blocks (oo, ov, vo, vv) into blocks over all occupied orbitals (frozen core first)."""
    oo, ov, vo, vv = blocks
    out.oo[..., nfz:, nfz:] += oo
    out.ov[..., nfz:, :] += ov
    out.vo[..., :, nfz:] += vo
    out.vv += vv


def _trace(M, Bs):
    """sum_pq M_pq B_x[p, q] over the four blocks."""
    return (np.einsum("pq,xpq->x", M.oo, Bs.oo, optimize=True) + np.einsum("pq,xpq->x", M.ov, Bs.ov, optimize=True)
            + np.einsum("pq,xpq->x", M.vo, Bs.vo, optimize=True) + np.einsum("pq,xpq->x", M.vv, Bs.vv, optimize=True))


def _add_separable(Y, g, P, Bs, PB, gB, scale):
    """Y_x += scale (g PB_x + P gB_x - g B_x^T P - P B_x^T g) for a projector-like P with only
    oo and ov blocks."""
    Y.oo += scale * (g.oo[None] * PB[:, None, None] + P.oo[None] * gB[:, None, None])
    Y.ov += scale * (g.ov[None] * PB[:, None, None] + P.ov[None] * gB[:, None, None])
    Y.vo += scale * g.vo[None] * PB[:, None, None]
    Y.vv += scale * g.vv[None] * PB[:, None, None]
    # (g B^T)_{p,o} = sum_s g_ps B_{o s}, then times P_{o r}
    gBo_o = np.einsum("ps,xqs->xpq", g.oo, Bs.oo, optimize=True) + np.einsum("ps,xqs->xpq", g.ov, Bs.ov, optimize=True)
    gBv_o = np.einsum("ps,xqs->xpq", g.vo, Bs.oo, optimize=True) + np.einsum("ps,xqs->xpq", g.vv, Bs.ov, optimize=True)
    Y.oo -= scale * gBo_o @ P.oo
    Y.ov -= scale * gBo_o @ P.ov
    Y.vo -= scale * gBv_o @ P.oo
    Y.vv -= scale * gBv_o @ P.ov
    # (B^T g)_{s,r} = sum_q B_{q s} g_{q r}, then P_{o s} times it
    BTg_oo = np.einsum("xqs,qr->xsr", Bs.oo, g.oo, optimize=True) + np.einsum("xqs,qr->xsr", Bs.vo, g.vo, optimize=True)
    BTg_ov = np.einsum("xqs,qr->xsr", Bs.oo, g.ov, optimize=True) + np.einsum("xqs,qr->xsr", Bs.vo, g.vv, optimize=True)
    BTg_vo = np.einsum("xqs,qr->xsr", Bs.ov, g.oo, optimize=True) + np.einsum("xqs,qr->xsr", Bs.vv, g.vo, optimize=True)
    BTg_vv = np.einsum("xqs,qr->xsr", Bs.ov, g.ov, optimize=True) + np.einsum("xqs,qr->xsr", Bs.vv, g.vv, optimize=True)
    Y.oo -= scale * (P.oo @ BTg_oo + P.ov @ BTg_vo)
    Y.ov -= scale * (P.oo @ BTg_ov + P.ov @ BTg_vv)


# ======================================================================================
# orbital rotations by a singles vector w (a (v, o) block over the correlated occupied)
# ======================================================================================
def commutator(M, wa, wb, nfz):
    """[M, W1] = M W1 - W1 M of block matrices M (h-like, or B-like with an auxiliary index)
    for W1 with only a (v, o_correlated) block w per spin."""
    out = zero_like(M)
    _commutator(out.a, M.a, wa, nfz)
    _commutator(out.b, M.b, wb, nfz)
    return out


def _commutator(C, M, w, nfz):
    C.oo[..., :, nfz:] += M.ov @ w
    C.vo[..., :, nfz:] += M.vv @ w
    C.vo -= w @ M.oo[..., nfz:, :]
    C.vv -= w @ M.ov[..., nfz:, :]


def rotate_density(M, wa, wb, nfz):
    """M W1^T - W1^T M of block densities M for W1 with only a (v, o_correlated) block w."""
    out = zero_like(M)
    _rotate_density(out.a, M.a, wa, nfz)
    _rotate_density(out.b, M.b, wb, nfz)
    return out


def _rotate_density(R, M, w, nfz):
    R.oo[..., nfz:, :] -= w.T @ M.vo
    R.ov += M.oo[..., :, nfz:] @ w.T
    R.ov[..., nfz:, :] -= w.T @ M.vv
    R.vv += M.vo[..., :, nfz:] @ w.T


def rotation_gradient(D1, Y, h, B, reference):
    """dF/dt_i^a of a functional F(h, B) with dF/dh = D1 and dF/dB = Y, where t_i^a rotates
    the Hamiltonian as H -> H + e [H, a+ i] ([M, E] = M E - E M for h and each B_x):
    (h^T D1 - D1 h^T)[a, i] + sum_x (B_x^T Y_x - Y_x B_x^T)[a, i], i correlated."""
    nfz = reference.nfrozen
    return {"a": _rotation_gradient(D1.a, Y.a, h.a, B.a)[:, nfz:],
            "b": _rotation_gradient(D1.b, Y.b, h.b, B.b)[:, nfz:]}


def _rotation_gradient(D1, Y, h, B):
    return (h.ov.T @ D1.oo + h.vv.T @ D1.vo - D1.vo @ h.oo.T - D1.vv @ h.ov.T
            + np.einsum("xqa,xqi->ai", B.ov, Y.oo, optimize=True) + np.einsum("xqa,xqi->ai", B.vv, Y.vo, optimize=True)
            - np.einsum("xaq,xiq->ai", Y.vo, B.oo, optimize=True) - np.einsum("xaq,xiq->ai", Y.vv, B.ov, optimize=True))
