"""CCS similarity-transformed Hamiltonian H(T1) = e^{-T1} H e^{T1} with Cholesky-decomposed
two-electron integrals, and its derivative along a singles direction W, [H(T1), W].

H(T1) is again a one- plus two-body operator: its Cholesky vectors are the T1-transformed
ones (exactly as in build_hbar_ccsd_chol), its one-body part is the CCS-dressed Fock matrix,
and any two-body block is composed from the transformed vectors. No vvvv block is formed;
contractions with vvvv go through H.chol.a.vv and H.chol.b.vv.
"""
import numpy as np

from ccpy.models.integrals import Integral


def build_hbar_ccs_chol(T, H0, system):
    """H(T1) = e^{-T1} H0 e^{T1} from the bare Cholesky Hamiltonian H0 (only T.a, T.b are used).
    H0 is not modified."""
    H = Integral.from_empty(system, 2, use_none=True)
    H.chol = Integral.from_empty(system, 1, use_none=True)

    # -------------------------------------------------------------------------
    ### T1-transformation of Cholesky vectors
    ### a ###
    H.chol.a.ov = H0.chol.a.ov.copy()
    H.chol.a.oo = H0.chol.a.oo.copy() + np.einsum("xme,ei->xmi", H.chol.a.ov, T.a, optimize=True)
    H.chol.a.vv = H0.chol.a.vv.copy() - np.einsum("xme,am->xae", H.chol.a.ov, T.a, optimize=True)
    H.chol.a.vo = (
            H0.chol.a.vo.copy()
            - np.einsum("xmi,am->xai", H.chol.a.oo, T.a, optimize=True)
            + np.einsum("xae,ei->xai", H.chol.a.vv, T.a, optimize=True)
            + np.einsum("xme,ei,am->xai", H.chol.a.ov, T.a, T.a, optimize=True)
    )
    ### b ###
    H.chol.b.ov = H0.chol.b.ov.copy()
    H.chol.b.oo = H0.chol.b.oo.copy() + np.einsum("xme,ei->xmi", H.chol.b.ov, T.b, optimize=True)
    H.chol.b.vv = H0.chol.b.vv.copy() - np.einsum("xme,am->xae", H.chol.b.ov, T.b, optimize=True)
    H.chol.b.vo = (
            H0.chol.b.vo.copy()
            - np.einsum("xmi,am->xai", H.chol.b.oo, T.b, optimize=True)
            + np.einsum("xae,ei->xai", H.chol.b.vv, T.b, optimize=True)
            + np.einsum("xme,ei,am->xai", H.chol.b.ov, T.b, T.b, optimize=True)
    )
    # -------------------------------------------------------------------------

    # One-body part: f(T1) = (1 - t1) [f + G(t1)] (1 + t1), with
    # G(t1)_pq = sum_{nf} <pn||qf> t_n^f + <pn~|qf~> t_n~^f~ built from the bare vectors
    bt1 = (
            np.einsum("xnf,fn->x", H0.chol.a.ov, T.a, optimize=True)
            + np.einsum("xnf,fn->x", H0.chol.b.ov, T.b, optimize=True)
    )
    for s, t1 in (("a", T.a), ("b", T.b)):
        f0, B, h = getattr(H0, s), getattr(H0.chol, s), getattr(H, s)
        F_oo = f0.oo + np.einsum("xmi,x->mi", B.oo, bt1, optimize=True) - np.einsum("xmf,fn,xni->mi", B.ov, t1, B.oo, optimize=True)
        F_ov = f0.ov + np.einsum("xme,x->me", B.ov, bt1, optimize=True) - np.einsum("xmf,fn,xne->me", B.ov, t1, B.ov, optimize=True)
        F_vo = f0.vo + np.einsum("xai,x->ai", B.vo, bt1, optimize=True) - np.einsum("xaf,fn,xni->ai", B.vv, t1, B.oo, optimize=True)
        F_vv = f0.vv + np.einsum("xae,x->ae", B.vv, bt1, optimize=True) - np.einsum("xaf,fn,xne->ae", B.vv, t1, B.ov, optimize=True)
        h.ov = F_ov
        h.oo = F_oo + np.einsum("me,ei->mi", F_ov, t1, optimize=True)
        h.vv = F_vv - np.einsum("am,me->ae", t1, F_ov, optimize=True)
        h.vo = (F_vo + np.einsum("ae,ei->ai", F_vv, t1, optimize=True) - np.einsum("am,mi->ai", t1, F_oo, optimize=True)
                - np.einsum("am,me,ei->ai", t1, F_ov, t1, optimize=True))

    # Two-body blocks from the transformed vectors
    _add_two_body(H, H.chol, H.chol, first=True)
    return H


def build_hbar_ccs_chol_derivative(W, H, system):
    """d/de H(T1 + e W) = [H(T1), W] for a singles direction W (W.a, W.b), with H from
    build_hbar_ccs_chol. Its Cholesky vectors are the pair (dB, B), dB = [B, W]: the two-body
    part is sum_x (dB_x B_x + B_x dB_x). H1.chol holds dB."""
    H1 = Integral.from_empty(system, 2, use_none=True)
    H1.chol = Integral.from_empty(system, 1, use_none=True)
    # dB = B W - W B (W: virtual <- occupied)
    for s, w in (("a", W.a), ("b", W.b)):
        B, dB = getattr(H.chol, s), getattr(H1.chol, s)
        dB.oo = np.einsum("xme,ei->xmi", B.ov, w, optimize=True)
        dB.ov = np.zeros_like(B.ov)
        dB.vv = -np.einsum("am,xme->xae", w, B.ov, optimize=True)
        dB.vo = np.einsum("xae,ei->xai", B.vv, w, optimize=True) - np.einsum("am,xmi->xai", w, B.oo, optimize=True)
    # One-body part: [f, W] + G(W), G(W)_pq = sum_{ai} <pi||qa> w_i^a (+ opposite spin)
    bw = (
            np.einsum("xia,ai->x", H.chol.a.ov, W.a, optimize=True)
            + np.einsum("xia,ai->x", H.chol.b.ov, W.b, optimize=True)
    )
    for s, w in (("a", W.a), ("b", W.b)):
        f, B, f1 = getattr(H, s), getattr(H.chol, s), getattr(H1, s)
        f1.oo = (np.einsum("me,ei->mi", f.ov, w, optimize=True)
                 + np.einsum("xmi,x->mi", B.oo, bw, optimize=True) - np.einsum("xme,en,xni->mi", B.ov, w, B.oo, optimize=True))
        f1.ov = np.einsum("xme,x->me", B.ov, bw, optimize=True) - np.einsum("xmf,fn,xne->me", B.ov, w, B.ov, optimize=True)
        f1.vv = (-np.einsum("am,me->ae", w, f.ov, optimize=True)
                 + np.einsum("xae,x->ae", B.vv, bw, optimize=True) - np.einsum("xaf,fn,xne->ae", B.vv, w, B.ov, optimize=True))
        f1.vo = (np.einsum("ae,ei->ai", f.vv, w, optimize=True) - np.einsum("am,mi->ai", w, f.oo, optimize=True)
                 + np.einsum("xai,x->ai", B.vo, bw, optimize=True) - np.einsum("xaf,fn,xni->ai", B.vv, w, B.oo, optimize=True))
    # Two-body blocks from the pair (dB, B)
    _add_two_body(H1, H1.chol, H.chol, first=True)
    _add_two_body(H1, H.chol, H1.chol, first=False)
    return H1


def _add_two_body(H, X, Y, first):
    """Two-body blocks <pq||rs> = X[p,r] Y[q,s] - X[p,s] Y[q,r] (aa, bb) and <pq|rs> = X.a[p,r] Y.b[q,s] (ab)
    from two sets of Cholesky vectors, assigned (first=True) or added to H, for the blocks used
    by the CC gradients (no vvvv)."""
    def put(target, name, value):
        setattr(target, name, value if first else getattr(target, name) + value)

    x, y = X.a, Y.a
    put(H.aa, "oooo", np.einsum("xmi,xnj->mnij", x.oo, y.oo, optimize=True)
                     - np.einsum("xmj,xni->mnij", x.oo, y.oo, optimize=True))
    put(H.aa, "ooov", np.einsum("xmi,xne->mnie", x.oo, y.ov, optimize=True)
                     - np.einsum("xme,xni->mnie", x.ov, y.oo, optimize=True))
    put(H.aa, "oovv", np.einsum("xme,xnf->mnef", x.ov, y.ov, optimize=True)
                     - np.einsum("xmf,xne->mnef", x.ov, y.ov, optimize=True))
    put(H.aa, "vooo", np.einsum("xai,xmj->amij", x.vo, y.oo, optimize=True)
                     - np.einsum("xaj,xmi->amij", x.vo, y.oo, optimize=True))
    put(H.aa, "voov", np.einsum("xai,xme->amie", x.vo, y.ov, optimize=True)
                     - np.einsum("xae,xmi->amie", x.vv, y.oo, optimize=True))
    put(H.aa, "vovv", np.einsum("xae,xmf->amef", x.vv, y.ov, optimize=True)
                     - np.einsum("xaf,xme->amef", x.vv, y.ov, optimize=True))
    put(H.aa, "vvov", np.einsum("xai,xbe->abie", x.vo, y.vv, optimize=True)
                     - np.einsum("xae,xbi->abie", x.vv, y.vo, optimize=True))

    x, y = X.b, Y.b
    put(H.bb, "oooo", np.einsum("xmi,xnj->mnij", x.oo, y.oo, optimize=True)
                     - np.einsum("xmj,xni->mnij", x.oo, y.oo, optimize=True))
    put(H.bb, "ooov", np.einsum("xmi,xne->mnie", x.oo, y.ov, optimize=True)
                     - np.einsum("xme,xni->mnie", x.ov, y.oo, optimize=True))
    put(H.bb, "oovv", np.einsum("xme,xnf->mnef", x.ov, y.ov, optimize=True)
                     - np.einsum("xmf,xne->mnef", x.ov, y.ov, optimize=True))
    put(H.bb, "vooo", np.einsum("xai,xmj->amij", x.vo, y.oo, optimize=True)
                     - np.einsum("xaj,xmi->amij", x.vo, y.oo, optimize=True))
    put(H.bb, "voov", np.einsum("xai,xme->amie", x.vo, y.ov, optimize=True)
                     - np.einsum("xae,xmi->amie", x.vv, y.oo, optimize=True))
    put(H.bb, "vovv", np.einsum("xae,xmf->amef", x.vv, y.ov, optimize=True)
                     - np.einsum("xaf,xme->amef", x.vv, y.ov, optimize=True))
    put(H.bb, "vvov", np.einsum("xai,xbe->abie", x.vo, y.vv, optimize=True)
                     - np.einsum("xae,xbi->abie", x.vv, y.vo, optimize=True))

    x, y = X.a, Y.b
    put(H.ab, "oooo", np.einsum("xmi,xnj->mnij", x.oo, y.oo, optimize=True))
    put(H.ab, "ooov", np.einsum("xmi,xne->mnie", x.oo, y.ov, optimize=True))
    put(H.ab, "oovo", np.einsum("xme,xni->mnei", x.ov, y.oo, optimize=True))
    put(H.ab, "oovv", np.einsum("xme,xnf->mnef", x.ov, y.ov, optimize=True))
    put(H.ab, "ovoo", np.einsum("xmi,xaj->maij", x.oo, y.vo, optimize=True))
    put(H.ab, "ovov", np.einsum("xmi,xae->maie", x.oo, y.vv, optimize=True))
    put(H.ab, "ovvo", np.einsum("xme,xai->maei", x.ov, y.vo, optimize=True))
    put(H.ab, "ovvv", np.einsum("xme,xaf->maef", x.ov, y.vv, optimize=True))
    put(H.ab, "vooo", np.einsum("xai,xmj->amij", x.vo, y.oo, optimize=True))
    put(H.ab, "voov", np.einsum("xai,xme->amie", x.vo, y.ov, optimize=True))
    put(H.ab, "vovo", np.einsum("xae,xmi->amei", x.vv, y.oo, optimize=True))
    put(H.ab, "vovv", np.einsum("xae,xmf->amef", x.vv, y.ov, optimize=True))
    put(H.ab, "vvov", np.einsum("xai,xbe->abie", x.vo, y.vv, optimize=True))
    put(H.ab, "vvvo", np.einsum("xae,xbi->abei", x.vv, y.vo, optimize=True))
