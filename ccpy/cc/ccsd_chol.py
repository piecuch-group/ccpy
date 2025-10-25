'''
Coupled-Cluster Method with Singles and Doubles (CCSD)
[This version uses Cholesky decomposition of the two-electron integrals]
'''

import numpy as np
from ccpy.utilities.linear_algebra import ccpy_einsum
from ccpy.lib.core import cc_loops2, vvvv_contraction

# @profile
def update(T, dT, H, X, shift, flag_RHF):

    # pre-CCS intermediates
    X.a.ov = H.a.ov + (
            ccpy_einsum("mnef,fn->me", H.aa.oovv, T.a)
            + ccpy_einsum("mnef,fn->me", H.ab.oovv, T.b)
    )
    X.a.vv = H.a.vv + (
            - 0.5 * ccpy_einsum("mnef,afmn->ae", H.aa.oovv, T.aa)  #
            - ccpy_einsum("mnef,afmn->ae", H.ab.oovv, T.ab)  #
    )
    bt1 = (
            ccpy_einsum("xnf,fn->x", H.chol.a.ov, T.a)
          + ccpy_einsum("xnf,fn->x", H.chol.b.ov, T.b)
    )
    bxt1 = -ccpy_einsum("xne,fn->xfe", H.chol.a.ov, T.a)
    X.a.vv += (
             ccpy_einsum("xae,x->ae", H.chol.a.vv, bt1)
            + ccpy_einsum("xaf,xfe->ae", H.chol.a.vv, bxt1)
    )
    X.a.oo = H.a.oo + (
            ccpy_einsum("mnif,fn->mi", H.aa.ooov, T.a)
            + ccpy_einsum("mnif,fn->mi", H.ab.ooov, T.b)
            + ccpy_einsum("me,ei->mi", X.a.ov, T.a)
            + 0.5 * ccpy_einsum("mnef,efin->mi", H.aa.oovv, T.aa)
            + ccpy_einsum("mnef,efin->mi", H.ab.oovv, T.ab)
    )
    if flag_RHF:
        X.b.ov = X.a.ov.copy()
        X.b.oo = X.a.oo.copy()
        X.b.vv = X.a.vv.copy()
    else:
        X.b.ov = H.b.ov + (
                ccpy_einsum("nmfe,fn->me", H.ab.oovv, T.a)
                + ccpy_einsum("mnef,fn->me", H.bb.oovv, T.b)
        )
        X.b.vv = H.b.vv + (
                - 0.5 * ccpy_einsum("mnef,afmn->ae", H.bb.oovv, T.bb)
                - ccpy_einsum("nmfe,fanm->ae", H.ab.oovv, T.ab)
        )
        bt1 = (
                ccpy_einsum("xnf,fn->x", H.chol.b.ov, T.b)
              + ccpy_einsum("xnf,fn->x", H.chol.a.ov, T.a)
        )
        bxt1 = -ccpy_einsum("xne,fn->xfe", H.chol.b.ov, T.b)
        X.b.vv += (
                 ccpy_einsum("xae,x->ae", H.chol.b.vv, bt1)
                + ccpy_einsum("xaf,xfe->ae", H.chol.b.vv, bxt1)
        )
        X.b.oo = H.b.oo + (
                + ccpy_einsum("mnif,fn->mi", H.bb.ooov, T.b)
                + ccpy_einsum("nmfi,fn->mi", H.ab.oovo, T.a)
                + ccpy_einsum("me,ei->mi", X.b.ov, T.b)
                + 0.5 * ccpy_einsum("mnef,efin->mi", H.bb.oovv, T.bb)  #
                + ccpy_einsum("nmfe,feni->mi", H.ab.oovv, T.ab)  #
        )

    # update T1
    T, dT = update_t1a(T, dT, X, H, shift)
    if flag_RHF:
        T.b = T.a.copy()
        dT.b = dT.a.copy()
    else:
        T, dT = update_t1b(T, dT, X, H, shift)

    # Adjust (vv) intermediates
    X.a.vv -= ccpy_einsum("me,am->ae", X.a.ov, T.a)
    if flag_RHF:
        X.b.vv = X.a.vv.copy()
    else:
        X.b.vv -= ccpy_einsum("me,am->ae", X.b.ov, T.b)
    # T1-transform Cholesky vectors
    X.chol.a.ov = H.chol.a.ov.copy()
    X.chol.a.oo = H.chol.a.oo.copy() + ccpy_einsum("xme,ei->xmi", X.chol.a.ov, T.a)
    X.chol.a.vv = H.chol.a.vv.copy() - ccpy_einsum("xme,am->xae", X.chol.a.ov, T.a)
    X.chol.a.vo = (
            H.chol.a.vo.copy()
            - ccpy_einsum("xmi,am->xai", X.chol.a.oo, T.a)
            + ccpy_einsum("xae,ei->xai", X.chol.a.vv, T.a)
            + ccpy_einsum("xme,ei,am->xai", X.chol.a.ov, T.a, T.a)
    )
    if flag_RHF:
        X.chol.b.ov = X.chol.a.ov.copy()
        X.chol.b.oo = X.chol.a.oo.copy()
        X.chol.b.vv = X.chol.a.vv.copy()
        X.chol.b.vo = X.chol.a.vo.copy()
    else:
        X.chol.b.ov = H.chol.b.ov.copy()
        X.chol.b.oo = H.chol.b.oo.copy() + ccpy_einsum("xme,ei->xmi", X.chol.b.ov, T.b)
        X.chol.b.vv = H.chol.b.vv.copy() - ccpy_einsum("xme,am->xae", X.chol.b.ov, T.b)
        X.chol.b.vo = (
                H.chol.b.vo.copy()
                - ccpy_einsum("xmi,am->xai", X.chol.b.oo, T.b)
                + ccpy_einsum("xae,ei->xai", X.chol.b.vv, T.b)
                + ccpy_einsum("xme,ei,am->xai", X.chol.b.ov, T.b, T.b)
        )

    # update T2
    T, dT = update_t2a(T, dT, X, H, shift)
    T, dT = update_t2b(T, dT, X, H, shift)
    if flag_RHF:
        T.bb = T.aa.copy()
        dT.bb = dT.aa.copy()
    else:
        T, dT = update_t2c(T, dT, X, H, shift)
    return T, dT

# @profile
def update_t1a(T, dT, X, H, shift):
    """
    Update t1a amplitudes by calculating the projection <ia|(H_N e^(T1+T2))_C|0>.
    """
    dT.a = -ccpy_einsum("mi,am->ai", X.a.oo, T.a)
    dT.a += ccpy_einsum("ae,ei->ai", X.a.vv, T.a)
    dT.a += ccpy_einsum("me,aeim->ai", X.a.ov, T.aa) # [+]
    dT.a += ccpy_einsum("me,aeim->ai", X.b.ov, T.ab) # [+]
    dT.a += ccpy_einsum("anif,fn->ai", H.aa.voov, T.a)
    dT.a += ccpy_einsum("anif,fn->ai", H.ab.voov, T.b)
    dT.a -= 0.5 * ccpy_einsum("mnif,afmn->ai", H.aa.ooov, T.aa)
    dT.a -= ccpy_einsum("mnif,afmn->ai", H.ab.ooov, T.ab)
    #
    b_vo = (
              0.5 * ccpy_einsum("xmf,efim->xei", H.chol.a.ov, T.aa)
            - 0.5 * ccpy_einsum("xme,efim->xfi", H.chol.a.ov, T.aa)
            + ccpy_einsum("xnf,efin->xei", H.chol.b.ov, T.ab)
    )
    dT.a += ccpy_einsum("xae,xei->ai", H.chol.a.vv, b_vo)
    #
    T.a, dT.a = cc_loops2.update_t1a(
        T.a, dT.a + H.a.vo, H.a.oo, H.a.vv, shift
    )
    return T, dT

# @profile
def update_t1b(T, dT, X, H, shift):
    """
    Update t1b amplitudes by calculating the projection <i~a~|(H_N e^(T1+T2))_C|0>.
    """
    dT.b = -ccpy_einsum("mi,am->ai", X.b.oo, T.b)
    dT.b += ccpy_einsum("ae,ei->ai", X.b.vv, T.b)
    dT.b += ccpy_einsum("anif,fn->ai", H.bb.voov, T.b)
    dT.b += ccpy_einsum("nafi,fn->ai", H.ab.ovvo, T.a)
    dT.b += ccpy_einsum("me,eami->ai", X.a.ov, T.ab)
    dT.b += ccpy_einsum("me,aeim->ai", X.b.ov, T.bb)
    dT.b -= 0.5 * ccpy_einsum("mnif,afmn->ai", H.bb.ooov, T.bb)
    dT.b -= ccpy_einsum("nmfi,fanm->ai", H.ab.oovo, T.ab)
    #
    b_vo = (
          0.5 * ccpy_einsum("xnf,efin->xei", H.chol.b.ov, T.bb)
        - 0.5 * ccpy_einsum("xne,efin->xfi", H.chol.b.ov, T.bb)
        + ccpy_einsum("xnf,feni->xei", H.chol.a.ov, T.ab)
    )
    dT.b += ccpy_einsum("xae,xei->ai", H.chol.b.vv, b_vo)
    #
    T.b, dT.b = cc_loops2.update_t1b(
        T.b, dT.b + H.b.vo, H.b.oo, H.b.vv, shift
    )
    return T, dT

# @profile
def update_t2a(T, dT, X, H, shift):
    """
    Update t2a amplitudes by calculating the projection <ijab|(H_N e^(T1+T2))_C|0>.
    """
    h2a_oooo = (
            ccpy_einsum("xmi,xnj->mnij", X.chol.a.oo, X.chol.a.oo)
            - ccpy_einsum("xmj,xni->mnij", X.chol.a.oo, X.chol.a.oo)
            + 0.5 * ccpy_einsum("mnef,efij->mnij", H.aa.oovv, T.aa)
    )
    h2a_voov = (
            ccpy_einsum("xai,xme->amie", X.chol.a.vo, X.chol.a.ov)
            - ccpy_einsum("xae,xmi->amie", X.chol.a.vv, X.chol.a.oo)
            + 0.5 * ccpy_einsum("mnef,afin->amie", H.aa.oovv, T.aa)
            + ccpy_einsum("mnef,afin->amie", H.ab.oovv, T.ab)
    )
    h2b_voov = (
            ccpy_einsum("xai,xme->amie", X.chol.a.vo, X.chol.b.ov)
            + 0.5 * ccpy_einsum("mnef,afin->amie", H.bb.oovv, T.ab)
    )
    # save some voov intermediates for T2B update
    X.aa.voov = h2a_voov
    X.ab.voov = h2b_voov
    # <abij|H(1)|0>
    dT.aa = 0.5 * ccpy_einsum("xai,xbj->abij", X.chol.a.vo, X.chol.a.vo)
    # <abij|[H(1)*T2]_C|0>
    dT.aa -= 0.5 * ccpy_einsum("mi,abmj->abij", X.a.oo, T.aa)
    dT.aa += 0.5 * ccpy_einsum("ae,ebij->abij", X.a.vv, T.aa)
    dT.aa += ccpy_einsum("amie,ebmj->abij", h2a_voov, T.aa)
    dT.aa += ccpy_einsum("amie,bejm->abij", h2b_voov, T.ab)
    dT.aa += 0.125 * ccpy_einsum("mnij,abmn->abij", h2a_oooo, T.aa)
    # for a in range(T.a.shape[0]):
    #    for b in range(a + 1, T.a.shape[0]):
    #        # <ab|ef> = <x|ae><x|bf>
    #        batch_ints = build_2index_batch_vvvv_aa(a, b, X)
    #        dT.aa[a, b, :, :] += 0.25 * ccpy_einsum("ef,efij->ij", batch_ints, T.aa)
    tmp = vvvv_contraction.vvvv_t2_sym(X.chol.a.vv.transpose(0, 2, 1), 0.5 * T.aa.transpose(3, 2, 1, 0))
    dT.aa += tmp.transpose(3, 2, 1, 0)
    T.aa, dT.aa = cc_loops2.update_t2a(
        T.aa, dT.aa, H.a.oo, H.a.vv, shift
    )
    return T, dT

# @profile
def update_t2b(T, dT, X, H, shift):
    """
    Update t2b amplitudes by calculating the projection <ij~ab~|(H_N e^(T1+T2))_C|0>.
    """
    # h2a_voov = (
    #         ccpy_einsum("xai,xme->amie", X.chol.a.vo, X.chol.a.ov)
    #         - ccpy_einsum("xae,xmi->amie", X.chol.a.vv, X.chol.a.oo)
    #         + 0.5 * ccpy_einsum("mnef,afin->amie", H.aa.oovv, T.aa)
    #         + ccpy_einsum("mnef,afin->amie", H.ab.oovv, T.ab)
    # )
    # h2b_voov = (
    #         ccpy_einsum("xai,xme->amie", X.chol.a.vo, X.chol.b.ov)
    #         + 0.5 * ccpy_einsum("mnef,afin->amie", H.bb.oovv, T.ab)
    # )
    X.aa.voov += 0.5 * ccpy_einsum("mnef,aeim->anif", H.aa.oovv, T.aa)
    X.ab.voov += (
            0.5 * ccpy_einsum("mnef,aeim->anif", H.bb.oovv, T.ab)
            + ccpy_einsum("mnef,aeim->anif", H.ab.oovv, T.aa)
    )
    h2b_oooo = (
            ccpy_einsum("xmi,xnj->mnij", X.chol.a.oo, X.chol.b.oo)
            + ccpy_einsum("mnef,efij->mnij", H.ab.oovv, T.ab)
    )
    h2b_ovvo = (
        ccpy_einsum("xme,xai->maei", X.chol.a.ov, X.chol.b.vo)
    )
    h2b_vovo = (
            ccpy_einsum("xae,xmi->amei", X.chol.a.vv, X.chol.b.oo)
            - ccpy_einsum("mnef,afmj->anej", H.ab.oovv, T.ab)
    )
    h2b_ovov = (
        ccpy_einsum("xmj,xbe->mbje", X.chol.a.oo, X.chol.b.vv)
    )
    h2c_voov = (
            ccpy_einsum("xai,xme->amie", X.chol.b.vo, X.chol.b.ov)
            - ccpy_einsum("xae,xmi->amie", X.chol.b.vv, X.chol.b.oo)
    )
    # <ab~ij~|H(1)|0>
    dT.ab = ccpy_einsum("xai,xbj->abij", X.chol.a.vo, X.chol.b.vo)
    # <ab~ij~|[H(1)*T2]_C|0>
    dT.ab += ccpy_einsum("ae,ebij->abij", X.a.vv, T.ab)
    dT.ab += ccpy_einsum("be,aeij->abij", X.b.vv, T.ab)
    dT.ab -= ccpy_einsum("mi,abmj->abij", X.a.oo, T.ab)
    dT.ab -= ccpy_einsum("mj,abim->abij", X.b.oo, T.ab)
    dT.ab += ccpy_einsum("amie,ebmj->abij", X.aa.voov, T.ab)
    dT.ab += ccpy_einsum("amie,ebmj->abij", X.ab.voov, T.bb)
    dT.ab += ccpy_einsum("mbej,aeim->abij", h2b_ovvo, T.aa)
    dT.ab += ccpy_einsum("bmje,aeim->abij", h2c_voov, T.ab)
    dT.ab -= ccpy_einsum("mbie,aemj->abij", h2b_ovov, T.ab)
    dT.ab -= ccpy_einsum("amej,ebim->abij", h2b_vovo, T.ab)
    dT.ab += ccpy_einsum("mnij,abmn->abij", h2b_oooo, T.ab)
    # the one-loop Python is faster than the two-loop Fortran
    # for a in range(T.a.shape[0]):
    #     batch_ints = build_3index_batch_vvvv_ab(a, X)
    #     dT.ab[a, :, :, :] += ccpy_einsum("bef,efij->bij", batch_ints, T.ab)
    tmp = vvvv_contraction.vvvv_t2(X.chol.a.vv.transpose(0, 2, 1), X.chol.b.vv.transpose(0, 2, 1), T.ab.transpose(3, 2, 1, 0))
    dT.ab += tmp.transpose(3, 2, 1, 0)
    # dT.ab = _contract_vvvv_ab(dT.ab, T.ab, X.chol.a.vv, X.chol.b.vv)
    T.ab, dT.ab = cc_loops2.update_t2b(
        T.ab, dT.ab, H.a.oo, H.a.vv, H.b.oo, H.b.vv, shift
    )
    return T, dT

# @profile
def update_t2c(T, dT, X, H, shift):
    """
    Update t2c amplitudes by calculating the projection <i~j~a~b~|(H_N e^(T1+T2))_C|0>.
    """
    h2c_oooo = (
            ccpy_einsum("xmi,xnj->mnij", X.chol.b.oo, X.chol.b.oo)
            - ccpy_einsum("xmj,xni->mnij", X.chol.b.oo, X.chol.b.oo)
            + 0.5 * ccpy_einsum("mnef,efij->mnij", H.bb.oovv, T.bb)
    )
    h2c_voov = (
            ccpy_einsum("xai,xme->amie", X.chol.b.vo, X.chol.b.ov)
            - ccpy_einsum("xae,xmi->amie", X.chol.b.vv, X.chol.b.oo)
            + 0.5 * ccpy_einsum("mnef,afin->amie", H.bb.oovv, T.bb)
            + ccpy_einsum("nmfe,fani->amie", H.ab.oovv, T.ab)
    )
    h2b_ovvo = (
            ccpy_einsum("xai,xme->maei", X.chol.b.vo, X.chol.a.ov)
            + 0.5 * ccpy_einsum("mnef,fani->maei", H.aa.oovv, T.ab)
    )
    # <abij|H(1)|0>
    dT.bb = 0.5 * ccpy_einsum("xai,xbj->abij", X.chol.b.vo, X.chol.b.vo)
    # <abij|[H(1)*T2]_C|0>
    dT.bb -= 0.5 * ccpy_einsum("mi,abmj->abij", X.b.oo, T.bb)
    dT.bb += 0.5 * ccpy_einsum("ae,ebij->abij", X.b.vv, T.bb)
    dT.bb += ccpy_einsum("amie,ebmj->abij", h2c_voov, T.bb)
    dT.bb += ccpy_einsum("maei,ebmj->abij", h2b_ovvo, T.ab)
    dT.bb += 0.125 * ccpy_einsum("mnij,abmn->abij", h2c_oooo, T.bb)
    # for a in range(T.b.shape[0]):
    #    for b in range(a + 1, T.b.shape[0]):
    #        # <ab|ef> = <x|ae><x|bf>
    #        batch_ints = build_2index_batch_vvvv_bb(a, b, X)
    #        dT.bb[a, b, :, :] += 0.25 * ccpy_einsum("ef,efij->ij", batch_ints, T.bb)
    tmp = vvvv_contraction.vvvv_t2_sym(X.chol.b.vv.transpose(0, 2, 1), 0.5 * T.bb.transpose(3, 2, 1, 0))
    dT.bb += tmp.transpose(3, 2, 1, 0)
    T.bb, dT.bb = cc_loops2.update_t2c(
        T.bb, dT.bb, H.b.oo, H.b.vv, shift
    )
    return T, dT
