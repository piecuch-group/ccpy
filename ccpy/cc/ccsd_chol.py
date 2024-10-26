"""
Module with functions to perform the coupled-cluster (CC) approach with singles and 
doubles, abbreviated as CCSD.

References:
    [1] G. D. Purvis and R. J. Bartlett, J. Chem. Phys. 76, 1910 (1982). 
    [2] J. M. Cullen and M. C. Zerner, J. Chem. Phys. 77, 4088 (1982).
    [3] G. E. Scuseria, A. C. Scheiner, T. J. Lee, J. E. Rice, and H. F. Schaefer, J. Chem. Phys. 86, 2881 (1987). 
    [4] P. Piecuch and J. Paldus, Int. J. Quantum Chem. 36, 429 (1989).
"""
import numpy as np
from ccpy.lib.core import cc_loops2, vvvv_contraction

# @profile
def update(T, dT, H, X, shift, flag_RHF, system):

    # pre-CCS intermediates
    X.a.ov = H.a.ov + (
            np.einsum("mnef,fn->me", H.aa.oovv, T.a, optimize=True)
            + np.einsum("mnef,fn->me", H.ab.oovv, T.b, optimize=True)
    )
    X.a.vv = H.a.vv + (
            - 0.5 * np.einsum("mnef,afmn->ae", H.aa.oovv, T.aa, optimize=True)  #
            - np.einsum("mnef,afmn->ae", H.ab.oovv, T.ab, optimize=True)  #
    )
    bt1 = (
            np.einsum("xnf,fn->x", H.chol.a.ov, T.a, optimize=True)
          + np.einsum("xnf,fn->x", H.chol.b.ov, T.b, optimize=True)
    )
    bxt1 = -np.einsum("xne,fn->xfe", H.chol.a.ov, T.a, optimize=True)
    X.a.vv += (
             np.einsum("xae,x->ae", H.chol.a.vv, bt1, optimize=True)
            + np.einsum("xaf,xfe->ae", H.chol.a.vv, bxt1, optimize=True)
    )
    X.a.oo = H.a.oo + (
            np.einsum("mnif,fn->mi", H.aa.ooov, T.a, optimize=True)
            + np.einsum("mnif,fn->mi", H.ab.ooov, T.b, optimize=True)
            + np.einsum("me,ei->mi", X.a.ov, T.a, optimize=True)
            + 0.5 * np.einsum("mnef,efin->mi", H.aa.oovv, T.aa, optimize=True)
            + np.einsum("mnef,efin->mi", H.ab.oovv, T.ab, optimize=True)
    )
    if flag_RHF:
        X.b.ov = X.a.ov.copy()
        X.b.oo = X.a.oo.copy()
        X.b.vv = X.a.vv.copy()
    else:
        X.b.ov = H.b.ov + (
                np.einsum("nmfe,fn->me", H.ab.oovv, T.a, optimize=True)
                + np.einsum("mnef,fn->me", H.bb.oovv, T.b, optimize=True)
        )
        X.b.vv = H.b.vv + (
                - 0.5 * np.einsum("mnef,afmn->ae", H.bb.oovv, T.bb, optimize=True)
                - np.einsum("nmfe,fanm->ae", H.ab.oovv, T.ab, optimize=True)
        )
        bt1 = (
                np.einsum("xnf,fn->x", H.chol.b.ov, T.b, optimize=True)
              + np.einsum("xnf,fn->x", H.chol.a.ov, T.a, optimize=True)
        )
        bxt1 = -np.einsum("xne,fn->xfe", H.chol.b.ov, T.b, optimize=True)
        X.b.vv += (
                 np.einsum("xae,x->ae", H.chol.b.vv, bt1, optimize=True)
                + np.einsum("xaf,xfe->ae", H.chol.b.vv, bxt1, optimize=True)
        )
        X.b.oo = H.b.oo + (
                + np.einsum("mnif,fn->mi", H.bb.ooov, T.b, optimize=True)
                + np.einsum("nmfi,fn->mi", H.ab.oovo, T.a, optimize=True)
                + np.einsum("me,ei->mi", X.b.ov, T.b, optimize=True)
                + 0.5 * np.einsum("mnef,efin->mi", H.bb.oovv, T.bb, optimize=True)  #
                + np.einsum("nmfe,feni->mi", H.ab.oovv, T.ab, optimize=True)  #
        )

    # update T1
    T, dT = update_t1a(T, dT, X, H, shift)
    if flag_RHF:
        T.b = T.a.copy()
        dT.b = dT.a.copy()
    else:
        T, dT = update_t1b(T, dT, X, H, shift)

    # Adjust (vv) intermediates
    X.a.vv -= np.einsum("me,am->ae", X.a.ov, T.a, optimize=True)
    if flag_RHF:
        X.b.vv = X.a.vv.copy()
    else:
        X.b.vv -= np.einsum("me,am->ae", X.b.ov, T.b, optimize=True)
    # T1-transform Cholesky vectors
    X.chol.a.ov = H.chol.a.ov.copy()
    X.chol.a.oo = H.chol.a.oo.copy() + np.einsum("xme,ei->xmi", X.chol.a.ov, T.a, optimize=True)
    X.chol.a.vv = H.chol.a.vv.copy() - np.einsum("xme,am->xae", X.chol.a.ov, T.a, optimize=True)
    X.chol.a.vo = (
            H.chol.a.vo.copy()
            - np.einsum("xmi,am->xai", X.chol.a.oo, T.a, optimize=True)
            + np.einsum("xae,ei->xai", X.chol.a.vv, T.a, optimize=True)
            + np.einsum("xme,ei,am->xai", X.chol.a.ov, T.a, T.a, optimize=True)
    )
    if flag_RHF:
        X.chol.b.ov = X.chol.a.ov.copy()
        X.chol.b.oo = X.chol.a.oo.copy()
        X.chol.b.vv = X.chol.a.vv.copy()
        X.chol.b.vo = X.chol.a.vo.copy()
    else:
        X.chol.b.ov = H.chol.b.ov.copy()
        X.chol.b.oo = H.chol.b.oo.copy() + np.einsum("xme,ei->xmi", X.chol.b.ov, T.b, optimize=True)
        X.chol.b.vv = H.chol.b.vv.copy() - np.einsum("xme,am->xae", X.chol.b.ov, T.b, optimize=True)
        X.chol.b.vo = (
                H.chol.b.vo.copy()
                - np.einsum("xmi,am->xai", X.chol.b.oo, T.b, optimize=True)
                + np.einsum("xae,ei->xai", X.chol.b.vv, T.b, optimize=True)
                + np.einsum("xme,ei,am->xai", X.chol.b.ov, T.b, T.b, optimize=True)
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
    dT.a = -np.einsum("mi,am->ai", X.a.oo, T.a, optimize=True)
    dT.a += np.einsum("ae,ei->ai", X.a.vv, T.a, optimize=True)
    dT.a += np.einsum("me,aeim->ai", X.a.ov, T.aa, optimize=True) # [+]
    dT.a += np.einsum("me,aeim->ai", X.b.ov, T.ab, optimize=True) # [+]
    dT.a += np.einsum("anif,fn->ai", H.aa.voov, T.a, optimize=True)
    dT.a += np.einsum("anif,fn->ai", H.ab.voov, T.b, optimize=True)
    dT.a -= 0.5 * np.einsum("mnif,afmn->ai", H.aa.ooov, T.aa, optimize=True)
    dT.a -= np.einsum("mnif,afmn->ai", H.ab.ooov, T.ab, optimize=True)
    #
    b_vo = (
              0.5 * np.einsum("xmf,efim->xei", H.chol.a.ov, T.aa, optimize=True)
            - 0.5 * np.einsum("xme,efim->xfi", H.chol.a.ov, T.aa, optimize=True)
            + np.einsum("xnf,efin->xei", H.chol.b.ov, T.ab, optimize=True)
    )
    dT.a += np.einsum("xae,xei->ai", H.chol.a.vv, b_vo, optimize=True)
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
    dT.b = -np.einsum("mi,am->ai", X.b.oo, T.b, optimize=True)
    dT.b += np.einsum("ae,ei->ai", X.b.vv, T.b, optimize=True)
    dT.b += np.einsum("anif,fn->ai", H.bb.voov, T.b, optimize=True)
    dT.b += np.einsum("nafi,fn->ai", H.ab.ovvo, T.a, optimize=True)
    dT.b += np.einsum("me,eami->ai", X.a.ov, T.ab, optimize=True)
    dT.b += np.einsum("me,aeim->ai", X.b.ov, T.bb, optimize=True)
    dT.b -= 0.5 * np.einsum("mnif,afmn->ai", H.bb.ooov, T.bb, optimize=True)
    dT.b -= np.einsum("nmfi,fanm->ai", H.ab.oovo, T.ab, optimize=True)
    #
    b_vo = (
          0.5 * np.einsum("xnf,efin->xei", H.chol.b.ov, T.bb, optimize=True)
        - 0.5 * np.einsum("xne,efin->xfi", H.chol.b.ov, T.bb, optimize=True)
        + np.einsum("xnf,feni->xei", H.chol.a.ov, T.ab, optimize=True)
    )
    dT.b += np.einsum("xae,xei->ai", H.chol.b.vv, b_vo, optimize=True)
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
            np.einsum("xmi,xnj->mnij", X.chol.a.oo, X.chol.a.oo, optimize=True)
            - np.einsum("xmj,xni->mnij", X.chol.a.oo, X.chol.a.oo, optimize=True)
            + 0.5 * np.einsum("mnef,efij->mnij", H.aa.oovv, T.aa, optimize=True)
    )
    h2a_voov = (
            np.einsum("xai,xme->amie", X.chol.a.vo, X.chol.a.ov, optimize=True)
            - np.einsum("xae,xmi->amie", X.chol.a.vv, X.chol.a.oo, optimize=True)
            + 0.5 * np.einsum("mnef,afin->amie", H.aa.oovv, T.aa, optimize=True)
            + np.einsum("mnef,afin->amie", H.ab.oovv, T.ab, optimize=True)
    )
    h2b_voov = (
            np.einsum("xai,xme->amie", X.chol.a.vo, X.chol.b.ov, optimize=True)
            + 0.5 * np.einsum("mnef,afin->amie", H.bb.oovv, T.ab, optimize=True)
    )
    # save some voov intermediates for T2B update
    X.aa.voov = h2a_voov
    X.ab.voov = h2b_voov
    # <abij|H(1)|0>
    dT.aa = 0.5 * np.einsum("xai,xbj->abij", X.chol.a.vo, X.chol.a.vo, optimize=True)
    # <abij|[H(1)*T2]_C|0>
    dT.aa -= 0.5 * np.einsum("mi,abmj->abij", X.a.oo, T.aa, optimize=True)
    dT.aa += 0.5 * np.einsum("ae,ebij->abij", X.a.vv, T.aa, optimize=True)
    dT.aa += np.einsum("amie,ebmj->abij", h2a_voov, T.aa, optimize=True)
    dT.aa += np.einsum("amie,bejm->abij", h2b_voov, T.ab, optimize=True)
    dT.aa += 0.125 * np.einsum("mnij,abmn->abij", h2a_oooo, T.aa, optimize=True)
    # for a in range(T.a.shape[0]):
    #    for b in range(a + 1, T.a.shape[0]):
    #        # <ab|ef> = <x|ae><x|bf>
    #        batch_ints = build_2index_batch_vvvv_aa(a, b, X)
    #        dT.aa[a, b, :, :] += 0.25 * np.einsum("ef,efij->ij", batch_ints, T.aa, optimize=True)
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
    #         np.einsum("xai,xme->amie", X.chol.a.vo, X.chol.a.ov, optimize=True)
    #         - np.einsum("xae,xmi->amie", X.chol.a.vv, X.chol.a.oo, optimize=True)
    #         + 0.5 * np.einsum("mnef,afin->amie", H.aa.oovv, T.aa, optimize=True)
    #         + np.einsum("mnef,afin->amie", H.ab.oovv, T.ab, optimize=True)
    # )
    # h2b_voov = (
    #         np.einsum("xai,xme->amie", X.chol.a.vo, X.chol.b.ov, optimize=True)
    #         + 0.5 * np.einsum("mnef,afin->amie", H.bb.oovv, T.ab, optimize=True)
    # )
    X.aa.voov += 0.5 * np.einsum("mnef,aeim->anif", H.aa.oovv, T.aa, optimize=True)
    X.ab.voov += (
            0.5 * np.einsum("mnef,aeim->anif", H.bb.oovv, T.ab, optimize=True)
            + np.einsum("mnef,aeim->anif", H.ab.oovv, T.aa, optimize=True)
    )
    h2b_oooo = (
            np.einsum("xmi,xnj->mnij", X.chol.a.oo, X.chol.b.oo, optimize=True)
            + np.einsum("mnef,efij->mnij", H.ab.oovv, T.ab, optimize=True)
    )
    h2b_ovvo = (
        np.einsum("xme,xai->maei", X.chol.a.ov, X.chol.b.vo, optimize=True)
    )
    h2b_vovo = (
            np.einsum("xae,xmi->amei", X.chol.a.vv, X.chol.b.oo, optimize=True)
            - np.einsum("mnef,afmj->anej", H.ab.oovv, T.ab, optimize=True)
    )
    h2b_ovov = (
        np.einsum("xmj,xbe->mbje", X.chol.a.oo, X.chol.b.vv, optimize=True)
    )
    h2c_voov = (
            np.einsum("xai,xme->amie", X.chol.b.vo, X.chol.b.ov, optimize=True)
            - np.einsum("xae,xmi->amie", X.chol.b.vv, X.chol.b.oo, optimize=True)
    )
    # <ab~ij~|H(1)|0>
    dT.ab = np.einsum("xai,xbj->abij", X.chol.a.vo, X.chol.b.vo, optimize=True)
    # <ab~ij~|[H(1)*T2]_C|0>
    dT.ab += np.einsum("ae,ebij->abij", X.a.vv, T.ab, optimize=True)
    dT.ab += np.einsum("be,aeij->abij", X.b.vv, T.ab, optimize=True)
    dT.ab -= np.einsum("mi,abmj->abij", X.a.oo, T.ab, optimize=True)
    dT.ab -= np.einsum("mj,abim->abij", X.b.oo, T.ab, optimize=True)
    dT.ab += np.einsum("amie,ebmj->abij", X.aa.voov, T.ab, optimize=True)
    dT.ab += np.einsum("amie,ebmj->abij", X.ab.voov, T.bb, optimize=True)
    dT.ab += np.einsum("mbej,aeim->abij", h2b_ovvo, T.aa, optimize=True)
    dT.ab += np.einsum("bmje,aeim->abij", h2c_voov, T.ab, optimize=True)
    dT.ab -= np.einsum("mbie,aemj->abij", h2b_ovov, T.ab, optimize=True)
    dT.ab -= np.einsum("amej,ebim->abij", h2b_vovo, T.ab, optimize=True)
    dT.ab += np.einsum("mnij,abmn->abij", h2b_oooo, T.ab, optimize=True)
    # the one-loop Python is faster than the two-loop Fortran
    # for a in range(T.a.shape[0]):
    #     batch_ints = build_3index_batch_vvvv_ab(a, X)
    #     dT.ab[a, :, :, :] += np.einsum("bef,efij->bij", batch_ints, T.ab, optimize=True)
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
            np.einsum("xmi,xnj->mnij", X.chol.b.oo, X.chol.b.oo, optimize=True)
            - np.einsum("xmj,xni->mnij", X.chol.b.oo, X.chol.b.oo, optimize=True)
            + 0.5 * np.einsum("mnef,efij->mnij", H.bb.oovv, T.bb, optimize=True)
    )
    h2c_voov = (
            np.einsum("xai,xme->amie", X.chol.b.vo, X.chol.b.ov, optimize=True)
            - np.einsum("xae,xmi->amie", X.chol.b.vv, X.chol.b.oo, optimize=True)
            + 0.5 * np.einsum("mnef,afin->amie", H.bb.oovv, T.bb, optimize=True)
            + np.einsum("nmfe,fani->amie", H.ab.oovv, T.ab, optimize=True)
    )
    h2b_ovvo = (
            np.einsum("xai,xme->maei", X.chol.b.vo, X.chol.a.ov, optimize=True)
            + 0.5 * np.einsum("mnef,fani->maei", H.aa.oovv, T.ab, optimize=True)
    )
    # <abij|H(1)|0>
    dT.bb = 0.5 * np.einsum("xai,xbj->abij", X.chol.b.vo, X.chol.b.vo, optimize=True)
    # <abij|[H(1)*T2]_C|0>
    dT.bb -= 0.5 * np.einsum("mi,abmj->abij", X.b.oo, T.bb, optimize=True)
    dT.bb += 0.5 * np.einsum("ae,ebij->abij", X.b.vv, T.bb, optimize=True)
    dT.bb += np.einsum("amie,ebmj->abij", h2c_voov, T.bb, optimize=True)
    dT.bb += np.einsum("maei,ebmj->abij", h2b_ovvo, T.ab, optimize=True)
    dT.bb += 0.125 * np.einsum("mnij,abmn->abij", h2c_oooo, T.bb, optimize=True)
    # for a in range(T.b.shape[0]):
    #    for b in range(a + 1, T.b.shape[0]):
    #        # <ab|ef> = <x|ae><x|bf>
    #        batch_ints = build_2index_batch_vvvv_bb(a, b, X)
    #        dT.bb[a, b, :, :] += 0.25 * np.einsum("ef,efij->ij", batch_ints, T.bb, optimize=True)
    tmp = vvvv_contraction.vvvv_t2_sym(X.chol.b.vv.transpose(0, 2, 1), 0.5 * T.bb.transpose(3, 2, 1, 0))
    dT.bb += tmp.transpose(3, 2, 1, 0)
    T.bb, dT.bb = cc_loops2.update_t2c(
        T.bb, dT.bb, H.b.oo, H.b.vv, shift
    )
    return T, dT
