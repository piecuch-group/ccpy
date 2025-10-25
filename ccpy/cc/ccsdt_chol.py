'''
Coupled-Cluster Method with Singles, Doubles, and Triples (CCSDT)
[This version uses Cholesky decomposition of the two-electron integrals]
'''

import numpy as np
from ccpy.utilities.linear_algebra import ccpy_einsum

from ccpy.hbar.hbar_ccsd_chol import get_ccsd_intermediates
from ccpy.lib.core import cc_loops2, vvvv_contraction

def update(T, dT, H, X, shift, flag_RHF):

    # pre-CCS intermediates
    X.a.ov = H.a.ov + (
            ccpy_einsum("mnef,fn->me", H.aa.oovv, T.a)
            + ccpy_einsum("mnef,fn->me", H.ab.oovv, T.b)
    )
    X.a.vv = H.a.vv + (
            ccpy_einsum("anef,fn->ae", H.aa.vovv, T.a)
            + ccpy_einsum("anef,fn->ae", H.ab.vovv, T.b)
            # - ccpy_einsum("me,am->ae", X.a.ov, T.a)
            - 0.5 * ccpy_einsum("mnef,afmn->ae", H.aa.oovv, T.aa)  #
            - ccpy_einsum("mnef,afmn->ae", H.ab.oovv, T.ab)  #
    )
    X.a.oo = H.a.oo + (
            ccpy_einsum("mnif,fn->mi", H.aa.ooov, T.a)
            + ccpy_einsum("mnif,fn->mi", H.ab.ooov, T.b)
            + ccpy_einsum("me,ei->mi", X.a.ov, T.a)
            + 0.5 * ccpy_einsum("mnef,efin->mi", H.aa.oovv, T.aa)  #
            + ccpy_einsum("mnef,efin->mi", H.ab.oovv, T.ab)  #
    )
    X.b.ov = H.b.ov + (
            ccpy_einsum("nmfe,fn->me", H.ab.oovv, T.a)
            + ccpy_einsum("mnef,fn->me", H.bb.oovv, T.b)
    )
    X.b.vv = H.b.vv + (
            + ccpy_einsum("anef,fn->ae", H.bb.vovv, T.b)
            + ccpy_einsum("nafe,fn->ae", H.ab.ovvv, T.a)
            # - ccpy_einsum("me,am->ae", X.b.ov, T.b)
            - 0.5 * ccpy_einsum("mnef,afmn->ae", H.bb.oovv, T.bb)  #
            - ccpy_einsum("nmfe,fanm->ae", H.ab.oovv, T.ab)  #
    )
    X.b.oo = H.b.oo + (
            + ccpy_einsum("mnif,fn->mi", H.bb.ooov, T.b)
            + ccpy_einsum("nmfi,fn->mi", H.ab.oovo, T.a)
            + ccpy_einsum("me,ei->mi", X.b.ov, T.b)
            + 0.5 * ccpy_einsum("mnef,efin->mi", H.bb.oovv, T.bb)  #
            + ccpy_einsum("nmfe,feni->mi", H.ab.oovv, T.ab)  #
    )

    # update T1
    T, dT = update_t1a(T, dT, H, X, shift)
    if flag_RHF:
        T.b = T.a.copy()
        dT.b = dT.a.copy()
    else:
        T, dT = update_t1b(T, dT, H, X, shift)

    # Adjust (vv) intermediates
    X.a.vv -= ccpy_einsum("me,am->ae", X.a.ov, T.a)
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

    # CCSD intermediates
    #[TODO]: Should accept CCS HBar as input and build only terms with T2 in it
    X = get_ccsd_intermediates(T, X, H, flag_RHF)

    # update T3
    T, dT = update_t3a(T, dT, X, H, shift)
    T, dT = update_t3b(T, dT, X, H, shift)
    if flag_RHF:
        T.abb = np.transpose(T.aab, (2, 1, 0, 5, 4, 3))
        dT.abb = np.transpose(dT.aab, (2, 1, 0, 5, 4, 3))
        T.bbb = T.aaa.copy()
        dT.bbb = dT.aaa.copy()
    else:
        T, dT = update_t3c(T, dT, X, H, shift)
        T, dT = update_t3d(T, dT, X, H, shift)

    return T, dT

def update_t1a(T, dT, H, X, shift):
    """
    Update t1a amplitudes by calculating the projection <ia|(H_N exp(T1+T2+T3))_C|0>.
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
    # T3 parts
    dT.a += 0.25 * ccpy_einsum("mnef,aefimn->ai", H.aa.oovv, T.aaa)
    dT.a += ccpy_einsum("mnef,aefimn->ai", H.ab.oovv, T.aab)
    dT.a += 0.25 * ccpy_einsum("mnef,aefimn->ai", H.bb.oovv, T.abb)
    T.a, dT.a = cc_loops2.cc_loops2.update_t1a(
        T.a,
        dT.a,
        H.a.oo,
        H.a.vv,
        shift,
    )
    return T, dT

def update_t1b(T, dT, H, X, shift):
    """
    Update t1b amplitudes by calculating the projection <i~a~|(H_N exp(T1+T2+T3))_C|0>.
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
    # T3 parts
    dT.b += 0.25 * ccpy_einsum("mnef,aefimn->ai", H.bb.oovv, T.bbb)
    dT.b += 0.25 * ccpy_einsum("mnef,efamni->ai", H.aa.oovv, T.aab)
    dT.b += ccpy_einsum("mnef,efamni->ai", H.ab.oovv, T.abb)
    T.b, dT.b = cc_loops2.cc_loops2.update_t1b(
        T.b,
        dT.b,
        H.b.oo,
        H.b.vv,
        shift,
    )
    return T, dT

def update_t2a(T, dT, X, H, shift):
    """
    Update t2a amplitudes by calculating the projection <ijab|(H_N exp(T1+T2+T3))_C|0>.
    """
    # intermediates
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
    h2b_ooov = ccpy_einsum("xmi,xne->mnie", X.chol.a.oo, X.chol.b.ov)
    h2a_ooov = (
                    ccpy_einsum("xmi,xne->mnie", X.chol.a.oo, X.chol.a.ov)
                    - ccpy_einsum("xni,xme->mnie", X.chol.a.oo, X.chol.a.ov)
    )
    h2a_vovv = (
                    ccpy_einsum("xae,xmf->amef", X.chol.a.vv, X.chol.a.ov)
                    - ccpy_einsum("xaf,xme->amef", X.chol.a.vv, X.chol.a.ov)
    )
    h2b_vovv = ccpy_einsum("xae,xmf->amef", X.chol.a.vv, X.chol.b.ov)
    # save some voov intermediates for T2B update
    X.aa.voov = h2a_voov
    X.ab.voov = h2b_voov
    X.ab.ooov = h2b_ooov
    X.aa.ooov = h2a_ooov
    X.aa.vovv = h2a_vovv
    X.ab.vovv = h2b_vovv
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
    tmp = vvvv_contraction.vvvv_contraction.vvvv_t2_sym(X.chol.a.vv.transpose(0, 2, 1), 0.5 * T.aa.transpose(3, 2, 1, 0))
    dT.aa += tmp.transpose(3, 2, 1, 0)
    # T3 parts
    dT.aa += 0.25 * ccpy_einsum("me,abeijm->abij", X.a.ov, T.aaa)
    dT.aa += 0.25 * ccpy_einsum("me,abeijm->abij", X.b.ov, T.aab)
    dT.aa -= 0.5 * ccpy_einsum("mnif,abfmjn->abij", h2b_ooov, T.aab)
    dT.aa -= 0.25 * ccpy_einsum("mnif,abfmjn->abij", h2a_ooov, T.aaa)
    dT.aa += 0.25 * ccpy_einsum("anef,ebfijn->abij", h2a_vovv, T.aaa)
    dT.aa += 0.5 * ccpy_einsum("anef,ebfijn->abij", h2b_vovv, T.aab)
    T.aa, dT.aa = cc_loops2.cc_loops2.update_t2a(
        T.aa, dT.aa, H.a.oo, H.a.vv, shift
    )
    return T, dT

def update_t2b(T, dT, X, H, shift):
    """
    Update t2b amplitudes by calculating the projection <ij~ab~|(H_N exp(T1+T2+T3))_C|0>.
    """
    # intermediates
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
    h2b_oovo = ccpy_einsum("xme,xni->mnei", X.chol.a.ov, X.chol.b.oo)
    h2c_ooov = (
                    ccpy_einsum("xmi,xne->mnie", X.chol.b.oo, X.chol.b.ov)
                    - ccpy_einsum("xni,xme->mnie", X.chol.b.oo, X.chol.b.ov)
    )
    h2b_ovvv = (
                    ccpy_einsum("xmf,xae->mafe", X.chol.a.ov, X.chol.b.vv)
    )
    h2c_vovv = (
                    ccpy_einsum("xae,xmf->amef", X.chol.b.vv, X.chol.b.ov)
                    - ccpy_einsum("xaf,xme->amef", X.chol.b.vv, X.chol.b.ov)
    )
    # save intermediates
    X.ab.ovvv = h2b_ovvv
    X.ab.oovo = h2b_oovo
    X.bb.vovv = h2c_vovv
    X.bb.ooov = h2c_ooov
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
    tmp = vvvv_contraction.vvvv_contraction.vvvv_t2(X.chol.a.vv.transpose(0, 2, 1), X.chol.b.vv.transpose(0, 2, 1), T.ab.transpose(3, 2, 1, 0))
    dT.ab += tmp.transpose(3, 2, 1, 0)
    # T3 parts
    dT.ab -= 0.5 * ccpy_einsum("mnif,afbmnj->abij", X.aa.ooov, T.aab)
    dT.ab -= ccpy_einsum("nmfj,afbinm->abij", h2b_oovo, T.aab)
    dT.ab -= 0.5 * ccpy_einsum("mnjf,afbinm->abij", h2c_ooov, T.abb)
    dT.ab -= ccpy_einsum("mnif,afbmnj->abij", X.ab.ooov, T.abb)
    dT.ab += 0.5 * ccpy_einsum("anef,efbinj->abij", X.aa.vovv, T.aab)
    dT.ab += ccpy_einsum("anef,efbinj->abij", X.ab.vovv, T.abb)
    dT.ab += ccpy_einsum("nbfe,afeinj->abij", h2b_ovvv, T.aab)
    dT.ab += 0.5 * ccpy_einsum("bnef,afeinj->abij", h2c_vovv, T.abb)
    dT.ab += ccpy_einsum("me,aebimj->abij", X.a.ov, T.aab)
    dT.ab += ccpy_einsum("me,aebimj->abij", X.b.ov, T.abb)
    T.ab, dT.ab = cc_loops2.cc_loops2.update_t2b(
        T.ab, dT.ab, H.a.oo, H.a.vv, H.b.oo, H.b.vv, shift
    )
    return T, dT

def update_t2c(T, dT, X, H, shift):
    """
    Update t2c amplitudes by calculating the projection <i~j~a~b~|(H_N exp(T1+T2+T3))_C|0>.
    """
    # intermediates
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
    tmp = vvvv_contraction.vvvv_contraction.vvvv_t2_sym(X.chol.b.vv.transpose(0, 2, 1), 0.5 * T.bb.transpose(3, 2, 1, 0))
    dT.bb += tmp.transpose(3, 2, 1, 0)
    # T3 parts
    dT.bb += 0.25 * ccpy_einsum("me,eabmij->abij", X.a.ov, T.abb)
    dT.bb += 0.25 * ccpy_einsum("me,abeijm->abij", X.b.ov, T.bbb)
    dT.bb += 0.25 * ccpy_einsum("anef,ebfijn->abij", X.bb.vovv, T.bbb)
    dT.bb += 0.5 * ccpy_einsum("nafe,febnij->abij", X.ab.ovvv, T.abb)
    dT.bb -= 0.25 * ccpy_einsum("mnif,abfmjn->abij", X.bb.ooov, T.bbb)
    dT.bb -= 0.5 * ccpy_einsum("nmfi,fabnmj->abij", X.ab.oovo, T.abb)
    T.bb, dT.bb = cc_loops2.cc_loops2.update_t2c(
        T.bb, dT.bb, H.b.oo, H.b.vv, shift
    )
    return T, dT

def update_t3a(T, dT, H, H0, shift):
    """
    Update t3a amplitudes by calculating the projection <ijkabc|(H_N exp(T1+T2+T3))_C|0>.
    """
    # <ijkabc | H(2) | 0 > + (VT3)_C intermediates
    I2A_vvov = -0.5 * ccpy_einsum("mnef,abfimn->abie", H0.aa.oovv, T.aaa)
    I2A_vvov -= ccpy_einsum("mnef,abfimn->abie", H0.ab.oovv, T.aab)
    I2A_vvov += H.aa.vvov
    
    I2A_vooo = 0.5 * ccpy_einsum("mnef,aefijn->amij", H0.aa.oovv, T.aaa)
    I2A_vooo += H.aa.vooo + ccpy_einsum("mnef,aefijn->amij", H0.ab.oovv, T.aab)
    I2A_vooo -= ccpy_einsum("me,aeij->amij", H.a.ov, T.aa)

    # MM(2,3)A
    dT.aaa = -0.25 * ccpy_einsum("amij,bcmk->abcijk", I2A_vooo, T.aa)
    dT.aaa += 0.25 * ccpy_einsum("abie,ecjk->abcijk", I2A_vvov, T.aa)
    # (HBar*T3)_C
    dT.aaa -= (1.0 / 12.0) * ccpy_einsum("mk,abcijm->abcijk", H.a.oo, T.aaa)
    dT.aaa += (1.0 / 12.0) * ccpy_einsum("ce,abeijk->abcijk", H.a.vv, T.aaa)
    dT.aaa += (1.0 / 24.0) * ccpy_einsum("mnij,abcmnk->abcijk", H.aa.oooo, T.aaa)
    dT.aaa += (1.0 / 24.0) * ccpy_einsum("abef,efcijk->abcijk", H.aa.vvvv, T.aaa)
    dT.aaa += 0.25 * ccpy_einsum("cmke,abeijm->abcijk", H.aa.voov, T.aaa)
    dT.aaa += 0.25 * ccpy_einsum("cmke,abeijm->abcijk", H.ab.voov, T.aab)
    T.aaa, dT.aaa = cc_loops2.cc_loops2.update_t3a_v2(
        T.aaa,
        dT.aaa,
        H0.a.oo,
        H0.a.vv,
        shift,
    )
    return T, dT

def update_t3b(T, dT, H, H0, shift):
    """
    Update t3b amplitudes by calculating the projection <ijk~abc~|(H_N exp(T1+T2+T3))_C|0>.
    """
    # <ijk~abc~ | H(2) | 0 > + (VT3)_C intermediates
    I2A_vvov = -0.5 * ccpy_einsum("mnef,abfimn->abie", H0.aa.oovv, T.aaa)
    I2A_vvov += -ccpy_einsum("mnef,abfimn->abie", H0.ab.oovv, T.aab)
    I2A_vvov += H.aa.vvov
    
    I2A_vooo = 0.5 * ccpy_einsum("mnef,aefijn->amij", H0.aa.oovv, T.aaa)
    I2A_vooo += ccpy_einsum("mnef,aefijn->amij", H0.ab.oovv, T.aab)
    I2A_vooo += -ccpy_einsum("me,aeij->amij", H.a.ov, T.aa)
    I2A_vooo += H.aa.vooo

    I2B_vvvo = -0.5 * ccpy_einsum("mnef,afbmnj->abej", H0.aa.oovv, T.aab)
    I2B_vvvo += -ccpy_einsum("mnef,afbmnj->abej", H0.ab.oovv, T.abb)
    I2B_vvvo += H.ab.vvvo
    
    I2B_ovoo = 0.5 * ccpy_einsum("mnef,efbinj->mbij", H0.aa.oovv, T.aab)
    I2B_ovoo += ccpy_einsum("mnef,efbinj->mbij", H0.ab.oovv, T.abb)
    I2B_ovoo += -ccpy_einsum("me,ecjk->mcjk", H.a.ov, T.ab)
    I2B_ovoo += H.ab.ovoo

    I2B_vvov = -ccpy_einsum("nmfe,afbinm->abie", H0.ab.oovv, T.aab)
    I2B_vvov += -0.5 * ccpy_einsum("nmfe,afbinm->abie", H0.bb.oovv, T.abb)
    I2B_vvov += H.ab.vvov
    
    I2B_vooo = ccpy_einsum("nmfe,afeinj->amij", H0.ab.oovv, T.aab)
    I2B_vooo += 0.5 * ccpy_einsum("nmfe,afeinj->amij", H0.bb.oovv, T.abb)
    I2B_vooo += -ccpy_einsum("me,aeik->amik", H.b.ov, T.ab)
    I2B_vooo += H.ab.vooo

    # MM(2,3)B
    dT.aab = 0.5 * ccpy_einsum("bcek,aeij->abcijk", I2B_vvvo, T.aa)
    dT.aab -= 0.5 * ccpy_einsum("mcjk,abim->abcijk", I2B_ovoo, T.aa)
    dT.aab += ccpy_einsum("acie,bejk->abcijk", I2B_vvov, T.ab)
    dT.aab -= ccpy_einsum("amik,bcjm->abcijk", I2B_vooo, T.ab)
    dT.aab += 0.5 * ccpy_einsum("abie,ecjk->abcijk", I2A_vvov, T.ab)
    dT.aab -= 0.5 * ccpy_einsum("amij,bcmk->abcijk", I2A_vooo, T.ab)
    # (HBar*T3)_C
    dT.aab -= 0.5 * ccpy_einsum("mi,abcmjk->abcijk", H.a.oo, T.aab)
    dT.aab -= 0.25 * ccpy_einsum("mk,abcijm->abcijk", H.b.oo, T.aab)
    dT.aab += 0.5 * ccpy_einsum("ae,ebcijk->abcijk", H.a.vv, T.aab)
    dT.aab += 0.25 * ccpy_einsum("ce,abeijk->abcijk", H.b.vv, T.aab)
    dT.aab += 0.125 * ccpy_einsum("mnij,abcmnk->abcijk", H.aa.oooo, T.aab)
    dT.aab += 0.5 * ccpy_einsum("mnjk,abcimn->abcijk", H.ab.oooo, T.aab)
    dT.aab += 0.125 * ccpy_einsum("abef,efcijk->abcijk", H.aa.vvvv, T.aab)
    dT.aab += 0.5 * ccpy_einsum("bcef,aefijk->abcijk", H.ab.vvvv, T.aab)
    dT.aab += ccpy_einsum("amie,ebcmjk->abcijk", H.aa.voov, T.aab)
    dT.aab += ccpy_einsum("amie,becjmk->abcijk", H.ab.voov, T.abb)
    dT.aab += 0.25 * ccpy_einsum("mcek,abeijm->abcijk", H.ab.ovvo, T.aaa)
    dT.aab += 0.25 * ccpy_einsum("cmke,abeijm->abcijk", H.bb.voov, T.aab)
    dT.aab -= 0.5 * ccpy_einsum("amek,ebcijm->abcijk", H.ab.vovo, T.aab)
    dT.aab -= 0.5 * ccpy_einsum("mcie,abemjk->abcijk", H.ab.ovov, T.aab)
    T.aab, dT.aab = cc_loops2.cc_loops2.update_t3b_v2(
        T.aab,
        dT.aab,
        H0.a.oo,
        H0.a.vv,
        H0.b.oo,
        H0.b.vv,
        shift,
    )
    return T, dT

def update_t3c(T, dT, H, H0, shift):
    """
    Update t3c amplitudes by calculating the projection <ij~k~ab~c~|(H_N exp(T1+T2+T3))_C|0>.
    """
    # <ij~k~ab~c~ | H(2) | 0 > + (VT3)_C intermediates
    I2B_vvvo = -0.5 * ccpy_einsum("mnef,afbmnj->abej", H0.aa.oovv, T.aab)
    I2B_vvvo += -ccpy_einsum("mnef,afbmnj->abej", H0.ab.oovv, T.abb)
    I2B_vvvo += H.ab.vvvo
    
    I2B_ovoo = 0.5 * ccpy_einsum("mnef,efbinj->mbij", H0.aa.oovv, T.aab)
    I2B_ovoo += ccpy_einsum("mnef,efbinj->mbij", H0.ab.oovv, T.abb)
    I2B_ovoo -= ccpy_einsum("me,ebij->mbij", H.a.ov, T.ab)
    I2B_ovoo += H.ab.ovoo

    I2B_vvov = -ccpy_einsum("nmfe,afbinm->abie", H0.ab.oovv, T.aab)
    I2B_vvov += -0.5 * ccpy_einsum("nmfe,afbinm->abie", H0.bb.oovv, T.abb)
    I2B_vvov += H.ab.vvov
    
    I2B_vooo = ccpy_einsum("nmfe,afeinj->amij", H0.ab.oovv, T.aab)
    I2B_vooo += 0.5 * ccpy_einsum("nmfe,afeinj->amij", H0.bb.oovv, T.abb)
    I2B_vooo -= ccpy_einsum("me,aeij->amij", H.b.ov, T.ab)
    I2B_vooo += H.ab.vooo

    I2C_vvov = -0.5 * ccpy_einsum("mnef,abfimn->abie", H0.bb.oovv, T.bbb)
    I2C_vvov += -ccpy_einsum("nmfe,fabnim->abie", H0.ab.oovv, T.abb)
    I2C_vvov += H.bb.vvov
    
    I2C_vooo = ccpy_einsum("nmfe,faenij->amij", H0.ab.oovv, T.abb)
    I2C_vooo += 0.5 * ccpy_einsum("mnef,aefijn->amij", H0.bb.oovv, T.bbb)
    I2C_vooo -= ccpy_einsum("me,cekj->cmkj", H.b.ov, T.bb)
    I2C_vooo += H.bb.vooo

    # MM(2,3)C
    dT.abb = 0.5 * ccpy_einsum("abie,ecjk->abcijk", I2B_vvov, T.bb)
    dT.abb -= 0.5 * ccpy_einsum("amij,bcmk->abcijk", I2B_vooo, T.bb)
    dT.abb += 0.5 * ccpy_einsum("cbke,aeij->abcijk", I2C_vvov, T.ab)
    dT.abb -= 0.5 * ccpy_einsum("cmkj,abim->abcijk", I2C_vooo, T.ab)
    dT.abb += ccpy_einsum("abej,ecik->abcijk", I2B_vvvo, T.ab)
    dT.abb -= ccpy_einsum("mbij,acmk->abcijk", I2B_ovoo, T.ab)
    # (HBar*T3)_C
    dT.abb -= 0.25 * ccpy_einsum("mi,abcmjk->abcijk", H.a.oo, T.abb)
    dT.abb -= 0.5 * ccpy_einsum("mj,abcimk->abcijk", H.b.oo, T.abb)
    dT.abb += 0.25 * ccpy_einsum("ae,ebcijk->abcijk", H.a.vv, T.abb)
    dT.abb += 0.5 * ccpy_einsum("be,aecijk->abcijk", H.b.vv, T.abb)
    dT.abb += 0.125 * ccpy_einsum("mnjk,abcimn->abcijk", H.bb.oooo, T.abb)
    dT.abb += 0.5 * ccpy_einsum("mnij,abcmnk->abcijk", H.ab.oooo, T.abb)
    dT.abb += 0.125 * ccpy_einsum("bcef,aefijk->abcijk", H.bb.vvvv, T.abb)
    dT.abb += 0.5 * ccpy_einsum("abef,efcijk->abcijk", H.ab.vvvv, T.abb)
    dT.abb += 0.25 * ccpy_einsum("amie,ebcmjk->abcijk", H.aa.voov, T.abb)
    dT.abb += 0.25 * ccpy_einsum("amie,ebcmjk->abcijk", H.ab.voov, T.bbb)
    dT.abb += ccpy_einsum("mbej,aecimk->abcijk", H.ab.ovvo, T.aab)
    dT.abb += ccpy_einsum("bmje,aecimk->abcijk", H.bb.voov, T.abb)
    dT.abb -= 0.5 * ccpy_einsum("mbie,aecmjk->abcijk", H.ab.ovov, T.abb)
    dT.abb -= 0.5 * ccpy_einsum("amej,ebcimk->abcijk", H.ab.vovo, T.abb)
    T.abb, dT.abb = cc_loops2.cc_loops2.update_t3c_v2(
        T.abb,
        dT.abb,
        H0.a.oo,
        H0.a.vv,
        H0.b.oo,
        H0.b.vv,
        shift,
    )
    return T, dT

def update_t3d(T, dT, H, H0, shift):
    """
    Update t3d amplitudes by calculating the projection <i~j~k~a~b~c~|(H_N exp(T1+T2+T3))_C|0>.
    """
    #  <i~j~k~a~b~c~ | H(2) | 0 > + (VT3)_C intermediates
    I2C_vvov = -0.5 * ccpy_einsum("mnef,abfimn->abie", H0.bb.oovv, T.bbb)
    I2C_vvov -= ccpy_einsum("nmfe,fabnim->abie", H0.ab.oovv, T.abb)
    I2C_vvov += H.bb.vvov
    
    I2C_vooo = 0.5 * ccpy_einsum("mnef,aefijn->amij", H0.bb.oovv, T.bbb)
    I2C_vooo += ccpy_einsum("nmfe,faenij->amij", H0.ab.oovv, T.abb)
    I2C_vooo -= ccpy_einsum("me,aeij->amij", H.b.ov, T.bb)
    I2C_vooo += H.bb.vooo

    # MM(2,3)D
    dT.bbb = -0.25 * ccpy_einsum("amij,bcmk->abcijk", I2C_vooo, T.bb)
    dT.bbb += 0.25 * ccpy_einsum("abie,ecjk->abcijk", I2C_vvov, T.bb)
    # (HBar*T3)_C
    dT.bbb -= (1.0 / 12.0) * ccpy_einsum("mk,abcijm->abcijk", H.b.oo, T.bbb)
    dT.bbb += (1.0 / 12.0) * ccpy_einsum("ce,abeijk->abcijk", H.b.vv, T.bbb)
    dT.bbb += (1.0 / 24.0) * ccpy_einsum("mnij,abcmnk->abcijk", H.bb.oooo, T.bbb)
    dT.bbb += (1.0 / 24.0) * ccpy_einsum("abef,efcijk->abcijk", H.bb.vvvv, T.bbb)
    dT.bbb += 0.25 * ccpy_einsum("maei,ebcmjk->abcijk", H.ab.ovvo, T.abb)
    dT.bbb += 0.25 * ccpy_einsum("amie,ebcmjk->abcijk", H.bb.voov, T.bbb)
    T.bbb, dT.bbb = cc_loops2.cc_loops2.update_t3d_v2(
        T.bbb, 
        dT.bbb, 
        H0.b.oo, 
        H0.b.vv, 
        shift,
    )
    return T, dT
