"""Module with functions that perform the CC with singles, doubles,
and triples (CCSDT) calculation for a molecular system."""

import numpy as np

from ccpy.hbar.hbar_ccsd_chol import get_ccsd_intermediates
from ccpy.lib.core import cc_loops2, vvvv_contraction

def update(T, dT, H, X, shift, flag_RHF):

    # pre-CCS intermediates
    X.a.ov = H.a.ov + (
            np.einsum("mnef,fn->me", H.aa.oovv, T.a, optimize=True)
            + np.einsum("mnef,fn->me", H.ab.oovv, T.b, optimize=True)
    )
    X.a.vv = H.a.vv + (
            np.einsum("anef,fn->ae", H.aa.vovv, T.a, optimize=True)
            + np.einsum("anef,fn->ae", H.ab.vovv, T.b, optimize=True)
            # - np.einsum("me,am->ae", X.a.ov, T.a, optimize=True)
            - 0.5 * np.einsum("mnef,afmn->ae", H.aa.oovv, T.aa, optimize=True)  #
            - np.einsum("mnef,afmn->ae", H.ab.oovv, T.ab, optimize=True)  #
    )
    X.a.oo = H.a.oo + (
            np.einsum("mnif,fn->mi", H.aa.ooov, T.a, optimize=True)
            + np.einsum("mnif,fn->mi", H.ab.ooov, T.b, optimize=True)
            + np.einsum("me,ei->mi", X.a.ov, T.a, optimize=True)
            + 0.5 * np.einsum("mnef,efin->mi", H.aa.oovv, T.aa, optimize=True)  #
            + np.einsum("mnef,efin->mi", H.ab.oovv, T.ab, optimize=True)  #
    )
    X.b.ov = H.b.ov + (
            np.einsum("nmfe,fn->me", H.ab.oovv, T.a, optimize=True)
            + np.einsum("mnef,fn->me", H.bb.oovv, T.b, optimize=True)
    )
    X.b.vv = H.b.vv + (
            + np.einsum("anef,fn->ae", H.bb.vovv, T.b, optimize=True)
            + np.einsum("nafe,fn->ae", H.ab.ovvv, T.a, optimize=True)
            # - np.einsum("me,am->ae", X.b.ov, T.b, optimize=True)
            - 0.5 * np.einsum("mnef,afmn->ae", H.bb.oovv, T.bb, optimize=True)  #
            - np.einsum("nmfe,fanm->ae", H.ab.oovv, T.ab, optimize=True)  #
    )
    X.b.oo = H.b.oo + (
            + np.einsum("mnif,fn->mi", H.bb.ooov, T.b, optimize=True)
            + np.einsum("nmfi,fn->mi", H.ab.oovo, T.a, optimize=True)
            + np.einsum("me,ei->mi", X.b.ov, T.b, optimize=True)
            + 0.5 * np.einsum("mnef,efin->mi", H.bb.oovv, T.bb, optimize=True)  #
            + np.einsum("nmfe,feni->mi", H.ab.oovv, T.ab, optimize=True)  #
    )

    # update T1
    T, dT = update_t1a(T, dT, H, X, shift)
    if flag_RHF:
        T.b = T.a.copy()
        dT.b = dT.a.copy()
    else:
        T, dT = update_t1b(T, dT, H, X, shift)

    # Adjust (vv) intermediates
    X.a.vv -= np.einsum("me,am->ae", X.a.ov, T.a, optimize=True)
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
    # T3 parts
    dT.a += 0.25 * np.einsum("mnef,aefimn->ai", H.aa.oovv, T.aaa, optimize=True)
    dT.a += np.einsum("mnef,aefimn->ai", H.ab.oovv, T.aab, optimize=True)
    dT.a += 0.25 * np.einsum("mnef,aefimn->ai", H.bb.oovv, T.abb, optimize=True)
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
    # T3 parts
    dT.b += 0.25 * np.einsum("mnef,aefimn->ai", H.bb.oovv, T.bbb, optimize=True)
    dT.b += 0.25 * np.einsum("mnef,efamni->ai", H.aa.oovv, T.aab, optimize=True)
    dT.b += np.einsum("mnef,efamni->ai", H.ab.oovv, T.abb, optimize=True)
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
    h2b_ooov = np.einsum("xmi,xne->mnie", X.chol.a.oo, X.chol.b.ov, optimize=True)
    h2a_ooov = (
                    np.einsum("xmi,xne->mnie", X.chol.a.oo, X.chol.a.ov, optimize=True)
                    - np.einsum("xni,xme->mnie", X.chol.a.oo, X.chol.a.ov, optimize=True)
    )
    h2a_vovv = (
                    np.einsum("xae,xmf->amef", X.chol.a.vv, X.chol.a.ov, optimize=True)
                    - np.einsum("xaf,xme->amef", X.chol.a.vv, X.chol.a.ov, optimize=True)
    )
    h2b_vovv = np.einsum("xae,xmf->amef", X.chol.a.vv, X.chol.b.ov, optimize=True)
    # save some voov intermediates for T2B update
    X.aa.voov = h2a_voov
    X.ab.voov = h2b_voov
    X.ab.ooov = h2b_ooov
    X.aa.ooov = h2a_ooov
    X.aa.vovv = h2a_vovv
    X.ab.vovv = h2b_vovv
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
    tmp = vvvv_contraction.vvvv_contraction.vvvv_t2_sym(X.chol.a.vv.transpose(0, 2, 1), 0.5 * T.aa.transpose(3, 2, 1, 0))
    dT.aa += tmp.transpose(3, 2, 1, 0)
    # T3 parts
    dT.aa += 0.25 * np.einsum("me,abeijm->abij", X.a.ov, T.aaa, optimize=True)
    dT.aa += 0.25 * np.einsum("me,abeijm->abij", X.b.ov, T.aab, optimize=True)
    dT.aa -= 0.5 * np.einsum("mnif,abfmjn->abij", h2b_ooov, T.aab, optimize=True)
    dT.aa -= 0.25 * np.einsum("mnif,abfmjn->abij", h2a_ooov, T.aaa, optimize=True)
    dT.aa += 0.25 * np.einsum("anef,ebfijn->abij", h2a_vovv, T.aaa, optimize=True)
    dT.aa += 0.5 * np.einsum("anef,ebfijn->abij", h2b_vovv, T.aab, optimize=True)
    T.aa, dT.aa = cc_loops2.cc_loops2.update_t2a(
        T.aa, dT.aa, H.a.oo, H.a.vv, shift
    )
    return T, dT

def update_t2b(T, dT, X, H, shift):
    """
    Update t2b amplitudes by calculating the projection <ij~ab~|(H_N exp(T1+T2+T3))_C|0>.
    """
    # intermediates
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
    h2b_oovo = np.einsum("xme,xni->mnei", X.chol.a.ov, X.chol.b.oo, optimize=True)
    h2c_ooov = (
                    np.einsum("xmi,xne->mnie", X.chol.b.oo, X.chol.b.ov, optimize=True)
                    - np.einsum("xni,xme->mnie", X.chol.b.oo, X.chol.b.ov, optimize=True)
    )
    h2b_ovvv = (
                    np.einsum("xmf,xae->mafe", X.chol.a.ov, X.chol.b.vv, optimize=True)
    )
    h2c_vovv = (
                    np.einsum("xae,xmf->amef", X.chol.b.vv, X.chol.b.ov, optimize=True)
                    - np.einsum("xaf,xme->amef", X.chol.b.vv, X.chol.b.ov, optimize=True)
    )
    # save intermediates
    X.ab.ovvv = h2b_ovvv
    X.ab.oovo = h2b_oovo
    X.bb.vovv = h2c_vovv
    X.bb.ooov = h2c_ooov
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
    tmp = vvvv_contraction.vvvv_contraction.vvvv_t2(X.chol.a.vv.transpose(0, 2, 1), X.chol.b.vv.transpose(0, 2, 1), T.ab.transpose(3, 2, 1, 0))
    dT.ab += tmp.transpose(3, 2, 1, 0)
    # T3 parts
    dT.ab -= 0.5 * np.einsum("mnif,afbmnj->abij", X.aa.ooov, T.aab, optimize=True)
    dT.ab -= np.einsum("nmfj,afbinm->abij", h2b_oovo, T.aab, optimize=True)
    dT.ab -= 0.5 * np.einsum("mnjf,afbinm->abij", h2c_ooov, T.abb, optimize=True)
    dT.ab -= np.einsum("mnif,afbmnj->abij", X.ab.ooov, T.abb, optimize=True)
    dT.ab += 0.5 * np.einsum("anef,efbinj->abij", X.aa.vovv, T.aab, optimize=True)
    dT.ab += np.einsum("anef,efbinj->abij", X.ab.vovv, T.abb, optimize=True)
    dT.ab += np.einsum("nbfe,afeinj->abij", h2b_ovvv, T.aab, optimize=True)
    dT.ab += 0.5 * np.einsum("bnef,afeinj->abij", h2c_vovv, T.abb, optimize=True)
    dT.ab += np.einsum("me,aebimj->abij", X.a.ov, T.aab, optimize=True)
    dT.ab += np.einsum("me,aebimj->abij", X.b.ov, T.abb, optimize=True)
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
    tmp = vvvv_contraction.vvvv_contraction.vvvv_t2_sym(X.chol.b.vv.transpose(0, 2, 1), 0.5 * T.bb.transpose(3, 2, 1, 0))
    dT.bb += tmp.transpose(3, 2, 1, 0)
    # T3 parts
    dT.bb += 0.25 * np.einsum("me,eabmij->abij", X.a.ov, T.abb, optimize=True)
    dT.bb += 0.25 * np.einsum("me,abeijm->abij", X.b.ov, T.bbb, optimize=True)
    dT.bb += 0.25 * np.einsum("anef,ebfijn->abij", X.bb.vovv, T.bbb, optimize=True)
    dT.bb += 0.5 * np.einsum("nafe,febnij->abij", X.ab.ovvv, T.abb, optimize=True)
    dT.bb -= 0.25 * np.einsum("mnif,abfmjn->abij", X.bb.ooov, T.bbb, optimize=True)
    dT.bb -= 0.5 * np.einsum("nmfi,fabnmj->abij", X.ab.oovo, T.abb, optimize=True)
    T.bb, dT.bb = cc_loops2.cc_loops2.update_t2c(
        T.bb, dT.bb, H.b.oo, H.b.vv, shift
    )
    return T, dT

def update_t3a(T, dT, H, H0, shift):
    """
    Update t3a amplitudes by calculating the projection <ijkabc|(H_N exp(T1+T2+T3))_C|0>.
    """
    # <ijkabc | H(2) | 0 > + (VT3)_C intermediates
    I2A_vvov = -0.5 * np.einsum("mnef,abfimn->abie", H0.aa.oovv, T.aaa, optimize=True)
    I2A_vvov -= np.einsum("mnef,abfimn->abie", H0.ab.oovv, T.aab, optimize=True)
    I2A_vvov += H.aa.vvov
    
    I2A_vooo = 0.5 * np.einsum("mnef,aefijn->amij", H0.aa.oovv, T.aaa, optimize=True)
    I2A_vooo += H.aa.vooo + np.einsum("mnef,aefijn->amij", H0.ab.oovv, T.aab, optimize=True)
    I2A_vooo -= np.einsum("me,aeij->amij", H.a.ov, T.aa, optimize=True)

    # MM(2,3)A
    dT.aaa = -0.25 * np.einsum("amij,bcmk->abcijk", I2A_vooo, T.aa, optimize=True)
    dT.aaa += 0.25 * np.einsum("abie,ecjk->abcijk", I2A_vvov, T.aa, optimize=True)
    # (HBar*T3)_C
    dT.aaa -= (1.0 / 12.0) * np.einsum("mk,abcijm->abcijk", H.a.oo, T.aaa, optimize=True)
    dT.aaa += (1.0 / 12.0) * np.einsum("ce,abeijk->abcijk", H.a.vv, T.aaa, optimize=True)
    dT.aaa += (1.0 / 24.0) * np.einsum("mnij,abcmnk->abcijk", H.aa.oooo, T.aaa, optimize=True)
    dT.aaa += (1.0 / 24.0) * np.einsum("abef,efcijk->abcijk", H.aa.vvvv, T.aaa, optimize=True)
    dT.aaa += 0.25 * np.einsum("cmke,abeijm->abcijk", H.aa.voov, T.aaa, optimize=True)
    dT.aaa += 0.25 * np.einsum("cmke,abeijm->abcijk", H.ab.voov, T.aab, optimize=True)
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
    I2A_vvov = -0.5 * np.einsum("mnef,abfimn->abie", H0.aa.oovv, T.aaa, optimize=True)
    I2A_vvov += -np.einsum("mnef,abfimn->abie", H0.ab.oovv, T.aab, optimize=True)
    I2A_vvov += H.aa.vvov
    
    I2A_vooo = 0.5 * np.einsum("mnef,aefijn->amij", H0.aa.oovv, T.aaa, optimize=True)
    I2A_vooo += np.einsum("mnef,aefijn->amij", H0.ab.oovv, T.aab, optimize=True)
    I2A_vooo += -np.einsum("me,aeij->amij", H.a.ov, T.aa, optimize=True)
    I2A_vooo += H.aa.vooo

    I2B_vvvo = -0.5 * np.einsum("mnef,afbmnj->abej", H0.aa.oovv, T.aab, optimize=True)
    I2B_vvvo += -np.einsum("mnef,afbmnj->abej", H0.ab.oovv, T.abb, optimize=True)
    I2B_vvvo += H.ab.vvvo
    
    I2B_ovoo = 0.5 * np.einsum("mnef,efbinj->mbij", H0.aa.oovv, T.aab, optimize=True)
    I2B_ovoo += np.einsum("mnef,efbinj->mbij", H0.ab.oovv, T.abb, optimize=True)
    I2B_ovoo += -np.einsum("me,ecjk->mcjk", H.a.ov, T.ab, optimize=True)
    I2B_ovoo += H.ab.ovoo

    I2B_vvov = -np.einsum("nmfe,afbinm->abie", H0.ab.oovv, T.aab, optimize=True)
    I2B_vvov += -0.5 * np.einsum("nmfe,afbinm->abie", H0.bb.oovv, T.abb, optimize=True)
    I2B_vvov += H.ab.vvov
    
    I2B_vooo = np.einsum("nmfe,afeinj->amij", H0.ab.oovv, T.aab, optimize=True)
    I2B_vooo += 0.5 * np.einsum("nmfe,afeinj->amij", H0.bb.oovv, T.abb, optimize=True)
    I2B_vooo += -np.einsum("me,aeik->amik", H.b.ov, T.ab, optimize=True)
    I2B_vooo += H.ab.vooo

    # MM(2,3)B
    dT.aab = 0.5 * np.einsum("bcek,aeij->abcijk", I2B_vvvo, T.aa, optimize=True)
    dT.aab -= 0.5 * np.einsum("mcjk,abim->abcijk", I2B_ovoo, T.aa, optimize=True)
    dT.aab += np.einsum("acie,bejk->abcijk", I2B_vvov, T.ab, optimize=True)
    dT.aab -= np.einsum("amik,bcjm->abcijk", I2B_vooo, T.ab, optimize=True)
    dT.aab += 0.5 * np.einsum("abie,ecjk->abcijk", I2A_vvov, T.ab, optimize=True)
    dT.aab -= 0.5 * np.einsum("amij,bcmk->abcijk", I2A_vooo, T.ab, optimize=True)
    # (HBar*T3)_C
    dT.aab -= 0.5 * np.einsum("mi,abcmjk->abcijk", H.a.oo, T.aab, optimize=True)
    dT.aab -= 0.25 * np.einsum("mk,abcijm->abcijk", H.b.oo, T.aab, optimize=True)
    dT.aab += 0.5 * np.einsum("ae,ebcijk->abcijk", H.a.vv, T.aab, optimize=True)
    dT.aab += 0.25 * np.einsum("ce,abeijk->abcijk", H.b.vv, T.aab, optimize=True)
    dT.aab += 0.125 * np.einsum("mnij,abcmnk->abcijk", H.aa.oooo, T.aab, optimize=True)
    dT.aab += 0.5 * np.einsum("mnjk,abcimn->abcijk", H.ab.oooo, T.aab, optimize=True)
    dT.aab += 0.125 * np.einsum("abef,efcijk->abcijk", H.aa.vvvv, T.aab, optimize=True)
    dT.aab += 0.5 * np.einsum("bcef,aefijk->abcijk", H.ab.vvvv, T.aab, optimize=True)
    dT.aab += np.einsum("amie,ebcmjk->abcijk", H.aa.voov, T.aab, optimize=True)
    dT.aab += np.einsum("amie,becjmk->abcijk", H.ab.voov, T.abb, optimize=True)
    dT.aab += 0.25 * np.einsum("mcek,abeijm->abcijk", H.ab.ovvo, T.aaa, optimize=True)
    dT.aab += 0.25 * np.einsum("cmke,abeijm->abcijk", H.bb.voov, T.aab, optimize=True)
    dT.aab -= 0.5 * np.einsum("amek,ebcijm->abcijk", H.ab.vovo, T.aab, optimize=True)
    dT.aab -= 0.5 * np.einsum("mcie,abemjk->abcijk", H.ab.ovov, T.aab, optimize=True)
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
    I2B_vvvo = -0.5 * np.einsum("mnef,afbmnj->abej", H0.aa.oovv, T.aab, optimize=True)
    I2B_vvvo += -np.einsum("mnef,afbmnj->abej", H0.ab.oovv, T.abb, optimize=True)
    I2B_vvvo += H.ab.vvvo
    
    I2B_ovoo = 0.5 * np.einsum("mnef,efbinj->mbij", H0.aa.oovv, T.aab, optimize=True)
    I2B_ovoo += np.einsum("mnef,efbinj->mbij", H0.ab.oovv, T.abb, optimize=True)
    I2B_ovoo -= np.einsum("me,ebij->mbij", H.a.ov, T.ab, optimize=True)
    I2B_ovoo += H.ab.ovoo

    I2B_vvov = -np.einsum("nmfe,afbinm->abie", H0.ab.oovv, T.aab, optimize=True)
    I2B_vvov += -0.5 * np.einsum("nmfe,afbinm->abie", H0.bb.oovv, T.abb, optimize=True)
    I2B_vvov += H.ab.vvov
    
    I2B_vooo = np.einsum("nmfe,afeinj->amij", H0.ab.oovv, T.aab, optimize=True)
    I2B_vooo += 0.5 * np.einsum("nmfe,afeinj->amij", H0.bb.oovv, T.abb, optimize=True)
    I2B_vooo -= np.einsum("me,aeij->amij", H.b.ov, T.ab, optimize=True)
    I2B_vooo += H.ab.vooo

    I2C_vvov = -0.5 * np.einsum("mnef,abfimn->abie", H0.bb.oovv, T.bbb, optimize=True)
    I2C_vvov += -np.einsum("nmfe,fabnim->abie", H0.ab.oovv, T.abb, optimize=True)
    I2C_vvov += H.bb.vvov
    
    I2C_vooo = np.einsum("nmfe,faenij->amij", H0.ab.oovv, T.abb, optimize=True)
    I2C_vooo += 0.5 * np.einsum("mnef,aefijn->amij", H0.bb.oovv, T.bbb, optimize=True)
    I2C_vooo -= np.einsum("me,cekj->cmkj", H.b.ov, T.bb, optimize=True)
    I2C_vooo += H.bb.vooo

    # MM(2,3)C
    dT.abb = 0.5 * np.einsum("abie,ecjk->abcijk", I2B_vvov, T.bb, optimize=True)
    dT.abb -= 0.5 * np.einsum("amij,bcmk->abcijk", I2B_vooo, T.bb, optimize=True)
    dT.abb += 0.5 * np.einsum("cbke,aeij->abcijk", I2C_vvov, T.ab, optimize=True)
    dT.abb -= 0.5 * np.einsum("cmkj,abim->abcijk", I2C_vooo, T.ab, optimize=True)
    dT.abb += np.einsum("abej,ecik->abcijk", I2B_vvvo, T.ab, optimize=True)
    dT.abb -= np.einsum("mbij,acmk->abcijk", I2B_ovoo, T.ab, optimize=True)
    # (HBar*T3)_C
    dT.abb -= 0.25 * np.einsum("mi,abcmjk->abcijk", H.a.oo, T.abb, optimize=True)
    dT.abb -= 0.5 * np.einsum("mj,abcimk->abcijk", H.b.oo, T.abb, optimize=True)
    dT.abb += 0.25 * np.einsum("ae,ebcijk->abcijk", H.a.vv, T.abb, optimize=True)
    dT.abb += 0.5 * np.einsum("be,aecijk->abcijk", H.b.vv, T.abb, optimize=True)
    dT.abb += 0.125 * np.einsum("mnjk,abcimn->abcijk", H.bb.oooo, T.abb, optimize=True)
    dT.abb += 0.5 * np.einsum("mnij,abcmnk->abcijk", H.ab.oooo, T.abb, optimize=True)
    dT.abb += 0.125 * np.einsum("bcef,aefijk->abcijk", H.bb.vvvv, T.abb, optimize=True)
    dT.abb += 0.5 * np.einsum("abef,efcijk->abcijk", H.ab.vvvv, T.abb, optimize=True)
    dT.abb += 0.25 * np.einsum("amie,ebcmjk->abcijk", H.aa.voov, T.abb, optimize=True)
    dT.abb += 0.25 * np.einsum("amie,ebcmjk->abcijk", H.ab.voov, T.bbb, optimize=True)
    dT.abb += np.einsum("mbej,aecimk->abcijk", H.ab.ovvo, T.aab, optimize=True)
    dT.abb += np.einsum("bmje,aecimk->abcijk", H.bb.voov, T.abb, optimize=True)
    dT.abb -= 0.5 * np.einsum("mbie,aecmjk->abcijk", H.ab.ovov, T.abb, optimize=True)
    dT.abb -= 0.5 * np.einsum("amej,ebcimk->abcijk", H.ab.vovo, T.abb, optimize=True)
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
    I2C_vvov = -0.5 * np.einsum("mnef,abfimn->abie", H0.bb.oovv, T.bbb, optimize=True)
    I2C_vvov -= np.einsum("nmfe,fabnim->abie", H0.ab.oovv, T.abb, optimize=True)
    I2C_vvov += H.bb.vvov
    
    I2C_vooo = 0.5 * np.einsum("mnef,aefijn->amij", H0.bb.oovv, T.bbb, optimize=True)
    I2C_vooo += np.einsum("nmfe,faenij->amij", H0.ab.oovv, T.abb, optimize=True)
    I2C_vooo -= np.einsum("me,aeij->amij", H.b.ov, T.bb, optimize=True)
    I2C_vooo += H.bb.vooo

    # MM(2,3)D
    dT.bbb = -0.25 * np.einsum("amij,bcmk->abcijk", I2C_vooo, T.bb, optimize=True)
    dT.bbb += 0.25 * np.einsum("abie,ecjk->abcijk", I2C_vvov, T.bb, optimize=True)
    # (HBar*T3)_C
    dT.bbb -= (1.0 / 12.0) * np.einsum("mk,abcijm->abcijk", H.b.oo, T.bbb, optimize=True)
    dT.bbb += (1.0 / 12.0) * np.einsum("ce,abeijk->abcijk", H.b.vv, T.bbb, optimize=True)
    dT.bbb += (1.0 / 24.0) * np.einsum("mnij,abcmnk->abcijk", H.bb.oooo, T.bbb, optimize=True)
    dT.bbb += (1.0 / 24.0) * np.einsum("abef,efcijk->abcijk", H.bb.vvvv, T.bbb, optimize=True)
    dT.bbb += 0.25 * np.einsum("maei,ebcmjk->abcijk", H.ab.ovvo, T.abb, optimize=True)
    dT.bbb += 0.25 * np.einsum("amie,ebcmjk->abcijk", H.bb.voov, T.bbb, optimize=True)
    T.bbb, dT.bbb = cc_loops2.cc_loops2.update_t3d_v2(
        T.bbb, 
        dT.bbb, 
        H0.b.oo, 
        H0.b.vv, 
        shift,
    )
    return T, dT
