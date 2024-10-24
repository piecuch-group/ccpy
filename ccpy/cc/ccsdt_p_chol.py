"""Module with functions that perform the CC with singles, doubles,
and triples (CCSDT) calculation for a molecular system."""

import numpy as np
# Modules for type checking
from typing import List, Tuple, Dict
from ccpy.models.operators import ClusterOperator
from ccpy.models.system import System
from ccpy.models.integrals import Integral
#
from ccpy.hbar.hbar_ccsd_chol import get_ccsd_intermediates
from ccpy.lib.core import vvvv_contraction
from ccpy.lib.core import ccsdt_p_loops, ccsdt_p_chol_loops

def update(T: ClusterOperator,
           dT: ClusterOperator,
           H: Integral,
           X: Integral,
           shift: float,
           flag_RHF: bool,
           system: System,
           t3_excitations: Dict[str, np.ndarray]) -> Tuple[ClusterOperator, ClusterOperator]:

    # Check for empty spincases in t3 list. Remember that [1., 1., 1., 1., 1., 1.]
    # is defined as the "empty" state in the Fortran modules.
    do_t3 = {"aaa" : True, "aab" : True, "abb" : True, "bbb" : True}
    if np.array_equal(t3_excitations["aaa"][0,:], np.array([1., 1., 1., 1., 1., 1.])):
        do_t3["aaa"] = False
    if np.array_equal(t3_excitations["aab"][0,:], np.array([1., 1., 1., 1., 1., 1.])):
        do_t3["aab"] = False
    if np.array_equal(t3_excitations["abb"][0,:], np.array([1., 1., 1., 1., 1., 1.])):
        do_t3["abb"] = False
    if np.array_equal(t3_excitations["bbb"][0,:], np.array([1., 1., 1., 1., 1., 1.])):
        do_t3["bbb"] = False
    build_hbar = do_t3["aaa"] or do_t3["aab"] or do_t3["abb"] or do_t3["bbb"]

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
    T, dT = update_t1a(T, dT, H, X, shift, t3_excitations)
    if flag_RHF:
        T.b = T.a.copy()
        dT.b = dT.a.copy()
    else:
        T, dT = update_t1b(T, dT, H, X, shift, t3_excitations)

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
    T, dT = update_t2a(T, dT, X, H, shift, t3_excitations)
    T, dT = update_t2b(T, dT, X, H, shift, t3_excitations)
    if flag_RHF:
        T.bb = T.aa.copy()
        dT.bb = dT.aa.copy()
    else:
        T, dT = update_t2c(T, dT, X, H, shift, t3_excitations)

    # CCSD intermediates
    #[TODO]: Should accept CCS HBar as input and build only terms with T2 in it
    if build_hbar:
        X = get_ccsd_intermediates(T, X, H, flag_RHF)
        # Transpose integrals appropriately
        X.a.vv = X.a.vv.T
        X.b.vv = X.b.vv.T
        #
        X.aa.vvov = X.aa.vvov.transpose(3, 0, 1, 2)
        X.ab.vvov = X.ab.vvov.transpose(3, 0, 1, 2)
        X.ab.vvvo = X.ab.vvvo.transpose(2, 0, 1, 3)
        X.bb.vvov = X.bb.vvov.transpose(3, 0, 1, 2)
        #
        X.aa.voov = X.aa.voov.transpose(1, 3, 0, 2)
        X.ab.voov = X.ab.voov.transpose(1, 3, 0, 2)
        X.ab.ovvo = X.ab.ovvo.transpose(0, 2, 1, 3)
        X.ab.vovo = X.ab.vovo.transpose(1, 2, 0, 3)
        X.ab.ovov = X.ab.ovov.transpose(0, 3, 1, 2)
        X.bb.voov = X.bb.voov.transpose(1, 3, 0, 2)

    # update T3
    if do_t3["aaa"]:
        T, dT, t3_excitations = update_t3a(T, dT, X, H, shift, t3_excitations)
    if do_t3["aab"]:
        T, dT, t3_excitations = update_t3b(T, dT, X, H, shift, t3_excitations)
    if flag_RHF:
       T.abb = T.aab.copy()
       dT.abb = dT.aab.copy()
       t3_excitations["abb"] = t3_excitations["aab"][:, np.array([2, 0, 1, 5, 3, 4])]
       T.bbb = T.aaa.copy()
       dT.bbb = dT.aaa.copy()
       t3_excitations["bbb"] = t3_excitations["aaa"].copy()
    else:
        if do_t3["abb"]:
            T, dT, t3_excitations = update_t3c(T, dT, X, H, shift, t3_excitations)
        if do_t3["bbb"]:
            T, dT, t3_excitations = update_t3d(T, dT, X, H, shift, t3_excitations)

    return T, dT

def update_t1a(T, dT, H, X, shift, t3_excitations):
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
    #
    T.a, dT.a = ccsdt_p_loops.update_t1a(
        T.a,
        dT.a,
        t3_excitations["aaa"], t3_excitations["aab"], t3_excitations["abb"],
        T.aaa, T.aab, T.abb,
        H.aa.oovv, H.ab.oovv, H.bb.oovv,
        H.a.oo, H.a.vv,
        shift
    )
    return T, dT

def update_t1b(T, dT, H, X, shift, t3_excitations):
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
    #
    T.b, dT.b = ccsdt_p_loops.update_t1b(
        T.b,
        dT.b,
        t3_excitations["aab"], t3_excitations["abb"], t3_excitations["bbb"],
        T.aab, T.abb, T.bbb,
        H.aa.oovv, H.ab.oovv, H.bb.oovv,
        H.b.oo, H.b.vv,
        shift
    )
    return T, dT

def update_t2a(T, dT, X, H, shift, t3_excitations):
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
    tmp = vvvv_contraction.vvvv_t2_sym(X.chol.a.vv.transpose(0, 2, 1), 0.5 * T.aa.transpose(3, 2, 1, 0))
    dT.aa += tmp.transpose(3, 2, 1, 0)
    #
    T.aa, dT.aa = ccsdt_p_loops.update_t2a(
        T.aa,
        dT.aa,
        t3_excitations["aaa"], t3_excitations["aab"],
        T.aaa, T.aab,
        X.a.ov, X.b.ov,
        X.aa.ooov, X.aa.vovv,
        X.ab.ooov, X.ab.vovv,
        H.a.oo, H.a.vv,
        shift
    )
    return T, dT

def update_t2b(T, dT, X, H, shift, t3_excitations):
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
    tmp = vvvv_contraction.vvvv_t2(X.chol.a.vv.transpose(0, 2, 1), X.chol.b.vv.transpose(0, 2, 1), T.ab.transpose(3, 2, 1, 0))
    dT.ab += tmp.transpose(3, 2, 1, 0)
    #
    T.ab, dT.ab = ccsdt_p_loops.update_t2b(
        T.ab,
        dT.ab,
        t3_excitations["aab"], t3_excitations["abb"],
        T.aab, T.abb,
        X.a.ov, X.b.ov,
        X.aa.ooov, X.aa.vovv,
        X.ab.ooov, X.ab.oovo, X.ab.vovv, X.ab.ovvv,
        X.bb.ooov, X.bb.vovv,
        H.a.oo, H.a.vv, H.b.oo, H.b.vv,
        shift
    )
    return T, dT

def update_t2c(T, dT, X, H, shift, t3_excitations):
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
    tmp = vvvv_contraction.vvvv_t2_sym(X.chol.b.vv.transpose(0, 2, 1), 0.5 * T.bb.transpose(3, 2, 1, 0))
    dT.bb += tmp.transpose(3, 2, 1, 0)
    #
    T.bb, dT.bb = ccsdt_p_loops.update_t2c(
        T.bb,
        dT.bb,
        t3_excitations["abb"], t3_excitations["bbb"],
        T.abb, T.bbb,
        X.a.ov, X.b.ov,
        X.ab.oovo, X.ab.ovvv,
        X.bb.ooov, X.bb.vovv,
        H.b.oo, H.b.vv,
        shift
    )
    return T, dT

def update_t3a(T, dT, H, H0, shift, t3_excitations):
    """
    Update t3a amplitudes by calculating the projection <ijkabc|(H_N exp(T1+T2+T3))_C|0>.
    """
    I2A_vooo = H.aa.vooo - np.einsum("me,aeij->amij", H.a.ov, T.aa, optimize=True)
    I2A_vooo = I2A_vooo.transpose(1, 0, 2, 3)

    dT.aaa, T.aaa, t3_excitations["aaa"] = ccsdt_p_chol_loops.update_t3a_p(
        T.aaa, t3_excitations["aaa"],
        T.aab, t3_excitations["aab"],
        T.aa,
        H.a.oo, H.a.vv,
        H0.aa.oovv, H.aa.vvov, I2A_vooo,
        H.aa.oooo, H.aa.voov, H.chol.a.vv.transpose(0, 2, 1),
        H0.ab.oovv, H.ab.voov,
        H0.a.oo, H0.a.vv,
        shift
    )
    return T, dT, t3_excitations

def update_t3b(T, dT, H, H0, shift, t3_excitations):
    """
    Update t3b amplitudes by calculating the projection <ijk~abc~|(H_N exp(T1+T2+T3))_C|0>.
    """
    I2A_vooo = H.aa.vooo - np.einsum("me,aeij->amij", H.a.ov, T.aa, optimize=True)
    I2B_ovoo = H.ab.ovoo - np.einsum("me,ecjk->mcjk", H.a.ov, T.ab, optimize=True)
    I2B_vooo = H.ab.vooo - np.einsum("me,aeik->amik", H.b.ov, T.ab, optimize=True)
    I2A_vooo = I2A_vooo.transpose(1, 0, 2, 3)
    I2B_vooo = I2B_vooo.transpose(1, 0, 2, 3)

    dT.aab, T.aab, t3_excitations["aab"] = ccsdt_p_chol_loops.update_t3b_p(
        T.aaa, t3_excitations["aaa"],
        T.aab, t3_excitations["aab"],
        T.abb, t3_excitations["abb"],
        T.aa, T.ab,
        H.a.oo, H.a.vv, H.b.oo, H.b.vv,
        H0.aa.oovv, H.aa.vvov, I2A_vooo, H.aa.oooo, H.aa.voov, H.chol.a.vv.transpose(0, 2, 1),
        H0.ab.oovv, H.ab.vvov, H.ab.vvvo, I2B_vooo, I2B_ovoo,
        H.ab.oooo, H.ab.voov, H.ab.vovo, H.ab.ovov, H.ab.ovvo, H.chol.b.vv.transpose(0, 2, 1),
        H0.bb.oovv, H.bb.voov,
        H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
        shift
    )
    return T, dT, t3_excitations

def update_t3c(T, dT, H, H0, shift, t3_excitations):
    """
    Update t3c amplitudes by calculating the projection <ij~k~ab~c~|(H_N exp(T1+T2+T3))_C|0>.
    """
    I2C_vooo = H.bb.vooo - np.einsum("me,aeij->amij", H.b.ov, T.bb, optimize=True)
    I2B_ovoo = H.ab.ovoo - np.einsum("me,ecjk->mcjk", H.a.ov, T.ab, optimize=True)
    I2B_vooo = H.ab.vooo - np.einsum("me,aeik->amik", H.b.ov, T.ab, optimize=True)
    I2B_vooo = I2B_vooo.transpose(1, 0, 2, 3)
    I2C_vooo = I2C_vooo.transpose(1, 0, 2, 3)

    dT.abb, T.abb, t3_excitations["abb"] = ccsdt_p_chol_loops.update_t3c_p(
        T.aab, t3_excitations["aab"],
        T.abb, t3_excitations["abb"],
        T.bbb, t3_excitations["bbb"],
        T.ab, T.bb,
        H.a.oo, H.a.vv, H.b.oo, H.b.vv,
        H0.aa.oovv, H.aa.voov,
        H0.ab.oovv, I2B_vooo, I2B_ovoo, H.ab.vvov, H.ab.vvvo, H.ab.oooo,
        H.ab.voov, H.ab.vovo, H.ab.ovov, H.ab.ovvo, H.chol.a.vv.transpose(0, 2, 1),
        H0.bb.oovv, I2C_vooo, H.bb.vvov, H.bb.oooo, H.bb.voov, H.chol.b.vv.transpose(0, 2, 1),
        H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
        shift
    )
    return T, dT, t3_excitations

def update_t3d(T, dT, H, H0, shift, t3_excitations):
    """
    Update t3d amplitudes by calculating the projection <i~j~k~a~b~c~|(H_N exp(T1+T2+T3))_C|0>.
    """
    I2C_vooo = H.bb.vooo - np.einsum("me,aeij->amij", H.b.ov, T.bb, optimize=True)
    I2C_vooo = I2C_vooo.transpose(1, 0, 2, 3)

    dT.bbb, T.bbb, t3_excitations["bbb"] = ccsdt_p_chol_loops.update_t3d_p(
        T.abb, t3_excitations["abb"],
        T.bbb, t3_excitations["bbb"],
        T.bb,
        H.b.oo, H.b.vv,
        H0.bb.oovv, H.bb.vvov, I2C_vooo,
        H.bb.oooo, H.bb.voov, H.chol.b.vv.transpose(0, 2, 1),
        H0.ab.oovv, H.ab.ovvo,
        H0.b.oo, H0.b.vv,
        shift
    )
    return T, dT, t3_excitations
