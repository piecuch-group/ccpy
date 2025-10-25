'''
Coupled-Cluster Method with Singles, Doubles, and an Arbitrary Subset of Triples [CC(P)]
[This version uses Cholesky decomposition of the two-electron integrals]
'''

import numpy as np
from ccpy.utilities.linear_algebra import ccpy_einsum
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
    T, dT = update_t1a(T, dT, H, X, shift, t3_excitations)
    if flag_RHF:
        T.b = T.a.copy()
        dT.b = dT.a.copy()
    else:
        T, dT = update_t1b(T, dT, H, X, shift, t3_excitations)

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
    I2A_vooo = H.aa.vooo - ccpy_einsum("me,aeij->amij", H.a.ov, T.aa)
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
    I2A_vooo = H.aa.vooo - ccpy_einsum("me,aeij->amij", H.a.ov, T.aa)
    I2B_ovoo = H.ab.ovoo - ccpy_einsum("me,ecjk->mcjk", H.a.ov, T.ab)
    I2B_vooo = H.ab.vooo - ccpy_einsum("me,aeik->amik", H.b.ov, T.ab)
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
    I2C_vooo = H.bb.vooo - ccpy_einsum("me,aeij->amij", H.b.ov, T.bb)
    I2B_ovoo = H.ab.ovoo - ccpy_einsum("me,ecjk->mcjk", H.a.ov, T.ab)
    I2B_vooo = H.ab.vooo - ccpy_einsum("me,aeik->amik", H.b.ov, T.ab)
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
    I2C_vooo = H.bb.vooo - ccpy_einsum("me,aeij->amij", H.b.ov, T.bb)
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
