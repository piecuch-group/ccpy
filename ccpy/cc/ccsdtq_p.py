'''
Coupled-Cluster Method with Singles, Doubles, and Arbitrary Subsets of Triples and Quadruples [CC(P)]
'''

import numpy as np
# Modules for type checking
from typing import List, Tuple, Dict
from ccpy.models.operators import ClusterOperator
from ccpy.models.system import System
from ccpy.models.integrals import Integral
# Modules for computation
from ccpy.hbar.hbar_ccs import get_pre_ccs_intermediates, get_ccs_intermediates_opt
from ccpy.hbar.hbar_ccsd import get_ccsd_intermediates
from ccpy.lib.core import ccsdtq_p_loops

def update(T: ClusterOperator,
           dT: ClusterOperator,
           H: Integral,
           X: Integral,
           shift: float,
           flag_RHF: bool,
           t3_excitations: Dict[str, np.ndarray],
           t4_excitations: Dict[str, np.ndarray]) -> Tuple[ClusterOperator, ClusterOperator]:
    """
    Performs one update of the CC amplitude equations for the CCSDTQ(P) method.

    Parameters
    ----------
    T : ClusterOperator
        Cluster operator containing T1, T2, and the P-space T3 and T4 components.
    dT : ClusterOperator
        Residual of the CC amplitude equations corresponding to projections onto singles, doubles, and the subsets of
        triples and quadruples included in the P space.
    H : Integral
        Bare Hamiltonian in the normal-ordered form.
    X : Integral
        Intermediates for the CC iterations that are roughly given by a CCSD-like similarity-transformed Hamiltonian.
    shift : float
        Energy denominator shift for stabilizing update in case of strong quasidegeneracy.
    flag_RHF : bool
        Flag to turn on/off RHF symmetry. Doing so skips updating components of T that are equivalent for closed shells.
    t3_excitations : Dict[str, np.ndarray]
        Dictionary with the keys 'aaa', 'aab', 'abb', and 'bbb', corresponding to distinct T3 spincases, which are
        asociated with values given by the 2D Numpy array containing the triple excitations [a, b, c, i, j, k] belonging
        to the P space.
    t4_excitations : Dict[str, np.ndarray]
        Dictionary with the keys 'aaaa', 'aaab', 'aabb', 'abbb', and 'bbbb', corresponding to distinct T4 spincases,
        which are associated with values given by the 2D Numpy array containing the quadruple excitations
        [a, b, c, d, i, j, k, l] belonging to the P space.

    Returns
    -------
    T : ClusterOperator
        Cluster operator with updated T1, T2, T3(P), and T4(P) components.
    dT : ClusterOperator
        Residual of the CCSDTQ(P) amplitude equations corresponding to projections onto singles, doubles, and the
        subsets of triples and quadruples included in the P space.
    """

    # Check for empty spincases in t3 and t4 lists. Remember that [1., 1., 1., 1., 1., 1.]
    # is defined as the "empty" state in the Fortran modules.
    do_t3 = {"aaa" : True, "aab" : True, "abb" : True, "bbb" : True}
    for key in do_t3.keys():
        if np.array_equal(t3_excitations[key][0,:], np.ones(6)):
            do_t3[key] = False
    do_t4 = {"aaaa" : True, "aaab" : True, "aabb" : True, "abbb" : True, "bbbb" : True}
    for key in do_t4.keys():
        if np.array_equal(t4_excitations[key][0,:], np.ones(8)):
            do_t4[key] = False

    X = get_pre_ccs_intermediates(X, T, H, flag_RHF)

    # update T1
    T, dT = update_t1a(T, dT, H, X, shift, t3_excitations)
    if flag_RHF:
        T.b = T.a.copy()
        dT.b = dT.a.copy()
    else:
        T, dT = update_t1b(T, dT, H, X, shift, t3_excitations)

    # CCS intermediates
    X = get_ccs_intermediates_opt(X, T, H, flag_RHF)

    # update T2
    T, dT = update_t2a(T, dT, X, H, shift, t3_excitations, t4_excitations)
    T, dT = update_t2b(T, dT, X, H, shift, t3_excitations, t4_excitations)
    if flag_RHF:
        T.bb = T.aa.copy()
        dT.bb = dT.aa.copy()
    else:
        T, dT = update_t2c(T, dT, X, H, shift, t3_excitations, t4_excitations)

    # CCSD intermediates
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
    X.aa.vvvv = X.aa.vvvv.transpose(3, 2, 1, 0)
    X.bb.vvvv = X.bb.vvvv.transpose(3, 2, 1, 0)

    # update T3
    if do_t3["aaa"]:
        T, dT, t3_excitations = update_t3a(T, dT, X, H, shift, t3_excitations, t4_excitations)
    if do_t3["aab"]:
        T, dT, t3_excitations = update_t3b(T, dT, X, H, shift, t3_excitations, t4_excitations)
    if flag_RHF:
        T.abb = T.aab.copy()
        dT.abb = dT.aab.copy()
        t3_excitations["abb"] = t3_excitations["aab"][:, np.array([2, 0, 1, 5, 3, 4])]
        T.bbb = T.aaa.copy()
        dT.bbb = dT.aaa.copy()
        t3_excitations["bbb"] = t3_excitations["aaa"].copy()
    else:
        if do_t3["abb"]:
            T, dT, t3_excitations = update_t3c(T, dT, X, H, shift, t3_excitations, t4_excitations)
        if do_t3["bbb"]:
            T, dT, t3_excitations = update_t3d(T, dT, X, H, shift, t3_excitations, t4_excitations)

    # Transpose integrals back
    X.a.vv = X.a.vv.T
    X.b.vv = X.b.vv.T
    #
    X.aa.vvov = X.aa.vvov.transpose(1, 2, 3, 0)
    X.ab.vvov = X.ab.vvov.transpose(1, 2, 3, 0)
    X.ab.vvvo = X.ab.vvvo.transpose(1, 2, 0, 3)
    X.bb.vvov = X.bb.vvov.transpose(1, 2, 3, 0)
    #
    X.aa.voov = X.aa.voov.transpose(2, 0, 3, 1)
    X.ab.voov = X.ab.voov.transpose(2, 0, 3, 1)
    X.ab.ovvo = X.ab.ovvo.transpose(0, 2, 1, 3)
    X.ab.vovo = X.ab.vovo.transpose(2, 0, 1, 3)
    X.ab.ovov = X.ab.ovov.transpose(0, 2, 3, 1)
    X.bb.voov = X.bb.voov.transpose(2, 0, 3, 1)
    X.aa.vvvv = X.aa.vvvv.transpose(3, 2, 1, 0)
    X.bb.vvvv = X.bb.vvvv.transpose(3, 2, 1, 0)

    # add VT3 intermediates to two-body part of HBar
    X = add_VT3_intermediates(T, X, H, t3_excitations)

    # update T4
    if do_t4["aaaa"]:
        T, dT, t4_excitations = update_t4a(T, dT, X, H, shift, t3_excitations, t4_excitations)
    if do_t4["aaab"]:
        T, dT, t4_excitations = update_t4b(T, dT, X, H, shift, t3_excitations, t4_excitations)
    if do_t4["aabb"]:
        T, dT, t4_excitations = update_t4c(T, dT, X, H, shift, t3_excitations, t4_excitations)
    if flag_RHF:
        T.abbb = T.aaab.copy()
        dT.abbb = dT.aaab.copy()
        t4_excitations["abbb"] = t4_excitations["aaab"][:, np.array([3, 0, 1, 2, 7, 4, 5, 6])]
        T.bbbb = T.aaaa.copy()
        dT.bbbb = dT.aaaa.copy()
        t4_excitations["bbbb"] = t4_excitations["aaaa"].copy()
    else:
        if do_t4["abbb"]:
            T, dT, t4_excitations = update_t4d(T, dT, X, H, shift, t3_excitations, t4_excitations)
        if do_t4["bbbb"]:
            T, dT, t4_excitations = update_t4e(T, dT, X, H, shift, t3_excitations, t4_excitations)

    return T, dT

def update_t1a(T, dT, H, X, shift, t3_excitations):
    """
    Update t1a amplitudes by calculating the projection <ia|(H_N exp(T1+T2+T3^(P)))_C|0>.
    """
    dT.a = -np.einsum("mi,am->ai", X.a.oo, T.a, optimize=True)
    dT.a += np.einsum("ae,ei->ai", X.a.vv, T.a, optimize=True)
    dT.a += np.einsum("me,aeim->ai", X.a.ov, T.aa, optimize=True) # [+]
    dT.a += np.einsum("me,aeim->ai", X.b.ov, T.ab, optimize=True) # [+]
    dT.a += np.einsum("anif,fn->ai", H.aa.voov, T.a, optimize=True)
    dT.a += np.einsum("anif,fn->ai", H.ab.voov, T.b, optimize=True)
    dT.a -= 0.5 * np.einsum("mnif,afmn->ai", H.aa.ooov, T.aa, optimize=True)
    dT.a -= np.einsum("mnif,afmn->ai", H.ab.ooov, T.ab, optimize=True)
    dT.a += 0.5 * np.einsum("anef,efin->ai", H.aa.vovv, T.aa, optimize=True)
    dT.a += np.einsum("anef,efin->ai", H.ab.vovv, T.ab, optimize=True)
    T.a, dT.a = ccsdtq_p_loops.update_t1a(
        T.a, 
        dT.a + H.a.vo,
        t3_excitations["aaa"], t3_excitations["aab"], t3_excitations["abb"],
        T.aaa, T.aab, T.abb,
        H.aa.oovv, H.ab.oovv, H.bb.oovv,
        H.a.oo, H.a.vv,
        shift
    )
    return T, dT

def update_t1b(T, dT, H, X, shift, t3_excitations):
    """
    Update t1b amplitudes by calculating the projection <i~a~|(H_N e^(T1+T2+t3^(P)))_C|0>.
    """
    dT.b = -np.einsum("mi,am->ai", X.b.oo, T.b, optimize=True)
    dT.b += np.einsum("ae,ei->ai", X.b.vv, T.b, optimize=True)
    dT.b += np.einsum("anif,fn->ai", H.bb.voov, T.b, optimize=True)
    dT.b += np.einsum("nafi,fn->ai", H.ab.ovvo, T.a, optimize=True)
    dT.b += np.einsum("me,eami->ai", X.a.ov, T.ab, optimize=True)
    dT.b += np.einsum("me,aeim->ai", X.b.ov, T.bb, optimize=True)
    dT.b -= 0.5 * np.einsum("mnif,afmn->ai", H.bb.ooov, T.bb, optimize=True)
    dT.b -= np.einsum("nmfi,fanm->ai", H.ab.oovo, T.ab, optimize=True)
    dT.b += 0.5 * np.einsum("anef,efin->ai", H.bb.vovv, T.bb, optimize=True)
    dT.b += np.einsum("nafe,feni->ai", H.ab.ovvv, T.ab, optimize=True)
    T.b, dT.b = ccsdtq_p_loops.update_t1b(
        T.b,
        dT.b + H.b.vo,
        t3_excitations["aab"], t3_excitations["abb"], t3_excitations["bbb"],
        T.aab, T.abb, T.bbb,
        H.aa.oovv, H.ab.oovv, H.bb.oovv,
        H.b.oo, H.b.vv,
        shift
    )
    return T, dT

# @profile
def update_t2a(T, dT, H, H0, shift, t3_excitations, t4_excitations):
    """
    Update t2a amplitudes by calculating the projection <ijab|(H_N exp(T1+T2+t3^(P)))_C|0>.
    """
    # intermediates
    I2A_voov = H.aa.voov + (
        + 0.5 * np.einsum("mnef,afin->amie", H0.aa.oovv, T.aa, optimize=True)
        + np.einsum("mnef,afin->amie", H0.ab.oovv, T.ab, optimize=True)
    )
    I2A_oooo = H.aa.oooo + 0.5 * np.einsum("mnef,efij->mnij", H0.aa.oovv, T.aa, optimize=True)
    I2B_voov = H.ab.voov + 0.5 * np.einsum("mnef,afin->amie", H0.bb.oovv, T.ab, optimize=True)
    I2A_vooo = H.aa.vooo + 0.5*np.einsum('anef,efij->anij', H0.aa.vovv + 0.5 * H.aa.vovv, T.aa, optimize=True)

    tau = 0.5 * T.aa + np.einsum('ai,bj->abij', T.a, T.a, optimize=True)

    dT.aa = -0.5 * np.einsum("amij,bm->abij", I2A_vooo, T.a, optimize=True)
    dT.aa += 0.5 * np.einsum("abie,ej->abij", H.aa.vvov, T.a, optimize=True)
    dT.aa += 0.5 * np.einsum("ae,ebij->abij", H.a.vv, T.aa, optimize=True)
    dT.aa -= 0.5 * np.einsum("mi,abmj->abij", H.a.oo, T.aa, optimize=True)
    dT.aa += np.einsum("amie,ebmj->abij", I2A_voov, T.aa, optimize=True)
    dT.aa += np.einsum("amie,bejm->abij", I2B_voov, T.ab, optimize=True)
    dT.aa += 0.25 * np.einsum("abef,efij->abij", H0.aa.vvvv, tau, optimize=True)
    dT.aa += 0.125 * np.einsum("mnij,abmn->abij", I2A_oooo, T.aa, optimize=True)

    T.aa, dT.aa = ccsdtq_p_loops.update_t2a(
        T.aa,
        dT.aa + 0.25 * H0.aa.vvoo,
        t3_excitations["aaa"], t3_excitations["aab"],
        T.aaa, T.aab,
        H.a.ov, H.b.ov,
        H0.aa.ooov + H.aa.ooov, H0.aa.vovv + H.aa.vovv,
        H0.ab.ooov + H.ab.ooov, H0.ab.vovv + H.ab.vovv,
        H0.a.oo, H0.a.vv,
        T.aaaa, t4_excitations["aaaa"],
        T.aaab, t4_excitations["aaab"],
        T.aabb, t4_excitations["aabb"],
        H0.aa.oovv, H0.ab.oovv, H0.bb.oovv,
        shift
    )
    return T, dT

# @profile
def update_t2b(T, dT, H, H0, shift, t3_excitations, t4_excitations):
    """
    Update t2b amplitudes by calculating the projection <ij~ab~|(H_N exp(T1+T2+t3^(P)))_C|0>.
    """
    # intermediates
    I2A_voov = H.aa.voov + (
        + np.einsum("mnef,aeim->anif", H0.aa.oovv, T.aa, optimize=True)
        + np.einsum("nmfe,aeim->anif", H0.ab.oovv, T.ab, optimize=True)
    )
    I2B_voov = H.ab.voov + (
        + np.einsum("mnef,aeim->anif", H0.ab.oovv, T.aa, optimize=True)
        + np.einsum("mnef,aeim->anif", H0.bb.oovv, T.ab, optimize=True)
    )
    I2B_oooo = H.ab.oooo + np.einsum("mnef,efij->mnij", H0.ab.oovv, T.ab, optimize=True)
    I2B_vovo = H.ab.vovo - np.einsum("mnef,afmj->anej", H0.ab.oovv, T.ab, optimize=True)
    I2B_ovoo = H.ab.ovoo + np.einsum("maef,efij->maij", H0.ab.ovvv + 0.5 * H.ab.ovvv, T.ab, optimize=True)
    I2B_vooo = H.ab.vooo + np.einsum("amef,efij->amij", H0.ab.vovv + 0.5 * H.ab.vovv, T.ab, optimize=True)

    tau = T.ab + np.einsum('ai,bj->abij', T.a, T.b, optimize=True)

    dT.ab = -np.einsum("mbij,am->abij", I2B_ovoo, T.a, optimize=True)
    dT.ab -= np.einsum("amij,bm->abij", I2B_vooo, T.b, optimize=True)
    dT.ab += np.einsum("abej,ei->abij", H.ab.vvvo, T.a, optimize=True)
    dT.ab += np.einsum("abie,ej->abij", H.ab.vvov, T.b, optimize=True)
    dT.ab += np.einsum("ae,ebij->abij", H.a.vv, T.ab, optimize=True)
    dT.ab += np.einsum("be,aeij->abij", H.b.vv, T.ab, optimize=True)
    dT.ab -= np.einsum("mi,abmj->abij", H.a.oo, T.ab, optimize=True)
    dT.ab -= np.einsum("mj,abim->abij", H.b.oo, T.ab, optimize=True)
    dT.ab += np.einsum("amie,ebmj->abij", I2A_voov, T.ab, optimize=True)
    dT.ab += np.einsum("amie,ebmj->abij", I2B_voov, T.bb, optimize=True)
    dT.ab += np.einsum("mbej,aeim->abij", H.ab.ovvo, T.aa, optimize=True)
    dT.ab += np.einsum("bmje,aeim->abij", H.bb.voov, T.ab, optimize=True)
    dT.ab -= np.einsum("mbie,aemj->abij", H.ab.ovov, T.ab, optimize=True)
    dT.ab -= np.einsum("amej,ebim->abij", I2B_vovo, T.ab, optimize=True)
    dT.ab += np.einsum("mnij,abmn->abij", I2B_oooo, T.ab, optimize=True)
    dT.ab += np.einsum("abef,efij->abij", H0.ab.vvvv, tau, optimize=True)

    T.ab, dT.ab = ccsdtq_p_loops.update_t2b(
        T.ab,
        dT.ab + H0.ab.vvoo,
        t3_excitations["aab"], t3_excitations["abb"],
        T.aab, T.abb,
        H.a.ov, H.b.ov,
        H.aa.ooov + H0.aa.ooov, H.aa.vovv + H0.aa.vovv,
        H.ab.ooov + H0.ab.ooov, H.ab.oovo + H0.ab.oovo, H.ab.vovv + H0.ab.vovv, H.ab.ovvv + H0.ab.ovvv,
        H.bb.ooov + H0.bb.ooov, H.bb.vovv + H0.bb.vovv,
        H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
        T.aaab, t4_excitations["aaab"],
        T.aabb, t4_excitations["aabb"],
        T.abbb, t4_excitations["abbb"],
        H0.aa.oovv, H0.ab.oovv, H0.bb.oovv,
        shift
    )

    return T, dT

# @profile
def update_t2c(T, dT, H, H0, shift, t3_excitations, t4_excitations):
    """
    Update t2c amplitudes by calculating the projection <i~j~a~b~|(H_N exp(T1+T2+t3^(P)))_C|0>.
    """
    # intermediates
    I2C_oooo = H.bb.oooo + 0.5 * np.einsum("mnef,efij->mnij", H0.bb.oovv, T.bb, optimize=True)

    I2B_ovvo = H.ab.ovvo + (
        + np.einsum("mnef,afin->maei", H0.ab.oovv, T.bb, optimize=True)
        + 0.5 * np.einsum("mnef,fani->maei", H0.aa.oovv, T.ab, optimize=True)
    )
    I2C_voov = H.bb.voov + 0.5 * np.einsum("mnef,afin->amie", H0.bb.oovv, T.bb, optimize=True)
    I2C_vooo = H.bb.vooo + 0.5 * np.einsum('anef,efij->anij', H0.bb.vovv + 0.5 * H.bb.vovv, T.bb, optimize=True)

    tau = 0.5 * T.bb + np.einsum('ai,bj->abij', T.b, T.b, optimize=True)

    dT.bb = -0.5 * np.einsum("amij,bm->abij", I2C_vooo, T.b, optimize=True)
    dT.bb += 0.5 * np.einsum("abie,ej->abij", H.bb.vvov, T.b, optimize=True)
    dT.bb += 0.5 * np.einsum("ae,ebij->abij", H.b.vv, T.bb, optimize=True)
    dT.bb -= 0.5 * np.einsum("mi,abmj->abij", H.b.oo, T.bb, optimize=True)
    dT.bb += np.einsum("amie,ebmj->abij", I2C_voov, T.bb, optimize=True)
    dT.bb += np.einsum("maei,ebmj->abij", I2B_ovvo, T.ab, optimize=True)
    dT.bb += 0.25 * np.einsum("abef,efij->abij", H0.bb.vvvv, tau, optimize=True)
    dT.bb += 0.125 * np.einsum("mnij,abmn->abij", I2C_oooo, T.bb, optimize=True)

    T.bb, dT.bb = ccsdtq_p_loops.update_t2c(
        T.bb,
        dT.bb + 0.25 * H0.bb.vvoo,
        t3_excitations["abb"], t3_excitations["bbb"],
        T.abb, T.bbb,
        H.a.ov, H.b.ov,
        H0.ab.oovo + H.ab.oovo, H0.ab.ovvv + H.ab.ovvv,
        H0.bb.ooov + H.bb.ooov, H0.bb.vovv + H.bb.vovv,
        H0.b.oo, H0.b.vv,
        T.bbbb, t4_excitations["bbbb"],
        T.abbb, t4_excitations["abbb"],
        T.aabb, t4_excitations["aabb"],
        H0.bb.oovv, H0.ab.oovv, H0.aa.oovv,
        shift
    )
    return T, dT

def update_t3a(T, dT, H, H0, shift, t3_excitations, t4_excitations):
    """
    Update t3a amplitudes by calculating the projection <ijkabc|(H_N exp(T1+T2+T3))_C|0>.
    """
    I2A_vooo = H.aa.vooo - np.einsum("me,aeij->amij", H.a.ov, T.aa, optimize=True)
    I2A_vooo = I2A_vooo.transpose(1, 0, 2, 3)

    dT.aaa, T.aaa, t3_excitations["aaa"] = ccsdtq_p_loops.update_t3a_p(
        T.aaa, t3_excitations["aaa"], 
        T.aab, t3_excitations["aab"],
        T.aa,
        H.a.oo, H.a.vv,
        H0.aa.oovv, H.aa.vvov, I2A_vooo,
        H.aa.oooo, H.aa.voov, H.aa.vvvv,
        H0.ab.oovv, H.ab.voov,
        H0.a.oo, H0.a.vv,
        T.aaaa, t4_excitations["aaaa"],
        T.aaab, t4_excitations["aaab"],
        H.a.ov, H.b.ov, H.aa.vovv, H.ab.vovv,
        H.aa.ooov, H.ab.ooov,
        shift
    )
    return T, dT, t3_excitations

def update_t3b(T, dT, H, H0, shift, t3_excitations, t4_excitations):
    """
    Update t3b amplitudes by calculating the projection <ijk~abc~|(H_N exp(T1+T2+T3))_C|0>.
    """
    I2A_vooo = H.aa.vooo - np.einsum("me,aeij->amij", H.a.ov, T.aa, optimize=True)
    I2B_ovoo = H.ab.ovoo - np.einsum("me,ecjk->mcjk", H.a.ov, T.ab, optimize=True)
    I2B_vooo = H.ab.vooo - np.einsum("me,aeik->amik", H.b.ov, T.ab, optimize=True)
    I2A_vooo = I2A_vooo.transpose(1, 0, 2, 3)
    I2B_vooo = I2B_vooo.transpose(1, 0, 2, 3)

    dT.aab, T.aab, t3_excitations["aab"] = ccsdtq_p_loops.update_t3b_p(
        T.aaa, t3_excitations["aaa"],
        T.aab, t3_excitations["aab"],
        T.abb, t3_excitations["abb"],
        T.aa, T.ab,
        H.a.oo, H.a.vv, H.b.oo, H.b.vv,
        H0.aa.oovv, H.aa.vvov, I2A_vooo, H.aa.oooo, H.aa.voov, H.aa.vvvv,
        H0.ab.oovv, H.ab.vvov, H.ab.vvvo, I2B_vooo, I2B_ovoo, 
        H.ab.oooo, H.ab.voov, H.ab.vovo, H.ab.ovov, H.ab.ovvo, H.ab.vvvv.transpose(3, 2, 1, 0),
        H0.bb.oovv, H.bb.voov,
        H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
        T.aaab, t4_excitations["aaab"],
        T.aabb, t4_excitations["aabb"],
        H.a.ov, H.b.ov, H.aa.ooov, H.ab.ooov,
        H.ab.oovo, H.bb.ooov, H.aa.vovv, H.ab.vovv,
        H.ab.ovvv, H.bb.vovv,
        shift
    )
    return T, dT, t3_excitations

def update_t3c(T, dT, H, H0, shift, t3_excitations, t4_excitations):
    """
    Update t3c amplitudes by calculating the projection <ij~k~ab~c~|(H_N exp(T1+T2+T3))_C|0>.
    """
    I2C_vooo = H.bb.vooo - np.einsum("me,aeij->amij", H.b.ov, T.bb, optimize=True)
    I2B_ovoo = H.ab.ovoo - np.einsum("me,ecjk->mcjk", H.a.ov, T.ab, optimize=True)
    I2B_vooo = H.ab.vooo - np.einsum("me,aeik->amik", H.b.ov, T.ab, optimize=True)
    I2B_vooo = I2B_vooo.transpose(1, 0, 2, 3)
    I2C_vooo = I2C_vooo.transpose(1, 0, 2, 3)

    dT.abb, T.abb, t3_excitations["abb"] = ccsdtq_p_loops.update_t3c_p(
        T.aab, t3_excitations["aab"],
        T.abb, t3_excitations["abb"],
        T.bbb, t3_excitations["bbb"],
        T.ab, T.bb,
        H.a.oo, H.a.vv, H.b.oo, H.b.vv,
        H0.aa.oovv, H.aa.voov,
        H0.ab.oovv, I2B_vooo, I2B_ovoo, H.ab.vvov, H.ab.vvvo, H.ab.oooo,
        H.ab.voov, H.ab.vovo, H.ab.ovov, H.ab.ovvo, H.ab.vvvv.transpose(2, 3, 0, 1),
        H0.bb.oovv, I2C_vooo, H.bb.vvov, H.bb.oooo, H.bb.voov, H.bb.vvvv,
        H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
        T.abbb, t4_excitations["abbb"],
        T.aabb, t4_excitations["aabb"],
        H.b.ov, H.a.ov, H.bb.ooov, H.ab.oovo,
        H.ab.ooov, H.aa.ooov, H.bb.vovv, H.ab.ovvv,
        H.ab.vovv, H.aa.vovv,
        shift
    )
    return T, dT, t3_excitations

def update_t3d(T, dT, H, H0, shift, t3_excitations, t4_excitations):
    """
    Update t3d amplitudes by calculating the projection <i~j~k~a~b~c~|(H_N exp(T1+T2+T3))_C|0>.
    """
    I2C_vooo = H.bb.vooo - np.einsum("me,aeij->amij", H.b.ov, T.bb, optimize=True)
    I2C_vooo = I2C_vooo.transpose(1, 0, 2, 3)

    dT.bbb, T.bbb, t3_excitations["bbb"] = ccsdtq_p_loops.update_t3d_p(
        T.abb, t3_excitations["abb"],
        T.bbb, t3_excitations["bbb"],
        T.bb,
        H.b.oo, H.b.vv,
        H0.bb.oovv, H.bb.vvov, I2C_vooo,
        H.bb.oooo, H.bb.voov, H.bb.vvvv,
        H0.ab.oovv, H.ab.ovvo,
        H0.b.oo, H0.b.vv,
        T.bbbb, t4_excitations["bbbb"],
        T.abbb, t4_excitations["abbb"],
        H.b.ov, H.a.ov, H.bb.vovv, H.ab.ovvv,
        H.bb.ooov, H.ab.oovo,
        shift
    )
    return T, dT, t3_excitations

def add_VT3_intermediates(T, H, H0, t3_excitations):
    """
    Add the (V_N T3)_C pieces to the vooo/vvov-type blocks of HBar that enter the T4 equations
    (cf. ccpy.hbar.hbar_ccsdt.add_VT3_intermediates).
    """
    H.aa.vooo, H.ab.vooo, H.ab.ovoo, H.aa.vvov,\
    H.ab.vvov, H.ab.vvvo, H.bb.vooo, H.bb.vvov = ccsdtq_p_loops.add_vt3_p(
        H.aa.vooo, H.ab.vooo, H.ab.ovoo, H.aa.vvov,
        H.ab.vvov, H.ab.vvvo, H.bb.vooo, H.bb.vvov,
        T.aaa, t3_excitations["aaa"],
        T.aab, t3_excitations["aab"],
        T.abb, t3_excitations["abb"],
        T.bbb, t3_excitations["bbb"],
        H0.aa.oovv, H0.ab.oovv, H0.bb.oovv
    )
    return H

def update_t4a(T, dT, H, H0, shift, t3_excitations, t4_excitations):
    """
    Update t4a amplitudes by calculating the projection <ijklabcd|(H_N exp(T1+T2+T3+T4))_C|0>.
    """
    dT.aaaa, T.aaaa, t4_excitations["aaaa"] = ccsdtq_p_loops.update_t4a_p(
        T.aaaa, t4_excitations["aaaa"],
        T.aaab, t4_excitations["aaab"],
        T.aaa, t3_excitations["aaa"],
        T.aab, t3_excitations["aab"],
        T.aa,
        H.a.oo, H.a.vv,
        H.aa.oooo, H.aa.ooov, H.aa.vooo, H.aa.voov,
        H.aa.vovv, H.aa.vvov, H.aa.vvvv, H.ab.ooov,
        H.ab.voov, H.ab.vovv,
        H0.aa.oovv, H0.ab.oovv, H0.bb.oovv,
        H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
        shift
    )
    return T, dT, t4_excitations

def update_t4b(T, dT, H, H0, shift, t3_excitations, t4_excitations):
    """
    Update t4b amplitudes by calculating the projection <ijkl~abcd~|(H_N exp(T1+T2+T3+T4))_C|0>.
    """
    dT.aaab, T.aaab, t4_excitations["aaab"] = ccsdtq_p_loops.update_t4b_p(
        T.aaab, t4_excitations["aaab"],
        T.aaaa, t4_excitations["aaaa"],
        T.aabb, t4_excitations["aabb"],
        T.aaa, t3_excitations["aaa"],
        T.aab, t3_excitations["aab"],
        T.abb, t3_excitations["abb"],
        T.aa, T.ab,
        H.a.oo, H.a.vv, H.b.oo, H.b.vv,
        H.aa.oooo, H.aa.ooov, H.aa.vooo, H.aa.voov,
        H.aa.vovv, H.aa.vvov, H.aa.vvvv, H.ab.oooo,
        H.ab.ooov, H.ab.oovo, H.ab.ovoo, H.ab.ovov,
        H.ab.ovvo, H.ab.ovvv, H.ab.vooo, H.ab.voov,
        H.ab.vovo, H.ab.vovv, H.ab.vvov, H.ab.vvvo,
        H.ab.vvvv, H.bb.ooov, H.bb.voov, H.bb.vovv,
        H0.aa.oovv, H0.ab.oovv, H0.bb.oovv,
        H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
        shift
    )
    return T, dT, t4_excitations

def update_t4c(T, dT, H, H0, shift, t3_excitations, t4_excitations):
    """
    Update t4c amplitudes by calculating the projection <ijk~l~abc~d~|(H_N exp(T1+T2+T3+T4))_C|0>.
    """
    dT.aabb, T.aabb, t4_excitations["aabb"] = ccsdtq_p_loops.update_t4c_p(
        T.aabb, t4_excitations["aabb"],
        T.aaab, t4_excitations["aaab"],
        T.abbb, t4_excitations["abbb"],
        T.aaa, t3_excitations["aaa"],
        T.aab, t3_excitations["aab"],
        T.abb, t3_excitations["abb"],
        T.bbb, t3_excitations["bbb"],
        T.aa, T.ab, T.bb,
        H.a.oo, H.a.vv, H.b.oo, H.b.vv,
        H.aa.oooo, H.aa.ooov, H.aa.vooo, H.aa.voov,
        H.aa.vovv, H.aa.vvov, H.aa.vvvv, H.ab.oooo,
        H.ab.ooov, H.ab.oovo, H.ab.ovoo, H.ab.ovov,
        H.ab.ovvo, H.ab.ovvv, H.ab.vooo, H.ab.voov,
        H.ab.vovo, H.ab.vovv, H.ab.vvov, H.ab.vvvo,
        H.ab.vvvv, H.bb.oooo, H.bb.ooov, H.bb.vooo,
        H.bb.voov, H.bb.vovv, H.bb.vvov, H.bb.vvvv,
        H0.aa.oovv, H0.ab.oovv, H0.bb.oovv,
        H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
        shift
    )
    return T, dT, t4_excitations

def update_t4d(T, dT, H, H0, shift, t3_excitations, t4_excitations):
    """
    Update t4d amplitudes by calculating the projection <ij~k~l~ab~c~d~|(H_N exp(T1+T2+T3+T4))_C|0>.
    """
    dT.abbb, T.abbb, t4_excitations["abbb"] = ccsdtq_p_loops.update_t4d_p(
        T.abbb, t4_excitations["abbb"],
        T.aabb, t4_excitations["aabb"],
        T.bbbb, t4_excitations["bbbb"],
        T.aab, t3_excitations["aab"],
        T.abb, t3_excitations["abb"],
        T.bbb, t3_excitations["bbb"],
        T.ab, T.bb,
        H.a.oo, H.a.vv, H.b.oo, H.b.vv,
        H.aa.ooov, H.aa.voov, H.aa.vovv, H.ab.oooo,
        H.ab.ooov, H.ab.oovo, H.ab.ovoo, H.ab.ovov,
        H.ab.ovvo, H.ab.ovvv, H.ab.vooo, H.ab.voov,
        H.ab.vovo, H.ab.vovv, H.ab.vvov, H.ab.vvvo,
        H.ab.vvvv, H.bb.oooo, H.bb.ooov, H.bb.vooo,
        H.bb.voov, H.bb.vovv, H.bb.vvov, H.bb.vvvv,
        H0.aa.oovv, H0.ab.oovv, H0.bb.oovv,
        H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
        shift
    )
    return T, dT, t4_excitations

def update_t4e(T, dT, H, H0, shift, t3_excitations, t4_excitations):
    """
    Update t4e amplitudes by calculating the projection <i~j~k~l~a~b~c~d~|(H_N exp(T1+T2+T3+T4))_C|0>.
    """
    dT.bbbb, T.bbbb, t4_excitations["bbbb"] = ccsdtq_p_loops.update_t4e_p(
        T.bbbb, t4_excitations["bbbb"],
        T.abbb, t4_excitations["abbb"],
        T.abb, t3_excitations["abb"],
        T.bbb, t3_excitations["bbb"],
        T.bb,
        H.b.oo, H.b.vv,
        H.ab.oovo, H.ab.ovvo, H.ab.ovvv, H.bb.oooo,
        H.bb.ooov, H.bb.vooo, H.bb.voov, H.bb.vovv,
        H.bb.vvov, H.bb.vvvv,
        H0.aa.oovv, H0.ab.oovv, H0.bb.oovv,
        H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
        shift
    )
    return T, dT, t4_excitations
