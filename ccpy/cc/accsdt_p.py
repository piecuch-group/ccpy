"""
Module with functions that help perform the approximate coupled-pair (ACP) coupled-cluster (CC)
approach with singles, doubles, and the subset of triples belonging to the P-space, abbreviated
as ACCSDT(P).
"""
import numpy as np
# Modules for type checking
from typing import List, Tuple, Dict
from ccpy.models.operators import ClusterOperator
from ccpy.models.system import System
from ccpy.models.integrals import Integral
# Modules for computation
from ccpy.hbar.hbar_ccs import get_pre_ccs_intermediates, get_ccs_intermediates_opt
from ccpy.hbar.hbar_ccsd import get_ccsd_intermediates
from ccpy.lib.core import ccsdt_p_loops


def update(T: ClusterOperator,
           dT: ClusterOperator,
           H: Integral,
           X: Integral,
           shift: float,
           flag_RHF: bool,
           system: System,
           t3_excitations: Dict[str, np.ndarray],
           acparray: List[float]) -> Tuple[ClusterOperator, ClusterOperator]:
    """
    Performs one update of the CC amplitude equations for the ACCSDT(P) method.

    Parameters
    ----------
    T : ClusterOperator
        Cluster operator containing T1 and T2 components.
    dT : ClusterOperator
        Residual of the CC amplitude equations corresponding to projections onto singles, doubles, and the subset of
        triples included in the P space.
    H : Integral
        Bare Hamiltonian in the normal-ordered form.
    X : Integral
        Intermediates for the CC iterations that are roughly given by a CCSD-like similarity-transformed Hamiltonian.
    shift : float
        Energy denominator shift for stabilizing update in case of strong quasidegeneracy.
    flag_RHF : bool
        Flag to turn on/off RHF symmetry. Doing so skips updating components of T that are equivalent for closed shells.
    system : System
        System object containing information about the molecular system, such as orbital dimensions.
    t3_excitations : Dict[str, np.ndarray]
        Dictionary with the keys 'aaa', 'aab', 'abb', and 'bbb', corresponding to distinct T3 spincases, which are
        asociated with values given by the 2D Numpy array containing the triple excitations [a, b, c, i, j, k] belonging
        to the P space.
    acparray : List[float]
        List containing the ACP scaling factors for the 5 T2**2 diagrams.

    Returns
    -------
    T : ClusterOperator
        Cluster operator with updated T1, T2, and T3(P) components.
    dT : ClusterOperator
        Residual of the ACCSDT(P) amplitude equations corresponding to projections onto singles, doubles, and the
        subset of triples included in the P space.
    """

    # Check for empty spincases in t3 list. Remember that [1., 1., 1., 1., 1., 1.]
    # is defined as the "empty" state in the Fortran modules.
    do_t3 = {"aaa": True, "aab": True, "abb": True, "bbb": True}
    if np.array_equal(t3_excitations["aaa"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_t3["aaa"] = False
    if np.array_equal(t3_excitations["aab"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_t3["aab"] = False
    if np.array_equal(t3_excitations["abb"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_t3["abb"] = False
    if np.array_equal(t3_excitations["bbb"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_t3["bbb"] = False
    build_hbar = do_t3["aaa"] or do_t3["aab"] or do_t3["abb"] or do_t3["bbb"]

    X = get_pre_ccs_intermediates(X, T, H, system, flag_RHF)

    # update T1
    T, dT = update_t1a(T, dT, H, X, shift, t3_excitations)
    if flag_RHF:
        T.b = T.a.copy()
        dT.b = dT.a.copy()
    else:
        T, dT = update_t1b(T, dT, H, X, shift, t3_excitations)

    # CCS intermediates
    X = get_ccs_intermediates_opt(X, T, H, system, flag_RHF)
    # Remove T2 parts from X.a.oo/X.b.oo and X.a.vv/X.b.vv
    X.a.vv += (
            + 0.5 * np.einsum("mnef,afmn->ae", H.aa.oovv, T.aa, optimize=True)
            + np.einsum("mnef,afmn->ae", H.ab.oovv, T.ab, optimize=True)
    )
    X.a.oo -= (
            + 0.5 * np.einsum("mnef,efin->mi", H.aa.oovv, T.aa, optimize=True)
            + np.einsum("mnef,efin->mi", H.ab.oovv, T.ab, optimize=True)
    )
    if flag_RHF:
        X.b.vv = X.a.vv
        X.b.oo = X.a.oo
    else:
        X.b.vv += (
                    + 0.5 * np.einsum("mnef,afmn->ae", H.bb.oovv, T.bb, optimize=True)
                    + np.einsum("nmfe,fanm->ae", H.ab.oovv, T.ab, optimize=True)
        )
        X.b.oo -= (
                    + 0.5 * np.einsum("mnef,efin->mi", H.bb.oovv, T.bb, optimize=True) 
                    + np.einsum("nmfe,feni->mi", H.ab.oovv, T.ab, optimize=True)
        )

    # update T2
    T, dT = update_t2a(T, dT, X, H, shift, t3_excitations, acparray)
    T, dT = update_t2b(T, dT, X, H, shift, t3_excitations, acparray)
    if flag_RHF:
        T.bb = T.aa.copy()
        dT.bb = dT.aa.copy()
    else:
        T, dT = update_t2c(T, dT, X, H, shift, t3_excitations, acparray)

    # Add T2 parts from X.a.oo/X.b.oo and X.a.vv/X.b.vv
    X.a.vv -= (
            + 0.5 * np.einsum("mnef,afmn->ae", H.aa.oovv, T.aa, optimize=True) #
            + np.einsum("mnef,afmn->ae", H.ab.oovv, T.ab, optimize=True) #
    )
    X.a.oo += (
            + 0.5 * np.einsum("mnef,efin->mi", H.aa.oovv, T.aa, optimize=True) # 
            + np.einsum("mnef,efin->mi", H.ab.oovv, T.ab, optimize=True) #
    )
    if flag_RHF:
        X.b.vv = X.a.vv
        X.b.oo = X.a.oo
    else:
        X.b.vv -= (
                    + 0.5 * np.einsum("mnef,afmn->ae", H.bb.oovv, T.bb, optimize=True) #
                    + np.einsum("nmfe,fanm->ae", H.ab.oovv, T.ab, optimize=True) #
        )
        X.b.oo += (
                    + 0.5 * np.einsum("mnef,efin->mi", H.bb.oovv, T.bb, optimize=True) # 
                    + np.einsum("nmfe,feni->mi", H.ab.oovv, T.ab, optimize=True) #
        )

    # CCSD intermediates
    # [TODO]: Should accept CCS HBar as input and build only terms with T2 in it
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
        X.aa.vvvv = X.aa.vvvv.transpose(3, 2, 1, 0)
        # X.ab.vvvv = X.ab.vvvv.transpose(3, 2, 1, 0)
        X.bb.vvvv = X.bb.vvvv.transpose(3, 2, 1, 0)

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

def update_t1a(T: ClusterOperator,
               dT: ClusterOperator,
               H: Integral,
               X: Integral,
               shift: float,
               t3_excitations: Dict[str, np.ndarray]) -> Tuple[ClusterOperator, ClusterOperator]:
    """Update t1a amplitudes as t1a(ai) <- t1(ai) + <ia|(H_N exp(T1+T2+T3(P)))_C|0>/D_MP(ai)

    Parameters
    ----------
    T : ClusterOperator
        Cluster operator containing T1, T2, and T3(P) components
    dT : ClusterOperator
        Residual for T1, T2, and T3(P) amplitude equations
    H : Integral
        Bare Hamiltonian in the normal-ordered form
    X : Integral
        Intermediates for the CC iterations that are roughly given by a CCSD-like similarity-transformed Hamiltonian
    shift : float
        Energy denominator shift for stabilizing update
    t3_excitations : Dict[str, np.ndarray]
        Dictionary with the keys 'aaa', 'aab', 'abb', and 'bbb', corresponding to distinct T3 spincases, which are
        asociated with values given by the 2D Numpy array containing the triple excitations [a, b, c, i, j, k] belonging
        to the P space.

    Returns
    -------
    T : ClusterOperator
        Cluster operator containing updated t1a(ai) component in T.a
    dT : ClusterOperator
        Residual of CC amplitude equation containing <ia|(H_N exp(T1+T2+T3(P))_C|0>/D_MP(ai) in dT.a
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
    T.a, dT.a = ccsdt_p_loops.update_t1a(
        T.a, 
        dT.a + H.a.vo,
        t3_excitations["aaa"], t3_excitations["aab"], t3_excitations["abb"],
        T.aaa, T.aab, T.abb,
        H.aa.oovv, H.ab.oovv, H.bb.oovv,
        H.a.oo, H.a.vv,
        shift
    )
    return T, dT

def update_t1b(T: ClusterOperator,
               dT: ClusterOperator,
               H: Integral,
               X: Integral,
               shift: float,
               t3_excitations: Dict[str, np.ndarray]):
    """
    Update t1b amplitudes by calculating the projection <i~a~|(H_N e^(T1+T2+T3(P)))_C|0>.
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
    T.b, dT.b = ccsdt_p_loops.update_t1b(
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
def update_t2a(T, dT, H, H0, shift, t3_excitations, acparray):
    """
    Update t2a amplitudes by calculating the projection <ijab|(H_N e^(T1+T2+T3(P)))_C|0>.
    """
    d1, d2, d3, d4, d5 = acparray
    # intermediates
    I2A_vooo = H.aa.vooo + 0.5*np.einsum('anef,efij->anij', H0.aa.vovv + 0.5 * H.aa.vovv, T.aa, optimize=True)
    tau = 0.5 * T.aa + np.einsum('ai,bj->abij', T.a, T.a, optimize=True)

    dT.aa = -0.5 * np.einsum("amij,bm->abij", I2A_vooo, T.a, optimize=True)
    dT.aa += 0.5 * np.einsum("abie,ej->abij", H.aa.vvov, T.a, optimize=True)
    dT.aa += 0.5 * np.einsum("ae,ebij->abij", H.a.vv, T.aa, optimize=True)
    dT.aa -= 0.5 * np.einsum("mi,abmj->abij", H.a.oo, T.aa, optimize=True)
    dT.aa += np.einsum("amie,ebmj->abij", H.aa.voov, T.aa, optimize=True)
    dT.aa += np.einsum("amie,bejm->abij", H.ab.voov, T.ab, optimize=True)
    dT.aa += 0.25 * np.einsum("abef,efij->abij", H0.aa.vvvv, tau, optimize=True)
    dT.aa += 0.125 * np.einsum("mnij,abmn->abij", H.aa.oooo, T.aa, optimize=True)

    # < ijab | (V T2**2)_C | 0 >
    #dT.aa += 0.5 * np.einsum(
    #    "mnef,aeim,bfjn->abij", H.aa.oovv, T.aa, T.aa, optimize=True
    #) # A(ij) [D1 + D2]
    dT.aa += d1 * 0.5 * np.einsum("mnef,aeim,bfjn->abij", H0.ab.oovv, T.aa, T.aa, optimize=True) # A(ij) [D1]
    dT.aa -= d2 * 0.5 * np.einsum("mnfe,aeim,bfjn->abij", H0.ab.oovv, T.aa, T.aa, optimize=True) # A(ij) [D2]
    dT.aa += d5 * 0.25 * 0.25 * np.einsum("mnef,efij,abmn->abij", H0.aa.oovv, T.aa, T.aa, optimize=True) # 1 [D5]
    dT.aa -= d4 * 0.25 * np.einsum("mnef,abim,efjn->abij", H0.aa.oovv, T.aa, T.aa, optimize=True) # A(ij) [D4]
    dT.aa -= d3 * 0.25 * np.einsum("mnef,aeij,bfmn->abij", H0.aa.oovv, T.aa, T.aa, optimize=True) # A(ab) [D3]
    dT.aa += d1 * np.einsum("mnef,aeim,bfjn->abij", H0.ab.oovv, T.aa, T.ab, optimize=True) # A(ij)A(ab) [D1]
    dT.aa -= d4 * 0.5 * np.einsum("mnef,abim,efjn->abij", H0.ab.oovv, T.aa, T.ab, optimize=True) # A(ij) [D4]
    dT.aa -= d3 * 0.5 * np.einsum("mnef,aeij,bfmn->abij", H0.ab.oovv, T.aa, T.ab, optimize=True) # A(ab) [D3]
    #dT.aa += 0.5 * np.einsum(
    #    "mnef,aeim,bfjn->abij", H.bb.oovv, T.ab, T.ab, optimize=True
    #) # A(ij) [D1 + D2]
    dT.aa += d1 * 0.5 * np.einsum("mnef,aeim,bfjn->abij", H0.ab.oovv, T.ab, T.ab, optimize=True) # A(ij) [D1]
    dT.aa -= d2 * 0.5 * np.einsum("mnfe,aeim,bfjn->abij", H0.ab.oovv, T.ab, T.ab, optimize=True) # A(ij) [D2]

    T.aa, dT.aa = ccsdt_p_loops.update_t2a(
        T.aa,
        dT.aa + 0.25 * H0.aa.vvoo,
        t3_excitations["aaa"], t3_excitations["aab"],
        T.aaa, T.aab,
        H.a.ov, H.b.ov,
        H0.aa.ooov + H.aa.ooov, H0.aa.vovv + H.aa.vovv,
        H0.ab.ooov + H.ab.ooov, H0.ab.vovv + H.ab.vovv,
        H0.a.oo, H0.a.vv,
        shift
    )
    return T, dT

# @profile
def update_t2b(T, dT, H, H0, shift, t3_excitations, acparray):
    """
    Update t2b amplitudes by calculating the projection <ij~ab~|(H_N e^(T1+T2+T3(P)))_C|0>.
    """
    d1, d2, d3, d4, d5 = acparray

    # intermediates
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
    dT.ab += np.einsum("amie,ebmj->abij", H.aa.voov, T.ab, optimize=True)
    dT.ab += np.einsum("amie,ebmj->abij", H.ab.voov, T.bb, optimize=True)
    dT.ab += np.einsum("mbej,aeim->abij", H.ab.ovvo, T.aa, optimize=True)
    dT.ab += np.einsum("bmje,aeim->abij", H.bb.voov, T.ab, optimize=True)
    dT.ab -= np.einsum("mbie,aemj->abij", H.ab.ovov, T.ab, optimize=True)
    dT.ab -= np.einsum("amej,ebim->abij", H.ab.vovo, T.ab, optimize=True)
    dT.ab += np.einsum("mnij,abmn->abij", H.ab.oooo, T.ab, optimize=True)
    dT.ab += np.einsum("abef,efij->abij", H0.ab.vvvv, tau, optimize=True)

    # < ij~ab~ | (V T2**2)_C | 0 >
    # dT.ab += 0.5 * (acparray[0] + acparray[1]) * np.einsum("mnef,aeim,fbnj->abij", H0.aa.oovv, T.aa, T.ab, optimize=True) #D1+D2 ##
    dT.ab += acparray[0] * np.einsum("mnef,aeim,fbnj->abij", H0.ab.oovv, T.aa, T.ab, optimize=True) #D1,DIR ##
    dT.ab -= acparray[1] * np.einsum("mnfe,aeim,fbnj->abij", H0.ab.oovv, T.aa, T.ab, optimize=True) #D2,EXCH ##
    dT.ab += acparray[0] * np.einsum("mnef,aeim,bfjn->abij", H0.ab.oovv, T.aa, T.bb, optimize=True) #D1 ##
    dT.ab += acparray[0] * np.einsum("mnef,afin,ebmj->abij", H0.ab.oovv, T.ab, T.ab, optimize=True) #D1 ##
    # dT.ab += 0.5 * (acparray[0] + acparray[1]) * np.einsum("mnef,aeim,bfjn->abij", H0.bb.oovv, T.ab, T.bb, optimize=True) #D1+D2 ##
    dT.ab += acparray[0] * np.einsum("mnef,aeim,bfjn->abij", H0.ab.oovv, T.ab, T.bb, optimize=True) #D1,DIR ##
    dT.ab -= acparray[1] * np.einsum("mnfe,aeim,bfjn->abij", H0.ab.oovv, T.ab, T.bb, optimize=True) #D2,EXCH ##
    dT.ab += acparray[1] * np.einsum("mnef,ebin,afmj->abij", H0.ab.oovv, T.ab, T.ab, optimize=True) #D2 ##
    dT.ab += acparray[4] * np.einsum("mnef,efij,abmn->abij", H0.ab.oovv, T.ab, T.ab, optimize=True) #D5 ##
    dT.ab += acparray[3] * 0.5 * np.einsum("mnef,efni,abmj->abij", H0.aa.oovv, T.aa, T.ab, optimize=True) #D4 ##
    dT.ab -= acparray[3] * np.einsum("mnef,efin,abmj->abij", H0.ab.oovv, T.ab, T.ab, optimize=True) #D4 ##
    dT.ab -= acparray[3] * np.einsum("mnef,efmj,abin->abij", H0.ab.oovv, T.ab, T.ab, optimize=True) #D4 ##
    dT.ab += acparray[3] * 0.5 * np.einsum("mnef,efnj,abim->abij", H0.bb.oovv, T.bb, T.ab, optimize=True) #D4 ##
    dT.ab += acparray[2] * 0.5 * np.einsum("mnef,afnm,ebij->abij", H0.aa.oovv, T.aa, T.ab, optimize=True) #D3 ##
    dT.ab -= acparray[2] * np.einsum("mnef,afmn,ebij->abij", H0.ab.oovv, T.ab, T.ab, optimize=True) #D3 ##
    dT.ab -= acparray[2] * np.einsum("mnef,ebmn,afij->abij", H0.ab.oovv, T.ab, T.ab, optimize=True) #D3 ##
    dT.ab += acparray[2] * 0.5 * np.einsum("mnef,bfnm,aeij->abij", H0.bb.oovv, T.bb, T.ab, optimize=True) #D3 ##

    T.ab, dT.ab = ccsdt_p_loops.update_t2b(
        T.ab,
        dT.ab + H0.ab.vvoo,
        t3_excitations["aab"], t3_excitations["abb"],
        T.aab, T.abb,
        H.a.ov, H.b.ov,
        H.aa.ooov + H0.aa.ooov, H.aa.vovv + H0.aa.vovv,
        H.ab.ooov + H0.ab.ooov, H.ab.oovo + H0.ab.oovo, H.ab.vovv + H0.ab.vovv, H.ab.ovvv + H0.ab.ovvv,
        H.bb.ooov + H0.bb.ooov, H.bb.vovv + H0.bb.vovv,
        H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
        shift
    )

    return T, dT

# @profile
def update_t2c(T, dT, H, H0, shift, t3_excitations, acparray):
    """
    Update t2c amplitudes by calculating the projection <i~j~a~b~|(H_N e^(T1+T2+T3(P)))_C|0>.
    """
    d1, d2, d3, d4, d5 = acparray
    # intermediates
    I2C_vooo = H.bb.vooo + 0.5 * np.einsum('anef,efij->anij', H0.bb.vovv + 0.5 * H.bb.vovv, T.bb, optimize=True)
    tau = 0.5 * T.bb + np.einsum('ai,bj->abij', T.b, T.b, optimize=True)

    dT.bb = -0.5 * np.einsum("amij,bm->abij", I2C_vooo, T.b, optimize=True)
    dT.bb += 0.5 * np.einsum("abie,ej->abij", H.bb.vvov, T.b, optimize=True)
    dT.bb += 0.5 * np.einsum("ae,ebij->abij", H.b.vv, T.bb, optimize=True)
    dT.bb -= 0.5 * np.einsum("mi,abmj->abij", H.b.oo, T.bb, optimize=True)
    dT.bb += np.einsum("amie,ebmj->abij", H.bb.voov, T.bb, optimize=True)
    dT.bb += np.einsum("maei,ebmj->abij", H.ab.ovvo, T.ab, optimize=True)
    dT.bb += 0.25 * np.einsum("abef,efij->abij", H0.bb.vvvv, tau, optimize=True)
    dT.bb += 0.125 * np.einsum("mnij,abmn->abij", H.bb.oooo, T.bb, optimize=True)

    # < ijab | (V T2**2)_C | 0 >
    #dT.aa += 0.5 * np.einsum(
    #    "mnef,aeim,bfjn->abij", H.aa.oovv, T.aa, T.aa, optimize=True
    #) # A(ij) [D1 + D2]
    dT.bb += d1 * 0.5 * np.einsum("nmfe,aeim,bfjn->abij", H0.ab.oovv, T.bb, T.bb, optimize=True) # A(ij) [D1]
    dT.bb -= d2 * 0.5 * np.einsum("nmef,aeim,bfjn->abij", H0.ab.oovv, T.bb, T.bb, optimize=True) # A(ij) [D2]
    dT.bb += d5 * 0.25 * 0.25 * np.einsum("mnef,efij,abmn->abij", H0.bb.oovv, T.bb, T.bb, optimize=True) # 1 [D5]
    dT.bb -= d4 * 0.25 * np.einsum("mnef,abim,efjn->abij", H0.bb.oovv, T.bb, T.bb, optimize=True) # A(ij) [D4]
    dT.bb -= d3 * 0.25 * np.einsum("mnef,aeij,bfmn->abij", H0.bb.oovv, T.bb, T.bb, optimize=True) # A(ab) [D3]
    dT.bb += d1 * np.einsum("nmfe,aeim,fbnj->abij", H0.ab.oovv, T.bb, T.ab, optimize=True) # A(ij)A(ab) [D1]
    dT.bb -= d4 * 0.5 * np.einsum("nmfe,abim,fenj->abij", H0.ab.oovv, T.bb, T.ab, optimize=True) # A(ij) [D4]
    dT.bb -= d3 * 0.5 * np.einsum("nmfe,aeij,fbnm->abij", H0.ab.oovv, T.bb, T.ab, optimize=True) # A(ab) [D3]
    #dT.bb += 0.5 * np.einsum(
    #    "mnef,aeim,bfjn->abij", H.bb.oovv, T.ab, T.ab, optimize=True
    #) # A(ij) [D1 + D2]
    dT.bb += d1 * 0.5 * np.einsum("nmfe,eami,fbnj->abij", H0.ab.oovv, T.ab, T.ab, optimize=True) # A(ij) [D1]
    dT.bb -= d2 * 0.5 * np.einsum("nmef,eami,fbnj->abij", H0.ab.oovv, T.ab, T.ab, optimize=True) # A(ij) [D2]

    T.bb, dT.bb = ccsdt_p_loops.update_t2c(
        T.bb,
        dT.bb + 0.25 * H0.bb.vvoo,
        t3_excitations["abb"], t3_excitations["bbb"],
        T.abb, T.bbb,
        H.a.ov, H.b.ov,
        H0.ab.oovo + H.ab.oovo, H0.ab.ovvv + H.ab.ovvv,
        H0.bb.ooov + H.bb.ooov, H0.bb.vovv + H.bb.vovv,
        H0.b.oo, H0.b.vv,
        shift
    )
    return T, dT

def update_t3a(T, dT, H, H0, shift, t3_excitations):
    """
    Update t3a amplitudes by calculating the projection <ijkabc|(H_N e^(T1+T2+T3(P)))_C|0>.
    """
    I2A_vooo = H.aa.vooo - np.einsum("me,aeij->amij", H.a.ov, T.aa, optimize=True)
    I2A_vooo = I2A_vooo.transpose(1, 0, 2, 3)

    dT.aaa, T.aaa, t3_excitations["aaa"] = ccsdt_p_loops.update_t3a_p(
        T.aaa, t3_excitations["aaa"], 
        T.aab, t3_excitations["aab"],
        T.aa,
        H.a.oo, H.a.vv,
        H0.aa.oovv, H.aa.vvov, I2A_vooo,
        H.aa.oooo, H.aa.voov, H.aa.vvvv,
        H0.ab.oovv, H.ab.voov,
        H0.a.oo, H0.a.vv,
        shift
    )
    return T, dT, t3_excitations

def update_t3b(T, dT, H, H0, shift, t3_excitations):
    """
    Update t3b amplitudes by calculating the projection <ijk~abc~|(H_N e^(T1+T2+T3(P)))_C|0>.
    """
    I2A_vooo = H.aa.vooo - np.einsum("me,aeij->amij", H.a.ov, T.aa, optimize=True)
    I2B_ovoo = H.ab.ovoo - np.einsum("me,ecjk->mcjk", H.a.ov, T.ab, optimize=True)
    I2B_vooo = H.ab.vooo - np.einsum("me,aeik->amik", H.b.ov, T.ab, optimize=True)
    I2A_vooo = I2A_vooo.transpose(1, 0, 2, 3)
    I2B_vooo = I2B_vooo.transpose(1, 0, 2, 3)

    dT.aab, T.aab, t3_excitations["aab"] = ccsdt_p_loops.update_t3b_p(
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
        shift
    )
    return T, dT, t3_excitations

def update_t3c(T, dT, H, H0, shift, t3_excitations):
    """
    Update t3c amplitudes by calculating the projection <ij~k~ab~c~|(H_N e^(T1+T2+T3(P)))_C|0>.
    """
    I2C_vooo = H.bb.vooo - np.einsum("me,aeij->amij", H.b.ov, T.bb, optimize=True)
    I2B_ovoo = H.ab.ovoo - np.einsum("me,ecjk->mcjk", H.a.ov, T.ab, optimize=True)
    I2B_vooo = H.ab.vooo - np.einsum("me,aeik->amik", H.b.ov, T.ab, optimize=True)
    I2B_vooo = I2B_vooo.transpose(1, 0, 2, 3)
    I2C_vooo = I2C_vooo.transpose(1, 0, 2, 3)

    dT.abb, T.abb, t3_excitations["abb"] = ccsdt_p_loops.update_t3c_p(
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
        shift
    )
    return T, dT, t3_excitations

def update_t3d(T, dT, H, H0, shift, t3_excitations):
    """
    Update t3d amplitudes by calculating the projection <i~j~k~a~b~c~|(H_N e^(T1+T2+T3(P)))_C|0>.
    """
    I2C_vooo = H.bb.vooo - np.einsum("me,aeij->amij", H.b.ov, T.bb, optimize=True)
    I2C_vooo = I2C_vooo.transpose(1, 0, 2, 3)

    dT.bbb, T.bbb, t3_excitations["bbb"] = ccsdt_p_loops.update_t3d_p(
        T.abb, t3_excitations["abb"],
        T.bbb, t3_excitations["bbb"],
        T.bb,
        H.b.oo, H.b.vv,
        H0.bb.oovv, H.bb.vvov, I2C_vooo,
        H.bb.oooo, H.bb.voov, H.bb.vvvv,
        H0.ab.oovv, H.ab.ovvo,
        H0.b.oo, H0.b.vv,
        shift
    )
    return T, dT, t3_excitations
