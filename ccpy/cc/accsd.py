"""
Module with functions that help perform the approximate coupled-pair (ACP) coupled-cluster (CC)
approach with singles and doubles, abbreviated as ACCSD.
"""
import numpy as np
# Modules for type checking
from typing import List, Tuple
from ccpy.models.operators import ClusterOperator
from ccpy.models.system import System
from ccpy.models.integrals import Integral
# Modules for computation
from ccpy.hbar.hbar_ccs import get_pre_ccs_intermediates, get_ccs_intermediates_opt
from ccpy.lib.core import cc_loops2


def update(T: ClusterOperator,
           dT: ClusterOperator,
           H: Integral,
           X: Integral,
           shift: float,
           flag_RHF: bool,
           system: System,
           acparray: List[float]) -> Tuple[ClusterOperator, ClusterOperator]:
    """
    Performs one update of the CC amplitude equations for the ACCSD method.

    Parameters
    ----------
    T : ClusterOperator
        Cluster operator containing T1 and T2 components
    dT : ClusterOperator
        Residual of the CC amplitude equations corresponding to projections onto singles and doubles
    H : Integral
        Bare Hamiltonian in the normal-ordered form
    X : Integral
        Intermediates for the CC iterations that are roughly given by a CCSD-like similarity-transformed Hamiltonian
    shift : float
        Energy denominator shift for stabilizing update in case of strong quasidegeneracy.
    flag_RHF : bool
        Flag to turn on/off RHF symmetry. Doing so skips updating components of T that are equivalent for closed shells.
    system : System
        System object containing information about the molecular system, such as orbital dimensions.
    acparray : List[float]
        List containing the ACP scaling factors for the 5 T2**2 diagrams

    Returns
    -------
    T : ClusterOperator
        Cluster operator with updated T1 and T2 components
    dT : ClusterOperator
        Residual of the ACCSD amplitude equations corresponding to projections onto singles and doubles
    """

    # pre-CCS intermediates
    X = get_pre_ccs_intermediates(X, T, H, system, flag_RHF)

    # update T1
    T, dT = update_t1a(T, dT, X, H, shift)
    if flag_RHF:
        T.b = T.a.copy()
        dT.b = dT.a.copy()
    else:
        T, dT = update_t1b(T, dT, X, H, shift)

    # CCS intermediates
    X = get_ccs_intermediates_opt(X, T, H, system, flag_RHF)
    # Remove T2 parts from X.a.oo/X.b.oo and X.a.vv/X.b.vv as these will be treated with ACP weighting later on
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
    T, dT = update_t2a(T, dT, X, H, shift, acparray)
    T, dT = update_t2b(T, dT, X, H, shift, acparray)
    if flag_RHF:
        T.bb = T.aa.copy()
        dT.bb = dT.aa.copy()
    else:
        T, dT = update_t2c(T, dT, X, H, shift, acparray)

    return T, dT


def update_t1a(T: ClusterOperator,
               dT: ClusterOperator,
               X: Integral,
               H: Integral,
               shift: float) -> Tuple[ClusterOperator, ClusterOperator]:
    """Update t1a amplitudes as t1a(ai) <- t1(ai) + <ia|(H_N exp(T1+T2))_C|0>/D_MP(ai),

    Parameters
    ----------
    T : ClusterOperator
        Cluster operator containing T1 and T2 components
    dT : ClusterOperator
        Residual for T1 and T2 amplitude equations
    X : Integral
        Intermediates for the CC iterations that are roughly given by a CCSD-like similarity-transformed Hamiltonian
    H : Integral
        Bare Hamiltonian in the normal-ordered form
    shift : float
        Energy denominator shift for stabilizing update
    acparray : List[float]
        List containing the ACP scaling factors for the 5 T2**2 diagrams

    Returns
    -------
    T : ClusterOperator
        Cluster operator containing updated t1a(ai) component in T.a
    dT : ClusterOperator
        Residual of CC amplitude equation containing <ia|(H_N exp(T1+T2)_C|0>/D_MP(ai) in dT.a
    """
    dT.a = -np.einsum("mi,am->ai", X.a.oo, T.a, optimize=True)
    dT.a += np.einsum("ae,ei->ai", X.a.vv, T.a, optimize=True)
    dT.a += np.einsum("me,aeim->ai", X.a.ov, T.aa, optimize=True)
    dT.a += np.einsum("me,aeim->ai", X.b.ov, T.ab, optimize=True)
    dT.a += np.einsum("anif,fn->ai", H.aa.voov, T.a, optimize=True)
    dT.a += np.einsum("anif,fn->ai", H.ab.voov, T.b, optimize=True)
    dT.a -= 0.5 * np.einsum("mnif,afmn->ai", H.aa.ooov, T.aa, optimize=True)
    dT.a -= np.einsum("mnif,afmn->ai", H.ab.ooov, T.ab, optimize=True)
    dT.a += 0.5 * np.einsum("anef,efin->ai", H.aa.vovv, T.aa, optimize=True)
    dT.a += np.einsum("anef,efin->ai", H.ab.vovv, T.ab, optimize=True)
    T.a, dT.a = cc_loops2.update_t1a(
        T.a, dT.a + H.a.vo, H.a.oo, H.a.vv, shift
    )
    return T, dT


def update_t1b(T: ClusterOperator,
               dT: ClusterOperator,
               X: Integral,
               H: Integral,
               shift: float) -> Tuple[ClusterOperator, ClusterOperator]:
    """Update t1b amplitudes as t1b(a~i~) <- t1b(a~i~) + <i~a~|(H_N exp(T1 + T2))_C|0>/D_MP(a~i~)

    Parameters
    ----------
    T : ClusterOperator
        Cluster operator containing T1 and T2 components
    dT : ClusterOperator
        Residual for T1 and T2 amplitude equations
    X : Integral
        Intermediates for the CC iterations that are roughly given by a CCSD-like similarity-transformed Hamiltonian
    H : Integral
        Bare Hamiltonian in the normal-ordered form
    shift : float
        Energy denominator shift for stabilizing update
    acparray : List[float]
        List containing the ACP scaling factors for the 5 T2**2 diagrams

    Returns
    -------
    T : ClusterOperator
        Cluster operator containing updated t1b(a~i~) component in T.b
    dT : ClusterOperator
        Residual of CC amplitude equation containing <i~a~|(H_N exp(T1+T2)_C|0>/D_MP(a~i~) in dT.b
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
    T.b, dT.b = cc_loops2.update_t1b(
        T.b, dT.b + H.b.vo, H.b.oo, H.b.vv, shift
    )
    return T, dT


def update_t2a(T: ClusterOperator,
               dT: ClusterOperator,
               H: Integral,
               H0: Integral,
               shift: float,
               acparray: List[float]) -> Tuple[ClusterOperator, ClusterOperator]:
    """Update t2aa amplitudes as t2aa(abij) <- t2ab(abij) + <ijab|(H_N exp(T1 + T2))_C|0>/D_MP(abij)

    Parameters
    ----------
    T : ClusterOperator
        Cluster operator containing T1 and T2 components
    dT : ClusterOperator
        Residual for T1 and T2 amplitude equations
    H : Integral
        Intermediates for the CC iterations that are roughly given by a CCSD-like similarity-transformed Hamiltonian
    H0 : Integral
        Bare Hamiltonian in the normal-ordered form
    shift : float
        Energy denominator shift for stabilizing update
    acparray : List[float]
        List containing the ACP scaling factors for the 5 T2**2 diagrams

    Returns
    -------
    T : ClusterOperator
        Cluster operator containing updated t2aa(abij) component in T.aa
    dT : ClusterOperator
        Residual of CC amplitude equation containing <ijab|(H_N exp(T1+T2)_C|0>/D_MP(abij) in dT.aa
    """
    d1, d2, d3, d4, d5 = acparray

    # intermediates
    I2A_vooo = H.aa.vooo + 0.5 * np.einsum('anef,efij->anij', H0.aa.vovv + 0.5 * H.aa.vovv, T.aa, optimize=True)
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
    # dT.aa += 0.5 * np.einsum(
    #     "mnef,aeim,bfjn->abij", H.aa.oovv, T.aa, T.aa, optimize=True
    # ) # A(ij) [D1 + D2]
    dT.aa += d1 * 0.5 * np.einsum("mnef,aeim,bfjn->abij", H0.ab.oovv, T.aa, T.aa, optimize=True)  # A(ij) [D1]
    dT.aa -= d2 * 0.5 * np.einsum("mnfe,aeim,bfjn->abij", H0.ab.oovv, T.aa, T.aa, optimize=True)  # A(ij) [D2]
    dT.aa += d5 * 0.25 * 0.25 * np.einsum("mnef,efij,abmn->abij", H0.aa.oovv, T.aa, T.aa, optimize=True)  # 1 [D5]
    dT.aa -= d4 * 0.25 * np.einsum("mnef,abim,efjn->abij", H0.aa.oovv, T.aa, T.aa, optimize=True)  # A(ij) [D4]
    dT.aa -= d3 * 0.25 * np.einsum("mnef,aeij,bfmn->abij", H0.aa.oovv, T.aa, T.aa, optimize=True)  # A(ab) [D3]
    dT.aa += d1 * np.einsum("mnef,aeim,bfjn->abij", H0.ab.oovv, T.aa, T.ab, optimize=True)  # A(ij)A(ab) [D1]
    dT.aa -= d4 * 0.5 * np.einsum("mnef,abim,efjn->abij", H0.ab.oovv, T.aa, T.ab, optimize=True)  # A(ij) [D4]
    dT.aa -= d3 * 0.5 * np.einsum("mnef,aeij,bfmn->abij", H0.ab.oovv, T.aa, T.ab, optimize=True)  # A(ab) [D3]
    # dT.aa += 0.5 * np.einsum(
    #     "mnef,aeim,bfjn->abij", H.bb.oovv, T.ab, T.ab, optimize=True
    # ) # A(ij) [D1 + D2]
    dT.aa += d1 * 0.5 * np.einsum("mnef,aeim,bfjn->abij", H0.ab.oovv, T.ab, T.ab, optimize=True)  # A(ij) [D1]
    dT.aa -= d2 * 0.5 * np.einsum("mnfe,aeim,bfjn->abij", H0.ab.oovv, T.ab, T.ab, optimize=True)  # A(ij) [D2]

    T.aa, dT.aa = cc_loops2.update_t2a(
        T.aa, dT.aa + 0.25 * H0.aa.vvoo, H0.a.oo, H0.a.vv, shift
    )
    return T, dT


def update_t2b(T: ClusterOperator,
               dT: ClusterOperator,
               H: Integral,
               H0: Integral,
               shift: float,
               acparray: List[float]) -> Tuple[ClusterOperator, ClusterOperator]:
    """Update t2ab amplitudes as t2ab(abij) <- t2ab(ab~ij~) + <ij~ab~|(H_N exp(T1 + T2))_C|0>/D_MP(ab~ij~)

    Parameters
    ----------
    T : ClusterOperator
        Cluster operator containing T1 and T2 components
    dT : ClusterOperator
        Residual for T1 and T2 amplitude equations
    H : Integral
        Intermediates for the CC iterations that are roughly given by a CCSD-like similarity-transformed Hamiltonian
    H0 : Integral
        Bare Hamiltonian in the normal-ordered form
    shift : float
        Energy denominator shift for stabilizing update
    acparray : List[float]
        List containing the ACP scaling factors for the 5 T2**2 diagrams

    Returns
    -------
    T : ClusterOperator
        Cluster operator containing updated t2ab(ab~ij~) component in T.ab
    dT : ClusterOperator
        Residual of CC amplitude equation containing <ij~ab~|(H_N exp(T1+T2)_C|0>/D_MP(ab~ij~) in dT.ab
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
    # dT.ab += 0.5 * (d1 + d2) * np.einsum("mnef,aeim,fbnj->abij", H0.aa.oovv, T.aa, T.ab, optimize=True) #D1+D2 ##
    dT.ab += acparray[0] * np.einsum("mnef,aeim,fbnj->abij", H0.ab.oovv, T.aa, T.ab, optimize=True)  # D1,DIR
    dT.ab -= acparray[1] * np.einsum("mnfe,aeim,fbnj->abij", H0.ab.oovv, T.aa, T.ab, optimize=True)  # D2,EXCH

    dT.ab += acparray[0] * np.einsum("mnef,aeim,bfjn->abij", H0.ab.oovv, T.aa, T.bb, optimize=True)  # D1
    dT.ab += acparray[0] * np.einsum("mnef,afin,ebmj->abij", H0.ab.oovv, T.ab, T.ab, optimize=True)  # D1

    # dT.ab += 0.5 * (d1 + s2) * np.einsum("mnef,aeim,bfjn->abij", H0.bb.oovv, T.ab, T.bb, optimize=True) # D1+D2
    dT.ab += acparray[0] * np.einsum("mnef,aeim,bfjn->abij", H0.ab.oovv, T.ab, T.bb, optimize=True)  # D1,DIR
    dT.ab -= acparray[1] * np.einsum("mnfe,aeim,bfjn->abij", H0.ab.oovv, T.ab, T.bb, optimize=True)  # D2,EXCH

    dT.ab += acparray[1] * np.einsum("mnef,ebin,afmj->abij", H0.ab.oovv, T.ab, T.ab, optimize=True)  # D2
    dT.ab += acparray[4] * np.einsum("mnef,efij,abmn->abij", H0.ab.oovv, T.ab, T.ab, optimize=True)  # D5

    dT.ab += acparray[3] * 0.5 * np.einsum("mnef,efni,abmj->abij", H0.aa.oovv, T.aa, T.ab, optimize=True)  # D4
    dT.ab -= acparray[3] * np.einsum("mnef,efin,abmj->abij", H0.ab.oovv, T.ab, T.ab, optimize=True)  # D4
    dT.ab -= acparray[3] * np.einsum("mnef,efmj,abin->abij", H0.ab.oovv, T.ab, T.ab, optimize=True)  # D4
    dT.ab += acparray[3] * 0.5 * np.einsum("mnef,efnj,abim->abij", H0.bb.oovv, T.bb, T.ab, optimize=True)  # D4

    dT.ab += acparray[2] * 0.5 * np.einsum("mnef,afnm,ebij->abij", H0.aa.oovv, T.aa, T.ab, optimize=True) #D3 ##
    dT.ab -= acparray[2] * np.einsum("mnef,afmn,ebij->abij", H0.ab.oovv, T.ab, T.ab, optimize=True) #D3 ##
    dT.ab -= acparray[2] * np.einsum("mnef,ebmn,afij->abij", H0.ab.oovv, T.ab, T.ab, optimize=True) #D3 ##
    dT.ab += acparray[2] * 0.5 * np.einsum("mnef,bfnm,aeij->abij", H0.bb.oovv, T.bb, T.ab, optimize=True) #D3 ##
    T.ab, dT.ab = cc_loops2.update_t2b(
        T.ab, dT.ab + H0.ab.vvoo, H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv, shift
    )
    return T, dT


def update_t2c(T: ClusterOperator,
               dT: ClusterOperator,
               H: Integral,
               H0: Integral,
               shift: float,
               acparray: List[float]) -> Tuple[ClusterOperator, ClusterOperator]:
    """Update t2bb amplitudes as t2bb(a~b~i~j~) <- t2bb(a~b~i~j~) + <i~j~a~b~|(H_N exp(T1 + T2))_C|0>/D(a~b~i~j~)

    Parameters
    ----------
    T : ClusterOperator
        Cluster operator containing T1 and T2 components
    dT : ClusterOperator
        Residual for T1 and T2 amplitude equations
    H : Integral
        Intermediates for the CC iterations that are roughly given by a CCSD-like similarity-transformed Hamiltonian
    H0 : Integral
        Bare Hamiltonian in the normal-ordered form
    shift : float
        Energy denominator shift for stabilizing update
    acparray : List[float]
        List containing the ACP scaling factors for the 5 T2**2 diagrams

    Returns
    -------
    T : ClusterOperator
        Cluster operator containing updated t2bb(a~b~i~j~) component in T.bb
    dT : ClusterOperator
        Residual of CC amplitude equation containing <i~j~a~b~|(H_N exp(T1+T2)_C|0>/D_MP(a~b~i~j~) in dT.bb
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
    # dT.aa += 0.5 * np.einsum(
    #     "mnef,aeim,bfjn->abij", H.aa.oovv, T.aa, T.aa, optimize=True
    # ) # A(ij) [D1 + D2]
    dT.bb += d1 * 0.5 * np.einsum("nmfe,aeim,bfjn->abij", H0.ab.oovv, T.bb, T.bb, optimize=True)  # A(ij) [D1]
    dT.bb -= d2 * 0.5 * np.einsum("nmef,aeim,bfjn->abij", H0.ab.oovv, T.bb, T.bb, optimize=True)  # A(ij) [D2]
    dT.bb += d5 * 0.25 * 0.25 * np.einsum("mnef,efij,abmn->abij", H0.bb.oovv, T.bb, T.bb, optimize=True)  # 1 [D5]
    dT.bb -= d4 * 0.25 * np.einsum("mnef,abim,efjn->abij", H0.bb.oovv, T.bb, T.bb, optimize=True)  # A(ij) [D4]
    dT.bb -= d3 * 0.25 * np.einsum("mnef,aeij,bfmn->abij", H0.bb.oovv, T.bb, T.bb, optimize=True)  # A(ab) [D3]
    dT.bb += d1 * np.einsum("nmfe,aeim,fbnj->abij", H0.ab.oovv, T.bb, T.ab, optimize=True)  # A(ij)A(ab) [D1]
    dT.bb -= d4 * 0.5 * np.einsum("nmfe,abim,fenj->abij", H0.ab.oovv, T.bb, T.ab, optimize=True)  # A(ij) [D4]
    dT.bb -= d3 * 0.5 * np.einsum("nmfe,aeij,fbnm->abij", H0.ab.oovv, T.bb, T.ab, optimize=True)  # A(ab) [D3]
    # dT.bb += 0.5 * np.einsum(
    #     "mnef,aeim,bfjn->abij", H.bb.oovv, T.ab, T.ab, optimize=True
    # ) # A(ij) [D1 + D2]
    dT.bb += d1 * 0.5 * np.einsum("nmfe,eami,fbnj->abij", H0.ab.oovv, T.ab, T.ab, optimize=True)  # A(ij) [D1]
    dT.bb -= d2 * 0.5 * np.einsum("nmef,eami,fbnj->abij", H0.ab.oovv, T.ab, T.ab, optimize=True)  # A(ij) [D2]

    T.bb, dT.bb = cc_loops2.update_t2c(
        T.bb, dT.bb + 0.25 * H0.bb.vvoo, H0.b.oo, H0.b.vv, shift
    )
    return T, dT
