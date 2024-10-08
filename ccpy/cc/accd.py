"""
Module with functions that help perform the approximate coupled-pair (ACP) coupled-cluster (CC)
approach with doubles, abbreviated as ACCD.
"""
import numpy as np
# Modules for type checking
from typing import List, Tuple
from ccpy.models.operators import ClusterOperator
from ccpy.models.system import System
from ccpy.models.integrals import Integral
# Modules for computation
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
    Performs one update of the CC amplitude equations for the ACCD method.

    Parameters
    ----------
    T : ClusterOperator
        Cluster operator containing  T2 components
    dT : ClusterOperator
        Residual of the CC amplitude equations corresponding to projections onto doubles
    H : Integral
        Bare Hamiltonian in the normal-ordered form
    X : Integral
        Intermediates for CC iterations. Not used in the ACCD routine.
    shift : float
        Energy denominator shift for stabilizing update in case of strong quasidegeneracy.
    flag_RHF : bool
        Flag to turn on/off RHF symmetry. Doing so skips updating components of T that are equivalent for closed shells.
    system : System
        System object containing information about the molecular system, such as orbital dimensions. Not used in ACCD routine.
    acparray : List[float]
        List containing the ACP scaling factors for the 5 T2**2 diagrams

    Returns
    -------
    T : ClusterOperator
        Cluster operator with updated T2 component
    dT : ClusterOperator
        Residual of the ACCD amplitude equations corresponding to projections onto doubles
    """
    # update T2
    T, dT = update_t2a(T, dT, H, shift, acparray)
    T, dT = update_t2b(T, dT, H, shift, acparray)
    if flag_RHF:
        T.bb = T.aa.copy()
        dT.bb = dT.aa.copy()
    else:
        T, dT = update_t2c(T, dT, H, shift, acparray)

    return T, dT


def update_t2a(T: ClusterOperator,
               dT: ClusterOperator,
               H: Integral,
               shift: float,
               acparray: List[float]) -> Tuple[ClusterOperator, ClusterOperator]:
    """Update t2aa amplitudes as t2aa(abij) <- t2ab(abij) + <ijab|(H_N exp(T2))_C|0>/D_MP(abij)

    Parameters
    ----------
    T : ClusterOperator
        Cluster operator containing T2 component
    dT : ClusterOperator
        Residual for T2 amplitude equations
    H : Integral
        Bare Hamiltonian in the normal-ordered form
    shift : float
        Energy denominator shift for stabilizing update
    acparray : List[float]
        List containing the ACP scaling factors for the 5 T2**2 diagrams

    Returns
    -------
    T : ClusterOperator
        Cluster operator with updated T.ab component
    dT : ClusterOperator
        Residual of CC amplitude equation with updated component dT.aa
    """
    d1, d2, d3, d4, d5 = acparray

    # < ijab | (F T2)_C | 0 >
    dT.aa = -0.5 * np.einsum("mi,abmj->abij", H.a.oo, T.aa, optimize=True)  # A(ij)
    dT.aa += 0.5 * np.einsum("ae,ebij->abij", H.a.vv, T.aa, optimize=True)  # A(ab)

    # < ijab | (V T2)_C | 0 >
    dT.aa += np.einsum("amie,ebmj->abij", H.aa.voov, T.aa, optimize=True)  # A(ab)A(ij)
    dT.aa += np.einsum("amie,bejm->abij", H.ab.voov, T.ab, optimize=True)  # A(ab)A(ij)
    dT.aa += 0.125 * np.einsum("mnij,abmn->abij", H.aa.oooo, T.aa, optimize=True)  # 1
    dT.aa += 0.125 * np.einsum("abef,efij->abij", H.aa.vvvv, T.aa, optimize=True)  # 1

    # < ijab | (V T2**2)_C | 0 >
    # dT.aa += 0.5 * np.einsum(
    #    "mnef,aeim,bfjn->abij", H.aa.oovv, T.aa, T.aa, optimize=True
    # ) # A(ij) [D1 + D2]
    dT.aa += d1 * 0.5 * np.einsum("mnef,aeim,bfjn->abij", H.ab.oovv, T.aa, T.aa, optimize=True)  # A(ij) [D1]
    dT.aa -= d2 * 0.5 * np.einsum("mnfe,aeim,bfjn->abij", H.ab.oovv, T.aa, T.aa, optimize=True)  # A(ij) [D2]
    dT.aa += d5 * 0.25 * 0.25 * np.einsum("mnef,efij,abmn->abij", H.aa.oovv, T.aa, T.aa, optimize=True)  # 1 [D5]
    dT.aa -= d4 * 0.25 * np.einsum("mnef,abim,efjn->abij", H.aa.oovv, T.aa, T.aa, optimize=True)  # A(ij) [D4]
    dT.aa -= d3 * 0.25 * np.einsum("mnef,aeij,bfmn->abij", H.aa.oovv, T.aa, T.aa, optimize=True)  # A(ab) [D3]
    dT.aa += d1 * np.einsum("mnef,aeim,bfjn->abij", H.ab.oovv, T.aa, T.ab, optimize=True)  # A(ij)A(ab) [D1]
    dT.aa -= d4 * 0.5 * np.einsum("mnef,abim,efjn->abij", H.ab.oovv, T.aa, T.ab, optimize=True)  # A(ij) [D4]
    dT.aa -= d3 * 0.5 * np.einsum("mnef,aeij,bfmn->abij", H.ab.oovv, T.aa, T.ab, optimize=True)  # A(ab) [D3]
    # dT.aa += 0.5 * np.einsum(
    #    "mnef,aeim,bfjn->abij", H.bb.oovv, T.ab, T.ab, optimize=True
    # ) # A(ij) [D1 + D2]
    dT.aa += d1 * 0.5 * np.einsum("mnef,aeim,bfjn->abij", H.ab.oovv, T.ab, T.ab, optimize=True)  # A(ij) [D1]
    dT.aa -= d2 * 0.5 * np.einsum("mnfe,aeim,bfjn->abij", H.ab.oovv, T.ab, T.ab, optimize=True)  # A(ij) [D2]

    T.aa, dT.aa = cc_loops2.update_t2a(
        T.aa, dT.aa + 0.25 * H.aa.vvoo, H.a.oo, H.a.vv, shift
    )
    return T, dT


def update_t2b(T: ClusterOperator,
               dT: ClusterOperator,
               H: Integral,
               shift: float,
               acparray: List[float]) -> Tuple[ClusterOperator, ClusterOperator]:
    """Update t2ab amplitudes as t2ab(abij) <- t2ab(ab~ij~) + <ij~ab~|(H_N exp(T2))_C|0>/D_MP(ab~ij~)

    Parameters
    ----------
    T : ClusterOperator
        Cluster operator containing T2 component
    dT : ClusterOperator
        Residual for T2 amplitude equations
    H : Integral
        Bare Hamiltonian in the normal-ordered form
    shift : float
        Energy denominator shift for stabilizing update
    acparray : List[float]
        List containing the ACP scaling factors for the 5 T2**2 diagrams

    Returns
    -------
    T : ClusterOperator
        Cluster operator with updated T.ab component
    dT : ClusterOperator
        Residual of CC amplitude equation with updated component dT.ab
    """
    d1, d2, d3, d4, d5 = acparray

    # < ijab | (F T2)_C | 0 >
    dT.ab = -np.einsum("mi,abmj->abij", H.a.oo, T.ab, optimize=True)
    dT.ab += np.einsum("ae,ebij->abij", H.a.vv, T.ab, optimize=True)
    dT.ab -= np.einsum("mj,abim->abij", H.b.oo, T.ab, optimize=True)
    dT.ab += np.einsum("be,aeij->abij", H.b.vv, T.ab, optimize=True)

    # < ijab | (V T2)_C | 0 >
    dT.ab += np.einsum("amie,ebmj->abij", H.aa.voov, T.ab, optimize=True)
    dT.ab += np.einsum("amie,ebmj->abij", H.ab.voov, T.bb, optimize=True)
    dT.ab += np.einsum("mbej,aeim->abij", H.ab.ovvo, T.aa, optimize=True)
    dT.ab += np.einsum("bmje,aeim->abij", H.bb.voov, T.ab, optimize=True)
    dT.ab -= np.einsum("mbie,aemj->abij", H.ab.ovov, T.ab, optimize=True)
    dT.ab -= np.einsum("amej,ebim->abij", H.ab.vovo, T.ab, optimize=True)
    dT.ab += np.einsum("mnij,abmn->abij", H.ab.oooo, T.ab, optimize=True)
    dT.ab += np.einsum("abef,efij->abij", H.ab.vvvv, T.ab, optimize=True)

    # < ijab | (V T2**2)_C | 0 >
    # dT.ab += np.einsum("mnef,aeim,fbnj->abij", H.aa.oovv, T.aa, T.ab, optimize=True) # [D1 + D2]
    dT.ab += d1 * np.einsum("mnef,aeim,fbnj->abij", H.ab.oovv, T.aa, T.ab, optimize=True)  # [D1]
    dT.ab -= d2 * np.einsum("mnfe,aeim,fbnj->abij", H.ab.oovv, T.aa, T.ab, optimize=True)  # [D2]
    dT.ab -= d4 * 0.5 * np.einsum("mnef,efin,abmj->abij", H.aa.oovv, T.aa, T.ab, optimize=True)  # [D4]
    dT.ab -= d3 * 0.5 * np.einsum("mnef,afmn,ebij->abij", H.aa.oovv, T.aa, T.ab, optimize=True)  # [D3]

    dT.ab += d1 * np.einsum("nmfe,aeim,fbnj->abij", H.ab.oovv, T.ab, T.ab, optimize=True)  # [D1]
    dT.ab += d2 * np.einsum("mnef,ebin,afmj->abij", H.ab.oovv, T.ab, T.ab, optimize=True)  # [D2]
    dT.ab += d4 * np.einsum("mnef,efij,abmn->abij", H.ab.oovv, T.ab, T.ab, optimize=True)  # [D4]
    dT.ab -= d4 * np.einsum("mnef,efin,abmj->abij", H.ab.oovv, T.ab, T.ab, optimize=True)  # [D4]
    dT.ab -= d5 * np.einsum("nmfe,fenj,abim->abij", H.ab.oovv, T.ab, T.ab, optimize=True)  # [D5]
    dT.ab -= d3 * np.einsum("mnef,afmn,ebij->abij", H.ab.oovv, T.ab, T.ab, optimize=True)  # [D3]
    dT.ab -= d3 * np.einsum("nmfe,fbnm,aeij->abij", H.ab.oovv, T.ab, T.ab, optimize=True)  # [D3]

    dT.ab += d1 * np.einsum("mnef,aeim,fbnj->abij", H.ab.oovv, T.aa, T.bb, optimize=True)  # [D1]

    # dT.ab += np.einsum("mnef,aeim,fbnj->abij", H.bb.oovv, T.ab, T.bb, optimize=True) # [D1 + D2]
    dT.ab += d1 * np.einsum("mnef,aeim,fbnj->abij", H.ab.oovv, T.ab, T.bb, optimize=True)  # [D1]
    dT.ab -= d2 * np.einsum("mnfe,aeim,fbnj->abij", H.ab.oovv, T.ab, T.bb, optimize=True)  # [D2]
    dT.ab -= d4 * 0.5 * np.einsum("mnef,efjn,abim->abij", H.bb.oovv, T.bb, T.ab, optimize=True)  # [D4]
    dT.ab -= d3 * 0.5 * np.einsum("mnef,bfmn,aeij->abij", H.bb.oovv, T.bb, T.ab, optimize=True)  # [D3]

    T.ab, dT.ab = cc_loops2.update_t2b(
        T.ab, dT.ab + H.ab.vvoo, H.a.oo, H.a.vv, H.b.oo, H.b.vv, shift
    )
    return T, dT


def update_t2c(T: ClusterOperator,
               dT: ClusterOperator,
               H: Integral,
               shift: float,
               acparray: List[float]) -> Tuple[ClusterOperator, ClusterOperator]:
    """Update t2bb amplitudes as t2bb(a~b~i~j~) <- t2bb(a~b~i~j~) + <i~j~a~b~|(H_N exp(T2))_C|0>/D_MP(a~b~i~j~)

    Parameters
    ----------
    T : ClusterOperator
        Cluster operator containing T2 component
    dT : ClusterOperator
        Residual for T2 amplitude equations
    H : Integral
        Bare Hamiltonian in the normal-ordered form
    shift : float
        Energy denominator shift for stabilizing update
    acparray : List[float]
        List containing the ACP scaling factors for the 5 T2**2 diagrams

    Returns
    -------
    T : ClusterOperator
        Cluster operator with updated T.bb component
    dT : ClusterOperator
        Residual of CC amplitude equation with updated component dT.bb
    """
    d1, d2, d3, d4, d5 = acparray

    # < ijab | (F T2)_C | 0 >
    dT.bb = -0.5 * np.einsum("mi,abmj->abij", H.b.oo, T.bb, optimize=True)  # A(ij)
    dT.bb += 0.5 * np.einsum("ae,ebij->abij", H.b.vv, T.bb, optimize=True)  # A(ab)

    # < ijab | (V T2)_C | 0 >
    dT.bb += np.einsum("amie,ebmj->abij", H.bb.voov, T.bb, optimize=True)  # A(ab)A(ij)
    dT.bb += np.einsum("maei,ebmj->abij", H.ab.ovvo, T.ab, optimize=True)  # A(ab)A(ij)
    dT.bb += 0.125 * np.einsum("mnij,abmn->abij", H.bb.oooo, T.bb, optimize=True)  # 1
    dT.bb += 0.125 * np.einsum("abef,efij->abij", H.bb.vvvv, T.bb, optimize=True)  # 1

    # < ijab | (V T2**2)_C | 0 >
    # dT.bb += 0.5 * np.einsum(
    #    "mnef,aeim,bfjn->abij", H.bb.oovv, T.bb, T.bb, optimize=True
    # ) # A(ij) [D1 + D2]
    dT.bb += d1 * 0.5 * np.einsum("nmfe,aeim,bfjn->abij", H.ab.oovv, T.bb, T.bb, optimize=True)  # A(ij) [D1]
    dT.bb -= d2 * 0.5 * np.einsum("nmef,aeim,bfjn->abij", H.ab.oovv, T.bb, T.bb, optimize=True)  # A(ij) [D2]
    dT.bb += d5 * 0.25 * 0.25 * np.einsum("mnef,efij,abmn->abij", H.bb.oovv, T.bb, T.bb, optimize=True)  # 1 [D5]
    dT.bb -= d4 * 0.25 * np.einsum("mnef,abim,efjn->abij", H.bb.oovv, T.bb, T.bb, optimize=True)  # A(ij) [D4]
    dT.bb -= d3 * 0.25 * np.einsum("mnef,aeij,bfmn->abij", H.bb.oovv, T.bb, T.bb, optimize=True)  # A(ab) [D3]
    dT.bb += d1 * np.einsum("nmfe,aeim,fbnj->abij", H.ab.oovv, T.bb, T.ab, optimize=True)  # A(ij)A(ab) [D1]
    dT.bb -= d4 * 0.5 * np.einsum("nmfe,abim,fenj->abij", H.ab.oovv, T.bb, T.ab, optimize=True)  # A(ij) [D4]
    dT.bb -= d3 * 0.5 * np.einsum("nmfe,aeij,fbnm->abij", H.ab.oovv, T.bb, T.ab, optimize=True)  # A(ab) [D3]
    # dT.bb += 0.5 * np.einsum(
    #    "mnef,aeim,bfjn->abij", H.bb.oovv, T.ab, T.ab, optimize=True
    # ) # A(ij) [D1 + D2]
    dT.bb += d1 * 0.5 * np.einsum("nmfe,eami,fbnj->abij", H.ab.oovv, T.ab, T.ab, optimize=True)  # A(ij) [D1]
    dT.bb -= d2 * 0.5 * np.einsum("nmef,eami,fbnj->abij", H.ab.oovv, T.ab, T.ab, optimize=True)  # A(ij) [D2]

    T.bb, dT.bb = cc_loops2.update_t2c(
        T.bb, dT.bb + 0.25 * H.bb.vvoo, H.b.oo, H.b.vv, shift
    )
    return T, dT
