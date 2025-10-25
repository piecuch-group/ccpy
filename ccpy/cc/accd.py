'''
Approximate Coupled-Pair Method with Doubles (ACCD)
'''

import numpy as np
from ccpy.utilities.linear_algebra import ccpy_einsum
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
    dT.aa = -0.5 * ccpy_einsum("mi,abmj->abij", H.a.oo, T.aa)  # A(ij)
    dT.aa += 0.5 * ccpy_einsum("ae,ebij->abij", H.a.vv, T.aa)  # A(ab)

    # < ijab | (V T2)_C | 0 >
    dT.aa += ccpy_einsum("amie,ebmj->abij", H.aa.voov, T.aa)  # A(ab)A(ij)
    dT.aa += ccpy_einsum("amie,bejm->abij", H.ab.voov, T.ab)  # A(ab)A(ij)
    dT.aa += 0.125 * ccpy_einsum("mnij,abmn->abij", H.aa.oooo, T.aa)  # 1
    dT.aa += 0.125 * ccpy_einsum("abef,efij->abij", H.aa.vvvv, T.aa)  # 1

    # < ijab | (V T2**2)_C | 0 >
    # dT.aa += 0.5 * np.einsum(
    #    "mnef,aeim,bfjn->abij", H.aa.oovv, T.aa, T.aa, optimize=True
    # ) # A(ij) [D1 + D2]
    dT.aa += d1 * 0.5 * ccpy_einsum("mnef,aeim,bfjn->abij", H.ab.oovv, T.aa, T.aa)  # A(ij) [D1]
    dT.aa -= d2 * 0.5 * ccpy_einsum("mnfe,aeim,bfjn->abij", H.ab.oovv, T.aa, T.aa)  # A(ij) [D2]
    dT.aa += d5 * 0.25 * 0.25 * ccpy_einsum("mnef,efij,abmn->abij", H.aa.oovv, T.aa, T.aa)  # 1 [D5]
    dT.aa -= d4 * 0.25 * ccpy_einsum("mnef,abim,efjn->abij", H.aa.oovv, T.aa, T.aa)  # A(ij) [D4]
    dT.aa -= d3 * 0.25 * ccpy_einsum("mnef,aeij,bfmn->abij", H.aa.oovv, T.aa, T.aa)  # A(ab) [D3]
    dT.aa += d1 * ccpy_einsum("mnef,aeim,bfjn->abij", H.ab.oovv, T.aa, T.ab)  # A(ij)A(ab) [D1]
    dT.aa -= d4 * 0.5 * ccpy_einsum("mnef,abim,efjn->abij", H.ab.oovv, T.aa, T.ab)  # A(ij) [D4]
    dT.aa -= d3 * 0.5 * ccpy_einsum("mnef,aeij,bfmn->abij", H.ab.oovv, T.aa, T.ab)  # A(ab) [D3]
    # dT.aa += 0.5 * np.einsum(
    #    "mnef,aeim,bfjn->abij", H.bb.oovv, T.ab, T.ab, optimize=True
    # ) # A(ij) [D1 + D2]
    dT.aa += d1 * 0.5 * ccpy_einsum("mnef,aeim,bfjn->abij", H.ab.oovv, T.ab, T.ab)  # A(ij) [D1]
    dT.aa -= d2 * 0.5 * ccpy_einsum("mnfe,aeim,bfjn->abij", H.ab.oovv, T.ab, T.ab)  # A(ij) [D2]

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
    dT.ab = -ccpy_einsum("mi,abmj->abij", H.a.oo, T.ab)
    dT.ab += ccpy_einsum("ae,ebij->abij", H.a.vv, T.ab)
    dT.ab -= ccpy_einsum("mj,abim->abij", H.b.oo, T.ab)
    dT.ab += ccpy_einsum("be,aeij->abij", H.b.vv, T.ab)

    # < ijab | (V T2)_C | 0 >
    dT.ab += ccpy_einsum("amie,ebmj->abij", H.aa.voov, T.ab)
    dT.ab += ccpy_einsum("amie,ebmj->abij", H.ab.voov, T.bb)
    dT.ab += ccpy_einsum("mbej,aeim->abij", H.ab.ovvo, T.aa)
    dT.ab += ccpy_einsum("bmje,aeim->abij", H.bb.voov, T.ab)
    dT.ab -= ccpy_einsum("mbie,aemj->abij", H.ab.ovov, T.ab)
    dT.ab -= ccpy_einsum("amej,ebim->abij", H.ab.vovo, T.ab)
    dT.ab += ccpy_einsum("mnij,abmn->abij", H.ab.oooo, T.ab)
    dT.ab += ccpy_einsum("abef,efij->abij", H.ab.vvvv, T.ab)

    # < ijab | (V T2**2)_C | 0 >
    # dT.ab += ccpy_einsum("mnef,aeim,fbnj->abij", H.aa.oovv, T.aa, T.ab) # [D1 + D2]
    dT.ab += d1 * ccpy_einsum("mnef,aeim,fbnj->abij", H.ab.oovv, T.aa, T.ab)  # [D1]
    dT.ab -= d2 * ccpy_einsum("mnfe,aeim,fbnj->abij", H.ab.oovv, T.aa, T.ab)  # [D2]
    dT.ab -= d4 * 0.5 * ccpy_einsum("mnef,efin,abmj->abij", H.aa.oovv, T.aa, T.ab)  # [D4]
    dT.ab -= d3 * 0.5 * ccpy_einsum("mnef,afmn,ebij->abij", H.aa.oovv, T.aa, T.ab)  # [D3]

    dT.ab += d1 * ccpy_einsum("nmfe,aeim,fbnj->abij", H.ab.oovv, T.ab, T.ab)  # [D1]
    dT.ab += d2 * ccpy_einsum("mnef,ebin,afmj->abij", H.ab.oovv, T.ab, T.ab)  # [D2]
    dT.ab += d4 * ccpy_einsum("mnef,efij,abmn->abij", H.ab.oovv, T.ab, T.ab)  # [D4]
    dT.ab -= d4 * ccpy_einsum("mnef,efin,abmj->abij", H.ab.oovv, T.ab, T.ab)  # [D4]
    dT.ab -= d5 * ccpy_einsum("nmfe,fenj,abim->abij", H.ab.oovv, T.ab, T.ab)  # [D5]
    dT.ab -= d3 * ccpy_einsum("mnef,afmn,ebij->abij", H.ab.oovv, T.ab, T.ab)  # [D3]
    dT.ab -= d3 * ccpy_einsum("nmfe,fbnm,aeij->abij", H.ab.oovv, T.ab, T.ab)  # [D3]

    dT.ab += d1 * ccpy_einsum("mnef,aeim,fbnj->abij", H.ab.oovv, T.aa, T.bb)  # [D1]

    # dT.ab += ccpy_einsum("mnef,aeim,fbnj->abij", H.bb.oovv, T.ab, T.bb) # [D1 + D2]
    dT.ab += d1 * ccpy_einsum("mnef,aeim,fbnj->abij", H.ab.oovv, T.ab, T.bb)  # [D1]
    dT.ab -= d2 * ccpy_einsum("mnfe,aeim,fbnj->abij", H.ab.oovv, T.ab, T.bb)  # [D2]
    dT.ab -= d4 * 0.5 * ccpy_einsum("mnef,efjn,abim->abij", H.bb.oovv, T.bb, T.ab)  # [D4]
    dT.ab -= d3 * 0.5 * ccpy_einsum("mnef,bfmn,aeij->abij", H.bb.oovv, T.bb, T.ab)  # [D3]

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
    dT.bb = -0.5 * ccpy_einsum("mi,abmj->abij", H.b.oo, T.bb)  # A(ij)
    dT.bb += 0.5 * ccpy_einsum("ae,ebij->abij", H.b.vv, T.bb)  # A(ab)

    # < ijab | (V T2)_C | 0 >
    dT.bb += ccpy_einsum("amie,ebmj->abij", H.bb.voov, T.bb)  # A(ab)A(ij)
    dT.bb += ccpy_einsum("maei,ebmj->abij", H.ab.ovvo, T.ab)  # A(ab)A(ij)
    dT.bb += 0.125 * ccpy_einsum("mnij,abmn->abij", H.bb.oooo, T.bb)  # 1
    dT.bb += 0.125 * ccpy_einsum("abef,efij->abij", H.bb.vvvv, T.bb)  # 1

    # < ijab | (V T2**2)_C | 0 >
    # dT.bb += 0.5 * np.einsum(
    #    "mnef,aeim,bfjn->abij", H.bb.oovv, T.bb, T.bb, optimize=True
    # ) # A(ij) [D1 + D2]
    dT.bb += d1 * 0.5 * ccpy_einsum("nmfe,aeim,bfjn->abij", H.ab.oovv, T.bb, T.bb)  # A(ij) [D1]
    dT.bb -= d2 * 0.5 * ccpy_einsum("nmef,aeim,bfjn->abij", H.ab.oovv, T.bb, T.bb)  # A(ij) [D2]
    dT.bb += d5 * 0.25 * 0.25 * ccpy_einsum("mnef,efij,abmn->abij", H.bb.oovv, T.bb, T.bb)  # 1 [D5]
    dT.bb -= d4 * 0.25 * ccpy_einsum("mnef,abim,efjn->abij", H.bb.oovv, T.bb, T.bb)  # A(ij) [D4]
    dT.bb -= d3 * 0.25 * ccpy_einsum("mnef,aeij,bfmn->abij", H.bb.oovv, T.bb, T.bb)  # A(ab) [D3]
    dT.bb += d1 * ccpy_einsum("nmfe,aeim,fbnj->abij", H.ab.oovv, T.bb, T.ab)  # A(ij)A(ab) [D1]
    dT.bb -= d4 * 0.5 * ccpy_einsum("nmfe,abim,fenj->abij", H.ab.oovv, T.bb, T.ab)  # A(ij) [D4]
    dT.bb -= d3 * 0.5 * ccpy_einsum("nmfe,aeij,fbnm->abij", H.ab.oovv, T.bb, T.ab)  # A(ab) [D3]
    # dT.bb += 0.5 * np.einsum(
    #    "mnef,aeim,bfjn->abij", H.bb.oovv, T.ab, T.ab, optimize=True
    # ) # A(ij) [D1 + D2]
    dT.bb += d1 * 0.5 * ccpy_einsum("nmfe,eami,fbnj->abij", H.ab.oovv, T.ab, T.ab)  # A(ij) [D1]
    dT.bb -= d2 * 0.5 * ccpy_einsum("nmef,eami,fbnj->abij", H.ab.oovv, T.ab, T.ab)  # A(ij) [D2]

    T.bb, dT.bb = cc_loops2.update_t2c(
        T.bb, dT.bb + 0.25 * H.bb.vvoo, H.b.oo, H.b.vv, shift
    )
    return T, dT
