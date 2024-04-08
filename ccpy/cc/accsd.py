"""Module with functions that perform the CC with singles and 
doubles (CCSD) calculation for a molecular system."""
import numpy as np
from ccpy.hbar.hbar_ccs import get_pre_ccs_intermediates, get_ccs_intermediates_opt
from ccpy.utilities.updates import cc_loops2

def update(T, dT, H, X, shift, flag_RHF, system, acparray):

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
    # Remove T2 parts from X.a.oo/X.b.oo and X.a.vv/X.b.vv
    X.a.vv += (
            + 0.5 * np.einsum("mnef,afmn->ae", H.aa.oovv, T.aa, optimize=True) #
            + np.einsum("mnef,afmn->ae", H.ab.oovv, T.ab, optimize=True) #
    )
    X.a.oo -= (
            + 0.5 * np.einsum("mnef,efin->mi", H.aa.oovv, T.aa, optimize=True) # 
            + np.einsum("mnef,efin->mi", H.ab.oovv, T.ab, optimize=True) #
    )
    if flag_RHF:
        X.b.vv = X.a.vv
        X.b.oo = X.a.oo
    else:
        X.b.vv += (
                    + 0.5 * np.einsum("mnef,afmn->ae", H.bb.oovv, T.bb, optimize=True) #
                    + np.einsum("nmfe,fanm->ae", H.ab.oovv, T.ab, optimize=True) #
        )
        X.b.oo -= (
                    + 0.5 * np.einsum("mnef,efin->mi", H.bb.oovv, T.bb, optimize=True) # 
                    + np.einsum("nmfe,feni->mi", H.ab.oovv, T.ab, optimize=True) #
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
    dT.a += 0.5 * np.einsum("anef,efin->ai", H.aa.vovv, T.aa, optimize=True)
    dT.a += np.einsum("anef,efin->ai", H.ab.vovv, T.ab, optimize=True)
    T.a, dT.a = cc_loops2.cc_loops2.update_t1a(
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
    dT.b += 0.5 * np.einsum("anef,efin->ai", H.bb.vovv, T.bb, optimize=True)
    dT.b += np.einsum("nafe,feni->ai", H.ab.ovvv, T.ab, optimize=True)
    T.b, dT.b = cc_loops2.cc_loops2.update_t1b(
        T.b, dT.b + H.b.vo, H.b.oo, H.b.vv, shift
    )
    return T, dT


# @profile
def update_t2a(T, dT, H, H0, shift, acparray):
    """
    Update t2a amplitudes by calculating the projection <ijab|(H_N e^(T1+T2))_C|0>.
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

    ## < ijab | (V T2**2)_C | 0 >
    ## dT.aa += 0.5 * (acparray[0] + acparray[1]) * 0.5 * np.einsum(
    ##     "mnef,aeim,bfjn->abij", H0.aa.oovv, T.aa, T.aa, optimize=True
    ## ) # A(ij),D1+D2 ##
    #dT.aa += acparray[0] * 0.5 * np.einsum(
    #    "mnef,aeim,bfjn->abij", H0.ab.oovv, T.aa, T.aa, optimize=True
    #) # A(ij),D1,DIR ##
    #dT.aa -= acparray[1] * 0.5 * np.einsum(
    #    "mnfe,aeim,bfjn->abij", H0.ab.oovv, T.aa, T.aa, optimize=True
    #) # A(ij),D2,EXCH ##

    #dT.aa += acparray[0] * np.einsum(
    #    "mnef,aeim,bfjn->abij", H0.ab.oovv, T.aa, T.ab, optimize=True
    #) # A(ij)A(ab),D1 ##

    ## dT.aa += 0.5 * (acparray[0] + acparray[1]) * 0.5 * np.einsum(
    ##     "mnef,aeim,bfjn->abij", H0.bb.oovv, T.ab, T.ab, optimize=True
    ## ) # A(ij),D1+D2 ##
    #dT.aa += acparray[0] * 0.5 * np.einsum(
    #    "mnef,aeim,bfjn->abij", H0.ab.oovv, T.ab, T.ab, optimize=True
    #) # A(ij),D1,DIR ##
    #dT.aa -= acparray[1] * 0.5 * np.einsum(
    #    "mnfe,aeim,bfjn->abij", H0.ab.oovv, T.ab, T.ab, optimize=True
    #) # A(ij),D2,EXCH ##

    #dT.aa += acparray[4] * 0.25 * 0.25 * np.einsum(
    #    "mnef,efij,abmn->abij", H0.aa.oovv, T.aa, T.aa, optimize=True
    #) # 1,D5 ##
    #dT.aa += acparray[3] * 0.25 * np.einsum(
    #    "mnef,efni,abmj->abij", H0.aa.oovv, T.aa, T.aa, optimize=True
    #) # A(ij),D4 ##
    #dT.aa -= acparray[3] * 0.5 * np.einsum(
    #    "mnef,abim,efjn->abij", H0.ab.oovv, T.aa, T.ab, optimize=True
    #) # A(ij),D4 ##
    #dT.aa += acparray[2] * 0.25 * np.einsum(
    #    "mnef,afnm,beji->abij", H0.aa.oovv, T.aa, T.aa, optimize=True
    #) # A(ab),D3 ##
    #dT.aa -= acparray[2] * 0.5 * np.einsum(
    #    "mnef,aeij,bfmn->abij", H0.ab.oovv, T.aa, T.ab, optimize=True
    #) # A(ab),D3 ##


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

    T.aa, dT.aa = cc_loops2.cc_loops2.update_t2a(
        T.aa, dT.aa + 0.25 * H0.aa.vvoo, H0.a.oo, H0.a.vv, shift
    )
    return T, dT


# @profile
def update_t2b(T, dT, H, H0, shift, acparray):
    """
    Update t2b amplitudes by calculating the projection <ij~ab~|(H_N e^(T1+T2))_C|0>.
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
    T.ab, dT.ab = cc_loops2.cc_loops2.update_t2b(
        T.ab, dT.ab + H0.ab.vvoo, H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv, shift
    )
    return T, dT


# @profile
def update_t2c(T, dT, H, H0, shift, acparray):
    """
    Update t2c amplitudes by calculating the projection <i~j~a~b~|(H_N e^(T1+T2))_C|0>.
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

    T.bb, dT.bb = cc_loops2.cc_loops2.update_t2c(
        T.bb, dT.bb + 0.25 * H0.bb.vvoo, H0.b.oo, H0.b.vv, shift
    )
    return T, dT
