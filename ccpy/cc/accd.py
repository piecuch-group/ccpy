"""Module with functions that perform the CC with doubles
(CCD) calculation for a molecular system."""
import numpy as np
import time

from ccpy.utilities.updates import cc_loops2


def update(T, dT, H, shift, flag_RHF):

    # update T2
    T, dT = update_t2a(T, dT, H, shift)
    T, dT = update_t2b(T, dT, H, shift)
    if flag_RHF:
        T.bb = T.aa.copy()
        dT.bb = dT.aa.copy()
    else:
        T, dT = update_t2c(T, dT, H, shift)

    return T, dT

def update_t2a(T, dT, H, shift):

    # < ijab | (F T2)_C | 0 >
    dT.aa = -0.5 * np.einsum("mi,abmj->abij", H.a.oo, T.aa, optimize=True)  # A(ij)
    dT.aa += 0.5*np.einsum("ae,ebij->abij", H.a.vv, T.aa, optimize=True)  # A(ab)

    # < ijab | (V T2)_C | 0 >
    dT.aa += np.einsum("amie,ebmj->abij", H.aa.voov, T.aa, optimize=True)  # A(ab)A(ij)
    dT.aa += np.einsum("amie,bejm->abij", H.ab.voov, T.ab, optimize=True)  # A(ab)A(ij)
    dT.aa += 0.125 * np.einsum("mnij,abmn->abij", H.aa.oooo, T.aa, optimize=True)  # 1
    dT.aa += 0.125 * np.einsum("abef,efij->abij", H.aa.vvvv, T.aa, optimize=True)  # 1

    # < ijab | (V T2**2)_C | 0 >
    dT.aa += 0.5 * np.einsum(
        "mnef,aeim,bfjn->abij", H.aa.oovv, T.aa, T.aa, optimize=True
    ) # A(ij)
    dT.aa += np.einsum(
        "mnef,aeim,bfjn->abij", H.ab.oovv, T.aa, T.ab, optimize=True
    ) # A(ij)A(ab)
    dT.aa += 0.5 * np.einsum(
        "mnef,aeim,bfjn->abij", H.bb.oovv, T.ab, T.ab, optimize=True
    ) # A(ij)
    dT.aa += 0.125 * np.einsum(
        "mnef,efij,abmn->abij", H.aa.oovv, T.aa, T.aa, optimize=True
    ) # 1
    dT.aa -= 0.25 * np.einsum(
        "mnef,abim,efjn->abij", H.aa.oovv, T.aa, T.aa, optimize=True
    ) # A(ij)
    dT.aa -= 0.5 * np.einsum(
        "mnef,abim,efjn->abij", H.ab.oovv, T.aa, T.ab, optimize=True
    ) # A(ij)
    dT.aa -= 0.25 * np.einsum(
        "mnef,aeij,bfmn->abij", H.aa.oovv, T.aa, T.aa, optimize=True
    ) # A(ab)
    dT.aa -= 0.5 * np.einsum(
        "mnef,aeij,bfmn->abij", H.ab.oovv, T.aa, T.ab, optimize=True
    ) # A(ab)


    T.aa, dT.aa = cc_loops2.cc_loops2.update_t2a(
        T.aa, dT.aa + 0.25 * H.aa.vvoo, H.a.oo, H.a.vv, shift
    )
    return T, dT


def update_t2b(T, dT, H, shift):

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
    dT.ab += np.einsum("mnef,aeim,fbnj->abij", H.aa.oovv, T.aa, T.ab, optimize=True)
    dT.ab += np.einsum("mnef,aeim,fbnj->abij", H.ab.oovv, T.aa, T.bb, optimize=True)
    dT.ab += np.einsum("nmfe,aeim,fbnj->abij", H.ab.oovv, T.ab, T.ab, optimize=True)
    dT.ab += np.einsum("mnef,aeim,fbnj->abij", H.bb.oovv, T.ab, T.bb, optimize=True)
    dT.ab += np.einsum("mnef,ebin,afmj->abij", H.ab.oovv, T.ab, T.ab, optimize=True)
    dT.ab += np.einsum("mnef,efij,abmn->abij", H.ab.oovv, T.ab, T.ab, optimize=True)

    dT.ab -= 0.5 * np.einsum("mnef,efin,abmj->abij", H.aa.oovv, T.aa, T.ab, optimize=True)
    dT.ab -= np.einsum("mnef,efin,abmj->abij", H.ab.oovv, T.ab, T.ab, optimize=True)
    dT.ab -= np.einsum("nmfe,fenj,abim->abij", H.ab.oovv, T.ab, T.ab, optimize=True)
    dT.ab -= 0.5 * np.einsum("mnef,efjn,abim->abij", H.bb.oovv, T.bb, T.ab, optimize=True)
    dT.ab -= 0.5 * np.einsum("mnef,afmn,ebij->abij", H.aa.oovv, T.aa, T.ab, optimize=True)
    dT.ab -= np.einsum("mnef,afmn,ebij->abij", H.ab.oovv, T.ab, T.ab, optimize=True)
    dT.ab -= np.einsum("nmfe,fbnm,aeij->abij", H.ab.oovv, T.ab, T.ab, optimize=True)
    dT.ab -= 0.5 * np.einsum("mnef,bfmn,aeij->abij", H.bb.oovv, T.bb, T.ab, optimize=True)

    T.ab, dT.ab = cc_loops2.cc_loops2.update_t2b(
        T.ab, dT.ab + H.ab.vvoo, H.a.oo, H.a.vv, H.b.oo, H.b.vv, shift
    )
    return T, dT


def update_t2c(T, dT, H, shift):



    # < ijab | (F T2)_C | 0 >
    dT.bb = -0.5 * np.einsum("mi,abmj->abij", H.b.oo, T.bb, optimize=True)  # A(ij)
    dT.bb += 0.5*np.einsum("ae,ebij->abij", H.b.vv, T.bb, optimize=True)  # A(ab)

    # < ijab | (V T2)_C | 0 >
    dT.bb += np.einsum("amie,ebmj->abij", H.bb.voov, T.bb, optimize=True)  # A(ab)A(ij)
    dT.bb += np.einsum("maei,bejm->abij", H.ab.ovvo, T.ab, optimize=True)  # A(ab)A(ij)
    dT.bb += 0.125 * np.einsum("mnij,abmn->abij", H.bb.oooo, T.bb, optimize=True)  # 1
    dT.bb += 0.125 * np.einsum("abef,efij->abij", H.bb.vvvv, T.bb, optimize=True)  # 1

    # < ijab | (V T2**2)_C | 0 >
    dT.bb += 0.5 * np.einsum(
        "mnef,aeim,bfjn->abij", H.bb.oovv, T.bb, T.bb, optimize=True
    ) # A(ij)
    dT.bb += np.einsum(
        "nmfe,aeim,fbnj->abij", H.ab.oovv, T.bb, T.ab, optimize=True
    ) # A(ij)A(ab)
    dT.bb += 0.5 * np.einsum(
        "nmfe,eami,fbnj->abij", H.aa.oovv, T.ab, T.ab, optimize=True
    ) # A(ij)
    dT.bb += 0.125 * np.einsum(
        "mnef,efij,abmn->abij", H.bb.oovv, T.bb, T.bb, optimize=True
    ) # 1
    dT.bb -= 0.25 * np.einsum(
        "mnef,abim,efjn->abij", H.bb.oovv, T.bb, T.bb, optimize=True
    ) # A(ij)
    dT.bb -= 0.5 * np.einsum(
        "nmfe,abim,fenj->abij", H.ab.oovv, T.bb, T.ab, optimize=True
    ) # A(ij)
    dT.bb -= 0.25 * np.einsum(
        "mnef,aeij,bfmn->abij", H.bb.oovv, T.bb, T.bb, optimize=True
    ) # A(ab)
    dT.bb -= 0.5 * np.einsum(
        "nmfe,aeij,fbnm->abij", H.ab.oovv, T.bb, T.ab, optimize=True
    ) # A(ab)


    T.bb, dT.bb = cc_loops2.cc_loops2.update_t2c(
        T.bb, dT.bb + 0.25 * H.bb.vvoo, H.b.oo, H.b.vv, shift
    )
    return T, dT

    # # < ijab | (F T2)_C | 0 >
    # D1 = -np.einsum("mi,abmj->abij", H.b.oo, T.bb, optimize=True)  # A(ij)
    # D2 = np.einsum("ae,ebij->abij", H.b.vv, T.bb, optimize=True)  # A(ab)
    #
    # # < ijab | (V T2)_C | 0 >
    # D3 = np.einsum("amie,ebmj->abij", H.bb.voov, T.bb, optimize=True)  # A(ab)A(ij)
    # D4 = np.einsum("maei,ebmj->abij", H.ab.ovvo, T.ab, optimize=True)  # A(ab)A(ij)
    # D5 = 0.5 * np.einsum("mnij,abmn->abij", vC["oooo"], T.bb, optimize=True)  # 1
    # D6 = 0.5 * np.einsum("abef,efij->abij", vC["vvvv"], T.bb, optimize=True)  # 1
    #
    # # < ijab | (V T2**2)_C | 0 >
    # D7 = np.einsum("mnef,aeim,bfjn->abij", H.bb.oovv, T.bb, T.bb, optimize=True)  # A(ij)
    # D8 = np.einsum(
    #     "nmfe,aeim,fbnj->abij", H.ab.oovv, T.bb, T.ab, optimize=True
    # )  # A(ij)A(ab)
    # D9 = np.einsum("mnef,eami,fbnj->abij", H.aa.oovv, T.ab, T.ab, optimize=True)  # A(ij)
    # D10 = 0.25 * np.einsum(
    #     "mnef,efij,abmn->abij", H.bb.oovv, T.bb, T.bb, optimize=True
    # )  # 1
    # D11 = -0.5 * np.einsum(
    #     "mnef,abim,efjn->abij", H.bb.oovv, T.bb, T.bb, optimize=True
    # )  # A(ij)
    # D12 = -np.einsum(
    #     "nmfe,abim,fenj->abij", H.ab.oovv, T.bb, T.ab, optimize=True
    # )  # A(ij)
    # D13 = -0.5 * np.einsum(
    #     "mnef,aeij,bfmn->abij", H.bb.oovv, T.bb, T.bb, optimize=True
    # )  # A(ab)
    # D14 = -np.einsum(
    #     "nmfe,aeij,fbnm->abij", H.ab.oovv, T.bb, T.ab, optimize=True
    # )  # A(ab)
    #
    # X2C = (
    #     0.5 * (D1 + D2 + D7 + D9 + D11 + D12 + D13 + D14)
    #     + 0.25 * (D5 + D6 + D10 + vC["vvoo"])
    #     + (D3 + D4 + D8)
    # )

