"""
Module with functions to perform the coupled-cluster (CC) approach with doubles, 
abbreviated as CCD (originally known as coupled-pair many-electron theory, or CPMET).

References:
    [1] J. Cizek, J. Chem. Phys. 45, 4256 (1966).
"""
import numpy as np

from ccpy.lib.core import cc_loops2


def update(T, dT, H, X, shift, flag_RHF, system):

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
    """
    Update t2a amplitudes by calculating the projection <ijab|(H_N e^T2)_C|0>.
    """
    # intermediates
    I1A_oo = (
        H.a.oo
        + 0.5 * np.einsum("mnef,efin->mi", H.aa.oovv, T.aa, optimize=True)
        + np.einsum("mnef,efin->mi", H.ab.oovv, T.ab, optimize=True)
    )

    I1A_vv = (
        H.a.vv
        - 0.5 * np.einsum("mnef,afmn->ae", H.aa.oovv, T.aa, optimize=True)
        - np.einsum("mnef,afmn->ae", H.ab.oovv, T.ab, optimize=True)
    )

    I2A_voov = (
        H.aa.voov
        + 0.5 * np.einsum("mnef,afin->amie", H.aa.oovv, T.aa, optimize=True)
        + np.einsum("mnef,afin->amie", H.ab.oovv, T.ab, optimize=True)
    )

    I2A_oooo = H.aa.oooo + 0.5 * np.einsum(
        "mnef,efij->mnij", H.aa.oovv, T.aa, optimize=True
    )

    I2B_voov = H.ab.voov + 0.5 * np.einsum(
        "mnef,afin->amie", H.bb.oovv, T.ab, optimize=True
    )

    dT.aa = 0.5 * np.einsum("ae,ebij->abij", I1A_vv, T.aa, optimize=True)
    dT.aa -= 0.5 * np.einsum("mi,abmj->abij", I1A_oo, T.aa, optimize=True)
    dT.aa += np.einsum("amie,ebmj->abij", I2A_voov, T.aa, optimize=True)
    dT.aa += np.einsum("amie,bejm->abij", I2B_voov, T.ab, optimize=True)
    dT.aa += 0.125 * np.einsum("abef,efij->abij", H.aa.vvvv, T.aa, optimize=True)
    dT.aa += 0.125 * np.einsum("mnij,abmn->abij", I2A_oooo, T.aa, optimize=True)

    T.aa, dT.aa = cc_loops2.update_t2a(
        T.aa, dT.aa + 0.25 * H.aa.vvoo, H.a.oo, H.a.vv, shift
    )
    return T, dT

def update_t2b(T, dT, H, shift):
    """
    Update t2b amplitudes by calculating the projection <ij~ab~|(H_N e^T2)_C|0>.
    """
    # intermediates
    I1A_vv = (
        H.a.vv
        - 0.5 * np.einsum("mnef,afmn->ae", H.aa.oovv, T.aa, optimize=True)
        - np.einsum("mnef,afmn->ae", H.ab.oovv, T.ab, optimize=True)
    )

    I1B_vv = (
        H.b.vv
        - np.einsum("nmfe,fbnm->be", H.ab.oovv, T.ab, optimize=True)
        - 0.5 * np.einsum("mnef,fbnm->be", H.bb.oovv, T.bb, optimize=True)
    )

    I1A_oo = (
        H.a.oo
        + 0.5 * np.einsum("mnef,efin->mi", H.aa.oovv, T.aa, optimize=True)
        + np.einsum("mnef,efin->mi", H.ab.oovv, T.ab, optimize=True)
    )

    I1B_oo = (
        H.b.oo
        + np.einsum("nmfe,fenj->mj", H.ab.oovv, T.ab, optimize=True)
        + 0.5 * np.einsum("mnef,efjn->mj", H.bb.oovv, T.bb, optimize=True)
    )

    I2A_voov = (
        H.aa.voov
        + np.einsum("mnef,aeim->anif", H.aa.oovv, T.aa, optimize=True)
        + np.einsum("nmfe,aeim->anif", H.ab.oovv, T.ab, optimize=True)
    )

    I2B_voov = (
        H.ab.voov
        + np.einsum("mnef,aeim->anif", H.ab.oovv, T.aa, optimize=True)
        + np.einsum("mnef,aeim->anif", H.bb.oovv, T.ab, optimize=True)
    )

    I2B_oooo = H.ab.oooo + np.einsum("mnef,efij->mnij", H.ab.oovv, T.ab, optimize=True)

    I2B_vovo = H.ab.vovo - np.einsum("mnef,afmj->anej", H.ab.oovv, T.ab, optimize=True)

    dT.ab = np.einsum("ae,ebij->abij", I1A_vv, T.ab, optimize=True)
    dT.ab += np.einsum("be,aeij->abij", I1B_vv, T.ab, optimize=True)
    dT.ab -= np.einsum("mi,abmj->abij", I1A_oo, T.ab, optimize=True)
    dT.ab -= np.einsum("mj,abim->abij", I1B_oo, T.ab, optimize=True)
    dT.ab += np.einsum("amie,ebmj->abij", I2A_voov, T.ab, optimize=True)
    dT.ab += np.einsum("amie,ebmj->abij", I2B_voov, T.bb, optimize=True)
    dT.ab += np.einsum("mbej,aeim->abij", H.ab.ovvo, T.aa, optimize=True)
    dT.ab += np.einsum("bmje,aeim->abij", H.bb.voov, T.ab, optimize=True)
    dT.ab -= np.einsum("mbie,aemj->abij", H.ab.ovov, T.ab, optimize=True)
    dT.ab -= np.einsum("amej,ebim->abij", I2B_vovo, T.ab, optimize=True)
    dT.ab += np.einsum("mnij,abmn->abij", I2B_oooo, T.ab, optimize=True)
    dT.ab += np.einsum("abef,efij->abij", H.ab.vvvv, T.ab, optimize=True)

    T.ab, dT.ab = cc_loops2.update_t2b(
        T.ab, dT.ab + H.ab.vvoo, H.a.oo, H.a.vv, H.b.oo, H.b.vv, shift
    )
    return T, dT

def update_t2c(T, dT, H, shift):
    """
    Update t2c amplitudes by calculating the projection <i~j~a~b~|(H_N e^T2)_C|0>.
    """
    # intermediates
    I1B_oo = (
        H.b.oo
        + 0.5 * np.einsum("mnef,efin->mi", H.bb.oovv, T.bb, optimize=True)
        + np.einsum("nmfe,feni->mi", H.ab.oovv, T.ab, optimize=True)
    )

    I1B_vv = (
        H.b.vv
        - 0.5 * np.einsum("mnef,afmn->ae", H.bb.oovv, T.bb, optimize=True)
        - np.einsum("nmfe,fanm->ae", H.ab.oovv, T.ab, optimize=True)
    )

    I2C_oooo = H.bb.oooo + 0.5 * np.einsum(
        "mnef,efij->mnij", H.bb.oovv, T.bb, optimize=True
    )

    I2B_ovvo = (
        H.ab.ovvo
        + np.einsum("mnef,afin->maei", H.ab.oovv, T.bb, optimize=True)
        + 0.5 * np.einsum("mnef,fani->maei", H.aa.oovv, T.ab, optimize=True)
    )

    I2C_voov = H.bb.voov + 0.5 * np.einsum(
        "mnef,afin->amie", H.bb.oovv, T.bb, optimize=True
    )

    dT.bb = 0.5 * np.einsum("ae,ebij->abij", I1B_vv, T.bb, optimize=True)
    dT.bb -= 0.5 * np.einsum("mi,abmj->abij", I1B_oo, T.bb, optimize=True)
    dT.bb += np.einsum("amie,ebmj->abij", I2C_voov, T.bb, optimize=True)
    dT.bb += np.einsum("maei,ebmj->abij", I2B_ovvo, T.ab, optimize=True)
    dT.bb += 0.125 * np.einsum("abef,efij->abij", H.bb.vvvv, T.bb, optimize=True)
    dT.bb += 0.125 * np.einsum("mnij,abmn->abij", I2C_oooo, T.bb, optimize=True)

    T.bb, dT.bb = cc_loops2.update_t2c(
        T.bb, dT.bb + 0.25 * H.bb.vvoo, H.b.oo, H.b.vv, shift
    )
    return T, dT
