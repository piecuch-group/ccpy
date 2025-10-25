'''
Coupled-Cluster Method with Doubles (CCD)
'''

import numpy as np
from ccpy.utilities.linear_algebra import ccpy_einsum

from ccpy.lib.core import cc_loops2


def update(T, dT, H, X, shift, flag_RHF):

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
        + 0.5 * ccpy_einsum("mnef,efin->mi", H.aa.oovv, T.aa)
        + ccpy_einsum("mnef,efin->mi", H.ab.oovv, T.ab)
    )

    I1A_vv = (
        H.a.vv
        - 0.5 * ccpy_einsum("mnef,afmn->ae", H.aa.oovv, T.aa)
        - ccpy_einsum("mnef,afmn->ae", H.ab.oovv, T.ab)
    )

    I2A_voov = (
        H.aa.voov
        + 0.5 * ccpy_einsum("mnef,afin->amie", H.aa.oovv, T.aa)
        + ccpy_einsum("mnef,afin->amie", H.ab.oovv, T.ab)
    )

    I2A_oooo = H.aa.oooo + 0.5 * np.einsum(
        "mnef,efij->mnij", H.aa.oovv, T.aa, optimize=True
    )

    I2B_voov = H.ab.voov + 0.5 * np.einsum(
        "mnef,afin->amie", H.bb.oovv, T.ab, optimize=True
    )

    dT.aa = 0.5 * ccpy_einsum("ae,ebij->abij", I1A_vv, T.aa)
    dT.aa -= 0.5 * ccpy_einsum("mi,abmj->abij", I1A_oo, T.aa)
    dT.aa += ccpy_einsum("amie,ebmj->abij", I2A_voov, T.aa)
    dT.aa += ccpy_einsum("amie,bejm->abij", I2B_voov, T.ab)
    dT.aa += 0.125 * ccpy_einsum("abef,efij->abij", H.aa.vvvv, T.aa)
    dT.aa += 0.125 * ccpy_einsum("mnij,abmn->abij", I2A_oooo, T.aa)

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
        - 0.5 * ccpy_einsum("mnef,afmn->ae", H.aa.oovv, T.aa)
        - ccpy_einsum("mnef,afmn->ae", H.ab.oovv, T.ab)
    )

    I1B_vv = (
        H.b.vv
        - ccpy_einsum("nmfe,fbnm->be", H.ab.oovv, T.ab)
        - 0.5 * ccpy_einsum("mnef,fbnm->be", H.bb.oovv, T.bb)
    )

    I1A_oo = (
        H.a.oo
        + 0.5 * ccpy_einsum("mnef,efin->mi", H.aa.oovv, T.aa)
        + ccpy_einsum("mnef,efin->mi", H.ab.oovv, T.ab)
    )

    I1B_oo = (
        H.b.oo
        + ccpy_einsum("nmfe,fenj->mj", H.ab.oovv, T.ab)
        + 0.5 * ccpy_einsum("mnef,efjn->mj", H.bb.oovv, T.bb)
    )

    I2A_voov = (
        H.aa.voov
        + ccpy_einsum("mnef,aeim->anif", H.aa.oovv, T.aa)
        + ccpy_einsum("nmfe,aeim->anif", H.ab.oovv, T.ab)
    )

    I2B_voov = (
        H.ab.voov
        + ccpy_einsum("mnef,aeim->anif", H.ab.oovv, T.aa)
        + ccpy_einsum("mnef,aeim->anif", H.bb.oovv, T.ab)
    )

    I2B_oooo = H.ab.oooo + ccpy_einsum("mnef,efij->mnij", H.ab.oovv, T.ab)

    I2B_vovo = H.ab.vovo - ccpy_einsum("mnef,afmj->anej", H.ab.oovv, T.ab)

    dT.ab = ccpy_einsum("ae,ebij->abij", I1A_vv, T.ab)
    dT.ab += ccpy_einsum("be,aeij->abij", I1B_vv, T.ab)
    dT.ab -= ccpy_einsum("mi,abmj->abij", I1A_oo, T.ab)
    dT.ab -= ccpy_einsum("mj,abim->abij", I1B_oo, T.ab)
    dT.ab += ccpy_einsum("amie,ebmj->abij", I2A_voov, T.ab)
    dT.ab += ccpy_einsum("amie,ebmj->abij", I2B_voov, T.bb)
    dT.ab += ccpy_einsum("mbej,aeim->abij", H.ab.ovvo, T.aa)
    dT.ab += ccpy_einsum("bmje,aeim->abij", H.bb.voov, T.ab)
    dT.ab -= ccpy_einsum("mbie,aemj->abij", H.ab.ovov, T.ab)
    dT.ab -= ccpy_einsum("amej,ebim->abij", I2B_vovo, T.ab)
    dT.ab += ccpy_einsum("mnij,abmn->abij", I2B_oooo, T.ab)
    dT.ab += ccpy_einsum("abef,efij->abij", H.ab.vvvv, T.ab)

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
        + 0.5 * ccpy_einsum("mnef,efin->mi", H.bb.oovv, T.bb)
        + ccpy_einsum("nmfe,feni->mi", H.ab.oovv, T.ab)
    )

    I1B_vv = (
        H.b.vv
        - 0.5 * ccpy_einsum("mnef,afmn->ae", H.bb.oovv, T.bb)
        - ccpy_einsum("nmfe,fanm->ae", H.ab.oovv, T.ab)
    )

    I2C_oooo = H.bb.oooo + 0.5 * np.einsum(
        "mnef,efij->mnij", H.bb.oovv, T.bb, optimize=True
    )

    I2B_ovvo = (
        H.ab.ovvo
        + ccpy_einsum("mnef,afin->maei", H.ab.oovv, T.bb)
        + 0.5 * ccpy_einsum("mnef,fani->maei", H.aa.oovv, T.ab)
    )

    I2C_voov = H.bb.voov + 0.5 * np.einsum(
        "mnef,afin->amie", H.bb.oovv, T.bb, optimize=True
    )

    dT.bb = 0.5 * ccpy_einsum("ae,ebij->abij", I1B_vv, T.bb)
    dT.bb -= 0.5 * ccpy_einsum("mi,abmj->abij", I1B_oo, T.bb)
    dT.bb += ccpy_einsum("amie,ebmj->abij", I2C_voov, T.bb)
    dT.bb += ccpy_einsum("maei,ebmj->abij", I2B_ovvo, T.ab)
    dT.bb += 0.125 * ccpy_einsum("abef,efij->abij", H.bb.vvvv, T.bb)
    dT.bb += 0.125 * ccpy_einsum("mnij,abmn->abij", I2C_oooo, T.bb)

    T.bb, dT.bb = cc_loops2.update_t2c(
        T.bb, dT.bb + 0.25 * H.bb.vvoo, H.b.oo, H.b.vv, shift
    )
    return T, dT
