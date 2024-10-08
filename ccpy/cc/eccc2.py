"""Module with functions that help perform the externally corrected (ec)
CC (ec-CC) computations that solve for T1 and T2 in the presence of T3
and T4 extracted from an external non-CC source."""

import numpy as np
from ccpy.hbar.hbar_ccs import get_pre_ccs_intermediates, get_ccs_intermediates_opt
from ccpy.lib.core import cc_loops2

def update(T, dT, H, X, shift, flag_RHF, system, T_ext, VT_ext):

    # pre-CCS intermediates
    X = get_pre_ccs_intermediates(X, T, H, system, flag_RHF)

    # update T1
    T, dT = update_t1a(T, dT, H, X, VT_ext, shift)
    if flag_RHF:
        T.b = T.a.copy()
        dT.b = dT.a.copy()
    else:
        T, dT = update_t1b(T, dT, H, X, VT_ext, shift)

    # CCS intermediates
    X = get_ccs_intermediates_opt(X, T, H, system, flag_RHF)

    # update T2
    T, dT = update_t2a(T, dT, X, H, VT_ext, shift, T_ext)
    T, dT = update_t2b(T, dT, X, H, VT_ext, shift, T_ext)
    if flag_RHF:
        T.bb = T.aa.copy()
        dT.bb = dT.aa.copy()
    else:
        T, dT = update_t2c(T, dT, X, H, VT_ext, shift, T_ext)

    return T, dT

def update_t1a(T, dT, H, X, VT_ext, shift):
    """
    Update t1a amplitudes by calculating the projection <ia|(H_N e^(T1+T2+T3(ext)))_C|0>.
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
    T.a, dT.a = cc_loops2.update_t1a(
        T.a,
        dT.a + H.a.vo + VT_ext.a.vo,
        H.a.oo,
        H.a.vv,
        shift,
    )
    return T, dT

def update_t1b(T, dT, H, X, VT_ext, shift):
    """
    Update t1b amplitudes by calculating the projection <i~a~|(H_N e^(T1+T2+T3(ext)))_C|0>.
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
        T.b,
        dT.b + H.b.vo + VT_ext.b.vo,
        H.b.oo,
        H.b.vv,
        shift,
    )
    return T, dT

def update_t2a(T, dT, H, H0, VT_ext, shift, T_ext):
    """
    Update t2a amplitudes by calculating the projection <ijab|(H_N e^(T1+T2+T3(ext)+T4(ext)))_C|0>.
    """
    # intermediates
    I2A_voov = H.aa.voov + (
        + 0.5 * np.einsum("mnef,afin->amie", H0.aa.oovv, T.aa, optimize=True)
        + np.einsum("mnef,afin->amie", H0.ab.oovv, T.ab, optimize=True)
    )
    I2A_oooo = H.aa.oooo + 0.5 * np.einsum("mnef,efij->mnij", H0.aa.oovv, T.aa, optimize=True)
    I2B_voov = H.ab.voov + 0.5 * np.einsum("mnef,afin->amie", H0.bb.oovv, T.ab, optimize=True)
    I2A_vooo = H.aa.vooo + 0.5 * np.einsum('anef,efij->anij', H0.aa.vovv + 0.5 * H.aa.vovv, T.aa, optimize=True)

    tau = 0.5 * T.aa + np.einsum('ai,bj->abij', T.a, T.a, optimize=True)

    dT.aa = -0.5 * np.einsum("amij,bm->abij", I2A_vooo, T.a, optimize=True)
    dT.aa += 0.5 * np.einsum("abie,ej->abij", H.aa.vvov, T.a, optimize=True)
    dT.aa += 0.5 * np.einsum("ae,ebij->abij", H.a.vv, T.aa, optimize=True)
    dT.aa -= 0.5 * np.einsum("mi,abmj->abij", H.a.oo, T.aa, optimize=True)
    dT.aa += np.einsum("amie,ebmj->abij", I2A_voov, T.aa, optimize=True)
    dT.aa += np.einsum("amie,bejm->abij", I2B_voov, T.ab, optimize=True)
    dT.aa += 0.25 * np.einsum("abef,efij->abij", H0.aa.vvvv, tau, optimize=True)
    dT.aa += 0.125 * np.einsum("mnij,abmn->abij", I2A_oooo, T.aa, optimize=True)
    # T3 parts
    dT.aa += 0.25 * np.einsum("me,abeijm->abij", H.a.ov, T_ext.aaa, optimize=True)
    dT.aa += 0.25 * np.einsum("me,abeijm->abij", H.b.ov, T_ext.aab, optimize=True)
    dT.aa -= 0.5 * np.einsum("mnif,abfmjn->abij", H0.ab.ooov + H.ab.ooov, T_ext.aab, optimize=True)
    dT.aa -= 0.25 * np.einsum("mnif,abfmjn->abij", H0.aa.ooov + H.aa.ooov, T_ext.aaa, optimize=True)
    dT.aa += 0.25 * np.einsum("anef,ebfijn->abij", H0.aa.vovv + H.aa.vovv, T_ext.aaa, optimize=True)
    dT.aa += 0.5 * np.einsum("anef,ebfijn->abij", H0.ab.vovv + H.ab.vovv, T_ext.aab, optimize=True)

    T.aa, dT.aa = cc_loops2.update_t2a(
        T.aa,
        dT.aa + 0.25 * (H0.aa.vvoo + VT_ext.aa.vvoo),
        H0.a.oo,
        H0.a.vv,
        shift
    )
    return T, dT

def update_t2b(T, dT, H, H0, VT_ext, shift, T_ext):
    """
    Update t2b amplitudes by calculating the projection <ij~ab~|(H_N e^(T1+T2+T3(ext)+T4(ext)))_C|0>.
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
    # T3 parts
    dT.ab -= 0.5 * np.einsum("mnif,afbmnj->abij", H0.aa.ooov + H.aa.ooov, T_ext.aab, optimize=True)
    dT.ab -= np.einsum("nmfj,afbinm->abij", H0.ab.oovo + H.ab.oovo, T_ext.aab, optimize=True)
    dT.ab -= 0.5 * np.einsum("mnjf,afbinm->abij", H0.bb.ooov + H.bb.ooov, T_ext.abb, optimize=True)
    dT.ab -= np.einsum("mnif,afbmnj->abij", H0.ab.ooov + H.ab.ooov, T_ext.abb, optimize=True)
    dT.ab += 0.5 * np.einsum("anef,efbinj->abij", H0.aa.vovv + H.aa.vovv, T_ext.aab, optimize=True)
    dT.ab += np.einsum("anef,efbinj->abij", H0.ab.vovv + H.ab.vovv, T_ext.abb, optimize=True)
    dT.ab += np.einsum("nbfe,afeinj->abij", H0.ab.ovvv + H.ab.ovvv, T_ext.aab, optimize=True)
    dT.ab += 0.5 * np.einsum("bnef,afeinj->abij", H0.bb.vovv + H.bb.vovv, T_ext.abb, optimize=True)
    dT.ab += np.einsum("me,aebimj->abij", H.a.ov, T_ext.aab, optimize=True)
    dT.ab += np.einsum("me,aebimj->abij", H.b.ov, T_ext.abb, optimize=True)
    T.ab, dT.ab = cc_loops2.update_t2b(
        T.ab,
        dT.ab + H0.ab.vvoo + VT_ext.ab.vvoo,
        H0.a.oo,
        H0.a.vv,
        H0.b.oo,
        H0.b.vv,
        shift
    )
    return T, dT

def update_t2c(T, dT, H, H0, VT_ext, shift, T_ext):
    """
    Update t2c amplitudes by calculating the projection <i~j~a~b~|(H_N e^(T1+T2+T3(ext)+T4(ext)))_C|0>.
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
    # T3 parts
    dT.bb += 0.25 * np.einsum("me,eabmij->abij", H.a.ov, T_ext.abb, optimize=True)
    dT.bb += 0.25 * np.einsum("me,abeijm->abij", H.b.ov, T_ext.bbb, optimize=True)
    dT.bb += 0.25 * np.einsum("anef,ebfijn->abij", H0.bb.vovv + H.bb.vovv, T_ext.bbb, optimize=True)
    dT.bb += 0.5 * np.einsum("nafe,febnij->abij", H0.ab.ovvv + H.ab.ovvv, T_ext.abb, optimize=True)
    dT.bb -= 0.25 * np.einsum("mnif,abfmjn->abij", H0.bb.ooov + H.bb.ooov, T_ext.bbb, optimize=True)
    dT.bb -= 0.5 * np.einsum("nmfi,fabnmj->abij", H0.ab.oovo + H.ab.oovo, T_ext.abb, optimize=True)
    T.bb, dT.bb = cc_loops2.update_t2c(
        T.bb,
        dT.bb + 0.25 * (H0.bb.vvoo + VT_ext.bb.vvoo),
        H0.b.oo,
        H0.b.vv,
        shift
    )
    return T, dT