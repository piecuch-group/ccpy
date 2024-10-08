import numpy as np

# NOTE:
# In order to get the expected behavior for submatrix slicing, use H.aa[np.ix_(arr1, arr2, arr3, arr4)]
# if you are using singleton dimensions and arbitrary slices, those singletons must be converted to arrays

from ccpy.mrcc.normal_order import shift_normal_order
from ccpy.hbar.hbar_ccs import get_ccs_intermediates_unsorted
from ccpy.lib.core import mrcc_loops

def update(T, dT, H, model_space, Heff, C, shift, flag_RHF, system):

    occ_prev_a = []
    occ_prev_b = []

    for p, det_p in enumerate(model_space):

        H, _ = shift_normal_order(H, occ_a, occ_b, occ_prev_a, occ_prev_b)
        occ_prev_a = occ_a.copy()
        occ_prev_b = occ_b.copy()

        # update T1
        dT[p] = calc_direct_t1a(T[p], dT[p], H, det_p)
        dT[p] = calc_coupling_t1a(T, dT, Heff, C, model_space, p)
        T[p], dT[p] = update_t1a(T[p], dT[p], H, det_p, shift)
        if flag_RHF:
            T[p].b = T[p].a.copy()
            dT[p].b = dT[p].a.copy()
        else:
            dT[p] = calc_direct_t1b(T[p], dT[p], H, det_p)
            dT[p] = calc_coupling_t1b(T, dT, Heff, C, model_space, p)
            T[p], dT[p] = update_t1b(T[p], dT[p], H, det_p, shift)

        # CCS intermediates
        hbar = get_ccs_intermediates_unsorted(T[p], H, det_p)

        # update T2
        dT[p] = calc_direct_t2a(T[p], dT[p], hbar, H, det_p)
        dT[p] = calc_coupling_t2a(T, dT, Heff, C, model_space, p)
        T[p], dT[p] = update_t2a(T[p], dT[p], H, det_p, shift)

        dT[p] = calc_direct_t2b(T[p], dT[p], H, det_p)
        dT[p] = calc_coupling_t2b(T, dT, Heff, C, model_space, p)
        T[p], dT[p] = update_t2b(T[p], dT[p], H, det_p, shift)
        if flag_RHF:
            T[p].bb = T[p].aa.copy()
            dT[p].bb = dT[p].aa.copy()
        else:
            dT[p] = calc_direct_t2c(T[p], dT[p], H, det_p)
            dT[p] = calc_coupling_t2c(T, dT, Heff, C, model_space, p)
            T[p], dT[p] = update_t2c(T[p], dT[p], H, det_p, shift)


    return T, dT

def calc_direct_t1a(T, dT, H, occ_a, unocc_a, occ_b, unocc_b):

    chi1A_vv = H.a[unocc_a, unocc_a].copy()
    chi1A_vv += np.einsum("anef,fn->ae", H.aa[unocc_a, occ_a, unocc_a, unocc_a], T.a, optimize=True)
    chi1A_vv += np.einsum("anef,fn->ae", H.ab[unocc_a, occ_b, unocc_a, unocc_b], T.b, optimize=True)

    chi1A_oo = H.a[occ_a, occ_a].copy()
    chi1A_oo += np.einsum("mnif,fn->mi", H.aa[occ_a, occ_a, occ_a, unocc_a], T.a, optimize=True)
    chi1A_oo += np.einsum("mnif,fn->mi", H.ab[occ_a, occ_b, occ_a, unocc_b], T.b, optimize=True)

    h1A_ov = H.a[occ_a, unocc_a].copy()
    h1A_ov += np.einsum("mnef,fn->me", H.aa[occ_a, occ_a, unocc_a, unocc_a], T.a, optimize=True)
    h1A_ov += np.einsum("mnef,fn->me", H.ab[occ_a, occ_b, unocc_a, unocc_b], T.b, optimize=True)

    h1B_ov = H.b[occ_b, unocc_b].copy()
    h1B_ov += np.einsum("nmfe,fn->me", H.ab[occ_a, occ_b, occ_a, unocc_b], T.a, optimize=True)
    h1B_ov += np.einsum("mnef,fn->me", H.bb[occ_b, occ_b, unocc_b, unocc_b], T.b, optimize=True)

    h1A_oo = chi1A_oo.copy()
    h1A_oo += np.einsum("me,ei->mi", h1A_ov[occ_a, unocc_a], T.a, optimize=True)

    h2A_ooov = H.aa[occ_a, occ_a, occ_a, unocc_a] + np.einsum("mnfe,fi->mnie", H.aa[occ_a, occ_a, unocc_a, unocc_a], T.a, optimize=True)
    h2B_ooov = H.ab[occ_a, occ_b, occ_a, unocc_b] + np.einsum("mnfe,fi->mnie", H.ab[occ_a, occ_b, unocc_a, unocc_b], T.a, optimize=True)
    h2A_vovv = H.aa[unocc_a, occ_a, unocc_a, unocc_a] - np.einsum("mnfe,an->amef", H.aa[occ_a, occ_a, unocc_a, unocc_a], T.a, optimize=True)
    h2B_vovv = H.ab[unocc_a, occ_b, unocc_a, unocc_b] - np.einsum("nmef,an->amef", H.ab[occ_a, occ_b, unocc_a, unocc_b], T.a, optimize=True)

    dT.a = H.a[unocc_a, occ_a].copy()
    dT.a -= np.einsum("mi,am->ai", h1A_oo, T.a, optimize=True)
    dT.a += np.einsum("ae,ei->ai", chi1A_vv, T.a, optimize=True)
    dT.a += np.einsum("anif,fn->ai", H.aa[unocc_a, occ_a, occ_a, unocc_a], T.a, optimize=True)
    dT.a += np.einsum("anif,fn->ai", H.ab[unocc_a, occ_b, occ_a, unocc_b], T.b, optimize=True)
    dT.a += np.einsum("me,aeim->ai", h1A_ov, T.aa, optimize=True)
    dT.a += np.einsum("me,aeim->ai", h1B_ov, T.ab, optimize=True)
    dT.a -= 0.5 * np.einsum("mnif,afmn->ai", h2A_ooov, T.aa, optimize=True)
    dT.a -= np.einsum("mnif,afmn->ai", h2B_ooov, T.ab, optimize=True)
    dT.a += 0.5 * np.einsum("anef,efin->ai", h2A_vovv, T.aa, optimize=True)
    dT.a += np.einsum("anef,efin->ai", h2B_vovv, T.ab, optimize=True)

    return dT

def calc_coupling_t1a(T, dT, Heff, C, model_space, p):

    for q in range(len(model_space)):

        if p == q: continue

        dT[p].a = mrcc_loops.mrcc_loops.compute_coupling_t1a_pq(
            dT[p].a, T[p].a, T[q].a,
            model_space[p].spinorbital_occupation, model_space[q].spinorbital_occupation,
            Heff[p, q], C[q]/C[p]
        )

    return dT


def update_t1a(T, dT, H, occ_a, unocc_a, shift):
    T.a, dT.a = mrcc_loops.mrcc_loops.update_t1a(
        T.a, dT.a, H.a[occ_a, occ_a], H.a[unocc_a, unocc_a], shift
    )
    return T, dT

# @profile
def calc_direct_t1b(T, dT, H, occ_a, unocc_a, occ_b, unocc_b):

    # Intermediates
    chi1B_vv = H.b[unocc_b, unocc_b].copy()
    chi1B_vv += np.einsum("anef,fn->ae", H.bb[unocc_b, occ_b, unocc_b, unocc_b], T.b, optimize=True)
    chi1B_vv += np.einsum("nafe,fn->ae", H.ab[occ_a, unocc_b, unocc_a, unocc_b], T.a, optimize=True)

    chi1B_oo = H.b[occ_b, occ_b].copy()
    chi1B_oo += np.einsum("mnif,fn->mi", H.bb[occ_b, occ_b, occ_b, unocc_b], T.b, optimize=True)
    chi1B_oo += np.einsum("nmfi,fn->mi", H.ab[occ_a, occ_b, unocc_a, occ_b], T.a, optimize=True)

    h1A_ov = H.a.ov.copy()
    h1A_ov += np.einsum("mnef,fn->me", H.aa[occ_a, occ_a, unocc_a, unocc_a], T.a, optimize=True)
    h1A_ov += np.einsum("mnef,fn->me", H.ab[occ_a, occ_b, unocc_a, unocc_b], T.b, optimize=True)

    h1B_ov = H.b.ov.copy()
    h1B_ov += np.einsum("nmfe,fn->me", H.ab[occ_a, occ_b, unocc_a, unocc_b], T.a, optimize=True)
    h1B_ov += np.einsum("mnef,fn->me", H.bb[occ_b, occ_b, unocc_b, unocc_b], T.b, optimize=True)

    h1B_oo = chi1B_oo + np.einsum("me,ei->mi", h1B_ov, T.b, optimize=True)

    h2C_ooov = H.bb[occ_b, occ_b, occ_b, unocc_b] + np.einsum("mnfe,fi->mnie", H.bb[occ_b, occ_b, unocc_b, unocc_b], T.b, optimize=True)
    h2B_oovo = H.ab[occ_a, occ_b, unocc_a, occ_b] + np.einsum("nmef,fi->nmei", H.ab[occ_a, occ_b, unocc_a, unocc_b], T.b, optimize=True)
    h2C_vovv = H.bb[unocc_b, occ_b, unocc_b, unocc_b] - np.einsum("mnfe,an->amef", H.bb[occ_b, occ_b, unocc_b, unocc_b], T.b, optimize=True)
    h2B_ovvv = H.ab[occ_a, unocc_b, unocc_a, unocc_b] - np.einsum("mnfe,an->mafe", H.ab[occ_a, occ_b, unocc_a, unocc_b], T.b, optimize=True)

    dT.b = H.b[unocc_b, occ_b].copy()
    dT.b -= np.einsum("mi,am->ai", h1B_oo, T.b, optimize=True)
    dT.b += np.einsum("ae,ei->ai", chi1B_vv, T.b, optimize=True)
    dT.b += np.einsum("anif,fn->ai", H.bb[unocc_b, occ_b, occ_b, unocc_b], T.b, optimize=True)
    dT.b += np.einsum("nafi,fn->ai", H.ab[occ_a, unocc_b, unocc_a, occ_b], T.a, optimize=True)
    dT.b += np.einsum("me,eami->ai", h1A_ov, T.ab, optimize=True)
    dT.b += np.einsum("me,aeim->ai", h1B_ov, T.bb, optimize=True)
    dT.b -= 0.5 * np.einsum("mnif,afmn->ai", h2C_ooov, T.bb, optimize=True)
    dT.b -= np.einsum("nmfi,fanm->ai", h2B_oovo, T.ab, optimize=True)
    dT.b += 0.5 * np.einsum("anef,efin->ai", h2C_vovv, T.bb, optimize=True)
    dT.b += np.einsum("nafe,feni->ai", h2B_ovvv, T.ab, optimize=True)

    return dT


def calc_coupling_t1b(T, dT, Heff, C, model_space, p):

    for q in range(len(model_space)):

        if p == q: continue

        dT[p].b = mrcc_loops.mrcc_loops.compute_coupling_t1b_pq(
            dT[p].b, T[p].b, T[q].b,
            model_space[p].spinorbital_occupation, model_space[q].spinorbital_occupation,
            Heff[p, q], C[q]/C[p]
        )

    return dT

def update_t1b(T, dT, H, occ_b, unocc_b, shift):
    T.b, dT.b = mrcc_loops.mrcc_loops.update_t1b(
        T.b, dT.b, H.b[occ_b, occ_b], H.b[unocc_b, unocc_b], shift
    )
    return T, dT


# @profile
def calc_direct_t2a(T, dT, H, H0, occ_a, unocc_a, occ_b, unocc_b):
    """
    Update t2a amplitudes by calculating the projection <ijab|(H_N e^(T1+T2))_C|0>.
    """
    # intermediates
    I1A_oo = (
        H.a[occ_a, occ_a]
        + 0.5 * np.einsum("mnef,efin->mi", H.aa[occ_a, occ_a, unocc_a, unocc_a], T.aa, optimize=True)
        + np.einsum("mnef,efin->mi", H.ab[occ_a, occ_b, unocc_a, unocc_b], T.ab, optimize=True)
    )

    I1A_vv = (
        H.a[unocc_a, unocc_a]
        - 0.5 * np.einsum("mnef,afmn->ae", H.aa[occ_a, occ_a, unocc_a, unocc_a], T.aa, optimize=True)
        - np.einsum("mnef,afmn->ae", H.ab[occ_a, occ_b, unocc_a, unocc_b], T.ab, optimize=True)
    )

    I2A_voov = (
        H.aa[unocc_a, occ_a, occ_a, unocc_a]
        + 0.5 * np.einsum("mnef,afin->amie", H.aa[occ_a, occ_a, unocc_a, unocc_a], T.aa, optimize=True)
        + np.einsum("mnef,afin->amie", H.ab[occ_a, occ_b, unocc_a, unocc_b], T.ab, optimize=True)
    )

    I2A_oooo = H.aa[occ_a, occ_a, occ_a, occ_a] + 0.5 * np.einsum(
        "mnef,efij->mnij", H.aa[occ_a, occ_a, unocc_a, unocc_a], T.aa, optimize=True
    )

    I2B_voov = H.ab[unocc_a, occ_b, occ_a, unocc_b] + 0.5 * np.einsum(
        "mnef,afin->amie", H.bb[occ_b, occ_b, unocc_b, unocc_b], T.ab, optimize=True
    )

    I2A_vooo = H.aa[unocc_a, occ_a, occ_a, occ_a] + (
            0.5*np.einsum('anef,efij->anij', H0.aa[unocc_a, occ_a, unocc_a, unocc_a] + 0.5 * H.aa[unocc_a, occ_a, unocc_a, unocc_a], T.aa, optimize=True)
    )

    tau = 0.5 * T.aa + np.einsum('ai,bj->abij', T.a, T.a, optimize=True)

    dT.aa = 0.25 * H0.aa[unocc_a, unocc_a, occ_a, occ_a]
    dT.aa -= 0.5 * np.einsum("amij,bm->abij", I2A_vooo, T.a, optimize=True)
    dT.aa += 0.5 * np.einsum("abie,ej->abij", H.aa[unocc_a, unocc_a, occ_a, unocc_a], T.a, optimize=True)
    dT.aa += 0.5 * np.einsum("ae,ebij->abij", I1A_vv, T.aa, optimize=True)
    dT.aa -= 0.5 * np.einsum("mi,abmj->abij", I1A_oo, T.aa, optimize=True)
    dT.aa += np.einsum("amie,ebmj->abij", I2A_voov, T.aa, optimize=True)
    dT.aa += np.einsum("amie,bejm->abij", I2B_voov, T.ab, optimize=True)
    dT.aa += 0.25 * np.einsum("abef,efij->abij", H.aa[unocc_a, unocc_a, unocc_a, unocc_a], tau, optimize=True)
    dT.aa += 0.125 * np.einsum("mnij,abmn->abij", I2A_oooo, T.aa, optimize=True)

    return dT


def calc_coupling_t2a(T, dT, Heff, C, model_space, p):

    for q in range(len(model_space)):

        if p == q: continue

        dT[p].aa = mrcc_loops.mrcc_loops.compute_coupling_t2a_pq(
            dT[p].aa, T[p].a, T[p].aa, T[q].a, T[q].aa,
            model_space[p].spinorbital_occupation, model_space[q].spinorbital_occupation,
            Heff[p, q], C[q]/C[p]
        )

    return dT

def update_t2a(T, dT, H, occ_a, unocc_a, shift):
    T.aa, dT.aa = mrcc_loops.mrcc_loops.update_t2a(
        T.aa, dT.aa, H.a[occ_a, occ_a], H.a[unocc_a, unocc_a], shift
    )
    return T, dT

# @profile
def calc_direct_t2b(T, dT, H, H0, occ_a, unocc_a, occ_b, unocc_b):
    """
    Update t2b amplitudes by calculating the projection <ij~ab~|(H_N e^(T1+T2))_C|0>.
    """
    # intermediates
    I1A_vv = (
        H.a[unocc_a, unocc_a]
        - 0.5 * np.einsum("mnef,afmn->ae", H.aa[occ_a, occ_a, unocc_a, unocc_a], T.aa, optimize=True)
        - np.einsum("mnef,afmn->ae", H.ab[occ_a, occ_b, unocc_a, unocc_b], T.ab, optimize=True)
    )

    I1B_vv = (
        H.b[unocc_b, unocc_b]
        - np.einsum("nmfe,fbnm->be", H.ab[occ_a, occ_b, unocc_a, unocc_b], T.ab, optimize=True)
        - 0.5 * np.einsum("mnef,fbnm->be", H.bb[occ_b, occ_b, unocc_b, unocc_b], T.bb, optimize=True)
    )

    I1A_oo = (
        H.a[occ_a, occ_a]
        + 0.5 * np.einsum("mnef,efin->mi", H.aa[occ_a, occ_a, unocc_a, unocc_a], T.aa, optimize=True)
        + np.einsum("mnef,efin->mi", H.ab[occ_a, occ_b, unocc_a, unocc_b], T.ab, optimize=True)
    )

    I1B_oo = (
        H.b[occ_b, occ_b]
        + np.einsum("nmfe,fenj->mj", H.ab[occ_a, occ_b, unocc_a, unocc_b], T.ab, optimize=True)
        + 0.5 * np.einsum("mnef,efjn->mj", H.bb[occ_b, occ_b, unocc_b, unocc_b], T.bb, optimize=True)
    )

    I2A_voov = (
        H.aa[unocc_a, occ_a, occ_a, unocc_a]
        + np.einsum("mnef,aeim->anif", H.aa[occ_a, occ_a, unocc_a, unocc_a], T.aa, optimize=True)
        + np.einsum("nmfe,aeim->anif", H.ab[occ_a, occ_b, unocc_a, unocc_b], T.ab, optimize=True)
    )

    I2B_voov = (
        H.ab[unocc_a, occ_b, occ_a, unocc_b]
        + np.einsum("mnef,aeim->anif", H.ab[occ_a, occ_b, unocc_a, unocc_b], T.aa, optimize=True)
        + np.einsum("mnef,aeim->anif", H.bb[occ_b, occ_b, unocc_b, unocc_b], T.ab, optimize=True)
    )

    I2B_oooo = H.ab[occ_a, occ_b, occ_a, occ_b] + np.einsum("mnef,efij->mnij", H.ab[occ_a, occ_b, unocc_a, unocc_b], T.ab, optimize=True)

    I2B_vovo = H.ab[unocc_a, occ_b, unocc_a, occ_b] - np.einsum("mnef,afmj->anej", H.ab[occ_a, occ_b, unocc_a, unocc_b], T.ab, optimize=True)

    I2B_ovoo = H.ab[occ_a, unocc_b, occ_a, occ_b] + np.einsum("maef,efij->maij", H0.ab[occ_a, unocc_b, unocc_a, unocc_b] + 0.5 * H.ab[occ_a, unocc_b, unocc_a, unocc_b], T.ab, optimize=True)
    I2B_vooo = H.ab[unocc_a, occ_b, occ_a, occ_b] + np.einsum("amef,efij->amij", H0.ab[unocc_a, occ_b, unocc_a, unocc_b] + 0.5 * H.ab[unocc_a, occ_b, unocc_a, unocc_b], T.ab, optimize=True)

    tau = T.ab + np.einsum('ai,bj->abij', T.a, T.b, optimize=True)

    dT.ab = H0.ab[unocc_a, unocc_b, occ_a, occ_b].copy()
    dT.ab -= np.einsum("mbij,am->abij", I2B_ovoo, T.a, optimize=True)
    dT.ab -= np.einsum("amij,bm->abij", I2B_vooo, T.b, optimize=True)
    dT.ab += np.einsum("abej,ei->abij", H.ab[unocc_a, unocc_b, unocc_a, occ_b], T.a, optimize=True)
    dT.ab += np.einsum("abie,ej->abij", H.ab[unocc_a, unocc_b, occ_a, unocc_b], T.b, optimize=True)
    dT.ab += np.einsum("ae,ebij->abij", I1A_vv, T.ab, optimize=True)
    dT.ab += np.einsum("be,aeij->abij", I1B_vv, T.ab, optimize=True)
    dT.ab -= np.einsum("mi,abmj->abij", I1A_oo, T.ab, optimize=True)
    dT.ab -= np.einsum("mj,abim->abij", I1B_oo, T.ab, optimize=True)
    dT.ab += np.einsum("amie,ebmj->abij", I2A_voov, T.ab, optimize=True)
    dT.ab += np.einsum("amie,ebmj->abij", I2B_voov, T.bb, optimize=True)
    dT.ab += np.einsum("mbej,aeim->abij", H.ab[occ_a, unocc_b, unocc_a, occ_b], T.aa, optimize=True)
    dT.ab += np.einsum("bmje,aeim->abij", H.bb[unocc_b, occ_b, occ_b, unocc_b], T.ab, optimize=True)
    dT.ab -= np.einsum("mbie,aemj->abij", H.ab[occ_a, unocc_b, occ_a, unocc_b], T.ab, optimize=True)
    dT.ab -= np.einsum("amej,ebim->abij", I2B_vovo, T.ab, optimize=True)
    dT.ab += np.einsum("mnij,abmn->abij", I2B_oooo, T.ab, optimize=True)
    dT.ab += np.einsum("abef,efij->abij", H.ab[unocc_a, unocc_b, unocc_a, unocc_b], tau, optimize=True)

    return dT


def calc_coupling_t2b(T, dT, Heff, C, model_space, p):

    for q in range(len(model_space)):

        if p == q: continue

        dT[p].ab = mrcc_loops.mrcc_loops.compute_coupling_t2b_pq(
            dT[p].ab, T[p].a, T[p].b, T[p].ab, T[q].a, T[q].b, T[q].ab,
            model_space[p].spinorbital_occupation, model_space[q].spinorbital_occupation,
            Heff[p, q], C[q]/C[p]
        )

    return dT

def update_t2b(T, dT, H, occ_a, unocc_a, occ_b, unocc_b, shift):
    T.ab, dT.ab = mrcc_loops.mrcc_loops.update_t2b(
        T.ab, dT.ab, H.a[occ_a, occ_a], H.a[unocc_a, unocc_a], H.b[occ_b, occ_b], H.b[unocc_b, unocc_b], shift
    )
    return T, dT

# @profile
def calc_direct_t2c(T, dT, H, H0, occ_a, unocc_a, occ_b, unocc_b):
    """
    Update t2c amplitudes by calculating the projection <i~j~a~b~|(H_N e^(T1+T2))_C|0>.
    """
    # intermediates
    I1B_oo = (
        H.b[occ_b, occ_b]
        + 0.5 * np.einsum("mnef,efin->mi", H.bb[occ_b, occ_b, unocc_b, unocc_b], T.bb, optimize=True)
        + np.einsum("nmfe,feni->mi", H.ab[occ_a, occ_b, unocc_a, unocc_b], T.ab, optimize=True)
    )

    I1B_vv = (
        H.b[unocc_b, unocc_b]
        - 0.5 * np.einsum("mnef,afmn->ae", H.bb[occ_b, occ_b, unocc_b, unocc_b], T.bb, optimize=True)
        - np.einsum("nmfe,fanm->ae", H.ab[occ_a, occ_b, unocc_a, unocc_b], T.ab, optimize=True)
    )

    I2C_oooo = H.bb[occ_b, occ_b, occ_b, occ_b] + 0.5 * np.einsum(
        "mnef,efij->mnij", H.bb[occ_b, occ_b, unocc_b, unocc_b], T.bb, optimize=True
    )

    I2B_ovvo = (
        H.ab[occ_a, unocc_b, unocc_a, occ_b]
        + np.einsum("mnef,afin->maei", H.ab[occ_a, occ_b, unocc_a, unocc_b], T.bb, optimize=True)
        + 0.5 * np.einsum("mnef,fani->maei", H.aa[occ_a, occ_a, unocc_a, unocc_a], T.ab, optimize=True)
    )

    I2C_voov = H.bb[unocc_b, occ_b, occ_b, unocc_b] + 0.5 * np.einsum(
        "mnef,afin->amie", H.bb[occ_b, occ_b, unocc_b, unocc_b], T.bb, optimize=True
    )

    I2C_vooo = H.bb[unocc_b, occ_b, occ_b, occ_b] + 0.5*np.einsum('anef,efij->anij', H0.bb[unocc_b, occ_b, unocc_b, unocc_b] + 0.5 * H.bb[unocc_b, occ_b, unocc_b, unocc_b], T.bb, optimize=True)

    tau = 0.5 * T.bb + np.einsum('ai,bj->abij', T.b, T.b, optimize=True)

    dT.bb = 0.25 * H0.bb[unocc_b, unocc_b, occ_b, occ_b]
    dT.bb -= 0.5 * np.einsum("amij,bm->abij", I2C_vooo, T.b, optimize=True)
    dT.bb += 0.5 * np.einsum("abie,ej->abij", H.bb[unocc_b, unocc_b, occ_b, unocc_b], T.b, optimize=True)
    dT.bb += 0.5 * np.einsum("ae,ebij->abij", I1B_vv, T.bb, optimize=True)
    dT.bb -= 0.5 * np.einsum("mi,abmj->abij", I1B_oo, T.bb, optimize=True)
    dT.bb += np.einsum("amie,ebmj->abij", I2C_voov, T.bb, optimize=True)
    dT.bb += np.einsum("maei,ebmj->abij", I2B_ovvo, T.ab, optimize=True)
    dT.bb += 0.25 * np.einsum("abef,efij->abij", H.bb[unocc_b, unocc_b, unocc_b, unocc_b], tau, optimize=True)
    dT.bb += 0.125 * np.einsum("mnij,abmn->abij", I2C_oooo, T.bb, optimize=True)

    return dT


def calc_coupling_t2c(T, dT, Heff, C, model_space, p):

    for q in range(len(model_space)):

        if p == q: continue

        dT[p].bb = mrcc_loops.mrcc_loops.compute_coupling_t2c_pq(
            dT[p].bb, T[p].b, T[p].bb, T[q].b, T[q].bb,
            model_space[p].spinorbital_occupation, model_space[q].spinorbital_occupation,
            Heff[p, q], C[q]/C[p]
        )

    return dT

def update_t2c(T, dT, H, occ_b, unocc_b, shift):
    T.bb, dT.bb = mrcc_loops.mrcc_loops.update_t2c(
        T.bb, dT.bb, H.b[occ_b, occ_b], H.b[unocc_b, unocc_b], shift
    )
    return T, dT
