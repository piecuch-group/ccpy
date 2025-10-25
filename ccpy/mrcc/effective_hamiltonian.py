import numpy as np
from ccpy.utilities.linear_algebra import ccpy_einsum
from ccpy.mrcc.normal_order import shift_normal_order
from ccpy.energy.cc_energy import get_cc_energy_unsorted


def compute_Heff_mkmrccsd(H, T, model_space, system):

    d = len(model_space)
    Heff = np.zeros((d, d))

    occ_prev_a = []
    occ_prev_b = []

    for q in range(d):

        occ_a, unocc_a, occ_b, unocc_b = model_space[q].get_orbital_partitioning(system)

        H, e_ref = shift_normal_order(H, occ_a, occ_b, occ_prev_a, occ_prev_b)

        occ_prev_a = occ_a.copy()
        occ_prev_b = occ_b.copy()

        for p in range(d):

            if p == q:
                Heff[p, p] = get_cc_energy_unsorted(T[p], H, occ_a, unocc_a, occ_b, unocc_b) + e_ref
            else:
                exc = model_space[p].get_excitation(model_space[q], system)

                if len(exc.spincase) == 1: # single excitation, need moments M_a^i(q) = < ia(q) | (H_q e^T(q))_C | 0(q) >
                    if exc.spincase == 'a':
                        i, = exc.from_alpha
                        a, = exc.to_alpha
                        Heff[p, q] = H.a[a, i]
                    if exc.spincase == 'b':
                        i, = exc.from_beta[1]
                        a, = exc.to_beta[1]
                        Heff[p, q] = H.b[a, i]

                if len(exc.spincase) == 2: # double excitation, need moments M_ab^ij(q) = < ijab(q) | (H e^T)_C | 0(q) >
                    if exc.spincase == 'aa':
                        i, j = exc.from_alpha
                        a, b = exc.to_alpha
                        Heff[p, q] = (H.aa[a, b, i, j]
                                    + T.a[a, i] * H.a[b, j]
                                    - T.a[a, j] * H.a[b, i]
                                    - T.a[b, i] * H.a[a, j]
                                    + T.a[b, j] * H.a[a, i])
                    if exc.spincase == 'ab':
                        i, = exc.from_alpha
                        j, = exc.from_beta
                        a, = exc.to_alpha
                        b, = exc.to_beta
                        Heff[p, q] = (H.ab[a, b, i, j]
                                    + T.a[a, i] * H.b[b, j]
                                    + T.b[b, j] * H.a[a, i])
                    if exc.spincase == 'bb':
                        i, j = exc.from_beta
                        a, b = exc.to_beta
                        Heff[p, q] = (H.bb[a, b, i, j]
                                    + T.b[a, i] * H.b[b, j]
                                    - T.b[a, j] * H.b[b, i]
                                    - T.b[b, i] * H.b[a, j]
                                    + T.b[b, j] * H.b[a, i])

                if len(exc.spincase == 3): # triple excitation, need moments M_abc^ijk(q) = < ijkabc(q) | (H e^T)_C | 0(q) >
                    if exc.spincase == 'aaa':
                        i, j, k = exc.from_alpha
                        a, b, c = exc.to_beta
                        Heff[p, q] = (H)
    return Heff


def calc_ccsd_moment_a(a, i, T, H, occ_a, unocc_a, occ_b, unocc_b):

    chi1A_vv = H.a[a, unocc_a]
    chi1A_vv += ccpy_einsum("nef,fn->e", H.aa[a, occ_a, unocc_a, unocc_a], T.a)
    chi1A_vv += ccpy_einsum("nef,fn->e", H.ab[a, occ_b, unocc_a, unocc_b], T.b)

    chi1A_oo = H.a[occ_a, i]
    chi1A_oo += ccpy_einsum("mnf,fn->m", H.aa[occ_a, occ_a, i, unocc_a], T.a)
    chi1A_oo += ccpy_einsum("mnf,fn->m", H.ab[occ_a, occ_b, i, unocc_b], T.b)

    h1A_ov = H.a[occ_a, unocc_a].copy()
    h1A_ov += ccpy_einsum("mnef,fn->me", H.aa[occ_a, occ_a, unocc_a, unocc_a], T.a)
    h1A_ov += ccpy_einsum("mnef,fn->me", H.ab[occ_a, occ_b, unocc_a, unocc_b], T.b)

    h1B_ov = H.b[occ_b, unocc_b].copy()
    h1B_ov += ccpy_einsum("nmfe,fn->me", H.ab[occ_a, occ_b, occ_a, unocc_b], T.a)
    h1B_ov += ccpy_einsum("mnef,fn->me", H.bb[occ_b, occ_b, unocc_b, unocc_b], T.b)

    h1A_oo = chi1A_oo + ccpy_einsum("me,ei->m", h1A_ov[occ_a, i], T.a)

    h2A_ooov = H.aa[occ_a, occ_a, i, unocc_a] + ccpy_einsum("mnfe,f->mne", H.aa[occ_a, occ_a, unocc_a, unocc_a], T.a[:, i])
    h2B_ooov = H.ab[occ_a, occ_b, i, unocc_b] + ccpy_einsum("mnfe,f->mne", H.ab[occ_a, occ_b, unocc_a, unocc_b], T.a[:, i])
    h2A_vovv = H.aa[a, occ_a, unocc_a, unocc_a] - ccpy_einsum("mnfe,n->mef", H.aa[occ_a, occ_a, unocc_a, unocc_a], T.a[a, :])
    h2B_vovv = H.ab[a, occ_b, unocc_a, unocc_b] - ccpy_einsum("nmef,n->mef", H.ab[occ_a, occ_b, unocc_a, unocc_b], T.a[a, :])

    val = H.a[a, i]
    val -= ccpy_einsum("m,m->", h1A_oo, T.a[a, :])
    val += ccpy_einsum("e,e->", chi1A_vv, T.a[:, i])
    val += ccpy_einsum("nf,fn->", H.aa[a, occ_a, i, unocc_a], T.a)
    val += ccpy_einsum("nf,fn->", H.ab[a, occ_b, i, unocc_b], T.b)
    val += ccpy_einsum("me,em->", h1A_ov, T.aa[a, :, i, :])
    val += ccpy_einsum("me,em->", h1B_ov, T.ab[a, :, i, :])
    val -= 0.5 * ccpy_einsum("mnf,fmn->", h2A_ooov, T.aa[a, :, :, :])
    val -= ccpy_einsum("mnf,fmn->", h2B_ooov, T.ab[a, :, :, :])
    val += 0.5 * ccpy_einsum("nef,efn->", h2A_vovv, T.aa[:, :, i, :])
    val += ccpy_einsum("nef,efn->", h2B_vovv, T.ab[:, :, i, :])

    return val