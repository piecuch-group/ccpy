"""Module with functions that perform the CC with singles, doubles,
and triples (CCSDT) calculation for a molecular system."""

# Further development steps:
# (1) Refactor all terms with T3 (re-implement in Numba)
#
# (2) Change structure of P space so that p[a, b, c, i, j, k] = idx, where T[idx] = t3[a, b, c, i, j, k] is
#     a compact, linear T vector used in the iterative steps.
#
# Remove dependence on full P space storage using one of the following (or a combination thereof):
#
# (3) Implement pre-screening to reduce loop sizes and checking of amplitudes for each diagram, thus removing dependence
#     on P space matrix. For instance, one could use prescreening on low-storage diagrams and some other method on the
#     high-storage ones.
#
# (3) Utilize a disk-based I/O so that for each i,j,k (and its permutations), the block p_abc[:, :, :] is loaded into
#     memory and used for all the loops.
#
# (3) Use full p_space storage for p[1:64, 1:64, 1:64, :, :, :], e.g., all occupied and the first 64 (or something)
#     virtual orbitals, as these will be the amplitudes accessed most often, and use a hashing storage for all
#     other amplitudes (look at Garniron thesis).
#
# (4) Alternative to (3), store p_space array on disk and load in p_ijk(a,b,c) and related permutations within the
#     loop over triples in p_space and perform usual checking with p_ijk which is nu**3 storage
#
# (5) Alternative to (3) and (4), do a naive checking of idx in p_space_list, where p_space_list only stores the ijkabc
#     for triples in P space. Can use good search algorithm that go as log(N) or even O(1)
#
# (6) Alternative to (5), use MurmurHash3 to hash each tuple in P space so that checking


import numpy as np

from ccpy.hbar.hbar_ccs import get_ccs_intermediates_opt
from ccpy.hbar.hbar_ccsd import get_ccsd_intermediates
from ccpy.utilities.updates import ccp_opt_loops_v3

def update(T, dT, H, shift, flag_RHF, system, pspace):

    # update T1
    T, dT = update_t1a(T, dT, H, shift, pspace[0]['aaa'], pspace[0]['aab'], pspace[0]['abb'])
    if flag_RHF:
        T.b = T.a.copy()
        dT.b = dT.a.copy()
    else:
        T, dT = update_t1b(T, dT, H, shift, pspace[0]['aab'], pspace[0]['abb'], pspace[0]['bbb'])

    # CCS intermediates
    hbar = get_ccs_intermediates_opt(T, H)

    # update T2
    T, dT = update_t2a(T, dT, hbar, H, shift, pspace[0]['aaa'], pspace[0]['aab'])
    T, dT = update_t2b(T, dT, hbar, H, shift, pspace[0]['aab'], pspace[0]['abb'])
    if flag_RHF:
        T.bb = T.aa.copy()
        dT.bb = dT.aa.copy()
    else:
        T, dT = update_t2c(T, dT, hbar, H, shift, pspace[0]['abb'], pspace[0]['bbb'])

    # CCSD intermediates
    #[TODO]: Should accept CCS HBar as input and build only terms with T2 in it
    hbar = get_ccsd_intermediates(T, H)

    # update T3
    T, dT = update_t3a(T, dT, hbar, H, shift, pspace[0]['aaa'], pspace[0]['aab'])
    T, dT = update_t3b(T, dT, hbar, H, shift, pspace[0]['aaa'], pspace[0]['aab'], pspace[0]['abb'])
    if flag_RHF:
        T.abb = np.transpose(T.aab, (2, 1, 0, 5, 4, 3))
        dT.abb = np.transpose(dT.abb, (2, 1, 0, 5, 4, 3))
        T.bbb = T.aaa.copy()
        dT.bbb = dT.aaa.copy()
    else:
        T, dT = update_t3c(T, dT, hbar, H, shift, pspace[0]['aab'], pspace[0]['abb'], pspace[0]['bbb'])
        T, dT = update_t3d(T, dT, hbar, H, shift, pspace[0]['abb'], pspace[0]['bbb'])

    return T, dT

def update_t1a(T, dT, H, shift, pspace_aaa, pspace_aab, pspace_abb):
    """
    Update t1a amplitudes by calculating the projection <ia|(H_N e^(T1+T2+T3))_C|0>.
    """
    chi1A_vv = H.a.vv.copy()
    chi1A_vv += np.einsum("anef,fn->ae", H.aa.vovv, T.a, optimize=True)
    chi1A_vv += np.einsum("anef,fn->ae", H.ab.vovv, T.b, optimize=True)

    chi1A_oo = H.a.oo.copy()
    chi1A_oo += np.einsum("mnif,fn->mi", H.aa.ooov, T.a, optimize=True)
    chi1A_oo += np.einsum("mnif,fn->mi", H.ab.ooov, T.b, optimize=True)

    h1A_ov = H.a.ov.copy()
    h1A_ov += np.einsum("mnef,fn->me", H.aa.oovv, T.a, optimize=True)
    h1A_ov += np.einsum("mnef,fn->me", H.ab.oovv, T.b, optimize=True)

    h1B_ov = H.b.ov.copy()
    h1B_ov += np.einsum("nmfe,fn->me", H.ab.oovv, T.a, optimize=True)
    h1B_ov += np.einsum("mnef,fn->me", H.bb.oovv, T.b, optimize=True)

    h1A_oo = chi1A_oo.copy()
    h1A_oo += np.einsum("me,ei->mi", h1A_ov, T.a, optimize=True)

    h2A_ooov = H.aa.ooov + np.einsum("mnfe,fi->mnie", H.aa.oovv, T.a, optimize=True)
    h2B_ooov = H.ab.ooov + np.einsum("mnfe,fi->mnie", H.ab.oovv, T.a, optimize=True)
    h2A_vovv = H.aa.vovv - np.einsum("mnfe,an->amef", H.aa.oovv, T.a, optimize=True)
    h2B_vovv = H.ab.vovv - np.einsum("nmef,an->amef", H.ab.oovv, T.a, optimize=True)

    dT.a = -np.einsum("mi,am->ai", h1A_oo, T.a, optimize=True)
    dT.a += np.einsum("ae,ei->ai", chi1A_vv, T.a, optimize=True)
    dT.a += np.einsum("anif,fn->ai", H.aa.voov, T.a, optimize=True)
    dT.a += np.einsum("anif,fn->ai", H.ab.voov, T.b, optimize=True)
    dT.a += np.einsum("me,aeim->ai", h1A_ov, T.aa, optimize=True)
    dT.a += np.einsum("me,aeim->ai", h1B_ov, T.ab, optimize=True)
    dT.a -= 0.5 * np.einsum("mnif,afmn->ai", h2A_ooov, T.aa, optimize=True)
    dT.a -= np.einsum("mnif,afmn->ai", h2B_ooov, T.ab, optimize=True)
    dT.a += 0.5 * np.einsum("anef,efin->ai", h2A_vovv, T.aa, optimize=True)
    dT.a += np.einsum("anef,efin->ai", h2B_vovv, T.ab, optimize=True)

    T.a, dT.a = ccp_opt_loops_v3.ccp_opt_loops_v3.update_t1a_opt(
         T.a,
         dT.a + H.a.vo,
         T.aaa, T.aab, T.abb,
         pspace_aaa, pspace_aab, pspace_abb,
         H.aa.oovv, H.ab.oovv, H.bb.oovv,
         H.a.oo, H.a.vv,
         shift,
    )

    return T, dT

def update_t1b(T, dT, H, shift, pspace_aab, pspace_abb, pspace_bbb, n3aab, n3abb, n3bbb):
    """
    Update t1b amplitudes by calculating the projection <i~a~|(H_N e^(T1+T2+T3))_C|0>.
    """
    # Intermediates
    chi1B_vv = H.b.vv.copy()
    chi1B_vv += np.einsum("anef,fn->ae", H.bb.vovv, T.b, optimize=True)
    chi1B_vv += np.einsum("nafe,fn->ae", H.ab.ovvv, T.a, optimize=True)

    chi1B_oo = H.b.oo.copy()
    chi1B_oo += np.einsum("mnif,fn->mi", H.bb.ooov, T.b, optimize=True)
    chi1B_oo += np.einsum("nmfi,fn->mi", H.ab.oovo, T.a, optimize=True)

    h1A_ov = H.a.ov.copy()
    h1A_ov += np.einsum("mnef,fn->me", H.aa.oovv, T.a, optimize=True)
    h1A_ov += np.einsum("mnef,fn->me", H.ab.oovv, T.b, optimize=True)

    h1B_ov = H.b.ov.copy()
    h1B_ov += np.einsum("nmfe,fn->me", H.ab.oovv, T.a, optimize=True)
    h1B_ov += np.einsum("mnef,fn->me", H.bb.oovv, T.b, optimize=True)

    h1B_oo = chi1B_oo + np.einsum("me,ei->mi", h1B_ov, T.b, optimize=True)

    h2C_ooov = H.bb.ooov + np.einsum("mnfe,fi->mnie", H.bb.oovv, T.b, optimize=True)
    h2B_oovo = H.ab.oovo + np.einsum("nmef,fi->nmei", H.ab.oovv, T.b, optimize=True)
    h2C_vovv = H.bb.vovv - np.einsum("mnfe,an->amef", H.bb.oovv, T.b, optimize=True)
    h2B_ovvv = H.ab.ovvv - np.einsum("mnfe,an->mafe", H.ab.oovv, T.b, optimize=True)

    dT.b = -np.einsum("mi,am->ai", h1B_oo, T.b, optimize=True)
    dT.b += np.einsum("ae,ei->ai", chi1B_vv, T.b, optimize=True)
    dT.b += np.einsum("anif,fn->ai", H.bb.voov, T.b, optimize=True)
    dT.b += np.einsum("nafi,fn->ai", H.ab.ovvo, T.a, optimize=True)
    dT.b += np.einsum("me,eami->ai", h1A_ov, T.ab, optimize=True)
    dT.b += np.einsum("me,aeim->ai", h1B_ov, T.bb, optimize=True)
    dT.b -= 0.5 * np.einsum("mnif,afmn->ai", h2C_ooov, T.bb, optimize=True)
    dT.b -= np.einsum("nmfi,fanm->ai", h2B_oovo, T.ab, optimize=True)
    dT.b += 0.5 * np.einsum("anef,efin->ai", h2C_vovv, T.bb, optimize=True)
    dT.b += np.einsum("nafe,feni->ai", h2B_ovvv, T.ab, optimize=True)

    T.b, dT.b = ccp_opt_loops_v3.ccp_opt_loops_v3.update_t1b_opt(
         T.b,
         dT.b + H.b.vo,
         T.aab, T.abb, T.bbb,
         pspace_aab, pspace_abb, pspace_bbb,
         H.aa.oovv, H.ab.oovv, H.bb.oovv,
         H.b.oo, H.b.vv,
         shift,
    )

    return T, dT

# @profile
def update_t2a(T, dT, H, H0, shift, pspace_aaa, pspace_aab, n3aaa, n3aab):
    """
    Update t2a amplitudes by calculating the projection <ijab|(H_N e^(T1+T2))_C|0>.
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

    I2A_vooo = H.aa.vooo + 0.5*np.einsum('anef,efij->anij', H0.aa.vovv + 0.5 * H.aa.vovv, T.aa, optimize=True)

    tau = 0.5 * T.aa + np.einsum('ai,bj->abij', T.a, T.a, optimize=True)

    dT.aa = -0.5 * np.einsum("amij,bm->abij", I2A_vooo, T.a, optimize=True)
    dT.aa += 0.5 * np.einsum("abie,ej->abij", H.aa.vvov, T.a, optimize=True)
    dT.aa += 0.5 * np.einsum("ae,ebij->abij", I1A_vv, T.aa, optimize=True)
    dT.aa -= 0.5 * np.einsum("mi,abmj->abij", I1A_oo, T.aa, optimize=True)
    dT.aa += np.einsum("amie,ebmj->abij", I2A_voov, T.aa, optimize=True)
    dT.aa += np.einsum("amie,bejm->abij", I2B_voov, T.ab, optimize=True)
    dT.aa += 0.25 * np.einsum("abef,efij->abij", H.aa.vvvv, tau, optimize=True)
    dT.aa += 0.125 * np.einsum("mnij,abmn->abij", I2A_oooo, T.aa, optimize=True)

    T.aa, dT.aa = ccp_opt_loops_v3.ccp_opt_loops_v3.update_t2a_opt(
        T.aa,
        dT.aa + 0.25 * H0.aa.vvoo,
        T.aaa, T.aab,
        pspace_aaa, pspace_aab,
        H.a.ov, H.b.ov,
        H.aa.ooov + H0.aa.ooov, H.aa.vovv + H0.aa.vovv,
        H.ab.ooov + H0.ab.ooov, H.ab.vovv + H0.ab.vovv,
        H0.a.oo, H0.a.vv,
        shift
    )

    return T, dT


# @profile
def update_t2b(T, dT, H, H0, shift, pspace_aab, pspace_abb, n3aab, n3abb):
    """
    Update t2b amplitudes by calculating the projection <ij~ab~|(H_N e^(T1+T2))_C|0>.
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

    I2B_ovoo = H.ab.ovoo + np.einsum("maef,efij->maij", H0.ab.ovvv + 0.5 * H.ab.ovvv, T.ab, optimize=True)
    I2B_vooo = H.ab.vooo + np.einsum("amef,efij->amij", H0.ab.vovv + 0.5 * H.ab.vovv, T.ab, optimize=True)

    tau = T.ab + np.einsum('ai,bj->abij', T.a, T.b, optimize=True)

    dT.ab = -np.einsum("mbij,am->abij", I2B_ovoo, T.a, optimize=True)
    dT.ab -= np.einsum("amij,bm->abij", I2B_vooo, T.b, optimize=True)
    dT.ab += np.einsum("abej,ei->abij", H.ab.vvvo, T.a, optimize=True)
    dT.ab += np.einsum("abie,ej->abij", H.ab.vvov, T.b, optimize=True)
    dT.ab += np.einsum("ae,ebij->abij", I1A_vv, T.ab, optimize=True)
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
    dT.ab += np.einsum("abef,efij->abij", H.ab.vvvv, tau, optimize=True)

    T.ab, dT.ab = ccp_opt_loops_v3.ccp_opt_loops_v3.update_t2b_opt(
        T.ab,
        dT.ab + H0.ab.vvoo,
        T.aab, T.abb,
        pspace_aab, pspace_abb,
        H.a.ov, H.b.ov,
        H.aa.ooov + H0.aa.ooov, H.aa.vovv + H0.aa.vovv,
        H.ab.ooov + H0.ab.ooov, H.ab.oovo + H0.ab.oovo, H.ab.vovv + H0.ab.vovv, H.ab.ovvv + H0.ab.ovvv,
        H.bb.ooov + H0.bb.ooov, H.bb.vovv + H0.bb.vovv,
        H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
        shift
    )

    return T, dT


# @profile
def update_t2c(T, dT, H, H0, shift, pspace_abb, pspace_bbb, n3abb, n3bbb):
    """
    Update t2c amplitudes by calculating the projection <i~j~a~b~|(H_N e^(T1+T2))_C|0>.
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

    I2C_vooo = H.bb.vooo + 0.5*np.einsum('anef,efij->anij', H0.bb.vovv + 0.5 * H.bb.vovv, T.bb, optimize=True)

    tau = 0.5 * T.bb + np.einsum('ai,bj->abij', T.b, T.b, optimize=True)

    dT.bb = -0.5 * np.einsum("amij,bm->abij", I2C_vooo, T.b, optimize=True)
    dT.bb += 0.5 * np.einsum("abie,ej->abij", H.bb.vvov, T.b, optimize=True)
    dT.bb += 0.5 * np.einsum("ae,ebij->abij", I1B_vv, T.bb, optimize=True)
    dT.bb -= 0.5 * np.einsum("mi,abmj->abij", I1B_oo, T.bb, optimize=True)
    dT.bb += np.einsum("amie,ebmj->abij", I2C_voov, T.bb, optimize=True)
    dT.bb += np.einsum("maei,ebmj->abij", I2B_ovvo, T.ab, optimize=True)
    dT.bb += 0.25 * np.einsum("abef,efij->abij", H.bb.vvvv, tau, optimize=True)
    dT.bb += 0.125 * np.einsum("mnij,abmn->abij", I2C_oooo, T.bb, optimize=True)

    T.bb, dT.bb = ccp_opt_loops_v3.ccp_opt_loops_v3.update_t2c_opt(
        T.bb,
        dT.bb + 0.25 * H0.bb.vvoo,
        T.abb, T.bbb,
        pspace_abb, pspace_bbb,
        H.a.ov, H.b.ov,
        H.ab.oovo + H0.ab.oovo, H.ab.ovvv + H0.ab.ovvv,
        H.bb.ooov + H0.bb.ooov, H.bb.vovv + H0.bb.vovv,
        H0.b.oo, H0.b.vv,
        shift
    )

    return T, dT


# @profile
def update_t3a(T, dT, H, H0, shift, pspace_aaa, pspace_aab):
    """
    Update t3a amplitudes by calculating the projection <ijkabc|(H_N e^(T1+T2+T3))_C|0>.
    """
    I2A_vooo = H.aa.vooo - np.einsum("me,aeij->amij", H.a.ov, T.aa, optimize=True)

    T.aaa, dT.aaa = ccp_opt_loops_v3.ccp_opt_loops_v3.update_t3a_p_opt2(
        T.aa, T.aaa, T.aab,
        pspace_aaa, pspace_aab,
        H.a.oo, H.a.vv,
        H.aa.oovv, H.aa.vvov, I2A_vooo,
        H.aa.oooo, H.aa.voov, H.aa.vvvv,
        H.ab.oovv, H.ab.voov,
        H0.a.oo, H0.a.vv,
        shift
    )

    return T, dT


# @profile
def update_t3b(T, dT, H, H0, shift, pspace_aaa, pspace_aab, pspace_abb):
    """
    Update t3b amplitudes by calculating the projection <ijk~abc~|(H_N e^(T1+T2+T3))_C|0>.
    """
    I2A_vooo = H.aa.vooo - np.einsum("me,aeij->amij", H.a.ov, T.aa, optimize=True)
    I2B_ovoo = H.ab.ovoo - np.einsum("me,ecjk->mcjk", H.a.ov, T.ab, optimize=True)
    I2B_vooo = H.ab.vooo - np.einsum("me,aeik->amik", H.b.ov, T.ab, optimize=True)

    T.aab, dT.aab = ccp_opt_loops_v3.ccp_opt_loops_v3.update_t3b_p_opt2(
        T.aa, T.ab, T.aaa, T.aab, T.abb,
        pspace_aaa, pspace_aab, pspace_abb,
        H.a.oo, H.a.vv, H.b.oo, H.b.vv,
        H.aa.oovv, H.aa.vvov, I2A_vooo, H.aa.oooo, H.aa.voov, H.aa.vvvv,
        H.ab.oovv, H.ab.vvov, H.ab.vvvo, I2B_vooo, I2B_ovoo,
        H.ab.oooo, H.ab.voov, H.ab.vovo, H.ab.ovov, H.ab.ovvo, H.ab.vvvv,
        H.bb.oovv, H.bb.voov,
        H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
        shift
    )

    return T, dT


# @profile
def update_t3c(T, dT, H, H0, shift, pspace_aab, pspace_abb, pspace_bbb):
    """
    Update t3c amplitudes by calculating the projection <ij~k~ab~c~|(H_N e^(T1+T2+T3))_C|0>.
    """
    I2C_vooo = H.bb.vooo - np.einsum("me,aeij->amij", H.b.ov, T.bb, optimize=True)
    I2B_ovoo = H.ab.ovoo - np.einsum("me,ecjk->mcjk", H.a.ov, T.ab, optimize=True)
    I2B_vooo = H.ab.vooo - np.einsum("me,aeik->amik", H.b.ov, T.ab, optimize=True)

    T.abb, dT.abb = ccp_opt_loops_v3.ccp_opt_loops_v3.update_t3c_p_opt2(
        T.ab, T.bb, T.aab, T.abb, T.bbb,
        pspace_aab, pspace_abb, pspace_bbb,
        H.a.oo, H.a.vv, H.b.oo, H.b.vv,
        H.aa.oovv, H.aa.voov,
        H.ab.oovv, I2B_vooo, I2B_ovoo, H.ab.vvov, H.ab.vvvo, H.ab.oooo,
        H.ab.voov, H.ab.vovo, H.ab.ovov, H.ab.ovvo, H.ab.vvvv,
        H.bb.oovv, I2C_vooo, H.bb.vvov, H.bb.oooo, H.bb.voov, H.bb.vvvv,
        H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
        shift
    )

    return T, dT


# @profile
def update_t3d(T, dT, H, H0, shift, pspace_abb, pspace_bbb):
    """
    Update t3d amplitudes by calculating the projection <i~j~k~a~b~c~|(H_N e^(T1+T2+T3))_C|0>.
    """
    I2C_vooo = H.bb.vooo - np.einsum("me,aeij->amij", H.b.ov, T.bb, optimize=True)

    T.bbb, dT.bbb = ccp_opt_loops_v3.ccp_opt_loops_v3.update_t3d_p_opt2(
        T.bb, T.abb, T.bbb,
        pspace_abb, pspace_bbb,
        H.b.oo, H.b.vv,
        H.ab.oovv, H.ab.ovvo,
        H.bb.oovv, I2C_vooo, H.bb.vvov, H.bb.oooo, H.bb.voov, H.bb.vvvv,
        H0.b.oo, H0.b.vv,
        shift
    )

    return T, dT

########################################################
### OLD VERSION - DEVECTORIZED ONLY EXPENSIVE TERMS ####
########################################################

# #@profile
# def update_t3a(T, dT, H, H0, shift, pspace_aaa, pspace_aab):
#     """
#     Update t3a amplitudes by calculating the projection <ijkabc|(H_N e^(T1+T2+T3))_C|0>.
#     """
#     # <ijkabc | H(2) | 0 > + (VT3)_C intermediates
#     I2A_vvov = H.aa.vvov - 0.5 * np.einsum("mnef,abfimn->abie", H0.aa.oovv, T.aaa, optimize=True)
#     I2A_vvov -= np.einsum("mnef,abfimn->abie", H0.ab.oovv, T.aab, optimize=True)
#
#     I2A_vooo = 0.5 * np.einsum("mnef,aefijn->amij", H0.aa.oovv, T.aaa, optimize=True)
#     I2A_vooo += H.aa.vooo + np.einsum("mnef,aefijn->amij", H0.ab.oovv, T.aab, optimize=True)
#     I2A_vooo -= np.einsum("me,aeij->amij", H.a.ov, T.aa, optimize=True)
#
#     # MM(2,3)A
#     dT.aaa = -0.25 * np.einsum("amij,bcmk->abcijk", I2A_vooo, T.aa, optimize=True)
#     #dT.aaa += 0.25 * np.einsum("abie,ecjk->abcijk", I2A_vvov, T.aa, optimize=True) #
#     # (HBar*T3)_C
#     dT.aaa -= (1.0 / 12.0) * np.einsum("mk,abcijm->abcijk", H.a.oo, T.aaa, optimize=True)
#     #dT.aaa += (1.0 / 12.0) * np.einsum("ce,abeijk->abcijk", H.a.vv, T.aaa, optimize=True) #
#     dT.aaa += (1.0 / 24.0) * np.einsum("mnij,abcmnk->abcijk", H.aa.oooo, T.aaa, optimize=True)
#     #dT.aaa += (1.0 / 24.0) * np.einsum("abef,efcijk->abcijk", H.aa.vvvv, T.aaa, optimize=True) #
#     #dT.aaa += 0.25 * np.einsum("cmke,abeijm->abcijk", H.aa.voov, T.aaa, optimize=True) #
#     #dT.aaa += 0.25 * np.einsum("cmke,abeijm->abcijk", H.ab.voov, T.aab, optimize=True) ##
#
#     T.aaa, dT.aaa = ccp_opt_loops.ccp_opt_loops.update_t3a_p_opt2(
#         dT.aaa, T.aa, T.aaa, T.aab,
#         pspace_aaa, pspace_aab,
#         H.a.vv,
#         H.aa.oovv, I2A_vvov, H.aa.voov, H.aa.vvvv,
#         H.ab.oovv, H.ab.voov,
#         H0.a.oo, H0.a.vv,
#         shift
#     )
#
#     return T, dT

# #@profile
# def update_t3b(T, dT, H, H0, shift, pspace_aaa, pspace_aab, pspace_abb):
#     """
#     Update t3b amplitudes by calculating the projection <ijk~abc~|(H_N e^(T1+T2+T3))_C|0>.
#     """
#     # <ijk~abc~ | H(2) | 0 > + (VT3)_C intermediates
#     I2A_vvov = -0.5 * np.einsum("mnef,abfimn->abie", H0.aa.oovv, T.aaa, optimize=True)
#     I2A_vvov += -np.einsum("mnef,abfimn->abie", H0.ab.oovv, T.aab, optimize=True)
#     I2A_vvov += H.aa.vvov
#
#     I2A_vooo = 0.5 * np.einsum("mnef,aefijn->amij", H0.aa.oovv, T.aaa, optimize=True)
#     I2A_vooo += np.einsum("mnef,aefijn->amij", H0.ab.oovv, T.aab, optimize=True)
#     I2A_vooo += -np.einsum("me,aeij->amij", H.a.ov, T.aa, optimize=True)
#     I2A_vooo += H.aa.vooo
#
#     I2B_vvvo = -0.5 * np.einsum("mnef,afbmnj->abej", H0.aa.oovv, T.aab, optimize=True)
#     I2B_vvvo += -np.einsum("mnef,afbmnj->abej", H0.ab.oovv, T.abb, optimize=True)
#     I2B_vvvo += H.ab.vvvo
#
#     I2B_ovoo = 0.5 * np.einsum("mnef,efbinj->mbij", H0.aa.oovv, T.aab, optimize=True)
#     I2B_ovoo += np.einsum("mnef,efbinj->mbij", H0.ab.oovv, T.abb, optimize=True)
#     I2B_ovoo += -np.einsum("me,ecjk->mcjk", H.a.ov, T.ab, optimize=True)
#     I2B_ovoo += H.ab.ovoo
#
#     I2B_vvov = -np.einsum("nmfe,afbinm->abie", H0.ab.oovv, T.aab, optimize=True)
#     I2B_vvov += -0.5 * np.einsum("nmfe,afbinm->abie", H0.bb.oovv, T.abb, optimize=True)
#     I2B_vvov += H.ab.vvov
#
#     I2B_vooo = np.einsum("nmfe,afeinj->amij", H0.ab.oovv, T.aab, optimize=True)
#     I2B_vooo += 0.5 * np.einsum("nmfe,afeinj->amij", H0.bb.oovv, T.abb, optimize=True)
#     I2B_vooo += -np.einsum("me,aeik->amik", H.b.ov, T.ab, optimize=True)
#     I2B_vooo += H.ab.vooo
#
#     # MM(2,3)B
#     dT.aab = -0.5 * np.einsum("mcjk,abim->abcijk", I2B_ovoo, T.aa, optimize=True)
#     #dT.aab += 0.5 * np.einsum("bcek,aeij->abcijk", I2B_vvvo, T.aa, optimize=True) #
#     dT.aab -= np.einsum("amik,bcjm->abcijk", I2B_vooo, T.ab, optimize=True)
#     #dT.aab += np.einsum("acie,bejk->abcijk", I2B_vvov, T.ab, optimize=True) ##
#     dT.aab -= 0.5 * np.einsum("amij,bcmk->abcijk", I2A_vooo, T.ab, optimize=True)
#     #dT.aab += 0.5 * np.einsum("abie,ecjk->abcijk", I2A_vvov, T.ab, optimize=True) #
#
#     # (HBar*T3)_C
#     dT.aab -= 0.5 * np.einsum("mi,abcmjk->abcijk", H.a.oo, T.aab, optimize=True)
#     dT.aab -= 0.25 * np.einsum("mk,abcijm->abcijk", H.b.oo, T.aab, optimize=True)
#     #dT.aab += 0.5 * np.einsum("ae,ebcijk->abcijk", H.a.vv, T.aab, optimize=True) #
#     #dT.aab += 0.25 * np.einsum("ce,abeijk->abcijk", H.b.vv, T.aab, optimize=True) ##
#     dT.aab += 0.125 * np.einsum("mnij,abcmnk->abcijk", H.aa.oooo, T.aab, optimize=True)
#     dT.aab += 0.5 * np.einsum("mnjk,abcimn->abcijk", H.ab.oooo, T.aab, optimize=True)
#     #dT.aab += 0.125 * np.einsum("abef,efcijk->abcijk", H.aa.vvvv, T.aab, optimize=True) #
#     #dT.aab += 0.5 * np.einsum("bcef,aefijk->abcijk", H.ab.vvvv, T.aab, optimize=True) #
#     #dT.aab += np.einsum("amie,ebcmjk->abcijk", H.aa.voov, T.aab, optimize=True) #
#     #dT.aab += np.einsum("amie,becjmk->abcijk", H.ab.voov, T.abb, optimize=True) ##
#     #dT.aab += 0.25 * np.einsum("mcek,abeijm->abcijk", H.ab.ovvo, T.aaa, optimize=True) #
#     #dT.aab += 0.25 * np.einsum("cmke,abeijm->abcijk", H.bb.voov, T.aab, optimize=True) ##
#     #dT.aab -= 0.5 * np.einsum("amek,ebcijm->abcijk", H.ab.vovo, T.aab, optimize=True) #
#     #dT.aab -= 0.5 * np.einsum("mcie,abemjk->abcijk", H.ab.ovov, T.aab, optimize=True) ##
#
#     T.aab, dT.aab = ccp_opt_loops.ccp_opt_loops.update_t3b_p_opt2(
#         dT.aab, T.aa, T.ab, T.aaa, T.aab, T.abb,
#         pspace_aaa, pspace_aab, pspace_abb,
#         H.a.vv, H.b.vv,
#         H.aa.oovv, I2A_vvov, H.aa.voov, H.aa.vvvv,
#         H.ab.oovv, I2B_vvov, I2B_vvvo, H.ab.voov, H.ab.vovo, H.ab.ovov, H.ab.ovvo, H.ab.vvvv,
#         H.bb.voov,
#         H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
#         shift
#     )
#
#     return T, dT


# #@profile
# def update_t3c(T, dT, H, H0, shift, pspace_aab, pspace_abb, pspace_bbb):
#     """
#     Update t3c amplitudes by calculating the projection <ij~k~ab~c~|(H_N e^(T1+T2+T3))_C|0>.
#     """
#     # <ij~k~ab~c~ | H(2) | 0 > + (VT3)_C intermediates
#     I2B_vvvo = -0.5 * np.einsum("mnef,afbmnj->abej", H0.aa.oovv, T.aab, optimize=True)
#     I2B_vvvo += -np.einsum("mnef,afbmnj->abej", H0.ab.oovv, T.abb, optimize=True)
#     I2B_vvvo += H.ab.vvvo
#
#     I2B_ovoo = 0.5 * np.einsum("mnef,efbinj->mbij", H0.aa.oovv, T.aab, optimize=True)
#     I2B_ovoo += np.einsum("mnef,efbinj->mbij", H0.ab.oovv, T.abb, optimize=True)
#     I2B_ovoo += H.ab.ovoo
#
#     I2B_ovoo -= np.einsum("me,ebij->mbij", H.a.ov, T.ab, optimize=True)
#     I2B_vvov = -np.einsum("nmfe,afbinm->abie", H0.ab.oovv, T.aab, optimize=True)
#     I2B_vvov += -0.5 * np.einsum("nmfe,afbinm->abie", H0.bb.oovv, T.abb, optimize=True)
#     I2B_vvov += H.ab.vvov
#
#     I2B_vooo = np.einsum("nmfe,afeinj->amij", H0.ab.oovv, T.aab, optimize=True)
#     I2B_vooo += 0.5 * np.einsum("nmfe,afeinj->amij", H0.bb.oovv, T.abb, optimize=True)
#     I2B_vooo += H.ab.vooo
#     I2B_vooo -= np.einsum("me,aeij->amij", H.b.ov, T.ab, optimize=True)
#
#     I2C_vvov = -0.5 * np.einsum("mnef,abfimn->abie", H0.bb.oovv, T.bbb, optimize=True)
#     I2C_vvov += -np.einsum("nmfe,fabnim->abie", H0.ab.oovv, T.abb, optimize=True)
#     I2C_vvov += H.bb.vvov
#
#     I2C_vooo = np.einsum("nmfe,faenij->amij", H0.ab.oovv, T.abb, optimize=True)
#     I2C_vooo += 0.5 * np.einsum("mnef,aefijn->amij", H0.bb.oovv, T.bbb, optimize=True)
#     I2C_vooo -= np.einsum("me,cekj->cmkj", H.b.ov, T.bb, optimize=True)
#     I2C_vooo += H.bb.vooo
#
#     # MM(2,3)C
#     dT.abb = -0.5 * np.einsum("amij,bcmk->abcijk", I2B_vooo, T.bb, optimize=True)
#     #dT.abb += 0.5 * np.einsum("abie,ecjk->abcijk", I2B_vvov, T.bb, optimize=True) ##
#     dT.abb -= 0.5 * np.einsum("cmkj,abim->abcijk", I2C_vooo, T.ab, optimize=True)
#     #dT.abb += 0.5 * np.einsum("cbke,aeij->abcijk", I2C_vvov, T.ab, optimize=True) ##
#     dT.abb -= np.einsum("mbij,acmk->abcijk", I2B_ovoo, T.ab, optimize=True)
#     #dT.abb += np.einsum("abej,ecik->abcijk", I2B_vvvo, T.ab, optimize=True) #
#
#     # (HBar*T3)_C
#     dT.abb -= 0.25 * np.einsum("mi,abcmjk->abcijk", H.a.oo, T.abb, optimize=True)
#     dT.abb -= 0.5 * np.einsum("mj,abcimk->abcijk", H.b.oo, T.abb, optimize=True)
#     #dT.abb += 0.25 * np.einsum("ae,ebcijk->abcijk", H.a.vv, T.abb, optimize=True) #
#     #dT.abb += 0.5 * np.einsum("be,aecijk->abcijk", H.b.vv, T.abb, optimize=True) ##
#     dT.abb += 0.125 * np.einsum("mnjk,abcimn->abcijk", H.bb.oooo, T.abb, optimize=True)
#     dT.abb += 0.5 * np.einsum("mnij,abcmnk->abcijk", H.ab.oooo, T.abb, optimize=True)
#     #dT.abb += 0.125 * np.einsum("bcef,aefijk->abcijk", H.bb.vvvv, T.abb, optimize=True) ##
#     #dT.abb += 0.5 * np.einsum("abef,efcijk->abcijk", H.ab.vvvv, T.abb, optimize=True) #
#     #dT.abb += 0.25 * np.einsum("amie,ebcmjk->abcijk", H.aa.voov, T.abb, optimize=True) #
#     #dT.abb += 0.25 * np.einsum("amie,ebcmjk->abcijk", H.ab.voov, T.bbb, optimize=True) ##
#     #dT.abb += np.einsum("mbej,aecimk->abcijk", H.ab.ovvo, T.aab, optimize=True) #
#     #dT.abb += np.einsum("bmje,aecimk->abcijk", H.bb.voov, T.abb, optimize=True) ##
#     #dT.abb -= 0.5 * np.einsum("mbie,aecmjk->abcijk", H.ab.ovov, T.abb, optimize=True) ##
#     #dT.abb -= 0.5 * np.einsum("amej,ebcimk->abcijk", H.ab.vovo, T.abb, optimize=True) #
#
#     T.abb, dT.abb = ccp_opt_loops.ccp_opt_loops.update_t3c_p_opt2(
#         dT.abb, T.ab, T.bb, T.aab, T.abb, T.bbb,
#         pspace_aab, pspace_abb, pspace_bbb,
#         H.a.vv, H.b.vv,
#         H.aa.voov,
#         H.ab.oovv, I2B_vvov, I2B_vvvo, H.ab.voov, H.ab.vovo, H.ab.ovov, H.ab.ovvo, H.ab.vvvv,
#         H.bb.oovv, I2C_vvov, H.bb.voov, H.bb.vvvv,
#         H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
#         shift
#     )
#
#     return T, dT


# #@profile
# def update_t3d(T, dT, H, H0, shift, pspace_abb, pspace_bbb):
#     """
#     Update t3d amplitudes by calculating the projection <i~j~k~a~b~c~|(H_N e^(T1+T2+T3))_C|0>.
#     """
#     #  <i~j~k~a~b~c~ | H(2) | 0 > + (VT3)_C intermediates
#     I2C_vvov = -0.5 * np.einsum("mnef,abfimn->abie", H0.bb.oovv, T.bbb, optimize=True)
#     I2C_vvov -= np.einsum("nmfe,fabnim->abie", H0.ab.oovv, T.abb, optimize=True)
#     I2C_vvov += H.bb.vvov
#
#     I2C_vooo = 0.5 * np.einsum("mnef,aefijn->amij", H0.bb.oovv, T.bbb, optimize=True)
#     I2C_vooo += np.einsum("nmfe,faenij->amij", H0.ab.oovv, T.abb, optimize=True)
#     I2C_vooo -= np.einsum("me,aeij->amij", H.b.ov, T.bb, optimize=True)
#     I2C_vooo += H.bb.vooo
#
#     # MM(2,3)D
#     dT.bbb = -0.25 * np.einsum("amij,bcmk->abcijk", I2C_vooo, T.bb, optimize=True)
#     #dT.bbb += 0.25 * np.einsum("abie,ecjk->abcijk", I2C_vvov, T.bb, optimize=True) ##
#     # (HBar*T3)_C
#     dT.bbb -= (1.0 / 12.0) * np.einsum("mk,abcijm->abcijk", H.b.oo, T.bbb, optimize=True)
#     #dT.bbb += (1.0 / 12.0) * np.einsum("ce,abeijk->abcijk", H.b.vv, T.bbb, optimize=True) ##
#     dT.bbb += (1.0 / 24.0) * np.einsum("mnij,abcmnk->abcijk", H.bb.oooo, T.bbb, optimize=True)
#     #dT.bbb += (1.0 / 24.0) * np.einsum("abef,efcijk->abcijk", H.bb.vvvv, T.bbb, optimize=True) ##
#     #dT.bbb += 0.25 * np.einsum("maei,ebcmjk->abcijk", H.ab.ovvo, T.abb, optimize=True) #
#     #dT.bbb += 0.25 * np.einsum("amie,ebcmjk->abcijk", H.bb.voov, T.bbb, optimize=True) ##
#
#     T.bbb, dT.bbb = ccp_opt_loops.ccp_opt_loops.update_t3d_p_opt2(
#         dT.bbb, T.bb, T.abb, T.bbb,
#         pspace_abb, pspace_bbb,
#         H.b.vv,
#         H.ab.oovv, H.ab.ovvo,
#         H.bb.oovv, I2C_vvov, H.bb.voov, H.bb.vvvv,
#         H0.b.oo, H0.b.vv,
#         shift
#     )
#
#     return T, dT