"""
Module with functions to perform the approximate coupled-cluster (CC) method with
singles, doubles, and triples, abbrevated as CC3, where the T3 operator is correct
through 2st order of perturbation theory and T1 clusters are treated as 0th order.

References:
    [1] J. Chem. Phys. 106, 1808 (1997); doi: 10.1063/1.473322 [CC3 method]
    [2] J. Chem. Phys. 103, 7429–7441 (1995); doi: 10.1063/1.470315 [CC3 response & excited states]
    [3] J. Chem. Phys. 122, 054110 (2005); doi: 10.1063/1.1835953 [CC3 for open shells]
"""
import numpy as np
# Modules for type checking
from typing import List, Tuple
from ccpy.models.operators import ClusterOperator
from ccpy.models.system import System
from ccpy.models.integrals import Integral
# Modules for computation
from ccpy.hbar.hbar_ccs import get_pre_ccs_intermediates, get_ccs_intermediates_opt
from ccpy.hbar.hbar_cc3 import get_cc3_intermediates
from ccpy.lib.core import cc3_loops


def update(T: ClusterOperator,
           dT: ClusterOperator,
           H: Integral,
           hbar: Integral,
           shift: float,
           flag_RHF: bool,
           system: System) -> Tuple[ClusterOperator, ClusterOperator]:
    """

    Parameters
    ----------
    T :
    dT :
    H :
    hbar :
    shift :
    flag_RHF :
    system :

    Returns
    -------

    """
    # pre-CCS intermediates
    hbar = get_pre_ccs_intermediates(hbar, T, H, system, flag_RHF)

    # update T1
    dT = build_t1a(T, dT, H, hbar)
    if flag_RHF:
        dT.b = dT.a.copy()
    else:
        dT = build_t1b(T, dT, H, hbar)

    # CCS intermediates
    hbar = get_ccs_intermediates_opt(hbar, T, H, system, flag_RHF)

    # update T2
    dT = build_t2a(T, dT, hbar, H)
    dT = build_t2b(T, dT, hbar, H)
    if flag_RHF:
        dT.bb = dT.aa.copy()
    else:
        dT = build_t2c(T, dT, hbar, H)

    # Adjust CCS intermediate values for contractions with T3 in T2 updates
    hbar.aa.ooov += H.aa.ooov
    hbar.aa.vovv += H.aa.vovv
    hbar.ab.ooov += H.ab.ooov
    hbar.ab.oovo += H.ab.oovo
    hbar.ab.vovv += H.ab.vovv
    hbar.ab.ovvv += H.ab.ovvv
    if flag_RHF:
        hbar.bb.ooov = hbar.aa.ooov
        hbar.bb.vovv = hbar.aa.vovv
    else:
        hbar.bb.ooov += H.bb.ooov
        hbar.bb.vovv += H.bb.vovv

    # CCS-like transformed intermediates for CC3
    X = get_cc3_intermediates(T, H)

    # Update all T1 and T2 clusters together by computing T3 on-the-fly once
    # it would be nice if the T1-transformed X intermediates here were simply the
    # elements of the CCS-transformed hbar computed before T2 builds.
    T.a, T.b, T.aa, T.ab, T.bb, dT.a, dT.b, dT.aa, dT.ab, dT.bb = cc3_loops.update_t(
        T.a, T.b, T.aa, T.ab, T.bb,
        dT.a, dT.b, dT.aa, dT.ab, dT.bb,
        H.a.oo, H.a.vv, H.b.oo, H.b.vv,
        hbar.a.ov, hbar.b.ov,
        H.aa.oovv, H.ab.oovv, H.bb.oovv,
        hbar.aa.ooov, hbar.aa.vovv,
        hbar.ab.ooov, hbar.ab.oovo, hbar.ab.vovv, hbar.ab.ovvv,
        hbar.bb.ooov, hbar.bb.vovv,
        X["aa"]["vooo"], X["aa"]["vvov"],
        X["ab"]["vooo"], X["ab"]["ovoo"], X["ab"]["vvov"], X["ab"]["vvvo"],
        X["bb"]["vooo"], X["bb"]["vvov"],
        shift)
    return T, dT


def build_t1a(T, dT, H, X):
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
    dT.a += H.a.vo
    return dT


def build_t1b(T, dT, H, X):
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
    dT.b += H.b.vo
    return dT


def build_t2a(T, dT, H, H0):
    """
    Update t2a amplitudes by calculating the projection <ijab|(H_N e^(T1+T2))_C|0>.
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
    dT.aa += 0.25 * H0.aa.vvoo
    return dT


def build_t2b(T, dT, H, H0):
    """
    Update t2b amplitudes by calculating the projection <ij~ab~|(H_N e^(T1+T2))_C|0>.
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
    dT.ab += H0.ab.vvoo
    return dT


def build_t2c(T, dT, H, H0):
    """
    Update t2c amplitudes by calculating the projection <i~j~a~b~|(H_N e^(T1+T2))_C|0>.
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
    dT.bb += 0.25 * H0.bb.vvoo
    return dT

### CODE FOR DOING BATCHED (abc) VECTORIZATION IN PYTHON ###
    # ### t3aaa ###
    # e_abc = -eps_a_v[:, n, n] - eps_a_v[n, :, n] - eps_a_v[n, n, :]
    # for i in range(system.noccupied_alpha):
    #     for j in range(i + 1, system.noccupied_alpha):
    #         for k in range(j + 1, system.noccupied_alpha):
    #             # fock denominator for occupied
    #             e_ijk = H.a.oo[i, i] + H.a.oo[j, j] + H.a.oo[k, k]
    #             # -1/2 A(k/ij)A(abc) I(amij) * t(bcmk)
    #             t3a_abc = -0.5 * np.einsum("am,bcm->abc", X["aa"]["vooo"][:, :, i, j], T.aa[:, :, :, k], optimize=True)
    #             t3a_abc += 0.5 * np.einsum("am,bcm->abc", X["aa"]["vooo"][:, :, k, j], T.aa[:, :, :, i], optimize=True)
    #             t3a_abc += 0.5 * np.einsum("am,bcm->abc", X["aa"]["vooo"][:, :, i, k], T.aa[:, :, :, j], optimize=True)
    #             # 1/2 A(i/jk)A(abc) I(abie) * t(ecjk)
    #             t3a_abc += 0.5 * np.einsum("abe,ec->abc", X["aa"]["vvov"][:, :, i, :], T.aa[:, :, j, k], optimize=True)
    #             t3a_abc -= 0.5 * np.einsum("abe,ec->abc", X["aa"]["vvov"][:, :, j, :], T.aa[:, :, i, k], optimize=True)
    #             t3a_abc -= 0.5 * np.einsum("abe,ec->abc", X["aa"]["vvov"][:, :, k, :], T.aa[:, :, j, i], optimize=True)
    #             # Antisymmetrize A(abc)
    #             t3a_abc -= np.transpose(t3a_abc, (1, 0, 2)) + np.transpose(t3a_abc, (2, 1, 0)) # A(a/bc)
    #             t3a_abc -= np.transpose(t3a_abc, (0, 2, 1)) # A(bc)
    #             # Divide t_abc by the denominator
    #             t3a_abc /= (e_ijk + e_abc)
    #             # Compute diagram: 1/2 A(i/jk) v(jkbc) * t(abcijk)
    #             dT.a[:, i] += 0.5 * np.einsum("bc,abc->a", H.aa.oovv[j, k, :, :], t3a_abc, optimize=True)
    #             dT.a[:, j] -= 0.5 * np.einsum("bc,abc->a", H.aa.oovv[i, k, :, :], t3a_abc, optimize=True)
    #             dT.a[:, k] -= 0.5 * np.einsum("bc,abc->a", H.aa.oovv[j, i, :, :], t3a_abc, optimize=True)
    #             # Compute diagram: A(ij) [A(k/ij) h(ke) * t3(abeijk)]
    #             dT.aa[:, :, i, j] += 0.5 * np.einsum("e,abe->ab", hbar.a.ov[k, :], t3a_abc, optimize=True) # (1)
    #             dT.aa[:, :, j, k] += 0.5 * np.einsum("e,abe->ab", hbar.a.ov[i, :], t3a_abc, optimize=True) # (ik)
    #             dT.aa[:, :, i, k] -= 0.5 * np.einsum("e,abe->ab", hbar.a.ov[j, :], t3a_abc, optimize=True) # (jk)
    #             # Compute diagram: -A(j/ik) h(ik:f) * t3(abfijk)
    #             dT.aa[:, :, :, j] -= 0.5 * np.einsum("mf,abf->abm", hbar.aa.ooov[i, k, :, :], t3a_abc, optimize=True)
    #             dT.aa[:, :, :, i] += 0.5 * np.einsum("mf,abf->abm", hbar.aa.ooov[j, k, :, :], t3a_abc, optimize=True)
    #             dT.aa[:, :, :, k] += 0.5 * np.einsum("mf,abf->abm", hbar.aa.ooov[i, j, :, :], t3a_abc, optimize=True)
    #             # Compute diagram: 1/2 A(k/ij) h(akef) * t3(ebfijk)
    #             dT.aa[:, :, i, j] += 0.5 * np.einsum("aef,ebf->ab", hbar.aa.vovv[:, k, :, :], t3a_abc, optimize=True)
    #             dT.aa[:, :, j, k] += 0.5 * np.einsum("aef,ebf->ab", hbar.aa.vovv[:, i, :, :], t3a_abc, optimize=True)
    #             dT.aa[:, :, i, k] -= 0.5 * np.einsum("aef,ebf->ab", hbar.aa.vovv[:, j, :, :], t3a_abc, optimize=True)
    # ### t3aab ###
    # e_abc = -eps_a_v[:, n, n] - eps_a_v[n, :, n] - eps_b_v[n, n, :]
    # for i in range(system.noccupied_alpha):
    #     for j in range(i + 1, system.noccupied_alpha):
    #         for k in range(system.noccupied_beta):
    #             # fock denominator for occupied
    #             e_ijk = H.a.oo[i, i] + H.a.oo[j, j] + H.b.oo[k, k]
    #             # Diagram 1: A(ab) H2B(bcek) * t2a(aeij)
    #             t3b_abc = np.einsum("bce,ae->abe", X["ab"]["vvvo"][:, :, :, k], T.aa[:, :, i, j,], optimize=True)
    #             # Diagram 2: -A(ij) I2B(mcjk) * t2a(abim)
    #             t3b_abc -= 0.5 * np.einsum("mc,abm->abc", X["ab"]["ovoo"][:, :, j, k], T.aa[:, :, i, :], optimize=True)
    #             t3b_abc += 0.5 * np.einsum("mc,abm->abc", X["ab"]["ovoo"][:, :, i, k], T.aa[:, :, j, :], optimize=True)
    #             # Diagram 3: A(ab)A(ij) H2B(acie)*t2b(bejk)
    #             t3b_abc += np.einsum("ace,be->abc", X["ab"]["vvov"][:, :, i, :], T.ab[:, :, j, k], optimize=True)
    #             t3b_abc -= np.einsum("ace,be->abc", X["ab"]["vvov"][:, :, j, :], T.ab[:, :, i, k], optimize=True)
    #             # Diagram 4: -A(ab)A(ij) I2B(amik)*t2b(bcjm)
    #             t3b_abc -= np.einsum("am,bcm->abc", X["ab"]["vooo"][:, :, i, k], T.ab[:, :, j, :], optimize=True)
    #             t3b_abc += np.einsum("am,bcm->abc", X["ab"]["vooo"][:, :, j, k], T.ab[:, :, i, :], optimize=True)
    #             # Diagram 5: A(ij) H2A(abie)*t2b(ecjk)
    #             t3b_abc += 0.5 * np.einsum("abe,ec->abc", X["ab"]["vvov"][:, :, i, :], T.ab[:, :, j, k], optimize=True)
    #             t3b_abc -= 0.5 * np.einsum("abe,ec->abc", X["ab"]["vvov"][:, :, j, :], T.ab[:, :, i, k], optimize=True)
    #             # Diagram 6: -A(ab) I2A(amij)*t2b(bcmk)
    #             t3b_abc -= np.einsum("am,bcm->abc", X["aa"]["vooo"][:, :, i, j], T.ab[:, :, :, k], optimize=True)
    #             # Antisymmetrize A(ab)
    #             t3b_abc -= np.transpose(t3b_abc, (1, 0, 2)) # A(ab)
    #             # Divide t_abc by the denominator
    #             t3b_abc /= (e_ijk + e_abc)
    #             # A(ij)A(ab) vB(jkbc) * t3b(abcijk)
    #             dT.a[:, i] += np.einsum("bc,abc->a", H.ab.oovv[j, k, :, :], t3b_abc, optimize=True)
    #             dT.a[:, j] -= np.einsum("bc,abc->a", H.ab.oovv[i, k, :, :], t3b_abc, optimize=True)
    #             # vA(ijab) * t3b(abcijk)
    #             dT.b[:, k] += np.einsum("ab,abc->c", H.aa.oovv[i, j, :, :], t3b_abc, optimize=True)
    #             # A(ij)A(ab) [h1b(me) * t3b(abeijm)]
    #             dT.aa[:, :, i, j] += np.einsum("c,abc->ab", hbar.b.ov[k, :], t3b_abc, optimize=True)
    #             # A(ij)A(ab) [A(jm) -h2b(mnif) * t3b(abfmjn)]
    #             dT.aa[:, :, :, j] -= np.einsum("mc,abc->abm", hbar.ab.ooov[i, k, :, :], t3b_abc, optimize=True)
    #             dT.aa[:, :, :, i] += np.einsum("mc,abc->abm", hbar.ab.ooov[j, k, :, :], t3b_abc, optimize=True)
    #             # A(ij)A(ab) [A(be) h2b(anef) * t3b(ebfijn)]
    #             dT.aa[:, :, i, j] += np.einsum("eac,abc->eb", hbar.ab.vovv[:, k, :, :], t3b_abc, optimize=True)
    #             # A(af) -h2a(mnif) * t3b(afbmnj)
    #             dT.ab[:, :, :, k] -= np.einsum("mb,abc->acm", hbar.aa.ooov[i, j, :, :], t3b_abc, optimize=True)
    #             # A(af)A(in) -h2b(nmfj) * t3b(afbinm)
    #             dT.ab[:, :, i, :] -= np.einsum("bm,abc->acm", hbar.ab.oovo[j, k, :, :], t3b_abc, optimize=True)
    #             dT.ab[:, :, j, :] += np.einsum("bm,abc->acm", hbar.ab.oovo[i, k, :, :], t3b_abc, optimize=True)
    #             # A(in) h2a(anef) * t3b(efbinj)
    #             dT.ab[:, :, i, k] += np.einsum("eab,abc->ec", hbar.aa.vovv[:, j, :, :], t3b_abc, optimize=True)
    #             dT.ab[:, :, j, k] -= np.einsum("eab,abc->ec", hbar.aa.vovv[:, i, :, :], t3b_abc, optimize=True)
    #             # A(af)A(in) h2b(nbfe) * t3b(afeinj)
    #             dT.ab[:, :, i, k] += np.einsum("ebc,abc->ae", hbar.ab.ovvv[j, :, :, :], t3b_abc, optimize=True)
    #             dT.ab[:, :, j, k] -= np.einsum("ebc,abc->ae", hbar.ab.ovvv[i, :, :, :], t3b_abc, optimize=True)
    #             # A(ae)A(im) h1a(me) * t3b(aebimj)
    #             dT.ab[:, :, i, k] += np.einsum("b,abc->ab", hbar.a.ov[j, :], t3b_abc, optimize=True)
    #             dT.ab[:, :, j, k] -= np.einsum("b,abc->ab", hbar.a.ov[i, :], t3b_abc, optimize=True)
