import time
from ccpy.utilities.linear_algebra import ccpy_einsum
import numpy as np
from ccpy.constants.constants import hartreetoeV

def calc_eomccsdta_star(T, R, L, omega, corr_energy, H, H0, system, use_RHF=False):
    """

    """
    t_start = time.perf_counter()
    t_cpu_start = time.process_time()

    n = np.newaxis
    eps_o_a = np.diagonal(H0.a.oo); eps_v_a = np.diagonal(H0.a.vv);
    eps_o_b = np.diagonal(H0.b.oo); eps_v_b = np.diagonal(H0.b.vv);
    e3_aaa = (eps_o_a[n, n, n, :, n, n] + eps_o_a[n, n, n, n, :, n] + eps_o_a[n, n, n, n, n, :]
              -eps_v_a[:, n, n, n, n, n] - eps_v_a[n, :, n, n, n, n] - eps_v_a[n, n, :, n, n, n])
    e3_aab = (eps_o_a[n, n, n, :, n, n] + eps_o_a[n, n, n, n, :, n] + eps_o_b[n, n, n, n, n, :]
              -eps_v_a[:, n, n, n, n, n] - eps_v_a[n, :, n, n, n, n] - eps_v_b[n, n, :, n, n, n])

    X = get_eomccsdta_intermediates(H0, R)

    # update R3
    M3A = build_HR_3A(R, T, H0, X)
    L3A = build_LH_3A(L, H0)
    L3A /= (omega + e3_aaa)
    dA_aaa = (1.0 / 36.0) * ccpy_einsum("abcijk,abcijk->", L3A, M3A)

    M3B = build_HR_3B(R, T, H0, X)
    L3B = build_LH_3B(L, H0)
    L3B /= (omega + e3_aab)
    dA_aab = (1.0 / 4.0) * ccpy_einsum("abcijk,abcijk->", L3B, M3B)

    if use_RHF:
        correction_A = 2.0 * dA_aaa + 2.0 * dA_aab
    else:
        e3_abb = (eps_o_a[n, n, n, :, n, n] + eps_o_b[n, n, n, n, :, n] + eps_o_b[n, n, n, n, n, :]
                  - eps_v_a[:, n, n, n, n, n] - eps_v_b[n, :, n, n, n, n] - eps_v_b[n, n, :, n, n, n])
        e3_bbb = (eps_o_b[n, n, n, :, n, n] + eps_o_b[n, n, n, n, :, n] + eps_o_b[n, n, n, n, n, :]
                  - eps_v_b[:, n, n, n, n, n] - eps_v_b[n, :, n, n, n, n] - eps_v_b[n, n, :, n, n, n])

        M3C = build_HR_3C(R, T, H0, X)
        L3C = build_LH_3C(L, H0)
        L3C /= (omega + e3_abb)
        dA_abb = (1.0 / 4.0) * ccpy_einsum("abcijk,abcijk->", L3C, M3C)

        M3D = build_HR_3D(R, T, H0, X)
        L3D = build_LH_3D(L, H0)
        L3D /= (omega + e3_bbb)
        dA_bbb = (1.0 / 36.0) * ccpy_einsum("abcijk,abcijk->", L3D, M3D)

        correction_A = dA_aaa + dA_aab + dA_abb + dA_bbb

    t_end = time.perf_counter()
    t_cpu_end = time.process_time()
    minutes, seconds = divmod(t_end - t_start, 60)

    energy_A = corr_energy + omega + correction_A
    total_energy_A = system.reference_energy + energy_A

    print("")
    print('   EOMCCSD(T)(a)* Calculation Summary')
    print('   -------------------------------------------------')
    print("   Total wall time: {:0.2f}m  {:0.2f}s".format(minutes, seconds))
    print(f"   Total CPU time: {t_cpu_end - t_cpu_start} seconds")
    print("   EOMCCSD(a) = {:>10.10f}    ω = {:>10.10f}     VEE = {:>10.5f} eV".format(system.reference_energy + corr_energy + omega, omega, hartreetoeV * omega))
    print(
        "   EOMCCSD(T)(a)* = {:>10.10f}     δ_A = {:>10.10f}     VEE = {:>10.5f} eV".format(
            total_energy_A, correction_A, (omega + correction_A) * hartreetoeV
        )
    )
    Ecrcc23 = {"A": total_energy_A, "B": 0.0, "C": 0.0, "D": 0.0}
    delta23 = {"A": correction_A, "B": 0.0, "C": 0.0, "D": 0.0}

    return Ecrcc23, delta23


def build_HR_3A(R, T, H, X):
    # <ijkabc| [H(R1+R2+R3)]_C | 0 >
    X3A = 0.25 * ccpy_einsum("baje,ecik->abcijk", X["aa"]["vvov"], T.aa)
    X3A += 0.25 * ccpy_einsum("baje,ecik->abcijk", H.aa.vvov, R.aa)
    X3A -= 0.25 * ccpy_einsum("bmji,acmk->abcijk", X["aa"]["vooo"], T.aa)
    X3A -= 0.25 * ccpy_einsum("bmji,acmk->abcijk", H.aa.vooo, R.aa)
    # antisymmetrize terms and add up: A(abc)A(ijk) = A(a/bc)A(bc)A(i/jk)A(jk)
    X3A -= np.transpose(X3A, (0, 1, 2, 3, 5, 4))
    X3A -= np.transpose(X3A, (0, 1, 2, 4, 3, 5)) + np.transpose(X3A, (0, 1, 2, 5, 4, 3))
    X3A -= np.transpose(X3A, (0, 2, 1, 3, 4, 5))
    X3A -= np.transpose(X3A, (1, 0, 2, 3, 4, 5)) + np.transpose(X3A, (2, 1, 0, 3, 4, 5))
    return X3A

def build_HR_3B(R, T, H, X):
    # < ijk~abc~ | [ H(R1+R2+R3) ]_C | 0 >
    # Intermediate 1: X2B(bcek)*Y2A(aeij) -> Z3B(abcijk)
    X3B = 0.5 * ccpy_einsum("bcek,aeij->abcijk", X["ab"]["vvvo"], T.aa)
    X3B += 0.5 * ccpy_einsum("bcek,aeij->abcijk", H.ab.vvvo, R.aa)
    # Intermediate 2: X2B(ncjk)*Y2A(abin) -> Z3B(abcijk)
    X3B -= 0.5 * ccpy_einsum("ncjk,abin->abcijk", X["ab"]["ovoo"], T.aa)
    X3B -= 0.5 * ccpy_einsum("mcjk,abim->abcijk", H.ab.ovoo, R.aa)
    # Intermediate 3: X2A(baje)*Y2B(ecik) -> Z3B(abcijk)
    X3B += 0.5 * ccpy_einsum("baje,ecik->abcijk", X["aa"]["vvov"], T.ab)
    X3B += 0.5 * ccpy_einsum("baje,ecik->abcijk", H.aa.vvov, R.ab)
    # Intermediate 4: X2A(bnji)*Y2B(acnk) -> Z3B(abcijk)
    X3B -= 0.5 * ccpy_einsum("bnji,acnk->abcijk", X["aa"]["vooo"], T.ab)
    X3B -= 0.5 * ccpy_einsum("bnji,acnk->abcijk", H.aa.vooo, R.ab)
    # Intermediate 5: X2B(bcje)*Y2B(aeik) -> Z3B(abcijk)
    X3B += ccpy_einsum("bcje,aeik->abcijk", X["ab"]["vvov"], T.ab)
    X3B += ccpy_einsum("bcje,aeik->abcijk", H.ab.vvov, R.ab)
    # Intermediate 6: X2B(bnjk)*Y2B(acin) -> Z3B(abcijk)
    X3B -= ccpy_einsum("bnjk,acin->abcijk", X["ab"]["vooo"], T.ab)
    X3B -= ccpy_einsum("bnjk,acin->abcijk", H.ab.vooo, R.ab)
    X3B -= np.transpose(X3B, (1, 0, 2, 3, 4, 5))
    X3B -= np.transpose(X3B, (0, 1, 2, 4, 3, 5))
    return X3B

def build_HR_3C(R, T, H, X):
    # < ij~k~ab~c~ | [ H(R1+R2+R3) ]_C | 0 >
    # Intermediate 1: X2B(cbke)*Y2C(aeij) -> Z3C(cbakji)
    X3C = 0.5 * ccpy_einsum("cbke,aeij->cbakji", X["ab"]["vvov"], T.bb)
    X3C += 0.5 * ccpy_einsum("cbke,aeij->cbakji", H.ab.vvov, R.bb)
    # Intermediate 2: X2B(cnkj)*Y2C(abin) -> Z3C(cbakji)
    X3C -= 0.5 * ccpy_einsum("cnkj,abin->cbakji", X["ab"]["vooo"], T.bb)
    X3C -= 0.5 * ccpy_einsum("cmkj,abim->cbakji", H.ab.vooo, R.bb)
    # Intermediate 3: X2C(baje)*Y2B(ceki) -> Z3C(cbakji)
    X3C += 0.5 * ccpy_einsum("baje,ceki->cbakji", X["bb"]["vvov"], T.ab)
    X3C += 0.5 * ccpy_einsum("baje,ceki->cbakji", H.bb.vvov, R.ab)
    # Intermediate 4: X2C(bnji)*Y2B(cakn) -> Z3C(cbakji)
    X3C -= 0.5 * ccpy_einsum("bnji,cakn->cbakji", X["bb"]["vooo"], T.ab)
    X3C -= 0.5 * ccpy_einsum("bnji,cakn->cbakji", H.bb.vooo, R.ab)
    # Intermediate 5: X2B(cbej)*Y2B(eaki) -> Z3C(cbakji)
    X3C += ccpy_einsum("cbej,eaki->cbakji", X["ab"]["vvvo"], T.ab)
    X3C += ccpy_einsum("cbej,eaki->cbakji", H.ab.vvvo, R.ab)
    # Intermediate 6: X2B(nbkj)*Y2B(cani) -> Z3C(cbakji)
    X3C -= ccpy_einsum("nbkj,cani->cbakji", X["ab"]["ovoo"], T.ab)
    X3C -= ccpy_einsum("nbkj,cani->cbakji", H.ab.ovoo, R.ab)
    X3C -= np.transpose(X3C, (0, 2, 1, 3, 4, 5))
    X3C -= np.transpose(X3C, (0, 1, 2, 3, 5, 4))
    return X3C

def build_HR_3D(R, T, H, X):
    # <i~j~k~a~b~c~| [H(R1+R2+R3)]_C | 0 >
    X3D = 0.25 * ccpy_einsum("baje,ecik->abcijk", X["bb"]["vvov"], T.bb)
    X3D += 0.25 * ccpy_einsum("baje,ecik->abcijk", H.bb.vvov, R.bb)
    X3D -= 0.25 * ccpy_einsum("bmji,acmk->abcijk", X["bb"]["vooo"], T.bb)
    X3D -= 0.25 * ccpy_einsum("bmji,acmk->abcijk", H.bb.vooo, R.bb)
    # antisymmetrize terms and add up: A(abc)A(ijk) = A(a/bc)A(bc)A(i/jk)A(jk)
    X3D -= np.transpose(X3D, (0, 1, 2, 3, 5, 4))
    X3D -= np.transpose(X3D, (0, 1, 2, 4, 3, 5)) + np.transpose(X3D, (0, 1, 2, 5, 4, 3))
    X3D -= np.transpose(X3D, (0, 2, 1, 3, 4, 5))
    X3D -= np.transpose(X3D, (1, 0, 2, 3, 4, 5)) + np.transpose(X3D, (2, 1, 0, 3, 4, 5))
    return X3D

def build_LH_3A(L, H):
    # < 0 | L1 * H(2) | ijkabc >
    L3A = (9.0 / 36.0) * ccpy_einsum("ai,jkbc->abcijk", L.a, H.aa.oovv)
    # < 0 | L2 * H(2) | ijkabc >
    L3A += (9.0 / 36.0) * ccpy_einsum("bcjk,ia->abcijk", L.aa, H.a.ov)
    L3A += (9.0 / 36.0) * ccpy_einsum("ebij,ekac->abcijk", L.aa, H.aa.vovv)
    L3A -= (9.0 / 36.0) * ccpy_einsum("abmj,ikmc->abcijk", L.aa, H.aa.ooov)

    L3A -= np.transpose(L3A, (0, 1, 2, 3, 5, 4)) # (jk)
    L3A -= np.transpose(L3A, (0, 1, 2, 4, 3, 5)) + np.transpose(L3A, (0, 1, 2, 5, 4, 3)) # (i/jk)
    L3A -= np.transpose(L3A, (0, 2, 1, 3, 4, 5)) # (bc)
    L3A -= np.transpose(L3A, (2, 1, 0, 3, 4, 5)) + np.transpose(L3A, (1, 0, 2, 3, 4, 5)) # (a/bc)
    return L3A

def build_LH_3B(L, H):
    # < 0 | L1 * H(2) | ijk~abc~ >
    L3B = ccpy_einsum("ai,jkbc->abcijk", L.a, H.ab.oovv)
    L3B += 0.25 * ccpy_einsum("ck,ijab->abcijk", L.b, H.aa.oovv)
    # < 0 | L2 * H(2) | ijk~abc~ >
    L3B += ccpy_einsum("bcjk,ia->abcijk", L.ab, H.a.ov)
    L3B += 0.25 * ccpy_einsum("abij,kc->abcijk", L.aa, H.b.ov)
    L3B += 0.5 * ccpy_einsum("ekbc,aeij->abcijk", H.ab.vovv, L.aa)
    L3B -= 0.5 * ccpy_einsum("jkmc,abim->abcijk", H.ab.ooov, L.aa)
    L3B += ccpy_einsum("ieac,bejk->abcijk", H.ab.ovvv, L.ab)
    L3B -= ccpy_einsum("ikam,bcjm->abcijk", H.ab.oovo, L.ab)
    L3B += 0.5 * ccpy_einsum("eiba,ecjk->abcijk", H.aa.vovv, L.ab)
    L3B -= 0.5 * ccpy_einsum("jima,bcmk->abcijk", H.aa.ooov, L.ab)

    L3B -= np.transpose(L3B, (1, 0, 2, 3, 4, 5))
    L3B -= np.transpose(L3B, (0, 1, 2, 4, 3, 5))
    return L3B

def build_LH_3C(L, H):
    # < 0 | L1 * H(2) | ijk~abc~ >
    L3C = ccpy_einsum("ai,kjcb->cbakji", L.b, H.ab.oovv)
    L3C += 0.25 * ccpy_einsum("ck,ijab->cbakji", L.a, H.bb.oovv)
    # < 0 | L2 * H(2) | ijk~abc~ >
    L3C += ccpy_einsum("cbkj,ia->cbakji", L.ab, H.b.ov)
    L3C += 0.25 * ccpy_einsum("abij,kc->cbakji", L.bb, H.a.ov)
    L3C += 0.5 * ccpy_einsum("kecb,aeij->cbakji", H.ab.ovvv, L.bb)
    L3C -= 0.5 * ccpy_einsum("kjcm,abim->cbakji", H.ab.oovo, L.bb)
    L3C += ccpy_einsum("eica,ebkj->cbakji", H.ab.vovv, L.ab)
    L3C -= ccpy_einsum("kima,cbmj->cbakji", H.ab.ooov, L.ab)
    L3C += 0.5 * ccpy_einsum("eiba,cekj->cbakji", H.bb.vovv, L.ab)
    L3C -= 0.5 * ccpy_einsum("jima,cbkm->cbakji", H.bb.ooov, L.ab)

    L3C -= np.transpose(L3C, (0, 2, 1, 3, 4, 5))
    L3C -= np.transpose(L3C, (0, 1, 2, 3, 5, 4))
    return L3C

def build_LH_3D(L, H):
    # < 0 | L1 * H(2) | ijkabc >
    L3D = (9.0 / 36.0) * ccpy_einsum("ai,jkbc->abcijk", L.b, H.bb.oovv)
    # < 0 | L2 * H(2) | ijkabc >
    L3D += (9.0 / 36.0) * ccpy_einsum("bcjk,ia->abcijk", L.bb, H.b.ov)
    L3D += (9.0 / 36.0) * ccpy_einsum("ebij,ekac->abcijk", L.bb, H.bb.vovv)
    L3D -= (9.0 / 36.0) * ccpy_einsum("abmj,ikmc->abcijk", L.bb, H.bb.ooov)
    L3D -= np.transpose(L3D, (0, 1, 2, 3, 5, 4)) # (jk)
    L3D -= np.transpose(L3D, (0, 1, 2, 4, 3, 5)) + np.transpose(L3D, (0, 1, 2, 5, 4, 3)) # (i/jk)
    L3D -= np.transpose(L3D, (0, 2, 1, 3, 4, 5)) # (bc)
    L3D -= np.transpose(L3D, (2, 1, 0, 3, 4, 5)) + np.transpose(L3D, (1, 0, 2, 3, 4, 5)) # (a/bc)
    return L3D

def get_eomccsdta_intermediates(H, R):
    """Calculate the CCSD-like intermediates for CCSDT. This routine
    should only calculate terms with T2 and any remaining terms outside of the CCS intermediate
    routine."""

    X = {"aa": {}, "ab": {}, "bb": {}}

    X["aa"]["vvov"] =(
        ccpy_einsum("amje,bm->baje", H.aa.voov, R.a)
        + 0.5 * ccpy_einsum("abfe,ej->bajf", H.aa.vvvv, R.a)
    )
    X["aa"]["vvov"] -= np.transpose(X["aa"]["vvov"], (1, 0, 2, 3))

    X["aa"]["vooo"] = (
        -ccpy_einsum("bmie,ej->bmji", H.aa.voov, R.a)
        - 0.5 * ccpy_einsum("nmij,bm->bnji", H.aa.oooo, R.a)
    )
    X["aa"]["vooo"] -= np.transpose(X["aa"]["vooo"], (0, 1, 3, 2))

    X["ab"]["vvvo"] = (
        - ccpy_einsum("mcek,bm->bcek", H.ab.ovvo, R.a)
        - ccpy_einsum("bmek,cm->bcek", H.ab.vovo, R.b)
        + ccpy_einsum("bcfe,ek->bcfk", H.ab.vvvv, R.b)
    )

    X["ab"]["ovoo"] = (
        - ccpy_einsum("nmjk,cm->ncjk", H.ab.oooo, R.b)
        + ccpy_einsum("mcje,ek->mcjk", H.ab.ovov, R.b)
        + ccpy_einsum("mcek,ej->mcjk", H.ab.ovvo, R.a)
    )

    X["ab"]["vvov"] = (
        - ccpy_einsum("mcje,bm->bcje", H.ab.ovov, R.a)
        - ccpy_einsum("bmje,cm->bcje", H.ab.voov, R.b)
        + ccpy_einsum("bcef,ej->bcjf", H.ab.vvvv, R.a)
    )

    X["ab"]["vooo"] = (
        - ccpy_einsum("mnjk,bm->bnjk", H.ab.oooo, R.a)
        + ccpy_einsum("bmje,ek->bmjk", H.ab.voov, R.b)
        + ccpy_einsum("bmek,ej->bmjk", H.ab.vovo, R.a)
    )

    X["bb"]["vvov"] = (
        ccpy_einsum("amje,bm->baje", H.bb.voov, R.b)
        + 0.5 * ccpy_einsum("abfe,ej->bajf", H.bb.vvvv, R.b)
    )
    X["bb"]["vvov"] -= np.transpose(X["bb"]["vvov"], (1, 0, 2, 3))

    X["bb"]["vooo"] = (
        -0.5 * ccpy_einsum("nmij,bm->bnji", H.bb.oooo, R.b)
        - ccpy_einsum("bmie,ej->bmji", H.bb.voov, R.b)
    )
    X["bb"]["vooo"] -= np.transpose(X["bb"]["vooo"], (0, 1, 3, 2))
    return X
