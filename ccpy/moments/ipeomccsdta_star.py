import time
import numpy as np
from ccpy.constants.constants import hartreetoeV

def calc_ipeomccsdta_star(T, R, L, omega, corr_energy, H, H0, system, use_RHF=False):
    """

    """
    t_start = time.perf_counter()
    t_cpu_start = time.process_time()

    n = np.newaxis
    eps_o_a = np.diagonal(H0.a.oo); eps_v_a = np.diagonal(H0.a.vv);
    eps_o_b = np.diagonal(H0.b.oo); eps_v_b = np.diagonal(H0.b.vv);
    e3_aaa = (-eps_o_a[:, n, n, n, n] + eps_v_a[n, :, n, n, n] + eps_v_a[n, n, :, n, n] - eps_o_a[n, n, n, :, n] - eps_o_a[n, n, n, n, :])
    e3_aab = (-eps_o_a[:, n, n, n, n] + eps_v_a[n, :, n, n, n] + eps_v_b[n, n, :, n, n] - eps_o_a[n, n, n, :, n] - eps_o_b[n, n, n, n, :])
    e3_abb = (-eps_o_a[:, n, n, n, n] + eps_v_b[n, :, n, n, n] + eps_v_b[n, n, :, n, n] - eps_o_b[n, n, n, :, n] - eps_o_b[n, n, n, n, :])

    X = get_ipeomccsdta_intermediates(H0, R)

    # update R3
    M3A = build_HR_3A(R, T, H0, X)
    L3A = build_LH_3A(L, H0)
    L3A /= (omega - e3_aaa)
    dA_aaa = (1.0 / 12.0) * np.einsum("ibcjk,ibcjk->", L3A, M3A, optimize=True)

    M3B = build_HR_3B(R, T, H0, X)
    L3B = build_LH_3B(L, H0)
    L3B /= (omega - e3_aab)
    dA_aab = (1.0 / 2.0) * np.einsum("ibcjk,ibcjk->", L3B, M3B, optimize=True)

    M3C = build_HR_3C(R, T, H0, X)
    L3C = build_LH_3C(L, H0)
    L3C /= (omega - e3_abb)
    dA_abb = (1.0 / 4.0) * np.einsum("ibcjk,ibcjk->", L3C, M3C, optimize=True)

    correction_A = dA_aaa + dA_aab + dA_abb

    t_end = time.perf_counter()
    t_cpu_end = time.process_time()
    minutes, seconds = divmod(t_end - t_start, 60)

    energy_A = corr_energy + omega + correction_A
    total_energy_A = system.reference_energy + energy_A

    print("")
    print('   IP-EOMCCSDT(a)* Calculation Summary')
    print('   -------------------------------------------------')
    print("   Total wall time: {:0.2f}m  {:0.2f}s".format(minutes, seconds))
    print(f"   Total CPU time: {t_cpu_end - t_cpu_start} seconds")
    print("   IP-EOMCCSD(T)(a) = {:>10.10f}    ω = {:>10.10f}     VEE = {:>10.5f} eV".format(system.reference_energy + corr_energy + omega, omega, hartreetoeV * omega))
    print(
        "   IP-EOMCCSDT(a)* = {:>10.10f}     δ_A = {:>10.10f}     VEE = {:>10.5f} eV".format(
            total_energy_A, correction_A, (omega + correction_A) * hartreetoeV
        )
    )
    Ecrcc23 = {"A": total_energy_A, "B": 0.0, "C": 0.0, "D": 0.0}
    delta23 = {"A": correction_A, "B": 0.0, "C": 0.0, "D": 0.0}

    return Ecrcc23, delta23

def build_HR_3A(R, T, H, X):
    # moment-like terms
    X3A = -(6.0 / 12.0) * np.einsum("cmkj,ibm->ibcjk", H.aa.vooo, R.aa, optimize=True)
    X3A += (3.0 / 12.0) * np.einsum("cbke,iej->ibcjk", H.aa.vvov, R.aa, optimize=True)
    # 3-body Hbar terms factorized using intermediates
    X3A -= (3.0 / 12.0) * np.einsum("imj,bcmk->ibcjk", X["aa"]["ooo"], T.aa, optimize=True)
    X3A += (6.0 / 12.0) * np.einsum("ibe,ecjk->ibcjk", X["aa"]["ovv"], T.aa, optimize=True)
    X3A -= np.transpose(X3A, (3, 1, 2, 0, 4)) + np.transpose(X3A, (4, 1, 2, 3, 0)) # antisymmetrize A(i/jk)
    X3A -= np.transpose(X3A, (0, 1, 2, 4, 3)) # antisymmetrize A(jk)
    X3A -= np.transpose(X3A, (0, 2, 1, 3, 4)) # antisymmetrize A(bc)
    return X3A

def build_HR_3B(R, T, H, X):
    # moment-like terms
    X3B = -np.einsum("mcjk,ibm->ibcjk", H.ab.ovoo, R.aa, optimize=True) # (14)
    X3B -= 0.5 * np.einsum("bmji,mck->ibcjk", H.aa.vooo, R.ab, optimize=True) # (15)
    X3B -= np.einsum("bmjk,icm->ibcjk", H.ab.vooo, R.ab, optimize=True) # (16)
    X3B += np.einsum("bcje,iek->ibcjk", H.ab.vvov, R.ab, optimize=True) # (17)
    X3B += 0.5 * np.einsum("bcek,iej->ibcjk", H.ab.vvvo, R.aa, optimize=True) # (18)
    # 3-body Hbar terms factorized using intermediates
    X3B += 0.5 * np.einsum("eck,ebij->ibcjk", X["ab"]["vvo"], T.aa, optimize=True) # (19)
    X3B -= 0.5 * np.einsum("imj,bcmk->ibcjk", X["aa"]["ooo"], T.ab, optimize=True) # (20)
    X3B -= np.einsum("imk,bcjm->ibcjk", X["ab"]["ooo"], T.ab, optimize=True) # (21)
    X3B += np.einsum("ice,bejk->ibcjk", X["ab"]["ovv"], T.ab, optimize=True) # (22)
    X3B += np.einsum("ibe,ecjk->ibcjk", X["aa"]["ovv"], T.ab, optimize=True) # (23)
    X3B -= np.transpose(X3B, (3, 1, 2, 0, 4)) # antisymmetrize (ij)
    return X3B

def build_HR_3C(R, T, H, X):
    # moment-like terms
    X3C = -np.einsum("mcik,mbj->ibcjk", H.ab.ovoo, R.ab, optimize=True) # (10)
    X3C -= (2.0 / 4.0) * np.einsum("cmkj,ibm->ibcjk", H.bb.vooo, R.ab, optimize=True) # (11)
    X3C += (2.0 / 4.0) * np.einsum("cbke,iej->ibcjk", H.bb.vvov, R.ab, optimize=True) # (12)
    # 3-body Hbar terms factorized using intermediates
    X3C -= (2.0 / 4.0) * np.einsum("imj,bcmk->ibcjk", X["ab"]["ooo"], T.bb, optimize=True) # (13)
    X3C += (2.0 / 4.0) * np.einsum("ibe,ecjk->ibcjk", X["ab"]["ovv"], T.bb, optimize=True) # (14)
    X3C += np.einsum("ebj,ecik->ibcjk", X["ab"]["vvo"], T.ab, optimize=True) # (15)
    X3C -= np.transpose(X3C, (0, 2, 1, 3, 4)) # antisymmetrize A(bc)
    X3C -= np.transpose(X3C, (0, 1, 2, 4, 3)) # antisymmetrize A(jk)
    return X3C

def build_LH_3A(L, H):
    # moment-like terms
    L3A = (3.0 / 12.0) * np.einsum("i,jkbc->ibcjk", L.a, H.aa.oovv, optimize=True)
    # L3A += (6.0 / 12.0) * np.einsum("ibj,kc->ibcjk", L.aa, H.a.ov, optimize=True)
    L3A += (3.0 / 12.0) * np.einsum("iej,ekbc->ibcjk", L.aa, H.aa.vovv, optimize=True)
    L3A -= (6.0 / 12.0) * np.einsum("mck,ijmb->ibcjk", L.aa, H.aa.ooov, optimize=True)
    L3A -= np.transpose(L3A, (3, 1, 2, 0, 4)) + np.transpose(L3A, (4, 1, 2, 3, 0)) # antisymmetrize A(i/jk)
    L3A -= np.transpose(L3A, (0, 1, 2, 4, 3)) # antisymmetrize A(jk)
    L3A -= np.transpose(L3A, (0, 2, 1, 3, 4)) # antisymmetrize A(bc)
    return L3A

def build_LH_3B(L, H):
    # moment-like terms
    L3B = np.einsum("i,jkbc->ibcjk", L.a, H.ab.oovv, optimize=True)
    L3B += (1.0 / 2.0) * np.einsum("ibj,kc->ibcjk", L.aa, H.b.ov, optimize=True)
    L3B += np.einsum("ick,jb->ibcjk", L.ab, H.a.ov, optimize=True)
    L3B += (1.0 / 2.0) * np.einsum("iej,ekbc->ibcjk", L.aa, H.ab.vovv, optimize=True)
    L3B += np.einsum("iek,jebc->ibcjk", L.ab, H.ab.ovvv, optimize=True)
    L3B -= np.einsum("mbj,ikmc->ibcjk", L.aa, H.ab.ooov, optimize=True)
    L3B -= (1.0 / 2.0) * np.einsum("mck,ijmb->ibcjk", L.ab, H.aa.ooov, optimize=True)
    L3B -= np.einsum("icm,jkbm->ibcjk", L.ab, H.ab.oovo, optimize=True)
    L3B -= np.transpose(L3B, (3, 1, 2, 0, 4)) # antisymmetrize (ij)
    return L3B

def build_LH_3C(L, H):
    # moment-like terms
    L3C = (1.0 / 4.0) * np.einsum("i,jkbc->ibcjk", L.a, H.bb.oovv, optimize=True)
    L3C += np.einsum("ibj,kc->ibcjk", L.ab, H.b.ov, optimize=True)
    L3C += (2.0 / 4.0) * np.einsum("iej,ekbc->ibcjk", L.ab, H.bb.vovv, optimize=True)
    L3C -= np.einsum("mck,ijmb->ibcjk", L.ab, H.ab.ooov, optimize=True)
    L3C -= (2.0 / 4.0) * np.einsum("ibm,jkmc->ibcjk", L.ab, H.bb.ooov, optimize=True)
    L3C -= np.transpose(L3C, (0, 2, 1, 3, 4)) # antisymmetrize A(bc)
    L3C -= np.transpose(L3C, (0, 1, 2, 4, 3)) # antisymmetrize A(jk)
    return L3C

def get_ipeomccsdta_intermediates(H, R):

    X = {"aa" : {}, "ab" : {}}

    # x2a(ibe)
    X["aa"]["ovv"] = (
            +np.einsum("bmie,m->ibe", H.aa.voov, R.a, optimize=True)
    )
    # x2b(eb~j~)
    X["ab"]["vvo"] = (
            -np.einsum("mbej,m->ebj", H.ab.ovvo, R.a, optimize=True)
    )
    # x2b(ib~e~)
    X["ab"]["ovv"] = (
            -np.einsum("mbie,m->ibe", H.ab.ovov, R.a, optimize=True)
    )

    # x2a(imj)
    X["aa"]["ooo"] = (
            -0.5 * np.einsum("mnji,n->imj", H.aa.oooo, R.a, optimize=True)
    )
    X["aa"]["ooo"] -= np.transpose(X["aa"]["ooo"], (2, 1, 0))
    # x2b(im~j~)
    X["ab"]["ooo"] = (
            -np.einsum("nmij,n->imj", H.ab.oooo, R.a, optimize=True)
    )
    return X
