import time
import numpy as np
from ccpy.constants.constants import hartreetoeV

def calc_eaeomccsdta_star(T, R, L, omega, corr_energy, H, H0, system, use_RHF=False):
    """

    """
    t_start = time.perf_counter()
    t_cpu_start = time.process_time()

    n = np.newaxis
    eps_o_a = np.diagonal(H0.a.oo); eps_v_a = np.diagonal(H0.a.vv);
    eps_o_b = np.diagonal(H0.b.oo); eps_v_b = np.diagonal(H0.b.vv);
    e3_aaa = (eps_v_a[:, n, n, n, n] + eps_v_a[n, :, n, n, n] + eps_v_a[n, n, :, n, n] - eps_o_a[n, n, n, :, n] - eps_o_a[n, n, n, n, :])
    e3_aab = (eps_v_a[:, n, n, n, n] + eps_v_a[n, :, n, n, n] + eps_v_b[n, n, :, n, n] - eps_o_a[n, n, n, :, n] - eps_o_b[n, n, n, n, :])
    e3_abb = (eps_v_a[:, n, n, n, n] + eps_v_b[n, :, n, n, n] + eps_v_b[n, n, :, n, n] - eps_o_b[n, n, n, :, n] - eps_o_b[n, n, n, n, :])

    X = get_eaeomccsdta_intermediates(H0, R)

    # update R3
    M3A = build_HR_3A(R, T, H0, X)
    L3A = build_LH_3A(L, H0)
    L3A /= (omega - e3_aaa)
    dA_aaa = (1.0 / 12.0) * np.einsum("abcjk,abcjk->", L3A, M3A, optimize=True)

    M3B = build_HR_3B(R, T, H0, X)
    L3B = build_LH_3B(L, H0)
    L3B /= (omega - e3_aab)
    dA_aab = (1.0 / 2.0) * np.einsum("abcjk,abcjk->", L3B, M3B, optimize=True)

    M3C = build_HR_3C(R, T, H0, X)
    L3C = build_LH_3C(L, H0)
    L3C /= (omega - e3_abb)
    dA_abb = (1.0 / 4.0) * np.einsum("abcjk,abcjk->", L3C, M3C, optimize=True)

    correction_A = dA_aaa + dA_aab + dA_abb

    t_end = time.perf_counter()
    t_cpu_end = time.process_time()
    minutes, seconds = divmod(t_end - t_start, 60)

    energy_A = corr_energy + omega + correction_A
    total_energy_A = system.reference_energy + energy_A

    print("")
    print('   EA-EOMCCSDT(a)* Calculation Summary')
    print('   -------------------------------------------------')
    print("   Total wall time: {:0.2f}m  {:0.2f}s".format(minutes, seconds))
    print(f"   Total CPU time: {t_cpu_end - t_cpu_start} seconds")
    print("   EA-EOMCCSD(T)(a) = {:>10.10f}    ω = {:>10.10f}     VEE = {:>10.5f} eV".format(system.reference_energy + corr_energy + omega, omega, hartreetoeV * omega))
    print(
        "   EA-EOMCCSDT(a)* = {:>10.10f}     δ_A = {:>10.10f}     VEE = {:>10.5f} eV".format(
            total_energy_A, correction_A, (omega + correction_A) * hartreetoeV
        )
    )
    Ecrcc23 = {"A": total_energy_A, "B": 0.0, "C": 0.0, "D": 0.0}
    delta23 = {"A": correction_A, "B": 0.0, "C": 0.0, "D": 0.0}

    return Ecrcc23, delta23

def build_HR_3A(R, T, H, X):
    # moment-like terms
    X3A = -(3.0 / 12.0) * np.einsum("cmkj,abm->abcjk", H.aa.vooo, R.aa, optimize=True)     # (7)
    X3A += (6.0 / 12.0) * np.einsum("cbke,aej->abcjk", H.aa.vvov, R.aa, optimize=True)     # (8)
    # 3-body Hbar terms factorized using intermediates
    X3A -= (6.0 / 12.0) * np.einsum("amj,bcmk->abcjk", X["aa"]["voo"], T.aa, optimize=True) # (9)
    X3A += (3.0 / 12.0) * np.einsum("abe,ecjk->abcjk", X["aa"]["vvv"], T.aa, optimize=True) # (10)
    X3A -= np.transpose(X3A, (1, 0, 2, 3, 4)) + np.transpose(X3A, (2, 1, 0, 3, 4)) # antisymmetrize A(a/bc)
    X3A -= np.transpose(X3A, (0, 2, 1, 3, 4)) # antisymmetrize A(bc)
    X3A -= np.transpose(X3A, (0, 1, 2, 4, 3)) # antisymmetrize A(jk)
    return X3A

def build_HR_3B(R, T, H, X):
    # moment-like terms
    X3B = -(1.0 / 2.0) * np.einsum("mcjk,abm->abcjk", H.ab.ovoo, R.aa, optimize=True) # (14)
    X3B -= np.einsum("bmjk,acm->abcjk", H.ab.vooo, R.ab, optimize=True) # (15)
    X3B += np.einsum("bcek,aej->abcjk", H.ab.vvvo, R.aa, optimize=True) # (16)
    X3B += np.einsum("bcje,aek->abcjk", H.ab.vvov, R.ab, optimize=True) # (17)
    X3B += (1.0 / 2.0) * np.einsum("baje,eck->abcjk", H.aa.vvov, R.ab, optimize=True) # (23)
    # 3-body Hbar terms factorized using intermediates
    X3B -= (1.0 / 2.0) * np.einsum("mck,abmj->abcjk", X["ab"]["ovo"], T.aa, optimize=True) # (18)
    X3B -= np.einsum("amj,bcmk->abcjk", X["aa"]["voo"], T.ab, optimize=True) # (19)
    X3B -= np.einsum("amk,bcjm->abcjk", X["ab"]["voo"], T.ab, optimize=True) # (20)
    X3B += (1.0 / 2.0) * np.einsum("abe,ecjk->abcjk", X["aa"]["vvv"], T.ab, optimize=True) # (21)
    X3B += np.einsum("ace,bejk->abcjk", X["ab"]["vvv"], T.ab, optimize=True) # (22)
    X3B -= np.transpose(X3B, (1, 0, 2, 3, 4)) # antisymmetrize A(ab)
    return X3B

def build_HR_3C(R, T, H, X):
    # moment-like terms
    X3C = -(2.0 / 4.0) * np.einsum("cmkj,abm->abcjk", H.bb.vooo, R.ab, optimize=True) # (10)
    X3C += (2.0 / 4.0) * np.einsum("cbke,aej->abcjk", H.bb.vvov, R.ab, optimize=True) # (11)
    X3C += np.einsum("acek,ebj->abcjk", H.ab.vvvo, R.ab, optimize=True) # (12)
    # 3-body Hbar terms factorized using intermediates
    X3C -= (2.0 / 4.0) * np.einsum("amj,bcmk->abcjk", X["ab"]["voo"], T.bb, optimize=True) # (13)
    X3C -= np.einsum("mck,abmj->abcjk", X["ab"]["ovo"], T.ab, optimize=True) # (14)
    X3C += (2.0 / 4.0) * np.einsum("abe,ecjk->abcjk", X["ab"]["vvv"], T.bb, optimize=True) # (15)
    X3C -= np.transpose(X3C, (0, 2, 1, 3, 4)) # antisymmetrize A(bc)
    X3C -= np.transpose(X3C, (0, 1, 2, 4, 3)) # antisymmetrize A(jk)
    return X3C

def build_LH_3A(L, H):
    # moment-like terms < 0 | (L1p+L2p1h)*(H_N e^(T1+T2))_C | jkabc >
    L3A = (3.0 / 12.0) * np.einsum("a,jkbc->abcjk", L.a, H.aa.oovv, optimize=True)
    L3A += (6.0 / 12.0) * np.einsum("abj,kc->abcjk", L.aa, H.a.ov, optimize=True)
    L3A -= (3.0 / 12.0) * np.einsum("abm,jkmc->abcjk", L.aa, H.aa.ooov, optimize=True)
    L3A += (6.0 / 12.0) * np.einsum("eck,ejab->abcjk", L.aa, H.aa.vovv, optimize=True)
    L3A -= np.transpose(L3A, (1, 0, 2, 3, 4)) + np.transpose(L3A, (2, 1, 0, 3, 4)) # antisymmetrize A(a/bc)
    L3A -= np.transpose(L3A, (0, 2, 1, 3, 4)) # antisymmetrize A(bc)
    L3A -= np.transpose(L3A, (0, 1, 2, 4, 3)) # antisymmetrize A(jk)
    return L3A

def build_LH_3B(L, H):
    # moment-like terms < 0 | (L1p+L2p1h)*(H_N e^(T1+T2))_C | jk~abc~ >
    L3B = np.einsum("a,jkbc->abcjk", L.a, H.ab.oovv, optimize=True)
    L3B += (1.0 / 2.0) * np.einsum("abj,kc->abcjk", L.aa, H.b.ov, optimize=True)
    L3B += np.einsum("ack,jb->abcjk", L.ab, H.a.ov, optimize=True)
    L3B -= (1.0 / 2.0) * np.einsum("abm,jkmc->abcjk", L.aa, H.ab.ooov, optimize=True)
    L3B -= np.einsum("acm,jkbm->abcjk", L.ab, H.ab.oovo, optimize=True)
    L3B += np.einsum("aej,ekbc->abcjk", L.aa, H.ab.vovv, optimize=True)
    L3B += np.einsum("aek,jebc->abcjk", L.ab, H.ab.ovvv, optimize=True)
    L3B += (1.0 / 2.0) * np.einsum("eck,ejab->abcjk", L.ab, H.aa.vovv, optimize=True)
    L3B -= np.transpose(L3B, (1, 0, 2, 3, 4)) # antisymmetrize A(ab)
    return L3B

def build_LH_3C(L, H):
    # moment-like terms < 0 | (L1p+L2p1h)*(H_N e^(T1+T2))_C | j~k~ab~c~ >
    L3C = (1.0 / 4.0) * np.einsum("a,jkbc->abcjk", L.a, H.bb.oovv, optimize=True)
    L3C += np.einsum("abj,kc->abcjk", L.ab, H.b.ov, optimize=True)
    L3C -= (2.0 / 4.0) * np.einsum("abm,jkmc->abcjk", L.ab, H.bb.ooov, optimize=True)
    L3C += (2.0 / 4.0) * np.einsum("aej,ekbc->abcjk", L.ab, H.bb.vovv, optimize=True)
    L3C += np.einsum("eck,ejab->abcjk", L.ab, H.ab.vovv, optimize=True) # !
    L3C -= np.transpose(L3C, (0, 2, 1, 3, 4)) # antisymmetrize A(bc)
    L3C -= np.transpose(L3C, (0, 1, 2, 4, 3)) # antisymmetrize A(jk)
    return L3C

def get_eaeomccsdta_intermediates(H, R):
    X = {"aa" : {}, "ab" : {}}

    # x2a(amj)
    X["aa"]["voo"] = (
                    -np.einsum("amje,e->amj", H.aa.voov, R.a, optimize=True) # CAREFUL: this is a minus sign
    )
    # x2b(mb~j~)
    X["ab"]["ovo"] = (
                    +np.einsum("mbfj,f->mbj", H.ab.ovvo, R.a, optimize=True)
    )
    # x2b(am~j~)
    X["ab"]["voo"] = (
                    +np.einsum("amej,e->amj", H.ab.vovo, R.a, optimize=True)
    )

    # x2a(abe)
    X["aa"]["vvv"] = (
                    +0.5*np.einsum("abfe,f->abe", H.aa.vvvv, R.a, optimize=True)
    )
    X["aa"]["vvv"] -= np.transpose(X["aa"]["vvv"], (1, 0, 2))
    # x2b(ab~e~)
    X["ab"]["vvv"] = (
                    +np.einsum("abfe,f->abe", H.ab.vvvv, R.a, optimize=True)
    )
    return X
