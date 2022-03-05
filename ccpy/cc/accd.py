"""Module with functions that perform the approximate CC with doubles
(ACCD) calculation for a molecular system."""
import numpy as np
import time

from ccpy.utilities.updates import cc_loops2

def update_t2a_v2(cc_t, ints, sys, shift):

    vA = ints["vA"]
    vB = ints["vB"]
    vC = ints["vC"]
    fA = ints["fA"]
    ints["fB"]
    t2a = cc_t["t2a"]
    t2b = cc_t["t2b"]
    cc_t["t2c"]

    # < ijab | (F T2)_C | 0 >
    D1 = -np.einsum("mi,abmj->abij", fA["oo"], t2a, optimize=True)  # A(ij)
    D2 = np.einsum("ae,ebij->abij", fA["vv"], t2a, optimize=True)  # A(ab)

    # < ijab | (V T2)_C | 0 >
    D3 = np.einsum("amie,ebmj->abij", vA["voov"], t2a, optimize=True)  # A(ab)A(ij)
    D4 = np.einsum("amie,bejm->abij", vB["voov"], t2b, optimize=True)  # A(ab)A(ij)
    D5 = 0.5 * np.einsum("mnij,abmn->abij", vA["oooo"], t2a, optimize=True)  # 1
    D6 = 0.5 * np.einsum("abef,efij->abij", vA["vvvv"], t2a, optimize=True)  # 1

    # < ijab | (V T2**2)_C | 0 >
    D7 = np.einsum(
        "mnef,aeim,bfjn->abij", vA["oovv"], t2a, t2a, optimize=True
    )  # A(ij), D1+D2
    D8 = np.einsum(
        "mnef,aeim,bfjn->abij", vB["oovv"], t2a, t2b, optimize=True
    )  # A(ij)A(ab), D1
    D9 = np.einsum(
        "mnef,aeim,bfjn->abij", vC["oovv"], t2b, t2b, optimize=True
    )  # A(ij), D1+D2
    D10 = 0.25 * np.einsum(
        "mnef,efij,abmn->abij", vA["oovv"], t2a, t2a, optimize=True
    )  # 1, D5
    D11 = -0.5 * np.einsum(
        "mnef,abim,efjn->abij", vA["oovv"], t2a, t2a, optimize=True
    )  # A(ij),
    D12 = -np.einsum(
        "mnef,abim,efjn->abij", vB["oovv"], t2a, t2b, optimize=True
    )  # A(ij)
    D13 = -0.5 * np.einsum(
        "mnef,aeij,bfmn->abij", vA["oovv"], t2a, t2a, optimize=True
    )  # A(ab)
    D14 = -np.einsum(
        "mnef,aeij,bfmn->abij", vB["oovv"], t2a, t2b, optimize=True
    )  # A(ab)

    X2A = (
        0.5 * (D1 + D2 + D7 + D9 + D11 + D12 + D13 + D14)
        + 0.25 * (D5 + D6 + D10 + vA["vvoo"])
        + (D3 + D4 + D8)
    )

    for a in range(sys["Nunocc_a"]):
        for b in range(a + 1, sys["Nunocc_a"]):
            for i in range(sys["Nocc_a"]):
                for j in range(i + 1, sys["Nocc_a"]):
                    denom = (
                        fA["oo"][i, i]
                        + fA["oo"][j, j]
                        - fA["vv"][a, a]
                        - fA["vv"][b, b]
                    )
                    t2a[a, b, i, j] += (
                        X2A[a, b, i, j]
                        - X2A[b, a, i, j]
                        - X2A[a, b, j, i]
                        + X2A[b, a, j, i]
                    ) / (denom - shift)
                    t2a[b, a, i, j] = -t2a[a, b, i, j]
                    t2a[a, b, j, i] = -t2a[a, b, i, j]
                    t2a[b, a, j, i] = t2a[a, b, i, j]
    cc_t["t2a"] = t2a

    return cc_t


def update_t2a(cc_t, ints, sys, shift):

    vA = ints["vA"]
    vB = ints["vB"]
    vC = ints["vC"]
    fA = ints["fA"]
    ints["fB"]
    t2a = cc_t["t2a"]
    t2b = cc_t["t2b"]
    cc_t["t2c"]

    # intermediates
    I1A_oo = 0.0
    I1A_oo += 0.5 * np.einsum("mnef,efin->mi", vA["oovv"], t2a, optimize=True)
    I1A_oo += np.einsum("mnef,efin->mi", vB["oovv"], t2b, optimize=True)
    I1A_oo += fA["oo"]

    I1A_vv = 0.0
    I1A_vv -= 0.5 * np.einsum("mnef,afmn->ae", vA["oovv"], t2a, optimize=True)
    I1A_vv -= np.einsum("mnef,afmn->ae", vB["oovv"], t2b, optimize=True)
    I1A_vv += fA["vv"]

    I2A_voov = 0.0
    I2A_voov += 0.5 * np.einsum("mnef,afin->amie", vA["oovv"], t2a, optimize=True)
    I2A_voov += np.einsum("mnef,afin->amie", vB["oovv"], t2b, optimize=True)
    I2A_voov += vA["voov"]

    I2A_oooo = 0.0
    I2A_oooo += 0.5 * np.einsum("mnef,efij->mnij", vA["oovv"], t2a, optimize=True)
    I2A_oooo += vA["oooo"]

    I2B_voov = 0.0
    I2B_voov += 0.5 * np.einsum("mnef,afin->amie", vC["oovv"], t2b, optimize=True)
    I2B_voov += vB["voov"]

    X2A = 0.0
    X2A += vA["vvoo"]
    D3 = np.einsum("ae,ebij->abij", I1A_vv, t2a, optimize=True)
    D4 = -np.einsum("mi,abmj->abij", I1A_oo, t2a, optimize=True)
    D5 = np.einsum("amie,ebmj->abij", I2A_voov, t2a, optimize=True)
    D6 = np.einsum("amie,bejm->abij", I2B_voov, t2b, optimize=True)
    X2A += 0.5 * np.einsum("abef,efij->abij", vA["vvvv"], t2a, optimize=True)
    X2A += 0.5 * np.einsum("mnij,abmn->abij", I2A_oooo, t2a, optimize=True)

    # diagrams that have A(ab)
    D13 = D3
    D13 = D13 - np.einsum("abij->baij", D13)

    # diagrams that have A(ij)
    D24 = D4
    D24 = D24 - np.einsum("abij->abji", D24)

    # diagrams that have A(ab)A(ij)
    D56 = D5 + D6
    D56 = (
        D56
        - np.einsum("abij->baij", D56)
        - np.einsum("abij->abji", D56)
        + np.einsum("abij->baji", D56)
    )

    # total contribution
    X2A += D13 + D24 + D56

    t2a = cc_loops.cc_loops.update_t2a(t2a, X2A, fA["oo"], fA["vv"], shift)

    cc_t["t2a"] = t2a
    return cc_t


def update_t2b_v2(cc_t, ints, sys, shift):

    vA = ints["vA"]
    vB = ints["vB"]
    vC = ints["vC"]
    fA = ints["fA"]
    fB = ints["fB"]
    t2a = cc_t["t2a"]
    t2b = cc_t["t2b"]
    t2c = cc_t["t2c"]

    # < ijab | (F T2)_C | 0 >
    D1 = -np.einsum("mi,abmj->abij", fA["oo"], t2b, optimize=True)
    D2 = np.einsum("ae,ebij->abij", fA["vv"], t2b, optimize=True)
    D3 = -np.einsum("mj,abim->abij", fB["oo"], t2b, optimize=True)
    D4 = np.einsum("be,aeij->abij", fB["vv"], t2b, optimize=True)

    # < ijab | (V T2)_C | 0 >
    D5 = np.einsum("amie,ebmj->abij", vA["voov"], t2b, optimize=True)
    D6 = np.einsum("amie,ebmj->abij", vB["voov"], t2c, optimize=True)
    D7 = np.einsum("mbej,aeim->abij", vB["ovvo"], t2a, optimize=True)
    D8 = np.einsum("bmje,aeim->abij", vC["voov"], t2b, optimize=True)
    D9 = -np.einsum("mbie,aemj->abij", vB["ovov"], t2b, optimize=True)
    D10 = -np.einsum("amej,ebim->abij", vB["vovo"], t2b, optimize=True)
    D11 = np.einsum("mnij,abmn->abij", vB["oooo"], t2b, optimize=True)
    D12 = np.einsum("abef,efij->abij", vB["vvvv"], t2b, optimize=True)

    # < ijab | (V T2**2)_C | 0 >
    D13 = np.einsum("mnef,aeim,fbnj->abij", vA["oovv"], t2a, t2b, optimize=True)
    D14 = np.einsum("mnef,aeim,fbnj->abij", vB["oovv"], t2a, t2c, optimize=True)
    D15 = np.einsum("nmfe,aeim,fbnj->abij", vB["oovv"], t2b, t2b, optimize=True)
    D16 = np.einsum("mnef,aeim,fbnj->abij", vC["oovv"], t2b, t2c, optimize=True)
    D17 = np.einsum("mnef,ebin,afmj->abij", vB["oovv"], t2b, t2b, optimize=True)
    D18 = np.einsum("mnef,efij,abmn->abij", vB["oovv"], t2b, t2b, optimize=True)

    D19 = -0.5 * np.einsum("mnef,efin,abmj->abij", vA["oovv"], t2a, t2b, optimize=True)
    D20 = -np.einsum("mnef,efin,abmj->abij", vB["oovv"], t2b, t2b, optimize=True)
    D21 = -np.einsum("nmfe,fenj,abim->abij", vB["oovv"], t2b, t2b, optimize=True)
    D22 = -0.5 * np.einsum("mnef,efjn,abim->abij", vC["oovv"], t2c, t2b, optimize=True)
    D23 = -0.5 * np.einsum("mnef,afmn,ebij->abij", vA["oovv"], t2a, t2b, optimize=True)
    D24 = -np.einsum("mnef,afmn,ebij->abij", vB["oovv"], t2b, t2b, optimize=True)
    D25 = -np.einsum("nmfe,fbnm,aeij->abij", vB["oovv"], t2b, t2b, optimize=True)
    D26 = -0.5 * np.einsum("mnef,bfmn,aeij->abij", vC["oovv"], t2c, t2b, optimize=True)

    X2B = (
        D1
        + D2
        + D3
        + D4
        + D5
        + D6
        + D7
        + D8
        + D9
        + D10
        + D11
        + D12
        + D13
        + D14
        + D15
        + D16
        + D17
        + D18
        + D19
        + D20
        + D21
        + D22
        + D23
        + D24
        + D25
        + D26
        + vB["vvoo"]
    )

    for a in range(sys["Nunocc_a"]):
        for b in range(sys["Nunocc_b"]):
            for i in range(sys["Nocc_a"]):
                for j in range(sys["Nocc_b"]):
                    denom = (
                        fA["oo"][i, i]
                        + fB["oo"][j, j]
                        - fA["vv"][a, a]
                        - fB["vv"][b, b]
                    )
                    t2b[a, b, i, j] += X2B[a, b, i, j] / (denom - shift)
    cc_t["t2b"] = t2b

    return cc_t


def update_t2b(cc_t, ints, sys, shift):

    vA = ints["vA"]
    vB = ints["vB"]
    vC = ints["vC"]
    fA = ints["fA"]
    fB = ints["fB"]
    t2a = cc_t["t2a"]
    t2b = cc_t["t2b"]
    t2c = cc_t["t2c"]

    # intermediates
    I1A_vv = 0.0
    I1A_vv -= 0.5 * np.einsum("mnef,afmn->ae", vA["oovv"], t2a, optimize=True)
    I1A_vv -= np.einsum("mnef,afmn->ae", vB["oovv"], t2b, optimize=True)
    I1A_vv += fA["vv"]

    I1B_vv = 0.0
    I1B_vv -= np.einsum("nmfe,fbnm->be", vB["oovv"], t2b, optimize=True)
    I1B_vv -= 0.5 * np.einsum("mnef,fbnm->be", vC["oovv"], t2c, optimize=True)
    I1B_vv += fB["vv"]

    I1A_oo = 0.0
    I1A_oo += 0.5 * np.einsum("mnef,efin->mi", vA["oovv"], t2a, optimize=True)
    I1A_oo += np.einsum("mnef,efin->mi", vB["oovv"], t2b, optimize=True)
    I1A_oo += fA["oo"]

    I1B_oo = 0.0
    I1B_oo += np.einsum("nmfe,fenj->mj", vB["oovv"], t2b, optimize=True)
    I1B_oo += 0.5 * np.einsum("mnef,efjn->mj", vC["oovv"], t2c, optimize=True)
    I1B_oo += fB["oo"]

    I2A_voov = 0.0
    I2A_voov += np.einsum("mnef,aeim->anif", vA["oovv"], t2a, optimize=True)
    I2A_voov += np.einsum("nmfe,aeim->anif", vB["oovv"], t2b, optimize=True)
    I2A_voov += vA["voov"]

    I2B_voov = 0.0
    I2B_voov += np.einsum("mnef,aeim->anif", vB["oovv"], t2a, optimize=True)
    I2B_voov += np.einsum("mnef,aeim->anif", vC["oovv"], t2b, optimize=True)
    I2B_voov += vB["voov"]

    I2B_oooo = 0.0
    I2B_oooo += np.einsum("mnef,efij->mnij", vB["oovv"], t2b, optimize=True)
    I2B_oooo += vB["oooo"]

    I2B_vovo = 0.0
    I2B_vovo -= np.einsum("mnef,afmj->anej", vB["oovv"], t2b, optimize=True)
    I2B_vovo += vB["vovo"]

    X2B = 0.0
    X2B += vB["vvoo"]
    X2B += np.einsum("ae,ebij->abij", I1A_vv, t2b, optimize=True)
    X2B += np.einsum("be,aeij->abij", I1B_vv, t2b, optimize=True)
    X2B -= np.einsum("mi,abmj->abij", I1A_oo, t2b, optimize=True)
    X2B -= np.einsum("mj,abim->abij", I1B_oo, t2b, optimize=True)
    X2B += np.einsum("amie,ebmj->abij", I2A_voov, t2b, optimize=True)
    X2B += np.einsum("amie,ebmj->abij", I2B_voov, t2c, optimize=True)
    X2B += np.einsum("mbej,aeim->abij", vB["ovvo"], t2a, optimize=True)
    X2B += np.einsum("bmje,aeim->abij", vC["voov"], t2b, optimize=True)
    X2B -= np.einsum("mbie,aemj->abij", vB["ovov"], t2b, optimize=True)
    X2B -= np.einsum("amej,ebim->abij", I2B_vovo, t2b, optimize=True)
    X2B += np.einsum("mnij,abmn->abij", I2B_oooo, t2b, optimize=True)
    X2B += np.einsum("abef,efij->abij", vB["vvvv"], t2b, optimize=True)

    t2b = cc_loops.cc_loops.update_t2b(
        t2b, X2B, fA["oo"], fA["vv"], fB["oo"], fB["vv"], shift
    )

    cc_t["t2b"] = t2b
    return cc_t


def update_t2c(cc_t, ints, sys, shift):

    vA = ints["vA"]
    vB = ints["vB"]
    vC = ints["vC"]
    ints["fA"]
    fB = ints["fB"]
    cc_t["t2a"]
    t2b = cc_t["t2b"]
    t2c = cc_t["t2c"]

    I1B_oo = 0.0
    I1B_oo += 0.5 * np.einsum("mnef,efin->mi", vC["oovv"], t2c, optimize=True)
    I1B_oo += np.einsum("nmfe,feni->mi", vB["oovv"], t2b, optimize=True)
    I1B_oo += fB["oo"]

    I1B_vv = 0.0
    I1B_vv -= 0.5 * np.einsum("mnef,afmn->ae", vC["oovv"], t2c, optimize=True)
    I1B_vv -= np.einsum("nmfe,fanm->ae", vB["oovv"], t2b, optimize=True)
    I1B_vv += fB["vv"]

    I2C_oooo = 0.0
    I2C_oooo += 0.5 * np.einsum("mnef,efij->mnij", vC["oovv"], t2c, optimize=True)
    I2C_oooo += vC["oooo"]

    I2B_ovvo = 0.0
    I2B_ovvo += np.einsum("mnef,afin->maei", vB["oovv"], t2c, optimize=True)
    I2B_ovvo += 0.5 * np.einsum("mnef,fani->maei", vA["oovv"], t2b, optimize=True)
    I2B_ovvo += vB["ovvo"]

    I2C_voov = 0.0
    I2C_voov += 0.5 * np.einsum("mnef,afin->amie", vC["oovv"], t2c, optimize=True)
    I2C_voov += vC["voov"]

    X2C = 0.0
    X2C += vC["vvoo"]
    D3 = np.einsum("ae,ebij->abij", I1B_vv, t2c, optimize=True)
    D4 = -np.einsum("mi,abmj->abij", I1B_oo, t2c, optimize=True)
    D5 = np.einsum("amie,ebmj->abij", I2C_voov, t2c, optimize=True)
    D6 = np.einsum("maei,ebmj->abij", I2B_ovvo, t2b, optimize=True)
    X2C += 0.5 * np.einsum("abef,efij->abij", vC["vvvv"], t2c, optimize=True)
    X2C += 0.5 * np.einsum("mnij,abmn->abij", I2C_oooo, t2c, optimize=True)

    # diagrams that have A(ab)
    D13 = D3
    D13 = D13 - np.einsum("abij->baij", D13)

    # diagrams that have A(ij)
    D24 = D4
    D24 = D24 - np.einsum("abij->abji", D24)

    # diagrams that have A(ab)A(ij)
    D56 = D5 + D6
    D56 = (
        D56
        - np.einsum("abij->baij", D56)
        - np.einsum("abij->abji", D56)
        + np.einsum("abij->baji", D56)
    )

    # total contribution
    X2C += D13 + D24 + D56

    t2c = cc_loops.cc_loops.update_t2c(t2c, X2C, fB["oo"], fB["vv"], shift)

    cc_t["t2c"] = t2c
    return cc_t


def update_t2c_v2(cc_t, ints, sys, shift):

    vA = ints["vA"]
    vB = ints["vB"]
    vC = ints["vC"]
    ints["fA"]
    fB = ints["fB"]
    cc_t["t2a"]
    t2b = cc_t["t2b"]
    t2c = cc_t["t2c"]

    # < ijab | (F T2)_C | 0 >
    D1 = -np.einsum("mi,abmj->abij", fB["oo"], t2c, optimize=True)  # A(ij)
    D2 = np.einsum("ae,ebij->abij", fB["vv"], t2c, optimize=True)  # A(ab)

    # < ijab | (V T2)_C | 0 >
    D3 = np.einsum("amie,ebmj->abij", vC["voov"], t2c, optimize=True)  # A(ab)A(ij)
    D4 = np.einsum("maei,ebmj->abij", vB["ovvo"], t2b, optimize=True)  # A(ab)A(ij)
    D5 = 0.5 * np.einsum("mnij,abmn->abij", vC["oooo"], t2c, optimize=True)  # 1
    D6 = 0.5 * np.einsum("abef,efij->abij", vC["vvvv"], t2c, optimize=True)  # 1

    # < ijab | (V T2**2)_C | 0 >
    D7 = np.einsum("mnef,aeim,bfjn->abij", vC["oovv"], t2c, t2c, optimize=True)  # A(ij)
    D8 = np.einsum(
        "nmfe,aeim,fbnj->abij", vB["oovv"], t2c, t2b, optimize=True
    )  # A(ij)A(ab)
    D9 = np.einsum("mnef,eami,fbnj->abij", vA["oovv"], t2b, t2b, optimize=True)  # A(ij)
    D10 = 0.25 * np.einsum(
        "mnef,efij,abmn->abij", vC["oovv"], t2c, t2c, optimize=True
    )  # 1
    D11 = -0.5 * np.einsum(
        "mnef,abim,efjn->abij", vC["oovv"], t2c, t2c, optimize=True
    )  # A(ij)
    D12 = -np.einsum(
        "nmfe,abim,fenj->abij", vB["oovv"], t2c, t2b, optimize=True
    )  # A(ij)
    D13 = -0.5 * np.einsum(
        "mnef,aeij,bfmn->abij", vC["oovv"], t2c, t2c, optimize=True
    )  # A(ab)
    D14 = -np.einsum(
        "nmfe,aeij,fbnm->abij", vB["oovv"], t2c, t2b, optimize=True
    )  # A(ab)

    X2C = (
        0.5 * (D1 + D2 + D7 + D9 + D11 + D12 + D13 + D14)
        + 0.25 * (D5 + D6 + D10 + vC["vvoo"])
        + (D3 + D4 + D8)
    )

    for a in range(sys["Nunocc_b"]):
        for b in range(a + 1, sys["Nunocc_b"]):
            for i in range(sys["Nocc_b"]):
                for j in range(i + 1, sys["Nocc_b"]):
                    denom = (
                        fB["oo"][i, i]
                        + fB["oo"][j, j]
                        - fB["vv"][a, a]
                        - fB["vv"][b, b]
                    )
                    t2c[a, b, i, j] += (
                        X2C[a, b, i, j]
                        - X2C[b, a, i, j]
                        - X2C[a, b, j, i]
                        + X2C[b, a, j, i]
                    ) / (denom - shift)
                    t2c[b, a, i, j] = -t2c[a, b, i, j]
                    t2c[a, b, j, i] = -t2c[a, b, i, j]
                    t2c[b, a, j, i] = t2c[a, b, i, j]
    cc_t["t2c"] = t2c

    return cc_t
