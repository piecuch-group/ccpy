"""Module containing functions to calculate the vertical excitation
energies and linear excitation amplitudes for excited states using
the equation-of-motion (EOM) CC with singles and doubles (EOMCCSD)."""
import numpy as np
from ccpy.utilities.updates import cc_loops

def update(dR, omega, H):

    dR.a, dR.b, dR.aa, dR.ab, dR.bb = cc_loops.cc_loops.update_r(
        dR.a,
        dR.b,
        dR.aa,
        dR.ab,
        dR.bb,
        omega,
        H.a.oo,
        H.a.vv,
        H.b.oo,
        H.b.vv,
        0.0
    )
    return dR


def HR(R, T, H, system, flag_RHF):


    if flag_RHF:
        X1A = build_HR_1A(R, T, H, system)
        X2A = build_HR_2A(R, T, H, system)
        X2B = build_HR_2B(R, T, H, system)
    else:
        X1A = build_HR_1A(
            r1a, r1b, r2a, r2b, r2c, cc_t, H1A, H1B, H2A, H2B, H2C, ints, sys
        )
        X1B = build_HR_1B(
            r1a, r1b, r2a, r2b, r2c, cc_t, H1A, H1B, H2A, H2B, H2C, ints, sys
        )
        X2A = build_HR_2A(
            r1a, r1b, r2a, r2b, r2c, cc_t, H1A, H1B, H2A, H2B, H2C, ints, sys
        )
        X2B = build_HR_2B(
            r1a, r1b, r2a, r2b, r2c, cc_t, H1A, H1B, H2A, H2B, H2C, ints, sys
        )
        X2C = build_HR_2C(
            r1a, r1b, r2a, r2b, r2c, cc_t, H1A, H1B, H2A, H2B, H2C, ints, sys
        )
    return np.concatenate( (X1A.flatten(), X1B.flatten(), X2A.flatten(), X2B.flatten(), X2C.flatten()), axis=0)


def build_HR_1A(r1a, r1b, r2a, r2b, r2c, cc_t, H1A, H1B, H2A, H2B, H2C, ints, sys):

    X1A = 0.0
    X1A -= np.einsum("mi,am->ai", H1A["oo"], r1a, optimize=True)
    X1A += np.einsum("ae,ei->ai", H1A["vv"], r1a, optimize=True)
    X1A += np.einsum("amie,em->ai", H2A["voov"], r1a, optimize=True)
    X1A += np.einsum("amie,em->ai", H2B["voov"], r1b, optimize=True)
    X1A -= 0.5 * np.einsum("mnif,afmn->ai", H2A["ooov"], r2a, optimize=True)
    X1A -= np.einsum("mnif,afmn->ai", H2B["ooov"], r2b, optimize=True)
    X1A += 0.5 * np.einsum("anef,efin->ai", H2A["vovv"], r2a, optimize=True)
    X1A += np.einsum("anef,efin->ai", H2B["vovv"], r2b, optimize=True)
    X1A += np.einsum("me,aeim->ai", H1A["ov"], r2a, optimize=True)
    X1A += np.einsum("me,aeim->ai", H1B["ov"], r2b, optimize=True)

    return X1A


def build_HR_1B(r1a, r1b, r2a, r2b, r2c, cc_t, H1A, H1B, H2A, H2B, H2C, ints, sys):

    X1B = 0.0
    X1B -= np.einsum("mi,am->ai", H1B["oo"], r1b, optimize=True)
    X1B += np.einsum("ae,ei->ai", H1B["vv"], r1b, optimize=True)
    X1B += np.einsum("maei,em->ai", H2B["ovvo"], r1a, optimize=True)
    X1B += np.einsum("amie,em->ai", H2C["voov"], r1b, optimize=True)
    X1B -= np.einsum("nmfi,fanm->ai", H2B["oovo"], r2b, optimize=True)
    X1B -= 0.5 * np.einsum("mnif,afmn->ai", H2C["ooov"], r2c, optimize=True)
    X1B += np.einsum("nafe,feni->ai", H2B["ovvv"], r2b, optimize=True)
    X1B += 0.5 * np.einsum("anef,efin->ai", H2C["vovv"], r2c, optimize=True)
    X1B += np.einsum("me,eami->ai", H1A["ov"], r2b, optimize=True)
    X1B += np.einsum("me,aeim->ai", H1B["ov"], r2c, optimize=True)

    return X1B


def build_HR_2A(r1a, r1b, r2a, r2b, r2c, cc_t, H1A, H1B, H2A, H2B, H2C, ints, sys):

    t2a = cc_t["t2a"]
    vA = ints["vA"]
    vB = ints["vB"]

    X2A = 0.0
    D1 = -np.einsum("mi,abmj->abij", H1A["oo"], r2a, optimize=True)  # A(ij)
    D2 = np.einsum("ae,ebij->abij", H1A["vv"], r2a, optimize=True)  # A(ab)
    X2A += 0.5 * np.einsum("mnij,abmn->abij", H2A["oooo"], r2a, optimize=True)
    X2A += 0.5 * np.einsum("abef,efij->abij", H2A["vvvv"], r2a, optimize=True)
    D3 = np.einsum("amie,ebmj->abij", H2A["voov"], r2a, optimize=True)  # A(ij)A(ab)
    D4 = np.einsum("amie,bejm->abij", H2B["voov"], r2b, optimize=True)  # A(ij)A(ab)
    D5 = -np.einsum("bmji,am->abij", H2A["vooo"], r1a, optimize=True)  # A(ab)
    D6 = np.einsum("baje,ei->abij", H2A["vvov"], r1a, optimize=True)  # A(ij)

    Q1 = -0.5 * np.einsum("mnef,bfmn->eb", vA["oovv"], r2a, optimize=True)
    D7 = np.einsum("eb,aeij->abij", Q1, t2a, optimize=True)  # A(ab)
    Q2 = -np.einsum("mnef,bfmn->eb", vB["oovv"], r2b, optimize=True)
    D8 = np.einsum("eb,aeij->abij", Q2, t2a, optimize=True)  # A(ab)

    Q1 = 0.5 * np.einsum("mnef,efjn->mj", vA["oovv"], r2a, optimize=True)
    D9 = -np.einsum("mj,abim->abij", Q1, t2a, optimize=True)  # A(ij)
    Q2 = np.einsum("mnef,efjn->mj", vB["oovv"], r2b, optimize=True)
    D10 = -np.einsum("mj,abim->abij", Q2, t2a, optimize=True)  # A(ij)

    Q1 = np.einsum("amfe,em->af", H2A["vovv"], r1a, optimize=True)
    D11 = np.einsum("af,fbij->abij", Q1, t2a, optimize=True)  # A(ab)
    Q2 = np.einsum("nmie,em->ni", H2A["ooov"], r1a, optimize=True)
    D12 = -np.einsum("ni,abnj->abij", Q2, t2a, optimize=True)  # A(ij)

    Q1 = np.einsum("amfe,em->af", H2B["vovv"], r1b, optimize=True)
    D13 = np.einsum("af,fbij->abij", Q1, t2a, optimize=True)  # A(ab)
    Q2 = np.einsum("nmie,em->ni", H2B["ooov"], r1b, optimize=True)
    D14 = -np.einsum("ni,abnj->abij", Q2, t2a, optimize=True)  # A(ij)

    D_ij = D1 + D6 + D9 + D10 + D12 + D14
    D_ab = D2 + D5 + D7 + D8 + D11 + D13
    D_abij = D3 + D4

    D_ij -= np.einsum("abij->abji", D_ij, optimize=True)
    D_ab -= np.einsum("abij->baij", D_ab, optimize=True)
    D_abij += (
        -np.einsum("abij->baij", D_abij, optimize=True)
        - np.einsum("abij->abji", D_abij, optimize=True)
        + np.einsum("abij->baji", D_abij, optimize=True)
    )

    X2A += D_ij + D_ab + D_abij

    return X2A


def build_HR_2B(r1a, r1b, r2a, r2b, r2c, cc_t, H1A, H1B, H2A, H2B, H2C, ints, sys):

    t2b = cc_t["t2b"]
    vA = ints["vA"]
    vB = ints["vB"]
    vC = ints["vC"]

    X2B = 0.0
    X2B += np.einsum("ae,ebij->abij", H1A["vv"], r2b, optimize=True)
    X2B += np.einsum("be,aeij->abij", H1B["vv"], r2b, optimize=True)
    X2B -= np.einsum("mi,abmj->abij", H1A["oo"], r2b, optimize=True)
    X2B -= np.einsum("mj,abim->abij", H1B["oo"], r2b, optimize=True)
    X2B += np.einsum("mnij,abmn->abij", H2B["oooo"], r2b, optimize=True)
    X2B += np.einsum("abef,efij->abij", H2B["vvvv"], r2b, optimize=True)
    X2B += np.einsum("amie,ebmj->abij", H2A["voov"], r2b, optimize=True)
    X2B += np.einsum("amie,ebmj->abij", H2B["voov"], r2c, optimize=True)
    X2B += np.einsum("mbej,aeim->abij", H2B["ovvo"], r2a, optimize=True)
    X2B += np.einsum("bmje,aeim->abij", H2C["voov"], r2b, optimize=True)
    X2B -= np.einsum("mbie,aemj->abij", H2B["ovov"], r2b, optimize=True)
    X2B -= np.einsum("amej,ebim->abij", H2B["vovo"], r2b, optimize=True)
    X2B += np.einsum("abej,ei->abij", H2B["vvvo"], r1a, optimize=True)
    X2B += np.einsum("abie,ej->abij", H2B["vvov"], r1b, optimize=True)
    X2B -= np.einsum("mbij,am->abij", H2B["ovoo"], r1a, optimize=True)
    X2B -= np.einsum("amij,bm->abij", H2B["vooo"], r1b, optimize=True)

    Q1 = -0.5 * np.einsum("mnef,afmn->ae", vA["oovv"], r2a, optimize=True)
    X2B += np.einsum("ae,ebij->abij", Q1, t2b, optimize=True)
    Q2 = 0.5 * np.einsum("mnef,efin->mi", vA["oovv"], r2a, optimize=True)
    X2B -= np.einsum("mi,abmj->abij", Q2, t2b, optimize=True)

    Q1 = -np.einsum("nmfe,fbnm->be", vB["oovv"], r2b, optimize=True)
    X2B += np.einsum("be,aeij->abij", Q1, t2b, optimize=True)
    Q2 = -np.einsum("mnef,afmn->ae", vB["oovv"], r2b, optimize=True)
    X2B += np.einsum("ae,ebij->abij", Q2, t2b, optimize=True)
    Q3 = np.einsum("nmfe,fenj->mj", vB["oovv"], r2b, optimize=True)
    X2B -= np.einsum("mj,abim->abij", Q3, t2b, optimize=True)
    Q4 = np.einsum("mnef,efin->mi", vB["oovv"], r2b, optimize=True)
    X2B -= np.einsum("mi,abmj->abij", Q4, t2b, optimize=True)

    Q1 = -0.5 * np.einsum("mnef,bfmn->be", vC["oovv"], r2c, optimize=True)
    X2B += np.einsum("be,aeij->abij", Q1, t2b, optimize=True)
    Q2 = 0.5 * np.einsum("mnef,efjn->mj", vC["oovv"], r2c, optimize=True)
    X2B -= np.einsum("mj,abim->abij", Q2, t2b, optimize=True)

    Q1 = np.einsum("mbef,em->bf", H2B["ovvv"], r1a, optimize=True)
    X2B += np.einsum("bf,afij->abij", Q1, t2b, optimize=True)
    Q2 = np.einsum("mnej,em->nj", H2B["oovo"], r1a, optimize=True)
    X2B -= np.einsum("nj,abin->abij", Q2, t2b, optimize=True)
    Q3 = np.einsum("amfe,em->af", H2A["vovv"], r1a, optimize=True)
    X2B += np.einsum("af,fbij->abij", Q3, t2b, optimize=True)
    Q4 = np.einsum("nmie,em->ni", H2A["ooov"], r1a, optimize=True)
    X2B -= np.einsum("ni,abnj->abij", Q4, t2b, optimize=True)

    Q1 = np.einsum("amfe,em->af", H2B["vovv"], r1b, optimize=True)
    X2B += np.einsum("af,fbij->abij", Q1, t2b, optimize=True)
    Q2 = np.einsum("nmie,em->ni", H2B["ooov"], r1b, optimize=True)
    X2B -= np.einsum("ni,abnj->abij", Q2, t2b, optimize=True)
    Q3 = np.einsum("bmfe,em->bf", H2C["vovv"], r1b, optimize=True)
    X2B += np.einsum("bf,afij->abij", Q3, t2b, optimize=True)
    Q4 = np.einsum("nmje,em->nj", H2C["ooov"], r1b, optimize=True)
    X2B -= np.einsum("nj,abin->abij", Q4, t2b, optimize=True)
    return X2B


def build_HR_2C(r1a, r1b, r2a, r2b, r2c, cc_t, H1A, H1B, H2A, H2B, H2C, ints, sys):

    t2c = cc_t["t2c"]
    vC = ints["vC"]
    vB = ints["vB"]

    X2C = 0.0
    D1 = -np.einsum("mi,abmj->abij", H1B["oo"], r2c, optimize=True)  # A(ij)
    D2 = np.einsum("ae,ebij->abij", H1B["vv"], r2c, optimize=True)  # A(ab)
    X2C += 0.5 * np.einsum("mnij,abmn->abij", H2C["oooo"], r2c, optimize=True)
    X2C += 0.5 * np.einsum("abef,efij->abij", H2C["vvvv"], r2c, optimize=True)
    D3 = np.einsum("amie,ebmj->abij", H2C["voov"], r2c, optimize=True)  # A(ij)A(ab)
    D4 = np.einsum("maei,ebmj->abij", H2B["ovvo"], r2b, optimize=True)  # A(ij)A(ab)
    D5 = -np.einsum("bmji,am->abij", H2C["vooo"], r1b, optimize=True)  # A(ab)
    D6 = np.einsum("baje,ei->abij", H2C["vvov"], r1b, optimize=True)  # A(ij)

    Q1 = -0.5 * np.einsum("mnef,bfmn->eb", vC["oovv"], r2c, optimize=True)
    D7 = np.einsum("eb,aeij->abij", Q1, t2c, optimize=True)  # A(ab)
    Q2 = -np.einsum("nmfe,fbnm->eb", vB["oovv"], r2b, optimize=True)
    D8 = np.einsum("eb,aeij->abij", Q2, t2c, optimize=True)  # A(ab)
    # D7 = 0.0
    # D8 = 0.0

    Q1 = 0.5 * np.einsum("mnef,efjn->mj", vC["oovv"], r2c, optimize=True)
    D9 = -np.einsum("mj,abim->abij", Q1, t2c, optimize=True)  # A(ij)
    Q2 = np.einsum("nmfe,fenj->mj", vB["oovv"], r2b, optimize=True)
    D10 = -np.einsum("mj,abim->abij", Q2, t2c, optimize=True)  # A(ij)
    # D9 = 0.0
    # D10 = 0.0

    Q1 = np.einsum("amfe,em->af", H2C["vovv"], r1b, optimize=True)
    D11 = np.einsum("af,fbij->abij", Q1, t2c, optimize=True)  # A(ab)
    Q2 = np.einsum("nmie,em->ni", H2C["ooov"], r1b, optimize=True)
    D12 = -np.einsum("ni,abnj->abij", Q2, t2c, optimize=True)  # A(ij)
    # D11 = 0.0
    # D12 = 0.0

    Q1 = np.einsum("maef,em->af", H2B["ovvv"], r1a, optimize=True)
    D13 = np.einsum("af,fbij->abij", Q1, t2c, optimize=True)  # A(ab)
    Q2 = np.einsum("mnei,em->ni", H2B["oovo"], r1a, optimize=True)
    D14 = -np.einsum("ni,abnj->abij", Q2, t2c, optimize=True)  # A(ij)
    # D13 = 0.0
    # D14 = 0.0

    D_ij = D1 + D6 + D9 + D10 + D12 + D14
    D_ab = D2 + D5 + D7 + D8 + D11 + D13
    D_abij = D3 + D4

    D_ij -= np.einsum("abij->abji", D_ij, optimize=True)
    D_ab -= np.einsum("abij->baij", D_ab, optimize=True)
    D_abij += (
        -np.einsum("abij->baij", D_abij, optimize=True)
        - np.einsum("abij->abji", D_abij, optimize=True)
        + np.einsum("abij->baji", D_abij, optimize=True)
    )

    X2C += D_ij + D_ab + D_abij

    return X2C
