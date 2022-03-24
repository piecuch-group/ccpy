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
        0.0,
    )
    return dR


def HR(R, T, H, flag_RHF, system):

    if flag_RHF:
        X1A = build_HR_1A(R, T, H)
        X2A = build_HR_2A(R, T, H)
        X2B = build_HR_2B(R, T, H)
        Xout = np.concatenate((X1A.flatten(), X1A.flatten(), X2A.flatten(), X2B.flatten(), X2A.flatten()), axis=0)
    else:
        X1A = build_HR_1A(R, T, H)
        X1B = build_HR_1B(R, T, H)
        X2A = build_HR_2A(R, T, H)
        X2B = build_HR_2B(R, T, H)
        X2C = build_HR_2C(R, T, H)
        Xout = np.concatenate( (X1A.flatten(), X1B.flatten(), X2A.flatten(), X2B.flatten(), X2C.flatten()), axis=0)

    return Xout


def build_HR_1A(R, T, H):

    X1A = -np.einsum("mi,am->ai", H.a.oo, R.a, optimize=True)
    X1A += np.einsum("ae,ei->ai", H.a.vv, R.a, optimize=True)
    X1A += np.einsum("amie,em->ai", H.aa.voov, R.a, optimize=True)
    X1A += np.einsum("amie,em->ai", H.ab.voov, R.b, optimize=True)
    X1A -= 0.5 * np.einsum("mnif,afmn->ai", H.aa.ooov, R.aa, optimize=True)
    X1A -= np.einsum("mnif,afmn->ai", H.ab.ooov, R.ab, optimize=True)
    X1A += 0.5 * np.einsum("anef,efin->ai", H.aa.vovv, R.aa, optimize=True)
    X1A += np.einsum("anef,efin->ai", H.ab.vovv, R.ab, optimize=True)
    X1A += np.einsum("me,aeim->ai", H.a.ov, R.aa, optimize=True)
    X1A += np.einsum("me,aeim->ai", H.b.ov, R.ab, optimize=True)

    return X1A


def build_HR_1B(R, T, H):

    X1B = -np.einsum("mi,am->ai", H.b.oo, R.b, optimize=True)
    X1B += np.einsum("ae,ei->ai", H.b.vv, R.b, optimize=True)
    X1B += np.einsum("maei,em->ai", H.ab.ovvo, R.a, optimize=True)
    X1B += np.einsum("amie,em->ai", H.bb.voov, R.b, optimize=True)
    X1B -= np.einsum("nmfi,fanm->ai", H.ab.oovo, R.ab, optimize=True)
    X1B -= 0.5 * np.einsum("mnif,afmn->ai", H.bb.ooov, R.bb, optimize=True)
    X1B += np.einsum("nafe,feni->ai", H.ab.ovvv, R.ab, optimize=True)
    X1B += 0.5 * np.einsum("anef,efin->ai", H.bb.vovv, R.bb, optimize=True)
    X1B += np.einsum("me,eami->ai", H.a.ov, R.ab, optimize=True)
    X1B += np.einsum("me,aeim->ai", H.b.ov, R.bb, optimize=True)

    return X1B


def build_HR_2A(R, T, H):

    D1 = -np.einsum("mi,abmj->abij", H.a.oo, R.aa, optimize=True)  # A(ij)
    D2 = np.einsum("ae,ebij->abij", H.a.vv, R.aa, optimize=True)  # A(ab)
    X2A = 0.5 * np.einsum("mnij,abmn->abij", H.aa.oooo, R.aa, optimize=True)
    X2A += 0.5 * np.einsum("abef,efij->abij", H.aa.vvvv, R.aa, optimize=True)
    D3 = np.einsum("amie,ebmj->abij", H.aa.voov, R.aa, optimize=True)  # A(ij)A(ab)
    D4 = np.einsum("amie,bejm->abij", H.ab.voov, R.ab, optimize=True)  # A(ij)A(ab)
    D5 = -np.einsum("bmji,am->abij", H.aa.vooo, R.a, optimize=True)  # A(ab)
    D6 = np.einsum("baje,ei->abij", H.aa.vvov, R.a, optimize=True)  # A(ij)

    Q1 = -0.5 * np.einsum("mnef,bfmn->eb", H.aa.oovv, R.aa, optimize=True)
    D7 = np.einsum("eb,aeij->abij", Q1, T.aa, optimize=True)  # A(ab)
    Q2 = -np.einsum("mnef,bfmn->eb", H.ab.oovv, R.ab, optimize=True)
    D8 = np.einsum("eb,aeij->abij", Q2, T.aa, optimize=True)  # A(ab)

    Q1 = 0.5 * np.einsum("mnef,efjn->mj", H.aa.oovv, R.aa, optimize=True)
    D9 = -np.einsum("mj,abim->abij", Q1, T.aa, optimize=True)  # A(ij)
    Q2 = np.einsum("mnef,efjn->mj", H.ab.oovv, R.ab, optimize=True)
    D10 = -np.einsum("mj,abim->abij", Q2, T.aa, optimize=True)  # A(ij)

    Q1 = np.einsum("amfe,em->af", H.aa.vovv, R.a, optimize=True)
    D11 = np.einsum("af,fbij->abij", Q1, T.aa, optimize=True)  # A(ab)
    Q2 = np.einsum("nmie,em->ni", H.aa.ooov, R.a, optimize=True)
    D12 = -np.einsum("ni,abnj->abij", Q2, T.aa, optimize=True)  # A(ij)

    Q1 = np.einsum("amfe,em->af", H.ab.vovv, R.b, optimize=True)
    D13 = np.einsum("af,fbij->abij", Q1, T.aa, optimize=True)  # A(ab)
    Q2 = np.einsum("nmie,em->ni", H.ab.ooov, R.b, optimize=True)
    D14 = -np.einsum("ni,abnj->abij", Q2, T.aa, optimize=True)  # A(ij)

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


def build_HR_2B(R, T, H):
    
    X2B = np.einsum("ae,ebij->abij", H.a.vv, R.ab, optimize=True)
    X2B += np.einsum("be,aeij->abij", H.b.vv, R.ab, optimize=True)
    X2B -= np.einsum("mi,abmj->abij", H.a.oo, R.ab, optimize=True)
    X2B -= np.einsum("mj,abim->abij", H.b.oo, R.ab, optimize=True)
    X2B += np.einsum("mnij,abmn->abij", H.ab.oooo, R.ab, optimize=True)
    X2B += np.einsum("abef,efij->abij", H.ab.vvvv, R.ab, optimize=True)
    X2B += np.einsum("amie,ebmj->abij", H.aa.voov, R.ab, optimize=True)
    X2B += np.einsum("amie,ebmj->abij", H.ab.voov, R.bb, optimize=True)
    X2B += np.einsum("mbej,aeim->abij", H.ab.ovvo, R.aa, optimize=True)
    X2B += np.einsum("bmje,aeim->abij", H.bb.voov, R.ab, optimize=True)
    X2B -= np.einsum("mbie,aemj->abij", H.ab.ovov, R.ab, optimize=True)
    X2B -= np.einsum("amej,ebim->abij", H.ab.vovo, R.ab, optimize=True)
    X2B += np.einsum("abej,ei->abij", H.ab.vvvo, R.a, optimize=True)
    X2B += np.einsum("abie,ej->abij", H.ab.vvov, R.b, optimize=True)
    X2B -= np.einsum("mbij,am->abij", H.ab.ovoo, R.a, optimize=True)
    X2B -= np.einsum("amij,bm->abij", H.ab.vooo, R.b, optimize=True)

    Q1 = -0.5 * np.einsum("mnef,afmn->ae", H.aa.oovv, R.aa, optimize=True)
    X2B += np.einsum("ae,ebij->abij", Q1, T.ab, optimize=True)
    Q2 = 0.5 * np.einsum("mnef,efin->mi", H.aa.oovv, R.aa, optimize=True)
    X2B -= np.einsum("mi,abmj->abij", Q2, T.ab, optimize=True)

    Q1 = -np.einsum("nmfe,fbnm->be", H.ab.oovv, R.ab, optimize=True)
    X2B += np.einsum("be,aeij->abij", Q1, T.ab, optimize=True)
    Q2 = -np.einsum("mnef,afmn->ae", H.ab.oovv, R.ab, optimize=True)
    X2B += np.einsum("ae,ebij->abij", Q2, T.ab, optimize=True)
    Q3 = np.einsum("nmfe,fenj->mj", H.ab.oovv, R.ab, optimize=True)
    X2B -= np.einsum("mj,abim->abij", Q3, T.ab, optimize=True)
    Q4 = np.einsum("mnef,efin->mi", H.ab.oovv, R.ab, optimize=True)
    X2B -= np.einsum("mi,abmj->abij", Q4, T.ab, optimize=True)

    Q1 = -0.5 * np.einsum("mnef,bfmn->be", H.bb.oovv, R.bb, optimize=True)
    X2B += np.einsum("be,aeij->abij", Q1, T.ab, optimize=True)
    Q2 = 0.5 * np.einsum("mnef,efjn->mj", H.bb.oovv, R.bb, optimize=True)
    X2B -= np.einsum("mj,abim->abij", Q2, T.ab, optimize=True)

    Q1 = np.einsum("mbef,em->bf", H.ab.ovvv, R.a, optimize=True)
    X2B += np.einsum("bf,afij->abij", Q1, T.ab, optimize=True)
    Q2 = np.einsum("mnej,em->nj", H.ab.oovo, R.a, optimize=True)
    X2B -= np.einsum("nj,abin->abij", Q2, T.ab, optimize=True)
    Q3 = np.einsum("amfe,em->af", H.aa.vovv, R.a, optimize=True)
    X2B += np.einsum("af,fbij->abij", Q3, T.ab, optimize=True)
    Q4 = np.einsum("nmie,em->ni", H.aa.ooov, R.a, optimize=True)
    X2B -= np.einsum("ni,abnj->abij", Q4, T.ab, optimize=True)

    Q1 = np.einsum("amfe,em->af", H.ab.vovv, R.b, optimize=True)
    X2B += np.einsum("af,fbij->abij", Q1, T.ab, optimize=True)
    Q2 = np.einsum("nmie,em->ni", H.ab.ooov, R.b, optimize=True)
    X2B -= np.einsum("ni,abnj->abij", Q2, T.ab, optimize=True)
    Q3 = np.einsum("bmfe,em->bf", H.bb.vovv, R.b, optimize=True)
    X2B += np.einsum("bf,afij->abij", Q3, T.ab, optimize=True)
    Q4 = np.einsum("nmje,em->nj", H.bb.ooov, R.b, optimize=True)
    X2B -= np.einsum("nj,abin->abij", Q4, T.ab, optimize=True)
    
    return X2B


def build_HR_2C(R, T, H):

    D1 = -np.einsum("mi,abmj->abij", H.b.oo, R.bb, optimize=True)  # A(ij)
    D2 = np.einsum("ae,ebij->abij", H.b.vv, R.bb, optimize=True)  # A(ab)
    X2C = 0.5 * np.einsum("mnij,abmn->abij", H.bb.oooo, R.bb, optimize=True)
    X2C += 0.5 * np.einsum("abef,efij->abij", H.bb.vvvv, R.bb, optimize=True)
    D3 = np.einsum("amie,ebmj->abij", H.bb.voov, R.bb, optimize=True)  # A(ij)A(ab)
    D4 = np.einsum("maei,ebmj->abij", H.ab.ovvo, R.ab, optimize=True)  # A(ij)A(ab)
    D5 = -np.einsum("bmji,am->abij", H.bb.vooo, R.b, optimize=True)  # A(ab)
    D6 = np.einsum("baje,ei->abij", H.bb.vvov, R.b, optimize=True)  # A(ij)

    Q1 = -0.5 * np.einsum("mnef,bfmn->eb", H.bb.oovv, R.bb, optimize=True)
    D7 = np.einsum("eb,aeij->abij", Q1, T.bb, optimize=True)  # A(ab)
    Q2 = -np.einsum("nmfe,fbnm->eb", H.ab.oovv, R.ab, optimize=True)
    D8 = np.einsum("eb,aeij->abij", Q2, T.bb, optimize=True)  # A(ab)

    Q1 = 0.5 * np.einsum("mnef,efjn->mj", H.bb.oovv, R.bb, optimize=True)
    D9 = -np.einsum("mj,abim->abij", Q1, T.bb, optimize=True)  # A(ij)
    Q2 = np.einsum("nmfe,fenj->mj", H.ab.oovv, R.ab, optimize=True)
    D10 = -np.einsum("mj,abim->abij", Q2, T.bb, optimize=True)  # A(ij)

    Q1 = np.einsum("amfe,em->af", H.bb.vovv, R.b, optimize=True)
    D11 = np.einsum("af,fbij->abij", Q1, T.bb, optimize=True)  # A(ab)
    Q2 = np.einsum("nmie,em->ni", H.bb.ooov, R.b, optimize=True)
    D12 = -np.einsum("ni,abnj->abij", Q2, T.bb, optimize=True)  # A(ij)

    Q1 = np.einsum("maef,em->af", H.ab.ovvv, R.a, optimize=True)
    D13 = np.einsum("af,fbij->abij", Q1, T.bb, optimize=True)  # A(ab)
    Q2 = np.einsum("mnei,em->ni", H.ab.oovo, R.a, optimize=True)
    D14 = -np.einsum("ni,abnj->abij", Q2, T.bb, optimize=True)  # A(ij)

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
