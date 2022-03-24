"""Module containing functions to calculate the vertical excitation
energies and linear excitation amplitudes for excited states using
the equation-of-motion (EOM) CC with singles, doubles, and triples (EOMCCSDT)."""
import numpy as np
from ccpy.utilities.updates import cc_loops

def update(dR, omega, H):

    dR.a, dR.b, dR.aa, dR.ab, dR.bb, dR.aaa, dR.aab, dR.abb, dR.bbb = cc_loops.cc_loops.update_r_ccsdt(
        dR.a,
        dR.b,
        dR.aa,
        dR.ab,
        dR.bb,
        dR.aaa,
        dR.aab,
        dR.abb,
        dR.bbb,
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
        X3A = build_HR_3A(R, T, H)
        X3B = build_HR_3B(R, T, H)
        Xout = np.concatenate((X1A.flatten(), X1A.flatten(), X2A.flatten(), X2B.flatten(), X2A.flatten(),
                               X3A.flatten(), X3B.flatten(), np.transpose(X3B, (2, 1, 0, 5, 4, 3)).flatten(), X3A.flatten()), axis=0)
    else:
        X1A = build_HR_1A(R, T, H)
        X1B = build_HR_1B(R, T, H)
        X2A = build_HR_2A(R, T, H)
        X2B = build_HR_2B(R, T, H)
        X2C = build_HR_2C(R, T, H)
        # [TODO]: Create an HR function to produce one- and two-body components arising from [Hbar * (R1 + R2)]_C
        X3A = build_HR_3A(R, T, H)
        X3B = build_HR_3B(R, T, H)
        X3C = build_HR_3C(R, T, H)
        X3D = build_HR_3D(R, T, H)
        Xout = np.concatenate( (X1A.flatten(), X1B.flatten(), X2A.flatten(), X2B.flatten(), X2C.flatten(),
                                X3A.flatten(), X3B.flatten(), X3C.flatten(), X3D.flatten()), axis=0)

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

    X1A += 0.25 * np.einsum("mnef,aefimn->ai", H.aa.oovv, R.aaa, optimize=True)
    X1A += np.einsum("mnef,aefimn->ai", H.ab.oovv, R.aab, optimize=True)
    X1A += 0.25 * np.einsum("mnef,aefimn->ai", H.bb.oovv, R.abb, optimize=True)

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

    X1B += 0.25 * np.einsum("mnef,efamni->ai", H.aa.oovv, R.aab, optimize=True)
    X1B += np.einsum("mnef,efamni->ai", H.ab.oovv, R.abb, optimize=True)
    X1B += 0.25 * np.einsum("mnef,aefimn->ai", H.bb.oovv, R.bbb, optimize=True)

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

    I1 = np.einsum("mnef,fn->me", H.aa.oovv, R.a, optimize=True) + np.einsum(
        "mnef,fn->me", H.ab.oovv, R.b, optimize=True
    )
    X2A += np.einsum("me,abeijm->abij", I1, T.aaa, optimize=True)

    I1 = np.einsum("nmfe,fn->me", H.ab.oovv, R.a, optimize=True) + np.einsum(
        "mnef,fn->me", H.bb.oovv, R.b, optimize=True
    )
    X2A += np.einsum("me,abeijm->abij", I1, T.aab, optimize=True)

    DR3_1 = np.einsum("me,abeijm->abij", H.a.ov, R.aaa, optimize=True)
    DR3_2 = np.einsum("me,abeijm->abij", H.b.ov, R.aab, optimize=True)
    DR3_3 = -0.5 * np.einsum("mnjf,abfimn->abij", H.aa.ooov, R.aaa, optimize=True)
    DR3_4 = -1.0 * np.einsum("mnjf,abfimn->abij", H.ab.ooov, R.aab, optimize=True)
    DR3_5 = 0.5 * np.einsum("bnef,aefijn->abij", H.aa.vovv, R.aaa, optimize=True)
    DR3_6 = np.einsum("bnef,aefijn->abij", H.ab.vovv, R.aab, optimize=True)

    D_ij = D1 + D6 + D9 + D10 + D12 + D14 + DR3_3 + DR3_4
    D_ab = D2 + D5 + D7 + D8 + D11 + D13 + DR3_5 + DR3_6
    D_abij = D3 + D4

    D_ij -= np.einsum("abij->abji", D_ij, optimize=True)
    D_ab -= np.einsum("abij->baij", D_ab, optimize=True)
    D_abij += (
        -np.einsum("abij->baij", D_abij, optimize=True)
        - np.einsum("abij->abji", D_abij, optimize=True)
        + np.einsum("abij->baji", D_abij, optimize=True)
    )

    X2A += D_ij + D_ab + D_abij + DR3_1 + DR3_2

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

    I1 = np.einsum("mnef,fn->me", H.aa.oovv, R.a, optimize=True) + np.einsum(
        "mnef,fn->me", H.ab.oovv, R.b, optimize=True
    )
    X2B += np.einsum("me,aebimj->abij", I1, T.aab, optimize=True)

    I1 = np.einsum("nmfe,fn->me", H.ab.oovv, R.a, optimize=True) + np.einsum(
        "mnef,fn->me", H.bb.oovv, R.b, optimize=True
    )
    X2B += np.einsum("me,aebimj->abij", I1, T.abb, optimize=True)

    X2B += np.einsum("me,aebimj->abij", H.a.ov, R.aab, optimize=True)
    X2B += np.einsum("me,aebimj->abij", H.b.ov, R.abb, optimize=True)
    X2B -= np.einsum("nmfj,afbinm->abij", H.ab.oovo, R.aab, optimize=True)
    X2B -= 0.5 * np.einsum("mnjf,abfimn->abij", H.bb.ooov, R.abb, optimize=True)
    X2B -= 0.5 * np.einsum("mnif,afbmnj->abij", H.aa.ooov, R.aab, optimize=True)
    X2B -= np.einsum("mnif,abfmjn->abij", H.ab.ooov, R.abb, optimize=True)
    X2B += np.einsum("nbfe,afeinj->abij", H.ab.ovvv, R.aab, optimize=True)
    X2B += 0.5 * np.einsum("bnef,aefijn->abij", H.bb.vovv, R.abb, optimize=True)
    X2B += 0.5 * np.einsum("anef,efbinj->abij", H.aa.vovv, R.aab, optimize=True)
    X2B += np.einsum("anef,efbinj->abij", H.ab.vovv, R.abb, optimize=True)

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

    I1 = np.einsum("mnef,fn->me", H.aa.oovv, R.a, optimize=True) + np.einsum(
        "mnef,fn->me", H.ab.oovv, R.b, optimize=True
    )
    X2C += np.einsum("me,eabmij->abij", I1, T.abb, optimize=True)

    I1 = np.einsum("nmfe,fn->me", H.ab.oovv, R.a, optimize=True) + np.einsum(
        "mnef,fn->me", H.bb.oovv, R.b, optimize=True
    )
    X2C += np.einsum("me,abeijm->abij", I1, T.bbb, optimize=True)

    DR3_1 = np.einsum("me,eabmij->abij", H.a.ov, R.abb, optimize=True)
    DR3_2 = np.einsum("me,abeijm->abij", H.b.ov, R.bbb, optimize=True)
    DR3_3 = -0.5 * np.einsum("mnjf,abfimn->abij", H.bb.ooov, R.bbb, optimize=True)
    DR3_4 = -1.0 * np.einsum("nmfj,fabnim->abij", H.ab.oovo, R.abb, optimize=True)
    DR3_5 = 0.5 * np.einsum("bnef,aefijn->abij", H.bb.vovv, R.bbb, optimize=True)
    DR3_6 = np.einsum("nbfe,faenij->abij", H.ab.ovvv, R.abb, optimize=True)

    D_ij = D1 + D6 + D9 + D10 + D12 + D14 + DR3_3 + DR3_4
    D_ab = D2 + D5 + D7 + D8 + D11 + D13 + DR3_5 + DR3_6
    D_abij = D3 + D4

    D_ij -= np.einsum("abij->abji", D_ij, optimize=True)
    D_ab -= np.einsum("abij->baij", D_ab, optimize=True)
    D_abij += (
        -np.einsum("abij->baij", D_abij, optimize=True)
        - np.einsum("abij->abji", D_abij, optimize=True)
        + np.einsum("abij->baji", D_abij, optimize=True)
    )

    X2C += D_ij + D_ab + D_abij + DR3_1 + DR3_2

    return X2C


def build_HR_3A(R, T, H):
    
    # <ijkabc| [H(R1+R2)]_C | 0 >
    Q1 = np.einsum("mnef,fn->me", H.aa.oovv, R.a, optimize=True)
    Q1 += np.einsum("mnef,fn->me", H.ab.oovv, R.b, optimize=True)
    I1 = np.einsum("amje,bm->abej", H.aa.voov, R.a, optimize=True)
    I1 += np.einsum("amfe,bejm->abfj", H.aa.vovv, R.aa, optimize=True)
    I1 += np.einsum("amfe,bejm->abfj", H.ab.vovv, R.ab, optimize=True)
    I1 -= np.transpose(I1, (1, 0, 2, 3))
    I2 = np.einsum("abfe,ej->abfj", H.aa.vvvv, R.a, optimize=True)
    I2 += 0.5 * np.einsum("nmje,abmn->abej", H.aa.ooov, R.aa, optimize=True)
    I2 -= np.einsum("me,abmj->abej", Q1, T.aa, optimize=True)
    I3 = -0.5 * np.einsum(
        "mnef,abfimn->baei", H.aa.oovv, R.aaa, optimize=True
    ) - np.einsum("mnef,abfimn->baei", H.ab.oovv, R.aab, optimize=True)
    X3A = 0.25 * np.einsum("abej,ecik->abcijk", I1 + I2 + I3, T.aa, optimize=True)
    X3A += 0.25 * np.einsum("baje,ecik->abcijk", H.aa.vvov, R.aa, optimize=True)

    I1 = -np.einsum("bmie,ej->mbij", H.aa.voov, R.a, optimize=True)
    I1 += np.einsum("nmie,bejm->nbij", H.aa.ooov, R.aa, optimize=True)
    I1 += np.einsum("nmie,bejm->nbij", H.ab.ooov, R.ab, optimize=True)
    I1 -= np.transpose(I1, (0, 1, 3, 2))
    I2 = -1.0 * np.einsum("nmij,bm->nbij", H.aa.oooo, R.a, optimize=True)
    I2 += 0.5 * np.einsum("bmfe,efij->mbij", H.aa.vovv, R.aa, optimize=True)
    I3 = 0.5 * np.einsum(
        "mnef,efcjnk->mcjk", H.aa.oovv, R.aaa, optimize=True
    ) + np.einsum("mnef,ecfjkn->mcjk", H.ab.oovv, R.aab, optimize=True)
    X3A -= 0.25 * np.einsum("mbij,acmk->abcijk", I1 + I2 + I3, T.aa, optimize=True)

    # additional terms with T3 in <ijkabc|[ H(R1+R2)]_C | 0>
    I1 = (
        -1.0 * np.einsum("me,bm->be", H.a.ov, R.a, optimize=True)
        + np.einsum("bnef,fn->be", H.aa.vovv, R.a, optimize=True)
        + np.einsum("bnef,fn->be", H.ab.vovv, R.b, optimize=True)
    )
    I2 = -0.5 * np.einsum("mnef,bfmn->be", H.aa.oovv, R.aa, optimize=True) - np.einsum(
        "mnef,bfmn->be", H.ab.oovv, R.ab, optimize=True
    )
    X3A += (1.0 / 12.0) * np.einsum(
        "be,aecijk->abcijk", I1 + I2, T.aaa, optimize=True
    )  # A(b/ac)

    I1 = (
        np.einsum("me,ej->mj", H.a.ov, R.a, optimize=True)
        + np.einsum("mnjf,fn->mj", H.aa.ooov, R.a, optimize=True)
        + np.einsum("mnjf,fn->mj", H.ab.ooov, R.b, optimize=True)
    )
    I2 = 0.5 * np.einsum("mnef,efjn->mj", H.aa.oovv, R.aa, optimize=True) + np.einsum(
        "mnef,efjn->mj", H.ab.oovv, R.ab, optimize=True
    )
    X3A -= (1.0 / 12.0) * np.einsum(
        "mj,abcimk->abcijk", I1 + I2, T.aaa, optimize=True
    )  # A(j/ik)

    I1 = np.einsum("nmje,ei->mnij", H.aa.ooov, R.a, optimize=True)
    I1 -= np.transpose(I1, (0, 1, 3, 2))
    I2 = 0.5 * np.einsum("mnef,efij->mnij", H.aa.oovv, R.aa, optimize=True)
    X3A += (1.0 / 24.0) * np.einsum(
        "mnij,abcmnk->abcijk", I1 + I2, T.aaa, optimize=True
    )  # A(k/ij)

    I1 = -1.0 * np.einsum("amef,bm->abef", H.aa.vovv, R.a, optimize=True)
    I1 -= np.transpose(I1, (1, 0, 2, 3))
    I2 = 0.5 * np.einsum("mnef,abmn->abef", H.aa.oovv, R.aa, optimize=True)
    X3A += (1.0 / 24.0) * np.einsum(
        "abef,efcijk->abcijk", I1 + I2, T.aaa, optimize=True
    )  # A(c/ab)

    I1 = -1.0 * np.einsum("nmje,bn->bmje", H.aa.ooov, R.a, optimize=True) + np.einsum(
        "bmfe,fj->bmje", H.aa.vovv, R.a, optimize=True
    )
    I2 = np.einsum("mnef,fcnk->cmke", H.aa.oovv, R.aa, optimize=True) + np.einsum(
        "mnef,cfkn->cmke", H.ab.oovv, R.ab, optimize=True
    )
    X3A += 0.25 * np.einsum(
        "bmje,aecimk->abcijk", I1 + I2, T.aaa, optimize=True
    )  # A(j/ik)A(b/ac)

    I1 = -1.0 * np.einsum("nmje,bn->bmje", H.ab.ooov, R.a, optimize=True) + np.einsum(
        "bmfe,fj->bmje", H.ab.vovv, R.a, optimize=True
    )
    I2 = np.einsum("nmfe,fcnk->cmke", H.ab.oovv, R.aa, optimize=True) + np.einsum(
        "mnef,cfkn->cmke", H.bb.oovv, R.ab, optimize=True
    )
    X3A += 0.25 * np.einsum(
        "bmje,aceikm->abcijk", I1 + I2, T.aab, optimize=True
    )  # A(j/ik)A(b/ac)

    # < ijkabc | (HR3)_C | 0 >
    X3A -= (1.0 / 12.0) * np.einsum("mj,abcimk->abcijk", H.a.oo, R.aaa, optimize=True)
    X3A += (1.0 / 12.0) * np.einsum("be,aecijk->abcijk", H.a.vv, R.aaa, optimize=True)
    X3A += (1.0 / 24.0) * np.einsum(
        "mnij,abcmnk->abcijk", H.aa.oooo, R.aaa, optimize=True
    )
    X3A += (1.0 / 24.0) * np.einsum(
        "abef,efcijk->abcijk", H.aa.vvvv, R.aaa, optimize=True
    )
    X3A += 0.25 * np.einsum("amie,ebcmjk->abcijk", H.aa.voov, R.aaa, optimize=True)
    X3A += 0.25 * np.einsum("amie,bcejkm->abcijk", H.ab.voov, R.aab, optimize=True)

    # antisymmetrize terms and add up: A(abc)A(ijk) = A(a/bc)A(bc)A(i/jk)A(jk)
    X3A -= np.transpose(X3A, (0, 1, 2, 3, 5, 4))
    X3A -= np.transpose(X3A, (0, 1, 2, 4, 3, 5)) + np.transpose(X3A, (0, 1, 2, 5, 4, 3))
    X3A -= np.transpose(X3A, (0, 2, 1, 3, 4, 5))
    X3A -= np.transpose(X3A, (1, 0, 2, 3, 4, 5)) + np.transpose(X3A, (2, 1, 0, 3, 4, 5))

    return X3A


def build_HR_3B(R, T, H):

    # < ijk~abc~ | [ H(R1+R2) ]_C | 0 >
    Q1 = np.einsum("mnef,fn->me", H.aa.oovv, R.a, optimize=True) + np.einsum(
        "mnef,fn->me", H.ab.oovv, R.b, optimize=True
    )
    Q2 = np.einsum("nmfe,fn->me", H.ab.oovv, R.a, optimize=True) + np.einsum(
        "nmfe,fn->me", H.bb.oovv, R.b, optimize=True
    )
    # Intermediate 1: X2B(bcek)*Y2A(aeij) -> Z3B(abcijk)
    Int1 = -1.0 * np.einsum("mcek,bm->bcek", H.ab.ovvo, R.a, optimize=True)
    Int1 -= np.einsum("bmek,cm->bcek", H.ab.vovo, R.b, optimize=True)
    Int1 += np.einsum("bcfe,ek->bcfk", H.ab.vvvv, R.b, optimize=True)
    Int1 += np.einsum("mnek,bcmn->bcek", H.ab.oovo, R.ab, optimize=True)
    Int1 += np.einsum("bmfe,ecmk->bcfk", H.aa.vovv, R.ab, optimize=True)
    Int1 += np.einsum("bmfe,ecmk->bcfk", H.ab.vovv, R.bb, optimize=True)
    Int1 -= np.einsum("mcfe,bemk->bcfk", H.ab.ovvv, R.ab, optimize=True)
    I1 = -0.5 * np.einsum(
        "mnef,bfcmnk->bcek", H.aa.oovv, R.aab, optimize=True
    ) - np.einsum("mnef,bfcmnk->bcek", H.ab.oovv, R.abb, optimize=True)
    X3B = 0.5 * np.einsum("bcek,aeij->abcijk", Int1 + I1, T.aa, optimize=True)
    X3B += 0.5 * np.einsum("bcek,aeij->abcijk", H.ab.vvvo, R.aa, optimize=True)
    # Intermediate 2: X2B(ncjk)*Y2A(abin) -> Z3B(abcijk)
    Int2 = -1.0 * np.einsum("nmjk,cm->ncjk", H.ab.oooo, R.b, optimize=True)
    Int2 += np.einsum("mcje,ek->mcjk", H.ab.ovov, R.b, optimize=True)
    Int2 += np.einsum("mcek,ej->mcjk", H.ab.ovvo, R.a, optimize=True)
    Int2 += np.einsum("mcef,efjk->mcjk", H.ab.ovvv, R.ab, optimize=True)
    Int2 += np.einsum("nmje,ecmk->ncjk", H.aa.ooov, R.ab, optimize=True)
    Int2 += np.einsum("nmje,ecmk->ncjk", H.ab.ooov, R.bb, optimize=True)
    Int2 -= np.einsum("nmek,ecjm->ncjk", H.ab.oovo, R.ab, optimize=True)
    I1 = 0.5 * np.einsum(
        "mnef,efcjnk->mcjk", H.aa.oovv, R.aab, optimize=True
    ) + np.einsum("mnef,efcjnk->mcjk", H.ab.oovv, R.abb, optimize=True)
    X3B -= 0.5 * np.einsum("ncjk,abin->abcijk", Int2 + I1, T.aa, optimize=True)
    X3B -= 0.5 * np.einsum("mcjk,abim->abcijk", H.ab.ovoo, R.aa, optimize=True)
    # Intermediate 3: X2A(abej)*Y2B(ecik) -> Z3B(abcijk)
    Int3 = np.einsum(
        "amje,bm->abej", H.aa.voov, R.a, optimize=True
    )  # (*) flipped sign to use H2A(voov) instead of H2A(vovo)
    Int3 += 0.5 * np.einsum(
        "abfe,ej->abfj", H.aa.vvvv, R.a, optimize=True
    )  # (*) added factor 1/2 to compensate A(ab)
    Int3 += 0.25 * np.einsum(
        "nmje,abmn->abej", H.aa.ooov, R.aa, optimize=True
    )  # (*) added factor 1/2 to compensate A(ab)
    Int3 += np.einsum("amfe,bejm->abfj", H.aa.vovv, R.aa, optimize=True)
    Int3 += np.einsum("amfe,bejm->abfj", H.ab.vovv, R.ab, optimize=True)
    Int3 -= 0.5 * np.einsum(
        "me,abmj->abej", Q1, T.aa, optimize=True
    )  # (*) added factor 1/2 to compensate A(ab)
    Int3 -= np.transpose(Int3, (1, 0, 2, 3))
    I1 = -0.5 * np.einsum(
        "mnef,abfmjn->abej", H.aa.oovv, R.aaa, optimize=True
    ) - np.einsum("mnef,abfmjn->abej", H.ab.oovv, R.aab, optimize=True)
    X3B += 0.5 * np.einsum("abej,ecik->abcijk", Int3 + I1, T.ab, optimize=True)
    X3B += 0.5 * np.einsum("baje,ecik->abcijk", H.aa.vvov, R.ab, optimize=True)
    # Intermediate 4: X2A(bnji)*Y2B(acnk) -> Z3B(abcijk)
    Int4 = -0.5 * np.einsum(
        "nmij,bm->bnji", H.aa.oooo, R.a, optimize=True
    )  # (*) added factor 1/2 to compensate A(ij)
    Int4 -= np.einsum(
        "bmie,ej->bmji", H.aa.voov, R.a, optimize=True
    )  # (*) flipped sign to use H2A(voov) instead of H2A(vovo)
    Int4 += 0.25 * np.einsum(
        "bmfe,efij->bmji", H.aa.vovv, R.aa, optimize=True
    )  # (*) added factor 1/2 to compensate A(ij)
    Int4 += np.einsum("nmie,bejm->bnji", H.aa.ooov, R.aa, optimize=True)
    Int4 += np.einsum("nmie,bejm->bnji", H.ab.ooov, R.ab, optimize=True)
    Int4 += 0.5 * np.einsum(
        "me,ebij->bmji", Q1, T.aa, optimize=True
    )  # (*) added factor 1/2 to compensate A(ij)
    Int4 -= np.transpose(Int4, (0, 1, 3, 2))
    I1 = 0.5 * np.einsum(
        "mnef,aefijn->amij", H.aa.oovv, R.aaa, optimize=True
    ) + np.einsum("mnef,aefijn->amij", H.ab.oovv, R.aab, optimize=True)
    X3B -= 0.5 * np.einsum("bnji,acnk->abcijk", Int4 + I1, T.ab, optimize=True)
    X3B -= 0.5 * np.einsum("bnji,acnk->abcijk", H.aa.vooo, R.ab, optimize=True)
    # Intermediate 5: X2B(bcje)*Y2B(aeik) -> Z3B(abcijk)
    Int5 = -1.0 * np.einsum("mcje,bm->bcje", H.ab.ovov, R.a, optimize=True)
    Int5 -= np.einsum("bmje,cm->bcje", H.ab.voov, R.b, optimize=True)
    Int5 += np.einsum("bcef,ej->bcjf", H.ab.vvvv, R.a, optimize=True)
    Int5 += np.einsum("mnjf,bcmn->bcjf", H.ab.ooov, R.ab, optimize=True)
    Int5 += np.einsum("mcef,bejm->bcjf", H.ab.ovvv, R.aa, optimize=True)
    Int5 += np.einsum("cmfe,bejm->bcjf", H.bb.vovv, R.ab, optimize=True)
    Int5 -= np.einsum("bmef,ecjm->bcjf", H.ab.vovv, R.ab, optimize=True)
    I1 = -1.0 * np.einsum(
        "nmfe,bfcjnm->bcje", H.ab.oovv, R.aab, optimize=True
    ) - 0.5 * np.einsum("mnef,bfcjnm->bcje", H.bb.oovv, R.abb, optimize=True)
    X3B += np.einsum("bcje,aeik->abcijk", Int5 + I1, T.ab, optimize=True)
    X3B += np.einsum("bcje,aeik->abcijk", H.ab.vvov, R.ab, optimize=True)
    # Intermediate 6: X2B(bnjk)*Y2B(acin) -> Z3B(abcijk)
    Int6 = -1.0 * np.einsum("mnjk,bm->bnjk", H.ab.oooo, R.a, optimize=True)
    Int6 += np.einsum("bmje,ek->bmjk", H.ab.voov, R.b, optimize=True)
    Int6 += np.einsum("bmek,ej->bmjk", H.ab.vovo, R.a, optimize=True)
    Int6 += np.einsum("bnef,efjk->bnjk", H.ab.vovv, R.ab, optimize=True)
    Int6 += np.einsum("mnek,bejm->bnjk", H.ab.oovo, R.aa, optimize=True)
    Int6 += np.einsum("nmke,bejm->bnjk", H.bb.ooov, R.ab, optimize=True)
    Int6 -= np.einsum("nmje,benk->bmjk", H.ab.ooov, R.ab, optimize=True)
    Int6 += np.einsum("me,bejk->bmjk", Q2, T.ab, optimize=True)
    I1 = np.einsum(
        "nmfe,bfejnk->bmjk", H.ab.oovv, R.aab, optimize=True
    ) + 0.5 * np.einsum("mnef,befjkn->bmjk", H.bb.oovv, R.abb, optimize=True)
    X3B -= np.einsum("bnjk,acin->abcijk", Int6 + I1, T.ab, optimize=True)
    X3B -= np.einsum("bnjk,acin->abcijk", H.ab.vooo, R.ab, optimize=True)

    # additional terms with T3 (these contractions mirror the form of
    # the ones with R3 later on)
    I1 = (
        -1.0 * np.einsum("me,bm->be", H.a.ov, R.a, optimize=True)
        + np.einsum("bnef,fn->be", H.aa.vovv, R.a, optimize=True)
        + np.einsum("bnef,fn->be", H.ab.vovv, R.b, optimize=True)
    )
    I2 = -0.5 * np.einsum("mnef,bfmn->be", H.aa.oovv, R.aa, optimize=True) - np.einsum(
        "mnef,bfmn->be", H.ab.oovv, R.ab, optimize=True
    )
    X3B += 0.5 * np.einsum("be,aecijk->abcijk", I1 + I2, T.aab, optimize=True)

    I1 = (
        -1.0 * np.einsum("me,cm->ce", H.b.ov, R.b, optimize=True)
        + np.einsum("ncfe,fn->ce", H.ab.ovvv, R.a, optimize=True)
        + np.einsum("cnef,fn->ce", H.bb.vovv, R.b, optimize=True)
    )
    I2 = -1.0 * np.einsum(
        "nmfe,fcnm->ce", H.ab.oovv, R.ab, optimize=True
    ) - 0.5 * np.einsum("mnef,fcnm->ce", H.bb.oovv, R.bb, optimize=True)
    X3B += 0.25 * np.einsum("ce,abeijk->abcijk", I1 + I2, T.aab, optimize=True)

    I1 = (
        np.einsum("me,ej->mj", H.a.ov, R.a, optimize=True)
        + np.einsum("mnjf,fn->mj", H.aa.ooov, R.a, optimize=True)
        + np.einsum("mnjf,fn->mj", H.ab.ooov, R.b, optimize=True)
    )
    I2 = 0.5 * np.einsum("mnef,efjn->mj", H.aa.oovv, R.aa, optimize=True) + np.einsum(
        "mnef,efjn->mj", H.ab.oovv, R.ab, optimize=True
    )
    X3B -= 0.5 * np.einsum("mj,abcimk->abcijk", I1 + I2, T.aab, optimize=True)

    I1 = (
        np.einsum("me,ek->mk", H.b.ov, R.b, optimize=True)
        + np.einsum("nmfk,fn->mk", H.ab.oovo, R.a, optimize=True)
        + np.einsum("mnkf,fn->mk", H.bb.ooov, R.b, optimize=True)
    )
    I2 = np.einsum("nmfe,fenk->mk", H.ab.oovv, R.ab, optimize=True) + 0.5 * np.einsum(
        "mnef,efkn->mk", H.bb.oovv, R.bb, optimize=True
    )
    X3B -= 0.25 * np.einsum("mk,abcijm->abcijk", I1 + I2, T.aab, optimize=True)

    I1 = np.einsum("nmje,ek->nmjk", H.ab.ooov, R.b, optimize=True) + np.einsum(
        "nmek,ej->nmjk", H.ab.oovo, R.a, optimize=True
    )
    I2 = np.einsum("mnef,efjk->mnjk", H.ab.oovv, R.ab, optimize=True)
    X3B += 0.5 * np.einsum("nmjk,abcinm->abcijk", I1 + I2, T.aab, optimize=True)

    I1 = np.einsum("mnie,ej->mnij", H.aa.ooov, R.a, optimize=True)
    I1 -= np.transpose(I1, (0, 1, 3, 2))
    I2 = 0.5 * np.einsum("mnef,efij->mnij", H.aa.oovv, R.aa, optimize=True)
    X3B += 0.125 * np.einsum("mnij,abcmnk->abcijk", I1 + I2, T.aab, optimize=True)

    I1 = -1.0 * np.einsum("bmfe,cm->bcfe", H.ab.vovv, R.b, optimize=True) - np.einsum(
        "mcfe,bm->bcfe", H.ab.ovvv, R.a, optimize=True
    )
    I2 = np.einsum("mnef,bcmn->bcef", H.ab.oovv, R.ab, optimize=True)
    X3B += 0.5 * np.einsum("bcfe,afeijk->abcijk", I1 + I2, T.aab, optimize=True)

    I1 = -1.0 * np.einsum("amef,bm->abef", H.aa.vovv, R.a, optimize=True)
    I1 -= np.transpose(I1, (1, 0, 2, 3))
    I2 = 0.5 * np.einsum("mnef,abmn->abef", H.aa.oovv, R.aa, optimize=True)
    X3B += 0.125 * np.einsum("abef,efcijk->abcijk", I1 + I2, T.aab, optimize=True)

    I1 = -1.0 * np.einsum("nmfk,cm->ncfk", H.ab.oovo, R.b, optimize=True) + np.einsum(
        "ncfe,ek->ncfk", H.ab.ovvv, R.b, optimize=True
    )
    I2 = np.einsum("mnef,ecmk->ncfk", H.aa.oovv, R.ab, optimize=True) + np.einsum(
        "nmfe,ecmk->ncfk", H.ab.oovv, R.bb, optimize=True
    )
    X3B += 0.25 * np.einsum("ncfk,abfijn->abcijk", I1 + I2, T.aaa, optimize=True)

    I1 = -1.0 * np.einsum("mnkf,cm->cnkf", H.bb.ooov, R.b, optimize=True) + np.einsum(
        "cnef,ek->cnkf", H.bb.vovv, R.b, optimize=True
    )
    I2 = np.einsum("mnef,ecmk->cnkf", H.ab.oovv, R.ab, optimize=True) + np.einsum(
        "mnef,ecmk->cnkf", H.bb.oovv, R.bb, optimize=True
    )
    X3B += 0.25 * np.einsum("cnkf,abfijn->abcijk", I1 + I2, T.aab, optimize=True)

    I1 = np.einsum("bmfe,ek->bmfk", H.ab.vovv, R.b, optimize=True) - np.einsum(
        "nmfk,bn->bmfk", H.ab.oovo, R.a, optimize=True
    )
    I2 = -1.0 * np.einsum("mnef,bfmk->bnek", H.ab.oovv, R.ab, optimize=True)
    X3B -= 0.5 * np.einsum("bmfk,afcijm->abcijk", I1 + I2, T.aab, optimize=True)

    I1 = -1.0 * np.einsum("nmje,cm->ncje", H.ab.ooov, R.b, optimize=True) + np.einsum(
        "ncfe,fj->ncje", H.ab.ovvv, R.a, optimize=True
    )
    I2 = -1.0 * np.einsum("mnef,ecjn->mcjf", H.ab.oovv, R.ab, optimize=True)
    X3B -= 0.5 * np.einsum("ncje,abeink->abcijk", I1 + I2, T.aab, optimize=True)

    I1 = -1.0 * np.einsum("nmje,bn->bmje", H.aa.ooov, R.a, optimize=True) + np.einsum(
        "bmfe,fj->bmje", H.aa.vovv, R.a, optimize=True
    )
    I2 = np.einsum("mnef,aeim->anif", H.aa.oovv, R.aa, optimize=True) + np.einsum(
        "nmfe,aeim->anif", H.ab.oovv, R.ab, optimize=True
    )
    X3B += np.einsum("bmje,aecimk->abcijk", I1 + I2, T.aab, optimize=True)

    I1 = -1.0 * np.einsum("nmje,bn->bmje", H.ab.ooov, R.a, optimize=True) + np.einsum(
        "bmfe,fj->bmje", H.ab.vovv, R.a, optimize=True
    )
    I2 = np.einsum("mnef,aeim->anif", H.ab.oovv, R.aa, optimize=True) + np.einsum(
        "mnef,aeim->anif", H.bb.oovv, R.ab, optimize=True
    )
    X3B += np.einsum("bmje,aecimk->abcijk", I1 + I2, T.abb, optimize=True)

    # < ijk~abc~ | (HR3)_C | 0 >
    X3B -= 0.5 * np.einsum("mj,abcimk->abcijk", H.a.oo, R.aab, optimize=True)
    X3B -= 0.25 * np.einsum("mk,abcijm->abcijk", H.b.oo, R.aab, optimize=True)
    X3B += 0.5 * np.einsum("be,aecijk->abcijk", H.a.vv, R.aab, optimize=True)
    X3B += 0.25 * np.einsum("ce,abeijk->abcijk", H.b.vv, R.aab, optimize=True)
    X3B += 0.125 * np.einsum("mnij,abcmnk->abcijk", H.aa.oooo, R.aab, optimize=True)
    X3B += 0.5 * np.einsum("mnjk,abcimn->abcijk", H.ab.oooo, R.aab, optimize=True)
    X3B += 0.125 * np.einsum("abef,efcijk->abcijk", H.aa.vvvv, R.aab, optimize=True)
    X3B += 0.5 * np.einsum("bcef,aefijk->abcijk", H.ab.vvvv, R.aab, optimize=True)
    X3B += np.einsum("amie,ebcmjk->abcijk", H.aa.voov, R.aab, optimize=True)
    X3B += np.einsum("amie,becjmk->abcijk", H.ab.voov, R.abb, optimize=True)
    X3B += 0.25 * np.einsum("mcek,abeijm->abcijk", H.ab.ovvo, R.aaa, optimize=True)
    X3B += 0.25 * np.einsum("cmke,abeijm->abcijk", H.bb.voov, R.aab, optimize=True)
    X3B -= 0.5 * np.einsum("bmek,aecijm->abcijk", H.ab.vovo, R.aab, optimize=True)
    X3B -= 0.5 * np.einsum("mcje,abeimk->abcijk", H.ab.ovov, R.aab, optimize=True)

    X3B -= (
        np.transpose(X3B, (0, 1, 2, 4, 3, 5))
        + np.transpose(X3B, (1, 0, 2, 3, 4, 5))
        - np.transpose(X3B, (1, 0, 2, 4, 3, 5))
    )
    return X3B


def build_HR_3C(R, T, H):

    # < ij~k~ab~c~ | [ H(R1+R2) ]_C | 0 >
    Q1 = np.einsum("mnef,fn->me", H.bb.oovv, R.b, optimize=True) + np.einsum(
        "nmfe,fn->me", H.ab.oovv, R.a, optimize=True
    )
    Q2 = np.einsum("mnef,fn->me", H.ab.oovv, R.b, optimize=True) + np.einsum(
        "nmfe,fn->me", H.aa.oovv, R.a, optimize=True
    )
    # Intermediate 1: X2B(cbke)*Y2C(aeij) -> Z3C(cbakji)
    Int1 = -1.0 * np.einsum("cmke,bm->cbke", H.ab.voov, R.b, optimize=True)
    Int1 -= np.einsum("mbke,cm->cbke", H.ab.ovov, R.a, optimize=True)
    Int1 += np.einsum("cbef,ek->cbkf", H.ab.vvvv, R.a, optimize=True)
    Int1 += np.einsum("nmke,cbnm->cbke", H.ab.ooov, R.ab, optimize=True)
    Int1 += np.einsum("bmfe,cekm->cbkf", H.bb.vovv, R.ab, optimize=True)
    Int1 += np.einsum("mbef,ecmk->cbkf", H.ab.ovvv, R.aa, optimize=True)
    Int1 -= np.einsum("cmef,ebkm->cbkf", H.ab.vovv, R.ab, optimize=True)
    I1 = -0.5 * np.einsum(
        "mnef,cfbknm->cbke", H.bb.oovv, R.abb, optimize=True
    ) - np.einsum("nmfe,cfbknm->cbke", H.ab.oovv, R.aab, optimize=True)
    X3C = 0.5 * np.einsum("cbke,aeij->cbakji", Int1 + I1, T.bb, optimize=True)
    X3C += 0.5 * np.einsum("cbke,aeij->cbakji", H.ab.vvov, R.bb, optimize=True)
    # Intermediate 2: X2B(cnkj)*Y2C(abin) -> Z3C(cbakji)
    Int2 = -1.0 * np.einsum("mnkj,cm->cnkj", H.ab.oooo, R.a, optimize=True)
    Int2 += np.einsum("cmej,ek->cmkj", H.ab.vovo, R.a, optimize=True)
    Int2 += np.einsum("cmke,ej->cmkj", H.ab.voov, R.b, optimize=True)
    Int2 += np.einsum("cmfe,fekj->cmkj", H.ab.vovv, R.ab, optimize=True)
    Int2 += np.einsum("nmje,cekm->cnkj", H.bb.ooov, R.ab, optimize=True)
    Int2 += np.einsum("mnej,ecmk->cnkj", H.ab.oovo, R.aa, optimize=True)
    Int2 -= np.einsum("mnke,cemj->cnkj", H.ab.ooov, R.ab, optimize=True)
    I1 = 0.5 * np.einsum(
        "mnef,cfeknj->cmkj", H.bb.oovv, R.abb, optimize=True
    ) + np.einsum("nmfe,cfeknj->cmkj", H.ab.oovv, R.aab, optimize=True)
    X3C -= 0.5 * np.einsum("cnkj,abin->cbakji", Int2 + I1, T.bb, optimize=True)
    X3C -= 0.5 * np.einsum("cmkj,abim->cbakji", H.ab.vooo, R.bb, optimize=True)
    # Intermediate 3: X2C(abej)*Y2B(ceki) -> Z3C(cbakji)
    Int3 = np.einsum(
        "amje,bm->abej", H.bb.voov, R.b, optimize=True
    )  # (*) flipped sign to use H2A(voov) instead of H2A(vovo)
    Int3 += 0.5 * np.einsum(
        "abfe,ej->abfj", H.bb.vvvv, R.b, optimize=True
    )  # (*) added factor 1/2 to compensate A(ab)
    Int3 += 0.25 * np.einsum(
        "nmje,abmn->abej", H.bb.ooov, R.bb, optimize=True
    )  # (*) added factor 1/2 to compensate A(ab)
    Int3 += np.einsum("amfe,bejm->abfj", H.bb.vovv, R.bb, optimize=True)
    Int3 += np.einsum("maef,ebmj->abfj", H.ab.ovvv, R.ab, optimize=True)
    Int3 -= 0.5 * np.einsum(
        "me,abmj->abej", Q1, T.bb, optimize=True
    )  # (*) added factor 1/2 to compensate A(ab)
    Int3 -= np.transpose(Int3, (1, 0, 2, 3))
    I1 = -0.5 * np.einsum(
        "mnef,abfmjn->abej", H.bb.oovv, R.bbb, optimize=True
    ) - np.einsum("nmfe,fbanjm->abej", H.ab.oovv, R.abb, optimize=True)
    X3C += 0.5 * np.einsum("abej,ceki->cbakji", Int3 + I1, T.ab, optimize=True)
    X3C += 0.5 * np.einsum("baje,ceki->cbakji", H.bb.vvov, R.ab, optimize=True)
    # Intermediate 4: X2C(bnji)*Y2B(cakn) -> Z3C(cbakji)
    Int4 = -0.5 * np.einsum(
        "nmij,bm->bnji", H.bb.oooo, R.b, optimize=True
    )  # (*) added factor 1/2 to compenate A(ij)
    Int4 -= np.einsum(
        "bmie,ej->bmji", H.bb.voov, R.b, optimize=True
    )  # (*) flipped sign to use H2A(voov) instead of H2A(vovo)
    Int4 += 0.25 * np.einsum(
        "bmfe,efij->bmji", H.bb.vovv, R.bb, optimize=True
    )  # (*) added factor 1/2 to compensate A(ij)
    Int4 += np.einsum("nmie,bejm->bnji", H.bb.ooov, R.bb, optimize=True)
    Int4 += np.einsum("mnei,ebmj->bnji", H.ab.oovo, R.ab, optimize=True)
    Int4 += 0.5 * np.einsum(
        "me,ebij->bmji", Q1, T.bb, optimize=True
    )  # (*) added factor 1/2 to compensate A(ij)
    Int4 -= np.transpose(Int4, (0, 1, 3, 2))
    I1 = 0.5 * np.einsum(
        "mnef,aefijn->amij", H.bb.oovv, R.bbb, optimize=True
    ) + np.einsum("nmfe,feanji->amij", H.ab.oovv, R.abb, optimize=True)
    X3C -= 0.5 * np.einsum("bnji,cakn->cbakji", Int4 + I1, T.ab, optimize=True)
    X3C -= 0.5 * np.einsum("bnji,cakn->cbakji", H.bb.vooo, R.ab, optimize=True)
    # Intermediate 5: X2B(cbej)*Y2B(eaki) -> Z3C(cbakji)
    Int5 = -1.0 * np.einsum("cmej,bm->cbej", H.ab.vovo, R.b, optimize=True)
    Int5 -= np.einsum("mbej,cm->cbej", H.ab.ovvo, R.a, optimize=True)
    Int5 += np.einsum("cbfe,ej->cbfj", H.ab.vvvv, R.b, optimize=True)
    Int5 += np.einsum("nmfj,cbnm->cbfj", H.ab.oovo, R.ab, optimize=True)
    Int5 += np.einsum("cmfe,bejm->cbfj", H.ab.vovv, R.bb, optimize=True)
    Int5 += np.einsum("cmfe,ebmj->cbfj", H.aa.vovv, R.ab, optimize=True)
    Int5 -= np.einsum("mbfe,cemj->cbfj", H.ab.ovvv, R.ab, optimize=True)
    I1 = -1.0 * np.einsum(
        "mnef,cfbmnj->cbej", H.ab.oovv, R.abb, optimize=True
    ) - 0.5 * np.einsum("mnef,cfbmnj->cbej", H.aa.oovv, R.aab, optimize=True)
    X3C += np.einsum("cbej,eaki->cbakji", Int5 + I1, T.ab, optimize=True)
    X3C += np.einsum("cbej,eaki->cbakji", H.ab.vvvo, R.ab, optimize=True)
    # Intermediate 6: X2B(nbkj)*Y2B(cani) -> Z3C(cbakji)
    Int6 = -1.0 * np.einsum("nmkj,bm->nbkj", H.ab.oooo, R.b, optimize=True)
    Int6 += np.einsum("mbej,ek->mbkj", H.ab.ovvo, R.a, optimize=True)
    Int6 += np.einsum("mbke,ej->mbkj", H.ab.ovov, R.b, optimize=True)
    Int6 += np.einsum("nbfe,fekj->nbkj", H.ab.ovvv, R.ab, optimize=True)
    Int6 += np.einsum("nmke,bejm->nbkj", H.ab.ooov, R.bb, optimize=True)
    Int6 += np.einsum("nmke,ebmj->nbkj", H.aa.ooov, R.ab, optimize=True)
    Int6 -= np.einsum("mnej,ebkn->mbkj", H.ab.oovo, R.ab, optimize=True)
    Int6 += np.einsum("me,ebkj->mbkj", Q2, T.ab, optimize=True)
    I1 = np.einsum(
        "mnef,efbknj->mbkj", H.ab.oovv, R.abb, optimize=True
    ) + 0.5 * np.einsum("mnef,febnkj->mbkj", H.aa.oovv, R.aab, optimize=True)
    X3C -= np.einsum("nbkj,cani->cbakji", Int6 + I1, T.ab, optimize=True)
    X3C -= np.einsum("nbkj,cani->cbakji", H.ab.ovoo, R.ab, optimize=True)

    # additional terms with T3
    I1 = (
        -1.0 * np.einsum("me,bm->be", H.b.ov, R.b, optimize=True)
        + np.einsum("bnef,fn->be", H.bb.vovv, R.b, optimize=True)
        + np.einsum("nbfe,fn->be", H.ab.ovvv, R.a, optimize=True)
    )
    I2 = -0.5 * np.einsum("mnef,bfmn->be", H.bb.oovv, R.bb, optimize=True) - np.einsum(
        "nmfe,fbnm->be", H.ab.oovv, R.ab, optimize=True
    )
    X3C += 0.5 * np.einsum("be,ceakji->cbakji", I1 + I2, T.abb, optimize=True)

    I1 = (
        -1.0 * np.einsum("me,cm->ce", H.a.ov, R.a, optimize=True)
        + np.einsum("cnef,fn->ce", H.ab.vovv, R.b, optimize=True)
        + np.einsum("cnef,fn->ce", H.aa.vovv, R.a, optimize=True)
    )
    I2 = -1.0 * np.einsum(
        "mnef,cfmn->ce", H.ab.oovv, R.ab, optimize=True
    ) - 0.5 * np.einsum("mnef,fcnm->ce", H.aa.oovv, R.aa, optimize=True)
    X3C += 0.25 * np.einsum("ce,ebakji->cbakji", I1 + I2, T.abb, optimize=True)

    I1 = (
        np.einsum("me,ej->mj", H.b.ov, R.b, optimize=True)
        + np.einsum("mnjf,fn->mj", H.bb.ooov, R.b, optimize=True)
        + np.einsum("nmfj,fn->mj", H.ab.oovo, R.a, optimize=True)
    )
    I2 = 0.5 * np.einsum("mnef,efjn->mj", H.bb.oovv, R.bb, optimize=True) + np.einsum(
        "nmfe,fenj->mj", H.ab.oovv, R.ab, optimize=True
    )
    X3C -= 0.5 * np.einsum("mj,cbakmi->cbakji", I1 + I2, T.abb, optimize=True)

    I1 = (
        np.einsum("me,ek->mk", H.a.ov, R.a, optimize=True)
        + np.einsum("mnkf,fn->mk", H.ab.ooov, R.b, optimize=True)
        + np.einsum("mnkf,fn->mk", H.aa.ooov, R.a, optimize=True)
    )
    I2 = np.einsum("mnef,efkn->mk", H.ab.oovv, R.ab, optimize=True) + 0.5 * np.einsum(
        "mnef,efkn->mk", H.aa.oovv, R.aa, optimize=True
    )
    X3C -= 0.25 * np.einsum("mk,cbamji->cbakji", I1 + I2, T.abb, optimize=True)

    I1 = np.einsum("mnej,ek->mnkj", H.ab.oovo, R.a, optimize=True) + np.einsum(
        "mnke,ej->mnkj", H.ab.ooov, R.b, optimize=True
    )
    I2 = np.einsum("nmfe,fekj->nmkj", H.ab.oovv, R.ab, optimize=True)
    X3C += 0.5 * np.einsum("mnkj,cbamni->cbakji", I1 + I2, T.abb, optimize=True)

    I1 = np.einsum("mnie,ej->mnij", H.bb.ooov, R.b, optimize=True)
    I1 -= np.transpose(I1, (0, 1, 3, 2))
    I2 = 0.5 * np.einsum("mnef,efij->mnij", H.bb.oovv, R.bb, optimize=True)
    X3C += 0.125 * np.einsum("mnij,cbaknm->cbakji", I1 + I2, T.abb, optimize=True)

    I1 = -1.0 * np.einsum("mbef,cm->cbef", H.ab.ovvv, R.a, optimize=True) - np.einsum(
        "cmef,bm->cbef", H.ab.vovv, R.b, optimize=True
    )
    I2 = np.einsum("nmfe,cbnm->cbfe", H.ab.oovv, R.ab, optimize=True)
    X3C += 0.5 * np.einsum("cbef,efakji->cbakji", I1 + I2, T.abb, optimize=True)

    I1 = -1.0 * np.einsum("amef,bm->abef", H.bb.vovv, R.b, optimize=True)
    I1 -= np.transpose(I1, (1, 0, 2, 3))
    I2 = 0.5 * np.einsum("mnef,abmn->abef", H.bb.oovv, R.bb, optimize=True)
    X3C += 0.125 * np.einsum("abef,cfekji->cbakji", I1 + I2, T.abb, optimize=True)

    I1 = -1.0 * np.einsum("mnkf,cm->cnkf", H.ab.ooov, R.a, optimize=True) + np.einsum(
        "cnef,ek->cnkf", H.ab.vovv, R.a, optimize=True
    )
    I2 = np.einsum("mnef,cekm->cnkf", H.bb.oovv, R.ab, optimize=True) + np.einsum(
        "mnef,ecmk->cnkf", H.ab.oovv, R.aa, optimize=True
    )
    X3C += 0.25 * np.einsum("cnkf,abfijn->cbakji", I1 + I2, T.bbb, optimize=True)

    I1 = -1.0 * np.einsum("mnkf,cm->ncfk", H.aa.ooov, R.a, optimize=True) + np.einsum(
        "cnef,ek->ncfk", H.aa.vovv, R.a, optimize=True
    )
    I2 = np.einsum("nmfe,cekm->ncfk", H.ab.oovv, R.ab, optimize=True) + np.einsum(
        "mnef,ecmk->ncfk", H.aa.oovv, R.aa, optimize=True
    )
    X3C += 0.25 * np.einsum("ncfk,fbanji->cbakji", I1 + I2, T.abb, optimize=True)

    I1 = np.einsum("mbef,ek->mbkf", H.ab.ovvv, R.a, optimize=True) - np.einsum(
        "mnkf,bn->mbkf", H.ab.ooov, R.b, optimize=True
    )
    I2 = -1.0 * np.einsum("nmfe,fbkm->nbke", H.ab.oovv, R.ab, optimize=True)
    X3C -= 0.5 * np.einsum("mbkf,cfamji->cbakji", I1 + I2, T.abb, optimize=True)

    I1 = -1.0 * np.einsum("mnej,cm->cnej", H.ab.oovo, R.a, optimize=True) + np.einsum(
        "cnef,fj->cnej", H.ab.vovv, R.b, optimize=True
    )
    I2 = -1.0 * np.einsum("nmfe,cenj->cmfj", H.ab.oovv, R.ab, optimize=True)
    X3C -= 0.5 * np.einsum("cnej,ebakni->cbakji", I1 + I2, T.abb, optimize=True)

    I1 = -1.0 * np.einsum("nmje,bn->mbej", H.bb.ooov, R.b, optimize=True) + np.einsum(
        "bmfe,fj->mbej", H.bb.vovv, R.b, optimize=True
    )
    I2 = np.einsum("mnef,aeim->nafi", H.bb.oovv, R.bb, optimize=True) + np.einsum(
        "mnef,eami->nafi", H.ab.oovv, R.ab, optimize=True
    )
    X3C += np.einsum("mbej,ceakmi->cbakji", I1 + I2, T.abb, optimize=True)

    I1 = -1.0 * np.einsum("mnej,bn->mbej", H.ab.oovo, R.b, optimize=True) + np.einsum(
        "mbef,fj->mbej", H.ab.ovvv, R.b, optimize=True
    )
    I2 = np.einsum("nmfe,aeim->nafi", H.ab.oovv, R.bb, optimize=True) + np.einsum(
        "mnef,eami->nafi", H.aa.oovv, R.ab, optimize=True
    )
    X3C += np.einsum("mbej,ceakmi->cbakji", I1 + I2, T.aab, optimize=True)

    # < ijk~abc~ | (HR3)_C | 0 >
    X3C -= 0.5 * np.einsum("mj,cbakmi->cbakji", H.b.oo, R.abb, optimize=True)
    X3C -= 0.25 * np.einsum("mk,cbamji->cbakji", H.a.oo, R.abb, optimize=True)
    X3C += 0.5 * np.einsum("be,ceakji->cbakji", H.b.vv, R.abb, optimize=True)
    X3C += 0.25 * np.einsum("ce,ebakji->cbakji", H.a.vv, R.abb, optimize=True)
    X3C += 0.125 * np.einsum("mnij,cbaknm->cbakji", H.bb.oooo, R.abb, optimize=True)
    X3C += 0.5 * np.einsum("nmkj,cbanmi->cbakji", H.ab.oooo, R.abb, optimize=True)
    X3C += 0.125 * np.einsum("abef,cfekji->cbakji", H.bb.vvvv, R.abb, optimize=True)
    X3C += 0.5 * np.einsum("cbfe,feakji->cbakji", H.ab.vvvv, R.abb, optimize=True)
    X3C += np.einsum("amie,cbekjm->cbakji", H.bb.voov, R.abb, optimize=True)
    X3C += np.einsum("maei,cebkmj->cbakji", H.ab.ovvo, R.aab, optimize=True)
    X3C += 0.25 * np.einsum("cmke,ebamji->cbakji", H.ab.voov, R.bbb, optimize=True)
    X3C += 0.25 * np.einsum("cmke,ebamji->cbakji", H.aa.voov, R.abb, optimize=True)
    X3C -= 0.5 * np.einsum("mbke,ceamji->cbakji", H.ab.ovov, R.abb, optimize=True)
    X3C -= 0.5 * np.einsum("cmej,ebakmi->cbakji", H.ab.vovo, R.abb, optimize=True)

    X3C -= (
        np.transpose(X3C, (0, 1, 2, 3, 5, 4))
        + np.transpose(X3C, (0, 2, 1, 3, 4, 5))
        - np.transpose(X3C, (0, 2, 1, 3, 5, 4))
    )
    return X3C


def build_HR_3D(R, T, H):
    
    # <i~j~k~a~b~c~| [H(R1+R2)]_C | 0 >
    Q1 = np.einsum("mnef,fn->me", H.bb.oovv, R.b, optimize=True)
    Q1 += np.einsum("nmfe,fn->me", H.ab.oovv, R.a, optimize=True)
    I1 = np.einsum("amje,bm->abej", H.bb.voov, R.b, optimize=True)
    I1 += np.einsum("amfe,bejm->abfj", H.bb.vovv, R.bb, optimize=True)
    I1 += np.einsum("maef,ebmj->abfj", H.ab.ovvv, R.ab, optimize=True)
    I1 -= np.transpose(I1, (1, 0, 2, 3))
    I2 = np.einsum("abfe,ej->abfj", H.bb.vvvv, R.b, optimize=True)
    I2 += 0.5 * np.einsum("nmje,abmn->abej", H.bb.ooov, R.bb, optimize=True)
    I2 -= np.einsum("me,abmj->abej", Q1, T.bb, optimize=True)
    I3 = -0.5 * np.einsum(
        "mnef,abfimn->baei", H.bb.oovv, R.bbb, optimize=True
    ) - np.einsum("nmfe,fbanmi->baei", H.ab.oovv, R.abb, optimize=True)
    X3D = 0.25 * np.einsum(
        "abej,ecik->abcijk", I1 + I2 + I3, T.bb, optimize=True
    )
    X3D += 0.25 * np.einsum("baje,ecik->abcijk", H.bb.vvov, R.bb, optimize=True)

    I1 = -np.einsum("bmie,ej->mbij", H.bb.voov, R.b, optimize=True)
    I1 += np.einsum("nmie,bejm->nbij", H.bb.ooov, R.bb, optimize=True)
    I1 += np.einsum("mnei,ebmj->nbij", H.ab.oovo, R.ab, optimize=True)
    I1 -= np.transpose(I1, (0, 1, 3, 2))
    I2 = -1.0 * np.einsum("nmij,bm->nbij", H.bb.oooo, R.b, optimize=True)
    I2 += 0.5 * np.einsum("bmfe,efij->mbij", H.bb.vovv, R.bb, optimize=True)
    I3 = 0.5 * np.einsum(
        "mnef,efcjnk->mcjk", H.bb.oovv, R.bbb, optimize=True
    ) + np.einsum("nmfe,fcenkj->mcjk", H.ab.oovv, R.abb, optimize=True)
    X3D -= 0.25 * np.einsum(
        "mbij,acmk->abcijk", I1 + I2 + I3, T.bb, optimize=True
    )
    X3D -= 0.25 * np.einsum("bmji,acmk->abcijk", H.bb.vooo, R.bb, optimize=True)

    # additional terms with T3 in <ijkabc|[ H(R1+R2)]_C | 0>
    I1 = (
        -1.0 * np.einsum("me,bm->be", H.b.ov, R.b, optimize=True)
        + np.einsum("bnef,fn->be", H.bb.vovv, R.b, optimize=True)
        + np.einsum("nbfe,fn->be", H.ab.ovvv, R.a, optimize=True)
    )
    I2 = -0.5 * np.einsum("mnef,bfmn->be", H.bb.oovv, R.bb, optimize=True) - np.einsum(
        "nmfe,fbnm->be", H.ab.oovv, R.ab, optimize=True
    )
    X3D += (1.0 / 12.0) * np.einsum(
        "be,aecijk->abcijk", I1 + I2, T.bbb, optimize=True
    )  # A(b/ac)

    I1 = (
        np.einsum("me,ej->mj", H.b.ov, R.b, optimize=True)
        + np.einsum("mnjf,fn->mj", H.bb.ooov, R.b, optimize=True)
        + np.einsum("nmfj,fn->mj", H.ab.oovo, R.a, optimize=True)
    )
    I2 = 0.5 * np.einsum("mnef,efjn->mj", H.bb.oovv, R.bb, optimize=True) + np.einsum(
        "nmfe,fenj->mj", H.ab.oovv, R.ab, optimize=True
    )
    X3D -= (1.0 / 12.0) * np.einsum(
        "mj,abcimk->abcijk", I1 + I2, T.bbb, optimize=True
    )  # A(j/ik)

    I1 = np.einsum("nmje,ei->mnij", H.bb.ooov, R.b, optimize=True)
    I1 -= np.transpose(I1, (0, 1, 3, 2))
    I2 = 0.5 * np.einsum("mnef,efij->mnij", H.bb.oovv, R.bb, optimize=True)
    X3D += (1.0 / 24.0) * np.einsum(
        "mnij,abcmnk->abcijk", I1 + I2, T.bbb, optimize=True
    )  # A(k/ij)

    I1 = -1.0 * np.einsum("amef,bm->abef", H.bb.vovv, R.b, optimize=True)
    I1 -= np.transpose(I1, (1, 0, 2, 3))
    I2 = 0.5 * np.einsum("mnef,abmn->abef", H.bb.oovv, R.bb, optimize=True)
    X3D += (1.0 / 24.0) * np.einsum(
        "abef,efcijk->abcijk", I1 + I2, T.bbb, optimize=True
    )  # A(c/ab)

    I1 = -1.0 * np.einsum("nmje,bn->bmje", H.bb.ooov, R.b, optimize=True) + np.einsum(
        "bmfe,fj->bmje", H.bb.vovv, R.b, optimize=True
    )
    I2 = np.einsum("mnef,fcnk->cmke", H.bb.oovv, R.bb, optimize=True) + np.einsum(
        "nmfe,fcnk->cmke", H.ab.oovv, R.ab, optimize=True
    )
    X3D += 0.25 * np.einsum(
        "bmje,aecimk->abcijk", I1 + I2, T.bbb, optimize=True
    )  # A(j/ik)A(b/ac)

    I1 = -1.0 * np.einsum("mnej,bn->bmje", H.ab.oovo, R.b, optimize=True) + np.einsum(
        "mbef,fj->bmje", H.ab.ovvv, R.b, optimize=True
    )
    I2 = np.einsum("mnef,fcnk->cmke", H.ab.oovv, R.bb, optimize=True) + np.einsum(
        "mnef,fcnk->cmke", H.aa.oovv, R.ab, optimize=True
    )
    X3D += 0.25 * np.einsum(
        "bmje,ecamki->abcijk", I1 + I2, T.abb, optimize=True
    )  # A(j/ik)A(b/ac)

    # < i~j~k~a~b~c~ | (HR3)_C | 0 >
    X3D -= (1.0 / 12.0) * np.einsum("mj,abcimk->abcijk", H.b.oo, R.bbb, optimize=True)
    X3D += (1.0 / 12.0) * np.einsum("be,aecijk->abcijk", H.b.vv, R.bbb, optimize=True)
    X3D += (1.0 / 24.0) * np.einsum(
        "mnij,abcmnk->abcijk", H.bb.oooo, R.bbb, optimize=True
    )
    X3D += (1.0 / 24.0) * np.einsum(
        "abef,efcijk->abcijk", H.bb.vvvv, R.bbb, optimize=True
    )
    X3D += 0.25 * np.einsum("amie,ebcmjk->abcijk", H.bb.voov, R.bbb, optimize=True)
    X3D += 0.25 * np.einsum("maei,ecbmkj->abcijk", H.ab.ovvo, R.abb, optimize=True)

    # antisymmetrize terms and add up: A(abc)A(ijk) = A(a/bc)A(bc)A(i/jk)A(jk)
    X3D -= np.transpose(X3D, (0, 1, 2, 3, 5, 4))
    X3D -= np.transpose(X3D, (0, 1, 2, 4, 3, 5)) + np.transpose(X3D, (0, 1, 2, 5, 4, 3))
    X3D -= np.transpose(X3D, (0, 2, 1, 3, 4, 5))
    X3D -= np.transpose(X3D, (1, 0, 2, 3, 4, 5)) + np.transpose(X3D, (2, 1, 0, 3, 4, 5))
    return X3D
