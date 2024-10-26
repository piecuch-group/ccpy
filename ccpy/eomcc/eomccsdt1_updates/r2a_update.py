import numpy as np

from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops

def update(R, omega, H):

    R.aa = eomcc_active_loops.update_r2a(R.aa, omega, H.a.oo, H.a.vv, H.b.oo, H.b.vv, 0.0)

    return R


def build(dR, R, T, H, X, system):
    
    x2 = build_eomccsd(R, T, H)   # base EOMCCSD part (separately antisymmetrized)
    
    # Add on T3 parts
    dR = build_1111(dR, R, T, H, X, system)
    dR = build_1101(dR, R, T, H, X, system)
    dR = build_1011(dR, R, T, H, X, system)
    dR = build_1100(dR, R, T, H, X, system)
    dR = build_0011(dR, R, T, H, X, system)
    dR = build_1001(dR, R, T, H, X, system)
    dR = build_1000(dR, R, T, H, X, system)
    dR = build_0001(dR, R, T, H, X, system)
    dR = build_0000(dR, R, T, H, X, system)
    dR.aa += x2
    
    return dR

def build_eomccsd(R, T, H):
    
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

   
def build_1111(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.aa[Va, Va, Oa, Oa] = (1.0 / 4.0) * (
            -1.0 * np.einsum('me,BAemIJ->ABIJ', X.a.ov[oa, va], T.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('Me,BAeIJM->ABIJ', X.a.ov[Oa, va], T.aaa.VVvOOO, optimize=True)
            - 1.0 * np.einsum('mE,EBAmIJ->ABIJ', X.a.ov[oa, Va], T.aaa.VVVoOO, optimize=True)
            - 1.0 * np.einsum('ME,EBAIJM->ABIJ', X.a.ov[Oa, Va], T.aaa.VVVOOO, optimize=True)
    )
    dR.aa[Va, Va, Oa, Oa] += (1.0 / 4.0) * (
            - 1.0 * np.einsum('me,BAeIJm->ABIJ', X.b.ov[ob, vb], T.aab.VVvOOo, optimize=True)
            - 1.0 * np.einsum('Me,BAeIJM->ABIJ', X.b.ov[Ob, vb], T.aab.VVvOOO, optimize=True)
            - 1.0 * np.einsum('mE,BAEIJm->ABIJ', X.b.ov[ob, Vb], T.aab.VVVOOo, optimize=True)
            - 1.0 * np.einsum('ME,BAEIJM->ABIJ', X.b.ov[Ob, Vb], T.aab.VVVOOO, optimize=True)
    )

    dR.aa[Va, Va, Oa, Oa] += (1.0 / 4.0) * (
            -1.0 * np.einsum('me,BAemIJ->ABIJ', H.a.ov[oa, va], R.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('Me,BAeIJM->ABIJ', H.a.ov[Oa, va], R.aaa.VVvOOO, optimize=True)
            - 1.0 * np.einsum('mE,EBAmIJ->ABIJ', H.a.ov[oa, Va], R.aaa.VVVoOO, optimize=True)
            - 1.0 * np.einsum('ME,EBAIJM->ABIJ', H.a.ov[Oa, Va], R.aaa.VVVOOO, optimize=True)
    )
    dR.aa[Va, Va, Oa, Oa] += (1.0 / 4.0) * (
            - 1.0 * np.einsum('me,BAeIJm->ABIJ', H.b.ov[ob, vb], R.aab.VVvOOo, optimize=True)
            - 1.0 * np.einsum('Me,BAeIJM->ABIJ', H.b.ov[Ob, vb], R.aab.VVvOOO, optimize=True)
            - 1.0 * np.einsum('mE,BAEIJm->ABIJ', H.b.ov[ob, Vb], R.aab.VVVOOo, optimize=True)
            - 1.0 * np.einsum('ME,BAEIJM->ABIJ', H.b.ov[Ob, Vb], R.aab.VVVOOO, optimize=True)
    )

    dR.aa[Va, Va, Oa, Oa] += (2.0 / 4.0) * (
            + 1.0 * np.einsum('mnIf,BAfmJn->ABIJ', H.ab.ooov[oa, ob, Oa, vb], R.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('mNIf,BAfmJN->ABIJ', H.ab.ooov[oa, Ob, Oa, vb], R.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('MnIf,BAfMJn->ABIJ', H.ab.ooov[Oa, ob, Oa, vb], R.aab.VVvOOo, optimize=True)
            + 1.0 * np.einsum('MNIf,BAfMJN->ABIJ', H.ab.ooov[Oa, Ob, Oa, vb], R.aab.VVvOOO, optimize=True)
            + 1.0 * np.einsum('mnIF,BAFmJn->ABIJ', H.ab.ooov[oa, ob, Oa, Vb], R.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('mNIF,BAFmJN->ABIJ', H.ab.ooov[oa, Ob, Oa, Vb], R.aab.VVVoOO, optimize=True)
            + 1.0 * np.einsum('MnIF,BAFMJn->ABIJ', H.ab.ooov[Oa, ob, Oa, Vb], R.aab.VVVOOo, optimize=True)
            + 1.0 * np.einsum('MNIF,BAFMJN->ABIJ', H.ab.ooov[Oa, Ob, Oa, Vb], R.aab.VVVOOO, optimize=True)
    )
    dR.aa[Va, Va, Oa, Oa] += (2.0 / 4.0) * (
            -0.5 * np.einsum('mnIf,BAfmnJ->ABIJ', H.aa.ooov[oa, oa, Oa, va], R.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('MnIf,BAfnMJ->ABIJ', H.aa.ooov[Oa, oa, Oa, va], R.aaa.VVvoOO, optimize=True)
            + 0.5 * np.einsum('MNIf,BAfMJN->ABIJ', H.aa.ooov[Oa, Oa, Oa, va], R.aaa.VVvOOO, optimize=True)
            - 0.5 * np.einsum('mnIF,FBAmnJ->ABIJ', H.aa.ooov[oa, oa, Oa, Va], R.aaa.VVVooO, optimize=True)
            + 1.0 * np.einsum('MnIF,FBAnMJ->ABIJ', H.aa.ooov[Oa, oa, Oa, Va], R.aaa.VVVoOO, optimize=True)
            + 0.5 * np.einsum('MNIF,FBAMJN->ABIJ', H.aa.ooov[Oa, Oa, Oa, Va], R.aaa.VVVOOO, optimize=True)
    )
    dR.aa[Va, Va, Oa, Oa] += (2.0 / 4.0) * (
            +0.5 * np.einsum('Anef,BfenIJ->ABIJ', H.aa.vovv[Va, oa, va, va], R.aaa.VvvoOO, optimize=True)
            + 0.5 * np.einsum('ANef,BfeIJN->ABIJ', H.aa.vovv[Va, Oa, va, va], R.aaa.VvvOOO, optimize=True)
            - 1.0 * np.einsum('AnEf,BEfnIJ->ABIJ', H.aa.vovv[Va, oa, Va, va], R.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('ANEf,BEfIJN->ABIJ', H.aa.vovv[Va, Oa, Va, va], R.aaa.VVvOOO, optimize=True)
            - 0.5 * np.einsum('AnEF,FBEnIJ->ABIJ', H.aa.vovv[Va, oa, Va, Va], R.aaa.VVVoOO, optimize=True)
            - 0.5 * np.einsum('ANEF,FBEIJN->ABIJ', H.aa.vovv[Va, Oa, Va, Va], R.aaa.VVVOOO, optimize=True)
    )
    dR.aa[Va, Va, Oa, Oa] += (2.0 / 4.0) * (
            -1.0 * np.einsum('Anef,BefIJn->ABIJ', H.ab.vovv[Va, ob, va, vb], R.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('ANef,BefIJN->ABIJ', H.ab.vovv[Va, Ob, va, vb], R.aab.VvvOOO, optimize=True)
            - 1.0 * np.einsum('AneF,BeFIJn->ABIJ', H.ab.vovv[Va, ob, va, Vb], R.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('ANeF,BeFIJN->ABIJ', H.ab.vovv[Va, Ob, va, Vb], R.aab.VvVOOO, optimize=True)
            - 1.0 * np.einsum('AnEf,BEfIJn->ABIJ', H.ab.vovv[Va, ob, Va, vb], R.aab.VVvOOo, optimize=True)
            - 1.0 * np.einsum('ANEf,BEfIJN->ABIJ', H.ab.vovv[Va, Ob, Va, vb], R.aab.VVvOOO, optimize=True)
            - 1.0 * np.einsum('AnEF,BEFIJn->ABIJ', H.ab.vovv[Va, ob, Va, Vb], R.aab.VVVOOo, optimize=True)
            - 1.0 * np.einsum('ANEF,BEFIJN->ABIJ', H.ab.vovv[Va, Ob, Va, Vb], R.aab.VVVOOO, optimize=True)
    )
    
    dR.aa[Va, Va, Oa, Oa] -= np.transpose(dR.aa[Va, Va, Oa, Oa], (1, 0, 2, 3))
    dR.aa[Va, Va, Oa, Oa] -= np.transpose(dR.aa[Va, Va, Oa, Oa], (0, 1, 3, 2))
    
    return dR

def build_1101(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    
    dR.aa[Va, Va, oa, Oa] = (1.0 / 2.0) * (
            +1.0 * np.einsum('me,BAeimJ->ABiJ', X.a.ov[oa, va], T.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('mE,EBAimJ->ABiJ', X.a.ov[oa, Va], T.aaa.VVVooO, optimize=True)
            - 1.0 * np.einsum('Me,BAeiJM->ABiJ', X.a.ov[Oa, va], T.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('ME,EBAiJM->ABiJ', X.a.ov[Oa, Va], T.aaa.VVVoOO, optimize=True)
    )
    dR.aa[Va, Va, oa, Oa] += (1.0 / 2.0) * (
            -1.0 * np.einsum('me,BAeiJm->ABiJ', X.b.ov[ob, vb], T.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('mE,BAEiJm->ABiJ', X.b.ov[ob, Vb], T.aab.VVVoOo, optimize=True)
            - 1.0 * np.einsum('Me,BAeiJM->ABiJ', X.b.ov[Ob, vb], T.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('ME,BAEiJM->ABiJ', X.b.ov[Ob, Vb], T.aab.VVVoOO, optimize=True)
    )

    dR.aa[Va, Va, oa, Oa] += (1.0 / 2.0) * (
            +1.0 * np.einsum('me,BAeimJ->ABiJ', H.a.ov[oa, va], R.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('mE,EBAimJ->ABiJ', H.a.ov[oa, Va], R.aaa.VVVooO, optimize=True)
            - 1.0 * np.einsum('Me,BAeiJM->ABiJ', H.a.ov[Oa, va], R.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('ME,EBAiJM->ABiJ', H.a.ov[Oa, Va], R.aaa.VVVoOO, optimize=True)
    )
    dR.aa[Va, Va, oa, Oa] += (1.0 / 2.0) * (
            -1.0 * np.einsum('me,BAeiJm->ABiJ', H.b.ov[ob, vb], R.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('mE,BAEiJm->ABiJ', H.b.ov[ob, Vb], R.aab.VVVoOo, optimize=True)
            - 1.0 * np.einsum('Me,BAeiJM->ABiJ', H.b.ov[Ob, vb], R.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('ME,BAEiJM->ABiJ', H.b.ov[Ob, Vb], R.aab.VVVoOO, optimize=True)
    )

    dR.aa[Va, Va, oa, Oa] += (1.0 / 2.0) * (
            +1.0 * np.einsum('mnif,BAfmJn->ABiJ', H.ab.ooov[oa, ob, oa, vb], R.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('Mnif,BAfMJn->ABiJ', H.ab.ooov[Oa, ob, oa, vb], R.aab.VVvOOo, optimize=True)
            + 1.0 * np.einsum('mNif,BAfmJN->ABiJ', H.ab.ooov[oa, Ob, oa, vb], R.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('MNif,BAfMJN->ABiJ', H.ab.ooov[Oa, Ob, oa, vb], R.aab.VVvOOO, optimize=True)
            + 1.0 * np.einsum('mniF,BAFmJn->ABiJ', H.ab.ooov[oa, ob, oa, Vb], R.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('MniF,BAFMJn->ABiJ', H.ab.ooov[Oa, ob, oa, Vb], R.aab.VVVOOo, optimize=True)
            + 1.0 * np.einsum('mNiF,BAFmJN->ABiJ', H.ab.ooov[oa, Ob, oa, Vb], R.aab.VVVoOO, optimize=True)
            + 1.0 * np.einsum('MNiF,BAFMJN->ABiJ', H.ab.ooov[Oa, Ob, oa, Vb], R.aab.VVVOOO, optimize=True)
    )
    dR.aa[Va, Va, oa, Oa] += (1.0 / 2.0) * (
            +1.0 * np.einsum('MnJf,BAfiMn->ABiJ', H.ab.ooov[Oa, ob, Oa, vb], R.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('mNJf,BAfmiN->ABiJ', H.ab.ooov[oa, Ob, Oa, vb], R.aab.VVvooO, optimize=True)
            + 1.0 * np.einsum('MNJf,BAfiMN->ABiJ', H.ab.ooov[Oa, Ob, Oa, vb], R.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('MnJF,BAFiMn->ABiJ', H.ab.ooov[Oa, ob, Oa, Vb], R.aab.VVVoOo, optimize=True)
            - 1.0 * np.einsum('mNJF,BAFmiN->ABiJ', H.ab.ooov[oa, Ob, Oa, Vb], R.aab.VVVooO, optimize=True)
            + 1.0 * np.einsum('MNJF,BAFiMN->ABiJ', H.ab.ooov[Oa, Ob, Oa, Vb], R.aab.VVVoOO, optimize=True)
    )
    dR.aa[Va, Va, oa, Oa] += (1.0 / 2.0) * (
            -0.5 * np.einsum('mnif,BAfmnJ->ABiJ', H.aa.ooov[oa, oa, oa, va], R.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('mNif,BAfmJN->ABiJ', H.aa.ooov[oa, Oa, oa, va], R.aaa.VVvoOO, optimize=True)
            + 0.5 * np.einsum('MNif,BAfMJN->ABiJ', H.aa.ooov[Oa, Oa, oa, va], R.aaa.VVvOOO, optimize=True)
            - 0.5 * np.einsum('mniF,FBAmnJ->ABiJ', H.aa.ooov[oa, oa, oa, Va], R.aaa.VVVooO, optimize=True)
            + 1.0 * np.einsum('mNiF,FBAmJN->ABiJ', H.aa.ooov[oa, Oa, oa, Va], R.aaa.VVVoOO, optimize=True)
            + 0.5 * np.einsum('MNiF,FBAMJN->ABiJ', H.aa.ooov[Oa, Oa, oa, Va], R.aaa.VVVOOO, optimize=True)
    )
    dR.aa[Va, Va, oa, Oa] += (1.0 / 2.0) * (
            -1.0 * np.einsum('mNJf,BAfmiN->ABiJ', H.aa.ooov[oa, Oa, Oa, va], R.aaa.VVvooO, optimize=True)
            + 0.5 * np.einsum('MNJf,BAfiMN->ABiJ', H.aa.ooov[Oa, Oa, Oa, va], R.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('mNJF,FBAmiN->ABiJ', H.aa.ooov[oa, Oa, Oa, Va], R.aaa.VVVooO, optimize=True)
            + 0.5 * np.einsum('MNJF,FBAiMN->ABiJ', H.aa.ooov[Oa, Oa, Oa, Va], R.aaa.VVVoOO, optimize=True)
    )
    dR.aa[Va, Va, oa, Oa] += (2.0 / 2.0) * (
            -0.5 * np.einsum('Anef,BfeinJ->ABiJ', H.aa.vovv[Va, oa, va, va], R.aaa.VvvooO, optimize=True)
            + 0.5 * np.einsum('ANef,BfeiJN->ABiJ', H.aa.vovv[Va, Oa, va, va], R.aaa.VvvoOO, optimize=True)
            + 1.0 * np.einsum('AneF,FBeinJ->ABiJ', H.aa.vovv[Va, oa, va, Va], R.aaa.VVvooO, optimize=True)
            + 0.5 * np.einsum('AnEF,FBEinJ->ABiJ', H.aa.vovv[Va, oa, Va, Va], R.aaa.VVVooO, optimize=True)
            - 1.0 * np.einsum('ANeF,FBeiJN->ABiJ', H.aa.vovv[Va, Oa, va, Va], R.aaa.VVvoOO, optimize=True)
            - 0.5 * np.einsum('ANEF,FBEiJN->ABiJ', H.aa.vovv[Va, Oa, Va, Va], R.aaa.VVVoOO, optimize=True)
    )
    dR.aa[Va, Va, oa, Oa] += (2.0 / 2.0) * (
            -1.0 * np.einsum('Anef,BefiJn->ABiJ', H.ab.vovv[Va, ob, va, vb], R.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('AnEf,BEfiJn->ABiJ', H.ab.vovv[Va, ob, Va, vb], R.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('ANef,BefiJN->ABiJ', H.ab.vovv[Va, Ob, va, vb], R.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('ANEf,BEfiJN->ABiJ', H.ab.vovv[Va, Ob, Va, vb], R.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('AneF,BeFiJn->ABiJ', H.ab.vovv[Va, ob, va, Vb], R.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('AnEF,BEFiJn->ABiJ', H.ab.vovv[Va, ob, Va, Vb], R.aab.VVVoOo, optimize=True)
            - 1.0 * np.einsum('ANeF,BeFiJN->ABiJ', H.ab.vovv[Va, Ob, va, Vb], R.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('ANEF,BEFiJN->ABiJ', H.ab.vovv[Va, Ob, Va, Vb], R.aab.VVVoOO, optimize=True)
    )
    
    dR.aa[Va, Va, oa, Oa] -= np.transpose(dR.aa[Va, Va, oa, Oa], (1, 0, 2, 3))
    dR.aa[Va, Va, Oa, oa] = -1.0 * np.transpose(dR.aa[Va, Va, oa, Oa], (0, 1, 3, 2))
    
    return dR

def build_1011(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.aa[Va, va, Oa, Oa] = (1.0 / 2.0) * (
            -1.0 * np.einsum('me,AebmIJ->AbIJ', X.a.ov[oa, va], T.aaa.VvvoOO, optimize=True)
            + 1.0 * np.einsum('mE,EAbmIJ->AbIJ', X.a.ov[oa, Va], T.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('Me,AebIJM->AbIJ', X.a.ov[Oa, va], T.aaa.VvvOOO, optimize=True)
            + 1.0 * np.einsum('ME,EAbIJM->AbIJ', X.a.ov[Oa, Va], T.aaa.VVvOOO, optimize=True)
    )
    dR.aa[Va, va, Oa, Oa] += (1.0 / 2.0) * (
            +1.0 * np.einsum('me,AbeIJm->AbIJ', X.b.ov[ob, vb], T.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('mE,AbEIJm->AbIJ', X.b.ov[ob, Vb], T.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('Me,AbeIJM->AbIJ', X.b.ov[Ob, vb], T.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('ME,AbEIJM->AbIJ', X.b.ov[Ob, Vb], T.aab.VvVOOO, optimize=True)
    )

    dR.aa[Va, va, Oa, Oa] += (1.0 / 2.0) * (
            -1.0 * np.einsum('me,AebmIJ->AbIJ', H.a.ov[oa, va], R.aaa.VvvoOO, optimize=True)
            + 1.0 * np.einsum('mE,EAbmIJ->AbIJ', H.a.ov[oa, Va], R.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('Me,AebIJM->AbIJ', H.a.ov[Oa, va], R.aaa.VvvOOO, optimize=True)
            + 1.0 * np.einsum('ME,EAbIJM->AbIJ', H.a.ov[Oa, Va], R.aaa.VVvOOO, optimize=True)
    )
    dR.aa[Va, va, Oa, Oa] += (1.0 / 2.0) * (
            +1.0 * np.einsum('me,AbeIJm->AbIJ', H.b.ov[ob, vb], R.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('mE,AbEIJm->AbIJ', H.b.ov[ob, Vb], R.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('Me,AbeIJM->AbIJ', H.b.ov[Ob, vb], R.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('ME,AbEIJM->AbIJ', H.b.ov[Ob, Vb], R.aab.VvVOOO, optimize=True)
    )

    dR.aa[Va, va, Oa, Oa] += (2.0 / 2.0) * (
            -1.0 * np.einsum('mnIf,AbfmJn->AbIJ', H.ab.ooov[oa, ob, Oa, vb], R.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('mnIF,AbFmJn->AbIJ', H.ab.ooov[oa, ob, Oa, Vb], R.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('mNIf,AbfmJN->AbIJ', H.ab.ooov[oa, Ob, Oa, vb], R.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('mNIF,AbFmJN->AbIJ', H.ab.ooov[oa, Ob, Oa, Vb], R.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('MnIf,AbfMJn->AbIJ', H.ab.ooov[Oa, ob, Oa, vb], R.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('MnIF,AbFMJn->AbIJ', H.ab.ooov[Oa, ob, Oa, Vb], R.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('MNIf,AbfMJN->AbIJ', H.ab.ooov[Oa, Ob, Oa, vb], R.aab.VvvOOO, optimize=True)
            - 1.0 * np.einsum('MNIF,AbFMJN->AbIJ', H.ab.ooov[Oa, Ob, Oa, Vb], R.aab.VvVOOO, optimize=True)
    )
    dR.aa[Va, va, Oa, Oa] += (2.0 / 2.0) * (
            -0.5 * np.einsum('mnIf,AfbmnJ->AbIJ', H.aa.ooov[oa, oa, Oa, va], R.aaa.VvvooO, optimize=True)
            + 0.5 * np.einsum('mnIF,FAbmnJ->AbIJ', H.aa.ooov[oa, oa, Oa, Va], R.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('MnIf,AfbnMJ->AbIJ', H.aa.ooov[Oa, oa, Oa, va], R.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('MnIF,FAbnMJ->AbIJ', H.aa.ooov[Oa, oa, Oa, Va], R.aaa.VVvoOO, optimize=True)
            + 0.5 * np.einsum('MNIf,AfbMJN->AbIJ', H.aa.ooov[Oa, Oa, Oa, va], R.aaa.VvvOOO, optimize=True)
            - 0.5 * np.einsum('MNIF,FAbMJN->AbIJ', H.aa.ooov[Oa, Oa, Oa, Va], R.aaa.VVvOOO, optimize=True)
    )
    dR.aa[Va, va, Oa, Oa] += (1.0 / 2.0) * (
            -1.0 * np.einsum('AnEf,EfbnIJ->AbIJ', H.aa.vovv[Va, oa, Va, va], R.aaa.VvvoOO, optimize=True)
            + 0.5 * np.einsum('AnEF,FEbnIJ->AbIJ', H.aa.vovv[Va, oa, Va, Va], R.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('ANEf,EfbIJN->AbIJ', H.aa.vovv[Va, Oa, Va, va], R.aaa.VvvOOO, optimize=True)
            + 0.5 * np.einsum('ANEF,FEbIJN->AbIJ', H.aa.vovv[Va, Oa, Va, Va], R.aaa.VVvOOO, optimize=True)
    )
    dR.aa[Va, va, Oa, Oa] += (1.0 / 2.0) * (
            -0.5 * np.einsum('bnef,AfenIJ->AbIJ', H.aa.vovv[va, oa, va, va], R.aaa.VvvoOO, optimize=True)
            - 0.5 * np.einsum('bNef,AfeIJN->AbIJ', H.aa.vovv[va, Oa, va, va], R.aaa.VvvOOO, optimize=True)
            + 1.0 * np.einsum('bnEf,AEfnIJ->AbIJ', H.aa.vovv[va, oa, Va, va], R.aaa.VVvoOO, optimize=True)
            + 0.5 * np.einsum('bnEF,FAEnIJ->AbIJ', H.aa.vovv[va, oa, Va, Va], R.aaa.VVVoOO, optimize=True)
            + 1.0 * np.einsum('bNEf,AEfIJN->AbIJ', H.aa.vovv[va, Oa, Va, va], R.aaa.VVvOOO, optimize=True)
            + 0.5 * np.einsum('bNEF,FAEIJN->AbIJ', H.aa.vovv[va, Oa, Va, Va], R.aaa.VVVOOO, optimize=True)
    )
    dR.aa[Va, va, Oa, Oa] += (1.0 / 2.0) * (
            -1.0 * np.einsum('AneF,beFIJn->AbIJ', H.ab.vovv[Va, ob, va, Vb], R.aab.vvVOOo, optimize=True)
            - 1.0 * np.einsum('ANeF,beFIJN->AbIJ', H.ab.vovv[Va, Ob, va, Vb], R.aab.vvVOOO, optimize=True)
            + 1.0 * np.einsum('AnEf,EbfIJn->AbIJ', H.ab.vovv[Va, ob, Va, vb], R.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('AnEF,EbFIJn->AbIJ', H.ab.vovv[Va, ob, Va, Vb], R.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('ANEf,EbfIJN->AbIJ', H.ab.vovv[Va, Ob, Va, vb], R.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('ANEF,EbFIJN->AbIJ', H.ab.vovv[Va, Ob, Va, Vb], R.aab.VvVOOO, optimize=True)
    )
    dR.aa[Va, va, Oa, Oa] += (1.0 / 2.0) * (
            +1.0 * np.einsum('bnef,AefIJn->AbIJ', H.ab.vovv[va, ob, va, vb], R.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('bneF,AeFIJn->AbIJ', H.ab.vovv[va, ob, va, Vb], R.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('bNef,AefIJN->AbIJ', H.ab.vovv[va, Ob, va, vb], R.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('bNeF,AeFIJN->AbIJ', H.ab.vovv[va, Ob, va, Vb], R.aab.VvVOOO, optimize=True)
            + 1.0 * np.einsum('bnEf,AEfIJn->AbIJ', H.ab.vovv[va, ob, Va, vb], R.aab.VVvOOo, optimize=True)
            + 1.0 * np.einsum('bnEF,AEFIJn->AbIJ', H.ab.vovv[va, ob, Va, Vb], R.aab.VVVOOo, optimize=True)
            + 1.0 * np.einsum('bNEf,AEfIJN->AbIJ', H.ab.vovv[va, Ob, Va, vb], R.aab.VVvOOO, optimize=True)
            + 1.0 * np.einsum('bNEF,AEFIJN->AbIJ', H.ab.vovv[va, Ob, Va, Vb], R.aab.VVVOOO, optimize=True)
    )
    
    dR.aa[Va, va, Oa, Oa] -= np.transpose(dR.aa[Va, va, Oa, Oa], (0, 1, 3, 2))
    dR.aa[va, Va, Oa, Oa] = -1.0 * np.transpose(dR.aa[Va, va, Oa, Oa], (1, 0, 2, 3))
    
    return dR

def build_1100(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    
    dR.aa[Va, Va, oa, oa] = (1.0 / 4.0) * (
            -1.0 * np.einsum('Me,BAeijM->ABij', X.a.ov[Oa, va], T.aaa.VVvooO, optimize=True)
            - 1.0 * np.einsum('ME,EBAijM->ABij', X.a.ov[Oa, Va], T.aaa.VVVooO, optimize=True)
    )
    dR.aa[Va, Va, oa, oa] += (1.0 / 4.0) * (
            -1.0 * np.einsum('Me,BAeijM->ABij', X.b.ov[Ob, vb], T.aab.VVvooO, optimize=True)
            - 1.0 * np.einsum('ME,BAEijM->ABij', X.b.ov[Ob, Vb], T.aab.VVVooO, optimize=True)
    )

    dR.aa[Va, Va, oa, oa] += (1.0 / 4.0) * (
            -1.0 * np.einsum('Me,BAeijM->ABij', H.a.ov[Oa, va], R.aaa.VVvooO, optimize=True)
            - 1.0 * np.einsum('ME,EBAijM->ABij', H.a.ov[Oa, Va], R.aaa.VVVooO, optimize=True)
    )
    dR.aa[Va, Va, oa, oa] += (1.0 / 4.0) * (
            -1.0 * np.einsum('Me,BAeijM->ABij', H.b.ov[Ob, vb], R.aab.VVvooO, optimize=True)
            - 1.0 * np.einsum('ME,BAEijM->ABij', H.b.ov[Ob, Vb], R.aab.VVVooO, optimize=True)
    )

    dR.aa[Va, Va, oa, oa] += (2.0 / 4.0) * (
            +1.0 * np.einsum('mNif,BAfmjN->ABij', H.ab.ooov[oa, Ob, oa, vb], R.aab.VVvooO, optimize=True)
            + 1.0 * np.einsum('mNiF,BAFmjN->ABij', H.ab.ooov[oa, Ob, oa, Vb], R.aab.VVVooO, optimize=True)
            - 1.0 * np.einsum('Mnif,BAfjMn->ABij', H.ab.ooov[Oa, ob, oa, vb], R.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('MniF,BAFjMn->ABij', H.ab.ooov[Oa, ob, oa, Vb], R.aab.VVVoOo, optimize=True)
            - 1.0 * np.einsum('MNif,BAfjMN->ABij', H.ab.ooov[Oa, Ob, oa, vb], R.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('MNiF,BAFjMN->ABij', H.ab.ooov[Oa, Ob, oa, Vb], R.aab.VVVoOO, optimize=True)
    )
    dR.aa[Va, Va, oa, oa] += (2.0 / 4.0) * (
            +1.0 * np.einsum('Mnif,BAfjnM->ABij', H.aa.ooov[Oa, oa, oa, va], R.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('MniF,FBAjnM->ABij', H.aa.ooov[Oa, oa, oa, Va], R.aaa.VVVooO, optimize=True)
            - 0.5 * np.einsum('MNif,BAfjMN->ABij', H.aa.ooov[Oa, Oa, oa, va], R.aaa.VVvoOO, optimize=True)
            - 0.5 * np.einsum('MNiF,FBAjMN->ABij', H.aa.ooov[Oa, Oa, oa, Va], R.aaa.VVVoOO, optimize=True)
    )
    dR.aa[Va, Va, oa, oa] += (2.0 / 4.0) * (
            +0.5 * np.einsum('ANef,BfeijN->ABij', H.aa.vovv[Va, Oa, va, va], R.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('ANEf,BEfijN->ABij', H.aa.vovv[Va, Oa, Va, va], R.aaa.VVvooO, optimize=True)
            - 0.5 * np.einsum('ANEF,FBEijN->ABij', H.aa.vovv[Va, Oa, Va, Va], R.aaa.VVVooO, optimize=True)
    )
    dR.aa[Va, Va, oa, oa] += (2.0 / 4.0) * (
            -1.0 * np.einsum('ANef,BefijN->ABij', H.ab.vovv[Va, Ob, va, vb], R.aab.VvvooO, optimize=True)
            - 1.0 * np.einsum('ANeF,BeFijN->ABij', H.ab.vovv[Va, Ob, va, Vb], R.aab.VvVooO, optimize=True)
            - 1.0 * np.einsum('ANEf,BEfijN->ABij', H.ab.vovv[Va, Ob, Va, vb], R.aab.VVvooO, optimize=True)
            - 1.0 * np.einsum('ANEF,BEFijN->ABij', H.ab.vovv[Va, Ob, Va, Vb], R.aab.VVVooO, optimize=True)
    )

    dR.aa[Va, Va, oa, oa] -= np.transpose(dR.aa[Va, Va, oa, oa], (1, 0, 2, 3))
    dR.aa[Va, Va, oa, oa] -= np.transpose(dR.aa[Va, Va, oa, oa], (0, 1, 3, 2))
    
    return dR

def build_0011(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    
    dR.aa[va, va, Oa, Oa] = (1.0 / 4.0) * (
            -1.0 * np.einsum('mE,EbamIJ->abIJ', X.a.ov[oa, Va], T.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('ME,EbaIJM->abIJ', X.a.ov[Oa, Va], T.aaa.VvvOOO, optimize=True)
    )
    dR.aa[va, va, Oa, Oa] += (1.0 / 4.0) * (
            -1.0 * np.einsum('mE,baEIJm->abIJ', X.b.ov[ob, Vb], T.aab.vvVOOo, optimize=True)
            - 1.0 * np.einsum('ME,baEIJM->abIJ', X.b.ov[Ob, Vb], T.aab.vvVOOO, optimize=True)
    )

    dR.aa[va, va, Oa, Oa] += (1.0 / 4.0) * (
            -1.0 * np.einsum('mE,EbamIJ->abIJ', H.a.ov[oa, Va], R.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('ME,EbaIJM->abIJ', H.a.ov[Oa, Va], R.aaa.VvvOOO, optimize=True)
    )
    dR.aa[va, va, Oa, Oa] += (1.0 / 4.0) * (
            -1.0 * np.einsum('mE,baEIJm->abIJ', H.b.ov[ob, Vb], R.aab.vvVOOo, optimize=True)
            - 1.0 * np.einsum('ME,baEIJM->abIJ', H.b.ov[Ob, Vb], R.aab.vvVOOO, optimize=True)
    )

    dR.aa[va, va, Oa, Oa] += (2.0 / 4.0) * (
            +1.0 * np.einsum('mnIF,baFmJn->abIJ', H.ab.ooov[oa, ob, Oa, Vb], R.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('mNIF,baFmJN->abIJ', H.ab.ooov[oa, Ob, Oa, Vb], R.aab.vvVoOO, optimize=True)
            + 1.0 * np.einsum('MnIF,baFMJn->abIJ', H.ab.ooov[Oa, ob, Oa, Vb], R.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('MNIF,baFMJN->abIJ', H.ab.ooov[Oa, Ob, Oa, Vb], R.aab.vvVOOO, optimize=True)
    )
    dR.aa[va, va, Oa, Oa] += (2.0 / 4.0) * (
            -0.5 * np.einsum('mnIF,FbamnJ->abIJ', H.aa.ooov[oa, oa, Oa, Va], R.aaa.VvvooO, optimize=True)
            + 1.0 * np.einsum('MnIF,FbanMJ->abIJ', H.aa.ooov[Oa, oa, Oa, Va], R.aaa.VvvoOO, optimize=True)
            + 0.5 * np.einsum('MNIF,FbaMJN->abIJ', H.aa.ooov[Oa, Oa, Oa, Va], R.aaa.VvvOOO, optimize=True)
    )
    dR.aa[va, va, Oa, Oa] += (2.0 / 4.0) * (
            -1.0 * np.einsum('anEf,EfbnIJ->abIJ', H.aa.vovv[va, oa, Va, va], R.aaa.VvvoOO, optimize=True)
            + 0.5 * np.einsum('anEF,FEbnIJ->abIJ', H.aa.vovv[va, oa, Va, Va], R.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('aNEf,EfbIJN->abIJ', H.aa.vovv[va, Oa, Va, va], R.aaa.VvvOOO, optimize=True)
            + 0.5 * np.einsum('aNEF,FEbIJN->abIJ', H.aa.vovv[va, Oa, Va, Va], R.aaa.VVvOOO, optimize=True)
    )
    dR.aa[va, va, Oa, Oa] += (2.0 / 4.0) * (
            -1.0 * np.einsum('aneF,beFIJn->abIJ', H.ab.vovv[va, ob, va, Vb], R.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('anEf,EbfIJn->abIJ', H.ab.vovv[va, ob, Va, vb], R.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('anEF,EbFIJn->abIJ', H.ab.vovv[va, ob, Va, Vb], R.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('aNeF,beFIJN->abIJ', H.ab.vovv[va, Ob, va, Vb], R.aab.vvVOOO, optimize=True)
            + 1.0 * np.einsum('aNEf,EbfIJN->abIJ', H.ab.vovv[va, Ob, Va, vb], R.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('aNEF,EbFIJN->abIJ', H.ab.vovv[va, Ob, Va, Vb], R.aab.VvVOOO, optimize=True)
    )
    
    dR.aa[va, va, Oa, Oa] -= np.transpose(dR.aa[va, va, Oa, Oa], (1, 0, 2, 3))
    dR.aa[va, va, Oa, Oa] -= np.transpose(dR.aa[va, va, Oa, Oa], (0, 1, 3, 2))

    return dR

def build_1001(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.aa[Va, va, oa, Oa] = (1.0 / 1.0) * (
            +1.0 * np.einsum('me,AebimJ->AbiJ', X.a.ov[oa, va], T.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('mE,EAbimJ->AbiJ', X.a.ov[oa, Va], T.aaa.VVvooO, optimize=True)
            - 1.0 * np.einsum('Me,AebiJM->AbiJ', X.a.ov[Oa, va], T.aaa.VvvoOO, optimize=True)
            + 1.0 * np.einsum('ME,EAbiJM->AbiJ', X.a.ov[Oa, Va], T.aaa.VVvoOO, optimize=True)
    )
    dR.aa[Va, va, oa, Oa] += (1.0 / 1.0) * (
            +1.0 * np.einsum('me,AbeiJm->AbiJ', X.b.ov[ob, vb], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('mE,AbEiJm->AbiJ', X.b.ov[ob, Vb], T.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('Me,AbeiJM->AbiJ', X.b.ov[Ob, vb], T.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('ME,AbEiJM->AbiJ', X.b.ov[Ob, Vb], T.aab.VvVoOO, optimize=True)
    )

    dR.aa[Va, va, oa, Oa] += (1.0 / 1.0) * (
            +1.0 * np.einsum('me,AebimJ->AbiJ', H.a.ov[oa, va], R.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('mE,EAbimJ->AbiJ', H.a.ov[oa, Va], R.aaa.VVvooO, optimize=True)
            - 1.0 * np.einsum('Me,AebiJM->AbiJ', H.a.ov[Oa, va], R.aaa.VvvoOO, optimize=True)
            + 1.0 * np.einsum('ME,EAbiJM->AbiJ', H.a.ov[Oa, Va], R.aaa.VVvoOO, optimize=True)
    )
    dR.aa[Va, va, oa, Oa] += (1.0 / 1.0) * (
            +1.0 * np.einsum('me,AbeiJm->AbiJ', H.b.ov[ob, vb], R.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('mE,AbEiJm->AbiJ', H.b.ov[ob, Vb], R.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('Me,AbeiJM->AbiJ', H.b.ov[Ob, vb], R.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('ME,AbEiJM->AbiJ', H.b.ov[Ob, Vb], R.aab.VvVoOO, optimize=True)
    )

    dR.aa[Va, va, oa, Oa] += (1.0 / 1.0) * (
            -1.0 * np.einsum('mnif,AbfmJn->AbiJ', H.ab.ooov[oa, ob, oa, vb], R.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('mNif,AbfmJN->AbiJ', H.ab.ooov[oa, Ob, oa, vb], R.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('Mnif,AbfMJn->AbiJ', H.ab.ooov[Oa, ob, oa, vb], R.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('MNif,AbfMJN->AbiJ', H.ab.ooov[Oa, Ob, oa, vb], R.aab.VvvOOO, optimize=True)
            - 1.0 * np.einsum('mniF,AbFmJn->AbiJ', H.ab.ooov[oa, ob, oa, Vb], R.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('mNiF,AbFmJN->AbiJ', H.ab.ooov[oa, Ob, oa, Vb], R.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('MniF,AbFMJn->AbiJ', H.ab.ooov[Oa, ob, oa, Vb], R.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('MNiF,AbFMJN->AbiJ', H.ab.ooov[Oa, Ob, oa, Vb], R.aab.VvVOOO, optimize=True)
    )
    dR.aa[Va, va, oa, Oa] += (1.0 / 1.0) * (
            +1.0 * np.einsum('mNJf,AbfmiN->AbiJ', H.ab.ooov[oa, Ob, Oa, vb], R.aab.VvvooO, optimize=True)
            - 1.0 * np.einsum('MnJf,AbfiMn->AbiJ', H.ab.ooov[Oa, ob, Oa, vb], R.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('MNJf,AbfiMN->AbiJ', H.ab.ooov[Oa, Ob, Oa, vb], R.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('mNJF,AbFmiN->AbiJ', H.ab.ooov[oa, Ob, Oa, Vb], R.aab.VvVooO, optimize=True)
            - 1.0 * np.einsum('MnJF,AbFiMn->AbiJ', H.ab.ooov[Oa, ob, Oa, Vb], R.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('MNJF,AbFiMN->AbiJ', H.ab.ooov[Oa, Ob, Oa, Vb], R.aab.VvVoOO, optimize=True)
    )
    dR.aa[Va, va, oa, Oa] += (1.0 / 1.0) * (
            -0.5 * np.einsum('mnif,AfbmnJ->AbiJ', H.aa.ooov[oa, oa, oa, va], R.aaa.VvvooO, optimize=True)
            + 1.0 * np.einsum('Mnif,AfbnMJ->AbiJ', H.aa.ooov[Oa, oa, oa, va], R.aaa.VvvoOO, optimize=True)
            + 0.5 * np.einsum('MNif,AfbMJN->AbiJ', H.aa.ooov[Oa, Oa, oa, va], R.aaa.VvvOOO, optimize=True)
            + 0.5 * np.einsum('mniF,FAbmnJ->AbiJ', H.aa.ooov[oa, oa, oa, Va], R.aaa.VVvooO, optimize=True)
            - 1.0 * np.einsum('MniF,FAbnMJ->AbiJ', H.aa.ooov[Oa, oa, oa, Va], R.aaa.VVvoOO, optimize=True)
            - 0.5 * np.einsum('MNiF,FAbMJN->AbiJ', H.aa.ooov[Oa, Oa, oa, Va], R.aaa.VVvOOO, optimize=True)
    )
    dR.aa[Va, va, oa, Oa] += (1.0 / 1.0) * (
            -1.0 * np.einsum('MnJf,AfbinM->AbiJ', H.aa.ooov[Oa, oa, Oa, va], R.aaa.VvvooO, optimize=True)
            + 0.5 * np.einsum('MNJf,AfbiMN->AbiJ', H.aa.ooov[Oa, Oa, Oa, va], R.aaa.VvvoOO, optimize=True)
            + 1.0 * np.einsum('MnJF,FAbinM->AbiJ', H.aa.ooov[Oa, oa, Oa, Va], R.aaa.VVvooO, optimize=True)
            - 0.5 * np.einsum('MNJF,FAbiMN->AbiJ', H.aa.ooov[Oa, Oa, Oa, Va], R.aaa.VVvoOO, optimize=True)
    )
    dR.aa[Va, va, oa, Oa] += (1.0 / 1.0) * (
            +1.0 * np.einsum('AneF,FbeinJ->AbiJ', H.aa.vovv[Va, oa, va, Va], R.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('ANeF,FbeiJN->AbiJ', H.aa.vovv[Va, Oa, va, Va], R.aaa.VvvoOO, optimize=True)
            - 0.5 * np.einsum('AnEF,FEbinJ->AbiJ', H.aa.vovv[Va, oa, Va, Va], R.aaa.VVvooO, optimize=True)
            + 0.5 * np.einsum('ANEF,FEbiJN->AbiJ', H.aa.vovv[Va, Oa, Va, Va], R.aaa.VVvoOO, optimize=True)
    )
    dR.aa[Va, va, oa, Oa] += (1.0 / 1.0) * (
            +0.5 * np.einsum('bnef,AfeinJ->AbiJ', H.aa.vovv[va, oa, va, va], R.aaa.VvvooO, optimize=True)
            - 0.5 * np.einsum('bNef,AfeiJN->AbiJ', H.aa.vovv[va, Oa, va, va], R.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('bneF,FAeinJ->AbiJ', H.aa.vovv[va, oa, va, Va], R.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('bNeF,FAeiJN->AbiJ', H.aa.vovv[va, Oa, va, Va], R.aaa.VVvoOO, optimize=True)
            - 0.5 * np.einsum('bnEF,FAEinJ->AbiJ', H.aa.vovv[va, oa, Va, Va], R.aaa.VVVooO, optimize=True)
            + 0.5 * np.einsum('bNEF,FAEiJN->AbiJ', H.aa.vovv[va, Oa, Va, Va], R.aaa.VVVoOO, optimize=True)
    )
    dR.aa[Va, va, oa, Oa] += (1.0 / 1.0) * (
            +1.0 * np.einsum('AnEf,EbfiJn->AbiJ', H.ab.vovv[Va, ob, Va, vb], R.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('ANEf,EbfiJN->AbiJ', H.ab.vovv[Va, Ob, Va, vb], R.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('AneF,beFiJn->AbiJ', H.ab.vovv[Va, ob, va, Vb], R.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('ANeF,beFiJN->AbiJ', H.ab.vovv[Va, Ob, va, Vb], R.aab.vvVoOO, optimize=True)
            + 1.0 * np.einsum('AnEF,EbFiJn->AbiJ', H.ab.vovv[Va, ob, Va, Vb], R.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('ANEF,EbFiJN->AbiJ', H.ab.vovv[Va, Ob, Va, Vb], R.aab.VvVoOO, optimize=True)
    )
    dR.aa[Va, va, oa, Oa] += (1.0 / 1.0) * (
            +1.0 * np.einsum('bnef,AefiJn->AbiJ', H.ab.vovv[va, ob, va, vb], R.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('bNef,AefiJN->AbiJ', H.ab.vovv[va, Ob, va, vb], R.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('bnEf,AEfiJn->AbiJ', H.ab.vovv[va, ob, Va, vb], R.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('bNEf,AEfiJN->AbiJ', H.ab.vovv[va, Ob, Va, vb], R.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('bneF,AeFiJn->AbiJ', H.ab.vovv[va, ob, va, Vb], R.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('bNeF,AeFiJN->AbiJ', H.ab.vovv[va, Ob, va, Vb], R.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('bnEF,AEFiJn->AbiJ', H.ab.vovv[va, ob, Va, Vb], R.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('bNEF,AEFiJN->AbiJ', H.ab.vovv[va, Ob, Va, Vb], R.aab.VVVoOO, optimize=True)
    )

    dR.aa[va, Va, oa, Oa] = -1.0 * np.transpose(dR.aa[Va, va, oa, Oa], (1, 0, 2, 3))
    dR.aa[Va, va, Oa, oa] = -1.0 * np.transpose(dR.aa[Va, va, oa, Oa], (0, 1, 3, 2))
    dR.aa[va, Va, Oa, oa] = np.transpose(dR.aa[Va, va, oa, Oa], (1, 0, 3, 2))
    
    return dR

def build_1000(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.aa[Va, va, oa, oa] = (1.0 / 2.0) * (
            -1.0 * np.einsum('Me,AebijM->Abij', X.a.ov[Oa, va], T.aaa.VvvooO, optimize=True)
            + 1.0 * np.einsum('ME,EAbijM->Abij', X.a.ov[Oa, Va], T.aaa.VVvooO, optimize=True)
    )
    dR.aa[Va, va, oa, oa] += (1.0 / 2.0) * (
            +1.0 * np.einsum('Me,AbeijM->Abij', X.b.ov[Ob, vb], T.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('ME,AbEijM->Abij', X.b.ov[Ob, Vb], T.aab.VvVooO, optimize=True)
    )
    dR.aa[Va, va, oa, oa] += (1.0 / 2.0) * (
            -1.0 * np.einsum('Me,AebijM->Abij', H.a.ov[Oa, va], R.aaa.VvvooO, optimize=True)
            + 1.0 * np.einsum('ME,EAbijM->Abij', H.a.ov[Oa, Va], R.aaa.VVvooO, optimize=True)
    )
    dR.aa[Va, va, oa, oa] += (1.0 / 2.0) * (
            +1.0 * np.einsum('Me,AbeijM->Abij', H.b.ov[Ob, vb], R.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('ME,AbEijM->Abij', H.b.ov[Ob, Vb], R.aab.VvVooO, optimize=True)
    )

    dR.aa[Va, va, oa, oa] += (2.0 / 2.0) * (
            +1.0 * np.einsum('Mnif,AbfjMn->Abij', H.ab.ooov[Oa, ob, oa, vb], R.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('MniF,AbFjMn->Abij', H.ab.ooov[Oa, ob, oa, Vb], R.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('mNif,AbfmjN->Abij', H.ab.ooov[oa, Ob, oa, vb], R.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('MNif,AbfjMN->Abij', H.ab.ooov[Oa, Ob, oa, vb], R.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('mNiF,AbFmjN->Abij', H.ab.ooov[oa, Ob, oa, Vb], R.aab.VvVooO, optimize=True)
            + 1.0 * np.einsum('MNiF,AbFjMN->Abij', H.ab.ooov[Oa, Ob, oa, Vb], R.aab.VvVoOO, optimize=True)
    )
    dR.aa[Va, va, oa, oa] += (2.0 / 2.0) * (
            +1.0 * np.einsum('mNif,AfbmjN->Abij', H.aa.ooov[oa, Oa, oa, va], R.aaa.VvvooO, optimize=True)
            - 0.5 * np.einsum('MNif,AfbjMN->Abij', H.aa.ooov[Oa, Oa, oa, va], R.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('mNiF,FAbmjN->Abij', H.aa.ooov[oa, Oa, oa, Va], R.aaa.VVvooO, optimize=True)
            + 0.5 * np.einsum('MNiF,FAbjMN->Abij', H.aa.ooov[Oa, Oa, oa, Va], R.aaa.VVvoOO, optimize=True)
    )
    dR.aa[Va, va, oa, oa] += (1.0 / 2.0) * (
            -1.0 * np.einsum('ANEf,EfbijN->Abij', H.aa.vovv[Va, Oa, Va, va], R.aaa.VvvooO, optimize=True)
            + 0.5 * np.einsum('ANEF,FEbijN->Abij', H.aa.vovv[Va, Oa, Va, Va], R.aaa.VVvooO, optimize=True)
    )
    dR.aa[Va, va, oa, oa] += (1.0 / 2.0) * (
            -0.5 * np.einsum('bNef,AfeijN->Abij', H.aa.vovv[va, Oa, va, va], R.aaa.VvvooO, optimize=True)
            + 1.0 * np.einsum('bNEf,AEfijN->Abij', H.aa.vovv[va, Oa, Va, va], R.aaa.VVvooO, optimize=True)
            + 0.5 * np.einsum('bNEF,FAEijN->Abij', H.aa.vovv[va, Oa, Va, Va], R.aaa.VVVooO, optimize=True)
    )
    dR.aa[Va, va, oa, oa] += (1.0 / 2.0) * (
            -1.0 * np.einsum('ANeF,beFijN->Abij', H.ab.vovv[Va, Ob, va, Vb], R.aab.vvVooO, optimize=True)
            + 1.0 * np.einsum('ANEf,EbfijN->Abij', H.ab.vovv[Va, Ob, Va, vb], R.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('ANEF,EbFijN->Abij', H.ab.vovv[Va, Ob, Va, Vb], R.aab.VvVooO, optimize=True)
    )
    dR.aa[Va, va, oa, oa] += (1.0 / 2.0) * (
            +1.0 * np.einsum('bNef,AefijN->Abij', H.ab.vovv[va, Ob, va, vb], R.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('bNeF,AeFijN->Abij', H.ab.vovv[va, Ob, va, Vb], R.aab.VvVooO, optimize=True)
            + 1.0 * np.einsum('bNEf,AEfijN->Abij', H.ab.vovv[va, Ob, Va, vb], R.aab.VVvooO, optimize=True)
            + 1.0 * np.einsum('bNEF,AEFijN->Abij', H.ab.vovv[va, Ob, Va, Vb], R.aab.VVVooO, optimize=True)
    )
    
    dR.aa[Va, va, oa, oa] -= np.transpose(dR.aa[Va, va, oa, oa], (0, 1, 3, 2))
    dR.aa[va, Va, oa, oa] = -1.0 * np.transpose(dR.aa[Va, va, oa, oa], (1, 0, 2, 3))
    
    return dR

def build_0001(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    
    dR.aa[va, va, oa, Oa] = (1.0 / 2.0) * (
            +1.0 * np.einsum('mE,EbaimJ->abiJ', X.a.ov[oa, Va], T.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('ME,EbaiJM->abiJ', X.a.ov[Oa, Va], T.aaa.VvvoOO, optimize=True)
    )
    dR.aa[va, va, oa, Oa] += (1.0 / 2.0) * (
            -1.0 * np.einsum('mE,baEiJm->abiJ', X.b.ov[ob, Vb], T.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('ME,baEiJM->abiJ', X.b.ov[Ob, Vb], T.aab.vvVoOO, optimize=True)
    )

    dR.aa[va, va, oa, Oa] += (1.0 / 2.0) * (
            +1.0 * np.einsum('mE,EbaimJ->abiJ', H.a.ov[oa, Va], R.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('ME,EbaiJM->abiJ', H.a.ov[Oa, Va], R.aaa.VvvoOO, optimize=True)
    )
    dR.aa[va, va, oa, Oa] += (1.0 / 2.0) * (
            -1.0 * np.einsum('mE,baEiJm->abiJ', H.b.ov[ob, Vb], R.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('ME,baEiJM->abiJ', H.b.ov[Ob, Vb], R.aab.vvVoOO, optimize=True)
    )
    dR.aa[va, va, oa, Oa] += (1.0 / 2.0) * (
            +1.0 * np.einsum('mniF,baFmJn->abiJ', H.ab.ooov[oa, ob, oa, Vb], R.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('MniF,baFMJn->abiJ', H.ab.ooov[Oa, ob, oa, Vb], R.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('mNiF,baFmJN->abiJ', H.ab.ooov[oa, Ob, oa, Vb], R.aab.vvVoOO, optimize=True)
            + 1.0 * np.einsum('MNiF,baFMJN->abiJ', H.ab.ooov[Oa, Ob, oa, Vb], R.aab.vvVOOO, optimize=True)
    )
    dR.aa[va, va, oa, Oa] += (1.0 / 2.0) * (
            +1.0 * np.einsum('MnJF,baFiMn->abiJ', H.ab.ooov[Oa, ob, Oa, Vb], R.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('mNJF,baFmiN->abiJ', H.ab.ooov[oa, Ob, Oa, Vb], R.aab.vvVooO, optimize=True)
            + 1.0 * np.einsum('MNJF,baFiMN->abiJ', H.ab.ooov[Oa, Ob, Oa, Vb], R.aab.vvVoOO, optimize=True)
    )
    dR.aa[va, va, oa, Oa] += (1.0 / 2.0) * (
            -0.5 * np.einsum('mniF,FbamnJ->abiJ', H.aa.ooov[oa, oa, oa, Va], R.aaa.VvvooO, optimize=True)
            + 1.0 * np.einsum('mNiF,FbamJN->abiJ', H.aa.ooov[oa, Oa, oa, Va], R.aaa.VvvoOO, optimize=True)
            + 0.5 * np.einsum('MNiF,FbaMJN->abiJ', H.aa.ooov[Oa, Oa, oa, Va], R.aaa.VvvOOO, optimize=True)
    )
    dR.aa[va, va, oa, Oa] += (1.0 / 2.0) * (
            -1.0 * np.einsum('mNJF,FbamiN->abiJ', H.aa.ooov[oa, Oa, Oa, Va], R.aaa.VvvooO, optimize=True)
            + 0.5 * np.einsum('MNJF,FbaiMN->abiJ', H.aa.ooov[Oa, Oa, Oa, Va], R.aaa.VvvoOO, optimize=True)
    )
    dR.aa[va, va, oa, Oa] += (2.0 / 2.0) * (
            +1.0 * np.einsum('aneF,FbeinJ->abiJ', H.aa.vovv[va, oa, va, Va], R.aaa.VvvooO, optimize=True)
            - 0.5 * np.einsum('anEF,FEbinJ->abiJ', H.aa.vovv[va, oa, Va, Va], R.aaa.VVvooO, optimize=True)
            - 1.0 * np.einsum('aNeF,FbeiJN->abiJ', H.aa.vovv[va, Oa, va, Va], R.aaa.VvvoOO, optimize=True)
            + 0.5 * np.einsum('aNEF,FEbiJN->abiJ', H.aa.vovv[va, Oa, Va, Va], R.aaa.VVvoOO, optimize=True)
    )
    dR.aa[va, va, oa, Oa] += (2.0 / 2.0) * (
            +1.0 * np.einsum('anEf,EbfiJn->abiJ', H.ab.vovv[va, ob, Va, vb], R.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('aneF,beFiJn->abiJ', H.ab.vovv[va, ob, va, Vb], R.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('anEF,EbFiJn->abiJ', H.ab.vovv[va, ob, Va, Vb], R.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('aNEf,EbfiJN->abiJ', H.ab.vovv[va, Ob, Va, vb], R.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('aNeF,beFiJN->abiJ', H.ab.vovv[va, Ob, va, Vb], R.aab.vvVoOO, optimize=True)
            + 1.0 * np.einsum('aNEF,EbFiJN->abiJ', H.ab.vovv[va, Ob, Va, Vb], R.aab.VvVoOO, optimize=True)
    )
    
    dR.aa[va, va, oa, Oa] -= np.transpose(dR.aa[va, va, oa, Oa], (1, 0, 2, 3))
    dR.aa[va, va, Oa, oa] = -1.0 * np.transpose(dR.aa[va, va, oa, Oa], (0, 1, 3, 2))
    
    return dR


def build_0000(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    
    dR.aa[va, va, oa, oa] = (1.0 / 4.0) * (
            -1.0 * np.einsum('ME,EbaijM->abij', X.a.ov[Oa, Va], T.aaa.VvvooO, optimize=True)
    )
    dR.aa[va, va, oa, oa] += (1.0 / 4.0) * (
            -1.0 * np.einsum('ME,baEijM->abij', X.b.ov[Ob, Vb], T.aab.vvVooO, optimize=True)
    )

    dR.aa[va, va, oa, oa] += (1.0 / 4.0) * (
            -1.0 * np.einsum('ME,EbaijM->abij', H.a.ov[Oa, Va], R.aaa.VvvooO, optimize=True)
    )
    dR.aa[va, va, oa, oa] += (1.0 / 4.0) * (
            -1.0 * np.einsum('ME,baEijM->abij', H.b.ov[Ob, Vb], R.aab.vvVooO, optimize=True)
    )

    dR.aa[va, va, oa, oa] += (2.0 / 4.0) * (
            +1.0 * np.einsum('mNiF,baFmjN->abij', H.ab.ooov[oa, Ob, oa, Vb], R.aab.vvVooO, optimize=True)
            - 1.0 * np.einsum('MniF,baFjMn->abij', H.ab.ooov[Oa, ob, oa, Vb], R.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('MNiF,baFjMN->abij', H.ab.ooov[Oa, Ob, oa, Vb], R.aab.vvVoOO, optimize=True)
    )
    dR.aa[va, va, oa, oa] += (2.0 / 4.0) * (
            +1.0 * np.einsum('MniF,FbajnM->abij', H.aa.ooov[Oa, oa, oa, Va], R.aaa.VvvooO, optimize=True)
            - 0.5 * np.einsum('MNiF,FbajMN->abij', H.aa.ooov[Oa, Oa, oa, Va], R.aaa.VvvoOO, optimize=True)
    )
    dR.aa[va, va, oa, oa] += (2.0 / 4.0) * (
            -1.0 * np.einsum('aNeF,FbeijN->abij', H.aa.vovv[va, Oa, va, Va], R.aaa.VvvooO, optimize=True)
            + 0.5 * np.einsum('aNEF,FEbijN->abij', H.aa.vovv[va, Oa, Va, Va], R.aaa.VVvooO, optimize=True)
    )
    dR.aa[va, va, oa, oa] += (2.0 / 4.0) * (
            +1.0 * np.einsum('aNEf,EbfijN->abij', H.ab.vovv[va, Ob, Va, vb], R.aab.VvvooO, optimize=True)
            - 1.0 * np.einsum('aNeF,beFijN->abij', H.ab.vovv[va, Ob, va, Vb], R.aab.vvVooO, optimize=True)
            + 1.0 * np.einsum('aNEF,EbFijN->abij', H.ab.vovv[va, Ob, Va, Vb], R.aab.VvVooO, optimize=True)
    )

    dR.aa[va, va, oa, oa] -= np.transpose(dR.aa[va, va, oa, oa], (1, 0, 2, 3))
    dR.aa[va, va, oa, oa] -= np.transpose(dR.aa[va, va, oa, oa], (0, 1, 3, 2))
    
    return dR
