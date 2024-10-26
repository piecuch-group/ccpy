import numpy as np

from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops


def update(R, omega, H):

    R.bb = eomcc_active_loops.update_r2c(R.bb, omega, H.a.oo, H.a.vv, H.b.oo, H.b.vv, 0.0)

    return R


def build(dR, R, T, H, X, system):
    x2 = build_eomccsd(R, T, H, X)  # base EOMCCSD part (separately antisymmetrized)

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
    dR.bb += x2

    return dR

def build_eomccsd(R, T, H, X):

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


def build_1111(dR, R, T, H, X, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    dR.bb[Vb, Vb, Ob, Ob] = (1.0 / 4.0) * (
            -1.0 * np.einsum('me,eBAmIJ->ABIJ', X.a.ov[oa, va], T.abb.vVVoOO, optimize=True)
            - 1.0 * np.einsum('Me,eBAMIJ->ABIJ', X.a.ov[Oa, va], T.abb.vVVOOO, optimize=True)
            - 1.0 * np.einsum('mE,EBAmIJ->ABIJ', X.a.ov[oa, Va], T.abb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('ME,EBAMIJ->ABIJ', X.a.ov[Oa, Va], T.abb.VVVOOO, optimize=True)
    )
    dR.bb[Vb, Vb, Ob, Ob] += (1.0 / 4.0) * (
            -1.0 * np.einsum('me,BAemIJ->ABIJ', X.b.ov[ob, vb], T.bbb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('Me,BAeIJM->ABIJ', X.b.ov[Ob, vb], T.bbb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('mE,EBAmIJ->ABIJ', X.b.ov[ob, Vb], T.bbb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('ME,EBAIJM->ABIJ', X.b.ov[Ob, Vb], T.bbb.VVVOOO, optimize=True)
    )
    dR.bb[Vb, Vb, Ob, Ob] += (1.0 / 4.0) * (
            -1.0 * np.einsum('me,eBAmIJ->ABIJ', H.a.ov[oa, va], R.abb.vVVoOO, optimize=True)
            - 1.0 * np.einsum('Me,eBAMIJ->ABIJ', H.a.ov[Oa, va], R.abb.vVVOOO, optimize=True)
            - 1.0 * np.einsum('mE,EBAmIJ->ABIJ', H.a.ov[oa, Va], R.abb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('ME,EBAMIJ->ABIJ', H.a.ov[Oa, Va], R.abb.VVVOOO, optimize=True)
    )
    dR.bb[Vb, Vb, Ob, Ob] += (1.0 / 4.0) * (
            -1.0 * np.einsum('me,BAemIJ->ABIJ', H.b.ov[ob, vb], R.bbb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('Me,BAeIJM->ABIJ', H.b.ov[Ob, vb], R.bbb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('mE,EBAmIJ->ABIJ', H.b.ov[ob, Vb], R.bbb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('ME,EBAIJM->ABIJ', H.b.ov[Ob, Vb], R.bbb.VVVOOO, optimize=True)
    )
    dR.bb[Vb, Vb, Ob, Ob] += (2.0 / 4.0) * (
            +0.5 * np.einsum('Anef,BfenIJ->ABIJ', H.bb.vovv[Vb, ob, vb, vb], R.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('AneF,FBenIJ->ABIJ', H.bb.vovv[Vb, ob, vb, Vb], R.bbb.VVvoOO, optimize=True)
            - 0.5 * np.einsum('AnEF,FBEnIJ->ABIJ', H.bb.vovv[Vb, ob, Vb, Vb], R.bbb.VVVoOO, optimize=True)
            + 0.5 * np.einsum('ANef,BfeIJN->ABIJ', H.bb.vovv[Vb, Ob, vb, vb], R.bbb.VvvOOO, optimize=True)
            - 1.0 * np.einsum('ANeF,FBeIJN->ABIJ', H.bb.vovv[Vb, Ob, vb, Vb], R.bbb.VVvOOO, optimize=True)
            - 0.5 * np.einsum('ANEF,FBEIJN->ABIJ', H.bb.vovv[Vb, Ob, Vb, Vb], R.bbb.VVVOOO, optimize=True)
    )
    dR.bb[Vb, Vb, Ob, Ob] += (2.0 / 4.0) * (
            -1.0 * np.einsum('nAfe,fBenIJ->ABIJ', H.ab.ovvv[oa, Vb, va, vb], R.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('nAfE,fBEnIJ->ABIJ', H.ab.ovvv[oa, Vb, va, Vb], R.abb.vVVoOO, optimize=True)
            - 1.0 * np.einsum('nAFe,FBenIJ->ABIJ', H.ab.ovvv[oa, Vb, Va, vb], R.abb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('nAFE,FBEnIJ->ABIJ', H.ab.ovvv[oa, Vb, Va, Vb], R.abb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('NAfe,fBeNIJ->ABIJ', H.ab.ovvv[Oa, Vb, va, vb], R.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('NAfE,fBENIJ->ABIJ', H.ab.ovvv[Oa, Vb, va, Vb], R.abb.vVVOOO, optimize=True)
            - 1.0 * np.einsum('NAFe,FBeNIJ->ABIJ', H.ab.ovvv[Oa, Vb, Va, vb], R.abb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('NAFE,FBENIJ->ABIJ', H.ab.ovvv[Oa, Vb, Va, Vb], R.abb.VVVOOO, optimize=True)
    )
    dR.bb[Vb, Vb, Ob, Ob] += (2.0 / 4.0) * (
            -0.5 * np.einsum('mnIf,BAfmnJ->ABIJ', H.bb.ooov[ob, ob, Ob, vb], R.bbb.VVvooO, optimize=True)
            + 1.0 * np.einsum('mNIf,BAfmJN->ABIJ', H.bb.ooov[ob, Ob, Ob, vb], R.bbb.VVvoOO, optimize=True)
            + 0.5 * np.einsum('MNIf,BAfMJN->ABIJ', H.bb.ooov[Ob, Ob, Ob, vb], R.bbb.VVvOOO, optimize=True)
            - 0.5 * np.einsum('mnIF,FBAmnJ->ABIJ', H.bb.ooov[ob, ob, Ob, Vb], R.bbb.VVVooO, optimize=True)
            + 1.0 * np.einsum('mNIF,FBAmJN->ABIJ', H.bb.ooov[ob, Ob, Ob, Vb], R.bbb.VVVoOO, optimize=True)
            + 0.5 * np.einsum('MNIF,FBAMJN->ABIJ', H.bb.ooov[Ob, Ob, Ob, Vb], R.bbb.VVVOOO, optimize=True)
    )
    dR.bb[Vb, Vb, Ob, Ob] += (2.0 / 4.0) * (
            +1.0 * np.einsum('nmfI,fBAnmJ->ABIJ', H.ab.oovo[oa, ob, va, Ob], R.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('nMfI,fBAnMJ->ABIJ', H.ab.oovo[oa, Ob, va, Ob], R.abb.vVVoOO, optimize=True)
            + 1.0 * np.einsum('nmFI,FBAnmJ->ABIJ', H.ab.oovo[oa, ob, Va, Ob], R.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('nMFI,FBAnMJ->ABIJ', H.ab.oovo[oa, Ob, Va, Ob], R.abb.VVVoOO, optimize=True)
            + 1.0 * np.einsum('NmfI,fBANmJ->ABIJ', H.ab.oovo[Oa, ob, va, Ob], R.abb.vVVOoO, optimize=True)
            + 1.0 * np.einsum('NMfI,fBANMJ->ABIJ', H.ab.oovo[Oa, Ob, va, Ob], R.abb.vVVOOO, optimize=True)
            + 1.0 * np.einsum('NmFI,FBANmJ->ABIJ', H.ab.oovo[Oa, ob, Va, Ob], R.abb.VVVOoO, optimize=True)
            + 1.0 * np.einsum('NMFI,FBANMJ->ABIJ', H.ab.oovo[Oa, Ob, Va, Ob], R.abb.VVVOOO, optimize=True)
    )

    dR.bb[Vb, Vb, Ob, Ob] -= np.transpose(dR.bb[Vb, Vb, Ob, Ob], (1, 0, 2, 3))
    dR.bb[Vb, Vb, Ob, Ob] -= np.transpose(dR.bb[Vb, Vb, Ob, Ob], (0, 1, 3, 2))

    return dR


def build_1101(dR, R, T, H, X, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    dR.bb[Vb, Vb, ob, Ob] = (1.0 / 2.0) * (
            -1.0 * np.einsum('me,eBAmiJ->ABiJ', X.a.ov[oa, va], T.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('mE,EBAmiJ->ABiJ', X.a.ov[oa, Va], T.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('Me,eBAMiJ->ABiJ', X.a.ov[Oa, va], T.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('ME,EBAMiJ->ABiJ', X.a.ov[Oa, Va], T.abb.VVVOoO, optimize=True)
    )
    dR.bb[Vb, Vb, ob, Ob] += (1.0 / 2.0) * (
            +1.0 * np.einsum('me,BAeimJ->ABiJ', X.b.ov[ob, vb], T.bbb.VVvooO, optimize=True)
            + 1.0 * np.einsum('mE,EBAimJ->ABiJ', X.b.ov[ob, Vb], T.bbb.VVVooO, optimize=True)
            - 1.0 * np.einsum('Me,BAeiJM->ABiJ', X.b.ov[Ob, vb], T.bbb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('ME,EBAiJM->ABiJ', X.b.ov[Ob, Vb], T.bbb.VVVoOO, optimize=True)
    )
    dR.bb[Vb, Vb, ob, Ob] += (1.0 / 2.0) * (
            -1.0 * np.einsum('me,eBAmiJ->ABiJ', H.a.ov[oa, va], R.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('mE,EBAmiJ->ABiJ', H.a.ov[oa, Va], R.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('Me,eBAMiJ->ABiJ', H.a.ov[Oa, va], R.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('ME,EBAMiJ->ABiJ', H.a.ov[Oa, Va], R.abb.VVVOoO, optimize=True)
    )
    dR.bb[Vb, Vb, ob, Ob] += (1.0 / 2.0) * (
            +1.0 * np.einsum('me,BAeimJ->ABiJ', H.b.ov[ob, vb], R.bbb.VVvooO, optimize=True)
            + 1.0 * np.einsum('mE,EBAimJ->ABiJ', H.b.ov[ob, Vb], R.bbb.VVVooO, optimize=True)
            - 1.0 * np.einsum('Me,BAeiJM->ABiJ', H.b.ov[Ob, vb], R.bbb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('ME,EBAiJM->ABiJ', H.b.ov[Ob, Vb], R.bbb.VVVoOO, optimize=True)
    )
    dR.bb[Vb, Vb, ob, Ob] += (2.0 / 2.0) * (
            -0.5 * np.einsum('Anef,BfeinJ->ABiJ', H.bb.vovv[Vb, ob, vb, vb], R.bbb.VvvooO, optimize=True)
            + 0.5 * np.einsum('ANef,BfeiJN->ABiJ', H.bb.vovv[Vb, Ob, vb, vb], R.bbb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('AnEf,BEfinJ->ABiJ', H.bb.vovv[Vb, ob, Vb, vb], R.bbb.VVvooO, optimize=True)
            + 0.5 * np.einsum('AnEF,FBEinJ->ABiJ', H.bb.vovv[Vb, ob, Vb, Vb], R.bbb.VVVooO, optimize=True)
            - 1.0 * np.einsum('ANEf,BEfiJN->ABiJ', H.bb.vovv[Vb, Ob, Vb, vb], R.bbb.VVvoOO, optimize=True)
            - 0.5 * np.einsum('ANEF,FBEiJN->ABiJ', H.bb.vovv[Vb, Ob, Vb, Vb], R.bbb.VVVoOO, optimize=True)
    )
    dR.bb[Vb, Vb, ob, Ob] += (2.0 / 2.0) * (
            -1.0 * np.einsum('nAfe,fBeniJ->ABiJ', H.ab.ovvv[oa, Vb, va, vb], R.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('nAFe,FBeniJ->ABiJ', H.ab.ovvv[oa, Vb, Va, vb], R.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('NAfe,fBeNiJ->ABiJ', H.ab.ovvv[Oa, Vb, va, vb], R.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('NAFe,FBeNiJ->ABiJ', H.ab.ovvv[Oa, Vb, Va, vb], R.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('nAfE,fBEniJ->ABiJ', H.ab.ovvv[oa, Vb, va, Vb], R.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('nAFE,FBEniJ->ABiJ', H.ab.ovvv[oa, Vb, Va, Vb], R.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('NAfE,fBENiJ->ABiJ', H.ab.ovvv[Oa, Vb, va, Vb], R.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('NAFE,FBENiJ->ABiJ', H.ab.ovvv[Oa, Vb, Va, Vb], R.abb.VVVOoO, optimize=True)
    )
    dR.bb[Vb, Vb, ob, Ob] += (1.0 / 2.0) * (
            -0.5 * np.einsum('mnif,BAfmnJ->ABiJ', H.bb.ooov[ob, ob, ob, vb], R.bbb.VVvooO, optimize=True)
            - 0.5 * np.einsum('mniF,FBAmnJ->ABiJ', H.bb.ooov[ob, ob, ob, Vb], R.bbb.VVVooO, optimize=True)
            + 1.0 * np.einsum('Mnif,BAfnMJ->ABiJ', H.bb.ooov[Ob, ob, ob, vb], R.bbb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('MniF,FBAnMJ->ABiJ', H.bb.ooov[Ob, ob, ob, Vb], R.bbb.VVVoOO, optimize=True)
            + 0.5 * np.einsum('MNif,BAfMJN->ABiJ', H.bb.ooov[Ob, Ob, ob, vb], R.bbb.VVvOOO, optimize=True)
            + 0.5 * np.einsum('MNiF,FBAMJN->ABiJ', H.bb.ooov[Ob, Ob, ob, Vb], R.bbb.VVVOOO, optimize=True)
    )
    dR.bb[Vb, Vb, ob, Ob] += (1.0 / 2.0) * (
            -1.0 * np.einsum('MnJf,BAfinM->ABiJ', H.bb.ooov[Ob, ob, Ob, vb], R.bbb.VVvooO, optimize=True)
            - 1.0 * np.einsum('MnJF,FBAinM->ABiJ', H.bb.ooov[Ob, ob, Ob, Vb], R.bbb.VVVooO, optimize=True)
            + 0.5 * np.einsum('MNJf,BAfiMN->ABiJ', H.bb.ooov[Ob, Ob, Ob, vb], R.bbb.VVvoOO, optimize=True)
            + 0.5 * np.einsum('MNJF,FBAiMN->ABiJ', H.bb.ooov[Ob, Ob, Ob, Vb], R.bbb.VVVoOO, optimize=True)
    )
    dR.bb[Vb, Vb, ob, Ob] += (1.0 / 2.0) * (
            +1.0 * np.einsum('nmfi,fBAnmJ->ABiJ', H.ab.oovo[oa, ob, va, ob], R.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('nmFi,FBAnmJ->ABiJ', H.ab.oovo[oa, ob, Va, ob], R.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('Nmfi,fBANmJ->ABiJ', H.ab.oovo[Oa, ob, va, ob], R.abb.vVVOoO, optimize=True)
            + 1.0 * np.einsum('NmFi,FBANmJ->ABiJ', H.ab.oovo[Oa, ob, Va, ob], R.abb.VVVOoO, optimize=True)
            + 1.0 * np.einsum('nMfi,fBAnMJ->ABiJ', H.ab.oovo[oa, Ob, va, ob], R.abb.vVVoOO, optimize=True)
            + 1.0 * np.einsum('nMFi,FBAnMJ->ABiJ', H.ab.oovo[oa, Ob, Va, ob], R.abb.VVVoOO, optimize=True)
            + 1.0 * np.einsum('NMfi,fBANMJ->ABiJ', H.ab.oovo[Oa, Ob, va, ob], R.abb.vVVOOO, optimize=True)
            + 1.0 * np.einsum('NMFi,FBANMJ->ABiJ', H.ab.oovo[Oa, Ob, Va, ob], R.abb.VVVOOO, optimize=True)
    )
    dR.bb[Vb, Vb, ob, Ob] += (1.0 / 2.0) * (
            -1.0 * np.einsum('NmfJ,fBANmi->ABiJ', H.ab.oovo[Oa, ob, va, Ob], R.abb.vVVOoo, optimize=True)
            - 1.0 * np.einsum('NmFJ,FBANmi->ABiJ', H.ab.oovo[Oa, ob, Va, Ob], R.abb.VVVOoo, optimize=True)
            + 1.0 * np.einsum('nMfJ,fBAniM->ABiJ', H.ab.oovo[oa, Ob, va, Ob], R.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('nMFJ,FBAniM->ABiJ', H.ab.oovo[oa, Ob, Va, Ob], R.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('NMfJ,fBANiM->ABiJ', H.ab.oovo[Oa, Ob, va, Ob], R.abb.vVVOoO, optimize=True)
            + 1.0 * np.einsum('NMFJ,FBANiM->ABiJ', H.ab.oovo[Oa, Ob, Va, Ob], R.abb.VVVOoO, optimize=True)
    )

    dR.bb[Vb, Vb, ob, Ob] -= np.transpose(dR.bb[Vb, Vb, ob, Ob], (1, 0, 2, 3))
    dR.bb[Vb, Vb, Ob, ob] = -1.0 * np.transpose(dR.bb[Vb, Vb, ob, Ob], (0, 1, 3, 2))
    return dR


def build_1011(dR, R, T, H, X, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    dR.bb[Vb, vb, Ob, Ob] = (1.0 / 2.0) * (
            +1.0 * np.einsum('me,eAbmIJ->AbIJ', X.a.ov[oa, va], T.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('mE,EAbmIJ->AbIJ', X.a.ov[oa, Va], T.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('Me,eAbMIJ->AbIJ', X.a.ov[Oa, va], T.abb.vVvOOO, optimize=True)
            + 1.0 * np.einsum('ME,EAbMIJ->AbIJ', X.a.ov[Oa, Va], T.abb.VVvOOO, optimize=True)
    )
    dR.bb[Vb, vb, Ob, Ob] += (1.0 / 2.0) * (
            -1.0 * np.einsum('me,AebmIJ->AbIJ', X.b.ov[ob, vb], T.bbb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('mE,EAbmIJ->AbIJ', X.b.ov[ob, Vb], T.bbb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('Me,AebIJM->AbIJ', X.b.ov[Ob, vb], T.bbb.VvvOOO, optimize=True)
            + 1.0 * np.einsum('ME,EAbIJM->AbIJ', X.b.ov[Ob, Vb], T.bbb.VVvOOO, optimize=True)
    )
    dR.bb[Vb, vb, Ob, Ob] += (1.0 / 2.0) * (
            +1.0 * np.einsum('me,eAbmIJ->AbIJ', H.a.ov[oa, va], R.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('mE,EAbmIJ->AbIJ', H.a.ov[oa, Va], R.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('Me,eAbMIJ->AbIJ', H.a.ov[Oa, va], R.abb.vVvOOO, optimize=True)
            + 1.0 * np.einsum('ME,EAbMIJ->AbIJ', H.a.ov[Oa, Va], R.abb.VVvOOO, optimize=True)
    )
    dR.bb[Vb, vb, Ob, Ob] += (1.0 / 2.0) * (
            -1.0 * np.einsum('me,AebmIJ->AbIJ', H.b.ov[ob, vb], R.bbb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('mE,EAbmIJ->AbIJ', H.b.ov[ob, Vb], R.bbb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('Me,AebIJM->AbIJ', H.b.ov[Ob, vb], R.bbb.VvvOOO, optimize=True)
            + 1.0 * np.einsum('ME,EAbIJM->AbIJ', H.b.ov[Ob, Vb], R.bbb.VVvOOO, optimize=True)
    )
    dR.bb[Vb, vb, Ob, Ob] += (1.0 / 2.0) * (
            -1.0 * np.einsum('AneF,FbenIJ->AbIJ', H.bb.vovv[Vb, ob, vb, Vb], R.bbb.VvvoOO, optimize=True)
            + 0.5 * np.einsum('AnEF,FEbnIJ->AbIJ', H.bb.vovv[Vb, ob, Vb, Vb], R.bbb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('ANeF,FbeIJN->AbIJ', H.bb.vovv[Vb, Ob, vb, Vb], R.bbb.VvvOOO, optimize=True)
            + 0.5 * np.einsum('ANEF,FEbIJN->AbIJ', H.bb.vovv[Vb, Ob, Vb, Vb], R.bbb.VVvOOO, optimize=True)
    )
    dR.bb[Vb, vb, Ob, Ob] += (1.0 / 2.0) * (
            -0.5 * np.einsum('bnef,AfenIJ->AbIJ', H.bb.vovv[vb, ob, vb, vb], R.bbb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('bneF,FAenIJ->AbIJ', H.bb.vovv[vb, ob, vb, Vb], R.bbb.VVvoOO, optimize=True)
            + 0.5 * np.einsum('bnEF,FAEnIJ->AbIJ', H.bb.vovv[vb, ob, Vb, Vb], R.bbb.VVVoOO, optimize=True)
            - 0.5 * np.einsum('bNef,AfeIJN->AbIJ', H.bb.vovv[vb, Ob, vb, vb], R.bbb.VvvOOO, optimize=True)
            + 1.0 * np.einsum('bNeF,FAeIJN->AbIJ', H.bb.vovv[vb, Ob, vb, Vb], R.bbb.VVvOOO, optimize=True)
            + 0.5 * np.einsum('bNEF,FAEIJN->AbIJ', H.bb.vovv[vb, Ob, Vb, Vb], R.bbb.VVVOOO, optimize=True)
    )
    dR.bb[Vb, vb, Ob, Ob] += (1.0 / 2.0) * (
            +1.0 * np.einsum('nAfE,fEbnIJ->AbIJ', H.ab.ovvv[oa, Vb, va, Vb], R.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('nAFe,FbenIJ->AbIJ', H.ab.ovvv[oa, Vb, Va, vb], R.abb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('nAFE,FEbnIJ->AbIJ', H.ab.ovvv[oa, Vb, Va, Vb], R.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('NAfE,fEbNIJ->AbIJ', H.ab.ovvv[Oa, Vb, va, Vb], R.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('NAFe,FbeNIJ->AbIJ', H.ab.ovvv[Oa, Vb, Va, vb], R.abb.VvvOOO, optimize=True)
            + 1.0 * np.einsum('NAFE,FEbNIJ->AbIJ', H.ab.ovvv[Oa, Vb, Va, Vb], R.abb.VVvOOO, optimize=True)
    )
    dR.bb[Vb, vb, Ob, Ob] += (1.0 / 2.0) * (
            +1.0 * np.einsum('nbfe,fAenIJ->AbIJ', H.ab.ovvv[oa, vb, va, vb], R.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('nbfE,fAEnIJ->AbIJ', H.ab.ovvv[oa, vb, va, Vb], R.abb.vVVoOO, optimize=True)
            + 1.0 * np.einsum('nbFe,FAenIJ->AbIJ', H.ab.ovvv[oa, vb, Va, vb], R.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('nbFE,FAEnIJ->AbIJ', H.ab.ovvv[oa, vb, Va, Vb], R.abb.VVVoOO, optimize=True)
            + 1.0 * np.einsum('Nbfe,fAeNIJ->AbIJ', H.ab.ovvv[Oa, vb, va, vb], R.abb.vVvOOO, optimize=True)
            + 1.0 * np.einsum('NbfE,fAENIJ->AbIJ', H.ab.ovvv[Oa, vb, va, Vb], R.abb.vVVOOO, optimize=True)
            + 1.0 * np.einsum('NbFe,FAeNIJ->AbIJ', H.ab.ovvv[Oa, vb, Va, vb], R.abb.VVvOOO, optimize=True)
            + 1.0 * np.einsum('NbFE,FAENIJ->AbIJ', H.ab.ovvv[Oa, vb, Va, Vb], R.abb.VVVOOO, optimize=True)
    )
    dR.bb[Vb, vb, Ob, Ob] += (2.0 / 2.0) * (
            -0.5 * np.einsum('mnIf,AfbmnJ->AbIJ', H.bb.ooov[ob, ob, Ob, vb], R.bbb.VvvooO, optimize=True)
            + 0.5 * np.einsum('mnIF,FAbmnJ->AbIJ', H.bb.ooov[ob, ob, Ob, Vb], R.bbb.VVvooO, optimize=True)
            + 1.0 * np.einsum('MnIf,AfbnMJ->AbIJ', H.bb.ooov[Ob, ob, Ob, vb], R.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('MnIF,FAbnMJ->AbIJ', H.bb.ooov[Ob, ob, Ob, Vb], R.bbb.VVvoOO, optimize=True)
            + 0.5 * np.einsum('MNIf,AfbMJN->AbIJ', H.bb.ooov[Ob, Ob, Ob, vb], R.bbb.VvvOOO, optimize=True)
            - 0.5 * np.einsum('MNIF,FAbMJN->AbIJ', H.bb.ooov[Ob, Ob, Ob, Vb], R.bbb.VVvOOO, optimize=True)
    )
    dR.bb[Vb, vb, Ob, Ob] += (2.0 / 2.0) * (
            -1.0 * np.einsum('nmfI,fAbnmJ->AbIJ', H.ab.oovo[oa, ob, va, Ob], R.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('nmFI,FAbnmJ->AbIJ', H.ab.oovo[oa, ob, Va, Ob], R.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('NmfI,fAbNmJ->AbIJ', H.ab.oovo[Oa, ob, va, Ob], R.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('NmFI,FAbNmJ->AbIJ', H.ab.oovo[Oa, ob, Va, Ob], R.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('nMfI,fAbnMJ->AbIJ', H.ab.oovo[oa, Ob, va, Ob], R.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('nMFI,FAbnMJ->AbIJ', H.ab.oovo[oa, Ob, Va, Ob], R.abb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('NMfI,fAbNMJ->AbIJ', H.ab.oovo[Oa, Ob, va, Ob], R.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('NMFI,FAbNMJ->AbIJ', H.ab.oovo[Oa, Ob, Va, Ob], R.abb.VVvOOO, optimize=True)
    )

    dR.bb[Vb, vb, Ob, Ob] -= np.transpose(dR.bb[Vb, vb, Ob, Ob], (0, 1, 3, 2))
    dR.bb[vb, Vb, Ob, Ob] = -1.0 * np.transpose(dR.bb[Vb, vb, Ob, Ob], (1, 0, 2, 3))
    return dR


def build_1100(dR, R, T, H, X, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    dR.bb[Vb, Vb, ob, ob] = (1.0 / 4.0) * (
            -1.0 * np.einsum('Me,eBAMij->ABij', X.a.ov[Oa, va], T.abb.vVVOoo, optimize=True)
            - 1.0 * np.einsum('ME,EBAMij->ABij', X.a.ov[Oa, Va], T.abb.VVVOoo, optimize=True)
    )
    dR.bb[Vb, Vb, ob, ob] += (1.0 / 4.0) * (
            -1.0 * np.einsum('Me,BAeijM->ABij', X.b.ov[Ob, vb], T.bbb.VVvooO, optimize=True)
            - 1.0 * np.einsum('ME,EBAijM->ABij', X.b.ov[Ob, Vb], T.bbb.VVVooO, optimize=True)
    )
    dR.bb[Vb, Vb, ob, ob] += (1.0 / 4.0) * (
            -1.0 * np.einsum('Me,eBAMij->ABij', H.a.ov[Oa, va], R.abb.vVVOoo, optimize=True)
            - 1.0 * np.einsum('ME,EBAMij->ABij', H.a.ov[Oa, Va], R.abb.VVVOoo, optimize=True)
    )
    dR.bb[Vb, Vb, ob, ob] += (1.0 / 4.0) * (
            -1.0 * np.einsum('Me,BAeijM->ABij', H.b.ov[Ob, vb], R.bbb.VVvooO, optimize=True)
            - 1.0 * np.einsum('ME,EBAijM->ABij', H.b.ov[Ob, Vb], R.bbb.VVVooO, optimize=True)
    )
    dR.bb[Vb, Vb, ob, ob] += (2.0 / 4.0) * (
            +0.5 * np.einsum('ANef,BfeijN->ABij', H.bb.vovv[Vb, Ob, vb, vb], R.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('ANeF,FBeijN->ABij', H.bb.vovv[Vb, Ob, vb, Vb], R.bbb.VVvooO, optimize=True)
            - 0.5 * np.einsum('ANEF,FBEijN->ABij', H.bb.vovv[Vb, Ob, Vb, Vb], R.bbb.VVVooO, optimize=True)
    )
    dR.bb[Vb, Vb, ob, ob] += (2.0 / 4.0) * (
            -1.0 * np.einsum('NAfe,fBeNij->ABij', H.ab.ovvv[Oa, Vb, va, vb], R.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('NAfE,fBENij->ABij', H.ab.ovvv[Oa, Vb, va, Vb], R.abb.vVVOoo, optimize=True)
            - 1.0 * np.einsum('NAFe,FBeNij->ABij', H.ab.ovvv[Oa, Vb, Va, vb], R.abb.VVvOoo, optimize=True)
            - 1.0 * np.einsum('NAFE,FBENij->ABij', H.ab.ovvv[Oa, Vb, Va, Vb], R.abb.VVVOoo, optimize=True)
    )
    dR.bb[Vb, Vb, ob, ob] += (2.0 / 4.0) * (
            +1.0 * np.einsum('mNif,BAfmjN->ABij', H.bb.ooov[ob, Ob, ob, vb], R.bbb.VVvooO, optimize=True)
            - 0.5 * np.einsum('MNif,BAfjMN->ABij', H.bb.ooov[Ob, Ob, ob, vb], R.bbb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('mNiF,FBAmjN->ABij', H.bb.ooov[ob, Ob, ob, Vb], R.bbb.VVVooO, optimize=True)
            - 0.5 * np.einsum('MNiF,FBAjMN->ABij', H.bb.ooov[Ob, Ob, ob, Vb], R.bbb.VVVoOO, optimize=True)
    )
    dR.bb[Vb, Vb, ob, ob] += (2.0 / 4.0) * (
            -1.0 * np.einsum('nMfi,fBAnjM->ABij', H.ab.oovo[oa, Ob, va, ob], R.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('nMFi,FBAnjM->ABij', H.ab.oovo[oa, Ob, Va, ob], R.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('Nmfi,fBANmj->ABij', H.ab.oovo[Oa, ob, va, ob], R.abb.vVVOoo, optimize=True)
            - 1.0 * np.einsum('NMfi,fBANjM->ABij', H.ab.oovo[Oa, Ob, va, ob], R.abb.vVVOoO, optimize=True)
            + 1.0 * np.einsum('NmFi,FBANmj->ABij', H.ab.oovo[Oa, ob, Va, ob], R.abb.VVVOoo, optimize=True)
            - 1.0 * np.einsum('NMFi,FBANjM->ABij', H.ab.oovo[Oa, Ob, Va, ob], R.abb.VVVOoO, optimize=True)
    )

    dR.bb[Vb, Vb, ob, ob] -= np.transpose(dR.bb[Vb, Vb, ob, ob], (1, 0, 2, 3))
    dR.bb[Vb, Vb, ob, ob] -= np.transpose(dR.bb[Vb, Vb, ob, ob], (0, 1, 3, 2))

    return dR


def build_0011(dR, R, T, H, X, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    dR.bb[vb, vb, Ob, Ob] = (1.0 / 4.0) * (
            -1.0 * np.einsum('mE,EbamIJ->abIJ', X.a.ov[oa, Va], T.abb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('ME,EbaMIJ->abIJ', X.a.ov[Oa, Va], T.abb.VvvOOO, optimize=True)
    )
    dR.bb[vb, vb, Ob, Ob] += (1.0 / 4.0) * (
            -1.0 * np.einsum('mE,EbamIJ->abIJ', X.b.ov[ob, Vb], T.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('ME,EbaIJM->abIJ', X.b.ov[Ob, Vb], T.bbb.VvvOOO, optimize=True)
    )
    dR.bb[vb, vb, Ob, Ob] += (1.0 / 4.0) * (
            -1.0 * np.einsum('mE,EbamIJ->abIJ', H.a.ov[oa, Va], R.abb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('ME,EbaMIJ->abIJ', H.a.ov[Oa, Va], R.abb.VvvOOO, optimize=True)
    )
    dR.bb[vb, vb, Ob, Ob] += (1.0 / 4.0) * (
            -1.0 * np.einsum('mE,EbamIJ->abIJ', H.b.ov[ob, Vb], R.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('ME,EbaIJM->abIJ', H.b.ov[Ob, Vb], R.bbb.VvvOOO, optimize=True)
    )
    dR.bb[vb, vb, Ob, Ob] += (2.0 / 4.0) * (
            -1.0 * np.einsum('anEf,EfbnIJ->abIJ', H.bb.vovv[vb, ob, Vb, vb], R.bbb.VvvoOO, optimize=True)
            + 0.5 * np.einsum('anEF,FEbnIJ->abIJ', H.bb.vovv[vb, ob, Vb, Vb], R.bbb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('aNEf,EfbIJN->abIJ', H.bb.vovv[vb, Ob, Vb, vb], R.bbb.VvvOOO, optimize=True)
            + 0.5 * np.einsum('aNEF,FEbIJN->abIJ', H.bb.vovv[vb, Ob, Vb, Vb], R.bbb.VVvOOO, optimize=True)
    )
    dR.bb[vb, vb, Ob, Ob] += (2.0 / 4.0) * (
            -1.0 * np.einsum('naFe,FbenIJ->abIJ', H.ab.ovvv[oa, vb, Va, vb], R.abb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('NaFe,FbeNIJ->abIJ', H.ab.ovvv[Oa, vb, Va, vb], R.abb.VvvOOO, optimize=True)
            + 1.0 * np.einsum('nafE,fEbnIJ->abIJ', H.ab.ovvv[oa, vb, va, Vb], R.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('naFE,FEbnIJ->abIJ', H.ab.ovvv[oa, vb, Va, Vb], R.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('NafE,fEbNIJ->abIJ', H.ab.ovvv[Oa, vb, va, Vb], R.abb.vVvOOO, optimize=True)
            + 1.0 * np.einsum('NaFE,FEbNIJ->abIJ', H.ab.ovvv[Oa, vb, Va, Vb], R.abb.VVvOOO, optimize=True)
    )
    dR.bb[vb, vb, Ob, Ob] += (2.0 / 4.0) * (
            -0.5 * np.einsum('mnIF,FbamnJ->abIJ', H.bb.ooov[ob, ob, Ob, Vb], R.bbb.VvvooO, optimize=True)
            + 1.0 * np.einsum('mNIF,FbamJN->abIJ', H.bb.ooov[ob, Ob, Ob, Vb], R.bbb.VvvoOO, optimize=True)
            + 0.5 * np.einsum('MNIF,FbaMJN->abIJ', H.bb.ooov[Ob, Ob, Ob, Vb], R.bbb.VvvOOO, optimize=True)
    )
    dR.bb[vb, vb, Ob, Ob] += (2.0 / 4.0) * (
            +1.0 * np.einsum('nmFI,FbanmJ->abIJ', H.ab.oovo[oa, ob, Va, Ob], R.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('nMFI,FbanMJ->abIJ', H.ab.oovo[oa, Ob, Va, Ob], R.abb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('NmFI,FbaNmJ->abIJ', H.ab.oovo[Oa, ob, Va, Ob], R.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('NMFI,FbaNMJ->abIJ', H.ab.oovo[Oa, Ob, Va, Ob], R.abb.VvvOOO, optimize=True)
    )

    dR.bb[vb, vb, Ob, Ob] -= np.transpose(dR.bb[vb, vb, Ob, Ob], (1, 0, 2, 3))
    dR.bb[vb, vb, Ob, Ob] -= np.transpose(dR.bb[vb, vb, Ob, Ob], (0, 1, 3, 2))

    return dR


def build_1001(dR, R, T, H, X, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.bb[Vb, vb, ob, Ob] = (1.0 / 1.0) * (
            +1.0 * np.einsum('me,eAbmiJ->AbiJ', X.a.ov[oa, va], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('mE,EAbmiJ->AbiJ', X.a.ov[oa, Va], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('Me,eAbMiJ->AbiJ', X.a.ov[Oa, va], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('ME,EAbMiJ->AbiJ', X.a.ov[Oa, Va], T.abb.VVvOoO, optimize=True)
    )
    dR.bb[Vb, vb, ob, Ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('me,AebimJ->AbiJ', X.b.ov[ob, vb], T.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('mE,EAbimJ->AbiJ', X.b.ov[ob, Vb], T.bbb.VVvooO, optimize=True)
            - 1.0 * np.einsum('Me,AebiJM->AbiJ', X.b.ov[Ob, vb], T.bbb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('ME,EAbiJM->AbiJ', X.b.ov[Ob, Vb], T.bbb.VVvoOO, optimize=True)
    )
    dR.bb[Vb, vb, ob, Ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('me,eAbmiJ->AbiJ', H.a.ov[oa, va], R.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('mE,EAbmiJ->AbiJ', H.a.ov[oa, Va], R.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('Me,eAbMiJ->AbiJ', H.a.ov[Oa, va], R.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('ME,EAbMiJ->AbiJ', H.a.ov[Oa, Va], R.abb.VVvOoO, optimize=True)
    )
    dR.bb[Vb, vb, ob, Ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('me,AebimJ->AbiJ', H.b.ov[ob, vb], R.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('mE,EAbimJ->AbiJ', H.b.ov[ob, Vb], R.bbb.VVvooO, optimize=True)
            - 1.0 * np.einsum('Me,AebiJM->AbiJ', H.b.ov[Ob, vb], R.bbb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('ME,EAbiJM->AbiJ', H.b.ov[Ob, Vb], R.bbb.VVvoOO, optimize=True)
    )
    dR.bb[Vb, vb, ob, Ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('AneF,FbeinJ->AbiJ', H.bb.vovv[Vb, ob, vb, Vb], R.bbb.VvvooO, optimize=True)
            - 0.5 * np.einsum('AnEF,FEbinJ->AbiJ', H.bb.vovv[Vb, ob, Vb, Vb], R.bbb.VVvooO, optimize=True)
            - 1.0 * np.einsum('ANeF,FbeiJN->AbiJ', H.bb.vovv[Vb, Ob, vb, Vb], R.bbb.VvvoOO, optimize=True)
            + 0.5 * np.einsum('ANEF,FEbiJN->AbiJ', H.bb.vovv[Vb, Ob, Vb, Vb], R.bbb.VVvoOO, optimize=True)
    )
    dR.bb[Vb, vb, ob, Ob] += (1.0 / 1.0) * (
            +0.5 * np.einsum('bnef,AfeinJ->AbiJ', H.bb.vovv[vb, ob, vb, vb], R.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('bneF,FAeinJ->AbiJ', H.bb.vovv[vb, ob, vb, Vb], R.bbb.VVvooO, optimize=True)
            - 0.5 * np.einsum('bnEF,FAEinJ->AbiJ', H.bb.vovv[vb, ob, Vb, Vb], R.bbb.VVVooO, optimize=True)
            - 0.5 * np.einsum('bNef,AfeiJN->AbiJ', H.bb.vovv[vb, Ob, vb, vb], R.bbb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('bNeF,FAeiJN->AbiJ', H.bb.vovv[vb, Ob, vb, Vb], R.bbb.VVvoOO, optimize=True)
            + 0.5 * np.einsum('bNEF,FAEiJN->AbiJ', H.bb.vovv[vb, Ob, Vb, Vb], R.bbb.VVVoOO, optimize=True)
    )
    dR.bb[Vb, vb, ob, Ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('nAfE,fEbniJ->AbiJ', H.ab.ovvv[oa, Vb, va, Vb], R.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('nAFe,FbeniJ->AbiJ', H.ab.ovvv[oa, Vb, Va, vb], R.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('nAFE,FEbniJ->AbiJ', H.ab.ovvv[oa, Vb, Va, Vb], R.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('NAfE,fEbNiJ->AbiJ', H.ab.ovvv[Oa, Vb, va, Vb], R.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('NAFe,FbeNiJ->AbiJ', H.ab.ovvv[Oa, Vb, Va, vb], R.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('NAFE,FEbNiJ->AbiJ', H.ab.ovvv[Oa, Vb, Va, Vb], R.abb.VVvOoO, optimize=True)
    )
    dR.bb[Vb, vb, ob, Ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('nbfe,fAeniJ->AbiJ', H.ab.ovvv[oa, vb, va, vb], R.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('nbfE,fAEniJ->AbiJ', H.ab.ovvv[oa, vb, va, Vb], R.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('nbFe,FAeniJ->AbiJ', H.ab.ovvv[oa, vb, Va, vb], R.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('nbFE,FAEniJ->AbiJ', H.ab.ovvv[oa, vb, Va, Vb], R.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('Nbfe,fAeNiJ->AbiJ', H.ab.ovvv[Oa, vb, va, vb], R.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('NbfE,fAENiJ->AbiJ', H.ab.ovvv[Oa, vb, va, Vb], R.abb.vVVOoO, optimize=True)
            + 1.0 * np.einsum('NbFe,FAeNiJ->AbiJ', H.ab.ovvv[Oa, vb, Va, vb], R.abb.VVvOoO, optimize=True)
            + 1.0 * np.einsum('NbFE,FAENiJ->AbiJ', H.ab.ovvv[Oa, vb, Va, Vb], R.abb.VVVOoO, optimize=True)
    )
    dR.bb[Vb, vb, ob, Ob] += (1.0 / 1.0) * (
            -0.5 * np.einsum('mnif,AfbmnJ->AbiJ', H.bb.ooov[ob, ob, ob, vb], R.bbb.VvvooO, optimize=True)
            + 0.5 * np.einsum('mniF,FAbmnJ->AbiJ', H.bb.ooov[ob, ob, ob, Vb], R.bbb.VVvooO, optimize=True)
            + 1.0 * np.einsum('mNif,AfbmJN->AbiJ', H.bb.ooov[ob, Ob, ob, vb], R.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('mNiF,FAbmJN->AbiJ', H.bb.ooov[ob, Ob, ob, Vb], R.bbb.VVvoOO, optimize=True)
            + 0.5 * np.einsum('MNif,AfbMJN->AbiJ', H.bb.ooov[Ob, Ob, ob, vb], R.bbb.VvvOOO, optimize=True)
            - 0.5 * np.einsum('MNiF,FAbMJN->AbiJ', H.bb.ooov[Ob, Ob, ob, Vb], R.bbb.VVvOOO, optimize=True)
    )
    dR.bb[Vb, vb, ob, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('mNJf,AfbmiN->AbiJ', H.bb.ooov[ob, Ob, Ob, vb], R.bbb.VvvooO, optimize=True)
            + 1.0 * np.einsum('mNJF,FAbmiN->AbiJ', H.bb.ooov[ob, Ob, Ob, Vb], R.bbb.VVvooO, optimize=True)
            + 0.5 * np.einsum('MNJf,AfbiMN->AbiJ', H.bb.ooov[Ob, Ob, Ob, vb], R.bbb.VvvoOO, optimize=True)
            - 0.5 * np.einsum('MNJF,FAbiMN->AbiJ', H.bb.ooov[Ob, Ob, Ob, Vb], R.bbb.VVvoOO, optimize=True)
    )
    dR.bb[Vb, vb, ob, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('nmfi,fAbnmJ->AbiJ', H.ab.oovo[oa, ob, va, ob], R.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('nmFi,FAbnmJ->AbiJ', H.ab.oovo[oa, ob, Va, ob], R.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('nMfi,fAbnMJ->AbiJ', H.ab.oovo[oa, Ob, va, ob], R.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('nMFi,FAbnMJ->AbiJ', H.ab.oovo[oa, Ob, Va, ob], R.abb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('Nmfi,fAbNmJ->AbiJ', H.ab.oovo[Oa, ob, va, ob], R.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('NmFi,FAbNmJ->AbiJ', H.ab.oovo[Oa, ob, Va, ob], R.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('NMfi,fAbNMJ->AbiJ', H.ab.oovo[Oa, Ob, va, ob], R.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('NMFi,FAbNMJ->AbiJ', H.ab.oovo[Oa, Ob, Va, ob], R.abb.VVvOOO, optimize=True)
    )
    dR.bb[Vb, vb, ob, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('nMfJ,fAbniM->AbiJ', H.ab.oovo[oa, Ob, va, Ob], R.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('nMFJ,FAbniM->AbiJ', H.ab.oovo[oa, Ob, Va, Ob], R.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('NmfJ,fAbNmi->AbiJ', H.ab.oovo[Oa, ob, va, Ob], R.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('NmFJ,FAbNmi->AbiJ', H.ab.oovo[Oa, ob, Va, Ob], R.abb.VVvOoo, optimize=True)
            - 1.0 * np.einsum('NMfJ,fAbNiM->AbiJ', H.ab.oovo[Oa, Ob, va, Ob], R.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('NMFJ,FAbNiM->AbiJ', H.ab.oovo[Oa, Ob, Va, Ob], R.abb.VVvOoO, optimize=True)
    )

    dR.bb[vb, Vb, ob, Ob] = -1.0 * np.transpose(dR.bb[Vb, vb, ob, Ob], (1, 0, 2, 3))
    dR.bb[Vb, vb, Ob, ob] = -1.0 * np.transpose(dR.bb[Vb, vb, ob, Ob], (0, 1, 3, 2))
    dR.bb[vb, Vb, Ob, ob] = np.transpose(dR.bb[Vb, vb, ob, Ob], (1, 0, 3, 2))
    return dR


def build_1000(dR, R, T, H, X, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    dR.bb[Vb, vb, ob, ob] = (1.0 / 2.0) * (
            +1.0 * np.einsum('Me,eAbMij->Abij', X.a.ov[Oa, va], T.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('ME,EAbMij->Abij', X.a.ov[Oa, Va], T.abb.VVvOoo, optimize=True)
    )
    dR.bb[Vb, vb, ob, ob] += (1.0 / 2.0) * (
            -1.0 * np.einsum('Me,AebijM->Abij', X.b.ov[Ob, vb], T.bbb.VvvooO, optimize=True)
            + 1.0 * np.einsum('ME,EAbijM->Abij', X.b.ov[Ob, Vb], T.bbb.VVvooO, optimize=True)
    )
    dR.bb[Vb, vb, ob, ob] += (1.0 / 2.0) * (
            +1.0 * np.einsum('Me,eAbMij->Abij', H.a.ov[Oa, va], R.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('ME,EAbMij->Abij', H.a.ov[Oa, Va], R.abb.VVvOoo, optimize=True)
    )
    dR.bb[Vb, vb, ob, ob] += (1.0 / 2.0) * (
            -1.0 * np.einsum('Me,AebijM->Abij', H.b.ov[Ob, vb], R.bbb.VvvooO, optimize=True)
            + 1.0 * np.einsum('ME,EAbijM->Abij', H.b.ov[Ob, Vb], R.bbb.VVvooO, optimize=True)
    )
    dR.bb[Vb, vb, ob, ob] += (1.0 / 2.0) * (
            -1.0 * np.einsum('ANEf,EfbijN->Abij', H.bb.vovv[Vb, Ob, Vb, vb], R.bbb.VvvooO, optimize=True)
            + 0.5 * np.einsum('ANEF,FEbijN->Abij', H.bb.vovv[Vb, Ob, Vb, Vb], R.bbb.VVvooO, optimize=True)
    )
    dR.bb[Vb, vb, ob, ob] += (1.0 / 2.0) * (
            -0.5 * np.einsum('bNef,AfeijN->Abij', H.bb.vovv[vb, Ob, vb, vb], R.bbb.VvvooO, optimize=True)
            + 1.0 * np.einsum('bNEf,AEfijN->Abij', H.bb.vovv[vb, Ob, Vb, vb], R.bbb.VVvooO, optimize=True)
            + 0.5 * np.einsum('bNEF,FAEijN->Abij', H.bb.vovv[vb, Ob, Vb, Vb], R.bbb.VVVooO, optimize=True)
    )
    dR.bb[Vb, vb, ob, ob] += (1.0 / 2.0) * (
            -1.0 * np.einsum('NAFe,FbeNij->Abij', H.ab.ovvv[Oa, Vb, Va, vb], R.abb.VvvOoo, optimize=True)
            + 1.0 * np.einsum('NAfE,fEbNij->Abij', H.ab.ovvv[Oa, Vb, va, Vb], R.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('NAFE,FEbNij->Abij', H.ab.ovvv[Oa, Vb, Va, Vb], R.abb.VVvOoo, optimize=True)
    )
    dR.bb[Vb, vb, ob, ob] += (1.0 / 2.0) * (
            +1.0 * np.einsum('Nbfe,fAeNij->Abij', H.ab.ovvv[Oa, vb, va, vb], R.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('NbFe,FAeNij->Abij', H.ab.ovvv[Oa, vb, Va, vb], R.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('NbfE,fAENij->Abij', H.ab.ovvv[Oa, vb, va, Vb], R.abb.vVVOoo, optimize=True)
            + 1.0 * np.einsum('NbFE,FAENij->Abij', H.ab.ovvv[Oa, vb, Va, Vb], R.abb.VVVOoo, optimize=True)
    )
    dR.bb[Vb, vb, ob, ob] += (2.0 / 2.0) * (
            +1.0 * np.einsum('Mnif,AfbjnM->Abij', H.bb.ooov[Ob, ob, ob, vb], R.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('MniF,FAbjnM->Abij', H.bb.ooov[Ob, ob, ob, Vb], R.bbb.VVvooO, optimize=True)
            - 0.5 * np.einsum('MNif,AfbjMN->Abij', H.bb.ooov[Ob, Ob, ob, vb], R.bbb.VvvoOO, optimize=True)
            + 0.5 * np.einsum('MNiF,FAbjMN->Abij', H.bb.ooov[Ob, Ob, ob, Vb], R.bbb.VVvoOO, optimize=True)
    )
    dR.bb[Vb, vb, ob, ob] += (2.0 / 2.0) * (
            -1.0 * np.einsum('Nmfi,fAbNmj->Abij', H.ab.oovo[Oa, ob, va, ob], R.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('NmFi,FAbNmj->Abij', H.ab.oovo[Oa, ob, Va, ob], R.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('nMfi,fAbnjM->Abij', H.ab.oovo[oa, Ob, va, ob], R.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('nMFi,FAbnjM->Abij', H.ab.oovo[oa, Ob, Va, ob], R.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('NMfi,fAbNjM->Abij', H.ab.oovo[Oa, Ob, va, ob], R.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('NMFi,FAbNjM->Abij', H.ab.oovo[Oa, Ob, Va, ob], R.abb.VVvOoO, optimize=True)
    )

    dR.bb[Vb, vb, ob, ob] -= np.transpose(dR.bb[Vb, vb, ob, ob], (0, 1, 3, 2))
    dR.bb[vb, Vb, ob, ob] = -1.0 * np.transpose(dR.bb[Vb, vb, ob, ob], (1, 0, 2, 3))
    return dR


def build_0001(dR, R, T, H, X, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    dR.bb[vb, vb, ob, Ob] = (1.0 / 2.0) * (
            -1.0 * np.einsum('mE,EbamiJ->abiJ', X.a.ov[oa, Va], T.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('ME,EbaMiJ->abiJ', X.a.ov[Oa, Va], T.abb.VvvOoO, optimize=True)
    )
    dR.bb[vb, vb, ob, Ob] += (1.0 / 2.0) * (
            +1.0 * np.einsum('mE,EbaimJ->abiJ', X.b.ov[ob, Vb], T.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('ME,EbaiJM->abiJ', X.b.ov[Ob, Vb], T.bbb.VvvoOO, optimize=True)
    )
    dR.bb[vb, vb, ob, Ob] += (1.0 / 2.0) * (
            -1.0 * np.einsum('mE,EbamiJ->abiJ', H.a.ov[oa, Va], R.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('ME,EbaMiJ->abiJ', H.a.ov[Oa, Va], R.abb.VvvOoO, optimize=True)
    )
    dR.bb[vb, vb, ob, Ob] += (1.0 / 2.0) * (
            +1.0 * np.einsum('mE,EbaimJ->abiJ', H.b.ov[ob, Vb], R.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('ME,EbaiJM->abiJ', H.b.ov[Ob, Vb], R.bbb.VvvoOO, optimize=True)
    )
    dR.bb[vb, vb, ob, Ob] += (2.0 / 2.0) * (
            +1.0 * np.einsum('anEf,EfbinJ->abiJ', H.bb.vovv[vb, ob, Vb, vb], R.bbb.VvvooO, optimize=True)
            - 0.5 * np.einsum('anEF,FEbinJ->abiJ', H.bb.vovv[vb, ob, Vb, Vb], R.bbb.VVvooO, optimize=True)
            - 1.0 * np.einsum('aNEf,EfbiJN->abiJ', H.bb.vovv[vb, Ob, Vb, vb], R.bbb.VvvoOO, optimize=True)
            + 0.5 * np.einsum('aNEF,FEbiJN->abiJ', H.bb.vovv[vb, Ob, Vb, Vb], R.bbb.VVvoOO, optimize=True)
    )
    dR.bb[vb, vb, ob, Ob] += (2.0 / 2.0) * (
            -1.0 * np.einsum('naFe,FbeniJ->abiJ', H.ab.ovvv[oa, vb, Va, vb], R.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('NaFe,FbeNiJ->abiJ', H.ab.ovvv[Oa, vb, Va, vb], R.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('nafE,fEbniJ->abiJ', H.ab.ovvv[oa, vb, va, Vb], R.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('naFE,FEbniJ->abiJ', H.ab.ovvv[oa, vb, Va, Vb], R.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('NafE,fEbNiJ->abiJ', H.ab.ovvv[Oa, vb, va, Vb], R.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('NaFE,FEbNiJ->abiJ', H.ab.ovvv[Oa, vb, Va, Vb], R.abb.VVvOoO, optimize=True)
    )
    dR.bb[vb, vb, ob, Ob] += (1.0 / 2.0) * (
            -0.5 * np.einsum('mniF,FbamnJ->abiJ', H.bb.ooov[ob, ob, ob, Vb], R.bbb.VvvooO, optimize=True)
            + 1.0 * np.einsum('MniF,FbanMJ->abiJ', H.bb.ooov[Ob, ob, ob, Vb], R.bbb.VvvoOO, optimize=True)
            + 0.5 * np.einsum('MNiF,FbaMJN->abiJ', H.bb.ooov[Ob, Ob, ob, Vb], R.bbb.VvvOOO, optimize=True)
    )
    dR.bb[vb, vb, ob, Ob] += (1.0 / 2.0) * (
            -1.0 * np.einsum('MnJF,FbainM->abiJ', H.bb.ooov[Ob, ob, Ob, Vb], R.bbb.VvvooO, optimize=True)
            + 0.5 * np.einsum('MNJF,FbaiMN->abiJ', H.bb.ooov[Ob, Ob, Ob, Vb], R.bbb.VvvoOO, optimize=True)
    )
    dR.bb[vb, vb, ob, Ob] += (1.0 / 2.0) * (
            +1.0 * np.einsum('nmFi,FbanmJ->abiJ', H.ab.oovo[oa, ob, Va, ob], R.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('NmFi,FbaNmJ->abiJ', H.ab.oovo[Oa, ob, Va, ob], R.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('nMFi,FbanMJ->abiJ', H.ab.oovo[oa, Ob, Va, ob], R.abb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('NMFi,FbaNMJ->abiJ', H.ab.oovo[Oa, Ob, Va, ob], R.abb.VvvOOO, optimize=True)
    )
    dR.bb[vb, vb, ob, Ob] += (1.0 / 2.0) * (
            -1.0 * np.einsum('NmFJ,FbaNmi->abiJ', H.ab.oovo[Oa, ob, Va, Ob], R.abb.VvvOoo, optimize=True)
            + 1.0 * np.einsum('nMFJ,FbaniM->abiJ', H.ab.oovo[oa, Ob, Va, Ob], R.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('NMFJ,FbaNiM->abiJ', H.ab.oovo[Oa, Ob, Va, Ob], R.abb.VvvOoO, optimize=True)
    )

    dR.bb[vb, vb, ob, Ob] -= np.transpose(dR.bb[vb, vb, ob, Ob], (1, 0, 2, 3))
    dR.bb[vb, vb, Ob, ob] = -1.0 * np.transpose(dR.bb[vb, vb, ob, Ob], (0, 1, 3, 2))
    return dR


def build_0000(dR, R, T, H, X, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    dR.bb[vb, vb, ob, ob] = (1.0 / 4.0) * (
            -1.0 * np.einsum('ME,EbaMij->abij', X.a.ov[Oa, Va], T.abb.VvvOoo, optimize=True)
    )
    dR.bb[vb, vb, ob, ob] += (1.0 / 4.0) * (
            -1.0 * np.einsum('ME,EbaijM->abij', X.b.ov[Ob, Vb], T.bbb.VvvooO, optimize=True)
    )
    dR.bb[vb, vb, ob, ob] += (1.0 / 4.0) * (
            -1.0 * np.einsum('ME,EbaMij->abij', H.a.ov[Oa, Va], R.abb.VvvOoo, optimize=True)
    )
    dR.bb[vb, vb, ob, ob] += (1.0 / 4.0) * (
            -1.0 * np.einsum('ME,EbaijM->abij', H.b.ov[Ob, Vb], R.bbb.VvvooO, optimize=True)
    )
    dR.bb[vb, vb, ob, ob] += (2.0 / 4.0) * (
            -1.0 * np.einsum('aNeF,FbeijN->abij', H.bb.vovv[vb, Ob, vb, Vb], R.bbb.VvvooO, optimize=True)
            + 0.5 * np.einsum('aNEF,FEbijN->abij', H.bb.vovv[vb, Ob, Vb, Vb], R.bbb.VVvooO, optimize=True)
    )
    dR.bb[vb, vb, ob, ob] += (2.0 / 4.0) * (
            +1.0 * np.einsum('NafE,fEbNij->abij', H.ab.ovvv[Oa, vb, va, Vb], R.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('NaFe,FbeNij->abij', H.ab.ovvv[Oa, vb, Va, vb], R.abb.VvvOoo, optimize=True)
            + 1.0 * np.einsum('NaFE,FEbNij->abij', H.ab.ovvv[Oa, vb, Va, Vb], R.abb.VVvOoo, optimize=True)
    )
    dR.bb[vb, vb, ob, ob] += (2.0 / 4.0) * (
            +1.0 * np.einsum('MniF,FbajnM->abij', H.bb.ooov[Ob, ob, ob, Vb], R.bbb.VvvooO, optimize=True)
            - 0.5 * np.einsum('MNiF,FbajMN->abij', H.bb.ooov[Ob, Ob, ob, Vb], R.bbb.VvvoOO, optimize=True)
    )
    dR.bb[vb, vb, ob, ob] += (2.0 / 4.0) * (
            +1.0 * np.einsum('NmFi,FbaNmj->abij', H.ab.oovo[Oa, ob, Va, ob], R.abb.VvvOoo, optimize=True)
            - 1.0 * np.einsum('nMFi,FbanjM->abij', H.ab.oovo[oa, Ob, Va, ob], R.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('NMFi,FbaNjM->abij', H.ab.oovo[Oa, Ob, Va, ob], R.abb.VvvOoO, optimize=True)
    )

    dR.bb[vb, vb, ob, ob] -= np.transpose(dR.bb[vb, vb, ob, ob], (1, 0, 2, 3))
    dR.bb[vb, vb, ob, ob] -= np.transpose(dR.bb[vb, vb, ob, ob], (0, 1, 3, 2))

    return dR
