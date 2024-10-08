import numpy as np

from ccpy.utilities.active_space import get_active_slices

from ccpy.lib.core import cc_active_loops

#@profile
def build_ccsd(T, H, H0):
    """
    Calculate CCSD parts of the projection <i~j~a~b~|(H_N e^(T1+T2+T3))_C|0>.
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

    x2c = -0.5 * np.einsum("amij,bm->abij", I2C_vooo, T.b, optimize=True)
    x2c += 0.5 * np.einsum("abie,ej->abij", H.bb.vvov, T.b, optimize=True)
    x2c += 0.5 * np.einsum("ae,ebij->abij", H.b.vv, T.bb, optimize=True)
    x2c -= 0.5 * np.einsum("mi,abmj->abij", H.b.oo, T.bb, optimize=True)
    x2c += np.einsum("amie,ebmj->abij", I2C_voov, T.bb, optimize=True)
    x2c += np.einsum("maei,ebmj->abij", I2B_ovvo, T.ab, optimize=True)
    x2c += 0.25 * np.einsum("abef,efij->abij", H0.bb.vvvv, tau, optimize=True)
    x2c += 0.125 * np.einsum("mnij,abmn->abij", I2C_oooo, T.bb, optimize=True)
    x2c += 0.25 * H0.bb.vvoo
    x2c -= np.transpose(x2c, (1, 0, 2, 3))
    x2c -= np.transpose(x2c, (0, 1, 3, 2))
    return x2c

def build_1111(T, dT, H, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    dT.bb[Vb, Vb, Ob, Ob] = (1.0 / 4.0) * (
            -1.0 * np.einsum('me,eBAmIJ->ABIJ', H.a.ov[oa, va], T.abb.vVVoOO, optimize=True)
            - 1.0 * np.einsum('Me,eBAMIJ->ABIJ', H.a.ov[Oa, va], T.abb.vVVOOO, optimize=True)
            - 1.0 * np.einsum('mE,EBAmIJ->ABIJ', H.a.ov[oa, Va], T.abb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('ME,EBAMIJ->ABIJ', H.a.ov[Oa, Va], T.abb.VVVOOO, optimize=True)
    )
    dT.bb[Vb, Vb, Ob, Ob] += (1.0 / 4.0) * (
            -1.0 * np.einsum('me,BAemIJ->ABIJ', H.b.ov[ob, vb], T.bbb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('Me,BAeIJM->ABIJ', H.b.ov[Ob, vb], T.bbb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('mE,EBAmIJ->ABIJ', H.b.ov[ob, Vb], T.bbb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('ME,EBAIJM->ABIJ', H.b.ov[Ob, Vb], T.bbb.VVVOOO, optimize=True)
    )
    dT.bb[Vb, Vb, Ob, Ob] += (2.0 / 4.0) * (
            +0.5 * np.einsum('Anef,BfenIJ->ABIJ', H.bb.vovv[Vb, ob, vb, vb], T.bbb.VvvoOO, optimize=True)
            + 0.5 * np.einsum('ANef,BfeIJN->ABIJ', H.bb.vovv[Vb, Ob, vb, vb], T.bbb.VvvOOO, optimize=True)
            - 1.0 * np.einsum('AnEf,BEfnIJ->ABIJ', H.bb.vovv[Vb, ob, Vb, vb], T.bbb.VVvoOO, optimize=True)
            - 0.5 * np.einsum('AnEF,FBEnIJ->ABIJ', H.bb.vovv[Vb, ob, Vb, Vb], T.bbb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('ANEf,BEfIJN->ABIJ', H.bb.vovv[Vb, Ob, Vb, vb], T.bbb.VVvOOO, optimize=True)
            - 0.5 * np.einsum('ANEF,FBEIJN->ABIJ', H.bb.vovv[Vb, Ob, Vb, Vb], T.bbb.VVVOOO, optimize=True)
    )
    dT.bb[Vb, Vb, Ob, Ob] += (2.0 / 4.0) * (
            -1.0 * np.einsum('nAfe,fBenIJ->ABIJ', H.ab.ovvv[oa, Vb, va, vb], T.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('nAFe,FBenIJ->ABIJ', H.ab.ovvv[oa, Vb, Va, vb], T.abb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('NAfe,fBeNIJ->ABIJ', H.ab.ovvv[Oa, Vb, va, vb], T.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('NAFe,FBeNIJ->ABIJ', H.ab.ovvv[Oa, Vb, Va, vb], T.abb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('nAfE,fBEnIJ->ABIJ', H.ab.ovvv[oa, Vb, va, Vb], T.abb.vVVoOO, optimize=True)
            - 1.0 * np.einsum('nAFE,FBEnIJ->ABIJ', H.ab.ovvv[oa, Vb, Va, Vb], T.abb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('NAfE,fBENIJ->ABIJ', H.ab.ovvv[Oa, Vb, va, Vb], T.abb.vVVOOO, optimize=True)
            - 1.0 * np.einsum('NAFE,FBENIJ->ABIJ', H.ab.ovvv[Oa, Vb, Va, Vb], T.abb.VVVOOO, optimize=True)
    )
    dT.bb[Vb, Vb, Ob, Ob] += (2.0 / 4.0) * (
            -0.5 * np.einsum('mnIf,BAfmnJ->ABIJ', H.bb.ooov[ob, ob, Ob, vb], T.bbb.VVvooO, optimize=True)
            - 0.5 * np.einsum('mnIF,FBAmnJ->ABIJ', H.bb.ooov[ob, ob, Ob, Vb], T.bbb.VVVooO, optimize=True)
            + 1.0 * np.einsum('MnIf,BAfnMJ->ABIJ', H.bb.ooov[Ob, ob, Ob, vb], T.bbb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('MnIF,FBAnMJ->ABIJ', H.bb.ooov[Ob, ob, Ob, Vb], T.bbb.VVVoOO, optimize=True)
            + 0.5 * np.einsum('MNIf,BAfMJN->ABIJ', H.bb.ooov[Ob, Ob, Ob, vb], T.bbb.VVvOOO, optimize=True)
            + 0.5 * np.einsum('MNIF,FBAMJN->ABIJ', H.bb.ooov[Ob, Ob, Ob, Vb], T.bbb.VVVOOO, optimize=True)
    )
    dT.bb[Vb, Vb, Ob, Ob] += (2.0 / 4.0) * (
            +1.0 * np.einsum('nmfI,fBAnmJ->ABIJ', H.ab.oovo[oa, ob, va, Ob], T.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('nmFI,FBAnmJ->ABIJ', H.ab.oovo[oa, ob, Va, Ob], T.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('NmfI,fBANmJ->ABIJ', H.ab.oovo[Oa, ob, va, Ob], T.abb.vVVOoO, optimize=True)
            + 1.0 * np.einsum('NmFI,FBANmJ->ABIJ', H.ab.oovo[Oa, ob, Va, Ob], T.abb.VVVOoO, optimize=True)
            + 1.0 * np.einsum('nMfI,fBAnMJ->ABIJ', H.ab.oovo[oa, Ob, va, Ob], T.abb.vVVoOO, optimize=True)
            + 1.0 * np.einsum('nMFI,FBAnMJ->ABIJ', H.ab.oovo[oa, Ob, Va, Ob], T.abb.VVVoOO, optimize=True)
            + 1.0 * np.einsum('NMfI,fBANMJ->ABIJ', H.ab.oovo[Oa, Ob, va, Ob], T.abb.vVVOOO, optimize=True)
            + 1.0 * np.einsum('NMFI,FBANMJ->ABIJ', H.ab.oovo[Oa, Ob, Va, Ob], T.abb.VVVOOO, optimize=True)
    )

    dT.bb[Vb, Vb, Ob, Ob] -= np.transpose(dT.bb[Vb, Vb, Ob, Ob], (1, 0, 2, 3))
    dT.bb[Vb, Vb, Ob, Ob] -= np.transpose(dT.bb[Vb, Vb, Ob, Ob], (0, 1, 3, 2))

    return dT


def build_1101(T, dT, H, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dT.bb[Vb, Vb, ob, Ob] = (1.0 / 2.0) * (
            -1.0 * np.einsum('me,eBAmiJ->ABiJ', H.a.ov[oa, va], T.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('mE,EBAmiJ->ABiJ', H.a.ov[oa, Va], T.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('Me,eBAMiJ->ABiJ', H.a.ov[Oa, va], T.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('ME,EBAMiJ->ABiJ', H.a.ov[Oa, Va], T.abb.VVVOoO, optimize=True)
    )
    dT.bb[Vb, Vb, ob, Ob] += (1.0 / 2.0) * (
            +1.0 * np.einsum('me,BAeimJ->ABiJ', H.b.ov[ob, vb], T.bbb.VVvooO, optimize=True)
            + 1.0 * np.einsum('mE,EBAimJ->ABiJ', H.b.ov[ob, Vb], T.bbb.VVVooO, optimize=True)
            - 1.0 * np.einsum('Me,BAeiJM->ABiJ', H.b.ov[Ob, vb], T.bbb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('ME,EBAiJM->ABiJ', H.b.ov[Ob, Vb], T.bbb.VVVoOO, optimize=True)
    )
    dT.bb[Vb, Vb, ob, Ob] += (2.0 / 2.0) * (
            -0.5 * np.einsum('Anef,BfeinJ->ABiJ', H.bb.vovv[Vb, ob, vb, vb], T.bbb.VvvooO, optimize=True)
            + 1.0 * np.einsum('AneF,FBeinJ->ABiJ', H.bb.vovv[Vb, ob, vb, Vb], T.bbb.VVvooO, optimize=True)
            + 0.5 * np.einsum('AnEF,FBEinJ->ABiJ', H.bb.vovv[Vb, ob, Vb, Vb], T.bbb.VVVooO, optimize=True)
            + 0.5 * np.einsum('ANef,BfeiJN->ABiJ', H.bb.vovv[Vb, Ob, vb, vb], T.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('ANeF,FBeiJN->ABiJ', H.bb.vovv[Vb, Ob, vb, Vb], T.bbb.VVvoOO, optimize=True)
            - 0.5 * np.einsum('ANEF,FBEiJN->ABiJ', H.bb.vovv[Vb, Ob, Vb, Vb], T.bbb.VVVoOO, optimize=True)
    )
    dT.bb[Vb, Vb, ob, Ob] += (2.0 / 2.0) * (
            -1.0 * np.einsum('nAfe,fBeniJ->ABiJ', H.ab.ovvv[oa, Vb, va, vb], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('nAfE,fBEniJ->ABiJ', H.ab.ovvv[oa, Vb, va, Vb], T.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('nAFe,FBeniJ->ABiJ', H.ab.ovvv[oa, Vb, Va, vb], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('nAFE,FBEniJ->ABiJ', H.ab.ovvv[oa, Vb, Va, Vb], T.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('NAfe,fBeNiJ->ABiJ', H.ab.ovvv[Oa, Vb, va, vb], T.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('NAfE,fBENiJ->ABiJ', H.ab.ovvv[Oa, Vb, va, Vb], T.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('NAFe,FBeNiJ->ABiJ', H.ab.ovvv[Oa, Vb, Va, vb], T.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('NAFE,FBENiJ->ABiJ', H.ab.ovvv[Oa, Vb, Va, Vb], T.abb.VVVOoO, optimize=True)
    )
    dT.bb[Vb, Vb, ob, Ob] += (1.0 / 2.0) * (
            -0.5 * np.einsum('mnif,BAfmnJ->ABiJ', H.bb.ooov[ob, ob, ob, vb], T.bbb.VVvooO, optimize=True)
            - 0.5 * np.einsum('mniF,FBAmnJ->ABiJ', H.bb.ooov[ob, ob, ob, Vb], T.bbb.VVVooO, optimize=True)
            + 1.0 * np.einsum('Mnif,BAfnMJ->ABiJ', H.bb.ooov[Ob, ob, ob, vb], T.bbb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('MniF,FBAnMJ->ABiJ', H.bb.ooov[Ob, ob, ob, Vb], T.bbb.VVVoOO, optimize=True)
            + 0.5 * np.einsum('MNif,BAfMJN->ABiJ', H.bb.ooov[Ob, Ob, ob, vb], T.bbb.VVvOOO, optimize=True)
            + 0.5 * np.einsum('MNiF,FBAMJN->ABiJ', H.bb.ooov[Ob, Ob, ob, Vb], T.bbb.VVVOOO, optimize=True)
    )
    dT.bb[Vb, Vb, ob, Ob] += (1.0 / 2.0) * (
            -1.0 * np.einsum('MnJf,BAfinM->ABiJ', H.bb.ooov[Ob, ob, Ob, vb], T.bbb.VVvooO, optimize=True)
            - 1.0 * np.einsum('MnJF,FBAinM->ABiJ', H.bb.ooov[Ob, ob, Ob, Vb], T.bbb.VVVooO, optimize=True)
            + 0.5 * np.einsum('MNJf,BAfiMN->ABiJ', H.bb.ooov[Ob, Ob, Ob, vb], T.bbb.VVvoOO, optimize=True)
            + 0.5 * np.einsum('MNJF,FBAiMN->ABiJ', H.bb.ooov[Ob, Ob, Ob, Vb], T.bbb.VVVoOO, optimize=True)
    )
    dT.bb[Vb, Vb, ob, Ob] += (1.0 / 2.0) * (
            +1.0 * np.einsum('nmfi,fBAnmJ->ABiJ', H.ab.oovo[oa, ob, va, ob], T.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('nmFi,FBAnmJ->ABiJ', H.ab.oovo[oa, ob, Va, ob], T.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('Nmfi,fBANmJ->ABiJ', H.ab.oovo[Oa, ob, va, ob], T.abb.vVVOoO, optimize=True)
            + 1.0 * np.einsum('NmFi,FBANmJ->ABiJ', H.ab.oovo[Oa, ob, Va, ob], T.abb.VVVOoO, optimize=True)
            + 1.0 * np.einsum('nMfi,fBAnMJ->ABiJ', H.ab.oovo[oa, Ob, va, ob], T.abb.vVVoOO, optimize=True)
            + 1.0 * np.einsum('nMFi,FBAnMJ->ABiJ', H.ab.oovo[oa, Ob, Va, ob], T.abb.VVVoOO, optimize=True)
            + 1.0 * np.einsum('NMfi,fBANMJ->ABiJ', H.ab.oovo[Oa, Ob, va, ob], T.abb.vVVOOO, optimize=True)
            + 1.0 * np.einsum('NMFi,FBANMJ->ABiJ', H.ab.oovo[Oa, Ob, Va, ob], T.abb.VVVOOO, optimize=True)
    )
    dT.bb[Vb, Vb, ob, Ob] += (1.0 / 2.0) * (
            -1.0 * np.einsum('NmfJ,fBANmi->ABiJ', H.ab.oovo[Oa, ob, va, Ob], T.abb.vVVOoo, optimize=True)
            - 1.0 * np.einsum('NmFJ,FBANmi->ABiJ', H.ab.oovo[Oa, ob, Va, Ob], T.abb.VVVOoo, optimize=True)
            + 1.0 * np.einsum('nMfJ,fBAniM->ABiJ', H.ab.oovo[oa, Ob, va, Ob], T.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('nMFJ,FBAniM->ABiJ', H.ab.oovo[oa, Ob, Va, Ob], T.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('NMfJ,fBANiM->ABiJ', H.ab.oovo[Oa, Ob, va, Ob], T.abb.vVVOoO, optimize=True)
            + 1.0 * np.einsum('NMFJ,FBANiM->ABiJ', H.ab.oovo[Oa, Ob, Va, Ob], T.abb.VVVOoO, optimize=True)
    )
    dT.bb[Vb, Vb, ob, Ob] -= np.transpose(dT.bb[Vb, Vb, ob, Ob], (1, 0, 2, 3))
    dT.bb[Vb, Vb, Ob, ob] = -1.0 * np.transpose(dT.bb[Vb, Vb, ob, Ob], (0, 1, 3, 2))
    return dT


def build_1011(T, dT, H, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dT.bb[Vb, vb, Ob, Ob] = (1.0 / 2.0) * (
            +1.0 * np.einsum('me,eAbmIJ->AbIJ', H.a.ov[oa, va], T.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('mE,EAbmIJ->AbIJ', H.a.ov[oa, Va], T.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('Me,eAbMIJ->AbIJ', H.a.ov[Oa, va], T.abb.vVvOOO, optimize=True)
            + 1.0 * np.einsum('ME,EAbMIJ->AbIJ', H.a.ov[Oa, Va], T.abb.VVvOOO, optimize=True)
    )
    dT.bb[Vb, vb, Ob, Ob] += (1.0 / 2.0) * (
            -1.0 * np.einsum('me,AebmIJ->AbIJ', H.b.ov[ob, vb], T.bbb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('mE,EAbmIJ->AbIJ', H.b.ov[ob, Vb], T.bbb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('Me,AebIJM->AbIJ', H.b.ov[Ob, vb], T.bbb.VvvOOO, optimize=True)
            + 1.0 * np.einsum('ME,EAbIJM->AbIJ', H.b.ov[Ob, Vb], T.bbb.VVvOOO, optimize=True)
    )
    dT.bb[Vb, vb, Ob, Ob] += (1.0 / 2.0) * (
            -1.0 * np.einsum('AnEf,EfbnIJ->AbIJ', H.bb.vovv[Vb, ob, Vb, vb], T.bbb.VvvoOO, optimize=True)
            + 0.5 * np.einsum('AnEF,FEbnIJ->AbIJ', H.bb.vovv[Vb, ob, Vb, Vb], T.bbb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('ANEf,EfbIJN->AbIJ', H.bb.vovv[Vb, Ob, Vb, vb], T.bbb.VvvOOO, optimize=True)
            + 0.5 * np.einsum('ANEF,FEbIJN->AbIJ', H.bb.vovv[Vb, Ob, Vb, Vb], T.bbb.VVvOOO, optimize=True)
    )
    dT.bb[Vb, vb, Ob, Ob] += (1.0 / 2.0) * (
            -0.5 * np.einsum('bnef,AfenIJ->AbIJ', H.bb.vovv[vb, ob, vb, vb], T.bbb.VvvoOO, optimize=True)
            - 0.5 * np.einsum('bNef,AfeIJN->AbIJ', H.bb.vovv[vb, Ob, vb, vb], T.bbb.VvvOOO, optimize=True)
            + 1.0 * np.einsum('bnEf,AEfnIJ->AbIJ', H.bb.vovv[vb, ob, Vb, vb], T.bbb.VVvoOO, optimize=True)
            + 0.5 * np.einsum('bnEF,FAEnIJ->AbIJ', H.bb.vovv[vb, ob, Vb, Vb], T.bbb.VVVoOO, optimize=True)
            + 1.0 * np.einsum('bNEf,AEfIJN->AbIJ', H.bb.vovv[vb, Ob, Vb, vb], T.bbb.VVvOOO, optimize=True)
            + 0.5 * np.einsum('bNEF,FAEIJN->AbIJ', H.bb.vovv[vb, Ob, Vb, Vb], T.bbb.VVVOOO, optimize=True)
    )
    dT.bb[Vb, vb, Ob, Ob] += (1.0 / 2.0) * (
            -1.0 * np.einsum('nAFe,FbenIJ->AbIJ', H.ab.ovvv[oa, Vb, Va, vb], T.abb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('NAFe,FbeNIJ->AbIJ', H.ab.ovvv[Oa, Vb, Va, vb], T.abb.VvvOOO, optimize=True)
            + 1.0 * np.einsum('nAfE,fEbnIJ->AbIJ', H.ab.ovvv[oa, Vb, va, Vb], T.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('nAFE,FEbnIJ->AbIJ', H.ab.ovvv[oa, Vb, Va, Vb], T.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('NAfE,fEbNIJ->AbIJ', H.ab.ovvv[Oa, Vb, va, Vb], T.abb.vVvOOO, optimize=True)
            + 1.0 * np.einsum('NAFE,FEbNIJ->AbIJ', H.ab.ovvv[Oa, Vb, Va, Vb], T.abb.VVvOOO, optimize=True)
    )
    dT.bb[Vb, vb, Ob, Ob] += (1.0 / 2.0) * (
            +1.0 * np.einsum('nbfe,fAenIJ->AbIJ', H.ab.ovvv[oa, vb, va, vb], T.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('nbFe,FAenIJ->AbIJ', H.ab.ovvv[oa, vb, Va, vb], T.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('Nbfe,fAeNIJ->AbIJ', H.ab.ovvv[Oa, vb, va, vb], T.abb.vVvOOO, optimize=True)
            + 1.0 * np.einsum('NbFe,FAeNIJ->AbIJ', H.ab.ovvv[Oa, vb, Va, vb], T.abb.VVvOOO, optimize=True)
            + 1.0 * np.einsum('nbfE,fAEnIJ->AbIJ', H.ab.ovvv[oa, vb, va, Vb], T.abb.vVVoOO, optimize=True)
            + 1.0 * np.einsum('nbFE,FAEnIJ->AbIJ', H.ab.ovvv[oa, vb, Va, Vb], T.abb.VVVoOO, optimize=True)
            + 1.0 * np.einsum('NbfE,fAENIJ->AbIJ', H.ab.ovvv[Oa, vb, va, Vb], T.abb.vVVOOO, optimize=True)
            + 1.0 * np.einsum('NbFE,FAENIJ->AbIJ', H.ab.ovvv[Oa, vb, Va, Vb], T.abb.VVVOOO, optimize=True)
    )
    dT.bb[Vb, vb, Ob, Ob] += (2.0 / 2.0) * (
            -0.5 * np.einsum('mnIf,AfbmnJ->AbIJ', H.bb.ooov[ob, ob, Ob, vb], T.bbb.VvvooO, optimize=True)
            + 0.5 * np.einsum('mnIF,FAbmnJ->AbIJ', H.bb.ooov[ob, ob, Ob, Vb], T.bbb.VVvooO, optimize=True)
            + 1.0 * np.einsum('MnIf,AfbnMJ->AbIJ', H.bb.ooov[Ob, ob, Ob, vb], T.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('MnIF,FAbnMJ->AbIJ', H.bb.ooov[Ob, ob, Ob, Vb], T.bbb.VVvoOO, optimize=True)
            + 0.5 * np.einsum('MNIf,AfbMJN->AbIJ', H.bb.ooov[Ob, Ob, Ob, vb], T.bbb.VvvOOO, optimize=True)
            - 0.5 * np.einsum('MNIF,FAbMJN->AbIJ', H.bb.ooov[Ob, Ob, Ob, Vb], T.bbb.VVvOOO, optimize=True)
    )
    dT.bb[Vb, vb, Ob, Ob] += (2.0 / 2.0) * (
            -1.0 * np.einsum('nmfI,fAbnmJ->AbIJ', H.ab.oovo[oa, ob, va, Ob], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('nmFI,FAbnmJ->AbIJ', H.ab.oovo[oa, ob, Va, Ob], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('NmfI,fAbNmJ->AbIJ', H.ab.oovo[Oa, ob, va, Ob], T.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('NmFI,FAbNmJ->AbIJ', H.ab.oovo[Oa, ob, Va, Ob], T.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('nMfI,fAbnMJ->AbIJ', H.ab.oovo[oa, Ob, va, Ob], T.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('nMFI,FAbnMJ->AbIJ', H.ab.oovo[oa, Ob, Va, Ob], T.abb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('NMfI,fAbNMJ->AbIJ', H.ab.oovo[Oa, Ob, va, Ob], T.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('NMFI,FAbNMJ->AbIJ', H.ab.oovo[Oa, Ob, Va, Ob], T.abb.VVvOOO, optimize=True)
    )
    dT.bb[Vb, vb, Ob, Ob] -= np.transpose(dT.bb[Vb, vb, Ob, Ob], (0, 1, 3, 2))
    dT.bb[vb, Vb, Ob, Ob] = -1.0 * np.transpose(dT.bb[Vb, vb, Ob, Ob], (1, 0, 2, 3))
    return dT


def build_1100(T, dT, H, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dT.bb[Vb, Vb, ob, ob] = (1.0 / 4.0) * (
            -1.0 * np.einsum('Me,eBAMij->ABij', H.a.ov[Oa, va], T.abb.vVVOoo, optimize=True)
            - 1.0 * np.einsum('ME,EBAMij->ABij', H.a.ov[Oa, Va], T.abb.VVVOoo, optimize=True)
    )
    dT.bb[Vb, Vb, ob, ob] += (1.0 / 4.0) * (
            -1.0 * np.einsum('Me,BAeijM->ABij', H.b.ov[Ob, vb], T.bbb.VVvooO, optimize=True)
            - 1.0 * np.einsum('ME,EBAijM->ABij', H.b.ov[Ob, Vb], T.bbb.VVVooO, optimize=True)
    )
    dT.bb[Vb, Vb, ob, ob] += (2.0 / 4.0) * (
            +0.5 * np.einsum('ANef,BfeijN->ABij', H.bb.vovv[Vb, Ob, vb, vb], T.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('ANEf,BEfijN->ABij', H.bb.vovv[Vb, Ob, Vb, vb], T.bbb.VVvooO, optimize=True)
            - 0.5 * np.einsum('ANEF,FBEijN->ABij', H.bb.vovv[Vb, Ob, Vb, Vb], T.bbb.VVVooO, optimize=True)
    )
    dT.bb[Vb, Vb, ob, ob] += (2.0 / 4.0) * (
            -1.0 * np.einsum('NAfe,fBeNij->ABij', H.ab.ovvv[Oa, Vb, va, vb], T.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('NAFe,FBeNij->ABij', H.ab.ovvv[Oa, Vb, Va, vb], T.abb.VVvOoo, optimize=True)
            - 1.0 * np.einsum('NAfE,fBENij->ABij', H.ab.ovvv[Oa, Vb, va, Vb], T.abb.vVVOoo, optimize=True)
            - 1.0 * np.einsum('NAFE,FBENij->ABij', H.ab.ovvv[Oa, Vb, Va, Vb], T.abb.VVVOoo, optimize=True)
    )
    dT.bb[Vb, Vb, ob, ob] += (2.0 / 4.0) * (
            +1.0 * np.einsum('mNif,BAfmjN->ABij', H.bb.ooov[ob, Ob, ob, vb], T.bbb.VVvooO, optimize=True)
            - 0.5 * np.einsum('MNif,BAfjMN->ABij', H.bb.ooov[Ob, Ob, ob, vb], T.bbb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('mNiF,FBAmjN->ABij', H.bb.ooov[ob, Ob, ob, Vb], T.bbb.VVVooO, optimize=True)
            - 0.5 * np.einsum('MNiF,FBAjMN->ABij', H.bb.ooov[Ob, Ob, ob, Vb], T.bbb.VVVoOO, optimize=True)
    )
    dT.bb[Vb, Vb, ob, ob] += (2.0 / 4.0) * (
            -1.0 * np.einsum('nMfi,fBAnjM->ABij', H.ab.oovo[oa, Ob, va, ob], T.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('nMFi,FBAnjM->ABij', H.ab.oovo[oa, Ob, Va, ob], T.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('Nmfi,fBANmj->ABij', H.ab.oovo[Oa, ob, va, ob], T.abb.vVVOoo, optimize=True)
            - 1.0 * np.einsum('NMfi,fBANjM->ABij', H.ab.oovo[Oa, Ob, va, ob], T.abb.vVVOoO, optimize=True)
            + 1.0 * np.einsum('NmFi,FBANmj->ABij', H.ab.oovo[Oa, ob, Va, ob], T.abb.VVVOoo, optimize=True)
            - 1.0 * np.einsum('NMFi,FBANjM->ABij', H.ab.oovo[Oa, Ob, Va, ob], T.abb.VVVOoO, optimize=True)
    )

    dT.bb[Vb, Vb, ob, ob] -= np.transpose(dT.bb[Vb, Vb, ob, ob], (1, 0, 2, 3))
    dT.bb[Vb, Vb, ob, ob] -= np.transpose(dT.bb[Vb, Vb, ob, ob], (0, 1, 3, 2))

    return dT


def build_0011(T, dT, H, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dT.bb[vb, vb, Ob, Ob] = (1.0 / 4.0) * (
            -1.0 * np.einsum('mE,EbamIJ->abIJ', H.a.ov[oa, Va], T.abb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('ME,EbaMIJ->abIJ', H.a.ov[Oa, Va], T.abb.VvvOOO, optimize=True)
    )
    dT.bb[vb, vb, Ob, Ob] += (1.0 / 4.0) * (
            -1.0 * np.einsum('mE,EbamIJ->abIJ', H.b.ov[ob, Vb], T.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('ME,EbaIJM->abIJ', H.b.ov[Ob, Vb], T.bbb.VvvOOO, optimize=True)
    )
    dT.bb[vb, vb, Ob, Ob] += (2.0 / 4.0) * (
            -1.0 * np.einsum('aneF,FbenIJ->abIJ', H.bb.vovv[vb, ob, vb, Vb], T.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('aNeF,FbeIJN->abIJ', H.bb.vovv[vb, Ob, vb, Vb], T.bbb.VvvOOO, optimize=True)
            + 0.5 * np.einsum('anEF,FEbnIJ->abIJ', H.bb.vovv[vb, ob, Vb, Vb], T.bbb.VVvoOO, optimize=True)
            + 0.5 * np.einsum('aNEF,FEbIJN->abIJ', H.bb.vovv[vb, Ob, Vb, Vb], T.bbb.VVvOOO, optimize=True)
    )
    dT.bb[vb, vb, Ob, Ob] += (2.0 / 4.0) * (
            +1.0 * np.einsum('nafE,fEbnIJ->abIJ', H.ab.ovvv[oa, vb, va, Vb], T.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('NafE,fEbNIJ->abIJ', H.ab.ovvv[Oa, vb, va, Vb], T.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('naFe,FbenIJ->abIJ', H.ab.ovvv[oa, vb, Va, vb], T.abb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('NaFe,FbeNIJ->abIJ', H.ab.ovvv[Oa, vb, Va, vb], T.abb.VvvOOO, optimize=True)
            + 1.0 * np.einsum('naFE,FEbnIJ->abIJ', H.ab.ovvv[oa, vb, Va, Vb], T.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('NaFE,FEbNIJ->abIJ', H.ab.ovvv[Oa, vb, Va, Vb], T.abb.VVvOOO, optimize=True)
    )
    dT.bb[vb, vb, Ob, Ob] += (2.0 / 4.0) * (
            -0.5 * np.einsum('mnIF,FbamnJ->abIJ', H.bb.ooov[ob, ob, Ob, Vb], T.bbb.VvvooO, optimize=True)
            + 1.0 * np.einsum('MnIF,FbanMJ->abIJ', H.bb.ooov[Ob, ob, Ob, Vb], T.bbb.VvvoOO, optimize=True)
            + 0.5 * np.einsum('MNIF,FbaMJN->abIJ', H.bb.ooov[Ob, Ob, Ob, Vb], T.bbb.VvvOOO, optimize=True)
    )
    dT.bb[vb, vb, Ob, Ob] += (2.0 / 4.0) * (
            +1.0 * np.einsum('nmFI,FbanmJ->abIJ', H.ab.oovo[oa, ob, Va, Ob], T.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('NmFI,FbaNmJ->abIJ', H.ab.oovo[Oa, ob, Va, Ob], T.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('nMFI,FbanMJ->abIJ', H.ab.oovo[oa, Ob, Va, Ob], T.abb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('NMFI,FbaNMJ->abIJ', H.ab.oovo[Oa, Ob, Va, Ob], T.abb.VvvOOO, optimize=True)
    )

    dT.bb[vb, vb, Ob, Ob] -= np.transpose(dT.bb[vb, vb, Ob, Ob], (1, 0, 2, 3))
    dT.bb[vb, vb, Ob, Ob] -= np.transpose(dT.bb[vb, vb, Ob, Ob], (0, 1, 3, 2))

    return dT


def build_1001(T, dT, H, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dT.bb[Vb, vb, ob, Ob] = (1.0 / 1.0) * (
            +1.0 * np.einsum('me,eAbmiJ->AbiJ', H.a.ov[oa, va], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('Me,eAbMiJ->AbiJ', H.a.ov[Oa, va], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('mE,EAbmiJ->AbiJ', H.a.ov[oa, Va], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('ME,EAbMiJ->AbiJ', H.a.ov[Oa, Va], T.abb.VVvOoO, optimize=True)
    )
    dT.bb[Vb, vb, ob, Ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('me,AebimJ->AbiJ', H.b.ov[ob, vb], T.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('Me,AebiJM->AbiJ', H.b.ov[Ob, vb], T.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('mE,EAbimJ->AbiJ', H.b.ov[ob, Vb], T.bbb.VVvooO, optimize=True)
            + 1.0 * np.einsum('ME,EAbiJM->AbiJ', H.b.ov[Ob, Vb], T.bbb.VVvoOO, optimize=True)
    )
    dT.bb[Vb, vb, ob, Ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('AnEf,EfbinJ->AbiJ', H.bb.vovv[Vb, ob, Vb, vb], T.bbb.VvvooO, optimize=True)
            - 0.5 * np.einsum('AnEF,FEbinJ->AbiJ', H.bb.vovv[Vb, ob, Vb, Vb], T.bbb.VVvooO, optimize=True)
            - 1.0 * np.einsum('ANEf,EfbiJN->AbiJ', H.bb.vovv[Vb, Ob, Vb, vb], T.bbb.VvvoOO, optimize=True)
            + 0.5 * np.einsum('ANEF,FEbiJN->AbiJ', H.bb.vovv[Vb, Ob, Vb, Vb], T.bbb.VVvoOO, optimize=True)
    )
    dT.bb[Vb, vb, ob, Ob] += (1.0 / 1.0) * (
            +0.5 * np.einsum('bnef,AfeinJ->AbiJ', H.bb.vovv[vb, ob, vb, vb], T.bbb.VvvooO, optimize=True)
            - 0.5 * np.einsum('bNef,AfeiJN->AbiJ', H.bb.vovv[vb, Ob, vb, vb], T.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('bnEf,AEfinJ->AbiJ', H.bb.vovv[vb, ob, Vb, vb], T.bbb.VVvooO, optimize=True)
            - 0.5 * np.einsum('bnEF,FAEinJ->AbiJ', H.bb.vovv[vb, ob, Vb, Vb], T.bbb.VVVooO, optimize=True)
            + 1.0 * np.einsum('bNEf,AEfiJN->AbiJ', H.bb.vovv[vb, Ob, Vb, vb], T.bbb.VVvoOO, optimize=True)
            + 0.5 * np.einsum('bNEF,FAEiJN->AbiJ', H.bb.vovv[vb, Ob, Vb, Vb], T.bbb.VVVoOO, optimize=True)
    )
    dT.bb[Vb, vb, ob, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('nAFe,FbeniJ->AbiJ', H.ab.ovvv[oa, Vb, Va, vb], T.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('NAFe,FbeNiJ->AbiJ', H.ab.ovvv[Oa, Vb, Va, vb], T.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('nAfE,fEbniJ->AbiJ', H.ab.ovvv[oa, Vb, va, Vb], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('nAFE,FEbniJ->AbiJ', H.ab.ovvv[oa, Vb, Va, Vb], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('NAfE,fEbNiJ->AbiJ', H.ab.ovvv[Oa, Vb, va, Vb], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('NAFE,FEbNiJ->AbiJ', H.ab.ovvv[Oa, Vb, Va, Vb], T.abb.VVvOoO, optimize=True)
    )
    dT.bb[Vb, vb, ob, Ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('nbfe,fAeniJ->AbiJ', H.ab.ovvv[oa, vb, va, vb], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('nbFe,FAeniJ->AbiJ', H.ab.ovvv[oa, vb, Va, vb], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('Nbfe,fAeNiJ->AbiJ', H.ab.ovvv[Oa, vb, va, vb], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('NbFe,FAeNiJ->AbiJ', H.ab.ovvv[Oa, vb, Va, vb], T.abb.VVvOoO, optimize=True)
            + 1.0 * np.einsum('nbfE,fAEniJ->AbiJ', H.ab.ovvv[oa, vb, va, Vb], T.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('nbFE,FAEniJ->AbiJ', H.ab.ovvv[oa, vb, Va, Vb], T.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('NbfE,fAENiJ->AbiJ', H.ab.ovvv[Oa, vb, va, Vb], T.abb.vVVOoO, optimize=True)
            + 1.0 * np.einsum('NbFE,FAENiJ->AbiJ', H.ab.ovvv[Oa, vb, Va, Vb], T.abb.VVVOoO, optimize=True)
    )
    dT.bb[Vb, vb, ob, Ob] += (1.0 / 1.0) * (
            -0.5 * np.einsum('mnif,AfbmnJ->AbiJ', H.bb.ooov[ob, ob, ob, vb], T.bbb.VvvooO, optimize=True)
            + 0.5 * np.einsum('mniF,FAbmnJ->AbiJ', H.bb.ooov[ob, ob, ob, Vb], T.bbb.VVvooO, optimize=True)
            + 1.0 * np.einsum('Mnif,AfbnMJ->AbiJ', H.bb.ooov[Ob, ob, ob, vb], T.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('MniF,FAbnMJ->AbiJ', H.bb.ooov[Ob, ob, ob, Vb], T.bbb.VVvoOO, optimize=True)
            + 0.5 * np.einsum('MNif,AfbMJN->AbiJ', H.bb.ooov[Ob, Ob, ob, vb], T.bbb.VvvOOO, optimize=True)
            - 0.5 * np.einsum('MNiF,FAbMJN->AbiJ', H.bb.ooov[Ob, Ob, ob, Vb], T.bbb.VVvOOO, optimize=True)
    )
    dT.bb[Vb, vb, ob, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('MnJf,AfbinM->AbiJ', H.bb.ooov[Ob, ob, Ob, vb], T.bbb.VvvooO, optimize=True)
            + 1.0 * np.einsum('MnJF,FAbinM->AbiJ', H.bb.ooov[Ob, ob, Ob, Vb], T.bbb.VVvooO, optimize=True)
            + 0.5 * np.einsum('MNJf,AfbiMN->AbiJ', H.bb.ooov[Ob, Ob, Ob, vb], T.bbb.VvvoOO, optimize=True)
            - 0.5 * np.einsum('MNJF,FAbiMN->AbiJ', H.bb.ooov[Ob, Ob, Ob, Vb], T.bbb.VVvoOO, optimize=True)
    )
    dT.bb[Vb, vb, ob, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('nmfi,fAbnmJ->AbiJ', H.ab.oovo[oa, ob, va, ob], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('nmFi,FAbnmJ->AbiJ', H.ab.oovo[oa, ob, Va, ob], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('Nmfi,fAbNmJ->AbiJ', H.ab.oovo[Oa, ob, va, ob], T.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('NmFi,FAbNmJ->AbiJ', H.ab.oovo[Oa, ob, Va, ob], T.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('nMfi,fAbnMJ->AbiJ', H.ab.oovo[oa, Ob, va, ob], T.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('nMFi,FAbnMJ->AbiJ', H.ab.oovo[oa, Ob, Va, ob], T.abb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('NMfi,fAbNMJ->AbiJ', H.ab.oovo[Oa, Ob, va, ob], T.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('NMFi,FAbNMJ->AbiJ', H.ab.oovo[Oa, Ob, Va, ob], T.abb.VVvOOO, optimize=True)
    )
    dT.bb[Vb, vb, ob, Ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('NmfJ,fAbNmi->AbiJ', H.ab.oovo[Oa, ob, va, Ob], T.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('NmFJ,FAbNmi->AbiJ', H.ab.oovo[Oa, ob, Va, Ob], T.abb.VVvOoo, optimize=True)
            - 1.0 * np.einsum('nMfJ,fAbniM->AbiJ', H.ab.oovo[oa, Ob, va, Ob], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('nMFJ,FAbniM->AbiJ', H.ab.oovo[oa, Ob, Va, Ob], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('NMfJ,fAbNiM->AbiJ', H.ab.oovo[Oa, Ob, va, Ob], T.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('NMFJ,FAbNiM->AbiJ', H.ab.oovo[Oa, Ob, Va, Ob], T.abb.VVvOoO, optimize=True)
    )

    dT.bb[vb, Vb, ob, Ob] = -1.0 * np.transpose(dT.bb[Vb, vb, ob, Ob], (1, 0, 2, 3))
    dT.bb[Vb, vb, Ob, ob] = -1.0 * np.transpose(dT.bb[Vb, vb, ob, Ob], (0, 1, 3, 2))
    dT.bb[vb, Vb, Ob, ob] = np.transpose(dT.bb[Vb, vb, ob, Ob], (1, 0, 3, 2))
    return dT


def build_1000(T, dT, H, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dT.bb[Vb, vb, ob, ob] = (1.0 / 2.0) * (
            +1.0 * np.einsum('Me,eAbMij->Abij', H.a.ov[Oa, va], T.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('ME,EAbMij->Abij', H.a.ov[Oa, Va], T.abb.VVvOoo, optimize=True)
    )
    dT.bb[Vb, vb, ob, ob] += (1.0 / 2.0) * (
            -1.0 * np.einsum('Me,AebijM->Abij', H.b.ov[Ob, vb], T.bbb.VvvooO, optimize=True)
            + 1.0 * np.einsum('ME,EAbijM->Abij', H.b.ov[Ob, Vb], T.bbb.VVvooO, optimize=True)
    )
    dT.bb[Vb, vb, ob, ob] += (1.0 / 2.0) * (
            -1.0 * np.einsum('ANeF,FbeijN->Abij', H.bb.vovv[Vb, Ob, vb, Vb], T.bbb.VvvooO, optimize=True)
            + 0.5 * np.einsum('ANEF,FEbijN->Abij', H.bb.vovv[Vb, Ob, Vb, Vb], T.bbb.VVvooO, optimize=True)
    )
    dT.bb[Vb, vb, ob, ob] += (1.0 / 2.0) * (
            -0.5 * np.einsum('bNef,AfeijN->Abij', H.bb.vovv[vb, Ob, vb, vb], T.bbb.VvvooO, optimize=True)
            + 1.0 * np.einsum('bNeF,FAeijN->Abij', H.bb.vovv[vb, Ob, vb, Vb], T.bbb.VVvooO, optimize=True)
            + 0.5 * np.einsum('bNEF,FAEijN->Abij', H.bb.vovv[vb, Ob, Vb, Vb], T.bbb.VVVooO, optimize=True)
    )
    dT.bb[Vb, vb, ob, ob] += (1.0 / 2.0) * (
            +1.0 * np.einsum('NAfE,fEbNij->Abij', H.ab.ovvv[Oa, Vb, va, Vb], T.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('NAFe,FbeNij->Abij', H.ab.ovvv[Oa, Vb, Va, vb], T.abb.VvvOoo, optimize=True)
            + 1.0 * np.einsum('NAFE,FEbNij->Abij', H.ab.ovvv[Oa, Vb, Va, Vb], T.abb.VVvOoo, optimize=True)
    )
    dT.bb[Vb, vb, ob, ob] += (1.0 / 2.0) * (
            +1.0 * np.einsum('Nbfe,fAeNij->Abij', H.ab.ovvv[Oa, vb, va, vb], T.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('NbfE,fAENij->Abij', H.ab.ovvv[Oa, vb, va, Vb], T.abb.vVVOoo, optimize=True)
            + 1.0 * np.einsum('NbFe,FAeNij->Abij', H.ab.ovvv[Oa, vb, Va, vb], T.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('NbFE,FAENij->Abij', H.ab.ovvv[Oa, vb, Va, Vb], T.abb.VVVOoo, optimize=True)
    )
    dT.bb[Vb, vb, ob, ob] += (2.0 / 2.0) * (
            +1.0 * np.einsum('Mnif,AfbjnM->Abij', H.bb.ooov[Ob, ob, ob, vb], T.bbb.VvvooO, optimize=True)
            - 0.5 * np.einsum('MNif,AfbjMN->Abij', H.bb.ooov[Ob, Ob, ob, vb], T.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('MniF,FAbjnM->Abij', H.bb.ooov[Ob, ob, ob, Vb], T.bbb.VVvooO, optimize=True)
            + 0.5 * np.einsum('MNiF,FAbjMN->Abij', H.bb.ooov[Ob, Ob, ob, Vb], T.bbb.VVvoOO, optimize=True)
    )
    dT.bb[Vb, vb, ob, ob] += (2.0 / 2.0) * (
            +1.0 * np.einsum('nMfi,fAbnjM->Abij', H.ab.oovo[oa, Ob, va, ob], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('Nmfi,fAbNmj->Abij', H.ab.oovo[Oa, ob, va, ob], T.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('NMfi,fAbNjM->Abij', H.ab.oovo[Oa, Ob, va, ob], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('nMFi,FAbnjM->Abij', H.ab.oovo[oa, Ob, Va, ob], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('NmFi,FAbNmj->Abij', H.ab.oovo[Oa, ob, Va, ob], T.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('NMFi,FAbNjM->Abij', H.ab.oovo[Oa, Ob, Va, ob], T.abb.VVvOoO, optimize=True)
    )
    dT.bb[Vb, vb, ob, ob] -= np.transpose(dT.bb[Vb, vb, ob, ob], (0, 1, 3, 2))
    dT.bb[vb, Vb, ob, ob] = -1.0 * np.transpose(dT.bb[Vb, vb, ob, ob], (1, 0, 2, 3))
    return dT


def build_0001(T, dT, H, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dT.bb[vb, vb, ob, Ob] = (1.0 / 2.0) * (
            -1.0 * np.einsum('mE,EbamiJ->abiJ', H.a.ov[oa, Va], T.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('ME,EbaMiJ->abiJ', H.a.ov[Oa, Va], T.abb.VvvOoO, optimize=True)
    )
    dT.bb[vb, vb, ob, Ob] += (1.0 / 2.0) * (
            +1.0 * np.einsum('mE,EbaimJ->abiJ', H.b.ov[ob, Vb], T.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('ME,EbaiJM->abiJ', H.b.ov[Ob, Vb], T.bbb.VvvoOO, optimize=True)
    )
    dT.bb[vb, vb, ob, Ob] += (2.0 / 2.0) * (
            +1.0 * np.einsum('anEf,EfbinJ->abiJ', H.bb.vovv[vb, ob, Vb, vb], T.bbb.VvvooO, optimize=True)
            - 0.5 * np.einsum('anEF,FEbinJ->abiJ', H.bb.vovv[vb, ob, Vb, Vb], T.bbb.VVvooO, optimize=True)
            - 1.0 * np.einsum('aNEf,EfbiJN->abiJ', H.bb.vovv[vb, Ob, Vb, vb], T.bbb.VvvoOO, optimize=True)
            + 0.5 * np.einsum('aNEF,FEbiJN->abiJ', H.bb.vovv[vb, Ob, Vb, Vb], T.bbb.VVvoOO, optimize=True)
    )
    dT.bb[vb, vb, ob, Ob] += (2.0 / 2.0) * (
            -1.0 * np.einsum('naFe,FbeniJ->abiJ', H.ab.ovvv[oa, vb, Va, vb], T.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('nafE,fEbniJ->abiJ', H.ab.ovvv[oa, vb, va, Vb], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('naFE,FEbniJ->abiJ', H.ab.ovvv[oa, vb, Va, Vb], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('NaFe,FbeNiJ->abiJ', H.ab.ovvv[Oa, vb, Va, vb], T.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('NafE,fEbNiJ->abiJ', H.ab.ovvv[Oa, vb, va, Vb], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('NaFE,FEbNiJ->abiJ', H.ab.ovvv[Oa, vb, Va, Vb], T.abb.VVvOoO, optimize=True)
    )
    dT.bb[vb, vb, ob, Ob] += (1.0 / 2.0) * (
            -0.5 * np.einsum('mniF,FbamnJ->abiJ', H.bb.ooov[ob, ob, ob, Vb], T.bbb.VvvooO, optimize=True)
            + 1.0 * np.einsum('mNiF,FbamJN->abiJ', H.bb.ooov[ob, Ob, ob, Vb], T.bbb.VvvoOO, optimize=True)
            + 0.5 * np.einsum('MNiF,FbaMJN->abiJ', H.bb.ooov[Ob, Ob, ob, Vb], T.bbb.VvvOOO, optimize=True)
    )
    dT.bb[vb, vb, ob, Ob] += (1.0 / 2.0) * (
            -1.0 * np.einsum('mNJF,FbamiN->abiJ', H.bb.ooov[ob, Ob, Ob, Vb], T.bbb.VvvooO, optimize=True)
            + 0.5 * np.einsum('MNJF,FbaiMN->abiJ', H.bb.ooov[Ob, Ob, Ob, Vb], T.bbb.VvvoOO, optimize=True)
    )
    dT.bb[vb, vb, ob, Ob] += (1.0 / 2.0) * (
            +1.0 * np.einsum('nmFi,FbanmJ->abiJ', H.ab.oovo[oa, ob, Va, ob], T.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('nMFi,FbanMJ->abiJ', H.ab.oovo[oa, Ob, Va, ob], T.abb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('NmFi,FbaNmJ->abiJ', H.ab.oovo[Oa, ob, Va, ob], T.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('NMFi,FbaNMJ->abiJ', H.ab.oovo[Oa, Ob, Va, ob], T.abb.VvvOOO, optimize=True)
    )
    dT.bb[vb, vb, ob, Ob] += (1.0 / 2.0) * (
            +1.0 * np.einsum('nMFJ,FbaniM->abiJ', H.ab.oovo[oa, Ob, Va, Ob], T.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('NmFJ,FbaNmi->abiJ', H.ab.oovo[Oa, ob, Va, Ob], T.abb.VvvOoo, optimize=True)
            + 1.0 * np.einsum('NMFJ,FbaNiM->abiJ', H.ab.oovo[Oa, Ob, Va, Ob], T.abb.VvvOoO, optimize=True)
    )

    dT.bb[vb, vb, ob, Ob] -= np.transpose(dT.bb[vb, vb, ob, Ob], (1, 0, 2, 3))
    dT.bb[vb, vb, Ob, ob] = -1.0 * np.transpose(dT.bb[vb, vb, ob, Ob], (0, 1, 3, 2))
    return dT


def build_0000(T, dT, H, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dT.bb[vb, vb, ob, ob] = (1.0 / 4.0) * (
            -1.0 * np.einsum('ME,EbaMij->abij', H.a.ov[Oa, Va], T.abb.VvvOoo, optimize=True)
    )
    dT.bb[vb, vb, ob, ob] += (1.0 / 4.0) * (
            -1.0 * np.einsum('ME,EbaijM->abij', H.b.ov[Ob, Vb], T.bbb.VvvooO, optimize=True)
    )
    dT.bb[vb, vb, ob, ob] += (2.0 / 4.0) * (
            -1.0 * np.einsum('aNeF,FbeijN->abij', H.bb.vovv[vb, Ob, vb, Vb], T.bbb.VvvooO, optimize=True)
            + 0.5 * np.einsum('aNEF,FEbijN->abij', H.bb.vovv[vb, Ob, Vb, Vb], T.bbb.VVvooO, optimize=True)
    )
    dT.bb[vb, vb, ob, ob] += (2.0 / 4.0) * (
            +1.0 * np.einsum('NafE,fEbNij->abij', H.ab.ovvv[Oa, vb, va, Vb], T.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('NaFe,FbeNij->abij', H.ab.ovvv[Oa, vb, Va, vb], T.abb.VvvOoo, optimize=True)
            + 1.0 * np.einsum('NaFE,FEbNij->abij', H.ab.ovvv[Oa, vb, Va, Vb], T.abb.VVvOoo, optimize=True)
    )
    dT.bb[vb, vb, ob, ob] += (2.0 / 4.0) * (
            +1.0 * np.einsum('MniF,FbajnM->abij', H.bb.ooov[Ob, ob, ob, Vb], T.bbb.VvvooO, optimize=True)
            - 0.5 * np.einsum('MNiF,FbajMN->abij', H.bb.ooov[Ob, Ob, ob, Vb], T.bbb.VvvoOO, optimize=True)
    )
    dT.bb[vb, vb, ob, ob] += (2.0 / 4.0) * (
            +1.0 * np.einsum('NmFi,FbaNmj->abij', H.ab.oovo[Oa, ob, Va, ob], T.abb.VvvOoo, optimize=True)
            - 1.0 * np.einsum('nMFi,FbanjM->abij', H.ab.oovo[oa, Ob, Va, ob], T.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('NMFi,FbaNjM->abij', H.ab.oovo[Oa, Ob, Va, ob], T.abb.VvvOoO, optimize=True)
    )

    dT.bb[vb, vb, ob, ob] -= np.transpose(dT.bb[vb, vb, ob, ob], (1, 0, 2, 3))
    dT.bb[vb, vb, ob, ob] -= np.transpose(dT.bb[vb, vb, ob, ob], (0, 1, 3, 2))

    return dT

def update(T, dT, H, shift):

    T.bb, dT.bb = cc_active_loops.update_t2c(
        T.bb,
        dT.bb,
        H.b.oo,
        H.b.vv,
        shift,
    )

    return T, dT