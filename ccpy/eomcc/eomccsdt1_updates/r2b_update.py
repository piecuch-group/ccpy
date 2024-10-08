import numpy as np

from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops


def update(R, omega, H):

    R.ab = eomcc_active_loops.update_r2b(R.ab, omega, H.a.oo, H.a.vv, H.b.oo, H.b.vv, 0.0)

    return R


def build(dR, R, T, H, X, system):
    x2 = build_eomccsd(R, T, H)  # base EOMCCSD part (separately antisymmetrized)

    # Add on T3 parts
    dR = build_1111(dR, R, T, H, X, system)
    dR = build_1101(dR, R, T, H, X, system)
    dR = build_1110(dR, R, T, H, X, system)
    dR = build_1011(dR, R, T, H, X, system)
    dR = build_0111(dR, R, T, H, X, system)
    dR = build_1100(dR, R, T, H, X, system)
    dR = build_0011(dR, R, T, H, X, system)
    dR = build_1001(dR, R, T, H, X, system)
    dR = build_0101(dR, R, T, H, X, system)
    dR = build_1010(dR, R, T, H, X, system)
    dR = build_0110(dR, R, T, H, X, system)
    dR = build_1000(dR, R, T, H, X, system)
    dR = build_0100(dR, R, T, H, X, system)
    dR = build_0001(dR, R, T, H, X, system)
    dR = build_0010(dR, R, T, H, X, system)
    dR = build_0000(dR, R, T, H, X, system)

    dR.ab += x2

    return dR


def build_eomccsd(R, T, H):
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


def build_1111(dR, R, T, H, X, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    dR.ab[Va, Vb, Oa, Ob] = (1.0 / 1.0) * (
            -1.0 * np.einsum('me,AeBmIJ->ABIJ', X.a.ov[oa, va], T.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('mE,EABmIJ->ABIJ', X.a.ov[oa, Va], T.aab.VVVoOO, optimize=True)
            + 1.0 * np.einsum('Me,AeBIMJ->ABIJ', X.a.ov[Oa, va], T.aab.VvVOOO, optimize=True)
            - 1.0 * np.einsum('ME,EABIMJ->ABIJ', X.a.ov[Oa, Va], T.aab.VVVOOO, optimize=True)
    )
    dR.ab[Va, Vb, Oa, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('me,ABeImJ->ABIJ', X.b.ov[ob, vb], T.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('mE,ABEImJ->ABIJ', X.b.ov[ob, Vb], T.abb.VVVOoO, optimize=True)
            - 1.0 * np.einsum('Me,ABeIMJ->ABIJ', X.b.ov[Ob, vb], T.abb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('ME,ABEIMJ->ABIJ', X.b.ov[Ob, Vb], T.abb.VVVOOO, optimize=True)
    )
    dR.ab[Va, Vb, Oa, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('me,AeBmIJ->ABIJ', H.a.ov[oa, va], R.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('mE,EABmIJ->ABIJ', H.a.ov[oa, Va], R.aab.VVVoOO, optimize=True)
            + 1.0 * np.einsum('Me,AeBIMJ->ABIJ', H.a.ov[Oa, va], R.aab.VvVOOO, optimize=True)
            - 1.0 * np.einsum('ME,EABIMJ->ABIJ', H.a.ov[Oa, Va], R.aab.VVVOOO, optimize=True)
    )
    dR.ab[Va, Vb, Oa, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('me,ABeImJ->ABIJ', H.b.ov[ob, vb], R.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('mE,ABEImJ->ABIJ', H.b.ov[ob, Vb], R.abb.VVVOoO, optimize=True)
            - 1.0 * np.einsum('Me,ABeIMJ->ABIJ', H.b.ov[Ob, vb], R.abb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('ME,ABEIMJ->ABIJ', H.b.ov[Ob, Vb], R.abb.VVVOOO, optimize=True)
    )
    dR.ab[Va, Vb, Oa, Ob] += (1.0 / 1.0) * (
            -0.5 * np.einsum('mnIf,AfBmnJ->ABIJ', H.aa.ooov[oa, oa, Oa, va], R.aab.VvVooO, optimize=True)
            + 1.0 * np.einsum('MnIf,AfBnMJ->ABIJ', H.aa.ooov[Oa, oa, Oa, va], R.aab.VvVoOO, optimize=True)
            - 0.5 * np.einsum('MNIf,AfBMNJ->ABIJ', H.aa.ooov[Oa, Oa, Oa, va], R.aab.VvVOOO, optimize=True)
            + 0.5 * np.einsum('mnIF,FABmnJ->ABIJ', H.aa.ooov[oa, oa, Oa, Va], R.aab.VVVooO, optimize=True)
            - 1.0 * np.einsum('MnIF,FABnMJ->ABIJ', H.aa.ooov[Oa, oa, Oa, Va], R.aab.VVVoOO, optimize=True)
            + 0.5 * np.einsum('MNIF,FABMNJ->ABIJ', H.aa.ooov[Oa, Oa, Oa, Va], R.aab.VVVOOO, optimize=True)
    )
    dR.ab[Va, Vb, Oa, Ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('nmfJ,AfBnIm->ABIJ', H.ab.oovo[oa, ob, va, Ob], R.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('nmFJ,FABnIm->ABIJ', H.ab.oovo[oa, ob, Va, Ob], R.aab.VVVoOo, optimize=True)
            - 1.0 * np.einsum('NmfJ,AfBINm->ABIJ', H.ab.oovo[Oa, ob, va, Ob], R.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('NmFJ,FABINm->ABIJ', H.ab.oovo[Oa, ob, Va, Ob], R.aab.VVVOOo, optimize=True)
            + 1.0 * np.einsum('nMfJ,AfBnIM->ABIJ', H.ab.oovo[oa, Ob, va, Ob], R.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('nMFJ,FABnIM->ABIJ', H.ab.oovo[oa, Ob, Va, Ob], R.aab.VVVoOO, optimize=True)
            - 1.0 * np.einsum('NMfJ,AfBINM->ABIJ', H.ab.oovo[Oa, Ob, va, Ob], R.aab.VvVOOO, optimize=True)
            + 1.0 * np.einsum('NMFJ,FABINM->ABIJ', H.ab.oovo[Oa, Ob, Va, Ob], R.aab.VVVOOO, optimize=True)
    )
    dR.ab[Va, Vb, Oa, Ob] += (1.0 / 1.0) * (
            +0.5 * np.einsum('mnJf,ABfInm->ABIJ', H.bb.ooov[ob, ob, Ob, vb], R.abb.VVvOoo, optimize=True)
            + 0.5 * np.einsum('mnJF,ABFInm->ABIJ', H.bb.ooov[ob, ob, Ob, Vb], R.abb.VVVOoo, optimize=True)
            + 1.0 * np.einsum('MnJf,ABfInM->ABIJ', H.bb.ooov[Ob, ob, Ob, vb], R.abb.VVvOoO, optimize=True)
            + 1.0 * np.einsum('MnJF,ABFInM->ABIJ', H.bb.ooov[Ob, ob, Ob, Vb], R.abb.VVVOoO, optimize=True)
            + 0.5 * np.einsum('MNJf,ABfINM->ABIJ', H.bb.ooov[Ob, Ob, Ob, vb], R.abb.VVvOOO, optimize=True)
            + 0.5 * np.einsum('MNJF,ABFINM->ABIJ', H.bb.ooov[Ob, Ob, Ob, Vb], R.abb.VVVOOO, optimize=True)
    )
    dR.ab[Va, Vb, Oa, Ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('mnIf,ABfmnJ->ABIJ', H.ab.ooov[oa, ob, Oa, vb], R.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('mNIf,ABfmNJ->ABIJ', H.ab.ooov[oa, Ob, Oa, vb], R.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('MnIf,ABfMnJ->ABIJ', H.ab.ooov[Oa, ob, Oa, vb], R.abb.VVvOoO, optimize=True)
            + 1.0 * np.einsum('MNIf,ABfMNJ->ABIJ', H.ab.ooov[Oa, Ob, Oa, vb], R.abb.VVvOOO, optimize=True)
            + 1.0 * np.einsum('mnIF,ABFmnJ->ABIJ', H.ab.ooov[oa, ob, Oa, Vb], R.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('mNIF,ABFmNJ->ABIJ', H.ab.ooov[oa, Ob, Oa, Vb], R.abb.VVVoOO, optimize=True)
            + 1.0 * np.einsum('MnIF,ABFMnJ->ABIJ', H.ab.ooov[Oa, ob, Oa, Vb], R.abb.VVVOoO, optimize=True)
            + 1.0 * np.einsum('MNIF,ABFMNJ->ABIJ', H.ab.ooov[Oa, Ob, Oa, Vb], R.abb.VVVOOO, optimize=True)
    )
    dR.ab[Va, Vb, Oa, Ob] += (1.0 / 1.0) * (
            +0.5 * np.einsum('Anef,feBnIJ->ABIJ', H.aa.vovv[Va, oa, va, va], R.aab.vvVoOO, optimize=True)
            - 0.5 * np.einsum('ANef,feBINJ->ABIJ', H.aa.vovv[Va, Oa, va, va], R.aab.vvVOOO, optimize=True)
            + 1.0 * np.einsum('AneF,FeBnIJ->ABIJ', H.aa.vovv[Va, oa, va, Va], R.aab.VvVoOO, optimize=True)
            + 0.5 * np.einsum('AnEF,FEBnIJ->ABIJ', H.aa.vovv[Va, oa, Va, Va], R.aab.VVVoOO, optimize=True)
            - 1.0 * np.einsum('ANeF,FeBINJ->ABIJ', H.aa.vovv[Va, Oa, va, Va], R.aab.VvVOOO, optimize=True)
            - 0.5 * np.einsum('ANEF,FEBINJ->ABIJ', H.aa.vovv[Va, Oa, Va, Va], R.aab.VVVOOO, optimize=True)
    )
    dR.ab[Va, Vb, Oa, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('Anef,eBfInJ->ABIJ', H.ab.vovv[Va, ob, va, vb], R.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('AnEf,EBfInJ->ABIJ', H.ab.vovv[Va, ob, Va, vb], R.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('ANef,eBfINJ->ABIJ', H.ab.vovv[Va, Ob, va, vb], R.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('ANEf,EBfINJ->ABIJ', H.ab.vovv[Va, Ob, Va, vb], R.abb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('AneF,eBFInJ->ABIJ', H.ab.vovv[Va, ob, va, Vb], R.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('AnEF,EBFInJ->ABIJ', H.ab.vovv[Va, ob, Va, Vb], R.abb.VVVOoO, optimize=True)
            - 1.0 * np.einsum('ANeF,eBFINJ->ABIJ', H.ab.vovv[Va, Ob, va, Vb], R.abb.vVVOOO, optimize=True)
            - 1.0 * np.einsum('ANEF,EBFINJ->ABIJ', H.ab.vovv[Va, Ob, Va, Vb], R.abb.VVVOOO, optimize=True)
    )
    dR.ab[Va, Vb, Oa, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('nBfe,AfenIJ->ABIJ', H.ab.ovvv[oa, Vb, va, vb], R.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('nBfE,AfEnIJ->ABIJ', H.ab.ovvv[oa, Vb, va, Vb], R.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('NBfe,AfeINJ->ABIJ', H.ab.ovvv[Oa, Vb, va, vb], R.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('NBfE,AfEINJ->ABIJ', H.ab.ovvv[Oa, Vb, va, Vb], R.aab.VvVOOO, optimize=True)
            + 1.0 * np.einsum('nBFe,FAenIJ->ABIJ', H.ab.ovvv[oa, Vb, Va, vb], R.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('nBFE,FAEnIJ->ABIJ', H.ab.ovvv[oa, Vb, Va, Vb], R.aab.VVVoOO, optimize=True)
            - 1.0 * np.einsum('NBFe,FAeINJ->ABIJ', H.ab.ovvv[Oa, Vb, Va, vb], R.aab.VVvOOO, optimize=True)
            - 1.0 * np.einsum('NBFE,FAEINJ->ABIJ', H.ab.ovvv[Oa, Vb, Va, Vb], R.aab.VVVOOO, optimize=True)
    )
    dR.ab[Va, Vb, Oa, Ob] += (1.0 / 1.0) * (
            -0.5 * np.einsum('Bnef,AefInJ->ABIJ', H.bb.vovv[Vb, ob, vb, vb], R.abb.VvvOoO, optimize=True)
            - 0.5 * np.einsum('BNef,AefINJ->ABIJ', H.bb.vovv[Vb, Ob, vb, vb], R.abb.VvvOOO, optimize=True)
            + 1.0 * np.einsum('BneF,AFeInJ->ABIJ', H.bb.vovv[Vb, ob, vb, Vb], R.abb.VVvOoO, optimize=True)
            - 0.5 * np.einsum('BnEF,AEFInJ->ABIJ', H.bb.vovv[Vb, ob, Vb, Vb], R.abb.VVVOoO, optimize=True)
            + 1.0 * np.einsum('BNeF,AFeINJ->ABIJ', H.bb.vovv[Vb, Ob, vb, Vb], R.abb.VVvOOO, optimize=True)
            - 0.5 * np.einsum('BNEF,AEFINJ->ABIJ', H.bb.vovv[Vb, Ob, Vb, Vb], R.abb.VVVOOO, optimize=True)
    )
    # of terms =  12

    return dR

def build_1011(dR, R, T, H, X, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    dR.ab[Va, vb, Oa, Ob] = (1.0 / 1.0) * (
            -1.0 * np.einsum('me,AebmIJ->AbIJ', X.a.ov[oa, va], T.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('mE,EAbmIJ->AbIJ', X.a.ov[oa, Va], T.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('Me,AebIMJ->AbIJ', X.a.ov[Oa, va], T.aab.VvvOOO, optimize=True)
            - 1.0 * np.einsum('ME,EAbIMJ->AbIJ', X.a.ov[Oa, Va], T.aab.VVvOOO, optimize=True)
    )
    dR.ab[Va, vb, Oa, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('me,AbeImJ->AbIJ', X.b.ov[ob, vb], T.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('mE,AEbImJ->AbIJ', X.b.ov[ob, Vb], T.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('Me,AbeIMJ->AbIJ', X.b.ov[Ob, vb], T.abb.VvvOOO, optimize=True)
            + 1.0 * np.einsum('ME,AEbIMJ->AbIJ', X.b.ov[Ob, Vb], T.abb.VVvOOO, optimize=True)
    )
    dR.ab[Va, vb, Oa, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('me,AebmIJ->AbIJ', H.a.ov[oa, va], R.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('mE,EAbmIJ->AbIJ', H.a.ov[oa, Va], R.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('Me,AebIMJ->AbIJ', H.a.ov[Oa, va], R.aab.VvvOOO, optimize=True)
            - 1.0 * np.einsum('ME,EAbIMJ->AbIJ', H.a.ov[Oa, Va], R.aab.VVvOOO, optimize=True)
    )
    dR.ab[Va, vb, Oa, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('me,AbeImJ->AbIJ', H.b.ov[ob, vb], R.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('mE,AEbImJ->AbIJ', H.b.ov[ob, Vb], R.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('Me,AbeIMJ->AbIJ', H.b.ov[Ob, vb], R.abb.VvvOOO, optimize=True)
            + 1.0 * np.einsum('ME,AEbIMJ->AbIJ', H.b.ov[Ob, Vb], R.abb.VVvOOO, optimize=True)
    )
    dR.ab[Va, vb, Oa, Ob] += (1.0 / 1.0) * (
            -0.5 * np.einsum('mnIf,AfbmnJ->AbIJ', H.aa.ooov[oa, oa, Oa, va], R.aab.VvvooO, optimize=True)
            + 0.5 * np.einsum('mnIF,FAbmnJ->AbIJ', H.aa.ooov[oa, oa, Oa, Va], R.aab.VVvooO, optimize=True)
            + 1.0 * np.einsum('MnIf,AfbnMJ->AbIJ', H.aa.ooov[Oa, oa, Oa, va], R.aab.VvvoOO, optimize=True)
            - 0.5 * np.einsum('MNIf,AfbMNJ->AbIJ', H.aa.ooov[Oa, Oa, Oa, va], R.aab.VvvOOO, optimize=True)
            - 1.0 * np.einsum('MnIF,FAbnMJ->AbIJ', H.aa.ooov[Oa, oa, Oa, Va], R.aab.VVvoOO, optimize=True)
            + 0.5 * np.einsum('MNIF,FAbMNJ->AbIJ', H.aa.ooov[Oa, Oa, Oa, Va], R.aab.VVvOOO, optimize=True)
    )
    dR.ab[Va, vb, Oa, Ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('nmfJ,AfbnIm->AbIJ', H.ab.oovo[oa, ob, va, Ob], R.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('NmfJ,AfbINm->AbIJ', H.ab.oovo[Oa, ob, va, Ob], R.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('nmFJ,FAbnIm->AbIJ', H.ab.oovo[oa, ob, Va, Ob], R.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('NmFJ,FAbINm->AbIJ', H.ab.oovo[Oa, ob, Va, Ob], R.aab.VVvOOo, optimize=True)
            + 1.0 * np.einsum('nMfJ,AfbnIM->AbIJ', H.ab.oovo[oa, Ob, va, Ob], R.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('NMfJ,AfbINM->AbIJ', H.ab.oovo[Oa, Ob, va, Ob], R.aab.VvvOOO, optimize=True)
            - 1.0 * np.einsum('nMFJ,FAbnIM->AbIJ', H.ab.oovo[oa, Ob, Va, Ob], R.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('NMFJ,FAbINM->AbIJ', H.ab.oovo[Oa, Ob, Va, Ob], R.aab.VVvOOO, optimize=True)
    )
    dR.ab[Va, vb, Oa, Ob] += (1.0 / 1.0) * (
            +0.5 * np.einsum('mnJf,AbfInm->AbIJ', H.bb.ooov[ob, ob, Ob, vb], R.abb.VvvOoo, optimize=True)
            - 0.5 * np.einsum('mnJF,AFbInm->AbIJ', H.bb.ooov[ob, ob, Ob, Vb], R.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('MnJf,AbfInM->AbIJ', H.bb.ooov[Ob, ob, Ob, vb], R.abb.VvvOoO, optimize=True)
            + 0.5 * np.einsum('MNJf,AbfINM->AbIJ', H.bb.ooov[Ob, Ob, Ob, vb], R.abb.VvvOOO, optimize=True)
            - 1.0 * np.einsum('MnJF,AFbInM->AbIJ', H.bb.ooov[Ob, ob, Ob, Vb], R.abb.VVvOoO, optimize=True)
            - 0.5 * np.einsum('MNJF,AFbINM->AbIJ', H.bb.ooov[Ob, Ob, Ob, Vb], R.abb.VVvOOO, optimize=True)
    )
    dR.ab[Va, vb, Oa, Ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('mnIf,AbfmnJ->AbIJ', H.ab.ooov[oa, ob, Oa, vb], R.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('mNIf,AbfmNJ->AbIJ', H.ab.ooov[oa, Ob, Oa, vb], R.abb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('mnIF,AFbmnJ->AbIJ', H.ab.ooov[oa, ob, Oa, Vb], R.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('mNIF,AFbmNJ->AbIJ', H.ab.ooov[oa, Ob, Oa, Vb], R.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('MnIf,AbfMnJ->AbIJ', H.ab.ooov[Oa, ob, Oa, vb], R.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('MNIf,AbfMNJ->AbIJ', H.ab.ooov[Oa, Ob, Oa, vb], R.abb.VvvOOO, optimize=True)
            - 1.0 * np.einsum('MnIF,AFbMnJ->AbIJ', H.ab.ooov[Oa, ob, Oa, Vb], R.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('MNIF,AFbMNJ->AbIJ', H.ab.ooov[Oa, Ob, Oa, Vb], R.abb.VVvOOO, optimize=True)
    )
    dR.ab[Va, vb, Oa, Ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('AneF,FebnIJ->AbIJ', H.aa.vovv[Va, oa, va, Va], R.aab.VvvoOO, optimize=True)
            + 0.5 * np.einsum('AnEF,FEbnIJ->AbIJ', H.aa.vovv[Va, oa, Va, Va], R.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('ANeF,FebINJ->AbIJ', H.aa.vovv[Va, Oa, va, Va], R.aab.VvvOOO, optimize=True)
            - 0.5 * np.einsum('ANEF,FEbINJ->AbIJ', H.aa.vovv[Va, Oa, Va, Va], R.aab.VVvOOO, optimize=True)
    )
    dR.ab[Va, vb, Oa, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('AnEf,EbfInJ->AbIJ', H.ab.vovv[Va, ob, Va, vb], R.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('AneF,eFbInJ->AbIJ', H.ab.vovv[Va, ob, va, Vb], R.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('AnEF,EFbInJ->AbIJ', H.ab.vovv[Va, ob, Va, Vb], R.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('ANEf,EbfINJ->AbIJ', H.ab.vovv[Va, Ob, Va, vb], R.abb.VvvOOO, optimize=True)
            + 1.0 * np.einsum('ANeF,eFbINJ->AbIJ', H.ab.vovv[Va, Ob, va, Vb], R.abb.vVvOOO, optimize=True)
            + 1.0 * np.einsum('ANEF,EFbINJ->AbIJ', H.ab.vovv[Va, Ob, Va, Vb], R.abb.VVvOOO, optimize=True)
    )
    dR.ab[Va, vb, Oa, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('nbfe,AfenIJ->AbIJ', H.ab.ovvv[oa, vb, va, vb], R.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('nbfE,AfEnIJ->AbIJ', H.ab.ovvv[oa, vb, va, Vb], R.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('nbFe,FAenIJ->AbIJ', H.ab.ovvv[oa, vb, Va, vb], R.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('nbFE,FAEnIJ->AbIJ', H.ab.ovvv[oa, vb, Va, Vb], R.aab.VVVoOO, optimize=True)
            + 1.0 * np.einsum('Nbfe,AfeINJ->AbIJ', H.ab.ovvv[Oa, vb, va, vb], R.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('NbfE,AfEINJ->AbIJ', H.ab.ovvv[Oa, vb, va, Vb], R.aab.VvVOOO, optimize=True)
            - 1.0 * np.einsum('NbFe,FAeINJ->AbIJ', H.ab.ovvv[Oa, vb, Va, vb], R.aab.VVvOOO, optimize=True)
            - 1.0 * np.einsum('NbFE,FAEINJ->AbIJ', H.ab.ovvv[Oa, vb, Va, Vb], R.aab.VVVOOO, optimize=True)
    )
    dR.ab[Va, vb, Oa, Ob] += (1.0 / 1.0) * (
            -0.5 * np.einsum('bnef,AefInJ->AbIJ', H.bb.vovv[vb, ob, vb, vb], R.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('bneF,AFeInJ->AbIJ', H.bb.vovv[vb, ob, vb, Vb], R.abb.VVvOoO, optimize=True)
            - 0.5 * np.einsum('bnEF,AEFInJ->AbIJ', H.bb.vovv[vb, ob, Vb, Vb], R.abb.VVVOoO, optimize=True)
            - 0.5 * np.einsum('bNef,AefINJ->AbIJ', H.bb.vovv[vb, Ob, vb, vb], R.abb.VvvOOO, optimize=True)
            + 1.0 * np.einsum('bNeF,AFeINJ->AbIJ', H.bb.vovv[vb, Ob, vb, Vb], R.abb.VVvOOO, optimize=True)
            - 0.5 * np.einsum('bNEF,AEFINJ->AbIJ', H.bb.vovv[vb, Ob, Vb, Vb], R.abb.VVVOOO, optimize=True)
    )
    # of terms =  12

    return dR

def build_0111(dR, R, T, H, X, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    dR.ab[va, Vb, Oa, Ob] = (1.0 / 1.0) * (
            +1.0 * np.einsum('me,eaBmIJ->aBIJ', X.a.ov[oa, va], T.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('Me,eaBIMJ->aBIJ', X.a.ov[Oa, va], T.aab.vvVOOO, optimize=True)
            + 1.0 * np.einsum('mE,EaBmIJ->aBIJ', X.a.ov[oa, Va], T.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('ME,EaBIMJ->aBIJ', X.a.ov[Oa, Va], T.aab.VvVOOO, optimize=True)
    )
    dR.ab[va, Vb, Oa, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('me,aBeImJ->aBIJ', X.b.ov[ob, vb], T.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('Me,aBeIMJ->aBIJ', X.b.ov[Ob, vb], T.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('mE,aBEImJ->aBIJ', X.b.ov[ob, Vb], T.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('ME,aBEIMJ->aBIJ', X.b.ov[Ob, Vb], T.abb.vVVOOO, optimize=True)
    )
    dR.ab[va, Vb, Oa, Ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('me,eaBmIJ->aBIJ', H.a.ov[oa, va], R.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('Me,eaBIMJ->aBIJ', H.a.ov[Oa, va], R.aab.vvVOOO, optimize=True)
            + 1.0 * np.einsum('mE,EaBmIJ->aBIJ', H.a.ov[oa, Va], R.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('ME,EaBIMJ->aBIJ', H.a.ov[Oa, Va], R.aab.VvVOOO, optimize=True)
    )
    dR.ab[va, Vb, Oa, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('me,aBeImJ->aBIJ', H.b.ov[ob, vb], R.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('Me,aBeIMJ->aBIJ', H.b.ov[Ob, vb], R.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('mE,aBEImJ->aBIJ', H.b.ov[ob, Vb], R.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('ME,aBEIMJ->aBIJ', H.b.ov[Ob, Vb], R.abb.vVVOOO, optimize=True)
    )
    dR.ab[va, Vb, Oa, Ob] += (1.0 / 1.0) * (
            +0.5 * np.einsum('mnIf,faBmnJ->aBIJ', H.aa.ooov[oa, oa, Oa, va], R.aab.vvVooO, optimize=True)
            + 0.5 * np.einsum('mnIF,FaBmnJ->aBIJ', H.aa.ooov[oa, oa, Oa, Va], R.aab.VvVooO, optimize=True)
            + 1.0 * np.einsum('mNIf,faBmNJ->aBIJ', H.aa.ooov[oa, Oa, Oa, va], R.aab.vvVoOO, optimize=True)
            + 0.5 * np.einsum('MNIf,faBMNJ->aBIJ', H.aa.ooov[Oa, Oa, Oa, va], R.aab.vvVOOO, optimize=True)
            + 1.0 * np.einsum('mNIF,FaBmNJ->aBIJ', H.aa.ooov[oa, Oa, Oa, Va], R.aab.VvVoOO, optimize=True)
            + 0.5 * np.einsum('MNIF,FaBMNJ->aBIJ', H.aa.ooov[Oa, Oa, Oa, Va], R.aab.VvVOOO, optimize=True)
    )
    dR.ab[va, Vb, Oa, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('nmfJ,faBnIm->aBIJ', H.ab.oovo[oa, ob, va, Ob], R.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('nMfJ,faBnIM->aBIJ', H.ab.oovo[oa, Ob, va, Ob], R.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('nmFJ,FaBnIm->aBIJ', H.ab.oovo[oa, ob, Va, Ob], R.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('nMFJ,FaBnIM->aBIJ', H.ab.oovo[oa, Ob, Va, Ob], R.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('NmfJ,faBINm->aBIJ', H.ab.oovo[Oa, ob, va, Ob], R.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('NMfJ,faBINM->aBIJ', H.ab.oovo[Oa, Ob, va, Ob], R.aab.vvVOOO, optimize=True)
            + 1.0 * np.einsum('NmFJ,FaBINm->aBIJ', H.ab.oovo[Oa, ob, Va, Ob], R.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('NMFJ,FaBINM->aBIJ', H.ab.oovo[Oa, Ob, Va, Ob], R.aab.VvVOOO, optimize=True)
    )
    dR.ab[va, Vb, Oa, Ob] += (1.0 / 1.0) * (
            +0.5 * np.einsum('mnJf,aBfInm->aBIJ', H.bb.ooov[ob, ob, Ob, vb], R.abb.vVvOoo, optimize=True)
            + 0.5 * np.einsum('mnJF,aBFInm->aBIJ', H.bb.ooov[ob, ob, Ob, Vb], R.abb.vVVOoo, optimize=True)
            - 1.0 * np.einsum('mNJf,aBfImN->aBIJ', H.bb.ooov[ob, Ob, Ob, vb], R.abb.vVvOoO, optimize=True)
            + 0.5 * np.einsum('MNJf,aBfINM->aBIJ', H.bb.ooov[Ob, Ob, Ob, vb], R.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('mNJF,aBFImN->aBIJ', H.bb.ooov[ob, Ob, Ob, Vb], R.abb.vVVOoO, optimize=True)
            + 0.5 * np.einsum('MNJF,aBFINM->aBIJ', H.bb.ooov[Ob, Ob, Ob, Vb], R.abb.vVVOOO, optimize=True)
    )
    dR.ab[va, Vb, Oa, Ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('mnIf,aBfmnJ->aBIJ', H.ab.ooov[oa, ob, Oa, vb], R.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('MnIf,aBfMnJ->aBIJ', H.ab.ooov[Oa, ob, Oa, vb], R.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('mnIF,aBFmnJ->aBIJ', H.ab.ooov[oa, ob, Oa, Vb], R.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('MnIF,aBFMnJ->aBIJ', H.ab.ooov[Oa, ob, Oa, Vb], R.abb.vVVOoO, optimize=True)
            + 1.0 * np.einsum('mNIf,aBfmNJ->aBIJ', H.ab.ooov[oa, Ob, Oa, vb], R.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('MNIf,aBfMNJ->aBIJ', H.ab.ooov[Oa, Ob, Oa, vb], R.abb.vVvOOO, optimize=True)
            + 1.0 * np.einsum('mNIF,aBFmNJ->aBIJ', H.ab.ooov[oa, Ob, Oa, Vb], R.abb.vVVoOO, optimize=True)
            + 1.0 * np.einsum('MNIF,aBFMNJ->aBIJ', H.ab.ooov[Oa, Ob, Oa, Vb], R.abb.vVVOOO, optimize=True)
    )
    dR.ab[va, Vb, Oa, Ob] += (1.0 / 1.0) * (
            +0.5 * np.einsum('anef,feBnIJ->aBIJ', H.aa.vovv[va, oa, va, va], R.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('anEf,EfBnIJ->aBIJ', H.aa.vovv[va, oa, Va, va], R.aab.VvVoOO, optimize=True)
            + 0.5 * np.einsum('anEF,FEBnIJ->aBIJ', H.aa.vovv[va, oa, Va, Va], R.aab.VVVoOO, optimize=True)
            - 0.5 * np.einsum('aNef,feBINJ->aBIJ', H.aa.vovv[va, Oa, va, va], R.aab.vvVOOO, optimize=True)
            + 1.0 * np.einsum('aNEf,EfBINJ->aBIJ', H.aa.vovv[va, Oa, Va, va], R.aab.VvVOOO, optimize=True)
            - 0.5 * np.einsum('aNEF,FEBINJ->aBIJ', H.aa.vovv[va, Oa, Va, Va], R.aab.VVVOOO, optimize=True)
    )
    dR.ab[va, Vb, Oa, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('anef,eBfInJ->aBIJ', H.ab.vovv[va, ob, va, vb], R.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('aneF,eBFInJ->aBIJ', H.ab.vovv[va, ob, va, Vb], R.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('anEf,EBfInJ->aBIJ', H.ab.vovv[va, ob, Va, vb], R.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('anEF,EBFInJ->aBIJ', H.ab.vovv[va, ob, Va, Vb], R.abb.VVVOoO, optimize=True)
            - 1.0 * np.einsum('aNef,eBfINJ->aBIJ', H.ab.vovv[va, Ob, va, vb], R.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('aNeF,eBFINJ->aBIJ', H.ab.vovv[va, Ob, va, Vb], R.abb.vVVOOO, optimize=True)
            - 1.0 * np.einsum('aNEf,EBfINJ->aBIJ', H.ab.vovv[va, Ob, Va, vb], R.abb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('aNEF,EBFINJ->aBIJ', H.ab.vovv[va, Ob, Va, Vb], R.abb.VVVOOO, optimize=True)
    )
    dR.ab[va, Vb, Oa, Ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('nBfE,faEnIJ->aBIJ', H.ab.ovvv[oa, Vb, va, Vb], R.aab.vvVoOO, optimize=True)
            + 1.0 * np.einsum('nBFe,FaenIJ->aBIJ', H.ab.ovvv[oa, Vb, Va, vb], R.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('nBFE,FaEnIJ->aBIJ', H.ab.ovvv[oa, Vb, Va, Vb], R.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('NBfE,faEINJ->aBIJ', H.ab.ovvv[Oa, Vb, va, Vb], R.aab.vvVOOO, optimize=True)
            - 1.0 * np.einsum('NBFe,FaeINJ->aBIJ', H.ab.ovvv[Oa, Vb, Va, vb], R.aab.VvvOOO, optimize=True)
            - 1.0 * np.einsum('NBFE,FaEINJ->aBIJ', H.ab.ovvv[Oa, Vb, Va, Vb], R.aab.VvVOOO, optimize=True)
    )
    dR.ab[va, Vb, Oa, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('BnEf,aEfInJ->aBIJ', H.bb.vovv[Vb, ob, Vb, vb], R.abb.vVvOoO, optimize=True)
            - 0.5 * np.einsum('BnEF,aEFInJ->aBIJ', H.bb.vovv[Vb, ob, Vb, Vb], R.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('BNEf,aEfINJ->aBIJ', H.bb.vovv[Vb, Ob, Vb, vb], R.abb.vVvOOO, optimize=True)
            - 0.5 * np.einsum('BNEF,aEFINJ->aBIJ', H.bb.vovv[Vb, Ob, Vb, Vb], R.abb.vVVOOO, optimize=True)
    )
    # of terms =  12

    return dR


def build_1101(dR, R, T, H, X, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    dR.ab[Va, Vb, oa, Ob] = (1.0 / 1.0) * (
            +1.0 * np.einsum('me,AeBimJ->ABiJ', X.a.ov[oa, va], T.aab.VvVooO, optimize=True)
            + 1.0 * np.einsum('Me,AeBiMJ->ABiJ', X.a.ov[Oa, va], T.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('mE,EABimJ->ABiJ', X.a.ov[oa, Va], T.aab.VVVooO, optimize=True)
            - 1.0 * np.einsum('ME,EABiMJ->ABiJ', X.a.ov[Oa, Va], T.aab.VVVoOO, optimize=True)
    )
    dR.ab[Va, Vb, oa, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('me,ABeimJ->ABiJ', X.b.ov[ob, vb], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('Me,ABeiMJ->ABiJ', X.b.ov[Ob, vb], T.abb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('mE,ABEimJ->ABiJ', X.b.ov[ob, Vb], T.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('ME,ABEiMJ->ABiJ', X.b.ov[Ob, Vb], T.abb.VVVoOO, optimize=True)
    )
    dR.ab[Va, Vb, oa, Ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('me,AeBimJ->ABiJ', H.a.ov[oa, va], R.aab.VvVooO, optimize=True)
            + 1.0 * np.einsum('Me,AeBiMJ->ABiJ', H.a.ov[Oa, va], R.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('mE,EABimJ->ABiJ', H.a.ov[oa, Va], R.aab.VVVooO, optimize=True)
            - 1.0 * np.einsum('ME,EABiMJ->ABiJ', H.a.ov[Oa, Va], R.aab.VVVoOO, optimize=True)
    )
    dR.ab[Va, Vb, oa, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('me,ABeimJ->ABiJ', H.b.ov[ob, vb], R.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('Me,ABeiMJ->ABiJ', H.b.ov[Ob, vb], R.abb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('mE,ABEimJ->ABiJ', H.b.ov[ob, Vb], R.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('ME,ABEiMJ->ABiJ', H.b.ov[Ob, Vb], R.abb.VVVoOO, optimize=True)
    )
    dR.ab[Va, Vb, oa, Ob] += (1.0 / 1.0) * (
            -0.5 * np.einsum('mnif,AfBmnJ->ABiJ', H.aa.ooov[oa, oa, oa, va], R.aab.VvVooO, optimize=True)
            - 1.0 * np.einsum('mNif,AfBmNJ->ABiJ', H.aa.ooov[oa, Oa, oa, va], R.aab.VvVoOO, optimize=True)
            - 0.5 * np.einsum('MNif,AfBMNJ->ABiJ', H.aa.ooov[Oa, Oa, oa, va], R.aab.VvVOOO, optimize=True)
            + 0.5 * np.einsum('mniF,FABmnJ->ABiJ', H.aa.ooov[oa, oa, oa, Va], R.aab.VVVooO, optimize=True)
            + 1.0 * np.einsum('mNiF,FABmNJ->ABiJ', H.aa.ooov[oa, Oa, oa, Va], R.aab.VVVoOO, optimize=True)
            + 0.5 * np.einsum('MNiF,FABMNJ->ABiJ', H.aa.ooov[Oa, Oa, oa, Va], R.aab.VVVOOO, optimize=True)
    )
    dR.ab[Va, Vb, oa, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('nMfJ,AfBinM->ABiJ', H.ab.oovo[oa, Ob, va, Ob], R.aab.VvVooO, optimize=True)
            - 1.0 * np.einsum('NmfJ,AfBiNm->ABiJ', H.ab.oovo[Oa, ob, va, Ob], R.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('NMfJ,AfBiNM->ABiJ', H.ab.oovo[Oa, Ob, va, Ob], R.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('nMFJ,FABinM->ABiJ', H.ab.oovo[oa, Ob, Va, Ob], R.aab.VVVooO, optimize=True)
            + 1.0 * np.einsum('NmFJ,FABiNm->ABiJ', H.ab.oovo[Oa, ob, Va, Ob], R.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('NMFJ,FABiNM->ABiJ', H.ab.oovo[Oa, Ob, Va, Ob], R.aab.VVVoOO, optimize=True)
    )
    dR.ab[Va, Vb, oa, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('mNJf,ABfimN->ABiJ', H.bb.ooov[ob, Ob, Ob, vb], R.abb.VVvooO, optimize=True)
            + 0.5 * np.einsum('MNJf,ABfiNM->ABiJ', H.bb.ooov[Ob, Ob, Ob, vb], R.abb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('mNJF,ABFimN->ABiJ', H.bb.ooov[ob, Ob, Ob, Vb], R.abb.VVVooO, optimize=True)
            + 0.5 * np.einsum('MNJF,ABFiNM->ABiJ', H.bb.ooov[Ob, Ob, Ob, Vb], R.abb.VVVoOO, optimize=True)
    )
    dR.ab[Va, Vb, oa, Ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('mnif,ABfmnJ->ABiJ', H.ab.ooov[oa, ob, oa, vb], R.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('Mnif,ABfMnJ->ABiJ', H.ab.ooov[Oa, ob, oa, vb], R.abb.VVvOoO, optimize=True)
            + 1.0 * np.einsum('mNif,ABfmNJ->ABiJ', H.ab.ooov[oa, Ob, oa, vb], R.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('MNif,ABfMNJ->ABiJ', H.ab.ooov[Oa, Ob, oa, vb], R.abb.VVvOOO, optimize=True)
            + 1.0 * np.einsum('mniF,ABFmnJ->ABiJ', H.ab.ooov[oa, ob, oa, Vb], R.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('MniF,ABFMnJ->ABiJ', H.ab.ooov[Oa, ob, oa, Vb], R.abb.VVVOoO, optimize=True)
            + 1.0 * np.einsum('mNiF,ABFmNJ->ABiJ', H.ab.ooov[oa, Ob, oa, Vb], R.abb.VVVoOO, optimize=True)
            + 1.0 * np.einsum('MNiF,ABFMNJ->ABiJ', H.ab.ooov[Oa, Ob, oa, Vb], R.abb.VVVOOO, optimize=True)
    )
    dR.ab[Va, Vb, oa, Ob] += (1.0 / 1.0) * (
            -0.5 * np.einsum('Anef,feBinJ->ABiJ', H.aa.vovv[Va, oa, va, va], R.aab.vvVooO, optimize=True)
            - 0.5 * np.einsum('ANef,feBiNJ->ABiJ', H.aa.vovv[Va, Oa, va, va], R.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('AneF,FeBinJ->ABiJ', H.aa.vovv[Va, oa, va, Va], R.aab.VvVooO, optimize=True)
            - 0.5 * np.einsum('AnEF,FEBinJ->ABiJ', H.aa.vovv[Va, oa, Va, Va], R.aab.VVVooO, optimize=True)
            - 1.0 * np.einsum('ANeF,FeBiNJ->ABiJ', H.aa.vovv[Va, Oa, va, Va], R.aab.VvVoOO, optimize=True)
            - 0.5 * np.einsum('ANEF,FEBiNJ->ABiJ', H.aa.vovv[Va, Oa, Va, Va], R.aab.VVVoOO, optimize=True)
    )
    dR.ab[Va, Vb, oa, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('Anef,eBfinJ->ABiJ', H.ab.vovv[Va, ob, va, vb], R.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('AnEf,EBfinJ->ABiJ', H.ab.vovv[Va, ob, Va, vb], R.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('ANef,eBfiNJ->ABiJ', H.ab.vovv[Va, Ob, va, vb], R.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('ANEf,EBfiNJ->ABiJ', H.ab.vovv[Va, Ob, Va, vb], R.abb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('AneF,eBFinJ->ABiJ', H.ab.vovv[Va, ob, va, Vb], R.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('AnEF,EBFinJ->ABiJ', H.ab.vovv[Va, ob, Va, Vb], R.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('ANeF,eBFiNJ->ABiJ', H.ab.vovv[Va, Ob, va, Vb], R.abb.vVVoOO, optimize=True)
            - 1.0 * np.einsum('ANEF,EBFiNJ->ABiJ', H.ab.vovv[Va, Ob, Va, Vb], R.abb.VVVoOO, optimize=True)
    )
    dR.ab[Va, Vb, oa, Ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('nBfe,AfeinJ->ABiJ', H.ab.ovvv[oa, Vb, va, vb], R.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('nBfE,AfEinJ->ABiJ', H.ab.ovvv[oa, Vb, va, Vb], R.aab.VvVooO, optimize=True)
            + 1.0 * np.einsum('NBfe,AfeiNJ->ABiJ', H.ab.ovvv[Oa, Vb, va, vb], R.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('NBfE,AfEiNJ->ABiJ', H.ab.ovvv[Oa, Vb, va, Vb], R.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('nBFe,FAeinJ->ABiJ', H.ab.ovvv[oa, Vb, Va, vb], R.aab.VVvooO, optimize=True)
            - 1.0 * np.einsum('nBFE,FAEinJ->ABiJ', H.ab.ovvv[oa, Vb, Va, Vb], R.aab.VVVooO, optimize=True)
            - 1.0 * np.einsum('NBFe,FAeiNJ->ABiJ', H.ab.ovvv[Oa, Vb, Va, vb], R.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('NBFE,FAEiNJ->ABiJ', H.ab.ovvv[Oa, Vb, Va, Vb], R.aab.VVVoOO, optimize=True)
    )
    dR.ab[Va, Vb, oa, Ob] += (1.0 / 1.0) * (
            -0.5 * np.einsum('Bnef,AefinJ->ABiJ', H.bb.vovv[Vb, ob, vb, vb], R.abb.VvvooO, optimize=True)
            - 0.5 * np.einsum('BNef,AefiNJ->ABiJ', H.bb.vovv[Vb, Ob, vb, vb], R.abb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('BneF,AFeinJ->ABiJ', H.bb.vovv[Vb, ob, vb, Vb], R.abb.VVvooO, optimize=True)
            - 0.5 * np.einsum('BnEF,AEFinJ->ABiJ', H.bb.vovv[Vb, ob, Vb, Vb], R.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('BNeF,AFeiNJ->ABiJ', H.bb.vovv[Vb, Ob, vb, Vb], R.abb.VVvoOO, optimize=True)
            - 0.5 * np.einsum('BNEF,AEFiNJ->ABiJ', H.bb.vovv[Vb, Ob, Vb, Vb], R.abb.VVVoOO, optimize=True)
    )
    # of terms =  12

    return dR


def build_1110(dR, R, T, H, X, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    dR.ab[Va, Vb, Oa, ob] = (1.0 / 1.0) * (
            -1.0 * np.einsum('me,AeBmIj->ABIj', X.a.ov[oa, va], T.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('Me,AeBIMj->ABIj', X.a.ov[Oa, va], T.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('mE,EABmIj->ABIj', X.a.ov[oa, Va], T.aab.VVVoOo, optimize=True)
            - 1.0 * np.einsum('ME,EABIMj->ABIj', X.a.ov[Oa, Va], T.aab.VVVOOo, optimize=True)
    )
    dR.ab[Va, Vb, Oa, ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('me,ABeImj->ABIj', X.b.ov[ob, vb], T.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('Me,ABeIjM->ABIj', X.b.ov[Ob, vb], T.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('mE,ABEImj->ABIj', X.b.ov[ob, Vb], T.abb.VVVOoo, optimize=True)
            + 1.0 * np.einsum('ME,ABEIjM->ABIj', X.b.ov[Ob, Vb], T.abb.VVVOoO, optimize=True)
    )
    dR.ab[Va, Vb, Oa, ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('me,AeBmIj->ABIj', H.a.ov[oa, va], R.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('Me,AeBIMj->ABIj', H.a.ov[Oa, va], R.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('mE,EABmIj->ABIj', H.a.ov[oa, Va], R.aab.VVVoOo, optimize=True)
            - 1.0 * np.einsum('ME,EABIMj->ABIj', H.a.ov[Oa, Va], R.aab.VVVOOo, optimize=True)
    )
    dR.ab[Va, Vb, Oa, ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('me,ABeImj->ABIj', H.b.ov[ob, vb], R.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('Me,ABeIjM->ABIj', H.b.ov[Ob, vb], R.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('mE,ABEImj->ABIj', H.b.ov[ob, Vb], R.abb.VVVOoo, optimize=True)
            + 1.0 * np.einsum('ME,ABEIjM->ABIj', H.b.ov[Ob, Vb], R.abb.VVVOoO, optimize=True)
    )
    dR.ab[Va, Vb, Oa, ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('mNIf,AfBmNj->ABIj', H.aa.ooov[oa, Oa, Oa, va], R.aab.VvVoOo, optimize=True)
            - 0.5 * np.einsum('MNIf,AfBMNj->ABIj', H.aa.ooov[Oa, Oa, Oa, va], R.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('mNIF,FABmNj->ABIj', H.aa.ooov[oa, Oa, Oa, Va], R.aab.VVVoOo, optimize=True)
            + 0.5 * np.einsum('MNIF,FABMNj->ABIj', H.aa.ooov[Oa, Oa, Oa, Va], R.aab.VVVOOo, optimize=True)
    )
    dR.ab[Va, Vb, Oa, ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('nmfj,AfBnIm->ABIj', H.ab.oovo[oa, ob, va, ob], R.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('nMfj,AfBnIM->ABIj', H.ab.oovo[oa, Ob, va, ob], R.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('nmFj,FABnIm->ABIj', H.ab.oovo[oa, ob, Va, ob], R.aab.VVVoOo, optimize=True)
            - 1.0 * np.einsum('nMFj,FABnIM->ABIj', H.ab.oovo[oa, Ob, Va, ob], R.aab.VVVoOO, optimize=True)
            - 1.0 * np.einsum('Nmfj,AfBINm->ABIj', H.ab.oovo[Oa, ob, va, ob], R.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('NMfj,AfBINM->ABIj', H.ab.oovo[Oa, Ob, va, ob], R.aab.VvVOOO, optimize=True)
            + 1.0 * np.einsum('NmFj,FABINm->ABIj', H.ab.oovo[Oa, ob, Va, ob], R.aab.VVVOOo, optimize=True)
            + 1.0 * np.einsum('NMFj,FABINM->ABIj', H.ab.oovo[Oa, Ob, Va, ob], R.aab.VVVOOO, optimize=True)
    )
    dR.ab[Va, Vb, Oa, ob] += (1.0 / 1.0) * (
            +0.5 * np.einsum('mnjf,ABfInm->ABIj', H.bb.ooov[ob, ob, ob, vb], R.abb.VVvOoo, optimize=True)
            + 0.5 * np.einsum('mnjF,ABFInm->ABIj', H.bb.ooov[ob, ob, ob, Vb], R.abb.VVVOoo, optimize=True)
            - 1.0 * np.einsum('mNjf,ABfImN->ABIj', H.bb.ooov[ob, Ob, ob, vb], R.abb.VVvOoO, optimize=True)
            + 0.5 * np.einsum('MNjf,ABfINM->ABIj', H.bb.ooov[Ob, Ob, ob, vb], R.abb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('mNjF,ABFImN->ABIj', H.bb.ooov[ob, Ob, ob, Vb], R.abb.VVVOoO, optimize=True)
            + 0.5 * np.einsum('MNjF,ABFINM->ABIj', H.bb.ooov[Ob, Ob, ob, Vb], R.abb.VVVOOO, optimize=True)
    )
    dR.ab[Va, Vb, Oa, ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('MnIf,ABfMnj->ABIj', H.ab.ooov[Oa, ob, Oa, vb], R.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('MnIF,ABFMnj->ABIj', H.ab.ooov[Oa, ob, Oa, Vb], R.abb.VVVOoo, optimize=True)
            - 1.0 * np.einsum('mNIf,ABfmjN->ABIj', H.ab.ooov[oa, Ob, Oa, vb], R.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('MNIf,ABfMjN->ABIj', H.ab.ooov[Oa, Ob, Oa, vb], R.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('mNIF,ABFmjN->ABIj', H.ab.ooov[oa, Ob, Oa, Vb], R.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('MNIF,ABFMjN->ABIj', H.ab.ooov[Oa, Ob, Oa, Vb], R.abb.VVVOoO, optimize=True)
    )
    dR.ab[Va, Vb, Oa, ob] += (1.0 / 1.0) * (
            +0.5 * np.einsum('Anef,feBnIj->ABIj', H.aa.vovv[Va, oa, va, va], R.aab.vvVoOo, optimize=True)
            - 0.5 * np.einsum('ANef,feBINj->ABIj', H.aa.vovv[Va, Oa, va, va], R.aab.vvVOOo, optimize=True)
            - 1.0 * np.einsum('AnEf,EfBnIj->ABIj', H.aa.vovv[Va, oa, Va, va], R.aab.VvVoOo, optimize=True)
            + 0.5 * np.einsum('AnEF,FEBnIj->ABIj', H.aa.vovv[Va, oa, Va, Va], R.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('ANEf,EfBINj->ABIj', H.aa.vovv[Va, Oa, Va, va], R.aab.VvVOOo, optimize=True)
            - 0.5 * np.einsum('ANEF,FEBINj->ABIj', H.aa.vovv[Va, Oa, Va, Va], R.aab.VVVOOo, optimize=True)
    )
    dR.ab[Va, Vb, Oa, ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('Anef,eBfInj->ABIj', H.ab.vovv[Va, ob, va, vb], R.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('AneF,eBFInj->ABIj', H.ab.vovv[Va, ob, va, Vb], R.abb.vVVOoo, optimize=True)
            + 1.0 * np.einsum('ANef,eBfIjN->ABIj', H.ab.vovv[Va, Ob, va, vb], R.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('ANeF,eBFIjN->ABIj', H.ab.vovv[Va, Ob, va, Vb], R.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('AnEf,EBfInj->ABIj', H.ab.vovv[Va, ob, Va, vb], R.abb.VVvOoo, optimize=True)
            - 1.0 * np.einsum('AnEF,EBFInj->ABIj', H.ab.vovv[Va, ob, Va, Vb], R.abb.VVVOoo, optimize=True)
            + 1.0 * np.einsum('ANEf,EBfIjN->ABIj', H.ab.vovv[Va, Ob, Va, vb], R.abb.VVvOoO, optimize=True)
            + 1.0 * np.einsum('ANEF,EBFIjN->ABIj', H.ab.vovv[Va, Ob, Va, Vb], R.abb.VVVOoO, optimize=True)
    )
    dR.ab[Va, Vb, Oa, ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('nBfe,AfenIj->ABIj', H.ab.ovvv[oa, Vb, va, vb], R.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('nBFe,FAenIj->ABIj', H.ab.ovvv[oa, Vb, Va, vb], R.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('NBfe,AfeINj->ABIj', H.ab.ovvv[Oa, Vb, va, vb], R.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('NBFe,FAeINj->ABIj', H.ab.ovvv[Oa, Vb, Va, vb], R.aab.VVvOOo, optimize=True)
            - 1.0 * np.einsum('nBfE,AfEnIj->ABIj', H.ab.ovvv[oa, Vb, va, Vb], R.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('nBFE,FAEnIj->ABIj', H.ab.ovvv[oa, Vb, Va, Vb], R.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('NBfE,AfEINj->ABIj', H.ab.ovvv[Oa, Vb, va, Vb], R.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('NBFE,FAEINj->ABIj', H.ab.ovvv[Oa, Vb, Va, Vb], R.aab.VVVOOo, optimize=True)
    )
    dR.ab[Va, Vb, Oa, ob] += (1.0 / 1.0) * (
            -0.5 * np.einsum('Bnef,AefInj->ABIj', H.bb.vovv[Vb, ob, vb, vb], R.abb.VvvOoo, optimize=True)
            + 0.5 * np.einsum('BNef,AefIjN->ABIj', H.bb.vovv[Vb, Ob, vb, vb], R.abb.VvvOoO, optimize=True)
            - 1.0 * np.einsum('BnEf,AEfInj->ABIj', H.bb.vovv[Vb, ob, Vb, vb], R.abb.VVvOoo, optimize=True)
            - 0.5 * np.einsum('BnEF,AEFInj->ABIj', H.bb.vovv[Vb, ob, Vb, Vb], R.abb.VVVOoo, optimize=True)
            + 1.0 * np.einsum('BNEf,AEfIjN->ABIj', H.bb.vovv[Vb, Ob, Vb, vb], R.abb.VVvOoO, optimize=True)
            + 0.5 * np.einsum('BNEF,AEFIjN->ABIj', H.bb.vovv[Vb, Ob, Vb, Vb], R.abb.VVVOoO, optimize=True)
    )
    # of terms =  12

    return dR


def build_1001(dR, R, T, H, X, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    dR.ab[Va, vb, oa, Ob] = (1.0 / 1.0) * (
            +1.0 * np.einsum('me,AebimJ->AbiJ', X.a.ov[oa, va], T.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('Me,AebiMJ->AbiJ', X.a.ov[Oa, va], T.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('mE,EAbimJ->AbiJ', X.a.ov[oa, Va], T.aab.VVvooO, optimize=True)
            - 1.0 * np.einsum('ME,EAbiMJ->AbiJ', X.a.ov[Oa, Va], T.aab.VVvoOO, optimize=True)
    )
    dR.ab[Va, vb, oa, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('me,AbeimJ->AbiJ', X.b.ov[ob, vb], T.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('Me,AbeiMJ->AbiJ', X.b.ov[Ob, vb], T.abb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('mE,AEbimJ->AbiJ', X.b.ov[ob, Vb], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('ME,AEbiMJ->AbiJ', X.b.ov[Ob, Vb], T.abb.VVvoOO, optimize=True)
    )
    dR.ab[Va, vb, oa, Ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('me,AebimJ->AbiJ', H.a.ov[oa, va], R.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('Me,AebiMJ->AbiJ', H.a.ov[Oa, va], R.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('mE,EAbimJ->AbiJ', H.a.ov[oa, Va], R.aab.VVvooO, optimize=True)
            - 1.0 * np.einsum('ME,EAbiMJ->AbiJ', H.a.ov[Oa, Va], R.aab.VVvoOO, optimize=True)
    )
    dR.ab[Va, vb, oa, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('me,AbeimJ->AbiJ', H.b.ov[ob, vb], R.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('Me,AbeiMJ->AbiJ', H.b.ov[Ob, vb], R.abb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('mE,AEbimJ->AbiJ', H.b.ov[ob, Vb], R.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('ME,AEbiMJ->AbiJ', H.b.ov[Ob, Vb], R.abb.VVvoOO, optimize=True)
    )
    dR.ab[Va, vb, oa, Ob] += (1.0 / 1.0) * (
            -0.5 * np.einsum('mnif,AfbmnJ->AbiJ', H.aa.ooov[oa, oa, oa, va], R.aab.VvvooO, optimize=True)
            + 0.5 * np.einsum('mniF,FAbmnJ->AbiJ', H.aa.ooov[oa, oa, oa, Va], R.aab.VVvooO, optimize=True)
            - 1.0 * np.einsum('mNif,AfbmNJ->AbiJ', H.aa.ooov[oa, Oa, oa, va], R.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('mNiF,FAbmNJ->AbiJ', H.aa.ooov[oa, Oa, oa, Va], R.aab.VVvoOO, optimize=True)
            - 0.5 * np.einsum('MNif,AfbMNJ->AbiJ', H.aa.ooov[Oa, Oa, oa, va], R.aab.VvvOOO, optimize=True)
            + 0.5 * np.einsum('MNiF,FAbMNJ->AbiJ', H.aa.ooov[Oa, Oa, oa, Va], R.aab.VVvOOO, optimize=True)
    )
    dR.ab[Va, vb, oa, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('nMfJ,AfbinM->AbiJ', H.ab.oovo[oa, Ob, va, Ob], R.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('nMFJ,FAbinM->AbiJ', H.ab.oovo[oa, Ob, Va, Ob], R.aab.VVvooO, optimize=True)
            - 1.0 * np.einsum('NmfJ,AfbiNm->AbiJ', H.ab.oovo[Oa, ob, va, Ob], R.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('NmFJ,FAbiNm->AbiJ', H.ab.oovo[Oa, ob, Va, Ob], R.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('NMfJ,AfbiNM->AbiJ', H.ab.oovo[Oa, Ob, va, Ob], R.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('NMFJ,FAbiNM->AbiJ', H.ab.oovo[Oa, Ob, Va, Ob], R.aab.VVvoOO, optimize=True)
    )
    dR.ab[Va, vb, oa, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('mNJf,AbfimN->AbiJ', H.bb.ooov[ob, Ob, Ob, vb], R.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('mNJF,AFbimN->AbiJ', H.bb.ooov[ob, Ob, Ob, Vb], R.abb.VVvooO, optimize=True)
            + 0.5 * np.einsum('MNJf,AbfiNM->AbiJ', H.bb.ooov[Ob, Ob, Ob, vb], R.abb.VvvoOO, optimize=True)
            - 0.5 * np.einsum('MNJF,AFbiNM->AbiJ', H.bb.ooov[Ob, Ob, Ob, Vb], R.abb.VVvoOO, optimize=True)
    )
    dR.ab[Va, vb, oa, Ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('mnif,AbfmnJ->AbiJ', H.ab.ooov[oa, ob, oa, vb], R.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('mniF,AFbmnJ->AbiJ', H.ab.ooov[oa, ob, oa, Vb], R.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('Mnif,AbfMnJ->AbiJ', H.ab.ooov[Oa, ob, oa, vb], R.abb.VvvOoO, optimize=True)
            - 1.0 * np.einsum('MniF,AFbMnJ->AbiJ', H.ab.ooov[Oa, ob, oa, Vb], R.abb.VVvOoO, optimize=True)
            + 1.0 * np.einsum('mNif,AbfmNJ->AbiJ', H.ab.ooov[oa, Ob, oa, vb], R.abb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('mNiF,AFbmNJ->AbiJ', H.ab.ooov[oa, Ob, oa, Vb], R.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('MNif,AbfMNJ->AbiJ', H.ab.ooov[Oa, Ob, oa, vb], R.abb.VvvOOO, optimize=True)
            - 1.0 * np.einsum('MNiF,AFbMNJ->AbiJ', H.ab.ooov[Oa, Ob, oa, Vb], R.abb.VVvOOO, optimize=True)
    )
    dR.ab[Va, vb, oa, Ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('AnEf,EfbinJ->AbiJ', H.aa.vovv[Va, oa, Va, va], R.aab.VvvooO, optimize=True)
            - 0.5 * np.einsum('AnEF,FEbinJ->AbiJ', H.aa.vovv[Va, oa, Va, Va], R.aab.VVvooO, optimize=True)
            + 1.0 * np.einsum('ANEf,EfbiNJ->AbiJ', H.aa.vovv[Va, Oa, Va, va], R.aab.VvvoOO, optimize=True)
            - 0.5 * np.einsum('ANEF,FEbiNJ->AbiJ', H.aa.vovv[Va, Oa, Va, Va], R.aab.VVvoOO, optimize=True)
    )
    dR.ab[Va, vb, oa, Ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('AneF,eFbinJ->AbiJ', H.ab.vovv[Va, ob, va, Vb], R.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('ANeF,eFbiNJ->AbiJ', H.ab.vovv[Va, Ob, va, Vb], R.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('AnEf,EbfinJ->AbiJ', H.ab.vovv[Va, ob, Va, vb], R.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('AnEF,EFbinJ->AbiJ', H.ab.vovv[Va, ob, Va, Vb], R.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('ANEf,EbfiNJ->AbiJ', H.ab.vovv[Va, Ob, Va, vb], R.abb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('ANEF,EFbiNJ->AbiJ', H.ab.vovv[Va, Ob, Va, Vb], R.abb.VVvoOO, optimize=True)
    )
    dR.ab[Va, vb, oa, Ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('nbfe,AfeinJ->AbiJ', H.ab.ovvv[oa, vb, va, vb], R.aab.VvvooO, optimize=True)
            - 1.0 * np.einsum('nbFe,FAeinJ->AbiJ', H.ab.ovvv[oa, vb, Va, vb], R.aab.VVvooO, optimize=True)
            + 1.0 * np.einsum('Nbfe,AfeiNJ->AbiJ', H.ab.ovvv[Oa, vb, va, vb], R.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('NbFe,FAeiNJ->AbiJ', H.ab.ovvv[Oa, vb, Va, vb], R.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('nbfE,AfEinJ->AbiJ', H.ab.ovvv[oa, vb, va, Vb], R.aab.VvVooO, optimize=True)
            - 1.0 * np.einsum('nbFE,FAEinJ->AbiJ', H.ab.ovvv[oa, vb, Va, Vb], R.aab.VVVooO, optimize=True)
            + 1.0 * np.einsum('NbfE,AfEiNJ->AbiJ', H.ab.ovvv[Oa, vb, va, Vb], R.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('NbFE,FAEiNJ->AbiJ', H.ab.ovvv[Oa, vb, Va, Vb], R.aab.VVVoOO, optimize=True)
    )
    dR.ab[Va, vb, oa, Ob] += (1.0 / 1.0) * (
            -0.5 * np.einsum('bnef,AefinJ->AbiJ', H.bb.vovv[vb, ob, vb, vb], R.abb.VvvooO, optimize=True)
            - 0.5 * np.einsum('bNef,AefiNJ->AbiJ', H.bb.vovv[vb, Ob, vb, vb], R.abb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('bnEf,AEfinJ->AbiJ', H.bb.vovv[vb, ob, Vb, vb], R.abb.VVvooO, optimize=True)
            - 0.5 * np.einsum('bnEF,AEFinJ->AbiJ', H.bb.vovv[vb, ob, Vb, Vb], R.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('bNEf,AEfiNJ->AbiJ', H.bb.vovv[vb, Ob, Vb, vb], R.abb.VVvoOO, optimize=True)
            - 0.5 * np.einsum('bNEF,AEFiNJ->AbiJ', H.bb.vovv[vb, Ob, Vb, Vb], R.abb.VVVoOO, optimize=True)
    )
    # of terms =  12

    return dR


def build_1100(dR, R, T, H, X, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    dR.ab[Va, Vb, oa, ob] = (1.0 / 1.0) * (
            +1.0 * np.einsum('Me,AeBiMj->ABij', X.a.ov[Oa, va], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('ME,EABiMj->ABij', X.a.ov[Oa, Va], T.aab.VVVoOo, optimize=True)
    )
    dR.ab[Va, Vb, oa, ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('Me,ABeijM->ABij', X.b.ov[Ob, vb], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('ME,ABEijM->ABij', X.b.ov[Ob, Vb], T.abb.VVVooO, optimize=True)
    )
    dR.ab[Va, Vb, oa, ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('Me,AeBiMj->ABij', H.a.ov[Oa, va], R.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('ME,EABiMj->ABij', H.a.ov[Oa, Va], R.aab.VVVoOo, optimize=True)
    )
    dR.ab[Va, Vb, oa, ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('Me,ABeijM->ABij', H.b.ov[Ob, vb], R.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('ME,ABEijM->ABij', H.b.ov[Ob, Vb], R.abb.VVVooO, optimize=True)
    )
    dR.ab[Va, Vb, oa, ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('Mnif,AfBnMj->ABij', H.aa.ooov[Oa, oa, oa, va], R.aab.VvVoOo, optimize=True)
            - 0.5 * np.einsum('MNif,AfBMNj->ABij', H.aa.ooov[Oa, Oa, oa, va], R.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('MniF,FABnMj->ABij', H.aa.ooov[Oa, oa, oa, Va], R.aab.VVVoOo, optimize=True)
            + 0.5 * np.einsum('MNiF,FABMNj->ABij', H.aa.ooov[Oa, Oa, oa, Va], R.aab.VVVOOo, optimize=True)
    )
    dR.ab[Va, Vb, oa, ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('Nmfj,AfBiNm->ABij', H.ab.oovo[Oa, ob, va, ob], R.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('nMfj,AfBinM->ABij', H.ab.oovo[oa, Ob, va, ob], R.aab.VvVooO, optimize=True)
            - 1.0 * np.einsum('NMfj,AfBiNM->ABij', H.ab.oovo[Oa, Ob, va, ob], R.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('NmFj,FABiNm->ABij', H.ab.oovo[Oa, ob, Va, ob], R.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('nMFj,FABinM->ABij', H.ab.oovo[oa, Ob, Va, ob], R.aab.VVVooO, optimize=True)
            + 1.0 * np.einsum('NMFj,FABiNM->ABij', H.ab.oovo[Oa, Ob, Va, ob], R.aab.VVVoOO, optimize=True)
    )
    dR.ab[Va, Vb, oa, ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('Mnjf,ABfinM->ABij', H.bb.ooov[Ob, ob, ob, vb], R.abb.VVvooO, optimize=True)
            + 0.5 * np.einsum('MNjf,ABfiNM->ABij', H.bb.ooov[Ob, Ob, ob, vb], R.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('MnjF,ABFinM->ABij', H.bb.ooov[Ob, ob, ob, Vb], R.abb.VVVooO, optimize=True)
            + 0.5 * np.einsum('MNjF,ABFiNM->ABij', H.bb.ooov[Ob, Ob, ob, Vb], R.abb.VVVoOO, optimize=True)
    )
    dR.ab[Va, Vb, oa, ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('mNif,ABfmjN->ABij', H.ab.ooov[oa, Ob, oa, vb], R.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('Mnif,ABfMnj->ABij', H.ab.ooov[Oa, ob, oa, vb], R.abb.VVvOoo, optimize=True)
            - 1.0 * np.einsum('MNif,ABfMjN->ABij', H.ab.ooov[Oa, Ob, oa, vb], R.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('mNiF,ABFmjN->ABij', H.ab.ooov[oa, Ob, oa, Vb], R.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('MniF,ABFMnj->ABij', H.ab.ooov[Oa, ob, oa, Vb], R.abb.VVVOoo, optimize=True)
            - 1.0 * np.einsum('MNiF,ABFMjN->ABij', H.ab.ooov[Oa, Ob, oa, Vb], R.abb.VVVOoO, optimize=True)
    )
    dR.ab[Va, Vb, oa, ob] += (1.0 / 1.0) * (
            -0.5 * np.einsum('ANef,feBiNj->ABij', H.aa.vovv[Va, Oa, va, va], R.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('ANEf,EfBiNj->ABij', H.aa.vovv[Va, Oa, Va, va], R.aab.VvVoOo, optimize=True)
            - 0.5 * np.einsum('ANEF,FEBiNj->ABij', H.aa.vovv[Va, Oa, Va, Va], R.aab.VVVoOo, optimize=True)
    )
    dR.ab[Va, Vb, oa, ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('ANef,eBfijN->ABij', H.ab.vovv[Va, Ob, va, vb], R.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('ANeF,eBFijN->ABij', H.ab.vovv[Va, Ob, va, Vb], R.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('ANEf,EBfijN->ABij', H.ab.vovv[Va, Ob, Va, vb], R.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('ANEF,EBFijN->ABij', H.ab.vovv[Va, Ob, Va, Vb], R.abb.VVVooO, optimize=True)
    )
    dR.ab[Va, Vb, oa, ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('NBfe,AfeiNj->ABij', H.ab.ovvv[Oa, Vb, va, vb], R.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('NBFe,FAeiNj->ABij', H.ab.ovvv[Oa, Vb, Va, vb], R.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('NBfE,AfEiNj->ABij', H.ab.ovvv[Oa, Vb, va, Vb], R.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('NBFE,FAEiNj->ABij', H.ab.ovvv[Oa, Vb, Va, Vb], R.aab.VVVoOo, optimize=True)
    )
    dR.ab[Va, Vb, oa, ob] += (1.0 / 1.0) * (
            +0.5 * np.einsum('BNef,AefijN->ABij', H.bb.vovv[Vb, Ob, vb, vb], R.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('BNEf,AEfijN->ABij', H.bb.vovv[Vb, Ob, Vb, vb], R.abb.VVvooO, optimize=True)
            + 0.5 * np.einsum('BNEF,AEFijN->ABij', H.bb.vovv[Vb, Ob, Vb, Vb], R.abb.VVVooO, optimize=True)
    )
    # of terms =  12

    return dR

def build_0011(dR, R, T, H, X, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    dR.ab[va, vb, Oa, Ob] = (1.0 / 1.0) * (
            +1.0 * np.einsum('mE,EabmIJ->abIJ', X.a.ov[oa, Va], T.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('ME,EabIMJ->abIJ', X.a.ov[Oa, Va], T.aab.VvvOOO, optimize=True)
    )
    dR.ab[va, vb, Oa, Ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('mE,aEbImJ->abIJ', X.b.ov[ob, Vb], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('ME,aEbIMJ->abIJ', X.b.ov[Ob, Vb], T.abb.vVvOOO, optimize=True)
    )
    dR.ab[va, vb, Oa, Ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('mE,EabmIJ->abIJ', H.a.ov[oa, Va], R.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('ME,EabIMJ->abIJ', H.a.ov[Oa, Va], R.aab.VvvOOO, optimize=True)
    )
    dR.ab[va, vb, Oa, Ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('mE,aEbImJ->abIJ', H.b.ov[ob, Vb], R.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('ME,aEbIMJ->abIJ', H.b.ov[Ob, Vb], R.abb.vVvOOO, optimize=True)
    )
    dR.ab[va, vb, Oa, Ob] += (1.0 / 1.0) * (
            +0.5 * np.einsum('mnIF,FabmnJ->abIJ', H.aa.ooov[oa, oa, Oa, Va], R.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('mNIF,FabmNJ->abIJ', H.aa.ooov[oa, Oa, Oa, Va], R.aab.VvvoOO, optimize=True)
            + 0.5 * np.einsum('MNIF,FabMNJ->abIJ', H.aa.ooov[Oa, Oa, Oa, Va], R.aab.VvvOOO, optimize=True)
    )
    dR.ab[va, vb, Oa, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('nmFJ,FabnIm->abIJ', H.ab.oovo[oa, ob, Va, Ob], R.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('nMFJ,FabnIM->abIJ', H.ab.oovo[oa, Ob, Va, Ob], R.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('NmFJ,FabINm->abIJ', H.ab.oovo[Oa, ob, Va, Ob], R.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('NMFJ,FabINM->abIJ', H.ab.oovo[Oa, Ob, Va, Ob], R.aab.VvvOOO, optimize=True)
    )
    dR.ab[va, vb, Oa, Ob] += (1.0 / 1.0) * (
            -0.5 * np.einsum('mnJF,aFbInm->abIJ', H.bb.ooov[ob, ob, Ob, Vb], R.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('mNJF,aFbImN->abIJ', H.bb.ooov[ob, Ob, Ob, Vb], R.abb.vVvOoO, optimize=True)
            - 0.5 * np.einsum('MNJF,aFbINM->abIJ', H.bb.ooov[Ob, Ob, Ob, Vb], R.abb.vVvOOO, optimize=True)
    )
    dR.ab[va, vb, Oa, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('mnIF,aFbmnJ->abIJ', H.ab.ooov[oa, ob, Oa, Vb], R.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('MnIF,aFbMnJ->abIJ', H.ab.ooov[Oa, ob, Oa, Vb], R.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('mNIF,aFbmNJ->abIJ', H.ab.ooov[oa, Ob, Oa, Vb], R.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('MNIF,aFbMNJ->abIJ', H.ab.ooov[Oa, Ob, Oa, Vb], R.abb.vVvOOO, optimize=True)
    )
    dR.ab[va, vb, Oa, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('anEf,EfbnIJ->abIJ', H.aa.vovv[va, oa, Va, va], R.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('aNEf,EfbINJ->abIJ', H.aa.vovv[va, Oa, Va, va], R.aab.VvvOOO, optimize=True)
            + 0.5 * np.einsum('anEF,FEbnIJ->abIJ', H.aa.vovv[va, oa, Va, Va], R.aab.VVvoOO, optimize=True)
            - 0.5 * np.einsum('aNEF,FEbINJ->abIJ', H.aa.vovv[va, Oa, Va, Va], R.aab.VVvOOO, optimize=True)
    )
    dR.ab[va, vb, Oa, Ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('aneF,eFbInJ->abIJ', H.ab.vovv[va, ob, va, Vb], R.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('aNeF,eFbINJ->abIJ', H.ab.vovv[va, Ob, va, Vb], R.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('anEf,EbfInJ->abIJ', H.ab.vovv[va, ob, Va, vb], R.abb.VvvOoO, optimize=True)
            - 1.0 * np.einsum('aNEf,EbfINJ->abIJ', H.ab.vovv[va, Ob, Va, vb], R.abb.VvvOOO, optimize=True)
            + 1.0 * np.einsum('anEF,EFbInJ->abIJ', H.ab.vovv[va, ob, Va, Vb], R.abb.VVvOoO, optimize=True)
            + 1.0 * np.einsum('aNEF,EFbINJ->abIJ', H.ab.vovv[va, Ob, Va, Vb], R.abb.VVvOOO, optimize=True)
    )
    dR.ab[va, vb, Oa, Ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('nbFe,FaenIJ->abIJ', H.ab.ovvv[oa, vb, Va, vb], R.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('NbFe,FaeINJ->abIJ', H.ab.ovvv[Oa, vb, Va, vb], R.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('nbfE,faEnIJ->abIJ', H.ab.ovvv[oa, vb, va, Vb], R.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('NbfE,faEINJ->abIJ', H.ab.ovvv[Oa, vb, va, Vb], R.aab.vvVOOO, optimize=True)
            + 1.0 * np.einsum('nbFE,FaEnIJ->abIJ', H.ab.ovvv[oa, vb, Va, Vb], R.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('NbFE,FaEINJ->abIJ', H.ab.ovvv[Oa, vb, Va, Vb], R.aab.VvVOOO, optimize=True)
    )
    dR.ab[va, vb, Oa, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('bnEf,aEfInJ->abIJ', H.bb.vovv[vb, ob, Vb, vb], R.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('bNEf,aEfINJ->abIJ', H.bb.vovv[vb, Ob, Vb, vb], R.abb.vVvOOO, optimize=True)
            - 0.5 * np.einsum('bnEF,aEFInJ->abIJ', H.bb.vovv[vb, ob, Vb, Vb], R.abb.vVVOoO, optimize=True)
            - 0.5 * np.einsum('bNEF,aEFINJ->abIJ', H.bb.vovv[vb, Ob, Vb, Vb], R.abb.vVVOOO, optimize=True)
    )
    # of terms =  12

    return dR


def build_1010(dR, R, T, H, X, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    dR.ab[Va, vb, Oa, ob] = (1.0 / 1.0) * (
            -1.0 * np.einsum('me,AebmIj->AbIj', X.a.ov[oa, va], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('mE,EAbmIj->AbIj', X.a.ov[oa, Va], T.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('Me,AebIMj->AbIj', X.a.ov[Oa, va], T.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('ME,EAbIMj->AbIj', X.a.ov[Oa, Va], T.aab.VVvOOo, optimize=True)
    )
    dR.ab[Va, vb, Oa, ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('me,AbeImj->AbIj', X.b.ov[ob, vb], T.abb.VvvOoo, optimize=True)
            + 1.0 * np.einsum('mE,AEbImj->AbIj', X.b.ov[ob, Vb], T.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('Me,AbeIjM->AbIj', X.b.ov[Ob, vb], T.abb.VvvOoO, optimize=True)
            - 1.0 * np.einsum('ME,AEbIjM->AbIj', X.b.ov[Ob, Vb], T.abb.VVvOoO, optimize=True)
    )
    dR.ab[Va, vb, Oa, ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('me,AebmIj->AbIj', H.a.ov[oa, va], R.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('mE,EAbmIj->AbIj', H.a.ov[oa, Va], R.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('Me,AebIMj->AbIj', H.a.ov[Oa, va], R.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('ME,EAbIMj->AbIj', H.a.ov[Oa, Va], R.aab.VVvOOo, optimize=True)
    )
    dR.ab[Va, vb, Oa, ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('me,AbeImj->AbIj', H.b.ov[ob, vb], R.abb.VvvOoo, optimize=True)
            + 1.0 * np.einsum('mE,AEbImj->AbIj', H.b.ov[ob, Vb], R.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('Me,AbeIjM->AbIj', H.b.ov[Ob, vb], R.abb.VvvOoO, optimize=True)
            - 1.0 * np.einsum('ME,AEbIjM->AbIj', H.b.ov[Ob, Vb], R.abb.VVvOoO, optimize=True)
    )
    dR.ab[Va, vb, Oa, ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('MnIf,AfbnMj->AbIj', H.aa.ooov[Oa, oa, Oa, va], R.aab.VvvoOo, optimize=True)
            - 0.5 * np.einsum('MNIf,AfbMNj->AbIj', H.aa.ooov[Oa, Oa, Oa, va], R.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('MnIF,FAbnMj->AbIj', H.aa.ooov[Oa, oa, Oa, Va], R.aab.VVvoOo, optimize=True)
            + 0.5 * np.einsum('MNIF,FAbMNj->AbIj', H.aa.ooov[Oa, Oa, Oa, Va], R.aab.VVvOOo, optimize=True)
    )
    dR.ab[Va, vb, Oa, ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('nmfj,AfbnIm->AbIj', H.ab.oovo[oa, ob, va, ob], R.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('Nmfj,AfbINm->AbIj', H.ab.oovo[Oa, ob, va, ob], R.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('nmFj,FAbnIm->AbIj', H.ab.oovo[oa, ob, Va, ob], R.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('NmFj,FAbINm->AbIj', H.ab.oovo[Oa, ob, Va, ob], R.aab.VVvOOo, optimize=True)
            + 1.0 * np.einsum('nMfj,AfbnIM->AbIj', H.ab.oovo[oa, Ob, va, ob], R.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('NMfj,AfbINM->AbIj', H.ab.oovo[Oa, Ob, va, ob], R.aab.VvvOOO, optimize=True)
            - 1.0 * np.einsum('nMFj,FAbnIM->AbIj', H.ab.oovo[oa, Ob, Va, ob], R.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('NMFj,FAbINM->AbIj', H.ab.oovo[Oa, Ob, Va, ob], R.aab.VVvOOO, optimize=True)
    )
    dR.ab[Va, vb, Oa, ob] += (1.0 / 1.0) * (
            +0.5 * np.einsum('mnjf,AbfInm->AbIj', H.bb.ooov[ob, ob, ob, vb], R.abb.VvvOoo, optimize=True)
            - 0.5 * np.einsum('mnjF,AFbInm->AbIj', H.bb.ooov[ob, ob, ob, Vb], R.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('Mnjf,AbfInM->AbIj', H.bb.ooov[Ob, ob, ob, vb], R.abb.VvvOoO, optimize=True)
            + 0.5 * np.einsum('MNjf,AbfINM->AbIj', H.bb.ooov[Ob, Ob, ob, vb], R.abb.VvvOOO, optimize=True)
            - 1.0 * np.einsum('MnjF,AFbInM->AbIj', H.bb.ooov[Ob, ob, ob, Vb], R.abb.VVvOoO, optimize=True)
            - 0.5 * np.einsum('MNjF,AFbINM->AbIj', H.bb.ooov[Ob, Ob, ob, Vb], R.abb.VVvOOO, optimize=True)
    )
    dR.ab[Va, vb, Oa, ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('mNIf,AbfmjN->AbIj', H.ab.ooov[oa, Ob, Oa, vb], R.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('mNIF,AFbmjN->AbIj', H.ab.ooov[oa, Ob, Oa, Vb], R.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('MnIf,AbfMnj->AbIj', H.ab.ooov[Oa, ob, Oa, vb], R.abb.VvvOoo, optimize=True)
            - 1.0 * np.einsum('MNIf,AbfMjN->AbIj', H.ab.ooov[Oa, Ob, Oa, vb], R.abb.VvvOoO, optimize=True)
            - 1.0 * np.einsum('MnIF,AFbMnj->AbIj', H.ab.ooov[Oa, ob, Oa, Vb], R.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('MNIF,AFbMjN->AbIj', H.ab.ooov[Oa, Ob, Oa, Vb], R.abb.VVvOoO, optimize=True)
    )
    dR.ab[Va, vb, Oa, ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('AnEf,EfbnIj->AbIj', H.aa.vovv[Va, oa, Va, va], R.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('ANEf,EfbINj->AbIj', H.aa.vovv[Va, Oa, Va, va], R.aab.VvvOOo, optimize=True)
            + 0.5 * np.einsum('AnEF,FEbnIj->AbIj', H.aa.vovv[Va, oa, Va, Va], R.aab.VVvoOo, optimize=True)
            - 0.5 * np.einsum('ANEF,FEbINj->AbIj', H.aa.vovv[Va, Oa, Va, Va], R.aab.VVvOOo, optimize=True)
    )
    dR.ab[Va, vb, Oa, ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('AneF,eFbInj->AbIj', H.ab.vovv[Va, ob, va, Vb], R.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('ANeF,eFbIjN->AbIj', H.ab.vovv[Va, Ob, va, Vb], R.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('AnEf,EbfInj->AbIj', H.ab.vovv[Va, ob, Va, vb], R.abb.VvvOoo, optimize=True)
            + 1.0 * np.einsum('ANEf,EbfIjN->AbIj', H.ab.vovv[Va, Ob, Va, vb], R.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('AnEF,EFbInj->AbIj', H.ab.vovv[Va, ob, Va, Vb], R.abb.VVvOoo, optimize=True)
            - 1.0 * np.einsum('ANEF,EFbIjN->AbIj', H.ab.vovv[Va, Ob, Va, Vb], R.abb.VVvOoO, optimize=True)
    )
    dR.ab[Va, vb, Oa, ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('nbfe,AfenIj->AbIj', H.ab.ovvv[oa, vb, va, vb], R.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('Nbfe,AfeINj->AbIj', H.ab.ovvv[Oa, vb, va, vb], R.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('nbFe,FAenIj->AbIj', H.ab.ovvv[oa, vb, Va, vb], R.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('NbFe,FAeINj->AbIj', H.ab.ovvv[Oa, vb, Va, vb], R.aab.VVvOOo, optimize=True)
            - 1.0 * np.einsum('nbfE,AfEnIj->AbIj', H.ab.ovvv[oa, vb, va, Vb], R.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('NbfE,AfEINj->AbIj', H.ab.ovvv[Oa, vb, va, Vb], R.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('nbFE,FAEnIj->AbIj', H.ab.ovvv[oa, vb, Va, Vb], R.aab.VVVoOo, optimize=True)
            - 1.0 * np.einsum('NbFE,FAEINj->AbIj', H.ab.ovvv[Oa, vb, Va, Vb], R.aab.VVVOOo, optimize=True)
    )
    dR.ab[Va, vb, Oa, ob] += (1.0 / 1.0) * (
            -0.5 * np.einsum('bnef,AefInj->AbIj', H.bb.vovv[vb, ob, vb, vb], R.abb.VvvOoo, optimize=True)
            + 0.5 * np.einsum('bNef,AefIjN->AbIj', H.bb.vovv[vb, Ob, vb, vb], R.abb.VvvOoO, optimize=True)
            - 1.0 * np.einsum('bnEf,AEfInj->AbIj', H.bb.vovv[vb, ob, Vb, vb], R.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('bNEf,AEfIjN->AbIj', H.bb.vovv[vb, Ob, Vb, vb], R.abb.VVvOoO, optimize=True)
            - 0.5 * np.einsum('bnEF,AEFInj->AbIj', H.bb.vovv[vb, ob, Vb, Vb], R.abb.VVVOoo, optimize=True)
            + 0.5 * np.einsum('bNEF,AEFIjN->AbIj', H.bb.vovv[vb, Ob, Vb, Vb], R.abb.VVVOoO, optimize=True)
    )
    # of terms =  12

    return dR


def build_0101(dR, R, T, H, X, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    dR.ab[va, Vb, oa, Ob] = (1.0 / 1.0) * (
            -1.0 * np.einsum('me,eaBimJ->aBiJ', X.a.ov[oa, va], T.aab.vvVooO, optimize=True)
            - 1.0 * np.einsum('mE,EaBimJ->aBiJ', X.a.ov[oa, Va], T.aab.VvVooO, optimize=True)
            - 1.0 * np.einsum('Me,eaBiMJ->aBiJ', X.a.ov[Oa, va], T.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('ME,EaBiMJ->aBiJ', X.a.ov[Oa, Va], T.aab.VvVoOO, optimize=True)
    )
    dR.ab[va, Vb, oa, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('me,aBeimJ->aBiJ', X.b.ov[ob, vb], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('mE,aBEimJ->aBiJ', X.b.ov[ob, Vb], T.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('Me,aBeiMJ->aBiJ', X.b.ov[Ob, vb], T.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('ME,aBEiMJ->aBiJ', X.b.ov[Ob, Vb], T.abb.vVVoOO, optimize=True)
    )
    dR.ab[va, Vb, oa, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('me,eaBimJ->aBiJ', H.a.ov[oa, va], R.aab.vvVooO, optimize=True)
            - 1.0 * np.einsum('mE,EaBimJ->aBiJ', H.a.ov[oa, Va], R.aab.VvVooO, optimize=True)
            - 1.0 * np.einsum('Me,eaBiMJ->aBiJ', H.a.ov[Oa, va], R.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('ME,EaBiMJ->aBiJ', H.a.ov[Oa, Va], R.aab.VvVoOO, optimize=True)
    )
    dR.ab[va, Vb, oa, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('me,aBeimJ->aBiJ', H.b.ov[ob, vb], R.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('mE,aBEimJ->aBiJ', H.b.ov[ob, Vb], R.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('Me,aBeiMJ->aBiJ', H.b.ov[Ob, vb], R.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('ME,aBEiMJ->aBiJ', H.b.ov[Ob, Vb], R.abb.vVVoOO, optimize=True)
    )
    dR.ab[va, Vb, oa, Ob] += (1.0 / 1.0) * (
            +0.5 * np.einsum('mnif,faBmnJ->aBiJ', H.aa.ooov[oa, oa, oa, va], R.aab.vvVooO, optimize=True)
            + 0.5 * np.einsum('mniF,FaBmnJ->aBiJ', H.aa.ooov[oa, oa, oa, Va], R.aab.VvVooO, optimize=True)
            + 1.0 * np.einsum('mNif,faBmNJ->aBiJ', H.aa.ooov[oa, Oa, oa, va], R.aab.vvVoOO, optimize=True)
            + 1.0 * np.einsum('mNiF,FaBmNJ->aBiJ', H.aa.ooov[oa, Oa, oa, Va], R.aab.VvVoOO, optimize=True)
            + 0.5 * np.einsum('MNif,faBMNJ->aBiJ', H.aa.ooov[Oa, Oa, oa, va], R.aab.vvVOOO, optimize=True)
            + 0.5 * np.einsum('MNiF,FaBMNJ->aBiJ', H.aa.ooov[Oa, Oa, oa, Va], R.aab.VvVOOO, optimize=True)
    )
    dR.ab[va, Vb, oa, Ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('nMfJ,faBinM->aBiJ', H.ab.oovo[oa, Ob, va, Ob], R.aab.vvVooO, optimize=True)
            + 1.0 * np.einsum('nMFJ,FaBinM->aBiJ', H.ab.oovo[oa, Ob, Va, Ob], R.aab.VvVooO, optimize=True)
            + 1.0 * np.einsum('NmfJ,faBiNm->aBiJ', H.ab.oovo[Oa, ob, va, Ob], R.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('NmFJ,FaBiNm->aBiJ', H.ab.oovo[Oa, ob, Va, Ob], R.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('NMfJ,faBiNM->aBiJ', H.ab.oovo[Oa, Ob, va, Ob], R.aab.vvVoOO, optimize=True)
            + 1.0 * np.einsum('NMFJ,FaBiNM->aBiJ', H.ab.oovo[Oa, Ob, Va, Ob], R.aab.VvVoOO, optimize=True)
    )
    dR.ab[va, Vb, oa, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('mNJf,aBfimN->aBiJ', H.bb.ooov[ob, Ob, Ob, vb], R.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('mNJF,aBFimN->aBiJ', H.bb.ooov[ob, Ob, Ob, Vb], R.abb.vVVooO, optimize=True)
            + 0.5 * np.einsum('MNJf,aBfiNM->aBiJ', H.bb.ooov[Ob, Ob, Ob, vb], R.abb.vVvoOO, optimize=True)
            + 0.5 * np.einsum('MNJF,aBFiNM->aBiJ', H.bb.ooov[Ob, Ob, Ob, Vb], R.abb.vVVoOO, optimize=True)
    )
    dR.ab[va, Vb, oa, Ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('mnif,aBfmnJ->aBiJ', H.ab.ooov[oa, ob, oa, vb], R.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('mniF,aBFmnJ->aBiJ', H.ab.ooov[oa, ob, oa, Vb], R.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('Mnif,aBfMnJ->aBiJ', H.ab.ooov[Oa, ob, oa, vb], R.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('MniF,aBFMnJ->aBiJ', H.ab.ooov[Oa, ob, oa, Vb], R.abb.vVVOoO, optimize=True)
            + 1.0 * np.einsum('mNif,aBfmNJ->aBiJ', H.ab.ooov[oa, Ob, oa, vb], R.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('mNiF,aBFmNJ->aBiJ', H.ab.ooov[oa, Ob, oa, Vb], R.abb.vVVoOO, optimize=True)
            + 1.0 * np.einsum('MNif,aBfMNJ->aBiJ', H.ab.ooov[Oa, Ob, oa, vb], R.abb.vVvOOO, optimize=True)
            + 1.0 * np.einsum('MNiF,aBFMNJ->aBiJ', H.ab.ooov[Oa, Ob, oa, Vb], R.abb.vVVOOO, optimize=True)
    )
    dR.ab[va, Vb, oa, Ob] += (1.0 / 1.0) * (
            -0.5 * np.einsum('anef,feBinJ->aBiJ', H.aa.vovv[va, oa, va, va], R.aab.vvVooO, optimize=True)
            - 1.0 * np.einsum('aneF,FeBinJ->aBiJ', H.aa.vovv[va, oa, va, Va], R.aab.VvVooO, optimize=True)
            - 0.5 * np.einsum('anEF,FEBinJ->aBiJ', H.aa.vovv[va, oa, Va, Va], R.aab.VVVooO, optimize=True)
            - 0.5 * np.einsum('aNef,feBiNJ->aBiJ', H.aa.vovv[va, Oa, va, va], R.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('aNeF,FeBiNJ->aBiJ', H.aa.vovv[va, Oa, va, Va], R.aab.VvVoOO, optimize=True)
            - 0.5 * np.einsum('aNEF,FEBiNJ->aBiJ', H.aa.vovv[va, Oa, Va, Va], R.aab.VVVoOO, optimize=True)
    )
    dR.ab[va, Vb, oa, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('anef,eBfinJ->aBiJ', H.ab.vovv[va, ob, va, vb], R.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('anEf,EBfinJ->aBiJ', H.ab.vovv[va, ob, Va, vb], R.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('aneF,eBFinJ->aBiJ', H.ab.vovv[va, ob, va, Vb], R.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('anEF,EBFinJ->aBiJ', H.ab.vovv[va, ob, Va, Vb], R.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('aNef,eBfiNJ->aBiJ', H.ab.vovv[va, Ob, va, vb], R.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('aNEf,EBfiNJ->aBiJ', H.ab.vovv[va, Ob, Va, vb], R.abb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('aNeF,eBFiNJ->aBiJ', H.ab.vovv[va, Ob, va, Vb], R.abb.vVVoOO, optimize=True)
            - 1.0 * np.einsum('aNEF,EBFiNJ->aBiJ', H.ab.vovv[va, Ob, Va, Vb], R.abb.VVVoOO, optimize=True)
    )
    dR.ab[va, Vb, oa, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('nBfE,faEinJ->aBiJ', H.ab.ovvv[oa, Vb, va, Vb], R.aab.vvVooO, optimize=True)
            - 1.0 * np.einsum('nBFe,FaeinJ->aBiJ', H.ab.ovvv[oa, Vb, Va, vb], R.aab.VvvooO, optimize=True)
            - 1.0 * np.einsum('nBFE,FaEinJ->aBiJ', H.ab.ovvv[oa, Vb, Va, Vb], R.aab.VvVooO, optimize=True)
            - 1.0 * np.einsum('NBfE,faEiNJ->aBiJ', H.ab.ovvv[Oa, Vb, va, Vb], R.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('NBFe,FaeiNJ->aBiJ', H.ab.ovvv[Oa, Vb, Va, vb], R.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('NBFE,FaEiNJ->aBiJ', H.ab.ovvv[Oa, Vb, Va, Vb], R.aab.VvVoOO, optimize=True)
    )
    dR.ab[va, Vb, oa, Ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('BneF,aFeinJ->aBiJ', H.bb.vovv[Vb, ob, vb, Vb], R.abb.vVvooO, optimize=True)
            - 0.5 * np.einsum('BnEF,aEFinJ->aBiJ', H.bb.vovv[Vb, ob, Vb, Vb], R.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('BNeF,aFeiNJ->aBiJ', H.bb.vovv[Vb, Ob, vb, Vb], R.abb.vVvoOO, optimize=True)
            - 0.5 * np.einsum('BNEF,aEFiNJ->aBiJ', H.bb.vovv[Vb, Ob, Vb, Vb], R.abb.vVVoOO, optimize=True)
    )
    # of terms =  12

    return dR


def build_0110(dR, R, T, H, X, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    dR.ab[va, Vb, Oa, ob] = (1.0 / 1.0) * (
            +1.0 * np.einsum('me,eaBmIj->aBIj', X.a.ov[oa, va], T.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('mE,EaBmIj->aBIj', X.a.ov[oa, Va], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('Me,eaBIMj->aBIj', X.a.ov[Oa, va], T.aab.vvVOOo, optimize=True)
            - 1.0 * np.einsum('ME,EaBIMj->aBIj', X.a.ov[Oa, Va], T.aab.VvVOOo, optimize=True)
    )
    dR.ab[va, Vb, Oa, ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('me,aBeImj->aBIj', X.b.ov[ob, vb], T.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('mE,aBEImj->aBIj', X.b.ov[ob, Vb], T.abb.vVVOoo, optimize=True)
            + 1.0 * np.einsum('Me,aBeIjM->aBIj', X.b.ov[Ob, vb], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('ME,aBEIjM->aBIj', X.b.ov[Ob, Vb], T.abb.vVVOoO, optimize=True)
    )
    dR.ab[va, Vb, Oa, ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('me,eaBmIj->aBIj', H.a.ov[oa, va], R.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('mE,EaBmIj->aBIj', H.a.ov[oa, Va], R.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('Me,eaBIMj->aBIj', H.a.ov[Oa, va], R.aab.vvVOOo, optimize=True)
            - 1.0 * np.einsum('ME,EaBIMj->aBIj', H.a.ov[Oa, Va], R.aab.VvVOOo, optimize=True)
    )
    dR.ab[va, Vb, Oa, ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('me,aBeImj->aBIj', H.b.ov[ob, vb], R.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('mE,aBEImj->aBIj', H.b.ov[ob, Vb], R.abb.vVVOoo, optimize=True)
            + 1.0 * np.einsum('Me,aBeIjM->aBIj', H.b.ov[Ob, vb], R.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('ME,aBEIjM->aBIj', H.b.ov[Ob, Vb], R.abb.vVVOoO, optimize=True)
    )
    dR.ab[va, Vb, Oa, ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('MnIf,faBnMj->aBIj', H.aa.ooov[Oa, oa, Oa, va], R.aab.vvVoOo, optimize=True)
            + 0.5 * np.einsum('MNIf,faBMNj->aBIj', H.aa.ooov[Oa, Oa, Oa, va], R.aab.vvVOOo, optimize=True)
            - 1.0 * np.einsum('MnIF,FaBnMj->aBIj', H.aa.ooov[Oa, oa, Oa, Va], R.aab.VvVoOo, optimize=True)
            + 0.5 * np.einsum('MNIF,FaBMNj->aBIj', H.aa.ooov[Oa, Oa, Oa, Va], R.aab.VvVOOo, optimize=True)
    )
    dR.ab[va, Vb, Oa, ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('nmfj,faBnIm->aBIj', H.ab.oovo[oa, ob, va, ob], R.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('Nmfj,faBINm->aBIj', H.ab.oovo[Oa, ob, va, ob], R.aab.vvVOOo, optimize=True)
            - 1.0 * np.einsum('nMfj,faBnIM->aBIj', H.ab.oovo[oa, Ob, va, ob], R.aab.vvVoOO, optimize=True)
            + 1.0 * np.einsum('NMfj,faBINM->aBIj', H.ab.oovo[Oa, Ob, va, ob], R.aab.vvVOOO, optimize=True)
            - 1.0 * np.einsum('nmFj,FaBnIm->aBIj', H.ab.oovo[oa, ob, Va, ob], R.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('NmFj,FaBINm->aBIj', H.ab.oovo[Oa, ob, Va, ob], R.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('nMFj,FaBnIM->aBIj', H.ab.oovo[oa, Ob, Va, ob], R.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('NMFj,FaBINM->aBIj', H.ab.oovo[Oa, Ob, Va, ob], R.aab.VvVOOO, optimize=True)
    )
    dR.ab[va, Vb, Oa, ob] += (1.0 / 1.0) * (
            +0.5 * np.einsum('mnjf,aBfInm->aBIj', H.bb.ooov[ob, ob, ob, vb], R.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('Mnjf,aBfInM->aBIj', H.bb.ooov[Ob, ob, ob, vb], R.abb.vVvOoO, optimize=True)
            + 0.5 * np.einsum('MNjf,aBfINM->aBIj', H.bb.ooov[Ob, Ob, ob, vb], R.abb.vVvOOO, optimize=True)
            + 0.5 * np.einsum('mnjF,aBFInm->aBIj', H.bb.ooov[ob, ob, ob, Vb], R.abb.vVVOoo, optimize=True)
            + 1.0 * np.einsum('MnjF,aBFInM->aBIj', H.bb.ooov[Ob, ob, ob, Vb], R.abb.vVVOoO, optimize=True)
            + 0.5 * np.einsum('MNjF,aBFINM->aBIj', H.bb.ooov[Ob, Ob, ob, Vb], R.abb.vVVOOO, optimize=True)
    )
    dR.ab[va, Vb, Oa, ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('mNIf,aBfmjN->aBIj', H.ab.ooov[oa, Ob, Oa, vb], R.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('MnIf,aBfMnj->aBIj', H.ab.ooov[Oa, ob, Oa, vb], R.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('MNIf,aBfMjN->aBIj', H.ab.ooov[Oa, Ob, Oa, vb], R.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('mNIF,aBFmjN->aBIj', H.ab.ooov[oa, Ob, Oa, Vb], R.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('MnIF,aBFMnj->aBIj', H.ab.ooov[Oa, ob, Oa, Vb], R.abb.vVVOoo, optimize=True)
            - 1.0 * np.einsum('MNIF,aBFMjN->aBIj', H.ab.ooov[Oa, Ob, Oa, Vb], R.abb.vVVOoO, optimize=True)
    )
    dR.ab[va, Vb, Oa, ob] += (1.0 / 1.0) * (
            +0.5 * np.einsum('anef,feBnIj->aBIj', H.aa.vovv[va, oa, va, va], R.aab.vvVoOo, optimize=True)
            - 0.5 * np.einsum('aNef,feBINj->aBIj', H.aa.vovv[va, Oa, va, va], R.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('aneF,FeBnIj->aBIj', H.aa.vovv[va, oa, va, Va], R.aab.VvVoOo, optimize=True)
            + 0.5 * np.einsum('anEF,FEBnIj->aBIj', H.aa.vovv[va, oa, Va, Va], R.aab.VVVoOo, optimize=True)
            - 1.0 * np.einsum('aNeF,FeBINj->aBIj', H.aa.vovv[va, Oa, va, Va], R.aab.VvVOOo, optimize=True)
            - 0.5 * np.einsum('aNEF,FEBINj->aBIj', H.aa.vovv[va, Oa, Va, Va], R.aab.VVVOOo, optimize=True)
    )
    dR.ab[va, Vb, Oa, ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('anef,eBfInj->aBIj', H.ab.vovv[va, ob, va, vb], R.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('anEf,EBfInj->aBIj', H.ab.vovv[va, ob, Va, vb], R.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('aNef,eBfIjN->aBIj', H.ab.vovv[va, Ob, va, vb], R.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('aNEf,EBfIjN->aBIj', H.ab.vovv[va, Ob, Va, vb], R.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('aneF,eBFInj->aBIj', H.ab.vovv[va, ob, va, Vb], R.abb.vVVOoo, optimize=True)
            - 1.0 * np.einsum('anEF,EBFInj->aBIj', H.ab.vovv[va, ob, Va, Vb], R.abb.VVVOoo, optimize=True)
            + 1.0 * np.einsum('aNeF,eBFIjN->aBIj', H.ab.vovv[va, Ob, va, Vb], R.abb.vVVOoO, optimize=True)
            + 1.0 * np.einsum('aNEF,EBFIjN->aBIj', H.ab.vovv[va, Ob, Va, Vb], R.abb.VVVOoO, optimize=True)
    )
    dR.ab[va, Vb, Oa, ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('nBfE,faEnIj->aBIj', H.ab.ovvv[oa, Vb, va, Vb], R.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('NBfE,faEINj->aBIj', H.ab.ovvv[Oa, Vb, va, Vb], R.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('nBFe,FaenIj->aBIj', H.ab.ovvv[oa, Vb, Va, vb], R.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('nBFE,FaEnIj->aBIj', H.ab.ovvv[oa, Vb, Va, Vb], R.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('NBFe,FaeINj->aBIj', H.ab.ovvv[Oa, Vb, Va, vb], R.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('NBFE,FaEINj->aBIj', H.ab.ovvv[Oa, Vb, Va, Vb], R.aab.VvVOOo, optimize=True)
    )
    dR.ab[va, Vb, Oa, ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('BneF,aFeInj->aBIj', H.bb.vovv[Vb, ob, vb, Vb], R.abb.vVvOoo, optimize=True)
            - 0.5 * np.einsum('BnEF,aEFInj->aBIj', H.bb.vovv[Vb, ob, Vb, Vb], R.abb.vVVOoo, optimize=True)
            - 1.0 * np.einsum('BNeF,aFeIjN->aBIj', H.bb.vovv[Vb, Ob, vb, Vb], R.abb.vVvOoO, optimize=True)
            + 0.5 * np.einsum('BNEF,aEFIjN->aBIj', H.bb.vovv[Vb, Ob, Vb, Vb], R.abb.vVVOoO, optimize=True)
    )
    # of terms =  12

    return dR

def build_0001(dR, R, T, H, X, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    dR.ab[va, vb, oa, Ob] = (1.0 / 1.0) * (
            -1.0 * np.einsum('mE,EabimJ->abiJ', X.a.ov[oa, Va], T.aab.VvvooO, optimize=True)
            - 1.0 * np.einsum('ME,EabiMJ->abiJ', X.a.ov[Oa, Va], T.aab.VvvoOO, optimize=True)
    )
    dR.ab[va, vb, oa, Ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('mE,aEbimJ->abiJ', X.b.ov[ob, Vb], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('ME,aEbiMJ->abiJ', X.b.ov[Ob, Vb], T.abb.vVvoOO, optimize=True)
    )
    dR.ab[va, vb, oa, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('mE,EabimJ->abiJ', H.a.ov[oa, Va], R.aab.VvvooO, optimize=True)
            - 1.0 * np.einsum('ME,EabiMJ->abiJ', H.a.ov[Oa, Va], R.aab.VvvoOO, optimize=True)
    )
    dR.ab[va, vb, oa, Ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('mE,aEbimJ->abiJ', H.b.ov[ob, Vb], R.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('ME,aEbiMJ->abiJ', H.b.ov[Ob, Vb], R.abb.vVvoOO, optimize=True)
    )
    dR.ab[va, vb, oa, Ob] += (1.0 / 1.0) * (
            +0.5 * np.einsum('mniF,FabmnJ->abiJ', H.aa.ooov[oa, oa, oa, Va], R.aab.VvvooO, optimize=True)
            - 1.0 * np.einsum('MniF,FabnMJ->abiJ', H.aa.ooov[Oa, oa, oa, Va], R.aab.VvvoOO, optimize=True)
            + 0.5 * np.einsum('MNiF,FabMNJ->abiJ', H.aa.ooov[Oa, Oa, oa, Va], R.aab.VvvOOO, optimize=True)
    )
    dR.ab[va, vb, oa, Ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('NmFJ,FabiNm->abiJ', H.ab.oovo[Oa, ob, Va, Ob], R.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('nMFJ,FabinM->abiJ', H.ab.oovo[oa, Ob, Va, Ob], R.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('NMFJ,FabiNM->abiJ', H.ab.oovo[Oa, Ob, Va, Ob], R.aab.VvvoOO, optimize=True)
    )
    dR.ab[va, vb, oa, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('MnJF,aFbinM->abiJ', H.bb.ooov[Ob, ob, Ob, Vb], R.abb.vVvooO, optimize=True)
            - 0.5 * np.einsum('MNJF,aFbiNM->abiJ', H.bb.ooov[Ob, Ob, Ob, Vb], R.abb.vVvoOO, optimize=True)
    )
    dR.ab[va, vb, oa, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('mniF,aFbmnJ->abiJ', H.ab.ooov[oa, ob, oa, Vb], R.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('mNiF,aFbmNJ->abiJ', H.ab.ooov[oa, Ob, oa, Vb], R.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('MniF,aFbMnJ->abiJ', H.ab.ooov[Oa, ob, oa, Vb], R.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('MNiF,aFbMNJ->abiJ', H.ab.ooov[Oa, Ob, oa, Vb], R.abb.vVvOOO, optimize=True)
    )
    dR.ab[va, vb, oa, Ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('anEf,EfbinJ->abiJ', H.aa.vovv[va, oa, Va, va], R.aab.VvvooO, optimize=True)
            - 0.5 * np.einsum('anEF,FEbinJ->abiJ', H.aa.vovv[va, oa, Va, Va], R.aab.VVvooO, optimize=True)
            + 1.0 * np.einsum('aNEf,EfbiNJ->abiJ', H.aa.vovv[va, Oa, Va, va], R.aab.VvvoOO, optimize=True)
            - 0.5 * np.einsum('aNEF,FEbiNJ->abiJ', H.aa.vovv[va, Oa, Va, Va], R.aab.VVvoOO, optimize=True)
    )
    dR.ab[va, vb, oa, Ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('aneF,eFbinJ->abiJ', H.ab.vovv[va, ob, va, Vb], R.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('anEf,EbfinJ->abiJ', H.ab.vovv[va, ob, Va, vb], R.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('anEF,EFbinJ->abiJ', H.ab.vovv[va, ob, Va, Vb], R.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('aNeF,eFbiNJ->abiJ', H.ab.vovv[va, Ob, va, Vb], R.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('aNEf,EbfiNJ->abiJ', H.ab.vovv[va, Ob, Va, vb], R.abb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('aNEF,EFbiNJ->abiJ', H.ab.vovv[va, Ob, Va, Vb], R.abb.VVvoOO, optimize=True)
    )
    dR.ab[va, vb, oa, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('nbFe,FaeinJ->abiJ', H.ab.ovvv[oa, vb, Va, vb], R.aab.VvvooO, optimize=True)
            - 1.0 * np.einsum('NbFe,FaeiNJ->abiJ', H.ab.ovvv[Oa, vb, Va, vb], R.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('nbfE,faEinJ->abiJ', H.ab.ovvv[oa, vb, va, Vb], R.aab.vvVooO, optimize=True)
            - 1.0 * np.einsum('nbFE,FaEinJ->abiJ', H.ab.ovvv[oa, vb, Va, Vb], R.aab.VvVooO, optimize=True)
            - 1.0 * np.einsum('NbfE,faEiNJ->abiJ', H.ab.ovvv[Oa, vb, va, Vb], R.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('NbFE,FaEiNJ->abiJ', H.ab.ovvv[Oa, vb, Va, Vb], R.aab.VvVoOO, optimize=True)
    )
    dR.ab[va, vb, oa, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('bnEf,aEfinJ->abiJ', H.bb.vovv[vb, ob, Vb, vb], R.abb.vVvooO, optimize=True)
            - 0.5 * np.einsum('bnEF,aEFinJ->abiJ', H.bb.vovv[vb, ob, Vb, Vb], R.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('bNEf,aEfiNJ->abiJ', H.bb.vovv[vb, Ob, Vb, vb], R.abb.vVvoOO, optimize=True)
            - 0.5 * np.einsum('bNEF,aEFiNJ->abiJ', H.bb.vovv[vb, Ob, Vb, Vb], R.abb.vVVoOO, optimize=True)
    )
    # of terms =  12

    return dR


def build_0010(dR, R, T, H, X, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    dR.ab[va, vb, Oa, ob] = (1.0 / 1.0) * (
            +1.0 * np.einsum('mE,EabmIj->abIj', X.a.ov[oa, Va], T.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('ME,EabIMj->abIj', X.a.ov[Oa, Va], T.aab.VvvOOo, optimize=True)
    )
    dR.ab[va, vb, Oa, ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('mE,aEbImj->abIj', X.b.ov[ob, Vb], T.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('ME,aEbIjM->abIj', X.b.ov[Ob, Vb], T.abb.vVvOoO, optimize=True)
    )
    dR.ab[va, vb, Oa, ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('mE,EabmIj->abIj', H.a.ov[oa, Va], R.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('ME,EabIMj->abIj', H.a.ov[Oa, Va], R.aab.VvvOOo, optimize=True)
    )
    dR.ab[va, vb, Oa, ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('mE,aEbImj->abIj', H.b.ov[ob, Vb], R.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('ME,aEbIjM->abIj', H.b.ov[Ob, Vb], R.abb.vVvOoO, optimize=True)
    )
    dR.ab[va, vb, Oa, ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('MnIF,FabnMj->abIj', H.aa.ooov[Oa, oa, Oa, Va], R.aab.VvvoOo, optimize=True)
            + 0.5 * np.einsum('MNIF,FabMNj->abIj', H.aa.ooov[Oa, Oa, Oa, Va], R.aab.VvvOOo, optimize=True)
    )
    dR.ab[va, vb, Oa, ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('nmFj,FabnIm->abIj', H.ab.oovo[oa, ob, Va, ob], R.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('NmFj,FabINm->abIj', H.ab.oovo[Oa, ob, Va, ob], R.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('nMFj,FabnIM->abIj', H.ab.oovo[oa, Ob, Va, ob], R.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('NMFj,FabINM->abIj', H.ab.oovo[Oa, Ob, Va, ob], R.aab.VvvOOO, optimize=True)
    )
    dR.ab[va, vb, Oa, ob] += (1.0 / 1.0) * (
            -0.5 * np.einsum('mnjF,aFbInm->abIj', H.bb.ooov[ob, ob, ob, Vb], R.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('MnjF,aFbInM->abIj', H.bb.ooov[Ob, ob, ob, Vb], R.abb.vVvOoO, optimize=True)
            - 0.5 * np.einsum('MNjF,aFbINM->abIj', H.bb.ooov[Ob, Ob, ob, Vb], R.abb.vVvOOO, optimize=True)
    )
    dR.ab[va, vb, Oa, ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('mNIF,aFbmjN->abIj', H.ab.ooov[oa, Ob, Oa, Vb], R.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('MnIF,aFbMnj->abIj', H.ab.ooov[Oa, ob, Oa, Vb], R.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('MNIF,aFbMjN->abIj', H.ab.ooov[Oa, Ob, Oa, Vb], R.abb.vVvOoO, optimize=True)
    )
    dR.ab[va, vb, Oa, ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('aneF,FebnIj->abIj', H.aa.vovv[va, oa, va, Va], R.aab.VvvoOo, optimize=True)
            + 0.5 * np.einsum('anEF,FEbnIj->abIj', H.aa.vovv[va, oa, Va, Va], R.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('aNeF,FebINj->abIj', H.aa.vovv[va, Oa, va, Va], R.aab.VvvOOo, optimize=True)
            - 0.5 * np.einsum('aNEF,FEbINj->abIj', H.aa.vovv[va, Oa, Va, Va], R.aab.VVvOOo, optimize=True)
    )
    dR.ab[va, vb, Oa, ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('anEf,EbfInj->abIj', H.ab.vovv[va, ob, Va, vb], R.abb.VvvOoo, optimize=True)
            + 1.0 * np.einsum('aneF,eFbInj->abIj', H.ab.vovv[va, ob, va, Vb], R.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('anEF,EFbInj->abIj', H.ab.vovv[va, ob, Va, Vb], R.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('aNEf,EbfIjN->abIj', H.ab.vovv[va, Ob, Va, vb], R.abb.VvvOoO, optimize=True)
            - 1.0 * np.einsum('aNeF,eFbIjN->abIj', H.ab.vovv[va, Ob, va, Vb], R.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('aNEF,EFbIjN->abIj', H.ab.vovv[va, Ob, Va, Vb], R.abb.VVvOoO, optimize=True)
    )
    dR.ab[va, vb, Oa, ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('nbfE,faEnIj->abIj', H.ab.ovvv[oa, vb, va, Vb], R.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('nbFe,FaenIj->abIj', H.ab.ovvv[oa, vb, Va, vb], R.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('nbFE,FaEnIj->abIj', H.ab.ovvv[oa, vb, Va, Vb], R.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('NbfE,faEINj->abIj', H.ab.ovvv[Oa, vb, va, Vb], R.aab.vvVOOo, optimize=True)
            - 1.0 * np.einsum('NbFe,FaeINj->abIj', H.ab.ovvv[Oa, vb, Va, vb], R.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('NbFE,FaEINj->abIj', H.ab.ovvv[Oa, vb, Va, Vb], R.aab.VvVOOo, optimize=True)
    )
    dR.ab[va, vb, Oa, ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('bneF,aFeInj->abIj', H.bb.vovv[vb, ob, vb, Vb], R.abb.vVvOoo, optimize=True)
            - 0.5 * np.einsum('bnEF,aEFInj->abIj', H.bb.vovv[vb, ob, Vb, Vb], R.abb.vVVOoo, optimize=True)
            - 1.0 * np.einsum('bNeF,aFeIjN->abIj', H.bb.vovv[vb, Ob, vb, Vb], R.abb.vVvOoO, optimize=True)
            + 0.5 * np.einsum('bNEF,aEFIjN->abIj', H.bb.vovv[vb, Ob, Vb, Vb], R.abb.vVVOoO, optimize=True)
    )
    # of terms =  12

    return dR


def build_0100(dR, R, T, H, X, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    dR.ab[va, Vb, oa, ob] = (1.0 / 1.0) * (
            -1.0 * np.einsum('Me,eaBiMj->aBij', X.a.ov[Oa, va], T.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('ME,EaBiMj->aBij', X.a.ov[Oa, Va], T.aab.VvVoOo, optimize=True)
    )
    dR.ab[va, Vb, oa, ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('Me,aBeijM->aBij', X.b.ov[Ob, vb], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('ME,aBEijM->aBij', X.b.ov[Ob, Vb], T.abb.vVVooO, optimize=True)
    )
    dR.ab[va, Vb, oa, ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('Me,eaBiMj->aBij', H.a.ov[Oa, va], R.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('ME,EaBiMj->aBij', H.a.ov[Oa, Va], R.aab.VvVoOo, optimize=True)
    )
    dR.ab[va, Vb, oa, ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('Me,aBeijM->aBij', H.b.ov[Ob, vb], R.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('ME,aBEijM->aBij', H.b.ov[Ob, Vb], R.abb.vVVooO, optimize=True)
    )
    dR.ab[va, Vb, oa, ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('Mnif,faBnMj->aBij', H.aa.ooov[Oa, oa, oa, va], R.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('MniF,FaBnMj->aBij', H.aa.ooov[Oa, oa, oa, Va], R.aab.VvVoOo, optimize=True)
            + 0.5 * np.einsum('MNif,faBMNj->aBij', H.aa.ooov[Oa, Oa, oa, va], R.aab.vvVOOo, optimize=True)
            + 0.5 * np.einsum('MNiF,FaBMNj->aBij', H.aa.ooov[Oa, Oa, oa, Va], R.aab.VvVOOo, optimize=True)
    )
    dR.ab[va, Vb, oa, ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('Nmfj,faBiNm->aBij', H.ab.oovo[Oa, ob, va, ob], R.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('NmFj,FaBiNm->aBij', H.ab.oovo[Oa, ob, Va, ob], R.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('nMfj,faBinM->aBij', H.ab.oovo[oa, Ob, va, ob], R.aab.vvVooO, optimize=True)
            + 1.0 * np.einsum('nMFj,FaBinM->aBij', H.ab.oovo[oa, Ob, Va, ob], R.aab.VvVooO, optimize=True)
            + 1.0 * np.einsum('NMfj,faBiNM->aBij', H.ab.oovo[Oa, Ob, va, ob], R.aab.vvVoOO, optimize=True)
            + 1.0 * np.einsum('NMFj,FaBiNM->aBij', H.ab.oovo[Oa, Ob, Va, ob], R.aab.VvVoOO, optimize=True)
    )
    dR.ab[va, Vb, oa, ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('Mnjf,aBfinM->aBij', H.bb.ooov[Ob, ob, ob, vb], R.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('MnjF,aBFinM->aBij', H.bb.ooov[Ob, ob, ob, Vb], R.abb.vVVooO, optimize=True)
            + 0.5 * np.einsum('MNjf,aBfiNM->aBij', H.bb.ooov[Ob, Ob, ob, vb], R.abb.vVvoOO, optimize=True)
            + 0.5 * np.einsum('MNjF,aBFiNM->aBij', H.bb.ooov[Ob, Ob, ob, Vb], R.abb.vVVoOO, optimize=True)
    )
    dR.ab[va, Vb, oa, ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('mNif,aBfmjN->aBij', H.ab.ooov[oa, Ob, oa, vb], R.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('mNiF,aBFmjN->aBij', H.ab.ooov[oa, Ob, oa, Vb], R.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('Mnif,aBfMnj->aBij', H.ab.ooov[Oa, ob, oa, vb], R.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('MniF,aBFMnj->aBij', H.ab.ooov[Oa, ob, oa, Vb], R.abb.vVVOoo, optimize=True)
            - 1.0 * np.einsum('MNif,aBfMjN->aBij', H.ab.ooov[Oa, Ob, oa, vb], R.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('MNiF,aBFMjN->aBij', H.ab.ooov[Oa, Ob, oa, Vb], R.abb.vVVOoO, optimize=True)
    )
    dR.ab[va, Vb, oa, ob] += (1.0 / 1.0) * (
            -0.5 * np.einsum('aNef,feBiNj->aBij', H.aa.vovv[va, Oa, va, va], R.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('aNEf,EfBiNj->aBij', H.aa.vovv[va, Oa, Va, va], R.aab.VvVoOo, optimize=True)
            - 0.5 * np.einsum('aNEF,FEBiNj->aBij', H.aa.vovv[va, Oa, Va, Va], R.aab.VVVoOo, optimize=True)
    )
    dR.ab[va, Vb, oa, ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('aNef,eBfijN->aBij', H.ab.vovv[va, Ob, va, vb], R.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('aNeF,eBFijN->aBij', H.ab.vovv[va, Ob, va, Vb], R.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('aNEf,EBfijN->aBij', H.ab.vovv[va, Ob, Va, vb], R.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('aNEF,EBFijN->aBij', H.ab.vovv[va, Ob, Va, Vb], R.abb.VVVooO, optimize=True)
    )
    dR.ab[va, Vb, oa, ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('NBFe,FaeiNj->aBij', H.ab.ovvv[Oa, Vb, Va, vb], R.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('NBfE,faEiNj->aBij', H.ab.ovvv[Oa, Vb, va, Vb], R.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('NBFE,FaEiNj->aBij', H.ab.ovvv[Oa, Vb, Va, Vb], R.aab.VvVoOo, optimize=True)
    )
    dR.ab[va, Vb, oa, ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('BNEf,aEfijN->aBij', H.bb.vovv[Vb, Ob, Vb, vb], R.abb.vVvooO, optimize=True)
            + 0.5 * np.einsum('BNEF,aEFijN->aBij', H.bb.vovv[Vb, Ob, Vb, Vb], R.abb.vVVooO, optimize=True)
    )
    # of terms =  12

    return dR


def build_1000(dR, R, T, H, X, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    dR.ab[Va, vb, oa, ob] = (1.0 / 1.0) * (
            +1.0 * np.einsum('Me,AebiMj->Abij', X.a.ov[Oa, va], T.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('ME,EAbiMj->Abij', X.a.ov[Oa, Va], T.aab.VVvoOo, optimize=True)
    )
    dR.ab[Va, vb, oa, ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('Me,AbeijM->Abij', X.b.ov[Ob, vb], T.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('ME,AEbijM->Abij', X.b.ov[Ob, Vb], T.abb.VVvooO, optimize=True)
    )
    dR.ab[Va, vb, oa, ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('Me,AebiMj->Abij', H.a.ov[Oa, va], R.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('ME,EAbiMj->Abij', H.a.ov[Oa, Va], R.aab.VVvoOo, optimize=True)
    )
    dR.ab[Va, vb, oa, ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('Me,AbeijM->Abij', H.b.ov[Ob, vb], R.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('ME,AEbijM->Abij', H.b.ov[Ob, Vb], R.abb.VVvooO, optimize=True)
    )
    dR.ab[Va, vb, oa, ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('mNif,AfbmNj->Abij', H.aa.ooov[oa, Oa, oa, va], R.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('mNiF,FAbmNj->Abij', H.aa.ooov[oa, Oa, oa, Va], R.aab.VVvoOo, optimize=True)
            - 0.5 * np.einsum('MNif,AfbMNj->Abij', H.aa.ooov[Oa, Oa, oa, va], R.aab.VvvOOo, optimize=True)
            + 0.5 * np.einsum('MNiF,FAbMNj->Abij', H.aa.ooov[Oa, Oa, oa, Va], R.aab.VVvOOo, optimize=True)
    )
    dR.ab[Va, vb, oa, ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('nMfj,AfbinM->Abij', H.ab.oovo[oa, Ob, va, ob], R.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('nMFj,FAbinM->Abij', H.ab.oovo[oa, Ob, Va, ob], R.aab.VVvooO, optimize=True)
            - 1.0 * np.einsum('Nmfj,AfbiNm->Abij', H.ab.oovo[Oa, ob, va, ob], R.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('NmFj,FAbiNm->Abij', H.ab.oovo[Oa, ob, Va, ob], R.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('NMfj,AfbiNM->Abij', H.ab.oovo[Oa, Ob, va, ob], R.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('NMFj,FAbiNM->Abij', H.ab.oovo[Oa, Ob, Va, ob], R.aab.VVvoOO, optimize=True)
    )
    dR.ab[Va, vb, oa, ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('mNjf,AbfimN->Abij', H.bb.ooov[ob, Ob, ob, vb], R.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('mNjF,AFbimN->Abij', H.bb.ooov[ob, Ob, ob, Vb], R.abb.VVvooO, optimize=True)
            + 0.5 * np.einsum('MNjf,AbfiNM->Abij', H.bb.ooov[Ob, Ob, ob, vb], R.abb.VvvoOO, optimize=True)
            - 0.5 * np.einsum('MNjF,AFbiNM->Abij', H.bb.ooov[Ob, Ob, ob, Vb], R.abb.VVvoOO, optimize=True)
    )
    dR.ab[Va, vb, oa, ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('Mnif,AbfMnj->Abij', H.ab.ooov[Oa, ob, oa, vb], R.abb.VvvOoo, optimize=True)
            - 1.0 * np.einsum('MniF,AFbMnj->Abij', H.ab.ooov[Oa, ob, oa, Vb], R.abb.VVvOoo, optimize=True)
            - 1.0 * np.einsum('mNif,AbfmjN->Abij', H.ab.ooov[oa, Ob, oa, vb], R.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('mNiF,AFbmjN->Abij', H.ab.ooov[oa, Ob, oa, Vb], R.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('MNif,AbfMjN->Abij', H.ab.ooov[Oa, Ob, oa, vb], R.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('MNiF,AFbMjN->Abij', H.ab.ooov[Oa, Ob, oa, Vb], R.abb.VVvOoO, optimize=True)
    )
    dR.ab[Va, vb, oa, ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('ANEf,EfbiNj->Abij', H.aa.vovv[Va, Oa, Va, va], R.aab.VvvoOo, optimize=True)
            - 0.5 * np.einsum('ANEF,FEbiNj->Abij', H.aa.vovv[Va, Oa, Va, Va], R.aab.VVvoOo, optimize=True)
    )
    dR.ab[Va, vb, oa, ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('ANeF,eFbijN->Abij', H.ab.vovv[Va, Ob, va, Vb], R.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('ANEf,EbfijN->Abij', H.ab.vovv[Va, Ob, Va, vb], R.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('ANEF,EFbijN->Abij', H.ab.vovv[Va, Ob, Va, Vb], R.abb.VVvooO, optimize=True)
    )
    dR.ab[Va, vb, oa, ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('Nbfe,AfeiNj->Abij', H.ab.ovvv[Oa, vb, va, vb], R.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('NbFe,FAeiNj->Abij', H.ab.ovvv[Oa, vb, Va, vb], R.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('NbfE,AfEiNj->Abij', H.ab.ovvv[Oa, vb, va, Vb], R.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('NbFE,FAEiNj->Abij', H.ab.ovvv[Oa, vb, Va, Vb], R.aab.VVVoOo, optimize=True)
    )
    dR.ab[Va, vb, oa, ob] += (1.0 / 1.0) * (
            +0.5 * np.einsum('bNef,AefijN->Abij', H.bb.vovv[vb, Ob, vb, vb], R.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('bNEf,AEfijN->Abij', H.bb.vovv[vb, Ob, Vb, vb], R.abb.VVvooO, optimize=True)
            + 0.5 * np.einsum('bNEF,AEFijN->Abij', H.bb.vovv[vb, Ob, Vb, Vb], R.abb.VVVooO, optimize=True)
    )
    # of terms =  12

    return dR


def build_0000(dR, R, T, H, X, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    dR.ab[va, vb, oa, ob] = (1.0 / 1.0) * (
            -1.0 * np.einsum('ME,EabiMj->abij', X.a.ov[Oa, Va], T.aab.VvvoOo, optimize=True)
    )
    dR.ab[va, vb, oa, ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('ME,aEbijM->abij', X.b.ov[Ob, Vb], T.abb.vVvooO, optimize=True)
    )
    dR.ab[va, vb, oa, ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('ME,EabiMj->abij', H.a.ov[Oa, Va], R.aab.VvvoOo, optimize=True)
    )
    dR.ab[va, vb, oa, ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('ME,aEbijM->abij', H.b.ov[Ob, Vb], R.abb.vVvooO, optimize=True)
    )
    dR.ab[va, vb, oa, ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('MniF,FabnMj->abij', H.aa.ooov[Oa, oa, oa, Va], R.aab.VvvoOo, optimize=True)
            + 0.5 * np.einsum('MNiF,FabMNj->abij', H.aa.ooov[Oa, Oa, oa, Va], R.aab.VvvOOo, optimize=True)
    )
    dR.ab[va, vb, oa, ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('NmFj,FabiNm->abij', H.ab.oovo[Oa, ob, Va, ob], R.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('nMFj,FabinM->abij', H.ab.oovo[oa, Ob, Va, ob], R.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('NMFj,FabiNM->abij', H.ab.oovo[Oa, Ob, Va, ob], R.aab.VvvoOO, optimize=True)
    )
    dR.ab[va, vb, oa, ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('MnjF,aFbinM->abij', H.bb.ooov[Ob, ob, ob, Vb], R.abb.vVvooO, optimize=True)
            - 0.5 * np.einsum('MNjF,aFbiNM->abij', H.bb.ooov[Ob, Ob, ob, Vb], R.abb.vVvoOO, optimize=True)
    )
    dR.ab[va, vb, oa, ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('mNiF,aFbmjN->abij', H.ab.ooov[oa, Ob, oa, Vb], R.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('MniF,aFbMnj->abij', H.ab.ooov[Oa, ob, oa, Vb], R.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('MNiF,aFbMjN->abij', H.ab.ooov[Oa, Ob, oa, Vb], R.abb.vVvOoO, optimize=True)
    )
    dR.ab[va, vb, oa, ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('aNEf,EfbiNj->abij', H.aa.vovv[va, Oa, Va, va], R.aab.VvvoOo, optimize=True)
            - 0.5 * np.einsum('aNEF,FEbiNj->abij', H.aa.vovv[va, Oa, Va, Va], R.aab.VVvoOo, optimize=True)
    )
    dR.ab[va, vb, oa, ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('aNeF,eFbijN->abij', H.ab.vovv[va, Ob, va, Vb], R.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('aNEf,EbfijN->abij', H.ab.vovv[va, Ob, Va, vb], R.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('aNEF,EFbijN->abij', H.ab.vovv[va, Ob, Va, Vb], R.abb.VVvooO, optimize=True)
    )
    dR.ab[va, vb, oa, ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('NbFe,FaeiNj->abij', H.ab.ovvv[Oa, vb, Va, vb], R.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('NbfE,faEiNj->abij', H.ab.ovvv[Oa, vb, va, Vb], R.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('NbFE,FaEiNj->abij', H.ab.ovvv[Oa, vb, Va, Vb], R.aab.VvVoOo, optimize=True)
    )
    dR.ab[va, vb, oa, ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('bNEf,aEfijN->abij', H.bb.vovv[vb, Ob, Vb, vb], R.abb.vVvooO, optimize=True)
            + 0.5 * np.einsum('bNEF,aEFijN->abij', H.bb.vovv[vb, Ob, Vb, Vb], R.abb.vVVooO, optimize=True)
    )

    return dR
