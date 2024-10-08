import numpy as np

from ccpy.utilities.active_space import get_active_slices

from ccpy.lib.core import eomcc_active_loops

def update(R, omega, H):
    R.b = eomcc_active_loops.update_r1b(R.b, omega, H.a.oo, H.a.vv, H.b.oo, H.b.vv, 0.0)
    return R

def build(dR, R, H, system):

    x1 = build_eomccsd(R, H)  # base EOMCCSD part (separately antisymmetrized)

    # Add on T3 parts
    dR = build_11(dR, R, H, system)
    dR = build_10(dR, R, H, system)
    dR = build_01(dR, R, H, system)
    dR = build_00(dR, R, H, system)

    dR.b += x1

    return dR


def build_eomccsd(R, H):
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


def build_11(dR, R, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    dR.b[Vb, Ob] = (1.0 / 1.0) * (
            -0.25 * np.einsum('mnef,AfemnI->AI', H.bb.oovv[ob, ob, vb, vb], R.bbb.VvvooO, optimize=True)
            + 0.5 * np.einsum('mneF,FAemnI->AI', H.bb.oovv[ob, ob, vb, Vb], R.bbb.VVvooO, optimize=True)
            - 0.25 * np.einsum('mnEF,FEAmnI->AI', H.bb.oovv[ob, ob, Vb, Vb], R.bbb.VVVooO, optimize=True)
            + 0.5 * np.einsum('mNef,AfemIN->AI', H.bb.oovv[ob, Ob, vb, vb], R.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('mNeF,FAemIN->AI', H.bb.oovv[ob, Ob, vb, Vb], R.bbb.VVvoOO, optimize=True)
            + 0.5 * np.einsum('mNEF,FEAmIN->AI', H.bb.oovv[ob, Ob, Vb, Vb], R.bbb.VVVoOO, optimize=True)
            - 0.25 * np.einsum('MNef,AfeIMN->AI', H.bb.oovv[Ob, Ob, vb, vb], R.bbb.VvvOOO, optimize=True)
            + 0.5 * np.einsum('MNeF,FAeIMN->AI', H.bb.oovv[Ob, Ob, vb, Vb], R.bbb.VVvOOO, optimize=True)
            - 0.25 * np.einsum('MNEF,FEAIMN->AI', H.bb.oovv[Ob, Ob, Vb, Vb], R.bbb.VVVOOO, optimize=True)
    )
    dR.b[Vb, Ob] += (1.0 / 1.0) * (
            -0.25 * np.einsum('mnef,feAmnI->AI', H.aa.oovv[oa, oa, va, va], R.aab.vvVooO, optimize=True)
            - 0.5 * np.einsum('mneF,FeAmnI->AI', H.aa.oovv[oa, oa, va, Va], R.aab.VvVooO, optimize=True)
            - 0.25 * np.einsum('mnEF,FEAmnI->AI', H.aa.oovv[oa, oa, Va, Va], R.aab.VVVooO, optimize=True)
            - 0.5 * np.einsum('mNef,feAmNI->AI', H.aa.oovv[oa, Oa, va, va], R.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('mNeF,FeAmNI->AI', H.aa.oovv[oa, Oa, va, Va], R.aab.VvVoOO, optimize=True)
            - 0.5 * np.einsum('mNEF,FEAmNI->AI', H.aa.oovv[oa, Oa, Va, Va], R.aab.VVVoOO, optimize=True)
            - 0.25 * np.einsum('MNef,feAMNI->AI', H.aa.oovv[Oa, Oa, va, va], R.aab.vvVOOO, optimize=True)
            - 0.5 * np.einsum('MNeF,FeAMNI->AI', H.aa.oovv[Oa, Oa, va, Va], R.aab.VvVOOO, optimize=True)
            - 0.25 * np.einsum('MNEF,FEAMNI->AI', H.aa.oovv[Oa, Oa, Va, Va], R.aab.VVVOOO, optimize=True)
    )
    dR.b[Vb, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('mnef,eAfmnI->AI', H.ab.oovv[oa, ob, va, vb], R.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('mnEf,EAfmnI->AI', H.ab.oovv[oa, ob, Va, vb], R.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('mneF,eAFmnI->AI', H.ab.oovv[oa, ob, va, Vb], R.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('mnEF,EAFmnI->AI', H.ab.oovv[oa, ob, Va, Vb], R.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('Mnef,eAfMnI->AI', H.ab.oovv[Oa, ob, va, vb], R.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('MnEf,EAfMnI->AI', H.ab.oovv[Oa, ob, Va, vb], R.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('MneF,eAFMnI->AI', H.ab.oovv[Oa, ob, va, Vb], R.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('MnEF,EAFMnI->AI', H.ab.oovv[Oa, ob, Va, Vb], R.abb.VVVOoO, optimize=True)
            - 1.0 * np.einsum('mNef,eAfmNI->AI', H.ab.oovv[oa, Ob, va, vb], R.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('mNEf,EAfmNI->AI', H.ab.oovv[oa, Ob, Va, vb], R.abb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('mNeF,eAFmNI->AI', H.ab.oovv[oa, Ob, va, Vb], R.abb.vVVoOO, optimize=True)
            - 1.0 * np.einsum('mNEF,EAFmNI->AI', H.ab.oovv[oa, Ob, Va, Vb], R.abb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('MNef,eAfMNI->AI', H.ab.oovv[Oa, Ob, va, vb], R.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('MNEf,EAfMNI->AI', H.ab.oovv[Oa, Ob, Va, vb], R.abb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('MNeF,eAFMNI->AI', H.ab.oovv[Oa, Ob, va, Vb], R.abb.vVVOOO, optimize=True)
            - 1.0 * np.einsum('MNEF,EAFMNI->AI', H.ab.oovv[Oa, Ob, Va, Vb], R.abb.VVVOOO, optimize=True)
    )
    # of terms =  3

    return dR

def build_10(dR, R, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    dR.b[Vb, ob] = (1.0 / 1.0) * (
            +0.5 * np.einsum('Mnef,AfeinM->Ai', H.bb.oovv[Ob, ob, vb, vb], R.bbb.VvvooO, optimize=True)
            - 0.25 * np.einsum('MNef,AfeiMN->Ai', H.bb.oovv[Ob, Ob, vb, vb], R.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('MneF,FAeinM->Ai', H.bb.oovv[Ob, ob, vb, Vb], R.bbb.VVvooO, optimize=True)
            + 0.5 * np.einsum('MnEF,FEAinM->Ai', H.bb.oovv[Ob, ob, Vb, Vb], R.bbb.VVVooO, optimize=True)
            + 0.5 * np.einsum('MNeF,FAeiMN->Ai', H.bb.oovv[Ob, Ob, vb, Vb], R.bbb.VVvoOO, optimize=True)
            - 0.25 * np.einsum('MNEF,FEAiMN->Ai', H.bb.oovv[Ob, Ob, Vb, Vb], R.bbb.VVVoOO, optimize=True)
    )
    dR.b[Vb, ob] += (1.0 / 1.0) * (
            +0.5 * np.einsum('Mnef,feAnMi->Ai', H.aa.oovv[Oa, oa, va, va], R.aab.vvVoOo, optimize=True)
            - 0.25 * np.einsum('MNef,feAMNi->Ai', H.aa.oovv[Oa, Oa, va, va], R.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('MneF,FeAnMi->Ai', H.aa.oovv[Oa, oa, va, Va], R.aab.VvVoOo, optimize=True)
            + 0.5 * np.einsum('MnEF,FEAnMi->Ai', H.aa.oovv[Oa, oa, Va, Va], R.aab.VVVoOo, optimize=True)
            - 0.5 * np.einsum('MNeF,FeAMNi->Ai', H.aa.oovv[Oa, Oa, va, Va], R.aab.VvVOOo, optimize=True)
            - 0.25 * np.einsum('MNEF,FEAMNi->Ai', H.aa.oovv[Oa, Oa, Va, Va], R.aab.VVVOOo, optimize=True)
    )
    dR.b[Vb, ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('mNef,eAfmiN->Ai', H.ab.oovv[oa, Ob, va, vb], R.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('mNEf,EAfmiN->Ai', H.ab.oovv[oa, Ob, Va, vb], R.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('Mnef,eAfMni->Ai', H.ab.oovv[Oa, ob, va, vb], R.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('MnEf,EAfMni->Ai', H.ab.oovv[Oa, ob, Va, vb], R.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('MNef,eAfMiN->Ai', H.ab.oovv[Oa, Ob, va, vb], R.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('MNEf,EAfMiN->Ai', H.ab.oovv[Oa, Ob, Va, vb], R.abb.VVvOoO, optimize=True)
            + 1.0 * np.einsum('mNeF,eAFmiN->Ai', H.ab.oovv[oa, Ob, va, Vb], R.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('mNEF,EAFmiN->Ai', H.ab.oovv[oa, Ob, Va, Vb], R.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('MneF,eAFMni->Ai', H.ab.oovv[Oa, ob, va, Vb], R.abb.vVVOoo, optimize=True)
            - 1.0 * np.einsum('MnEF,EAFMni->Ai', H.ab.oovv[Oa, ob, Va, Vb], R.abb.VVVOoo, optimize=True)
            + 1.0 * np.einsum('MNeF,eAFMiN->Ai', H.ab.oovv[Oa, Ob, va, Vb], R.abb.vVVOoO, optimize=True)
            + 1.0 * np.einsum('MNEF,EAFMiN->Ai', H.ab.oovv[Oa, Ob, Va, Vb], R.abb.VVVOoO, optimize=True)
    )
    # of terms =  3

    return dR

def build_01(dR, R, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    dR.b[vb, Ob] = (1.0 / 1.0) * (
            -0.5 * np.einsum('mneF,FeamnI->aI', H.bb.oovv[ob, ob, vb, Vb], R.bbb.VvvooO, optimize=True)
            - 0.25 * np.einsum('mnEF,FEamnI->aI', H.bb.oovv[ob, ob, Vb, Vb], R.bbb.VVvooO, optimize=True)
            - 1.0 * np.einsum('MneF,FeanIM->aI', H.bb.oovv[Ob, ob, vb, Vb], R.bbb.VvvoOO, optimize=True)
            - 0.5 * np.einsum('MnEF,FEanIM->aI', H.bb.oovv[Ob, ob, Vb, Vb], R.bbb.VVvoOO, optimize=True)
            - 0.5 * np.einsum('MNeF,FeaIMN->aI', H.bb.oovv[Ob, Ob, vb, Vb], R.bbb.VvvOOO, optimize=True)
            - 0.25 * np.einsum('MNEF,FEaIMN->aI', H.bb.oovv[Ob, Ob, Vb, Vb], R.bbb.VVvOOO, optimize=True)
    )
    dR.b[vb, Ob] += (1.0 / 1.0) * (
            -0.5 * np.einsum('mneF,FeamnI->aI', H.aa.oovv[oa, oa, va, Va], R.aab.VvvooO, optimize=True)
            - 0.25 * np.einsum('mnEF,FEamnI->aI', H.aa.oovv[oa, oa, Va, Va], R.aab.VVvooO, optimize=True)
            + 1.0 * np.einsum('MneF,FeanMI->aI', H.aa.oovv[Oa, oa, va, Va], R.aab.VvvoOO, optimize=True)
            + 0.5 * np.einsum('MnEF,FEanMI->aI', H.aa.oovv[Oa, oa, Va, Va], R.aab.VVvoOO, optimize=True)
            - 0.5 * np.einsum('MNeF,FeaMNI->aI', H.aa.oovv[Oa, Oa, va, Va], R.aab.VvvOOO, optimize=True)
            - 0.25 * np.einsum('MNEF,FEaMNI->aI', H.aa.oovv[Oa, Oa, Va, Va], R.aab.VVvOOO, optimize=True)
    )
    dR.b[vb, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('mnEf,EafmnI->aI', H.ab.oovv[oa, ob, Va, vb], R.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('mneF,eFamnI->aI', H.ab.oovv[oa, ob, va, Vb], R.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('mnEF,EFamnI->aI', H.ab.oovv[oa, ob, Va, Vb], R.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('mNEf,EafmNI->aI', H.ab.oovv[oa, Ob, Va, vb], R.abb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('mNeF,eFamNI->aI', H.ab.oovv[oa, Ob, va, Vb], R.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('mNEF,EFamNI->aI', H.ab.oovv[oa, Ob, Va, Vb], R.abb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('MnEf,EafMnI->aI', H.ab.oovv[Oa, ob, Va, vb], R.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('MneF,eFaMnI->aI', H.ab.oovv[Oa, ob, va, Vb], R.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('MnEF,EFaMnI->aI', H.ab.oovv[Oa, ob, Va, Vb], R.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('MNEf,EafMNI->aI', H.ab.oovv[Oa, Ob, Va, vb], R.abb.VvvOOO, optimize=True)
            + 1.0 * np.einsum('MNeF,eFaMNI->aI', H.ab.oovv[Oa, Ob, va, Vb], R.abb.vVvOOO, optimize=True)
            + 1.0 * np.einsum('MNEF,EFaMNI->aI', H.ab.oovv[Oa, Ob, Va, Vb], R.abb.VVvOOO, optimize=True)
    )
    # of terms =  3

    return dR

def build_00(dR, R, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    dR.b[vb, ob] = (1.0 / 1.0) * (
            +1.0 * np.einsum('mNEf,EfaimN->ai', H.bb.oovv[ob, Ob, Vb, vb], R.bbb.VvvooO, optimize=True)
            + 0.5 * np.einsum('MNEf,EfaiMN->ai', H.bb.oovv[Ob, Ob, Vb, vb], R.bbb.VvvoOO, optimize=True)
            - 0.5 * np.einsum('mNEF,FEaimN->ai', H.bb.oovv[ob, Ob, Vb, Vb], R.bbb.VVvooO, optimize=True)
            - 0.25 * np.einsum('MNEF,FEaiMN->ai', H.bb.oovv[Ob, Ob, Vb, Vb], R.bbb.VVvoOO, optimize=True)
    )
    dR.b[vb, ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('mNEf,EfamNi->ai', H.aa.oovv[oa, Oa, Va, va], R.aab.VvvoOo, optimize=True)
            + 0.5 * np.einsum('MNEf,EfaMNi->ai', H.aa.oovv[Oa, Oa, Va, va], R.aab.VvvOOo, optimize=True)
            - 0.5 * np.einsum('mNEF,FEamNi->ai', H.aa.oovv[oa, Oa, Va, Va], R.aab.VVvoOo, optimize=True)
            - 0.25 * np.einsum('MNEF,FEaMNi->ai', H.aa.oovv[Oa, Oa, Va, Va], R.aab.VVvOOo, optimize=True)
    )
    dR.b[vb, ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('MneF,eFaMni->ai', H.ab.oovv[Oa, ob, va, Vb], R.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('mNeF,eFamiN->ai', H.ab.oovv[oa, Ob, va, Vb], R.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('MNeF,eFaMiN->ai', H.ab.oovv[Oa, Ob, va, Vb], R.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('MnEf,EafMni->ai', H.ab.oovv[Oa, ob, Va, vb], R.abb.VvvOoo, optimize=True)
            + 1.0 * np.einsum('mNEf,EafmiN->ai', H.ab.oovv[oa, Ob, Va, vb], R.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('MNEf,EafMiN->ai', H.ab.oovv[Oa, Ob, Va, vb], R.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('MnEF,EFaMni->ai', H.ab.oovv[Oa, ob, Va, Vb], R.abb.VVvOoo, optimize=True)
            - 1.0 * np.einsum('mNEF,EFamiN->ai', H.ab.oovv[oa, Ob, Va, Vb], R.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('MNEF,EFaMiN->ai', H.ab.oovv[Oa, Ob, Va, Vb], R.abb.VVvOoO, optimize=True)
    )

    return dR
