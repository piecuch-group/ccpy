import numpy as np

from ccpy.utilities.active_space import get_active_slices

from ccpy.lib.core import cc_active_loops

def build_ccsd(T, dT, H, X):
    """
    Calculate CCSD parts of the projection <i~a~|(H_N e^(T1+T2+T3))_C|0>.
    """
    dT.b = -np.einsum("mi,am->ai", X.b.oo, T.b, optimize=True)
    dT.b += np.einsum("ae,ei->ai", X.b.vv, T.b, optimize=True)
    dT.b += np.einsum("anif,fn->ai", H.bb.voov, T.b, optimize=True)
    dT.b += np.einsum("nafi,fn->ai", H.ab.ovvo, T.a, optimize=True)
    dT.b += np.einsum("me,eami->ai", X.a.ov, T.ab, optimize=True)
    dT.b += np.einsum("me,aeim->ai", X.b.ov, T.bb, optimize=True)
    dT.b -= 0.5 * np.einsum("mnif,afmn->ai", H.bb.ooov, T.bb, optimize=True)
    dT.b -= np.einsum("nmfi,fanm->ai", H.ab.oovo, T.ab, optimize=True)
    dT.b += 0.5 * np.einsum("anef,efin->ai", H.bb.vovv, T.bb, optimize=True)
    dT.b += np.einsum("nafe,feni->ai", H.ab.ovvv, T.ab, optimize=True)
    dT.b += H.b.vo
    return dT

def build_11(T, dT, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dT.b[Vb, Ob] += (1.0 / 1.0) * (
            -0.25 * np.einsum('mnef,AfemnI->AI', H.bb.oovv[ob, ob, vb, vb], T.bbb.VvvooO, optimize=True)
            - 0.5 * np.einsum('mnEf,EAfmnI->AI', H.bb.oovv[ob, ob, Vb, vb], T.bbb.VVvooO, optimize=True)
            - 0.25 * np.einsum('mnEF,FEAmnI->AI', H.bb.oovv[ob, ob, Vb, Vb], T.bbb.VVVooO, optimize=True)
            - 0.5 * np.einsum('Mnef,AfenIM->AI', H.bb.oovv[Ob, ob, vb, vb], T.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('MnEf,EAfnIM->AI', H.bb.oovv[Ob, ob, Vb, vb], T.bbb.VVvoOO, optimize=True)
            - 0.5 * np.einsum('MnEF,FEAnIM->AI', H.bb.oovv[Ob, ob, Vb, Vb], T.bbb.VVVoOO, optimize=True)
            - 0.25 * np.einsum('MNef,AfeIMN->AI', H.bb.oovv[Ob, Ob, vb, vb], T.bbb.VvvOOO, optimize=True)
            - 0.5 * np.einsum('MNEf,EAfIMN->AI', H.bb.oovv[Ob, Ob, Vb, vb], T.bbb.VVvOOO, optimize=True)
            - 0.25 * np.einsum('MNEF,FEAIMN->AI', H.bb.oovv[Ob, Ob, Vb, Vb], T.bbb.VVVOOO, optimize=True)
    )
    dT.b[Vb, Ob] += (1.0 / 1.0) * (
            -0.25 * np.einsum('mnef,feAmnI->AI', H.aa.oovv[oa, oa, va, va], T.aab.vvVooO, optimize=True)
            + 0.5 * np.einsum('mnEf,EfAmnI->AI', H.aa.oovv[oa, oa, Va, va], T.aab.VvVooO, optimize=True)
            - 0.25 * np.einsum('mnEF,FEAmnI->AI', H.aa.oovv[oa, oa, Va, Va], T.aab.VVVooO, optimize=True)
            + 0.5 * np.einsum('Mnef,feAnMI->AI', H.aa.oovv[Oa, oa, va, va], T.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('MnEf,EfAnMI->AI', H.aa.oovv[Oa, oa, Va, va], T.aab.VvVoOO, optimize=True)
            + 0.5 * np.einsum('MnEF,FEAnMI->AI', H.aa.oovv[Oa, oa, Va, Va], T.aab.VVVoOO, optimize=True)
            - 0.25 * np.einsum('MNef,feAMNI->AI', H.aa.oovv[Oa, Oa, va, va], T.aab.vvVOOO, optimize=True)
            + 0.5 * np.einsum('MNEf,EfAMNI->AI', H.aa.oovv[Oa, Oa, Va, va], T.aab.VvVOOO, optimize=True)
            - 0.25 * np.einsum('MNEF,FEAMNI->AI', H.aa.oovv[Oa, Oa, Va, Va], T.aab.VVVOOO, optimize=True)
    )
    dT.b[Vb, Ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('mnef,eAfmnI->AI', H.ab.oovv[oa, ob, va, vb], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('mneF,eAFmnI->AI', H.ab.oovv[oa, ob, va, Vb], T.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('mnEf,EAfmnI->AI', H.ab.oovv[oa, ob, Va, vb], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('mnEF,EAFmnI->AI', H.ab.oovv[oa, ob, Va, Vb], T.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('mNef,eAfmNI->AI', H.ab.oovv[oa, Ob, va, vb], T.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('mNeF,eAFmNI->AI', H.ab.oovv[oa, Ob, va, Vb], T.abb.vVVoOO, optimize=True)
            - 1.0 * np.einsum('mNEf,EAfmNI->AI', H.ab.oovv[oa, Ob, Va, vb], T.abb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('mNEF,EAFmNI->AI', H.ab.oovv[oa, Ob, Va, Vb], T.abb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('Mnef,eAfMnI->AI', H.ab.oovv[Oa, ob, va, vb], T.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('MneF,eAFMnI->AI', H.ab.oovv[Oa, ob, va, Vb], T.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('MnEf,EAfMnI->AI', H.ab.oovv[Oa, ob, Va, vb], T.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('MnEF,EAFMnI->AI', H.ab.oovv[Oa, ob, Va, Vb], T.abb.VVVOoO, optimize=True)
            - 1.0 * np.einsum('MNef,eAfMNI->AI', H.ab.oovv[Oa, Ob, va, vb], T.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('MNeF,eAFMNI->AI', H.ab.oovv[Oa, Ob, va, Vb], T.abb.vVVOOO, optimize=True)
            - 1.0 * np.einsum('MNEf,EAfMNI->AI', H.ab.oovv[Oa, Ob, Va, vb], T.abb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('MNEF,EAFMNI->AI', H.ab.oovv[Oa, Ob, Va, Vb], T.abb.VVVOOO, optimize=True)
    )

    return dT

def build_10(T, dT, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dT.b[Vb, ob] += (1.0 / 1.0) * (
            -0.5 * np.einsum('mNef,AfeimN->Ai', H.bb.oovv[ob, Ob, vb, vb], T.bbb.VvvooO, optimize=True)
            - 0.25 * np.einsum('MNef,AfeiMN->Ai', H.bb.oovv[Ob, Ob, vb, vb], T.bbb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('mNeF,FAeimN->Ai', H.bb.oovv[ob, Ob, vb, Vb], T.bbb.VVvooO, optimize=True)
            - 0.5 * np.einsum('mNEF,FEAimN->Ai', H.bb.oovv[ob, Ob, Vb, Vb], T.bbb.VVVooO, optimize=True)
            + 0.5 * np.einsum('MNeF,FAeiMN->Ai', H.bb.oovv[Ob, Ob, vb, Vb], T.bbb.VVvoOO, optimize=True)
            - 0.25 * np.einsum('MNEF,FEAiMN->Ai', H.bb.oovv[Ob, Ob, Vb, Vb], T.bbb.VVVoOO, optimize=True)
    )
    dT.b[Vb, ob] += (1.0 / 1.0) * (
            -0.5 * np.einsum('mNef,feAmNi->Ai', H.aa.oovv[oa, Oa, va, va], T.aab.vvVoOo, optimize=True)
            - 0.25 * np.einsum('MNef,feAMNi->Ai', H.aa.oovv[Oa, Oa, va, va], T.aab.vvVOOo, optimize=True)
            - 1.0 * np.einsum('mNeF,FeAmNi->Ai', H.aa.oovv[oa, Oa, va, Va], T.aab.VvVoOo, optimize=True)
            - 0.5 * np.einsum('mNEF,FEAmNi->Ai', H.aa.oovv[oa, Oa, Va, Va], T.aab.VVVoOo, optimize=True)
            - 0.5 * np.einsum('MNeF,FeAMNi->Ai', H.aa.oovv[Oa, Oa, va, Va], T.aab.VvVOOo, optimize=True)
            - 0.25 * np.einsum('MNEF,FEAMNi->Ai', H.aa.oovv[Oa, Oa, Va, Va], T.aab.VVVOOo, optimize=True)
    )
    dT.b[Vb, ob] += (1.0 / 1.0) * (
            -1.0 * np.einsum('Mnef,eAfMni->Ai', H.ab.oovv[Oa, ob, va, vb], T.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('MnEf,EAfMni->Ai', H.ab.oovv[Oa, ob, Va, vb], T.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('mNef,eAfmiN->Ai', H.ab.oovv[oa, Ob, va, vb], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('mNEf,EAfmiN->Ai', H.ab.oovv[oa, Ob, Va, vb], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('MNef,eAfMiN->Ai', H.ab.oovv[Oa, Ob, va, vb], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('MNEf,EAfMiN->Ai', H.ab.oovv[Oa, Ob, Va, vb], T.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('MneF,eAFMni->Ai', H.ab.oovv[Oa, ob, va, Vb], T.abb.vVVOoo, optimize=True)
            - 1.0 * np.einsum('MnEF,EAFMni->Ai', H.ab.oovv[Oa, ob, Va, Vb], T.abb.VVVOoo, optimize=True)
            + 1.0 * np.einsum('mNeF,eAFmiN->Ai', H.ab.oovv[oa, Ob, va, Vb], T.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('mNEF,EAFmiN->Ai', H.ab.oovv[oa, Ob, Va, Vb], T.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('MNeF,eAFMiN->Ai', H.ab.oovv[Oa, Ob, va, Vb], T.abb.vVVOoO, optimize=True)
            + 1.0 * np.einsum('MNEF,EAFMiN->Ai', H.ab.oovv[Oa, Ob, Va, Vb], T.abb.VVVOoO, optimize=True)
    )

    return dT

def build_01(T, dT, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dT.b[vb, Ob] += (1.0 / 1.0) * (
            +0.5 * np.einsum('mnEf,EfamnI->aI', H.bb.oovv[ob, ob, Vb, vb], T.bbb.VvvooO, optimize=True)
            - 0.25 * np.einsum('mnEF,FEamnI->aI', H.bb.oovv[ob, ob, Vb, Vb], T.bbb.VVvooO, optimize=True)
            + 1.0 * np.einsum('MnEf,EfanIM->aI', H.bb.oovv[Ob, ob, Vb, vb], T.bbb.VvvoOO, optimize=True)
            - 0.5 * np.einsum('MnEF,FEanIM->aI', H.bb.oovv[Ob, ob, Vb, Vb], T.bbb.VVvoOO, optimize=True)
            + 0.5 * np.einsum('MNEf,EfaIMN->aI', H.bb.oovv[Ob, Ob, Vb, vb], T.bbb.VvvOOO, optimize=True)
            - 0.25 * np.einsum('MNEF,FEaIMN->aI', H.bb.oovv[Ob, Ob, Vb, Vb], T.bbb.VVvOOO, optimize=True)
    )
    dT.b[vb, Ob] += (1.0 / 1.0) * (
            +0.5 * np.einsum('mnEf,EfamnI->aI', H.aa.oovv[oa, oa, Va, va], T.aab.VvvooO, optimize=True)
            - 0.25 * np.einsum('mnEF,FEamnI->aI', H.aa.oovv[oa, oa, Va, Va], T.aab.VVvooO, optimize=True)
            - 1.0 * np.einsum('MnEf,EfanMI->aI', H.aa.oovv[Oa, oa, Va, va], T.aab.VvvoOO, optimize=True)
            + 0.5 * np.einsum('MnEF,FEanMI->aI', H.aa.oovv[Oa, oa, Va, Va], T.aab.VVvoOO, optimize=True)
            + 0.5 * np.einsum('MNEf,EfaMNI->aI', H.aa.oovv[Oa, Oa, Va, va], T.aab.VvvOOO, optimize=True)
            - 0.25 * np.einsum('MNEF,FEaMNI->aI', H.aa.oovv[Oa, Oa, Va, Va], T.aab.VVvOOO, optimize=True)
    )
    dT.b[vb, Ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('mneF,eFamnI->aI', H.ab.oovv[oa, ob, va, Vb], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('mnEf,EafmnI->aI', H.ab.oovv[oa, ob, Va, vb], T.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('mnEF,EFamnI->aI', H.ab.oovv[oa, ob, Va, Vb], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('mNeF,eFamNI->aI', H.ab.oovv[oa, Ob, va, Vb], T.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('mNEf,EafmNI->aI', H.ab.oovv[oa, Ob, Va, vb], T.abb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('mNEF,EFamNI->aI', H.ab.oovv[oa, Ob, Va, Vb], T.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('MneF,eFaMnI->aI', H.ab.oovv[Oa, ob, va, Vb], T.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('MnEf,EafMnI->aI', H.ab.oovv[Oa, ob, Va, vb], T.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('MnEF,EFaMnI->aI', H.ab.oovv[Oa, ob, Va, Vb], T.abb.VVvOoO, optimize=True)
            + 1.0 * np.einsum('MNeF,eFaMNI->aI', H.ab.oovv[Oa, Ob, va, Vb], T.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('MNEf,EafMNI->aI', H.ab.oovv[Oa, Ob, Va, vb], T.abb.VvvOOO, optimize=True)
            + 1.0 * np.einsum('MNEF,EFaMNI->aI', H.ab.oovv[Oa, Ob, Va, Vb], T.abb.VVvOOO, optimize=True)
    )

    return dT

def build_00(T, dT, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dT.b[vb, ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('mNEf,EfaimN->ai', H.bb.oovv[ob, Ob, Vb, vb], T.bbb.VvvooO, optimize=True)
            + 0.5 * np.einsum('MNEf,EfaiMN->ai', H.bb.oovv[Ob, Ob, Vb, vb], T.bbb.VvvoOO, optimize=True)
            - 0.5 * np.einsum('mNEF,FEaimN->ai', H.bb.oovv[ob, Ob, Vb, Vb], T.bbb.VVvooO, optimize=True)
            - 0.25 * np.einsum('MNEF,FEaiMN->ai', H.bb.oovv[Ob, Ob, Vb, Vb], T.bbb.VVvoOO, optimize=True)
    )
    dT.b[vb, ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('mNEf,EfamNi->ai', H.aa.oovv[oa, Oa, Va, va], T.aab.VvvoOo, optimize=True)
            + 0.5 * np.einsum('MNEf,EfaMNi->ai', H.aa.oovv[Oa, Oa, Va, va], T.aab.VvvOOo, optimize=True)
            - 0.5 * np.einsum('mNEF,FEamNi->ai', H.aa.oovv[oa, Oa, Va, Va], T.aab.VVvoOo, optimize=True)
            - 0.25 * np.einsum('MNEF,FEaMNi->ai', H.aa.oovv[Oa, Oa, Va, Va], T.aab.VVvOOo, optimize=True)
    )
    dT.b[vb, ob] += (1.0 / 1.0) * (
            +1.0 * np.einsum('MneF,eFaMni->ai', H.ab.oovv[Oa, ob, va, Vb], T.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('MnEf,EafMni->ai', H.ab.oovv[Oa, ob, Va, vb], T.abb.VvvOoo, optimize=True)
            + 1.0 * np.einsum('MnEF,EFaMni->ai', H.ab.oovv[Oa, ob, Va, Vb], T.abb.VVvOoo, optimize=True)
            - 1.0 * np.einsum('mNeF,eFamiN->ai', H.ab.oovv[oa, Ob, va, Vb], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('MNeF,eFaMiN->ai', H.ab.oovv[Oa, Ob, va, Vb], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('mNEf,EafmiN->ai', H.ab.oovv[oa, Ob, Va, vb], T.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('MNEf,EafMiN->ai', H.ab.oovv[Oa, Ob, Va, vb], T.abb.VvvOoO, optimize=True)
            - 1.0 * np.einsum('mNEF,EFamiN->ai', H.ab.oovv[oa, Ob, Va, Vb], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('MNEF,EFaMiN->ai', H.ab.oovv[Oa, Ob, Va, Vb], T.abb.VVvOoO, optimize=True)
    )

    return dT

def update(T, dT, H, shift):

    T.b, dT.b = cc_active_loops.update_t1b(
        T.b,
        dT.b,
        H.b.oo,
        H.b.vv,
        shift,
    )
    return T, dT