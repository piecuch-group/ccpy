import numpy as np

from ccpy.utilities.active_space import get_active_slices

from ccpy.lib.core import cc_active_loops

def build_ccsd(T, dT, H, X):
    """
    Calcualte CCSD parts of the projection <ia|(H_N e^(T1+T2+T3))_C|0>.
    """
    dT.a = -np.einsum("mi,am->ai", X.a.oo, T.a, optimize=True)
    dT.a += np.einsum("ae,ei->ai", X.a.vv, T.a, optimize=True)
    dT.a += np.einsum("me,aeim->ai", X.a.ov, T.aa, optimize=True) # [+]
    dT.a += np.einsum("me,aeim->ai", X.b.ov, T.ab, optimize=True) # [+]
    dT.a += np.einsum("anif,fn->ai", H.aa.voov, T.a, optimize=True)
    dT.a += np.einsum("anif,fn->ai", H.ab.voov, T.b, optimize=True)
    dT.a -= 0.5 * np.einsum("mnif,afmn->ai", H.aa.ooov, T.aa, optimize=True)
    dT.a -= np.einsum("mnif,afmn->ai", H.ab.ooov, T.ab, optimize=True)
    dT.a += 0.5 * np.einsum("anef,efin->ai", H.aa.vovv, T.aa, optimize=True)
    dT.a += np.einsum("anef,efin->ai", H.ab.vovv, T.ab, optimize=True)
    dT.a += H.a.vo
    return dT

def build_11(T, dT, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dT.a[Va, Oa] += (1.0 / 1.0) * (
            -0.25 * np.einsum('mnef,AfemnI->AI', H.aa.oovv[oa, oa, va, va], T.aaa.VvvooO, optimize=True)
            - 0.5 * np.einsum('mnEf,EAfmnI->AI', H.aa.oovv[oa, oa, Va, va], T.aaa.VVvooO, optimize=True)
            - 0.25 * np.einsum('mnEF,FEAmnI->AI', H.aa.oovv[oa, oa, Va, Va], T.aaa.VVVooO, optimize=True)
            + 0.5 * np.einsum('mNef,AfemIN->AI', H.aa.oovv[oa, Oa, va, va], T.aaa.VvvoOO, optimize=True)
            - 0.25 * np.einsum('MNef,AfeIMN->AI', H.aa.oovv[Oa, Oa, va, va], T.aaa.VvvOOO, optimize=True)
            + 1.0 * np.einsum('mNEf,EAfmIN->AI', H.aa.oovv[oa, Oa, Va, va], T.aaa.VVvoOO, optimize=True)
            + 0.5 * np.einsum('mNEF,FEAmIN->AI', H.aa.oovv[oa, Oa, Va, Va], T.aaa.VVVoOO, optimize=True)
            - 0.5 * np.einsum('MNEf,EAfIMN->AI', H.aa.oovv[Oa, Oa, Va, va], T.aaa.VVvOOO, optimize=True)
            - 0.25 * np.einsum('MNEF,FEAIMN->AI', H.aa.oovv[Oa, Oa, Va, Va], T.aaa.VVVOOO, optimize=True)
    )
    dT.a[Va, Oa] += (1.0 / 1.0) * (
            -1.0 * np.einsum('mnef,AefmIn->AI', H.ab.oovv[oa, ob, va, vb], T.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('mneF,AeFmIn->AI', H.ab.oovv[oa, ob, va, Vb], T.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('Mnef,AefIMn->AI', H.ab.oovv[Oa, ob, va, vb], T.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('MneF,AeFIMn->AI', H.ab.oovv[Oa, ob, va, Vb], T.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('mnEf,EAfmIn->AI', H.ab.oovv[oa, ob, Va, vb], T.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('mnEF,EAFmIn->AI', H.ab.oovv[oa, ob, Va, Vb], T.aab.VVVoOo, optimize=True)
            - 1.0 * np.einsum('MnEf,EAfIMn->AI', H.ab.oovv[Oa, ob, Va, vb], T.aab.VVvOOo, optimize=True)
            - 1.0 * np.einsum('MnEF,EAFIMn->AI', H.ab.oovv[Oa, ob, Va, Vb], T.aab.VVVOOo, optimize=True)
            - 1.0 * np.einsum('mNef,AefmIN->AI', H.ab.oovv[oa, Ob, va, vb], T.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('mNeF,AeFmIN->AI', H.ab.oovv[oa, Ob, va, Vb], T.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('MNef,AefIMN->AI', H.ab.oovv[Oa, Ob, va, vb], T.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('MNeF,AeFIMN->AI', H.ab.oovv[Oa, Ob, va, Vb], T.aab.VvVOOO, optimize=True)
            + 1.0 * np.einsum('mNEf,EAfmIN->AI', H.ab.oovv[oa, Ob, Va, vb], T.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('mNEF,EAFmIN->AI', H.ab.oovv[oa, Ob, Va, Vb], T.aab.VVVoOO, optimize=True)
            - 1.0 * np.einsum('MNEf,EAfIMN->AI', H.ab.oovv[Oa, Ob, Va, vb], T.aab.VVvOOO, optimize=True)
            - 1.0 * np.einsum('MNEF,EAFIMN->AI', H.ab.oovv[Oa, Ob, Va, Vb], T.aab.VVVOOO, optimize=True)
    )
    dT.a[Va, Oa] += (1.0 / 1.0) * (
            -0.25 * np.einsum('mnef,AfeImn->AI', H.bb.oovv[ob, ob, vb, vb], T.abb.VvvOoo, optimize=True)
            + 0.5 * np.einsum('mnEf,AEfImn->AI', H.bb.oovv[ob, ob, Vb, vb], T.abb.VVvOoo, optimize=True)
            - 0.25 * np.einsum('mnEF,AFEImn->AI', H.bb.oovv[ob, ob, Vb, Vb], T.abb.VVVOoo, optimize=True)
            - 0.5 * np.einsum('mNef,AfeImN->AI', H.bb.oovv[ob, Ob, vb, vb], T.abb.VvvOoO, optimize=True)
            - 0.25 * np.einsum('MNef,AfeIMN->AI', H.bb.oovv[Ob, Ob, vb, vb], T.abb.VvvOOO, optimize=True)
            + 1.0 * np.einsum('mNEf,AEfImN->AI', H.bb.oovv[ob, Ob, Vb, vb], T.abb.VVvOoO, optimize=True)
            - 0.5 * np.einsum('mNEF,AFEImN->AI', H.bb.oovv[ob, Ob, Vb, Vb], T.abb.VVVOoO, optimize=True)
            + 0.5 * np.einsum('MNEf,AEfIMN->AI', H.bb.oovv[Ob, Ob, Vb, vb], T.abb.VVvOOO, optimize=True)
            - 0.25 * np.einsum('MNEF,AFEIMN->AI', H.bb.oovv[Ob, Ob, Vb, Vb], T.abb.VVVOOO, optimize=True)
    )

    return dT

def build_10(T, dT, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dT.a[Va, oa] += (1.0 / 1.0) * (
            -0.5 * np.einsum('mNef,AfeimN->Ai', H.aa.oovv[oa, Oa, va, va], T.aaa.VvvooO, optimize=True)
            - 0.25 * np.einsum('MNef,AfeiMN->Ai', H.aa.oovv[Oa, Oa, va, va], T.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('mNEf,EAfimN->Ai', H.aa.oovv[oa, Oa, Va, va], T.aaa.VVvooO, optimize=True)
            - 0.5 * np.einsum('mNEF,FEAimN->Ai', H.aa.oovv[oa, Oa, Va, Va], T.aaa.VVVooO, optimize=True)
            - 0.5 * np.einsum('MNEf,EAfiMN->Ai', H.aa.oovv[Oa, Oa, Va, va], T.aaa.VVvoOO, optimize=True)
            - 0.25 * np.einsum('MNEF,FEAiMN->Ai', H.aa.oovv[Oa, Oa, Va, Va], T.aaa.VVVoOO, optimize=True)
    )
    dT.a[Va, oa] += (1.0 / 1.0) * (
            +1.0 * np.einsum('Mnef,AefiMn->Ai', H.ab.oovv[Oa, ob, va, vb], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('MneF,AeFiMn->Ai', H.ab.oovv[Oa, ob, va, Vb], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('MnEf,EAfiMn->Ai', H.ab.oovv[Oa, ob, Va, vb], T.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('MnEF,EAFiMn->Ai', H.ab.oovv[Oa, ob, Va, Vb], T.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('mNef,AefimN->Ai', H.ab.oovv[oa, Ob, va, vb], T.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('mNeF,AeFimN->Ai', H.ab.oovv[oa, Ob, va, Vb], T.aab.VvVooO, optimize=True)
            + 1.0 * np.einsum('MNef,AefiMN->Ai', H.ab.oovv[Oa, Ob, va, vb], T.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('MNeF,AeFiMN->Ai', H.ab.oovv[Oa, Ob, va, Vb], T.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('mNEf,EAfimN->Ai', H.ab.oovv[oa, Ob, Va, vb], T.aab.VVvooO, optimize=True)
            - 1.0 * np.einsum('mNEF,EAFimN->Ai', H.ab.oovv[oa, Ob, Va, Vb], T.aab.VVVooO, optimize=True)
            - 1.0 * np.einsum('MNEf,EAfiMN->Ai', H.ab.oovv[Oa, Ob, Va, vb], T.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('MNEF,EAFiMN->Ai', H.ab.oovv[Oa, Ob, Va, Vb], T.aab.VVVoOO, optimize=True)
    )
    dT.a[Va, oa] += (1.0 / 1.0) * (
            -0.5 * np.einsum('mNef,AfeimN->Ai', H.bb.oovv[ob, Ob, vb, vb], T.abb.VvvooO, optimize=True)
            - 0.25 * np.einsum('MNef,AfeiMN->Ai', H.bb.oovv[Ob, Ob, vb, vb], T.abb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('mNEf,AEfimN->Ai', H.bb.oovv[ob, Ob, Vb, vb], T.abb.VVvooO, optimize=True)
            - 0.5 * np.einsum('mNEF,AFEimN->Ai', H.bb.oovv[ob, Ob, Vb, Vb], T.abb.VVVooO, optimize=True)
            + 0.5 * np.einsum('MNEf,AEfiMN->Ai', H.bb.oovv[Ob, Ob, Vb, vb], T.abb.VVvoOO, optimize=True)
            - 0.25 * np.einsum('MNEF,AFEiMN->Ai', H.bb.oovv[Ob, Ob, Vb, Vb], T.abb.VVVoOO, optimize=True)
    )

    return dT

def build_01(T, dT, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dT.a[va, Oa] += (1.0 / 1.0) * (
            -0.5 * np.einsum('mneF,FeamnI->aI', H.aa.oovv[oa, oa, va, Va], T.aaa.VvvooO, optimize=True)
            - 0.25 * np.einsum('mnEF,FEamnI->aI', H.aa.oovv[oa, oa, Va, Va], T.aaa.VVvooO, optimize=True)
            - 1.0 * np.einsum('MneF,FeanIM->aI', H.aa.oovv[Oa, oa, va, Va], T.aaa.VvvoOO, optimize=True)
            - 0.5 * np.einsum('MnEF,FEanIM->aI', H.aa.oovv[Oa, oa, Va, Va], T.aaa.VVvoOO, optimize=True)
            - 0.5 * np.einsum('MNeF,FeaIMN->aI', H.aa.oovv[Oa, Oa, va, Va], T.aaa.VvvOOO, optimize=True)
            - 0.25 * np.einsum('MNEF,FEaIMN->aI', H.aa.oovv[Oa, Oa, Va, Va], T.aaa.VVvOOO, optimize=True)
    )
    dT.a[va, Oa] += (1.0 / 1.0) * (
            +1.0 * np.einsum('mnEf,EafmIn->aI', H.ab.oovv[oa, ob, Va, vb], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('mNEf,EafmIN->aI', H.ab.oovv[oa, Ob, Va, vb], T.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('mneF,eaFmIn->aI', H.ab.oovv[oa, ob, va, Vb], T.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('mnEF,EaFmIn->aI', H.ab.oovv[oa, ob, Va, Vb], T.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('mNeF,eaFmIN->aI', H.ab.oovv[oa, Ob, va, Vb], T.aab.vvVoOO, optimize=True)
            + 1.0 * np.einsum('mNEF,EaFmIN->aI', H.ab.oovv[oa, Ob, Va, Vb], T.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('MnEf,EafIMn->aI', H.ab.oovv[Oa, ob, Va, vb], T.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('MNEf,EafIMN->aI', H.ab.oovv[Oa, Ob, Va, vb], T.aab.VvvOOO, optimize=True)
            - 1.0 * np.einsum('MneF,eaFIMn->aI', H.ab.oovv[Oa, ob, va, Vb], T.aab.vvVOOo, optimize=True)
            - 1.0 * np.einsum('MnEF,EaFIMn->aI', H.ab.oovv[Oa, ob, Va, Vb], T.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('MNeF,eaFIMN->aI', H.ab.oovv[Oa, Ob, va, Vb], T.aab.vvVOOO, optimize=True)
            - 1.0 * np.einsum('MNEF,EaFIMN->aI', H.ab.oovv[Oa, Ob, Va, Vb], T.aab.VvVOOO, optimize=True)
    )
    dT.a[va, Oa] += (1.0 / 1.0) * (
            -0.5 * np.einsum('mneF,aFeImn->aI', H.bb.oovv[ob, ob, vb, Vb], T.abb.vVvOoo, optimize=True)
            - 0.25 * np.einsum('mnEF,aFEImn->aI', H.bb.oovv[ob, ob, Vb, Vb], T.abb.vVVOoo, optimize=True)
            + 1.0 * np.einsum('MneF,aFeInM->aI', H.bb.oovv[Ob, ob, vb, Vb], T.abb.vVvOoO, optimize=True)
            + 0.5 * np.einsum('MnEF,aFEInM->aI', H.bb.oovv[Ob, ob, Vb, Vb], T.abb.vVVOoO, optimize=True)
            - 0.5 * np.einsum('MNeF,aFeIMN->aI', H.bb.oovv[Ob, Ob, vb, Vb], T.abb.vVvOOO, optimize=True)
            - 0.25 * np.einsum('MNEF,aFEIMN->aI', H.bb.oovv[Ob, Ob, Vb, Vb], T.abb.vVVOOO, optimize=True)
    )

    return dT

def build_00(T, dT, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dT.a[va, oa] += (1.0 / 1.0) * (
            +1.0 * np.einsum('mNEf,EfaimN->ai', H.aa.oovv[oa, Oa, Va, va], T.aaa.VvvooO, optimize=True)
            + 0.5 * np.einsum('MNEf,EfaiMN->ai', H.aa.oovv[Oa, Oa, Va, va], T.aaa.VvvoOO, optimize=True)
            - 0.5 * np.einsum('mNEF,FEaimN->ai', H.aa.oovv[oa, Oa, Va, Va], T.aaa.VVvooO, optimize=True)
            - 0.25 * np.einsum('MNEF,FEaiMN->ai', H.aa.oovv[Oa, Oa, Va, Va], T.aaa.VVvoOO, optimize=True)
    )
    dT.a[va, oa] += (1.0 / 1.0) * (
            -1.0 * np.einsum('MneF,eaFiMn->ai', H.ab.oovv[Oa, ob, va, Vb], T.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('mNeF,eaFimN->ai', H.ab.oovv[oa, Ob, va, Vb], T.aab.vvVooO, optimize=True)
            - 1.0 * np.einsum('MNeF,eaFiMN->ai', H.ab.oovv[Oa, Ob, va, Vb], T.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('MnEf,EafiMn->ai', H.ab.oovv[Oa, ob, Va, vb], T.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('MnEF,EaFiMn->ai', H.ab.oovv[Oa, ob, Va, Vb], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('mNEf,EafimN->ai', H.ab.oovv[oa, Ob, Va, vb], T.aab.VvvooO, optimize=True)
            - 1.0 * np.einsum('MNEf,EafiMN->ai', H.ab.oovv[Oa, Ob, Va, vb], T.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('mNEF,EaFimN->ai', H.ab.oovv[oa, Ob, Va, Vb], T.aab.VvVooO, optimize=True)
            - 1.0 * np.einsum('MNEF,EaFiMN->ai', H.ab.oovv[Oa, Ob, Va, Vb], T.aab.VvVoOO, optimize=True)
    )
    dT.a[va, oa] += (1.0 / 1.0) * (
            +1.0 * np.einsum('mNEf,aEfimN->ai', H.bb.oovv[ob, Ob, Vb, vb], T.abb.vVvooO, optimize=True)
            + 0.5 * np.einsum('MNEf,aEfiMN->ai', H.bb.oovv[Ob, Ob, Vb, vb], T.abb.vVvoOO, optimize=True)
            - 0.5 * np.einsum('mNEF,aFEimN->ai', H.bb.oovv[ob, Ob, Vb, Vb], T.abb.vVVooO, optimize=True)
            - 0.25 * np.einsum('MNEF,aFEiMN->ai', H.bb.oovv[Ob, Ob, Vb, Vb], T.abb.vVVoOO, optimize=True)
    )

    return dT

def update(T, dT, H, shift):

    T.a, dT.a = cc_active_loops.update_t1a(
        T.a,
        dT.a,
        H.a.oo,
        H.a.vv,
        shift,
    )
    return T, dT