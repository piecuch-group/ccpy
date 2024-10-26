import numpy as np

from ccpy.utilities.active_space import get_active_slices

from ccpy.lib.core import eomcc_active_loops

def update(R, omega, H):
    R.a = eomcc_active_loops.update_r1a(R.a, omega, H.a.oo, H.a.vv, H.b.oo, H.b.vv, 0.0)
    return R

def build(dR, R, H, system):

    x1 = build_eomccsd(R, H)  # base EOMCCSD part (separately antisymmetrized)

    # Add on T3 parts
    dR = build_11(dR, R, H, system)
    dR = build_10(dR, R, H, system)
    dR = build_01(dR, R, H, system)
    dR = build_00(dR, R, H, system)

    dR.a += x1

    return dR


def build_eomccsd(R, H):

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

def build_11(dR, R, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    dR.a[Va, Oa] = (1.0 / 1.0) * (
            -0.25 * np.einsum('mnef,AfemnI->AI', H.aa.oovv[oa, oa, va, va], R.aaa.VvvooO, optimize=True)
            - 0.5 * np.einsum('Mnef,AfenIM->AI', H.aa.oovv[Oa, oa, va, va], R.aaa.VvvoOO, optimize=True)
            - 0.25 * np.einsum('MNef,AfeIMN->AI', H.aa.oovv[Oa, Oa, va, va], R.aaa.VvvOOO, optimize=True)
            - 0.5 * np.einsum('mnEf,EAfmnI->AI', H.aa.oovv[oa, oa, Va, va], R.aaa.VVvooO, optimize=True)
            - 0.25 * np.einsum('mnEF,FEAmnI->AI', H.aa.oovv[oa, oa, Va, Va], R.aaa.VVVooO, optimize=True)
            - 1.0 * np.einsum('MnEf,EAfnIM->AI', H.aa.oovv[Oa, oa, Va, va], R.aaa.VVvoOO, optimize=True)
            - 0.5 * np.einsum('MnEF,FEAnIM->AI', H.aa.oovv[Oa, oa, Va, Va], R.aaa.VVVoOO, optimize=True)
            - 0.5 * np.einsum('MNEf,EAfIMN->AI', H.aa.oovv[Oa, Oa, Va, va], R.aaa.VVvOOO, optimize=True)
            - 0.25 * np.einsum('MNEF,FEAIMN->AI', H.aa.oovv[Oa, Oa, Va, Va], R.aaa.VVVOOO, optimize=True)
    )
    dR.a[Va, Oa] += (1.0 / 1.0) * (
            -1.0 * np.einsum('mnef,AefmIn->AI', H.ab.oovv[oa, ob, va, vb], R.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('mneF,AeFmIn->AI', H.ab.oovv[oa, ob, va, Vb], R.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('mNef,AefmIN->AI', H.ab.oovv[oa, Ob, va, vb], R.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('mNeF,AeFmIN->AI', H.ab.oovv[oa, Ob, va, Vb], R.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('Mnef,AefIMn->AI', H.ab.oovv[Oa, ob, va, vb], R.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('MneF,AeFIMn->AI', H.ab.oovv[Oa, ob, va, Vb], R.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('MNef,AefIMN->AI', H.ab.oovv[Oa, Ob, va, vb], R.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('MNeF,AeFIMN->AI', H.ab.oovv[Oa, Ob, va, Vb], R.aab.VvVOOO, optimize=True)
            + 1.0 * np.einsum('mnEf,EAfmIn->AI', H.ab.oovv[oa, ob, Va, vb], R.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('mnEF,EAFmIn->AI', H.ab.oovv[oa, ob, Va, Vb], R.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('mNEf,EAfmIN->AI', H.ab.oovv[oa, Ob, Va, vb], R.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('mNEF,EAFmIN->AI', H.ab.oovv[oa, Ob, Va, Vb], R.aab.VVVoOO, optimize=True)
            - 1.0 * np.einsum('MnEf,EAfIMn->AI', H.ab.oovv[Oa, ob, Va, vb], R.aab.VVvOOo, optimize=True)
            - 1.0 * np.einsum('MnEF,EAFIMn->AI', H.ab.oovv[Oa, ob, Va, Vb], R.aab.VVVOOo, optimize=True)
            - 1.0 * np.einsum('MNEf,EAfIMN->AI', H.ab.oovv[Oa, Ob, Va, vb], R.aab.VVvOOO, optimize=True)
            - 1.0 * np.einsum('MNEF,EAFIMN->AI', H.ab.oovv[Oa, Ob, Va, Vb], R.aab.VVVOOO, optimize=True)
    )
    dR.a[Va, Oa] += (1.0 / 1.0) * (
            -0.25 * np.einsum('mnef,AfeImn->AI', H.bb.oovv[ob, ob, vb, vb], R.abb.VvvOoo, optimize=True)
            + 0.5 * np.einsum('Mnef,AfeInM->AI', H.bb.oovv[Ob, ob, vb, vb], R.abb.VvvOoO, optimize=True)
            - 0.25 * np.einsum('MNef,AfeIMN->AI', H.bb.oovv[Ob, Ob, vb, vb], R.abb.VvvOOO, optimize=True)
            + 0.5 * np.einsum('mnEf,AEfImn->AI', H.bb.oovv[ob, ob, Vb, vb], R.abb.VVvOoo, optimize=True)
            - 0.25 * np.einsum('mnEF,AFEImn->AI', H.bb.oovv[ob, ob, Vb, Vb], R.abb.VVVOoo, optimize=True)
            - 1.0 * np.einsum('MnEf,AEfInM->AI', H.bb.oovv[Ob, ob, Vb, vb], R.abb.VVvOoO, optimize=True)
            + 0.5 * np.einsum('MnEF,AFEInM->AI', H.bb.oovv[Ob, ob, Vb, Vb], R.abb.VVVOoO, optimize=True)
            + 0.5 * np.einsum('MNEf,AEfIMN->AI', H.bb.oovv[Ob, Ob, Vb, vb], R.abb.VVvOOO, optimize=True)
            - 0.25 * np.einsum('MNEF,AFEIMN->AI', H.bb.oovv[Ob, Ob, Vb, Vb], R.abb.VVVOOO, optimize=True)
    )
    # of terms =  3

    return dR

def build_10(dR, R, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    dR.a[Va, oa] = (1.0 / 1.0) * (
            +0.5 * np.einsum('Mnef,AfeinM->Ai', H.aa.oovv[Oa, oa, va, va], R.aaa.VvvooO, optimize=True)
            - 0.25 * np.einsum('MNef,AfeiMN->Ai', H.aa.oovv[Oa, Oa, va, va], R.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('MneF,FAeinM->Ai', H.aa.oovv[Oa, oa, va, Va], R.aaa.VVvooO, optimize=True)
            + 0.5 * np.einsum('MnEF,FEAinM->Ai', H.aa.oovv[Oa, oa, Va, Va], R.aaa.VVVooO, optimize=True)
            + 0.5 * np.einsum('MNeF,FAeiMN->Ai', H.aa.oovv[Oa, Oa, va, Va], R.aaa.VVvoOO, optimize=True)
            - 0.25 * np.einsum('MNEF,FEAiMN->Ai', H.aa.oovv[Oa, Oa, Va, Va], R.aaa.VVVoOO, optimize=True)
    )
    dR.a[Va, oa] += (1.0 / 1.0) * (
            +1.0 * np.einsum('mNef,AefimN->Ai', H.ab.oovv[oa, Ob, va, vb], R.aab.VvvooO, optimize=True)
            - 1.0 * np.einsum('mNEf,EAfimN->Ai', H.ab.oovv[oa, Ob, Va, vb], R.aab.VVvooO, optimize=True)
            + 1.0 * np.einsum('mNeF,AeFimN->Ai', H.ab.oovv[oa, Ob, va, Vb], R.aab.VvVooO, optimize=True)
            - 1.0 * np.einsum('mNEF,EAFimN->Ai', H.ab.oovv[oa, Ob, Va, Vb], R.aab.VVVooO, optimize=True)
            + 1.0 * np.einsum('Mnef,AefiMn->Ai', H.ab.oovv[Oa, ob, va, vb], R.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('MnEf,EAfiMn->Ai', H.ab.oovv[Oa, ob, Va, vb], R.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('MNef,AefiMN->Ai', H.ab.oovv[Oa, Ob, va, vb], R.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('MNEf,EAfiMN->Ai', H.ab.oovv[Oa, Ob, Va, vb], R.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('MneF,AeFiMn->Ai', H.ab.oovv[Oa, ob, va, Vb], R.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('MnEF,EAFiMn->Ai', H.ab.oovv[Oa, ob, Va, Vb], R.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('MNeF,AeFiMN->Ai', H.ab.oovv[Oa, Ob, va, Vb], R.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('MNEF,EAFiMN->Ai', H.ab.oovv[Oa, Ob, Va, Vb], R.aab.VVVoOO, optimize=True)
    )
    dR.a[Va, oa] += (1.0 / 1.0) * (
            +0.5 * np.einsum('Mnef,AfeinM->Ai', H.bb.oovv[Ob, ob, vb, vb], R.abb.VvvooO, optimize=True)
            - 0.25 * np.einsum('MNef,AfeiMN->Ai', H.bb.oovv[Ob, Ob, vb, vb], R.abb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('MneF,AFeinM->Ai', H.bb.oovv[Ob, ob, vb, Vb], R.abb.VVvooO, optimize=True)
            + 0.5 * np.einsum('MnEF,AFEinM->Ai', H.bb.oovv[Ob, ob, Vb, Vb], R.abb.VVVooO, optimize=True)
            - 0.5 * np.einsum('MNeF,AFeiMN->Ai', H.bb.oovv[Ob, Ob, vb, Vb], R.abb.VVvoOO, optimize=True)
            - 0.25 * np.einsum('MNEF,AFEiMN->Ai', H.bb.oovv[Ob, Ob, Vb, Vb], R.abb.VVVoOO, optimize=True)
    )
    # of terms =  3

    return dR

def build_01(dR, R, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    dR.a[va, Oa] = (1.0 / 1.0) * (
            +0.5 * np.einsum('mnEf,EfamnI->aI', H.aa.oovv[oa, oa, Va, va], R.aaa.VvvooO, optimize=True)
            - 0.25 * np.einsum('mnEF,FEamnI->aI', H.aa.oovv[oa, oa, Va, Va], R.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('MnEf,EfanIM->aI', H.aa.oovv[Oa, oa, Va, va], R.aaa.VvvoOO, optimize=True)
            + 0.5 * np.einsum('MNEf,EfaIMN->aI', H.aa.oovv[Oa, Oa, Va, va], R.aaa.VvvOOO, optimize=True)
            - 0.5 * np.einsum('MnEF,FEanIM->aI', H.aa.oovv[Oa, oa, Va, Va], R.aaa.VVvoOO, optimize=True)
            - 0.25 * np.einsum('MNEF,FEaIMN->aI', H.aa.oovv[Oa, Oa, Va, Va], R.aaa.VVvOOO, optimize=True)
    )
    dR.a[va, Oa] += (1.0 / 1.0) * (
            +1.0 * np.einsum('mneF,eaFmIn->aI', H.ab.oovv[oa, ob, va, Vb], R.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('mNeF,eaFmIN->aI', H.ab.oovv[oa, Ob, va, Vb], R.aab.vvVoOO, optimize=True)
            + 1.0 * np.einsum('mnEf,EafmIn->aI', H.ab.oovv[oa, ob, Va, vb], R.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('mNEf,EafmIN->aI', H.ab.oovv[oa, Ob, Va, vb], R.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('mnEF,EaFmIn->aI', H.ab.oovv[oa, ob, Va, Vb], R.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('mNEF,EaFmIN->aI', H.ab.oovv[oa, Ob, Va, Vb], R.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('MneF,eaFIMn->aI', H.ab.oovv[Oa, ob, va, Vb], R.aab.vvVOOo, optimize=True)
            - 1.0 * np.einsum('MNeF,eaFIMN->aI', H.ab.oovv[Oa, Ob, va, Vb], R.aab.vvVOOO, optimize=True)
            - 1.0 * np.einsum('MnEf,EafIMn->aI', H.ab.oovv[Oa, ob, Va, vb], R.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('MNEf,EafIMN->aI', H.ab.oovv[Oa, Ob, Va, vb], R.aab.VvvOOO, optimize=True)
            - 1.0 * np.einsum('MnEF,EaFIMn->aI', H.ab.oovv[Oa, ob, Va, Vb], R.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('MNEF,EaFIMN->aI', H.ab.oovv[Oa, Ob, Va, Vb], R.aab.VvVOOO, optimize=True)
    )
    dR.a[va, Oa] += (1.0 / 1.0) * (
            +0.5 * np.einsum('mnEf,aEfImn->aI', H.bb.oovv[ob, ob, Vb, vb], R.abb.vVvOoo, optimize=True)
            - 0.25 * np.einsum('mnEF,aFEImn->aI', H.bb.oovv[ob, ob, Vb, Vb], R.abb.vVVOoo, optimize=True)
            - 1.0 * np.einsum('MnEf,aEfInM->aI', H.bb.oovv[Ob, ob, Vb, vb], R.abb.vVvOoO, optimize=True)
            + 0.5 * np.einsum('MNEf,aEfIMN->aI', H.bb.oovv[Ob, Ob, Vb, vb], R.abb.vVvOOO, optimize=True)
            + 0.5 * np.einsum('MnEF,aFEInM->aI', H.bb.oovv[Ob, ob, Vb, Vb], R.abb.vVVOoO, optimize=True)
            - 0.25 * np.einsum('MNEF,aFEIMN->aI', H.bb.oovv[Ob, Ob, Vb, Vb], R.abb.vVVOOO, optimize=True)
    )
    # of terms =  3

    return dR

def build_00(dR, R, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    dR.a[va, oa] = (1.0 / 1.0) * (
            +1.0 * np.einsum('mNEf,EfaimN->ai', H.aa.oovv[oa, Oa, Va, va], R.aaa.VvvooO, optimize=True)
            - 0.5 * np.einsum('mNEF,FEaimN->ai', H.aa.oovv[oa, Oa, Va, Va], R.aaa.VVvooO, optimize=True)
            + 0.5 * np.einsum('MNEf,EfaiMN->ai', H.aa.oovv[Oa, Oa, Va, va], R.aaa.VvvoOO, optimize=True)
            - 0.25 * np.einsum('MNEF,FEaiMN->ai', H.aa.oovv[Oa, Oa, Va, Va], R.aaa.VVvoOO, optimize=True)
    )
    dR.a[va, oa] += (1.0 / 1.0) * (
            -1.0 * np.einsum('MneF,eaFiMn->ai', H.ab.oovv[Oa, ob, va, Vb], R.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('MnEf,EafiMn->ai', H.ab.oovv[Oa, ob, Va, vb], R.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('MnEF,EaFiMn->ai', H.ab.oovv[Oa, ob, Va, Vb], R.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('mNeF,eaFimN->ai', H.ab.oovv[oa, Ob, va, Vb], R.aab.vvVooO, optimize=True)
            - 1.0 * np.einsum('mNEf,EafimN->ai', H.ab.oovv[oa, Ob, Va, vb], R.aab.VvvooO, optimize=True)
            - 1.0 * np.einsum('mNEF,EaFimN->ai', H.ab.oovv[oa, Ob, Va, Vb], R.aab.VvVooO, optimize=True)
            - 1.0 * np.einsum('MNeF,eaFiMN->ai', H.ab.oovv[Oa, Ob, va, Vb], R.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('MNEf,EafiMN->ai', H.ab.oovv[Oa, Ob, Va, vb], R.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('MNEF,EaFiMN->ai', H.ab.oovv[Oa, Ob, Va, Vb], R.aab.VvVoOO, optimize=True)
    )
    dR.a[va, oa] += (1.0 / 1.0) * (
            +1.0 * np.einsum('mNEf,aEfimN->ai', H.bb.oovv[ob, Ob, Vb, vb], R.abb.vVvooO, optimize=True)
            - 0.5 * np.einsum('mNEF,aFEimN->ai', H.bb.oovv[ob, Ob, Vb, Vb], R.abb.vVVooO, optimize=True)
            + 0.5 * np.einsum('MNEf,aEfiMN->ai', H.bb.oovv[Ob, Ob, Vb, vb], R.abb.vVvoOO, optimize=True)
            - 0.25 * np.einsum('MNEF,aFEiMN->ai', H.bb.oovv[Ob, Ob, Vb, Vb], R.abb.vVVoOO, optimize=True)
    )

    return dR
