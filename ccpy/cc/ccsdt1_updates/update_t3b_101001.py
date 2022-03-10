import numpy as np
from ccpy.utilities.active_space import get_active_slices

def update(T, dT, H, H0, shift, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dT.aab.VvVooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mi,AbCmjK->AbCijK', H.a.oo[oa, oa], T.aab.VvVooO, optimize=True)
            + 1.0 * np.einsum('Mi,AbCjMK->AbCijK', H.a.oo[Oa, oa], T.aab.VvVoOO, optimize=True)
    )
    dT.aab.VvVooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('MK,AbCijM->AbCijK', H.b.oo[Ob, Ob], T.aab.VvVooO, optimize=True)
    )
    dT.aab.VvVooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('Ae,beCijK->AbCijK', H.a.vv[Va, va], T.aab.vvVooO, optimize=True)
            + 1.0 * np.einsum('AE,EbCijK->AbCijK', H.a.vv[Va, Va], T.aab.VvVooO, optimize=True)
    )
    dT.aab.VvVooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('be,AeCijK->AbCijK', H.a.vv[va, va], T.aab.VvVooO, optimize=True)
            + 1.0 * np.einsum('bE,AECijK->AbCijK', H.a.vv[va, Va], T.aab.VVVooO, optimize=True)
    )
    dT.aab.VvVooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('Ce,AbeijK->AbCijK', H.b.vv[Vb, vb], T.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('CE,AbEijK->AbCijK', H.b.vv[Vb, Vb], T.aab.VvVooO, optimize=True)
    )
    dT.aab.VvVooO += (1.0 / 2.0) * (
            +0.5 * np.einsum('mnij,AbCmnK->AbCijK', H.aa.oooo[oa, oa, oa, oa], T.aab.VvVooO, optimize=True)
            - 1.0 * np.einsum('Mnij,AbCnMK->AbCijK', H.aa.oooo[Oa, oa, oa, oa], T.aab.VvVoOO, optimize=True)
            + 0.5 * np.einsum('MNij,AbCMNK->AbCijK', H.aa.oooo[Oa, Oa, oa, oa], T.aab.VvVOOO, optimize=True)
    )
    dT.aab.VvVooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mNjK,AbCimN->AbCijK', H.ab.oooo[oa, Ob, oa, Ob], T.aab.VvVooO, optimize=True)
            + 1.0 * np.einsum('MnjK,AbCiMn->AbCijK', H.ab.oooo[Oa, ob, oa, Ob], T.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('MNjK,AbCiMN->AbCijK', H.ab.oooo[Oa, Ob, oa, Ob], T.aab.VvVoOO, optimize=True)
    )
    dT.aab.VvVooO += (1.0 / 2.0) * (
            -0.5 * np.einsum('Abef,feCijK->AbCijK', H.aa.vvvv[Va, va, va, va], T.aab.vvVooO, optimize=True)
            - 1.0 * np.einsum('AbeF,FeCijK->AbCijK', H.aa.vvvv[Va, va, va, Va], T.aab.VvVooO, optimize=True)
            - 0.5 * np.einsum('AbEF,FECijK->AbCijK', H.aa.vvvv[Va, va, Va, Va], T.aab.VVVooO, optimize=True)
    )
    dT.aab.VvVooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('bCef,AefijK->AbCijK', H.ab.vvvv[va, Vb, va, vb], T.aab.VvvooO, optimize=True)
            - 1.0 * np.einsum('bCEf,EAfijK->AbCijK', H.ab.vvvv[va, Vb, Va, vb], T.aab.VVvooO, optimize=True)
            + 1.0 * np.einsum('bCeF,AeFijK->AbCijK', H.ab.vvvv[va, Vb, va, Vb], T.aab.VvVooO, optimize=True)
            - 1.0 * np.einsum('bCEF,EAFijK->AbCijK', H.ab.vvvv[va, Vb, Va, Vb], T.aab.VVVooO, optimize=True)
    )
    dT.aab.VvVooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('ACEf,EbfijK->AbCijK', H.ab.vvvv[Va, Vb, Va, vb], T.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('ACeF,ebFijK->AbCijK', H.ab.vvvv[Va, Vb, va, Vb], T.aab.vvVooO, optimize=True)
            + 1.0 * np.einsum('ACEF,EbFijK->AbCijK', H.ab.vvvv[Va, Vb, Va, Vb], T.aab.VvVooO, optimize=True)
    )
    dT.aab.VvVooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('Amie,beCmjK->AbCijK', H.aa.voov[Va, oa, oa, va], T.aab.vvVooO, optimize=True)
            + 1.0 * np.einsum('AMie,beCjMK->AbCijK', H.aa.voov[Va, Oa, oa, va], T.aab.vvVoOO, optimize=True)
            + 1.0 * np.einsum('AmiE,EbCmjK->AbCijK', H.aa.voov[Va, oa, oa, Va], T.aab.VvVooO, optimize=True)
            - 1.0 * np.einsum('AMiE,EbCjMK->AbCijK', H.aa.voov[Va, Oa, oa, Va], T.aab.VvVoOO, optimize=True)
    )
    dT.aab.VvVooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('bmie,AeCmjK->AbCijK', H.aa.voov[va, oa, oa, va], T.aab.VvVooO, optimize=True)
            - 1.0 * np.einsum('bMie,AeCjMK->AbCijK', H.aa.voov[va, Oa, oa, va], T.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('bmiE,AECmjK->AbCijK', H.aa.voov[va, oa, oa, Va], T.aab.VVVooO, optimize=True)
            - 1.0 * np.einsum('bMiE,AECjMK->AbCijK', H.aa.voov[va, Oa, oa, Va], T.aab.VVVoOO, optimize=True)
    )
    dT.aab.VvVooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('Amie,bCejmK->AbCijK', H.ab.voov[Va, ob, oa, vb], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('AMie,bCejMK->AbCijK', H.ab.voov[Va, Ob, oa, vb], T.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('AmiE,bCEjmK->AbCijK', H.ab.voov[Va, ob, oa, Vb], T.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('AMiE,bCEjMK->AbCijK', H.ab.voov[Va, Ob, oa, Vb], T.abb.vVVoOO, optimize=True)
    )
    dT.aab.VvVooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('bmie,ACejmK->AbCijK', H.ab.voov[va, ob, oa, vb], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('bMie,ACejMK->AbCijK', H.ab.voov[va, Ob, oa, vb], T.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('bmiE,ACEjmK->AbCijK', H.ab.voov[va, ob, oa, Vb], T.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('bMiE,ACEjMK->AbCijK', H.ab.voov[va, Ob, oa, Vb], T.abb.VVVoOO, optimize=True)
    )
    dT.aab.VvVooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('MCeK,AebijM->AbCijK', H.ab.ovvo[Oa, Vb, va, Ob], T.aaa.VvvooO, optimize=True)
            + 1.0 * np.einsum('MCEK,EAbijM->AbCijK', H.ab.ovvo[Oa, Vb, Va, Ob], T.aaa.VVvooO, optimize=True)
    )
    dT.aab.VvVooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('CMKe,AbeijM->AbCijK', H.bb.voov[Vb, Ob, Ob, vb], T.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('CMKE,AbEijM->AbCijK', H.bb.voov[Vb, Ob, Ob, Vb], T.aab.VvVooO, optimize=True)
    )
    dT.aab.VvVooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('AMeK,beCijM->AbCijK', H.ab.vovo[Va, Ob, va, Ob], T.aab.vvVooO, optimize=True)
            - 1.0 * np.einsum('AMEK,EbCijM->AbCijK', H.ab.vovo[Va, Ob, Va, Ob], T.aab.VvVooO, optimize=True)
    )
    dT.aab.VvVooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('bMeK,AeCijM->AbCijK', H.ab.vovo[va, Ob, va, Ob], T.aab.VvVooO, optimize=True)
            - 1.0 * np.einsum('bMEK,AECijM->AbCijK', H.ab.vovo[va, Ob, Va, Ob], T.aab.VVVooO, optimize=True)
    )
    dT.aab.VvVooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mCie,AbemjK->AbCijK', H.ab.ovov[oa, Vb, oa, vb], T.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('MCie,AbejMK->AbCijK', H.ab.ovov[Oa, Vb, oa, vb], T.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('mCiE,AbEmjK->AbCijK', H.ab.ovov[oa, Vb, oa, Vb], T.aab.VvVooO, optimize=True)
            + 1.0 * np.einsum('MCiE,AbEjMK->AbCijK', H.ab.ovov[Oa, Vb, oa, Vb], T.aab.VvVoOO, optimize=True)
    )

    dT.aab.VvVooO -= np.transpose(dT.aab.VvVooO, (0, 1, 2, 4, 3, 5))

    return T, dT