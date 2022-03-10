import numpy as np
from ccpy.utilities.active_space import get_active_slices

def update(T, dT, H, H0, shift, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dT.aab.vvVooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mi,baCmjK->abCijK', H.a.oo[oa, oa], T.aab.vvVooO, optimize=True)
            - 1.0 * np.einsum('Mi,baCjMK->abCijK', H.a.oo[Oa, oa], T.aab.vvVoOO, optimize=True)
    )
    dT.aab.vvVooO += (1.0 / 4.0) * (
            +1.0 * np.einsum('MK,baCijM->abCijK', H.b.oo[Ob, Ob], T.aab.vvVooO, optimize=True)
    )
    dT.aab.vvVooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('ae,beCijK->abCijK', H.a.vv[va, va], T.aab.vvVooO, optimize=True)
            + 1.0 * np.einsum('aE,EbCijK->abCijK', H.a.vv[va, Va], T.aab.VvVooO, optimize=True)
    )
    dT.aab.vvVooO += (1.0 / 4.0) * (
            -1.0 * np.einsum('CE,baEijK->abCijK', H.b.vv[Vb, Vb], T.aab.vvVooO, optimize=True)
    )
    dT.aab.vvVooO += (1.0 / 4.0) * (
            -0.5 * np.einsum('mnij,baCmnK->abCijK', H.aa.oooo[oa, oa, oa, oa], T.aab.vvVooO, optimize=True)
            - 1.0 * np.einsum('mNij,baCmNK->abCijK', H.aa.oooo[oa, Oa, oa, oa], T.aab.vvVoOO, optimize=True)
            - 0.5 * np.einsum('MNij,baCMNK->abCijK', H.aa.oooo[Oa, Oa, oa, oa], T.aab.vvVOOO, optimize=True)
    )
    dT.aab.vvVooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('MnjK,baCiMn->abCijK', H.ab.oooo[Oa, ob, oa, Ob], T.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('mNjK,baCimN->abCijK', H.ab.oooo[oa, Ob, oa, Ob], T.aab.vvVooO, optimize=True)
            - 1.0 * np.einsum('MNjK,baCiMN->abCijK', H.ab.oooo[Oa, Ob, oa, Ob], T.aab.vvVoOO, optimize=True)
    )
    dT.aab.vvVooO += (1.0 / 4.0) * (
            -0.5 * np.einsum('abef,feCijK->abCijK', H.aa.vvvv[va, va, va, va], T.aab.vvVooO, optimize=True)
            - 1.0 * np.einsum('abeF,FeCijK->abCijK', H.aa.vvvv[va, va, va, Va], T.aab.VvVooO, optimize=True)
            - 0.5 * np.einsum('abEF,FECijK->abCijK', H.aa.vvvv[va, va, Va, Va], T.aab.VVVooO, optimize=True)
    )
    dT.aab.vvVooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('bCEf,EafijK->abCijK', H.ab.vvvv[va, Vb, Va, vb], T.aab.VvvooO, optimize=True)
            - 1.0 * np.einsum('bCeF,eaFijK->abCijK', H.ab.vvvv[va, Vb, va, Vb], T.aab.vvVooO, optimize=True)
            - 1.0 * np.einsum('bCEF,EaFijK->abCijK', H.ab.vvvv[va, Vb, Va, Vb], T.aab.VvVooO, optimize=True)
    )
    dT.aab.vvVooO += (4.0 / 4.0) * (
            -1.0 * np.einsum('amie,beCmjK->abCijK', H.aa.voov[va, oa, oa, va], T.aab.vvVooO, optimize=True)
            + 1.0 * np.einsum('amiE,EbCmjK->abCijK', H.aa.voov[va, oa, oa, Va], T.aab.VvVooO, optimize=True)
            + 1.0 * np.einsum('aMie,beCjMK->abCijK', H.aa.voov[va, Oa, oa, va], T.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('aMiE,EbCjMK->abCijK', H.aa.voov[va, Oa, oa, Va], T.aab.VvVoOO, optimize=True)
    )
    dT.aab.vvVooO += (4.0 / 4.0) * (
            -1.0 * np.einsum('amie,bCejmK->abCijK', H.ab.voov[va, ob, oa, vb], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('amiE,bCEjmK->abCijK', H.ab.voov[va, ob, oa, Vb], T.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('aMie,bCejMK->abCijK', H.ab.voov[va, Ob, oa, vb], T.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('aMiE,bCEjMK->abCijK', H.ab.voov[va, Ob, oa, Vb], T.abb.vVVoOO, optimize=True)
    )
    dT.aab.vvVooO += (1.0 / 4.0) * (
            -1.0 * np.einsum('MCEK,EbaijM->abCijK', H.ab.ovvo[Oa, Vb, Va, Ob], T.aaa.VvvooO, optimize=True)
    )
    dT.aab.vvVooO += (1.0 / 4.0) * (
            -1.0 * np.einsum('CMKE,baEijM->abCijK', H.bb.voov[Vb, Ob, Ob, Vb], T.aab.vvVooO, optimize=True)
    )
    dT.aab.vvVooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('aMeK,beCijM->abCijK', H.ab.vovo[va, Ob, va, Ob], T.aab.vvVooO, optimize=True)
            - 1.0 * np.einsum('aMEK,EbCijM->abCijK', H.ab.vovo[va, Ob, Va, Ob], T.aab.VvVooO, optimize=True)
    )
    dT.aab.vvVooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mCiE,baEmjK->abCijK', H.ab.ovov[oa, Vb, oa, Vb], T.aab.vvVooO, optimize=True)
            - 1.0 * np.einsum('MCiE,baEjMK->abCijK', H.ab.ovov[Oa, Vb, oa, Vb], T.aab.vvVoOO, optimize=True)
    )

    dT.aab.vvVooO -= np.transpose(dT.aab.vvVooO, (1, 0, 2, 3, 4, 5))
    dT.aab.vvVooO -= np.transpose(dT.aab.vvVooO, (0, 1, 2, 4, 3, 5))

    return T, dT