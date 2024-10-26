import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops

def build(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.aab.vvVooO = (2.0 / 4.0) * (
            +1.0 * np.einsum('bCeK,aeij->abCijK', X.ab.vvvo[va, Vb, :, Ob], T.aa[va, :, oa, oa], optimize=True)
    )
    dR.aab.vvVooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('mCjK,abim->abCijK', X.ab.ovoo[:, Vb, oa, Ob], T.aa[va, va, oa, :], optimize=True)
    )
    dR.aab.vvVooO += (4.0 / 4.0) * (
            +1.0 * np.einsum('aCie,bejK->abCijK', X.ab.vvov[va, Vb, oa, :], T.ab[va, :, oa, Ob], optimize=True)
    )
    dR.aab.vvVooO += (4.0 / 4.0) * (
            -1.0 * np.einsum('amiK,bCjm->abCijK', X.ab.vooo[va, :, oa, Ob], T.ab[va, Vb, oa, :], optimize=True)
    )
    dR.aab.vvVooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('abie,eCjK->abCijK', X.aa.vvov[va, va, oa, :], T.ab[:, Vb, oa, Ob], optimize=True)
    )
    dR.aab.vvVooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('amij,bCmK->abCijK', X.aa.vooo[va, :, oa, oa], T.ab[va, Vb, :, Ob], optimize=True)
    )
    dR.aab.vvVooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('bCeK,aeij->abCijK', H.ab.vvvo[va, Vb, :, Ob], R.aa[va, :, oa, oa], optimize=True)
    )
    dR.aab.vvVooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('mCjK,abim->abCijK', H.ab.ovoo[:, Vb, oa, Ob], R.aa[va, va, oa, :], optimize=True)
    )
    dR.aab.vvVooO += (4.0 / 4.0) * (
            +1.0 * np.einsum('aCie,bejK->abCijK', H.ab.vvov[va, Vb, oa, :], R.ab[va, :, oa, Ob], optimize=True)
    )
    dR.aab.vvVooO += (4.0 / 4.0) * (
            -1.0 * np.einsum('amiK,bCjm->abCijK', H.ab.vooo[va, :, oa, Ob], R.ab[va, Vb, oa, :], optimize=True)
    )
    dR.aab.vvVooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('abie,eCjK->abCijK', H.aa.vvov[va, va, oa, :], R.ab[:, Vb, oa, Ob], optimize=True)
    )
    dR.aab.vvVooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('amij,bCmK->abCijK', H.aa.vooo[va, :, oa, oa], R.ab[va, Vb, :, Ob], optimize=True)
    )

    dR.aab.vvVooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mi,baCmjK->abCijK', X.a.oo[oa, oa], T.aab.vvVooO, optimize=True)
            - 1.0 * np.einsum('Mi,baCjMK->abCijK', X.a.oo[Oa, oa], T.aab.vvVoOO, optimize=True)
    )
    dR.aab.vvVooO += (1.0 / 4.0) * (
            +1.0 * np.einsum('MK,baCijM->abCijK', X.b.oo[Ob, Ob], T.aab.vvVooO, optimize=True)
    )
    dR.aab.vvVooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('ae,beCijK->abCijK', X.a.vv[va, va], T.aab.vvVooO, optimize=True)
            + 1.0 * np.einsum('aE,EbCijK->abCijK', X.a.vv[va, Va], T.aab.VvVooO, optimize=True)
    )
    dR.aab.vvVooO += (1.0 / 4.0) * (
            -1.0 * np.einsum('CE,baEijK->abCijK', X.b.vv[Vb, Vb], T.aab.vvVooO, optimize=True)
    )
    dR.aab.vvVooO += (1.0 / 4.0) * (
            -0.5 * np.einsum('mnij,baCmnK->abCijK', X.aa.oooo[oa, oa, oa, oa], T.aab.vvVooO, optimize=True)
            - 1.0 * np.einsum('mNij,baCmNK->abCijK', X.aa.oooo[oa, Oa, oa, oa], T.aab.vvVoOO, optimize=True)
            - 0.5 * np.einsum('MNij,baCMNK->abCijK', X.aa.oooo[Oa, Oa, oa, oa], T.aab.vvVOOO, optimize=True)
    )
    dR.aab.vvVooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('MnjK,baCiMn->abCijK', X.ab.oooo[Oa, ob, oa, Ob], T.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('mNjK,baCimN->abCijK', X.ab.oooo[oa, Ob, oa, Ob], T.aab.vvVooO, optimize=True)
            - 1.0 * np.einsum('MNjK,baCiMN->abCijK', X.ab.oooo[Oa, Ob, oa, Ob], T.aab.vvVoOO, optimize=True)
    )
    dR.aab.vvVooO += (1.0 / 4.0) * (
            -0.5 * np.einsum('abef,feCijK->abCijK', X.aa.vvvv[va, va, va, va], T.aab.vvVooO, optimize=True)
            + 1.0 * np.einsum('abEf,EfCijK->abCijK', X.aa.vvvv[va, va, Va, va], T.aab.VvVooO, optimize=True)
            - 0.5 * np.einsum('abEF,FECijK->abCijK', X.aa.vvvv[va, va, Va, Va], T.aab.VVVooO, optimize=True)
    )
    dR.aab.vvVooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('bCeF,eaFijK->abCijK', X.ab.vvvv[va, Vb, va, Vb], T.aab.vvVooO, optimize=True)
            - 1.0 * np.einsum('bCEf,EafijK->abCijK', X.ab.vvvv[va, Vb, Va, vb], T.aab.VvvooO, optimize=True)
            - 1.0 * np.einsum('bCEF,EaFijK->abCijK', X.ab.vvvv[va, Vb, Va, Vb], T.aab.VvVooO, optimize=True)
    )
    dR.aab.vvVooO += (4.0 / 4.0) * (
            -1.0 * np.einsum('amie,beCmjK->abCijK', X.aa.voov[va, oa, oa, va], T.aab.vvVooO, optimize=True)
            + 1.0 * np.einsum('amiE,EbCmjK->abCijK', X.aa.voov[va, oa, oa, Va], T.aab.VvVooO, optimize=True)
            + 1.0 * np.einsum('aMie,beCjMK->abCijK', X.aa.voov[va, Oa, oa, va], T.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('aMiE,EbCjMK->abCijK', X.aa.voov[va, Oa, oa, Va], T.aab.VvVoOO, optimize=True)
    )
    dR.aab.vvVooO += (4.0 / 4.0) * (
            -1.0 * np.einsum('amie,bCejmK->abCijK', X.ab.voov[va, ob, oa, vb], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('amiE,bCEjmK->abCijK', X.ab.voov[va, ob, oa, Vb], T.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('aMie,bCejMK->abCijK', X.ab.voov[va, Ob, oa, vb], T.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('aMiE,bCEjMK->abCijK', X.ab.voov[va, Ob, oa, Vb], T.abb.vVVoOO, optimize=True)
    )
    dR.aab.vvVooO += (1.0 / 4.0) * (
            -1.0 * np.einsum('MCEK,EbaijM->abCijK', X.ab.ovvo[Oa, Vb, Va, Ob], T.aaa.VvvooO, optimize=True)
    )
    dR.aab.vvVooO += (1.0 / 4.0) * (
            -1.0 * np.einsum('CMKE,baEijM->abCijK', X.bb.voov[Vb, Ob, Ob, Vb], T.aab.vvVooO, optimize=True)
    )
    dR.aab.vvVooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('aMeK,beCijM->abCijK', X.ab.vovo[va, Ob, va, Ob], T.aab.vvVooO, optimize=True)
            - 1.0 * np.einsum('aMEK,EbCijM->abCijK', X.ab.vovo[va, Ob, Va, Ob], T.aab.VvVooO, optimize=True)
    )
    dR.aab.vvVooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mCiE,baEmjK->abCijK', X.ab.ovov[oa, Vb, oa, Vb], T.aab.vvVooO, optimize=True)
            - 1.0 * np.einsum('MCiE,baEjMK->abCijK', X.ab.ovov[Oa, Vb, oa, Vb], T.aab.vvVoOO, optimize=True)
    )
    dR.aab.vvVooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mi,baCmjK->abCijK', H.a.oo[oa, oa], R.aab.vvVooO, optimize=True)
            - 1.0 * np.einsum('Mi,baCjMK->abCijK', H.a.oo[Oa, oa], R.aab.vvVoOO, optimize=True)
    )
    dR.aab.vvVooO += (1.0 / 4.0) * (
            +1.0 * np.einsum('MK,baCijM->abCijK', H.b.oo[Ob, Ob], R.aab.vvVooO, optimize=True)
    )
    dR.aab.vvVooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('ae,beCijK->abCijK', H.a.vv[va, va], R.aab.vvVooO, optimize=True)
            + 1.0 * np.einsum('aE,EbCijK->abCijK', H.a.vv[va, Va], R.aab.VvVooO, optimize=True)
    )
    dR.aab.vvVooO += (1.0 / 4.0) * (
            -1.0 * np.einsum('CE,baEijK->abCijK', H.b.vv[Vb, Vb], R.aab.vvVooO, optimize=True)
    )
    dR.aab.vvVooO += (1.0 / 4.0) * (
            -0.5 * np.einsum('mnij,baCmnK->abCijK', H.aa.oooo[oa, oa, oa, oa], R.aab.vvVooO, optimize=True)
            - 1.0 * np.einsum('mNij,baCmNK->abCijK', H.aa.oooo[oa, Oa, oa, oa], R.aab.vvVoOO, optimize=True)
            - 0.5 * np.einsum('MNij,baCMNK->abCijK', H.aa.oooo[Oa, Oa, oa, oa], R.aab.vvVOOO, optimize=True)
    )
    dR.aab.vvVooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('MnjK,baCiMn->abCijK', H.ab.oooo[Oa, ob, oa, Ob], R.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('mNjK,baCimN->abCijK', H.ab.oooo[oa, Ob, oa, Ob], R.aab.vvVooO, optimize=True)
            - 1.0 * np.einsum('MNjK,baCiMN->abCijK', H.ab.oooo[Oa, Ob, oa, Ob], R.aab.vvVoOO, optimize=True)
    )
    dR.aab.vvVooO += (1.0 / 4.0) * (
            -0.5 * np.einsum('abef,feCijK->abCijK', H.aa.vvvv[va, va, va, va], R.aab.vvVooO, optimize=True)
            + 1.0 * np.einsum('abEf,EfCijK->abCijK', H.aa.vvvv[va, va, Va, va], R.aab.VvVooO, optimize=True)
            - 0.5 * np.einsum('abEF,FECijK->abCijK', H.aa.vvvv[va, va, Va, Va], R.aab.VVVooO, optimize=True)
    )
    dR.aab.vvVooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('bCeF,eaFijK->abCijK', H.ab.vvvv[va, Vb, va, Vb], R.aab.vvVooO, optimize=True)
            - 1.0 * np.einsum('bCEf,EafijK->abCijK', H.ab.vvvv[va, Vb, Va, vb], R.aab.VvvooO, optimize=True)
            - 1.0 * np.einsum('bCEF,EaFijK->abCijK', H.ab.vvvv[va, Vb, Va, Vb], R.aab.VvVooO, optimize=True)
    )
    dR.aab.vvVooO += (4.0 / 4.0) * (
            -1.0 * np.einsum('amie,beCmjK->abCijK', H.aa.voov[va, oa, oa, va], R.aab.vvVooO, optimize=True)
            + 1.0 * np.einsum('amiE,EbCmjK->abCijK', H.aa.voov[va, oa, oa, Va], R.aab.VvVooO, optimize=True)
            + 1.0 * np.einsum('aMie,beCjMK->abCijK', H.aa.voov[va, Oa, oa, va], R.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('aMiE,EbCjMK->abCijK', H.aa.voov[va, Oa, oa, Va], R.aab.VvVoOO, optimize=True)
    )
    dR.aab.vvVooO += (4.0 / 4.0) * (
            -1.0 * np.einsum('amie,bCejmK->abCijK', H.ab.voov[va, ob, oa, vb], R.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('amiE,bCEjmK->abCijK', H.ab.voov[va, ob, oa, Vb], R.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('aMie,bCejMK->abCijK', H.ab.voov[va, Ob, oa, vb], R.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('aMiE,bCEjMK->abCijK', H.ab.voov[va, Ob, oa, Vb], R.abb.vVVoOO, optimize=True)
    )
    dR.aab.vvVooO += (1.0 / 4.0) * (
            -1.0 * np.einsum('MCEK,EbaijM->abCijK', H.ab.ovvo[Oa, Vb, Va, Ob], R.aaa.VvvooO, optimize=True)
    )
    dR.aab.vvVooO += (1.0 / 4.0) * (
            -1.0 * np.einsum('CMKE,baEijM->abCijK', H.bb.voov[Vb, Ob, Ob, Vb], R.aab.vvVooO, optimize=True)
    )
    dR.aab.vvVooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('aMeK,beCijM->abCijK', H.ab.vovo[va, Ob, va, Ob], R.aab.vvVooO, optimize=True)
            - 1.0 * np.einsum('aMEK,EbCijM->abCijK', H.ab.vovo[va, Ob, Va, Ob], R.aab.VvVooO, optimize=True)
    )
    dR.aab.vvVooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mCiE,baEmjK->abCijK', H.ab.ovov[oa, Vb, oa, Vb], R.aab.vvVooO, optimize=True)
            - 1.0 * np.einsum('MCiE,baEjMK->abCijK', H.ab.ovov[Oa, Vb, oa, Vb], R.aab.vvVoOO, optimize=True)
    )

    dR.aab.vvVooO -= np.transpose(dR.aab.vvVooO, (1, 0, 2, 3, 4, 5))
    dR.aab.vvVooO -= np.transpose(dR.aab.vvVooO, (0, 1, 2, 4, 3, 5))

    return dR

def update(R, omega, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    R.aab.vvVooO = eomcc_active_loops.update_r3b_001001(
        R.aab.vvVooO,
        omega,
        H.a.oo[Oa, Oa],
        H.a.vv[Va, Va],
        H.a.oo[oa, oa],
        H.a.vv[va, va],
        H.b.oo[Ob, Ob],
        H.b.vv[Vb, Vb],
        H.b.oo[ob, ob],
        H.b.vv[vb, vb],
        0.0,
    )

    return R
