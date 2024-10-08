import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops

def build(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.aaa.VvvooO = (2.0 / 4.0) * (
            -1.0 * np.einsum('bmji,AcmK->AbcijK', X.aa.vooo[va, :, oa, oa], T.aa[Va, va, :, Oa], optimize=True)
    )
    dR.aaa.VvvooO += (1.0 / 4.0) * (
            +1.0 * np.einsum('Amji,bcmK->AbcijK', X.aa.vooo[Va, :, oa, oa], T.aa[va, va, :, Oa], optimize=True)
    )
    dR.aaa.VvvooO += (4.0 / 4.0) * (
            +1.0 * np.einsum('bmjK,Acmi->AbcijK', X.aa.vooo[va, :, oa, Oa], T.aa[Va, va, :, oa], optimize=True)
    )
    dR.aaa.VvvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('AmjK,bcmi->AbcijK', X.aa.vooo[Va, :, oa, Oa], T.aa[va, va, :, oa], optimize=True)
    )
    dR.aaa.VvvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('bmji,AcmK->AbcijK', H.aa.vooo[va, :, oa, oa], R.aa[Va, va, :, Oa], optimize=True)
    )
    dR.aaa.VvvooO += (1.0 / 4.0) * (
            +1.0 * np.einsum('Amji,bcmK->AbcijK', H.aa.vooo[Va, :, oa, oa], R.aa[va, va, :, Oa], optimize=True)
    )
    dR.aaa.VvvooO += (4.0 / 4.0) * (
            +1.0 * np.einsum('bmjK,Acmi->AbcijK', H.aa.vooo[va, :, oa, Oa], R.aa[Va, va, :, oa], optimize=True)
    )
    dR.aaa.VvvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('AmjK,bcmi->AbcijK', H.aa.vooo[Va, :, oa, Oa], R.aa[va, va, :, oa], optimize=True)
    )
    dR.aaa.VvvooO += (4.0 / 4.0) * (
            +1.0 * np.einsum('bAje,eciK->AbcijK', X.aa.vvov[va, Va, oa, :], T.aa[:, va, oa, Oa], optimize=True)
    )
    dR.aaa.VvvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('bcje,eAiK->AbcijK', X.aa.vvov[va, va, oa, :], T.aa[:, Va, oa, Oa], optimize=True)
    )
    dR.aaa.VvvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('bAKe,ecij->AbcijK', X.aa.vvov[va, Va, Oa, :], T.aa[:, va, oa, oa], optimize=True)
    )
    dR.aaa.VvvooO += (1.0 / 4.0) * (
            +1.0 * np.einsum('bcKe,eAij->AbcijK', X.aa.vvov[va, va, Oa, :], T.aa[:, Va, oa, oa], optimize=True)
    )
    dR.aaa.VvvooO += (4.0 / 4.0) * (
            +1.0 * np.einsum('bAje,eciK->AbcijK', H.aa.vvov[va, Va, oa, :], R.aa[:, va, oa, Oa], optimize=True)
    )
    dR.aaa.VvvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('bcje,eAiK->AbcijK', H.aa.vvov[va, va, oa, :], R.aa[:, Va, oa, Oa], optimize=True)
    )
    dR.aaa.VvvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('bAKe,ecij->AbcijK', H.aa.vvov[va, Va, Oa, :], R.aa[:, va, oa, oa], optimize=True)
    )
    dR.aaa.VvvooO += (1.0 / 4.0) * (
            +1.0 * np.einsum('bcKe,eAij->AbcijK', H.aa.vvov[va, va, Oa, :], R.aa[:, Va, oa, oa], optimize=True)
    )
    dR.aaa.VvvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('be,AceijK->AbcijK', X.a.vv[va, va], T.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('bE,EAcijK->AbcijK', X.a.vv[va, Va], T.aaa.VVvooO, optimize=True)
    )
    dR.aaa.VvvooO += (1.0 / 4.0) * (
            -1.0 * np.einsum('AE,EcbijK->AbcijK', X.a.vv[Va, Va], T.aaa.VvvooO, optimize=True)
    )
    dR.aaa.VvvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mj,AcbimK->AbcijK', X.a.oo[oa, oa], T.aaa.VvvooO, optimize=True)
            + 1.0 * np.einsum('Mj,AcbiMK->AbcijK', X.a.oo[Oa, oa], T.aaa.VvvoOO, optimize=True)
    )
    dR.aaa.VvvooO += (1.0 / 4.0) * (
            +1.0 * np.einsum('MK,AcbijM->AbcijK', X.a.oo[Oa, Oa], T.aaa.VvvooO, optimize=True)
    )
    dR.aaa.VvvooO += (1.0 / 4.0) * (
            -0.5 * np.einsum('mnij,AcbmnK->AbcijK', X.aa.oooo[oa, oa, oa, oa], T.aaa.VvvooO, optimize=True)
            + 1.0 * np.einsum('Mnij,AcbnMK->AbcijK', X.aa.oooo[Oa, oa, oa, oa], T.aaa.VvvoOO, optimize=True)
            - 0.5 * np.einsum('MNij,AcbMNK->AbcijK', X.aa.oooo[Oa, Oa, oa, oa], T.aaa.VvvOOO, optimize=True)
    )
    dR.aaa.VvvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('MnKj,AcbniM->AbcijK', X.aa.oooo[Oa, oa, Oa, oa], T.aaa.VvvooO, optimize=True)
            + 0.5 * np.einsum('MNKj,AcbiMN->AbcijK', X.aa.oooo[Oa, Oa, Oa, oa], T.aaa.VvvoOO, optimize=True)
    )
    dR.aaa.VvvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('AbEf,EcfijK->AbcijK', X.aa.vvvv[Va, va, Va, va], T.aaa.VvvooO, optimize=True)
            - 0.5 * np.einsum('AbEF,FEcijK->AbcijK', X.aa.vvvv[Va, va, Va, Va], T.aaa.VVvooO, optimize=True)
    )
    dR.aaa.VvvooO += (1.0 / 4.0) * (
            +0.5 * np.einsum('cbef,AfeijK->AbcijK', X.aa.vvvv[va, va, va, va], T.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('cbEf,AEfijK->AbcijK', X.aa.vvvv[va, va, Va, va], T.aaa.VVvooO, optimize=True)
            + 0.5 * np.einsum('cbEF,AFEijK->AbcijK', X.aa.vvvv[va, va, Va, Va], T.aaa.VVVooO, optimize=True)
    )
    dR.aaa.VvvooO += (4.0 / 4.0) * (
            -1.0 * np.einsum('bmje,AceimK->AbcijK', X.aa.voov[va, oa, oa, va], T.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('bMje,AceiMK->AbcijK', X.aa.voov[va, Oa, oa, va], T.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('bmjE,EAcimK->AbcijK', X.aa.voov[va, oa, oa, Va], T.aaa.VVvooO, optimize=True)
            - 1.0 * np.einsum('bMjE,EAciMK->AbcijK', X.aa.voov[va, Oa, oa, Va], T.aaa.VVvoOO, optimize=True)
    )
    dR.aaa.VvvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('AmjE,EcbimK->AbcijK', X.aa.voov[Va, oa, oa, Va], T.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('AMjE,EcbiMK->AbcijK', X.aa.voov[Va, Oa, oa, Va], T.aaa.VvvoOO, optimize=True)
    )
    dR.aaa.VvvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('bMKe,AceijM->AbcijK', X.aa.voov[va, Oa, Oa, va], T.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('bMKE,EAcijM->AbcijK', X.aa.voov[va, Oa, Oa, Va], T.aaa.VVvooO, optimize=True)
    )
    dR.aaa.VvvooO += (1.0 / 4.0) * (
            -1.0 * np.einsum('AMKE,EcbijM->AbcijK', X.aa.voov[Va, Oa, Oa, Va], T.aaa.VvvooO, optimize=True)
    )
    dR.aaa.VvvooO += (4.0 / 4.0) * (
            +1.0 * np.einsum('bmje,AceiKm->AbcijK', X.ab.voov[va, ob, oa, vb], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('bMje,AceiKM->AbcijK', X.ab.voov[va, Ob, oa, vb], T.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('bmjE,AcEiKm->AbcijK', X.ab.voov[va, ob, oa, Vb], T.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('bMjE,AcEiKM->AbcijK', X.ab.voov[va, Ob, oa, Vb], T.aab.VvVoOO, optimize=True)
    )
    dR.aaa.VvvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('AmjE,cbEiKm->AbcijK', X.ab.voov[Va, ob, oa, Vb], T.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('AMjE,cbEiKM->AbcijK', X.ab.voov[Va, Ob, oa, Vb], T.aab.vvVoOO, optimize=True)
    )
    dR.aaa.VvvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('bMKe,AceijM->AbcijK', X.ab.voov[va, Ob, Oa, vb], T.aab.VvvooO, optimize=True)
            - 1.0 * np.einsum('bMKE,AcEijM->AbcijK', X.ab.voov[va, Ob, Oa, Vb], T.aab.VvVooO, optimize=True)
    )
    dR.aaa.VvvooO += (1.0 / 4.0) * (
            -1.0 * np.einsum('AMKE,cbEijM->AbcijK', X.ab.voov[Va, Ob, Oa, Vb], T.aab.vvVooO, optimize=True)
    )
    dR.aaa.VvvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mj,AcbimK->AbcijK', H.a.oo[oa, oa], R.aaa.VvvooO, optimize=True)
            + 1.0 * np.einsum('Mj,AcbiMK->AbcijK', H.a.oo[Oa, oa], R.aaa.VvvoOO, optimize=True)
    )
    dR.aaa.VvvooO += (1.0 / 4.0) * (
            +1.0 * np.einsum('MK,AcbijM->AbcijK', H.a.oo[Oa, Oa], R.aaa.VvvooO, optimize=True)
    )
    dR.aaa.VvvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('be,AceijK->AbcijK', H.a.vv[va, va], R.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('bE,EAcijK->AbcijK', H.a.vv[va, Va], R.aaa.VVvooO, optimize=True)
    )
    dR.aaa.VvvooO += (1.0 / 4.0) * (
            -1.0 * np.einsum('AE,EcbijK->AbcijK', H.a.vv[Va, Va], R.aaa.VvvooO, optimize=True)
    )
    dR.aaa.VvvooO += (1.0 / 4.0) * (
            -0.5 * np.einsum('mnij,AcbmnK->AbcijK', H.aa.oooo[oa, oa, oa, oa], R.aaa.VvvooO, optimize=True)
            + 1.0 * np.einsum('Mnij,AcbnMK->AbcijK', H.aa.oooo[Oa, oa, oa, oa], R.aaa.VvvoOO, optimize=True)
            - 0.5 * np.einsum('MNij,AcbMNK->AbcijK', H.aa.oooo[Oa, Oa, oa, oa], R.aaa.VvvOOO, optimize=True)
    )
    dR.aaa.VvvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('MnKj,AcbniM->AbcijK', H.aa.oooo[Oa, oa, Oa, oa], R.aaa.VvvooO, optimize=True)
            + 0.5 * np.einsum('MNKj,AcbiMN->AbcijK', H.aa.oooo[Oa, Oa, Oa, oa], R.aaa.VvvoOO, optimize=True)
    )
    dR.aaa.VvvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('AbEf,EcfijK->AbcijK', H.aa.vvvv[Va, va, Va, va], R.aaa.VvvooO, optimize=True)
            - 0.5 * np.einsum('AbEF,FEcijK->AbcijK', H.aa.vvvv[Va, va, Va, Va], R.aaa.VVvooO, optimize=True)
    )
    dR.aaa.VvvooO += (1.0 / 4.0) * (
            +0.5 * np.einsum('cbef,AfeijK->AbcijK', H.aa.vvvv[va, va, va, va], R.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('cbEf,AEfijK->AbcijK', H.aa.vvvv[va, va, Va, va], R.aaa.VVvooO, optimize=True)
            + 0.5 * np.einsum('cbEF,AFEijK->AbcijK', H.aa.vvvv[va, va, Va, Va], R.aaa.VVVooO, optimize=True)
    )
    dR.aaa.VvvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('AmiE,EcbmjK->AbcijK', H.aa.voov[Va, oa, oa, Va], R.aaa.VvvooO, optimize=True)
            + 1.0 * np.einsum('AMiE,EcbjMK->AbcijK', H.aa.voov[Va, Oa, oa, Va], R.aaa.VvvoOO, optimize=True)
    )
    dR.aaa.VvvooO += (4.0 / 4.0) * (
            -1.0 * np.einsum('bmie,AcemjK->AbcijK', H.aa.voov[va, oa, oa, va], R.aaa.VvvooO, optimize=True)
            + 1.0 * np.einsum('bMie,AcejMK->AbcijK', H.aa.voov[va, Oa, oa, va], R.aaa.VvvoOO, optimize=True)
            + 1.0 * np.einsum('bmiE,AEcmjK->AbcijK', H.aa.voov[va, oa, oa, Va], R.aaa.VVvooO, optimize=True)
            - 1.0 * np.einsum('bMiE,AEcjMK->AbcijK', H.aa.voov[va, Oa, oa, Va], R.aaa.VVvoOO, optimize=True)
    )
    dR.aaa.VvvooO += (1.0 / 4.0) * (
            +1.0 * np.einsum('AMKE,EcbjiM->AbcijK', H.aa.voov[Va, Oa, Oa, Va], R.aaa.VvvooO, optimize=True)
    )
    dR.aaa.VvvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('bMKe,AcejiM->AbcijK', H.aa.voov[va, Oa, Oa, va], R.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('bMKE,AEcjiM->AbcijK', H.aa.voov[va, Oa, Oa, Va], R.aaa.VVvooO, optimize=True)
    )
    dR.aaa.VvvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('AmiE,cbEjKm->AbcijK', H.ab.voov[Va, ob, oa, Vb], R.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('AMiE,cbEjKM->AbcijK', H.ab.voov[Va, Ob, oa, Vb], R.aab.vvVoOO, optimize=True)
    )
    dR.aaa.VvvooO += (4.0 / 4.0) * (
            -1.0 * np.einsum('bmie,AcejKm->AbcijK', H.ab.voov[va, ob, oa, vb], R.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('bMie,AcejKM->AbcijK', H.ab.voov[va, Ob, oa, vb], R.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('bmiE,AcEjKm->AbcijK', H.ab.voov[va, ob, oa, Vb], R.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('bMiE,AcEjKM->AbcijK', H.ab.voov[va, Ob, oa, Vb], R.aab.VvVoOO, optimize=True)
    )
    dR.aaa.VvvooO += (1.0 / 4.0) * (
            +1.0 * np.einsum('AMKE,cbEjiM->AbcijK', H.ab.voov[Va, Ob, Oa, Vb], R.aab.vvVooO, optimize=True)
    )
    dR.aaa.VvvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('bMKe,AcejiM->AbcijK', H.ab.voov[va, Ob, Oa, vb], R.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('bMKE,AcEjiM->AbcijK', H.ab.voov[va, Ob, Oa, Vb], R.aab.VvVooO, optimize=True)
    )

    dR.aaa.VvvooO -= np.transpose(dR.aaa.VvvooO, (0, 2, 1, 3, 4, 5))

    dR.aaa.VvvooO -= np.transpose(dR.aaa.VvvooO, (0, 1, 2, 4, 3, 5))

    return dR

def update(R, omega, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    R.aaa.VvvooO = eomcc_active_loops.update_r3a_100001(
        R.aaa.VvvooO,
        omega,
        H.a.oo[Oa, Oa],
        H.a.vv[Va, Va],
        H.a.oo[oa, oa],
        H.a.vv[va, va],
        0.0,
    )

    return R