import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops

def build(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.aaa.VvvoOO = (4.0 / 4.0) * (
            -1.0 * np.einsum('bmJi,AcmK->AbciJK', X.aa.vooo[va, :, Oa, oa], T.aa[Va, va, :, Oa], optimize=True)
    )
    dR.aaa.VvvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('AmJi,bcmK->AbciJK', X.aa.vooo[Va, :, Oa, oa], T.aa[va, va, :, Oa], optimize=True)
    )
    dR.aaa.VvvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('bmJK,Acmi->AbciJK', X.aa.vooo[va, :, Oa, Oa], T.aa[Va, va, :, oa], optimize=True)
    )
    dR.aaa.VvvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('AmJK,bcmi->AbciJK', X.aa.vooo[Va, :, Oa, Oa], T.aa[va, va, :, oa], optimize=True)
    )
    dR.aaa.VvvoOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('bmJi,AcmK->AbciJK', H.aa.vooo[va, :, Oa, oa], R.aa[Va, va, :, Oa], optimize=True)
    )
    dR.aaa.VvvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('AmJi,bcmK->AbciJK', H.aa.vooo[Va, :, Oa, oa], R.aa[va, va, :, Oa], optimize=True)
    )
    dR.aaa.VvvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('bmJK,Acmi->AbciJK', H.aa.vooo[va, :, Oa, Oa], R.aa[Va, va, :, oa], optimize=True)
    )
    dR.aaa.VvvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('AmJK,bcmi->AbciJK', H.aa.vooo[Va, :, Oa, Oa], R.aa[va, va, :, oa], optimize=True)
    )
    dR.aaa.VvvoOO += (4.0 / 4.0) * (
            +1.0 * np.einsum('bAJe,eciK->AbciJK', X.aa.vvov[va, Va, Oa, :], T.aa[:, va, oa, Oa], optimize=True)
    )
    dR.aaa.VvvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('bcJe,eAiK->AbciJK', X.aa.vvov[va, va, Oa, :], T.aa[:, Va, oa, Oa], optimize=True)
    )
    dR.aaa.VvvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('bAie,ecJK->AbciJK', X.aa.vvov[va, Va, oa, :], T.aa[:, va, Oa, Oa], optimize=True)
    )
    dR.aaa.VvvoOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('bcie,eAJK->AbciJK', X.aa.vvov[va, va, oa, :], T.aa[:, Va, Oa, Oa], optimize=True)
    )
    dR.aaa.VvvoOO += (4.0 / 4.0) * (
            +1.0 * np.einsum('bAJe,eciK->AbciJK', H.aa.vvov[va, Va, Oa, :], R.aa[:, va, oa, Oa], optimize=True)
    )
    dR.aaa.VvvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('bcJe,eAiK->AbciJK', H.aa.vvov[va, va, Oa, :], R.aa[:, Va, oa, Oa], optimize=True)
    )
    dR.aaa.VvvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('bAie,ecJK->AbciJK', H.aa.vvov[va, Va, oa, :], R.aa[:, va, Oa, Oa], optimize=True)
    )
    dR.aaa.VvvoOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('bcie,eAJK->AbciJK', H.aa.vvov[va, va, oa, :], R.aa[:, Va, Oa, Oa], optimize=True)
    )

    dR.aaa.VvvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('be,AceiJK->AbciJK', X.a.vv[va, va], T.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('bE,EAciJK->AbciJK', X.a.vv[va, Va], T.aaa.VVvoOO, optimize=True)
    )
    dR.aaa.VvvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('AE,EcbiJK->AbciJK', X.a.vv[Va, Va], T.aaa.VvvoOO, optimize=True)
    )
    dR.aaa.VvvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mJ,AcbimK->AbciJK', X.a.oo[oa, Oa], T.aaa.VvvooO, optimize=True)
            + 1.0 * np.einsum('MJ,AcbiMK->AbciJK', X.a.oo[Oa, Oa], T.aaa.VvvoOO, optimize=True)
    )
    dR.aaa.VvvoOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('mi,AcbmJK->AbciJK', X.a.oo[oa, oa], T.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('Mi,AcbJMK->AbciJK', X.a.oo[Oa, oa], T.aaa.VvvOOO, optimize=True)
    )
    dR.aaa.VvvoOO += (2.0 / 4.0) * (
            -0.5 * np.einsum('mniJ,AcbmnK->AbciJK', X.aa.oooo[oa, oa, oa, Oa], T.aaa.VvvooO, optimize=True)
            + 1.0 * np.einsum('MniJ,AcbnMK->AbciJK', X.aa.oooo[Oa, oa, oa, Oa], T.aaa.VvvoOO, optimize=True)
            - 0.5 * np.einsum('MNiJ,AcbMNK->AbciJK', X.aa.oooo[Oa, Oa, oa, Oa], T.aaa.VvvOOO, optimize=True)
    )
    dR.aaa.VvvoOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('MnKJ,AcbniM->AbciJK', X.aa.oooo[Oa, oa, Oa, Oa], T.aaa.VvvooO, optimize=True)
            + 0.5 * np.einsum('MNKJ,AcbiMN->AbciJK', X.aa.oooo[Oa, Oa, Oa, Oa], T.aaa.VvvoOO, optimize=True)
    )
    dR.aaa.VvvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('AbeF,FceiJK->AbciJK', X.aa.vvvv[Va, va, va, Va], T.aaa.VvvoOO, optimize=True)
            - 0.5 * np.einsum('AbEF,FEciJK->AbciJK', X.aa.vvvv[Va, va, Va, Va], T.aaa.VVvoOO, optimize=True)
    )
    dR.aaa.VvvoOO += (1.0 / 4.0) * (
            +0.5 * np.einsum('cbef,AfeiJK->AbciJK', X.aa.vvvv[va, va, va, va], T.aaa.VvvoOO, optimize=True)
            + 1.0 * np.einsum('cbeF,AFeiJK->AbciJK', X.aa.vvvv[va, va, va, Va], T.aaa.VVvoOO, optimize=True)
            + 0.5 * np.einsum('cbEF,AFEiJK->AbciJK', X.aa.vvvv[va, va, Va, Va], T.aaa.VVVoOO, optimize=True)
    )
    dR.aaa.VvvoOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('bmJe,AceimK->AbciJK', X.aa.voov[va, oa, Oa, va], T.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('bmJE,EAcimK->AbciJK', X.aa.voov[va, oa, Oa, Va], T.aaa.VVvooO, optimize=True)
            - 1.0 * np.einsum('bMJe,AceiMK->AbciJK', X.aa.voov[va, Oa, Oa, va], T.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('bMJE,EAciMK->AbciJK', X.aa.voov[va, Oa, Oa, Va], T.aaa.VVvoOO, optimize=True)
    )
    dR.aaa.VvvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('AmJE,EcbimK->AbciJK', X.aa.voov[Va, oa, Oa, Va], T.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('AMJE,EcbiMK->AbciJK', X.aa.voov[Va, Oa, Oa, Va], T.aaa.VvvoOO, optimize=True)
    )
    dR.aaa.VvvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('bmie,AcemJK->AbciJK', X.aa.voov[va, oa, oa, va], T.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('bmiE,EAcmJK->AbciJK', X.aa.voov[va, oa, oa, Va], T.aaa.VVvoOO, optimize=True)
            + 1.0 * np.einsum('bMie,AceJMK->AbciJK', X.aa.voov[va, Oa, oa, va], T.aaa.VvvOOO, optimize=True)
            + 1.0 * np.einsum('bMiE,EAcJMK->AbciJK', X.aa.voov[va, Oa, oa, Va], T.aaa.VVvOOO, optimize=True)
    )
    dR.aaa.VvvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('AmiE,EcbmJK->AbciJK', X.aa.voov[Va, oa, oa, Va], T.aaa.VvvoOO, optimize=True)
            + 1.0 * np.einsum('AMiE,EcbJMK->AbciJK', X.aa.voov[Va, Oa, oa, Va], T.aaa.VvvOOO, optimize=True)
    )
    dR.aaa.VvvoOO += (4.0 / 4.0) * (
            +1.0 * np.einsum('bmJe,AceiKm->AbciJK', X.ab.voov[va, ob, Oa, vb], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('bmJE,AcEiKm->AbciJK', X.ab.voov[va, ob, Oa, Vb], T.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('bMJe,AceiKM->AbciJK', X.ab.voov[va, Ob, Oa, vb], T.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('bMJE,AcEiKM->AbciJK', X.ab.voov[va, Ob, Oa, Vb], T.aab.VvVoOO, optimize=True)
    )
    dR.aaa.VvvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('AmJE,cbEiKm->AbciJK', X.ab.voov[Va, ob, Oa, Vb], T.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('AMJE,cbEiKM->AbciJK', X.ab.voov[Va, Ob, Oa, Vb], T.aab.vvVoOO, optimize=True)
    )
    dR.aaa.VvvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('bmie,AceJKm->AbciJK', X.ab.voov[va, ob, oa, vb], T.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('bmiE,AcEJKm->AbciJK', X.ab.voov[va, ob, oa, Vb], T.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('bMie,AceJKM->AbciJK', X.ab.voov[va, Ob, oa, vb], T.aab.VvvOOO, optimize=True)
            - 1.0 * np.einsum('bMiE,AcEJKM->AbciJK', X.ab.voov[va, Ob, oa, Vb], T.aab.VvVOOO, optimize=True)
    )
    dR.aaa.VvvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('AmiE,cbEJKm->AbciJK', X.ab.voov[Va, ob, oa, Vb], T.aab.vvVOOo, optimize=True)
            - 1.0 * np.einsum('AMiE,cbEJKM->AbciJK', X.ab.voov[Va, Ob, oa, Vb], T.aab.vvVOOO, optimize=True)
    )
    dR.aaa.VvvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mJ,AcbimK->AbciJK', H.a.oo[oa, Oa], R.aaa.VvvooO, optimize=True)
            + 1.0 * np.einsum('MJ,AcbiMK->AbciJK', H.a.oo[Oa, Oa], R.aaa.VvvoOO, optimize=True)
    )
    dR.aaa.VvvoOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('mi,AcbmJK->AbciJK', H.a.oo[oa, oa], R.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('Mi,AcbJMK->AbciJK', H.a.oo[Oa, oa], R.aaa.VvvOOO, optimize=True)
    )
    dR.aaa.VvvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('be,AceiJK->AbciJK', H.a.vv[va, va], R.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('bE,EAciJK->AbciJK', H.a.vv[va, Va], R.aaa.VVvoOO, optimize=True)
    )
    dR.aaa.VvvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('AE,EcbiJK->AbciJK', H.a.vv[Va, Va], R.aaa.VvvoOO, optimize=True)
    )
    dR.aaa.VvvoOO += (2.0 / 4.0) * (
            -0.5 * np.einsum('mniJ,AcbmnK->AbciJK', H.aa.oooo[oa, oa, oa, Oa], R.aaa.VvvooO, optimize=True)
            + 1.0 * np.einsum('MniJ,AcbnMK->AbciJK', H.aa.oooo[Oa, oa, oa, Oa], R.aaa.VvvoOO, optimize=True)
            - 0.5 * np.einsum('MNiJ,AcbMNK->AbciJK', H.aa.oooo[Oa, Oa, oa, Oa], R.aaa.VvvOOO, optimize=True)
    )
    dR.aaa.VvvoOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('MnKJ,AcbniM->AbciJK', H.aa.oooo[Oa, oa, Oa, Oa], R.aaa.VvvooO, optimize=True)
            + 0.5 * np.einsum('MNKJ,AcbiMN->AbciJK', H.aa.oooo[Oa, Oa, Oa, Oa], R.aaa.VvvoOO, optimize=True)
    )
    dR.aaa.VvvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('AbeF,FceiJK->AbciJK', H.aa.vvvv[Va, va, va, Va], R.aaa.VvvoOO, optimize=True)
            - 0.5 * np.einsum('AbEF,FEciJK->AbciJK', H.aa.vvvv[Va, va, Va, Va], R.aaa.VVvoOO, optimize=True)
    )
    dR.aaa.VvvoOO += (1.0 / 4.0) * (
            +0.5 * np.einsum('cbef,AfeiJK->AbciJK', H.aa.vvvv[va, va, va, va], R.aaa.VvvoOO, optimize=True)
            + 1.0 * np.einsum('cbeF,AFeiJK->AbciJK', H.aa.vvvv[va, va, va, Va], R.aaa.VVvoOO, optimize=True)
            + 0.5 * np.einsum('cbEF,AFEiJK->AbciJK', H.aa.vvvv[va, va, Va, Va], R.aaa.VVVoOO, optimize=True)
    )
    dR.aaa.VvvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('AmiE,EcbmJK->AbciJK', H.aa.voov[Va, oa, oa, Va], R.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('AMiE,EcbMJK->AbciJK', H.aa.voov[Va, Oa, oa, Va], R.aaa.VvvOOO, optimize=True)
    )
    dR.aaa.VvvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('bmie,AcemJK->AbciJK', H.aa.voov[va, oa, oa, va], R.aaa.VvvoOO, optimize=True)
            + 1.0 * np.einsum('bmiE,AEcmJK->AbciJK', H.aa.voov[va, oa, oa, Va], R.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('bMie,AceMJK->AbciJK', H.aa.voov[va, Oa, oa, va], R.aaa.VvvOOO, optimize=True)
            + 1.0 * np.einsum('bMiE,AEcMJK->AbciJK', H.aa.voov[va, Oa, oa, Va], R.aaa.VVvOOO, optimize=True)
    )
    dR.aaa.VvvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('AmJE,EcbmiK->AbciJK', H.aa.voov[Va, oa, Oa, Va], R.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('AMJE,EcbiMK->AbciJK', H.aa.voov[Va, Oa, Oa, Va], R.aaa.VvvoOO, optimize=True)
    )
    dR.aaa.VvvoOO += (4.0 / 4.0) * (
            +1.0 * np.einsum('bmJe,AcemiK->AbciJK', H.aa.voov[va, oa, Oa, va], R.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('bmJE,AEcmiK->AbciJK', H.aa.voov[va, oa, Oa, Va], R.aaa.VVvooO, optimize=True)
            - 1.0 * np.einsum('bMJe,AceiMK->AbciJK', H.aa.voov[va, Oa, Oa, va], R.aaa.VvvoOO, optimize=True)
            + 1.0 * np.einsum('bMJE,AEciMK->AbciJK', H.aa.voov[va, Oa, Oa, Va], R.aaa.VVvoOO, optimize=True)
    )
    dR.aaa.VvvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('AmiE,cbEJKm->AbciJK', H.ab.voov[Va, ob, oa, Vb], R.aab.vvVOOo, optimize=True)
            - 1.0 * np.einsum('AMiE,cbEJKM->AbciJK', H.ab.voov[Va, Ob, oa, Vb], R.aab.vvVOOO, optimize=True)
    )
    dR.aaa.VvvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('bmie,AceJKm->AbciJK', H.ab.voov[va, ob, oa, vb], R.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('bmiE,AcEJKm->AbciJK', H.ab.voov[va, ob, oa, Vb], R.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('bMie,AceJKM->AbciJK', H.ab.voov[va, Ob, oa, vb], R.aab.VvvOOO, optimize=True)
            - 1.0 * np.einsum('bMiE,AcEJKM->AbciJK', H.ab.voov[va, Ob, oa, Vb], R.aab.VvVOOO, optimize=True)
    )
    dR.aaa.VvvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('AmJE,cbEiKm->AbciJK', H.ab.voov[Va, ob, Oa, Vb], R.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('AMJE,cbEiKM->AbciJK', H.ab.voov[Va, Ob, Oa, Vb], R.aab.vvVoOO, optimize=True)
    )
    dR.aaa.VvvoOO += (4.0 / 4.0) * (
            +1.0 * np.einsum('bmJe,AceiKm->AbciJK', H.ab.voov[va, ob, Oa, vb], R.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('bmJE,AcEiKm->AbciJK', H.ab.voov[va, ob, Oa, Vb], R.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('bMJe,AceiKM->AbciJK', H.ab.voov[va, Ob, Oa, vb], R.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('bMJE,AcEiKM->AbciJK', H.ab.voov[va, Ob, Oa, Vb], R.aab.VvVoOO, optimize=True)
    )

    dR.aaa.VvvoOO -= np.transpose(dR.aaa.VvvoOO, (0, 2, 1, 3, 4, 5))

    dR.aaa.VvvoOO -= np.transpose(dR.aaa.VvvoOO, (0, 1, 2, 3, 5, 4))


    return dR

def update(R, omega, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    R.aaa.VvvoOO = eomcc_active_loops.update_r3a_100011(
        R.aaa.VvvoOO,
        omega,
        H.a.oo[Oa, Oa],
        H.a.vv[Va, Va],
        H.a.oo[oa, oa],
        H.a.vv[va, va],
        0.0,
    )

    return R