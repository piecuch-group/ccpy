import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops

def build(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.bbb.VvvoOO = (4.0 / 4.0) * (
            -1.0 * np.einsum('bmJi,AcmK->AbciJK', X.bb.vooo[vb, :, Ob, ob], T.bb[Vb, vb, :, Ob], optimize=True)
    )
    dR.bbb.VvvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('AmJi,bcmK->AbciJK', X.bb.vooo[Vb, :, Ob, ob], T.bb[vb, vb, :, Ob], optimize=True)
    )
    dR.bbb.VvvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('bmJK,Acmi->AbciJK', X.bb.vooo[vb, :, Ob, Ob], T.bb[Vb, vb, :, ob], optimize=True)
    )
    dR.bbb.VvvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('AmJK,bcmi->AbciJK', X.bb.vooo[Vb, :, Ob, Ob], T.bb[vb, vb, :, ob], optimize=True)
    )
    dR.bbb.VvvoOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('bmJi,AcmK->AbciJK', H.bb.vooo[vb, :, Ob, ob], R.bb[Vb, vb, :, Ob], optimize=True)
    )
    dR.bbb.VvvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('AmJi,bcmK->AbciJK', H.bb.vooo[Vb, :, Ob, ob], R.bb[vb, vb, :, Ob], optimize=True)
    )
    dR.bbb.VvvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('bmJK,Acmi->AbciJK', H.bb.vooo[vb, :, Ob, Ob], R.bb[Vb, vb, :, ob], optimize=True)
    )
    dR.bbb.VvvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('AmJK,bcmi->AbciJK', H.bb.vooo[Vb, :, Ob, Ob], R.bb[vb, vb, :, ob], optimize=True)
    )
    dR.bbb.VvvoOO += (4.0 / 4.0) * (
            +1.0 * np.einsum('bAJe,eciK->AbciJK', X.bb.vvov[vb, Vb, Ob, :], T.bb[:, vb, ob, Ob], optimize=True)
    )
    dR.bbb.VvvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('bcJe,eAiK->AbciJK', X.bb.vvov[vb, vb, Ob, :], T.bb[:, Vb, ob, Ob], optimize=True)
    )
    dR.bbb.VvvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('bAie,ecJK->AbciJK', X.bb.vvov[vb, Vb, ob, :], T.bb[:, vb, Ob, Ob], optimize=True)
    )
    dR.bbb.VvvoOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('bcie,eAJK->AbciJK', X.bb.vvov[vb, vb, ob, :], T.bb[:, Vb, Ob, Ob], optimize=True)
    )
    dR.bbb.VvvoOO += (4.0 / 4.0) * (
            +1.0 * np.einsum('bAJe,eciK->AbciJK', H.bb.vvov[vb, Vb, Ob, :], R.bb[:, vb, ob, Ob], optimize=True)
    )
    dR.bbb.VvvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('bcJe,eAiK->AbciJK', H.bb.vvov[vb, vb, Ob, :], R.bb[:, Vb, ob, Ob], optimize=True)
    )
    dR.bbb.VvvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('bAie,ecJK->AbciJK', H.bb.vvov[vb, Vb, ob, :], R.bb[:, vb, Ob, Ob], optimize=True)
    )
    dR.bbb.VvvoOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('bcie,eAJK->AbciJK', H.bb.vvov[vb, vb, ob, :], R.bb[:, Vb, Ob, Ob], optimize=True)
    )
    # of terms =  16
    dR.bbb.VvvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('be,AceiJK->AbciJK', X.b.vv[vb, vb], T.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('bE,EAciJK->AbciJK', X.b.vv[vb, Vb], T.bbb.VVvoOO, optimize=True)
    )
    dR.bbb.VvvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('AE,EcbiJK->AbciJK', X.b.vv[Vb, Vb], T.bbb.VvvoOO, optimize=True)
    )
    dR.bbb.VvvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mJ,AcbimK->AbciJK', X.b.oo[ob, Ob], T.bbb.VvvooO, optimize=True)
            + 1.0 * np.einsum('MJ,AcbiMK->AbciJK', X.b.oo[Ob, Ob], T.bbb.VvvoOO, optimize=True)
    )
    dR.bbb.VvvoOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('mi,AcbmJK->AbciJK', X.b.oo[ob, ob], T.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('Mi,AcbJMK->AbciJK', X.b.oo[Ob, ob], T.bbb.VvvOOO, optimize=True)
    )
    dR.bbb.VvvoOO += (2.0 / 4.0) * (
            -0.5 * np.einsum('mniJ,AcbmnK->AbciJK', X.bb.oooo[ob, ob, ob, Ob], T.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('mNiJ,AcbmNK->AbciJK', X.bb.oooo[ob, Ob, ob, Ob], T.bbb.VvvoOO, optimize=True)
            - 0.5 * np.einsum('MNiJ,AcbMNK->AbciJK', X.bb.oooo[Ob, Ob, ob, Ob], T.bbb.VvvOOO, optimize=True)
    )
    dR.bbb.VvvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('mNKJ,AcbmiN->AbciJK', X.bb.oooo[ob, Ob, Ob, Ob], T.bbb.VvvooO, optimize=True)
            + 0.5 * np.einsum('MNKJ,AcbiMN->AbciJK', X.bb.oooo[Ob, Ob, Ob, Ob], T.bbb.VvvoOO, optimize=True)
    )
    dR.bbb.VvvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('AbeF,FceiJK->AbciJK', X.bb.vvvv[Vb, vb, vb, Vb], T.bbb.VvvoOO, optimize=True)
            - 0.5 * np.einsum('AbEF,FEciJK->AbciJK', X.bb.vvvv[Vb, vb, Vb, Vb], T.bbb.VVvoOO, optimize=True)
    )
    dR.bbb.VvvoOO += (1.0 / 4.0) * (
            +0.5 * np.einsum('cbef,AfeiJK->AbciJK', X.bb.vvvv[vb, vb, vb, vb], T.bbb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('cbeF,AFeiJK->AbciJK', X.bb.vvvv[vb, vb, vb, Vb], T.bbb.VVvoOO, optimize=True)
            + 0.5 * np.einsum('cbEF,AFEiJK->AbciJK', X.bb.vvvv[vb, vb, Vb, Vb], T.bbb.VVVoOO, optimize=True)
    )
    dR.bbb.VvvoOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('bmJe,AceimK->AbciJK', X.bb.voov[vb, ob, Ob, vb], T.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('bmJE,EAcimK->AbciJK', X.bb.voov[vb, ob, Ob, Vb], T.bbb.VVvooO, optimize=True)
            - 1.0 * np.einsum('bMJe,AceiMK->AbciJK', X.bb.voov[vb, Ob, Ob, vb], T.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('bMJE,EAciMK->AbciJK', X.bb.voov[vb, Ob, Ob, Vb], T.bbb.VVvoOO, optimize=True)
    )
    dR.bbb.VvvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('AmJE,EcbimK->AbciJK', X.bb.voov[Vb, ob, Ob, Vb], T.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('AMJE,EcbiMK->AbciJK', X.bb.voov[Vb, Ob, Ob, Vb], T.bbb.VvvoOO, optimize=True)
    )
    dR.bbb.VvvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('bmie,AcemJK->AbciJK', X.bb.voov[vb, ob, ob, vb], T.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('bmiE,EAcmJK->AbciJK', X.bb.voov[vb, ob, ob, Vb], T.bbb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('bMie,AceJMK->AbciJK', X.bb.voov[vb, Ob, ob, vb], T.bbb.VvvOOO, optimize=True)
            + 1.0 * np.einsum('bMiE,EAcJMK->AbciJK', X.bb.voov[vb, Ob, ob, Vb], T.bbb.VVvOOO, optimize=True)
    )
    dR.bbb.VvvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('AmiE,EcbmJK->AbciJK', X.bb.voov[Vb, ob, ob, Vb], T.bbb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('AMiE,EcbJMK->AbciJK', X.bb.voov[Vb, Ob, ob, Vb], T.bbb.VvvOOO, optimize=True)
    )
    dR.bbb.VvvoOO += (4.0 / 4.0) * (
            +1.0 * np.einsum('mbeJ,eAcmiK->AbciJK', X.ab.ovvo[oa, vb, va, Ob], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('mbEJ,EAcmiK->AbciJK', X.ab.ovvo[oa, vb, Va, Ob], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('MbeJ,eAcMiK->AbciJK', X.ab.ovvo[Oa, vb, va, Ob], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('MbEJ,EAcMiK->AbciJK', X.ab.ovvo[Oa, vb, Va, Ob], T.abb.VVvOoO, optimize=True)
    )
    dR.bbb.VvvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('mAEJ,EbcmiK->AbciJK', X.ab.ovvo[oa, Vb, Va, Ob], T.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('MAEJ,EbcMiK->AbciJK', X.ab.ovvo[Oa, Vb, Va, Ob], T.abb.VvvOoO, optimize=True)
    )
    dR.bbb.VvvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mbei,eAcmKJ->AbciJK', X.ab.ovvo[oa, vb, va, ob], T.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('mbEi,EAcmKJ->AbciJK', X.ab.ovvo[oa, vb, Va, ob], T.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('Mbei,eAcMKJ->AbciJK', X.ab.ovvo[Oa, vb, va, ob], T.abb.vVvOOO, optimize=True)
            + 1.0 * np.einsum('MbEi,EAcMKJ->AbciJK', X.ab.ovvo[Oa, vb, Va, ob], T.abb.VVvOOO, optimize=True)
    )
    dR.bbb.VvvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('mAEi,EbcmKJ->AbciJK', X.ab.ovvo[oa, Vb, Va, ob], T.abb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('MAEi,EbcMKJ->AbciJK', X.ab.ovvo[Oa, Vb, Va, ob], T.abb.VvvOOO, optimize=True)
    )
    dR.bbb.VvvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mJ,AcbimK->AbciJK', H.b.oo[ob, Ob], R.bbb.VvvooO, optimize=True)
            + 1.0 * np.einsum('MJ,AcbiMK->AbciJK', H.b.oo[Ob, Ob], R.bbb.VvvoOO, optimize=True)
    )
    dR.bbb.VvvoOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('mi,AcbmJK->AbciJK', H.b.oo[ob, ob], R.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('Mi,AcbJMK->AbciJK', H.b.oo[Ob, ob], R.bbb.VvvOOO, optimize=True)
    )
    dR.bbb.VvvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('be,AceiJK->AbciJK', H.b.vv[vb, vb], R.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('bE,EAciJK->AbciJK', H.b.vv[vb, Vb], R.bbb.VVvoOO, optimize=True)
    )
    dR.bbb.VvvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('AE,EcbiJK->AbciJK', H.b.vv[Vb, Vb], R.bbb.VvvoOO, optimize=True)
    )
    dR.bbb.VvvoOO += (2.0 / 4.0) * (
            -0.5 * np.einsum('mniJ,AcbmnK->AbciJK', H.bb.oooo[ob, ob, ob, Ob], R.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('mNiJ,AcbmNK->AbciJK', H.bb.oooo[ob, Ob, ob, Ob], R.bbb.VvvoOO, optimize=True)
            - 0.5 * np.einsum('MNiJ,AcbMNK->AbciJK', H.bb.oooo[Ob, Ob, ob, Ob], R.bbb.VvvOOO, optimize=True)
    )
    dR.bbb.VvvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('mNKJ,AcbmiN->AbciJK', H.bb.oooo[ob, Ob, Ob, Ob], R.bbb.VvvooO, optimize=True)
            + 0.5 * np.einsum('MNKJ,AcbiMN->AbciJK', H.bb.oooo[Ob, Ob, Ob, Ob], R.bbb.VvvoOO, optimize=True)
    )
    dR.bbb.VvvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('AbeF,FceiJK->AbciJK', H.bb.vvvv[Vb, vb, vb, Vb], R.bbb.VvvoOO, optimize=True)
            - 0.5 * np.einsum('AbEF,FEciJK->AbciJK', H.bb.vvvv[Vb, vb, Vb, Vb], R.bbb.VVvoOO, optimize=True)
    )
    dR.bbb.VvvoOO += (1.0 / 4.0) * (
            +0.5 * np.einsum('cbef,AfeiJK->AbciJK', H.bb.vvvv[vb, vb, vb, vb], R.bbb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('cbeF,AFeiJK->AbciJK', H.bb.vvvv[vb, vb, vb, Vb], R.bbb.VVvoOO, optimize=True)
            + 0.5 * np.einsum('cbEF,AFEiJK->AbciJK', H.bb.vvvv[vb, vb, Vb, Vb], R.bbb.VVVoOO, optimize=True)
    )
    dR.bbb.VvvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('AmiE,EcbmJK->AbciJK', H.bb.voov[Vb, ob, ob, Vb], R.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('AMiE,EcbMJK->AbciJK', H.bb.voov[Vb, Ob, ob, Vb], R.bbb.VvvOOO, optimize=True)
    )
    dR.bbb.VvvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('bmie,AcemJK->AbciJK', H.bb.voov[vb, ob, ob, vb], R.bbb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('bmiE,AEcmJK->AbciJK', H.bb.voov[vb, ob, ob, Vb], R.bbb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('bMie,AceMJK->AbciJK', H.bb.voov[vb, Ob, ob, vb], R.bbb.VvvOOO, optimize=True)
            + 1.0 * np.einsum('bMiE,AEcMJK->AbciJK', H.bb.voov[vb, Ob, ob, Vb], R.bbb.VVvOOO, optimize=True)
    )
    dR.bbb.VvvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('AmJE,EcbmiK->AbciJK', H.bb.voov[Vb, ob, Ob, Vb], R.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('AMJE,EcbiMK->AbciJK', H.bb.voov[Vb, Ob, Ob, Vb], R.bbb.VvvoOO, optimize=True)
    )
    dR.bbb.VvvoOO += (4.0 / 4.0) * (
            +1.0 * np.einsum('bmJe,AcemiK->AbciJK', H.bb.voov[vb, ob, Ob, vb], R.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('bmJE,AEcmiK->AbciJK', H.bb.voov[vb, ob, Ob, Vb], R.bbb.VVvooO, optimize=True)
            - 1.0 * np.einsum('bMJe,AceiMK->AbciJK', H.bb.voov[vb, Ob, Ob, vb], R.bbb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('bMJE,AEciMK->AbciJK', H.bb.voov[vb, Ob, Ob, Vb], R.bbb.VVvoOO, optimize=True)
    )
    dR.bbb.VvvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('mAEi,EbcmKJ->AbciJK', H.ab.ovvo[oa, Vb, Va, ob], R.abb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('MAEi,EbcMKJ->AbciJK', H.ab.ovvo[Oa, Vb, Va, ob], R.abb.VvvOOO, optimize=True)
    )
    dR.bbb.VvvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mbei,eAcmKJ->AbciJK', H.ab.ovvo[oa, vb, va, ob], R.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('mbEi,EAcmKJ->AbciJK', H.ab.ovvo[oa, vb, Va, ob], R.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('Mbei,eAcMKJ->AbciJK', H.ab.ovvo[Oa, vb, va, ob], R.abb.vVvOOO, optimize=True)
            + 1.0 * np.einsum('MbEi,EAcMKJ->AbciJK', H.ab.ovvo[Oa, vb, Va, ob], R.abb.VVvOOO, optimize=True)
    )
    dR.bbb.VvvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('mAEJ,EbcmiK->AbciJK', H.ab.ovvo[oa, Vb, Va, Ob], R.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('MAEJ,EbcMiK->AbciJK', H.ab.ovvo[Oa, Vb, Va, Ob], R.abb.VvvOoO, optimize=True)
    )
    dR.bbb.VvvoOO += (4.0 / 4.0) * (
            +1.0 * np.einsum('mbeJ,eAcmiK->AbciJK', H.ab.ovvo[oa, vb, va, Ob], R.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('mbEJ,EAcmiK->AbciJK', H.ab.ovvo[oa, vb, Va, Ob], R.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('MbeJ,eAcMiK->AbciJK', H.ab.ovvo[Oa, vb, va, Ob], R.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('MbEJ,EAcMiK->AbciJK', H.ab.ovvo[Oa, vb, Va, Ob], R.abb.VVvOoO, optimize=True)
    )
    # of terms =  32

    dR.bbb.VvvoOO -= np.transpose(dR.bbb.VvvoOO, (0, 1, 2, 3, 5, 4))
    dR.bbb.VvvoOO -= np.transpose(dR.bbb.VvvoOO, (0, 2, 1, 3, 4, 5))

    return dR

def update(R, omega, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    R.bbb.VvvoOO = eomcc_active_loops.update_r3d_100011(
        R.bbb.VvvoOO,
        omega,
        H.b.oo[Ob, Ob],
        H.b.vv[Vb, Vb],
        H.b.oo[ob, ob],
        H.b.vv[vb, vb],
        0.0,
    )
    return R
