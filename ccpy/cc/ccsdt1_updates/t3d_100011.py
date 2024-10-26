import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import cc_active_loops

def build(T, dT, H, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    # MM(2,3)
    dT.bbb.VvvoOO = (2.0 / 4.0) * (
            -1.0 * np.einsum('AmiJ,bcmK->AbciJK', H.bb.vooo[Vb, :, ob, Ob], T.bb[vb, vb, :, Ob], optimize=True)
    )
    dT.bbb.VvvoOO += (4.0 / 4.0) * (
            +1.0 * np.einsum('bmiJ,AcmK->AbciJK', H.bb.vooo[vb, :, ob, Ob], T.bb[Vb, vb, :, Ob], optimize=True)
    )
    dT.bbb.VvvoOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('AmKJ,bcmi->AbciJK', H.bb.vooo[Vb, :, Ob, Ob], T.bb[vb, vb, :, ob], optimize=True)
    )
    dT.bbb.VvvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('bmKJ,Acmi->AbciJK', H.bb.vooo[vb, :, Ob, Ob], T.bb[Vb, vb, :, ob], optimize=True)
    )
    dT.bbb.VvvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('Abie,ecJK->AbciJK', H.bb.vvov[Vb, vb, ob, :], T.bb[:, vb, Ob, Ob], optimize=True)
    )
    dT.bbb.VvvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('cbie,eAJK->AbciJK', H.bb.vvov[vb, vb, ob, :], T.bb[:, Vb, Ob, Ob], optimize=True)
    )
    dT.bbb.VvvoOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('AbJe,eciK->AbciJK', H.bb.vvov[Vb, vb, Ob, :], T.bb[:, vb, ob, Ob], optimize=True)
    )
    dT.bbb.VvvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('cbJe,eAiK->AbciJK', H.bb.vvov[vb, vb, Ob, :], T.bb[:, Vb, ob, Ob], optimize=True)
    )
    # (H(2) * T3)_C
    dT.bbb.VvvoOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('mi,AcbmJK->AbciJK', H.b.oo[ob, ob], T.bbb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('Mi,AcbMJK->AbciJK', H.b.oo[Ob, ob], T.bbb.VvvOOO, optimize=True)
    )
    dT.bbb.VvvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('mJ,AcbmiK->AbciJK', H.b.oo[ob, Ob], T.bbb.VvvooO, optimize=True)
            + 1.0 * np.einsum('MJ,AcbiMK->AbciJK', H.b.oo[Ob, Ob], T.bbb.VvvoOO, optimize=True)
    )
    dT.bbb.VvvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('AE,EcbiJK->AbciJK', H.b.vv[Vb, Vb], T.bbb.VvvoOO, optimize=True)
    )
    dT.bbb.VvvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('be,AceiJK->AbciJK', H.b.vv[vb, vb], T.bbb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('bE,AEciJK->AbciJK', H.b.vv[vb, Vb], T.bbb.VVvoOO, optimize=True)
    )
    dT.bbb.VvvoOO += (2.0 / 4.0) * (
            -0.5 * np.einsum('mniJ,AcbmnK->AbciJK', H.bb.oooo[ob, ob, ob, Ob], T.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('mNiJ,AcbmNK->AbciJK', H.bb.oooo[ob, Ob, ob, Ob], T.bbb.VvvoOO, optimize=True)
            - 0.5 * np.einsum('MNiJ,AcbMNK->AbciJK', H.bb.oooo[Ob, Ob, ob, Ob], T.bbb.VvvOOO, optimize=True)
    )
    dT.bbb.VvvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('mNKJ,AcbmiN->AbciJK', H.bb.oooo[ob, Ob, Ob, Ob], T.bbb.VvvooO, optimize=True)
            + 0.5 * np.einsum('MNKJ,AcbiMN->AbciJK', H.bb.oooo[Ob, Ob, Ob, Ob], T.bbb.VvvoOO, optimize=True)
    )
    dT.bbb.VvvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('AbEf,EcfiJK->AbciJK', H.bb.vvvv[Vb, vb, Vb, vb], T.bbb.VvvoOO, optimize=True)
            - 0.5 * np.einsum('AbEF,FEciJK->AbciJK', H.bb.vvvv[Vb, vb, Vb, Vb], T.bbb.VVvoOO, optimize=True)
    )
    dT.bbb.VvvoOO += (1.0 / 4.0) * (
            +0.5 * np.einsum('cbef,AfeiJK->AbciJK', H.bb.vvvv[vb, vb, vb, vb], T.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('cbEf,AEfiJK->AbciJK', H.bb.vvvv[vb, vb, Vb, vb], T.bbb.VVvoOO, optimize=True)
            + 0.5 * np.einsum('cbEF,AFEiJK->AbciJK', H.bb.vvvv[vb, vb, Vb, Vb], T.bbb.VVVoOO, optimize=True)
    )
    dT.bbb.VvvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('AmiE,EcbmJK->AbciJK', H.bb.voov[Vb, ob, ob, Vb], T.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('AMiE,EcbMJK->AbciJK', H.bb.voov[Vb, Ob, ob, Vb], T.bbb.VvvOOO, optimize=True)
    )
    dT.bbb.VvvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('bmie,AcemJK->AbciJK', H.bb.voov[vb, ob, ob, vb], T.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('bMie,AceMJK->AbciJK', H.bb.voov[vb, Ob, ob, vb], T.bbb.VvvOOO, optimize=True)
            + 1.0 * np.einsum('bmiE,AEcmJK->AbciJK', H.bb.voov[vb, ob, ob, Vb], T.bbb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('bMiE,AEcMJK->AbciJK', H.bb.voov[vb, Ob, ob, Vb], T.bbb.VVvOOO, optimize=True)
    )
    dT.bbb.VvvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('AmJE,EcbmiK->AbciJK', H.bb.voov[Vb, ob, Ob, Vb], T.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('AMJE,EcbiMK->AbciJK', H.bb.voov[Vb, Ob, Ob, Vb], T.bbb.VvvoOO, optimize=True)
    )
    dT.bbb.VvvoOO += (4.0 / 4.0) * (
            +1.0 * np.einsum('bmJe,AcemiK->AbciJK', H.bb.voov[vb, ob, Ob, vb], T.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('bMJe,AceiMK->AbciJK', H.bb.voov[vb, Ob, Ob, vb], T.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('bmJE,AEcmiK->AbciJK', H.bb.voov[vb, ob, Ob, Vb], T.bbb.VVvooO, optimize=True)
            + 1.0 * np.einsum('bMJE,AEciMK->AbciJK', H.bb.voov[vb, Ob, Ob, Vb], T.bbb.VVvoOO, optimize=True)
    )
    dT.bbb.VvvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('mAEi,EcbmJK->AbciJK', H.ab.ovvo[oa, Vb, Va, ob], T.abb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('MAEi,EcbMJK->AbciJK', H.ab.ovvo[Oa, Vb, Va, ob], T.abb.VvvOOO, optimize=True)
    )
    dT.bbb.VvvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('mbei,eAcmJK->AbciJK', H.ab.ovvo[oa, vb, va, ob], T.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('Mbei,eAcMJK->AbciJK', H.ab.ovvo[Oa, vb, va, ob], T.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('mbEi,EAcmJK->AbciJK', H.ab.ovvo[oa, vb, Va, ob], T.abb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('MbEi,EAcMJK->AbciJK', H.ab.ovvo[Oa, vb, Va, ob], T.abb.VVvOOO, optimize=True)
    )
    dT.bbb.VvvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mAEJ,EcbmiK->AbciJK', H.ab.ovvo[oa, Vb, Va, Ob], T.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('MAEJ,EcbMiK->AbciJK', H.ab.ovvo[Oa, Vb, Va, Ob], T.abb.VvvOoO, optimize=True)
    )
    dT.bbb.VvvoOO += (4.0 / 4.0) * (
            +1.0 * np.einsum('mbeJ,eAcmiK->AbciJK', H.ab.ovvo[oa, vb, va, Ob], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('MbeJ,eAcMiK->AbciJK', H.ab.ovvo[Oa, vb, va, Ob], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('mbEJ,EAcmiK->AbciJK', H.ab.ovvo[oa, vb, Va, Ob], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('MbEJ,EAcMiK->AbciJK', H.ab.ovvo[Oa, vb, Va, Ob], T.abb.VVvOoO, optimize=True)
    )

    dT.bbb.VvvoOO -= np.transpose(dT.bbb.VvvoOO, (0, 2, 1, 3, 4, 5))
    dT.bbb.VvvoOO -= np.transpose(dT.bbb.VvvoOO, (0, 1, 2, 3, 5, 4))

    return dT

def update(T, dT, H, shift, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.bbb.VvvoOO, dT.bbb.VvvoOO = cc_active_loops.update_t3d_100011(
        T.bbb.VvvoOO,
        dT.bbb.VvvoOO,
        H.b.oo[Ob, Ob],
        H.b.vv[Vb, Vb],
        H.b.oo[ob, ob],
        H.b.vv[vb, vb],
        shift,
    )

    return T, dT