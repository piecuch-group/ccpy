import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops

def build(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.bbb.VVVooO = (3.0 / 12.0) * (
            -1.0 * np.einsum('Bmji,ACmK->ABCijK', X.bb.vooo[Vb, :, ob, ob], T.bb[Vb, Vb, :, Ob], optimize=True)
    )
    dR.bbb.VVVooO += (6.0 / 12.0) * (
            +1.0 * np.einsum('BmjK,ACmi->ABCijK', X.bb.vooo[Vb, :, ob, Ob], T.bb[Vb, Vb, :, ob], optimize=True)
    )
    dR.bbb.VVVooO += (3.0 / 12.0) * (
            -1.0 * np.einsum('Bmji,ACmK->ABCijK', H.bb.vooo[Vb, :, ob, ob], R.bb[Vb, Vb, :, Ob], optimize=True)
    )
    dR.bbb.VVVooO += (6.0 / 12.0) * (
            +1.0 * np.einsum('BmjK,ACmi->ABCijK', H.bb.vooo[Vb, :, ob, Ob], R.bb[Vb, Vb, :, ob], optimize=True)
    )
    dR.bbb.VVVooO += (6.0 / 12.0) * (
            +1.0 * np.einsum('BAje,eCiK->ABCijK', X.bb.vvov[Vb, Vb, ob, :], T.bb[:, Vb, ob, Ob], optimize=True)
    )
    dR.bbb.VVVooO += (3.0 / 12.0) * (
            -1.0 * np.einsum('BAKe,eCij->ABCijK', X.bb.vvov[Vb, Vb, Ob, :], T.bb[:, Vb, ob, ob], optimize=True)
    )
    dR.bbb.VVVooO += (6.0 / 12.0) * (
            +1.0 * np.einsum('BAje,eCiK->ABCijK', H.bb.vvov[Vb, Vb, ob, :], R.bb[:, Vb, ob, Ob], optimize=True)
    )
    dR.bbb.VVVooO += (3.0 / 12.0) * (
            -1.0 * np.einsum('BAKe,eCij->ABCijK', H.bb.vvov[Vb, Vb, Ob, :], R.bb[:, Vb, ob, ob], optimize=True)
    )
    # of terms =  8
    dR.bbb.VVVooO += (3.0 / 12.0) * (
            +1.0 * np.einsum('Be,CAeijK->ABCijK', X.b.vv[Vb, vb], T.bbb.VVvooO, optimize=True)
            - 1.0 * np.einsum('BE,CEAijK->ABCijK', X.b.vv[Vb, Vb], T.bbb.VVVooO, optimize=True)
    )
    dR.bbb.VVVooO += (2.0 / 12.0) * (
            +1.0 * np.einsum('mj,CBAimK->ABCijK', X.b.oo[ob, ob], T.bbb.VVVooO, optimize=True)
            + 1.0 * np.einsum('Mj,CBAiMK->ABCijK', X.b.oo[Ob, ob], T.bbb.VVVoOO, optimize=True)
    )
    dR.bbb.VVVooO += (1.0 / 12.0) * (
            +1.0 * np.einsum('MK,CBAijM->ABCijK', X.b.oo[Ob, Ob], T.bbb.VVVooO, optimize=True)
    )
    dR.bbb.VVVooO += (1.0 / 12.0) * (
            -0.5 * np.einsum('mnij,CBAmnK->ABCijK', X.bb.oooo[ob, ob, ob, ob], T.bbb.VVVooO, optimize=True)
            + 1.0 * np.einsum('Mnij,CBAnMK->ABCijK', X.bb.oooo[Ob, ob, ob, ob], T.bbb.VVVoOO, optimize=True)
            - 0.5 * np.einsum('MNij,CBAMNK->ABCijK', X.bb.oooo[Ob, Ob, ob, ob], T.bbb.VVVOOO, optimize=True)
    )
    dR.bbb.VVVooO += (2.0 / 12.0) * (
            +1.0 * np.einsum('MnKj,CBAniM->ABCijK', X.bb.oooo[Ob, ob, Ob, ob], T.bbb.VVVooO, optimize=True)
            + 0.5 * np.einsum('MNKj,CBAiMN->ABCijK', X.bb.oooo[Ob, Ob, Ob, ob], T.bbb.VVVoOO, optimize=True)
    )
    dR.bbb.VVVooO += (3.0 / 12.0) * (
            -0.5 * np.einsum('ABef,CfeijK->ABCijK', X.bb.vvvv[Vb, Vb, vb, vb], T.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('ABeF,CFeijK->ABCijK', X.bb.vvvv[Vb, Vb, vb, Vb], T.bbb.VVvooO, optimize=True)
            - 0.5 * np.einsum('ABEF,CFEijK->ABCijK', X.bb.vvvv[Vb, Vb, Vb, Vb], T.bbb.VVVooO, optimize=True)
    )
    dR.bbb.VVVooO += (6.0 / 12.0) * (
            +1.0 * np.einsum('Bmje,CAeimK->ABCijK', X.bb.voov[Vb, ob, ob, vb], T.bbb.VVvooO, optimize=True)
            - 1.0 * np.einsum('BmjE,CEAimK->ABCijK', X.bb.voov[Vb, ob, ob, Vb], T.bbb.VVVooO, optimize=True)
            + 1.0 * np.einsum('BMje,CAeiMK->ABCijK', X.bb.voov[Vb, Ob, ob, vb], T.bbb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('BMjE,CEAiMK->ABCijK', X.bb.voov[Vb, Ob, ob, Vb], T.bbb.VVVoOO, optimize=True)
    )
    dR.bbb.VVVooO += (3.0 / 12.0) * (
            +1.0 * np.einsum('BMKe,CAeijM->ABCijK', X.bb.voov[Vb, Ob, Ob, vb], T.bbb.VVvooO, optimize=True)
            - 1.0 * np.einsum('BMKE,CEAijM->ABCijK', X.bb.voov[Vb, Ob, Ob, Vb], T.bbb.VVVooO, optimize=True)
    )
    dR.bbb.VVVooO += (6.0 / 12.0) * (
            +1.0 * np.einsum('mBej,eACmiK->ABCijK', X.ab.ovvo[oa, Vb, va, ob], T.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('mBEj,EACmiK->ABCijK', X.ab.ovvo[oa, Vb, Va, ob], T.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('MBej,eACMiK->ABCijK', X.ab.ovvo[Oa, Vb, va, ob], T.abb.vVVOoO, optimize=True)
            + 1.0 * np.einsum('MBEj,EACMiK->ABCijK', X.ab.ovvo[Oa, Vb, Va, ob], T.abb.VVVOoO, optimize=True)
    )
    dR.bbb.VVVooO += (3.0 / 12.0) * (
            +1.0 * np.einsum('MBeK,eACMji->ABCijK', X.ab.ovvo[Oa, Vb, va, Ob], T.abb.vVVOoo, optimize=True)
            + 1.0 * np.einsum('MBEK,EACMji->ABCijK', X.ab.ovvo[Oa, Vb, Va, Ob], T.abb.VVVOoo, optimize=True)
    )
    dR.bbb.VVVooO += (2.0 / 12.0) * (
            +1.0 * np.einsum('mj,CBAimK->ABCijK', H.b.oo[ob, ob], R.bbb.VVVooO, optimize=True)
            + 1.0 * np.einsum('Mj,CBAiMK->ABCijK', H.b.oo[Ob, ob], R.bbb.VVVoOO, optimize=True)
    )
    dR.bbb.VVVooO += (1.0 / 12.0) * (
            +1.0 * np.einsum('MK,CBAijM->ABCijK', H.b.oo[Ob, Ob], R.bbb.VVVooO, optimize=True)
    )
    dR.bbb.VVVooO += (3.0 / 12.0) * (
            +1.0 * np.einsum('Be,CAeijK->ABCijK', H.b.vv[Vb, vb], R.bbb.VVvooO, optimize=True)
            - 1.0 * np.einsum('BE,CEAijK->ABCijK', H.b.vv[Vb, Vb], R.bbb.VVVooO, optimize=True)
    )
    dR.bbb.VVVooO += (1.0 / 12.0) * (
            -0.5 * np.einsum('mnij,CBAmnK->ABCijK', H.bb.oooo[ob, ob, ob, ob], R.bbb.VVVooO, optimize=True)
            + 1.0 * np.einsum('Mnij,CBAnMK->ABCijK', H.bb.oooo[Ob, ob, ob, ob], R.bbb.VVVoOO, optimize=True)
            - 0.5 * np.einsum('MNij,CBAMNK->ABCijK', H.bb.oooo[Ob, Ob, ob, ob], R.bbb.VVVOOO, optimize=True)
    )
    dR.bbb.VVVooO += (2.0 / 12.0) * (
            +1.0 * np.einsum('MnKj,CBAniM->ABCijK', H.bb.oooo[Ob, ob, Ob, ob], R.bbb.VVVooO, optimize=True)
            + 0.5 * np.einsum('MNKj,CBAiMN->ABCijK', H.bb.oooo[Ob, Ob, Ob, ob], R.bbb.VVVoOO, optimize=True)
    )
    dR.bbb.VVVooO += (3.0 / 12.0) * (
            -0.5 * np.einsum('ABef,CfeijK->ABCijK', H.bb.vvvv[Vb, Vb, vb, vb], R.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('ABeF,CFeijK->ABCijK', H.bb.vvvv[Vb, Vb, vb, Vb], R.bbb.VVvooO, optimize=True)
            - 0.5 * np.einsum('ABEF,CFEijK->ABCijK', H.bb.vvvv[Vb, Vb, Vb, Vb], R.bbb.VVVooO, optimize=True)
    )
    dR.bbb.VVVooO += (6.0 / 12.0) * (
            -1.0 * np.einsum('Amie,CBemjK->ABCijK', H.bb.voov[Vb, ob, ob, vb], R.bbb.VVvooO, optimize=True)
            - 1.0 * np.einsum('AmiE,CBEmjK->ABCijK', H.bb.voov[Vb, ob, ob, Vb], R.bbb.VVVooO, optimize=True)
            + 1.0 * np.einsum('AMie,CBejMK->ABCijK', H.bb.voov[Vb, Ob, ob, vb], R.bbb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('AMiE,CBEjMK->ABCijK', H.bb.voov[Vb, Ob, ob, Vb], R.bbb.VVVoOO, optimize=True)
    )
    dR.bbb.VVVooO += (3.0 / 12.0) * (
            +1.0 * np.einsum('AMKe,CBejiM->ABCijK', H.bb.voov[Vb, Ob, Ob, vb], R.bbb.VVvooO, optimize=True)
            + 1.0 * np.einsum('AMKE,CBEjiM->ABCijK', H.bb.voov[Vb, Ob, Ob, Vb], R.bbb.VVVooO, optimize=True)
    )
    dR.bbb.VVVooO += (6.0 / 12.0) * (
            +1.0 * np.einsum('mAei,eBCmjK->ABCijK', H.ab.ovvo[oa, Vb, va, ob], R.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('mAEi,EBCmjK->ABCijK', H.ab.ovvo[oa, Vb, Va, ob], R.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('MAei,eBCMjK->ABCijK', H.ab.ovvo[Oa, Vb, va, ob], R.abb.vVVOoO, optimize=True)
            + 1.0 * np.einsum('MAEi,EBCMjK->ABCijK', H.ab.ovvo[Oa, Vb, Va, ob], R.abb.VVVOoO, optimize=True)
    )
    dR.bbb.VVVooO += (3.0 / 12.0) * (
            +1.0 * np.einsum('MAeK,eBCMij->ABCijK', H.ab.ovvo[Oa, Vb, va, Ob], R.abb.vVVOoo, optimize=True)
            + 1.0 * np.einsum('MAEK,EBCMij->ABCijK', H.ab.ovvo[Oa, Vb, Va, Ob], R.abb.VVVOoo, optimize=True)
    )
    # of terms =  20

    dR.bbb.VVVooO -= np.transpose(dR.bbb.VVVooO, (0, 1, 2, 4, 3, 5))
    dR.bbb.VVVooO -= np.transpose(dR.bbb.VVVooO, (0, 2, 1, 3, 4, 5))
    dR.bbb.VVVooO -= np.transpose(dR.bbb.VVVooO, (1, 0, 2, 3, 4, 5)) + np.transpose(dR.bbb.VVVooO, (2, 1, 0, 3, 4, 5))

    return dR

def update(R, omega, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    R.bbb.VVVooO = eomcc_active_loops.update_r3d_111001(
        R.bbb.VVVooO,
        omega,
        H.b.oo[Ob, Ob],
        H.b.vv[Vb, Vb],
        H.b.oo[ob, ob],
        H.b.vv[vb, vb],
        0.0,
    )
    return R
