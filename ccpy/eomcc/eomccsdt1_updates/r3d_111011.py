import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops

def build(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.bbb.VVVoOO = (6.0 / 12.0) * (
            -1.0 * np.einsum('BmJi,ACmK->ABCiJK', X.bb.vooo[Vb, :, Ob, ob], T.bb[Vb, Vb, :, Ob], optimize=True)
    )
    dR.bbb.VVVoOO += (3.0 / 12.0) * (
            +1.0 * np.einsum('BmJK,ACmi->ABCiJK', X.bb.vooo[Vb, :, Ob, Ob], T.bb[Vb, Vb, :, ob], optimize=True)
    )
    dR.bbb.VVVoOO += (6.0 / 12.0) * (
            -1.0 * np.einsum('BmJi,ACmK->ABCiJK', H.bb.vooo[Vb, :, Ob, ob], R.bb[Vb, Vb, :, Ob], optimize=True)
    )
    dR.bbb.VVVoOO += (3.0 / 12.0) * (
            +1.0 * np.einsum('BmJK,ACmi->ABCiJK', H.bb.vooo[Vb, :, Ob, Ob], R.bb[Vb, Vb, :, ob], optimize=True)
    )
    dR.bbb.VVVoOO += (6.0 / 12.0) * (
            +1.0 * np.einsum('BAJe,eCiK->ABCiJK', X.bb.vvov[Vb, Vb, Ob, :], T.bb[:, Vb, ob, Ob], optimize=True)
    )
    dR.bbb.VVVoOO += (3.0 / 12.0) * (
            -1.0 * np.einsum('BAie,eCJK->ABCiJK', X.bb.vvov[Vb, Vb, ob, :], T.bb[:, Vb, Ob, Ob], optimize=True)
    )
    dR.bbb.VVVoOO += (6.0 / 12.0) * (
            +1.0 * np.einsum('BAJe,eCiK->ABCiJK', H.bb.vvov[Vb, Vb, Ob, :], R.bb[:, Vb, ob, Ob], optimize=True)
    )
    dR.bbb.VVVoOO += (3.0 / 12.0) * (
            -1.0 * np.einsum('BAie,eCJK->ABCiJK', H.bb.vvov[Vb, Vb, ob, :], R.bb[:, Vb, Ob, Ob], optimize=True)
    )
    # of terms =  8
    dR.bbb.VVVoOO += (3.0 / 12.0) * (
            +1.0 * np.einsum('Be,CAeiJK->ABCiJK', X.b.vv[Vb, vb], T.bbb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('BE,CEAiJK->ABCiJK', X.b.vv[Vb, Vb], T.bbb.VVVoOO, optimize=True)
    )
    dR.bbb.VVVoOO += (2.0 / 12.0) * (
            +1.0 * np.einsum('mJ,CBAimK->ABCiJK', X.b.oo[ob, Ob], T.bbb.VVVooO, optimize=True)
            + 1.0 * np.einsum('MJ,CBAiMK->ABCiJK', X.b.oo[Ob, Ob], T.bbb.VVVoOO, optimize=True)
    )
    dR.bbb.VVVoOO += (1.0 / 12.0) * (
            +1.0 * np.einsum('mi,CBAmJK->ABCiJK', X.b.oo[ob, ob], T.bbb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('Mi,CBAJMK->ABCiJK', X.b.oo[Ob, ob], T.bbb.VVVOOO, optimize=True)
    )
    dR.bbb.VVVoOO += (2.0 / 12.0) * (
            -0.5 * np.einsum('mniJ,CBAmnK->ABCiJK', X.bb.oooo[ob, ob, ob, Ob], T.bbb.VVVooO, optimize=True)
            + 1.0 * np.einsum('MniJ,CBAnMK->ABCiJK', X.bb.oooo[Ob, ob, ob, Ob], T.bbb.VVVoOO, optimize=True)
            - 0.5 * np.einsum('MNiJ,CBAMNK->ABCiJK', X.bb.oooo[Ob, Ob, ob, Ob], T.bbb.VVVOOO, optimize=True)
    )
    dR.bbb.VVVoOO += (1.0 / 12.0) * (
            +1.0 * np.einsum('MnKJ,CBAniM->ABCiJK', X.bb.oooo[Ob, ob, Ob, Ob], T.bbb.VVVooO, optimize=True)
            + 0.5 * np.einsum('MNKJ,CBAiMN->ABCiJK', X.bb.oooo[Ob, Ob, Ob, Ob], T.bbb.VVVoOO, optimize=True)
    )
    dR.bbb.VVVoOO += (3.0 / 12.0) * (
            -0.5 * np.einsum('ABef,CfeiJK->ABCiJK', X.bb.vvvv[Vb, Vb, vb, vb], T.bbb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('ABEf,CEfiJK->ABCiJK', X.bb.vvvv[Vb, Vb, Vb, vb], T.bbb.VVvoOO, optimize=True)
            - 0.5 * np.einsum('ABEF,CFEiJK->ABCiJK', X.bb.vvvv[Vb, Vb, Vb, Vb], T.bbb.VVVoOO, optimize=True)
    )
    dR.bbb.VVVoOO += (6.0 / 12.0) * (
            +1.0 * np.einsum('BmJe,CAeimK->ABCiJK', X.bb.voov[Vb, ob, Ob, vb], T.bbb.VVvooO, optimize=True)
            + 1.0 * np.einsum('BMJe,CAeiMK->ABCiJK', X.bb.voov[Vb, Ob, Ob, vb], T.bbb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('BmJE,CEAimK->ABCiJK', X.bb.voov[Vb, ob, Ob, Vb], T.bbb.VVVooO, optimize=True)
            - 1.0 * np.einsum('BMJE,CEAiMK->ABCiJK', X.bb.voov[Vb, Ob, Ob, Vb], T.bbb.VVVoOO, optimize=True)
    )
    dR.bbb.VVVoOO += (3.0 / 12.0) * (
            +1.0 * np.einsum('Bmie,CAemJK->ABCiJK', X.bb.voov[Vb, ob, ob, vb], T.bbb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('BMie,CAeJMK->ABCiJK', X.bb.voov[Vb, Ob, ob, vb], T.bbb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('BmiE,CEAmJK->ABCiJK', X.bb.voov[Vb, ob, ob, Vb], T.bbb.VVVoOO, optimize=True)
            + 1.0 * np.einsum('BMiE,CEAJMK->ABCiJK', X.bb.voov[Vb, Ob, ob, Vb], T.bbb.VVVOOO, optimize=True)
    )
    dR.bbb.VVVoOO += (6.0 / 12.0) * (
            +1.0 * np.einsum('mBeJ,eACmiK->ABCiJK', X.ab.ovvo[oa, Vb, va, Ob], T.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('MBeJ,eACMiK->ABCiJK', X.ab.ovvo[Oa, Vb, va, Ob], T.abb.vVVOoO, optimize=True)
            + 1.0 * np.einsum('mBEJ,EACmiK->ABCiJK', X.ab.ovvo[oa, Vb, Va, Ob], T.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('MBEJ,EACMiK->ABCiJK', X.ab.ovvo[Oa, Vb, Va, Ob], T.abb.VVVOoO, optimize=True)
    )
    dR.bbb.VVVoOO += (3.0 / 12.0) * (
            +1.0 * np.einsum('mBei,eACmKJ->ABCiJK', X.ab.ovvo[oa, Vb, va, ob], T.abb.vVVoOO, optimize=True)
            + 1.0 * np.einsum('MBei,eACMKJ->ABCiJK', X.ab.ovvo[Oa, Vb, va, ob], T.abb.vVVOOO, optimize=True)
            + 1.0 * np.einsum('mBEi,EACmKJ->ABCiJK', X.ab.ovvo[oa, Vb, Va, ob], T.abb.VVVoOO, optimize=True)
            + 1.0 * np.einsum('MBEi,EACMKJ->ABCiJK', X.ab.ovvo[Oa, Vb, Va, ob], T.abb.VVVOOO, optimize=True)
    )
    dR.bbb.VVVoOO += (2.0 / 12.0) * (
            +1.0 * np.einsum('mJ,CBAimK->ABCiJK', H.b.oo[ob, Ob], R.bbb.VVVooO, optimize=True)
            + 1.0 * np.einsum('MJ,CBAiMK->ABCiJK', H.b.oo[Ob, Ob], R.bbb.VVVoOO, optimize=True)
    )
    dR.bbb.VVVoOO += (1.0 / 12.0) * (
            +1.0 * np.einsum('mi,CBAmJK->ABCiJK', H.b.oo[ob, ob], R.bbb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('Mi,CBAJMK->ABCiJK', H.b.oo[Ob, ob], R.bbb.VVVOOO, optimize=True)
    )
    dR.bbb.VVVoOO += (3.0 / 12.0) * (
            +1.0 * np.einsum('Be,CAeiJK->ABCiJK', H.b.vv[Vb, vb], R.bbb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('BE,CEAiJK->ABCiJK', H.b.vv[Vb, Vb], R.bbb.VVVoOO, optimize=True)
    )
    dR.bbb.VVVoOO += (2.0 / 12.0) * (
            -0.5 * np.einsum('mniJ,CBAmnK->ABCiJK', H.bb.oooo[ob, ob, ob, Ob], R.bbb.VVVooO, optimize=True)
            + 1.0 * np.einsum('MniJ,CBAnMK->ABCiJK', H.bb.oooo[Ob, ob, ob, Ob], R.bbb.VVVoOO, optimize=True)
            - 0.5 * np.einsum('MNiJ,CBAMNK->ABCiJK', H.bb.oooo[Ob, Ob, ob, Ob], R.bbb.VVVOOO, optimize=True)
    )
    dR.bbb.VVVoOO += (1.0 / 12.0) * (
            +1.0 * np.einsum('MnKJ,CBAniM->ABCiJK', H.bb.oooo[Ob, ob, Ob, Ob], R.bbb.VVVooO, optimize=True)
            + 0.5 * np.einsum('MNKJ,CBAiMN->ABCiJK', H.bb.oooo[Ob, Ob, Ob, Ob], R.bbb.VVVoOO, optimize=True)
    )
    dR.bbb.VVVoOO += (3.0 / 12.0) * (
            -0.5 * np.einsum('ABef,CfeiJK->ABCiJK', H.bb.vvvv[Vb, Vb, vb, vb], R.bbb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('ABEf,CEfiJK->ABCiJK', H.bb.vvvv[Vb, Vb, Vb, vb], R.bbb.VVvoOO, optimize=True)
            - 0.5 * np.einsum('ABEF,CFEiJK->ABCiJK', H.bb.vvvv[Vb, Vb, Vb, Vb], R.bbb.VVVoOO, optimize=True)
    )
    dR.bbb.VVVoOO += (3.0 / 12.0) * (
            -1.0 * np.einsum('Amie,CBemJK->ABCiJK', H.bb.voov[Vb, ob, ob, vb], R.bbb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('AMie,CBeMJK->ABCiJK', H.bb.voov[Vb, Ob, ob, vb], R.bbb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('AmiE,CBEmJK->ABCiJK', H.bb.voov[Vb, ob, ob, Vb], R.bbb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('AMiE,CBEMJK->ABCiJK', H.bb.voov[Vb, Ob, ob, Vb], R.bbb.VVVOOO, optimize=True)
    )
    dR.bbb.VVVoOO += (6.0 / 12.0) * (
            +1.0 * np.einsum('AmJe,CBemiK->ABCiJK', H.bb.voov[Vb, ob, Ob, vb], R.bbb.VVvooO, optimize=True)
            - 1.0 * np.einsum('AMJe,CBeiMK->ABCiJK', H.bb.voov[Vb, Ob, Ob, vb], R.bbb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('AmJE,CBEmiK->ABCiJK', H.bb.voov[Vb, ob, Ob, Vb], R.bbb.VVVooO, optimize=True)
            - 1.0 * np.einsum('AMJE,CBEiMK->ABCiJK', H.bb.voov[Vb, Ob, Ob, Vb], R.bbb.VVVoOO, optimize=True)
    )
    dR.bbb.VVVoOO += (3.0 / 12.0) * (
            -1.0 * np.einsum('mAei,eBCmKJ->ABCiJK', H.ab.ovvo[oa, Vb, va, ob], R.abb.vVVoOO, optimize=True)
            - 1.0 * np.einsum('MAei,eBCMKJ->ABCiJK', H.ab.ovvo[Oa, Vb, va, ob], R.abb.vVVOOO, optimize=True)
            - 1.0 * np.einsum('mAEi,EBCmKJ->ABCiJK', H.ab.ovvo[oa, Vb, Va, ob], R.abb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('MAEi,EBCMKJ->ABCiJK', H.ab.ovvo[Oa, Vb, Va, ob], R.abb.VVVOOO, optimize=True)
    )
    dR.bbb.VVVoOO += (6.0 / 12.0) * (
            -1.0 * np.einsum('mAeJ,eBCmiK->ABCiJK', H.ab.ovvo[oa, Vb, va, Ob], R.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('MAeJ,eBCMiK->ABCiJK', H.ab.ovvo[Oa, Vb, va, Ob], R.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('mAEJ,EBCmiK->ABCiJK', H.ab.ovvo[oa, Vb, Va, Ob], R.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('MAEJ,EBCMiK->ABCiJK', H.ab.ovvo[Oa, Vb, Va, Ob], R.abb.VVVOoO, optimize=True)
    )
    # of terms =  20

    dR.bbb.VVVoOO -= np.transpose(dR.bbb.VVVoOO, (0, 1, 2, 3, 5, 4))
    dR.bbb.VVVoOO -= np.transpose(dR.bbb.VVVoOO, (0, 2, 1, 3, 4, 5))
    dR.bbb.VVVoOO -= np.transpose(dR.bbb.VVVoOO, (1, 0, 2, 3, 4, 5)) + np.transpose(dR.bbb.VVVoOO, (2, 1, 0, 3, 4, 5))

    return dR

def update(R, omega, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    R.bbb.VVVoOO = eomcc_active_loops.update_r3d_111011(
        R.bbb.VVVoOO,
        omega,
        H.b.oo[Ob, Ob],
        H.b.vv[Vb, Vb],
        H.b.oo[ob, ob],
        H.b.vv[vb, vb],
        0.0,
    )
    return R
