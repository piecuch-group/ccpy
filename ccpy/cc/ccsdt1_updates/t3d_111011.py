import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import cc_active_loops

def build(T, dT, H, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    # MM(2,3)
    dT.bbb.VVVoOO = (6.0 / 12.0) * (
            -1.0 * np.einsum('AmiJ,BCmK->ABCiJK', H.bb.vooo[Vb, :, ob, Ob], T.bb[Vb, Vb, :, Ob], optimize=True)
    )
    dT.bbb.VVVoOO += (3.0 / 12.0) * (
            +1.0 * np.einsum('AmKJ,BCmi->ABCiJK', H.bb.vooo[Vb, :, Ob, Ob], T.bb[Vb, Vb, :, ob], optimize=True)
    )
    dT.bbb.VVVoOO += (3.0 / 12.0) * (
            +1.0 * np.einsum('ABie,eCJK->ABCiJK', H.bb.vvov[Vb, Vb, ob, :], T.bb[:, Vb, Ob, Ob], optimize=True)
    )
    dT.bbb.VVVoOO += (6.0 / 12.0) * (
            -1.0 * np.einsum('ABJe,eCiK->ABCiJK', H.bb.vvov[Vb, Vb, Ob, :], T.bb[:, Vb, ob, Ob], optimize=True)
    )
    # (H(2) * T3)_C
    dT.bbb.VVVoOO += (1.0 / 12.0) * (
            +1.0 * np.einsum('mi,CBAmJK->ABCiJK', H.b.oo[ob, ob], T.bbb.VVVoOO, optimize=True)
            + 1.0 * np.einsum('Mi,CBAMJK->ABCiJK', H.b.oo[Ob, ob], T.bbb.VVVOOO, optimize=True)
    )
    dT.bbb.VVVoOO += (2.0 / 12.0) * (
            -1.0 * np.einsum('mJ,CBAmiK->ABCiJK', H.b.oo[ob, Ob], T.bbb.VVVooO, optimize=True)
            + 1.0 * np.einsum('MJ,CBAiMK->ABCiJK', H.b.oo[Ob, Ob], T.bbb.VVVoOO, optimize=True)
    )
    dT.bbb.VVVoOO += (3.0 / 12.0) * (
            -1.0 * np.einsum('Ae,CBeiJK->ABCiJK', H.b.vv[Vb, vb], T.bbb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('AE,CBEiJK->ABCiJK', H.b.vv[Vb, Vb], T.bbb.VVVoOO, optimize=True)
    )
    dT.bbb.VVVoOO += (2.0 / 12.0) * (
            -0.5 * np.einsum('mniJ,CBAmnK->ABCiJK', H.bb.oooo[ob, ob, ob, Ob], T.bbb.VVVooO, optimize=True)
            + 1.0 * np.einsum('MniJ,CBAnMK->ABCiJK', H.bb.oooo[Ob, ob, ob, Ob], T.bbb.VVVoOO, optimize=True)
            - 0.5 * np.einsum('MNiJ,CBAMNK->ABCiJK', H.bb.oooo[Ob, Ob, ob, Ob], T.bbb.VVVOOO, optimize=True)
    )
    dT.bbb.VVVoOO += (1.0 / 12.0) * (
            +1.0 * np.einsum('MnKJ,CBAniM->ABCiJK', H.bb.oooo[Ob, ob, Ob, Ob], T.bbb.VVVooO, optimize=True)
            + 0.5 * np.einsum('MNKJ,CBAiMN->ABCiJK', H.bb.oooo[Ob, Ob, Ob, Ob], T.bbb.VVVoOO, optimize=True)
    )
    dT.bbb.VVVoOO += (3.0 / 12.0) * (
            -0.5 * np.einsum('ABef,CfeiJK->ABCiJK', H.bb.vvvv[Vb, Vb, vb, vb], T.bbb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('ABEf,CEfiJK->ABCiJK', H.bb.vvvv[Vb, Vb, Vb, vb], T.bbb.VVvoOO, optimize=True)
            - 0.5 * np.einsum('ABEF,CFEiJK->ABCiJK', H.bb.vvvv[Vb, Vb, Vb, Vb], T.bbb.VVVoOO, optimize=True)
    )
    dT.bbb.VVVoOO += (3.0 / 12.0) * (
            -1.0 * np.einsum('Amie,CBemJK->ABCiJK', H.bb.voov[Vb, ob, ob, vb], T.bbb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('AmiE,CBEmJK->ABCiJK', H.bb.voov[Vb, ob, ob, Vb], T.bbb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('AMie,CBeMJK->ABCiJK', H.bb.voov[Vb, Ob, ob, vb], T.bbb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('AMiE,CBEMJK->ABCiJK', H.bb.voov[Vb, Ob, ob, Vb], T.bbb.VVVOOO, optimize=True)
    )
    dT.bbb.VVVoOO += (6.0 / 12.0) * (
            +1.0 * np.einsum('AmJe,CBemiK->ABCiJK', H.bb.voov[Vb, ob, Ob, vb], T.bbb.VVvooO, optimize=True)
            + 1.0 * np.einsum('AmJE,CBEmiK->ABCiJK', H.bb.voov[Vb, ob, Ob, Vb], T.bbb.VVVooO, optimize=True)
            - 1.0 * np.einsum('AMJe,CBeiMK->ABCiJK', H.bb.voov[Vb, Ob, Ob, vb], T.bbb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('AMJE,CBEiMK->ABCiJK', H.bb.voov[Vb, Ob, Ob, Vb], T.bbb.VVVoOO, optimize=True)
    )
    dT.bbb.VVVoOO += (3.0 / 12.0) * (
            -1.0 * np.einsum('mAei,eCBmJK->ABCiJK', H.ab.ovvo[oa, Vb, va, ob], T.abb.vVVoOO, optimize=True)
            - 1.0 * np.einsum('mAEi,ECBmJK->ABCiJK', H.ab.ovvo[oa, Vb, Va, ob], T.abb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('MAei,eCBMJK->ABCiJK', H.ab.ovvo[Oa, Vb, va, ob], T.abb.vVVOOO, optimize=True)
            - 1.0 * np.einsum('MAEi,ECBMJK->ABCiJK', H.ab.ovvo[Oa, Vb, Va, ob], T.abb.VVVOOO, optimize=True)
    )
    dT.bbb.VVVoOO += (6.0 / 12.0) * (
            +1.0 * np.einsum('mAeJ,eCBmiK->ABCiJK', H.ab.ovvo[oa, Vb, va, Ob], T.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('mAEJ,ECBmiK->ABCiJK', H.ab.ovvo[oa, Vb, Va, Ob], T.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('MAeJ,eCBMiK->ABCiJK', H.ab.ovvo[Oa, Vb, va, Ob], T.abb.vVVOoO, optimize=True)
            + 1.0 * np.einsum('MAEJ,ECBMiK->ABCiJK', H.ab.ovvo[Oa, Vb, Va, Ob], T.abb.VVVOoO, optimize=True)
    )

    dT.bbb.VVVoOO -= np.transpose(dT.bbb.VVVoOO, (1, 0, 2, 3, 4, 5)) + np.transpose(dT.bbb.VVVoOO, (0, 2, 1, 3, 4, 5)) \
           + np.transpose(dT.bbb.VVVoOO, (2, 1, 0, 3, 4, 5)) - np.transpose(dT.bbb.VVVoOO, (1, 2, 0, 3, 4, 5)) \
           - np.transpose(dT.bbb.VVVoOO, (2, 0, 1, 3, 4, 5))

    dT.bbb.VVVoOO -= np.transpose(dT.bbb.VVVoOO, (0, 1, 2, 3, 5, 4))

    return dT

def update(T, dT, H, shift, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.bbb.VVVoOO, dT.bbb.VVVoOO = cc_active_loops.update_t3d_111011(
        T.bbb.VVVoOO,
        dT.bbb.VVVoOO,
        H.b.oo[Ob, Ob],
        H.b.vv[Vb, Vb],
        H.b.oo[ob, ob],
        H.b.vv[vb, vb],
        shift,
    )

    return T, dT