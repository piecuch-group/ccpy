import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import cc_active_loops

def build(T, dT, H, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    # MM(2,3)
    dT.bbb.VVVooO = (3.0 / 12.0) * (
            -1.0 * np.einsum('Amij,BCmK->ABCijK', H.bb.vooo[Vb, :, ob, ob], T.bb[Vb, Vb, :, Ob], optimize=True)
    )
    dT.bbb.VVVooO += (6.0 / 12.0) * (
            +1.0 * np.einsum('AmKj,BCmi->ABCijK', H.bb.vooo[Vb, :, Ob, ob], T.bb[Vb, Vb, :, ob], optimize=True)
    )
    dT.bbb.VVVooO += (6.0 / 12.0) * (
            +1.0 * np.einsum('ABie,eCjK->ABCijK', H.bb.vvov[Vb, Vb, ob, :], T.bb[:, Vb, ob, Ob], optimize=True)
    )
    dT.bbb.VVVooO += (3.0 / 12.0) * (
            -1.0 * np.einsum('ABKe,eCji->ABCijK', H.bb.vvov[Vb, Vb, Ob, :], T.bb[:, Vb, ob, ob], optimize=True)
    )
    # (H(2) * T3)_C
    dT.bbb.VVVooO += (2.0 / 12.0) * (
            +1.0 * np.einsum('mi,CBAmjK->ABCijK', H.b.oo[ob, ob], T.bbb.VVVooO, optimize=True)
            - 1.0 * np.einsum('Mi,CBAjMK->ABCijK', H.b.oo[Ob, ob], T.bbb.VVVoOO, optimize=True)
    )
    dT.bbb.VVVooO += (1.0 / 12.0) * (
            -1.0 * np.einsum('MK,CBAjiM->ABCijK', H.b.oo[Ob, Ob], T.bbb.VVVooO, optimize=True)
    )
    dT.bbb.VVVooO += (3.0 / 12.0) * (
            -1.0 * np.einsum('Ae,CBeijK->ABCijK', H.b.vv[Vb, vb], T.bbb.VVvooO, optimize=True)
            - 1.0 * np.einsum('AE,CBEijK->ABCijK', H.b.vv[Vb, Vb], T.bbb.VVVooO, optimize=True)
    )
    dT.bbb.VVVooO += (1.0 / 12.0) * (
            -0.5 * np.einsum('mnij,CBAmnK->ABCijK', H.bb.oooo[ob, ob, ob, ob], T.bbb.VVVooO, optimize=True)
            + 1.0 * np.einsum('Mnij,CBAnMK->ABCijK', H.bb.oooo[Ob, ob, ob, ob], T.bbb.VVVoOO, optimize=True)
            - 0.5 * np.einsum('MNij,CBAMNK->ABCijK', H.bb.oooo[Ob, Ob, ob, ob], T.bbb.VVVOOO, optimize=True)
    )
    dT.bbb.VVVooO += (2.0 / 12.0) * (
            +1.0 * np.einsum('MnKj,CBAniM->ABCijK', H.bb.oooo[Ob, ob, Ob, ob], T.bbb.VVVooO, optimize=True)
            + 0.5 * np.einsum('MNKj,CBAiMN->ABCijK', H.bb.oooo[Ob, Ob, Ob, ob], T.bbb.VVVoOO, optimize=True)
    )
    dT.bbb.VVVooO += (3.0 / 12.0) * (
            -0.5 * np.einsum('ABef,CfeijK->ABCijK', H.bb.vvvv[Vb, Vb, vb, vb], T.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('ABeF,CFeijK->ABCijK', H.bb.vvvv[Vb, Vb, vb, Vb], T.bbb.VVvooO, optimize=True)
            - 0.5 * np.einsum('ABEF,CFEijK->ABCijK', H.bb.vvvv[Vb, Vb, Vb, Vb], T.bbb.VVVooO, optimize=True)
    )
    dT.bbb.VVVooO += (6.0 / 12.0) * (
            -1.0 * np.einsum('Amie,CBemjK->ABCijK', H.bb.voov[Vb, ob, ob, vb], T.bbb.VVvooO, optimize=True)
            - 1.0 * np.einsum('AmiE,CBEmjK->ABCijK', H.bb.voov[Vb, ob, ob, Vb], T.bbb.VVVooO, optimize=True)
            + 1.0 * np.einsum('AMie,CBejMK->ABCijK', H.bb.voov[Vb, Ob, ob, vb], T.bbb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('AMiE,CBEjMK->ABCijK', H.bb.voov[Vb, Ob, ob, Vb], T.bbb.VVVoOO, optimize=True)
    )
    dT.bbb.VVVooO += (3.0 / 12.0) * (
            +1.0 * np.einsum('AMKe,CBejiM->ABCijK', H.bb.voov[Vb, Ob, Ob, vb], T.bbb.VVvooO, optimize=True)
            + 1.0 * np.einsum('AMKE,CBEjiM->ABCijK', H.bb.voov[Vb, Ob, Ob, Vb], T.bbb.VVVooO, optimize=True)
    )
    dT.bbb.VVVooO += (6.0 / 12.0) * (
            -1.0 * np.einsum('mAei,eCBmjK->ABCijK', H.ab.ovvo[oa, Vb, va, ob], T.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('mAEi,ECBmjK->ABCijK', H.ab.ovvo[oa, Vb, Va, ob], T.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('MAei,eCBMjK->ABCijK', H.ab.ovvo[Oa, Vb, va, ob], T.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('MAEi,ECBMjK->ABCijK', H.ab.ovvo[Oa, Vb, Va, ob], T.abb.VVVOoO, optimize=True)
    )
    dT.bbb.VVVooO += (3.0 / 12.0) * (
            +1.0 * np.einsum('MAeK,eCBMji->ABCijK', H.ab.ovvo[Oa, Vb, va, Ob], T.abb.vVVOoo, optimize=True)
            + 1.0 * np.einsum('MAEK,ECBMji->ABCijK', H.ab.ovvo[Oa, Vb, Va, Ob], T.abb.VVVOoo, optimize=True)
    )

    dT.bbb.VVVooO -= np.transpose(dT.bbb.VVVooO, (1, 0, 2, 3, 4, 5)) + np.transpose(dT.bbb.VVVooO, (0, 2, 1, 3, 4, 5)) \
           + np.transpose(dT.bbb.VVVooO, (2, 1, 0, 3, 4, 5)) - np.transpose(dT.bbb.VVVooO, (1, 2, 0, 3, 4, 5)) \
           - np.transpose(dT.bbb.VVVooO, (2, 0, 1, 3, 4, 5))

    dT.bbb.VVVooO -= np.transpose(dT.bbb.VVVooO, (0, 1, 2, 4, 3, 5))

    return dT

def update(T, dT, H, shift, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.bbb.VVVooO, dT.bbb.VVVooO = cc_active_loops.update_t3d_111001(
        T.bbb.VVVooO,
        dT.bbb.VVVooO,
        H.b.oo[Ob, Ob],
        H.b.vv[Vb, Vb],
        H.b.oo[ob, ob],
        H.b.vv[vb, vb],
        shift,
    )

    return T, dT