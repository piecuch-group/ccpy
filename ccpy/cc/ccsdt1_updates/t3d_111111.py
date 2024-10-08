import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import cc_active_loops

def build(T, dT, H, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)\
    # MM(2,3)
    dT.bbb.VVVOOO = (9.0 / 36.0) * (
            -1.0 * np.einsum('AmIJ,BCmK->ABCIJK', H.bb.vooo[Vb, :, Ob, Ob], T.bb[Vb, Vb, :, Ob], optimize=True)
    )
    dT.bbb.VVVOOO += (9.0 / 36.0) * (
            +1.0 * np.einsum('ABIe,eCJK->ABCIJK', H.bb.vvov[Vb, Vb, Ob, :], T.bb[:, Vb, Ob, Ob], optimize=True)
    )
    # (H(2) * T3)_C
    dT.bbb.VVVOOO += (3.0 / 36.0) * (
            +1.0 * np.einsum('mI,CBAmJK->ABCIJK', H.b.oo[ob, Ob], T.bbb.VVVoOO, optimize=True)
            + 1.0 * np.einsum('MI,CBAMJK->ABCIJK', H.b.oo[Ob, Ob], T.bbb.VVVOOO, optimize=True)
    )
    dT.bbb.VVVOOO += (3.0 / 36.0) * (
            -1.0 * np.einsum('Ae,CBeIJK->ABCIJK', H.b.vv[Vb, vb], T.bbb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('AE,CBEIJK->ABCIJK', H.b.vv[Vb, Vb], T.bbb.VVVOOO, optimize=True)
    )
    dT.bbb.VVVOOO += (3.0 / 36.0) * (
            -0.5 * np.einsum('mnIJ,CBAmnK->ABCIJK', H.bb.oooo[ob, ob, Ob, Ob], T.bbb.VVVooO, optimize=True)
            + 1.0 * np.einsum('MnIJ,CBAnMK->ABCIJK', H.bb.oooo[Ob, ob, Ob, Ob], T.bbb.VVVoOO, optimize=True)
            - 0.5 * np.einsum('MNIJ,CBAMNK->ABCIJK', H.bb.oooo[Ob, Ob, Ob, Ob], T.bbb.VVVOOO, optimize=True)
    )
    dT.bbb.VVVOOO += (3.0 / 36.0) * (
            -0.5 * np.einsum('ABef,CfeIJK->ABCIJK', H.bb.vvvv[Vb, Vb, vb, vb], T.bbb.VvvOOO, optimize=True)
            + 1.0 * np.einsum('ABEf,CEfIJK->ABCIJK', H.bb.vvvv[Vb, Vb, Vb, vb], T.bbb.VVvOOO, optimize=True)
            - 0.5 * np.einsum('ABEF,CFEIJK->ABCIJK', H.bb.vvvv[Vb, Vb, Vb, Vb], T.bbb.VVVOOO, optimize=True)
    )
    dT.bbb.VVVOOO += (9.0 / 36.0) * (
            -1.0 * np.einsum('AmIe,CBemJK->ABCIJK', H.bb.voov[Vb, ob, Ob, vb], T.bbb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('AMIe,CBeMJK->ABCIJK', H.bb.voov[Vb, Ob, Ob, vb], T.bbb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('AmIE,CBEmJK->ABCIJK', H.bb.voov[Vb, ob, Ob, Vb], T.bbb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('AMIE,CBEMJK->ABCIJK', H.bb.voov[Vb, Ob, Ob, Vb], T.bbb.VVVOOO, optimize=True)
    )
    dT.bbb.VVVOOO += (9.0 / 36.0) * (
            -1.0 * np.einsum('mAeI,eCBmJK->ABCIJK', H.ab.ovvo[oa, Vb, va, Ob], T.abb.vVVoOO, optimize=True)
            - 1.0 * np.einsum('MAeI,eCBMJK->ABCIJK', H.ab.ovvo[Oa, Vb, va, Ob], T.abb.vVVOOO, optimize=True)
            - 1.0 * np.einsum('mAEI,ECBmJK->ABCIJK', H.ab.ovvo[oa, Vb, Va, Ob], T.abb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('MAEI,ECBMJK->ABCIJK', H.ab.ovvo[Oa, Vb, Va, Ob], T.abb.VVVOOO, optimize=True)
    )

    dT.bbb.VVVOOO -= np.transpose(dT.bbb.VVVOOO, (1, 0, 2, 3, 4, 5)) + np.transpose(dT.bbb.VVVOOO, (0, 2, 1, 3, 4, 5)) \
           + np.transpose(dT.bbb.VVVOOO, (2, 1, 0, 3, 4, 5)) - np.transpose(dT.bbb.VVVOOO, (1, 2, 0, 3, 4, 5)) \
           - np.transpose(dT.bbb.VVVOOO, (2, 0, 1, 3, 4, 5))

    dT.bbb.VVVOOO -= np.transpose(dT.bbb.VVVOOO, (0, 1, 2, 4, 3, 5)) + np.transpose(dT.bbb.VVVOOO, (0, 1, 2, 3, 5, 4)) \
           + np.transpose(dT.bbb.VVVOOO, (0, 1, 2, 5, 4, 3)) - np.transpose(dT.bbb.VVVOOO, (0, 1, 2, 4, 5, 3)) \
           - np.transpose(dT.bbb.VVVOOO, (0, 1, 2, 5, 3, 4))

    return dT

def update(T, dT, H, shift, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.bbb.VVVOOO, dT.bbb.VVVOOO = cc_active_loops.update_t3d_111111(
        T.bbb.VVVOOO,
        dT.bbb.VVVOOO,
        H.b.oo[Ob, Ob],
        H.b.vv[Vb, Vb],
        H.b.oo[ob, ob],
        H.b.vv[vb, vb],
        shift,
    )

    return T, dT