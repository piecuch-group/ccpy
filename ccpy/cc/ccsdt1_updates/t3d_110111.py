import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import cc_active_loops

def build(T, dT, H, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    # MM(2,3)
    dT.bbb.VVvOOO = (6.0 / 12.0) * (
            -1.0 * np.einsum('AmIJ,BcmK->ABcIJK', H.bb.vooo[Vb, :, Ob, Ob], T.bb[Vb, vb, :, Ob], optimize=True)
    )
    dT.bbb.VVvOOO += (3.0 / 12.0) * (
            +1.0 * np.einsum('cmIJ,BAmK->ABcIJK', H.bb.vooo[vb, :, Ob, Ob], T.bb[Vb, Vb, :, Ob], optimize=True)
    )
    dT.bbb.VVvOOO += (3.0 / 12.0) * (
            +1.0 * np.einsum('ABIe,ecJK->ABcIJK', H.bb.vvov[Vb, Vb, Ob, :], T.bb[:, vb, Ob, Ob], optimize=True)
    )
    dT.bbb.VVvOOO += (6.0 / 12.0) * (
            -1.0 * np.einsum('cBIe,eAJK->ABcIJK', H.bb.vvov[vb, Vb, Ob, :], T.bb[:, Vb, Ob, Ob], optimize=True)
    )
    # (H(2) * T3)_C
    dT.bbb.VVvOOO += (3.0 / 12.0) * (
            +1.0 * np.einsum('mI,BAcmJK->ABcIJK', H.b.oo[ob, Ob], T.bbb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('MI,BAcMJK->ABcIJK', H.b.oo[Ob, Ob], T.bbb.VVvOOO, optimize=True)
    )
    dT.bbb.VVvOOO += (2.0 / 12.0) * (
            +1.0 * np.einsum('Ae,BceIJK->ABcIJK', H.b.vv[Vb, vb], T.bbb.VvvOOO, optimize=True)
            - 1.0 * np.einsum('AE,BEcIJK->ABcIJK', H.b.vv[Vb, Vb], T.bbb.VVvOOO, optimize=True)
    )
    dT.bbb.VVvOOO += (1.0 / 12.0) * (
            +1.0 * np.einsum('ce,ABeIJK->ABcIJK', H.b.vv[vb, vb], T.bbb.VVvOOO, optimize=True)
            + 1.0 * np.einsum('cE,ABEIJK->ABcIJK', H.b.vv[vb, Vb], T.bbb.VVVOOO, optimize=True)
    )
    dT.bbb.VVvOOO += (3.0 / 12.0) * (
            -0.5 * np.einsum('mnIJ,BAcmnK->ABcIJK', H.bb.oooo[ob, ob, Ob, Ob], T.bbb.VVvooO, optimize=True)
            - 1.0 * np.einsum('mNIJ,BAcmNK->ABcIJK', H.bb.oooo[ob, Ob, Ob, Ob], T.bbb.VVvoOO, optimize=True)
            - 0.5 * np.einsum('MNIJ,BAcMNK->ABcIJK', H.bb.oooo[Ob, Ob, Ob, Ob], T.bbb.VVvOOO, optimize=True)
    )
    dT.bbb.VVvOOO += (1.0 / 12.0) * (
            +1.0 * np.einsum('ABeF,FceIJK->ABcIJK', H.bb.vvvv[Vb, Vb, vb, Vb], T.bbb.VvvOOO, optimize=True)
            - 0.5 * np.einsum('ABEF,FEcIJK->ABcIJK', H.bb.vvvv[Vb, Vb, Vb, Vb], T.bbb.VVvOOO, optimize=True)
    )
    dT.bbb.VVvOOO += (2.0 / 12.0) * (
            +0.5 * np.einsum('cBef,AfeIJK->ABcIJK', H.bb.vvvv[vb, Vb, vb, vb], T.bbb.VvvOOO, optimize=True)
            + 1.0 * np.einsum('cBeF,AFeIJK->ABcIJK', H.bb.vvvv[vb, Vb, vb, Vb], T.bbb.VVvOOO, optimize=True)
            + 0.5 * np.einsum('cBEF,AFEIJK->ABcIJK', H.bb.vvvv[vb, Vb, Vb, Vb], T.bbb.VVVOOO, optimize=True)
    )
    dT.bbb.VVvOOO += (6.0 / 12.0) * (
            +1.0 * np.einsum('AmIe,BcemJK->ABcIJK', H.bb.voov[Vb, ob, Ob, vb], T.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('AmIE,BEcmJK->ABcIJK', H.bb.voov[Vb, ob, Ob, Vb], T.bbb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('AMIe,BceMJK->ABcIJK', H.bb.voov[Vb, Ob, Ob, vb], T.bbb.VvvOOO, optimize=True)
            - 1.0 * np.einsum('AMIE,BEcMJK->ABcIJK', H.bb.voov[Vb, Ob, Ob, Vb], T.bbb.VVvOOO, optimize=True)
    )
    dT.bbb.VVvOOO += (3.0 / 12.0) * (
            +1.0 * np.einsum('cmIe,ABemJK->ABcIJK', H.bb.voov[vb, ob, Ob, vb], T.bbb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('cmIE,ABEmJK->ABcIJK', H.bb.voov[vb, ob, Ob, Vb], T.bbb.VVVoOO, optimize=True)
            + 1.0 * np.einsum('cMIe,ABeMJK->ABcIJK', H.bb.voov[vb, Ob, Ob, vb], T.bbb.VVvOOO, optimize=True)
            + 1.0 * np.einsum('cMIE,ABEMJK->ABcIJK', H.bb.voov[vb, Ob, Ob, Vb], T.bbb.VVVOOO, optimize=True)
    )
    dT.bbb.VVvOOO += (6.0 / 12.0) * (
            +1.0 * np.einsum('mAeI,eBcmJK->ABcIJK', H.ab.ovvo[oa, Vb, va, Ob], T.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('mAEI,EBcmJK->ABcIJK', H.ab.ovvo[oa, Vb, Va, Ob], T.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('MAeI,eBcMJK->ABcIJK', H.ab.ovvo[Oa, Vb, va, Ob], T.abb.vVvOOO, optimize=True)
            + 1.0 * np.einsum('MAEI,EBcMJK->ABcIJK', H.ab.ovvo[Oa, Vb, Va, Ob], T.abb.VVvOOO, optimize=True)
    )
    dT.bbb.VVvOOO += (3.0 / 12.0) * (
            +1.0 * np.einsum('mceI,eABmJK->ABcIJK', H.ab.ovvo[oa, vb, va, Ob], T.abb.vVVoOO, optimize=True)
            + 1.0 * np.einsum('mcEI,EABmJK->ABcIJK', H.ab.ovvo[oa, vb, Va, Ob], T.abb.VVVoOO, optimize=True)
            + 1.0 * np.einsum('MceI,eABMJK->ABcIJK', H.ab.ovvo[Oa, vb, va, Ob], T.abb.vVVOOO, optimize=True)
            + 1.0 * np.einsum('McEI,EABMJK->ABcIJK', H.ab.ovvo[Oa, vb, Va, Ob], T.abb.VVVOOO, optimize=True)
    )

    dT.bbb.VVvOOO -= np.transpose(dT.bbb.VVvOOO, (1, 0, 2, 3, 4, 5))

    dT.bbb.VVvOOO -= np.transpose(dT.bbb.VVvOOO, (0, 1, 2, 4, 3, 5)) + np.transpose(dT.bbb.VVvOOO, (0, 1, 2, 3, 5, 4)) \
           + np.transpose(dT.bbb.VVvOOO, (0, 1, 2, 5, 4, 3)) - np.transpose(dT.bbb.VVvOOO, (0, 1, 2, 4, 5, 3)) \
           - np.transpose(dT.bbb.VVvOOO, (0, 1, 2, 5, 3, 4))

    return dT

def update(T, dT, H, shift, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.bbb.VVvOOO, dT.bbb.VVvOOO = cc_active_loops.update_t3d_110111(
        T.bbb.VVvOOO,
        dT.bbb.VVvOOO,
        H.b.oo[Ob, Ob],
        H.b.vv[Vb, Vb],
        H.b.oo[ob, ob],
        H.b.vv[vb, vb],
        shift,
    )

    return T, dT