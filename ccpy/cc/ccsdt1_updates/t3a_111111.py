import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import cc_active_loops

#@profile
def build(T, dT, H, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    # MM(2,3)A
    dT.aaa.VVVOOO = -0.25 * np.einsum("AmIJ,BCmK->ABCIJK", H.aa.vooo[Va, :, Oa, Oa], T.aa[Va, Va, :, Oa], optimize=True)
    dT.aaa.VVVOOO += 0.25 * np.einsum("ABIe,eCJK->ABCIJK", H.aa.vvov[Va, Va, Oa, :], T.aa[:, Va, Oa, Oa], optimize=True)

    # (H(2) * T3)_C
    dT.aaa.VVVOOO += (3.0 / 36.0) * (
            +1.0 * np.einsum('mI,CBAmJK->ABCIJK', H.a.oo[oa, Oa], T.aaa.VVVoOO, optimize=True)
            + 1.0 * np.einsum('MI,CBAMJK->ABCIJK', H.a.oo[Oa, Oa], T.aaa.VVVOOO, optimize=True)
    )
    dT.aaa.VVVOOO += (3.0 / 36.0) * (
            -1.0 * np.einsum('Ae,CBeIJK->ABCIJK', H.a.vv[Va, va], T.aaa.VVvOOO, optimize=True)
            - 1.0 * np.einsum('AE,CBEIJK->ABCIJK', H.a.vv[Va, Va], T.aaa.VVVOOO, optimize=True)
    )
    dT.aaa.VVVOOO += (3.0 / 36.0) * (
            -0.5 * np.einsum('mnIJ,CBAmnK->ABCIJK', H.aa.oooo[oa, oa, Oa, Oa], T.aaa.VVVooO, optimize=True)
            + 1.0 * np.einsum('MnIJ,CBAnMK->ABCIJK', H.aa.oooo[Oa, oa, Oa, Oa], T.aaa.VVVoOO, optimize=True)
            - 0.5 * np.einsum('MNIJ,CBAMNK->ABCIJK', H.aa.oooo[Oa, Oa, Oa, Oa], T.aaa.VVVOOO, optimize=True)
    )
    dT.aaa.VVVOOO += (3.0 / 36.0) * (
            -0.5 * np.einsum('ABef,CfeIJK->ABCIJK', H.aa.vvvv[Va, Va, va, va], T.aaa.VvvOOO, optimize=True)
            - 1.0 * np.einsum('ABeF,CFeIJK->ABCIJK', H.aa.vvvv[Va, Va, va, Va], T.aaa.VVvOOO, optimize=True)
            - 0.5 * np.einsum('ABEF,CFEIJK->ABCIJK', H.aa.vvvv[Va, Va, Va, Va], T.aaa.VVVOOO, optimize=True)
    )
    dT.aaa.VVVOOO += (9.0 / 36.0) * (
            -1.0 * np.einsum('AmIe,CBemJK->ABCIJK', H.aa.voov[Va, oa, Oa, va], T.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('AmIE,CBEmJK->ABCIJK', H.aa.voov[Va, oa, Oa, Va], T.aaa.VVVoOO, optimize=True)
            - 1.0 * np.einsum('AMIe,CBeMJK->ABCIJK', H.aa.voov[Va, Oa, Oa, va], T.aaa.VVvOOO, optimize=True)
            - 1.0 * np.einsum('AMIE,CBEMJK->ABCIJK', H.aa.voov[Va, Oa, Oa, Va], T.aaa.VVVOOO, optimize=True)
    )
    dT.aaa.VVVOOO += (9.0 / 36.0) * (
            -1.0 * np.einsum('AmIe,CBeJKm->ABCIJK', H.ab.voov[Va, ob, Oa, vb], T.aab.VVvOOo, optimize=True)
            - 1.0 * np.einsum('AmIE,CBEJKm->ABCIJK', H.ab.voov[Va, ob, Oa, Vb], T.aab.VVVOOo, optimize=True)
            - 1.0 * np.einsum('AMIe,CBeJKM->ABCIJK', H.ab.voov[Va, Ob, Oa, vb], T.aab.VVvOOO, optimize=True)
            - 1.0 * np.einsum('AMIE,CBEJKM->ABCIJK', H.ab.voov[Va, Ob, Oa, Vb], T.aab.VVVOOO, optimize=True)
    )

    dT.aaa.VVVOOO -= np.transpose(dT.aaa.VVVOOO, (1, 0, 2, 3, 4, 5)) + np.transpose(dT.aaa.VVVOOO, (0, 2, 1, 3, 4, 5)) \
           + np.transpose(dT.aaa.VVVOOO, (2, 1, 0, 3, 4, 5)) - np.transpose(dT.aaa.VVVOOO, (1, 2, 0, 3, 4, 5)) \
           - np.transpose(dT.aaa.VVVOOO, (2, 0, 1, 3, 4, 5))

    dT.aaa.VVVOOO -= np.transpose(dT.aaa.VVVOOO, (0, 1, 2, 4, 3, 5)) + np.transpose(dT.aaa.VVVOOO, (0, 1, 2, 3, 5, 4)) \
           + np.transpose(dT.aaa.VVVOOO, (0, 1, 2, 5, 4, 3)) - np.transpose(dT.aaa.VVVOOO, (0, 1, 2, 4, 5, 3)) \
           - np.transpose(dT.aaa.VVVOOO, (0, 1, 2, 5, 3, 4))

    return dT

def update(T, dT, H, shift, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.aaa.VVVOOO, dT.aaa.VVVOOO = cc_active_loops.update_t3a_111111(
        T.aaa.VVVOOO,
        dT.aaa.VVVOOO,
        H.a.oo[Oa, Oa],
        H.a.vv[Va, Va],
        H.a.oo[oa, oa],
        H.a.vv[va, va],
        shift,
    )

    return T, dT