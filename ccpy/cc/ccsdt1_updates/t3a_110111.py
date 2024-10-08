import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import cc_active_loops

#@profile
def build(T, dT, H, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    # MM(2,3)
    dT.aaa.VVvOOO = (6.0 / 12.0) * (
            -1.0 * np.einsum('AmIJ,BcmK->ABcIJK', H.aa.vooo[Va, :, Oa, Oa], T.aa[Va, va, :, Oa], optimize=True)
    )
    dT.aaa.VVvOOO += (3.0 / 12.0) * (
            +1.0 * np.einsum('cmIJ,BAmK->ABcIJK', H.aa.vooo[va, :, Oa, Oa], T.aa[Va, Va, :, Oa], optimize=True)
    )
    dT.aaa.VVvOOO += (3.0 / 12.0) * (
            +1.0 * np.einsum('ABIe,ecJK->ABcIJK', H.aa.vvov[Va, Va, Oa, :], T.aa[:, va, Oa, Oa], optimize=True)
    )
    dT.aaa.VVvOOO += (6.0 / 12.0) * (
            -1.0 * np.einsum('cBIe,eAJK->ABcIJK', H.aa.vvov[va, Va, Oa, :], T.aa[:, Va, Oa, Oa], optimize=True)
    )
    # (H(2) * T3)_C
    dT.aaa.VVvOOO += (3.0 / 12.0) * (
            +1.0 * np.einsum('mI,BAcmJK->ABcIJK', H.a.oo[oa, Oa], T.aaa.VVvoOO, optimize=True)
            + 1.0 * np.einsum('MI,BAcMJK->ABcIJK', H.a.oo[Oa, Oa], T.aaa.VVvOOO, optimize=True)
    )
    dT.aaa.VVvOOO += (2.0 / 12.0) * (
            +1.0 * np.einsum('Ae,BceIJK->ABcIJK', H.a.vv[Va, va], T.aaa.VvvOOO, optimize=True)
            - 1.0 * np.einsum('AE,BEcIJK->ABcIJK', H.a.vv[Va, Va], T.aaa.VVvOOO, optimize=True)
    )
    dT.aaa.VVvOOO += (1.0 / 12.0) * (
            +1.0 * np.einsum('ce,ABeIJK->ABcIJK', H.a.vv[va, va], T.aaa.VVvOOO, optimize=True)
            + 1.0 * np.einsum('cE,ABEIJK->ABcIJK', H.a.vv[va, Va], T.aaa.VVVOOO, optimize=True)
    )
    dT.aaa.VVvOOO += (3.0 / 12.0) * (
            -0.5 * np.einsum('mnIJ,BAcmnK->ABcIJK', H.aa.oooo[oa, oa, Oa, Oa], T.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('MnIJ,BAcnMK->ABcIJK', H.aa.oooo[Oa, oa, Oa, Oa], T.aaa.VVvoOO, optimize=True)
            - 0.5 * np.einsum('MNIJ,BAcMNK->ABcIJK', H.aa.oooo[Oa, Oa, Oa, Oa], T.aaa.VVvOOO, optimize=True)
    )
    dT.aaa.VVvOOO += (1.0 / 12.0) * (
            +1.0 * np.einsum('ABeF,FceIJK->ABcIJK', H.aa.vvvv[Va, Va, va, Va], T.aaa.VvvOOO, optimize=True)
            - 0.5 * np.einsum('ABEF,FEcIJK->ABcIJK', H.aa.vvvv[Va, Va, Va, Va], T.aaa.VVvOOO, optimize=True)
    )
    dT.aaa.VVvOOO += (2.0 / 12.0) * (
            +0.5 * np.einsum('cBef,AfeIJK->ABcIJK', H.aa.vvvv[va, Va, va, va], T.aaa.VvvOOO, optimize=True)
            + 1.0 * np.einsum('cBeF,AFeIJK->ABcIJK', H.aa.vvvv[va, Va, va, Va], T.aaa.VVvOOO, optimize=True)
            + 0.5 * np.einsum('cBEF,AFEIJK->ABcIJK', H.aa.vvvv[va, Va, Va, Va], T.aaa.VVVOOO, optimize=True)
    )
    dT.aaa.VVvOOO += (6.0 / 12.0) * (
            +1.0 * np.einsum('AmIe,BcemJK->ABcIJK', H.aa.voov[Va, oa, Oa, va], T.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('AmIE,BEcmJK->ABcIJK', H.aa.voov[Va, oa, Oa, Va], T.aaa.VVvoOO, optimize=True)
            + 1.0 * np.einsum('AMIe,BceMJK->ABcIJK', H.aa.voov[Va, Oa, Oa, va], T.aaa.VvvOOO, optimize=True)
            - 1.0 * np.einsum('AMIE,BEcMJK->ABcIJK', H.aa.voov[Va, Oa, Oa, Va], T.aaa.VVvOOO, optimize=True)
    )
    dT.aaa.VVvOOO += (3.0 / 12.0) * (
            +1.0 * np.einsum('cmIe,ABemJK->ABcIJK', H.aa.voov[va, oa, Oa, va], T.aaa.VVvoOO, optimize=True)
            + 1.0 * np.einsum('cmIE,ABEmJK->ABcIJK', H.aa.voov[va, oa, Oa, Va], T.aaa.VVVoOO, optimize=True)
            + 1.0 * np.einsum('cMIe,ABeMJK->ABcIJK', H.aa.voov[va, Oa, Oa, va], T.aaa.VVvOOO, optimize=True)
            + 1.0 * np.einsum('cMIE,ABEMJK->ABcIJK', H.aa.voov[va, Oa, Oa, Va], T.aaa.VVVOOO, optimize=True)
    )
    dT.aaa.VVvOOO += (6.0 / 12.0) * (
            +1.0 * np.einsum('AmIe,BceJKm->ABcIJK', H.ab.voov[Va, ob, Oa, vb], T.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('AmIE,BcEJKm->ABcIJK', H.ab.voov[Va, ob, Oa, Vb], T.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('AMIe,BceJKM->ABcIJK', H.ab.voov[Va, Ob, Oa, vb], T.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('AMIE,BcEJKM->ABcIJK', H.ab.voov[Va, Ob, Oa, Vb], T.aab.VvVOOO, optimize=True)
    )
    dT.aaa.VVvOOO += (3.0 / 12.0) * (
            +1.0 * np.einsum('cmIe,ABeJKm->ABcIJK', H.ab.voov[va, ob, Oa, vb], T.aab.VVvOOo, optimize=True)
            + 1.0 * np.einsum('cmIE,ABEJKm->ABcIJK', H.ab.voov[va, ob, Oa, Vb], T.aab.VVVOOo, optimize=True)
            + 1.0 * np.einsum('cMIe,ABeJKM->ABcIJK', H.ab.voov[va, Ob, Oa, vb], T.aab.VVvOOO, optimize=True)
            + 1.0 * np.einsum('cMIE,ABEJKM->ABcIJK', H.ab.voov[va, Ob, Oa, Vb], T.aab.VVVOOO, optimize=True)
    )

    dT.aaa.VVvOOO -= np.transpose(dT.aaa.VVvOOO, (1, 0, 2, 3, 4, 5))

    dT.aaa.VVvOOO -= np.transpose(dT.aaa.VVvOOO, (0, 1, 2, 4, 3, 5)) + np.transpose(dT.aaa.VVvOOO, (0, 1, 2, 3, 5, 4)) \
           + np.transpose(dT.aaa.VVvOOO, (0, 1, 2, 5, 4, 3)) - np.transpose(dT.aaa.VVvOOO, (0, 1, 2, 4, 5, 3)) \
           - np.transpose(dT.aaa.VVvOOO, (0, 1, 2, 5, 3, 4))

    return dT

def update(T, dT, H, shift, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.aaa.VVvOOO, dT.aaa.VVvOOO = cc_active_loops.update_t3a_110111(
        T.aaa.VVvOOO,
        dT.aaa.VVvOOO,
        H.a.oo[Oa, Oa],
        H.a.vv[Va, Va],
        H.a.oo[oa, oa],
        H.a.vv[va, va],
        shift,
    )

    return T, dT