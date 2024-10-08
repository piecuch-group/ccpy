import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import cc_active_loops

#@profile
def build(T, dT, H, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    # MM(2,3)
    dT.aaa.VVVooO = (3.0 / 12.0) * (
            -1.0 * np.einsum('Amij,BCmK->ABCijK', H.aa.vooo[Va, :, oa, oa], T.aa[Va, Va, :, Oa], optimize=True)
    )
    dT.aaa.VVVooO += (6.0 / 12.0) * (
            +1.0 * np.einsum('AmKj,BCmi->ABCijK', H.aa.vooo[Va, :, Oa, oa], T.aa[Va, Va, :, oa], optimize=True)
    )
    dT.aaa.VVVooO += (6.0 / 12.0) * (
            +1.0 * np.einsum('ABie,eCjK->ABCijK', H.aa.vvov[Va, Va, oa, :], T.aa[:, Va, oa, Oa], optimize=True)
    )
    dT.aaa.VVVooO += (3.0 / 12.0) * (
            -1.0 * np.einsum('ABKe,eCji->ABCijK', H.aa.vvov[Va, Va, Oa, :], T.aa[:, Va, oa, oa], optimize=True)
    )
    # (H(2) * T3)_C
    dT.aaa.VVVooO += (2.0 / 12.0) * (
            +1.0 * np.einsum('mi,CBAmjK->ABCijK', H.a.oo[oa, oa], T.aaa.VVVooO, optimize=True)
            - 1.0 * np.einsum('Mi,CBAjMK->ABCijK', H.a.oo[Oa, oa], T.aaa.VVVoOO, optimize=True)
    )
    dT.aaa.VVVooO += (1.0 / 12.0) * (
            -1.0 * np.einsum('MK,CBAjiM->ABCijK', H.a.oo[Oa, Oa], T.aaa.VVVooO, optimize=True)
    )
    dT.aaa.VVVooO += (3.0 / 12.0) * (
            -1.0 * np.einsum('Ae,CBeijK->ABCijK', H.a.vv[Va, va], T.aaa.VVvooO, optimize=True)
            - 1.0 * np.einsum('AE,CBEijK->ABCijK', H.a.vv[Va, Va], T.aaa.VVVooO, optimize=True)
    )
    dT.aaa.VVVooO += (1.0 / 12.0) * (
            -0.5 * np.einsum('mnij,CBAmnK->ABCijK', H.aa.oooo[oa, oa, oa, oa], T.aaa.VVVooO, optimize=True)
            + 1.0 * np.einsum('Mnij,CBAnMK->ABCijK', H.aa.oooo[Oa, oa, oa, oa], T.aaa.VVVoOO, optimize=True)
            - 0.5 * np.einsum('MNij,CBAMNK->ABCijK', H.aa.oooo[Oa, Oa, oa, oa], T.aaa.VVVOOO, optimize=True)
    )
    dT.aaa.VVVooO += (2.0 / 12.0) * (
            +1.0 * np.einsum('MnKj,CBAniM->ABCijK', H.aa.oooo[Oa, oa, Oa, oa], T.aaa.VVVooO, optimize=True)
            + 0.5 * np.einsum('MNKj,CBAiMN->ABCijK', H.aa.oooo[Oa, Oa, Oa, oa], T.aaa.VVVoOO, optimize=True)
    )
    dT.aaa.VVVooO += (3.0 / 12.0) * (
            -0.5 * np.einsum('ABef,CfeijK->ABCijK', H.aa.vvvv[Va, Va, va, va], T.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('ABeF,CFeijK->ABCijK', H.aa.vvvv[Va, Va, va, Va], T.aaa.VVvooO, optimize=True)
            - 0.5 * np.einsum('ABEF,CFEijK->ABCijK', H.aa.vvvv[Va, Va, Va, Va], T.aaa.VVVooO, optimize=True)
    )
    dT.aaa.VVVooO += (6.0 / 12.0) * (
            -1.0 * np.einsum('Amie,CBemjK->ABCijK', H.aa.voov[Va, oa, oa, va], T.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('AMie,CBejMK->ABCijK', H.aa.voov[Va, Oa, oa, va], T.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('AmiE,CBEmjK->ABCijK', H.aa.voov[Va, oa, oa, Va], T.aaa.VVVooO, optimize=True)
            + 1.0 * np.einsum('AMiE,CBEjMK->ABCijK', H.aa.voov[Va, Oa, oa, Va], T.aaa.VVVoOO, optimize=True)
    )
    dT.aaa.VVVooO += (3.0 / 12.0) * (
            +1.0 * np.einsum('AMKe,CBejiM->ABCijK', H.aa.voov[Va, Oa, Oa, va], T.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('AMKE,CBEjiM->ABCijK', H.aa.voov[Va, Oa, Oa, Va], T.aaa.VVVooO, optimize=True)
    )
    dT.aaa.VVVooO += (6.0 / 12.0) * (
            -1.0 * np.einsum('Amie,CBejKm->ABCijK', H.ab.voov[Va, ob, oa, vb], T.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('AMie,CBejKM->ABCijK', H.ab.voov[Va, Ob, oa, vb], T.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('AmiE,CBEjKm->ABCijK', H.ab.voov[Va, ob, oa, Vb], T.aab.VVVoOo, optimize=True)
            - 1.0 * np.einsum('AMiE,CBEjKM->ABCijK', H.ab.voov[Va, Ob, oa, Vb], T.aab.VVVoOO, optimize=True)
    )
    dT.aaa.VVVooO += (3.0 / 12.0) * (
            +1.0 * np.einsum('AMKe,CBejiM->ABCijK', H.ab.voov[Va, Ob, Oa, vb], T.aab.VVvooO, optimize=True)
            + 1.0 * np.einsum('AMKE,CBEjiM->ABCijK', H.ab.voov[Va, Ob, Oa, Vb], T.aab.VVVooO, optimize=True)
    )

    dT.aaa.VVVooO -= np.transpose(dT.aaa.VVVooO, (1, 0, 2, 3, 4, 5)) + np.transpose(dT.aaa.VVVooO, (0, 2, 1, 3, 4, 5)) \
           + np.transpose(dT.aaa.VVVooO, (2, 1, 0, 3, 4, 5)) - np.transpose(dT.aaa.VVVooO, (1, 2, 0, 3, 4, 5)) \
           - np.transpose(dT.aaa.VVVooO, (2, 0, 1, 3, 4, 5))

    dT.aaa.VVVooO -= np.transpose(dT.aaa.VVVooO, (0, 1, 2, 4, 3, 5))

    return dT

def update(T, dT, H, shift, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.aaa.VVVooO, dT.aaa.VVVooO = cc_active_loops.update_t3a_111001(
        T.aaa.VVVooO,
        dT.aaa.VVVooO,
        H.a.oo[Oa, Oa],
        H.a.vv[Va, Va],
        H.a.oo[oa, oa],
        H.a.vv[va, va],
        shift,
    )

    return T, dT