import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops

def build(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.aaa.VVVooO = (3.0 / 12.0) * (
            -1.0 * np.einsum('Bmji,ACmK->ABCijK', X.aa.vooo[Va, :, oa, oa], T.aa[Va, Va, :, Oa], optimize=True)
    )
    dR.aaa.VVVooO += (6.0 / 12.0) * (
            +1.0 * np.einsum('BmjK,ACmi->ABCijK', X.aa.vooo[Va, :, oa, Oa], T.aa[Va, Va, :, oa], optimize=True)
    )
    dR.aaa.VVVooO += (3.0 / 12.0) * (
            -1.0 * np.einsum('Bmji,ACmK->ABCijK', H.aa.vooo[Va, :, oa, oa], R.aa[Va, Va, :, Oa], optimize=True)
    )
    dR.aaa.VVVooO += (6.0 / 12.0) * (
            +1.0 * np.einsum('BmjK,ACmi->ABCijK', H.aa.vooo[Va, :, oa, Oa], R.aa[Va, Va, :, oa], optimize=True)
    )
    dR.aaa.VVVooO += (6.0 / 12.0) * (
            +1.0 * np.einsum('BAje,eCiK->ABCijK', X.aa.vvov[Va, Va, oa, :], T.aa[:, Va, oa, Oa], optimize=True)
    )
    dR.aaa.VVVooO += (3.0 / 12.0) * (
            -1.0 * np.einsum('BAKe,eCij->ABCijK', X.aa.vvov[Va, Va, Oa, :], T.aa[:, Va, oa, oa], optimize=True)
    )
    dR.aaa.VVVooO += (6.0 / 12.0) * (
            +1.0 * np.einsum('BAje,eCiK->ABCijK', H.aa.vvov[Va, Va, oa, :], R.aa[:, Va, oa, Oa], optimize=True)
    )
    dR.aaa.VVVooO += (3.0 / 12.0) * (
            -1.0 * np.einsum('BAKe,eCij->ABCijK', H.aa.vvov[Va, Va, Oa, :], R.aa[:, Va, oa, oa], optimize=True)
    )

    dR.aaa.VVVooO += (3.0 / 12.0) * (
            +1.0 * np.einsum('Be,CAeijK->ABCijK', X.a.vv[Va, va], T.aaa.VVvooO, optimize=True)
            - 1.0 * np.einsum('BE,CEAijK->ABCijK', X.a.vv[Va, Va], T.aaa.VVVooO, optimize=True)
    )
    dR.aaa.VVVooO += (2.0 / 12.0) * (
            +1.0 * np.einsum('mj,CBAimK->ABCijK', X.a.oo[oa, oa], T.aaa.VVVooO, optimize=True)
            + 1.0 * np.einsum('Mj,CBAiMK->ABCijK', X.a.oo[Oa, oa], T.aaa.VVVoOO, optimize=True)
    )
    dR.aaa.VVVooO += (1.0 / 12.0) * (
            +1.0 * np.einsum('MK,CBAijM->ABCijK', X.a.oo[Oa, Oa], T.aaa.VVVooO, optimize=True)
    )
    dR.aaa.VVVooO += (1.0 / 12.0) * (
            -0.5 * np.einsum('mnij,CBAmnK->ABCijK', X.aa.oooo[oa, oa, oa, oa], T.aaa.VVVooO, optimize=True)
            + 1.0 * np.einsum('Mnij,CBAnMK->ABCijK', X.aa.oooo[Oa, oa, oa, oa], T.aaa.VVVoOO, optimize=True)
            - 0.5 * np.einsum('MNij,CBAMNK->ABCijK', X.aa.oooo[Oa, Oa, oa, oa], T.aaa.VVVOOO, optimize=True)
    )
    dR.aaa.VVVooO += (2.0 / 12.0) * (
            +1.0 * np.einsum('MnKj,CBAniM->ABCijK', X.aa.oooo[Oa, oa, Oa, oa], T.aaa.VVVooO, optimize=True)
            + 0.5 * np.einsum('MNKj,CBAiMN->ABCijK', X.aa.oooo[Oa, Oa, Oa, oa], T.aaa.VVVoOO, optimize=True)
    )
    dR.aaa.VVVooO += (3.0 / 12.0) * (
            -0.5 * np.einsum('ABef,CfeijK->ABCijK', X.aa.vvvv[Va, Va, va, va], T.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('ABeF,CFeijK->ABCijK', X.aa.vvvv[Va, Va, va, Va], T.aaa.VVvooO, optimize=True)
            - 0.5 * np.einsum('ABEF,CFEijK->ABCijK', X.aa.vvvv[Va, Va, Va, Va], T.aaa.VVVooO, optimize=True)
    )
    dR.aaa.VVVooO += (6.0 / 12.0) * (
            +1.0 * np.einsum('Bmje,CAeimK->ABCijK', X.aa.voov[Va, oa, oa, va], T.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('BMje,CAeiMK->ABCijK', X.aa.voov[Va, Oa, oa, va], T.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('BmjE,CEAimK->ABCijK', X.aa.voov[Va, oa, oa, Va], T.aaa.VVVooO, optimize=True)
            - 1.0 * np.einsum('BMjE,CEAiMK->ABCijK', X.aa.voov[Va, Oa, oa, Va], T.aaa.VVVoOO, optimize=True)
    )
    dR.aaa.VVVooO += (3.0 / 12.0) * (
            +1.0 * np.einsum('BMKe,CAeijM->ABCijK', X.aa.voov[Va, Oa, Oa, va], T.aaa.VVvooO, optimize=True)
            - 1.0 * np.einsum('BMKE,CEAijM->ABCijK', X.aa.voov[Va, Oa, Oa, Va], T.aaa.VVVooO, optimize=True)
    )
    dR.aaa.VVVooO += (6.0 / 12.0) * (
            -1.0 * np.einsum('Bmje,CAeiKm->ABCijK', X.ab.voov[Va, ob, oa, vb], T.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('BMje,CAeiKM->ABCijK', X.ab.voov[Va, Ob, oa, vb], T.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('BmjE,CAEiKm->ABCijK', X.ab.voov[Va, ob, oa, Vb], T.aab.VVVoOo, optimize=True)
            - 1.0 * np.einsum('BMjE,CAEiKM->ABCijK', X.ab.voov[Va, Ob, oa, Vb], T.aab.VVVoOO, optimize=True)
    )
    dR.aaa.VVVooO += (3.0 / 12.0) * (
            +1.0 * np.einsum('BMKe,CAeijM->ABCijK', X.ab.voov[Va, Ob, Oa, vb], T.aab.VVvooO, optimize=True)
            + 1.0 * np.einsum('BMKE,CAEijM->ABCijK', X.ab.voov[Va, Ob, Oa, Vb], T.aab.VVVooO, optimize=True)
    )
    dR.aaa.VVVooO += (2.0 / 12.0) * (
            +1.0 * np.einsum('mj,CBAimK->ABCijK', H.a.oo[oa, oa], R.aaa.VVVooO, optimize=True)
            + 1.0 * np.einsum('Mj,CBAiMK->ABCijK', H.a.oo[Oa, oa], R.aaa.VVVoOO, optimize=True)
    )
    dR.aaa.VVVooO += (1.0 / 12.0) * (
            +1.0 * np.einsum('MK,CBAijM->ABCijK', H.a.oo[Oa, Oa], R.aaa.VVVooO, optimize=True)
    )
    dR.aaa.VVVooO += (3.0 / 12.0) * (
            +1.0 * np.einsum('Be,CAeijK->ABCijK', H.a.vv[Va, va], R.aaa.VVvooO, optimize=True)
            - 1.0 * np.einsum('BE,CEAijK->ABCijK', H.a.vv[Va, Va], R.aaa.VVVooO, optimize=True)
    )
    dR.aaa.VVVooO += (1.0 / 12.0) * (
            -0.5 * np.einsum('mnij,CBAmnK->ABCijK', H.aa.oooo[oa, oa, oa, oa], R.aaa.VVVooO, optimize=True)
            + 1.0 * np.einsum('Mnij,CBAnMK->ABCijK', H.aa.oooo[Oa, oa, oa, oa], R.aaa.VVVoOO, optimize=True)
            - 0.5 * np.einsum('MNij,CBAMNK->ABCijK', H.aa.oooo[Oa, Oa, oa, oa], R.aaa.VVVOOO, optimize=True)
    )
    dR.aaa.VVVooO += (2.0 / 12.0) * (
            +1.0 * np.einsum('MnKj,CBAniM->ABCijK', H.aa.oooo[Oa, oa, Oa, oa], R.aaa.VVVooO, optimize=True)
            + 0.5 * np.einsum('MNKj,CBAiMN->ABCijK', H.aa.oooo[Oa, Oa, Oa, oa], R.aaa.VVVoOO, optimize=True)
    )
    dR.aaa.VVVooO += (3.0 / 12.0) * (
            -0.5 * np.einsum('ABef,CfeijK->ABCijK', H.aa.vvvv[Va, Va, va, va], R.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('ABeF,CFeijK->ABCijK', H.aa.vvvv[Va, Va, va, Va], R.aaa.VVvooO, optimize=True)
            - 0.5 * np.einsum('ABEF,CFEijK->ABCijK', H.aa.vvvv[Va, Va, Va, Va], R.aaa.VVVooO, optimize=True)
    )
    dR.aaa.VVVooO += (6.0 / 12.0) * (
            -1.0 * np.einsum('Amie,CBemjK->ABCijK', H.aa.voov[Va, oa, oa, va], R.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('AMie,CBejMK->ABCijK', H.aa.voov[Va, Oa, oa, va], R.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('AmiE,CBEmjK->ABCijK', H.aa.voov[Va, oa, oa, Va], R.aaa.VVVooO, optimize=True)
            + 1.0 * np.einsum('AMiE,CBEjMK->ABCijK', H.aa.voov[Va, Oa, oa, Va], R.aaa.VVVoOO, optimize=True)
    )
    dR.aaa.VVVooO += (3.0 / 12.0) * (
            +1.0 * np.einsum('AMKe,CBejiM->ABCijK', H.aa.voov[Va, Oa, Oa, va], R.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('AMKE,CBEjiM->ABCijK', H.aa.voov[Va, Oa, Oa, Va], R.aaa.VVVooO, optimize=True)
    )
    dR.aaa.VVVooO += (6.0 / 12.0) * (
            -1.0 * np.einsum('Amie,CBejKm->ABCijK', H.ab.voov[Va, ob, oa, vb], R.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('AMie,CBejKM->ABCijK', H.ab.voov[Va, Ob, oa, vb], R.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('AmiE,CBEjKm->ABCijK', H.ab.voov[Va, ob, oa, Vb], R.aab.VVVoOo, optimize=True)
            - 1.0 * np.einsum('AMiE,CBEjKM->ABCijK', H.ab.voov[Va, Ob, oa, Vb], R.aab.VVVoOO, optimize=True)
    )
    dR.aaa.VVVooO += (3.0 / 12.0) * (
            +1.0 * np.einsum('AMKe,CBejiM->ABCijK', H.ab.voov[Va, Ob, Oa, vb], R.aab.VVvooO, optimize=True)
            + 1.0 * np.einsum('AMKE,CBEjiM->ABCijK', H.ab.voov[Va, Ob, Oa, Vb], R.aab.VVVooO, optimize=True)
    )

    dR.aaa.VVVooO -= np.transpose(dR.aaa.VVVooO, (1, 0, 2, 3, 4, 5)) + np.transpose(dR.aaa.VVVooO, (0, 2, 1, 3, 4, 5)) \
           + np.transpose(dR.aaa.VVVooO, (2, 1, 0, 3, 4, 5)) - np.transpose(dR.aaa.VVVooO, (1, 2, 0, 3, 4, 5)) \
           - np.transpose(dR.aaa.VVVooO, (2, 0, 1, 3, 4, 5))

    dR.aaa.VVVooO -= np.transpose(dR.aaa.VVVooO, (0, 1, 2, 4, 3, 5))

    return dR

def update(R, omega, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    R.aaa.VVVooO = eomcc_active_loops.update_r3a_111001(
        R.aaa.VVVooO,
        omega,
        H.a.oo[Oa, Oa],
        H.a.vv[Va, Va],
        H.a.oo[oa, oa],
        H.a.vv[va, va],
        0.0,
    )

    return R