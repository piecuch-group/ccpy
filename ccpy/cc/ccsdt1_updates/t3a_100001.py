import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import cc_active_loops

import time

#@profile
def build(T, dT, H, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    # MM(2,3)
    dT.aaa.VvvooO = (1.0 / 4.0) * (
            -1.0 * np.einsum('Amij,bcmK->AbcijK', H.aa.vooo[Va, :, oa, oa], T.aa[va, va, :, Oa], optimize=True)
    )
    dT.aaa.VvvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('bmij,AcmK->AbcijK', H.aa.vooo[va, :, oa, oa], T.aa[Va, va, :, Oa], optimize=True)
    )
    dT.aaa.VvvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('AmKj,bcmi->AbcijK', H.aa.vooo[Va, :, Oa, oa], T.aa[va, va, :, oa], optimize=True)
    )
    dT.aaa.VvvooO += (4.0 / 4.0) * (
            -1.0 * np.einsum('bmKj,Acmi->AbcijK', H.aa.vooo[va, :, Oa, oa], T.aa[Va, va, :, oa], optimize=True)
    )
    dT.aaa.VvvooO += (4.0 / 4.0) * (
            +1.0 * np.einsum('Abie,ecjK->AbcijK', H.aa.vvov[Va, va, oa, :], T.aa[:, va, oa, Oa], optimize=True)
    )
    dT.aaa.VvvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('cbie,eAjK->AbcijK', H.aa.vvov[va, va, oa, :], T.aa[:, Va, oa, Oa], optimize=True)
    )
    dT.aaa.VvvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('AbKe,ecji->AbcijK', H.aa.vvov[Va, va, Oa, :], T.aa[:, va, oa, oa], optimize=True)
    )
    dT.aaa.VvvooO += (1.0 / 4.0) * (
            +1.0 * np.einsum('cbKe,eAji->AbcijK', H.aa.vvov[va, va, Oa, :], T.aa[:, Va, oa, oa], optimize=True)
    )
    # (H(2) * T3)_C
    dT.aaa.VvvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mi,AcbmjK->AbcijK', H.a.oo[oa, oa], T.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('Mi,AcbjMK->AbcijK', H.a.oo[Oa, oa], T.aaa.VvvoOO, optimize=True)
    )
    dT.aaa.VvvooO += (1.0 / 4.0) * (
            -1.0 * np.einsum('MK,AcbjiM->AbcijK', H.a.oo[Oa, Oa], T.aaa.VvvooO, optimize=True)
    )
    dT.aaa.VvvooO += (1.0 / 4.0) * (
            -1.0 * np.einsum('AE,EcbijK->AbcijK', H.a.vv[Va, Va], T.aaa.VvvooO, optimize=True)
    )
    dT.aaa.VvvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('be,AceijK->AbcijK', H.a.vv[va, va], T.aaa.VvvooO, optimize=True)
            + 1.0 * np.einsum('bE,AEcijK->AbcijK', H.a.vv[va, Va], T.aaa.VVvooO, optimize=True)
    )
    dT.aaa.VvvooO += (1.0 / 4.0) * (
            -0.5 * np.einsum('mnij,AcbmnK->AbcijK', H.aa.oooo[oa, oa, oa, oa], T.aaa.VvvooO, optimize=True)
            + 1.0 * np.einsum('Mnij,AcbnMK->AbcijK', H.aa.oooo[Oa, oa, oa, oa], T.aaa.VvvoOO, optimize=True)
            - 0.5 * np.einsum('MNij,AcbMNK->AbcijK', H.aa.oooo[Oa, Oa, oa, oa], T.aaa.VvvOOO, optimize=True)
    )
    dT.aaa.VvvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('MnKj,AcbniM->AbcijK', H.aa.oooo[Oa, oa, Oa, oa], T.aaa.VvvooO, optimize=True)
            + 0.5 * np.einsum('MNKj,AcbiMN->AbcijK', H.aa.oooo[Oa, Oa, Oa, oa], T.aaa.VvvoOO, optimize=True)
    )
    #t1 = time.time()
    dT.aaa.VvvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('AbeF,FceijK->AbcijK', H.aa.vvvv[Va, va, va, Va], T.aaa.VvvooO, optimize=True)
            - 0.5 * np.einsum('AbEF,FEcijK->AbcijK', H.aa.vvvv[Va, va, Va, Va], T.aaa.VVvooO, optimize=True)
    )
    dT.aaa.VvvooO += (1.0 / 4.0) * (
            +0.5 * np.einsum('cbef,AfeijK->AbcijK', H.aa.vvvv[va, va, va, va], T.aaa.VvvooO, optimize=True)
            + 1.0 * np.einsum('cbeF,AFeijK->AbcijK', H.aa.vvvv[va, va, va, Va], T.aaa.VVvooO, optimize=True)
            + 0.5 * np.einsum('cbEF,AFEijK->AbcijK', H.aa.vvvv[va, va, Va, Va], T.aaa.VVVooO, optimize=True)
    )
    #print('Time for t3a vvvv =', time.time() - t1)
    dT.aaa.VvvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('AmiE,EcbmjK->AbcijK', H.aa.voov[Va, oa, oa, Va], T.aaa.VvvooO, optimize=True)
            + 1.0 * np.einsum('AMiE,EcbjMK->AbcijK', H.aa.voov[Va, Oa, oa, Va], T.aaa.VvvoOO, optimize=True)
    )
    dT.aaa.VvvooO += (4.0 / 4.0) * (
            -1.0 * np.einsum('bmie,AcemjK->AbcijK', H.aa.voov[va, oa, oa, va], T.aaa.VvvooO, optimize=True)
            + 1.0 * np.einsum('bmiE,AEcmjK->AbcijK', H.aa.voov[va, oa, oa, Va], T.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('bMie,AcejMK->AbcijK', H.aa.voov[va, Oa, oa, va], T.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('bMiE,AEcjMK->AbcijK', H.aa.voov[va, Oa, oa, Va], T.aaa.VVvoOO, optimize=True)
    )
    dT.aaa.VvvooO += (1.0 / 4.0) * (
            +1.0 * np.einsum('AMKE,EcbjiM->AbcijK', H.aa.voov[Va, Oa, Oa, Va], T.aaa.VvvooO, optimize=True)
    )
    dT.aaa.VvvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('bMKe,AcejiM->AbcijK', H.aa.voov[va, Oa, Oa, va], T.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('bMKE,AEcjiM->AbcijK', H.aa.voov[va, Oa, Oa, Va], T.aaa.VVvooO, optimize=True)
    )
    dT.aaa.VvvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('AmiE,cbEjKm->AbcijK', H.ab.voov[Va, ob, oa, Vb], T.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('AMiE,cbEjKM->AbcijK', H.ab.voov[Va, Ob, oa, Vb], T.aab.vvVoOO, optimize=True)
    )
    dT.aaa.VvvooO += (4.0 / 4.0) * (
            -1.0 * np.einsum('bmie,AcejKm->AbcijK', H.ab.voov[va, ob, oa, vb], T.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('bmiE,AcEjKm->AbcijK', H.ab.voov[va, ob, oa, Vb], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('bMie,AcejKM->AbcijK', H.ab.voov[va, Ob, oa, vb], T.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('bMiE,AcEjKM->AbcijK', H.ab.voov[va, Ob, oa, Vb], T.aab.VvVoOO, optimize=True)
    )
    dT.aaa.VvvooO += (1.0 / 4.0) * (
            +1.0 * np.einsum('AMKE,cbEjiM->AbcijK', H.ab.voov[Va, Ob, Oa, Vb], T.aab.vvVooO, optimize=True)
    )
    dT.aaa.VvvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('bMKe,AcejiM->AbcijK', H.ab.voov[va, Ob, Oa, vb], T.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('bMKE,AcEjiM->AbcijK', H.ab.voov[va, Ob, Oa, Vb], T.aab.VvVooO, optimize=True)
    )

    dT.aaa.VvvooO -= np.transpose(dT.aaa.VvvooO, (0, 2, 1, 3, 4, 5))
    dT.aaa.VvvooO -= np.transpose(dT.aaa.VvvooO, (0, 1, 2, 4, 3, 5))

    return dT


def update(T, dT, H, shift, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.aaa.VvvooO, dT.aaa.VvvooO = cc_active_loops.update_t3a_100001(
        T.aaa.VvvooO,
        dT.aaa.VvvooO,
        H.a.oo[Oa, Oa],
        H.a.vv[Va, Va],
        H.a.oo[oa, oa],
        H.a.vv[va, va],
        shift,
    )

    return T, dT