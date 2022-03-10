import numpy as np
from ccpy.utilities.active_space import get_active_slices

def update(T, dT, H, H0, shift, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

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
    dT.aaa.VvvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('AbeF,FceijK->AbcijK', H.aa.vvvv[Va, va, va, Va], T.aaa.VvvooO, optimize=True)
            - 0.5 * np.einsum('AbEF,FEcijK->AbcijK', H.aa.vvvv[Va, va, Va, Va], T.aaa.VVvooO, optimize=True)
    )
    dT.aaa.VvvooO += (1.0 / 4.0) * (
            +0.5 * np.einsum('cbef,AfeijK->AbcijK', H.aa.vvvv[va, va, va, va], T.aaa.VvvooO, optimize=True)
            + 1.0 * np.einsum('cbeF,AFeijK->AbcijK', H.aa.vvvv[va, va, va, Va], T.aaa.VVvooO, optimize=True)
            + 0.5 * np.einsum('cbEF,AFEijK->AbcijK', H.aa.vvvv[va, va, Va, Va], T.aaa.VVVooO, optimize=True)
    )
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

    return T, dT