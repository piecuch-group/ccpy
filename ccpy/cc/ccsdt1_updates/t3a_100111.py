import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import cc_active_loops

#@profile
def build(T, dT, H, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    # MM(2,3)
    dT.aaa.VvvOOO = (3.0 / 12.0) * (
            -1.0 * np.einsum('AmIJ,bcmK->AbcIJK', H.aa.vooo[Va, :, Oa, Oa], T.aa[va, va, :, Oa], optimize=True)
    )
    dT.aaa.VvvOOO += (6.0 / 12.0) * (
            +1.0 * np.einsum('bmIJ,AcmK->AbcIJK', H.aa.vooo[va, :, Oa, Oa], T.aa[Va, va, :, Oa], optimize=True)
    )
    dT.aaa.VvvOOO += (6.0 / 12.0) * (
            +1.0 * np.einsum('AbIe,ecJK->AbcIJK', H.aa.vvov[Va, va, Oa, :], T.aa[:, va, Oa, Oa], optimize=True)
    )
    dT.aaa.VvvOOO += (3.0 / 12.0) * (
            -1.0 * np.einsum('cbIe,eAJK->AbcIJK', H.aa.vvov[va, va, Oa, :], T.aa[:, Va, Oa, Oa], optimize=True)
    )
    # (H(2) * T3)_C
    dT.aaa.VvvOOO += (3.0 / 12.0) * (
            +1.0 * np.einsum('mI,AcbmJK->AbcIJK', H.a.oo[oa, Oa], T.aaa.VvvoOO, optimize=True)
            + 1.0 * np.einsum('MI,AcbMJK->AbcIJK', H.a.oo[Oa, Oa], T.aaa.VvvOOO, optimize=True)
    )
    dT.aaa.VvvOOO += (1.0 / 12.0) * (
            -1.0 * np.einsum('AE,EcbIJK->AbcIJK', H.a.vv[Va, Va], T.aaa.VvvOOO, optimize=True)
    )
    dT.aaa.VvvOOO += (2.0 / 12.0) * (
            -1.0 * np.einsum('be,AceIJK->AbcIJK', H.a.vv[va, va], T.aaa.VvvOOO, optimize=True)
            + 1.0 * np.einsum('bE,AEcIJK->AbcIJK', H.a.vv[va, Va], T.aaa.VVvOOO, optimize=True)
    )
    dT.aaa.VvvOOO += (3.0 / 12.0) * (
            -0.5 * np.einsum('mnIJ,AcbmnK->AbcIJK', H.aa.oooo[oa, oa, Oa, Oa], T.aaa.VvvooO, optimize=True)
            + 1.0 * np.einsum('MnIJ,AcbnMK->AbcIJK', H.aa.oooo[Oa, oa, Oa, Oa], T.aaa.VvvoOO, optimize=True)
            - 0.5 * np.einsum('MNIJ,AcbMNK->AbcIJK', H.aa.oooo[Oa, Oa, Oa, Oa], T.aaa.VvvOOO, optimize=True)
    )
    dT.aaa.VvvOOO += (2.0 / 12.0) * (
            -1.0 * np.einsum('AbEf,EcfIJK->AbcIJK', H.aa.vvvv[Va, va, Va, va], T.aaa.VvvOOO, optimize=True)
            - 0.5 * np.einsum('AbEF,FEcIJK->AbcIJK', H.aa.vvvv[Va, va, Va, Va], T.aaa.VVvOOO, optimize=True)
    )
    dT.aaa.VvvOOO += (1.0 / 12.0) * (
            +0.5 * np.einsum('cbef,AfeIJK->AbcIJK', H.aa.vvvv[va, va, va, va], T.aaa.VvvOOO, optimize=True)
            - 1.0 * np.einsum('cbEf,AEfIJK->AbcIJK', H.aa.vvvv[va, va, Va, va], T.aaa.VVvOOO, optimize=True)
            + 0.5 * np.einsum('cbEF,AFEIJK->AbcIJK', H.aa.vvvv[va, va, Va, Va], T.aaa.VVVOOO, optimize=True)
    )
    dT.aaa.VvvOOO += (3.0 / 12.0) * (
            -1.0 * np.einsum('AmIE,EcbmJK->AbcIJK', H.aa.voov[Va, oa, Oa, Va], T.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('AMIE,EcbMJK->AbcIJK', H.aa.voov[Va, Oa, Oa, Va], T.aaa.VvvOOO, optimize=True)
    )
    dT.aaa.VvvOOO += (6.0 / 12.0) * (
            -1.0 * np.einsum('bmIe,AcemJK->AbcIJK', H.aa.voov[va, oa, Oa, va], T.aaa.VvvoOO, optimize=True)
            + 1.0 * np.einsum('bmIE,AEcmJK->AbcIJK', H.aa.voov[va, oa, Oa, Va], T.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('bMIe,AceMJK->AbcIJK', H.aa.voov[va, Oa, Oa, va], T.aaa.VvvOOO, optimize=True)
            + 1.0 * np.einsum('bMIE,AEcMJK->AbcIJK', H.aa.voov[va, Oa, Oa, Va], T.aaa.VVvOOO, optimize=True)
    )
    dT.aaa.VvvOOO += (3.0 / 12.0) * (
            -1.0 * np.einsum('AmIE,cbEJKm->AbcIJK', H.ab.voov[Va, ob, Oa, Vb], T.aab.vvVOOo, optimize=True)
            - 1.0 * np.einsum('AMIE,cbEJKM->AbcIJK', H.ab.voov[Va, Ob, Oa, Vb], T.aab.vvVOOO, optimize=True)
    )
    dT.aaa.VvvOOO += (6.0 / 12.0) * (
            -1.0 * np.einsum('bmIe,AceJKm->AbcIJK', H.ab.voov[va, ob, Oa, vb], T.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('bmIE,AcEJKm->AbcIJK', H.ab.voov[va, ob, Oa, Vb], T.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('bMIe,AceJKM->AbcIJK', H.ab.voov[va, Ob, Oa, vb], T.aab.VvvOOO, optimize=True)
            - 1.0 * np.einsum('bMIE,AcEJKM->AbcIJK', H.ab.voov[va, Ob, Oa, Vb], T.aab.VvVOOO, optimize=True)
    )

    dT.aaa.VvvOOO -= np.transpose(dT.aaa.VvvOOO, (0, 1, 2, 4, 3, 5)) + np.transpose(dT.aaa.VvvOOO, (0, 1, 2, 3, 5, 4)) \
           + np.transpose(dT.aaa.VvvOOO, (0, 1, 2, 5, 4, 3)) - np.transpose(dT.aaa.VvvOOO, (0, 1, 2, 4, 5, 3)) \
           - np.transpose(dT.aaa.VvvOOO, (0, 1, 2, 5, 3, 4))

    dT.aaa.VvvOOO -= np.transpose(dT.aaa.VvvOOO, (0, 2, 1, 3, 4, 5))

    return dT

def update(T, dT, H, shift, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.aaa.VvvOOO, dT.aaa.VvvOOO = cc_active_loops.update_t3a_100111(
        T.aaa.VvvOOO,
        dT.aaa.VvvOOO,
        H.a.oo[Oa, Oa],
        H.a.vv[Va, Va],
        H.a.oo[oa, oa],
        H.a.vv[va, va],
        shift,
    )

    return T, dT