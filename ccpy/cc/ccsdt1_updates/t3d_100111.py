import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import cc_active_loops

def build(T, dT, H, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    # MM(2,3)
    dT.bbb.VvvOOO = (3.0 / 12.0) * (
            -1.0 * np.einsum('AmIJ,bcmK->AbcIJK', H.bb.vooo[Vb, :, Ob, Ob], T.bb[vb, vb, :, Ob], optimize=True)
    )
    dT.bbb.VvvOOO += (6.0 / 12.0) * (
            +1.0 * np.einsum('bmIJ,AcmK->AbcIJK', H.bb.vooo[vb, :, Ob, Ob], T.bb[Vb, vb, :, Ob], optimize=True)
    )
    dT.bbb.VvvOOO += (6.0 / 12.0) * (
            +1.0 * np.einsum('AbIe,ecJK->AbcIJK', H.bb.vvov[Vb, vb, Ob, :], T.bb[:, vb, Ob, Ob], optimize=True)
    )
    dT.bbb.VvvOOO += (3.0 / 12.0) * (
            -1.0 * np.einsum('cbIe,eAJK->AbcIJK', H.bb.vvov[vb, vb, Ob, :], T.bb[:, Vb, Ob, Ob], optimize=True)
    )
    # (H(2) * T3)_C
    dT.bbb.VvvOOO += (3.0 / 12.0) * (
            +1.0 * np.einsum('mI,AcbmJK->AbcIJK', H.b.oo[ob, Ob], T.bbb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('MI,AcbMJK->AbcIJK', H.b.oo[Ob, Ob], T.bbb.VvvOOO, optimize=True)
    )
    dT.bbb.VvvOOO += (1.0 / 12.0) * (
            -1.0 * np.einsum('AE,EcbIJK->AbcIJK', H.b.vv[Vb, Vb], T.bbb.VvvOOO, optimize=True)
    )
    dT.bbb.VvvOOO += (2.0 / 12.0) * (
            -1.0 * np.einsum('be,AceIJK->AbcIJK', H.b.vv[vb, vb], T.bbb.VvvOOO, optimize=True)
            + 1.0 * np.einsum('bE,AEcIJK->AbcIJK', H.b.vv[vb, Vb], T.bbb.VVvOOO, optimize=True)
    )
    dT.bbb.VvvOOO += (3.0 / 12.0) * (
            -0.5 * np.einsum('mnIJ,AcbmnK->AbcIJK', H.bb.oooo[ob, ob, Ob, Ob], T.bbb.VvvooO, optimize=True)
            + 1.0 * np.einsum('MnIJ,AcbnMK->AbcIJK', H.bb.oooo[Ob, ob, Ob, Ob], T.bbb.VvvoOO, optimize=True)
            - 0.5 * np.einsum('MNIJ,AcbMNK->AbcIJK', H.bb.oooo[Ob, Ob, Ob, Ob], T.bbb.VvvOOO, optimize=True)
    )
    dT.bbb.VvvOOO += (2.0 / 12.0) * (
            -1.0 * np.einsum('AbEf,EcfIJK->AbcIJK', H.bb.vvvv[Vb, vb, Vb, vb], T.bbb.VvvOOO, optimize=True)
            - 0.5 * np.einsum('AbEF,FEcIJK->AbcIJK', H.bb.vvvv[Vb, vb, Vb, Vb], T.bbb.VVvOOO, optimize=True)
    )
    dT.bbb.VvvOOO += (1.0 / 12.0) * (
            +0.5 * np.einsum('cbef,AfeIJK->AbcIJK', H.bb.vvvv[vb, vb, vb, vb], T.bbb.VvvOOO, optimize=True)
            - 1.0 * np.einsum('cbEf,AEfIJK->AbcIJK', H.bb.vvvv[vb, vb, Vb, vb], T.bbb.VVvOOO, optimize=True)
            + 0.5 * np.einsum('cbEF,AFEIJK->AbcIJK', H.bb.vvvv[vb, vb, Vb, Vb], T.bbb.VVVOOO, optimize=True)
    )
    dT.bbb.VvvOOO += (3.0 / 12.0) * (
            -1.0 * np.einsum('AmIE,EcbmJK->AbcIJK', H.bb.voov[Vb, ob, Ob, Vb], T.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('AMIE,EcbMJK->AbcIJK', H.bb.voov[Vb, Ob, Ob, Vb], T.bbb.VvvOOO, optimize=True)
    )
    dT.bbb.VvvOOO += (6.0 / 12.0) * (
            -1.0 * np.einsum('bmIe,AcemJK->AbcIJK', H.bb.voov[vb, ob, Ob, vb], T.bbb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('bmIE,AEcmJK->AbcIJK', H.bb.voov[vb, ob, Ob, Vb], T.bbb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('bMIe,AceMJK->AbcIJK', H.bb.voov[vb, Ob, Ob, vb], T.bbb.VvvOOO, optimize=True)
            + 1.0 * np.einsum('bMIE,AEcMJK->AbcIJK', H.bb.voov[vb, Ob, Ob, Vb], T.bbb.VVvOOO, optimize=True)
    )
    dT.bbb.VvvOOO += (3.0 / 12.0) * (
            -1.0 * np.einsum('mAEI,EcbmJK->AbcIJK', H.ab.ovvo[oa, Vb, Va, Ob], T.abb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('MAEI,EcbMJK->AbcIJK', H.ab.ovvo[Oa, Vb, Va, Ob], T.abb.VvvOOO, optimize=True)
    )
    dT.bbb.VvvOOO += (6.0 / 12.0) * (
            -1.0 * np.einsum('mbeI,eAcmJK->AbcIJK', H.ab.ovvo[oa, vb, va, Ob], T.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('mbEI,EAcmJK->AbcIJK', H.ab.ovvo[oa, vb, Va, Ob], T.abb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('MbeI,eAcMJK->AbcIJK', H.ab.ovvo[Oa, vb, va, Ob], T.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('MbEI,EAcMJK->AbcIJK', H.ab.ovvo[Oa, vb, Va, Ob], T.abb.VVvOOO, optimize=True)
    )

    dT.bbb.VvvOOO -= np.transpose(dT.bbb.VvvOOO, (0, 1, 2, 4, 3, 5)) + np.transpose(dT.bbb.VvvOOO, (0, 1, 2, 3, 5, 4)) \
           + np.transpose(dT.bbb.VvvOOO, (0, 1, 2, 5, 4, 3)) - np.transpose(dT.bbb.VvvOOO, (0, 1, 2, 4, 5, 3)) \
           - np.transpose(dT.bbb.VvvOOO, (0, 1, 2, 5, 3, 4))

    dT.bbb.VvvOOO -= np.transpose(dT.bbb.VvvOOO, (0, 2, 1, 3, 4, 5))

    return dT

def update(T, dT, H, shift, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.bbb.VvvOOO, dT.bbb.VvvOOO = cc_active_loops.update_t3d_100111(
        T.bbb.VvvOOO,
        dT.bbb.VvvOOO,
        H.b.oo[Ob, Ob],
        H.b.vv[Vb, Vb],
        H.b.oo[ob, ob],
        H.b.vv[vb, vb],
        shift,
    )

    return T, dT