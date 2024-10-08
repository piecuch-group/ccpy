import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops

def build(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.bbb.VvvOOO = (6.0 / 12.0) * (
            -1.0 * np.einsum('bmJI,AcmK->AbcIJK', X.bb.vooo[vb, :, Ob, Ob], T.bb[Vb, vb, :, Ob], optimize=True)
    )
    dR.bbb.VvvOOO += (3.0 / 12.0) * (
            +1.0 * np.einsum('AmJI,bcmK->AbcIJK', X.bb.vooo[Vb, :, Ob, Ob], T.bb[vb, vb, :, Ob], optimize=True)
    )
    dR.bbb.VvvOOO += (6.0 / 12.0) * (
            -1.0 * np.einsum('bmJI,AcmK->AbcIJK', H.bb.vooo[vb, :, Ob, Ob], R.bb[Vb, vb, :, Ob], optimize=True)
    )
    dR.bbb.VvvOOO += (3.0 / 12.0) * (
            +1.0 * np.einsum('AmJI,bcmK->AbcIJK', H.bb.vooo[Vb, :, Ob, Ob], R.bb[vb, vb, :, Ob], optimize=True)
    )
    dR.bbb.VvvOOO += (6.0 / 12.0) * (
            +1.0 * np.einsum('bAJe,ecIK->AbcIJK', X.bb.vvov[vb, Vb, Ob, :], T.bb[:, vb, Ob, Ob], optimize=True)
    )
    dR.bbb.VvvOOO += (3.0 / 12.0) * (
            -1.0 * np.einsum('bcJe,eAIK->AbcIJK', X.bb.vvov[vb, vb, Ob, :], T.bb[:, Vb, Ob, Ob], optimize=True)
    )
    dR.bbb.VvvOOO += (6.0 / 12.0) * (
            +1.0 * np.einsum('bAJe,ecIK->AbcIJK', H.bb.vvov[vb, Vb, Ob, :], R.bb[:, vb, Ob, Ob], optimize=True)
    )
    dR.bbb.VvvOOO += (3.0 / 12.0) * (
            -1.0 * np.einsum('bcJe,eAIK->AbcIJK', H.bb.vvov[vb, vb, Ob, :], R.bb[:, Vb, Ob, Ob], optimize=True)
    )
    # of terms =  8
    dR.bbb.VvvOOO += (2.0 / 12.0) * (
            -1.0 * np.einsum('be,AceIJK->AbcIJK', X.b.vv[vb, vb], T.bbb.VvvOOO, optimize=True)
            - 1.0 * np.einsum('bE,EAcIJK->AbcIJK', X.b.vv[vb, Vb], T.bbb.VVvOOO, optimize=True)
    )
    dR.bbb.VvvOOO += (1.0 / 12.0) * (
            -1.0 * np.einsum('AE,EcbIJK->AbcIJK', X.b.vv[Vb, Vb], T.bbb.VvvOOO, optimize=True)
    )
    dR.bbb.VvvOOO += (3.0 / 12.0) * (
            -1.0 * np.einsum('mJ,AcbmIK->AbcIJK', X.b.oo[ob, Ob], T.bbb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('MJ,AcbIMK->AbcIJK', X.b.oo[Ob, Ob], T.bbb.VvvOOO, optimize=True)
    )
    dR.bbb.VvvOOO += (3.0 / 12.0) * (
            -0.5 * np.einsum('mnIJ,AcbmnK->AbcIJK', X.bb.oooo[ob, ob, Ob, Ob], T.bbb.VvvooO, optimize=True)
            + 1.0 * np.einsum('MnIJ,AcbnMK->AbcIJK', X.bb.oooo[Ob, ob, Ob, Ob], T.bbb.VvvoOO, optimize=True)
            - 0.5 * np.einsum('MNIJ,AcbMNK->AbcIJK', X.bb.oooo[Ob, Ob, Ob, Ob], T.bbb.VvvOOO, optimize=True)
    )
    dR.bbb.VvvOOO += (2.0 / 12.0) * (
            -1.0 * np.einsum('AbEf,EcfIJK->AbcIJK', X.bb.vvvv[Vb, vb, Vb, vb], T.bbb.VvvOOO, optimize=True)
            - 0.5 * np.einsum('AbEF,FEcIJK->AbcIJK', X.bb.vvvv[Vb, vb, Vb, Vb], T.bbb.VVvOOO, optimize=True)
    )
    dR.bbb.VvvOOO += (1.0 / 12.0) * (
            +0.5 * np.einsum('cbef,AfeIJK->AbcIJK', X.bb.vvvv[vb, vb, vb, vb], T.bbb.VvvOOO, optimize=True)
            - 1.0 * np.einsum('cbEf,AEfIJK->AbcIJK', X.bb.vvvv[vb, vb, Vb, vb], T.bbb.VVvOOO, optimize=True)
            + 0.5 * np.einsum('cbEF,AFEIJK->AbcIJK', X.bb.vvvv[vb, vb, Vb, Vb], T.bbb.VVVOOO, optimize=True)
    )
    dR.bbb.VvvOOO += (6.0 / 12.0) * (
            +1.0 * np.einsum('bmJe,AcemIK->AbcIJK', X.bb.voov[vb, ob, Ob, vb], T.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('bMJe,AceIMK->AbcIJK', X.bb.voov[vb, Ob, Ob, vb], T.bbb.VvvOOO, optimize=True)
            + 1.0 * np.einsum('bmJE,EAcmIK->AbcIJK', X.bb.voov[vb, ob, Ob, Vb], T.bbb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('bMJE,EAcIMK->AbcIJK', X.bb.voov[vb, Ob, Ob, Vb], T.bbb.VVvOOO, optimize=True)
    )
    dR.bbb.VvvOOO += (3.0 / 12.0) * (
            +1.0 * np.einsum('AmJE,EcbmIK->AbcIJK', X.bb.voov[Vb, ob, Ob, Vb], T.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('AMJE,EcbIMK->AbcIJK', X.bb.voov[Vb, Ob, Ob, Vb], T.bbb.VvvOOO, optimize=True)
    )
    dR.bbb.VvvOOO += (6.0 / 12.0) * (
            -1.0 * np.einsum('mbeJ,eAcmKI->AbcIJK', X.ab.ovvo[oa, vb, va, Ob], T.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('MbeJ,eAcMKI->AbcIJK', X.ab.ovvo[Oa, vb, va, Ob], T.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('mbEJ,EAcmKI->AbcIJK', X.ab.ovvo[oa, vb, Va, Ob], T.abb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('MbEJ,EAcMKI->AbcIJK', X.ab.ovvo[Oa, vb, Va, Ob], T.abb.VVvOOO, optimize=True)
    )
    dR.bbb.VvvOOO += (3.0 / 12.0) * (
            +1.0 * np.einsum('mAEJ,EbcmKI->AbcIJK', X.ab.ovvo[oa, Vb, Va, Ob], T.abb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('MAEJ,EbcMKI->AbcIJK', X.ab.ovvo[Oa, Vb, Va, Ob], T.abb.VvvOOO, optimize=True)
    )
    dR.bbb.VvvOOO += (3.0 / 12.0) * (
            -1.0 * np.einsum('mJ,AcbmIK->AbcIJK', H.b.oo[ob, Ob], R.bbb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('MJ,AcbIMK->AbcIJK', H.b.oo[Ob, Ob], R.bbb.VvvOOO, optimize=True)
    )
    dR.bbb.VvvOOO += (2.0 / 12.0) * (
            -1.0 * np.einsum('be,AceIJK->AbcIJK', H.b.vv[vb, vb], R.bbb.VvvOOO, optimize=True)
            - 1.0 * np.einsum('bE,EAcIJK->AbcIJK', H.b.vv[vb, Vb], R.bbb.VVvOOO, optimize=True)
    )
    dR.bbb.VvvOOO += (1.0 / 12.0) * (
            -1.0 * np.einsum('AE,EcbIJK->AbcIJK', H.b.vv[Vb, Vb], R.bbb.VvvOOO, optimize=True)
    )
    dR.bbb.VvvOOO += (3.0 / 12.0) * (
            -0.5 * np.einsum('mnIJ,AcbmnK->AbcIJK', H.bb.oooo[ob, ob, Ob, Ob], R.bbb.VvvooO, optimize=True)
            + 1.0 * np.einsum('MnIJ,AcbnMK->AbcIJK', H.bb.oooo[Ob, ob, Ob, Ob], R.bbb.VvvoOO, optimize=True)
            - 0.5 * np.einsum('MNIJ,AcbMNK->AbcIJK', H.bb.oooo[Ob, Ob, Ob, Ob], R.bbb.VvvOOO, optimize=True)
    )
    dR.bbb.VvvOOO += (2.0 / 12.0) * (
            -1.0 * np.einsum('AbEf,EcfIJK->AbcIJK', H.bb.vvvv[Vb, vb, Vb, vb], R.bbb.VvvOOO, optimize=True)
            - 0.5 * np.einsum('AbEF,FEcIJK->AbcIJK', H.bb.vvvv[Vb, vb, Vb, Vb], R.bbb.VVvOOO, optimize=True)
    )
    dR.bbb.VvvOOO += (1.0 / 12.0) * (
            +0.5 * np.einsum('cbef,AfeIJK->AbcIJK', H.bb.vvvv[vb, vb, vb, vb], R.bbb.VvvOOO, optimize=True)
            - 1.0 * np.einsum('cbEf,AEfIJK->AbcIJK', H.bb.vvvv[vb, vb, Vb, vb], R.bbb.VVvOOO, optimize=True)
            + 0.5 * np.einsum('cbEF,AFEIJK->AbcIJK', H.bb.vvvv[vb, vb, Vb, Vb], R.bbb.VVVOOO, optimize=True)
    )
    dR.bbb.VvvOOO += (3.0 / 12.0) * (
            -1.0 * np.einsum('AmIE,EcbmJK->AbcIJK', H.bb.voov[Vb, ob, Ob, Vb], R.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('AMIE,EcbMJK->AbcIJK', H.bb.voov[Vb, Ob, Ob, Vb], R.bbb.VvvOOO, optimize=True)
    )
    dR.bbb.VvvOOO += (6.0 / 12.0) * (
            -1.0 * np.einsum('bmIe,AcemJK->AbcIJK', H.bb.voov[vb, ob, Ob, vb], R.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('bMIe,AceMJK->AbcIJK', H.bb.voov[vb, Ob, Ob, vb], R.bbb.VvvOOO, optimize=True)
            + 1.0 * np.einsum('bmIE,AEcmJK->AbcIJK', H.bb.voov[vb, ob, Ob, Vb], R.bbb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('bMIE,AEcMJK->AbcIJK', H.bb.voov[vb, Ob, Ob, Vb], R.bbb.VVvOOO, optimize=True)
    )
    dR.bbb.VvvOOO += (3.0 / 12.0) * (
            -1.0 * np.einsum('mAEI,EbcmKJ->AbcIJK', H.ab.ovvo[oa, Vb, Va, Ob], R.abb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('MAEI,EbcMKJ->AbcIJK', H.ab.ovvo[Oa, Vb, Va, Ob], R.abb.VvvOOO, optimize=True)
    )
    dR.bbb.VvvOOO += (6.0 / 12.0) * (
            +1.0 * np.einsum('mbeI,eAcmKJ->AbcIJK', H.ab.ovvo[oa, vb, va, Ob], R.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('MbeI,eAcMKJ->AbcIJK', H.ab.ovvo[Oa, vb, va, Ob], R.abb.vVvOOO, optimize=True)
            + 1.0 * np.einsum('mbEI,EAcmKJ->AbcIJK', H.ab.ovvo[oa, vb, Va, Ob], R.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('MbEI,EAcMKJ->AbcIJK', H.ab.ovvo[Oa, vb, Va, Ob], R.abb.VVvOOO, optimize=True)
    )
    # of terms =  20

    dR.bbb.VvvOOO -= np.transpose(dR.bbb.VvvOOO, (0, 1, 2, 3, 5, 4))
    dR.bbb.VvvOOO -= np.transpose(dR.bbb.VvvOOO, (0, 1, 2, 4, 3, 5)) + np.transpose(dR.bbb.VvvOOO, (0, 1, 2, 5, 4, 3))
    dR.bbb.VvvOOO -= np.transpose(dR.bbb.VvvOOO, (0, 2, 1, 3, 4, 5))

    return dR

def update(R, omega, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    R.bbb.VvvOOO = eomcc_active_loops.update_r3d_100111(
        R.bbb.VvvOOO,
        omega,
        H.b.oo[Ob, Ob],
        H.b.vv[Vb, Vb],
        H.b.oo[ob, ob],
        H.b.vv[vb, vb],
        0.0,
    )
    return R
