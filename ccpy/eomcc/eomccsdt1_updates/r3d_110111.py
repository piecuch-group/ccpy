import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops

def build(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.bbb.VVvOOO = (6.0 / 12.0) * (
            -1.0 * np.einsum('BmJI,AcmK->ABcIJK', X.bb.vooo[Vb, :, Ob, Ob], T.bb[Vb, vb, :, Ob], optimize=True)
    )
    dR.bbb.VVvOOO += (3.0 / 12.0) * (
            +1.0 * np.einsum('cmJI,ABmK->ABcIJK', X.bb.vooo[vb, :, Ob, Ob], T.bb[Vb, Vb, :, Ob], optimize=True)
    )
    dR.bbb.VVvOOO += (6.0 / 12.0) * (
            -1.0 * np.einsum('BmJI,AcmK->ABcIJK', H.bb.vooo[Vb, :, Ob, Ob], R.bb[Vb, vb, :, Ob], optimize=True)
    )
    dR.bbb.VVvOOO += (3.0 / 12.0) * (
            +1.0 * np.einsum('cmJI,ABmK->ABcIJK', H.bb.vooo[vb, :, Ob, Ob], R.bb[Vb, Vb, :, Ob], optimize=True)
    )
    dR.bbb.VVvOOO += (3.0 / 12.0) * (
            +1.0 * np.einsum('BAJe,ecIK->ABcIJK', X.bb.vvov[Vb, Vb, Ob, :], T.bb[:, vb, Ob, Ob], optimize=True)
    )
    dR.bbb.VVvOOO += (6.0 / 12.0) * (
            -1.0 * np.einsum('BcJe,eAIK->ABcIJK', X.bb.vvov[Vb, vb, Ob, :], T.bb[:, Vb, Ob, Ob], optimize=True)
    )
    dR.bbb.VVvOOO += (3.0 / 12.0) * (
            +1.0 * np.einsum('BAJe,ecIK->ABcIJK', H.bb.vvov[Vb, Vb, Ob, :], R.bb[:, vb, Ob, Ob], optimize=True)
    )
    dR.bbb.VVvOOO += (6.0 / 12.0) * (
            -1.0 * np.einsum('BcJe,eAIK->ABcIJK', H.bb.vvov[Vb, vb, Ob, :], R.bb[:, Vb, Ob, Ob], optimize=True)
    )
    # of terms =  8
    dR.bbb.VVvOOO += (2.0 / 12.0) * (
            -1.0 * np.einsum('Be,AceIJK->ABcIJK', X.b.vv[Vb, vb], T.bbb.VvvOOO, optimize=True)
            - 1.0 * np.einsum('BE,EAcIJK->ABcIJK', X.b.vv[Vb, Vb], T.bbb.VVvOOO, optimize=True)
    )
    dR.bbb.VVvOOO += (1.0 / 12.0) * (
            -1.0 * np.einsum('ce,BAeIJK->ABcIJK', X.b.vv[vb, vb], T.bbb.VVvOOO, optimize=True)
            + 1.0 * np.einsum('cE,BEAIJK->ABcIJK', X.b.vv[vb, Vb], T.bbb.VVVOOO, optimize=True)
    )
    dR.bbb.VVvOOO += (3.0 / 12.0) * (
            -1.0 * np.einsum('mJ,BAcmIK->ABcIJK', X.b.oo[ob, Ob], T.bbb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('MJ,BAcIMK->ABcIJK', X.b.oo[Ob, Ob], T.bbb.VVvOOO, optimize=True)
    )
    dR.bbb.VVvOOO += (3.0 / 12.0) * (
            -0.5 * np.einsum('mnIJ,BAcmnK->ABcIJK', X.bb.oooo[ob, ob, Ob, Ob], T.bbb.VVvooO, optimize=True)
            + 1.0 * np.einsum('MnIJ,BAcnMK->ABcIJK', X.bb.oooo[Ob, ob, Ob, Ob], T.bbb.VVvoOO, optimize=True)
            - 0.5 * np.einsum('MNIJ,BAcMNK->ABcIJK', X.bb.oooo[Ob, Ob, Ob, Ob], T.bbb.VVvOOO, optimize=True)
    )
    dR.bbb.VVvOOO += (1.0 / 12.0) * (
            -1.0 * np.einsum('ABEf,EcfIJK->ABcIJK', X.bb.vvvv[Vb, Vb, Vb, vb], T.bbb.VvvOOO, optimize=True)
            - 0.5 * np.einsum('ABEF,FEcIJK->ABcIJK', X.bb.vvvv[Vb, Vb, Vb, Vb], T.bbb.VVvOOO, optimize=True)
    )
    dR.bbb.VVvOOO += (2.0 / 12.0) * (
            +0.5 * np.einsum('cBef,AfeIJK->ABcIJK', X.bb.vvvv[vb, Vb, vb, vb], T.bbb.VvvOOO, optimize=True)
            - 1.0 * np.einsum('cBEf,AEfIJK->ABcIJK', X.bb.vvvv[vb, Vb, Vb, vb], T.bbb.VVvOOO, optimize=True)
            + 0.5 * np.einsum('cBEF,AFEIJK->ABcIJK', X.bb.vvvv[vb, Vb, Vb, Vb], T.bbb.VVVOOO, optimize=True)
    )
    dR.bbb.VVvOOO += (6.0 / 12.0) * (
            +1.0 * np.einsum('BmJe,AcemIK->ABcIJK', X.bb.voov[Vb, ob, Ob, vb], T.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('BMJe,AceIMK->ABcIJK', X.bb.voov[Vb, Ob, Ob, vb], T.bbb.VvvOOO, optimize=True)
            + 1.0 * np.einsum('BmJE,EAcmIK->ABcIJK', X.bb.voov[Vb, ob, Ob, Vb], T.bbb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('BMJE,EAcIMK->ABcIJK', X.bb.voov[Vb, Ob, Ob, Vb], T.bbb.VVvOOO, optimize=True)
    )
    dR.bbb.VVvOOO += (3.0 / 12.0) * (
            +1.0 * np.einsum('cmJe,BAemIK->ABcIJK', X.bb.voov[vb, ob, Ob, vb], T.bbb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('cMJe,BAeIMK->ABcIJK', X.bb.voov[vb, Ob, Ob, vb], T.bbb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('cmJE,BEAmIK->ABcIJK', X.bb.voov[vb, ob, Ob, Vb], T.bbb.VVVoOO, optimize=True)
            + 1.0 * np.einsum('cMJE,BEAIMK->ABcIJK', X.bb.voov[vb, Ob, Ob, Vb], T.bbb.VVVOOO, optimize=True)
    )
    dR.bbb.VVvOOO += (6.0 / 12.0) * (
            -1.0 * np.einsum('mBeJ,eAcmKI->ABcIJK', X.ab.ovvo[oa, Vb, va, Ob], T.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('MBeJ,eAcMKI->ABcIJK', X.ab.ovvo[Oa, Vb, va, Ob], T.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('mBEJ,EAcmKI->ABcIJK', X.ab.ovvo[oa, Vb, Va, Ob], T.abb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('MBEJ,EAcMKI->ABcIJK', X.ab.ovvo[Oa, Vb, Va, Ob], T.abb.VVvOOO, optimize=True)
    )
    dR.bbb.VVvOOO += (3.0 / 12.0) * (
            +1.0 * np.einsum('mceJ,eABmKI->ABcIJK', X.ab.ovvo[oa, vb, va, Ob], T.abb.vVVoOO, optimize=True)
            + 1.0 * np.einsum('MceJ,eABMKI->ABcIJK', X.ab.ovvo[Oa, vb, va, Ob], T.abb.vVVOOO, optimize=True)
            + 1.0 * np.einsum('mcEJ,EABmKI->ABcIJK', X.ab.ovvo[oa, vb, Va, Ob], T.abb.VVVoOO, optimize=True)
            + 1.0 * np.einsum('McEJ,EABMKI->ABcIJK', X.ab.ovvo[Oa, vb, Va, Ob], T.abb.VVVOOO, optimize=True)
    )
    dR.bbb.VVvOOO += (3.0 / 12.0) * (
            -1.0 * np.einsum('mJ,BAcmIK->ABcIJK', H.b.oo[ob, Ob], R.bbb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('MJ,BAcIMK->ABcIJK', H.b.oo[Ob, Ob], R.bbb.VVvOOO, optimize=True)
    )
    dR.bbb.VVvOOO += (2.0 / 12.0) * (
            -1.0 * np.einsum('Be,AceIJK->ABcIJK', H.b.vv[Vb, vb], R.bbb.VvvOOO, optimize=True)
            - 1.0 * np.einsum('BE,EAcIJK->ABcIJK', H.b.vv[Vb, Vb], R.bbb.VVvOOO, optimize=True)
    )
    dR.bbb.VVvOOO += (1.0 / 12.0) * (
            -1.0 * np.einsum('ce,BAeIJK->ABcIJK', H.b.vv[vb, vb], R.bbb.VVvOOO, optimize=True)
            + 1.0 * np.einsum('cE,BEAIJK->ABcIJK', H.b.vv[vb, Vb], R.bbb.VVVOOO, optimize=True)
    )
    dR.bbb.VVvOOO += (3.0 / 12.0) * (
            -0.5 * np.einsum('mnIJ,BAcmnK->ABcIJK', H.bb.oooo[ob, ob, Ob, Ob], R.bbb.VVvooO, optimize=True)
            + 1.0 * np.einsum('MnIJ,BAcnMK->ABcIJK', H.bb.oooo[Ob, ob, Ob, Ob], R.bbb.VVvoOO, optimize=True)
            - 0.5 * np.einsum('MNIJ,BAcMNK->ABcIJK', H.bb.oooo[Ob, Ob, Ob, Ob], R.bbb.VVvOOO, optimize=True)
    )
    dR.bbb.VVvOOO += (1.0 / 12.0) * (
            -1.0 * np.einsum('ABEf,EcfIJK->ABcIJK', H.bb.vvvv[Vb, Vb, Vb, vb], R.bbb.VvvOOO, optimize=True)
            - 0.5 * np.einsum('ABEF,FEcIJK->ABcIJK', H.bb.vvvv[Vb, Vb, Vb, Vb], R.bbb.VVvOOO, optimize=True)
    )
    dR.bbb.VVvOOO += (2.0 / 12.0) * (
            +0.5 * np.einsum('cBef,AfeIJK->ABcIJK', H.bb.vvvv[vb, Vb, vb, vb], R.bbb.VvvOOO, optimize=True)
            - 1.0 * np.einsum('cBEf,AEfIJK->ABcIJK', H.bb.vvvv[vb, Vb, Vb, vb], R.bbb.VVvOOO, optimize=True)
            + 0.5 * np.einsum('cBEF,AFEIJK->ABcIJK', H.bb.vvvv[vb, Vb, Vb, Vb], R.bbb.VVVOOO, optimize=True)
    )
    dR.bbb.VVvOOO += (6.0 / 12.0) * (
            +1.0 * np.einsum('AmIe,BcemJK->ABcIJK', H.bb.voov[Vb, ob, Ob, vb], R.bbb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('AMIe,BceMJK->ABcIJK', H.bb.voov[Vb, Ob, Ob, vb], R.bbb.VvvOOO, optimize=True)
            - 1.0 * np.einsum('AmIE,BEcmJK->ABcIJK', H.bb.voov[Vb, ob, Ob, Vb], R.bbb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('AMIE,BEcMJK->ABcIJK', H.bb.voov[Vb, Ob, Ob, Vb], R.bbb.VVvOOO, optimize=True)
    )
    dR.bbb.VVvOOO += (3.0 / 12.0) * (
            +1.0 * np.einsum('cmIe,ABemJK->ABcIJK', H.bb.voov[vb, ob, Ob, vb], R.bbb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('cMIe,ABeMJK->ABcIJK', H.bb.voov[vb, Ob, Ob, vb], R.bbb.VVvOOO, optimize=True)
            + 1.0 * np.einsum('cmIE,ABEmJK->ABcIJK', H.bb.voov[vb, ob, Ob, Vb], R.bbb.VVVoOO, optimize=True)
            + 1.0 * np.einsum('cMIE,ABEMJK->ABcIJK', H.bb.voov[vb, Ob, Ob, Vb], R.bbb.VVVOOO, optimize=True)
    )
    dR.bbb.VVvOOO += (6.0 / 12.0) * (
            -1.0 * np.einsum('mAeI,eBcmKJ->ABcIJK', H.ab.ovvo[oa, Vb, va, Ob], R.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('MAeI,eBcMKJ->ABcIJK', H.ab.ovvo[Oa, Vb, va, Ob], R.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('mAEI,EBcmKJ->ABcIJK', H.ab.ovvo[oa, Vb, Va, Ob], R.abb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('MAEI,EBcMKJ->ABcIJK', H.ab.ovvo[Oa, Vb, Va, Ob], R.abb.VVvOOO, optimize=True)
    )
    dR.bbb.VVvOOO += (3.0 / 12.0) * (
            +1.0 * np.einsum('mceI,eBAmKJ->ABcIJK', H.ab.ovvo[oa, vb, va, Ob], R.abb.vVVoOO, optimize=True)
            + 1.0 * np.einsum('MceI,eBAMKJ->ABcIJK', H.ab.ovvo[Oa, vb, va, Ob], R.abb.vVVOOO, optimize=True)
            + 1.0 * np.einsum('mcEI,EBAmKJ->ABcIJK', H.ab.ovvo[oa, vb, Va, Ob], R.abb.VVVoOO, optimize=True)
            + 1.0 * np.einsum('McEI,EBAMKJ->ABcIJK', H.ab.ovvo[Oa, vb, Va, Ob], R.abb.VVVOOO, optimize=True)
    )
    # of terms =  20

    dR.bbb.VVvOOO -= np.transpose(dR.bbb.VVvOOO, (0, 1, 2, 3, 5, 4))
    dR.bbb.VVvOOO -= np.transpose(dR.bbb.VVvOOO, (0, 1, 2, 4, 3, 5)) + np.transpose(dR.bbb.VVvOOO, (0, 1, 2, 5, 4, 3))
    dR.bbb.VVvOOO -= np.transpose(dR.bbb.VVvOOO, (1, 0, 2, 3, 4, 5))

    return dR

def update(R, omega, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    R.bbb.VVvOOO = eomcc_active_loops.update_r3d_110111(
        R.bbb.VVvOOO,
        omega,
        H.b.oo[Ob, Ob],
        H.b.vv[Vb, Vb],
        H.b.oo[ob, ob],
        H.b.vv[vb, vb],
        0.0,
    )
    return R
