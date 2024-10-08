import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops

def build(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.bbb.VVVOOO = (9.0 / 36.0) * (
            -1.0 * np.einsum('BmJI,ACmK->ABCIJK', X.bb.vooo[Vb, :, Ob, Ob], T.bb[Vb, Vb, :, Ob], optimize=True)
    )
    dR.bbb.VVVOOO += (9.0 / 36.0) * (
            -1.0 * np.einsum('BmJI,ACmK->ABCIJK', H.bb.vooo[Vb, :, Ob, Ob], R.bb[Vb, Vb, :, Ob], optimize=True)
    )
    dR.bbb.VVVOOO += (9.0 / 36.0) * (
            +1.0 * np.einsum('BAJe,eCIK->ABCIJK', X.bb.vvov[Vb, Vb, Ob, :], T.bb[:, Vb, Ob, Ob], optimize=True)
    )
    dR.bbb.VVVOOO += (9.0 / 36.0) * (
            +1.0 * np.einsum('BAJe,eCIK->ABCIJK', H.bb.vvov[Vb, Vb, Ob, :], R.bb[:, Vb, Ob, Ob], optimize=True)
    )
    # of terms =  4
    dR.bbb.VVVOOO += (3.0 / 36.0) * (
            +1.0 * np.einsum('Be,CAeIJK->ABCIJK', X.b.vv[Vb, vb], T.bbb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('BE,CEAIJK->ABCIJK', X.b.vv[Vb, Vb], T.bbb.VVVOOO, optimize=True)
    )
    dR.bbb.VVVOOO += (3.0 / 36.0) * (
            -1.0 * np.einsum('mJ,CBAmIK->ABCIJK', X.b.oo[ob, Ob], T.bbb.VVVoOO, optimize=True)
            + 1.0 * np.einsum('MJ,CBAIMK->ABCIJK', X.b.oo[Ob, Ob], T.bbb.VVVOOO, optimize=True)
    )
    dR.bbb.VVVOOO += (3.0 / 36.0) * (
            -0.5 * np.einsum('mnIJ,CBAmnK->ABCIJK', X.bb.oooo[ob, ob, Ob, Ob], T.bbb.VVVooO, optimize=True)
            - 1.0 * np.einsum('mNIJ,CBAmNK->ABCIJK', X.bb.oooo[ob, Ob, Ob, Ob], T.bbb.VVVoOO, optimize=True)
            - 0.5 * np.einsum('MNIJ,CBAMNK->ABCIJK', X.bb.oooo[Ob, Ob, Ob, Ob], T.bbb.VVVOOO, optimize=True)
    )
    dR.bbb.VVVOOO += (3.0 / 36.0) * (
            -0.5 * np.einsum('ABef,CfeIJK->ABCIJK', X.bb.vvvv[Vb, Vb, vb, vb], T.bbb.VvvOOO, optimize=True)
            + 1.0 * np.einsum('ABEf,CEfIJK->ABCIJK', X.bb.vvvv[Vb, Vb, Vb, vb], T.bbb.VVvOOO, optimize=True)
            - 0.5 * np.einsum('ABEF,CFEIJK->ABCIJK', X.bb.vvvv[Vb, Vb, Vb, Vb], T.bbb.VVVOOO, optimize=True)
    )
    dR.bbb.VVVOOO += (9.0 / 36.0) * (
            -1.0 * np.einsum('BmJe,CAemIK->ABCIJK', X.bb.voov[Vb, ob, Ob, vb], T.bbb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('BMJe,CAeIMK->ABCIJK', X.bb.voov[Vb, Ob, Ob, vb], T.bbb.VVvOOO, optimize=True)
            + 1.0 * np.einsum('BmJE,CEAmIK->ABCIJK', X.bb.voov[Vb, ob, Ob, Vb], T.bbb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('BMJE,CEAIMK->ABCIJK', X.bb.voov[Vb, Ob, Ob, Vb], T.bbb.VVVOOO, optimize=True)
    )
    dR.bbb.VVVOOO += (9.0 / 36.0) * (
            -1.0 * np.einsum('mBeJ,eACmKI->ABCIJK', X.ab.ovvo[oa, Vb, va, Ob], T.abb.vVVoOO, optimize=True)
            - 1.0 * np.einsum('MBeJ,eACMKI->ABCIJK', X.ab.ovvo[Oa, Vb, va, Ob], T.abb.vVVOOO, optimize=True)
            - 1.0 * np.einsum('mBEJ,EACmKI->ABCIJK', X.ab.ovvo[oa, Vb, Va, Ob], T.abb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('MBEJ,EACMKI->ABCIJK', X.ab.ovvo[Oa, Vb, Va, Ob], T.abb.VVVOOO, optimize=True)
    )
    dR.bbb.VVVOOO += (3.0 / 36.0) * (
            -1.0 * np.einsum('mJ,CBAmIK->ABCIJK', H.b.oo[ob, Ob], R.bbb.VVVoOO, optimize=True)
            + 1.0 * np.einsum('MJ,CBAIMK->ABCIJK', H.b.oo[Ob, Ob], R.bbb.VVVOOO, optimize=True)
    )
    dR.bbb.VVVOOO += (3.0 / 36.0) * (
            +1.0 * np.einsum('Be,CAeIJK->ABCIJK', H.b.vv[Vb, vb], R.bbb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('BE,CEAIJK->ABCIJK', H.b.vv[Vb, Vb], R.bbb.VVVOOO, optimize=True)
    )
    dR.bbb.VVVOOO += (3.0 / 36.0) * (
            -0.5 * np.einsum('mnIJ,CBAmnK->ABCIJK', H.bb.oooo[ob, ob, Ob, Ob], R.bbb.VVVooO, optimize=True)
            - 1.0 * np.einsum('mNIJ,CBAmNK->ABCIJK', H.bb.oooo[ob, Ob, Ob, Ob], R.bbb.VVVoOO, optimize=True)
            - 0.5 * np.einsum('MNIJ,CBAMNK->ABCIJK', H.bb.oooo[Ob, Ob, Ob, Ob], R.bbb.VVVOOO, optimize=True)
    )
    dR.bbb.VVVOOO += (3.0 / 36.0) * (
            -0.5 * np.einsum('ABef,CfeIJK->ABCIJK', H.bb.vvvv[Vb, Vb, vb, vb], R.bbb.VvvOOO, optimize=True)
            + 1.0 * np.einsum('ABEf,CEfIJK->ABCIJK', H.bb.vvvv[Vb, Vb, Vb, vb], R.bbb.VVvOOO, optimize=True)
            - 0.5 * np.einsum('ABEF,CFEIJK->ABCIJK', H.bb.vvvv[Vb, Vb, Vb, Vb], R.bbb.VVVOOO, optimize=True)
    )
    dR.bbb.VVVOOO += (9.0 / 36.0) * (
            -1.0 * np.einsum('AmIe,CBemJK->ABCIJK', H.bb.voov[Vb, ob, Ob, vb], R.bbb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('AMIe,CBeMJK->ABCIJK', H.bb.voov[Vb, Ob, Ob, vb], R.bbb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('AmIE,CBEmJK->ABCIJK', H.bb.voov[Vb, ob, Ob, Vb], R.bbb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('AMIE,CBEMJK->ABCIJK', H.bb.voov[Vb, Ob, Ob, Vb], R.bbb.VVVOOO, optimize=True)
    )
    dR.bbb.VVVOOO += (9.0 / 36.0) * (
            -1.0 * np.einsum('mAeI,eBCmKJ->ABCIJK', H.ab.ovvo[oa, Vb, va, Ob], R.abb.vVVoOO, optimize=True)
            - 1.0 * np.einsum('MAeI,eBCMKJ->ABCIJK', H.ab.ovvo[Oa, Vb, va, Ob], R.abb.vVVOOO, optimize=True)
            - 1.0 * np.einsum('mAEI,EBCmKJ->ABCIJK', H.ab.ovvo[oa, Vb, Va, Ob], R.abb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('MAEI,EBCMKJ->ABCIJK', H.ab.ovvo[Oa, Vb, Va, Ob], R.abb.VVVOOO, optimize=True)
    )
    # of terms =  12

    dR.bbb.VVVOOO -= np.transpose(dR.bbb.VVVOOO, (0, 1, 2, 3, 5, 4))
    dR.bbb.VVVOOO -= np.transpose(dR.bbb.VVVOOO, (0, 1, 2, 4, 3, 5)) + np.transpose(dR.bbb.VVVOOO, (0, 1, 2, 5, 4, 3))
    dR.bbb.VVVOOO -= np.transpose(dR.bbb.VVVOOO, (0, 2, 1, 3, 4, 5))
    dR.bbb.VVVOOO -= np.transpose(dR.bbb.VVVOOO, (1, 0, 2, 3, 4, 5)) + np.transpose(dR.bbb.VVVOOO, (2, 1, 0, 3, 4, 5))

    return dR

def update(R, omega, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    R.bbb.VVVOOO = eomcc_active_loops.update_r3d_111111(
        R.bbb.VVVOOO,
        omega,
        H.b.oo[Ob, Ob],
        H.b.vv[Vb, Vb],
        H.b.oo[ob, ob],
        H.b.vv[vb, vb],
        0.0,
    )
    return R

