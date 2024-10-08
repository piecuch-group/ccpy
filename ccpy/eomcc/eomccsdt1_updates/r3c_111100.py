import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops

def build(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.abb.VVVOoo = (2.0 / 4.0) * (
            +1.0 * np.einsum('ABIe,eCjk->ABCIjk', X.ab.vvov[Va, Vb, Oa, :], T.bb[:, Vb, ob, ob], optimize=True)
    )
    dR.abb.VVVOoo += (2.0 / 4.0) * (
            -1.0 * np.einsum('AmIj,BCmk->ABCIjk', X.ab.vooo[Va, :, Oa, ob], T.bb[Vb, Vb, :, ob], optimize=True)
    )
    dR.abb.VVVOoo += (2.0 / 4.0) * (
            +1.0 * np.einsum('CBke,AeIj->ABCIjk', X.bb.vvov[Vb, Vb, ob, :], T.ab[Va, :, Oa, ob], optimize=True)
    )
    dR.abb.VVVOoo += (2.0 / 4.0) * (
            -1.0 * np.einsum('Cmkj,ABIm->ABCIjk', X.bb.vooo[Vb, :, ob, ob], T.ab[Va, Vb, Oa, :], optimize=True)
    )
    dR.abb.VVVOoo += (4.0 / 4.0) * (
            +1.0 * np.einsum('ABej,eCIk->ABCIjk', X.ab.vvvo[Va, Vb, :, ob], T.ab[:, Vb, Oa, ob], optimize=True)
    )
    dR.abb.VVVOoo += (4.0 / 4.0) * (
            -1.0 * np.einsum('mBIj,ACmk->ABCIjk', X.ab.ovoo[:, Vb, Oa, ob], T.ab[Va, Vb, :, ob], optimize=True)
    )
    dR.abb.VVVOoo += (2.0 / 4.0) * (
            +1.0 * np.einsum('ABIe,eCjk->ABCIjk', H.ab.vvov[Va, Vb, Oa, :], R.bb[:, Vb, ob, ob], optimize=True)
    )
    dR.abb.VVVOoo += (2.0 / 4.0) * (
            -1.0 * np.einsum('AmIj,BCmk->ABCIjk', H.ab.vooo[Va, :, Oa, ob], R.bb[Vb, Vb, :, ob], optimize=True)
    )
    dR.abb.VVVOoo += (2.0 / 4.0) * (
            +1.0 * np.einsum('CBke,AeIj->ABCIjk', H.bb.vvov[Vb, Vb, ob, :], R.ab[Va, :, Oa, ob], optimize=True)
    )
    dR.abb.VVVOoo += (2.0 / 4.0) * (
            -1.0 * np.einsum('Cmkj,ABIm->ABCIjk', H.bb.vooo[Vb, :, ob, ob], R.ab[Va, Vb, Oa, :], optimize=True)
    )
    dR.abb.VVVOoo += (4.0 / 4.0) * (
            +1.0 * np.einsum('ABej,eCIk->ABCIjk', H.ab.vvvo[Va, Vb, :, ob], R.ab[:, Vb, Oa, ob], optimize=True)
    )
    dR.abb.VVVOoo += (4.0 / 4.0) * (
            -1.0 * np.einsum('mBIj,ACmk->ABCIjk', H.ab.ovoo[:, Vb, Oa, ob], R.ab[Va, Vb, :, ob], optimize=True)
    )
    # of terms =  12
    dR.abb.VVVOoo += (1.0 / 4.0) * (
            +1.0 * np.einsum('MI,ACBMjk->ABCIjk', X.a.oo[Oa, Oa], T.abb.VVVOoo, optimize=True)
    )
    dR.abb.VVVOoo += (2.0 / 4.0) * (
            +1.0 * np.einsum('mj,ACBImk->ABCIjk', X.b.oo[ob, ob], T.abb.VVVOoo, optimize=True)
            - 1.0 * np.einsum('Mj,ACBIkM->ABCIjk', X.b.oo[Ob, ob], T.abb.VVVOoO, optimize=True)
    )
    dR.abb.VVVOoo += (1.0 / 4.0) * (
            -1.0 * np.einsum('Ae,eCBIjk->ABCIjk', X.a.vv[Va, va], T.abb.vVVOoo, optimize=True)
            - 1.0 * np.einsum('AE,ECBIjk->ABCIjk', X.a.vv[Va, Va], T.abb.VVVOoo, optimize=True)
    )
    dR.abb.VVVOoo += (2.0 / 4.0) * (
            -1.0 * np.einsum('Be,ACeIjk->ABCIjk', X.b.vv[Vb, vb], T.abb.VVvOoo, optimize=True)
            - 1.0 * np.einsum('BE,ACEIjk->ABCIjk', X.b.vv[Vb, Vb], T.abb.VVVOoo, optimize=True)
    )
    dR.abb.VVVOoo += (1.0 / 4.0) * (
            -0.5 * np.einsum('mnjk,ACBImn->ABCIjk', X.bb.oooo[ob, ob, ob, ob], T.abb.VVVOoo, optimize=True)
            + 1.0 * np.einsum('Mnjk,ACBInM->ABCIjk', X.bb.oooo[Ob, ob, ob, ob], T.abb.VVVOoO, optimize=True)
            - 0.5 * np.einsum('MNjk,ACBIMN->ABCIjk', X.bb.oooo[Ob, Ob, ob, ob], T.abb.VVVOOO, optimize=True)
    )
    dR.abb.VVVOoo += (2.0 / 4.0) * (
            +1.0 * np.einsum('mNIj,ACBmkN->ABCIjk', X.ab.oooo[oa, Ob, Oa, ob], T.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('MnIj,ACBMnk->ABCIjk', X.ab.oooo[Oa, ob, Oa, ob], T.abb.VVVOoo, optimize=True)
            + 1.0 * np.einsum('MNIj,ACBMkN->ABCIjk', X.ab.oooo[Oa, Ob, Oa, ob], T.abb.VVVOoO, optimize=True)
    )
    dR.abb.VVVOoo += (1.0 / 4.0) * (
            -0.5 * np.einsum('BCef,AfeIjk->ABCIjk', X.bb.vvvv[Vb, Vb, vb, vb], T.abb.VvvOoo, optimize=True)
            + 1.0 * np.einsum('BCEf,AEfIjk->ABCIjk', X.bb.vvvv[Vb, Vb, Vb, vb], T.abb.VVvOoo, optimize=True)
            - 0.5 * np.einsum('BCEF,AFEIjk->ABCIjk', X.bb.vvvv[Vb, Vb, Vb, Vb], T.abb.VVVOoo, optimize=True)
    )
    dR.abb.VVVOoo += (2.0 / 4.0) * (
            -1.0 * np.einsum('ABef,eCfIjk->ABCIjk', X.ab.vvvv[Va, Vb, va, vb], T.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('ABeF,eCFIjk->ABCIjk', X.ab.vvvv[Va, Vb, va, Vb], T.abb.vVVOoo, optimize=True)
            - 1.0 * np.einsum('ABEf,ECfIjk->ABCIjk', X.ab.vvvv[Va, Vb, Va, vb], T.abb.VVvOoo, optimize=True)
            - 1.0 * np.einsum('ABEF,ECFIjk->ABCIjk', X.ab.vvvv[Va, Vb, Va, Vb], T.abb.VVVOoo, optimize=True)
    )
    dR.abb.VVVOoo += (1.0 / 4.0) * (
            -1.0 * np.einsum('AMIe,eCBMjk->ABCIjk', X.aa.voov[Va, Oa, Oa, va], T.abb.vVVOoo, optimize=True)
            - 1.0 * np.einsum('AMIE,ECBMjk->ABCIjk', X.aa.voov[Va, Oa, Oa, Va], T.abb.VVVOoo, optimize=True)
    )
    dR.abb.VVVOoo += (1.0 / 4.0) * (
            -1.0 * np.einsum('AMIe,CBejkM->ABCIjk', X.ab.voov[Va, Ob, Oa, vb], T.bbb.VVvooO, optimize=True)
            - 1.0 * np.einsum('AMIE,CBEjkM->ABCIjk', X.ab.voov[Va, Ob, Oa, Vb], T.bbb.VVVooO, optimize=True)
    )
    dR.abb.VVVOoo += (4.0 / 4.0) * (
            -1.0 * np.einsum('mBej,AeCmIk->ABCIjk', X.ab.ovvo[oa, Vb, va, ob], T.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('MBej,AeCIMk->ABCIjk', X.ab.ovvo[Oa, Vb, va, ob], T.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('mBEj,EACmIk->ABCIjk', X.ab.ovvo[oa, Vb, Va, ob], T.aab.VVVoOo, optimize=True)
            - 1.0 * np.einsum('MBEj,EACIMk->ABCIjk', X.ab.ovvo[Oa, Vb, Va, ob], T.aab.VVVOOo, optimize=True)
    )
    dR.abb.VVVOoo += (4.0 / 4.0) * (
            -1.0 * np.einsum('Bmje,ACeImk->ABCIjk', X.bb.voov[Vb, ob, ob, vb], T.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('BMje,ACeIkM->ABCIjk', X.bb.voov[Vb, Ob, ob, vb], T.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('BmjE,ACEImk->ABCIjk', X.bb.voov[Vb, ob, ob, Vb], T.abb.VVVOoo, optimize=True)
            + 1.0 * np.einsum('BMjE,ACEIkM->ABCIjk', X.bb.voov[Vb, Ob, ob, Vb], T.abb.VVVOoO, optimize=True)
    )
    dR.abb.VVVOoo += (2.0 / 4.0) * (
            +1.0 * np.einsum('MBIe,ACeMjk->ABCIjk', X.ab.ovov[Oa, Vb, Oa, vb], T.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('MBIE,ACEMjk->ABCIjk', X.ab.ovov[Oa, Vb, Oa, Vb], T.abb.VVVOoo, optimize=True)
    )
    dR.abb.VVVOoo += (2.0 / 4.0) * (
            +1.0 * np.einsum('Amej,eCBImk->ABCIjk', X.ab.vovo[Va, ob, va, ob], T.abb.vVVOoo, optimize=True)
            - 1.0 * np.einsum('AMej,eCBIkM->ABCIjk', X.ab.vovo[Va, Ob, va, ob], T.abb.vVVOoO, optimize=True)
            + 1.0 * np.einsum('AmEj,ECBImk->ABCIjk', X.ab.vovo[Va, ob, Va, ob], T.abb.VVVOoo, optimize=True)
            - 1.0 * np.einsum('AMEj,ECBIkM->ABCIjk', X.ab.vovo[Va, Ob, Va, ob], T.abb.VVVOoO, optimize=True)
    )
    dR.abb.VVVOoo += (1.0 / 4.0) * (
            +1.0 * np.einsum('MI,ACBMjk->ABCIjk', H.a.oo[Oa, Oa], R.abb.VVVOoo, optimize=True)
    )
    dR.abb.VVVOoo += (2.0 / 4.0) * (
            +1.0 * np.einsum('mj,ACBImk->ABCIjk', H.b.oo[ob, ob], R.abb.VVVOoo, optimize=True)
            - 1.0 * np.einsum('Mj,ACBIkM->ABCIjk', H.b.oo[Ob, ob], R.abb.VVVOoO, optimize=True)
    )
    dR.abb.VVVOoo += (1.0 / 4.0) * (
            -1.0 * np.einsum('Ae,eCBIjk->ABCIjk', H.a.vv[Va, va], R.abb.vVVOoo, optimize=True)
            - 1.0 * np.einsum('AE,ECBIjk->ABCIjk', H.a.vv[Va, Va], R.abb.VVVOoo, optimize=True)
    )
    dR.abb.VVVOoo += (2.0 / 4.0) * (
            -1.0 * np.einsum('Be,ACeIjk->ABCIjk', H.b.vv[Vb, vb], R.abb.VVvOoo, optimize=True)
            - 1.0 * np.einsum('BE,ACEIjk->ABCIjk', H.b.vv[Vb, Vb], R.abb.VVVOoo, optimize=True)
    )
    dR.abb.VVVOoo += (1.0 / 4.0) * (
            -0.5 * np.einsum('mnjk,ACBImn->ABCIjk', H.bb.oooo[ob, ob, ob, ob], R.abb.VVVOoo, optimize=True)
            + 1.0 * np.einsum('Mnjk,ACBInM->ABCIjk', H.bb.oooo[Ob, ob, ob, ob], R.abb.VVVOoO, optimize=True)
            - 0.5 * np.einsum('MNjk,ACBIMN->ABCIjk', H.bb.oooo[Ob, Ob, ob, ob], R.abb.VVVOOO, optimize=True)
    )
    dR.abb.VVVOoo += (2.0 / 4.0) * (
            +1.0 * np.einsum('mNIj,ACBmkN->ABCIjk', H.ab.oooo[oa, Ob, Oa, ob], R.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('MnIj,ACBMnk->ABCIjk', H.ab.oooo[Oa, ob, Oa, ob], R.abb.VVVOoo, optimize=True)
            + 1.0 * np.einsum('MNIj,ACBMkN->ABCIjk', H.ab.oooo[Oa, Ob, Oa, ob], R.abb.VVVOoO, optimize=True)
    )
    dR.abb.VVVOoo += (1.0 / 4.0) * (
            -0.5 * np.einsum('BCef,AfeIjk->ABCIjk', H.bb.vvvv[Vb, Vb, vb, vb], R.abb.VvvOoo, optimize=True)
            + 1.0 * np.einsum('BCEf,AEfIjk->ABCIjk', H.bb.vvvv[Vb, Vb, Vb, vb], R.abb.VVvOoo, optimize=True)
            - 0.5 * np.einsum('BCEF,AFEIjk->ABCIjk', H.bb.vvvv[Vb, Vb, Vb, Vb], R.abb.VVVOoo, optimize=True)
    )
    dR.abb.VVVOoo += (2.0 / 4.0) * (
            -1.0 * np.einsum('ABef,eCfIjk->ABCIjk', H.ab.vvvv[Va, Vb, va, vb], R.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('ABeF,eCFIjk->ABCIjk', H.ab.vvvv[Va, Vb, va, Vb], R.abb.vVVOoo, optimize=True)
            - 1.0 * np.einsum('ABEf,ECfIjk->ABCIjk', H.ab.vvvv[Va, Vb, Va, vb], R.abb.VVvOoo, optimize=True)
            - 1.0 * np.einsum('ABEF,ECFIjk->ABCIjk', H.ab.vvvv[Va, Vb, Va, Vb], R.abb.VVVOoo, optimize=True)
    )
    dR.abb.VVVOoo += (1.0 / 4.0) * (
            -1.0 * np.einsum('AMIe,eCBMjk->ABCIjk', H.aa.voov[Va, Oa, Oa, va], R.abb.vVVOoo, optimize=True)
            - 1.0 * np.einsum('AMIE,ECBMjk->ABCIjk', H.aa.voov[Va, Oa, Oa, Va], R.abb.VVVOoo, optimize=True)
    )
    dR.abb.VVVOoo += (1.0 / 4.0) * (
            -1.0 * np.einsum('AMIe,CBejkM->ABCIjk', H.ab.voov[Va, Ob, Oa, vb], R.bbb.VVvooO, optimize=True)
            - 1.0 * np.einsum('AMIE,CBEjkM->ABCIjk', H.ab.voov[Va, Ob, Oa, Vb], R.bbb.VVVooO, optimize=True)
    )
    dR.abb.VVVOoo += (4.0 / 4.0) * (
            -1.0 * np.einsum('mBej,AeCmIk->ABCIjk', H.ab.ovvo[oa, Vb, va, ob], R.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('MBej,AeCIMk->ABCIjk', H.ab.ovvo[Oa, Vb, va, ob], R.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('mBEj,EACmIk->ABCIjk', H.ab.ovvo[oa, Vb, Va, ob], R.aab.VVVoOo, optimize=True)
            - 1.0 * np.einsum('MBEj,EACIMk->ABCIjk', H.ab.ovvo[Oa, Vb, Va, ob], R.aab.VVVOOo, optimize=True)
    )
    dR.abb.VVVOoo += (4.0 / 4.0) * (
            -1.0 * np.einsum('Bmje,ACeImk->ABCIjk', H.bb.voov[Vb, ob, ob, vb], R.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('BMje,ACeIkM->ABCIjk', H.bb.voov[Vb, Ob, ob, vb], R.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('BmjE,ACEImk->ABCIjk', H.bb.voov[Vb, ob, ob, Vb], R.abb.VVVOoo, optimize=True)
            + 1.0 * np.einsum('BMjE,ACEIkM->ABCIjk', H.bb.voov[Vb, Ob, ob, Vb], R.abb.VVVOoO, optimize=True)
    )
    dR.abb.VVVOoo += (2.0 / 4.0) * (
            +1.0 * np.einsum('MBIe,ACeMjk->ABCIjk', H.ab.ovov[Oa, Vb, Oa, vb], R.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('MBIE,ACEMjk->ABCIjk', H.ab.ovov[Oa, Vb, Oa, Vb], R.abb.VVVOoo, optimize=True)
    )
    dR.abb.VVVOoo += (2.0 / 4.0) * (
            +1.0 * np.einsum('Amej,eCBImk->ABCIjk', H.ab.vovo[Va, ob, va, ob], R.abb.vVVOoo, optimize=True)
            - 1.0 * np.einsum('AMej,eCBIkM->ABCIjk', H.ab.vovo[Va, Ob, va, ob], R.abb.vVVOoO, optimize=True)
            + 1.0 * np.einsum('AmEj,ECBImk->ABCIjk', H.ab.vovo[Va, ob, Va, ob], R.abb.VVVOoo, optimize=True)
            - 1.0 * np.einsum('AMEj,ECBIkM->ABCIjk', H.ab.vovo[Va, Ob, Va, ob], R.abb.VVVOoO, optimize=True)
    )
    # of terms =  28

    dR.abb.VVVOoo -= np.transpose(dR.abb.VVVOoo, (0, 2, 1, 3, 4, 5))
    dR.abb.VVVOoo -= np.transpose(dR.abb.VVVOoo, (0, 1, 2, 3, 5, 4))

    return dR

def update(R, omega, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    R.abb.VVVOoo = eomcc_active_loops.update_r3c_111100(
        R.abb.VVVOoo,
        omega,
        H.a.oo[Oa, Oa],
        H.a.vv[Va, Va],
        H.a.oo[oa, oa],
        H.a.vv[va, va],
        H.b.oo[Ob, Ob],
        H.b.vv[Vb, Vb],
        H.b.oo[ob, ob],
        H.b.vv[vb, vb],
        0.0,
    )
    return R
