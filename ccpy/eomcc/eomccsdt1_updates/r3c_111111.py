import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops

def build(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.abb.VVVOOO = (2.0 / 4.0) * (
            +1.0 * np.einsum('ABIe,eCJK->ABCIJK', X.ab.vvov[Va, Vb, Oa, :], T.bb[:, Vb, Ob, Ob], optimize=True)
    )
    dR.abb.VVVOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('AmIJ,BCmK->ABCIJK', X.ab.vooo[Va, :, Oa, Ob], T.bb[Vb, Vb, :, Ob], optimize=True)
    )
    dR.abb.VVVOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('CBKe,AeIJ->ABCIJK', X.bb.vvov[Vb, Vb, Ob, :], T.ab[Va, :, Oa, Ob], optimize=True)
    )
    dR.abb.VVVOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('CmKJ,ABIm->ABCIJK', X.bb.vooo[Vb, :, Ob, Ob], T.ab[Va, Vb, Oa, :], optimize=True)
    )
    dR.abb.VVVOOO += (4.0 / 4.0) * (
            +1.0 * np.einsum('ABeJ,eCIK->ABCIJK', X.ab.vvvo[Va, Vb, :, Ob], T.ab[:, Vb, Oa, Ob], optimize=True)
    )
    dR.abb.VVVOOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('mBIJ,ACmK->ABCIJK', X.ab.ovoo[:, Vb, Oa, Ob], T.ab[Va, Vb, :, Ob], optimize=True)
    )
    dR.abb.VVVOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('ABIe,eCJK->ABCIJK', H.ab.vvov[Va, Vb, Oa, :], R.bb[:, Vb, Ob, Ob], optimize=True)
    )
    dR.abb.VVVOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('AmIJ,BCmK->ABCIJK', H.ab.vooo[Va, :, Oa, Ob], R.bb[Vb, Vb, :, Ob], optimize=True)
    )
    dR.abb.VVVOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('CBKe,AeIJ->ABCIJK', H.bb.vvov[Vb, Vb, Ob, :], R.ab[Va, :, Oa, Ob], optimize=True)
    )
    dR.abb.VVVOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('CmKJ,ABIm->ABCIJK', H.bb.vooo[Vb, :, Ob, Ob], R.ab[Va, Vb, Oa, :], optimize=True)
    )
    dR.abb.VVVOOO += (4.0 / 4.0) * (
            +1.0 * np.einsum('ABeJ,eCIK->ABCIJK', H.ab.vvvo[Va, Vb, :, Ob], R.ab[:, Vb, Oa, Ob], optimize=True)
    )
    dR.abb.VVVOOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('mBIJ,ACmK->ABCIJK', H.ab.ovoo[:, Vb, Oa, Ob], R.ab[Va, Vb, :, Ob], optimize=True)
    )

    dR.abb.VVVOOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('mI,ACBmJK->ABCIJK', X.a.oo[oa, Oa], T.abb.VVVoOO, optimize=True)
            + 1.0 * np.einsum('MI,ACBMJK->ABCIJK', X.a.oo[Oa, Oa], T.abb.VVVOOO, optimize=True)
    )
    dR.abb.VVVOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mJ,ACBImK->ABCIJK', X.b.oo[ob, Ob], T.abb.VVVOoO, optimize=True)
            + 1.0 * np.einsum('MJ,ACBIMK->ABCIJK', X.b.oo[Ob, Ob], T.abb.VVVOOO, optimize=True)
    )
    dR.abb.VVVOOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('Ae,eCBIJK->ABCIJK', X.a.vv[Va, va], T.abb.vVVOOO, optimize=True)
            - 1.0 * np.einsum('AE,ECBIJK->ABCIJK', X.a.vv[Va, Va], T.abb.VVVOOO, optimize=True)
    )
    dR.abb.VVVOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('Be,ACeIJK->ABCIJK', X.b.vv[Vb, vb], T.abb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('BE,ACEIJK->ABCIJK', X.b.vv[Vb, Vb], T.abb.VVVOOO, optimize=True)
    )
    dR.abb.VVVOOO += (1.0 / 4.0) * (
            -0.5 * np.einsum('mnJK,ACBImn->ABCIJK', X.bb.oooo[ob, ob, Ob, Ob], T.abb.VVVOoo, optimize=True)
            - 1.0 * np.einsum('mNJK,ACBImN->ABCIJK', X.bb.oooo[ob, Ob, Ob, Ob], T.abb.VVVOoO, optimize=True)
            - 0.5 * np.einsum('MNJK,ACBIMN->ABCIJK', X.bb.oooo[Ob, Ob, Ob, Ob], T.abb.VVVOOO, optimize=True)
    )
    dR.abb.VVVOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('mnIJ,ACBmnK->ABCIJK', X.ab.oooo[oa, ob, Oa, Ob], T.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('MnIJ,ACBMnK->ABCIJK', X.ab.oooo[Oa, ob, Oa, Ob], T.abb.VVVOoO, optimize=True)
            - 1.0 * np.einsum('mNIJ,ACBmNK->ABCIJK', X.ab.oooo[oa, Ob, Oa, Ob], T.abb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('MNIJ,ACBMNK->ABCIJK', X.ab.oooo[Oa, Ob, Oa, Ob], T.abb.VVVOOO, optimize=True)
    )
    dR.abb.VVVOOO += (1.0 / 4.0) * (
            -0.5 * np.einsum('BCef,AfeIJK->ABCIJK', X.bb.vvvv[Vb, Vb, vb, vb], T.abb.VvvOOO, optimize=True)
            + 1.0 * np.einsum('BCEf,AEfIJK->ABCIJK', X.bb.vvvv[Vb, Vb, Vb, vb], T.abb.VVvOOO, optimize=True)
            - 0.5 * np.einsum('BCEF,AFEIJK->ABCIJK', X.bb.vvvv[Vb, Vb, Vb, Vb], T.abb.VVVOOO, optimize=True)
    )
    dR.abb.VVVOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('ABef,eCfIJK->ABCIJK', X.ab.vvvv[Va, Vb, va, vb], T.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('ABeF,eCFIJK->ABCIJK', X.ab.vvvv[Va, Vb, va, Vb], T.abb.vVVOOO, optimize=True)
            - 1.0 * np.einsum('ABEf,ECfIJK->ABCIJK', X.ab.vvvv[Va, Vb, Va, vb], T.abb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('ABEF,ECFIJK->ABCIJK', X.ab.vvvv[Va, Vb, Va, Vb], T.abb.VVVOOO, optimize=True)
    )
    dR.abb.VVVOOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('AmIe,eCBmJK->ABCIJK', X.aa.voov[Va, oa, Oa, va], T.abb.vVVoOO, optimize=True)
            - 1.0 * np.einsum('AMIe,eCBMJK->ABCIJK', X.aa.voov[Va, Oa, Oa, va], T.abb.vVVOOO, optimize=True)
            - 1.0 * np.einsum('AmIE,ECBmJK->ABCIJK', X.aa.voov[Va, oa, Oa, Va], T.abb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('AMIE,ECBMJK->ABCIJK', X.aa.voov[Va, Oa, Oa, Va], T.abb.VVVOOO, optimize=True)
    )
    dR.abb.VVVOOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('AmIe,CBemJK->ABCIJK', X.ab.voov[Va, ob, Oa, vb], T.bbb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('AMIe,CBeMJK->ABCIJK', X.ab.voov[Va, Ob, Oa, vb], T.bbb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('AmIE,CBEmJK->ABCIJK', X.ab.voov[Va, ob, Oa, Vb], T.bbb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('AMIE,CBEMJK->ABCIJK', X.ab.voov[Va, Ob, Oa, Vb], T.bbb.VVVOOO, optimize=True)
    )
    dR.abb.VVVOOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('mBeJ,AeCmIK->ABCIJK', X.ab.ovvo[oa, Vb, va, Ob], T.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('MBeJ,AeCIMK->ABCIJK', X.ab.ovvo[Oa, Vb, va, Ob], T.aab.VvVOOO, optimize=True)
            + 1.0 * np.einsum('mBEJ,EACmIK->ABCIJK', X.ab.ovvo[oa, Vb, Va, Ob], T.aab.VVVoOO, optimize=True)
            - 1.0 * np.einsum('MBEJ,EACIMK->ABCIJK', X.ab.ovvo[Oa, Vb, Va, Ob], T.aab.VVVOOO, optimize=True)
    )
    dR.abb.VVVOOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('BmJe,ACeImK->ABCIJK', X.bb.voov[Vb, ob, Ob, vb], T.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('BMJe,ACeIMK->ABCIJK', X.bb.voov[Vb, Ob, Ob, vb], T.abb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('BmJE,ACEImK->ABCIJK', X.bb.voov[Vb, ob, Ob, Vb], T.abb.VVVOoO, optimize=True)
            - 1.0 * np.einsum('BMJE,ACEIMK->ABCIJK', X.bb.voov[Vb, Ob, Ob, Vb], T.abb.VVVOOO, optimize=True)
    )
    dR.abb.VVVOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mBIe,ACemJK->ABCIJK', X.ab.ovov[oa, Vb, Oa, vb], T.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('MBIe,ACeMJK->ABCIJK', X.ab.ovov[Oa, Vb, Oa, vb], T.abb.VVvOOO, optimize=True)
            + 1.0 * np.einsum('mBIE,ACEmJK->ABCIJK', X.ab.ovov[oa, Vb, Oa, Vb], T.abb.VVVoOO, optimize=True)
            + 1.0 * np.einsum('MBIE,ACEMJK->ABCIJK', X.ab.ovov[Oa, Vb, Oa, Vb], T.abb.VVVOOO, optimize=True)
    )
    dR.abb.VVVOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('AmeJ,eCBImK->ABCIJK', X.ab.vovo[Va, ob, va, Ob], T.abb.vVVOoO, optimize=True)
            + 1.0 * np.einsum('AMeJ,eCBIMK->ABCIJK', X.ab.vovo[Va, Ob, va, Ob], T.abb.vVVOOO, optimize=True)
            + 1.0 * np.einsum('AmEJ,ECBImK->ABCIJK', X.ab.vovo[Va, ob, Va, Ob], T.abb.VVVOoO, optimize=True)
            + 1.0 * np.einsum('AMEJ,ECBIMK->ABCIJK', X.ab.vovo[Va, Ob, Va, Ob], T.abb.VVVOOO, optimize=True)
    )
    dR.abb.VVVOOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('mI,ACBmJK->ABCIJK', H.a.oo[oa, Oa], R.abb.VVVoOO, optimize=True)
            + 1.0 * np.einsum('MI,ACBMJK->ABCIJK', H.a.oo[Oa, Oa], R.abb.VVVOOO, optimize=True)
    )
    dR.abb.VVVOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mJ,ACBImK->ABCIJK', H.b.oo[ob, Ob], R.abb.VVVOoO, optimize=True)
            + 1.0 * np.einsum('MJ,ACBIMK->ABCIJK', H.b.oo[Ob, Ob], R.abb.VVVOOO, optimize=True)
    )
    dR.abb.VVVOOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('Ae,eCBIJK->ABCIJK', H.a.vv[Va, va], R.abb.vVVOOO, optimize=True)
            - 1.0 * np.einsum('AE,ECBIJK->ABCIJK', H.a.vv[Va, Va], R.abb.VVVOOO, optimize=True)
    )
    dR.abb.VVVOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('Be,ACeIJK->ABCIJK', H.b.vv[Vb, vb], R.abb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('BE,ACEIJK->ABCIJK', H.b.vv[Vb, Vb], R.abb.VVVOOO, optimize=True)
    )
    dR.abb.VVVOOO += (1.0 / 4.0) * (
            -0.5 * np.einsum('mnJK,ACBImn->ABCIJK', H.bb.oooo[ob, ob, Ob, Ob], R.abb.VVVOoo, optimize=True)
            - 1.0 * np.einsum('mNJK,ACBImN->ABCIJK', H.bb.oooo[ob, Ob, Ob, Ob], R.abb.VVVOoO, optimize=True)
            - 0.5 * np.einsum('MNJK,ACBIMN->ABCIJK', H.bb.oooo[Ob, Ob, Ob, Ob], R.abb.VVVOOO, optimize=True)
    )
    dR.abb.VVVOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('mnIJ,ACBmnK->ABCIJK', H.ab.oooo[oa, ob, Oa, Ob], R.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('MnIJ,ACBMnK->ABCIJK', H.ab.oooo[Oa, ob, Oa, Ob], R.abb.VVVOoO, optimize=True)
            - 1.0 * np.einsum('mNIJ,ACBmNK->ABCIJK', H.ab.oooo[oa, Ob, Oa, Ob], R.abb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('MNIJ,ACBMNK->ABCIJK', H.ab.oooo[Oa, Ob, Oa, Ob], R.abb.VVVOOO, optimize=True)
    )
    dR.abb.VVVOOO += (1.0 / 4.0) * (
            -0.5 * np.einsum('BCef,AfeIJK->ABCIJK', H.bb.vvvv[Vb, Vb, vb, vb], R.abb.VvvOOO, optimize=True)
            + 1.0 * np.einsum('BCEf,AEfIJK->ABCIJK', H.bb.vvvv[Vb, Vb, Vb, vb], R.abb.VVvOOO, optimize=True)
            - 0.5 * np.einsum('BCEF,AFEIJK->ABCIJK', H.bb.vvvv[Vb, Vb, Vb, Vb], R.abb.VVVOOO, optimize=True)
    )
    dR.abb.VVVOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('ABef,eCfIJK->ABCIJK', H.ab.vvvv[Va, Vb, va, vb], R.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('ABeF,eCFIJK->ABCIJK', H.ab.vvvv[Va, Vb, va, Vb], R.abb.vVVOOO, optimize=True)
            - 1.0 * np.einsum('ABEf,ECfIJK->ABCIJK', H.ab.vvvv[Va, Vb, Va, vb], R.abb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('ABEF,ECFIJK->ABCIJK', H.ab.vvvv[Va, Vb, Va, Vb], R.abb.VVVOOO, optimize=True)
    )
    dR.abb.VVVOOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('AmIe,eCBmJK->ABCIJK', H.aa.voov[Va, oa, Oa, va], R.abb.vVVoOO, optimize=True)
            - 1.0 * np.einsum('AMIe,eCBMJK->ABCIJK', H.aa.voov[Va, Oa, Oa, va], R.abb.vVVOOO, optimize=True)
            - 1.0 * np.einsum('AmIE,ECBmJK->ABCIJK', H.aa.voov[Va, oa, Oa, Va], R.abb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('AMIE,ECBMJK->ABCIJK', H.aa.voov[Va, Oa, Oa, Va], R.abb.VVVOOO, optimize=True)
    )
    dR.abb.VVVOOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('AmIe,CBemJK->ABCIJK', H.ab.voov[Va, ob, Oa, vb], R.bbb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('AMIe,CBeMJK->ABCIJK', H.ab.voov[Va, Ob, Oa, vb], R.bbb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('AmIE,CBEmJK->ABCIJK', H.ab.voov[Va, ob, Oa, Vb], R.bbb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('AMIE,CBEMJK->ABCIJK', H.ab.voov[Va, Ob, Oa, Vb], R.bbb.VVVOOO, optimize=True)
    )
    dR.abb.VVVOOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('mBeJ,AeCmIK->ABCIJK', H.ab.ovvo[oa, Vb, va, Ob], R.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('MBeJ,AeCIMK->ABCIJK', H.ab.ovvo[Oa, Vb, va, Ob], R.aab.VvVOOO, optimize=True)
            + 1.0 * np.einsum('mBEJ,EACmIK->ABCIJK', H.ab.ovvo[oa, Vb, Va, Ob], R.aab.VVVoOO, optimize=True)
            - 1.0 * np.einsum('MBEJ,EACIMK->ABCIJK', H.ab.ovvo[Oa, Vb, Va, Ob], R.aab.VVVOOO, optimize=True)
    )
    dR.abb.VVVOOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('BmJe,ACeImK->ABCIJK', H.bb.voov[Vb, ob, Ob, vb], R.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('BMJe,ACeIMK->ABCIJK', H.bb.voov[Vb, Ob, Ob, vb], R.abb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('BmJE,ACEImK->ABCIJK', H.bb.voov[Vb, ob, Ob, Vb], R.abb.VVVOoO, optimize=True)
            - 1.0 * np.einsum('BMJE,ACEIMK->ABCIJK', H.bb.voov[Vb, Ob, Ob, Vb], R.abb.VVVOOO, optimize=True)
    )
    dR.abb.VVVOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mBIe,ACemJK->ABCIJK', H.ab.ovov[oa, Vb, Oa, vb], R.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('MBIe,ACeMJK->ABCIJK', H.ab.ovov[Oa, Vb, Oa, vb], R.abb.VVvOOO, optimize=True)
            + 1.0 * np.einsum('mBIE,ACEmJK->ABCIJK', H.ab.ovov[oa, Vb, Oa, Vb], R.abb.VVVoOO, optimize=True)
            + 1.0 * np.einsum('MBIE,ACEMJK->ABCIJK', H.ab.ovov[Oa, Vb, Oa, Vb], R.abb.VVVOOO, optimize=True)
    )
    dR.abb.VVVOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('AmeJ,eCBImK->ABCIJK', H.ab.vovo[Va, ob, va, Ob], R.abb.vVVOoO, optimize=True)
            + 1.0 * np.einsum('AMeJ,eCBIMK->ABCIJK', H.ab.vovo[Va, Ob, va, Ob], R.abb.vVVOOO, optimize=True)
            + 1.0 * np.einsum('AmEJ,ECBImK->ABCIJK', H.ab.vovo[Va, ob, Va, Ob], R.abb.VVVOoO, optimize=True)
            + 1.0 * np.einsum('AMEJ,ECBIMK->ABCIJK', H.ab.vovo[Va, Ob, Va, Ob], R.abb.VVVOOO, optimize=True)
    )

    dR.abb.VVVOOO -= np.transpose(dR.abb.VVVOOO, (0, 2, 1, 3, 4, 5))
    dR.abb.VVVOOO -= np.transpose(dR.abb.VVVOOO, (0, 1, 2, 3, 5, 4))

    return dR

def update(R, omega, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    R.abb.VVVOOO = eomcc_active_loops.update_r3c_111111(
        R.abb.VVVOOO,
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
