import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import cc_active_loops

def build(T, dT, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    # MM(2,3)
    dT.abb.VVVOOO = (2.0 / 4.0) * (
            +1.0 * np.einsum('ABIe,eCJK->ABCIJK', H.ab.vvov[Va, Vb, Oa, :], T.bb[:, Vb, Ob, Ob], optimize=True)
    )
    dT.abb.VVVOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('AmIJ,BCmK->ABCIJK', H.ab.vooo[Va, :, Oa, Ob], T.bb[Vb, Vb, :, Ob], optimize=True)
    )
    dT.abb.VVVOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('CBKe,AeIJ->ABCIJK', H.bb.vvov[Vb, Vb, Ob, :], T.ab[Va, :, Oa, Ob], optimize=True)
    )
    dT.abb.VVVOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('CmKJ,ABIm->ABCIJK', H.bb.vooo[Vb, :, Ob, Ob], T.ab[Va, Vb, Oa, :], optimize=True)
    )
    dT.abb.VVVOOO += (4.0 / 4.0) * (
            +1.0 * np.einsum('ABeJ,eCIK->ABCIJK', H.ab.vvvo[Va, Vb, :, Ob], T.ab[:, Vb, Oa, Ob], optimize=True)
    )
    dT.abb.VVVOOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('mBIJ,ACmK->ABCIJK', H.ab.ovoo[:, Vb, Oa, Ob], T.ab[Va, Vb, :, Ob], optimize=True)
    )
    # (H(2) * T3)_C
    dT.abb.VVVOOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('mI,ACBmJK->ABCIJK', H.a.oo[oa, Oa], T.abb.VVVoOO, optimize=True)
            + 1.0 * np.einsum('MI,ACBMJK->ABCIJK', H.a.oo[Oa, Oa], T.abb.VVVOOO, optimize=True)
    )
    dT.abb.VVVOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mJ,ACBImK->ABCIJK', H.b.oo[ob, Ob], T.abb.VVVOoO, optimize=True)
            + 1.0 * np.einsum('MJ,ACBIMK->ABCIJK', H.b.oo[Ob, Ob], T.abb.VVVOOO, optimize=True)
    )
    dT.abb.VVVOOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('Ae,eCBIJK->ABCIJK', H.a.vv[Va, va], T.abb.vVVOOO, optimize=True)
            - 1.0 * np.einsum('AE,ECBIJK->ABCIJK', H.a.vv[Va, Va], T.abb.VVVOOO, optimize=True)
    )
    dT.abb.VVVOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('Be,ACeIJK->ABCIJK', H.b.vv[Vb, vb], T.abb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('BE,ACEIJK->ABCIJK', H.b.vv[Vb, Vb], T.abb.VVVOOO, optimize=True)
    )
    dT.abb.VVVOOO += (1.0 / 4.0) * (
            -0.5 * np.einsum('mnJK,ACBImn->ABCIJK', H.bb.oooo[ob, ob, Ob, Ob], T.abb.VVVOoo, optimize=True)
            + 1.0 * np.einsum('MnJK,ACBInM->ABCIJK', H.bb.oooo[Ob, ob, Ob, Ob], T.abb.VVVOoO, optimize=True)
            - 0.5 * np.einsum('MNJK,ACBIMN->ABCIJK', H.bb.oooo[Ob, Ob, Ob, Ob], T.abb.VVVOOO, optimize=True)
    )
    dT.abb.VVVOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('mnIJ,ACBmnK->ABCIJK', H.ab.oooo[oa, ob, Oa, Ob], T.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('mNIJ,ACBmNK->ABCIJK', H.ab.oooo[oa, Ob, Oa, Ob], T.abb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('MnIJ,ACBMnK->ABCIJK', H.ab.oooo[Oa, ob, Oa, Ob], T.abb.VVVOoO, optimize=True)
            - 1.0 * np.einsum('MNIJ,ACBMNK->ABCIJK', H.ab.oooo[Oa, Ob, Oa, Ob], T.abb.VVVOOO, optimize=True)
    )
    dT.abb.VVVOOO += (1.0 / 4.0) * (
            -0.5 * np.einsum('BCef,AfeIJK->ABCIJK', H.bb.vvvv[Vb, Vb, vb, vb], T.abb.VvvOOO, optimize=True)
            + 1.0 * np.einsum('BCEf,AEfIJK->ABCIJK', H.bb.vvvv[Vb, Vb, Vb, vb], T.abb.VVvOOO, optimize=True)
            - 0.5 * np.einsum('BCEF,AFEIJK->ABCIJK', H.bb.vvvv[Vb, Vb, Vb, Vb], T.abb.VVVOOO, optimize=True)
    )
    dT.abb.VVVOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('ABef,eCfIJK->ABCIJK', H.ab.vvvv[Va, Vb, va, vb], T.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('ABeF,eCFIJK->ABCIJK', H.ab.vvvv[Va, Vb, va, Vb], T.abb.vVVOOO, optimize=True)
            - 1.0 * np.einsum('ABEf,ECfIJK->ABCIJK', H.ab.vvvv[Va, Vb, Va, vb], T.abb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('ABEF,ECFIJK->ABCIJK', H.ab.vvvv[Va, Vb, Va, Vb], T.abb.VVVOOO, optimize=True)
    )
    dT.abb.VVVOOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('AmIe,eCBmJK->ABCIJK', H.aa.voov[Va, oa, Oa, va], T.abb.vVVoOO, optimize=True)
            - 1.0 * np.einsum('AmIE,ECBmJK->ABCIJK', H.aa.voov[Va, oa, Oa, Va], T.abb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('AMIe,eCBMJK->ABCIJK', H.aa.voov[Va, Oa, Oa, va], T.abb.vVVOOO, optimize=True)
            - 1.0 * np.einsum('AMIE,ECBMJK->ABCIJK', H.aa.voov[Va, Oa, Oa, Va], T.abb.VVVOOO, optimize=True)
    )
    dT.abb.VVVOOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('AmIe,CBemJK->ABCIJK', H.ab.voov[Va, ob, Oa, vb], T.bbb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('AmIE,CBEmJK->ABCIJK', H.ab.voov[Va, ob, Oa, Vb], T.bbb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('AMIe,CBeMJK->ABCIJK', H.ab.voov[Va, Ob, Oa, vb], T.bbb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('AMIE,CBEMJK->ABCIJK', H.ab.voov[Va, Ob, Oa, Vb], T.bbb.VVVOOO, optimize=True)
    )
    dT.abb.VVVOOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('mBeJ,AeCmIK->ABCIJK', H.ab.ovvo[oa, Vb, va, Ob], T.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('mBEJ,EACmIK->ABCIJK', H.ab.ovvo[oa, Vb, Va, Ob], T.aab.VVVoOO, optimize=True)
            + 1.0 * np.einsum('MBeJ,AeCIMK->ABCIJK', H.ab.ovvo[Oa, Vb, va, Ob], T.aab.VvVOOO, optimize=True)
            - 1.0 * np.einsum('MBEJ,EACIMK->ABCIJK', H.ab.ovvo[Oa, Vb, Va, Ob], T.aab.VVVOOO, optimize=True)
    )
    dT.abb.VVVOOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('BmJe,ACeImK->ABCIJK', H.bb.voov[Vb, ob, Ob, vb], T.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('BmJE,ACEImK->ABCIJK', H.bb.voov[Vb, ob, Ob, Vb], T.abb.VVVOoO, optimize=True)
            - 1.0 * np.einsum('BMJe,ACeIMK->ABCIJK', H.bb.voov[Vb, Ob, Ob, vb], T.abb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('BMJE,ACEIMK->ABCIJK', H.bb.voov[Vb, Ob, Ob, Vb], T.abb.VVVOOO, optimize=True)
    )
    dT.abb.VVVOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mBIe,ACemJK->ABCIJK', H.ab.ovov[oa, Vb, Oa, vb], T.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('mBIE,ACEmJK->ABCIJK', H.ab.ovov[oa, Vb, Oa, Vb], T.abb.VVVoOO, optimize=True)
            + 1.0 * np.einsum('MBIe,ACeMJK->ABCIJK', H.ab.ovov[Oa, Vb, Oa, vb], T.abb.VVvOOO, optimize=True)
            + 1.0 * np.einsum('MBIE,ACEMJK->ABCIJK', H.ab.ovov[Oa, Vb, Oa, Vb], T.abb.VVVOOO, optimize=True)
    )
    dT.abb.VVVOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('AmeJ,eCBImK->ABCIJK', H.ab.vovo[Va, ob, va, Ob], T.abb.vVVOoO, optimize=True)
            + 1.0 * np.einsum('AmEJ,ECBImK->ABCIJK', H.ab.vovo[Va, ob, Va, Ob], T.abb.VVVOoO, optimize=True)
            + 1.0 * np.einsum('AMeJ,eCBIMK->ABCIJK', H.ab.vovo[Va, Ob, va, Ob], T.abb.vVVOOO, optimize=True)
            + 1.0 * np.einsum('AMEJ,ECBIMK->ABCIJK', H.ab.vovo[Va, Ob, Va, Ob], T.abb.VVVOOO, optimize=True)
    )

    dT.abb.VVVOOO -= np.transpose(dT.abb.VVVOOO, (0, 2, 1, 3, 4, 5))
    dT.abb.VVVOOO -= np.transpose(dT.abb.VVVOOO, (0, 1, 2, 3, 5, 4))

    return dT

def update(T, dT, H, shift, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.abb.VVVOOO, dT.abb.VVVOOO = cc_active_loops.update_t3c_111111(
        T.abb.VVVOOO,
        dT.abb.VVVOOO,
        H.a.oo[Oa, Oa],
        H.a.vv[Va, Va],
        H.a.oo[oa, oa],
        H.a.vv[va, va],
        H.b.oo[Ob, Ob],
        H.b.vv[Vb, Vb],
        H.b.oo[ob, ob],
        H.b.vv[vb, vb],
        shift,
    )

    return T, dT