import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import cc_active_loops

def build(T, dT, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    # MM(2,3)
    dT.abb.VVVOoo = (2.0 / 4.0) * (
            +1.0 * np.einsum('ABIe,eCjk->ABCIjk', H.ab.vvov[Va, Vb, Oa, :], T.bb[:, Vb, ob, ob], optimize=True)
    )
    dT.abb.VVVOoo += (2.0 / 4.0) * (
            -1.0 * np.einsum('AmIj,BCmk->ABCIjk', H.ab.vooo[Va, :, Oa, ob], T.bb[Vb, Vb, :, ob], optimize=True)
    )
    dT.abb.VVVOoo += (2.0 / 4.0) * (
            +1.0 * np.einsum('CBke,AeIj->ABCIjk', H.bb.vvov[Vb, Vb, ob, :], T.ab[Va, :, Oa, ob], optimize=True)
    )
    dT.abb.VVVOoo += (2.0 / 4.0) * (
            -1.0 * np.einsum('Cmkj,ABIm->ABCIjk', H.bb.vooo[Vb, :, ob, ob], T.ab[Va, Vb, Oa, :], optimize=True)
    )
    dT.abb.VVVOoo += (4.0 / 4.0) * (
            +1.0 * np.einsum('ABej,eCIk->ABCIjk', H.ab.vvvo[Va, Vb, :, ob], T.ab[:, Vb, Oa, ob], optimize=True)
    )
    dT.abb.VVVOoo += (4.0 / 4.0) * (
            -1.0 * np.einsum('mBIj,ACmk->ABCIjk', H.ab.ovoo[:, Vb, Oa, ob], T.ab[Va, Vb, :, ob], optimize=True)
    )
    # (H(2) * T3)_C
    dT.abb.VVVOoo += (1.0 / 4.0) * (
            +1.0 * np.einsum('MI,ACBMjk->ABCIjk', H.a.oo[Oa, Oa], T.abb.VVVOoo, optimize=True)
    )
    dT.abb.VVVOoo += (2.0 / 4.0) * (
            +1.0 * np.einsum('mj,ACBImk->ABCIjk', H.b.oo[ob, ob], T.abb.VVVOoo, optimize=True)
            - 1.0 * np.einsum('Mj,ACBIkM->ABCIjk', H.b.oo[Ob, ob], T.abb.VVVOoO, optimize=True)
    )
    dT.abb.VVVOoo += (1.0 / 4.0) * (
            -1.0 * np.einsum('Ae,eCBIjk->ABCIjk', H.a.vv[Va, va], T.abb.vVVOoo, optimize=True)
            - 1.0 * np.einsum('AE,ECBIjk->ABCIjk', H.a.vv[Va, Va], T.abb.VVVOoo, optimize=True)
    )
    dT.abb.VVVOoo += (2.0 / 4.0) * (
            -1.0 * np.einsum('Be,ACeIjk->ABCIjk', H.b.vv[Vb, vb], T.abb.VVvOoo, optimize=True)
            - 1.0 * np.einsum('BE,ACEIjk->ABCIjk', H.b.vv[Vb, Vb], T.abb.VVVOoo, optimize=True)
    )
    dT.abb.VVVOoo += (1.0 / 4.0) * (
            -0.5 * np.einsum('mnjk,ACBImn->ABCIjk', H.bb.oooo[ob, ob, ob, ob], T.abb.VVVOoo, optimize=True)
            + 1.0 * np.einsum('Mnjk,ACBInM->ABCIjk', H.bb.oooo[Ob, ob, ob, ob], T.abb.VVVOoO, optimize=True)
            - 0.5 * np.einsum('MNjk,ACBIMN->ABCIjk', H.bb.oooo[Ob, Ob, ob, ob], T.abb.VVVOOO, optimize=True)
    )
    dT.abb.VVVOoo += (2.0 / 4.0) * (
            +1.0 * np.einsum('mNIj,ACBmkN->ABCIjk', H.ab.oooo[oa, Ob, Oa, ob], T.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('MnIj,ACBMnk->ABCIjk', H.ab.oooo[Oa, ob, Oa, ob], T.abb.VVVOoo, optimize=True)
            + 1.0 * np.einsum('MNIj,ACBMkN->ABCIjk', H.ab.oooo[Oa, Ob, Oa, ob], T.abb.VVVOoO, optimize=True)
    )
    dT.abb.VVVOoo += (1.0 / 4.0) * (
            -0.5 * np.einsum('BCef,AfeIjk->ABCIjk', H.bb.vvvv[Vb, Vb, vb, vb], T.abb.VvvOoo, optimize=True)
            - 1.0 * np.einsum('BCeF,AFeIjk->ABCIjk', H.bb.vvvv[Vb, Vb, vb, Vb], T.abb.VVvOoo, optimize=True)
            - 0.5 * np.einsum('BCEF,AFEIjk->ABCIjk', H.bb.vvvv[Vb, Vb, Vb, Vb], T.abb.VVVOoo, optimize=True)
    )
    dT.abb.VVVOoo += (2.0 / 4.0) * (
            -1.0 * np.einsum('ABef,eCfIjk->ABCIjk', H.ab.vvvv[Va, Vb, va, vb], T.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('ABEf,ECfIjk->ABCIjk', H.ab.vvvv[Va, Vb, Va, vb], T.abb.VVvOoo, optimize=True)
            - 1.0 * np.einsum('ABeF,eCFIjk->ABCIjk', H.ab.vvvv[Va, Vb, va, Vb], T.abb.vVVOoo, optimize=True)
            - 1.0 * np.einsum('ABEF,ECFIjk->ABCIjk', H.ab.vvvv[Va, Vb, Va, Vb], T.abb.VVVOoo, optimize=True)
    )
    dT.abb.VVVOoo += (1.0 / 4.0) * (
            -1.0 * np.einsum('AMIe,eCBMjk->ABCIjk', H.aa.voov[Va, Oa, Oa, va], T.abb.vVVOoo, optimize=True)
            - 1.0 * np.einsum('AMIE,ECBMjk->ABCIjk', H.aa.voov[Va, Oa, Oa, Va], T.abb.VVVOoo, optimize=True)
    )
    dT.abb.VVVOoo += (1.0 / 4.0) * (
            -1.0 * np.einsum('AMIe,CBejkM->ABCIjk', H.ab.voov[Va, Ob, Oa, vb], T.bbb.VVvooO, optimize=True)
            - 1.0 * np.einsum('AMIE,CBEjkM->ABCIjk', H.ab.voov[Va, Ob, Oa, Vb], T.bbb.VVVooO, optimize=True)
    )
    dT.abb.VVVOoo += (4.0 / 4.0) * (
            -1.0 * np.einsum('mBej,AeCmIk->ABCIjk', H.ab.ovvo[oa, Vb, va, ob], T.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('mBEj,EACmIk->ABCIjk', H.ab.ovvo[oa, Vb, Va, ob], T.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('MBej,AeCIMk->ABCIjk', H.ab.ovvo[Oa, Vb, va, ob], T.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('MBEj,EACIMk->ABCIjk', H.ab.ovvo[Oa, Vb, Va, ob], T.aab.VVVOOo, optimize=True)
    )
    dT.abb.VVVOoo += (4.0 / 4.0) * (
            -1.0 * np.einsum('Bmje,ACeImk->ABCIjk', H.bb.voov[Vb, ob, ob, vb], T.abb.VVvOoo, optimize=True)
            - 1.0 * np.einsum('BmjE,ACEImk->ABCIjk', H.bb.voov[Vb, ob, ob, Vb], T.abb.VVVOoo, optimize=True)
            + 1.0 * np.einsum('BMje,ACeIkM->ABCIjk', H.bb.voov[Vb, Ob, ob, vb], T.abb.VVvOoO, optimize=True)
            + 1.0 * np.einsum('BMjE,ACEIkM->ABCIjk', H.bb.voov[Vb, Ob, ob, Vb], T.abb.VVVOoO, optimize=True)
    )
    dT.abb.VVVOoo += (2.0 / 4.0) * (
            +1.0 * np.einsum('MBIe,ACeMjk->ABCIjk', H.ab.ovov[Oa, Vb, Oa, vb], T.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('MBIE,ACEMjk->ABCIjk', H.ab.ovov[Oa, Vb, Oa, Vb], T.abb.VVVOoo, optimize=True)
    )
    dT.abb.VVVOoo += (2.0 / 4.0) * (
            +1.0 * np.einsum('Amej,eCBImk->ABCIjk', H.ab.vovo[Va, ob, va, ob], T.abb.vVVOoo, optimize=True)
            + 1.0 * np.einsum('AmEj,ECBImk->ABCIjk', H.ab.vovo[Va, ob, Va, ob], T.abb.VVVOoo, optimize=True)
            - 1.0 * np.einsum('AMej,eCBIkM->ABCIjk', H.ab.vovo[Va, Ob, va, ob], T.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('AMEj,ECBIkM->ABCIjk', H.ab.vovo[Va, Ob, Va, ob], T.abb.VVVOoO, optimize=True)
    )

    dT.abb.VVVOoo -= np.transpose(dT.abb.VVVOoo, (0, 2, 1, 3, 4, 5))
    dT.abb.VVVOoo -= np.transpose(dT.abb.VVVOoo, (0, 1, 2, 3, 5, 4))

    return dT


def update(T, dT, H, shift, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.abb.VVVOoo, dT.abb.VVVOoo = cc_active_loops.update_t3c_111100(
        T.abb.VVVOoo,
        dT.abb.VVVOoo,
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
