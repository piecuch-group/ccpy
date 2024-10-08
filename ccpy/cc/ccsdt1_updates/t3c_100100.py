import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import cc_active_loops

def build(T, dT, H, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    # MM(2,3)
    dT.abb.VvvOoo = (2.0 / 4.0) * (
            +1.0 * np.einsum('AbIe,ecjk->AbcIjk', H.ab.vvov[Va, vb, Oa, :], T.bb[:, vb, ob, ob], optimize=True)
    )
    dT.abb.VvvOoo += (2.0 / 4.0) * (
            -1.0 * np.einsum('AmIj,bcmk->AbcIjk', H.ab.vooo[Va, :, Oa, ob], T.bb[vb, vb, :, ob], optimize=True)
    )
    dT.abb.VvvOoo += (2.0 / 4.0) * (
            +1.0 * np.einsum('cbke,AeIj->AbcIjk', H.bb.vvov[vb, vb, ob, :], T.ab[Va, :, Oa, ob], optimize=True)
    )
    dT.abb.VvvOoo += (2.0 / 4.0) * (
            -1.0 * np.einsum('cmkj,AbIm->AbcIjk', H.bb.vooo[vb, :, ob, ob], T.ab[Va, vb, Oa, :], optimize=True)
    )
    dT.abb.VvvOoo += (4.0 / 4.0) * (
            +1.0 * np.einsum('Abej,ecIk->AbcIjk', H.ab.vvvo[Va, vb, :, ob], T.ab[:, vb, Oa, ob], optimize=True)
    )
    dT.abb.VvvOoo += (4.0 / 4.0) * (
            -1.0 * np.einsum('mbIj,Acmk->AbcIjk', H.ab.ovoo[:, vb, Oa, ob], T.ab[Va, vb, :, ob], optimize=True)
    )
    # (H(2) * T3)_C
    dT.abb.VvvOoo += (1.0 / 4.0) * (
            +1.0 * np.einsum('MI,AcbMjk->AbcIjk', H.a.oo[Oa, Oa], T.abb.VvvOoo, optimize=True)
    )
    dT.abb.VvvOoo += (2.0 / 4.0) * (
            +1.0 * np.einsum('mj,AcbImk->AbcIjk', H.b.oo[ob, ob], T.abb.VvvOoo, optimize=True)
            - 1.0 * np.einsum('Mj,AcbIkM->AbcIjk', H.b.oo[Ob, ob], T.abb.VvvOoO, optimize=True)
    )
    dT.abb.VvvOoo += (1.0 / 4.0) * (
            -1.0 * np.einsum('AE,EcbIjk->AbcIjk', H.a.vv[Va, Va], T.abb.VvvOoo, optimize=True)
    )
    dT.abb.VvvOoo += (2.0 / 4.0) * (
            -1.0 * np.einsum('be,AceIjk->AbcIjk', H.b.vv[vb, vb], T.abb.VvvOoo, optimize=True)
            + 1.0 * np.einsum('bE,AEcIjk->AbcIjk', H.b.vv[vb, Vb], T.abb.VVvOoo, optimize=True)
    )
    dT.abb.VvvOoo += (1.0 / 4.0) * (
            -0.5 * np.einsum('mnjk,AcbImn->AbcIjk', H.bb.oooo[ob, ob, ob, ob], T.abb.VvvOoo, optimize=True)
            - 1.0 * np.einsum('mNjk,AcbImN->AbcIjk', H.bb.oooo[ob, Ob, ob, ob], T.abb.VvvOoO, optimize=True)
            - 0.5 * np.einsum('MNjk,AcbIMN->AbcIjk', H.bb.oooo[Ob, Ob, ob, ob], T.abb.VvvOOO, optimize=True)
    )
    dT.abb.VvvOoo += (2.0 / 4.0) * (
            -1.0 * np.einsum('MnIj,AcbMnk->AbcIjk', H.ab.oooo[Oa, ob, Oa, ob], T.abb.VvvOoo, optimize=True)
            + 1.0 * np.einsum('mNIj,AcbmkN->AbcIjk', H.ab.oooo[oa, Ob, Oa, ob], T.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('MNIj,AcbMkN->AbcIjk', H.ab.oooo[Oa, Ob, Oa, ob], T.abb.VvvOoO, optimize=True)
    )
    dT.abb.VvvOoo += (1.0 / 4.0) * (
            -0.5 * np.einsum('bcef,AfeIjk->AbcIjk', H.bb.vvvv[vb, vb, vb, vb], T.abb.VvvOoo, optimize=True)
            - 1.0 * np.einsum('bceF,AFeIjk->AbcIjk', H.bb.vvvv[vb, vb, vb, Vb], T.abb.VVvOoo, optimize=True)
            - 0.5 * np.einsum('bcEF,AFEIjk->AbcIjk', H.bb.vvvv[vb, vb, Vb, Vb], T.abb.VVVOoo, optimize=True)
    )
    dT.abb.VvvOoo += (2.0 / 4.0) * (
            -1.0 * np.einsum('AbEf,EcfIjk->AbcIjk', H.ab.vvvv[Va, vb, Va, vb], T.abb.VvvOoo, optimize=True)
            + 1.0 * np.einsum('AbeF,eFcIjk->AbcIjk', H.ab.vvvv[Va, vb, va, Vb], T.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('AbEF,EFcIjk->AbcIjk', H.ab.vvvv[Va, vb, Va, Vb], T.abb.VVvOoo, optimize=True)
    )
    dT.abb.VvvOoo += (1.0 / 4.0) * (
            -1.0 * np.einsum('AMIE,EcbMjk->AbcIjk', H.aa.voov[Va, Oa, Oa, Va], T.abb.VvvOoo, optimize=True)
    )
    dT.abb.VvvOoo += (1.0 / 4.0) * (
            -1.0 * np.einsum('AMIE,EcbjkM->AbcIjk', H.ab.voov[Va, Ob, Oa, Vb], T.bbb.VvvooO, optimize=True)
    )
    dT.abb.VvvOoo += (4.0 / 4.0) * (
            -1.0 * np.einsum('mbej,AecmIk->AbcIjk', H.ab.ovvo[oa, vb, va, ob], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('Mbej,AecIMk->AbcIjk', H.ab.ovvo[Oa, vb, va, ob], T.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('mbEj,EAcmIk->AbcIjk', H.ab.ovvo[oa, vb, Va, ob], T.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('MbEj,EAcIMk->AbcIjk', H.ab.ovvo[Oa, vb, Va, ob], T.aab.VVvOOo, optimize=True)
    )
    dT.abb.VvvOoo += (4.0 / 4.0) * (
            -1.0 * np.einsum('bmje,AceImk->AbcIjk', H.bb.voov[vb, ob, ob, vb], T.abb.VvvOoo, optimize=True)
            + 1.0 * np.einsum('bMje,AceIkM->AbcIjk', H.bb.voov[vb, Ob, ob, vb], T.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('bmjE,AEcImk->AbcIjk', H.bb.voov[vb, ob, ob, Vb], T.abb.VVvOoo, optimize=True)
            - 1.0 * np.einsum('bMjE,AEcIkM->AbcIjk', H.bb.voov[vb, Ob, ob, Vb], T.abb.VVvOoO, optimize=True)
    )
    dT.abb.VvvOoo += (2.0 / 4.0) * (
            +1.0 * np.einsum('MbIe,AceMjk->AbcIjk', H.ab.ovov[Oa, vb, Oa, vb], T.abb.VvvOoo, optimize=True)
            - 1.0 * np.einsum('MbIE,AEcMjk->AbcIjk', H.ab.ovov[Oa, vb, Oa, Vb], T.abb.VVvOoo, optimize=True)
    )
    dT.abb.VvvOoo += (2.0 / 4.0) * (
            +1.0 * np.einsum('AmEj,EcbImk->AbcIjk', H.ab.vovo[Va, ob, Va, ob], T.abb.VvvOoo, optimize=True)
            - 1.0 * np.einsum('AMEj,EcbIkM->AbcIjk', H.ab.vovo[Va, Ob, Va, ob], T.abb.VvvOoO, optimize=True)
    )

    dT.abb.VvvOoo -= np.transpose(dT.abb.VvvOoo, (0, 2, 1, 3, 4, 5))
    dT.abb.VvvOoo -= np.transpose(dT.abb.VvvOoo, (0, 1, 2, 3, 5, 4))

    return dT


def update(T, dT, H, shift, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.abb.VvvOoo, dT.abb.VvvOoo = cc_active_loops.update_t3c_100100(
        T.abb.VvvOoo,
        dT.abb.VvvOoo,
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