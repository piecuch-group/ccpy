import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import cc_active_loops

def build(T, dT, H, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    # MM(2,3)
    dT.abb.vVVOoo = (2.0 / 4.0) * (
            +1.0 * np.einsum('aBIe,eCjk->aBCIjk', H.ab.vvov[va, Vb, Oa, :], T.bb[:, Vb, ob, ob], optimize=True)
    )
    dT.abb.vVVOoo += (2.0 / 4.0) * (
            -1.0 * np.einsum('amIj,BCmk->aBCIjk', H.ab.vooo[va, :, Oa, ob], T.bb[Vb, Vb, :, ob], optimize=True)
    )
    dT.abb.vVVOoo += (2.0 / 4.0) * (
            +1.0 * np.einsum('CBke,aeIj->aBCIjk', H.bb.vvov[Vb, Vb, ob, :], T.ab[va, :, Oa, ob], optimize=True)
    )
    dT.abb.vVVOoo += (2.0 / 4.0) * (
            -1.0 * np.einsum('Cmkj,aBIm->aBCIjk', H.bb.vooo[Vb, :, ob, ob], T.ab[va, Vb, Oa, :], optimize=True)
    )
    dT.abb.vVVOoo += (4.0 / 4.0) * (
            +1.0 * np.einsum('aBej,eCIk->aBCIjk', H.ab.vvvo[va, Vb, :, ob], T.ab[:, Vb, Oa, ob], optimize=True)
    )
    dT.abb.vVVOoo += (4.0 / 4.0) * (
            -1.0 * np.einsum('mBIj,aCmk->aBCIjk', H.ab.ovoo[:, Vb, Oa, ob], T.ab[va, Vb, :, ob], optimize=True)
    )
    # (H(2) * T3)_C
    dT.abb.vVVOoo += (1.0 / 4.0) * (
            +1.0 * np.einsum('MI,aCBMjk->aBCIjk', H.a.oo[Oa, Oa], T.abb.vVVOoo, optimize=True)
    )
    dT.abb.vVVOoo += (2.0 / 4.0) * (
            +1.0 * np.einsum('mj,aCBImk->aBCIjk', H.b.oo[ob, ob], T.abb.vVVOoo, optimize=True)
            - 1.0 * np.einsum('Mj,aCBIkM->aBCIjk', H.b.oo[Ob, ob], T.abb.vVVOoO, optimize=True)
    )
    dT.abb.vVVOoo += (1.0 / 4.0) * (
            -1.0 * np.einsum('ae,eCBIjk->aBCIjk', H.a.vv[va, va], T.abb.vVVOoo, optimize=True)
            - 1.0 * np.einsum('aE,ECBIjk->aBCIjk', H.a.vv[va, Va], T.abb.VVVOoo, optimize=True)
    )
    dT.abb.vVVOoo += (2.0 / 4.0) * (
            -1.0 * np.einsum('Be,aCeIjk->aBCIjk', H.b.vv[Vb, vb], T.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('BE,aCEIjk->aBCIjk', H.b.vv[Vb, Vb], T.abb.vVVOoo, optimize=True)
    )
    dT.abb.vVVOoo += (1.0 / 4.0) * (
            -0.5 * np.einsum('mnjk,aCBImn->aBCIjk', H.bb.oooo[ob, ob, ob, ob], T.abb.vVVOoo, optimize=True)
            + 1.0 * np.einsum('Mnjk,aCBInM->aBCIjk', H.bb.oooo[Ob, ob, ob, ob], T.abb.vVVOoO, optimize=True)
            - 0.5 * np.einsum('MNjk,aCBIMN->aBCIjk', H.bb.oooo[Ob, Ob, ob, ob], T.abb.vVVOOO, optimize=True)
    )
    dT.abb.vVVOoo += (2.0 / 4.0) * (
            +1.0 * np.einsum('mNIj,aCBmkN->aBCIjk', H.ab.oooo[oa, Ob, Oa, ob], T.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('MnIj,aCBMnk->aBCIjk', H.ab.oooo[Oa, ob, Oa, ob], T.abb.vVVOoo, optimize=True)
            + 1.0 * np.einsum('MNIj,aCBMkN->aBCIjk', H.ab.oooo[Oa, Ob, Oa, ob], T.abb.vVVOoO, optimize=True)
    )
    dT.abb.vVVOoo += (1.0 / 4.0) * (
            +1.0 * np.einsum('BCEf,aEfIjk->aBCIjk', H.bb.vvvv[Vb, Vb, Vb, vb], T.abb.vVvOoo, optimize=True)
            - 0.5 * np.einsum('BCEF,aFEIjk->aBCIjk', H.bb.vvvv[Vb, Vb, Vb, Vb], T.abb.vVVOoo, optimize=True)
    )
    dT.abb.vVVOoo += (2.0 / 4.0) * (
            -1.0 * np.einsum('aBef,eCfIjk->aBCIjk', H.ab.vvvv[va, Vb, va, vb], T.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('aBeF,eCFIjk->aBCIjk', H.ab.vvvv[va, Vb, va, Vb], T.abb.vVVOoo, optimize=True)
            - 1.0 * np.einsum('aBEf,ECfIjk->aBCIjk', H.ab.vvvv[va, Vb, Va, vb], T.abb.VVvOoo, optimize=True)
            - 1.0 * np.einsum('aBEF,ECFIjk->aBCIjk', H.ab.vvvv[va, Vb, Va, Vb], T.abb.VVVOoo, optimize=True)
    )
    dT.abb.vVVOoo += (1.0 / 4.0) * (
            -1.0 * np.einsum('aMIe,eCBMjk->aBCIjk', H.aa.voov[va, Oa, Oa, va], T.abb.vVVOoo, optimize=True)
            - 1.0 * np.einsum('aMIE,ECBMjk->aBCIjk', H.aa.voov[va, Oa, Oa, Va], T.abb.VVVOoo, optimize=True)
    )
    dT.abb.vVVOoo += (1.0 / 4.0) * (
            -1.0 * np.einsum('aMIe,CBejkM->aBCIjk', H.ab.voov[va, Ob, Oa, vb], T.bbb.VVvooO, optimize=True)
            - 1.0 * np.einsum('aMIE,CBEjkM->aBCIjk', H.ab.voov[va, Ob, Oa, Vb], T.bbb.VVVooO, optimize=True)
    )
    dT.abb.vVVOoo += (4.0 / 4.0) * (
            +1.0 * np.einsum('mBej,eaCmIk->aBCIjk', H.ab.ovvo[oa, Vb, va, ob], T.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('mBEj,EaCmIk->aBCIjk', H.ab.ovvo[oa, Vb, Va, ob], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('MBej,eaCIMk->aBCIjk', H.ab.ovvo[Oa, Vb, va, ob], T.aab.vvVOOo, optimize=True)
            - 1.0 * np.einsum('MBEj,EaCIMk->aBCIjk', H.ab.ovvo[Oa, Vb, Va, ob], T.aab.VvVOOo, optimize=True)
    )
    dT.abb.vVVOoo += (4.0 / 4.0) * (
            -1.0 * np.einsum('Bmje,aCeImk->aBCIjk', H.bb.voov[Vb, ob, ob, vb], T.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('BmjE,aCEImk->aBCIjk', H.bb.voov[Vb, ob, ob, Vb], T.abb.vVVOoo, optimize=True)
            + 1.0 * np.einsum('BMje,aCeIkM->aBCIjk', H.bb.voov[Vb, Ob, ob, vb], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('BMjE,aCEIkM->aBCIjk', H.bb.voov[Vb, Ob, ob, Vb], T.abb.vVVOoO, optimize=True)
    )
    dT.abb.vVVOoo += (2.0 / 4.0) * (
            +1.0 * np.einsum('MBIe,aCeMjk->aBCIjk', H.ab.ovov[Oa, Vb, Oa, vb], T.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('MBIE,aCEMjk->aBCIjk', H.ab.ovov[Oa, Vb, Oa, Vb], T.abb.vVVOoo, optimize=True)
    )
    dT.abb.vVVOoo += (2.0 / 4.0) * (
            +1.0 * np.einsum('amej,eCBImk->aBCIjk', H.ab.vovo[va, ob, va, ob], T.abb.vVVOoo, optimize=True)
            + 1.0 * np.einsum('amEj,ECBImk->aBCIjk', H.ab.vovo[va, ob, Va, ob], T.abb.VVVOoo, optimize=True)
            - 1.0 * np.einsum('aMej,eCBIkM->aBCIjk', H.ab.vovo[va, Ob, va, ob], T.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('aMEj,ECBIkM->aBCIjk', H.ab.vovo[va, Ob, Va, ob], T.abb.VVVOoO, optimize=True)
    )

    dT.abb.vVVOoo -= np.transpose(dT.abb.vVVOoo, (0, 2, 1, 3, 4, 5))
    dT.abb.vVVOoo -= np.transpose(dT.abb.vVVOoo, (0, 1, 2, 3, 5, 4))

    return dT


def update(T, dT, H, shift, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.abb.vVVOoo, dT.abb.vVVOoo = cc_active_loops.update_t3c_011100(
        T.abb.vVVOoo,
        dT.abb.vVVOoo,
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