import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import cc_active_loops

def build(T, dT, H, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    # MM(2,3)
    dT.abb.vVvOoo = (1.0 / 2.0) * (
            +1.0 * np.einsum('aBIe,ecjk->aBcIjk', H.ab.vvov[va, Vb, Oa, :], T.bb[:, vb, ob, ob], optimize=True)
    )
    dT.abb.vVvOoo += (1.0 / 2.0) * (
            -1.0 * np.einsum('acIe,eBjk->aBcIjk', H.ab.vvov[va, vb, Oa, :], T.bb[:, Vb, ob, ob], optimize=True)
    )
    dT.abb.vVvOoo += (2.0 / 2.0) * (
            -1.0 * np.einsum('amIj,Bcmk->aBcIjk', H.ab.vooo[va, :, Oa, ob], T.bb[Vb, vb, :, ob], optimize=True)
    )
    dT.abb.vVvOoo += (2.0 / 2.0) * (
            +1.0 * np.einsum('cBke,aeIj->aBcIjk', H.bb.vvov[vb, Vb, ob, :], T.ab[va, :, Oa, ob], optimize=True)
    )
    dT.abb.vVvOoo += (1.0 / 2.0) * (
            -1.0 * np.einsum('cmkj,aBIm->aBcIjk', H.bb.vooo[vb, :, ob, ob], T.ab[va, Vb, Oa, :], optimize=True)
    )
    dT.abb.vVvOoo += (1.0 / 2.0) * (
            +1.0 * np.einsum('Bmkj,acIm->aBcIjk', H.bb.vooo[Vb, :, ob, ob], T.ab[va, vb, Oa, :], optimize=True)
    )
    dT.abb.vVvOoo += (2.0 / 2.0) * (
            +1.0 * np.einsum('aBej,ecIk->aBcIjk', H.ab.vvvo[va, Vb, :, ob], T.ab[:, vb, Oa, ob], optimize=True)
    )
    dT.abb.vVvOoo += (2.0 / 2.0) * (
            -1.0 * np.einsum('acej,eBIk->aBcIjk', H.ab.vvvo[va, vb, :, ob], T.ab[:, Vb, Oa, ob], optimize=True)
    )
    dT.abb.vVvOoo += (2.0 / 2.0) * (
            -1.0 * np.einsum('mBIj,acmk->aBcIjk', H.ab.ovoo[:, Vb, Oa, ob], T.ab[va, vb, :, ob], optimize=True)
    )
    dT.abb.vVvOoo += (2.0 / 2.0) * (
            +1.0 * np.einsum('mcIj,aBmk->aBcIjk', H.ab.ovoo[:, vb, Oa, ob], T.ab[va, Vb, :, ob], optimize=True)
    )
    # (H(2) * T3)_C
    dT.abb.vVvOoo += (1.0 / 2.0) * (
            -1.0 * np.einsum('MI,aBcMjk->aBcIjk', H.a.oo[Oa, Oa], T.abb.vVvOoo, optimize=True)
    )
    dT.abb.vVvOoo += (2.0 / 2.0) * (
            -1.0 * np.einsum('mj,aBcImk->aBcIjk', H.b.oo[ob, ob], T.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('Mj,aBcIkM->aBcIjk', H.b.oo[Ob, ob], T.abb.vVvOoO, optimize=True)
    )
    dT.abb.vVvOoo += (1.0 / 2.0) * (
            +1.0 * np.einsum('ae,eBcIjk->aBcIjk', H.a.vv[va, va], T.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('aE,EBcIjk->aBcIjk', H.a.vv[va, Va], T.abb.VVvOoo, optimize=True)
    )
    dT.abb.vVvOoo += (1.0 / 2.0) * (
            +1.0 * np.einsum('BE,aEcIjk->aBcIjk', H.b.vv[Vb, Vb], T.abb.vVvOoo, optimize=True)
    )
    dT.abb.vVvOoo += (1.0 / 2.0) * (
            +1.0 * np.einsum('ce,aBeIjk->aBcIjk', H.b.vv[vb, vb], T.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('cE,aBEIjk->aBcIjk', H.b.vv[vb, Vb], T.abb.vVVOoo, optimize=True)
    )
    dT.abb.vVvOoo += (1.0 / 2.0) * (
            +0.5 * np.einsum('mnjk,aBcImn->aBcIjk', H.bb.oooo[ob, ob, ob, ob], T.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('mNjk,aBcImN->aBcIjk', H.bb.oooo[ob, Ob, ob, ob], T.abb.vVvOoO, optimize=True)
            + 0.5 * np.einsum('MNjk,aBcIMN->aBcIjk', H.bb.oooo[Ob, Ob, ob, ob], T.abb.vVvOOO, optimize=True)
    )
    dT.abb.vVvOoo += (2.0 / 2.0) * (
            +1.0 * np.einsum('MnIj,aBcMnk->aBcIjk', H.ab.oooo[Oa, ob, Oa, ob], T.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('mNIj,aBcmkN->aBcIjk', H.ab.oooo[oa, Ob, Oa, ob], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('MNIj,aBcMkN->aBcIjk', H.ab.oooo[Oa, Ob, Oa, ob], T.abb.vVvOoO, optimize=True)
    )
    dT.abb.vVvOoo += (1.0 / 2.0) * (
            -1.0 * np.einsum('BceF,aFeIjk->aBcIjk', H.bb.vvvv[Vb, vb, vb, Vb], T.abb.vVvOoo, optimize=True)
            - 0.5 * np.einsum('BcEF,aFEIjk->aBcIjk', H.bb.vvvv[Vb, vb, Vb, Vb], T.abb.vVVOoo, optimize=True)
    )
    dT.abb.vVvOoo += (1.0 / 2.0) * (
            -1.0 * np.einsum('aBEf,EcfIjk->aBcIjk', H.ab.vvvv[va, Vb, Va, vb], T.abb.VvvOoo, optimize=True)
            + 1.0 * np.einsum('aBeF,eFcIjk->aBcIjk', H.ab.vvvv[va, Vb, va, Vb], T.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('aBEF,EFcIjk->aBcIjk', H.ab.vvvv[va, Vb, Va, Vb], T.abb.VVvOoo, optimize=True)
    )
    dT.abb.vVvOoo += (1.0 / 2.0) * (
            +1.0 * np.einsum('acef,eBfIjk->aBcIjk', H.ab.vvvv[va, vb, va, vb], T.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('acEf,EBfIjk->aBcIjk', H.ab.vvvv[va, vb, Va, vb], T.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('aceF,eBFIjk->aBcIjk', H.ab.vvvv[va, vb, va, Vb], T.abb.vVVOoo, optimize=True)
            + 1.0 * np.einsum('acEF,EBFIjk->aBcIjk', H.ab.vvvv[va, vb, Va, Vb], T.abb.VVVOoo, optimize=True)
    )
    dT.abb.vVvOoo += (1.0 / 2.0) * (
            +1.0 * np.einsum('aMIe,eBcMjk->aBcIjk', H.aa.voov[va, Oa, Oa, va], T.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('aMIE,EBcMjk->aBcIjk', H.aa.voov[va, Oa, Oa, Va], T.abb.VVvOoo, optimize=True)
    )
    dT.abb.vVvOoo += (1.0 / 2.0) * (
            +1.0 * np.einsum('aMIe,BcejkM->aBcIjk', H.ab.voov[va, Ob, Oa, vb], T.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('aMIE,BEcjkM->aBcIjk', H.ab.voov[va, Ob, Oa, Vb], T.bbb.VVvooO, optimize=True)
    )
    dT.abb.vVvOoo += (2.0 / 2.0) * (
            +1.0 * np.einsum('mBEj,EacmIk->aBcIjk', H.ab.ovvo[oa, Vb, Va, ob], T.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('MBEj,EacIMk->aBcIjk', H.ab.ovvo[Oa, Vb, Va, ob], T.aab.VvvOOo, optimize=True)
    )
    dT.abb.vVvOoo += (2.0 / 2.0) * (
            -1.0 * np.einsum('mcej,eaBmIk->aBcIjk', H.ab.ovvo[oa, vb, va, ob], T.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('mcEj,EaBmIk->aBcIjk', H.ab.ovvo[oa, vb, Va, ob], T.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('Mcej,eaBIMk->aBcIjk', H.ab.ovvo[Oa, vb, va, ob], T.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('McEj,EaBIMk->aBcIjk', H.ab.ovvo[Oa, vb, Va, ob], T.aab.VvVOOo, optimize=True)
    )
    dT.abb.vVvOoo += (2.0 / 2.0) * (
            +1.0 * np.einsum('BmjE,aEcImk->aBcIjk', H.bb.voov[Vb, ob, ob, Vb], T.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('BMjE,aEcIkM->aBcIjk', H.bb.voov[Vb, Ob, ob, Vb], T.abb.vVvOoO, optimize=True)
    )
    dT.abb.vVvOoo += (2.0 / 2.0) * (
            +1.0 * np.einsum('cmje,aBeImk->aBcIjk', H.bb.voov[vb, ob, ob, vb], T.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('cmjE,aBEImk->aBcIjk', H.bb.voov[vb, ob, ob, Vb], T.abb.vVVOoo, optimize=True)
            - 1.0 * np.einsum('cMje,aBeIkM->aBcIjk', H.bb.voov[vb, Ob, ob, vb], T.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('cMjE,aBEIkM->aBcIjk', H.bb.voov[vb, Ob, ob, Vb], T.abb.vVVOoO, optimize=True)
    )
    dT.abb.vVvOoo += (1.0 / 2.0) * (
            -1.0 * np.einsum('MBIE,aEcMjk->aBcIjk', H.ab.ovov[Oa, Vb, Oa, Vb], T.abb.vVvOoo, optimize=True)
    )
    dT.abb.vVvOoo += (1.0 / 2.0) * (
            -1.0 * np.einsum('McIe,aBeMjk->aBcIjk', H.ab.ovov[Oa, vb, Oa, vb], T.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('McIE,aBEMjk->aBcIjk', H.ab.ovov[Oa, vb, Oa, Vb], T.abb.vVVOoo, optimize=True)
    )
    dT.abb.vVvOoo += (2.0 / 2.0) * (
            -1.0 * np.einsum('amej,eBcImk->aBcIjk', H.ab.vovo[va, ob, va, ob], T.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('amEj,EBcImk->aBcIjk', H.ab.vovo[va, ob, Va, ob], T.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('aMej,eBcIkM->aBcIjk', H.ab.vovo[va, Ob, va, ob], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('aMEj,EBcIkM->aBcIjk', H.ab.vovo[va, Ob, Va, ob], T.abb.VVvOoO, optimize=True)
    )

    dT.abb.vVvOoo -= np.transpose(dT.abb.vVvOoo, (0, 1, 2, 3, 5, 4))

    return dT


def update(T, dT, H, shift, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.abb.vVvOoo, dT.abb.vVvOoo = cc_active_loops.update_t3c_010100(
        T.abb.vVvOoo,
        dT.abb.vVvOoo,
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