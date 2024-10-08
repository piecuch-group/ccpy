import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops

def build(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.abb.vVvOoo = (1.0 / 2.0) * (
            +1.0 * np.einsum('aBIe,ecjk->aBcIjk', X.ab.vvov[va, Vb, Oa, :], T.bb[:, vb, ob, ob], optimize=True)
    )
    dR.abb.vVvOoo += (1.0 / 2.0) * (
            -1.0 * np.einsum('acIe,eBjk->aBcIjk', X.ab.vvov[va, vb, Oa, :], T.bb[:, Vb, ob, ob], optimize=True)
    )
    dR.abb.vVvOoo += (2.0 / 2.0) * (
            -1.0 * np.einsum('amIj,Bcmk->aBcIjk', X.ab.vooo[va, :, Oa, ob], T.bb[Vb, vb, :, ob], optimize=True)
    )
    dR.abb.vVvOoo += (2.0 / 2.0) * (
            +1.0 * np.einsum('cBke,aeIj->aBcIjk', X.bb.vvov[vb, Vb, ob, :], T.ab[va, :, Oa, ob], optimize=True)
    )
    dR.abb.vVvOoo += (1.0 / 2.0) * (
            -1.0 * np.einsum('cmkj,aBIm->aBcIjk', X.bb.vooo[vb, :, ob, ob], T.ab[va, Vb, Oa, :], optimize=True)
    )
    dR.abb.vVvOoo += (1.0 / 2.0) * (
            +1.0 * np.einsum('Bmkj,acIm->aBcIjk', X.bb.vooo[Vb, :, ob, ob], T.ab[va, vb, Oa, :], optimize=True)
    )
    dR.abb.vVvOoo += (2.0 / 2.0) * (
            +1.0 * np.einsum('aBej,ecIk->aBcIjk', X.ab.vvvo[va, Vb, :, ob], T.ab[:, vb, Oa, ob], optimize=True)
    )
    dR.abb.vVvOoo += (2.0 / 2.0) * (
            -1.0 * np.einsum('acej,eBIk->aBcIjk', X.ab.vvvo[va, vb, :, ob], T.ab[:, Vb, Oa, ob], optimize=True)
    )
    dR.abb.vVvOoo += (2.0 / 2.0) * (
            -1.0 * np.einsum('mBIj,acmk->aBcIjk', X.ab.ovoo[:, Vb, Oa, ob], T.ab[va, vb, :, ob], optimize=True)
    )
    dR.abb.vVvOoo += (2.0 / 2.0) * (
            +1.0 * np.einsum('mcIj,aBmk->aBcIjk', X.ab.ovoo[:, vb, Oa, ob], T.ab[va, Vb, :, ob], optimize=True)
    )
    dR.abb.vVvOoo += (1.0 / 2.0) * (
            +1.0 * np.einsum('aBIe,ecjk->aBcIjk', H.ab.vvov[va, Vb, Oa, :], R.bb[:, vb, ob, ob], optimize=True)
    )
    dR.abb.vVvOoo += (1.0 / 2.0) * (
            -1.0 * np.einsum('acIe,eBjk->aBcIjk', H.ab.vvov[va, vb, Oa, :], R.bb[:, Vb, ob, ob], optimize=True)
    )
    dR.abb.vVvOoo += (2.0 / 2.0) * (
            -1.0 * np.einsum('amIj,Bcmk->aBcIjk', H.ab.vooo[va, :, Oa, ob], R.bb[Vb, vb, :, ob], optimize=True)
    )
    dR.abb.vVvOoo += (2.0 / 2.0) * (
            +1.0 * np.einsum('cBke,aeIj->aBcIjk', H.bb.vvov[vb, Vb, ob, :], R.ab[va, :, Oa, ob], optimize=True)
    )
    dR.abb.vVvOoo += (1.0 / 2.0) * (
            -1.0 * np.einsum('cmkj,aBIm->aBcIjk', H.bb.vooo[vb, :, ob, ob], R.ab[va, Vb, Oa, :], optimize=True)
    )
    dR.abb.vVvOoo += (1.0 / 2.0) * (
            +1.0 * np.einsum('Bmkj,acIm->aBcIjk', H.bb.vooo[Vb, :, ob, ob], R.ab[va, vb, Oa, :], optimize=True)
    )
    dR.abb.vVvOoo += (2.0 / 2.0) * (
            +1.0 * np.einsum('aBej,ecIk->aBcIjk', H.ab.vvvo[va, Vb, :, ob], R.ab[:, vb, Oa, ob], optimize=True)
    )
    dR.abb.vVvOoo += (2.0 / 2.0) * (
            -1.0 * np.einsum('acej,eBIk->aBcIjk', H.ab.vvvo[va, vb, :, ob], R.ab[:, Vb, Oa, ob], optimize=True)
    )
    dR.abb.vVvOoo += (2.0 / 2.0) * (
            -1.0 * np.einsum('mBIj,acmk->aBcIjk', H.ab.ovoo[:, Vb, Oa, ob], R.ab[va, vb, :, ob], optimize=True)
    )
    dR.abb.vVvOoo += (2.0 / 2.0) * (
            +1.0 * np.einsum('mcIj,aBmk->aBcIjk', H.ab.ovoo[:, vb, Oa, ob], R.ab[va, Vb, :, ob], optimize=True)
    )
    # of terms =  20
    dR.abb.vVvOoo += (1.0 / 2.0) * (
            -1.0 * np.einsum('MI,aBcMjk->aBcIjk', X.a.oo[Oa, Oa], T.abb.vVvOoo, optimize=True)
    )
    dR.abb.vVvOoo += (2.0 / 2.0) * (
            -1.0 * np.einsum('mj,aBcImk->aBcIjk', X.b.oo[ob, ob], T.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('Mj,aBcIkM->aBcIjk', X.b.oo[Ob, ob], T.abb.vVvOoO, optimize=True)
    )
    dR.abb.vVvOoo += (1.0 / 2.0) * (
            +1.0 * np.einsum('ae,eBcIjk->aBcIjk', X.a.vv[va, va], T.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('aE,EBcIjk->aBcIjk', X.a.vv[va, Va], T.abb.VVvOoo, optimize=True)
    )
    dR.abb.vVvOoo += (1.0 / 2.0) * (
            +1.0 * np.einsum('BE,aEcIjk->aBcIjk', X.b.vv[Vb, Vb], T.abb.vVvOoo, optimize=True)
    )
    dR.abb.vVvOoo += (1.0 / 2.0) * (
            +1.0 * np.einsum('ce,aBeIjk->aBcIjk', X.b.vv[vb, vb], T.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('cE,aBEIjk->aBcIjk', X.b.vv[vb, Vb], T.abb.vVVOoo, optimize=True)
    )
    dR.abb.vVvOoo += (1.0 / 2.0) * (
            +0.5 * np.einsum('mnjk,aBcImn->aBcIjk', X.bb.oooo[ob, ob, ob, ob], T.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('Mnjk,aBcInM->aBcIjk', X.bb.oooo[Ob, ob, ob, ob], T.abb.vVvOoO, optimize=True)
            + 0.5 * np.einsum('MNjk,aBcIMN->aBcIjk', X.bb.oooo[Ob, Ob, ob, ob], T.abb.vVvOOO, optimize=True)
    )
    dR.abb.vVvOoo += (2.0 / 2.0) * (
            -1.0 * np.einsum('mNIj,aBcmkN->aBcIjk', X.ab.oooo[oa, Ob, Oa, ob], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('MnIj,aBcMnk->aBcIjk', X.ab.oooo[Oa, ob, Oa, ob], T.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('MNIj,aBcMkN->aBcIjk', X.ab.oooo[Oa, Ob, Oa, ob], T.abb.vVvOoO, optimize=True)
    )
    dR.abb.vVvOoo += (1.0 / 2.0) * (
            -1.0 * np.einsum('BceF,aFeIjk->aBcIjk', X.bb.vvvv[Vb, vb, vb, Vb], T.abb.vVvOoo, optimize=True)
            - 0.5 * np.einsum('BcEF,aFEIjk->aBcIjk', X.bb.vvvv[Vb, vb, Vb, Vb], T.abb.vVVOoo, optimize=True)
    )
    dR.abb.vVvOoo += (1.0 / 2.0) * (
            -1.0 * np.einsum('aBEf,EcfIjk->aBcIjk', X.ab.vvvv[va, Vb, Va, vb], T.abb.VvvOoo, optimize=True)
            + 1.0 * np.einsum('aBeF,eFcIjk->aBcIjk', X.ab.vvvv[va, Vb, va, Vb], T.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('aBEF,EFcIjk->aBcIjk', X.ab.vvvv[va, Vb, Va, Vb], T.abb.VVvOoo, optimize=True)
    )
    dR.abb.vVvOoo += (1.0 / 2.0) * (
            +1.0 * np.einsum('acef,eBfIjk->aBcIjk', X.ab.vvvv[va, vb, va, vb], T.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('acEf,EBfIjk->aBcIjk', X.ab.vvvv[va, vb, Va, vb], T.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('aceF,eBFIjk->aBcIjk', X.ab.vvvv[va, vb, va, Vb], T.abb.vVVOoo, optimize=True)
            + 1.0 * np.einsum('acEF,EBFIjk->aBcIjk', X.ab.vvvv[va, vb, Va, Vb], T.abb.VVVOoo, optimize=True)
    )
    dR.abb.vVvOoo += (1.0 / 2.0) * (
            +1.0 * np.einsum('aMIe,eBcMjk->aBcIjk', X.aa.voov[va, Oa, Oa, va], T.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('aMIE,EBcMjk->aBcIjk', X.aa.voov[va, Oa, Oa, Va], T.abb.VVvOoo, optimize=True)
    )
    dR.abb.vVvOoo += (1.0 / 2.0) * (
            +1.0 * np.einsum('aMIe,BcejkM->aBcIjk', X.ab.voov[va, Ob, Oa, vb], T.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('aMIE,BEcjkM->aBcIjk', X.ab.voov[va, Ob, Oa, Vb], T.bbb.VVvooO, optimize=True)
    )
    dR.abb.vVvOoo += (2.0 / 2.0) * (
            +1.0 * np.einsum('mBEj,EacmIk->aBcIjk', X.ab.ovvo[oa, Vb, Va, ob], T.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('MBEj,EacIMk->aBcIjk', X.ab.ovvo[Oa, Vb, Va, ob], T.aab.VvvOOo, optimize=True)
    )
    dR.abb.vVvOoo += (2.0 / 2.0) * (
            -1.0 * np.einsum('mcej,eaBmIk->aBcIjk', X.ab.ovvo[oa, vb, va, ob], T.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('mcEj,EaBmIk->aBcIjk', X.ab.ovvo[oa, vb, Va, ob], T.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('Mcej,eaBIMk->aBcIjk', X.ab.ovvo[Oa, vb, va, ob], T.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('McEj,EaBIMk->aBcIjk', X.ab.ovvo[Oa, vb, Va, ob], T.aab.VvVOOo, optimize=True)
    )
    dR.abb.vVvOoo += (2.0 / 2.0) * (
            +1.0 * np.einsum('BmjE,aEcImk->aBcIjk', X.bb.voov[Vb, ob, ob, Vb], T.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('BMjE,aEcIkM->aBcIjk', X.bb.voov[Vb, Ob, ob, Vb], T.abb.vVvOoO, optimize=True)
    )
    dR.abb.vVvOoo += (2.0 / 2.0) * (
            +1.0 * np.einsum('cmje,aBeImk->aBcIjk', X.bb.voov[vb, ob, ob, vb], T.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('cmjE,aBEImk->aBcIjk', X.bb.voov[vb, ob, ob, Vb], T.abb.vVVOoo, optimize=True)
            - 1.0 * np.einsum('cMje,aBeIkM->aBcIjk', X.bb.voov[vb, Ob, ob, vb], T.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('cMjE,aBEIkM->aBcIjk', X.bb.voov[vb, Ob, ob, Vb], T.abb.vVVOoO, optimize=True)
    )
    dR.abb.vVvOoo += (1.0 / 2.0) * (
            -1.0 * np.einsum('MBIE,aEcMjk->aBcIjk', X.ab.ovov[Oa, Vb, Oa, Vb], T.abb.vVvOoo, optimize=True)
    )
    dR.abb.vVvOoo += (1.0 / 2.0) * (
            -1.0 * np.einsum('McIe,aBeMjk->aBcIjk', X.ab.ovov[Oa, vb, Oa, vb], T.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('McIE,aBEMjk->aBcIjk', X.ab.ovov[Oa, vb, Oa, Vb], T.abb.vVVOoo, optimize=True)
    )
    dR.abb.vVvOoo += (2.0 / 2.0) * (
            -1.0 * np.einsum('amej,eBcImk->aBcIjk', X.ab.vovo[va, ob, va, ob], T.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('amEj,EBcImk->aBcIjk', X.ab.vovo[va, ob, Va, ob], T.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('aMej,eBcIkM->aBcIjk', X.ab.vovo[va, Ob, va, ob], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('aMEj,EBcIkM->aBcIjk', X.ab.vovo[va, Ob, Va, ob], T.abb.VVvOoO, optimize=True)
    )
    dR.abb.vVvOoo += (1.0 / 2.0) * (
            -1.0 * np.einsum('MI,aBcMjk->aBcIjk', H.a.oo[Oa, Oa], R.abb.vVvOoo, optimize=True)
    )
    dR.abb.vVvOoo += (2.0 / 2.0) * (
            -1.0 * np.einsum('mj,aBcImk->aBcIjk', H.b.oo[ob, ob], R.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('Mj,aBcIkM->aBcIjk', H.b.oo[Ob, ob], R.abb.vVvOoO, optimize=True)
    )
    dR.abb.vVvOoo += (1.0 / 2.0) * (
            +1.0 * np.einsum('ae,eBcIjk->aBcIjk', H.a.vv[va, va], R.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('aE,EBcIjk->aBcIjk', H.a.vv[va, Va], R.abb.VVvOoo, optimize=True)
    )
    dR.abb.vVvOoo += (1.0 / 2.0) * (
            +1.0 * np.einsum('BE,aEcIjk->aBcIjk', H.b.vv[Vb, Vb], R.abb.vVvOoo, optimize=True)
    )
    dR.abb.vVvOoo += (1.0 / 2.0) * (
            +1.0 * np.einsum('ce,aBeIjk->aBcIjk', H.b.vv[vb, vb], R.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('cE,aBEIjk->aBcIjk', H.b.vv[vb, Vb], R.abb.vVVOoo, optimize=True)
    )
    dR.abb.vVvOoo += (1.0 / 2.0) * (
            +0.5 * np.einsum('mnjk,aBcImn->aBcIjk', H.bb.oooo[ob, ob, ob, ob], R.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('Mnjk,aBcInM->aBcIjk', H.bb.oooo[Ob, ob, ob, ob], R.abb.vVvOoO, optimize=True)
            + 0.5 * np.einsum('MNjk,aBcIMN->aBcIjk', H.bb.oooo[Ob, Ob, ob, ob], R.abb.vVvOOO, optimize=True)
    )
    dR.abb.vVvOoo += (2.0 / 2.0) * (
            -1.0 * np.einsum('mNIj,aBcmkN->aBcIjk', H.ab.oooo[oa, Ob, Oa, ob], R.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('MnIj,aBcMnk->aBcIjk', H.ab.oooo[Oa, ob, Oa, ob], R.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('MNIj,aBcMkN->aBcIjk', H.ab.oooo[Oa, Ob, Oa, ob], R.abb.vVvOoO, optimize=True)
    )
    dR.abb.vVvOoo += (1.0 / 2.0) * (
            -1.0 * np.einsum('BceF,aFeIjk->aBcIjk', H.bb.vvvv[Vb, vb, vb, Vb], R.abb.vVvOoo, optimize=True)
            - 0.5 * np.einsum('BcEF,aFEIjk->aBcIjk', H.bb.vvvv[Vb, vb, Vb, Vb], R.abb.vVVOoo, optimize=True)
    )
    dR.abb.vVvOoo += (1.0 / 2.0) * (
            -1.0 * np.einsum('aBEf,EcfIjk->aBcIjk', H.ab.vvvv[va, Vb, Va, vb], R.abb.VvvOoo, optimize=True)
            + 1.0 * np.einsum('aBeF,eFcIjk->aBcIjk', H.ab.vvvv[va, Vb, va, Vb], R.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('aBEF,EFcIjk->aBcIjk', H.ab.vvvv[va, Vb, Va, Vb], R.abb.VVvOoo, optimize=True)
    )
    dR.abb.vVvOoo += (1.0 / 2.0) * (
            +1.0 * np.einsum('acef,eBfIjk->aBcIjk', H.ab.vvvv[va, vb, va, vb], R.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('acEf,EBfIjk->aBcIjk', H.ab.vvvv[va, vb, Va, vb], R.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('aceF,eBFIjk->aBcIjk', H.ab.vvvv[va, vb, va, Vb], R.abb.vVVOoo, optimize=True)
            + 1.0 * np.einsum('acEF,EBFIjk->aBcIjk', H.ab.vvvv[va, vb, Va, Vb], R.abb.VVVOoo, optimize=True)
    )
    dR.abb.vVvOoo += (1.0 / 2.0) * (
            +1.0 * np.einsum('aMIe,eBcMjk->aBcIjk', H.aa.voov[va, Oa, Oa, va], R.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('aMIE,EBcMjk->aBcIjk', H.aa.voov[va, Oa, Oa, Va], R.abb.VVvOoo, optimize=True)
    )
    dR.abb.vVvOoo += (1.0 / 2.0) * (
            +1.0 * np.einsum('aMIe,BcejkM->aBcIjk', H.ab.voov[va, Ob, Oa, vb], R.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('aMIE,BEcjkM->aBcIjk', H.ab.voov[va, Ob, Oa, Vb], R.bbb.VVvooO, optimize=True)
    )
    dR.abb.vVvOoo += (2.0 / 2.0) * (
            +1.0 * np.einsum('mBEj,EacmIk->aBcIjk', H.ab.ovvo[oa, Vb, Va, ob], R.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('MBEj,EacIMk->aBcIjk', H.ab.ovvo[Oa, Vb, Va, ob], R.aab.VvvOOo, optimize=True)
    )
    dR.abb.vVvOoo += (2.0 / 2.0) * (
            -1.0 * np.einsum('mcej,eaBmIk->aBcIjk', H.ab.ovvo[oa, vb, va, ob], R.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('mcEj,EaBmIk->aBcIjk', H.ab.ovvo[oa, vb, Va, ob], R.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('Mcej,eaBIMk->aBcIjk', H.ab.ovvo[Oa, vb, va, ob], R.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('McEj,EaBIMk->aBcIjk', H.ab.ovvo[Oa, vb, Va, ob], R.aab.VvVOOo, optimize=True)
    )
    dR.abb.vVvOoo += (2.0 / 2.0) * (
            +1.0 * np.einsum('BmjE,aEcImk->aBcIjk', H.bb.voov[Vb, ob, ob, Vb], R.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('BMjE,aEcIkM->aBcIjk', H.bb.voov[Vb, Ob, ob, Vb], R.abb.vVvOoO, optimize=True)
    )
    dR.abb.vVvOoo += (2.0 / 2.0) * (
            +1.0 * np.einsum('cmje,aBeImk->aBcIjk', H.bb.voov[vb, ob, ob, vb], R.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('cmjE,aBEImk->aBcIjk', H.bb.voov[vb, ob, ob, Vb], R.abb.vVVOoo, optimize=True)
            - 1.0 * np.einsum('cMje,aBeIkM->aBcIjk', H.bb.voov[vb, Ob, ob, vb], R.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('cMjE,aBEIkM->aBcIjk', H.bb.voov[vb, Ob, ob, Vb], R.abb.vVVOoO, optimize=True)
    )
    dR.abb.vVvOoo += (1.0 / 2.0) * (
            -1.0 * np.einsum('MBIE,aEcMjk->aBcIjk', H.ab.ovov[Oa, Vb, Oa, Vb], R.abb.vVvOoo, optimize=True)
    )
    dR.abb.vVvOoo += (1.0 / 2.0) * (
            -1.0 * np.einsum('McIe,aBeMjk->aBcIjk', H.ab.ovov[Oa, vb, Oa, vb], R.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('McIE,aBEMjk->aBcIjk', H.ab.ovov[Oa, vb, Oa, Vb], R.abb.vVVOoo, optimize=True)
    )
    dR.abb.vVvOoo += (2.0 / 2.0) * (
            -1.0 * np.einsum('amej,eBcImk->aBcIjk', H.ab.vovo[va, ob, va, ob], R.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('amEj,EBcImk->aBcIjk', H.ab.vovo[va, ob, Va, ob], R.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('aMej,eBcIkM->aBcIjk', H.ab.vovo[va, Ob, va, ob], R.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('aMEj,EBcIkM->aBcIjk', H.ab.vovo[va, Ob, Va, ob], R.abb.VVvOoO, optimize=True)
    )
    # of terms =  38

    dR.abb.vVvOoo -= np.transpose(dR.abb.vVvOoo, (0, 1, 2, 3, 5, 4))

    return dR

def update(R, omega, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    R.abb.vVvOoo = eomcc_active_loops.update_r3c_010100(
        R.abb.vVvOoo,
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
