import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops

def build(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.abb.VVvOoo = (1.0 / 2.0) * (
            +1.0 * np.einsum('ABIe,ecjk->ABcIjk', X.ab.vvov[Va, Vb, Oa, :], T.bb[:, vb, ob, ob], optimize=True)
    )
    dR.abb.VVvOoo += (1.0 / 2.0) * (
            -1.0 * np.einsum('AcIe,eBjk->ABcIjk', X.ab.vvov[Va, vb, Oa, :], T.bb[:, Vb, ob, ob], optimize=True)
    )
    dR.abb.VVvOoo += (2.0 / 2.0) * (
            -1.0 * np.einsum('AmIj,Bcmk->ABcIjk', X.ab.vooo[Va, :, Oa, ob], T.bb[Vb, vb, :, ob], optimize=True)
    )
    dR.abb.VVvOoo += (2.0 / 2.0) * (
            +1.0 * np.einsum('cBke,AeIj->ABcIjk', X.bb.vvov[vb, Vb, ob, :], T.ab[Va, :, Oa, ob], optimize=True)
    )
    dR.abb.VVvOoo += (1.0 / 2.0) * (
            -1.0 * np.einsum('cmkj,ABIm->ABcIjk', X.bb.vooo[vb, :, ob, ob], T.ab[Va, Vb, Oa, :], optimize=True)
    )
    dR.abb.VVvOoo += (1.0 / 2.0) * (
            +1.0 * np.einsum('Bmkj,AcIm->ABcIjk', X.bb.vooo[Vb, :, ob, ob], T.ab[Va, vb, Oa, :], optimize=True)
    )
    dR.abb.VVvOoo += (2.0 / 2.0) * (
            +1.0 * np.einsum('ABej,ecIk->ABcIjk', X.ab.vvvo[Va, Vb, :, ob], T.ab[:, vb, Oa, ob], optimize=True)
    )
    dR.abb.VVvOoo += (2.0 / 2.0) * (
            -1.0 * np.einsum('Acej,eBIk->ABcIjk', X.ab.vvvo[Va, vb, :, ob], T.ab[:, Vb, Oa, ob], optimize=True)
    )
    dR.abb.VVvOoo += (2.0 / 2.0) * (
            -1.0 * np.einsum('mBIj,Acmk->ABcIjk', X.ab.ovoo[:, Vb, Oa, ob], T.ab[Va, vb, :, ob], optimize=True)
    )
    dR.abb.VVvOoo += (2.0 / 2.0) * (
            +1.0 * np.einsum('mcIj,ABmk->ABcIjk', X.ab.ovoo[:, vb, Oa, ob], T.ab[Va, Vb, :, ob], optimize=True)
    )
    dR.abb.VVvOoo += (1.0 / 2.0) * (
            +1.0 * np.einsum('ABIe,ecjk->ABcIjk', H.ab.vvov[Va, Vb, Oa, :], R.bb[:, vb, ob, ob], optimize=True)
    )
    dR.abb.VVvOoo += (1.0 / 2.0) * (
            -1.0 * np.einsum('AcIe,eBjk->ABcIjk', H.ab.vvov[Va, vb, Oa, :], R.bb[:, Vb, ob, ob], optimize=True)
    )
    dR.abb.VVvOoo += (2.0 / 2.0) * (
            -1.0 * np.einsum('AmIj,Bcmk->ABcIjk', H.ab.vooo[Va, :, Oa, ob], R.bb[Vb, vb, :, ob], optimize=True)
    )
    dR.abb.VVvOoo += (2.0 / 2.0) * (
            +1.0 * np.einsum('cBke,AeIj->ABcIjk', H.bb.vvov[vb, Vb, ob, :], R.ab[Va, :, Oa, ob], optimize=True)
    )
    dR.abb.VVvOoo += (1.0 / 2.0) * (
            -1.0 * np.einsum('cmkj,ABIm->ABcIjk', H.bb.vooo[vb, :, ob, ob], R.ab[Va, Vb, Oa, :], optimize=True)
    )
    dR.abb.VVvOoo += (1.0 / 2.0) * (
            +1.0 * np.einsum('Bmkj,AcIm->ABcIjk', H.bb.vooo[Vb, :, ob, ob], R.ab[Va, vb, Oa, :], optimize=True)
    )
    dR.abb.VVvOoo += (2.0 / 2.0) * (
            +1.0 * np.einsum('ABej,ecIk->ABcIjk', H.ab.vvvo[Va, Vb, :, ob], R.ab[:, vb, Oa, ob], optimize=True)
    )
    dR.abb.VVvOoo += (2.0 / 2.0) * (
            -1.0 * np.einsum('Acej,eBIk->ABcIjk', H.ab.vvvo[Va, vb, :, ob], R.ab[:, Vb, Oa, ob], optimize=True)
    )
    dR.abb.VVvOoo += (2.0 / 2.0) * (
            -1.0 * np.einsum('mBIj,Acmk->ABcIjk', H.ab.ovoo[:, Vb, Oa, ob], R.ab[Va, vb, :, ob], optimize=True)
    )
    dR.abb.VVvOoo += (2.0 / 2.0) * (
            +1.0 * np.einsum('mcIj,ABmk->ABcIjk', H.ab.ovoo[:, vb, Oa, ob], R.ab[Va, Vb, :, ob], optimize=True)
    )
    # of terms =  20
    dR.abb.VVvOoo += (1.0 / 2.0) * (
            -1.0 * np.einsum('MI,ABcMjk->ABcIjk', X.a.oo[Oa, Oa], T.abb.VVvOoo, optimize=True)
    )
    dR.abb.VVvOoo += (2.0 / 2.0) * (
            -1.0 * np.einsum('mj,ABcImk->ABcIjk', X.b.oo[ob, ob], T.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('Mj,ABcIkM->ABcIjk', X.b.oo[Ob, ob], T.abb.VVvOoO, optimize=True)
    )
    dR.abb.VVvOoo += (1.0 / 2.0) * (
            +1.0 * np.einsum('Ae,eBcIjk->ABcIjk', X.a.vv[Va, va], T.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('AE,EBcIjk->ABcIjk', X.a.vv[Va, Va], T.abb.VVvOoo, optimize=True)
    )
    dR.abb.VVvOoo += (1.0 / 2.0) * (
            -1.0 * np.einsum('Be,AceIjk->ABcIjk', X.b.vv[Vb, vb], T.abb.VvvOoo, optimize=True)
            + 1.0 * np.einsum('BE,AEcIjk->ABcIjk', X.b.vv[Vb, Vb], T.abb.VVvOoo, optimize=True)
    )
    dR.abb.VVvOoo += (1.0 / 2.0) * (
            +1.0 * np.einsum('ce,ABeIjk->ABcIjk', X.b.vv[vb, vb], T.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('cE,ABEIjk->ABcIjk', X.b.vv[vb, Vb], T.abb.VVVOoo, optimize=True)
    )
    dR.abb.VVvOoo += (1.0 / 2.0) * (
            +0.5 * np.einsum('mnjk,ABcImn->ABcIjk', X.bb.oooo[ob, ob, ob, ob], T.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('mNjk,ABcImN->ABcIjk', X.bb.oooo[ob, Ob, ob, ob], T.abb.VVvOoO, optimize=True)
            + 0.5 * np.einsum('MNjk,ABcIMN->ABcIjk', X.bb.oooo[Ob, Ob, ob, ob], T.abb.VVvOOO, optimize=True)
    )
    dR.abb.VVvOoo += (2.0 / 2.0) * (
            +1.0 * np.einsum('MnIj,ABcMnk->ABcIjk', X.ab.oooo[Oa, ob, Oa, ob], T.abb.VVvOoo, optimize=True)
            - 1.0 * np.einsum('mNIj,ABcmkN->ABcIjk', X.ab.oooo[oa, Ob, Oa, ob], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('MNIj,ABcMkN->ABcIjk', X.ab.oooo[Oa, Ob, Oa, ob], T.abb.VVvOoO, optimize=True)
    )
    dR.abb.VVvOoo += (1.0 / 2.0) * (
            -0.5 * np.einsum('Bcef,AfeIjk->ABcIjk', X.bb.vvvv[Vb, vb, vb, vb], T.abb.VvvOoo, optimize=True)
            + 1.0 * np.einsum('BcEf,AEfIjk->ABcIjk', X.bb.vvvv[Vb, vb, Vb, vb], T.abb.VVvOoo, optimize=True)
            - 0.5 * np.einsum('BcEF,AFEIjk->ABcIjk', X.bb.vvvv[Vb, vb, Vb, Vb], T.abb.VVVOoo, optimize=True)
    )
    dR.abb.VVvOoo += (1.0 / 2.0) * (
            +1.0 * np.einsum('ABeF,eFcIjk->ABcIjk', X.ab.vvvv[Va, Vb, va, Vb], T.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('ABEf,EcfIjk->ABcIjk', X.ab.vvvv[Va, Vb, Va, vb], T.abb.VvvOoo, optimize=True)
            + 1.0 * np.einsum('ABEF,EFcIjk->ABcIjk', X.ab.vvvv[Va, Vb, Va, Vb], T.abb.VVvOoo, optimize=True)
    )
    dR.abb.VVvOoo += (1.0 / 2.0) * (
            +1.0 * np.einsum('Acef,eBfIjk->ABcIjk', X.ab.vvvv[Va, vb, va, vb], T.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('AceF,eBFIjk->ABcIjk', X.ab.vvvv[Va, vb, va, Vb], T.abb.vVVOoo, optimize=True)
            + 1.0 * np.einsum('AcEf,EBfIjk->ABcIjk', X.ab.vvvv[Va, vb, Va, vb], T.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('AcEF,EBFIjk->ABcIjk', X.ab.vvvv[Va, vb, Va, Vb], T.abb.VVVOoo, optimize=True)
    )
    dR.abb.VVvOoo += (1.0 / 2.0) * (
            +1.0 * np.einsum('AMIe,eBcMjk->ABcIjk', X.aa.voov[Va, Oa, Oa, va], T.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('AMIE,EBcMjk->ABcIjk', X.aa.voov[Va, Oa, Oa, Va], T.abb.VVvOoo, optimize=True)
    )
    dR.abb.VVvOoo += (1.0 / 2.0) * (
            +1.0 * np.einsum('AMIe,BcejkM->ABcIjk', X.ab.voov[Va, Ob, Oa, vb], T.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('AMIE,BEcjkM->ABcIjk', X.ab.voov[Va, Ob, Oa, Vb], T.bbb.VVvooO, optimize=True)
    )
    dR.abb.VVvOoo += (2.0 / 2.0) * (
            -1.0 * np.einsum('mBej,AecmIk->ABcIjk', X.ab.ovvo[oa, Vb, va, ob], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('MBej,AecIMk->ABcIjk', X.ab.ovvo[Oa, Vb, va, ob], T.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('mBEj,EAcmIk->ABcIjk', X.ab.ovvo[oa, Vb, Va, ob], T.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('MBEj,EAcIMk->ABcIjk', X.ab.ovvo[Oa, Vb, Va, ob], T.aab.VVvOOo, optimize=True)
    )
    dR.abb.VVvOoo += (2.0 / 2.0) * (
            +1.0 * np.einsum('mcej,AeBmIk->ABcIjk', X.ab.ovvo[oa, vb, va, ob], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('Mcej,AeBIMk->ABcIjk', X.ab.ovvo[Oa, vb, va, ob], T.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('mcEj,EABmIk->ABcIjk', X.ab.ovvo[oa, vb, Va, ob], T.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('McEj,EABIMk->ABcIjk', X.ab.ovvo[Oa, vb, Va, ob], T.aab.VVVOOo, optimize=True)
    )
    dR.abb.VVvOoo += (2.0 / 2.0) * (
            -1.0 * np.einsum('Bmje,AceImk->ABcIjk', X.bb.voov[Vb, ob, ob, vb], T.abb.VvvOoo, optimize=True)
            + 1.0 * np.einsum('BMje,AceIkM->ABcIjk', X.bb.voov[Vb, Ob, ob, vb], T.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('BmjE,AEcImk->ABcIjk', X.bb.voov[Vb, ob, ob, Vb], T.abb.VVvOoo, optimize=True)
            - 1.0 * np.einsum('BMjE,AEcIkM->ABcIjk', X.bb.voov[Vb, Ob, ob, Vb], T.abb.VVvOoO, optimize=True)
    )
    dR.abb.VVvOoo += (2.0 / 2.0) * (
            +1.0 * np.einsum('cmje,ABeImk->ABcIjk', X.bb.voov[vb, ob, ob, vb], T.abb.VVvOoo, optimize=True)
            - 1.0 * np.einsum('cMje,ABeIkM->ABcIjk', X.bb.voov[vb, Ob, ob, vb], T.abb.VVvOoO, optimize=True)
            + 1.0 * np.einsum('cmjE,ABEImk->ABcIjk', X.bb.voov[vb, ob, ob, Vb], T.abb.VVVOoo, optimize=True)
            - 1.0 * np.einsum('cMjE,ABEIkM->ABcIjk', X.bb.voov[vb, Ob, ob, Vb], T.abb.VVVOoO, optimize=True)
    )
    dR.abb.VVvOoo += (1.0 / 2.0) * (
            +1.0 * np.einsum('MBIe,AceMjk->ABcIjk', X.ab.ovov[Oa, Vb, Oa, vb], T.abb.VvvOoo, optimize=True)
            - 1.0 * np.einsum('MBIE,AEcMjk->ABcIjk', X.ab.ovov[Oa, Vb, Oa, Vb], T.abb.VVvOoo, optimize=True)
    )
    dR.abb.VVvOoo += (1.0 / 2.0) * (
            -1.0 * np.einsum('McIe,ABeMjk->ABcIjk', X.ab.ovov[Oa, vb, Oa, vb], T.abb.VVvOoo, optimize=True)
            - 1.0 * np.einsum('McIE,ABEMjk->ABcIjk', X.ab.ovov[Oa, vb, Oa, Vb], T.abb.VVVOoo, optimize=True)
    )
    dR.abb.VVvOoo += (2.0 / 2.0) * (
            -1.0 * np.einsum('Amej,eBcImk->ABcIjk', X.ab.vovo[Va, ob, va, ob], T.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('AMej,eBcIkM->ABcIjk', X.ab.vovo[Va, Ob, va, ob], T.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('AmEj,EBcImk->ABcIjk', X.ab.vovo[Va, ob, Va, ob], T.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('AMEj,EBcIkM->ABcIjk', X.ab.vovo[Va, Ob, Va, ob], T.abb.VVvOoO, optimize=True)
    )
    dR.abb.VVvOoo += (1.0 / 2.0) * (
            -1.0 * np.einsum('MI,ABcMjk->ABcIjk', H.a.oo[Oa, Oa], R.abb.VVvOoo, optimize=True)
    )
    dR.abb.VVvOoo += (2.0 / 2.0) * (
            -1.0 * np.einsum('mj,ABcImk->ABcIjk', H.b.oo[ob, ob], R.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('Mj,ABcIkM->ABcIjk', H.b.oo[Ob, ob], R.abb.VVvOoO, optimize=True)
    )
    dR.abb.VVvOoo += (1.0 / 2.0) * (
            +1.0 * np.einsum('Ae,eBcIjk->ABcIjk', H.a.vv[Va, va], R.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('AE,EBcIjk->ABcIjk', H.a.vv[Va, Va], R.abb.VVvOoo, optimize=True)
    )
    dR.abb.VVvOoo += (1.0 / 2.0) * (
            -1.0 * np.einsum('Be,AceIjk->ABcIjk', H.b.vv[Vb, vb], R.abb.VvvOoo, optimize=True)
            + 1.0 * np.einsum('BE,AEcIjk->ABcIjk', H.b.vv[Vb, Vb], R.abb.VVvOoo, optimize=True)
    )
    dR.abb.VVvOoo += (1.0 / 2.0) * (
            +1.0 * np.einsum('ce,ABeIjk->ABcIjk', H.b.vv[vb, vb], R.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('cE,ABEIjk->ABcIjk', H.b.vv[vb, Vb], R.abb.VVVOoo, optimize=True)
    )
    dR.abb.VVvOoo += (1.0 / 2.0) * (
            +0.5 * np.einsum('mnjk,ABcImn->ABcIjk', H.bb.oooo[ob, ob, ob, ob], R.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('mNjk,ABcImN->ABcIjk', H.bb.oooo[ob, Ob, ob, ob], R.abb.VVvOoO, optimize=True)
            + 0.5 * np.einsum('MNjk,ABcIMN->ABcIjk', H.bb.oooo[Ob, Ob, ob, ob], R.abb.VVvOOO, optimize=True)
    )
    dR.abb.VVvOoo += (2.0 / 2.0) * (
            +1.0 * np.einsum('MnIj,ABcMnk->ABcIjk', H.ab.oooo[Oa, ob, Oa, ob], R.abb.VVvOoo, optimize=True)
            - 1.0 * np.einsum('mNIj,ABcmkN->ABcIjk', H.ab.oooo[oa, Ob, Oa, ob], R.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('MNIj,ABcMkN->ABcIjk', H.ab.oooo[Oa, Ob, Oa, ob], R.abb.VVvOoO, optimize=True)
    )
    dR.abb.VVvOoo += (1.0 / 2.0) * (
            -0.5 * np.einsum('Bcef,AfeIjk->ABcIjk', H.bb.vvvv[Vb, vb, vb, vb], R.abb.VvvOoo, optimize=True)
            + 1.0 * np.einsum('BcEf,AEfIjk->ABcIjk', H.bb.vvvv[Vb, vb, Vb, vb], R.abb.VVvOoo, optimize=True)
            - 0.5 * np.einsum('BcEF,AFEIjk->ABcIjk', H.bb.vvvv[Vb, vb, Vb, Vb], R.abb.VVVOoo, optimize=True)
    )
    dR.abb.VVvOoo += (1.0 / 2.0) * (
            +1.0 * np.einsum('ABeF,eFcIjk->ABcIjk', H.ab.vvvv[Va, Vb, va, Vb], R.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('ABEf,EcfIjk->ABcIjk', H.ab.vvvv[Va, Vb, Va, vb], R.abb.VvvOoo, optimize=True)
            + 1.0 * np.einsum('ABEF,EFcIjk->ABcIjk', H.ab.vvvv[Va, Vb, Va, Vb], R.abb.VVvOoo, optimize=True)
    )
    dR.abb.VVvOoo += (1.0 / 2.0) * (
            +1.0 * np.einsum('Acef,eBfIjk->ABcIjk', H.ab.vvvv[Va, vb, va, vb], R.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('AceF,eBFIjk->ABcIjk', H.ab.vvvv[Va, vb, va, Vb], R.abb.vVVOoo, optimize=True)
            + 1.0 * np.einsum('AcEf,EBfIjk->ABcIjk', H.ab.vvvv[Va, vb, Va, vb], R.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('AcEF,EBFIjk->ABcIjk', H.ab.vvvv[Va, vb, Va, Vb], R.abb.VVVOoo, optimize=True)
    )
    dR.abb.VVvOoo += (1.0 / 2.0) * (
            +1.0 * np.einsum('AMIe,eBcMjk->ABcIjk', H.aa.voov[Va, Oa, Oa, va], R.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('AMIE,EBcMjk->ABcIjk', H.aa.voov[Va, Oa, Oa, Va], R.abb.VVvOoo, optimize=True)
    )
    dR.abb.VVvOoo += (1.0 / 2.0) * (
            +1.0 * np.einsum('AMIe,BcejkM->ABcIjk', H.ab.voov[Va, Ob, Oa, vb], R.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('AMIE,BEcjkM->ABcIjk', H.ab.voov[Va, Ob, Oa, Vb], R.bbb.VVvooO, optimize=True)
    )
    dR.abb.VVvOoo += (2.0 / 2.0) * (
            -1.0 * np.einsum('mBej,AecmIk->ABcIjk', H.ab.ovvo[oa, Vb, va, ob], R.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('MBej,AecIMk->ABcIjk', H.ab.ovvo[Oa, Vb, va, ob], R.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('mBEj,EAcmIk->ABcIjk', H.ab.ovvo[oa, Vb, Va, ob], R.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('MBEj,EAcIMk->ABcIjk', H.ab.ovvo[Oa, Vb, Va, ob], R.aab.VVvOOo, optimize=True)
    )
    dR.abb.VVvOoo += (2.0 / 2.0) * (
            +1.0 * np.einsum('mcej,AeBmIk->ABcIjk', H.ab.ovvo[oa, vb, va, ob], R.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('Mcej,AeBIMk->ABcIjk', H.ab.ovvo[Oa, vb, va, ob], R.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('mcEj,EABmIk->ABcIjk', H.ab.ovvo[oa, vb, Va, ob], R.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('McEj,EABIMk->ABcIjk', H.ab.ovvo[Oa, vb, Va, ob], R.aab.VVVOOo, optimize=True)
    )
    dR.abb.VVvOoo += (2.0 / 2.0) * (
            -1.0 * np.einsum('Bmje,AceImk->ABcIjk', H.bb.voov[Vb, ob, ob, vb], R.abb.VvvOoo, optimize=True)
            + 1.0 * np.einsum('BMje,AceIkM->ABcIjk', H.bb.voov[Vb, Ob, ob, vb], R.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('BmjE,AEcImk->ABcIjk', H.bb.voov[Vb, ob, ob, Vb], R.abb.VVvOoo, optimize=True)
            - 1.0 * np.einsum('BMjE,AEcIkM->ABcIjk', H.bb.voov[Vb, Ob, ob, Vb], R.abb.VVvOoO, optimize=True)
    )
    dR.abb.VVvOoo += (2.0 / 2.0) * (
            +1.0 * np.einsum('cmje,ABeImk->ABcIjk', H.bb.voov[vb, ob, ob, vb], R.abb.VVvOoo, optimize=True)
            - 1.0 * np.einsum('cMje,ABeIkM->ABcIjk', H.bb.voov[vb, Ob, ob, vb], R.abb.VVvOoO, optimize=True)
            + 1.0 * np.einsum('cmjE,ABEImk->ABcIjk', H.bb.voov[vb, ob, ob, Vb], R.abb.VVVOoo, optimize=True)
            - 1.0 * np.einsum('cMjE,ABEIkM->ABcIjk', H.bb.voov[vb, Ob, ob, Vb], R.abb.VVVOoO, optimize=True)
    )
    dR.abb.VVvOoo += (1.0 / 2.0) * (
            +1.0 * np.einsum('MBIe,AceMjk->ABcIjk', H.ab.ovov[Oa, Vb, Oa, vb], R.abb.VvvOoo, optimize=True)
            - 1.0 * np.einsum('MBIE,AEcMjk->ABcIjk', H.ab.ovov[Oa, Vb, Oa, Vb], R.abb.VVvOoo, optimize=True)
    )
    dR.abb.VVvOoo += (1.0 / 2.0) * (
            -1.0 * np.einsum('McIe,ABeMjk->ABcIjk', H.ab.ovov[Oa, vb, Oa, vb], R.abb.VVvOoo, optimize=True)
            - 1.0 * np.einsum('McIE,ABEMjk->ABcIjk', H.ab.ovov[Oa, vb, Oa, Vb], R.abb.VVVOoo, optimize=True)
    )
    dR.abb.VVvOoo += (2.0 / 2.0) * (
            -1.0 * np.einsum('Amej,eBcImk->ABcIjk', H.ab.vovo[Va, ob, va, ob], R.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('AMej,eBcIkM->ABcIjk', H.ab.vovo[Va, Ob, va, ob], R.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('AmEj,EBcImk->ABcIjk', H.ab.vovo[Va, ob, Va, ob], R.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('AMEj,EBcIkM->ABcIjk', H.ab.vovo[Va, Ob, Va, ob], R.abb.VVvOoO, optimize=True)
    )
    # of terms =  38

    dR.abb.VVvOoo -= np.transpose(dR.abb.VVvOoo, (0, 1, 2, 3, 5, 4))

    return dR

def update(R, omega, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    R.abb.VVvOoo = eomcc_active_loops.update_r3c_110100(
        R.abb.VVvOoo,
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
