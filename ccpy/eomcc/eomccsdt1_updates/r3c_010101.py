import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops

def build(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.abb.vVvOoO = (1.0 / 1.0) * (
            +1.0 * np.einsum('aBIe,ecjK->aBcIjK', X.ab.vvov[va, Vb, Oa, :], T.bb[:, vb, ob, Ob], optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('acIe,eBjK->aBcIjK', X.ab.vvov[va, vb, Oa, :], T.bb[:, Vb, ob, Ob], optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('amIj,BcmK->aBcIjK', X.ab.vooo[va, :, Oa, ob], T.bb[Vb, vb, :, Ob], optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('amIK,Bcmj->aBcIjK', X.ab.vooo[va, :, Oa, Ob], T.bb[Vb, vb, :, ob], optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('cBKe,aeIj->aBcIjK', X.bb.vvov[vb, Vb, Ob, :], T.ab[va, :, Oa, ob], optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('cBje,aeIK->aBcIjK', X.bb.vvov[vb, Vb, ob, :], T.ab[va, :, Oa, Ob], optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('cmKj,aBIm->aBcIjK', X.bb.vooo[vb, :, Ob, ob], T.ab[va, Vb, Oa, :], optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('BmKj,acIm->aBcIjK', X.bb.vooo[Vb, :, Ob, ob], T.ab[va, vb, Oa, :], optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('aBej,ecIK->aBcIjK', X.ab.vvvo[va, Vb, :, ob], T.ab[:, vb, Oa, Ob], optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('acej,eBIK->aBcIjK', X.ab.vvvo[va, vb, :, ob], T.ab[:, Vb, Oa, Ob], optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('aBeK,ecIj->aBcIjK', X.ab.vvvo[va, Vb, :, Ob], T.ab[:, vb, Oa, ob], optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('aceK,eBIj->aBcIjK', X.ab.vvvo[va, vb, :, Ob], T.ab[:, Vb, Oa, ob], optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mBIj,acmK->aBcIjK', X.ab.ovoo[:, Vb, Oa, ob], T.ab[va, vb, :, Ob], optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mcIj,aBmK->aBcIjK', X.ab.ovoo[:, vb, Oa, ob], T.ab[va, Vb, :, Ob], optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mBIK,acmj->aBcIjK', X.ab.ovoo[:, Vb, Oa, Ob], T.ab[va, vb, :, ob], optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mcIK,aBmj->aBcIjK', X.ab.ovoo[:, vb, Oa, Ob], T.ab[va, Vb, :, ob], optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('aBIe,ecjK->aBcIjK', H.ab.vvov[va, Vb, Oa, :], R.bb[:, vb, ob, Ob], optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('acIe,eBjK->aBcIjK', H.ab.vvov[va, vb, Oa, :], R.bb[:, Vb, ob, Ob], optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('amIj,BcmK->aBcIjK', H.ab.vooo[va, :, Oa, ob], R.bb[Vb, vb, :, Ob], optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('amIK,Bcmj->aBcIjK', H.ab.vooo[va, :, Oa, Ob], R.bb[Vb, vb, :, ob], optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('cBKe,aeIj->aBcIjK', H.bb.vvov[vb, Vb, Ob, :], R.ab[va, :, Oa, ob], optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('cBje,aeIK->aBcIjK', H.bb.vvov[vb, Vb, ob, :], R.ab[va, :, Oa, Ob], optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('cmKj,aBIm->aBcIjK', H.bb.vooo[vb, :, Ob, ob], R.ab[va, Vb, Oa, :], optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('BmKj,acIm->aBcIjK', H.bb.vooo[Vb, :, Ob, ob], R.ab[va, vb, Oa, :], optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('aBej,ecIK->aBcIjK', H.ab.vvvo[va, Vb, :, ob], R.ab[:, vb, Oa, Ob], optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('acej,eBIK->aBcIjK', H.ab.vvvo[va, vb, :, ob], R.ab[:, Vb, Oa, Ob], optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('aBeK,ecIj->aBcIjK', H.ab.vvvo[va, Vb, :, Ob], R.ab[:, vb, Oa, ob], optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('aceK,eBIj->aBcIjK', H.ab.vvvo[va, vb, :, Ob], R.ab[:, Vb, Oa, ob], optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mBIj,acmK->aBcIjK', H.ab.ovoo[:, Vb, Oa, ob], R.ab[va, vb, :, Ob], optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mcIj,aBmK->aBcIjK', H.ab.ovoo[:, vb, Oa, ob], R.ab[va, Vb, :, Ob], optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mBIK,acmj->aBcIjK', H.ab.ovoo[:, Vb, Oa, Ob], R.ab[va, vb, :, ob], optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mcIK,aBmj->aBcIjK', H.ab.ovoo[:, vb, Oa, Ob], R.ab[va, Vb, :, ob], optimize=True)
    )
    # of terms =  32
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mI,aBcmjK->aBcIjK', X.a.oo[oa, Oa], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('MI,aBcMjK->aBcIjK', X.a.oo[Oa, Oa], T.abb.vVvOoO, optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mj,aBcImK->aBcIjK', X.b.oo[ob, ob], T.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('Mj,aBcIMK->aBcIjK', X.b.oo[Ob, ob], T.abb.vVvOOO, optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mK,aBcImj->aBcIjK', X.b.oo[ob, Ob], T.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('MK,aBcIjM->aBcIjK', X.b.oo[Ob, Ob], T.abb.vVvOoO, optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('ae,eBcIjK->aBcIjK', X.a.vv[va, va], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('aE,EBcIjK->aBcIjK', X.a.vv[va, Va], T.abb.VVvOoO, optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('BE,aEcIjK->aBcIjK', X.b.vv[Vb, Vb], T.abb.vVvOoO, optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('ce,aBeIjK->aBcIjK', X.b.vv[vb, vb], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('cE,aBEIjK->aBcIjK', X.b.vv[vb, Vb], T.abb.vVVOoO, optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            +0.5 * np.einsum('mnjK,aBcImn->aBcIjK', X.bb.oooo[ob, ob, ob, Ob], T.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('MnjK,aBcInM->aBcIjK', X.bb.oooo[Ob, ob, ob, Ob], T.abb.vVvOoO, optimize=True)
            + 0.5 * np.einsum('MNjK,aBcIMN->aBcIjK', X.bb.oooo[Ob, Ob, ob, Ob], T.abb.vVvOOO, optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mnIj,aBcmnK->aBcIjK', X.ab.oooo[oa, ob, Oa, ob], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('mNIj,aBcmNK->aBcIjK', X.ab.oooo[oa, Ob, Oa, ob], T.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('MnIj,aBcMnK->aBcIjK', X.ab.oooo[Oa, ob, Oa, ob], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('MNIj,aBcMNK->aBcIjK', X.ab.oooo[Oa, Ob, Oa, ob], T.abb.vVvOOO, optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mNIK,aBcmjN->aBcIjK', X.ab.oooo[oa, Ob, Oa, Ob], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('MnIK,aBcMnj->aBcIjK', X.ab.oooo[Oa, ob, Oa, Ob], T.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('MNIK,aBcMjN->aBcIjK', X.ab.oooo[Oa, Ob, Oa, Ob], T.abb.vVvOoO, optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('BcEf,aEfIjK->aBcIjK', X.bb.vvvv[Vb, vb, Vb, vb], T.abb.vVvOoO, optimize=True)
            - 0.5 * np.einsum('BcEF,aFEIjK->aBcIjK', X.bb.vvvv[Vb, vb, Vb, Vb], T.abb.vVVOoO, optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('aBeF,eFcIjK->aBcIjK', X.ab.vvvv[va, Vb, va, Vb], T.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('aBEf,EcfIjK->aBcIjK', X.ab.vvvv[va, Vb, Va, vb], T.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('aBEF,EFcIjK->aBcIjK', X.ab.vvvv[va, Vb, Va, Vb], T.abb.VVvOoO, optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('acef,eBfIjK->aBcIjK', X.ab.vvvv[va, vb, va, vb], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('aceF,eBFIjK->aBcIjK', X.ab.vvvv[va, vb, va, Vb], T.abb.vVVOoO, optimize=True)
            + 1.0 * np.einsum('acEf,EBfIjK->aBcIjK', X.ab.vvvv[va, vb, Va, vb], T.abb.VVvOoO, optimize=True)
            + 1.0 * np.einsum('acEF,EBFIjK->aBcIjK', X.ab.vvvv[va, vb, Va, Vb], T.abb.VVVOoO, optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('amIe,eBcmjK->aBcIjK', X.aa.voov[va, oa, Oa, va], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('aMIe,eBcMjK->aBcIjK', X.aa.voov[va, Oa, Oa, va], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('amIE,EBcmjK->aBcIjK', X.aa.voov[va, oa, Oa, Va], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('aMIE,EBcMjK->aBcIjK', X.aa.voov[va, Oa, Oa, Va], T.abb.VVvOoO, optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('amIe,BcemjK->aBcIjK', X.ab.voov[va, ob, Oa, vb], T.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('aMIe,BcejMK->aBcIjK', X.ab.voov[va, Ob, Oa, vb], T.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('amIE,BEcmjK->aBcIjK', X.ab.voov[va, ob, Oa, Vb], T.bbb.VVvooO, optimize=True)
            + 1.0 * np.einsum('aMIE,BEcjMK->aBcIjK', X.ab.voov[va, Ob, Oa, Vb], T.bbb.VVvoOO, optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mBEj,EacmIK->aBcIjK', X.ab.ovvo[oa, Vb, Va, ob], T.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('MBEj,EacIMK->aBcIjK', X.ab.ovvo[Oa, Vb, Va, ob], T.aab.VvvOOO, optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mcej,eaBmIK->aBcIjK', X.ab.ovvo[oa, vb, va, ob], T.aab.vvVoOO, optimize=True)
            + 1.0 * np.einsum('Mcej,eaBIMK->aBcIjK', X.ab.ovvo[Oa, vb, va, ob], T.aab.vvVOOO, optimize=True)
            - 1.0 * np.einsum('mcEj,EaBmIK->aBcIjK', X.ab.ovvo[oa, vb, Va, ob], T.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('McEj,EaBIMK->aBcIjK', X.ab.ovvo[Oa, vb, Va, ob], T.aab.VvVOOO, optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mBEK,EacmIj->aBcIjK', X.ab.ovvo[oa, Vb, Va, Ob], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('MBEK,EacIMj->aBcIjK', X.ab.ovvo[Oa, Vb, Va, Ob], T.aab.VvvOOo, optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mceK,eaBmIj->aBcIjK', X.ab.ovvo[oa, vb, va, Ob], T.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('MceK,eaBIMj->aBcIjK', X.ab.ovvo[Oa, vb, va, Ob], T.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('mcEK,EaBmIj->aBcIjK', X.ab.ovvo[oa, vb, Va, Ob], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('McEK,EaBIMj->aBcIjK', X.ab.ovvo[Oa, vb, Va, Ob], T.aab.VvVOOo, optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('BmjE,aEcImK->aBcIjK', X.bb.voov[Vb, ob, ob, Vb], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('BMjE,aEcIMK->aBcIjK', X.bb.voov[Vb, Ob, ob, Vb], T.abb.vVvOOO, optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('cmje,aBeImK->aBcIjK', X.bb.voov[vb, ob, ob, vb], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('cMje,aBeIMK->aBcIjK', X.bb.voov[vb, Ob, ob, vb], T.abb.vVvOOO, optimize=True)
            + 1.0 * np.einsum('cmjE,aBEImK->aBcIjK', X.bb.voov[vb, ob, ob, Vb], T.abb.vVVOoO, optimize=True)
            + 1.0 * np.einsum('cMjE,aBEIMK->aBcIjK', X.bb.voov[vb, Ob, ob, Vb], T.abb.vVVOOO, optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('BmKE,aEcImj->aBcIjK', X.bb.voov[Vb, ob, Ob, Vb], T.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('BMKE,aEcIjM->aBcIjK', X.bb.voov[Vb, Ob, Ob, Vb], T.abb.vVvOoO, optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('cmKe,aBeImj->aBcIjK', X.bb.voov[vb, ob, Ob, vb], T.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('cMKe,aBeIjM->aBcIjK', X.bb.voov[vb, Ob, Ob, vb], T.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('cmKE,aBEImj->aBcIjK', X.bb.voov[vb, ob, Ob, Vb], T.abb.vVVOoo, optimize=True)
            + 1.0 * np.einsum('cMKE,aBEIjM->aBcIjK', X.bb.voov[vb, Ob, Ob, Vb], T.abb.vVVOoO, optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mBIE,aEcmjK->aBcIjK', X.ab.ovov[oa, Vb, Oa, Vb], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('MBIE,aEcMjK->aBcIjK', X.ab.ovov[Oa, Vb, Oa, Vb], T.abb.vVvOoO, optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mcIe,aBemjK->aBcIjK', X.ab.ovov[oa, vb, Oa, vb], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('McIe,aBeMjK->aBcIjK', X.ab.ovov[Oa, vb, Oa, vb], T.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('mcIE,aBEmjK->aBcIjK', X.ab.ovov[oa, vb, Oa, Vb], T.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('McIE,aBEMjK->aBcIjK', X.ab.ovov[Oa, vb, Oa, Vb], T.abb.vVVOoO, optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('amej,eBcImK->aBcIjK', X.ab.vovo[va, ob, va, ob], T.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('aMej,eBcIMK->aBcIjK', X.ab.vovo[va, Ob, va, ob], T.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('amEj,EBcImK->aBcIjK', X.ab.vovo[va, ob, Va, ob], T.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('aMEj,EBcIMK->aBcIjK', X.ab.vovo[va, Ob, Va, ob], T.abb.VVvOOO, optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('ameK,eBcImj->aBcIjK', X.ab.vovo[va, ob, va, Ob], T.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('aMeK,eBcIjM->aBcIjK', X.ab.vovo[va, Ob, va, Ob], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('amEK,EBcImj->aBcIjK', X.ab.vovo[va, ob, Va, Ob], T.abb.VVvOoo, optimize=True)
            - 1.0 * np.einsum('aMEK,EBcIjM->aBcIjK', X.ab.vovo[va, Ob, Va, Ob], T.abb.VVvOoO, optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mI,aBcmjK->aBcIjK', H.a.oo[oa, Oa], R.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('MI,aBcMjK->aBcIjK', H.a.oo[Oa, Oa], R.abb.vVvOoO, optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mj,aBcImK->aBcIjK', H.b.oo[ob, ob], R.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('Mj,aBcIMK->aBcIjK', H.b.oo[Ob, ob], R.abb.vVvOOO, optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mK,aBcImj->aBcIjK', H.b.oo[ob, Ob], R.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('MK,aBcIjM->aBcIjK', H.b.oo[Ob, Ob], R.abb.vVvOoO, optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('ae,eBcIjK->aBcIjK', H.a.vv[va, va], R.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('aE,EBcIjK->aBcIjK', H.a.vv[va, Va], R.abb.VVvOoO, optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('BE,aEcIjK->aBcIjK', H.b.vv[Vb, Vb], R.abb.vVvOoO, optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('ce,aBeIjK->aBcIjK', H.b.vv[vb, vb], R.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('cE,aBEIjK->aBcIjK', H.b.vv[vb, Vb], R.abb.vVVOoO, optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            +0.5 * np.einsum('mnjK,aBcImn->aBcIjK', H.bb.oooo[ob, ob, ob, Ob], R.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('MnjK,aBcInM->aBcIjK', H.bb.oooo[Ob, ob, ob, Ob], R.abb.vVvOoO, optimize=True)
            + 0.5 * np.einsum('MNjK,aBcIMN->aBcIjK', H.bb.oooo[Ob, Ob, ob, Ob], R.abb.vVvOOO, optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mnIj,aBcmnK->aBcIjK', H.ab.oooo[oa, ob, Oa, ob], R.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('mNIj,aBcmNK->aBcIjK', H.ab.oooo[oa, Ob, Oa, ob], R.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('MnIj,aBcMnK->aBcIjK', H.ab.oooo[Oa, ob, Oa, ob], R.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('MNIj,aBcMNK->aBcIjK', H.ab.oooo[Oa, Ob, Oa, ob], R.abb.vVvOOO, optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mNIK,aBcmjN->aBcIjK', H.ab.oooo[oa, Ob, Oa, Ob], R.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('MnIK,aBcMnj->aBcIjK', H.ab.oooo[Oa, ob, Oa, Ob], R.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('MNIK,aBcMjN->aBcIjK', H.ab.oooo[Oa, Ob, Oa, Ob], R.abb.vVvOoO, optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('BcEf,aEfIjK->aBcIjK', H.bb.vvvv[Vb, vb, Vb, vb], R.abb.vVvOoO, optimize=True)
            - 0.5 * np.einsum('BcEF,aFEIjK->aBcIjK', H.bb.vvvv[Vb, vb, Vb, Vb], R.abb.vVVOoO, optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('aBeF,eFcIjK->aBcIjK', H.ab.vvvv[va, Vb, va, Vb], R.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('aBEf,EcfIjK->aBcIjK', H.ab.vvvv[va, Vb, Va, vb], R.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('aBEF,EFcIjK->aBcIjK', H.ab.vvvv[va, Vb, Va, Vb], R.abb.VVvOoO, optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('acef,eBfIjK->aBcIjK', H.ab.vvvv[va, vb, va, vb], R.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('aceF,eBFIjK->aBcIjK', H.ab.vvvv[va, vb, va, Vb], R.abb.vVVOoO, optimize=True)
            + 1.0 * np.einsum('acEf,EBfIjK->aBcIjK', H.ab.vvvv[va, vb, Va, vb], R.abb.VVvOoO, optimize=True)
            + 1.0 * np.einsum('acEF,EBFIjK->aBcIjK', H.ab.vvvv[va, vb, Va, Vb], R.abb.VVVOoO, optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('amIe,eBcmjK->aBcIjK', H.aa.voov[va, oa, Oa, va], R.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('aMIe,eBcMjK->aBcIjK', H.aa.voov[va, Oa, Oa, va], R.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('amIE,EBcmjK->aBcIjK', H.aa.voov[va, oa, Oa, Va], R.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('aMIE,EBcMjK->aBcIjK', H.aa.voov[va, Oa, Oa, Va], R.abb.VVvOoO, optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('amIe,BcemjK->aBcIjK', H.ab.voov[va, ob, Oa, vb], R.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('aMIe,BcejMK->aBcIjK', H.ab.voov[va, Ob, Oa, vb], R.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('amIE,BEcmjK->aBcIjK', H.ab.voov[va, ob, Oa, Vb], R.bbb.VVvooO, optimize=True)
            + 1.0 * np.einsum('aMIE,BEcjMK->aBcIjK', H.ab.voov[va, Ob, Oa, Vb], R.bbb.VVvoOO, optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mBEj,EacmIK->aBcIjK', H.ab.ovvo[oa, Vb, Va, ob], R.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('MBEj,EacIMK->aBcIjK', H.ab.ovvo[Oa, Vb, Va, ob], R.aab.VvvOOO, optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mcej,eaBmIK->aBcIjK', H.ab.ovvo[oa, vb, va, ob], R.aab.vvVoOO, optimize=True)
            + 1.0 * np.einsum('Mcej,eaBIMK->aBcIjK', H.ab.ovvo[Oa, vb, va, ob], R.aab.vvVOOO, optimize=True)
            - 1.0 * np.einsum('mcEj,EaBmIK->aBcIjK', H.ab.ovvo[oa, vb, Va, ob], R.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('McEj,EaBIMK->aBcIjK', H.ab.ovvo[Oa, vb, Va, ob], R.aab.VvVOOO, optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mBEK,EacmIj->aBcIjK', H.ab.ovvo[oa, Vb, Va, Ob], R.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('MBEK,EacIMj->aBcIjK', H.ab.ovvo[Oa, Vb, Va, Ob], R.aab.VvvOOo, optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mceK,eaBmIj->aBcIjK', H.ab.ovvo[oa, vb, va, Ob], R.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('MceK,eaBIMj->aBcIjK', H.ab.ovvo[Oa, vb, va, Ob], R.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('mcEK,EaBmIj->aBcIjK', H.ab.ovvo[oa, vb, Va, Ob], R.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('McEK,EaBIMj->aBcIjK', H.ab.ovvo[Oa, vb, Va, Ob], R.aab.VvVOOo, optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('BmjE,aEcImK->aBcIjK', H.bb.voov[Vb, ob, ob, Vb], R.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('BMjE,aEcIMK->aBcIjK', H.bb.voov[Vb, Ob, ob, Vb], R.abb.vVvOOO, optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('cmje,aBeImK->aBcIjK', H.bb.voov[vb, ob, ob, vb], R.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('cMje,aBeIMK->aBcIjK', H.bb.voov[vb, Ob, ob, vb], R.abb.vVvOOO, optimize=True)
            + 1.0 * np.einsum('cmjE,aBEImK->aBcIjK', H.bb.voov[vb, ob, ob, Vb], R.abb.vVVOoO, optimize=True)
            + 1.0 * np.einsum('cMjE,aBEIMK->aBcIjK', H.bb.voov[vb, Ob, ob, Vb], R.abb.vVVOOO, optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('BmKE,aEcImj->aBcIjK', H.bb.voov[Vb, ob, Ob, Vb], R.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('BMKE,aEcIjM->aBcIjK', H.bb.voov[Vb, Ob, Ob, Vb], R.abb.vVvOoO, optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('cmKe,aBeImj->aBcIjK', H.bb.voov[vb, ob, Ob, vb], R.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('cMKe,aBeIjM->aBcIjK', H.bb.voov[vb, Ob, Ob, vb], R.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('cmKE,aBEImj->aBcIjK', H.bb.voov[vb, ob, Ob, Vb], R.abb.vVVOoo, optimize=True)
            + 1.0 * np.einsum('cMKE,aBEIjM->aBcIjK', H.bb.voov[vb, Ob, Ob, Vb], R.abb.vVVOoO, optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mBIE,aEcmjK->aBcIjK', H.ab.ovov[oa, Vb, Oa, Vb], R.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('MBIE,aEcMjK->aBcIjK', H.ab.ovov[Oa, Vb, Oa, Vb], R.abb.vVvOoO, optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mcIe,aBemjK->aBcIjK', H.ab.ovov[oa, vb, Oa, vb], R.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('McIe,aBeMjK->aBcIjK', H.ab.ovov[Oa, vb, Oa, vb], R.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('mcIE,aBEmjK->aBcIjK', H.ab.ovov[oa, vb, Oa, Vb], R.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('McIE,aBEMjK->aBcIjK', H.ab.ovov[Oa, vb, Oa, Vb], R.abb.vVVOoO, optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('amej,eBcImK->aBcIjK', H.ab.vovo[va, ob, va, ob], R.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('aMej,eBcIMK->aBcIjK', H.ab.vovo[va, Ob, va, ob], R.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('amEj,EBcImK->aBcIjK', H.ab.vovo[va, ob, Va, ob], R.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('aMEj,EBcIMK->aBcIjK', H.ab.vovo[va, Ob, Va, ob], R.abb.VVvOOO, optimize=True)
    )
    dR.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('ameK,eBcImj->aBcIjK', H.ab.vovo[va, ob, va, Ob], R.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('aMeK,eBcIjM->aBcIjK', H.ab.vovo[va, Ob, va, Ob], R.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('amEK,EBcImj->aBcIjK', H.ab.vovo[va, ob, Va, Ob], R.abb.VVvOoo, optimize=True)
            - 1.0 * np.einsum('aMEK,EBcIjM->aBcIjK', H.ab.vovo[va, Ob, Va, Ob], R.abb.VVvOoO, optimize=True)
    )
    # of terms =  52

    return dR

def update(R, omega, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    R.abb.vVvOoO = eomcc_active_loops.update_r3c_010101(
        R.abb.vVvOoO,
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
