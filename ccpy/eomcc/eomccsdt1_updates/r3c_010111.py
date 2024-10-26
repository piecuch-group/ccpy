import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops

def build(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.abb.vVvOOO = (1.0 / 2.0) * (
            +1.0 * np.einsum('aBIe,ecJK->aBcIJK', X.ab.vvov[va, Vb, Oa, :], T.bb[:, vb, Ob, Ob], optimize=True)
    )
    dR.abb.vVvOOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('acIe,eBJK->aBcIJK', X.ab.vvov[va, vb, Oa, :], T.bb[:, Vb, Ob, Ob], optimize=True)
    )
    dR.abb.vVvOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('amIJ,BcmK->aBcIJK', X.ab.vooo[va, :, Oa, Ob], T.bb[Vb, vb, :, Ob], optimize=True)
    )
    dR.abb.vVvOOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('cBKe,aeIJ->aBcIJK', X.bb.vvov[vb, Vb, Ob, :], T.ab[va, :, Oa, Ob], optimize=True)
    )
    dR.abb.vVvOOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('cmKJ,aBIm->aBcIJK', X.bb.vooo[vb, :, Ob, Ob], T.ab[va, Vb, Oa, :], optimize=True)
    )
    dR.abb.vVvOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('BmKJ,acIm->aBcIJK', X.bb.vooo[Vb, :, Ob, Ob], T.ab[va, vb, Oa, :], optimize=True)
    )
    dR.abb.vVvOOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('aBeJ,ecIK->aBcIJK', X.ab.vvvo[va, Vb, :, Ob], T.ab[:, vb, Oa, Ob], optimize=True)
    )
    dR.abb.vVvOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('aceJ,eBIK->aBcIJK', X.ab.vvvo[va, vb, :, Ob], T.ab[:, Vb, Oa, Ob], optimize=True)
    )
    dR.abb.vVvOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mBIJ,acmK->aBcIJK', X.ab.ovoo[:, Vb, Oa, Ob], T.ab[va, vb, :, Ob], optimize=True)
    )
    dR.abb.vVvOOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mcIJ,aBmK->aBcIJK', X.ab.ovoo[:, vb, Oa, Ob], T.ab[va, Vb, :, Ob], optimize=True)
    )
    dR.abb.vVvOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('aBIe,ecJK->aBcIJK', H.ab.vvov[va, Vb, Oa, :], R.bb[:, vb, Ob, Ob], optimize=True)
    )
    dR.abb.vVvOOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('acIe,eBJK->aBcIJK', H.ab.vvov[va, vb, Oa, :], R.bb[:, Vb, Ob, Ob], optimize=True)
    )
    dR.abb.vVvOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('amIJ,BcmK->aBcIJK', H.ab.vooo[va, :, Oa, Ob], R.bb[Vb, vb, :, Ob], optimize=True)
    )
    dR.abb.vVvOOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('cBKe,aeIJ->aBcIJK', H.bb.vvov[vb, Vb, Ob, :], R.ab[va, :, Oa, Ob], optimize=True)
    )
    dR.abb.vVvOOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('cmKJ,aBIm->aBcIJK', H.bb.vooo[vb, :, Ob, Ob], R.ab[va, Vb, Oa, :], optimize=True)
    )
    dR.abb.vVvOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('BmKJ,acIm->aBcIJK', H.bb.vooo[Vb, :, Ob, Ob], R.ab[va, vb, Oa, :], optimize=True)
    )
    dR.abb.vVvOOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('aBeJ,ecIK->aBcIJK', H.ab.vvvo[va, Vb, :, Ob], R.ab[:, vb, Oa, Ob], optimize=True)
    )
    dR.abb.vVvOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('aceJ,eBIK->aBcIJK', H.ab.vvvo[va, vb, :, Ob], R.ab[:, Vb, Oa, Ob], optimize=True)
    )
    dR.abb.vVvOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mBIJ,acmK->aBcIJK', H.ab.ovoo[:, Vb, Oa, Ob], R.ab[va, vb, :, Ob], optimize=True)
    )
    dR.abb.vVvOOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mcIJ,aBmK->aBcIJK', H.ab.ovoo[:, vb, Oa, Ob], R.ab[va, Vb, :, Ob], optimize=True)
    )
    # of terms =  20
    dR.abb.vVvOOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mI,aBcmJK->aBcIJK', X.a.oo[oa, Oa], T.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('MI,aBcMJK->aBcIJK', X.a.oo[Oa, Oa], T.abb.vVvOOO, optimize=True)
    )
    dR.abb.vVvOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mJ,aBcImK->aBcIJK', X.b.oo[ob, Ob], T.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('MJ,aBcIMK->aBcIJK', X.b.oo[Ob, Ob], T.abb.vVvOOO, optimize=True)
    )
    dR.abb.vVvOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('ae,eBcIJK->aBcIJK', X.a.vv[va, va], T.abb.vVvOOO, optimize=True)
            + 1.0 * np.einsum('aE,EBcIJK->aBcIJK', X.a.vv[va, Va], T.abb.VVvOOO, optimize=True)
    )
    dR.abb.vVvOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('BE,aEcIJK->aBcIJK', X.b.vv[Vb, Vb], T.abb.vVvOOO, optimize=True)
    )
    dR.abb.vVvOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('ce,aBeIJK->aBcIJK', X.b.vv[vb, vb], T.abb.vVvOOO, optimize=True)
            + 1.0 * np.einsum('cE,aBEIJK->aBcIJK', X.b.vv[vb, Vb], T.abb.vVVOOO, optimize=True)
    )
    dR.abb.vVvOOO += (1.0 / 2.0) * (
            +0.5 * np.einsum('mnJK,aBcImn->aBcIJK', X.bb.oooo[ob, ob, Ob, Ob], T.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('mNJK,aBcImN->aBcIJK', X.bb.oooo[ob, Ob, Ob, Ob], T.abb.vVvOoO, optimize=True)
            + 0.5 * np.einsum('MNJK,aBcIMN->aBcIJK', X.bb.oooo[Ob, Ob, Ob, Ob], T.abb.vVvOOO, optimize=True)
    )
    dR.abb.vVvOOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mnIJ,aBcmnK->aBcIJK', X.ab.oooo[oa, ob, Oa, Ob], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('MnIJ,aBcMnK->aBcIJK', X.ab.oooo[Oa, ob, Oa, Ob], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('mNIJ,aBcmNK->aBcIJK', X.ab.oooo[oa, Ob, Oa, Ob], T.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('MNIJ,aBcMNK->aBcIJK', X.ab.oooo[Oa, Ob, Oa, Ob], T.abb.vVvOOO, optimize=True)
    )
    dR.abb.vVvOOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('BceF,aFeIJK->aBcIJK', X.bb.vvvv[Vb, vb, vb, Vb], T.abb.vVvOOO, optimize=True)
            - 0.5 * np.einsum('BcEF,aFEIJK->aBcIJK', X.bb.vvvv[Vb, vb, Vb, Vb], T.abb.vVVOOO, optimize=True)
    )
    dR.abb.vVvOOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('aBEf,EcfIJK->aBcIJK', X.ab.vvvv[va, Vb, Va, vb], T.abb.VvvOOO, optimize=True)
            + 1.0 * np.einsum('aBeF,eFcIJK->aBcIJK', X.ab.vvvv[va, Vb, va, Vb], T.abb.vVvOOO, optimize=True)
            + 1.0 * np.einsum('aBEF,EFcIJK->aBcIJK', X.ab.vvvv[va, Vb, Va, Vb], T.abb.VVvOOO, optimize=True)
    )
    dR.abb.vVvOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('acef,eBfIJK->aBcIJK', X.ab.vvvv[va, vb, va, vb], T.abb.vVvOOO, optimize=True)
            + 1.0 * np.einsum('acEf,EBfIJK->aBcIJK', X.ab.vvvv[va, vb, Va, vb], T.abb.VVvOOO, optimize=True)
            + 1.0 * np.einsum('aceF,eBFIJK->aBcIJK', X.ab.vvvv[va, vb, va, Vb], T.abb.vVVOOO, optimize=True)
            + 1.0 * np.einsum('acEF,EBFIJK->aBcIJK', X.ab.vvvv[va, vb, Va, Vb], T.abb.VVVOOO, optimize=True)
    )
    dR.abb.vVvOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('amIe,eBcmJK->aBcIJK', X.aa.voov[va, oa, Oa, va], T.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('amIE,EBcmJK->aBcIJK', X.aa.voov[va, oa, Oa, Va], T.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('aMIe,eBcMJK->aBcIJK', X.aa.voov[va, Oa, Oa, va], T.abb.vVvOOO, optimize=True)
            + 1.0 * np.einsum('aMIE,EBcMJK->aBcIJK', X.aa.voov[va, Oa, Oa, Va], T.abb.VVvOOO, optimize=True)
    )
    dR.abb.vVvOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('amIe,BcemJK->aBcIJK', X.ab.voov[va, ob, Oa, vb], T.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('amIE,BEcmJK->aBcIJK', X.ab.voov[va, ob, Oa, Vb], T.bbb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('aMIe,BceMJK->aBcIJK', X.ab.voov[va, Ob, Oa, vb], T.bbb.VvvOOO, optimize=True)
            - 1.0 * np.einsum('aMIE,BEcMJK->aBcIJK', X.ab.voov[va, Ob, Oa, Vb], T.bbb.VVvOOO, optimize=True)
    )
    dR.abb.vVvOOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mBEJ,EacmIK->aBcIJK', X.ab.ovvo[oa, Vb, Va, Ob], T.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('MBEJ,EacIMK->aBcIJK', X.ab.ovvo[Oa, Vb, Va, Ob], T.aab.VvvOOO, optimize=True)
    )
    dR.abb.vVvOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mceJ,eaBmIK->aBcIJK', X.ab.ovvo[oa, vb, va, Ob], T.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('mcEJ,EaBmIK->aBcIJK', X.ab.ovvo[oa, vb, Va, Ob], T.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('MceJ,eaBIMK->aBcIJK', X.ab.ovvo[Oa, vb, va, Ob], T.aab.vvVOOO, optimize=True)
            + 1.0 * np.einsum('McEJ,EaBIMK->aBcIJK', X.ab.ovvo[Oa, vb, Va, Ob], T.aab.VvVOOO, optimize=True)
    )
    dR.abb.vVvOOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('BmJE,aEcImK->aBcIJK', X.bb.voov[Vb, ob, Ob, Vb], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('BMJE,aEcIMK->aBcIJK', X.bb.voov[Vb, Ob, Ob, Vb], T.abb.vVvOOO, optimize=True)
    )
    dR.abb.vVvOOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('cmJe,aBeImK->aBcIJK', X.bb.voov[vb, ob, Ob, vb], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('cmJE,aBEImK->aBcIJK', X.bb.voov[vb, ob, Ob, Vb], T.abb.vVVOoO, optimize=True)
            + 1.0 * np.einsum('cMJe,aBeIMK->aBcIJK', X.bb.voov[vb, Ob, Ob, vb], T.abb.vVvOOO, optimize=True)
            + 1.0 * np.einsum('cMJE,aBEIMK->aBcIJK', X.bb.voov[vb, Ob, Ob, Vb], T.abb.vVVOOO, optimize=True)
    )
    dR.abb.vVvOOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mBIE,aEcmJK->aBcIJK', X.ab.ovov[oa, Vb, Oa, Vb], T.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('MBIE,aEcMJK->aBcIJK', X.ab.ovov[Oa, Vb, Oa, Vb], T.abb.vVvOOO, optimize=True)
    )
    dR.abb.vVvOOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mcIe,aBemJK->aBcIJK', X.ab.ovov[oa, vb, Oa, vb], T.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('mcIE,aBEmJK->aBcIJK', X.ab.ovov[oa, vb, Oa, Vb], T.abb.vVVoOO, optimize=True)
            - 1.0 * np.einsum('McIe,aBeMJK->aBcIJK', X.ab.ovov[Oa, vb, Oa, vb], T.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('McIE,aBEMJK->aBcIJK', X.ab.ovov[Oa, vb, Oa, Vb], T.abb.vVVOOO, optimize=True)
    )
    dR.abb.vVvOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('ameJ,eBcImK->aBcIJK', X.ab.vovo[va, ob, va, Ob], T.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('amEJ,EBcImK->aBcIJK', X.ab.vovo[va, ob, Va, Ob], T.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('aMeJ,eBcIMK->aBcIJK', X.ab.vovo[va, Ob, va, Ob], T.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('aMEJ,EBcIMK->aBcIJK', X.ab.vovo[va, Ob, Va, Ob], T.abb.VVvOOO, optimize=True)
    )
    dR.abb.vVvOOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mI,aBcmJK->aBcIJK', H.a.oo[oa, Oa], R.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('MI,aBcMJK->aBcIJK', H.a.oo[Oa, Oa], R.abb.vVvOOO, optimize=True)
    )
    dR.abb.vVvOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mJ,aBcImK->aBcIJK', H.b.oo[ob, Ob], R.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('MJ,aBcIMK->aBcIJK', H.b.oo[Ob, Ob], R.abb.vVvOOO, optimize=True)
    )
    dR.abb.vVvOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('ae,eBcIJK->aBcIJK', H.a.vv[va, va], R.abb.vVvOOO, optimize=True)
            + 1.0 * np.einsum('aE,EBcIJK->aBcIJK', H.a.vv[va, Va], R.abb.VVvOOO, optimize=True)
    )
    dR.abb.vVvOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('BE,aEcIJK->aBcIJK', H.b.vv[Vb, Vb], R.abb.vVvOOO, optimize=True)
    )
    dR.abb.vVvOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('ce,aBeIJK->aBcIJK', H.b.vv[vb, vb], R.abb.vVvOOO, optimize=True)
            + 1.0 * np.einsum('cE,aBEIJK->aBcIJK', H.b.vv[vb, Vb], R.abb.vVVOOO, optimize=True)
    )
    dR.abb.vVvOOO += (1.0 / 2.0) * (
            +0.5 * np.einsum('mnJK,aBcImn->aBcIJK', H.bb.oooo[ob, ob, Ob, Ob], R.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('mNJK,aBcImN->aBcIJK', H.bb.oooo[ob, Ob, Ob, Ob], R.abb.vVvOoO, optimize=True)
            + 0.5 * np.einsum('MNJK,aBcIMN->aBcIJK', H.bb.oooo[Ob, Ob, Ob, Ob], R.abb.vVvOOO, optimize=True)
    )
    dR.abb.vVvOOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mnIJ,aBcmnK->aBcIJK', H.ab.oooo[oa, ob, Oa, Ob], R.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('MnIJ,aBcMnK->aBcIJK', H.ab.oooo[Oa, ob, Oa, Ob], R.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('mNIJ,aBcmNK->aBcIJK', H.ab.oooo[oa, Ob, Oa, Ob], R.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('MNIJ,aBcMNK->aBcIJK', H.ab.oooo[Oa, Ob, Oa, Ob], R.abb.vVvOOO, optimize=True)
    )
    dR.abb.vVvOOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('BceF,aFeIJK->aBcIJK', H.bb.vvvv[Vb, vb, vb, Vb], R.abb.vVvOOO, optimize=True)
            - 0.5 * np.einsum('BcEF,aFEIJK->aBcIJK', H.bb.vvvv[Vb, vb, Vb, Vb], R.abb.vVVOOO, optimize=True)
    )
    dR.abb.vVvOOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('aBEf,EcfIJK->aBcIJK', H.ab.vvvv[va, Vb, Va, vb], R.abb.VvvOOO, optimize=True)
            + 1.0 * np.einsum('aBeF,eFcIJK->aBcIJK', H.ab.vvvv[va, Vb, va, Vb], R.abb.vVvOOO, optimize=True)
            + 1.0 * np.einsum('aBEF,EFcIJK->aBcIJK', H.ab.vvvv[va, Vb, Va, Vb], R.abb.VVvOOO, optimize=True)
    )
    dR.abb.vVvOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('acef,eBfIJK->aBcIJK', H.ab.vvvv[va, vb, va, vb], R.abb.vVvOOO, optimize=True)
            + 1.0 * np.einsum('acEf,EBfIJK->aBcIJK', H.ab.vvvv[va, vb, Va, vb], R.abb.VVvOOO, optimize=True)
            + 1.0 * np.einsum('aceF,eBFIJK->aBcIJK', H.ab.vvvv[va, vb, va, Vb], R.abb.vVVOOO, optimize=True)
            + 1.0 * np.einsum('acEF,EBFIJK->aBcIJK', H.ab.vvvv[va, vb, Va, Vb], R.abb.VVVOOO, optimize=True)
    )
    dR.abb.vVvOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('amIe,eBcmJK->aBcIJK', H.aa.voov[va, oa, Oa, va], R.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('amIE,EBcmJK->aBcIJK', H.aa.voov[va, oa, Oa, Va], R.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('aMIe,eBcMJK->aBcIJK', H.aa.voov[va, Oa, Oa, va], R.abb.vVvOOO, optimize=True)
            + 1.0 * np.einsum('aMIE,EBcMJK->aBcIJK', H.aa.voov[va, Oa, Oa, Va], R.abb.VVvOOO, optimize=True)
    )
    dR.abb.vVvOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('amIe,BcemJK->aBcIJK', H.ab.voov[va, ob, Oa, vb], R.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('amIE,BEcmJK->aBcIJK', H.ab.voov[va, ob, Oa, Vb], R.bbb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('aMIe,BceMJK->aBcIJK', H.ab.voov[va, Ob, Oa, vb], R.bbb.VvvOOO, optimize=True)
            - 1.0 * np.einsum('aMIE,BEcMJK->aBcIJK', H.ab.voov[va, Ob, Oa, Vb], R.bbb.VVvOOO, optimize=True)
    )
    dR.abb.vVvOOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mBEJ,EacmIK->aBcIJK', H.ab.ovvo[oa, Vb, Va, Ob], R.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('MBEJ,EacIMK->aBcIJK', H.ab.ovvo[Oa, Vb, Va, Ob], R.aab.VvvOOO, optimize=True)
    )
    dR.abb.vVvOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mceJ,eaBmIK->aBcIJK', H.ab.ovvo[oa, vb, va, Ob], R.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('mcEJ,EaBmIK->aBcIJK', H.ab.ovvo[oa, vb, Va, Ob], R.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('MceJ,eaBIMK->aBcIJK', H.ab.ovvo[Oa, vb, va, Ob], R.aab.vvVOOO, optimize=True)
            + 1.0 * np.einsum('McEJ,EaBIMK->aBcIJK', H.ab.ovvo[Oa, vb, Va, Ob], R.aab.VvVOOO, optimize=True)
    )
    dR.abb.vVvOOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('BmJE,aEcImK->aBcIJK', H.bb.voov[Vb, ob, Ob, Vb], R.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('BMJE,aEcIMK->aBcIJK', H.bb.voov[Vb, Ob, Ob, Vb], R.abb.vVvOOO, optimize=True)
    )
    dR.abb.vVvOOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('cmJe,aBeImK->aBcIJK', H.bb.voov[vb, ob, Ob, vb], R.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('cmJE,aBEImK->aBcIJK', H.bb.voov[vb, ob, Ob, Vb], R.abb.vVVOoO, optimize=True)
            + 1.0 * np.einsum('cMJe,aBeIMK->aBcIJK', H.bb.voov[vb, Ob, Ob, vb], R.abb.vVvOOO, optimize=True)
            + 1.0 * np.einsum('cMJE,aBEIMK->aBcIJK', H.bb.voov[vb, Ob, Ob, Vb], R.abb.vVVOOO, optimize=True)
    )
    dR.abb.vVvOOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mBIE,aEcmJK->aBcIJK', H.ab.ovov[oa, Vb, Oa, Vb], R.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('MBIE,aEcMJK->aBcIJK', H.ab.ovov[Oa, Vb, Oa, Vb], R.abb.vVvOOO, optimize=True)
    )
    dR.abb.vVvOOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mcIe,aBemJK->aBcIJK', H.ab.ovov[oa, vb, Oa, vb], R.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('mcIE,aBEmJK->aBcIJK', H.ab.ovov[oa, vb, Oa, Vb], R.abb.vVVoOO, optimize=True)
            - 1.0 * np.einsum('McIe,aBeMJK->aBcIJK', H.ab.ovov[Oa, vb, Oa, vb], R.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('McIE,aBEMJK->aBcIJK', H.ab.ovov[Oa, vb, Oa, Vb], R.abb.vVVOOO, optimize=True)
    )
    dR.abb.vVvOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('ameJ,eBcImK->aBcIJK', H.ab.vovo[va, ob, va, Ob], R.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('amEJ,EBcImK->aBcIJK', H.ab.vovo[va, ob, Va, Ob], R.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('aMeJ,eBcIMK->aBcIJK', H.ab.vovo[va, Ob, va, Ob], R.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('aMEJ,EBcIMK->aBcIJK', H.ab.vovo[va, Ob, Va, Ob], R.abb.VVvOOO, optimize=True)
    )
    # of terms =  38

    dR.abb.vVvOOO -= np.transpose(dR.abb.vVvOOO, (0, 1, 2, 3, 5, 4))

    return dR

def update(R, omega, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    R.abb.vVvOOO = eomcc_active_loops.update_r3c_010111(
        R.abb.vVvOOO,
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
