import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops

def build(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.abb.vVVOOO = (2.0 / 4.0) * (
            +1.0 * np.einsum('aBIe,eCJK->aBCIJK', X.ab.vvov[va, Vb, Oa, :], T.bb[:, Vb, Ob, Ob], optimize=True)
    )
    dR.abb.vVVOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('amIJ,BCmK->aBCIJK', X.ab.vooo[va, :, Oa, Ob], T.bb[Vb, Vb, :, Ob], optimize=True)
    )
    dR.abb.vVVOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('CBKe,aeIJ->aBCIJK', X.bb.vvov[Vb, Vb, Ob, :], T.ab[va, :, Oa, Ob], optimize=True)
    )
    dR.abb.vVVOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('CmKJ,aBIm->aBCIJK', X.bb.vooo[Vb, :, Ob, Ob], T.ab[va, Vb, Oa, :], optimize=True)
    )
    dR.abb.vVVOOO += (4.0 / 4.0) * (
            +1.0 * np.einsum('aBeJ,eCIK->aBCIJK', X.ab.vvvo[va, Vb, :, Ob], T.ab[:, Vb, Oa, Ob], optimize=True)
    )
    dR.abb.vVVOOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('mBIJ,aCmK->aBCIJK', X.ab.ovoo[:, Vb, Oa, Ob], T.ab[va, Vb, :, Ob], optimize=True)
    )
    dR.abb.vVVOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('aBIe,eCJK->aBCIJK', H.ab.vvov[va, Vb, Oa, :], R.bb[:, Vb, Ob, Ob], optimize=True)
    )
    dR.abb.vVVOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('amIJ,BCmK->aBCIJK', H.ab.vooo[va, :, Oa, Ob], R.bb[Vb, Vb, :, Ob], optimize=True)
    )
    dR.abb.vVVOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('CBKe,aeIJ->aBCIJK', H.bb.vvov[Vb, Vb, Ob, :], R.ab[va, :, Oa, Ob], optimize=True)
    )
    dR.abb.vVVOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('CmKJ,aBIm->aBCIJK', H.bb.vooo[Vb, :, Ob, Ob], R.ab[va, Vb, Oa, :], optimize=True)
    )
    dR.abb.vVVOOO += (4.0 / 4.0) * (
            +1.0 * np.einsum('aBeJ,eCIK->aBCIJK', H.ab.vvvo[va, Vb, :, Ob], R.ab[:, Vb, Oa, Ob], optimize=True)
    )
    dR.abb.vVVOOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('mBIJ,aCmK->aBCIJK', H.ab.ovoo[:, Vb, Oa, Ob], R.ab[va, Vb, :, Ob], optimize=True)
    )
    # of terms =  12
    dR.abb.vVVOOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('mI,aCBmJK->aBCIJK', X.a.oo[oa, Oa], T.abb.vVVoOO, optimize=True)
            + 1.0 * np.einsum('MI,aCBMJK->aBCIJK', X.a.oo[Oa, Oa], T.abb.vVVOOO, optimize=True)
    )
    dR.abb.vVVOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mJ,aCBImK->aBCIJK', X.b.oo[ob, Ob], T.abb.vVVOoO, optimize=True)
            + 1.0 * np.einsum('MJ,aCBIMK->aBCIJK', X.b.oo[Ob, Ob], T.abb.vVVOOO, optimize=True)
    )
    dR.abb.vVVOOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('ae,eCBIJK->aBCIJK', X.a.vv[va, va], T.abb.vVVOOO, optimize=True)
            - 1.0 * np.einsum('aE,ECBIJK->aBCIJK', X.a.vv[va, Va], T.abb.VVVOOO, optimize=True)
    )
    dR.abb.vVVOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('Be,aCeIJK->aBCIJK', X.b.vv[Vb, vb], T.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('BE,aCEIJK->aBCIJK', X.b.vv[Vb, Vb], T.abb.vVVOOO, optimize=True)
    )
    dR.abb.vVVOOO += (1.0 / 4.0) * (
            -0.5 * np.einsum('mnJK,aCBImn->aBCIJK', X.bb.oooo[ob, ob, Ob, Ob], T.abb.vVVOoo, optimize=True)
            - 1.0 * np.einsum('mNJK,aCBImN->aBCIJK', X.bb.oooo[ob, Ob, Ob, Ob], T.abb.vVVOoO, optimize=True)
            - 0.5 * np.einsum('MNJK,aCBIMN->aBCIJK', X.bb.oooo[Ob, Ob, Ob, Ob], T.abb.vVVOOO, optimize=True)
    )
    dR.abb.vVVOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('mnIJ,aCBmnK->aBCIJK', X.ab.oooo[oa, ob, Oa, Ob], T.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('MnIJ,aCBMnK->aBCIJK', X.ab.oooo[Oa, ob, Oa, Ob], T.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('mNIJ,aCBmNK->aBCIJK', X.ab.oooo[oa, Ob, Oa, Ob], T.abb.vVVoOO, optimize=True)
            - 1.0 * np.einsum('MNIJ,aCBMNK->aBCIJK', X.ab.oooo[Oa, Ob, Oa, Ob], T.abb.vVVOOO, optimize=True)
    )
    dR.abb.vVVOOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('BCEf,aEfIJK->aBCIJK', X.bb.vvvv[Vb, Vb, Vb, vb], T.abb.vVvOOO, optimize=True)
            - 0.5 * np.einsum('BCEF,aFEIJK->aBCIJK', X.bb.vvvv[Vb, Vb, Vb, Vb], T.abb.vVVOOO, optimize=True)
    )
    dR.abb.vVVOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('aBef,eCfIJK->aBCIJK', X.ab.vvvv[va, Vb, va, vb], T.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('aBeF,eCFIJK->aBCIJK', X.ab.vvvv[va, Vb, va, Vb], T.abb.vVVOOO, optimize=True)
            - 1.0 * np.einsum('aBEf,ECfIJK->aBCIJK', X.ab.vvvv[va, Vb, Va, vb], T.abb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('aBEF,ECFIJK->aBCIJK', X.ab.vvvv[va, Vb, Va, Vb], T.abb.VVVOOO, optimize=True)
    )
    dR.abb.vVVOOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('amIe,eCBmJK->aBCIJK', X.aa.voov[va, oa, Oa, va], T.abb.vVVoOO, optimize=True)
            - 1.0 * np.einsum('aMIe,eCBMJK->aBCIJK', X.aa.voov[va, Oa, Oa, va], T.abb.vVVOOO, optimize=True)
            - 1.0 * np.einsum('amIE,ECBmJK->aBCIJK', X.aa.voov[va, oa, Oa, Va], T.abb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('aMIE,ECBMJK->aBCIJK', X.aa.voov[va, Oa, Oa, Va], T.abb.VVVOOO, optimize=True)
    )
    dR.abb.vVVOOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('amIe,CBemJK->aBCIJK', X.ab.voov[va, ob, Oa, vb], T.bbb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('aMIe,CBeMJK->aBCIJK', X.ab.voov[va, Ob, Oa, vb], T.bbb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('amIE,CBEmJK->aBCIJK', X.ab.voov[va, ob, Oa, Vb], T.bbb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('aMIE,CBEMJK->aBCIJK', X.ab.voov[va, Ob, Oa, Vb], T.bbb.VVVOOO, optimize=True)
    )
    dR.abb.vVVOOO += (4.0 / 4.0) * (
            +1.0 * np.einsum('mBeJ,eaCmIK->aBCIJK', X.ab.ovvo[oa, Vb, va, Ob], T.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('MBeJ,eaCIMK->aBCIJK', X.ab.ovvo[Oa, Vb, va, Ob], T.aab.vvVOOO, optimize=True)
            + 1.0 * np.einsum('mBEJ,EaCmIK->aBCIJK', X.ab.ovvo[oa, Vb, Va, Ob], T.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('MBEJ,EaCIMK->aBCIJK', X.ab.ovvo[Oa, Vb, Va, Ob], T.aab.VvVOOO, optimize=True)
    )
    dR.abb.vVVOOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('BmJe,aCeImK->aBCIJK', X.bb.voov[Vb, ob, Ob, vb], T.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('BMJe,aCeIMK->aBCIJK', X.bb.voov[Vb, Ob, Ob, vb], T.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('BmJE,aCEImK->aBCIJK', X.bb.voov[Vb, ob, Ob, Vb], T.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('BMJE,aCEIMK->aBCIJK', X.bb.voov[Vb, Ob, Ob, Vb], T.abb.vVVOOO, optimize=True)
    )
    dR.abb.vVVOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mBIe,aCemJK->aBCIJK', X.ab.ovov[oa, Vb, Oa, vb], T.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('MBIe,aCeMJK->aBCIJK', X.ab.ovov[Oa, Vb, Oa, vb], T.abb.vVvOOO, optimize=True)
            + 1.0 * np.einsum('mBIE,aCEmJK->aBCIJK', X.ab.ovov[oa, Vb, Oa, Vb], T.abb.vVVoOO, optimize=True)
            + 1.0 * np.einsum('MBIE,aCEMJK->aBCIJK', X.ab.ovov[Oa, Vb, Oa, Vb], T.abb.vVVOOO, optimize=True)
    )
    dR.abb.vVVOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('ameJ,eCBImK->aBCIJK', X.ab.vovo[va, ob, va, Ob], T.abb.vVVOoO, optimize=True)
            + 1.0 * np.einsum('aMeJ,eCBIMK->aBCIJK', X.ab.vovo[va, Ob, va, Ob], T.abb.vVVOOO, optimize=True)
            + 1.0 * np.einsum('amEJ,ECBImK->aBCIJK', X.ab.vovo[va, ob, Va, Ob], T.abb.VVVOoO, optimize=True)
            + 1.0 * np.einsum('aMEJ,ECBIMK->aBCIJK', X.ab.vovo[va, Ob, Va, Ob], T.abb.VVVOOO, optimize=True)
    )
    dR.abb.vVVOOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('mI,aCBmJK->aBCIJK', H.a.oo[oa, Oa], R.abb.vVVoOO, optimize=True)
            + 1.0 * np.einsum('MI,aCBMJK->aBCIJK', H.a.oo[Oa, Oa], R.abb.vVVOOO, optimize=True)
    )
    dR.abb.vVVOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mJ,aCBImK->aBCIJK', H.b.oo[ob, Ob], R.abb.vVVOoO, optimize=True)
            + 1.0 * np.einsum('MJ,aCBIMK->aBCIJK', H.b.oo[Ob, Ob], R.abb.vVVOOO, optimize=True)
    )
    dR.abb.vVVOOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('ae,eCBIJK->aBCIJK', H.a.vv[va, va], R.abb.vVVOOO, optimize=True)
            - 1.0 * np.einsum('aE,ECBIJK->aBCIJK', H.a.vv[va, Va], R.abb.VVVOOO, optimize=True)
    )
    dR.abb.vVVOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('Be,aCeIJK->aBCIJK', H.b.vv[Vb, vb], R.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('BE,aCEIJK->aBCIJK', H.b.vv[Vb, Vb], R.abb.vVVOOO, optimize=True)
    )
    dR.abb.vVVOOO += (1.0 / 4.0) * (
            -0.5 * np.einsum('mnJK,aCBImn->aBCIJK', H.bb.oooo[ob, ob, Ob, Ob], R.abb.vVVOoo, optimize=True)
            - 1.0 * np.einsum('mNJK,aCBImN->aBCIJK', H.bb.oooo[ob, Ob, Ob, Ob], R.abb.vVVOoO, optimize=True)
            - 0.5 * np.einsum('MNJK,aCBIMN->aBCIJK', H.bb.oooo[Ob, Ob, Ob, Ob], R.abb.vVVOOO, optimize=True)
    )
    dR.abb.vVVOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('mnIJ,aCBmnK->aBCIJK', H.ab.oooo[oa, ob, Oa, Ob], R.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('MnIJ,aCBMnK->aBCIJK', H.ab.oooo[Oa, ob, Oa, Ob], R.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('mNIJ,aCBmNK->aBCIJK', H.ab.oooo[oa, Ob, Oa, Ob], R.abb.vVVoOO, optimize=True)
            - 1.0 * np.einsum('MNIJ,aCBMNK->aBCIJK', H.ab.oooo[Oa, Ob, Oa, Ob], R.abb.vVVOOO, optimize=True)
    )
    dR.abb.vVVOOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('BCEf,aEfIJK->aBCIJK', H.bb.vvvv[Vb, Vb, Vb, vb], R.abb.vVvOOO, optimize=True)
            - 0.5 * np.einsum('BCEF,aFEIJK->aBCIJK', H.bb.vvvv[Vb, Vb, Vb, Vb], R.abb.vVVOOO, optimize=True)
    )
    dR.abb.vVVOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('aBef,eCfIJK->aBCIJK', H.ab.vvvv[va, Vb, va, vb], R.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('aBeF,eCFIJK->aBCIJK', H.ab.vvvv[va, Vb, va, Vb], R.abb.vVVOOO, optimize=True)
            - 1.0 * np.einsum('aBEf,ECfIJK->aBCIJK', H.ab.vvvv[va, Vb, Va, vb], R.abb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('aBEF,ECFIJK->aBCIJK', H.ab.vvvv[va, Vb, Va, Vb], R.abb.VVVOOO, optimize=True)
    )
    dR.abb.vVVOOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('amIe,eCBmJK->aBCIJK', H.aa.voov[va, oa, Oa, va], R.abb.vVVoOO, optimize=True)
            - 1.0 * np.einsum('aMIe,eCBMJK->aBCIJK', H.aa.voov[va, Oa, Oa, va], R.abb.vVVOOO, optimize=True)
            - 1.0 * np.einsum('amIE,ECBmJK->aBCIJK', H.aa.voov[va, oa, Oa, Va], R.abb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('aMIE,ECBMJK->aBCIJK', H.aa.voov[va, Oa, Oa, Va], R.abb.VVVOOO, optimize=True)
    )
    dR.abb.vVVOOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('amIe,CBemJK->aBCIJK', H.ab.voov[va, ob, Oa, vb], R.bbb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('aMIe,CBeMJK->aBCIJK', H.ab.voov[va, Ob, Oa, vb], R.bbb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('amIE,CBEmJK->aBCIJK', H.ab.voov[va, ob, Oa, Vb], R.bbb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('aMIE,CBEMJK->aBCIJK', H.ab.voov[va, Ob, Oa, Vb], R.bbb.VVVOOO, optimize=True)
    )
    dR.abb.vVVOOO += (4.0 / 4.0) * (
            +1.0 * np.einsum('mBeJ,eaCmIK->aBCIJK', H.ab.ovvo[oa, Vb, va, Ob], R.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('MBeJ,eaCIMK->aBCIJK', H.ab.ovvo[Oa, Vb, va, Ob], R.aab.vvVOOO, optimize=True)
            + 1.0 * np.einsum('mBEJ,EaCmIK->aBCIJK', H.ab.ovvo[oa, Vb, Va, Ob], R.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('MBEJ,EaCIMK->aBCIJK', H.ab.ovvo[Oa, Vb, Va, Ob], R.aab.VvVOOO, optimize=True)
    )
    dR.abb.vVVOOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('BmJe,aCeImK->aBCIJK', H.bb.voov[Vb, ob, Ob, vb], R.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('BMJe,aCeIMK->aBCIJK', H.bb.voov[Vb, Ob, Ob, vb], R.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('BmJE,aCEImK->aBCIJK', H.bb.voov[Vb, ob, Ob, Vb], R.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('BMJE,aCEIMK->aBCIJK', H.bb.voov[Vb, Ob, Ob, Vb], R.abb.vVVOOO, optimize=True)
    )
    dR.abb.vVVOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mBIe,aCemJK->aBCIJK', H.ab.ovov[oa, Vb, Oa, vb], R.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('MBIe,aCeMJK->aBCIJK', H.ab.ovov[Oa, Vb, Oa, vb], R.abb.vVvOOO, optimize=True)
            + 1.0 * np.einsum('mBIE,aCEmJK->aBCIJK', H.ab.ovov[oa, Vb, Oa, Vb], R.abb.vVVoOO, optimize=True)
            + 1.0 * np.einsum('MBIE,aCEMJK->aBCIJK', H.ab.ovov[Oa, Vb, Oa, Vb], R.abb.vVVOOO, optimize=True)
    )
    dR.abb.vVVOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('ameJ,eCBImK->aBCIJK', H.ab.vovo[va, ob, va, Ob], R.abb.vVVOoO, optimize=True)
            + 1.0 * np.einsum('aMeJ,eCBIMK->aBCIJK', H.ab.vovo[va, Ob, va, Ob], R.abb.vVVOOO, optimize=True)
            + 1.0 * np.einsum('amEJ,ECBImK->aBCIJK', H.ab.vovo[va, ob, Va, Ob], R.abb.VVVOoO, optimize=True)
            + 1.0 * np.einsum('aMEJ,ECBIMK->aBCIJK', H.ab.vovo[va, Ob, Va, Ob], R.abb.VVVOOO, optimize=True)
    )
    # of terms =  28

    dR.abb.vVVOOO -= np.transpose(dR.abb.vVVOOO, (0, 2, 1, 3, 4, 5))
    dR.abb.vVVOOO -= np.transpose(dR.abb.vVVOOO, (0, 1, 2, 3, 5, 4))

    return dR

def update(R, omega, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    R.abb.vVVOOO = eomcc_active_loops.update_r3c_011111(
        R.abb.vVVOOO,
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
