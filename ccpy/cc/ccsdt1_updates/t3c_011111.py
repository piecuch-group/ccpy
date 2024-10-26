import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import cc_active_loops

def build(T, dT, H, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    # MM(2,3)
    dT.abb.vVVOOO = (2.0 / 4.0) * (
            +1.0 * np.einsum('aBIe,eCJK->aBCIJK', H.ab.vvov[va, Vb, Oa, :], T.bb[:, Vb, Ob, Ob], optimize=True)
    )
    dT.abb.vVVOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('amIJ,BCmK->aBCIJK', H.ab.vooo[va, :, Oa, Ob], T.bb[Vb, Vb, :, Ob], optimize=True)
    )
    dT.abb.vVVOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('CBKe,aeIJ->aBCIJK', H.bb.vvov[Vb, Vb, Ob, :], T.ab[va, :, Oa, Ob], optimize=True)
    )
    dT.abb.vVVOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('CmKJ,aBIm->aBCIJK', H.bb.vooo[Vb, :, Ob, Ob], T.ab[va, Vb, Oa, :], optimize=True)
    )
    dT.abb.vVVOOO += (4.0 / 4.0) * (
            +1.0 * np.einsum('aBeJ,eCIK->aBCIJK', H.ab.vvvo[va, Vb, :, Ob], T.ab[:, Vb, Oa, Ob], optimize=True)
    )
    dT.abb.vVVOOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('mBIJ,aCmK->aBCIJK', H.ab.ovoo[:, Vb, Oa, Ob], T.ab[va, Vb, :, Ob], optimize=True)
    )
    # (H(2) * T3)_C
    dT.abb.vVVOOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('mI,aCBmJK->aBCIJK', H.a.oo[oa, Oa], T.abb.vVVoOO, optimize=True)
            + 1.0 * np.einsum('MI,aCBMJK->aBCIJK', H.a.oo[Oa, Oa], T.abb.vVVOOO, optimize=True)
    )
    dT.abb.vVVOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mJ,aCBImK->aBCIJK', H.b.oo[ob, Ob], T.abb.vVVOoO, optimize=True)
            + 1.0 * np.einsum('MJ,aCBIMK->aBCIJK', H.b.oo[Ob, Ob], T.abb.vVVOOO, optimize=True)
    )
    dT.abb.vVVOOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('ae,eCBIJK->aBCIJK', H.a.vv[va, va], T.abb.vVVOOO, optimize=True)
            - 1.0 * np.einsum('aE,ECBIJK->aBCIJK', H.a.vv[va, Va], T.abb.VVVOOO, optimize=True)
    )
    dT.abb.vVVOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('Be,aCeIJK->aBCIJK', H.b.vv[Vb, vb], T.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('BE,aCEIJK->aBCIJK', H.b.vv[Vb, Vb], T.abb.vVVOOO, optimize=True)
    )
    dT.abb.vVVOOO += (1.0 / 4.0) * (
            -0.5 * np.einsum('mnJK,aCBImn->aBCIJK', H.bb.oooo[ob, ob, Ob, Ob], T.abb.vVVOoo, optimize=True)
            + 1.0 * np.einsum('MnJK,aCBInM->aBCIJK', H.bb.oooo[Ob, ob, Ob, Ob], T.abb.vVVOoO, optimize=True)
            - 0.5 * np.einsum('MNJK,aCBIMN->aBCIJK', H.bb.oooo[Ob, Ob, Ob, Ob], T.abb.vVVOOO, optimize=True)
    )
    dT.abb.vVVOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('mnIJ,aCBmnK->aBCIJK', H.ab.oooo[oa, ob, Oa, Ob], T.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('mNIJ,aCBmNK->aBCIJK', H.ab.oooo[oa, Ob, Oa, Ob], T.abb.vVVoOO, optimize=True)
            - 1.0 * np.einsum('MnIJ,aCBMnK->aBCIJK', H.ab.oooo[Oa, ob, Oa, Ob], T.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('MNIJ,aCBMNK->aBCIJK', H.ab.oooo[Oa, Ob, Oa, Ob], T.abb.vVVOOO, optimize=True)
    )
    dT.abb.vVVOOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('BCEf,aEfIJK->aBCIJK', H.bb.vvvv[Vb, Vb, Vb, vb], T.abb.vVvOOO, optimize=True)
            - 0.5 * np.einsum('BCEF,aFEIJK->aBCIJK', H.bb.vvvv[Vb, Vb, Vb, Vb], T.abb.vVVOOO, optimize=True)
    )
    dT.abb.vVVOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('aBef,eCfIJK->aBCIJK', H.ab.vvvv[va, Vb, va, vb], T.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('aBeF,eCFIJK->aBCIJK', H.ab.vvvv[va, Vb, va, Vb], T.abb.vVVOOO, optimize=True)
            - 1.0 * np.einsum('aBEf,ECfIJK->aBCIJK', H.ab.vvvv[va, Vb, Va, vb], T.abb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('aBEF,ECFIJK->aBCIJK', H.ab.vvvv[va, Vb, Va, Vb], T.abb.VVVOOO, optimize=True)
    )
    dT.abb.vVVOOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('amIe,eCBmJK->aBCIJK', H.aa.voov[va, oa, Oa, va], T.abb.vVVoOO, optimize=True)
            - 1.0 * np.einsum('amIE,ECBmJK->aBCIJK', H.aa.voov[va, oa, Oa, Va], T.abb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('aMIe,eCBMJK->aBCIJK', H.aa.voov[va, Oa, Oa, va], T.abb.vVVOOO, optimize=True)
            - 1.0 * np.einsum('aMIE,ECBMJK->aBCIJK', H.aa.voov[va, Oa, Oa, Va], T.abb.VVVOOO, optimize=True)
    )
    dT.abb.vVVOOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('amIe,CBemJK->aBCIJK', H.ab.voov[va, ob, Oa, vb], T.bbb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('amIE,CBEmJK->aBCIJK', H.ab.voov[va, ob, Oa, Vb], T.bbb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('aMIe,CBeMJK->aBCIJK', H.ab.voov[va, Ob, Oa, vb], T.bbb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('aMIE,CBEMJK->aBCIJK', H.ab.voov[va, Ob, Oa, Vb], T.bbb.VVVOOO, optimize=True)
    )
    dT.abb.vVVOOO += (4.0 / 4.0) * (
            +1.0 * np.einsum('mBeJ,eaCmIK->aBCIJK', H.ab.ovvo[oa, Vb, va, Ob], T.aab.vvVoOO, optimize=True)
            + 1.0 * np.einsum('mBEJ,EaCmIK->aBCIJK', H.ab.ovvo[oa, Vb, Va, Ob], T.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('MBeJ,eaCIMK->aBCIJK', H.ab.ovvo[Oa, Vb, va, Ob], T.aab.vvVOOO, optimize=True)
            - 1.0 * np.einsum('MBEJ,EaCIMK->aBCIJK', H.ab.ovvo[Oa, Vb, Va, Ob], T.aab.VvVOOO, optimize=True)
    )
    dT.abb.vVVOOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('BmJe,aCeImK->aBCIJK', H.bb.voov[Vb, ob, Ob, vb], T.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('BmJE,aCEImK->aBCIJK', H.bb.voov[Vb, ob, Ob, Vb], T.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('BMJe,aCeIMK->aBCIJK', H.bb.voov[Vb, Ob, Ob, vb], T.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('BMJE,aCEIMK->aBCIJK', H.bb.voov[Vb, Ob, Ob, Vb], T.abb.vVVOOO, optimize=True)
    )
    dT.abb.vVVOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mBIe,aCemJK->aBCIJK', H.ab.ovov[oa, Vb, Oa, vb], T.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('mBIE,aCEmJK->aBCIJK', H.ab.ovov[oa, Vb, Oa, Vb], T.abb.vVVoOO, optimize=True)
            + 1.0 * np.einsum('MBIe,aCeMJK->aBCIJK', H.ab.ovov[Oa, Vb, Oa, vb], T.abb.vVvOOO, optimize=True)
            + 1.0 * np.einsum('MBIE,aCEMJK->aBCIJK', H.ab.ovov[Oa, Vb, Oa, Vb], T.abb.vVVOOO, optimize=True)
    )
    dT.abb.vVVOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('ameJ,eCBImK->aBCIJK', H.ab.vovo[va, ob, va, Ob], T.abb.vVVOoO, optimize=True)
            + 1.0 * np.einsum('amEJ,ECBImK->aBCIJK', H.ab.vovo[va, ob, Va, Ob], T.abb.VVVOoO, optimize=True)
            + 1.0 * np.einsum('aMeJ,eCBIMK->aBCIJK', H.ab.vovo[va, Ob, va, Ob], T.abb.vVVOOO, optimize=True)
            + 1.0 * np.einsum('aMEJ,ECBIMK->aBCIJK', H.ab.vovo[va, Ob, Va, Ob], T.abb.VVVOOO, optimize=True)
    )

    dT.abb.vVVOOO -= np.transpose(dT.abb.vVVOOO, (0, 2, 1, 3, 4, 5))
    dT.abb.vVVOOO -= np.transpose(dT.abb.vVVOOO, (0, 1, 2, 3, 5, 4))

    return dT


def update(T, dT, H, shift, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.abb.vVVOOO, dT.abb.vVVOOO = cc_active_loops.update_t3c_011111(
        T.abb.vVVOOO,
        dT.abb.vVVOOO,
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