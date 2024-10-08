import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import cc_active_loops

def build(T, dT, H, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    # MM(2,3)
    dT.abb.vVVOoO = (2.0 / 2.0) * (
            +1.0 * np.einsum('aBIe,eCjK->aBCIjK', H.ab.vvov[va, Vb, Oa, :], T.bb[:, Vb, ob, Ob], optimize=True)
    )
    dT.abb.vVVOoO += (1.0 / 2.0) * (
            -1.0 * np.einsum('amIj,BCmK->aBCIjK', H.ab.vooo[va, :, Oa, ob], T.bb[Vb, Vb, :, Ob], optimize=True)
    )
    dT.abb.vVVOoO += (1.0 / 2.0) * (
            +1.0 * np.einsum('amIK,BCmj->aBCIjK', H.ab.vooo[va, :, Oa, Ob], T.bb[Vb, Vb, :, ob], optimize=True)
    )
    dT.abb.vVVOoO += (1.0 / 2.0) * (
            +1.0 * np.einsum('CBKe,aeIj->aBCIjK', H.bb.vvov[Vb, Vb, Ob, :], T.ab[va, :, Oa, ob], optimize=True)
    )
    dT.abb.vVVOoO += (1.0 / 2.0) * (
            -1.0 * np.einsum('CBje,aeIK->aBCIjK', H.bb.vvov[Vb, Vb, ob, :], T.ab[va, :, Oa, Ob], optimize=True)
    )
    dT.abb.vVVOoO += (2.0 / 2.0) * (
            -1.0 * np.einsum('CmKj,aBIm->aBCIjK', H.bb.vooo[Vb, :, Ob, ob], T.ab[va, Vb, Oa, :], optimize=True)
    )
    dT.abb.vVVOoO += (2.0 / 2.0) * (
            +1.0 * np.einsum('aBej,eCIK->aBCIjK', H.ab.vvvo[va, Vb, :, ob], T.ab[:, Vb, Oa, Ob], optimize=True)
    )
    dT.abb.vVVOoO += (2.0 / 2.0) * (
            -1.0 * np.einsum('aBeK,eCIj->aBCIjK', H.ab.vvvo[va, Vb, :, Ob], T.ab[:, Vb, Oa, ob], optimize=True)
    )
    dT.abb.vVVOoO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mBIj,aCmK->aBCIjK', H.ab.ovoo[:, Vb, Oa, ob], T.ab[va, Vb, :, Ob], optimize=True)
    )
    dT.abb.vVVOoO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mBIK,aCmj->aBCIjK', H.ab.ovoo[:, Vb, Oa, Ob], T.ab[va, Vb, :, ob], optimize=True)
    )
    # (H(2) * T3)_C
    dT.abb.vVVOoO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mI,aCBmjK->aBCIjK', H.a.oo[oa, Oa], T.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('MI,aCBMjK->aBCIjK', H.a.oo[Oa, Oa], T.abb.vVVOoO, optimize=True)
    )
    dT.abb.vVVOoO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mj,aCBImK->aBCIjK', H.b.oo[ob, ob], T.abb.vVVOoO, optimize=True)
            + 1.0 * np.einsum('Mj,aCBIMK->aBCIjK', H.b.oo[Ob, ob], T.abb.vVVOOO, optimize=True)
    )
    dT.abb.vVVOoO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mK,aCBImj->aBCIjK', H.b.oo[ob, Ob], T.abb.vVVOoo, optimize=True)
            + 1.0 * np.einsum('MK,aCBIjM->aBCIjK', H.b.oo[Ob, Ob], T.abb.vVVOoO, optimize=True)
    )
    dT.abb.vVVOoO += (1.0 / 2.0) * (
            -1.0 * np.einsum('ae,eCBIjK->aBCIjK', H.a.vv[va, va], T.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('aE,ECBIjK->aBCIjK', H.a.vv[va, Va], T.abb.VVVOoO, optimize=True)
    )
    dT.abb.vVVOoO += (2.0 / 2.0) * (
            -1.0 * np.einsum('Be,aCeIjK->aBCIjK', H.b.vv[Vb, vb], T.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('BE,aCEIjK->aBCIjK', H.b.vv[Vb, Vb], T.abb.vVVOoO, optimize=True)
    )
    dT.abb.vVVOoO += (1.0 / 2.0) * (
            -0.5 * np.einsum('mnjK,aCBImn->aBCIjK', H.bb.oooo[ob, ob, ob, Ob], T.abb.vVVOoo, optimize=True)
            - 1.0 * np.einsum('mNjK,aCBImN->aBCIjK', H.bb.oooo[ob, Ob, ob, Ob], T.abb.vVVOoO, optimize=True)
            - 0.5 * np.einsum('MNjK,aCBIMN->aBCIjK', H.bb.oooo[Ob, Ob, ob, Ob], T.abb.vVVOOO, optimize=True)
    )
    dT.abb.vVVOoO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mnIj,aCBmnK->aBCIjK', H.ab.oooo[oa, ob, Oa, ob], T.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('MnIj,aCBMnK->aBCIjK', H.ab.oooo[Oa, ob, Oa, ob], T.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('mNIj,aCBmNK->aBCIjK', H.ab.oooo[oa, Ob, Oa, ob], T.abb.vVVoOO, optimize=True)
            - 1.0 * np.einsum('MNIj,aCBMNK->aBCIjK', H.ab.oooo[Oa, Ob, Oa, ob], T.abb.vVVOOO, optimize=True)
    )
    dT.abb.vVVOoO += (1.0 / 2.0) * (
            +1.0 * np.einsum('MnIK,aCBMnj->aBCIjK', H.ab.oooo[Oa, ob, Oa, Ob], T.abb.vVVOoo, optimize=True)
            - 1.0 * np.einsum('mNIK,aCBmjN->aBCIjK', H.ab.oooo[oa, Ob, Oa, Ob], T.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('MNIK,aCBMjN->aBCIjK', H.ab.oooo[Oa, Ob, Oa, Ob], T.abb.vVVOoO, optimize=True)
    )
    dT.abb.vVVOoO += (1.0 / 2.0) * (
            +1.0 * np.einsum('BCEf,aEfIjK->aBCIjK', H.bb.vvvv[Vb, Vb, Vb, vb], T.abb.vVvOoO, optimize=True)
            - 0.5 * np.einsum('BCEF,aFEIjK->aBCIjK', H.bb.vvvv[Vb, Vb, Vb, Vb], T.abb.vVVOoO, optimize=True)
    )
    dT.abb.vVVOoO += (2.0 / 2.0) * (
            -1.0 * np.einsum('aBef,eCfIjK->aBCIjK', H.ab.vvvv[va, Vb, va, vb], T.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('aBeF,eCFIjK->aBCIjK', H.ab.vvvv[va, Vb, va, Vb], T.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('aBEf,ECfIjK->aBCIjK', H.ab.vvvv[va, Vb, Va, vb], T.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('aBEF,ECFIjK->aBCIjK', H.ab.vvvv[va, Vb, Va, Vb], T.abb.VVVOoO, optimize=True)
    )
    dT.abb.vVVOoO += (1.0 / 2.0) * (
            -1.0 * np.einsum('amIe,eCBmjK->aBCIjK', H.aa.voov[va, oa, Oa, va], T.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('aMIe,eCBMjK->aBCIjK', H.aa.voov[va, Oa, Oa, va], T.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('amIE,ECBmjK->aBCIjK', H.aa.voov[va, oa, Oa, Va], T.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('aMIE,ECBMjK->aBCIjK', H.aa.voov[va, Oa, Oa, Va], T.abb.VVVOoO, optimize=True)
    )
    dT.abb.vVVOoO += (1.0 / 2.0) * (
            -1.0 * np.einsum('amIe,CBemjK->aBCIjK', H.ab.voov[va, ob, Oa, vb], T.bbb.VVvooO, optimize=True)
            + 1.0 * np.einsum('aMIe,CBejMK->aBCIjK', H.ab.voov[va, Ob, Oa, vb], T.bbb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('amIE,CBEmjK->aBCIjK', H.ab.voov[va, ob, Oa, Vb], T.bbb.VVVooO, optimize=True)
            + 1.0 * np.einsum('aMIE,CBEjMK->aBCIjK', H.ab.voov[va, Ob, Oa, Vb], T.bbb.VVVoOO, optimize=True)
    )
    dT.abb.vVVOoO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mBej,eaCmIK->aBCIjK', H.ab.ovvo[oa, Vb, va, ob], T.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('MBej,eaCIMK->aBCIjK', H.ab.ovvo[Oa, Vb, va, ob], T.aab.vvVOOO, optimize=True)
            + 1.0 * np.einsum('mBEj,EaCmIK->aBCIjK', H.ab.ovvo[oa, Vb, Va, ob], T.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('MBEj,EaCIMK->aBCIjK', H.ab.ovvo[Oa, Vb, Va, ob], T.aab.VvVOOO, optimize=True)
    )
    dT.abb.vVVOoO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mBeK,eaCmIj->aBCIjK', H.ab.ovvo[oa, Vb, va, Ob], T.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('MBeK,eaCIMj->aBCIjK', H.ab.ovvo[Oa, Vb, va, Ob], T.aab.vvVOOo, optimize=True)
            - 1.0 * np.einsum('mBEK,EaCmIj->aBCIjK', H.ab.ovvo[oa, Vb, Va, Ob], T.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('MBEK,EaCIMj->aBCIjK', H.ab.ovvo[Oa, Vb, Va, Ob], T.aab.VvVOOo, optimize=True)
    )
    dT.abb.vVVOoO += (2.0 / 2.0) * (
            -1.0 * np.einsum('Bmje,aCeImK->aBCIjK', H.bb.voov[Vb, ob, ob, vb], T.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('BMje,aCeIMK->aBCIjK', H.bb.voov[Vb, Ob, ob, vb], T.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('BmjE,aCEImK->aBCIjK', H.bb.voov[Vb, ob, ob, Vb], T.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('BMjE,aCEIMK->aBCIjK', H.bb.voov[Vb, Ob, ob, Vb], T.abb.vVVOOO, optimize=True)
    )
    dT.abb.vVVOoO += (2.0 / 2.0) * (
            +1.0 * np.einsum('BmKe,aCeImj->aBCIjK', H.bb.voov[Vb, ob, Ob, vb], T.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('BMKe,aCeIjM->aBCIjK', H.bb.voov[Vb, Ob, Ob, vb], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('BmKE,aCEImj->aBCIjK', H.bb.voov[Vb, ob, Ob, Vb], T.abb.vVVOoo, optimize=True)
            - 1.0 * np.einsum('BMKE,aCEIjM->aBCIjK', H.bb.voov[Vb, Ob, Ob, Vb], T.abb.vVVOoO, optimize=True)
    )
    dT.abb.vVVOoO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mBIe,aCemjK->aBCIjK', H.ab.ovov[oa, Vb, Oa, vb], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('MBIe,aCeMjK->aBCIjK', H.ab.ovov[Oa, Vb, Oa, vb], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('mBIE,aCEmjK->aBCIjK', H.ab.ovov[oa, Vb, Oa, Vb], T.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('MBIE,aCEMjK->aBCIjK', H.ab.ovov[Oa, Vb, Oa, Vb], T.abb.vVVOoO, optimize=True)
    )
    dT.abb.vVVOoO += (1.0 / 2.0) * (
            +1.0 * np.einsum('amej,eCBImK->aBCIjK', H.ab.vovo[va, ob, va, ob], T.abb.vVVOoO, optimize=True)
            + 1.0 * np.einsum('aMej,eCBIMK->aBCIjK', H.ab.vovo[va, Ob, va, ob], T.abb.vVVOOO, optimize=True)
            + 1.0 * np.einsum('amEj,ECBImK->aBCIjK', H.ab.vovo[va, ob, Va, ob], T.abb.VVVOoO, optimize=True)
            + 1.0 * np.einsum('aMEj,ECBIMK->aBCIjK', H.ab.vovo[va, Ob, Va, ob], T.abb.VVVOOO, optimize=True)
    )
    dT.abb.vVVOoO += (1.0 / 2.0) * (
            -1.0 * np.einsum('ameK,eCBImj->aBCIjK', H.ab.vovo[va, ob, va, Ob], T.abb.vVVOoo, optimize=True)
            + 1.0 * np.einsum('aMeK,eCBIjM->aBCIjK', H.ab.vovo[va, Ob, va, Ob], T.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('amEK,ECBImj->aBCIjK', H.ab.vovo[va, ob, Va, Ob], T.abb.VVVOoo, optimize=True)
            + 1.0 * np.einsum('aMEK,ECBIjM->aBCIjK', H.ab.vovo[va, Ob, Va, Ob], T.abb.VVVOoO, optimize=True)
    )

    dT.abb.vVVOoO -= np.transpose(dT.abb.vVVOoO, (0, 2, 1, 3, 4, 5))

    return dT


def update(T, dT, H, shift, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.abb.vVVOoO, dT.abb.vVVOoO = cc_active_loops.update_t3c_011101(
        T.abb.vVVOoO,
        dT.abb.vVVOoO,
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