import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import cc_active_loops

def build(T, dT, H, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    # MM(2,3)
    dT.abb.vVvOoO = (1.0 / 1.0) * (
            +1.0 * np.einsum('aBIe,ecjK->aBcIjK', H.ab.vvov[va, Vb, Oa, :], T.bb[:, vb, ob, Ob], optimize=True)
    )
    dT.abb.vVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('acIe,eBjK->aBcIjK', H.ab.vvov[va, vb, Oa, :], T.bb[:, Vb, ob, Ob], optimize=True)
    )
    dT.abb.vVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('amIj,BcmK->aBcIjK', H.ab.vooo[va, :, Oa, ob], T.bb[Vb, vb, :, Ob], optimize=True)
    )
    dT.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('amIK,Bcmj->aBcIjK', H.ab.vooo[va, :, Oa, Ob], T.bb[Vb, vb, :, ob], optimize=True)
    )
    dT.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('cBKe,aeIj->aBcIjK', H.bb.vvov[vb, Vb, Ob, :], T.ab[va, :, Oa, ob], optimize=True)
    )
    dT.abb.vVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('cBje,aeIK->aBcIjK', H.bb.vvov[vb, Vb, ob, :], T.ab[va, :, Oa, Ob], optimize=True)
    )
    dT.abb.vVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('cmKj,aBIm->aBcIjK', H.bb.vooo[vb, :, Ob, ob], T.ab[va, Vb, Oa, :], optimize=True)
    )
    dT.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('BmKj,acIm->aBcIjK', H.bb.vooo[Vb, :, Ob, ob], T.ab[va, vb, Oa, :], optimize=True)
    )
    dT.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('aBej,ecIK->aBcIjK', H.ab.vvvo[va, Vb, :, ob], T.ab[:, vb, Oa, Ob], optimize=True)
    )
    dT.abb.vVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('acej,eBIK->aBcIjK', H.ab.vvvo[va, vb, :, ob], T.ab[:, Vb, Oa, Ob], optimize=True)
    )
    dT.abb.vVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('aBeK,ecIj->aBcIjK', H.ab.vvvo[va, Vb, :, Ob], T.ab[:, vb, Oa, ob], optimize=True)
    )
    dT.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('aceK,eBIj->aBcIjK', H.ab.vvvo[va, vb, :, Ob], T.ab[:, Vb, Oa, ob], optimize=True)
    )
    dT.abb.vVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mBIj,acmK->aBcIjK', H.ab.ovoo[:, Vb, Oa, ob], T.ab[va, vb, :, Ob], optimize=True)
    )
    dT.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mcIj,aBmK->aBcIjK', H.ab.ovoo[:, vb, Oa, ob], T.ab[va, Vb, :, Ob], optimize=True)
    )
    dT.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mBIK,acmj->aBcIjK', H.ab.ovoo[:, Vb, Oa, Ob], T.ab[va, vb, :, ob], optimize=True)
    )
    dT.abb.vVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mcIK,aBmj->aBcIjK', H.ab.ovoo[:, vb, Oa, Ob], T.ab[va, Vb, :, ob], optimize=True)
    )
    # (H(2) * T3)_C
    dT.abb.vVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mI,aBcmjK->aBcIjK', H.a.oo[oa, Oa], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('MI,aBcMjK->aBcIjK', H.a.oo[Oa, Oa], T.abb.vVvOoO, optimize=True)
    )
    dT.abb.vVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mj,aBcImK->aBcIjK', H.b.oo[ob, ob], T.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('Mj,aBcIMK->aBcIjK', H.b.oo[Ob, ob], T.abb.vVvOOO, optimize=True)
    )
    dT.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mK,aBcImj->aBcIjK', H.b.oo[ob, Ob], T.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('MK,aBcIjM->aBcIjK', H.b.oo[Ob, Ob], T.abb.vVvOoO, optimize=True)
    )
    dT.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('ae,eBcIjK->aBcIjK', H.a.vv[va, va], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('aE,EBcIjK->aBcIjK', H.a.vv[va, Va], T.abb.VVvOoO, optimize=True)
    )
    dT.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('BE,aEcIjK->aBcIjK', H.b.vv[Vb, Vb], T.abb.vVvOoO, optimize=True)
    )
    dT.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('ce,aBeIjK->aBcIjK', H.b.vv[vb, vb], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('cE,aBEIjK->aBcIjK', H.b.vv[vb, Vb], T.abb.vVVOoO, optimize=True)
    )
    dT.abb.vVvOoO += (1.0 / 1.0) * (
            +0.5 * np.einsum('mnjK,aBcImn->aBcIjK', H.bb.oooo[ob, ob, ob, Ob], T.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('mNjK,aBcImN->aBcIjK', H.bb.oooo[ob, Ob, ob, Ob], T.abb.vVvOoO, optimize=True)
            + 0.5 * np.einsum('MNjK,aBcIMN->aBcIjK', H.bb.oooo[Ob, Ob, ob, Ob], T.abb.vVvOOO, optimize=True)
    )
    dT.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mnIj,aBcmnK->aBcIjK', H.ab.oooo[oa, ob, Oa, ob], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('MnIj,aBcMnK->aBcIjK', H.ab.oooo[Oa, ob, Oa, ob], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('mNIj,aBcmNK->aBcIjK', H.ab.oooo[oa, Ob, Oa, ob], T.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('MNIj,aBcMNK->aBcIjK', H.ab.oooo[Oa, Ob, Oa, ob], T.abb.vVvOOO, optimize=True)
    )
    dT.abb.vVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('MnIK,aBcMnj->aBcIjK', H.ab.oooo[Oa, ob, Oa, Ob], T.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('mNIK,aBcmjN->aBcIjK', H.ab.oooo[oa, Ob, Oa, Ob], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('MNIK,aBcMjN->aBcIjK', H.ab.oooo[Oa, Ob, Oa, Ob], T.abb.vVvOoO, optimize=True)
    )
    dT.abb.vVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('BceF,aFeIjK->aBcIjK', H.bb.vvvv[Vb, vb, vb, Vb], T.abb.vVvOoO, optimize=True)
            - 0.5 * np.einsum('BcEF,aFEIjK->aBcIjK', H.bb.vvvv[Vb, vb, Vb, Vb], T.abb.vVVOoO, optimize=True)
    )
    dT.abb.vVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('aBEf,EcfIjK->aBcIjK', H.ab.vvvv[va, Vb, Va, vb], T.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('aBeF,eFcIjK->aBcIjK', H.ab.vvvv[va, Vb, va, Vb], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('aBEF,EFcIjK->aBcIjK', H.ab.vvvv[va, Vb, Va, Vb], T.abb.VVvOoO, optimize=True)
    )
    dT.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('acef,eBfIjK->aBcIjK', H.ab.vvvv[va, vb, va, vb], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('acEf,EBfIjK->aBcIjK', H.ab.vvvv[va, vb, Va, vb], T.abb.VVvOoO, optimize=True)
            + 1.0 * np.einsum('aceF,eBFIjK->aBcIjK', H.ab.vvvv[va, vb, va, Vb], T.abb.vVVOoO, optimize=True)
            + 1.0 * np.einsum('acEF,EBFIjK->aBcIjK', H.ab.vvvv[va, vb, Va, Vb], T.abb.VVVOoO, optimize=True)
    )
    dT.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('amIe,eBcmjK->aBcIjK', H.aa.voov[va, oa, Oa, va], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('aMIe,eBcMjK->aBcIjK', H.aa.voov[va, Oa, Oa, va], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('amIE,EBcmjK->aBcIjK', H.aa.voov[va, oa, Oa, Va], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('aMIE,EBcMjK->aBcIjK', H.aa.voov[va, Oa, Oa, Va], T.abb.VVvOoO, optimize=True)
    )
    dT.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('amIe,BcemjK->aBcIjK', H.ab.voov[va, ob, Oa, vb], T.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('aMIe,BcejMK->aBcIjK', H.ab.voov[va, Ob, Oa, vb], T.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('amIE,BEcmjK->aBcIjK', H.ab.voov[va, ob, Oa, Vb], T.bbb.VVvooO, optimize=True)
            + 1.0 * np.einsum('aMIE,BEcjMK->aBcIjK', H.ab.voov[va, Ob, Oa, Vb], T.bbb.VVvoOO, optimize=True)
    )
    dT.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mBEj,EacmIK->aBcIjK', H.ab.ovvo[oa, Vb, Va, ob], T.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('MBEj,EacIMK->aBcIjK', H.ab.ovvo[Oa, Vb, Va, ob], T.aab.VvvOOO, optimize=True)
    )
    dT.abb.vVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mcej,eaBmIK->aBcIjK', H.ab.ovvo[oa, vb, va, ob], T.aab.vvVoOO, optimize=True)
            + 1.0 * np.einsum('Mcej,eaBIMK->aBcIjK', H.ab.ovvo[Oa, vb, va, ob], T.aab.vvVOOO, optimize=True)
            - 1.0 * np.einsum('mcEj,EaBmIK->aBcIjK', H.ab.ovvo[oa, vb, Va, ob], T.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('McEj,EaBIMK->aBcIjK', H.ab.ovvo[Oa, vb, Va, ob], T.aab.VvVOOO, optimize=True)
    )
    dT.abb.vVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mBEK,EacmIj->aBcIjK', H.ab.ovvo[oa, Vb, Va, Ob], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('MBEK,EacIMj->aBcIjK', H.ab.ovvo[Oa, Vb, Va, Ob], T.aab.VvvOOo, optimize=True)
    )
    dT.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mceK,eaBmIj->aBcIjK', H.ab.ovvo[oa, vb, va, Ob], T.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('MceK,eaBIMj->aBcIjK', H.ab.ovvo[Oa, vb, va, Ob], T.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('mcEK,EaBmIj->aBcIjK', H.ab.ovvo[oa, vb, Va, Ob], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('McEK,EaBIMj->aBcIjK', H.ab.ovvo[Oa, vb, Va, Ob], T.aab.VvVOOo, optimize=True)
    )
    dT.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('BmjE,aEcImK->aBcIjK', H.bb.voov[Vb, ob, ob, Vb], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('BMjE,aEcIMK->aBcIjK', H.bb.voov[Vb, Ob, ob, Vb], T.abb.vVvOOO, optimize=True)
    )
    dT.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('cmje,aBeImK->aBcIjK', H.bb.voov[vb, ob, ob, vb], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('cMje,aBeIMK->aBcIjK', H.bb.voov[vb, Ob, ob, vb], T.abb.vVvOOO, optimize=True)
            + 1.0 * np.einsum('cmjE,aBEImK->aBcIjK', H.bb.voov[vb, ob, ob, Vb], T.abb.vVVOoO, optimize=True)
            + 1.0 * np.einsum('cMjE,aBEIMK->aBcIjK', H.bb.voov[vb, Ob, ob, Vb], T.abb.vVVOOO, optimize=True)
    )
    dT.abb.vVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('BmKE,aEcImj->aBcIjK', H.bb.voov[Vb, ob, Ob, Vb], T.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('BMKE,aEcIjM->aBcIjK', H.bb.voov[Vb, Ob, Ob, Vb], T.abb.vVvOoO, optimize=True)
    )
    dT.abb.vVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('cmKe,aBeImj->aBcIjK', H.bb.voov[vb, ob, Ob, vb], T.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('cMKe,aBeIjM->aBcIjK', H.bb.voov[vb, Ob, Ob, vb], T.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('cmKE,aBEImj->aBcIjK', H.bb.voov[vb, ob, Ob, Vb], T.abb.vVVOoo, optimize=True)
            + 1.0 * np.einsum('cMKE,aBEIjM->aBcIjK', H.bb.voov[vb, Ob, Ob, Vb], T.abb.vVVOoO, optimize=True)
    )
    dT.abb.vVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mBIE,aEcmjK->aBcIjK', H.ab.ovov[oa, Vb, Oa, Vb], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('MBIE,aEcMjK->aBcIjK', H.ab.ovov[Oa, Vb, Oa, Vb], T.abb.vVvOoO, optimize=True)
    )
    dT.abb.vVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mcIe,aBemjK->aBcIjK', H.ab.ovov[oa, vb, Oa, vb], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('McIe,aBeMjK->aBcIjK', H.ab.ovov[Oa, vb, Oa, vb], T.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('mcIE,aBEmjK->aBcIjK', H.ab.ovov[oa, vb, Oa, Vb], T.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('McIE,aBEMjK->aBcIjK', H.ab.ovov[Oa, vb, Oa, Vb], T.abb.vVVOoO, optimize=True)
    )
    dT.abb.vVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('amej,eBcImK->aBcIjK', H.ab.vovo[va, ob, va, ob], T.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('aMej,eBcIMK->aBcIjK', H.ab.vovo[va, Ob, va, ob], T.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('amEj,EBcImK->aBcIjK', H.ab.vovo[va, ob, Va, ob], T.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('aMEj,EBcIMK->aBcIjK', H.ab.vovo[va, Ob, Va, ob], T.abb.VVvOOO, optimize=True)
    )
    dT.abb.vVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('ameK,eBcImj->aBcIjK', H.ab.vovo[va, ob, va, Ob], T.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('aMeK,eBcIjM->aBcIjK', H.ab.vovo[va, Ob, va, Ob], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('amEK,EBcImj->aBcIjK', H.ab.vovo[va, ob, Va, Ob], T.abb.VVvOoo, optimize=True)
            - 1.0 * np.einsum('aMEK,EBcIjM->aBcIjK', H.ab.vovo[va, Ob, Va, Ob], T.abb.VVvOoO, optimize=True)
    )

    return dT


def update(T, dT, H, shift, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.abb.vVvOoO, dT.abb.vVvOoO = cc_active_loops.update_t3c_010101(
        T.abb.vVvOoO,
        dT.abb.vVvOoO,
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