import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import cc_active_loops

def build(T, dT, H, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    # MM(2,3)
    dT.abb.vVvOOO = (1.0 / 2.0) * (
            +1.0 * np.einsum('aBIe,ecJK->aBcIJK', H.ab.vvov[va, Vb, Oa, :], T.bb[:, vb, Ob, Ob], optimize=True)
    )
    dT.abb.vVvOOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('acIe,eBJK->aBcIJK', H.ab.vvov[va, vb, Oa, :], T.bb[:, Vb, Ob, Ob], optimize=True)
    )
    dT.abb.vVvOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('amIJ,BcmK->aBcIJK', H.ab.vooo[va, :, Oa, Ob], T.bb[Vb, vb, :, Ob], optimize=True)
    )
    dT.abb.vVvOOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('cBKe,aeIJ->aBcIJK', H.bb.vvov[vb, Vb, Ob, :], T.ab[va, :, Oa, Ob], optimize=True)
    )
    dT.abb.vVvOOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('cmKJ,aBIm->aBcIJK', H.bb.vooo[vb, :, Ob, Ob], T.ab[va, Vb, Oa, :], optimize=True)
    )
    dT.abb.vVvOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('BmKJ,acIm->aBcIJK', H.bb.vooo[Vb, :, Ob, Ob], T.ab[va, vb, Oa, :], optimize=True)
    )
    dT.abb.vVvOOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('aBeJ,ecIK->aBcIJK', H.ab.vvvo[va, Vb, :, Ob], T.ab[:, vb, Oa, Ob], optimize=True)
    )
    dT.abb.vVvOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('aceJ,eBIK->aBcIJK', H.ab.vvvo[va, vb, :, Ob], T.ab[:, Vb, Oa, Ob], optimize=True)
    )
    dT.abb.vVvOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mBIJ,acmK->aBcIJK', H.ab.ovoo[:, Vb, Oa, Ob], T.ab[va, vb, :, Ob], optimize=True)
    )
    dT.abb.vVvOOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mcIJ,aBmK->aBcIJK', H.ab.ovoo[:, vb, Oa, Ob], T.ab[va, Vb, :, Ob], optimize=True)
    )
    # (H(2) * T3)_C
    dT.abb.vVvOOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mI,aBcmJK->aBcIJK', H.a.oo[oa, Oa], T.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('MI,aBcMJK->aBcIJK', H.a.oo[Oa, Oa], T.abb.vVvOOO, optimize=True)
    )
    dT.abb.vVvOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mJ,aBcImK->aBcIJK', H.b.oo[ob, Ob], T.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('MJ,aBcIMK->aBcIJK', H.b.oo[Ob, Ob], T.abb.vVvOOO, optimize=True)
    )
    dT.abb.vVvOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('ae,eBcIJK->aBcIJK', H.a.vv[va, va], T.abb.vVvOOO, optimize=True)
            + 1.0 * np.einsum('aE,EBcIJK->aBcIJK', H.a.vv[va, Va], T.abb.VVvOOO, optimize=True)
    )
    dT.abb.vVvOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('BE,aEcIJK->aBcIJK', H.b.vv[Vb, Vb], T.abb.vVvOOO, optimize=True)
    )
    dT.abb.vVvOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('ce,aBeIJK->aBcIJK', H.b.vv[vb, vb], T.abb.vVvOOO, optimize=True)
            + 1.0 * np.einsum('cE,aBEIJK->aBcIJK', H.b.vv[vb, Vb], T.abb.vVVOOO, optimize=True)
    )
    dT.abb.vVvOOO += (1.0 / 2.0) * (
            +0.5 * np.einsum('mnJK,aBcImn->aBcIJK', H.bb.oooo[ob, ob, Ob, Ob], T.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('mNJK,aBcImN->aBcIJK', H.bb.oooo[ob, Ob, Ob, Ob], T.abb.vVvOoO, optimize=True)
            + 0.5 * np.einsum('MNJK,aBcIMN->aBcIJK', H.bb.oooo[Ob, Ob, Ob, Ob], T.abb.vVvOOO, optimize=True)
    )
    dT.abb.vVvOOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mnIJ,aBcmnK->aBcIJK', H.ab.oooo[oa, ob, Oa, Ob], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('MnIJ,aBcMnK->aBcIJK', H.ab.oooo[Oa, ob, Oa, Ob], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('mNIJ,aBcmNK->aBcIJK', H.ab.oooo[oa, Ob, Oa, Ob], T.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('MNIJ,aBcMNK->aBcIJK', H.ab.oooo[Oa, Ob, Oa, Ob], T.abb.vVvOOO, optimize=True)
    )
    dT.abb.vVvOOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('BceF,aFeIJK->aBcIJK', H.bb.vvvv[Vb, vb, vb, Vb], T.abb.vVvOOO, optimize=True)
            - 0.5 * np.einsum('BcEF,aFEIJK->aBcIJK', H.bb.vvvv[Vb, vb, Vb, Vb], T.abb.vVVOOO, optimize=True)
    )
    dT.abb.vVvOOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('aBEf,EcfIJK->aBcIJK', H.ab.vvvv[va, Vb, Va, vb], T.abb.VvvOOO, optimize=True)
            + 1.0 * np.einsum('aBeF,eFcIJK->aBcIJK', H.ab.vvvv[va, Vb, va, Vb], T.abb.vVvOOO, optimize=True)
            + 1.0 * np.einsum('aBEF,EFcIJK->aBcIJK', H.ab.vvvv[va, Vb, Va, Vb], T.abb.VVvOOO, optimize=True)
    )
    dT.abb.vVvOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('acef,eBfIJK->aBcIJK', H.ab.vvvv[va, vb, va, vb], T.abb.vVvOOO, optimize=True)
            + 1.0 * np.einsum('acEf,EBfIJK->aBcIJK', H.ab.vvvv[va, vb, Va, vb], T.abb.VVvOOO, optimize=True)
            + 1.0 * np.einsum('aceF,eBFIJK->aBcIJK', H.ab.vvvv[va, vb, va, Vb], T.abb.vVVOOO, optimize=True)
            + 1.0 * np.einsum('acEF,EBFIJK->aBcIJK', H.ab.vvvv[va, vb, Va, Vb], T.abb.VVVOOO, optimize=True)
    )
    dT.abb.vVvOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('amIe,eBcmJK->aBcIJK', H.aa.voov[va, oa, Oa, va], T.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('amIE,EBcmJK->aBcIJK', H.aa.voov[va, oa, Oa, Va], T.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('aMIe,eBcMJK->aBcIJK', H.aa.voov[va, Oa, Oa, va], T.abb.vVvOOO, optimize=True)
            + 1.0 * np.einsum('aMIE,EBcMJK->aBcIJK', H.aa.voov[va, Oa, Oa, Va], T.abb.VVvOOO, optimize=True)
    )
    dT.abb.vVvOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('amIe,BcemJK->aBcIJK', H.ab.voov[va, ob, Oa, vb], T.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('amIE,BEcmJK->aBcIJK', H.ab.voov[va, ob, Oa, Vb], T.bbb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('aMIe,BceMJK->aBcIJK', H.ab.voov[va, Ob, Oa, vb], T.bbb.VvvOOO, optimize=True)
            - 1.0 * np.einsum('aMIE,BEcMJK->aBcIJK', H.ab.voov[va, Ob, Oa, Vb], T.bbb.VVvOOO, optimize=True)
    )
    dT.abb.vVvOOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mBEJ,EacmIK->aBcIJK', H.ab.ovvo[oa, Vb, Va, Ob], T.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('MBEJ,EacIMK->aBcIJK', H.ab.ovvo[Oa, Vb, Va, Ob], T.aab.VvvOOO, optimize=True)
    )
    dT.abb.vVvOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mceJ,eaBmIK->aBcIJK', H.ab.ovvo[oa, vb, va, Ob], T.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('mcEJ,EaBmIK->aBcIJK', H.ab.ovvo[oa, vb, Va, Ob], T.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('MceJ,eaBIMK->aBcIJK', H.ab.ovvo[Oa, vb, va, Ob], T.aab.vvVOOO, optimize=True)
            + 1.0 * np.einsum('McEJ,EaBIMK->aBcIJK', H.ab.ovvo[Oa, vb, Va, Ob], T.aab.VvVOOO, optimize=True)
    )
    dT.abb.vVvOOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('BmJE,aEcImK->aBcIJK', H.bb.voov[Vb, ob, Ob, Vb], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('BMJE,aEcIMK->aBcIJK', H.bb.voov[Vb, Ob, Ob, Vb], T.abb.vVvOOO, optimize=True)
    )
    dT.abb.vVvOOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('cmJe,aBeImK->aBcIJK', H.bb.voov[vb, ob, Ob, vb], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('cmJE,aBEImK->aBcIJK', H.bb.voov[vb, ob, Ob, Vb], T.abb.vVVOoO, optimize=True)
            + 1.0 * np.einsum('cMJe,aBeIMK->aBcIJK', H.bb.voov[vb, Ob, Ob, vb], T.abb.vVvOOO, optimize=True)
            + 1.0 * np.einsum('cMJE,aBEIMK->aBcIJK', H.bb.voov[vb, Ob, Ob, Vb], T.abb.vVVOOO, optimize=True)
    )
    dT.abb.vVvOOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mBIE,aEcmJK->aBcIJK', H.ab.ovov[oa, Vb, Oa, Vb], T.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('MBIE,aEcMJK->aBcIJK', H.ab.ovov[Oa, Vb, Oa, Vb], T.abb.vVvOOO, optimize=True)
    )
    dT.abb.vVvOOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mcIe,aBemJK->aBcIJK', H.ab.ovov[oa, vb, Oa, vb], T.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('mcIE,aBEmJK->aBcIJK', H.ab.ovov[oa, vb, Oa, Vb], T.abb.vVVoOO, optimize=True)
            - 1.0 * np.einsum('McIe,aBeMJK->aBcIJK', H.ab.ovov[Oa, vb, Oa, vb], T.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('McIE,aBEMJK->aBcIJK', H.ab.ovov[Oa, vb, Oa, Vb], T.abb.vVVOOO, optimize=True)
    )
    dT.abb.vVvOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('ameJ,eBcImK->aBcIJK', H.ab.vovo[va, ob, va, Ob], T.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('amEJ,EBcImK->aBcIJK', H.ab.vovo[va, ob, Va, Ob], T.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('aMeJ,eBcIMK->aBcIJK', H.ab.vovo[va, Ob, va, Ob], T.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('aMEJ,EBcIMK->aBcIJK', H.ab.vovo[va, Ob, Va, Ob], T.abb.VVvOOO, optimize=True)
    )

    dT.abb.vVvOOO -= np.transpose(dT.abb.vVvOOO, (0, 1, 2, 3, 5, 4))

    return dT


def update(T, dT, H, shift, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.abb.vVvOOO, dT.abb.vVvOOO = cc_active_loops.update_t3c_010111(
        T.abb.vVvOOO,
        dT.abb.vVvOOO,
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