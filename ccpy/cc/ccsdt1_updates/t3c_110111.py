import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import cc_active_loops

def build(T, dT, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    # MM(2,3)
    dT.abb.VVvOOO = (1.0 / 2.0) * (
            +1.0 * np.einsum('ABIe,ecJK->ABcIJK', H.ab.vvov[Va, Vb, Oa, :], T.bb[:, vb, Ob, Ob], optimize=True)
    )
    dT.abb.VVvOOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('AcIe,eBJK->ABcIJK', H.ab.vvov[Va, vb, Oa, :], T.bb[:, Vb, Ob, Ob], optimize=True)
    )
    dT.abb.VVvOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('AmIJ,BcmK->ABcIJK', H.ab.vooo[Va, :, Oa, Ob], T.bb[Vb, vb, :, Ob], optimize=True)
    )
    dT.abb.VVvOOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('cBKe,AeIJ->ABcIJK', H.bb.vvov[vb, Vb, Ob, :], T.ab[Va, :, Oa, Ob], optimize=True)
    )
    dT.abb.VVvOOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('cmKJ,ABIm->ABcIJK', H.bb.vooo[vb, :, Ob, Ob], T.ab[Va, Vb, Oa, :], optimize=True)
    )
    dT.abb.VVvOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('BmKJ,AcIm->ABcIJK', H.bb.vooo[Vb, :, Ob, Ob], T.ab[Va, vb, Oa, :], optimize=True)
    )
    dT.abb.VVvOOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('ABeJ,ecIK->ABcIJK', H.ab.vvvo[Va, Vb, :, Ob], T.ab[:, vb, Oa, Ob], optimize=True)
    )
    dT.abb.VVvOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('AceJ,eBIK->ABcIJK', H.ab.vvvo[Va, vb, :, Ob], T.ab[:, Vb, Oa, Ob], optimize=True)
    )
    dT.abb.VVvOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mBIJ,AcmK->ABcIJK', H.ab.ovoo[:, Vb, Oa, Ob], T.ab[Va, vb, :, Ob], optimize=True)
    )
    dT.abb.VVvOOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mcIJ,ABmK->ABcIJK', H.ab.ovoo[:, vb, Oa, Ob], T.ab[Va, Vb, :, Ob], optimize=True)
    )
    # (H(2) * T3)_C
    dT.abb.VVvOOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mI,ABcmJK->ABcIJK', H.a.oo[oa, Oa], T.abb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('MI,ABcMJK->ABcIJK', H.a.oo[Oa, Oa], T.abb.VVvOOO, optimize=True)
    )
    dT.abb.VVvOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mJ,ABcImK->ABcIJK', H.b.oo[ob, Ob], T.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('MJ,ABcIMK->ABcIJK', H.b.oo[Ob, Ob], T.abb.VVvOOO, optimize=True)
    )
    dT.abb.VVvOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('Ae,eBcIJK->ABcIJK', H.a.vv[Va, va], T.abb.vVvOOO, optimize=True)
            + 1.0 * np.einsum('AE,EBcIJK->ABcIJK', H.a.vv[Va, Va], T.abb.VVvOOO, optimize=True)
    )
    dT.abb.VVvOOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('Be,AceIJK->ABcIJK', H.b.vv[Vb, vb], T.abb.VvvOOO, optimize=True)
            + 1.0 * np.einsum('BE,AEcIJK->ABcIJK', H.b.vv[Vb, Vb], T.abb.VVvOOO, optimize=True)
    )
    dT.abb.VVvOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('ce,ABeIJK->ABcIJK', H.b.vv[vb, vb], T.abb.VVvOOO, optimize=True)
            + 1.0 * np.einsum('cE,ABEIJK->ABcIJK', H.b.vv[vb, Vb], T.abb.VVVOOO, optimize=True)
    )
    dT.abb.VVvOOO += (1.0 / 2.0) * (
            +0.5 * np.einsum('mnJK,ABcImn->ABcIJK', H.bb.oooo[ob, ob, Ob, Ob], T.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('mNJK,ABcImN->ABcIJK', H.bb.oooo[ob, Ob, Ob, Ob], T.abb.VVvOoO, optimize=True)
            + 0.5 * np.einsum('MNJK,ABcIMN->ABcIJK', H.bb.oooo[Ob, Ob, Ob, Ob], T.abb.VVvOOO, optimize=True)
    )
    dT.abb.VVvOOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mnIJ,ABcmnK->ABcIJK', H.ab.oooo[oa, ob, Oa, Ob], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('MnIJ,ABcMnK->ABcIJK', H.ab.oooo[Oa, ob, Oa, Ob], T.abb.VVvOoO, optimize=True)
            + 1.0 * np.einsum('mNIJ,ABcmNK->ABcIJK', H.ab.oooo[oa, Ob, Oa, Ob], T.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('MNIJ,ABcMNK->ABcIJK', H.ab.oooo[Oa, Ob, Oa, Ob], T.abb.VVvOOO, optimize=True)
    )
    dT.abb.VVvOOO += (1.0 / 2.0) * (
            -0.5 * np.einsum('Bcef,AfeIJK->ABcIJK', H.bb.vvvv[Vb, vb, vb, vb], T.abb.VvvOOO, optimize=True)
            + 1.0 * np.einsum('BcEf,AEfIJK->ABcIJK', H.bb.vvvv[Vb, vb, Vb, vb], T.abb.VVvOOO, optimize=True)
            - 0.5 * np.einsum('BcEF,AFEIJK->ABcIJK', H.bb.vvvv[Vb, vb, Vb, Vb], T.abb.VVVOOO, optimize=True)
    )
    dT.abb.VVvOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('ABeF,eFcIJK->ABcIJK', H.ab.vvvv[Va, Vb, va, Vb], T.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('ABEf,EcfIJK->ABcIJK', H.ab.vvvv[Va, Vb, Va, vb], T.abb.VvvOOO, optimize=True)
            + 1.0 * np.einsum('ABEF,EFcIJK->ABcIJK', H.ab.vvvv[Va, Vb, Va, Vb], T.abb.VVvOOO, optimize=True)
    )
    dT.abb.VVvOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('Acef,eBfIJK->ABcIJK', H.ab.vvvv[Va, vb, va, vb], T.abb.vVvOOO, optimize=True)
            + 1.0 * np.einsum('AceF,eBFIJK->ABcIJK', H.ab.vvvv[Va, vb, va, Vb], T.abb.vVVOOO, optimize=True)
            + 1.0 * np.einsum('AcEf,EBfIJK->ABcIJK', H.ab.vvvv[Va, vb, Va, vb], T.abb.VVvOOO, optimize=True)
            + 1.0 * np.einsum('AcEF,EBFIJK->ABcIJK', H.ab.vvvv[Va, vb, Va, Vb], T.abb.VVVOOO, optimize=True)
    )
    dT.abb.VVvOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('AmIe,eBcmJK->ABcIJK', H.aa.voov[Va, oa, Oa, va], T.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('AMIe,eBcMJK->ABcIJK', H.aa.voov[Va, Oa, Oa, va], T.abb.vVvOOO, optimize=True)
            + 1.0 * np.einsum('AmIE,EBcmJK->ABcIJK', H.aa.voov[Va, oa, Oa, Va], T.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('AMIE,EBcMJK->ABcIJK', H.aa.voov[Va, Oa, Oa, Va], T.abb.VVvOOO, optimize=True)
    )
    dT.abb.VVvOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('AmIe,BcemJK->ABcIJK', H.ab.voov[Va, ob, Oa, vb], T.bbb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('AMIe,BceMJK->ABcIJK', H.ab.voov[Va, Ob, Oa, vb], T.bbb.VvvOOO, optimize=True)
            - 1.0 * np.einsum('AmIE,BEcmJK->ABcIJK', H.ab.voov[Va, ob, Oa, Vb], T.bbb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('AMIE,BEcMJK->ABcIJK', H.ab.voov[Va, Ob, Oa, Vb], T.bbb.VVvOOO, optimize=True)
    )
    dT.abb.VVvOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mBeJ,AecmIK->ABcIJK', H.ab.ovvo[oa, Vb, va, Ob], T.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('MBeJ,AecIMK->ABcIJK', H.ab.ovvo[Oa, Vb, va, Ob], T.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('mBEJ,EAcmIK->ABcIJK', H.ab.ovvo[oa, Vb, Va, Ob], T.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('MBEJ,EAcIMK->ABcIJK', H.ab.ovvo[Oa, Vb, Va, Ob], T.aab.VVvOOO, optimize=True)
    )
    dT.abb.VVvOOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mceJ,AeBmIK->ABcIJK', H.ab.ovvo[oa, vb, va, Ob], T.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('MceJ,AeBIMK->ABcIJK', H.ab.ovvo[Oa, vb, va, Ob], T.aab.VvVOOO, optimize=True)
            - 1.0 * np.einsum('mcEJ,EABmIK->ABcIJK', H.ab.ovvo[oa, vb, Va, Ob], T.aab.VVVoOO, optimize=True)
            + 1.0 * np.einsum('McEJ,EABIMK->ABcIJK', H.ab.ovvo[Oa, vb, Va, Ob], T.aab.VVVOOO, optimize=True)
    )
    dT.abb.VVvOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('BmJe,AceImK->ABcIJK', H.bb.voov[Vb, ob, Ob, vb], T.abb.VvvOoO, optimize=True)
            - 1.0 * np.einsum('BMJe,AceIMK->ABcIJK', H.bb.voov[Vb, Ob, Ob, vb], T.abb.VvvOOO, optimize=True)
            + 1.0 * np.einsum('BmJE,AEcImK->ABcIJK', H.bb.voov[Vb, ob, Ob, Vb], T.abb.VVvOoO, optimize=True)
            + 1.0 * np.einsum('BMJE,AEcIMK->ABcIJK', H.bb.voov[Vb, Ob, Ob, Vb], T.abb.VVvOOO, optimize=True)
    )
    dT.abb.VVvOOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('cmJe,ABeImK->ABcIJK', H.bb.voov[vb, ob, Ob, vb], T.abb.VVvOoO, optimize=True)
            + 1.0 * np.einsum('cMJe,ABeIMK->ABcIJK', H.bb.voov[vb, Ob, Ob, vb], T.abb.VVvOOO, optimize=True)
            + 1.0 * np.einsum('cmJE,ABEImK->ABcIJK', H.bb.voov[vb, ob, Ob, Vb], T.abb.VVVOoO, optimize=True)
            + 1.0 * np.einsum('cMJE,ABEIMK->ABcIJK', H.bb.voov[vb, Ob, Ob, Vb], T.abb.VVVOOO, optimize=True)
    )
    dT.abb.VVvOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mBIe,AcemJK->ABcIJK', H.ab.ovov[oa, Vb, Oa, vb], T.abb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('MBIe,AceMJK->ABcIJK', H.ab.ovov[Oa, Vb, Oa, vb], T.abb.VvvOOO, optimize=True)
            - 1.0 * np.einsum('mBIE,AEcmJK->ABcIJK', H.ab.ovov[oa, Vb, Oa, Vb], T.abb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('MBIE,AEcMJK->ABcIJK', H.ab.ovov[Oa, Vb, Oa, Vb], T.abb.VVvOOO, optimize=True)
    )
    dT.abb.VVvOOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mcIe,ABemJK->ABcIJK', H.ab.ovov[oa, vb, Oa, vb], T.abb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('McIe,ABeMJK->ABcIJK', H.ab.ovov[Oa, vb, Oa, vb], T.abb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('mcIE,ABEmJK->ABcIJK', H.ab.ovov[oa, vb, Oa, Vb], T.abb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('McIE,ABEMJK->ABcIJK', H.ab.ovov[Oa, vb, Oa, Vb], T.abb.VVVOOO, optimize=True)
    )
    dT.abb.VVvOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('AmeJ,eBcImK->ABcIJK', H.ab.vovo[Va, ob, va, Ob], T.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('AMeJ,eBcIMK->ABcIJK', H.ab.vovo[Va, Ob, va, Ob], T.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('AmEJ,EBcImK->ABcIJK', H.ab.vovo[Va, ob, Va, Ob], T.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('AMEJ,EBcIMK->ABcIJK', H.ab.vovo[Va, Ob, Va, Ob], T.abb.VVvOOO, optimize=True)
    )

    dT.abb.VVvOOO -= np.transpose(dT.abb.VVvOOO, (0, 1, 2, 3, 5, 4))

    return dT


def update(T, dT, H, shift, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.abb.VVvOOO, dT.abb.VVvOOO = cc_active_loops.update_t3c_110111(
        T.abb.VVvOOO,
        dT.abb.VVvOOO,
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