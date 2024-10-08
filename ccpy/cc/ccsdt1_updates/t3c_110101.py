import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import cc_active_loops

def build(T, dT, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    # MM(2,3)
    dT.abb.VVvOoO = (1.0 / 1.0) * (
            +1.0 * np.einsum('ABIe,ecjK->ABcIjK', H.ab.vvov[Va, Vb, Oa, :], T.bb[:, vb, ob, Ob], optimize=True)
    )
    dT.abb.VVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('AcIe,eBjK->ABcIjK', H.ab.vvov[Va, vb, Oa, :], T.bb[:, Vb, ob, Ob], optimize=True)
    )
    dT.abb.VVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('AmIj,BcmK->ABcIjK', H.ab.vooo[Va, :, Oa, ob], T.bb[Vb, vb, :, Ob], optimize=True)
    )
    dT.abb.VVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('AmIK,Bcmj->ABcIjK', H.ab.vooo[Va, :, Oa, Ob], T.bb[Vb, vb, :, ob], optimize=True)
    )
    dT.abb.VVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('cBKe,AeIj->ABcIjK', H.bb.vvov[vb, Vb, Ob, :], T.ab[Va, :, Oa, ob], optimize=True)
    )
    dT.abb.VVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('cBje,AeIK->ABcIjK', H.bb.vvov[vb, Vb, ob, :], T.ab[Va, :, Oa, Ob], optimize=True)
    )
    dT.abb.VVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('cmKj,ABIm->ABcIjK', H.bb.vooo[vb, :, Ob, ob], T.ab[Va, Vb, Oa, :], optimize=True)
    )
    dT.abb.VVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('BmKj,AcIm->ABcIjK', H.bb.vooo[Vb, :, Ob, ob], T.ab[Va, vb, Oa, :], optimize=True)
    )
    dT.abb.VVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('ABej,ecIK->ABcIjK', H.ab.vvvo[Va, Vb, :, ob], T.ab[:, vb, Oa, Ob], optimize=True)
    )
    dT.abb.VVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('Acej,eBIK->ABcIjK', H.ab.vvvo[Va, vb, :, ob], T.ab[:, Vb, Oa, Ob], optimize=True)
    )
    dT.abb.VVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('ABeK,ecIj->ABcIjK', H.ab.vvvo[Va, Vb, :, Ob], T.ab[:, vb, Oa, ob], optimize=True)
    )
    dT.abb.VVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('AceK,eBIj->ABcIjK', H.ab.vvvo[Va, vb, :, Ob], T.ab[:, Vb, Oa, ob], optimize=True)
    )
    dT.abb.VVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mBIj,AcmK->ABcIjK', H.ab.ovoo[:, Vb, Oa, ob], T.ab[Va, vb, :, Ob], optimize=True)
    )
    dT.abb.VVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mcIj,ABmK->ABcIjK', H.ab.ovoo[:, vb, Oa, ob], T.ab[Va, Vb, :, Ob], optimize=True)
    )
    dT.abb.VVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mBIK,Acmj->ABcIjK', H.ab.ovoo[:, Vb, Oa, Ob], T.ab[Va, vb, :, ob], optimize=True)
    )
    dT.abb.VVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mcIK,ABmj->ABcIjK', H.ab.ovoo[:, vb, Oa, Ob], T.ab[Va, Vb, :, ob], optimize=True)
    )
    # (H(2) * T3)_C
    dT.abb.VVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mI,ABcmjK->ABcIjK', H.a.oo[oa, Oa], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('MI,ABcMjK->ABcIjK', H.a.oo[Oa, Oa], T.abb.VVvOoO, optimize=True)
    )
    dT.abb.VVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mj,ABcImK->ABcIjK', H.b.oo[ob, ob], T.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('Mj,ABcIMK->ABcIjK', H.b.oo[Ob, ob], T.abb.VVvOOO, optimize=True)
    )
    dT.abb.VVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mK,ABcImj->ABcIjK', H.b.oo[ob, Ob], T.abb.VVvOoo, optimize=True)
            - 1.0 * np.einsum('MK,ABcIjM->ABcIjK', H.b.oo[Ob, Ob], T.abb.VVvOoO, optimize=True)
    )
    dT.abb.VVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('Ae,eBcIjK->ABcIjK', H.a.vv[Va, va], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('AE,EBcIjK->ABcIjK', H.a.vv[Va, Va], T.abb.VVvOoO, optimize=True)
    )
    dT.abb.VVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('Be,AceIjK->ABcIjK', H.b.vv[Vb, vb], T.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('BE,AEcIjK->ABcIjK', H.b.vv[Vb, Vb], T.abb.VVvOoO, optimize=True)
    )
    dT.abb.VVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('ce,ABeIjK->ABcIjK', H.b.vv[vb, vb], T.abb.VVvOoO, optimize=True)
            + 1.0 * np.einsum('cE,ABEIjK->ABcIjK', H.b.vv[vb, Vb], T.abb.VVVOoO, optimize=True)
    )
    dT.abb.VVvOoO += (1.0 / 1.0) * (
            +0.5 * np.einsum('mnjK,ABcImn->ABcIjK', H.bb.oooo[ob, ob, ob, Ob], T.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('mNjK,ABcImN->ABcIjK', H.bb.oooo[ob, Ob, ob, Ob], T.abb.VVvOoO, optimize=True)
            + 0.5 * np.einsum('MNjK,ABcIMN->ABcIjK', H.bb.oooo[Ob, Ob, ob, Ob], T.abb.VVvOOO, optimize=True)
    )
    dT.abb.VVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mnIj,ABcmnK->ABcIjK', H.ab.oooo[oa, ob, Oa, ob], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('MnIj,ABcMnK->ABcIjK', H.ab.oooo[Oa, ob, Oa, ob], T.abb.VVvOoO, optimize=True)
            + 1.0 * np.einsum('mNIj,ABcmNK->ABcIjK', H.ab.oooo[oa, Ob, Oa, ob], T.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('MNIj,ABcMNK->ABcIjK', H.ab.oooo[Oa, Ob, Oa, ob], T.abb.VVvOOO, optimize=True)
    )
    dT.abb.VVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('MnIK,ABcMnj->ABcIjK', H.ab.oooo[Oa, ob, Oa, Ob], T.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('mNIK,ABcmjN->ABcIjK', H.ab.oooo[oa, Ob, Oa, Ob], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('MNIK,ABcMjN->ABcIjK', H.ab.oooo[Oa, Ob, Oa, Ob], T.abb.VVvOoO, optimize=True)
    )
    dT.abb.VVvOoO += (1.0 / 1.0) * (
            -0.5 * np.einsum('Bcef,AfeIjK->ABcIjK', H.bb.vvvv[Vb, vb, vb, vb], T.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('BcEf,AEfIjK->ABcIjK', H.bb.vvvv[Vb, vb, Vb, vb], T.abb.VVvOoO, optimize=True)
            - 0.5 * np.einsum('BcEF,AFEIjK->ABcIjK', H.bb.vvvv[Vb, vb, Vb, Vb], T.abb.VVVOoO, optimize=True)
    )
    dT.abb.VVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('ABeF,eFcIjK->ABcIjK', H.ab.vvvv[Va, Vb, va, Vb], T.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('ABEf,EcfIjK->ABcIjK', H.ab.vvvv[Va, Vb, Va, vb], T.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('ABEF,EFcIjK->ABcIjK', H.ab.vvvv[Va, Vb, Va, Vb], T.abb.VVvOoO, optimize=True)
    )
    dT.abb.VVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('Acef,eBfIjK->ABcIjK', H.ab.vvvv[Va, vb, va, vb], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('AceF,eBFIjK->ABcIjK', H.ab.vvvv[Va, vb, va, Vb], T.abb.vVVOoO, optimize=True)
            + 1.0 * np.einsum('AcEf,EBfIjK->ABcIjK', H.ab.vvvv[Va, vb, Va, vb], T.abb.VVvOoO, optimize=True)
            + 1.0 * np.einsum('AcEF,EBFIjK->ABcIjK', H.ab.vvvv[Va, vb, Va, Vb], T.abb.VVVOoO, optimize=True)
    )
    dT.abb.VVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('AmIe,eBcmjK->ABcIjK', H.aa.voov[Va, oa, Oa, va], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('AMIe,eBcMjK->ABcIjK', H.aa.voov[Va, Oa, Oa, va], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('AmIE,EBcmjK->ABcIjK', H.aa.voov[Va, oa, Oa, Va], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('AMIE,EBcMjK->ABcIjK', H.aa.voov[Va, Oa, Oa, Va], T.abb.VVvOoO, optimize=True)
    )
    dT.abb.VVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('AmIe,BcemjK->ABcIjK', H.ab.voov[Va, ob, Oa, vb], T.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('AMIe,BcejMK->ABcIjK', H.ab.voov[Va, Ob, Oa, vb], T.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('AmIE,BEcmjK->ABcIjK', H.ab.voov[Va, ob, Oa, Vb], T.bbb.VVvooO, optimize=True)
            + 1.0 * np.einsum('AMIE,BEcjMK->ABcIjK', H.ab.voov[Va, Ob, Oa, Vb], T.bbb.VVvoOO, optimize=True)
    )
    dT.abb.VVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mBej,AecmIK->ABcIjK', H.ab.ovvo[oa, Vb, va, ob], T.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('MBej,AecIMK->ABcIjK', H.ab.ovvo[Oa, Vb, va, ob], T.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('mBEj,EAcmIK->ABcIjK', H.ab.ovvo[oa, Vb, Va, ob], T.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('MBEj,EAcIMK->ABcIjK', H.ab.ovvo[Oa, Vb, Va, ob], T.aab.VVvOOO, optimize=True)
    )
    dT.abb.VVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mcej,AeBmIK->ABcIjK', H.ab.ovvo[oa, vb, va, ob], T.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('Mcej,AeBIMK->ABcIjK', H.ab.ovvo[Oa, vb, va, ob], T.aab.VvVOOO, optimize=True)
            - 1.0 * np.einsum('mcEj,EABmIK->ABcIjK', H.ab.ovvo[oa, vb, Va, ob], T.aab.VVVoOO, optimize=True)
            + 1.0 * np.einsum('McEj,EABIMK->ABcIjK', H.ab.ovvo[Oa, vb, Va, ob], T.aab.VVVOOO, optimize=True)
    )
    dT.abb.VVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mBeK,AecmIj->ABcIjK', H.ab.ovvo[oa, Vb, va, Ob], T.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('MBeK,AecIMj->ABcIjK', H.ab.ovvo[Oa, Vb, va, Ob], T.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('mBEK,EAcmIj->ABcIjK', H.ab.ovvo[oa, Vb, Va, Ob], T.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('MBEK,EAcIMj->ABcIjK', H.ab.ovvo[Oa, Vb, Va, Ob], T.aab.VVvOOo, optimize=True)
    )
    dT.abb.VVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mceK,AeBmIj->ABcIjK', H.ab.ovvo[oa, vb, va, Ob], T.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('MceK,AeBIMj->ABcIjK', H.ab.ovvo[Oa, vb, va, Ob], T.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('mcEK,EABmIj->ABcIjK', H.ab.ovvo[oa, vb, Va, Ob], T.aab.VVVoOo, optimize=True)
            - 1.0 * np.einsum('McEK,EABIMj->ABcIjK', H.ab.ovvo[Oa, vb, Va, Ob], T.aab.VVVOOo, optimize=True)
    )
    dT.abb.VVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('Bmje,AceImK->ABcIjK', H.bb.voov[Vb, ob, ob, vb], T.abb.VvvOoO, optimize=True)
            - 1.0 * np.einsum('BMje,AceIMK->ABcIjK', H.bb.voov[Vb, Ob, ob, vb], T.abb.VvvOOO, optimize=True)
            + 1.0 * np.einsum('BmjE,AEcImK->ABcIjK', H.bb.voov[Vb, ob, ob, Vb], T.abb.VVvOoO, optimize=True)
            + 1.0 * np.einsum('BMjE,AEcIMK->ABcIjK', H.bb.voov[Vb, Ob, ob, Vb], T.abb.VVvOOO, optimize=True)
    )
    dT.abb.VVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('cmje,ABeImK->ABcIjK', H.bb.voov[vb, ob, ob, vb], T.abb.VVvOoO, optimize=True)
            + 1.0 * np.einsum('cMje,ABeIMK->ABcIjK', H.bb.voov[vb, Ob, ob, vb], T.abb.VVvOOO, optimize=True)
            + 1.0 * np.einsum('cmjE,ABEImK->ABcIjK', H.bb.voov[vb, ob, ob, Vb], T.abb.VVVOoO, optimize=True)
            + 1.0 * np.einsum('cMjE,ABEIMK->ABcIjK', H.bb.voov[vb, Ob, ob, Vb], T.abb.VVVOOO, optimize=True)
    )
    dT.abb.VVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('BmKe,AceImj->ABcIjK', H.bb.voov[Vb, ob, Ob, vb], T.abb.VvvOoo, optimize=True)
            - 1.0 * np.einsum('BMKe,AceIjM->ABcIjK', H.bb.voov[Vb, Ob, Ob, vb], T.abb.VvvOoO, optimize=True)
            - 1.0 * np.einsum('BmKE,AEcImj->ABcIjK', H.bb.voov[Vb, ob, Ob, Vb], T.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('BMKE,AEcIjM->ABcIjK', H.bb.voov[Vb, Ob, Ob, Vb], T.abb.VVvOoO, optimize=True)
    )
    dT.abb.VVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('cmKe,ABeImj->ABcIjK', H.bb.voov[vb, ob, Ob, vb], T.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('cMKe,ABeIjM->ABcIjK', H.bb.voov[vb, Ob, Ob, vb], T.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('cmKE,ABEImj->ABcIjK', H.bb.voov[vb, ob, Ob, Vb], T.abb.VVVOoo, optimize=True)
            + 1.0 * np.einsum('cMKE,ABEIjM->ABcIjK', H.bb.voov[vb, Ob, Ob, Vb], T.abb.VVVOoO, optimize=True)
    )
    dT.abb.VVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mBIe,AcemjK->ABcIjK', H.ab.ovov[oa, Vb, Oa, vb], T.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('MBIe,AceMjK->ABcIjK', H.ab.ovov[Oa, Vb, Oa, vb], T.abb.VvvOoO, optimize=True)
            - 1.0 * np.einsum('mBIE,AEcmjK->ABcIjK', H.ab.ovov[oa, Vb, Oa, Vb], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('MBIE,AEcMjK->ABcIjK', H.ab.ovov[Oa, Vb, Oa, Vb], T.abb.VVvOoO, optimize=True)
    )
    dT.abb.VVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mcIe,ABemjK->ABcIjK', H.ab.ovov[oa, vb, Oa, vb], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('McIe,ABeMjK->ABcIjK', H.ab.ovov[Oa, vb, Oa, vb], T.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('mcIE,ABEmjK->ABcIjK', H.ab.ovov[oa, vb, Oa, Vb], T.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('McIE,ABEMjK->ABcIjK', H.ab.ovov[Oa, vb, Oa, Vb], T.abb.VVVOoO, optimize=True)
    )
    dT.abb.VVvOoO += (1.0 / 1.0) * (
            -1.0 * np.einsum('Amej,eBcImK->ABcIjK', H.ab.vovo[Va, ob, va, ob], T.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('AMej,eBcIMK->ABcIjK', H.ab.vovo[Va, Ob, va, ob], T.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('AmEj,EBcImK->ABcIjK', H.ab.vovo[Va, ob, Va, ob], T.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('AMEj,EBcIMK->ABcIjK', H.ab.vovo[Va, Ob, Va, ob], T.abb.VVvOOO, optimize=True)
    )
    dT.abb.VVvOoO += (1.0 / 1.0) * (
            +1.0 * np.einsum('AmeK,eBcImj->ABcIjK', H.ab.vovo[Va, ob, va, Ob], T.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('AMeK,eBcIjM->ABcIjK', H.ab.vovo[Va, Ob, va, Ob], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('AmEK,EBcImj->ABcIjK', H.ab.vovo[Va, ob, Va, Ob], T.abb.VVvOoo, optimize=True)
            - 1.0 * np.einsum('AMEK,EBcIjM->ABcIjK', H.ab.vovo[Va, Ob, Va, Ob], T.abb.VVvOoO, optimize=True)
    )

    return dT


def update(T, dT, H, shift, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.abb.VVvOoO, dT.abb.VVvOoO = cc_active_loops.update_t3c_110101(
        T.abb.VVvOoO,
        dT.abb.VVvOoO,
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