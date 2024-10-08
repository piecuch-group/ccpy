import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import cc_active_loops

def build(T, dT, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    # MM(2,3)
    dT.abb.VVVOoO = (2.0 / 2.0) * (
            +1.0 * np.einsum('ABIe,eCjK->ABCIjK', H.ab.vvov[Va, Vb, Oa, :], T.bb[:, Vb, ob, Ob], optimize=True)
    )
    dT.abb.VVVOoO += (1.0 / 2.0) * (
            -1.0 * np.einsum('AmIj,BCmK->ABCIjK', H.ab.vooo[Va, :, Oa, ob], T.bb[Vb, Vb, :, Ob], optimize=True)
    )
    dT.abb.VVVOoO += (1.0 / 2.0) * (
            +1.0 * np.einsum('AmIK,BCmj->ABCIjK', H.ab.vooo[Va, :, Oa, Ob], T.bb[Vb, Vb, :, ob], optimize=True)
    )
    dT.abb.VVVOoO += (1.0 / 2.0) * (
            +1.0 * np.einsum('CBKe,AeIj->ABCIjK', H.bb.vvov[Vb, Vb, Ob, :], T.ab[Va, :, Oa, ob], optimize=True)
    )
    dT.abb.VVVOoO += (1.0 / 2.0) * (
            -1.0 * np.einsum('CBje,AeIK->ABCIjK', H.bb.vvov[Vb, Vb, ob, :], T.ab[Va, :, Oa, Ob], optimize=True)
    )
    dT.abb.VVVOoO += (2.0 / 2.0) * (
            -1.0 * np.einsum('CmKj,ABIm->ABCIjK', H.bb.vooo[Vb, :, Ob, ob], T.ab[Va, Vb, Oa, :], optimize=True)
    )
    dT.abb.VVVOoO += (2.0 / 2.0) * (
            +1.0 * np.einsum('ABej,eCIK->ABCIjK', H.ab.vvvo[Va, Vb, :, ob], T.ab[:, Vb, Oa, Ob], optimize=True)
    )
    dT.abb.VVVOoO += (2.0 / 2.0) * (
            -1.0 * np.einsum('ABeK,eCIj->ABCIjK', H.ab.vvvo[Va, Vb, :, Ob], T.ab[:, Vb, Oa, ob], optimize=True)
    )
    dT.abb.VVVOoO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mBIj,ACmK->ABCIjK', H.ab.ovoo[:, Vb, Oa, ob], T.ab[Va, Vb, :, Ob], optimize=True)
    )
    dT.abb.VVVOoO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mBIK,ACmj->ABCIjK', H.ab.ovoo[:, Vb, Oa, Ob], T.ab[Va, Vb, :, ob], optimize=True)
    )
    # (H(2) * T3)_C
    dT.abb.VVVOoO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mI,ACBmjK->ABCIjK', H.a.oo[oa, Oa], T.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('MI,ACBMjK->ABCIjK', H.a.oo[Oa, Oa], T.abb.VVVOoO, optimize=True)
    )
    dT.abb.VVVOoO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mj,ACBImK->ABCIjK', H.b.oo[ob, ob], T.abb.VVVOoO, optimize=True)
            + 1.0 * np.einsum('Mj,ACBIMK->ABCIjK', H.b.oo[Ob, ob], T.abb.VVVOOO, optimize=True)
    )
    dT.abb.VVVOoO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mK,ACBImj->ABCIjK', H.b.oo[ob, Ob], T.abb.VVVOoo, optimize=True)
            + 1.0 * np.einsum('MK,ACBIjM->ABCIjK', H.b.oo[Ob, Ob], T.abb.VVVOoO, optimize=True)
    )
    dT.abb.VVVOoO += (1.0 / 2.0) * (
            -1.0 * np.einsum('Ae,eCBIjK->ABCIjK', H.a.vv[Va, va], T.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('AE,ECBIjK->ABCIjK', H.a.vv[Va, Va], T.abb.VVVOoO, optimize=True)
    )
    dT.abb.VVVOoO += (2.0 / 2.0) * (
            -1.0 * np.einsum('Be,ACeIjK->ABCIjK', H.b.vv[Vb, vb], T.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('BE,ACEIjK->ABCIjK', H.b.vv[Vb, Vb], T.abb.VVVOoO, optimize=True)
    )
    dT.abb.VVVOoO += (1.0 / 2.0) * (
            -0.5 * np.einsum('mnjK,ACBImn->ABCIjK', H.bb.oooo[ob, ob, ob, Ob], T.abb.VVVOoo, optimize=True)
            - 1.0 * np.einsum('mNjK,ACBImN->ABCIjK', H.bb.oooo[ob, Ob, ob, Ob], T.abb.VVVOoO, optimize=True)
            - 0.5 * np.einsum('MNjK,ACBIMN->ABCIjK', H.bb.oooo[Ob, Ob, ob, Ob], T.abb.VVVOOO, optimize=True)
    )
    dT.abb.VVVOoO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mnIj,ACBmnK->ABCIjK', H.ab.oooo[oa, ob, Oa, ob], T.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('MnIj,ACBMnK->ABCIjK', H.ab.oooo[Oa, ob, Oa, ob], T.abb.VVVOoO, optimize=True)
            - 1.0 * np.einsum('mNIj,ACBmNK->ABCIjK', H.ab.oooo[oa, Ob, Oa, ob], T.abb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('MNIj,ACBMNK->ABCIjK', H.ab.oooo[Oa, Ob, Oa, ob], T.abb.VVVOOO, optimize=True)
    )
    dT.abb.VVVOoO += (1.0 / 2.0) * (
            +1.0 * np.einsum('MnIK,ACBMnj->ABCIjK', H.ab.oooo[Oa, ob, Oa, Ob], T.abb.VVVOoo, optimize=True)
            - 1.0 * np.einsum('mNIK,ACBmjN->ABCIjK', H.ab.oooo[oa, Ob, Oa, Ob], T.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('MNIK,ACBMjN->ABCIjK', H.ab.oooo[Oa, Ob, Oa, Ob], T.abb.VVVOoO, optimize=True)
    )
    dT.abb.VVVOoO += (1.0 / 2.0) * (
            -0.5 * np.einsum('BCef,AfeIjK->ABCIjK', H.bb.vvvv[Vb, Vb, vb, vb], T.abb.VvvOoO, optimize=True)
            - 1.0 * np.einsum('BCeF,AFeIjK->ABCIjK', H.bb.vvvv[Vb, Vb, vb, Vb], T.abb.VVvOoO, optimize=True)
            - 0.5 * np.einsum('BCEF,AFEIjK->ABCIjK', H.bb.vvvv[Vb, Vb, Vb, Vb], T.abb.VVVOoO, optimize=True)
    )
    dT.abb.VVVOoO += (2.0 / 2.0) * (
            -1.0 * np.einsum('ABef,eCfIjK->ABCIjK', H.ab.vvvv[Va, Vb, va, vb], T.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('ABEf,ECfIjK->ABCIjK', H.ab.vvvv[Va, Vb, Va, vb], T.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('ABeF,eCFIjK->ABCIjK', H.ab.vvvv[Va, Vb, va, Vb], T.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('ABEF,ECFIjK->ABCIjK', H.ab.vvvv[Va, Vb, Va, Vb], T.abb.VVVOoO, optimize=True)
    )
    dT.abb.VVVOoO += (1.0 / 2.0) * (
            -1.0 * np.einsum('AmIe,eCBmjK->ABCIjK', H.aa.voov[Va, oa, Oa, va], T.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('AMIe,eCBMjK->ABCIjK', H.aa.voov[Va, Oa, Oa, va], T.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('AmIE,ECBmjK->ABCIjK', H.aa.voov[Va, oa, Oa, Va], T.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('AMIE,ECBMjK->ABCIjK', H.aa.voov[Va, Oa, Oa, Va], T.abb.VVVOoO, optimize=True)
    )
    dT.abb.VVVOoO += (1.0 / 2.0) * (
            -1.0 * np.einsum('AmIe,CBemjK->ABCIjK', H.ab.voov[Va, ob, Oa, vb], T.bbb.VVvooO, optimize=True)
            + 1.0 * np.einsum('AMIe,CBejMK->ABCIjK', H.ab.voov[Va, Ob, Oa, vb], T.bbb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('AmIE,CBEmjK->ABCIjK', H.ab.voov[Va, ob, Oa, Vb], T.bbb.VVVooO, optimize=True)
            + 1.0 * np.einsum('AMIE,CBEjMK->ABCIjK', H.ab.voov[Va, Ob, Oa, Vb], T.bbb.VVVoOO, optimize=True)
    )
    dT.abb.VVVOoO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mBej,AeCmIK->ABCIjK', H.ab.ovvo[oa, Vb, va, ob], T.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('MBej,AeCIMK->ABCIjK', H.ab.ovvo[Oa, Vb, va, ob], T.aab.VvVOOO, optimize=True)
            + 1.0 * np.einsum('mBEj,EACmIK->ABCIjK', H.ab.ovvo[oa, Vb, Va, ob], T.aab.VVVoOO, optimize=True)
            - 1.0 * np.einsum('MBEj,EACIMK->ABCIjK', H.ab.ovvo[Oa, Vb, Va, ob], T.aab.VVVOOO, optimize=True)
    )
    dT.abb.VVVOoO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mBeK,AeCmIj->ABCIjK', H.ab.ovvo[oa, Vb, va, Ob], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('MBeK,AeCIMj->ABCIjK', H.ab.ovvo[Oa, Vb, va, Ob], T.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('mBEK,EACmIj->ABCIjK', H.ab.ovvo[oa, Vb, Va, Ob], T.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('MBEK,EACIMj->ABCIjK', H.ab.ovvo[Oa, Vb, Va, Ob], T.aab.VVVOOo, optimize=True)
    )
    dT.abb.VVVOoO += (2.0 / 2.0) * (
            -1.0 * np.einsum('Bmje,ACeImK->ABCIjK', H.bb.voov[Vb, ob, ob, vb], T.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('BMje,ACeIMK->ABCIjK', H.bb.voov[Vb, Ob, ob, vb], T.abb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('BmjE,ACEImK->ABCIjK', H.bb.voov[Vb, ob, ob, Vb], T.abb.VVVOoO, optimize=True)
            - 1.0 * np.einsum('BMjE,ACEIMK->ABCIjK', H.bb.voov[Vb, Ob, ob, Vb], T.abb.VVVOOO, optimize=True)
    )
    dT.abb.VVVOoO += (2.0 / 2.0) * (
            +1.0 * np.einsum('BmKe,ACeImj->ABCIjK', H.bb.voov[Vb, ob, Ob, vb], T.abb.VVvOoo, optimize=True)
            - 1.0 * np.einsum('BMKe,ACeIjM->ABCIjK', H.bb.voov[Vb, Ob, Ob, vb], T.abb.VVvOoO, optimize=True)
            + 1.0 * np.einsum('BmKE,ACEImj->ABCIjK', H.bb.voov[Vb, ob, Ob, Vb], T.abb.VVVOoo, optimize=True)
            - 1.0 * np.einsum('BMKE,ACEIjM->ABCIjK', H.bb.voov[Vb, Ob, Ob, Vb], T.abb.VVVOoO, optimize=True)
    )
    dT.abb.VVVOoO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mBIe,ACemjK->ABCIjK', H.ab.ovov[oa, Vb, Oa, vb], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('MBIe,ACeMjK->ABCIjK', H.ab.ovov[Oa, Vb, Oa, vb], T.abb.VVvOoO, optimize=True)
            + 1.0 * np.einsum('mBIE,ACEmjK->ABCIjK', H.ab.ovov[oa, Vb, Oa, Vb], T.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('MBIE,ACEMjK->ABCIjK', H.ab.ovov[Oa, Vb, Oa, Vb], T.abb.VVVOoO, optimize=True)
    )
    dT.abb.VVVOoO += (1.0 / 2.0) * (
            +1.0 * np.einsum('Amej,eCBImK->ABCIjK', H.ab.vovo[Va, ob, va, ob], T.abb.vVVOoO, optimize=True)
            + 1.0 * np.einsum('AMej,eCBIMK->ABCIjK', H.ab.vovo[Va, Ob, va, ob], T.abb.vVVOOO, optimize=True)
            + 1.0 * np.einsum('AmEj,ECBImK->ABCIjK', H.ab.vovo[Va, ob, Va, ob], T.abb.VVVOoO, optimize=True)
            + 1.0 * np.einsum('AMEj,ECBIMK->ABCIjK', H.ab.vovo[Va, Ob, Va, ob], T.abb.VVVOOO, optimize=True)
    )
    dT.abb.VVVOoO += (1.0 / 2.0) * (
            -1.0 * np.einsum('AmeK,eCBImj->ABCIjK', H.ab.vovo[Va, ob, va, Ob], T.abb.vVVOoo, optimize=True)
            + 1.0 * np.einsum('AMeK,eCBIjM->ABCIjK', H.ab.vovo[Va, Ob, va, Ob], T.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('AmEK,ECBImj->ABCIjK', H.ab.vovo[Va, ob, Va, Ob], T.abb.VVVOoo, optimize=True)
            + 1.0 * np.einsum('AMEK,ECBIjM->ABCIjK', H.ab.vovo[Va, Ob, Va, Ob], T.abb.VVVOoO, optimize=True)
    )

    dT.abb.VVVOoO -= np.transpose(dT.abb.VVVOoO, (0, 2, 1, 3, 4, 5))

    return dT


def update(T, dT, H, shift, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.abb.VVVOoO, dT.abb.VVVOoO = cc_active_loops.update_t3c_111101(
        T.abb.VVVOoO,
        dT.abb.VVVOoO,
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