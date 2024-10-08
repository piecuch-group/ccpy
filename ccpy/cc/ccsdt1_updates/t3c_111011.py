import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import cc_active_loops

def build(T, dT, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    # MM(2,3)
    dT.abb.VVVoOO = (2.0 / 4.0) * (
            +1.0 * np.einsum('ABie,eCJK->ABCiJK', H.ab.vvov[Va, Vb, oa, :], T.bb[:, Vb, Ob, Ob], optimize=True)
    )
    dT.abb.VVVoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('AmiJ,BCmK->ABCiJK', H.ab.vooo[Va, :, oa, Ob], T.bb[Vb, Vb, :, Ob], optimize=True)
    )
    dT.abb.VVVoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('CBKe,AeiJ->ABCiJK', H.bb.vvov[Vb, Vb, Ob, :], T.ab[Va, :, oa, Ob], optimize=True)
    )
    dT.abb.VVVoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('CmKJ,ABim->ABCiJK', H.bb.vooo[Vb, :, Ob, Ob], T.ab[Va, Vb, oa, :], optimize=True)
    )
    dT.abb.VVVoOO += (4.0 / 4.0) * (
            +1.0 * np.einsum('ABeJ,eCiK->ABCiJK', H.ab.vvvo[Va, Vb, :, Ob], T.ab[:, Vb, oa, Ob], optimize=True)
    )
    dT.abb.VVVoOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('mBiJ,ACmK->ABCiJK', H.ab.ovoo[:, Vb, oa, Ob], T.ab[Va, Vb, :, Ob], optimize=True)
    )
    # (H(2) * T3)_C
    dT.abb.VVVoOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('mi,ACBmJK->ABCiJK', H.a.oo[oa, oa], T.abb.VVVoOO, optimize=True)
            + 1.0 * np.einsum('Mi,ACBMJK->ABCiJK', H.a.oo[Oa, oa], T.abb.VVVOOO, optimize=True)
    )
    dT.abb.VVVoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mJ,ACBimK->ABCiJK', H.b.oo[ob, Ob], T.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('MJ,ACBiMK->ABCiJK', H.b.oo[Ob, Ob], T.abb.VVVoOO, optimize=True)
    )
    dT.abb.VVVoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('Ae,eCBiJK->ABCiJK', H.a.vv[Va, va], T.abb.vVVoOO, optimize=True)
            - 1.0 * np.einsum('AE,ECBiJK->ABCiJK', H.a.vv[Va, Va], T.abb.VVVoOO, optimize=True)
    )
    dT.abb.VVVoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('Be,ACeiJK->ABCiJK', H.b.vv[Vb, vb], T.abb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('BE,ACEiJK->ABCiJK', H.b.vv[Vb, Vb], T.abb.VVVoOO, optimize=True)
    )
    dT.abb.VVVoOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('MnJK,ACBinM->ABCiJK', H.bb.oooo[Ob, ob, Ob, Ob], T.abb.VVVooO, optimize=True)
            - 0.5 * np.einsum('MNJK,ACBiMN->ABCiJK', H.bb.oooo[Ob, Ob, Ob, Ob], T.abb.VVVoOO, optimize=True)
    )
    dT.abb.VVVoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('mniJ,ACBmnK->ABCiJK', H.ab.oooo[oa, ob, oa, Ob], T.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('mNiJ,ACBmNK->ABCiJK', H.ab.oooo[oa, Ob, oa, Ob], T.abb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('MniJ,ACBMnK->ABCiJK', H.ab.oooo[Oa, ob, oa, Ob], T.abb.VVVOoO, optimize=True)
            - 1.0 * np.einsum('MNiJ,ACBMNK->ABCiJK', H.ab.oooo[Oa, Ob, oa, Ob], T.abb.VVVOOO, optimize=True)
    )
    dT.abb.VVVoOO += (1.0 / 4.0) * (
            -0.5 * np.einsum('BCef,AfeiJK->ABCiJK', H.bb.vvvv[Vb, Vb, vb, vb], T.abb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('BCEf,AEfiJK->ABCiJK', H.bb.vvvv[Vb, Vb, Vb, vb], T.abb.VVvoOO, optimize=True)
            - 0.5 * np.einsum('BCEF,AFEiJK->ABCiJK', H.bb.vvvv[Vb, Vb, Vb, Vb], T.abb.VVVoOO, optimize=True)
    )
    dT.abb.VVVoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('ABef,eCfiJK->ABCiJK', H.ab.vvvv[Va, Vb, va, vb], T.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('ABeF,eCFiJK->ABCiJK', H.ab.vvvv[Va, Vb, va, Vb], T.abb.vVVoOO, optimize=True)
            - 1.0 * np.einsum('ABEf,ECfiJK->ABCiJK', H.ab.vvvv[Va, Vb, Va, vb], T.abb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('ABEF,ECFiJK->ABCiJK', H.ab.vvvv[Va, Vb, Va, Vb], T.abb.VVVoOO, optimize=True)
    )
    dT.abb.VVVoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('Amie,eCBmJK->ABCiJK', H.aa.voov[Va, oa, oa, va], T.abb.vVVoOO, optimize=True)
            - 1.0 * np.einsum('AMie,eCBMJK->ABCiJK', H.aa.voov[Va, Oa, oa, va], T.abb.vVVOOO, optimize=True)
            - 1.0 * np.einsum('AmiE,ECBmJK->ABCiJK', H.aa.voov[Va, oa, oa, Va], T.abb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('AMiE,ECBMJK->ABCiJK', H.aa.voov[Va, Oa, oa, Va], T.abb.VVVOOO, optimize=True)
    )
    dT.abb.VVVoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('Amie,CBemJK->ABCiJK', H.ab.voov[Va, ob, oa, vb], T.bbb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('AMie,CBeMJK->ABCiJK', H.ab.voov[Va, Ob, oa, vb], T.bbb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('AmiE,CBEmJK->ABCiJK', H.ab.voov[Va, ob, oa, Vb], T.bbb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('AMiE,CBEMJK->ABCiJK', H.ab.voov[Va, Ob, oa, Vb], T.bbb.VVVOOO, optimize=True)
    )
    dT.abb.VVVoOO += (4.0 / 4.0) * (
            +1.0 * np.einsum('mBeJ,AeCimK->ABCiJK', H.ab.ovvo[oa, Vb, va, Ob], T.aab.VvVooO, optimize=True)
            + 1.0 * np.einsum('MBeJ,AeCiMK->ABCiJK', H.ab.ovvo[Oa, Vb, va, Ob], T.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('mBEJ,EACimK->ABCiJK', H.ab.ovvo[oa, Vb, Va, Ob], T.aab.VVVooO, optimize=True)
            - 1.0 * np.einsum('MBEJ,EACiMK->ABCiJK', H.ab.ovvo[Oa, Vb, Va, Ob], T.aab.VVVoOO, optimize=True)
    )
    dT.abb.VVVoOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('BmJe,ACeimK->ABCiJK', H.bb.voov[Vb, ob, Ob, vb], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('BMJe,ACeiMK->ABCiJK', H.bb.voov[Vb, Ob, Ob, vb], T.abb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('BmJE,ACEimK->ABCiJK', H.bb.voov[Vb, ob, Ob, Vb], T.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('BMJE,ACEiMK->ABCiJK', H.bb.voov[Vb, Ob, Ob, Vb], T.abb.VVVoOO, optimize=True)
    )
    dT.abb.VVVoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mBie,ACemJK->ABCiJK', H.ab.ovov[oa, Vb, oa, vb], T.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('MBie,ACeMJK->ABCiJK', H.ab.ovov[Oa, Vb, oa, vb], T.abb.VVvOOO, optimize=True)
            + 1.0 * np.einsum('mBiE,ACEmJK->ABCiJK', H.ab.ovov[oa, Vb, oa, Vb], T.abb.VVVoOO, optimize=True)
            + 1.0 * np.einsum('MBiE,ACEMJK->ABCiJK', H.ab.ovov[Oa, Vb, oa, Vb], T.abb.VVVOOO, optimize=True)
    )
    dT.abb.VVVoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('AmeJ,eCBimK->ABCiJK', H.ab.vovo[Va, ob, va, Ob], T.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('AMeJ,eCBiMK->ABCiJK', H.ab.vovo[Va, Ob, va, Ob], T.abb.vVVoOO, optimize=True)
            + 1.0 * np.einsum('AmEJ,ECBimK->ABCiJK', H.ab.vovo[Va, ob, Va, Ob], T.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('AMEJ,ECBiMK->ABCiJK', H.ab.vovo[Va, Ob, Va, Ob], T.abb.VVVoOO, optimize=True)
    )

    dT.abb.VVVoOO -= np.transpose(dT.abb.VVVoOO, (0, 2, 1, 3, 4, 5))
    dT.abb.VVVoOO -= np.transpose(dT.abb.VVVoOO, (0, 1, 2, 3, 5, 4))

    return dT


def update(T, dT, H, shift, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.abb.VVVoOO, dT.abb.VVVoOO = cc_active_loops.update_t3c_111011(
        T.abb.VVVoOO,
        dT.abb.VVVoOO,
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
