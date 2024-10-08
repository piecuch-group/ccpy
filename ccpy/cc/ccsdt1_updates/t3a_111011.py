import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import cc_active_loops

#@profile
def build(T, dT, H, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    # MM(2,3)
    dT.aaa.VVVoOO =  -(6.0 / 12.0) * np.einsum("AmiJ,BCmK->ABCiJK", H.aa.vooo[Va, :, oa, Oa], T.aa[Va, Va, :, Oa], optimize=True)
    dT.aaa.VVVoOO -= -(3.0 / 12.0) * np.einsum("AmKJ,BCmi->ABCiJK", H.aa.vooo[Va, :, Oa, Oa], T.aa[Va, Va, :, oa], optimize=True)

    dT.aaa.VVVoOO += (3.0 / 12.0) * np.einsum("ABie,eCJK->ABCiJK", H.aa.vvov[Va, Va, oa, :], T.aa[:, Va, Oa, Oa], optimize=True)
    dT.aaa.VVVoOO -= (6.0 / 12.0) * np.einsum("ABJe,eCiK->ABCiJK", H.aa.vvov[Va, Va, Oa, :], T.aa[:, Va, oa, Oa], optimize=True)
    # (H(2) * T3)_C
    dT.aaa.VVVoOO += (1.0 / 12.0) * (
            +1.0 * np.einsum('mi,CBAmJK->ABCiJK', H.a.oo[oa, oa], T.aaa.VVVoOO, optimize=True)
            + 1.0 * np.einsum('Mi,CBAMJK->ABCiJK', H.a.oo[Oa, oa], T.aaa.VVVOOO, optimize=True)
    )
    dT.aaa.VVVoOO += (2.0 / 12.0) * (
            -1.0 * np.einsum('mJ,CBAmiK->ABCiJK', H.a.oo[oa, Oa], T.aaa.VVVooO, optimize=True)
            + 1.0 * np.einsum('MJ,CBAiMK->ABCiJK', H.a.oo[Oa, Oa], T.aaa.VVVoOO, optimize=True)
    )
    dT.aaa.VVVoOO += (3.0 / 12.0) * (
            -1.0 * np.einsum('Ae,CBeiJK->ABCiJK', H.a.vv[Va, va], T.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('AE,CBEiJK->ABCiJK', H.a.vv[Va, Va], T.aaa.VVVoOO, optimize=True)
    )
    dT.aaa.VVVoOO += (2.0 / 12.0) * (
            -0.5 * np.einsum('mniJ,CBAmnK->ABCiJK', H.aa.oooo[oa, oa, oa, Oa], T.aaa.VVVooO, optimize=True)
            + 1.0 * np.einsum('MniJ,CBAnMK->ABCiJK', H.aa.oooo[Oa, oa, oa, Oa], T.aaa.VVVoOO, optimize=True)
            - 0.5 * np.einsum('MNiJ,CBAMNK->ABCiJK', H.aa.oooo[Oa, Oa, oa, Oa], T.aaa.VVVOOO, optimize=True)
    )
    dT.aaa.VVVoOO += (1.0 / 12.0) * (
            +1.0 * np.einsum('MnKJ,CBAniM->ABCiJK', H.aa.oooo[Oa, oa, Oa, Oa], T.aaa.VVVooO, optimize=True)
            + 0.5 * np.einsum('MNKJ,CBAiMN->ABCiJK', H.aa.oooo[Oa, Oa, Oa, Oa], T.aaa.VVVoOO, optimize=True)
    )
    dT.aaa.VVVoOO += (3.0 / 12.0) * (
            -0.5 * np.einsum('ABef,CfeiJK->ABCiJK', H.aa.vvvv[Va, Va, va, va], T.aaa.VvvoOO, optimize=True)
            + 1.0 * np.einsum('ABEf,CEfiJK->ABCiJK', H.aa.vvvv[Va, Va, Va, va], T.aaa.VVvoOO, optimize=True)
            - 0.5 * np.einsum('ABEF,CFEiJK->ABCiJK', H.aa.vvvv[Va, Va, Va, Va], T.aaa.VVVoOO, optimize=True)
    )
    dT.aaa.VVVoOO += (3.0 / 12.0) * (
            -1.0 * np.einsum('Amie,CBemJK->ABCiJK', H.aa.voov[Va, oa, oa, va], T.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('AmiE,CBEmJK->ABCiJK', H.aa.voov[Va, oa, oa, Va], T.aaa.VVVoOO, optimize=True)
            - 1.0 * np.einsum('AMie,CBeMJK->ABCiJK', H.aa.voov[Va, Oa, oa, va], T.aaa.VVvOOO, optimize=True)
            - 1.0 * np.einsum('AMiE,CBEMJK->ABCiJK', H.aa.voov[Va, Oa, oa, Va], T.aaa.VVVOOO, optimize=True)
    )
    dT.aaa.VVVoOO += (6.0 / 12.0) * (
            +1.0 * np.einsum('AmJe,CBemiK->ABCiJK', H.aa.voov[Va, oa, Oa, va], T.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('AmJE,CBEmiK->ABCiJK', H.aa.voov[Va, oa, Oa, Va], T.aaa.VVVooO, optimize=True)
            - 1.0 * np.einsum('AMJe,CBeiMK->ABCiJK', H.aa.voov[Va, Oa, Oa, va], T.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('AMJE,CBEiMK->ABCiJK', H.aa.voov[Va, Oa, Oa, Va], T.aaa.VVVoOO, optimize=True)
    )
    dT.aaa.VVVoOO += (3.0 / 12.0) * (
            -1.0 * np.einsum('Amie,CBeJKm->ABCiJK', H.ab.voov[Va, ob, oa, vb], T.aab.VVvOOo, optimize=True)
            - 1.0 * np.einsum('AmiE,CBEJKm->ABCiJK', H.ab.voov[Va, ob, oa, Vb], T.aab.VVVOOo, optimize=True)
            - 1.0 * np.einsum('AMie,CBeJKM->ABCiJK', H.ab.voov[Va, Ob, oa, vb], T.aab.VVvOOO, optimize=True)
            - 1.0 * np.einsum('AMiE,CBEJKM->ABCiJK', H.ab.voov[Va, Ob, oa, Vb], T.aab.VVVOOO, optimize=True)
    )
    dT.aaa.VVVoOO += (6.0 / 12.0) * (
            +1.0 * np.einsum('AmJe,CBeiKm->ABCiJK', H.ab.voov[Va, ob, Oa, vb], T.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('AmJE,CBEiKm->ABCiJK', H.ab.voov[Va, ob, Oa, Vb], T.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('AMJe,CBeiKM->ABCiJK', H.ab.voov[Va, Ob, Oa, vb], T.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('AMJE,CBEiKM->ABCiJK', H.ab.voov[Va, Ob, Oa, Vb], T.aab.VVVoOO, optimize=True)
    )

    dT.aaa.VVVoOO -= np.transpose(dT.aaa.VVVoOO, (1, 0, 2, 3, 4, 5)) + np.transpose(dT.aaa.VVVoOO, (0, 2, 1, 3, 4, 5)) \
           + np.transpose(dT.aaa.VVVoOO, (2, 1, 0, 3, 4, 5)) - np.transpose(dT.aaa.VVVoOO, (1, 2, 0, 3, 4, 5)) \
           - np.transpose(dT.aaa.VVVoOO, (2, 0, 1, 3, 4, 5))

    dT.aaa.VVVoOO -= np.transpose(dT.aaa.VVVoOO, (0, 1, 2, 3, 5, 4))

    return dT

def update(T, dT, H, shift, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.aaa.VVVoOO, dT.aaa.VVVoOO = cc_active_loops.update_t3a_111011(
        T.aaa.VVVoOO,
        dT.aaa.VVVoOO,
        H.a.oo[Oa, Oa],
        H.a.vv[Va, Va],
        H.a.oo[oa, oa],
        H.a.vv[va, va],
        shift,
    )

    return T, dT