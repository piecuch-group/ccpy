import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import cc_active_loops

#@profile
def build(T, dT, H, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    # MM(2,3)
    dT.aaa.VVvoOO = (4.0 / 4.0) * (
            -1.0 * np.einsum('AmiJ,BcmK->ABciJK', H.aa.vooo[Va, :, oa, Oa], T.aa[Va, va, :, Oa], optimize=True)
    )
    dT.aaa.VVvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('cmiJ,BAmK->ABciJK', H.aa.vooo[va, :, oa, Oa], T.aa[Va, Va, :, Oa], optimize=True)
    )
    dT.aaa.VVvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('AmKJ,Bcmi->ABciJK', H.aa.vooo[Va, :, Oa, Oa], T.aa[Va, va, :, oa], optimize=True)
    )
    dT.aaa.VVvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('cmKJ,BAmi->ABciJK', H.aa.vooo[va, :, Oa, Oa], T.aa[Va, Va, :, oa], optimize=True)
    )
    dT.aaa.VVvoOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('ABie,ecJK->ABciJK', H.aa.vvov[Va, Va, oa, :], T.aa[:, va, Oa, Oa], optimize=True)
    )
    dT.aaa.VVvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('cBie,eAJK->ABciJK', H.aa.vvov[va, Va, oa, :], T.aa[:, Va, Oa, Oa], optimize=True)
    )
    dT.aaa.VVvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('ABJe,eciK->ABciJK', H.aa.vvov[Va, Va, Oa, :], T.aa[:, va, oa, Oa], optimize=True)
    )
    dT.aaa.VVvoOO += (4.0 / 4.0) * (
            +1.0 * np.einsum('cBJe,eAiK->ABciJK', H.aa.vvov[va, Va, Oa, :], T.aa[:, Va, oa, Oa], optimize=True)
    )
    # (H(2) * T3)_C
    dT.aaa.VVvoOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('mi,BAcmJK->ABciJK', H.a.oo[oa, oa], T.aaa.VVvoOO, optimize=True)
            + 1.0 * np.einsum('Mi,BAcMJK->ABciJK', H.a.oo[Oa, oa], T.aaa.VVvOOO, optimize=True)
    )
    dT.aaa.VVvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('mJ,BAcmiK->ABciJK', H.a.oo[oa, Oa], T.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('MJ,BAciMK->ABciJK', H.a.oo[Oa, Oa], T.aaa.VVvoOO, optimize=True)
    )
    dT.aaa.VVvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('Ae,BceiJK->ABciJK', H.a.vv[Va, va], T.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('AE,BEciJK->ABciJK', H.a.vv[Va, Va], T.aaa.VVvoOO, optimize=True)
    )
    dT.aaa.VVvoOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('ce,ABeiJK->ABciJK', H.a.vv[va, va], T.aaa.VVvoOO, optimize=True)
            + 1.0 * np.einsum('cE,ABEiJK->ABciJK', H.a.vv[va, Va], T.aaa.VVVoOO, optimize=True)
    )
    dT.aaa.VVvoOO += (2.0 / 4.0) * (
            -0.5 * np.einsum('mniJ,BAcmnK->ABciJK', H.aa.oooo[oa, oa, oa, Oa], T.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('MniJ,BAcnMK->ABciJK', H.aa.oooo[Oa, oa, oa, Oa], T.aaa.VVvoOO, optimize=True)
            - 0.5 * np.einsum('MNiJ,BAcMNK->ABciJK', H.aa.oooo[Oa, Oa, oa, Oa], T.aaa.VVvOOO, optimize=True)
    )
    dT.aaa.VVvoOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('MnKJ,BAcniM->ABciJK', H.aa.oooo[Oa, oa, Oa, Oa], T.aaa.VVvooO, optimize=True)
            + 0.5 * np.einsum('MNKJ,BAciMN->ABciJK', H.aa.oooo[Oa, Oa, Oa, Oa], T.aaa.VVvoOO, optimize=True)
    )
    dT.aaa.VVvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('ABEf,EcfiJK->ABciJK', H.aa.vvvv[Va, Va, Va, va], T.aaa.VvvoOO, optimize=True)
            - 0.5 * np.einsum('ABEF,FEciJK->ABciJK', H.aa.vvvv[Va, Va, Va, Va], T.aaa.VVvoOO, optimize=True)
    )
    dT.aaa.VVvoOO += (2.0 / 4.0) * (
            +0.5 * np.einsum('cBef,AfeiJK->ABciJK', H.aa.vvvv[va, Va, va, va], T.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('cBEf,AEfiJK->ABciJK', H.aa.vvvv[va, Va, Va, va], T.aaa.VVvoOO, optimize=True)
            + 0.5 * np.einsum('cBEF,AFEiJK->ABciJK', H.aa.vvvv[va, Va, Va, Va], T.aaa.VVVoOO, optimize=True)
    )
    dT.aaa.VVvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('Amie,BcemJK->ABciJK', H.aa.voov[Va, oa, oa, va], T.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('AmiE,BEcmJK->ABciJK', H.aa.voov[Va, oa, oa, Va], T.aaa.VVvoOO, optimize=True)
            + 1.0 * np.einsum('AMie,BceMJK->ABciJK', H.aa.voov[Va, Oa, oa, va], T.aaa.VvvOOO, optimize=True)
            - 1.0 * np.einsum('AMiE,BEcMJK->ABciJK', H.aa.voov[Va, Oa, oa, Va], T.aaa.VVvOOO, optimize=True)
    )
    dT.aaa.VVvoOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('cmie,ABemJK->ABciJK', H.aa.voov[va, oa, oa, va], T.aaa.VVvoOO, optimize=True)
            + 1.0 * np.einsum('cmiE,ABEmJK->ABciJK', H.aa.voov[va, oa, oa, Va], T.aaa.VVVoOO, optimize=True)
            + 1.0 * np.einsum('cMie,ABeMJK->ABciJK', H.aa.voov[va, Oa, oa, va], T.aaa.VVvOOO, optimize=True)
            + 1.0 * np.einsum('cMiE,ABEMJK->ABciJK', H.aa.voov[va, Oa, oa, Va], T.aaa.VVVOOO, optimize=True)
    )
    dT.aaa.VVvoOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('AmJe,BcemiK->ABciJK', H.aa.voov[Va, oa, Oa, va], T.aaa.VvvooO, optimize=True)
            + 1.0 * np.einsum('AmJE,BEcmiK->ABciJK', H.aa.voov[Va, oa, Oa, Va], T.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('AMJe,BceiMK->ABciJK', H.aa.voov[Va, Oa, Oa, va], T.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('AMJE,BEciMK->ABciJK', H.aa.voov[Va, Oa, Oa, Va], T.aaa.VVvoOO, optimize=True)
    )
    dT.aaa.VVvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('cmJe,ABemiK->ABciJK', H.aa.voov[va, oa, Oa, va], T.aaa.VVvooO, optimize=True)
            - 1.0 * np.einsum('cmJE,ABEmiK->ABciJK', H.aa.voov[va, oa, Oa, Va], T.aaa.VVVooO, optimize=True)
            + 1.0 * np.einsum('cMJe,ABeiMK->ABciJK', H.aa.voov[va, Oa, Oa, va], T.aaa.VVvoOO, optimize=True)
            + 1.0 * np.einsum('cMJE,ABEiMK->ABciJK', H.aa.voov[va, Oa, Oa, Va], T.aaa.VVVoOO, optimize=True)
    )
    dT.aaa.VVvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('Amie,BceJKm->ABciJK', H.ab.voov[Va, ob, oa, vb], T.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('AmiE,BcEJKm->ABciJK', H.ab.voov[Va, ob, oa, Vb], T.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('AMie,BceJKM->ABciJK', H.ab.voov[Va, Ob, oa, vb], T.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('AMiE,BcEJKM->ABciJK', H.ab.voov[Va, Ob, oa, Vb], T.aab.VvVOOO, optimize=True)
    )
    dT.aaa.VVvoOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('cmie,ABeJKm->ABciJK', H.ab.voov[va, ob, oa, vb], T.aab.VVvOOo, optimize=True)
            + 1.0 * np.einsum('cmiE,ABEJKm->ABciJK', H.ab.voov[va, ob, oa, Vb], T.aab.VVVOOo, optimize=True)
            + 1.0 * np.einsum('cMie,ABeJKM->ABciJK', H.ab.voov[va, Ob, oa, vb], T.aab.VVvOOO, optimize=True)
            + 1.0 * np.einsum('cMiE,ABEJKM->ABciJK', H.ab.voov[va, Ob, oa, Vb], T.aab.VVVOOO, optimize=True)
    )
    dT.aaa.VVvoOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('AmJe,BceiKm->ABciJK', H.ab.voov[Va, ob, Oa, vb], T.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('AmJE,BcEiKm->ABciJK', H.ab.voov[Va, ob, Oa, Vb], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('AMJe,BceiKM->ABciJK', H.ab.voov[Va, Ob, Oa, vb], T.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('AMJE,BcEiKM->ABciJK', H.ab.voov[Va, Ob, Oa, Vb], T.aab.VvVoOO, optimize=True)
    )
    dT.aaa.VVvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('cmJe,ABeiKm->ABciJK', H.ab.voov[va, ob, Oa, vb], T.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('cmJE,ABEiKm->ABciJK', H.ab.voov[va, ob, Oa, Vb], T.aab.VVVoOo, optimize=True)
            - 1.0 * np.einsum('cMJe,ABeiKM->ABciJK', H.ab.voov[va, Ob, Oa, vb], T.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('cMJE,ABEiKM->ABciJK', H.ab.voov[va, Ob, Oa, Vb], T.aab.VVVoOO, optimize=True)
    )

    dT.aaa.VVvoOO -= np.transpose(dT.aaa.VVvoOO, (1, 0, 2, 3, 4, 5))
    dT.aaa.VVvoOO -= np.transpose(dT.aaa.VVvoOO, (0, 1, 2, 3, 5, 4))

    return dT

def update(T, dT, H, shift, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.aaa.VVvoOO, dT.aaa.VVvoOO = cc_active_loops.update_t3a_110011(
        T.aaa.VVvoOO,
        dT.aaa.VVvoOO,
        H.a.oo[Oa, Oa],
        H.a.vv[Va, Va],
        H.a.oo[oa, oa],
        H.a.vv[va, va],
        shift,
    )

    return T, dT