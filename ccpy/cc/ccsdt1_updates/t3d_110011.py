import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import cc_active_loops

def build(T, dT, H, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    # MM(2,3)
    dT.bbb.VVvoOO = (4.0 / 4.0) * (
            -1.0 * np.einsum('AmiJ,BcmK->ABciJK', H.bb.vooo[Vb, :, ob, Ob], T.bb[Vb, vb, :, Ob], optimize=True)
    )
    dT.bbb.VVvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('cmiJ,BAmK->ABciJK', H.bb.vooo[vb, :, ob, Ob], T.bb[Vb, Vb, :, Ob], optimize=True)
    )
    dT.bbb.VVvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('AmKJ,Bcmi->ABciJK', H.bb.vooo[Vb, :, Ob, Ob], T.bb[Vb, vb, :, ob], optimize=True)
    )
    dT.bbb.VVvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('cmKJ,BAmi->ABciJK', H.bb.vooo[vb, :, Ob, Ob], T.bb[Vb, Vb, :, ob], optimize=True)
    )
    dT.bbb.VVvoOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('ABie,ecJK->ABciJK', H.bb.vvov[Vb, Vb, ob, :], T.bb[:, vb, Ob, Ob], optimize=True)
    )
    dT.bbb.VVvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('cBie,eAJK->ABciJK', H.bb.vvov[vb, Vb, ob, :], T.bb[:, Vb, Ob, Ob], optimize=True)
    )
    dT.bbb.VVvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('ABJe,eciK->ABciJK', H.bb.vvov[Vb, Vb, Ob, :], T.bb[:, vb, ob, Ob], optimize=True)
    )
    dT.bbb.VVvoOO += (4.0 / 4.0) * (
            +1.0 * np.einsum('cBJe,eAiK->ABciJK', H.bb.vvov[vb, Vb, Ob, :], T.bb[:, Vb, ob, Ob], optimize=True)
    )
    # (H(2) * T3)_C
    dT.bbb.VVvoOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('mi,BAcmJK->ABciJK', H.b.oo[ob, ob], T.bbb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('Mi,BAcMJK->ABciJK', H.b.oo[Ob, ob], T.bbb.VVvOOO, optimize=True)
    )
    dT.bbb.VVvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('mJ,BAcmiK->ABciJK', H.b.oo[ob, Ob], T.bbb.VVvooO, optimize=True)
            + 1.0 * np.einsum('MJ,BAciMK->ABciJK', H.b.oo[Ob, Ob], T.bbb.VVvoOO, optimize=True)
    )
    dT.bbb.VVvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('Ae,BceiJK->ABciJK', H.b.vv[Vb, vb], T.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('AE,BEciJK->ABciJK', H.b.vv[Vb, Vb], T.bbb.VVvoOO, optimize=True)
    )
    dT.bbb.VVvoOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('ce,ABeiJK->ABciJK', H.b.vv[vb, vb], T.bbb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('cE,ABEiJK->ABciJK', H.b.vv[vb, Vb], T.bbb.VVVoOO, optimize=True)
    )
    dT.bbb.VVvoOO += (2.0 / 4.0) * (
            -0.5 * np.einsum('mniJ,BAcmnK->ABciJK', H.bb.oooo[ob, ob, ob, Ob], T.bbb.VVvooO, optimize=True)
            + 1.0 * np.einsum('MniJ,BAcnMK->ABciJK', H.bb.oooo[Ob, ob, ob, Ob], T.bbb.VVvoOO, optimize=True)
            - 0.5 * np.einsum('MNiJ,BAcMNK->ABciJK', H.bb.oooo[Ob, Ob, ob, Ob], T.bbb.VVvOOO, optimize=True)
    )
    dT.bbb.VVvoOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('MnKJ,BAcniM->ABciJK', H.bb.oooo[Ob, ob, Ob, Ob], T.bbb.VVvooO, optimize=True)
            + 0.5 * np.einsum('MNKJ,BAciMN->ABciJK', H.bb.oooo[Ob, Ob, Ob, Ob], T.bbb.VVvoOO, optimize=True)
    )
    dT.bbb.VVvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('ABEf,EcfiJK->ABciJK', H.bb.vvvv[Vb, Vb, Vb, vb], T.bbb.VvvoOO, optimize=True)
            - 0.5 * np.einsum('ABEF,FEciJK->ABciJK', H.bb.vvvv[Vb, Vb, Vb, Vb], T.bbb.VVvoOO, optimize=True)
    )
    dT.bbb.VVvoOO += (2.0 / 4.0) * (
            +0.5 * np.einsum('cBef,AfeiJK->ABciJK', H.bb.vvvv[vb, Vb, vb, vb], T.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('cBEf,AEfiJK->ABciJK', H.bb.vvvv[vb, Vb, Vb, vb], T.bbb.VVvoOO, optimize=True)
            + 0.5 * np.einsum('cBEF,AFEiJK->ABciJK', H.bb.vvvv[vb, Vb, Vb, Vb], T.bbb.VVVoOO, optimize=True)
    )
    dT.bbb.VVvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('Amie,BcemJK->ABciJK', H.bb.voov[Vb, ob, ob, vb], T.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('AmiE,BEcmJK->ABciJK', H.bb.voov[Vb, ob, ob, Vb], T.bbb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('AMie,BceMJK->ABciJK', H.bb.voov[Vb, Ob, ob, vb], T.bbb.VvvOOO, optimize=True)
            - 1.0 * np.einsum('AMiE,BEcMJK->ABciJK', H.bb.voov[Vb, Ob, ob, Vb], T.bbb.VVvOOO, optimize=True)
    )
    dT.bbb.VVvoOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('cmie,ABemJK->ABciJK', H.bb.voov[vb, ob, ob, vb], T.bbb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('cmiE,ABEmJK->ABciJK', H.bb.voov[vb, ob, ob, Vb], T.bbb.VVVoOO, optimize=True)
            + 1.0 * np.einsum('cMie,ABeMJK->ABciJK', H.bb.voov[vb, Ob, ob, vb], T.bbb.VVvOOO, optimize=True)
            + 1.0 * np.einsum('cMiE,ABEMJK->ABciJK', H.bb.voov[vb, Ob, ob, Vb], T.bbb.VVVOOO, optimize=True)
    )
    dT.bbb.VVvoOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('AmJe,BcemiK->ABciJK', H.bb.voov[Vb, ob, Ob, vb], T.bbb.VvvooO, optimize=True)
            + 1.0 * np.einsum('AmJE,BEcmiK->ABciJK', H.bb.voov[Vb, ob, Ob, Vb], T.bbb.VVvooO, optimize=True)
            + 1.0 * np.einsum('AMJe,BceiMK->ABciJK', H.bb.voov[Vb, Ob, Ob, vb], T.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('AMJE,BEciMK->ABciJK', H.bb.voov[Vb, Ob, Ob, Vb], T.bbb.VVvoOO, optimize=True)
    )
    dT.bbb.VVvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('cmJe,ABemiK->ABciJK', H.bb.voov[vb, ob, Ob, vb], T.bbb.VVvooO, optimize=True)
            - 1.0 * np.einsum('cmJE,ABEmiK->ABciJK', H.bb.voov[vb, ob, Ob, Vb], T.bbb.VVVooO, optimize=True)
            + 1.0 * np.einsum('cMJe,ABeiMK->ABciJK', H.bb.voov[vb, Ob, Ob, vb], T.bbb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('cMJE,ABEiMK->ABciJK', H.bb.voov[vb, Ob, Ob, Vb], T.bbb.VVVoOO, optimize=True)
    )
    dT.bbb.VVvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mAei,eBcmJK->ABciJK', H.ab.ovvo[oa, Vb, va, ob], T.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('mAEi,EBcmJK->ABciJK', H.ab.ovvo[oa, Vb, Va, ob], T.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('MAei,eBcMJK->ABciJK', H.ab.ovvo[Oa, Vb, va, ob], T.abb.vVvOOO, optimize=True)
            + 1.0 * np.einsum('MAEi,EBcMJK->ABciJK', H.ab.ovvo[Oa, Vb, Va, ob], T.abb.VVvOOO, optimize=True)
    )
    dT.bbb.VVvoOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('mcei,eABmJK->ABciJK', H.ab.ovvo[oa, vb, va, ob], T.abb.vVVoOO, optimize=True)
            + 1.0 * np.einsum('mcEi,EABmJK->ABciJK', H.ab.ovvo[oa, vb, Va, ob], T.abb.VVVoOO, optimize=True)
            + 1.0 * np.einsum('Mcei,eABMJK->ABciJK', H.ab.ovvo[Oa, vb, va, ob], T.abb.vVVOOO, optimize=True)
            + 1.0 * np.einsum('McEi,EABMJK->ABciJK', H.ab.ovvo[Oa, vb, Va, ob], T.abb.VVVOOO, optimize=True)
    )
    dT.bbb.VVvoOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('mAeJ,eBcmiK->ABciJK', H.ab.ovvo[oa, Vb, va, Ob], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('mAEJ,EBcmiK->ABciJK', H.ab.ovvo[oa, Vb, Va, Ob], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('MAeJ,eBcMiK->ABciJK', H.ab.ovvo[Oa, Vb, va, Ob], T.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('MAEJ,EBcMiK->ABciJK', H.ab.ovvo[Oa, Vb, Va, Ob], T.abb.VVvOoO, optimize=True)
    )
    dT.bbb.VVvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('mceJ,eABmiK->ABciJK', H.ab.ovvo[oa, vb, va, Ob], T.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('mcEJ,EABmiK->ABciJK', H.ab.ovvo[oa, vb, Va, Ob], T.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('MceJ,eABMiK->ABciJK', H.ab.ovvo[Oa, vb, va, Ob], T.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('McEJ,EABMiK->ABciJK', H.ab.ovvo[Oa, vb, Va, Ob], T.abb.VVVOoO, optimize=True)
    )

    dT.bbb.VVvoOO -= np.transpose(dT.bbb.VVvoOO, (1, 0, 2, 3, 4, 5))
    dT.bbb.VVvoOO -= np.transpose(dT.bbb.VVvoOO, (0, 1, 2, 3, 5, 4))

    return dT

def update(T, dT, H, shift, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.bbb.VVvoOO, dT.bbb.VVvoOO = cc_active_loops.update_t3d_110011(
        T.bbb.VVvoOO,
        dT.bbb.VVvoOO,
        H.b.oo[Ob, Ob],
        H.b.vv[Vb, Vb],
        H.b.oo[ob, ob],
        H.b.vv[vb, vb],
        shift,
    )

    return T, dT