import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import cc_active_loops

def build(T, dT, H, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    # MM(2,3)
    dT.bbb.VVvooO = (2.0 / 4.0) * (
            -1.0 * np.einsum('Amij,BcmK->ABcijK', H.bb.vooo[Vb, :, ob, ob], T.bb[Vb, vb, :, Ob], optimize=True)
    )
    dT.bbb.VVvooO += (1.0 / 4.0) * (
            +1.0 * np.einsum('cmij,BAmK->ABcijK', H.bb.vooo[vb, :, ob, ob], T.bb[Vb, Vb, :, Ob], optimize=True)
    )
    dT.bbb.VVvooO += (4.0 / 4.0) * (
            +1.0 * np.einsum('AmKj,Bcmi->ABcijK', H.bb.vooo[Vb, :, Ob, ob], T.bb[Vb, vb, :, ob], optimize=True)
    )
    dT.bbb.VVvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('cmKj,BAmi->ABcijK', H.bb.vooo[vb, :, Ob, ob], T.bb[Vb, Vb, :, ob], optimize=True)
    )
    dT.bbb.VVvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('ABie,ecjK->ABcijK', H.bb.vvov[Vb, Vb, ob, :], T.bb[:, vb, ob, Ob], optimize=True)
    )
    dT.bbb.VVvooO += (4.0 / 4.0) * (
            -1.0 * np.einsum('cBie,eAjK->ABcijK', H.bb.vvov[vb, Vb, ob, :], T.bb[:, Vb, ob, Ob], optimize=True)
    )
    dT.bbb.VVvooO += (1.0 / 4.0) * (
            -1.0 * np.einsum('ABKe,ecji->ABcijK', H.bb.vvov[Vb, Vb, Ob, :], T.bb[:, vb, ob, ob], optimize=True)
    )
    dT.bbb.VVvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('cBKe,eAji->ABcijK', H.bb.vvov[vb, Vb, Ob, :], T.bb[:, Vb, ob, ob], optimize=True)
    )
    # (H(2) * T3)_C
    dT.bbb.VVvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mi,BAcmjK->ABcijK', H.b.oo[ob, ob], T.bbb.VVvooO, optimize=True)
            - 1.0 * np.einsum('Mi,BAcjMK->ABcijK', H.b.oo[Ob, ob], T.bbb.VVvoOO, optimize=True)
    )
    dT.bbb.VVvooO += (1.0 / 4.0) * (
            -1.0 * np.einsum('MK,BAcjiM->ABcijK', H.b.oo[Ob, Ob], T.bbb.VVvooO, optimize=True)
    )
    dT.bbb.VVvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('Ae,BceijK->ABcijK', H.b.vv[Vb, vb], T.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('AE,BEcijK->ABcijK', H.b.vv[Vb, Vb], T.bbb.VVvooO, optimize=True)
    )
    dT.bbb.VVvooO += (1.0 / 4.0) * (
            +1.0 * np.einsum('ce,ABeijK->ABcijK', H.b.vv[vb, vb], T.bbb.VVvooO, optimize=True)
            + 1.0 * np.einsum('cE,ABEijK->ABcijK', H.b.vv[vb, Vb], T.bbb.VVVooO, optimize=True)
    )
    dT.bbb.VVvooO += (1.0 / 4.0) * (
            -0.5 * np.einsum('mnij,BAcmnK->ABcijK', H.bb.oooo[ob, ob, ob, ob], T.bbb.VVvooO, optimize=True)
            + 1.0 * np.einsum('Mnij,BAcnMK->ABcijK', H.bb.oooo[Ob, ob, ob, ob], T.bbb.VVvoOO, optimize=True)
            - 0.5 * np.einsum('MNij,BAcMNK->ABcijK', H.bb.oooo[Ob, Ob, ob, ob], T.bbb.VVvOOO, optimize=True)
    )
    dT.bbb.VVvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('MnKj,BAcniM->ABcijK', H.bb.oooo[Ob, ob, Ob, ob], T.bbb.VVvooO, optimize=True)
            + 0.5 * np.einsum('MNKj,BAciMN->ABcijK', H.bb.oooo[Ob, Ob, Ob, ob], T.bbb.VVvoOO, optimize=True)
    )
    dT.bbb.VVvooO += (1.0 / 4.0) * (
            +1.0 * np.einsum('ABeF,FceijK->ABcijK', H.bb.vvvv[Vb, Vb, vb, Vb], T.bbb.VvvooO, optimize=True)
            - 0.5 * np.einsum('ABEF,FEcijK->ABcijK', H.bb.vvvv[Vb, Vb, Vb, Vb], T.bbb.VVvooO, optimize=True)
    )
    dT.bbb.VVvooO += (2.0 / 4.0) * (
            +0.5 * np.einsum('cBef,AfeijK->ABcijK', H.bb.vvvv[vb, Vb, vb, vb], T.bbb.VvvooO, optimize=True)
            + 1.0 * np.einsum('cBeF,AFeijK->ABcijK', H.bb.vvvv[vb, Vb, vb, Vb], T.bbb.VVvooO, optimize=True)
            + 0.5 * np.einsum('cBEF,AFEijK->ABcijK', H.bb.vvvv[vb, Vb, Vb, Vb], T.bbb.VVVooO, optimize=True)
    )
    dT.bbb.VVvooO += (4.0 / 4.0) * (
            +1.0 * np.einsum('Amie,BcemjK->ABcijK', H.bb.voov[Vb, ob, ob, vb], T.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('AmiE,BEcmjK->ABcijK', H.bb.voov[Vb, ob, ob, Vb], T.bbb.VVvooO, optimize=True)
            - 1.0 * np.einsum('AMie,BcejMK->ABcijK', H.bb.voov[Vb, Ob, ob, vb], T.bbb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('AMiE,BEcjMK->ABcijK', H.bb.voov[Vb, Ob, ob, Vb], T.bbb.VVvoOO, optimize=True)
    )
    dT.bbb.VVvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('cmie,ABemjK->ABcijK', H.bb.voov[vb, ob, ob, vb], T.bbb.VVvooO, optimize=True)
            + 1.0 * np.einsum('cmiE,ABEmjK->ABcijK', H.bb.voov[vb, ob, ob, Vb], T.bbb.VVVooO, optimize=True)
            - 1.0 * np.einsum('cMie,ABejMK->ABcijK', H.bb.voov[vb, Ob, ob, vb], T.bbb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('cMiE,ABEjMK->ABcijK', H.bb.voov[vb, Ob, ob, Vb], T.bbb.VVVoOO, optimize=True)
    )
    dT.bbb.VVvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('AMKe,BcejiM->ABcijK', H.bb.voov[Vb, Ob, Ob, vb], T.bbb.VvvooO, optimize=True)
            + 1.0 * np.einsum('AMKE,BEcjiM->ABcijK', H.bb.voov[Vb, Ob, Ob, Vb], T.bbb.VVvooO, optimize=True)
    )
    dT.bbb.VVvooO += (1.0 / 4.0) * (
            -1.0 * np.einsum('cMKe,ABejiM->ABcijK', H.bb.voov[vb, Ob, Ob, vb], T.bbb.VVvooO, optimize=True)
            - 1.0 * np.einsum('cMKE,ABEjiM->ABcijK', H.bb.voov[vb, Ob, Ob, Vb], T.bbb.VVVooO, optimize=True)
    )
    dT.bbb.VVvooO += (4.0 / 4.0) * (
            +1.0 * np.einsum('mAei,eBcmjK->ABcijK', H.ab.ovvo[oa, Vb, va, ob], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('mAEi,EBcmjK->ABcijK', H.ab.ovvo[oa, Vb, Va, ob], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('MAei,eBcMjK->ABcijK', H.ab.ovvo[Oa, Vb, va, ob], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('MAEi,EBcMjK->ABcijK', H.ab.ovvo[Oa, Vb, Va, ob], T.abb.VVvOoO, optimize=True)
    )
    dT.bbb.VVvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mcei,eABmjK->ABcijK', H.ab.ovvo[oa, vb, va, ob], T.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('mcEi,EABmjK->ABcijK', H.ab.ovvo[oa, vb, Va, ob], T.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('Mcei,eABMjK->ABcijK', H.ab.ovvo[Oa, vb, va, ob], T.abb.vVVOoO, optimize=True)
            + 1.0 * np.einsum('McEi,EABMjK->ABcijK', H.ab.ovvo[Oa, vb, Va, ob], T.abb.VVVOoO, optimize=True)
    )
    dT.bbb.VVvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('MAeK,eBcMji->ABcijK', H.ab.ovvo[Oa, Vb, va, Ob], T.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('MAEK,EBcMji->ABcijK', H.ab.ovvo[Oa, Vb, Va, Ob], T.abb.VVvOoo, optimize=True)
    )
    dT.bbb.VVvooO += (1.0 / 4.0) * (
            -1.0 * np.einsum('MceK,eABMji->ABcijK', H.ab.ovvo[Oa, vb, va, Ob], T.abb.vVVOoo, optimize=True)
            - 1.0 * np.einsum('McEK,EABMji->ABcijK', H.ab.ovvo[Oa, vb, Va, Ob], T.abb.VVVOoo, optimize=True)
    )

    dT.bbb.VVvooO -= np.transpose(dT.bbb.VVvooO, (1, 0, 2, 3, 4, 5))
    dT.bbb.VVvooO -= np.transpose(dT.bbb.VVvooO, (0, 1, 2, 4, 3, 5))

    return dT

def update(T, dT, H, shift, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.bbb.VVvooO, dT.bbb.VVvooO = cc_active_loops.update_t3d_110001(
        T.bbb.VVvooO,
        dT.bbb.VVvooO,
        H.b.oo[Ob, Ob],
        H.b.vv[Vb, Vb],
        H.b.oo[ob, ob],
        H.b.vv[vb, vb],
        shift,
    )

    return T, dT