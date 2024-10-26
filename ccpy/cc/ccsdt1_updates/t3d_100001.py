import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import cc_active_loops

def build(T, dT, H, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    # MM(2,3)
    dT.bbb.VvvooO = (1.0 / 4.0) * (
            -1.0 * np.einsum('Amij,bcmK->AbcijK', H.bb.vooo[Vb, :, ob, ob], T.bb[vb, vb, :, Ob], optimize=True)
    )
    dT.bbb.VvvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('bmij,AcmK->AbcijK', H.bb.vooo[vb, :, ob, ob], T.bb[Vb, vb, :, Ob], optimize=True)
    )
    dT.bbb.VvvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('AmKj,bcmi->AbcijK', H.bb.vooo[Vb, :, Ob, ob], T.bb[vb, vb, :, ob], optimize=True)
    )
    dT.bbb.VvvooO += (4.0 / 4.0) * (
            -1.0 * np.einsum('bmKj,Acmi->AbcijK', H.bb.vooo[vb, :, Ob, ob], T.bb[Vb, vb, :, ob], optimize=True)
    )
    dT.bbb.VvvooO += (4.0 / 4.0) * (
            +1.0 * np.einsum('Abie,ecjK->AbcijK', H.bb.vvov[Vb, vb, ob, :], T.bb[:, vb, ob, Ob], optimize=True)
    )
    dT.bbb.VvvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('cbie,eAjK->AbcijK', H.bb.vvov[vb, vb, ob, :], T.bb[:, Vb, ob, Ob], optimize=True)
    )
    dT.bbb.VvvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('AbKe,ecji->AbcijK', H.bb.vvov[Vb, vb, Ob, :], T.bb[:, vb, ob, ob], optimize=True)
    )
    dT.bbb.VvvooO += (1.0 / 4.0) * (
            +1.0 * np.einsum('cbKe,eAji->AbcijK', H.bb.vvov[vb, vb, Ob, :], T.bb[:, Vb, ob, ob], optimize=True)
    )
    # (H(2) * T3)_C
    dT.bbb.VvvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mi,AcbmjK->AbcijK', H.b.oo[ob, ob], T.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('Mi,AcbjMK->AbcijK', H.b.oo[Ob, ob], T.bbb.VvvoOO, optimize=True)
    )
    dT.bbb.VvvooO += (1.0 / 4.0) * (
            -1.0 * np.einsum('MK,AcbjiM->AbcijK', H.b.oo[Ob, Ob], T.bbb.VvvooO, optimize=True)
    )
    dT.bbb.VvvooO += (1.0 / 4.0) * (
            -1.0 * np.einsum('AE,EcbijK->AbcijK', H.b.vv[Vb, Vb], T.bbb.VvvooO, optimize=True)
    )
    dT.bbb.VvvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('be,AceijK->AbcijK', H.b.vv[vb, vb], T.bbb.VvvooO, optimize=True)
            + 1.0 * np.einsum('bE,AEcijK->AbcijK', H.b.vv[vb, Vb], T.bbb.VVvooO, optimize=True)
    )
    dT.bbb.VvvooO += (1.0 / 4.0) * (
            -0.5 * np.einsum('mnij,AcbmnK->AbcijK', H.bb.oooo[ob, ob, ob, ob], T.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('mNij,AcbmNK->AbcijK', H.bb.oooo[ob, Ob, ob, ob], T.bbb.VvvoOO, optimize=True)
            - 0.5 * np.einsum('MNij,AcbMNK->AbcijK', H.bb.oooo[Ob, Ob, ob, ob], T.bbb.VvvOOO, optimize=True)
    )
    dT.bbb.VvvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('mNKj,AcbmiN->AbcijK', H.bb.oooo[ob, Ob, Ob, ob], T.bbb.VvvooO, optimize=True)
            + 0.5 * np.einsum('MNKj,AcbiMN->AbcijK', H.bb.oooo[Ob, Ob, Ob, ob], T.bbb.VvvoOO, optimize=True)
    )
    dT.bbb.VvvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('AbEf,EcfijK->AbcijK', H.bb.vvvv[Vb, vb, Vb, vb], T.bbb.VvvooO, optimize=True)
            - 0.5 * np.einsum('AbEF,FEcijK->AbcijK', H.bb.vvvv[Vb, vb, Vb, Vb], T.bbb.VVvooO, optimize=True)
    )
    dT.bbb.VvvooO += (1.0 / 4.0) * (
            +0.5 * np.einsum('cbef,AfeijK->AbcijK', H.bb.vvvv[vb, vb, vb, vb], T.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('cbEf,AEfijK->AbcijK', H.bb.vvvv[vb, vb, Vb, vb], T.bbb.VVvooO, optimize=True)
            + 0.5 * np.einsum('cbEF,AFEijK->AbcijK', H.bb.vvvv[vb, vb, Vb, Vb], T.bbb.VVVooO, optimize=True)
    )
    dT.bbb.VvvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('AmiE,EcbmjK->AbcijK', H.bb.voov[Vb, ob, ob, Vb], T.bbb.VvvooO, optimize=True)
            + 1.0 * np.einsum('AMiE,EcbjMK->AbcijK', H.bb.voov[Vb, Ob, ob, Vb], T.bbb.VvvoOO, optimize=True)
    )
    dT.bbb.VvvooO += (4.0 / 4.0) * (
            -1.0 * np.einsum('bmie,AcemjK->AbcijK', H.bb.voov[vb, ob, ob, vb], T.bbb.VvvooO, optimize=True)
            + 1.0 * np.einsum('bMie,AcejMK->AbcijK', H.bb.voov[vb, Ob, ob, vb], T.bbb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('bmiE,AEcmjK->AbcijK', H.bb.voov[vb, ob, ob, Vb], T.bbb.VVvooO, optimize=True)
            - 1.0 * np.einsum('bMiE,AEcjMK->AbcijK', H.bb.voov[vb, Ob, ob, Vb], T.bbb.VVvoOO, optimize=True)
    )
    dT.bbb.VvvooO += (1.0 / 4.0) * (
            +1.0 * np.einsum('AMKE,EcbjiM->AbcijK', H.bb.voov[Vb, Ob, Ob, Vb], T.bbb.VvvooO, optimize=True)
    )
    dT.bbb.VvvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('bMKe,AcejiM->AbcijK', H.bb.voov[vb, Ob, Ob, vb], T.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('bMKE,AEcjiM->AbcijK', H.bb.voov[vb, Ob, Ob, Vb], T.bbb.VVvooO, optimize=True)
    )
    dT.bbb.VvvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('mAEi,EcbmjK->AbcijK', H.ab.ovvo[oa, Vb, Va, ob], T.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('MAEi,EcbMjK->AbcijK', H.ab.ovvo[Oa, Vb, Va, ob], T.abb.VvvOoO, optimize=True)
    )
    dT.bbb.VvvooO += (4.0 / 4.0) * (
            -1.0 * np.einsum('mbei,eAcmjK->AbcijK', H.ab.ovvo[oa, vb, va, ob], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('Mbei,eAcMjK->AbcijK', H.ab.ovvo[Oa, vb, va, ob], T.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('mbEi,EAcmjK->AbcijK', H.ab.ovvo[oa, vb, Va, ob], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('MbEi,EAcMjK->AbcijK', H.ab.ovvo[Oa, vb, Va, ob], T.abb.VVvOoO, optimize=True)
    )
    dT.bbb.VvvooO += (1.0 / 4.0) * (
            +1.0 * np.einsum('MAEK,EcbMji->AbcijK', H.ab.ovvo[Oa, Vb, Va, Ob], T.abb.VvvOoo, optimize=True)
    )
    dT.bbb.VvvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('MbeK,eAcMji->AbcijK', H.ab.ovvo[Oa, vb, va, Ob], T.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('MbEK,EAcMji->AbcijK', H.ab.ovvo[Oa, vb, Va, Ob], T.abb.VVvOoo, optimize=True)
    )

    dT.bbb.VvvooO -= np.transpose(dT.bbb.VvvooO, (0, 2, 1, 3, 4, 5))
    dT.bbb.VvvooO -= np.transpose(dT.bbb.VvvooO, (0, 1, 2, 4, 3, 5))

    return dT

def update(T, dT, H, shift, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.bbb.VvvooO, dT.bbb.VvvooO = cc_active_loops.update_t3d_100001(
        T.bbb.VvvooO,
        dT.bbb.VvvooO,
        H.b.oo[Ob, Ob],
        H.b.vv[Vb, Vb],
        H.b.oo[ob, ob],
        H.b.vv[vb, vb],
        shift,
    )

    return T, dT