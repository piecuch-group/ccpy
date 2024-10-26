import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops

def build(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.bbb.VvvooO = (2.0 / 4.0) * (
            -1.0 * np.einsum('bmji,AcmK->AbcijK', X.bb.vooo[vb, :, ob, ob], T.bb[Vb, vb, :, Ob], optimize=True)
    )
    dR.bbb.VvvooO += (1.0 / 4.0) * (
            +1.0 * np.einsum('Amji,bcmK->AbcijK', X.bb.vooo[Vb, :, ob, ob], T.bb[vb, vb, :, Ob], optimize=True)
    )
    dR.bbb.VvvooO += (4.0 / 4.0) * (
            +1.0 * np.einsum('bmjK,Acmi->AbcijK', X.bb.vooo[vb, :, ob, Ob], T.bb[Vb, vb, :, ob], optimize=True)
    )
    dR.bbb.VvvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('AmjK,bcmi->AbcijK', X.bb.vooo[Vb, :, ob, Ob], T.bb[vb, vb, :, ob], optimize=True)
    )
    dR.bbb.VvvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('bmji,AcmK->AbcijK', H.bb.vooo[vb, :, ob, ob], R.bb[Vb, vb, :, Ob], optimize=True)
    )
    dR.bbb.VvvooO += (1.0 / 4.0) * (
            +1.0 * np.einsum('Amji,bcmK->AbcijK', H.bb.vooo[Vb, :, ob, ob], R.bb[vb, vb, :, Ob], optimize=True)
    )
    dR.bbb.VvvooO += (4.0 / 4.0) * (
            +1.0 * np.einsum('bmjK,Acmi->AbcijK', H.bb.vooo[vb, :, ob, Ob], R.bb[Vb, vb, :, ob], optimize=True)
    )
    dR.bbb.VvvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('AmjK,bcmi->AbcijK', H.bb.vooo[Vb, :, ob, Ob], R.bb[vb, vb, :, ob], optimize=True)
    )
    dR.bbb.VvvooO += (4.0 / 4.0) * (
            +1.0 * np.einsum('bAje,eciK->AbcijK', X.bb.vvov[vb, Vb, ob, :], T.bb[:, vb, ob, Ob], optimize=True)
    )
    dR.bbb.VvvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('bcje,eAiK->AbcijK', X.bb.vvov[vb, vb, ob, :], T.bb[:, Vb, ob, Ob], optimize=True)
    )
    dR.bbb.VvvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('bAKe,ecij->AbcijK', X.bb.vvov[vb, Vb, Ob, :], T.bb[:, vb, ob, ob], optimize=True)
    )
    dR.bbb.VvvooO += (1.0 / 4.0) * (
            +1.0 * np.einsum('bcKe,eAij->AbcijK', X.bb.vvov[vb, vb, Ob, :], T.bb[:, Vb, ob, ob], optimize=True)
    )
    dR.bbb.VvvooO += (4.0 / 4.0) * (
            +1.0 * np.einsum('bAje,eciK->AbcijK', H.bb.vvov[vb, Vb, ob, :], R.bb[:, vb, ob, Ob], optimize=True)
    )
    dR.bbb.VvvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('bcje,eAiK->AbcijK', H.bb.vvov[vb, vb, ob, :], R.bb[:, Vb, ob, Ob], optimize=True)
    )
    dR.bbb.VvvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('bAKe,ecij->AbcijK', H.bb.vvov[vb, Vb, Ob, :], R.bb[:, vb, ob, ob], optimize=True)
    )
    dR.bbb.VvvooO += (1.0 / 4.0) * (
            +1.0 * np.einsum('bcKe,eAij->AbcijK', H.bb.vvov[vb, vb, Ob, :], R.bb[:, Vb, ob, ob], optimize=True)
    )
    # of terms =  16
    dR.bbb.VvvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('be,AceijK->AbcijK', X.b.vv[vb, vb], T.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('bE,EAcijK->AbcijK', X.b.vv[vb, Vb], T.bbb.VVvooO, optimize=True)
    )
    dR.bbb.VvvooO += (1.0 / 4.0) * (
            -1.0 * np.einsum('AE,EcbijK->AbcijK', X.b.vv[Vb, Vb], T.bbb.VvvooO, optimize=True)
    )
    dR.bbb.VvvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mj,AcbimK->AbcijK', X.b.oo[ob, ob], T.bbb.VvvooO, optimize=True)
            + 1.0 * np.einsum('Mj,AcbiMK->AbcijK', X.b.oo[Ob, ob], T.bbb.VvvoOO, optimize=True)
    )
    dR.bbb.VvvooO += (1.0 / 4.0) * (
            +1.0 * np.einsum('MK,AcbijM->AbcijK', X.b.oo[Ob, Ob], T.bbb.VvvooO, optimize=True)
    )
    dR.bbb.VvvooO += (1.0 / 4.0) * (
            -0.5 * np.einsum('mnij,AcbmnK->AbcijK', X.bb.oooo[ob, ob, ob, ob], T.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('mNij,AcbmNK->AbcijK', X.bb.oooo[ob, Ob, ob, ob], T.bbb.VvvoOO, optimize=True)
            - 0.5 * np.einsum('MNij,AcbMNK->AbcijK', X.bb.oooo[Ob, Ob, ob, ob], T.bbb.VvvOOO, optimize=True)
    )
    dR.bbb.VvvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('mNKj,AcbmiN->AbcijK', X.bb.oooo[ob, Ob, Ob, ob], T.bbb.VvvooO, optimize=True)
            + 0.5 * np.einsum('MNKj,AcbiMN->AbcijK', X.bb.oooo[Ob, Ob, Ob, ob], T.bbb.VvvoOO, optimize=True)
    )
    dR.bbb.VvvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('AbeF,FceijK->AbcijK', X.bb.vvvv[Vb, vb, vb, Vb], T.bbb.VvvooO, optimize=True)
            - 0.5 * np.einsum('AbEF,FEcijK->AbcijK', X.bb.vvvv[Vb, vb, Vb, Vb], T.bbb.VVvooO, optimize=True)
    )
    dR.bbb.VvvooO += (1.0 / 4.0) * (
            +0.5 * np.einsum('cbef,AfeijK->AbcijK', X.bb.vvvv[vb, vb, vb, vb], T.bbb.VvvooO, optimize=True)
            + 1.0 * np.einsum('cbeF,AFeijK->AbcijK', X.bb.vvvv[vb, vb, vb, Vb], T.bbb.VVvooO, optimize=True)
            + 0.5 * np.einsum('cbEF,AFEijK->AbcijK', X.bb.vvvv[vb, vb, Vb, Vb], T.bbb.VVVooO, optimize=True)
    )
    dR.bbb.VvvooO += (4.0 / 4.0) * (
            -1.0 * np.einsum('bmje,AceimK->AbcijK', X.bb.voov[vb, ob, ob, vb], T.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('bmjE,EAcimK->AbcijK', X.bb.voov[vb, ob, ob, Vb], T.bbb.VVvooO, optimize=True)
            - 1.0 * np.einsum('bMje,AceiMK->AbcijK', X.bb.voov[vb, Ob, ob, vb], T.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('bMjE,EAciMK->AbcijK', X.bb.voov[vb, Ob, ob, Vb], T.bbb.VVvoOO, optimize=True)
    )
    dR.bbb.VvvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('AmjE,EcbimK->AbcijK', X.bb.voov[Vb, ob, ob, Vb], T.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('AMjE,EcbiMK->AbcijK', X.bb.voov[Vb, Ob, ob, Vb], T.bbb.VvvoOO, optimize=True)
    )
    dR.bbb.VvvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('bMKe,AceijM->AbcijK', X.bb.voov[vb, Ob, Ob, vb], T.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('bMKE,EAcijM->AbcijK', X.bb.voov[vb, Ob, Ob, Vb], T.bbb.VVvooO, optimize=True)
    )
    dR.bbb.VvvooO += (1.0 / 4.0) * (
            -1.0 * np.einsum('AMKE,EcbijM->AbcijK', X.bb.voov[Vb, Ob, Ob, Vb], T.bbb.VvvooO, optimize=True)
    )
    dR.bbb.VvvooO += (4.0 / 4.0) * (
            +1.0 * np.einsum('mbej,eAcmiK->AbcijK', X.ab.ovvo[oa, vb, va, ob], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('mbEj,EAcmiK->AbcijK', X.ab.ovvo[oa, vb, Va, ob], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('Mbej,eAcMiK->AbcijK', X.ab.ovvo[Oa, vb, va, ob], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('MbEj,EAcMiK->AbcijK', X.ab.ovvo[Oa, vb, Va, ob], T.abb.VVvOoO, optimize=True)
    )
    dR.bbb.VvvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('mAEj,EbcmiK->AbcijK', X.ab.ovvo[oa, Vb, Va, ob], T.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('MAEj,EbcMiK->AbcijK', X.ab.ovvo[Oa, Vb, Va, ob], T.abb.VvvOoO, optimize=True)
    )
    dR.bbb.VvvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('MbeK,eAcMji->AbcijK', X.ab.ovvo[Oa, vb, va, Ob], T.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('MbEK,EAcMji->AbcijK', X.ab.ovvo[Oa, vb, Va, Ob], T.abb.VVvOoo, optimize=True)
    )
    dR.bbb.VvvooO += (1.0 / 4.0) * (
            -1.0 * np.einsum('MAEK,EbcMji->AbcijK', X.ab.ovvo[Oa, Vb, Va, Ob], T.abb.VvvOoo, optimize=True)
    )
    dR.bbb.VvvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mj,AcbimK->AbcijK', H.b.oo[ob, ob], R.bbb.VvvooO, optimize=True)
            + 1.0 * np.einsum('Mj,AcbiMK->AbcijK', H.b.oo[Ob, ob], R.bbb.VvvoOO, optimize=True)
    )
    dR.bbb.VvvooO += (1.0 / 4.0) * (
            +1.0 * np.einsum('MK,AcbijM->AbcijK', H.b.oo[Ob, Ob], R.bbb.VvvooO, optimize=True)
    )
    dR.bbb.VvvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('be,AceijK->AbcijK', H.b.vv[vb, vb], R.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('bE,EAcijK->AbcijK', H.b.vv[vb, Vb], R.bbb.VVvooO, optimize=True)
    )
    dR.bbb.VvvooO += (1.0 / 4.0) * (
            -1.0 * np.einsum('AE,EcbijK->AbcijK', H.b.vv[Vb, Vb], R.bbb.VvvooO, optimize=True)
    )
    dR.bbb.VvvooO += (1.0 / 4.0) * (
            -0.5 * np.einsum('mnij,AcbmnK->AbcijK', H.bb.oooo[ob, ob, ob, ob], R.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('mNij,AcbmNK->AbcijK', H.bb.oooo[ob, Ob, ob, ob], R.bbb.VvvoOO, optimize=True)
            - 0.5 * np.einsum('MNij,AcbMNK->AbcijK', H.bb.oooo[Ob, Ob, ob, ob], R.bbb.VvvOOO, optimize=True)
    )
    dR.bbb.VvvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('mNKj,AcbmiN->AbcijK', H.bb.oooo[ob, Ob, Ob, ob], R.bbb.VvvooO, optimize=True)
            + 0.5 * np.einsum('MNKj,AcbiMN->AbcijK', H.bb.oooo[Ob, Ob, Ob, ob], R.bbb.VvvoOO, optimize=True)
    )
    dR.bbb.VvvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('AbeF,FceijK->AbcijK', H.bb.vvvv[Vb, vb, vb, Vb], R.bbb.VvvooO, optimize=True)
            - 0.5 * np.einsum('AbEF,FEcijK->AbcijK', H.bb.vvvv[Vb, vb, Vb, Vb], R.bbb.VVvooO, optimize=True)
    )
    dR.bbb.VvvooO += (1.0 / 4.0) * (
            +0.5 * np.einsum('cbef,AfeijK->AbcijK', H.bb.vvvv[vb, vb, vb, vb], R.bbb.VvvooO, optimize=True)
            + 1.0 * np.einsum('cbeF,AFeijK->AbcijK', H.bb.vvvv[vb, vb, vb, Vb], R.bbb.VVvooO, optimize=True)
            + 0.5 * np.einsum('cbEF,AFEijK->AbcijK', H.bb.vvvv[vb, vb, Vb, Vb], R.bbb.VVVooO, optimize=True)
    )
    dR.bbb.VvvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('AmiE,EcbmjK->AbcijK', H.bb.voov[Vb, ob, ob, Vb], R.bbb.VvvooO, optimize=True)
            + 1.0 * np.einsum('AMiE,EcbjMK->AbcijK', H.bb.voov[Vb, Ob, ob, Vb], R.bbb.VvvoOO, optimize=True)
    )
    dR.bbb.VvvooO += (4.0 / 4.0) * (
            -1.0 * np.einsum('bmie,AcemjK->AbcijK', H.bb.voov[vb, ob, ob, vb], R.bbb.VvvooO, optimize=True)
            + 1.0 * np.einsum('bmiE,AEcmjK->AbcijK', H.bb.voov[vb, ob, ob, Vb], R.bbb.VVvooO, optimize=True)
            + 1.0 * np.einsum('bMie,AcejMK->AbcijK', H.bb.voov[vb, Ob, ob, vb], R.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('bMiE,AEcjMK->AbcijK', H.bb.voov[vb, Ob, ob, Vb], R.bbb.VVvoOO, optimize=True)
    )
    dR.bbb.VvvooO += (1.0 / 4.0) * (
            +1.0 * np.einsum('AMKE,EcbjiM->AbcijK', H.bb.voov[Vb, Ob, Ob, Vb], R.bbb.VvvooO, optimize=True)
    )
    dR.bbb.VvvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('bMKe,AcejiM->AbcijK', H.bb.voov[vb, Ob, Ob, vb], R.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('bMKE,AEcjiM->AbcijK', H.bb.voov[vb, Ob, Ob, Vb], R.bbb.VVvooO, optimize=True)
    )
    dR.bbb.VvvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mAEi,EbcmjK->AbcijK', H.ab.ovvo[oa, Vb, Va, ob], R.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('MAEi,EbcMjK->AbcijK', H.ab.ovvo[Oa, Vb, Va, ob], R.abb.VvvOoO, optimize=True)
    )
    dR.bbb.VvvooO += (4.0 / 4.0) * (
            -1.0 * np.einsum('mbei,eAcmjK->AbcijK', H.ab.ovvo[oa, vb, va, ob], R.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('mbEi,EAcmjK->AbcijK', H.ab.ovvo[oa, vb, Va, ob], R.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('Mbei,eAcMjK->AbcijK', H.ab.ovvo[Oa, vb, va, ob], R.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('MbEi,EAcMjK->AbcijK', H.ab.ovvo[Oa, vb, Va, ob], R.abb.VVvOoO, optimize=True)
    )
    dR.bbb.VvvooO += (1.0 / 4.0) * (
            +1.0 * np.einsum('MAEK,EbcMij->AbcijK', H.ab.ovvo[Oa, Vb, Va, Ob], R.abb.VvvOoo, optimize=True)
    )
    dR.bbb.VvvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('MbeK,eAcMij->AbcijK', H.ab.ovvo[Oa, vb, va, Ob], R.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('MbEK,EAcMij->AbcijK', H.ab.ovvo[Oa, vb, Va, Ob], R.abb.VVvOoo, optimize=True)
    )
    # of terms =  32

    dR.bbb.VvvooO -= np.transpose(dR.bbb.VvvooO, (0, 1, 2, 4, 3, 5))
    dR.bbb.VvvooO -= np.transpose(dR.bbb.VvvooO, (0, 2, 1, 3, 4, 5))

    return dR

def update(R, omega, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    R.bbb.VvvooO = eomcc_active_loops.update_r3d_100001(
        R.bbb.VvvooO,
        omega,
        H.b.oo[Ob, Ob],
        H.b.vv[Vb, Vb],
        H.b.oo[ob, ob],
        H.b.vv[vb, vb],
        0.0,
    )
    return R
