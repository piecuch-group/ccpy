import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops

def build(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.bbb.VVvooO = (2.0 / 4.0) * (
            -1.0 * np.einsum('Bmji,AcmK->ABcijK', X.bb.vooo[Vb, :, ob, ob], T.bb[Vb, vb, :, Ob], optimize=True)
    )
    dR.bbb.VVvooO += (1.0 / 4.0) * (
            +1.0 * np.einsum('cmji,ABmK->ABcijK', X.bb.vooo[vb, :, ob, ob], T.bb[Vb, Vb, :, Ob], optimize=True)
    )
    dR.bbb.VVvooO += (4.0 / 4.0) * (
            +1.0 * np.einsum('BmjK,Acmi->ABcijK', X.bb.vooo[Vb, :, ob, Ob], T.bb[Vb, vb, :, ob], optimize=True)
    )
    dR.bbb.VVvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('cmjK,ABmi->ABcijK', X.bb.vooo[vb, :, ob, Ob], T.bb[Vb, Vb, :, ob], optimize=True)
    )
    dR.bbb.VVvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('Bmji,AcmK->ABcijK', H.bb.vooo[Vb, :, ob, ob], R.bb[Vb, vb, :, Ob], optimize=True)
    )
    dR.bbb.VVvooO += (1.0 / 4.0) * (
            +1.0 * np.einsum('cmji,ABmK->ABcijK', H.bb.vooo[vb, :, ob, ob], R.bb[Vb, Vb, :, Ob], optimize=True)
    )
    dR.bbb.VVvooO += (4.0 / 4.0) * (
            +1.0 * np.einsum('BmjK,Acmi->ABcijK', H.bb.vooo[Vb, :, ob, Ob], R.bb[Vb, vb, :, ob], optimize=True)
    )
    dR.bbb.VVvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('cmjK,ABmi->ABcijK', H.bb.vooo[vb, :, ob, Ob], R.bb[Vb, Vb, :, ob], optimize=True)
    )
    dR.bbb.VVvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('BAje,eciK->ABcijK', X.bb.vvov[Vb, Vb, ob, :], T.bb[:, vb, ob, Ob], optimize=True)
    )
    dR.bbb.VVvooO += (4.0 / 4.0) * (
            -1.0 * np.einsum('Bcje,eAiK->ABcijK', X.bb.vvov[Vb, vb, ob, :], T.bb[:, Vb, ob, Ob], optimize=True)
    )
    dR.bbb.VVvooO += (1.0 / 4.0) * (
            -1.0 * np.einsum('BAKe,ecij->ABcijK', X.bb.vvov[Vb, Vb, Ob, :], T.bb[:, vb, ob, ob], optimize=True)
    )
    dR.bbb.VVvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('BcKe,eAij->ABcijK', X.bb.vvov[Vb, vb, Ob, :], T.bb[:, Vb, ob, ob], optimize=True)
    )
    dR.bbb.VVvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('BAje,eciK->ABcijK', H.bb.vvov[Vb, Vb, ob, :], R.bb[:, vb, ob, Ob], optimize=True)
    )
    dR.bbb.VVvooO += (4.0 / 4.0) * (
            -1.0 * np.einsum('Bcje,eAiK->ABcijK', H.bb.vvov[Vb, vb, ob, :], R.bb[:, Vb, ob, Ob], optimize=True)
    )
    dR.bbb.VVvooO += (1.0 / 4.0) * (
            -1.0 * np.einsum('BAKe,ecij->ABcijK', H.bb.vvov[Vb, Vb, Ob, :], R.bb[:, vb, ob, ob], optimize=True)
    )
    dR.bbb.VVvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('BcKe,eAij->ABcijK', H.bb.vvov[Vb, vb, Ob, :], R.bb[:, Vb, ob, ob], optimize=True)
    )
    # of terms =  16
    dR.bbb.VVvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('Be,AceijK->ABcijK', X.b.vv[Vb, vb], T.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('BE,EAcijK->ABcijK', X.b.vv[Vb, Vb], T.bbb.VVvooO, optimize=True)
    )
    dR.bbb.VVvooO += (1.0 / 4.0) * (
            -1.0 * np.einsum('ce,BAeijK->ABcijK', X.b.vv[vb, vb], T.bbb.VVvooO, optimize=True)
            + 1.0 * np.einsum('cE,BEAijK->ABcijK', X.b.vv[vb, Vb], T.bbb.VVVooO, optimize=True)
    )
    dR.bbb.VVvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mj,BAcimK->ABcijK', X.b.oo[ob, ob], T.bbb.VVvooO, optimize=True)
            + 1.0 * np.einsum('Mj,BAciMK->ABcijK', X.b.oo[Ob, ob], T.bbb.VVvoOO, optimize=True)
    )
    dR.bbb.VVvooO += (1.0 / 4.0) * (
            +1.0 * np.einsum('MK,BAcijM->ABcijK', X.b.oo[Ob, Ob], T.bbb.VVvooO, optimize=True)
    )
    dR.bbb.VVvooO += (1.0 / 4.0) * (
            -0.5 * np.einsum('mnij,BAcmnK->ABcijK', X.bb.oooo[ob, ob, ob, ob], T.bbb.VVvooO, optimize=True)
            - 1.0 * np.einsum('mNij,BAcmNK->ABcijK', X.bb.oooo[ob, Ob, ob, ob], T.bbb.VVvoOO, optimize=True)
            - 0.5 * np.einsum('MNij,BAcMNK->ABcijK', X.bb.oooo[Ob, Ob, ob, ob], T.bbb.VVvOOO, optimize=True)
    )
    dR.bbb.VVvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('mNKj,BAcmiN->ABcijK', X.bb.oooo[ob, Ob, Ob, ob], T.bbb.VVvooO, optimize=True)
            + 0.5 * np.einsum('MNKj,BAciMN->ABcijK', X.bb.oooo[Ob, Ob, Ob, ob], T.bbb.VVvoOO, optimize=True)
    )
    dR.bbb.VVvooO += (1.0 / 4.0) * (
            +1.0 * np.einsum('ABeF,FceijK->ABcijK', X.bb.vvvv[Vb, Vb, vb, Vb], T.bbb.VvvooO, optimize=True)
            - 0.5 * np.einsum('ABEF,FEcijK->ABcijK', X.bb.vvvv[Vb, Vb, Vb, Vb], T.bbb.VVvooO, optimize=True)
    )
    dR.bbb.VVvooO += (2.0 / 4.0) * (
            +0.5 * np.einsum('cBef,AfeijK->ABcijK', X.bb.vvvv[vb, Vb, vb, vb], T.bbb.VvvooO, optimize=True)
            + 1.0 * np.einsum('cBeF,AFeijK->ABcijK', X.bb.vvvv[vb, Vb, vb, Vb], T.bbb.VVvooO, optimize=True)
            + 0.5 * np.einsum('cBEF,AFEijK->ABcijK', X.bb.vvvv[vb, Vb, Vb, Vb], T.bbb.VVVooO, optimize=True)
    )
    dR.bbb.VVvooO += (4.0 / 4.0) * (
            -1.0 * np.einsum('Bmje,AceimK->ABcijK', X.bb.voov[Vb, ob, ob, vb], T.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('BMje,AceiMK->ABcijK', X.bb.voov[Vb, Ob, ob, vb], T.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('BmjE,EAcimK->ABcijK', X.bb.voov[Vb, ob, ob, Vb], T.bbb.VVvooO, optimize=True)
            - 1.0 * np.einsum('BMjE,EAciMK->ABcijK', X.bb.voov[Vb, Ob, ob, Vb], T.bbb.VVvoOO, optimize=True)
    )
    dR.bbb.VVvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('cmje,BAeimK->ABcijK', X.bb.voov[vb, ob, ob, vb], T.bbb.VVvooO, optimize=True)
            - 1.0 * np.einsum('cMje,BAeiMK->ABcijK', X.bb.voov[vb, Ob, ob, vb], T.bbb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('cmjE,BEAimK->ABcijK', X.bb.voov[vb, ob, ob, Vb], T.bbb.VVVooO, optimize=True)
            + 1.0 * np.einsum('cMjE,BEAiMK->ABcijK', X.bb.voov[vb, Ob, ob, Vb], T.bbb.VVVoOO, optimize=True)
    )
    dR.bbb.VVvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('BMKe,AceijM->ABcijK', X.bb.voov[Vb, Ob, Ob, vb], T.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('BMKE,EAcijM->ABcijK', X.bb.voov[Vb, Ob, Ob, Vb], T.bbb.VVvooO, optimize=True)
    )
    dR.bbb.VVvooO += (1.0 / 4.0) * (
            -1.0 * np.einsum('cMKe,BAeijM->ABcijK', X.bb.voov[vb, Ob, Ob, vb], T.bbb.VVvooO, optimize=True)
            + 1.0 * np.einsum('cMKE,BEAijM->ABcijK', X.bb.voov[vb, Ob, Ob, Vb], T.bbb.VVVooO, optimize=True)
    )
    dR.bbb.VVvooO += (4.0 / 4.0) * (
            +1.0 * np.einsum('mBej,eAcmiK->ABcijK', X.ab.ovvo[oa, Vb, va, ob], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('MBej,eAcMiK->ABcijK', X.ab.ovvo[Oa, Vb, va, ob], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('mBEj,EAcmiK->ABcijK', X.ab.ovvo[oa, Vb, Va, ob], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('MBEj,EAcMiK->ABcijK', X.ab.ovvo[Oa, Vb, Va, ob], T.abb.VVvOoO, optimize=True)
    )
    dR.bbb.VVvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('mcej,eABmiK->ABcijK', X.ab.ovvo[oa, vb, va, ob], T.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('Mcej,eABMiK->ABcijK', X.ab.ovvo[Oa, vb, va, ob], T.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('mcEj,EABmiK->ABcijK', X.ab.ovvo[oa, vb, Va, ob], T.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('McEj,EABMiK->ABcijK', X.ab.ovvo[Oa, vb, Va, ob], T.abb.VVVOoO, optimize=True)
    )
    dR.bbb.VVvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('MBeK,eAcMji->ABcijK', X.ab.ovvo[Oa, Vb, va, Ob], T.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('MBEK,EAcMji->ABcijK', X.ab.ovvo[Oa, Vb, Va, Ob], T.abb.VVvOoo, optimize=True)
    )
    dR.bbb.VVvooO += (1.0 / 4.0) * (
            -1.0 * np.einsum('MceK,eABMji->ABcijK', X.ab.ovvo[Oa, vb, va, Ob], T.abb.vVVOoo, optimize=True)
            - 1.0 * np.einsum('McEK,EABMji->ABcijK', X.ab.ovvo[Oa, vb, Va, Ob], T.abb.VVVOoo, optimize=True)
    )
    dR.bbb.VVvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mj,BAcimK->ABcijK', H.b.oo[ob, ob], R.bbb.VVvooO, optimize=True)
            + 1.0 * np.einsum('Mj,BAciMK->ABcijK', H.b.oo[Ob, ob], R.bbb.VVvoOO, optimize=True)
    )
    dR.bbb.VVvooO += (1.0 / 4.0) * (
            +1.0 * np.einsum('MK,BAcijM->ABcijK', H.b.oo[Ob, Ob], R.bbb.VVvooO, optimize=True)
    )
    dR.bbb.VVvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('Be,AceijK->ABcijK', H.b.vv[Vb, vb], R.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('BE,EAcijK->ABcijK', H.b.vv[Vb, Vb], R.bbb.VVvooO, optimize=True)
    )
    dR.bbb.VVvooO += (1.0 / 4.0) * (
            -1.0 * np.einsum('ce,BAeijK->ABcijK', H.b.vv[vb, vb], R.bbb.VVvooO, optimize=True)
            + 1.0 * np.einsum('cE,BEAijK->ABcijK', H.b.vv[vb, Vb], R.bbb.VVVooO, optimize=True)
    )
    dR.bbb.VVvooO += (1.0 / 4.0) * (
            -0.5 * np.einsum('mnij,BAcmnK->ABcijK', H.bb.oooo[ob, ob, ob, ob], R.bbb.VVvooO, optimize=True)
            - 1.0 * np.einsum('mNij,BAcmNK->ABcijK', H.bb.oooo[ob, Ob, ob, ob], R.bbb.VVvoOO, optimize=True)
            - 0.5 * np.einsum('MNij,BAcMNK->ABcijK', H.bb.oooo[Ob, Ob, ob, ob], R.bbb.VVvOOO, optimize=True)
    )
    dR.bbb.VVvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('mNKj,BAcmiN->ABcijK', H.bb.oooo[ob, Ob, Ob, ob], R.bbb.VVvooO, optimize=True)
            + 0.5 * np.einsum('MNKj,BAciMN->ABcijK', H.bb.oooo[Ob, Ob, Ob, ob], R.bbb.VVvoOO, optimize=True)
    )
    dR.bbb.VVvooO += (1.0 / 4.0) * (
            +1.0 * np.einsum('ABeF,FceijK->ABcijK', H.bb.vvvv[Vb, Vb, vb, Vb], R.bbb.VvvooO, optimize=True)
            - 0.5 * np.einsum('ABEF,FEcijK->ABcijK', H.bb.vvvv[Vb, Vb, Vb, Vb], R.bbb.VVvooO, optimize=True)
    )
    dR.bbb.VVvooO += (2.0 / 4.0) * (
            +0.5 * np.einsum('cBef,AfeijK->ABcijK', H.bb.vvvv[vb, Vb, vb, vb], R.bbb.VvvooO, optimize=True)
            + 1.0 * np.einsum('cBeF,AFeijK->ABcijK', H.bb.vvvv[vb, Vb, vb, Vb], R.bbb.VVvooO, optimize=True)
            + 0.5 * np.einsum('cBEF,AFEijK->ABcijK', H.bb.vvvv[vb, Vb, Vb, Vb], R.bbb.VVVooO, optimize=True)
    )
    dR.bbb.VVvooO += (4.0 / 4.0) * (
            +1.0 * np.einsum('Amie,BcemjK->ABcijK', H.bb.voov[Vb, ob, ob, vb], R.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('AMie,BcejMK->ABcijK', H.bb.voov[Vb, Ob, ob, vb], R.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('AmiE,BEcmjK->ABcijK', H.bb.voov[Vb, ob, ob, Vb], R.bbb.VVvooO, optimize=True)
            + 1.0 * np.einsum('AMiE,BEcjMK->ABcijK', H.bb.voov[Vb, Ob, ob, Vb], R.bbb.VVvoOO, optimize=True)
    )
    dR.bbb.VVvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('cmie,ABemjK->ABcijK', H.bb.voov[vb, ob, ob, vb], R.bbb.VVvooO, optimize=True)
            - 1.0 * np.einsum('cMie,ABejMK->ABcijK', H.bb.voov[vb, Ob, ob, vb], R.bbb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('cmiE,ABEmjK->ABcijK', H.bb.voov[vb, ob, ob, Vb], R.bbb.VVVooO, optimize=True)
            - 1.0 * np.einsum('cMiE,ABEjMK->ABcijK', H.bb.voov[vb, Ob, ob, Vb], R.bbb.VVVoOO, optimize=True)
    )
    dR.bbb.VVvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('AMKe,BcejiM->ABcijK', H.bb.voov[Vb, Ob, Ob, vb], R.bbb.VvvooO, optimize=True)
            + 1.0 * np.einsum('AMKE,BEcjiM->ABcijK', H.bb.voov[Vb, Ob, Ob, Vb], R.bbb.VVvooO, optimize=True)
    )
    dR.bbb.VVvooO += (1.0 / 4.0) * (
            -1.0 * np.einsum('cMKe,ABejiM->ABcijK', H.bb.voov[vb, Ob, Ob, vb], R.bbb.VVvooO, optimize=True)
            - 1.0 * np.einsum('cMKE,ABEjiM->ABcijK', H.bb.voov[vb, Ob, Ob, Vb], R.bbb.VVVooO, optimize=True)
    )
    dR.bbb.VVvooO += (4.0 / 4.0) * (
            +1.0 * np.einsum('mAei,eBcmjK->ABcijK', H.ab.ovvo[oa, Vb, va, ob], R.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('MAei,eBcMjK->ABcijK', H.ab.ovvo[Oa, Vb, va, ob], R.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('mAEi,EBcmjK->ABcijK', H.ab.ovvo[oa, Vb, Va, ob], R.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('MAEi,EBcMjK->ABcijK', H.ab.ovvo[Oa, Vb, Va, ob], R.abb.VVvOoO, optimize=True)
    )
    dR.bbb.VVvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('mcei,eBAmjK->ABcijK', H.ab.ovvo[oa, vb, va, ob], R.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('Mcei,eBAMjK->ABcijK', H.ab.ovvo[Oa, vb, va, ob], R.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('mcEi,EBAmjK->ABcijK', H.ab.ovvo[oa, vb, Va, ob], R.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('McEi,EBAMjK->ABcijK', H.ab.ovvo[Oa, vb, Va, ob], R.abb.VVVOoO, optimize=True)
    )
    dR.bbb.VVvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('MAeK,eBcMij->ABcijK', H.ab.ovvo[Oa, Vb, va, Ob], R.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('MAEK,EBcMij->ABcijK', H.ab.ovvo[Oa, Vb, Va, Ob], R.abb.VVvOoo, optimize=True)
    )
    dR.bbb.VVvooO += (1.0 / 4.0) * (
            -1.0 * np.einsum('MceK,eBAMij->ABcijK', H.ab.ovvo[Oa, vb, va, Ob], R.abb.vVVOoo, optimize=True)
            - 1.0 * np.einsum('McEK,EBAMij->ABcijK', H.ab.ovvo[Oa, vb, Va, Ob], R.abb.VVVOoo, optimize=True)
    )
    # of terms =  32

    dR.bbb.VVvooO -= np.transpose(dR.bbb.VVvooO, (0, 1, 2, 4, 3, 5))
    dR.bbb.VVvooO -= np.transpose(dR.bbb.VVvooO, (1, 0, 2, 3, 4, 5))

    return dR

def update(R, omega, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    R.bbb.VVvooO = eomcc_active_loops.update_r3d_110001(
        R.bbb.VVvooO,
        omega,
        H.b.oo[Ob, Ob],
        H.b.vv[Vb, Vb],
        H.b.oo[ob, ob],
        H.b.vv[vb, vb],
        0.0,
    )
    return R
