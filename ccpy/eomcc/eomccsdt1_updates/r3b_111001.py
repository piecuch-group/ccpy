import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops

def build(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.aab.VVVooO = (2.0 / 4.0) * (
            +1.0 * np.einsum('BCeK,Aeij->ABCijK', X.ab.vvvo[Va, Vb, :, Ob], T.aa[Va, :, oa, oa], optimize=True)
    )
    dR.aab.VVVooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('mCjK,ABim->ABCijK', X.ab.ovoo[:, Vb, oa, Ob], T.aa[Va, Va, oa, :], optimize=True)
    )
    dR.aab.VVVooO += (4.0 / 4.0) * (
            +1.0 * np.einsum('ACie,BejK->ABCijK', X.ab.vvov[Va, Vb, oa, :], T.ab[Va, :, oa, Ob], optimize=True)
    )
    dR.aab.VVVooO += (4.0 / 4.0) * (
            -1.0 * np.einsum('AmiK,BCjm->ABCijK', X.ab.vooo[Va, :, oa, Ob], T.ab[Va, Vb, oa, :], optimize=True)
    )
    dR.aab.VVVooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('ABie,eCjK->ABCijK', X.aa.vvov[Va, Va, oa, :], T.ab[:, Vb, oa, Ob], optimize=True)
    )
    dR.aab.VVVooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('Amij,BCmK->ABCijK', X.aa.vooo[Va, :, oa, oa], T.ab[Va, Vb, :, Ob], optimize=True)
    )
    dR.aab.VVVooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('BCeK,Aeij->ABCijK', H.ab.vvvo[Va, Vb, :, Ob], R.aa[Va, :, oa, oa], optimize=True)
    )
    dR.aab.VVVooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('mCjK,ABim->ABCijK', H.ab.ovoo[:, Vb, oa, Ob], R.aa[Va, Va, oa, :], optimize=True)
    )
    dR.aab.VVVooO += (4.0 / 4.0) * (
            +1.0 * np.einsum('ACie,BejK->ABCijK', H.ab.vvov[Va, Vb, oa, :], R.ab[Va, :, oa, Ob], optimize=True)
    )
    dR.aab.VVVooO += (4.0 / 4.0) * (
            -1.0 * np.einsum('AmiK,BCjm->ABCijK', H.ab.vooo[Va, :, oa, Ob], R.ab[Va, Vb, oa, :], optimize=True)
    )
    dR.aab.VVVooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('ABie,eCjK->ABCijK', H.aa.vvov[Va, Va, oa, :], R.ab[:, Vb, oa, Ob], optimize=True)
    )
    dR.aab.VVVooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('Amij,BCmK->ABCijK', H.aa.vooo[Va, :, oa, oa], R.ab[Va, Vb, :, Ob], optimize=True)
    )
    dR.aab.VVVooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mi,BACmjK->ABCijK', X.a.oo[oa, oa], T.aab.VVVooO, optimize=True)
            - 1.0 * np.einsum('Mi,BACjMK->ABCijK', X.a.oo[Oa, oa], T.aab.VVVoOO, optimize=True)
    )
    dR.aab.VVVooO += (1.0 / 4.0) * (
            +1.0 * np.einsum('MK,BACijM->ABCijK', X.b.oo[Ob, Ob], T.aab.VVVooO, optimize=True)
    )
    dR.aab.VVVooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('Ae,BeCijK->ABCijK', X.a.vv[Va, va], T.aab.VvVooO, optimize=True)
            - 1.0 * np.einsum('AE,BECijK->ABCijK', X.a.vv[Va, Va], T.aab.VVVooO, optimize=True)
    )
    dR.aab.VVVooO += (1.0 / 4.0) * (
            -1.0 * np.einsum('Ce,BAeijK->ABCijK', X.b.vv[Vb, vb], T.aab.VVvooO, optimize=True)
            - 1.0 * np.einsum('CE,BAEijK->ABCijK', X.b.vv[Vb, Vb], T.aab.VVVooO, optimize=True)
    )
    dR.aab.VVVooO += (1.0 / 4.0) * (
            -0.5 * np.einsum('mnij,BACmnK->ABCijK', X.aa.oooo[oa, oa, oa, oa], T.aab.VVVooO, optimize=True)
            + 1.0 * np.einsum('Mnij,BACnMK->ABCijK', X.aa.oooo[Oa, oa, oa, oa], T.aab.VVVoOO, optimize=True)
            - 0.5 * np.einsum('MNij,BACMNK->ABCijK', X.aa.oooo[Oa, Oa, oa, oa], T.aab.VVVOOO, optimize=True)
    )
    dR.aab.VVVooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('mNjK,BACimN->ABCijK', X.ab.oooo[oa, Ob, oa, Ob], T.aab.VVVooO, optimize=True)
            - 1.0 * np.einsum('MnjK,BACiMn->ABCijK', X.ab.oooo[Oa, ob, oa, Ob], T.aab.VVVoOo, optimize=True)
            - 1.0 * np.einsum('MNjK,BACiMN->ABCijK', X.ab.oooo[Oa, Ob, oa, Ob], T.aab.VVVoOO, optimize=True)
    )
    dR.aab.VVVooO += (1.0 / 4.0) * (
            -0.5 * np.einsum('ABef,feCijK->ABCijK', X.aa.vvvv[Va, Va, va, va], T.aab.vvVooO, optimize=True)
            - 1.0 * np.einsum('ABeF,FeCijK->ABCijK', X.aa.vvvv[Va, Va, va, Va], T.aab.VvVooO, optimize=True)
            - 0.5 * np.einsum('ABEF,FECijK->ABCijK', X.aa.vvvv[Va, Va, Va, Va], T.aab.VVVooO, optimize=True)
    )
    dR.aab.VVVooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('BCef,AefijK->ABCijK', X.ab.vvvv[Va, Vb, va, vb], T.aab.VvvooO, optimize=True)
            - 1.0 * np.einsum('BCEf,EAfijK->ABCijK', X.ab.vvvv[Va, Vb, Va, vb], T.aab.VVvooO, optimize=True)
            + 1.0 * np.einsum('BCeF,AeFijK->ABCijK', X.ab.vvvv[Va, Vb, va, Vb], T.aab.VvVooO, optimize=True)
            - 1.0 * np.einsum('BCEF,EAFijK->ABCijK', X.ab.vvvv[Va, Vb, Va, Vb], T.aab.VVVooO, optimize=True)
    )
    dR.aab.VVVooO += (4.0 / 4.0) * (
            -1.0 * np.einsum('Amie,BeCmjK->ABCijK', X.aa.voov[Va, oa, oa, va], T.aab.VvVooO, optimize=True)
            + 1.0 * np.einsum('AMie,BeCjMK->ABCijK', X.aa.voov[Va, Oa, oa, va], T.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('AmiE,BECmjK->ABCijK', X.aa.voov[Va, oa, oa, Va], T.aab.VVVooO, optimize=True)
            + 1.0 * np.einsum('AMiE,BECjMK->ABCijK', X.aa.voov[Va, Oa, oa, Va], T.aab.VVVoOO, optimize=True)
    )
    dR.aab.VVVooO += (4.0 / 4.0) * (
            -1.0 * np.einsum('Amie,BCejmK->ABCijK', X.ab.voov[Va, ob, oa, vb], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('AMie,BCejMK->ABCijK', X.ab.voov[Va, Ob, oa, vb], T.abb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('AmiE,BCEjmK->ABCijK', X.ab.voov[Va, ob, oa, Vb], T.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('AMiE,BCEjMK->ABCijK', X.ab.voov[Va, Ob, oa, Vb], T.abb.VVVoOO, optimize=True)
    )
    dR.aab.VVVooO += (1.0 / 4.0) * (
            -1.0 * np.einsum('MCeK,BAeijM->ABCijK', X.ab.ovvo[Oa, Vb, va, Ob], T.aaa.VVvooO, optimize=True)
            - 1.0 * np.einsum('MCEK,EBAijM->ABCijK', X.ab.ovvo[Oa, Vb, Va, Ob], T.aaa.VVVooO, optimize=True)
    )
    dR.aab.VVVooO += (1.0 / 4.0) * (
            -1.0 * np.einsum('CMKe,BAeijM->ABCijK', X.bb.voov[Vb, Ob, Ob, vb], T.aab.VVvooO, optimize=True)
            - 1.0 * np.einsum('CMKE,BAEijM->ABCijK', X.bb.voov[Vb, Ob, Ob, Vb], T.aab.VVVooO, optimize=True)
    )
    dR.aab.VVVooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('AMeK,BeCijM->ABCijK', X.ab.vovo[Va, Ob, va, Ob], T.aab.VvVooO, optimize=True)
            + 1.0 * np.einsum('AMEK,BECijM->ABCijK', X.ab.vovo[Va, Ob, Va, Ob], T.aab.VVVooO, optimize=True)
    )
    dR.aab.VVVooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mCie,BAemjK->ABCijK', X.ab.ovov[oa, Vb, oa, vb], T.aab.VVvooO, optimize=True)
            - 1.0 * np.einsum('MCie,BAejMK->ABCijK', X.ab.ovov[Oa, Vb, oa, vb], T.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('mCiE,BAEmjK->ABCijK', X.ab.ovov[oa, Vb, oa, Vb], T.aab.VVVooO, optimize=True)
            - 1.0 * np.einsum('MCiE,BAEjMK->ABCijK', X.ab.ovov[Oa, Vb, oa, Vb], T.aab.VVVoOO, optimize=True)
    )
    dR.aab.VVVooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mi,BACmjK->ABCijK', H.a.oo[oa, oa], R.aab.VVVooO, optimize=True)
            - 1.0 * np.einsum('Mi,BACjMK->ABCijK', H.a.oo[Oa, oa], R.aab.VVVoOO, optimize=True)
    )
    dR.aab.VVVooO += (1.0 / 4.0) * (
            +1.0 * np.einsum('MK,BACijM->ABCijK', H.b.oo[Ob, Ob], R.aab.VVVooO, optimize=True)
    )
    dR.aab.VVVooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('Ae,BeCijK->ABCijK', H.a.vv[Va, va], R.aab.VvVooO, optimize=True)
            - 1.0 * np.einsum('AE,BECijK->ABCijK', H.a.vv[Va, Va], R.aab.VVVooO, optimize=True)
    )
    dR.aab.VVVooO += (1.0 / 4.0) * (
            -1.0 * np.einsum('Ce,BAeijK->ABCijK', H.b.vv[Vb, vb], R.aab.VVvooO, optimize=True)
            - 1.0 * np.einsum('CE,BAEijK->ABCijK', H.b.vv[Vb, Vb], R.aab.VVVooO, optimize=True)
    )
    dR.aab.VVVooO += (1.0 / 4.0) * (
            -0.5 * np.einsum('mnij,BACmnK->ABCijK', H.aa.oooo[oa, oa, oa, oa], R.aab.VVVooO, optimize=True)
            + 1.0 * np.einsum('Mnij,BACnMK->ABCijK', H.aa.oooo[Oa, oa, oa, oa], R.aab.VVVoOO, optimize=True)
            - 0.5 * np.einsum('MNij,BACMNK->ABCijK', H.aa.oooo[Oa, Oa, oa, oa], R.aab.VVVOOO, optimize=True)
    )
    dR.aab.VVVooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('mNjK,BACimN->ABCijK', H.ab.oooo[oa, Ob, oa, Ob], R.aab.VVVooO, optimize=True)
            - 1.0 * np.einsum('MnjK,BACiMn->ABCijK', H.ab.oooo[Oa, ob, oa, Ob], R.aab.VVVoOo, optimize=True)
            - 1.0 * np.einsum('MNjK,BACiMN->ABCijK', H.ab.oooo[Oa, Ob, oa, Ob], R.aab.VVVoOO, optimize=True)
    )
    dR.aab.VVVooO += (1.0 / 4.0) * (
            -0.5 * np.einsum('ABef,feCijK->ABCijK', H.aa.vvvv[Va, Va, va, va], R.aab.vvVooO, optimize=True)
            - 1.0 * np.einsum('ABeF,FeCijK->ABCijK', H.aa.vvvv[Va, Va, va, Va], R.aab.VvVooO, optimize=True)
            - 0.5 * np.einsum('ABEF,FECijK->ABCijK', H.aa.vvvv[Va, Va, Va, Va], R.aab.VVVooO, optimize=True)
    )
    dR.aab.VVVooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('BCef,AefijK->ABCijK', H.ab.vvvv[Va, Vb, va, vb], R.aab.VvvooO, optimize=True)
            - 1.0 * np.einsum('BCEf,EAfijK->ABCijK', H.ab.vvvv[Va, Vb, Va, vb], R.aab.VVvooO, optimize=True)
            + 1.0 * np.einsum('BCeF,AeFijK->ABCijK', H.ab.vvvv[Va, Vb, va, Vb], R.aab.VvVooO, optimize=True)
            - 1.0 * np.einsum('BCEF,EAFijK->ABCijK', H.ab.vvvv[Va, Vb, Va, Vb], R.aab.VVVooO, optimize=True)
    )
    dR.aab.VVVooO += (4.0 / 4.0) * (
            -1.0 * np.einsum('Amie,BeCmjK->ABCijK', H.aa.voov[Va, oa, oa, va], R.aab.VvVooO, optimize=True)
            + 1.0 * np.einsum('AMie,BeCjMK->ABCijK', H.aa.voov[Va, Oa, oa, va], R.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('AmiE,BECmjK->ABCijK', H.aa.voov[Va, oa, oa, Va], R.aab.VVVooO, optimize=True)
            + 1.0 * np.einsum('AMiE,BECjMK->ABCijK', H.aa.voov[Va, Oa, oa, Va], R.aab.VVVoOO, optimize=True)
    )
    dR.aab.VVVooO += (4.0 / 4.0) * (
            -1.0 * np.einsum('Amie,BCejmK->ABCijK', H.ab.voov[Va, ob, oa, vb], R.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('AMie,BCejMK->ABCijK', H.ab.voov[Va, Ob, oa, vb], R.abb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('AmiE,BCEjmK->ABCijK', H.ab.voov[Va, ob, oa, Vb], R.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('AMiE,BCEjMK->ABCijK', H.ab.voov[Va, Ob, oa, Vb], R.abb.VVVoOO, optimize=True)
    )
    dR.aab.VVVooO += (1.0 / 4.0) * (
            -1.0 * np.einsum('MCeK,BAeijM->ABCijK', H.ab.ovvo[Oa, Vb, va, Ob], R.aaa.VVvooO, optimize=True)
            - 1.0 * np.einsum('MCEK,EBAijM->ABCijK', H.ab.ovvo[Oa, Vb, Va, Ob], R.aaa.VVVooO, optimize=True)
    )
    dR.aab.VVVooO += (1.0 / 4.0) * (
            -1.0 * np.einsum('CMKe,BAeijM->ABCijK', H.bb.voov[Vb, Ob, Ob, vb], R.aab.VVvooO, optimize=True)
            - 1.0 * np.einsum('CMKE,BAEijM->ABCijK', H.bb.voov[Vb, Ob, Ob, Vb], R.aab.VVVooO, optimize=True)
    )
    dR.aab.VVVooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('AMeK,BeCijM->ABCijK', H.ab.vovo[Va, Ob, va, Ob], R.aab.VvVooO, optimize=True)
            + 1.0 * np.einsum('AMEK,BECijM->ABCijK', H.ab.vovo[Va, Ob, Va, Ob], R.aab.VVVooO, optimize=True)
    )
    dR.aab.VVVooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mCie,BAemjK->ABCijK', H.ab.ovov[oa, Vb, oa, vb], R.aab.VVvooO, optimize=True)
            - 1.0 * np.einsum('MCie,BAejMK->ABCijK', H.ab.ovov[Oa, Vb, oa, vb], R.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('mCiE,BAEmjK->ABCijK', H.ab.ovov[oa, Vb, oa, Vb], R.aab.VVVooO, optimize=True)
            - 1.0 * np.einsum('MCiE,BAEjMK->ABCijK', H.ab.ovov[Oa, Vb, oa, Vb], R.aab.VVVoOO, optimize=True)
    )

    dR.aab.VVVooO -= np.transpose(dR.aab.VVVooO, (1, 0, 2, 3, 4, 5))
    dR.aab.VVVooO -= np.transpose(dR.aab.VVVooO, (0, 1, 2, 4, 3, 5))

    return dR

def update(R, omega, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    R.aab.VVVooO = eomcc_active_loops.update_r3b_111001(
        R.aab.VVVooO,
        omega,
        H.a.oo[Oa, Oa],
        H.a.vv[Va, Va],
        H.a.oo[oa, oa],
        H.a.vv[va, va],
        H.b.oo[Ob, Ob],
        H.b.vv[Vb, Vb],
        H.b.oo[ob, ob],
        H.b.vv[vb, vb],
        0.0,
    )
    return R
