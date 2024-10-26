import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops

def build(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.aaa.VVvooO = (2.0 / 4.0) * (
            -1.0 * np.einsum('Bmji,AcmK->ABcijK', X.aa.vooo[Va, :, oa, oa], T.aa[Va, va, :, Oa], optimize=True)
    )
    dR.aaa.VVvooO += (1.0 / 4.0) * (
            +1.0 * np.einsum('cmji,ABmK->ABcijK', X.aa.vooo[va, :, oa, oa], T.aa[Va, Va, :, Oa], optimize=True)
    )
    dR.aaa.VVvooO += (4.0 / 4.0) * (
            +1.0 * np.einsum('BmjK,Acmi->ABcijK', X.aa.vooo[Va, :, oa, Oa], T.aa[Va, va, :, oa], optimize=True)
    )
    dR.aaa.VVvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('cmjK,ABmi->ABcijK', X.aa.vooo[va, :, oa, Oa], T.aa[Va, Va, :, oa], optimize=True)
    )
    dR.aaa.VVvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('Bmji,AcmK->ABcijK', H.aa.vooo[Va, :, oa, oa], R.aa[Va, va, :, Oa], optimize=True)
    )
    dR.aaa.VVvooO += (1.0 / 4.0) * (
            +1.0 * np.einsum('cmji,ABmK->ABcijK', H.aa.vooo[va, :, oa, oa], R.aa[Va, Va, :, Oa], optimize=True)
    )
    dR.aaa.VVvooO += (4.0 / 4.0) * (
            +1.0 * np.einsum('BmjK,Acmi->ABcijK', H.aa.vooo[Va, :, oa, Oa], R.aa[Va, va, :, oa], optimize=True)
    )
    dR.aaa.VVvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('cmjK,ABmi->ABcijK', H.aa.vooo[va, :, oa, Oa], R.aa[Va, Va, :, oa], optimize=True)
    )
    dR.aaa.VVvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('BAje,eciK->ABcijK', X.aa.vvov[Va, Va, oa, :], T.aa[:, va, oa, Oa], optimize=True)
    )
    dR.aaa.VVvooO += (4.0 / 4.0) * (
            -1.0 * np.einsum('Bcje,eAiK->ABcijK', X.aa.vvov[Va, va, oa, :], T.aa[:, Va, oa, Oa], optimize=True)
    )
    dR.aaa.VVvooO += (1.0 / 4.0) * (
            -1.0 * np.einsum('BAKe,ecij->ABcijK', X.aa.vvov[Va, Va, Oa, :], T.aa[:, va, oa, oa], optimize=True)
    )
    dR.aaa.VVvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('BcKe,eAij->ABcijK', X.aa.vvov[Va, va, Oa, :], T.aa[:, Va, oa, oa], optimize=True)
    )
    dR.aaa.VVvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('BAje,eciK->ABcijK', H.aa.vvov[Va, Va, oa, :], R.aa[:, va, oa, Oa], optimize=True)
    )
    dR.aaa.VVvooO += (4.0 / 4.0) * (
            -1.0 * np.einsum('Bcje,eAiK->ABcijK', H.aa.vvov[Va, va, oa, :], R.aa[:, Va, oa, Oa], optimize=True)
    )
    dR.aaa.VVvooO += (1.0 / 4.0) * (
            -1.0 * np.einsum('BAKe,ecij->ABcijK', H.aa.vvov[Va, Va, Oa, :], R.aa[:, va, oa, oa], optimize=True)
    )
    dR.aaa.VVvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('BcKe,eAij->ABcijK', H.aa.vvov[Va, va, Oa, :], R.aa[:, Va, oa, oa], optimize=True)
    )
    dR.aaa.VVvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('Be,AceijK->ABcijK', X.a.vv[Va, va], T.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('BE,EAcijK->ABcijK', X.a.vv[Va, Va], T.aaa.VVvooO, optimize=True)
    )
    dR.aaa.VVvooO += (1.0 / 4.0) * (
            -1.0 * np.einsum('ce,BAeijK->ABcijK', X.a.vv[va, va], T.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('cE,BEAijK->ABcijK', X.a.vv[va, Va], T.aaa.VVVooO, optimize=True)
    )
    dR.aaa.VVvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mj,BAcimK->ABcijK', X.a.oo[oa, oa], T.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('Mj,BAciMK->ABcijK', X.a.oo[Oa, oa], T.aaa.VVvoOO, optimize=True)
    )
    dR.aaa.VVvooO += (1.0 / 4.0) * (
            +1.0 * np.einsum('MK,BAcijM->ABcijK', X.a.oo[Oa, Oa], T.aaa.VVvooO, optimize=True)
    )
    dR.aaa.VVvooO += (1.0 / 4.0) * (
            -0.5 * np.einsum('mnij,BAcmnK->ABcijK', X.aa.oooo[oa, oa, oa, oa], T.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('Mnij,BAcnMK->ABcijK', X.aa.oooo[Oa, oa, oa, oa], T.aaa.VVvoOO, optimize=True)
            - 0.5 * np.einsum('MNij,BAcMNK->ABcijK', X.aa.oooo[Oa, Oa, oa, oa], T.aaa.VVvOOO, optimize=True)
    )
    dR.aaa.VVvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('MnKj,BAcniM->ABcijK', X.aa.oooo[Oa, oa, Oa, oa], T.aaa.VVvooO, optimize=True)
            + 0.5 * np.einsum('MNKj,BAciMN->ABcijK', X.aa.oooo[Oa, Oa, Oa, oa], T.aaa.VVvoOO, optimize=True)
    )
    dR.aaa.VVvooO += (1.0 / 4.0) * (
            -1.0 * np.einsum('ABEf,EcfijK->ABcijK', X.aa.vvvv[Va, Va, Va, va], T.aaa.VvvooO, optimize=True)
            - 0.5 * np.einsum('ABEF,FEcijK->ABcijK', X.aa.vvvv[Va, Va, Va, Va], T.aaa.VVvooO, optimize=True)
    )
    dR.aaa.VVvooO += (2.0 / 4.0) * (
            +0.5 * np.einsum('cBef,AfeijK->ABcijK', X.aa.vvvv[va, Va, va, va], T.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('cBEf,AEfijK->ABcijK', X.aa.vvvv[va, Va, Va, va], T.aaa.VVvooO, optimize=True)
            + 0.5 * np.einsum('cBEF,AFEijK->ABcijK', X.aa.vvvv[va, Va, Va, Va], T.aaa.VVVooO, optimize=True)
    )
    dR.aaa.VVvooO += (4.0 / 4.0) * (
            -1.0 * np.einsum('Bmje,AceimK->ABcijK', X.aa.voov[Va, oa, oa, va], T.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('BmjE,EAcimK->ABcijK', X.aa.voov[Va, oa, oa, Va], T.aaa.VVvooO, optimize=True)
            - 1.0 * np.einsum('BMje,AceiMK->ABcijK', X.aa.voov[Va, Oa, oa, va], T.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('BMjE,EAciMK->ABcijK', X.aa.voov[Va, Oa, oa, Va], T.aaa.VVvoOO, optimize=True)
    )
    dR.aaa.VVvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('cmje,BAeimK->ABcijK', X.aa.voov[va, oa, oa, va], T.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('cmjE,BEAimK->ABcijK', X.aa.voov[va, oa, oa, Va], T.aaa.VVVooO, optimize=True)
            - 1.0 * np.einsum('cMje,BAeiMK->ABcijK', X.aa.voov[va, Oa, oa, va], T.aaa.VVvoOO, optimize=True)
            + 1.0 * np.einsum('cMjE,BEAiMK->ABcijK', X.aa.voov[va, Oa, oa, Va], T.aaa.VVVoOO, optimize=True)
    )
    dR.aaa.VVvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('BMKe,AceijM->ABcijK', X.aa.voov[Va, Oa, Oa, va], T.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('BMKE,EAcijM->ABcijK', X.aa.voov[Va, Oa, Oa, Va], T.aaa.VVvooO, optimize=True)
    )
    dR.aaa.VVvooO += (1.0 / 4.0) * (
            -1.0 * np.einsum('cMKe,BAeijM->ABcijK', X.aa.voov[va, Oa, Oa, va], T.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('cMKE,BEAijM->ABcijK', X.aa.voov[va, Oa, Oa, Va], T.aaa.VVVooO, optimize=True)
    )
    dR.aaa.VVvooO += (4.0 / 4.0) * (
            +1.0 * np.einsum('Bmje,AceiKm->ABcijK', X.ab.voov[Va, ob, oa, vb], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('BmjE,AcEiKm->ABcijK', X.ab.voov[Va, ob, oa, Vb], T.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('BMje,AceiKM->ABcijK', X.ab.voov[Va, Ob, oa, vb], T.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('BMjE,AcEiKM->ABcijK', X.ab.voov[Va, Ob, oa, Vb], T.aab.VvVoOO, optimize=True)
    )
    dR.aaa.VVvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('cmje,BAeiKm->ABcijK', X.ab.voov[va, ob, oa, vb], T.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('cmjE,BAEiKm->ABcijK', X.ab.voov[va, ob, oa, Vb], T.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('cMje,BAeiKM->ABcijK', X.ab.voov[va, Ob, oa, vb], T.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('cMjE,BAEiKM->ABcijK', X.ab.voov[va, Ob, oa, Vb], T.aab.VVVoOO, optimize=True)
    )
    dR.aaa.VVvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('BMKe,AceijM->ABcijK', X.ab.voov[Va, Ob, Oa, vb], T.aab.VvvooO, optimize=True)
            - 1.0 * np.einsum('BMKE,AcEijM->ABcijK', X.ab.voov[Va, Ob, Oa, Vb], T.aab.VvVooO, optimize=True)
    )
    dR.aaa.VVvooO += (1.0 / 4.0) * (
            -1.0 * np.einsum('cMKe,BAeijM->ABcijK', X.ab.voov[va, Ob, Oa, vb], T.aab.VVvooO, optimize=True)
            - 1.0 * np.einsum('cMKE,BAEijM->ABcijK', X.ab.voov[va, Ob, Oa, Vb], T.aab.VVVooO, optimize=True)
    )
    dR.aaa.VVvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mj,BAcimK->ABcijK', H.a.oo[oa, oa], R.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('Mj,BAciMK->ABcijK', H.a.oo[Oa, oa], R.aaa.VVvoOO, optimize=True)
    )
    dR.aaa.VVvooO += (1.0 / 4.0) * (
            +1.0 * np.einsum('MK,BAcijM->ABcijK', H.a.oo[Oa, Oa], R.aaa.VVvooO, optimize=True)
    )
    dR.aaa.VVvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('Be,AceijK->ABcijK', H.a.vv[Va, va], R.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('BE,EAcijK->ABcijK', H.a.vv[Va, Va], R.aaa.VVvooO, optimize=True)
    )
    dR.aaa.VVvooO += (1.0 / 4.0) * (
            -1.0 * np.einsum('ce,BAeijK->ABcijK', H.a.vv[va, va], R.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('cE,BEAijK->ABcijK', H.a.vv[va, Va], R.aaa.VVVooO, optimize=True)
    )
    dR.aaa.VVvooO += (1.0 / 4.0) * (
            -0.5 * np.einsum('mnij,BAcmnK->ABcijK', H.aa.oooo[oa, oa, oa, oa], R.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('Mnij,BAcnMK->ABcijK', H.aa.oooo[Oa, oa, oa, oa], R.aaa.VVvoOO, optimize=True)
            - 0.5 * np.einsum('MNij,BAcMNK->ABcijK', H.aa.oooo[Oa, Oa, oa, oa], R.aaa.VVvOOO, optimize=True)
    )
    dR.aaa.VVvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('MnKj,BAcniM->ABcijK', H.aa.oooo[Oa, oa, Oa, oa], R.aaa.VVvooO, optimize=True)
            + 0.5 * np.einsum('MNKj,BAciMN->ABcijK', H.aa.oooo[Oa, Oa, Oa, oa], R.aaa.VVvoOO, optimize=True)
    )
    dR.aaa.VVvooO += (1.0 / 4.0) * (
            -1.0 * np.einsum('ABEf,EcfijK->ABcijK', H.aa.vvvv[Va, Va, Va, va], R.aaa.VvvooO, optimize=True)
            - 0.5 * np.einsum('ABEF,FEcijK->ABcijK', H.aa.vvvv[Va, Va, Va, Va], R.aaa.VVvooO, optimize=True)
    )
    dR.aaa.VVvooO += (2.0 / 4.0) * (
            +0.5 * np.einsum('cBef,AfeijK->ABcijK', H.aa.vvvv[va, Va, va, va], R.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('cBEf,AEfijK->ABcijK', H.aa.vvvv[va, Va, Va, va], R.aaa.VVvooO, optimize=True)
            + 0.5 * np.einsum('cBEF,AFEijK->ABcijK', H.aa.vvvv[va, Va, Va, Va], R.aaa.VVVooO, optimize=True)
    )
    dR.aaa.VVvooO += (4.0 / 4.0) * (
            +1.0 * np.einsum('Amie,BcemjK->ABcijK', H.aa.voov[Va, oa, oa, va], R.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('AmiE,BEcmjK->ABcijK', H.aa.voov[Va, oa, oa, Va], R.aaa.VVvooO, optimize=True)
            - 1.0 * np.einsum('AMie,BcejMK->ABcijK', H.aa.voov[Va, Oa, oa, va], R.aaa.VvvoOO, optimize=True)
            + 1.0 * np.einsum('AMiE,BEcjMK->ABcijK', H.aa.voov[Va, Oa, oa, Va], R.aaa.VVvoOO, optimize=True)
    )
    dR.aaa.VVvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('cmie,ABemjK->ABcijK', H.aa.voov[va, oa, oa, va], R.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('cmiE,ABEmjK->ABcijK', H.aa.voov[va, oa, oa, Va], R.aaa.VVVooO, optimize=True)
            - 1.0 * np.einsum('cMie,ABejMK->ABcijK', H.aa.voov[va, Oa, oa, va], R.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('cMiE,ABEjMK->ABcijK', H.aa.voov[va, Oa, oa, Va], R.aaa.VVVoOO, optimize=True)
    )
    dR.aaa.VVvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('AMKe,BcejiM->ABcijK', H.aa.voov[Va, Oa, Oa, va], R.aaa.VvvooO, optimize=True)
            + 1.0 * np.einsum('AMKE,BEcjiM->ABcijK', H.aa.voov[Va, Oa, Oa, Va], R.aaa.VVvooO, optimize=True)
    )
    dR.aaa.VVvooO += (1.0 / 4.0) * (
            -1.0 * np.einsum('cMKe,ABejiM->ABcijK', H.aa.voov[va, Oa, Oa, va], R.aaa.VVvooO, optimize=True)
            - 1.0 * np.einsum('cMKE,ABEjiM->ABcijK', H.aa.voov[va, Oa, Oa, Va], R.aaa.VVVooO, optimize=True)
    )
    dR.aaa.VVvooO += (4.0 / 4.0) * (
            +1.0 * np.einsum('Amie,BcejKm->ABcijK', H.ab.voov[Va, ob, oa, vb], R.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('AmiE,BcEjKm->ABcijK', H.ab.voov[Va, ob, oa, Vb], R.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('AMie,BcejKM->ABcijK', H.ab.voov[Va, Ob, oa, vb], R.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('AMiE,BcEjKM->ABcijK', H.ab.voov[Va, Ob, oa, Vb], R.aab.VvVoOO, optimize=True)
    )
    dR.aaa.VVvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('cmie,ABejKm->ABcijK', H.ab.voov[va, ob, oa, vb], R.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('cmiE,ABEjKm->ABcijK', H.ab.voov[va, ob, oa, Vb], R.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('cMie,ABejKM->ABcijK', H.ab.voov[va, Ob, oa, vb], R.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('cMiE,ABEjKM->ABcijK', H.ab.voov[va, Ob, oa, Vb], R.aab.VVVoOO, optimize=True)
    )
    dR.aaa.VVvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('AMKe,BcejiM->ABcijK', H.ab.voov[Va, Ob, Oa, vb], R.aab.VvvooO, optimize=True)
            - 1.0 * np.einsum('AMKE,BcEjiM->ABcijK', H.ab.voov[Va, Ob, Oa, Vb], R.aab.VvVooO, optimize=True)
    )
    dR.aaa.VVvooO += (1.0 / 4.0) * (
            -1.0 * np.einsum('cMKe,ABejiM->ABcijK', H.ab.voov[va, Ob, Oa, vb], R.aab.VVvooO, optimize=True)
            - 1.0 * np.einsum('cMKE,ABEjiM->ABcijK', H.ab.voov[va, Ob, Oa, Vb], R.aab.VVVooO, optimize=True)
    )

    dR.aaa.VVvooO -= np.transpose(dR.aaa.VVvooO, (1, 0, 2, 3, 4, 5))

    dR.aaa.VVvooO -= np.transpose(dR.aaa.VVvooO, (0, 1, 2, 4, 3, 5))

    return dR

def update(R, omega, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    R.aaa.VVvooO = eomcc_active_loops.update_r3a_110001(
        R.aaa.VVvooO,
        omega,
        H.a.oo[Oa, Oa],
        H.a.vv[Va, Va],
        H.a.oo[oa, oa],
        H.a.vv[va, va],
        0.0,
    )

    return R