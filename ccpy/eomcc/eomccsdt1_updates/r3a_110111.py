import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops


def build(dR, R, T, H, X, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.aaa.VVvOOO = (6.0 / 12.0) * (
            -1.0 * np.einsum('BmJI,AcmK->ABcIJK', X.aa.vooo[Va, :, Oa, Oa], T.aa[Va, va, :, Oa], optimize=True)
    )
    dR.aaa.VVvOOO += (3.0 / 12.0) * (
            +1.0 * np.einsum('cmJI,ABmK->ABcIJK', X.aa.vooo[va, :, Oa, Oa], T.aa[Va, Va, :, Oa], optimize=True)
    )
    dR.aaa.VVvOOO += (6.0 / 12.0) * (
            -1.0 * np.einsum('BmJI,AcmK->ABcIJK', H.aa.vooo[Va, :, Oa, Oa], R.aa[Va, va, :, Oa], optimize=True)
    )
    dR.aaa.VVvOOO += (3.0 / 12.0) * (
            +1.0 * np.einsum('cmJI,ABmK->ABcIJK', H.aa.vooo[va, :, Oa, Oa], R.aa[Va, Va, :, Oa], optimize=True)
    )
    dR.aaa.VVvOOO += (3.0 / 12.0) * (
            +1.0 * np.einsum('BAJe,ecIK->ABcIJK', X.aa.vvov[Va, Va, Oa, :], T.aa[:, va, Oa, Oa], optimize=True)
    )
    dR.aaa.VVvOOO += (6.0 / 12.0) * (
            -1.0 * np.einsum('BcJe,eAIK->ABcIJK', X.aa.vvov[Va, va, Oa, :], T.aa[:, Va, Oa, Oa], optimize=True)
    )
    dR.aaa.VVvOOO += (3.0 / 12.0) * (
            +1.0 * np.einsum('BAJe,ecIK->ABcIJK', H.aa.vvov[Va, Va, Oa, :], R.aa[:, va, Oa, Oa], optimize=True)
    )
    dR.aaa.VVvOOO += (6.0 / 12.0) * (
            -1.0 * np.einsum('BcJe,eAIK->ABcIJK', H.aa.vvov[Va, va, Oa, :], R.aa[:, Va, Oa, Oa], optimize=True)
    )

    dR.aaa.VVvOOO += (2.0 / 12.0) * (
            -1.0 * np.einsum('Be,AceIJK->ABcIJK', X.a.vv[Va, va], T.aaa.VvvOOO, optimize=True)
            - 1.0 * np.einsum('BE,EAcIJK->ABcIJK', X.a.vv[Va, Va], T.aaa.VVvOOO, optimize=True)
    )
    dR.aaa.VVvOOO += (1.0 / 12.0) * (
            -1.0 * np.einsum('ce,BAeIJK->ABcIJK', X.a.vv[va, va], T.aaa.VVvOOO, optimize=True)
            + 1.0 * np.einsum('cE,BEAIJK->ABcIJK', X.a.vv[va, Va], T.aaa.VVVOOO, optimize=True)
    )
    dR.aaa.VVvOOO += (3.0 / 12.0) * (
            -1.0 * np.einsum('mJ,BAcmIK->ABcIJK', X.a.oo[oa, Oa], T.aaa.VVvoOO, optimize=True)
            + 1.0 * np.einsum('MJ,BAcIMK->ABcIJK', X.a.oo[Oa, Oa], T.aaa.VVvOOO, optimize=True)
    )
    dR.aaa.VVvOOO += (3.0 / 12.0) * (
            -0.5 * np.einsum('mnIJ,BAcmnK->ABcIJK', X.aa.oooo[oa, oa, Oa, Oa], T.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('MnIJ,BAcnMK->ABcIJK', X.aa.oooo[Oa, oa, Oa, Oa], T.aaa.VVvoOO, optimize=True)
            - 0.5 * np.einsum('MNIJ,BAcMNK->ABcIJK', X.aa.oooo[Oa, Oa, Oa, Oa], T.aaa.VVvOOO, optimize=True)
    )
    dR.aaa.VVvOOO += (1.0 / 12.0) * (
            -1.0 * np.einsum('ABEf,EcfIJK->ABcIJK', X.aa.vvvv[Va, Va, Va, va], T.aaa.VvvOOO, optimize=True)
            - 0.5 * np.einsum('ABEF,FEcIJK->ABcIJK', X.aa.vvvv[Va, Va, Va, Va], T.aaa.VVvOOO, optimize=True)
    )
    dR.aaa.VVvOOO += (2.0 / 12.0) * (
            +0.5 * np.einsum('cBef,AfeIJK->ABcIJK', X.aa.vvvv[va, Va, va, va], T.aaa.VvvOOO, optimize=True)
            - 1.0 * np.einsum('cBEf,AEfIJK->ABcIJK', X.aa.vvvv[va, Va, Va, va], T.aaa.VVvOOO, optimize=True)
            + 0.5 * np.einsum('cBEF,AFEIJK->ABcIJK', X.aa.vvvv[va, Va, Va, Va], T.aaa.VVVOOO, optimize=True)
    )
    dR.aaa.VVvOOO += (6.0 / 12.0) * (
            +1.0 * np.einsum('BmJe,AcemIK->ABcIJK', X.aa.voov[Va, oa, Oa, va], T.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('BMJe,AceIMK->ABcIJK', X.aa.voov[Va, Oa, Oa, va], T.aaa.VvvOOO, optimize=True)
            + 1.0 * np.einsum('BmJE,EAcmIK->ABcIJK', X.aa.voov[Va, oa, Oa, Va], T.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('BMJE,EAcIMK->ABcIJK', X.aa.voov[Va, Oa, Oa, Va], T.aaa.VVvOOO, optimize=True)
    )
    dR.aaa.VVvOOO += (3.0 / 12.0) * (
            +1.0 * np.einsum('cmJe,BAemIK->ABcIJK', X.aa.voov[va, oa, Oa, va], T.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('cMJe,BAeIMK->ABcIJK', X.aa.voov[va, Oa, Oa, va], T.aaa.VVvOOO, optimize=True)
            - 1.0 * np.einsum('cmJE,BEAmIK->ABcIJK', X.aa.voov[va, oa, Oa, Va], T.aaa.VVVoOO, optimize=True)
            + 1.0 * np.einsum('cMJE,BEAIMK->ABcIJK', X.aa.voov[va, Oa, Oa, Va], T.aaa.VVVOOO, optimize=True)
    )
    dR.aaa.VVvOOO += (6.0 / 12.0) * (
            +1.0 * np.einsum('BmJe,AceIKm->ABcIJK', X.ab.voov[Va, ob, Oa, vb], T.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('BMJe,AceIKM->ABcIJK', X.ab.voov[Va, Ob, Oa, vb], T.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('BmJE,AcEIKm->ABcIJK', X.ab.voov[Va, ob, Oa, Vb], T.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('BMJE,AcEIKM->ABcIJK', X.ab.voov[Va, Ob, Oa, Vb], T.aab.VvVOOO, optimize=True)
    )
    dR.aaa.VVvOOO += (3.0 / 12.0) * (
            +1.0 * np.einsum('cmJe,BAeIKm->ABcIJK', X.ab.voov[va, ob, Oa, vb], T.aab.VVvOOo, optimize=True)
            + 1.0 * np.einsum('cMJe,BAeIKM->ABcIJK', X.ab.voov[va, Ob, Oa, vb], T.aab.VVvOOO, optimize=True)
            + 1.0 * np.einsum('cmJE,BAEIKm->ABcIJK', X.ab.voov[va, ob, Oa, Vb], T.aab.VVVOOo, optimize=True)
            + 1.0 * np.einsum('cMJE,BAEIKM->ABcIJK', X.ab.voov[va, Ob, Oa, Vb], T.aab.VVVOOO, optimize=True)
    )
    dR.aaa.VVvOOO += (3.0 / 12.0) * (
            -1.0 * np.einsum('mJ,BAcmIK->ABcIJK', H.a.oo[oa, Oa], R.aaa.VVvoOO, optimize=True)
            + 1.0 * np.einsum('MJ,BAcIMK->ABcIJK', H.a.oo[Oa, Oa], R.aaa.VVvOOO, optimize=True)
    )
    dR.aaa.VVvOOO += (2.0 / 12.0) * (
            -1.0 * np.einsum('Be,AceIJK->ABcIJK', H.a.vv[Va, va], R.aaa.VvvOOO, optimize=True)
            - 1.0 * np.einsum('BE,EAcIJK->ABcIJK', H.a.vv[Va, Va], R.aaa.VVvOOO, optimize=True)
    )
    dR.aaa.VVvOOO += (1.0 / 12.0) * (
            -1.0 * np.einsum('ce,BAeIJK->ABcIJK', H.a.vv[va, va], R.aaa.VVvOOO, optimize=True)
            + 1.0 * np.einsum('cE,BEAIJK->ABcIJK', H.a.vv[va, Va], R.aaa.VVVOOO, optimize=True)
    )
    dR.aaa.VVvOOO += (3.0 / 12.0) * (
            -0.5 * np.einsum('mnIJ,BAcmnK->ABcIJK', H.aa.oooo[oa, oa, Oa, Oa], R.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('MnIJ,BAcnMK->ABcIJK', H.aa.oooo[Oa, oa, Oa, Oa], R.aaa.VVvoOO, optimize=True)
            - 0.5 * np.einsum('MNIJ,BAcMNK->ABcIJK', H.aa.oooo[Oa, Oa, Oa, Oa], R.aaa.VVvOOO, optimize=True)
    )
    dR.aaa.VVvOOO += (1.0 / 12.0) * (
            -1.0 * np.einsum('ABEf,EcfIJK->ABcIJK', H.aa.vvvv[Va, Va, Va, va], R.aaa.VvvOOO, optimize=True)
            - 0.5 * np.einsum('ABEF,FEcIJK->ABcIJK', H.aa.vvvv[Va, Va, Va, Va], R.aaa.VVvOOO, optimize=True)
    )
    dR.aaa.VVvOOO += (2.0 / 12.0) * (
            +0.5 * np.einsum('cBef,AfeIJK->ABcIJK', H.aa.vvvv[va, Va, va, va], R.aaa.VvvOOO, optimize=True)
            - 1.0 * np.einsum('cBEf,AEfIJK->ABcIJK', H.aa.vvvv[va, Va, Va, va], R.aaa.VVvOOO, optimize=True)
            + 0.5 * np.einsum('cBEF,AFEIJK->ABcIJK', H.aa.vvvv[va, Va, Va, Va], R.aaa.VVVOOO, optimize=True)
    )
    dR.aaa.VVvOOO += (6.0 / 12.0) * (
            +1.0 * np.einsum('AmIe,BcemJK->ABcIJK', H.aa.voov[Va, oa, Oa, va], R.aaa.VvvoOO, optimize=True)
            + 1.0 * np.einsum('AMIe,BceMJK->ABcIJK', H.aa.voov[Va, Oa, Oa, va], R.aaa.VvvOOO, optimize=True)
            - 1.0 * np.einsum('AmIE,BEcmJK->ABcIJK', H.aa.voov[Va, oa, Oa, Va], R.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('AMIE,BEcMJK->ABcIJK', H.aa.voov[Va, Oa, Oa, Va], R.aaa.VVvOOO, optimize=True)
    )
    dR.aaa.VVvOOO += (3.0 / 12.0) * (
            +1.0 * np.einsum('cmIe,ABemJK->ABcIJK', H.aa.voov[va, oa, Oa, va], R.aaa.VVvoOO, optimize=True)
            + 1.0 * np.einsum('cMIe,ABeMJK->ABcIJK', H.aa.voov[va, Oa, Oa, va], R.aaa.VVvOOO, optimize=True)
            + 1.0 * np.einsum('cmIE,ABEmJK->ABcIJK', H.aa.voov[va, oa, Oa, Va], R.aaa.VVVoOO, optimize=True)
            + 1.0 * np.einsum('cMIE,ABEMJK->ABcIJK', H.aa.voov[va, Oa, Oa, Va], R.aaa.VVVOOO, optimize=True)
    )
    dR.aaa.VVvOOO += (6.0 / 12.0) * (
            +1.0 * np.einsum('AmIe,BceJKm->ABcIJK', H.ab.voov[Va, ob, Oa, vb], R.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('AMIe,BceJKM->ABcIJK', H.ab.voov[Va, Ob, Oa, vb], R.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('AmIE,BcEJKm->ABcIJK', H.ab.voov[Va, ob, Oa, Vb], R.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('AMIE,BcEJKM->ABcIJK', H.ab.voov[Va, Ob, Oa, Vb], R.aab.VvVOOO, optimize=True)
    )
    dR.aaa.VVvOOO += (3.0 / 12.0) * (
            +1.0 * np.einsum('cmIe,ABeJKm->ABcIJK', H.ab.voov[va, ob, Oa, vb], R.aab.VVvOOo, optimize=True)
            + 1.0 * np.einsum('cMIe,ABeJKM->ABcIJK', H.ab.voov[va, Ob, Oa, vb], R.aab.VVvOOO, optimize=True)
            + 1.0 * np.einsum('cmIE,ABEJKm->ABcIJK', H.ab.voov[va, ob, Oa, Vb], R.aab.VVVOOo, optimize=True)
            + 1.0 * np.einsum('cMIE,ABEJKM->ABcIJK', H.ab.voov[va, Ob, Oa, Vb], R.aab.VVVOOO, optimize=True)
    )

    dR.aaa.VVvOOO -= np.transpose(dR.aaa.VVvOOO, (1, 0, 2, 3, 4, 5))

    dR.aaa.VVvOOO -= np.transpose(dR.aaa.VVvOOO, (0, 1, 2, 4, 3, 5)) + np.transpose(dR.aaa.VVvOOO, (0, 1, 2, 3, 5, 4)) \
           + np.transpose(dR.aaa.VVvOOO, (0, 1, 2, 5, 4, 3)) - np.transpose(dR.aaa.VVvOOO, (0, 1, 2, 4, 5, 3)) \
           - np.transpose(dR.aaa.VVvOOO, (0, 1, 2, 5, 3, 4))

    return dR

def update(R, omega, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    R.aaa.VVvOOO = eomcc_active_loops.update_r3a_110111(
        R.aaa.VVvOOO,
        omega,
        H.a.oo[Oa, Oa],
        H.a.vv[Va, Va],
        H.a.oo[oa, oa],
        H.a.vv[va, va],
        0.0,
    )

    return R