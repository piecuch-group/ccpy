import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import cc_active_loops

#@profile
def build(T, dT, H, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    # MM(2,3)
    dT.aaa.VvvoOO = (2.0 / 4.0) * (
            -1.0 * np.einsum('AmiJ,bcmK->AbciJK', H.aa.vooo[Va, :, oa, Oa], T.aa[va, va, :, Oa], optimize=True)
    )
    dT.aaa.VvvoOO += (4.0 / 4.0) * (
            +1.0 * np.einsum('bmiJ,AcmK->AbciJK', H.aa.vooo[va, :, oa, Oa], T.aa[Va, va, :, Oa], optimize=True)
    )
    dT.aaa.VvvoOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('AmKJ,bcmi->AbciJK', H.aa.vooo[Va, :, Oa, Oa], T.aa[va, va, :, oa], optimize=True)
    )
    dT.aaa.VvvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('bmKJ,Acmi->AbciJK', H.aa.vooo[va, :, Oa, Oa], T.aa[Va, va, :, oa], optimize=True)
    )
    dT.aaa.VvvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('Abie,ecJK->AbciJK', H.aa.vvov[Va, va, oa, :], T.aa[:, va, Oa, Oa], optimize=True)
    )
    dT.aaa.VvvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('cbie,eAJK->AbciJK', H.aa.vvov[va, va, oa, :], T.aa[:, Va, Oa, Oa], optimize=True)
    )
    dT.aaa.VvvoOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('AbJe,eciK->AbciJK', H.aa.vvov[Va, va, Oa, :], T.aa[:, va, oa, Oa], optimize=True)
    )
    dT.aaa.VvvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('cbJe,eAiK->AbciJK', H.aa.vvov[va, va, Oa, :], T.aa[:, Va, oa, Oa], optimize=True)
    )
    # (H(2) * T3)_C
    dT.aaa.VvvoOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('mi,AcbmJK->AbciJK', H.a.oo[oa, oa], T.aaa.VvvoOO, optimize=True)
            + 1.0 * np.einsum('Mi,AcbMJK->AbciJK', H.a.oo[Oa, oa], T.aaa.VvvOOO, optimize=True)
    )
    dT.aaa.VvvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('mJ,AcbmiK->AbciJK', H.a.oo[oa, Oa], T.aaa.VvvooO, optimize=True)
            + 1.0 * np.einsum('MJ,AcbiMK->AbciJK', H.a.oo[Oa, Oa], T.aaa.VvvoOO, optimize=True)
    )
    dT.aaa.VvvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('AE,EcbiJK->AbciJK', H.a.vv[Va, Va], T.aaa.VvvoOO, optimize=True)
    )
    dT.aaa.VvvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('be,AceiJK->AbciJK', H.a.vv[va, va], T.aaa.VvvoOO, optimize=True)
            + 1.0 * np.einsum('bE,AEciJK->AbciJK', H.a.vv[va, Va], T.aaa.VVvoOO, optimize=True)
    )
    dT.aaa.VvvoOO += (2.0 / 4.0) * (
            -0.5 * np.einsum('mniJ,AcbmnK->AbciJK', H.aa.oooo[oa, oa, oa, Oa], T.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('mNiJ,AcbmNK->AbciJK', H.aa.oooo[oa, Oa, oa, Oa], T.aaa.VvvoOO, optimize=True)
            - 0.5 * np.einsum('MNiJ,AcbMNK->AbciJK', H.aa.oooo[Oa, Oa, oa, Oa], T.aaa.VvvOOO, optimize=True)
    )
    dT.aaa.VvvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('mNKJ,AcbmiN->AbciJK', H.aa.oooo[oa, Oa, Oa, Oa], T.aaa.VvvooO, optimize=True)
            + 0.5 * np.einsum('MNKJ,AcbiMN->AbciJK', H.aa.oooo[Oa, Oa, Oa, Oa], T.aaa.VvvoOO, optimize=True)
    )
    dT.aaa.VvvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('AbEf,EcfiJK->AbciJK', H.aa.vvvv[Va, va, Va, va], T.aaa.VvvoOO, optimize=True)
            - 0.5 * np.einsum('AbEF,FEciJK->AbciJK', H.aa.vvvv[Va, va, Va, Va], T.aaa.VVvoOO, optimize=True)
    )
    dT.aaa.VvvoOO += (1.0 / 4.0) * (
            +0.5 * np.einsum('cbef,AfeiJK->AbciJK', H.aa.vvvv[va, va, va, va], T.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('cbEf,AEfiJK->AbciJK', H.aa.vvvv[va, va, Va, va], T.aaa.VVvoOO, optimize=True)
            + 0.5 * np.einsum('cbEF,AFEiJK->AbciJK', H.aa.vvvv[va, va, Va, Va], T.aaa.VVVoOO, optimize=True)
    )
    dT.aaa.VvvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('AmiE,EcbmJK->AbciJK', H.aa.voov[Va, oa, oa, Va], T.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('AMiE,EcbMJK->AbciJK', H.aa.voov[Va, Oa, oa, Va], T.aaa.VvvOOO, optimize=True)
    )
    dT.aaa.VvvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('bmie,AcemJK->AbciJK', H.aa.voov[va, oa, oa, va], T.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('bMie,AceMJK->AbciJK', H.aa.voov[va, Oa, oa, va], T.aaa.VvvOOO, optimize=True)
            + 1.0 * np.einsum('bmiE,AEcmJK->AbciJK', H.aa.voov[va, oa, oa, Va], T.aaa.VVvoOO, optimize=True)
            + 1.0 * np.einsum('bMiE,AEcMJK->AbciJK', H.aa.voov[va, Oa, oa, Va], T.aaa.VVvOOO, optimize=True)
    )
    dT.aaa.VvvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('AmJE,EcbmiK->AbciJK', H.aa.voov[Va, oa, Oa, Va], T.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('AMJE,EcbiMK->AbciJK', H.aa.voov[Va, Oa, Oa, Va], T.aaa.VvvoOO, optimize=True)
    )
    dT.aaa.VvvoOO += (4.0 / 4.0) * (
            +1.0 * np.einsum('bmJe,AcemiK->AbciJK', H.aa.voov[va, oa, Oa, va], T.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('bMJe,AceiMK->AbciJK', H.aa.voov[va, Oa, Oa, va], T.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('bmJE,AEcmiK->AbciJK', H.aa.voov[va, oa, Oa, Va], T.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('bMJE,AEciMK->AbciJK', H.aa.voov[va, Oa, Oa, Va], T.aaa.VVvoOO, optimize=True)
    )
    dT.aaa.VvvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('AmiE,cbEJKm->AbciJK', H.ab.voov[Va, ob, oa, Vb], T.aab.vvVOOo, optimize=True)
            - 1.0 * np.einsum('AMiE,cbEJKM->AbciJK', H.ab.voov[Va, Ob, oa, Vb], T.aab.vvVOOO, optimize=True)
    )
    dT.aaa.VvvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('bmie,AceJKm->AbciJK', H.ab.voov[va, ob, oa, vb], T.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('bMie,AceJKM->AbciJK', H.ab.voov[va, Ob, oa, vb], T.aab.VvvOOO, optimize=True)
            - 1.0 * np.einsum('bmiE,AcEJKm->AbciJK', H.ab.voov[va, ob, oa, Vb], T.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('bMiE,AcEJKM->AbciJK', H.ab.voov[va, Ob, oa, Vb], T.aab.VvVOOO, optimize=True)
    )
    dT.aaa.VvvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('AmJE,cbEiKm->AbciJK', H.ab.voov[Va, ob, Oa, Vb], T.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('AMJE,cbEiKM->AbciJK', H.ab.voov[Va, Ob, Oa, Vb], T.aab.vvVoOO, optimize=True)
    )
    dT.aaa.VvvoOO += (4.0 / 4.0) * (
            +1.0 * np.einsum('bmJe,AceiKm->AbciJK', H.ab.voov[va, ob, Oa, vb], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('bMJe,AceiKM->AbciJK', H.ab.voov[va, Ob, Oa, vb], T.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('bmJE,AcEiKm->AbciJK', H.ab.voov[va, ob, Oa, Vb], T.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('bMJE,AcEiKM->AbciJK', H.ab.voov[va, Ob, Oa, Vb], T.aab.VvVoOO, optimize=True)
    )

    dT.aaa.VvvoOO -= np.transpose(dT.aaa.VvvoOO, (0, 2, 1, 3, 4, 5))
    dT.aaa.VvvoOO -= np.transpose(dT.aaa.VvvoOO, (0, 1, 2, 3, 5, 4))

    return dT

def update(T, dT, H, shift, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.aaa.VvvoOO, dT.aaa.VvvoOO = cc_active_loops.update_t3a_100011(
        T.aaa.VvvoOO,
        dT.aaa.VvvoOO,
        H.a.oo[Oa, Oa],
        H.a.vv[Va, Va],
        H.a.oo[oa, oa],
        H.a.vv[va, va],
        shift,
    )

    return T, dT