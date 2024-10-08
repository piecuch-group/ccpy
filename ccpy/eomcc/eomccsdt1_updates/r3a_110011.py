import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops


def build(dR, R, T, H, X, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.aaa.VVvoOO = (4.0 / 4.0) * (
            -1.0 * np.einsum('BmJi,AcmK->ABciJK', X.aa.vooo[Va, :, Oa, oa], T.aa[Va, va, :, Oa], optimize=True)
    )
    dR.aaa.VVvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('cmJi,ABmK->ABciJK', X.aa.vooo[va, :, Oa, oa], T.aa[Va, Va, :, Oa], optimize=True)
    )
    dR.aaa.VVvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('BmJK,Acmi->ABciJK', X.aa.vooo[Va, :, Oa, Oa], T.aa[Va, va, :, oa], optimize=True)
    )
    dR.aaa.VVvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('cmJK,ABmi->ABciJK', X.aa.vooo[va, :, Oa, Oa], T.aa[Va, Va, :, oa], optimize=True)
    )
    dR.aaa.VVvoOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('BmJi,AcmK->ABciJK', H.aa.vooo[Va, :, Oa, oa], R.aa[Va, va, :, Oa], optimize=True)
    )
    dR.aaa.VVvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('cmJi,ABmK->ABciJK', H.aa.vooo[va, :, Oa, oa], R.aa[Va, Va, :, Oa], optimize=True)
    )
    dR.aaa.VVvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('BmJK,Acmi->ABciJK', H.aa.vooo[Va, :, Oa, Oa], R.aa[Va, va, :, oa], optimize=True)
    )
    dR.aaa.VVvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('cmJK,ABmi->ABciJK', H.aa.vooo[va, :, Oa, Oa], R.aa[Va, Va, :, oa], optimize=True)
    )
    dR.aaa.VVvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('BAJe,eciK->ABciJK', X.aa.vvov[Va, Va, Oa, :], T.aa[:, va, oa, Oa], optimize=True)
    )
    dR.aaa.VVvoOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('BcJe,eAiK->ABciJK', X.aa.vvov[Va, va, Oa, :], T.aa[:, Va, oa, Oa], optimize=True)
    )
    dR.aaa.VVvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('BAie,ecJK->ABciJK', X.aa.vvov[Va, Va, oa, :], T.aa[:, va, Oa, Oa], optimize=True)
    )
    dR.aaa.VVvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('Bcie,eAJK->ABciJK', X.aa.vvov[Va, va, oa, :], T.aa[:, Va, Oa, Oa], optimize=True)
    )
    dR.aaa.VVvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('BAJe,eciK->ABciJK', H.aa.vvov[Va, Va, Oa, :], R.aa[:, va, oa, Oa], optimize=True)
    )
    dR.aaa.VVvoOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('BcJe,eAiK->ABciJK', H.aa.vvov[Va, va, Oa, :], R.aa[:, Va, oa, Oa], optimize=True)
    )
    dR.aaa.VVvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('BAie,ecJK->ABciJK', H.aa.vvov[Va, Va, oa, :], R.aa[:, va, Oa, Oa], optimize=True)
    )
    dR.aaa.VVvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('Bcie,eAJK->ABciJK', H.aa.vvov[Va, va, oa, :], R.aa[:, Va, Oa, Oa], optimize=True)
    )

    dR.aaa.VVvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('Be,AceiJK->ABciJK', X.a.vv[Va, va], T.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('BE,EAciJK->ABciJK', X.a.vv[Va, Va], T.aaa.VVvoOO, optimize=True)
    )
    dR.aaa.VVvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('ce,BAeiJK->ABciJK', X.a.vv[va, va], T.aaa.VVvoOO, optimize=True)
            + 1.0 * np.einsum('cE,BEAiJK->ABciJK', X.a.vv[va, Va], T.aaa.VVVoOO, optimize=True)
    )
    dR.aaa.VVvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mJ,BAcimK->ABciJK', X.a.oo[oa, Oa], T.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('MJ,BAciMK->ABciJK', X.a.oo[Oa, Oa], T.aaa.VVvoOO, optimize=True)
    )
    dR.aaa.VVvoOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('mi,BAcmJK->ABciJK', X.a.oo[oa, oa], T.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('Mi,BAcJMK->ABciJK', X.a.oo[Oa, oa], T.aaa.VVvOOO, optimize=True)
    )
    dR.aaa.VVvoOO += (2.0 / 4.0) * (
            -0.5 * np.einsum('mniJ,BAcmnK->ABciJK', X.aa.oooo[oa, oa, oa, Oa], T.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('MniJ,BAcnMK->ABciJK', X.aa.oooo[Oa, oa, oa, Oa], T.aaa.VVvoOO, optimize=True)
            - 0.5 * np.einsum('MNiJ,BAcMNK->ABciJK', X.aa.oooo[Oa, Oa, oa, Oa], T.aaa.VVvOOO, optimize=True)
    )
    dR.aaa.VVvoOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('MnKJ,BAcniM->ABciJK', X.aa.oooo[Oa, oa, Oa, Oa], T.aaa.VVvooO, optimize=True)
            + 0.5 * np.einsum('MNKJ,BAciMN->ABciJK', X.aa.oooo[Oa, Oa, Oa, Oa], T.aaa.VVvoOO, optimize=True)
    )
    dR.aaa.VVvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('ABEf,EcfiJK->ABciJK', X.aa.vvvv[Va, Va, Va, va], T.aaa.VvvoOO, optimize=True)
            - 0.5 * np.einsum('ABEF,FEciJK->ABciJK', X.aa.vvvv[Va, Va, Va, Va], T.aaa.VVvoOO, optimize=True)
    )
    dR.aaa.VVvoOO += (2.0 / 4.0) * (
            +0.5 * np.einsum('cBef,AfeiJK->ABciJK', X.aa.vvvv[va, Va, va, va], T.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('cBEf,AEfiJK->ABciJK', X.aa.vvvv[va, Va, Va, va], T.aaa.VVvoOO, optimize=True)
            + 0.5 * np.einsum('cBEF,AFEiJK->ABciJK', X.aa.vvvv[va, Va, Va, Va], T.aaa.VVVoOO, optimize=True)
    )
    dR.aaa.VVvoOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('BmJe,AceimK->ABciJK', X.aa.voov[Va, oa, Oa, va], T.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('BmJE,EAcimK->ABciJK', X.aa.voov[Va, oa, Oa, Va], T.aaa.VVvooO, optimize=True)
            - 1.0 * np.einsum('BMJe,AceiMK->ABciJK', X.aa.voov[Va, Oa, Oa, va], T.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('BMJE,EAciMK->ABciJK', X.aa.voov[Va, Oa, Oa, Va], T.aaa.VVvoOO, optimize=True)
    )
    dR.aaa.VVvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('cmJe,BAeimK->ABciJK', X.aa.voov[va, oa, Oa, va], T.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('cmJE,BEAimK->ABciJK', X.aa.voov[va, oa, Oa, Va], T.aaa.VVVooO, optimize=True)
            - 1.0 * np.einsum('cMJe,BAeiMK->ABciJK', X.aa.voov[va, Oa, Oa, va], T.aaa.VVvoOO, optimize=True)
            + 1.0 * np.einsum('cMJE,BEAiMK->ABciJK', X.aa.voov[va, Oa, Oa, Va], T.aaa.VVVoOO, optimize=True)
    )
    dR.aaa.VVvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('Bmie,AcemJK->ABciJK', X.aa.voov[Va, oa, oa, va], T.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('BmiE,EAcmJK->ABciJK', X.aa.voov[Va, oa, oa, Va], T.aaa.VVvoOO, optimize=True)
            + 1.0 * np.einsum('BMie,AceJMK->ABciJK', X.aa.voov[Va, Oa, oa, va], T.aaa.VvvOOO, optimize=True)
            + 1.0 * np.einsum('BMiE,EAcJMK->ABciJK', X.aa.voov[Va, Oa, oa, Va], T.aaa.VVvOOO, optimize=True)
    )
    dR.aaa.VVvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('cmie,BAemJK->ABciJK', X.aa.voov[va, oa, oa, va], T.aaa.VVvoOO, optimize=True)
            + 1.0 * np.einsum('cmiE,BEAmJK->ABciJK', X.aa.voov[va, oa, oa, Va], T.aaa.VVVoOO, optimize=True)
            + 1.0 * np.einsum('cMie,BAeJMK->ABciJK', X.aa.voov[va, Oa, oa, va], T.aaa.VVvOOO, optimize=True)
            - 1.0 * np.einsum('cMiE,BEAJMK->ABciJK', X.aa.voov[va, Oa, oa, Va], T.aaa.VVVOOO, optimize=True)
    )
    dR.aaa.VVvoOO += (4.0 / 4.0) * (
            +1.0 * np.einsum('BmJe,AceiKm->ABciJK', X.ab.voov[Va, ob, Oa, vb], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('BmJE,AcEiKm->ABciJK', X.ab.voov[Va, ob, Oa, Vb], T.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('BMJe,AceiKM->ABciJK', X.ab.voov[Va, Ob, Oa, vb], T.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('BMJE,AcEiKM->ABciJK', X.ab.voov[Va, Ob, Oa, Vb], T.aab.VvVoOO, optimize=True)
    )
    dR.aaa.VVvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('cmJe,BAeiKm->ABciJK', X.ab.voov[va, ob, Oa, vb], T.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('cmJE,BAEiKm->ABciJK', X.ab.voov[va, ob, Oa, Vb], T.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('cMJe,BAeiKM->ABciJK', X.ab.voov[va, Ob, Oa, vb], T.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('cMJE,BAEiKM->ABciJK', X.ab.voov[va, Ob, Oa, Vb], T.aab.VVVoOO, optimize=True)
    )
    dR.aaa.VVvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('Bmie,AceJKm->ABciJK', X.ab.voov[Va, ob, oa, vb], T.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('BmiE,AcEJKm->ABciJK', X.ab.voov[Va, ob, oa, Vb], T.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('BMie,AceJKM->ABciJK', X.ab.voov[Va, Ob, oa, vb], T.aab.VvvOOO, optimize=True)
            - 1.0 * np.einsum('BMiE,AcEJKM->ABciJK', X.ab.voov[Va, Ob, oa, Vb], T.aab.VvVOOO, optimize=True)
    )
    dR.aaa.VVvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('cmie,BAeJKm->ABciJK', X.ab.voov[va, ob, oa, vb], T.aab.VVvOOo, optimize=True)
            - 1.0 * np.einsum('cmiE,BAEJKm->ABciJK', X.ab.voov[va, ob, oa, Vb], T.aab.VVVOOo, optimize=True)
            - 1.0 * np.einsum('cMie,BAeJKM->ABciJK', X.ab.voov[va, Ob, oa, vb], T.aab.VVvOOO, optimize=True)
            - 1.0 * np.einsum('cMiE,BAEJKM->ABciJK', X.ab.voov[va, Ob, oa, Vb], T.aab.VVVOOO, optimize=True)
    )
    dR.aaa.VVvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mJ,BAcimK->ABciJK', H.a.oo[oa, Oa], R.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('MJ,BAciMK->ABciJK', H.a.oo[Oa, Oa], R.aaa.VVvoOO, optimize=True)
    )
    dR.aaa.VVvoOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('mi,BAcmJK->ABciJK', H.a.oo[oa, oa], R.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('Mi,BAcJMK->ABciJK', H.a.oo[Oa, oa], R.aaa.VVvOOO, optimize=True)
    )
    dR.aaa.VVvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('Be,AceiJK->ABciJK', H.a.vv[Va, va], R.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('BE,EAciJK->ABciJK', H.a.vv[Va, Va], R.aaa.VVvoOO, optimize=True)
    )
    dR.aaa.VVvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('ce,BAeiJK->ABciJK', H.a.vv[va, va], R.aaa.VVvoOO, optimize=True)
            + 1.0 * np.einsum('cE,BEAiJK->ABciJK', H.a.vv[va, Va], R.aaa.VVVoOO, optimize=True)
    )
    dR.aaa.VVvoOO += (2.0 / 4.0) * (
            -0.5 * np.einsum('mniJ,BAcmnK->ABciJK', H.aa.oooo[oa, oa, oa, Oa], R.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('MniJ,BAcnMK->ABciJK', H.aa.oooo[Oa, oa, oa, Oa], R.aaa.VVvoOO, optimize=True)
            - 0.5 * np.einsum('MNiJ,BAcMNK->ABciJK', H.aa.oooo[Oa, Oa, oa, Oa], R.aaa.VVvOOO, optimize=True)
    )
    dR.aaa.VVvoOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('MnKJ,BAcniM->ABciJK', H.aa.oooo[Oa, oa, Oa, Oa], R.aaa.VVvooO, optimize=True)
            + 0.5 * np.einsum('MNKJ,BAciMN->ABciJK', H.aa.oooo[Oa, Oa, Oa, Oa], R.aaa.VVvoOO, optimize=True)
    )
    dR.aaa.VVvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('ABEf,EcfiJK->ABciJK', H.aa.vvvv[Va, Va, Va, va], R.aaa.VvvoOO, optimize=True)
            - 0.5 * np.einsum('ABEF,FEciJK->ABciJK', H.aa.vvvv[Va, Va, Va, Va], R.aaa.VVvoOO, optimize=True)
    )
    dR.aaa.VVvoOO += (2.0 / 4.0) * (
            +0.5 * np.einsum('cBef,AfeiJK->ABciJK', H.aa.vvvv[va, Va, va, va], R.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('cBEf,AEfiJK->ABciJK', H.aa.vvvv[va, Va, Va, va], R.aaa.VVvoOO, optimize=True)
            + 0.5 * np.einsum('cBEF,AFEiJK->ABciJK', H.aa.vvvv[va, Va, Va, Va], R.aaa.VVVoOO, optimize=True)
    )
    dR.aaa.VVvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('Amie,BcemJK->ABciJK', H.aa.voov[Va, oa, oa, va], R.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('AmiE,BEcmJK->ABciJK', H.aa.voov[Va, oa, oa, Va], R.aaa.VVvoOO, optimize=True)
            + 1.0 * np.einsum('AMie,BceMJK->ABciJK', H.aa.voov[Va, Oa, oa, va], R.aaa.VvvOOO, optimize=True)
            - 1.0 * np.einsum('AMiE,BEcMJK->ABciJK', H.aa.voov[Va, Oa, oa, Va], R.aaa.VVvOOO, optimize=True)
    )
    dR.aaa.VVvoOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('cmie,ABemJK->ABciJK', H.aa.voov[va, oa, oa, va], R.aaa.VVvoOO, optimize=True)
            + 1.0 * np.einsum('cmiE,ABEmJK->ABciJK', H.aa.voov[va, oa, oa, Va], R.aaa.VVVoOO, optimize=True)
            + 1.0 * np.einsum('cMie,ABeMJK->ABciJK', H.aa.voov[va, Oa, oa, va], R.aaa.VVvOOO, optimize=True)
            + 1.0 * np.einsum('cMiE,ABEMJK->ABciJK', H.aa.voov[va, Oa, oa, Va], R.aaa.VVVOOO, optimize=True)
    )
    dR.aaa.VVvoOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('AmJe,BcemiK->ABciJK', H.aa.voov[Va, oa, Oa, va], R.aaa.VvvooO, optimize=True)
            + 1.0 * np.einsum('AmJE,BEcmiK->ABciJK', H.aa.voov[Va, oa, Oa, Va], R.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('AMJe,BceiMK->ABciJK', H.aa.voov[Va, Oa, Oa, va], R.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('AMJE,BEciMK->ABciJK', H.aa.voov[Va, Oa, Oa, Va], R.aaa.VVvoOO, optimize=True)
    )
    dR.aaa.VVvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('cmJe,ABemiK->ABciJK', H.aa.voov[va, oa, Oa, va], R.aaa.VVvooO, optimize=True)
            - 1.0 * np.einsum('cmJE,ABEmiK->ABciJK', H.aa.voov[va, oa, Oa, Va], R.aaa.VVVooO, optimize=True)
            + 1.0 * np.einsum('cMJe,ABeiMK->ABciJK', H.aa.voov[va, Oa, Oa, va], R.aaa.VVvoOO, optimize=True)
            + 1.0 * np.einsum('cMJE,ABEiMK->ABciJK', H.aa.voov[va, Oa, Oa, Va], R.aaa.VVVoOO, optimize=True)
    )
    dR.aaa.VVvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('Amie,BceJKm->ABciJK', H.ab.voov[Va, ob, oa, vb], R.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('AmiE,BcEJKm->ABciJK', H.ab.voov[Va, ob, oa, Vb], R.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('AMie,BceJKM->ABciJK', H.ab.voov[Va, Ob, oa, vb], R.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('AMiE,BcEJKM->ABciJK', H.ab.voov[Va, Ob, oa, Vb], R.aab.VvVOOO, optimize=True)
    )
    dR.aaa.VVvoOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('cmie,ABeJKm->ABciJK', H.ab.voov[va, ob, oa, vb], R.aab.VVvOOo, optimize=True)
            + 1.0 * np.einsum('cmiE,ABEJKm->ABciJK', H.ab.voov[va, ob, oa, Vb], R.aab.VVVOOo, optimize=True)
            + 1.0 * np.einsum('cMie,ABeJKM->ABciJK', H.ab.voov[va, Ob, oa, vb], R.aab.VVvOOO, optimize=True)
            + 1.0 * np.einsum('cMiE,ABEJKM->ABciJK', H.ab.voov[va, Ob, oa, Vb], R.aab.VVVOOO, optimize=True)
    )
    dR.aaa.VVvoOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('AmJe,BceiKm->ABciJK', H.ab.voov[Va, ob, Oa, vb], R.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('AmJE,BcEiKm->ABciJK', H.ab.voov[Va, ob, Oa, Vb], R.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('AMJe,BceiKM->ABciJK', H.ab.voov[Va, Ob, Oa, vb], R.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('AMJE,BcEiKM->ABciJK', H.ab.voov[Va, Ob, Oa, Vb], R.aab.VvVoOO, optimize=True)
    )
    dR.aaa.VVvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('cmJe,ABeiKm->ABciJK', H.ab.voov[va, ob, Oa, vb], R.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('cmJE,ABEiKm->ABciJK', H.ab.voov[va, ob, Oa, Vb], R.aab.VVVoOo, optimize=True)
            - 1.0 * np.einsum('cMJe,ABeiKM->ABciJK', H.ab.voov[va, Ob, Oa, vb], R.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('cMJE,ABEiKM->ABciJK', H.ab.voov[va, Ob, Oa, Vb], R.aab.VVVoOO, optimize=True)
    )

    dR.aaa.VVvoOO -= np.transpose(dR.aaa.VVvoOO, (1, 0, 2, 3, 4, 5))
    dR.aaa.VVvoOO -= np.transpose(dR.aaa.VVvoOO, (0, 1, 2, 3, 5, 4))

    return dR

def update(R, omega, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    R.aaa.VVvoOO = eomcc_active_loops.update_r3a_110011(
        R.aaa.VVvoOO,
        omega,
        H.a.oo[Oa, Oa],
        H.a.vv[Va, Va],
        H.a.oo[oa, oa],
        H.a.vv[va, va],
        0.0,
    )

    return R