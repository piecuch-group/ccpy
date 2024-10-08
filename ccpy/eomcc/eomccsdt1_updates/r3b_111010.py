import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops

def build(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.aab.VVVoOo = (2.0 / 2.0) * (
            +1.0 * np.einsum('BCek,AeiJ->ABCiJk', X.ab.vvvo[Va, Vb, :, ob], T.aa[Va, :, oa, Oa], optimize=True)
    )
    dR.aab.VVVoOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('mCJk,ABim->ABCiJk', X.ab.ovoo[:, Vb, Oa, ob], T.aa[Va, Va, oa, :], optimize=True)
    )
    dR.aab.VVVoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('mCik,ABJm->ABCiJk', X.ab.ovoo[:, Vb, oa, ob], T.aa[Va, Va, Oa, :], optimize=True)
    )
    dR.aab.VVVoOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('ACie,BeJk->ABCiJk', X.ab.vvov[Va, Vb, oa, :], T.ab[Va, :, Oa, ob], optimize=True)
    )
    dR.aab.VVVoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('ACJe,Beik->ABCiJk', X.ab.vvov[Va, Vb, Oa, :], T.ab[Va, :, oa, ob], optimize=True)
    )
    dR.aab.VVVoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('Amik,BCJm->ABCiJk', X.ab.vooo[Va, :, oa, ob], T.ab[Va, Vb, Oa, :], optimize=True)
    )
    dR.aab.VVVoOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('AmJk,BCim->ABCiJk', X.ab.vooo[Va, :, Oa, ob], T.ab[Va, Vb, oa, :], optimize=True)
    )
    dR.aab.VVVoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('ABie,eCJk->ABCiJk', X.aa.vvov[Va, Va, oa, :], T.ab[:, Vb, Oa, ob], optimize=True)
    )
    dR.aab.VVVoOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('ABJe,eCik->ABCiJk', X.aa.vvov[Va, Va, Oa, :], T.ab[:, Vb, oa, ob], optimize=True)
    )
    dR.aab.VVVoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('AmiJ,BCmk->ABCiJk', X.aa.vooo[Va, :, oa, Oa], T.ab[Va, Vb, :, ob], optimize=True)
    )
    dR.aab.VVVoOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('BCek,AeiJ->ABCiJk', H.ab.vvvo[Va, Vb, :, ob], R.aa[Va, :, oa, Oa], optimize=True)
    )
    dR.aab.VVVoOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('mCJk,ABim->ABCiJk', H.ab.ovoo[:, Vb, Oa, ob], R.aa[Va, Va, oa, :], optimize=True)
    )
    dR.aab.VVVoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('mCik,ABJm->ABCiJk', H.ab.ovoo[:, Vb, oa, ob], R.aa[Va, Va, Oa, :], optimize=True)
    )
    dR.aab.VVVoOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('ACie,BeJk->ABCiJk', H.ab.vvov[Va, Vb, oa, :], R.ab[Va, :, Oa, ob], optimize=True)
    )
    dR.aab.VVVoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('ACJe,Beik->ABCiJk', H.ab.vvov[Va, Vb, Oa, :], R.ab[Va, :, oa, ob], optimize=True)
    )
    dR.aab.VVVoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('Amik,BCJm->ABCiJk', H.ab.vooo[Va, :, oa, ob], R.ab[Va, Vb, Oa, :], optimize=True)
    )
    dR.aab.VVVoOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('AmJk,BCim->ABCiJk', H.ab.vooo[Va, :, Oa, ob], R.ab[Va, Vb, oa, :], optimize=True)
    )
    dR.aab.VVVoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('ABie,eCJk->ABCiJk', H.aa.vvov[Va, Va, oa, :], R.ab[:, Vb, Oa, ob], optimize=True)
    )
    dR.aab.VVVoOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('ABJe,eCik->ABCiJk', H.aa.vvov[Va, Va, Oa, :], R.ab[:, Vb, oa, ob], optimize=True)
    )
    dR.aab.VVVoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('AmiJ,BCmk->ABCiJk', H.aa.vooo[Va, :, oa, Oa], R.ab[Va, Vb, :, ob], optimize=True)
    )

    dR.aab.VVVoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('mi,BACmJk->ABCiJk', X.a.oo[oa, oa], T.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('Mi,BACMJk->ABCiJk', X.a.oo[Oa, oa], T.aab.VVVOOo, optimize=True)
    )
    dR.aab.VVVoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('MJ,BACiMk->ABCiJk', X.a.oo[Oa, Oa], T.aab.VVVoOo, optimize=True)
    )
    dR.aab.VVVoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('mk,BACiJm->ABCiJk', X.b.oo[ob, ob], T.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('Mk,BACiJM->ABCiJk', X.b.oo[Ob, ob], T.aab.VVVoOO, optimize=True)
    )
    dR.aab.VVVoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('Ae,BeCiJk->ABCiJk', X.a.vv[Va, va], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('AE,BECiJk->ABCiJk', X.a.vv[Va, Va], T.aab.VVVoOo, optimize=True)
    )
    dR.aab.VVVoOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('Ce,BAeiJk->ABCiJk', X.b.vv[Vb, vb], T.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('CE,BAEiJk->ABCiJk', X.b.vv[Vb, Vb], T.aab.VVVoOo, optimize=True)
    )
    dR.aab.VVVoOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('mNiJ,BACmNk->ABCiJk', X.aa.oooo[oa, Oa, oa, Oa], T.aab.VVVoOo, optimize=True)
            - 0.5 * np.einsum('MNiJ,BACMNk->ABCiJk', X.aa.oooo[Oa, Oa, oa, Oa], T.aab.VVVOOo, optimize=True)
    )
    dR.aab.VVVoOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('MnJk,BACiMn->ABCiJk', X.ab.oooo[Oa, ob, Oa, ob], T.aab.VVVoOo, optimize=True)
            - 1.0 * np.einsum('mNJk,BACimN->ABCiJk', X.ab.oooo[oa, Ob, Oa, ob], T.aab.VVVooO, optimize=True)
            - 1.0 * np.einsum('MNJk,BACiMN->ABCiJk', X.ab.oooo[Oa, Ob, Oa, ob], T.aab.VVVoOO, optimize=True)
    )
    dR.aab.VVVoOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('mnik,BACmJn->ABCiJk', X.ab.oooo[oa, ob, oa, ob], T.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('Mnik,BACJMn->ABCiJk', X.ab.oooo[Oa, ob, oa, ob], T.aab.VVVOOo, optimize=True)
            - 1.0 * np.einsum('mNik,BACmJN->ABCiJk', X.ab.oooo[oa, Ob, oa, ob], T.aab.VVVoOO, optimize=True)
            + 1.0 * np.einsum('MNik,BACJMN->ABCiJk', X.ab.oooo[Oa, Ob, oa, ob], T.aab.VVVOOO, optimize=True)
    )
    dR.aab.VVVoOo += (1.0 / 2.0) * (
            -0.5 * np.einsum('ABef,feCiJk->ABCiJk', X.aa.vvvv[Va, Va, va, va], T.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('ABEf,EfCiJk->ABCiJk', X.aa.vvvv[Va, Va, Va, va], T.aab.VvVoOo, optimize=True)
            - 0.5 * np.einsum('ABEF,FECiJk->ABCiJk', X.aa.vvvv[Va, Va, Va, Va], T.aab.VVVoOo, optimize=True)
    )
    dR.aab.VVVoOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('BCef,AefiJk->ABCiJk', X.ab.vvvv[Va, Vb, va, vb], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('BCeF,AeFiJk->ABCiJk', X.ab.vvvv[Va, Vb, va, Vb], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('BCEf,EAfiJk->ABCiJk', X.ab.vvvv[Va, Vb, Va, vb], T.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('BCEF,EAFiJk->ABCiJk', X.ab.vvvv[Va, Vb, Va, Vb], T.aab.VVVoOo, optimize=True)
    )
    dR.aab.VVVoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('Amie,BeCmJk->ABCiJk', X.aa.voov[Va, oa, oa, va], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('AMie,BeCMJk->ABCiJk', X.aa.voov[Va, Oa, oa, va], T.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('AmiE,BECmJk->ABCiJk', X.aa.voov[Va, oa, oa, Va], T.aab.VVVoOo, optimize=True)
            - 1.0 * np.einsum('AMiE,BECMJk->ABCiJk', X.aa.voov[Va, Oa, oa, Va], T.aab.VVVOOo, optimize=True)
    )
    dR.aab.VVVoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('AMJe,BeCiMk->ABCiJk', X.aa.voov[Va, Oa, Oa, va], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('AMJE,BECiMk->ABCiJk', X.aa.voov[Va, Oa, Oa, Va], T.aab.VVVoOo, optimize=True)
    )
    dR.aab.VVVoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('Amie,BCeJmk->ABCiJk', X.ab.voov[Va, ob, oa, vb], T.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('AMie,BCeJkM->ABCiJk', X.ab.voov[Va, Ob, oa, vb], T.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('AmiE,BCEJmk->ABCiJk', X.ab.voov[Va, ob, oa, Vb], T.abb.VVVOoo, optimize=True)
            + 1.0 * np.einsum('AMiE,BCEJkM->ABCiJk', X.ab.voov[Va, Ob, oa, Vb], T.abb.VVVOoO, optimize=True)
    )
    dR.aab.VVVoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('AMJe,BCeikM->ABCiJk', X.ab.voov[Va, Ob, Oa, vb], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('AMJE,BCEikM->ABCiJk', X.ab.voov[Va, Ob, Oa, Vb], T.abb.VVVooO, optimize=True)
    )
    dR.aab.VVVoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('mCek,BAeimJ->ABCiJk', X.ab.ovvo[oa, Vb, va, ob], T.aaa.VVvooO, optimize=True)
            - 1.0 * np.einsum('MCek,BAeiJM->ABCiJk', X.ab.ovvo[Oa, Vb, va, ob], T.aaa.VVvoOO, optimize=True)
            + 1.0 * np.einsum('mCEk,EBAimJ->ABCiJk', X.ab.ovvo[oa, Vb, Va, ob], T.aaa.VVVooO, optimize=True)
            - 1.0 * np.einsum('MCEk,EBAiJM->ABCiJk', X.ab.ovvo[Oa, Vb, Va, ob], T.aaa.VVVoOO, optimize=True)
    )
    dR.aab.VVVoOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('Cmke,BAeiJm->ABCiJk', X.bb.voov[Vb, ob, ob, vb], T.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('CMke,BAeiJM->ABCiJk', X.bb.voov[Vb, Ob, ob, vb], T.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('CmkE,BAEiJm->ABCiJk', X.bb.voov[Vb, ob, ob, Vb], T.aab.VVVoOo, optimize=True)
            - 1.0 * np.einsum('CMkE,BAEiJM->ABCiJk', X.bb.voov[Vb, Ob, ob, Vb], T.aab.VVVoOO, optimize=True)
    )
    dR.aab.VVVoOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('Amek,BeCiJm->ABCiJk', X.ab.vovo[Va, ob, va, ob], T.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('AMek,BeCiJM->ABCiJk', X.ab.vovo[Va, Ob, va, ob], T.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('AmEk,BECiJm->ABCiJk', X.ab.vovo[Va, ob, Va, ob], T.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('AMEk,BECiJM->ABCiJk', X.ab.vovo[Va, Ob, Va, ob], T.aab.VVVoOO, optimize=True)
    )
    dR.aab.VVVoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('mCie,BAemJk->ABCiJk', X.ab.ovov[oa, Vb, oa, vb], T.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('MCie,BAeMJk->ABCiJk', X.ab.ovov[Oa, Vb, oa, vb], T.aab.VVvOOo, optimize=True)
            + 1.0 * np.einsum('mCiE,BAEmJk->ABCiJk', X.ab.ovov[oa, Vb, oa, Vb], T.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('MCiE,BAEMJk->ABCiJk', X.ab.ovov[Oa, Vb, oa, Vb], T.aab.VVVOOo, optimize=True)
    )
    dR.aab.VVVoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('MCJe,BAeiMk->ABCiJk', X.ab.ovov[Oa, Vb, Oa, vb], T.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('MCJE,BAEiMk->ABCiJk', X.ab.ovov[Oa, Vb, Oa, Vb], T.aab.VVVoOo, optimize=True)
    )
    dR.aab.VVVoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('mi,BACmJk->ABCiJk', H.a.oo[oa, oa], R.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('Mi,BACMJk->ABCiJk', H.a.oo[Oa, oa], R.aab.VVVOOo, optimize=True)
    )
    dR.aab.VVVoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('MJ,BACiMk->ABCiJk', H.a.oo[Oa, Oa], R.aab.VVVoOo, optimize=True)
    )
    dR.aab.VVVoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('mk,BACiJm->ABCiJk', H.b.oo[ob, ob], R.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('Mk,BACiJM->ABCiJk', H.b.oo[Ob, ob], R.aab.VVVoOO, optimize=True)
    )
    dR.aab.VVVoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('Ae,BeCiJk->ABCiJk', H.a.vv[Va, va], R.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('AE,BECiJk->ABCiJk', H.a.vv[Va, Va], R.aab.VVVoOo, optimize=True)
    )
    dR.aab.VVVoOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('Ce,BAeiJk->ABCiJk', H.b.vv[Vb, vb], R.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('CE,BAEiJk->ABCiJk', H.b.vv[Vb, Vb], R.aab.VVVoOo, optimize=True)
    )
    dR.aab.VVVoOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('mNiJ,BACmNk->ABCiJk', H.aa.oooo[oa, Oa, oa, Oa], R.aab.VVVoOo, optimize=True)
            - 0.5 * np.einsum('MNiJ,BACMNk->ABCiJk', H.aa.oooo[Oa, Oa, oa, Oa], R.aab.VVVOOo, optimize=True)
    )
    dR.aab.VVVoOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('MnJk,BACiMn->ABCiJk', H.ab.oooo[Oa, ob, Oa, ob], R.aab.VVVoOo, optimize=True)
            - 1.0 * np.einsum('mNJk,BACimN->ABCiJk', H.ab.oooo[oa, Ob, Oa, ob], R.aab.VVVooO, optimize=True)
            - 1.0 * np.einsum('MNJk,BACiMN->ABCiJk', H.ab.oooo[Oa, Ob, Oa, ob], R.aab.VVVoOO, optimize=True)
    )
    dR.aab.VVVoOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('mnik,BACmJn->ABCiJk', H.ab.oooo[oa, ob, oa, ob], R.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('Mnik,BACJMn->ABCiJk', H.ab.oooo[Oa, ob, oa, ob], R.aab.VVVOOo, optimize=True)
            - 1.0 * np.einsum('mNik,BACmJN->ABCiJk', H.ab.oooo[oa, Ob, oa, ob], R.aab.VVVoOO, optimize=True)
            + 1.0 * np.einsum('MNik,BACJMN->ABCiJk', H.ab.oooo[Oa, Ob, oa, ob], R.aab.VVVOOO, optimize=True)
    )
    dR.aab.VVVoOo += (1.0 / 2.0) * (
            -0.5 * np.einsum('ABef,feCiJk->ABCiJk', H.aa.vvvv[Va, Va, va, va], R.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('ABEf,EfCiJk->ABCiJk', H.aa.vvvv[Va, Va, Va, va], R.aab.VvVoOo, optimize=True)
            - 0.5 * np.einsum('ABEF,FECiJk->ABCiJk', H.aa.vvvv[Va, Va, Va, Va], R.aab.VVVoOo, optimize=True)
    )
    dR.aab.VVVoOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('BCef,AefiJk->ABCiJk', H.ab.vvvv[Va, Vb, va, vb], R.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('BCeF,AeFiJk->ABCiJk', H.ab.vvvv[Va, Vb, va, Vb], R.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('BCEf,EAfiJk->ABCiJk', H.ab.vvvv[Va, Vb, Va, vb], R.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('BCEF,EAFiJk->ABCiJk', H.ab.vvvv[Va, Vb, Va, Vb], R.aab.VVVoOo, optimize=True)
    )
    dR.aab.VVVoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('Amie,BeCmJk->ABCiJk', H.aa.voov[Va, oa, oa, va], R.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('AMie,BeCMJk->ABCiJk', H.aa.voov[Va, Oa, oa, va], R.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('AmiE,BECmJk->ABCiJk', H.aa.voov[Va, oa, oa, Va], R.aab.VVVoOo, optimize=True)
            - 1.0 * np.einsum('AMiE,BECMJk->ABCiJk', H.aa.voov[Va, Oa, oa, Va], R.aab.VVVOOo, optimize=True)
    )
    dR.aab.VVVoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('AMJe,BeCiMk->ABCiJk', H.aa.voov[Va, Oa, Oa, va], R.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('AMJE,BECiMk->ABCiJk', H.aa.voov[Va, Oa, Oa, Va], R.aab.VVVoOo, optimize=True)
    )
    dR.aab.VVVoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('Amie,BCeJmk->ABCiJk', H.ab.voov[Va, ob, oa, vb], R.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('AMie,BCeJkM->ABCiJk', H.ab.voov[Va, Ob, oa, vb], R.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('AmiE,BCEJmk->ABCiJk', H.ab.voov[Va, ob, oa, Vb], R.abb.VVVOoo, optimize=True)
            + 1.0 * np.einsum('AMiE,BCEJkM->ABCiJk', H.ab.voov[Va, Ob, oa, Vb], R.abb.VVVOoO, optimize=True)
    )
    dR.aab.VVVoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('AMJe,BCeikM->ABCiJk', H.ab.voov[Va, Ob, Oa, vb], R.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('AMJE,BCEikM->ABCiJk', H.ab.voov[Va, Ob, Oa, Vb], R.abb.VVVooO, optimize=True)
    )
    dR.aab.VVVoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('mCek,BAeimJ->ABCiJk', H.ab.ovvo[oa, Vb, va, ob], R.aaa.VVvooO, optimize=True)
            - 1.0 * np.einsum('MCek,BAeiJM->ABCiJk', H.ab.ovvo[Oa, Vb, va, ob], R.aaa.VVvoOO, optimize=True)
            + 1.0 * np.einsum('mCEk,EBAimJ->ABCiJk', H.ab.ovvo[oa, Vb, Va, ob], R.aaa.VVVooO, optimize=True)
            - 1.0 * np.einsum('MCEk,EBAiJM->ABCiJk', H.ab.ovvo[Oa, Vb, Va, ob], R.aaa.VVVoOO, optimize=True)
    )
    dR.aab.VVVoOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('Cmke,BAeiJm->ABCiJk', H.bb.voov[Vb, ob, ob, vb], R.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('CMke,BAeiJM->ABCiJk', H.bb.voov[Vb, Ob, ob, vb], R.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('CmkE,BAEiJm->ABCiJk', H.bb.voov[Vb, ob, ob, Vb], R.aab.VVVoOo, optimize=True)
            - 1.0 * np.einsum('CMkE,BAEiJM->ABCiJk', H.bb.voov[Vb, Ob, ob, Vb], R.aab.VVVoOO, optimize=True)
    )
    dR.aab.VVVoOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('Amek,BeCiJm->ABCiJk', H.ab.vovo[Va, ob, va, ob], R.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('AMek,BeCiJM->ABCiJk', H.ab.vovo[Va, Ob, va, ob], R.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('AmEk,BECiJm->ABCiJk', H.ab.vovo[Va, ob, Va, ob], R.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('AMEk,BECiJM->ABCiJk', H.ab.vovo[Va, Ob, Va, ob], R.aab.VVVoOO, optimize=True)
    )
    dR.aab.VVVoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('mCie,BAemJk->ABCiJk', H.ab.ovov[oa, Vb, oa, vb], R.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('MCie,BAeMJk->ABCiJk', H.ab.ovov[Oa, Vb, oa, vb], R.aab.VVvOOo, optimize=True)
            + 1.0 * np.einsum('mCiE,BAEmJk->ABCiJk', H.ab.ovov[oa, Vb, oa, Vb], R.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('MCiE,BAEMJk->ABCiJk', H.ab.ovov[Oa, Vb, oa, Vb], R.aab.VVVOOo, optimize=True)
    )
    dR.aab.VVVoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('MCJe,BAeiMk->ABCiJk', H.ab.ovov[Oa, Vb, Oa, vb], R.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('MCJE,BAEiMk->ABCiJk', H.ab.ovov[Oa, Vb, Oa, Vb], R.aab.VVVoOo, optimize=True)
    )

    dR.aab.VVVoOo -= np.transpose(dR.aab.VVVoOo, (1, 0, 2, 3, 4, 5))

    return dR

def update(R, omega, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    R.aab.VVVoOo = eomcc_active_loops.update_r3b_111010(
        R.aab.VVVoOo,
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
