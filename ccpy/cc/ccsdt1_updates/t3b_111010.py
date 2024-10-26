import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import cc_active_loops

#@profile
def build(T, dT, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    # MM(2,3)
    dT.aab.VVVoOo = (2.0 / 2.0) * (
            +1.0 * np.einsum('BCek,AeiJ->ABCiJk', H.ab.vvvo[Va, Vb, :, ob], T.aa[Va, :, oa, Oa], optimize=True)
    )
    dT.aab.VVVoOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('mCJk,ABim->ABCiJk', H.ab.ovoo[:, Vb, Oa, ob], T.aa[Va, Va, oa, :], optimize=True)
    )
    dT.aab.VVVoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('mCik,ABJm->ABCiJk', H.ab.ovoo[:, Vb, oa, ob], T.aa[Va, Va, Oa, :], optimize=True)
    )
    dT.aab.VVVoOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('ACie,BeJk->ABCiJk', H.ab.vvov[Va, Vb, oa, :], T.ab[Va, :, Oa, ob], optimize=True)
    )
    dT.aab.VVVoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('ACJe,Beik->ABCiJk', H.ab.vvov[Va, Vb, Oa, :], T.ab[Va, :, oa, ob], optimize=True)
    )
    dT.aab.VVVoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('Amik,BCJm->ABCiJk', H.ab.vooo[Va, :, oa, ob], T.ab[Va, Vb, Oa, :], optimize=True)
    )
    dT.aab.VVVoOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('AmJk,BCim->ABCiJk', H.ab.vooo[Va, :, Oa, ob], T.ab[Va, Vb, oa, :], optimize=True)
    )
    dT.aab.VVVoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('ABie,eCJk->ABCiJk', H.aa.vvov[Va, Va, oa, :], T.ab[:, Vb, Oa, ob], optimize=True)
    )
    dT.aab.VVVoOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('ABJe,eCik->ABCiJk', H.aa.vvov[Va, Va, Oa, :], T.ab[:, Vb, oa, ob], optimize=True)
    )
    dT.aab.VVVoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('AmiJ,BCmk->ABCiJk', H.aa.vooo[Va, :, oa, Oa], T.ab[Va, Vb, :, ob], optimize=True)
    )
    # (H(2) * T3)_C
    dT.aab.VVVoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('mi,BACmJk->ABCiJk', H.a.oo[oa, oa], T.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('Mi,BACMJk->ABCiJk', H.a.oo[Oa, oa], T.aab.VVVOOo, optimize=True)
    )
    dT.aab.VVVoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('MJ,BACiMk->ABCiJk', H.a.oo[Oa, Oa], T.aab.VVVoOo, optimize=True)
    )
    dT.aab.VVVoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('mk,BACiJm->ABCiJk', H.b.oo[ob, ob], T.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('Mk,BACiJM->ABCiJk', H.b.oo[Ob, ob], T.aab.VVVoOO, optimize=True)
    )
    dT.aab.VVVoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('Ae,BeCiJk->ABCiJk', H.a.vv[Va, va], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('AE,BECiJk->ABCiJk', H.a.vv[Va, Va], T.aab.VVVoOo, optimize=True)
    )
    dT.aab.VVVoOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('Ce,BAeiJk->ABCiJk', H.b.vv[Vb, vb], T.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('CE,BAEiJk->ABCiJk', H.b.vv[Vb, Vb], T.aab.VVVoOo, optimize=True)
    )
    dT.aab.VVVoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('MniJ,BACnMk->ABCiJk', H.aa.oooo[Oa, oa, oa, Oa], T.aab.VVVoOo, optimize=True)
            - 0.5 * np.einsum('MNiJ,BACMNk->ABCiJk', H.aa.oooo[Oa, Oa, oa, Oa], T.aab.VVVOOo, optimize=True)
    )
    dT.aab.VVVoOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('mNJk,BACimN->ABCiJk', H.ab.oooo[oa, Ob, Oa, ob], T.aab.VVVooO, optimize=True)
            - 1.0 * np.einsum('MnJk,BACiMn->ABCiJk', H.ab.oooo[Oa, ob, Oa, ob], T.aab.VVVoOo, optimize=True)
            - 1.0 * np.einsum('MNJk,BACiMN->ABCiJk', H.ab.oooo[Oa, Ob, Oa, ob], T.aab.VVVoOO, optimize=True)
    )
    dT.aab.VVVoOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('mnik,BACmJn->ABCiJk', H.ab.oooo[oa, ob, oa, ob], T.aab.VVVoOo, optimize=True)
            - 1.0 * np.einsum('mNik,BACmJN->ABCiJk', H.ab.oooo[oa, Ob, oa, ob], T.aab.VVVoOO, optimize=True)
            + 1.0 * np.einsum('Mnik,BACJMn->ABCiJk', H.ab.oooo[Oa, ob, oa, ob], T.aab.VVVOOo, optimize=True)
            + 1.0 * np.einsum('MNik,BACJMN->ABCiJk', H.ab.oooo[Oa, Ob, oa, ob], T.aab.VVVOOO, optimize=True)
    )
    dT.aab.VVVoOo += (1.0 / 2.0) * (
            -0.5 * np.einsum('ABef,feCiJk->ABCiJk', H.aa.vvvv[Va, Va, va, va], T.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('ABEf,EfCiJk->ABCiJk', H.aa.vvvv[Va, Va, Va, va], T.aab.VvVoOo, optimize=True)
            - 0.5 * np.einsum('ABEF,FECiJk->ABCiJk', H.aa.vvvv[Va, Va, Va, Va], T.aab.VVVoOo, optimize=True)
    )
    dT.aab.VVVoOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('BCef,AefiJk->ABCiJk', H.ab.vvvv[Va, Vb, va, vb], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('BCeF,AeFiJk->ABCiJk', H.ab.vvvv[Va, Vb, va, Vb], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('BCEf,EAfiJk->ABCiJk', H.ab.vvvv[Va, Vb, Va, vb], T.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('BCEF,EAFiJk->ABCiJk', H.ab.vvvv[Va, Vb, Va, Vb], T.aab.VVVoOo, optimize=True)
    )
    dT.aab.VVVoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('Amie,BeCmJk->ABCiJk', H.aa.voov[Va, oa, oa, va], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('AMie,BeCMJk->ABCiJk', H.aa.voov[Va, Oa, oa, va], T.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('AmiE,BECmJk->ABCiJk', H.aa.voov[Va, oa, oa, Va], T.aab.VVVoOo, optimize=True)
            - 1.0 * np.einsum('AMiE,BECMJk->ABCiJk', H.aa.voov[Va, Oa, oa, Va], T.aab.VVVOOo, optimize=True)
    )
    dT.aab.VVVoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('AMJe,BeCiMk->ABCiJk', H.aa.voov[Va, Oa, Oa, va], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('AMJE,BECiMk->ABCiJk', H.aa.voov[Va, Oa, Oa, Va], T.aab.VVVoOo, optimize=True)
    )
    dT.aab.VVVoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('Amie,BCeJmk->ABCiJk', H.ab.voov[Va, ob, oa, vb], T.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('AMie,BCeJkM->ABCiJk', H.ab.voov[Va, Ob, oa, vb], T.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('AmiE,BCEJmk->ABCiJk', H.ab.voov[Va, ob, oa, Vb], T.abb.VVVOoo, optimize=True)
            + 1.0 * np.einsum('AMiE,BCEJkM->ABCiJk', H.ab.voov[Va, Ob, oa, Vb], T.abb.VVVOoO, optimize=True)
    )
    dT.aab.VVVoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('AMJe,BCeikM->ABCiJk', H.ab.voov[Va, Ob, Oa, vb], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('AMJE,BCEikM->ABCiJk', H.ab.voov[Va, Ob, Oa, Vb], T.abb.VVVooO, optimize=True)
    )
    dT.aab.VVVoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('mCek,BAeimJ->ABCiJk', H.ab.ovvo[oa, Vb, va, ob], T.aaa.VVvooO, optimize=True)
            - 1.0 * np.einsum('MCek,BAeiJM->ABCiJk', H.ab.ovvo[Oa, Vb, va, ob], T.aaa.VVvoOO, optimize=True)
            + 1.0 * np.einsum('mCEk,EBAimJ->ABCiJk', H.ab.ovvo[oa, Vb, Va, ob], T.aaa.VVVooO, optimize=True)
            - 1.0 * np.einsum('MCEk,EBAiJM->ABCiJk', H.ab.ovvo[Oa, Vb, Va, ob], T.aaa.VVVoOO, optimize=True)
    )
    dT.aab.VVVoOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('Cmke,BAeiJm->ABCiJk', H.bb.voov[Vb, ob, ob, vb], T.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('CMke,BAeiJM->ABCiJk', H.bb.voov[Vb, Ob, ob, vb], T.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('CmkE,BAEiJm->ABCiJk', H.bb.voov[Vb, ob, ob, Vb], T.aab.VVVoOo, optimize=True)
            - 1.0 * np.einsum('CMkE,BAEiJM->ABCiJk', H.bb.voov[Vb, Ob, ob, Vb], T.aab.VVVoOO, optimize=True)
    )
    dT.aab.VVVoOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('Amek,BeCiJm->ABCiJk', H.ab.vovo[Va, ob, va, ob], T.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('AMek,BeCiJM->ABCiJk', H.ab.vovo[Va, Ob, va, ob], T.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('AmEk,BECiJm->ABCiJk', H.ab.vovo[Va, ob, Va, ob], T.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('AMEk,BECiJM->ABCiJk', H.ab.vovo[Va, Ob, Va, ob], T.aab.VVVoOO, optimize=True)
    )
    dT.aab.VVVoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('mCie,BAemJk->ABCiJk', H.ab.ovov[oa, Vb, oa, vb], T.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('MCie,BAeMJk->ABCiJk', H.ab.ovov[Oa, Vb, oa, vb], T.aab.VVvOOo, optimize=True)
            + 1.0 * np.einsum('mCiE,BAEmJk->ABCiJk', H.ab.ovov[oa, Vb, oa, Vb], T.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('MCiE,BAEMJk->ABCiJk', H.ab.ovov[Oa, Vb, oa, Vb], T.aab.VVVOOo, optimize=True)
    )
    dT.aab.VVVoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('MCJe,BAeiMk->ABCiJk', H.ab.ovov[Oa, Vb, Oa, vb], T.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('MCJE,BAEiMk->ABCiJk', H.ab.ovov[Oa, Vb, Oa, Vb], T.aab.VVVoOo, optimize=True)
    )

    dT.aab.VVVoOo -= np.transpose(dT.aab.VVVoOo, (1, 0, 2, 3, 4, 5))

    return dT

def update(T, dT, H, shift, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.aab.VVVoOo, dT.aab.VVVoOo = cc_active_loops.update_t3b_111010(
        T.aab.VVVoOo,
        dT.aab.VVVoOo,
        H.a.oo[Oa, Oa],
        H.a.vv[Va, Va],
        H.a.oo[oa, oa],
        H.a.vv[va, va],
        H.b.oo[Ob, Ob],
        H.b.vv[Vb, Vb],
        H.b.oo[ob, ob],
        H.b.vv[vb, vb],
        shift,
    )

    return T, dT