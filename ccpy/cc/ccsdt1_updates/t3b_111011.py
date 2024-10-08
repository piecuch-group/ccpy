import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import cc_active_loops

#@profile
def build(T, dT, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    # MM(2,3)
    dT.aab.VVVoOO = (2.0 / 2.0) * (
            +1.0 * np.einsum('BCeK,AeiJ->ABCiJK', H.ab.vvvo[Va, Vb, :, Ob], T.aa[Va, :, oa, Oa], optimize=True)
    )
    dT.aab.VVVoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mCJK,ABim->ABCiJK', H.ab.ovoo[:, Vb, Oa, Ob], T.aa[Va, Va, oa, :], optimize=True)
    )
    dT.aab.VVVoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mCiK,ABJm->ABCiJK', H.ab.ovoo[:, Vb, oa, Ob], T.aa[Va, Va, Oa, :], optimize=True)
    )
    dT.aab.VVVoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('ACie,BeJK->ABCiJK', H.ab.vvov[Va, Vb, oa, :], T.ab[Va, :, Oa, Ob], optimize=True)
    )
    dT.aab.VVVoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('ACJe,BeiK->ABCiJK', H.ab.vvov[Va, Vb, Oa, :], T.ab[Va, :, oa, Ob], optimize=True)
    )
    dT.aab.VVVoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('AmiK,BCJm->ABCiJK', H.ab.vooo[Va, :, oa, Ob], T.ab[Va, Vb, Oa, :], optimize=True)
    )
    dT.aab.VVVoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('AmJK,BCim->ABCiJK', H.ab.vooo[Va, :, Oa, Ob], T.ab[Va, Vb, oa, :], optimize=True)
    )
    dT.aab.VVVoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('ABie,eCJK->ABCiJK', H.aa.vvov[Va, Va, oa, :], T.ab[:, Vb, Oa, Ob], optimize=True)
    )
    dT.aab.VVVoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('ABJe,eCiK->ABCiJK', H.aa.vvov[Va, Va, Oa, :], T.ab[:, Vb, oa, Ob], optimize=True)
    )
    dT.aab.VVVoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('AmiJ,BCmK->ABCiJK', H.aa.vooo[Va, :, oa, Oa], T.ab[Va, Vb, :, Ob], optimize=True)
    )
    # (H(2) * T3)_C
    dT.aab.VVVoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mi,BACmJK->ABCiJK', H.a.oo[oa, oa], T.aab.VVVoOO, optimize=True)
            + 1.0 * np.einsum('Mi,BACMJK->ABCiJK', H.a.oo[Oa, oa], T.aab.VVVOOO, optimize=True)
    )
    dT.aab.VVVoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mJ,BACmiK->ABCiJK', H.a.oo[oa, Oa], T.aab.VVVooO, optimize=True)
            + 1.0 * np.einsum('MJ,BACiMK->ABCiJK', H.a.oo[Oa, Oa], T.aab.VVVoOO, optimize=True)
    )
    dT.aab.VVVoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mK,BACiJm->ABCiJK', H.b.oo[ob, Ob], T.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('MK,BACiJM->ABCiJK', H.b.oo[Ob, Ob], T.aab.VVVoOO, optimize=True)
    )
    dT.aab.VVVoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('Ae,BeCiJK->ABCiJK', H.a.vv[Va, va], T.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('AE,BECiJK->ABCiJK', H.a.vv[Va, Va], T.aab.VVVoOO, optimize=True)
    )
    dT.aab.VVVoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('Ce,BAeiJK->ABCiJK', H.b.vv[Vb, vb], T.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('CE,BAEiJK->ABCiJK', H.b.vv[Vb, Vb], T.aab.VVVoOO, optimize=True)
    )
    dT.aab.VVVoOO += (1.0 / 2.0) * (
            -0.5 * np.einsum('mniJ,BACmnK->ABCiJK', H.aa.oooo[oa, oa, oa, Oa], T.aab.VVVooO, optimize=True)
            - 1.0 * np.einsum('mNiJ,BACmNK->ABCiJK', H.aa.oooo[oa, Oa, oa, Oa], T.aab.VVVoOO, optimize=True)
            - 0.5 * np.einsum('MNiJ,BACMNK->ABCiJK', H.aa.oooo[Oa, Oa, oa, Oa], T.aab.VVVOOO, optimize=True)
    )
    dT.aab.VVVoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('MnJK,BACiMn->ABCiJK', H.ab.oooo[Oa, ob, Oa, Ob], T.aab.VVVoOo, optimize=True)
            - 1.0 * np.einsum('mNJK,BACimN->ABCiJK', H.ab.oooo[oa, Ob, Oa, Ob], T.aab.VVVooO, optimize=True)
            - 1.0 * np.einsum('MNJK,BACiMN->ABCiJK', H.ab.oooo[Oa, Ob, Oa, Ob], T.aab.VVVoOO, optimize=True)
    )
    dT.aab.VVVoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mniK,BACmJn->ABCiJK', H.ab.oooo[oa, ob, oa, Ob], T.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('MniK,BACJMn->ABCiJK', H.ab.oooo[Oa, ob, oa, Ob], T.aab.VVVOOo, optimize=True)
            - 1.0 * np.einsum('mNiK,BACmJN->ABCiJK', H.ab.oooo[oa, Ob, oa, Ob], T.aab.VVVoOO, optimize=True)
            + 1.0 * np.einsum('MNiK,BACJMN->ABCiJK', H.ab.oooo[Oa, Ob, oa, Ob], T.aab.VVVOOO, optimize=True)
    )
    dT.aab.VVVoOO += (1.0 / 2.0) * (
            -0.5 * np.einsum('ABef,feCiJK->ABCiJK', H.aa.vvvv[Va, Va, va, va], T.aab.vvVoOO, optimize=True)
            + 1.0 * np.einsum('ABEf,EfCiJK->ABCiJK', H.aa.vvvv[Va, Va, Va, va], T.aab.VvVoOO, optimize=True)
            - 0.5 * np.einsum('ABEF,FECiJK->ABCiJK', H.aa.vvvv[Va, Va, Va, Va], T.aab.VVVoOO, optimize=True)
    )
    dT.aab.VVVoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('BCef,AefiJK->ABCiJK', H.ab.vvvv[Va, Vb, va, vb], T.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('BCeF,AeFiJK->ABCiJK', H.ab.vvvv[Va, Vb, va, Vb], T.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('BCEf,EAfiJK->ABCiJK', H.ab.vvvv[Va, Vb, Va, vb], T.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('BCEF,EAFiJK->ABCiJK', H.ab.vvvv[Va, Vb, Va, Vb], T.aab.VVVoOO, optimize=True)
    )
    dT.aab.VVVoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('Amie,BeCmJK->ABCiJK', H.aa.voov[Va, oa, oa, va], T.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('AMie,BeCMJK->ABCiJK', H.aa.voov[Va, Oa, oa, va], T.aab.VvVOOO, optimize=True)
            - 1.0 * np.einsum('AmiE,BECmJK->ABCiJK', H.aa.voov[Va, oa, oa, Va], T.aab.VVVoOO, optimize=True)
            - 1.0 * np.einsum('AMiE,BECMJK->ABCiJK', H.aa.voov[Va, Oa, oa, Va], T.aab.VVVOOO, optimize=True)
    )
    dT.aab.VVVoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('AmJe,BeCmiK->ABCiJK', H.aa.voov[Va, oa, Oa, va], T.aab.VvVooO, optimize=True)
            - 1.0 * np.einsum('AMJe,BeCiMK->ABCiJK', H.aa.voov[Va, Oa, Oa, va], T.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('AmJE,BECmiK->ABCiJK', H.aa.voov[Va, oa, Oa, Va], T.aab.VVVooO, optimize=True)
            - 1.0 * np.einsum('AMJE,BECiMK->ABCiJK', H.aa.voov[Va, Oa, Oa, Va], T.aab.VVVoOO, optimize=True)
    )
    dT.aab.VVVoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('Amie,BCeJmK->ABCiJK', H.ab.voov[Va, ob, oa, vb], T.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('AMie,BCeJMK->ABCiJK', H.ab.voov[Va, Ob, oa, vb], T.abb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('AmiE,BCEJmK->ABCiJK', H.ab.voov[Va, ob, oa, Vb], T.abb.VVVOoO, optimize=True)
            - 1.0 * np.einsum('AMiE,BCEJMK->ABCiJK', H.ab.voov[Va, Ob, oa, Vb], T.abb.VVVOOO, optimize=True)
    )
    dT.aab.VVVoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('AmJe,BCeimK->ABCiJK', H.ab.voov[Va, ob, Oa, vb], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('AMJe,BCeiMK->ABCiJK', H.ab.voov[Va, Ob, Oa, vb], T.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('AmJE,BCEimK->ABCiJK', H.ab.voov[Va, ob, Oa, Vb], T.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('AMJE,BCEiMK->ABCiJK', H.ab.voov[Va, Ob, Oa, Vb], T.abb.VVVoOO, optimize=True)
    )
    dT.aab.VVVoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mCeK,BAeimJ->ABCiJK', H.ab.ovvo[oa, Vb, va, Ob], T.aaa.VVvooO, optimize=True)
            - 1.0 * np.einsum('MCeK,BAeiJM->ABCiJK', H.ab.ovvo[Oa, Vb, va, Ob], T.aaa.VVvoOO, optimize=True)
            + 1.0 * np.einsum('mCEK,EBAimJ->ABCiJK', H.ab.ovvo[oa, Vb, Va, Ob], T.aaa.VVVooO, optimize=True)
            - 1.0 * np.einsum('MCEK,EBAiJM->ABCiJK', H.ab.ovvo[Oa, Vb, Va, Ob], T.aaa.VVVoOO, optimize=True)
    )
    dT.aab.VVVoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('CmKe,BAeiJm->ABCiJK', H.bb.voov[Vb, ob, Ob, vb], T.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('CMKe,BAeiJM->ABCiJK', H.bb.voov[Vb, Ob, Ob, vb], T.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('CmKE,BAEiJm->ABCiJK', H.bb.voov[Vb, ob, Ob, Vb], T.aab.VVVoOo, optimize=True)
            - 1.0 * np.einsum('CMKE,BAEiJM->ABCiJK', H.bb.voov[Vb, Ob, Ob, Vb], T.aab.VVVoOO, optimize=True)
    )
    dT.aab.VVVoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('AmeK,BeCiJm->ABCiJK', H.ab.vovo[Va, ob, va, Ob], T.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('AMeK,BeCiJM->ABCiJK', H.ab.vovo[Va, Ob, va, Ob], T.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('AmEK,BECiJm->ABCiJK', H.ab.vovo[Va, ob, Va, Ob], T.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('AMEK,BECiJM->ABCiJK', H.ab.vovo[Va, Ob, Va, Ob], T.aab.VVVoOO, optimize=True)
    )
    dT.aab.VVVoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mCie,BAemJK->ABCiJK', H.ab.ovov[oa, Vb, oa, vb], T.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('MCie,BAeMJK->ABCiJK', H.ab.ovov[Oa, Vb, oa, vb], T.aab.VVvOOO, optimize=True)
            + 1.0 * np.einsum('mCiE,BAEmJK->ABCiJK', H.ab.ovov[oa, Vb, oa, Vb], T.aab.VVVoOO, optimize=True)
            + 1.0 * np.einsum('MCiE,BAEMJK->ABCiJK', H.ab.ovov[Oa, Vb, oa, Vb], T.aab.VVVOOO, optimize=True)
    )
    dT.aab.VVVoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mCJe,BAemiK->ABCiJK', H.ab.ovov[oa, Vb, Oa, vb], T.aab.VVvooO, optimize=True)
            + 1.0 * np.einsum('MCJe,BAeiMK->ABCiJK', H.ab.ovov[Oa, Vb, Oa, vb], T.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('mCJE,BAEmiK->ABCiJK', H.ab.ovov[oa, Vb, Oa, Vb], T.aab.VVVooO, optimize=True)
            + 1.0 * np.einsum('MCJE,BAEiMK->ABCiJK', H.ab.ovov[Oa, Vb, Oa, Vb], T.aab.VVVoOO, optimize=True)
    )

    dT.aab.VVVoOO -= np.transpose(dT.aab.VVVoOO, (1, 0, 2, 3, 4, 5))

    return dT

def update(T, dT, H, shift, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.aab.VVVoOO, dT.aab.VVVoOO = cc_active_loops.update_t3b_111011(
        T.aab.VVVoOO,
        dT.aab.VVVoOO,
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