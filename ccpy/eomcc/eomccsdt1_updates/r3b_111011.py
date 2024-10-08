import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops

def build(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.aab.VVVoOO = (2.0 / 2.0) * (
            +1.0 * np.einsum('BCeK,AeiJ->ABCiJK', X.ab.vvvo[Va, Vb, :, Ob], T.aa[Va, :, oa, Oa], optimize=True)
    )
    dR.aab.VVVoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mCJK,ABim->ABCiJK', X.ab.ovoo[:, Vb, Oa, Ob], T.aa[Va, Va, oa, :], optimize=True)
    )
    dR.aab.VVVoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mCiK,ABJm->ABCiJK', X.ab.ovoo[:, Vb, oa, Ob], T.aa[Va, Va, Oa, :], optimize=True)
    )
    dR.aab.VVVoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('ACie,BeJK->ABCiJK', X.ab.vvov[Va, Vb, oa, :], T.ab[Va, :, Oa, Ob], optimize=True)
    )
    dR.aab.VVVoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('ACJe,BeiK->ABCiJK', X.ab.vvov[Va, Vb, Oa, :], T.ab[Va, :, oa, Ob], optimize=True)
    )
    dR.aab.VVVoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('AmiK,BCJm->ABCiJK', X.ab.vooo[Va, :, oa, Ob], T.ab[Va, Vb, Oa, :], optimize=True)
    )
    dR.aab.VVVoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('AmJK,BCim->ABCiJK', X.ab.vooo[Va, :, Oa, Ob], T.ab[Va, Vb, oa, :], optimize=True)
    )
    dR.aab.VVVoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('ABie,eCJK->ABCiJK', X.aa.vvov[Va, Va, oa, :], T.ab[:, Vb, Oa, Ob], optimize=True)
    )
    dR.aab.VVVoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('ABJe,eCiK->ABCiJK', X.aa.vvov[Va, Va, Oa, :], T.ab[:, Vb, oa, Ob], optimize=True)
    )
    dR.aab.VVVoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('AmiJ,BCmK->ABCiJK', X.aa.vooo[Va, :, oa, Oa], T.ab[Va, Vb, :, Ob], optimize=True)
    )
    dR.aab.VVVoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('BCeK,AeiJ->ABCiJK', H.ab.vvvo[Va, Vb, :, Ob], R.aa[Va, :, oa, Oa], optimize=True)
    )
    dR.aab.VVVoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mCJK,ABim->ABCiJK', H.ab.ovoo[:, Vb, Oa, Ob], R.aa[Va, Va, oa, :], optimize=True)
    )
    dR.aab.VVVoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mCiK,ABJm->ABCiJK', H.ab.ovoo[:, Vb, oa, Ob], R.aa[Va, Va, Oa, :], optimize=True)
    )
    dR.aab.VVVoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('ACie,BeJK->ABCiJK', H.ab.vvov[Va, Vb, oa, :], R.ab[Va, :, Oa, Ob], optimize=True)
    )
    dR.aab.VVVoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('ACJe,BeiK->ABCiJK', H.ab.vvov[Va, Vb, Oa, :], R.ab[Va, :, oa, Ob], optimize=True)
    )
    dR.aab.VVVoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('AmiK,BCJm->ABCiJK', H.ab.vooo[Va, :, oa, Ob], R.ab[Va, Vb, Oa, :], optimize=True)
    )
    dR.aab.VVVoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('AmJK,BCim->ABCiJK', H.ab.vooo[Va, :, Oa, Ob], R.ab[Va, Vb, oa, :], optimize=True)
    )
    dR.aab.VVVoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('ABie,eCJK->ABCiJK', H.aa.vvov[Va, Va, oa, :], R.ab[:, Vb, Oa, Ob], optimize=True)
    )
    dR.aab.VVVoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('ABJe,eCiK->ABCiJK', H.aa.vvov[Va, Va, Oa, :], R.ab[:, Vb, oa, Ob], optimize=True)
    )
    dR.aab.VVVoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('AmiJ,BCmK->ABCiJK', H.aa.vooo[Va, :, oa, Oa], R.ab[Va, Vb, :, Ob], optimize=True)
    )

    dR.aab.VVVoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mi,BACmJK->ABCiJK', X.a.oo[oa, oa], T.aab.VVVoOO, optimize=True)
            + 1.0 * np.einsum('Mi,BACMJK->ABCiJK', X.a.oo[Oa, oa], T.aab.VVVOOO, optimize=True)
    )
    dR.aab.VVVoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mJ,BACmiK->ABCiJK', X.a.oo[oa, Oa], T.aab.VVVooO, optimize=True)
            + 1.0 * np.einsum('MJ,BACiMK->ABCiJK', X.a.oo[Oa, Oa], T.aab.VVVoOO, optimize=True)
    )
    dR.aab.VVVoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mK,BACiJm->ABCiJK', X.b.oo[ob, Ob], T.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('MK,BACiJM->ABCiJK', X.b.oo[Ob, Ob], T.aab.VVVoOO, optimize=True)
    )
    dR.aab.VVVoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('Ae,BeCiJK->ABCiJK', X.a.vv[Va, va], T.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('AE,BECiJK->ABCiJK', X.a.vv[Va, Va], T.aab.VVVoOO, optimize=True)
    )
    dR.aab.VVVoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('Ce,BAeiJK->ABCiJK', X.b.vv[Vb, vb], T.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('CE,BAEiJK->ABCiJK', X.b.vv[Vb, Vb], T.aab.VVVoOO, optimize=True)
    )
    dR.aab.VVVoOO += (1.0 / 2.0) * (
            -0.5 * np.einsum('mniJ,BACmnK->ABCiJK', X.aa.oooo[oa, oa, oa, Oa], T.aab.VVVooO, optimize=True)
            + 1.0 * np.einsum('MniJ,BACnMK->ABCiJK', X.aa.oooo[Oa, oa, oa, Oa], T.aab.VVVoOO, optimize=True)
            - 0.5 * np.einsum('MNiJ,BACMNK->ABCiJK', X.aa.oooo[Oa, Oa, oa, Oa], T.aab.VVVOOO, optimize=True)
    )
    dR.aab.VVVoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mNJK,BACimN->ABCiJK', X.ab.oooo[oa, Ob, Oa, Ob], T.aab.VVVooO, optimize=True)
            - 1.0 * np.einsum('MnJK,BACiMn->ABCiJK', X.ab.oooo[Oa, ob, Oa, Ob], T.aab.VVVoOo, optimize=True)
            - 1.0 * np.einsum('MNJK,BACiMN->ABCiJK', X.ab.oooo[Oa, Ob, Oa, Ob], T.aab.VVVoOO, optimize=True)
    )
    dR.aab.VVVoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mniK,BACmJn->ABCiJK', X.ab.oooo[oa, ob, oa, Ob], T.aab.VVVoOo, optimize=True)
            - 1.0 * np.einsum('mNiK,BACmJN->ABCiJK', X.ab.oooo[oa, Ob, oa, Ob], T.aab.VVVoOO, optimize=True)
            + 1.0 * np.einsum('MniK,BACJMn->ABCiJK', X.ab.oooo[Oa, ob, oa, Ob], T.aab.VVVOOo, optimize=True)
            + 1.0 * np.einsum('MNiK,BACJMN->ABCiJK', X.ab.oooo[Oa, Ob, oa, Ob], T.aab.VVVOOO, optimize=True)
    )
    dR.aab.VVVoOO += (1.0 / 2.0) * (
            -0.5 * np.einsum('ABef,feCiJK->ABCiJK', X.aa.vvvv[Va, Va, va, va], T.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('ABeF,FeCiJK->ABCiJK', X.aa.vvvv[Va, Va, va, Va], T.aab.VvVoOO, optimize=True)
            - 0.5 * np.einsum('ABEF,FECiJK->ABCiJK', X.aa.vvvv[Va, Va, Va, Va], T.aab.VVVoOO, optimize=True)
    )
    dR.aab.VVVoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('BCef,AefiJK->ABCiJK', X.ab.vvvv[Va, Vb, va, vb], T.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('BCEf,EAfiJK->ABCiJK', X.ab.vvvv[Va, Vb, Va, vb], T.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('BCeF,AeFiJK->ABCiJK', X.ab.vvvv[Va, Vb, va, Vb], T.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('BCEF,EAFiJK->ABCiJK', X.ab.vvvv[Va, Vb, Va, Vb], T.aab.VVVoOO, optimize=True)
    )
    dR.aab.VVVoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('Amie,BeCmJK->ABCiJK', X.aa.voov[Va, oa, oa, va], T.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('AmiE,BECmJK->ABCiJK', X.aa.voov[Va, oa, oa, Va], T.aab.VVVoOO, optimize=True)
            - 1.0 * np.einsum('AMie,BeCMJK->ABCiJK', X.aa.voov[Va, Oa, oa, va], T.aab.VvVOOO, optimize=True)
            - 1.0 * np.einsum('AMiE,BECMJK->ABCiJK', X.aa.voov[Va, Oa, oa, Va], T.aab.VVVOOO, optimize=True)
    )
    dR.aab.VVVoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('AmJe,BeCmiK->ABCiJK', X.aa.voov[Va, oa, Oa, va], T.aab.VvVooO, optimize=True)
            + 1.0 * np.einsum('AmJE,BECmiK->ABCiJK', X.aa.voov[Va, oa, Oa, Va], T.aab.VVVooO, optimize=True)
            - 1.0 * np.einsum('AMJe,BeCiMK->ABCiJK', X.aa.voov[Va, Oa, Oa, va], T.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('AMJE,BECiMK->ABCiJK', X.aa.voov[Va, Oa, Oa, Va], T.aab.VVVoOO, optimize=True)
    )
    dR.aab.VVVoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('Amie,BCeJmK->ABCiJK', X.ab.voov[Va, ob, oa, vb], T.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('AmiE,BCEJmK->ABCiJK', X.ab.voov[Va, ob, oa, Vb], T.abb.VVVOoO, optimize=True)
            - 1.0 * np.einsum('AMie,BCeJMK->ABCiJK', X.ab.voov[Va, Ob, oa, vb], T.abb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('AMiE,BCEJMK->ABCiJK', X.ab.voov[Va, Ob, oa, Vb], T.abb.VVVOOO, optimize=True)
    )
    dR.aab.VVVoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('AmJe,BCeimK->ABCiJK', X.ab.voov[Va, ob, Oa, vb], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('AmJE,BCEimK->ABCiJK', X.ab.voov[Va, ob, Oa, Vb], T.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('AMJe,BCeiMK->ABCiJK', X.ab.voov[Va, Ob, Oa, vb], T.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('AMJE,BCEiMK->ABCiJK', X.ab.voov[Va, Ob, Oa, Vb], T.abb.VVVoOO, optimize=True)
    )
    dR.aab.VVVoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mCeK,BAeimJ->ABCiJK', X.ab.ovvo[oa, Vb, va, Ob], T.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('mCEK,EBAimJ->ABCiJK', X.ab.ovvo[oa, Vb, Va, Ob], T.aaa.VVVooO, optimize=True)
            - 1.0 * np.einsum('MCeK,BAeiJM->ABCiJK', X.ab.ovvo[Oa, Vb, va, Ob], T.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('MCEK,EBAiJM->ABCiJK', X.ab.ovvo[Oa, Vb, Va, Ob], T.aaa.VVVoOO, optimize=True)
    )
    dR.aab.VVVoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('CmKe,BAeiJm->ABCiJK', X.bb.voov[Vb, ob, Ob, vb], T.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('CmKE,BAEiJm->ABCiJK', X.bb.voov[Vb, ob, Ob, Vb], T.aab.VVVoOo, optimize=True)
            - 1.0 * np.einsum('CMKe,BAeiJM->ABCiJK', X.bb.voov[Vb, Ob, Ob, vb], T.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('CMKE,BAEiJM->ABCiJK', X.bb.voov[Vb, Ob, Ob, Vb], T.aab.VVVoOO, optimize=True)
    )
    dR.aab.VVVoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('AmeK,BeCiJm->ABCiJK', X.ab.vovo[Va, ob, va, Ob], T.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('AmEK,BECiJm->ABCiJK', X.ab.vovo[Va, ob, Va, Ob], T.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('AMeK,BeCiJM->ABCiJK', X.ab.vovo[Va, Ob, va, Ob], T.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('AMEK,BECiJM->ABCiJK', X.ab.vovo[Va, Ob, Va, Ob], T.aab.VVVoOO, optimize=True)
    )
    dR.aab.VVVoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mCie,BAemJK->ABCiJK', X.ab.ovov[oa, Vb, oa, vb], T.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('mCiE,BAEmJK->ABCiJK', X.ab.ovov[oa, Vb, oa, Vb], T.aab.VVVoOO, optimize=True)
            + 1.0 * np.einsum('MCie,BAeMJK->ABCiJK', X.ab.ovov[Oa, Vb, oa, vb], T.aab.VVvOOO, optimize=True)
            + 1.0 * np.einsum('MCiE,BAEMJK->ABCiJK', X.ab.ovov[Oa, Vb, oa, Vb], T.aab.VVVOOO, optimize=True)
    )
    dR.aab.VVVoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mCJe,BAemiK->ABCiJK', X.ab.ovov[oa, Vb, Oa, vb], T.aab.VVvooO, optimize=True)
            - 1.0 * np.einsum('mCJE,BAEmiK->ABCiJK', X.ab.ovov[oa, Vb, Oa, Vb], T.aab.VVVooO, optimize=True)
            + 1.0 * np.einsum('MCJe,BAeiMK->ABCiJK', X.ab.ovov[Oa, Vb, Oa, vb], T.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('MCJE,BAEiMK->ABCiJK', X.ab.ovov[Oa, Vb, Oa, Vb], T.aab.VVVoOO, optimize=True)
    )
    dR.aab.VVVoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mi,BACmJK->ABCiJK', H.a.oo[oa, oa], R.aab.VVVoOO, optimize=True)
            + 1.0 * np.einsum('Mi,BACMJK->ABCiJK', H.a.oo[Oa, oa], R.aab.VVVOOO, optimize=True)
    )
    dR.aab.VVVoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mJ,BACmiK->ABCiJK', H.a.oo[oa, Oa], R.aab.VVVooO, optimize=True)
            + 1.0 * np.einsum('MJ,BACiMK->ABCiJK', H.a.oo[Oa, Oa], R.aab.VVVoOO, optimize=True)
    )
    dR.aab.VVVoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mK,BACiJm->ABCiJK', H.b.oo[ob, Ob], R.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('MK,BACiJM->ABCiJK', H.b.oo[Ob, Ob], R.aab.VVVoOO, optimize=True)
    )
    dR.aab.VVVoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('Ae,BeCiJK->ABCiJK', H.a.vv[Va, va], R.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('AE,BECiJK->ABCiJK', H.a.vv[Va, Va], R.aab.VVVoOO, optimize=True)
    )
    dR.aab.VVVoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('Ce,BAeiJK->ABCiJK', H.b.vv[Vb, vb], R.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('CE,BAEiJK->ABCiJK', H.b.vv[Vb, Vb], R.aab.VVVoOO, optimize=True)
    )
    dR.aab.VVVoOO += (1.0 / 2.0) * (
            -0.5 * np.einsum('mniJ,BACmnK->ABCiJK', H.aa.oooo[oa, oa, oa, Oa], R.aab.VVVooO, optimize=True)
            + 1.0 * np.einsum('MniJ,BACnMK->ABCiJK', H.aa.oooo[Oa, oa, oa, Oa], R.aab.VVVoOO, optimize=True)
            - 0.5 * np.einsum('MNiJ,BACMNK->ABCiJK', H.aa.oooo[Oa, Oa, oa, Oa], R.aab.VVVOOO, optimize=True)
    )
    dR.aab.VVVoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mNJK,BACimN->ABCiJK', H.ab.oooo[oa, Ob, Oa, Ob], R.aab.VVVooO, optimize=True)
            - 1.0 * np.einsum('MnJK,BACiMn->ABCiJK', H.ab.oooo[Oa, ob, Oa, Ob], R.aab.VVVoOo, optimize=True)
            - 1.0 * np.einsum('MNJK,BACiMN->ABCiJK', H.ab.oooo[Oa, Ob, Oa, Ob], R.aab.VVVoOO, optimize=True)
    )
    dR.aab.VVVoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mniK,BACmJn->ABCiJK', H.ab.oooo[oa, ob, oa, Ob], R.aab.VVVoOo, optimize=True)
            - 1.0 * np.einsum('mNiK,BACmJN->ABCiJK', H.ab.oooo[oa, Ob, oa, Ob], R.aab.VVVoOO, optimize=True)
            + 1.0 * np.einsum('MniK,BACJMn->ABCiJK', H.ab.oooo[Oa, ob, oa, Ob], R.aab.VVVOOo, optimize=True)
            + 1.0 * np.einsum('MNiK,BACJMN->ABCiJK', H.ab.oooo[Oa, Ob, oa, Ob], R.aab.VVVOOO, optimize=True)
    )
    dR.aab.VVVoOO += (1.0 / 2.0) * (
            -0.5 * np.einsum('ABef,feCiJK->ABCiJK', H.aa.vvvv[Va, Va, va, va], R.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('ABeF,FeCiJK->ABCiJK', H.aa.vvvv[Va, Va, va, Va], R.aab.VvVoOO, optimize=True)
            - 0.5 * np.einsum('ABEF,FECiJK->ABCiJK', H.aa.vvvv[Va, Va, Va, Va], R.aab.VVVoOO, optimize=True)
    )
    dR.aab.VVVoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('BCef,AefiJK->ABCiJK', H.ab.vvvv[Va, Vb, va, vb], R.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('BCEf,EAfiJK->ABCiJK', H.ab.vvvv[Va, Vb, Va, vb], R.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('BCeF,AeFiJK->ABCiJK', H.ab.vvvv[Va, Vb, va, Vb], R.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('BCEF,EAFiJK->ABCiJK', H.ab.vvvv[Va, Vb, Va, Vb], R.aab.VVVoOO, optimize=True)
    )
    dR.aab.VVVoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('Amie,BeCmJK->ABCiJK', H.aa.voov[Va, oa, oa, va], R.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('AmiE,BECmJK->ABCiJK', H.aa.voov[Va, oa, oa, Va], R.aab.VVVoOO, optimize=True)
            - 1.0 * np.einsum('AMie,BeCMJK->ABCiJK', H.aa.voov[Va, Oa, oa, va], R.aab.VvVOOO, optimize=True)
            - 1.0 * np.einsum('AMiE,BECMJK->ABCiJK', H.aa.voov[Va, Oa, oa, Va], R.aab.VVVOOO, optimize=True)
    )
    dR.aab.VVVoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('AmJe,BeCmiK->ABCiJK', H.aa.voov[Va, oa, Oa, va], R.aab.VvVooO, optimize=True)
            + 1.0 * np.einsum('AmJE,BECmiK->ABCiJK', H.aa.voov[Va, oa, Oa, Va], R.aab.VVVooO, optimize=True)
            - 1.0 * np.einsum('AMJe,BeCiMK->ABCiJK', H.aa.voov[Va, Oa, Oa, va], R.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('AMJE,BECiMK->ABCiJK', H.aa.voov[Va, Oa, Oa, Va], R.aab.VVVoOO, optimize=True)
    )
    dR.aab.VVVoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('Amie,BCeJmK->ABCiJK', H.ab.voov[Va, ob, oa, vb], R.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('AmiE,BCEJmK->ABCiJK', H.ab.voov[Va, ob, oa, Vb], R.abb.VVVOoO, optimize=True)
            - 1.0 * np.einsum('AMie,BCeJMK->ABCiJK', H.ab.voov[Va, Ob, oa, vb], R.abb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('AMiE,BCEJMK->ABCiJK', H.ab.voov[Va, Ob, oa, Vb], R.abb.VVVOOO, optimize=True)
    )
    dR.aab.VVVoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('AmJe,BCeimK->ABCiJK', H.ab.voov[Va, ob, Oa, vb], R.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('AmJE,BCEimK->ABCiJK', H.ab.voov[Va, ob, Oa, Vb], R.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('AMJe,BCeiMK->ABCiJK', H.ab.voov[Va, Ob, Oa, vb], R.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('AMJE,BCEiMK->ABCiJK', H.ab.voov[Va, Ob, Oa, Vb], R.abb.VVVoOO, optimize=True)
    )
    dR.aab.VVVoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mCeK,BAeimJ->ABCiJK', H.ab.ovvo[oa, Vb, va, Ob], R.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('mCEK,EBAimJ->ABCiJK', H.ab.ovvo[oa, Vb, Va, Ob], R.aaa.VVVooO, optimize=True)
            - 1.0 * np.einsum('MCeK,BAeiJM->ABCiJK', H.ab.ovvo[Oa, Vb, va, Ob], R.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('MCEK,EBAiJM->ABCiJK', H.ab.ovvo[Oa, Vb, Va, Ob], R.aaa.VVVoOO, optimize=True)
    )
    dR.aab.VVVoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('CmKe,BAeiJm->ABCiJK', H.bb.voov[Vb, ob, Ob, vb], R.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('CmKE,BAEiJm->ABCiJK', H.bb.voov[Vb, ob, Ob, Vb], R.aab.VVVoOo, optimize=True)
            - 1.0 * np.einsum('CMKe,BAeiJM->ABCiJK', H.bb.voov[Vb, Ob, Ob, vb], R.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('CMKE,BAEiJM->ABCiJK', H.bb.voov[Vb, Ob, Ob, Vb], R.aab.VVVoOO, optimize=True)
    )
    dR.aab.VVVoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('AmeK,BeCiJm->ABCiJK', H.ab.vovo[Va, ob, va, Ob], R.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('AmEK,BECiJm->ABCiJK', H.ab.vovo[Va, ob, Va, Ob], R.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('AMeK,BeCiJM->ABCiJK', H.ab.vovo[Va, Ob, va, Ob], R.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('AMEK,BECiJM->ABCiJK', H.ab.vovo[Va, Ob, Va, Ob], R.aab.VVVoOO, optimize=True)
    )
    dR.aab.VVVoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mCie,BAemJK->ABCiJK', H.ab.ovov[oa, Vb, oa, vb], R.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('mCiE,BAEmJK->ABCiJK', H.ab.ovov[oa, Vb, oa, Vb], R.aab.VVVoOO, optimize=True)
            + 1.0 * np.einsum('MCie,BAeMJK->ABCiJK', H.ab.ovov[Oa, Vb, oa, vb], R.aab.VVvOOO, optimize=True)
            + 1.0 * np.einsum('MCiE,BAEMJK->ABCiJK', H.ab.ovov[Oa, Vb, oa, Vb], R.aab.VVVOOO, optimize=True)
    )
    dR.aab.VVVoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mCJe,BAemiK->ABCiJK', H.ab.ovov[oa, Vb, Oa, vb], R.aab.VVvooO, optimize=True)
            - 1.0 * np.einsum('mCJE,BAEmiK->ABCiJK', H.ab.ovov[oa, Vb, Oa, Vb], R.aab.VVVooO, optimize=True)
            + 1.0 * np.einsum('MCJe,BAeiMK->ABCiJK', H.ab.ovov[Oa, Vb, Oa, vb], R.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('MCJE,BAEiMK->ABCiJK', H.ab.ovov[Oa, Vb, Oa, Vb], R.aab.VVVoOO, optimize=True)
    )

    dR.aab.VVVoOO -= np.transpose(dR.aab.VVVoOO, (1, 0, 2, 3, 4, 5))

    return dR

def update(R, omega, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    R.aab.VVVoOO = eomcc_active_loops.update_r3b_111011(
        R.aab.VVVoOO,
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
