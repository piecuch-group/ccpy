import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import cc_active_loops

#@profile
def build(T, dT, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    # MM(2,3)
    dT.aab.VVvoOO = (2.0 / 2.0) * (
            +1.0 * np.einsum('BceK,AeiJ->ABciJK', H.ab.vvvo[Va, vb, :, Ob], T.aa[Va, :, oa, Oa], optimize=True)
    )
    dT.aab.VVvoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mcJK,ABim->ABciJK', H.ab.ovoo[:, vb, Oa, Ob], T.aa[Va, Va, oa, :], optimize=True)
    )
    dT.aab.VVvoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mciK,ABJm->ABciJK', H.ab.ovoo[:, vb, oa, Ob], T.aa[Va, Va, Oa, :], optimize=True)
    )
    dT.aab.VVvoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('Acie,BeJK->ABciJK', H.ab.vvov[Va, vb, oa, :], T.ab[Va, :, Oa, Ob], optimize=True)
    )
    dT.aab.VVvoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('AcJe,BeiK->ABciJK', H.ab.vvov[Va, vb, Oa, :], T.ab[Va, :, oa, Ob], optimize=True)
    )
    dT.aab.VVvoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('AmiK,BcJm->ABciJK', H.ab.vooo[Va, :, oa, Ob], T.ab[Va, vb, Oa, :], optimize=True)
    )
    dT.aab.VVvoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('AmJK,Bcim->ABciJK', H.ab.vooo[Va, :, Oa, Ob], T.ab[Va, vb, oa, :], optimize=True)
    )
    dT.aab.VVvoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('ABie,ecJK->ABciJK', H.aa.vvov[Va, Va, oa, :], T.ab[:, vb, Oa, Ob], optimize=True)
    )
    dT.aab.VVvoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('ABJe,eciK->ABciJK', H.aa.vvov[Va, Va, Oa, :], T.ab[:, vb, oa, Ob], optimize=True)
    )
    dT.aab.VVvoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('AmiJ,BcmK->ABciJK', H.aa.vooo[Va, :, oa, Oa], T.ab[Va, vb, :, Ob], optimize=True)
    )
    # (H(2) * T3)_C
    dT.aab.VVvoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mi,BAcmJK->ABciJK', H.a.oo[oa, oa], T.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('Mi,BAcMJK->ABciJK', H.a.oo[Oa, oa], T.aab.VVvOOO, optimize=True)
    )
    dT.aab.VVvoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mJ,BAcmiK->ABciJK', H.a.oo[oa, Oa], T.aab.VVvooO, optimize=True)
            + 1.0 * np.einsum('MJ,BAciMK->ABciJK', H.a.oo[Oa, Oa], T.aab.VVvoOO, optimize=True)
    )
    dT.aab.VVvoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mK,BAciJm->ABciJK', H.b.oo[ob, Ob], T.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('MK,BAciJM->ABciJK', H.b.oo[Ob, Ob], T.aab.VVvoOO, optimize=True)
    )
    dT.aab.VVvoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('Ae,BeciJK->ABciJK', H.a.vv[Va, va], T.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('AE,BEciJK->ABciJK', H.a.vv[Va, Va], T.aab.VVvoOO, optimize=True)
    )
    dT.aab.VVvoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('ce,BAeiJK->ABciJK', H.b.vv[vb, vb], T.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('cE,BAEiJK->ABciJK', H.b.vv[vb, Vb], T.aab.VVVoOO, optimize=True)
    )
    dT.aab.VVvoOO += (1.0 / 2.0) * (
            -0.5 * np.einsum('mniJ,BAcmnK->ABciJK', H.aa.oooo[oa, oa, oa, Oa], T.aab.VVvooO, optimize=True)
            + 1.0 * np.einsum('MniJ,BAcnMK->ABciJK', H.aa.oooo[Oa, oa, oa, Oa], T.aab.VVvoOO, optimize=True)
            - 0.5 * np.einsum('MNiJ,BAcMNK->ABciJK', H.aa.oooo[Oa, Oa, oa, Oa], T.aab.VVvOOO, optimize=True)
    )
    dT.aab.VVvoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mNJK,BAcimN->ABciJK', H.ab.oooo[oa, Ob, Oa, Ob], T.aab.VVvooO, optimize=True)
            - 1.0 * np.einsum('MnJK,BAciMn->ABciJK', H.ab.oooo[Oa, ob, Oa, Ob], T.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('MNJK,BAciMN->ABciJK', H.ab.oooo[Oa, Ob, Oa, Ob], T.aab.VVvoOO, optimize=True)
    )
    dT.aab.VVvoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mniK,BAcmJn->ABciJK', H.ab.oooo[oa, ob, oa, Ob], T.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('mNiK,BAcmJN->ABciJK', H.ab.oooo[oa, Ob, oa, Ob], T.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('MniK,BAcJMn->ABciJK', H.ab.oooo[Oa, ob, oa, Ob], T.aab.VVvOOo, optimize=True)
            + 1.0 * np.einsum('MNiK,BAcJMN->ABciJK', H.ab.oooo[Oa, Ob, oa, Ob], T.aab.VVvOOO, optimize=True)
    )
    dT.aab.VVvoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('ABeF,FeciJK->ABciJK', H.aa.vvvv[Va, Va, va, Va], T.aab.VvvoOO, optimize=True)
            - 0.5 * np.einsum('ABEF,FEciJK->ABciJK', H.aa.vvvv[Va, Va, Va, Va], T.aab.VVvoOO, optimize=True)
    )
    dT.aab.VVvoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('Bcef,AefiJK->ABciJK', H.ab.vvvv[Va, vb, va, vb], T.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('BcEf,EAfiJK->ABciJK', H.ab.vvvv[Va, vb, Va, vb], T.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('BceF,AeFiJK->ABciJK', H.ab.vvvv[Va, vb, va, Vb], T.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('BcEF,EAFiJK->ABciJK', H.ab.vvvv[Va, vb, Va, Vb], T.aab.VVVoOO, optimize=True)
    )
    dT.aab.VVvoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('Amie,BecmJK->ABciJK', H.aa.voov[Va, oa, oa, va], T.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('AmiE,BEcmJK->ABciJK', H.aa.voov[Va, oa, oa, Va], T.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('AMie,BecMJK->ABciJK', H.aa.voov[Va, Oa, oa, va], T.aab.VvvOOO, optimize=True)
            - 1.0 * np.einsum('AMiE,BEcMJK->ABciJK', H.aa.voov[Va, Oa, oa, Va], T.aab.VVvOOO, optimize=True)
    )
    dT.aab.VVvoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('AmJe,BecmiK->ABciJK', H.aa.voov[Va, oa, Oa, va], T.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('AmJE,BEcmiK->ABciJK', H.aa.voov[Va, oa, Oa, Va], T.aab.VVvooO, optimize=True)
            - 1.0 * np.einsum('AMJe,BeciMK->ABciJK', H.aa.voov[Va, Oa, Oa, va], T.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('AMJE,BEciMK->ABciJK', H.aa.voov[Va, Oa, Oa, Va], T.aab.VVvoOO, optimize=True)
    )
    dT.aab.VVvoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('Amie,BceJmK->ABciJK', H.ab.voov[Va, ob, oa, vb], T.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('AmiE,BEcJmK->ABciJK', H.ab.voov[Va, ob, oa, Vb], T.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('AMie,BceJMK->ABciJK', H.ab.voov[Va, Ob, oa, vb], T.abb.VvvOOO, optimize=True)
            + 1.0 * np.einsum('AMiE,BEcJMK->ABciJK', H.ab.voov[Va, Ob, oa, Vb], T.abb.VVvOOO, optimize=True)
    )
    dT.aab.VVvoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('AmJe,BceimK->ABciJK', H.ab.voov[Va, ob, Oa, vb], T.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('AmJE,BEcimK->ABciJK', H.ab.voov[Va, ob, Oa, Vb], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('AMJe,BceiMK->ABciJK', H.ab.voov[Va, Ob, Oa, vb], T.abb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('AMJE,BEciMK->ABciJK', H.ab.voov[Va, Ob, Oa, Vb], T.abb.VVvoOO, optimize=True)
    )
    dT.aab.VVvoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mceK,BAeimJ->ABciJK', H.ab.ovvo[oa, vb, va, Ob], T.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('mcEK,EBAimJ->ABciJK', H.ab.ovvo[oa, vb, Va, Ob], T.aaa.VVVooO, optimize=True)
            - 1.0 * np.einsum('MceK,BAeiJM->ABciJK', H.ab.ovvo[Oa, vb, va, Ob], T.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('McEK,EBAiJM->ABciJK', H.ab.ovvo[Oa, vb, Va, Ob], T.aaa.VVVoOO, optimize=True)
    )
    dT.aab.VVvoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('cmKe,BAeiJm->ABciJK', H.bb.voov[vb, ob, Ob, vb], T.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('cmKE,BAEiJm->ABciJK', H.bb.voov[vb, ob, Ob, Vb], T.aab.VVVoOo, optimize=True)
            - 1.0 * np.einsum('cMKe,BAeiJM->ABciJK', H.bb.voov[vb, Ob, Ob, vb], T.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('cMKE,BAEiJM->ABciJK', H.bb.voov[vb, Ob, Ob, Vb], T.aab.VVVoOO, optimize=True)
    )
    dT.aab.VVvoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('AmeK,BeciJm->ABciJK', H.ab.vovo[Va, ob, va, Ob], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('AmEK,BEciJm->ABciJK', H.ab.vovo[Va, ob, Va, Ob], T.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('AMeK,BeciJM->ABciJK', H.ab.vovo[Va, Ob, va, Ob], T.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('AMEK,BEciJM->ABciJK', H.ab.vovo[Va, Ob, Va, Ob], T.aab.VVvoOO, optimize=True)
    )
    dT.aab.VVvoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mcie,BAemJK->ABciJK', H.ab.ovov[oa, vb, oa, vb], T.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('mciE,BAEmJK->ABciJK', H.ab.ovov[oa, vb, oa, Vb], T.aab.VVVoOO, optimize=True)
            + 1.0 * np.einsum('Mcie,BAeMJK->ABciJK', H.ab.ovov[Oa, vb, oa, vb], T.aab.VVvOOO, optimize=True)
            + 1.0 * np.einsum('MciE,BAEMJK->ABciJK', H.ab.ovov[Oa, vb, oa, Vb], T.aab.VVVOOO, optimize=True)
    )
    dT.aab.VVvoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mcJe,BAemiK->ABciJK', H.ab.ovov[oa, vb, Oa, vb], T.aab.VVvooO, optimize=True)
            - 1.0 * np.einsum('mcJE,BAEmiK->ABciJK', H.ab.ovov[oa, vb, Oa, Vb], T.aab.VVVooO, optimize=True)
            + 1.0 * np.einsum('McJe,BAeiMK->ABciJK', H.ab.ovov[Oa, vb, Oa, vb], T.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('McJE,BAEiMK->ABciJK', H.ab.ovov[Oa, vb, Oa, Vb], T.aab.VVVoOO, optimize=True)
    )

    dT.aab.VVvoOO -= np.transpose(dT.aab.VVvoOO, (1, 0, 2, 3, 4, 5))

    return dT

def update(T, dT, H, shift, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.aab.VVvoOO, dT.aab.VVvoOO = cc_active_loops.update_t3b_110011(
        T.aab.VVvoOO,
        dT.aab.VVvoOO,
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