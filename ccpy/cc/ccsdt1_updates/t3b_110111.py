import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import cc_active_loops

#@profile
def build(T, dT, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    # MM(2,3)
    dT.aab.VVvOOO = (2.0 / 4.0) * (
            +1.0 * np.einsum('BceK,AeIJ->ABcIJK', H.ab.vvvo[Va, vb, :, Ob], T.aa[Va, :, Oa, Oa], optimize=True)
    )
    dT.aab.VVvOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('mcJK,ABIm->ABcIJK', H.ab.ovoo[:, vb, Oa, Ob], T.aa[Va, Va, Oa, :], optimize=True)
    )
    dT.aab.VVvOOO += (4.0 / 4.0) * (
            +1.0 * np.einsum('AcIe,BeJK->ABcIJK', H.ab.vvov[Va, vb, Oa, :], T.ab[Va, :, Oa, Ob], optimize=True)
    )
    dT.aab.VVvOOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('AmIK,BcJm->ABcIJK', H.ab.vooo[Va, :, Oa, Ob], T.ab[Va, vb, Oa, :], optimize=True)
    )
    dT.aab.VVvOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('ABIe,ecJK->ABcIJK', H.aa.vvov[Va, Va, Oa, :], T.ab[:, vb, Oa, Ob], optimize=True)
    )
    dT.aab.VVvOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('AmIJ,BcmK->ABcIJK', H.aa.vooo[Va, :, Oa, Oa], T.ab[Va, vb, :, Ob], optimize=True)
    )
    # (H(2) * T3)_C
    dT.aab.VVvOOO += (2.0/4.0) * (
            +1.0*np.einsum('mI,BAcmJK->ABcIJK', H.a.oo[oa, Oa], T.aab.VVvoOO, optimize=True)
            +1.0*np.einsum('MI,BAcMJK->ABcIJK', H.a.oo[Oa, Oa], T.aab.VVvOOO, optimize=True)
    )
    dT.aab.VVvOOO += (1.0/4.0) * (
            +1.0*np.einsum('mK,BAcIJm->ABcIJK', H.b.oo[ob, Ob], T.aab.VVvOOo, optimize=True)
            +1.0*np.einsum('MK,BAcIJM->ABcIJK', H.b.oo[Ob, Ob], T.aab.VVvOOO, optimize=True)
    )
    dT.aab.VVvOOO += (2.0/4.0) * (
            -1.0*np.einsum('Ae,BecIJK->ABcIJK', H.a.vv[Va, va], T.aab.VvvOOO, optimize=True)
            -1.0*np.einsum('AE,BEcIJK->ABcIJK', H.a.vv[Va, Va], T.aab.VVvOOO, optimize=True)
    )
    dT.aab.VVvOOO += (1.0/4.0) * (
            -1.0*np.einsum('ce,BAeIJK->ABcIJK', H.b.vv[vb, vb], T.aab.VVvOOO, optimize=True)
            -1.0*np.einsum('cE,BAEIJK->ABcIJK', H.b.vv[vb, Vb], T.aab.VVVOOO, optimize=True)
    )
    dT.aab.VVvOOO += (1.0/4.0) * (
            -0.5*np.einsum('mnIJ,BAcmnK->ABcIJK', H.aa.oooo[oa, oa, Oa, Oa], T.aab.VVvooO, optimize=True)
            -1.0*np.einsum('mNIJ,BAcmNK->ABcIJK', H.aa.oooo[oa, Oa, Oa, Oa], T.aab.VVvoOO, optimize=True)
            -0.5*np.einsum('MNIJ,BAcMNK->ABcIJK', H.aa.oooo[Oa, Oa, Oa, Oa], T.aab.VVvOOO, optimize=True)
    )
    dT.aab.VVvOOO += (2.0/4.0) * (
            +1.0*np.einsum('mnJK,BAcmIn->ABcIJK', H.ab.oooo[oa, ob, Oa, Ob], T.aab.VVvoOo, optimize=True)
            -1.0*np.einsum('MnJK,BAcIMn->ABcIJK', H.ab.oooo[Oa, ob, Oa, Ob], T.aab.VVvOOo, optimize=True)
            +1.0*np.einsum('mNJK,BAcmIN->ABcIJK', H.ab.oooo[oa, Ob, Oa, Ob], T.aab.VVvoOO, optimize=True)
            -1.0*np.einsum('MNJK,BAcIMN->ABcIJK', H.ab.oooo[Oa, Ob, Oa, Ob], T.aab.VVvOOO, optimize=True)
    )
    dT.aab.VVvOOO += (1.0/4.0) * (
            -1.0*np.einsum('ABeF,FecIJK->ABcIJK', H.aa.vvvv[Va, Va, va, Va], T.aab.VvvOOO, optimize=True)
            -0.5*np.einsum('ABEF,FEcIJK->ABcIJK', H.aa.vvvv[Va, Va, Va, Va], T.aab.VVvOOO, optimize=True)
    )
    dT.aab.VVvOOO += (2.0/4.0) * (
            +1.0*np.einsum('Bcef,AefIJK->ABcIJK', H.ab.vvvv[Va, vb, va, vb], T.aab.VvvOOO, optimize=True)
            -1.0*np.einsum('BcEf,EAfIJK->ABcIJK', H.ab.vvvv[Va, vb, Va, vb], T.aab.VVvOOO, optimize=True)
            +1.0*np.einsum('BceF,AeFIJK->ABcIJK', H.ab.vvvv[Va, vb, va, Vb], T.aab.VvVOOO, optimize=True)
            -1.0*np.einsum('BcEF,EAFIJK->ABcIJK', H.ab.vvvv[Va, vb, Va, Vb], T.aab.VVVOOO, optimize=True)
    )
    dT.aab.VVvOOO += (4.0/4.0) * (
            -1.0*np.einsum('AmIe,BecmJK->ABcIJK', H.aa.voov[Va, oa, Oa, va], T.aab.VvvoOO, optimize=True)
            -1.0*np.einsum('AMIe,BecMJK->ABcIJK', H.aa.voov[Va, Oa, Oa, va], T.aab.VvvOOO, optimize=True)
            -1.0*np.einsum('AmIE,BEcmJK->ABcIJK', H.aa.voov[Va, oa, Oa, Va], T.aab.VVvoOO, optimize=True)
            -1.0*np.einsum('AMIE,BEcMJK->ABcIJK', H.aa.voov[Va, Oa, Oa, Va], T.aab.VVvOOO, optimize=True)
    )
    dT.aab.VVvOOO += (4.0/4.0) * (
            -1.0*np.einsum('AmIe,BceJmK->ABcIJK', H.ab.voov[Va, ob, Oa, vb], T.abb.VvvOoO, optimize=True)
            -1.0*np.einsum('AMIe,BceJMK->ABcIJK', H.ab.voov[Va, Ob, Oa, vb], T.abb.VvvOOO, optimize=True)
            +1.0*np.einsum('AmIE,BEcJmK->ABcIJK', H.ab.voov[Va, ob, Oa, Vb], T.abb.VVvOoO, optimize=True)
            +1.0*np.einsum('AMIE,BEcJMK->ABcIJK', H.ab.voov[Va, Ob, Oa, Vb], T.abb.VVvOOO, optimize=True)
    )
    dT.aab.VVvOOO += (1.0/4.0) * (
            -1.0*np.einsum('mceK,BAemIJ->ABcIJK', H.ab.ovvo[oa, vb, va, Ob], T.aaa.VVvoOO, optimize=True)
            -1.0*np.einsum('MceK,BAeIJM->ABcIJK', H.ab.ovvo[Oa, vb, va, Ob], T.aaa.VVvOOO, optimize=True)
            -1.0*np.einsum('mcEK,EBAmIJ->ABcIJK', H.ab.ovvo[oa, vb, Va, Ob], T.aaa.VVVoOO, optimize=True)
            -1.0*np.einsum('McEK,EBAIJM->ABcIJK', H.ab.ovvo[Oa, vb, Va, Ob], T.aaa.VVVOOO, optimize=True)
    )
    dT.aab.VVvOOO += (1.0/4.0) * (
            -1.0*np.einsum('cmKe,BAeIJm->ABcIJK', H.bb.voov[vb, ob, Ob, vb], T.aab.VVvOOo, optimize=True)
            -1.0*np.einsum('cMKe,BAeIJM->ABcIJK', H.bb.voov[vb, Ob, Ob, vb], T.aab.VVvOOO, optimize=True)
            -1.0*np.einsum('cmKE,BAEIJm->ABcIJK', H.bb.voov[vb, ob, Ob, Vb], T.aab.VVVOOo, optimize=True)
            -1.0*np.einsum('cMKE,BAEIJM->ABcIJK', H.bb.voov[vb, Ob, Ob, Vb], T.aab.VVVOOO, optimize=True)
    )
    dT.aab.VVvOOO += (2.0/4.0) * (
            +1.0*np.einsum('AmeK,BecIJm->ABcIJK', H.ab.vovo[Va, ob, va, Ob], T.aab.VvvOOo, optimize=True)
            +1.0*np.einsum('AMeK,BecIJM->ABcIJK', H.ab.vovo[Va, Ob, va, Ob], T.aab.VvvOOO, optimize=True)
            +1.0*np.einsum('AmEK,BEcIJm->ABcIJK', H.ab.vovo[Va, ob, Va, Ob], T.aab.VVvOOo, optimize=True)
            +1.0*np.einsum('AMEK,BEcIJM->ABcIJK', H.ab.vovo[Va, Ob, Va, Ob], T.aab.VVvOOO, optimize=True)
    )
    dT.aab.VVvOOO += (2.0/4.0) * (
            +1.0*np.einsum('mcIe,BAemJK->ABcIJK', H.ab.ovov[oa, vb, Oa, vb], T.aab.VVvoOO, optimize=True)
            +1.0*np.einsum('McIe,BAeMJK->ABcIJK', H.ab.ovov[Oa, vb, Oa, vb], T.aab.VVvOOO, optimize=True)
            +1.0*np.einsum('mcIE,BAEmJK->ABcIJK', H.ab.ovov[oa, vb, Oa, Vb], T.aab.VVVoOO, optimize=True)
            +1.0*np.einsum('McIE,BAEMJK->ABcIJK', H.ab.ovov[Oa, vb, Oa, Vb], T.aab.VVVOOO, optimize=True)
    )

    dT.aab.VVvOOO -= np.transpose(dT.aab.VVvOOO, (1, 0, 2, 3, 4, 5))
    dT.aab.VVvOOO -= np.transpose(dT.aab.VVvOOO, (0, 1, 2, 4, 3, 5))

    return dT

def update(T, dT, H, shift, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.aab.VVvOOO, dT.aab.VVvOOO = cc_active_loops.update_t3b_110111(
        T.aab.VVvOOO,
        dT.aab.VVvOOO,
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