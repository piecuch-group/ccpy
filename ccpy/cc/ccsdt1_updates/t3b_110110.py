import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import cc_active_loops

#@profile
def build(T, dT, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    # MM(2,3)
    dT.aab.VVvOOo = (2.0 / 4.0) * (
            +1.0 * np.einsum('Bcek,AeIJ->ABcIJk', H.ab.vvvo[Va, vb, :, ob], T.aa[Va, :, Oa, Oa], optimize=True)
    )
    dT.aab.VVvOOo += (2.0 / 4.0) * (
            -1.0 * np.einsum('mcJk,ABIm->ABcIJk', H.ab.ovoo[:, vb, Oa, ob], T.aa[Va, Va, Oa, :], optimize=True)
    )
    dT.aab.VVvOOo += (4.0 / 4.0) * (
            +1.0 * np.einsum('AcIe,BeJk->ABcIJk', H.ab.vvov[Va, vb, Oa, :], T.ab[Va, :, Oa, ob], optimize=True)
    )
    dT.aab.VVvOOo += (4.0 / 4.0) * (
            -1.0 * np.einsum('AmIk,BcJm->ABcIJk', H.ab.vooo[Va, :, Oa, ob], T.ab[Va, vb, Oa, :], optimize=True)
    )
    dT.aab.VVvOOo += (2.0 / 4.0) * (
            +1.0 * np.einsum('ABIe,ecJk->ABcIJk', H.aa.vvov[Va, Va, Oa, :], T.ab[:, vb, Oa, ob], optimize=True)
    )
    dT.aab.VVvOOo += (2.0 / 4.0) * (
            -1.0 * np.einsum('AmIJ,Bcmk->ABcIJk', H.aa.vooo[Va, :, Oa, Oa], T.ab[Va, vb, :, ob], optimize=True)
    )
    # (H(2) * T3)_C
    dT.aab.VVvOOo += (2.0 / 4.0) * (
            +1.0 * np.einsum('mI,BAcmJk->ABcIJk', H.a.oo[oa, Oa], T.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('MI,BAcMJk->ABcIJk', H.a.oo[Oa, Oa], T.aab.VVvOOo, optimize=True)
    )
    dT.aab.VVvOOo += (1.0 / 4.0) * (
            +1.0 * np.einsum('mk,BAcIJm->ABcIJk', H.b.oo[ob, ob], T.aab.VVvOOo, optimize=True)
            + 1.0 * np.einsum('Mk,BAcIJM->ABcIJk', H.b.oo[Ob, ob], T.aab.VVvOOO, optimize=True)
    )
    dT.aab.VVvOOo += (2.0 / 4.0) * (
            -1.0 * np.einsum('Ae,BecIJk->ABcIJk', H.a.vv[Va, va], T.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('AE,BEcIJk->ABcIJk', H.a.vv[Va, Va], T.aab.VVvOOo, optimize=True)
    )
    dT.aab.VVvOOo += (1.0 / 4.0) * (
            -1.0 * np.einsum('ce,BAeIJk->ABcIJk', H.b.vv[vb, vb], T.aab.VVvOOo, optimize=True)
            - 1.0 * np.einsum('cE,BAEIJk->ABcIJk', H.b.vv[vb, Vb], T.aab.VVVOOo, optimize=True)
    )
    dT.aab.VVvOOo += (1.0 / 4.0) * (
            +1.0 * np.einsum('MnIJ,BAcnMk->ABcIJk', H.aa.oooo[Oa, oa, Oa, Oa], T.aab.VVvoOo, optimize=True)
            - 0.5 * np.einsum('MNIJ,BAcMNk->ABcIJk', H.aa.oooo[Oa, Oa, Oa, Oa], T.aab.VVvOOo, optimize=True)
    )
    dT.aab.VVvOOo += (2.0 / 4.0) * (
            +1.0 * np.einsum('mnJk,BAcmIn->ABcIJk', H.ab.oooo[oa, ob, Oa, ob], T.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('mNJk,BAcmIN->ABcIJk', H.ab.oooo[oa, Ob, Oa, ob], T.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('MnJk,BAcIMn->ABcIJk', H.ab.oooo[Oa, ob, Oa, ob], T.aab.VVvOOo, optimize=True)
            - 1.0 * np.einsum('MNJk,BAcIMN->ABcIJk', H.ab.oooo[Oa, Ob, Oa, ob], T.aab.VVvOOO, optimize=True)
    )
    dT.aab.VVvOOo += (1.0 / 4.0) * (
            +1.0 * np.einsum('ABEf,EfcIJk->ABcIJk', H.aa.vvvv[Va, Va, Va, va], T.aab.VvvOOo, optimize=True)
            - 0.5 * np.einsum('ABEF,FEcIJk->ABcIJk', H.aa.vvvv[Va, Va, Va, Va], T.aab.VVvOOo, optimize=True)
    )
    dT.aab.VVvOOo += (2.0 / 4.0) * (
            +1.0 * np.einsum('Bcef,AefIJk->ABcIJk', H.ab.vvvv[Va, vb, va, vb], T.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('BceF,AeFIJk->ABcIJk', H.ab.vvvv[Va, vb, va, Vb], T.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('BcEf,EAfIJk->ABcIJk', H.ab.vvvv[Va, vb, Va, vb], T.aab.VVvOOo, optimize=True)
            - 1.0 * np.einsum('BcEF,EAFIJk->ABcIJk', H.ab.vvvv[Va, vb, Va, Vb], T.aab.VVVOOo, optimize=True)
    )
    dT.aab.VVvOOo += (4.0 / 4.0) * (
            -1.0 * np.einsum('AmIe,BecmJk->ABcIJk', H.aa.voov[Va, oa, Oa, va], T.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('AMIe,BecMJk->ABcIJk', H.aa.voov[Va, Oa, Oa, va], T.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('AmIE,BEcmJk->ABcIJk', H.aa.voov[Va, oa, Oa, Va], T.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('AMIE,BEcMJk->ABcIJk', H.aa.voov[Va, Oa, Oa, Va], T.aab.VVvOOo, optimize=True)
    )
    dT.aab.VVvOOo += (4.0 / 4.0) * (
            -1.0 * np.einsum('AmIe,BceJmk->ABcIJk', H.ab.voov[Va, ob, Oa, vb], T.abb.VvvOoo, optimize=True)
            + 1.0 * np.einsum('AMIe,BceJkM->ABcIJk', H.ab.voov[Va, Ob, Oa, vb], T.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('AmIE,BEcJmk->ABcIJk', H.ab.voov[Va, ob, Oa, Vb], T.abb.VVvOoo, optimize=True)
            - 1.0 * np.einsum('AMIE,BEcJkM->ABcIJk', H.ab.voov[Va, Ob, Oa, Vb], T.abb.VVvOoO, optimize=True)
    )
    dT.aab.VVvOOo += (1.0 / 4.0) * (
            -1.0 * np.einsum('mcek,BAemIJ->ABcIJk', H.ab.ovvo[oa, vb, va, ob], T.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('Mcek,BAeIJM->ABcIJk', H.ab.ovvo[Oa, vb, va, ob], T.aaa.VVvOOO, optimize=True)
            - 1.0 * np.einsum('mcEk,EBAmIJ->ABcIJk', H.ab.ovvo[oa, vb, Va, ob], T.aaa.VVVoOO, optimize=True)
            - 1.0 * np.einsum('McEk,EBAIJM->ABcIJk', H.ab.ovvo[Oa, vb, Va, ob], T.aaa.VVVOOO, optimize=True)
    )
    dT.aab.VVvOOo += (1.0 / 4.0) * (
            -1.0 * np.einsum('cmke,BAeIJm->ABcIJk', H.bb.voov[vb, ob, ob, vb], T.aab.VVvOOo, optimize=True)
            - 1.0 * np.einsum('cMke,BAeIJM->ABcIJk', H.bb.voov[vb, Ob, ob, vb], T.aab.VVvOOO, optimize=True)
            - 1.0 * np.einsum('cmkE,BAEIJm->ABcIJk', H.bb.voov[vb, ob, ob, Vb], T.aab.VVVOOo, optimize=True)
            - 1.0 * np.einsum('cMkE,BAEIJM->ABcIJk', H.bb.voov[vb, Ob, ob, Vb], T.aab.VVVOOO, optimize=True)
    )
    dT.aab.VVvOOo += (2.0 / 4.0) * (
            +1.0 * np.einsum('Amek,BecIJm->ABcIJk', H.ab.vovo[Va, ob, va, ob], T.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('AMek,BecIJM->ABcIJk', H.ab.vovo[Va, Ob, va, ob], T.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('AmEk,BEcIJm->ABcIJk', H.ab.vovo[Va, ob, Va, ob], T.aab.VVvOOo, optimize=True)
            + 1.0 * np.einsum('AMEk,BEcIJM->ABcIJk', H.ab.vovo[Va, Ob, Va, ob], T.aab.VVvOOO, optimize=True)
    )
    dT.aab.VVvOOo += (2.0 / 4.0) * (
            +1.0 * np.einsum('mcIe,BAemJk->ABcIJk', H.ab.ovov[oa, vb, Oa, vb], T.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('McIe,BAeMJk->ABcIJk', H.ab.ovov[Oa, vb, Oa, vb], T.aab.VVvOOo, optimize=True)
            + 1.0 * np.einsum('mcIE,BAEmJk->ABcIJk', H.ab.ovov[oa, vb, Oa, Vb], T.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('McIE,BAEMJk->ABcIJk', H.ab.ovov[Oa, vb, Oa, Vb], T.aab.VVVOOo, optimize=True)
    )

    dT.aab.VVvOOo -= np.transpose(dT.aab.VVvOOo, (1, 0, 2, 3, 4, 5))
    dT.aab.VVvOOo -= np.transpose(dT.aab.VVvOOo, (0, 1, 2, 4, 3, 5))

    return dT

def update(T, dT, H, shift, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.aab.VVvOOo, dT.aab.VVvOOo = cc_active_loops.update_t3b_110110(
        T.aab.VVvOOo,
        dT.aab.VVvOOo,
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