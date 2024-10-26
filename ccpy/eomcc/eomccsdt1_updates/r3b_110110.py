import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops

def build(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.aab.VVvOOo = (2.0 / 4.0) * (
            +1.0 * np.einsum('Bcek,AeIJ->ABcIJk', X.ab.vvvo[Va, vb, :, ob], T.aa[Va, :, Oa, Oa], optimize=True)
    )
    dR.aab.VVvOOo += (2.0 / 4.0) * (
            -1.0 * np.einsum('mcJk,ABIm->ABcIJk', X.ab.ovoo[:, vb, Oa, ob], T.aa[Va, Va, Oa, :], optimize=True)
    )
    dR.aab.VVvOOo += (4.0 / 4.0) * (
            +1.0 * np.einsum('AcIe,BeJk->ABcIJk', X.ab.vvov[Va, vb, Oa, :], T.ab[Va, :, Oa, ob], optimize=True)
    )
    dR.aab.VVvOOo += (4.0 / 4.0) * (
            -1.0 * np.einsum('AmIk,BcJm->ABcIJk', X.ab.vooo[Va, :, Oa, ob], T.ab[Va, vb, Oa, :], optimize=True)
    )
    dR.aab.VVvOOo += (2.0 / 4.0) * (
            +1.0 * np.einsum('ABIe,ecJk->ABcIJk', X.aa.vvov[Va, Va, Oa, :], T.ab[:, vb, Oa, ob], optimize=True)
    )
    dR.aab.VVvOOo += (2.0 / 4.0) * (
            -1.0 * np.einsum('AmIJ,Bcmk->ABcIJk', X.aa.vooo[Va, :, Oa, Oa], T.ab[Va, vb, :, ob], optimize=True)
    )
    dR.aab.VVvOOo += (2.0 / 4.0) * (
            +1.0 * np.einsum('Bcek,AeIJ->ABcIJk', H.ab.vvvo[Va, vb, :, ob], R.aa[Va, :, Oa, Oa], optimize=True)
    )
    dR.aab.VVvOOo += (2.0 / 4.0) * (
            -1.0 * np.einsum('mcJk,ABIm->ABcIJk', H.ab.ovoo[:, vb, Oa, ob], R.aa[Va, Va, Oa, :], optimize=True)
    )
    dR.aab.VVvOOo += (4.0 / 4.0) * (
            +1.0 * np.einsum('AcIe,BeJk->ABcIJk', H.ab.vvov[Va, vb, Oa, :], R.ab[Va, :, Oa, ob], optimize=True)
    )
    dR.aab.VVvOOo += (4.0 / 4.0) * (
            -1.0 * np.einsum('AmIk,BcJm->ABcIJk', H.ab.vooo[Va, :, Oa, ob], R.ab[Va, vb, Oa, :], optimize=True)
    )
    dR.aab.VVvOOo += (2.0 / 4.0) * (
            +1.0 * np.einsum('ABIe,ecJk->ABcIJk', H.aa.vvov[Va, Va, Oa, :], R.ab[:, vb, Oa, ob], optimize=True)
    )
    dR.aab.VVvOOo += (2.0 / 4.0) * (
            -1.0 * np.einsum('AmIJ,Bcmk->ABcIJk', H.aa.vooo[Va, :, Oa, Oa], R.ab[Va, vb, :, ob], optimize=True)
    )

    dR.aab.VVvOOo += (2.0 / 4.0) * (
            +1.0 * np.einsum('mI,BAcmJk->ABcIJk', X.a.oo[oa, Oa], T.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('MI,BAcMJk->ABcIJk', X.a.oo[Oa, Oa], T.aab.VVvOOo, optimize=True)
    )
    dR.aab.VVvOOo += (1.0 / 4.0) * (
            +1.0 * np.einsum('mk,BAcIJm->ABcIJk', X.b.oo[ob, ob], T.aab.VVvOOo, optimize=True)
            + 1.0 * np.einsum('Mk,BAcIJM->ABcIJk', X.b.oo[Ob, ob], T.aab.VVvOOO, optimize=True)
    )
    dR.aab.VVvOOo += (2.0 / 4.0) * (
            -1.0 * np.einsum('Ae,BecIJk->ABcIJk', X.a.vv[Va, va], T.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('AE,BEcIJk->ABcIJk', X.a.vv[Va, Va], T.aab.VVvOOo, optimize=True)
    )
    dR.aab.VVvOOo += (1.0 / 4.0) * (
            -1.0 * np.einsum('ce,BAeIJk->ABcIJk', X.b.vv[vb, vb], T.aab.VVvOOo, optimize=True)
            - 1.0 * np.einsum('cE,BAEIJk->ABcIJk', X.b.vv[vb, Vb], T.aab.VVVOOo, optimize=True)
    )
    dR.aab.VVvOOo += (1.0 / 4.0) * (
            -1.0 * np.einsum('mNIJ,BAcmNk->ABcIJk', X.aa.oooo[oa, Oa, Oa, Oa], T.aab.VVvoOo, optimize=True)
            - 0.5 * np.einsum('MNIJ,BAcMNk->ABcIJk', X.aa.oooo[Oa, Oa, Oa, Oa], T.aab.VVvOOo, optimize=True)
    )
    dR.aab.VVvOOo += (2.0 / 4.0) * (
            +1.0 * np.einsum('mnJk,BAcmIn->ABcIJk', X.ab.oooo[oa, ob, Oa, ob], T.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('MnJk,BAcIMn->ABcIJk', X.ab.oooo[Oa, ob, Oa, ob], T.aab.VVvOOo, optimize=True)
            + 1.0 * np.einsum('mNJk,BAcmIN->ABcIJk', X.ab.oooo[oa, Ob, Oa, ob], T.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('MNJk,BAcIMN->ABcIJk', X.ab.oooo[Oa, Ob, Oa, ob], T.aab.VVvOOO, optimize=True)
    )
    dR.aab.VVvOOo += (1.0 / 4.0) * (
            +1.0 * np.einsum('ABEf,EfcIJk->ABcIJk', X.aa.vvvv[Va, Va, Va, va], T.aab.VvvOOo, optimize=True)
            - 0.5 * np.einsum('ABEF,FEcIJk->ABcIJk', X.aa.vvvv[Va, Va, Va, Va], T.aab.VVvOOo, optimize=True)
    )
    dR.aab.VVvOOo += (2.0 / 4.0) * (
            +1.0 * np.einsum('Bcef,AefIJk->ABcIJk', X.ab.vvvv[Va, vb, va, vb], T.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('BceF,AeFIJk->ABcIJk', X.ab.vvvv[Va, vb, va, Vb], T.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('BcEf,EAfIJk->ABcIJk', X.ab.vvvv[Va, vb, Va, vb], T.aab.VVvOOo, optimize=True)
            - 1.0 * np.einsum('BcEF,EAFIJk->ABcIJk', X.ab.vvvv[Va, vb, Va, Vb], T.aab.VVVOOo, optimize=True)
    )
    dR.aab.VVvOOo += (4.0 / 4.0) * (
            -1.0 * np.einsum('AmIe,BecmJk->ABcIJk', X.aa.voov[Va, oa, Oa, va], T.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('AMIe,BecMJk->ABcIJk', X.aa.voov[Va, Oa, Oa, va], T.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('AmIE,BEcmJk->ABcIJk', X.aa.voov[Va, oa, Oa, Va], T.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('AMIE,BEcMJk->ABcIJk', X.aa.voov[Va, Oa, Oa, Va], T.aab.VVvOOo, optimize=True)
    )
    dR.aab.VVvOOo += (4.0 / 4.0) * (
            -1.0 * np.einsum('AmIe,BceJmk->ABcIJk', X.ab.voov[Va, ob, Oa, vb], T.abb.VvvOoo, optimize=True)
            + 1.0 * np.einsum('AMIe,BceJkM->ABcIJk', X.ab.voov[Va, Ob, Oa, vb], T.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('AmIE,BEcJmk->ABcIJk', X.ab.voov[Va, ob, Oa, Vb], T.abb.VVvOoo, optimize=True)
            - 1.0 * np.einsum('AMIE,BEcJkM->ABcIJk', X.ab.voov[Va, Ob, Oa, Vb], T.abb.VVvOoO, optimize=True)
    )
    dR.aab.VVvOOo += (1.0 / 4.0) * (
            -1.0 * np.einsum('mcek,BAemIJ->ABcIJk', X.ab.ovvo[oa, vb, va, ob], T.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('Mcek,BAeIJM->ABcIJk', X.ab.ovvo[Oa, vb, va, ob], T.aaa.VVvOOO, optimize=True)
            - 1.0 * np.einsum('mcEk,EBAmIJ->ABcIJk', X.ab.ovvo[oa, vb, Va, ob], T.aaa.VVVoOO, optimize=True)
            - 1.0 * np.einsum('McEk,EBAIJM->ABcIJk', X.ab.ovvo[Oa, vb, Va, ob], T.aaa.VVVOOO, optimize=True)
    )
    dR.aab.VVvOOo += (1.0 / 4.0) * (
            -1.0 * np.einsum('cmke,BAeIJm->ABcIJk', X.bb.voov[vb, ob, ob, vb], T.aab.VVvOOo, optimize=True)
            - 1.0 * np.einsum('cMke,BAeIJM->ABcIJk', X.bb.voov[vb, Ob, ob, vb], T.aab.VVvOOO, optimize=True)
            - 1.0 * np.einsum('cmkE,BAEIJm->ABcIJk', X.bb.voov[vb, ob, ob, Vb], T.aab.VVVOOo, optimize=True)
            - 1.0 * np.einsum('cMkE,BAEIJM->ABcIJk', X.bb.voov[vb, Ob, ob, Vb], T.aab.VVVOOO, optimize=True)
    )
    dR.aab.VVvOOo += (2.0 / 4.0) * (
            +1.0 * np.einsum('Amek,BecIJm->ABcIJk', X.ab.vovo[Va, ob, va, ob], T.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('AMek,BecIJM->ABcIJk', X.ab.vovo[Va, Ob, va, ob], T.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('AmEk,BEcIJm->ABcIJk', X.ab.vovo[Va, ob, Va, ob], T.aab.VVvOOo, optimize=True)
            + 1.0 * np.einsum('AMEk,BEcIJM->ABcIJk', X.ab.vovo[Va, Ob, Va, ob], T.aab.VVvOOO, optimize=True)
    )
    dR.aab.VVvOOo += (2.0 / 4.0) * (
            +1.0 * np.einsum('mcIe,BAemJk->ABcIJk', X.ab.ovov[oa, vb, Oa, vb], T.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('McIe,BAeMJk->ABcIJk', X.ab.ovov[Oa, vb, Oa, vb], T.aab.VVvOOo, optimize=True)
            + 1.0 * np.einsum('mcIE,BAEmJk->ABcIJk', X.ab.ovov[oa, vb, Oa, Vb], T.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('McIE,BAEMJk->ABcIJk', X.ab.ovov[Oa, vb, Oa, Vb], T.aab.VVVOOo, optimize=True)
    )
    dR.aab.VVvOOo += (2.0 / 4.0) * (
            +1.0 * np.einsum('mI,BAcmJk->ABcIJk', H.a.oo[oa, Oa], R.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('MI,BAcMJk->ABcIJk', H.a.oo[Oa, Oa], R.aab.VVvOOo, optimize=True)
    )
    dR.aab.VVvOOo += (1.0 / 4.0) * (
            +1.0 * np.einsum('mk,BAcIJm->ABcIJk', H.b.oo[ob, ob], R.aab.VVvOOo, optimize=True)
            + 1.0 * np.einsum('Mk,BAcIJM->ABcIJk', H.b.oo[Ob, ob], R.aab.VVvOOO, optimize=True)
    )
    dR.aab.VVvOOo += (2.0 / 4.0) * (
            -1.0 * np.einsum('Ae,BecIJk->ABcIJk', H.a.vv[Va, va], R.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('AE,BEcIJk->ABcIJk', H.a.vv[Va, Va], R.aab.VVvOOo, optimize=True)
    )
    dR.aab.VVvOOo += (1.0 / 4.0) * (
            -1.0 * np.einsum('ce,BAeIJk->ABcIJk', H.b.vv[vb, vb], R.aab.VVvOOo, optimize=True)
            - 1.0 * np.einsum('cE,BAEIJk->ABcIJk', H.b.vv[vb, Vb], R.aab.VVVOOo, optimize=True)
    )
    dR.aab.VVvOOo += (1.0 / 4.0) * (
            -1.0 * np.einsum('mNIJ,BAcmNk->ABcIJk', H.aa.oooo[oa, Oa, Oa, Oa], R.aab.VVvoOo, optimize=True)
            - 0.5 * np.einsum('MNIJ,BAcMNk->ABcIJk', H.aa.oooo[Oa, Oa, Oa, Oa], R.aab.VVvOOo, optimize=True)
    )
    dR.aab.VVvOOo += (2.0 / 4.0) * (
            +1.0 * np.einsum('mnJk,BAcmIn->ABcIJk', H.ab.oooo[oa, ob, Oa, ob], R.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('MnJk,BAcIMn->ABcIJk', H.ab.oooo[Oa, ob, Oa, ob], R.aab.VVvOOo, optimize=True)
            + 1.0 * np.einsum('mNJk,BAcmIN->ABcIJk', H.ab.oooo[oa, Ob, Oa, ob], R.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('MNJk,BAcIMN->ABcIJk', H.ab.oooo[Oa, Ob, Oa, ob], R.aab.VVvOOO, optimize=True)
    )
    dR.aab.VVvOOo += (1.0 / 4.0) * (
            +1.0 * np.einsum('ABEf,EfcIJk->ABcIJk', H.aa.vvvv[Va, Va, Va, va], R.aab.VvvOOo, optimize=True)
            - 0.5 * np.einsum('ABEF,FEcIJk->ABcIJk', H.aa.vvvv[Va, Va, Va, Va], R.aab.VVvOOo, optimize=True)
    )
    dR.aab.VVvOOo += (2.0 / 4.0) * (
            +1.0 * np.einsum('Bcef,AefIJk->ABcIJk', H.ab.vvvv[Va, vb, va, vb], R.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('BceF,AeFIJk->ABcIJk', H.ab.vvvv[Va, vb, va, Vb], R.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('BcEf,EAfIJk->ABcIJk', H.ab.vvvv[Va, vb, Va, vb], R.aab.VVvOOo, optimize=True)
            - 1.0 * np.einsum('BcEF,EAFIJk->ABcIJk', H.ab.vvvv[Va, vb, Va, Vb], R.aab.VVVOOo, optimize=True)
    )
    dR.aab.VVvOOo += (4.0 / 4.0) * (
            -1.0 * np.einsum('AmIe,BecmJk->ABcIJk', H.aa.voov[Va, oa, Oa, va], R.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('AMIe,BecMJk->ABcIJk', H.aa.voov[Va, Oa, Oa, va], R.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('AmIE,BEcmJk->ABcIJk', H.aa.voov[Va, oa, Oa, Va], R.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('AMIE,BEcMJk->ABcIJk', H.aa.voov[Va, Oa, Oa, Va], R.aab.VVvOOo, optimize=True)
    )
    dR.aab.VVvOOo += (4.0 / 4.0) * (
            -1.0 * np.einsum('AmIe,BceJmk->ABcIJk', H.ab.voov[Va, ob, Oa, vb], R.abb.VvvOoo, optimize=True)
            + 1.0 * np.einsum('AMIe,BceJkM->ABcIJk', H.ab.voov[Va, Ob, Oa, vb], R.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('AmIE,BEcJmk->ABcIJk', H.ab.voov[Va, ob, Oa, Vb], R.abb.VVvOoo, optimize=True)
            - 1.0 * np.einsum('AMIE,BEcJkM->ABcIJk', H.ab.voov[Va, Ob, Oa, Vb], R.abb.VVvOoO, optimize=True)
    )
    dR.aab.VVvOOo += (1.0 / 4.0) * (
            -1.0 * np.einsum('mcek,BAemIJ->ABcIJk', H.ab.ovvo[oa, vb, va, ob], R.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('Mcek,BAeIJM->ABcIJk', H.ab.ovvo[Oa, vb, va, ob], R.aaa.VVvOOO, optimize=True)
            - 1.0 * np.einsum('mcEk,EBAmIJ->ABcIJk', H.ab.ovvo[oa, vb, Va, ob], R.aaa.VVVoOO, optimize=True)
            - 1.0 * np.einsum('McEk,EBAIJM->ABcIJk', H.ab.ovvo[Oa, vb, Va, ob], R.aaa.VVVOOO, optimize=True)
    )
    dR.aab.VVvOOo += (1.0 / 4.0) * (
            -1.0 * np.einsum('cmke,BAeIJm->ABcIJk', H.bb.voov[vb, ob, ob, vb], R.aab.VVvOOo, optimize=True)
            - 1.0 * np.einsum('cMke,BAeIJM->ABcIJk', H.bb.voov[vb, Ob, ob, vb], R.aab.VVvOOO, optimize=True)
            - 1.0 * np.einsum('cmkE,BAEIJm->ABcIJk', H.bb.voov[vb, ob, ob, Vb], R.aab.VVVOOo, optimize=True)
            - 1.0 * np.einsum('cMkE,BAEIJM->ABcIJk', H.bb.voov[vb, Ob, ob, Vb], R.aab.VVVOOO, optimize=True)
    )
    dR.aab.VVvOOo += (2.0 / 4.0) * (
            +1.0 * np.einsum('Amek,BecIJm->ABcIJk', H.ab.vovo[Va, ob, va, ob], R.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('AMek,BecIJM->ABcIJk', H.ab.vovo[Va, Ob, va, ob], R.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('AmEk,BEcIJm->ABcIJk', H.ab.vovo[Va, ob, Va, ob], R.aab.VVvOOo, optimize=True)
            + 1.0 * np.einsum('AMEk,BEcIJM->ABcIJk', H.ab.vovo[Va, Ob, Va, ob], R.aab.VVvOOO, optimize=True)
    )
    dR.aab.VVvOOo += (2.0 / 4.0) * (
            +1.0 * np.einsum('mcIe,BAemJk->ABcIJk', H.ab.ovov[oa, vb, Oa, vb], R.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('McIe,BAeMJk->ABcIJk', H.ab.ovov[Oa, vb, Oa, vb], R.aab.VVvOOo, optimize=True)
            + 1.0 * np.einsum('mcIE,BAEmJk->ABcIJk', H.ab.ovov[oa, vb, Oa, Vb], R.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('McIE,BAEMJk->ABcIJk', H.ab.ovov[Oa, vb, Oa, Vb], R.aab.VVVOOo, optimize=True)
    )

    dR.aab.VVvOOo -= np.transpose(dR.aab.VVvOOo, (1, 0, 2, 3, 4, 5))
    dR.aab.VVvOOo -= np.transpose(dR.aab.VVvOOo, (0, 1, 2, 4, 3, 5))

    return dR

def update(R, omega, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    R.aab.VVvOOo = eomcc_active_loops.update_r3b_110110(
        R.aab.VVvOOo,
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
