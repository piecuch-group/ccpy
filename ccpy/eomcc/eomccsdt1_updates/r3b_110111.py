import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops

def build(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.aab.VVvOOO = (2.0 / 4.0) * (
            +1.0 * np.einsum('BceK,AeIJ->ABcIJK', X.ab.vvvo[Va, vb, :, Ob], T.aa[Va, :, Oa, Oa], optimize=True)
    )
    dR.aab.VVvOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('mcJK,ABIm->ABcIJK', X.ab.ovoo[:, vb, Oa, Ob], T.aa[Va, Va, Oa, :], optimize=True)
    )
    dR.aab.VVvOOO += (4.0 / 4.0) * (
            +1.0 * np.einsum('AcIe,BeJK->ABcIJK', X.ab.vvov[Va, vb, Oa, :], T.ab[Va, :, Oa, Ob], optimize=True)
    )
    dR.aab.VVvOOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('AmIK,BcJm->ABcIJK', X.ab.vooo[Va, :, Oa, Ob], T.ab[Va, vb, Oa, :], optimize=True)
    )
    dR.aab.VVvOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('ABIe,ecJK->ABcIJK', X.aa.vvov[Va, Va, Oa, :], T.ab[:, vb, Oa, Ob], optimize=True)
    )
    dR.aab.VVvOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('AmIJ,BcmK->ABcIJK', X.aa.vooo[Va, :, Oa, Oa], T.ab[Va, vb, :, Ob], optimize=True)
    )
    dR.aab.VVvOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('BceK,AeIJ->ABcIJK', H.ab.vvvo[Va, vb, :, Ob], R.aa[Va, :, Oa, Oa], optimize=True)
    )
    dR.aab.VVvOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('mcJK,ABIm->ABcIJK', H.ab.ovoo[:, vb, Oa, Ob], R.aa[Va, Va, Oa, :], optimize=True)
    )
    dR.aab.VVvOOO += (4.0 / 4.0) * (
            +1.0 * np.einsum('AcIe,BeJK->ABcIJK', H.ab.vvov[Va, vb, Oa, :], R.ab[Va, :, Oa, Ob], optimize=True)
    )
    dR.aab.VVvOOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('AmIK,BcJm->ABcIJK', H.ab.vooo[Va, :, Oa, Ob], R.ab[Va, vb, Oa, :], optimize=True)
    )
    dR.aab.VVvOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('ABIe,ecJK->ABcIJK', H.aa.vvov[Va, Va, Oa, :], R.ab[:, vb, Oa, Ob], optimize=True)
    )
    dR.aab.VVvOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('AmIJ,BcmK->ABcIJK', H.aa.vooo[Va, :, Oa, Oa], R.ab[Va, vb, :, Ob], optimize=True)
    )

    dR.aab.VVvOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mI,BAcmJK->ABcIJK', X.a.oo[oa, Oa], T.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('MI,BAcMJK->ABcIJK', X.a.oo[Oa, Oa], T.aab.VVvOOO, optimize=True)
    )
    dR.aab.VVvOOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('mK,BAcIJm->ABcIJK', X.b.oo[ob, Ob], T.aab.VVvOOo, optimize=True)
            + 1.0 * np.einsum('MK,BAcIJM->ABcIJK', X.b.oo[Ob, Ob], T.aab.VVvOOO, optimize=True)
    )
    dR.aab.VVvOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('Ae,BecIJK->ABcIJK', X.a.vv[Va, va], T.aab.VvvOOO, optimize=True)
            - 1.0 * np.einsum('AE,BEcIJK->ABcIJK', X.a.vv[Va, Va], T.aab.VVvOOO, optimize=True)
    )
    dR.aab.VVvOOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('ce,BAeIJK->ABcIJK', X.b.vv[vb, vb], T.aab.VVvOOO, optimize=True)
            - 1.0 * np.einsum('cE,BAEIJK->ABcIJK', X.b.vv[vb, Vb], T.aab.VVVOOO, optimize=True)
    )
    dR.aab.VVvOOO += (1.0 / 4.0) * (
            -0.5 * np.einsum('mnIJ,BAcmnK->ABcIJK', X.aa.oooo[oa, oa, Oa, Oa], T.aab.VVvooO, optimize=True)
            + 1.0 * np.einsum('MnIJ,BAcnMK->ABcIJK', X.aa.oooo[Oa, oa, Oa, Oa], T.aab.VVvoOO, optimize=True)
            - 0.5 * np.einsum('MNIJ,BAcMNK->ABcIJK', X.aa.oooo[Oa, Oa, Oa, Oa], T.aab.VVvOOO, optimize=True)
    )
    dR.aab.VVvOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mnJK,BAcmIn->ABcIJK', X.ab.oooo[oa, ob, Oa, Ob], T.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('mNJK,BAcmIN->ABcIJK', X.ab.oooo[oa, Ob, Oa, Ob], T.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('MnJK,BAcIMn->ABcIJK', X.ab.oooo[Oa, ob, Oa, Ob], T.aab.VVvOOo, optimize=True)
            - 1.0 * np.einsum('MNJK,BAcIMN->ABcIJK', X.ab.oooo[Oa, Ob, Oa, Ob], T.aab.VVvOOO, optimize=True)
    )
    dR.aab.VVvOOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('ABeF,FecIJK->ABcIJK', X.aa.vvvv[Va, Va, va, Va], T.aab.VvvOOO, optimize=True)
            - 0.5 * np.einsum('ABEF,FEcIJK->ABcIJK', X.aa.vvvv[Va, Va, Va, Va], T.aab.VVvOOO, optimize=True)
    )
    dR.aab.VVvOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('Bcef,AefIJK->ABcIJK', X.ab.vvvv[Va, vb, va, vb], T.aab.VvvOOO, optimize=True)
            - 1.0 * np.einsum('BcEf,EAfIJK->ABcIJK', X.ab.vvvv[Va, vb, Va, vb], T.aab.VVvOOO, optimize=True)
            + 1.0 * np.einsum('BceF,AeFIJK->ABcIJK', X.ab.vvvv[Va, vb, va, Vb], T.aab.VvVOOO, optimize=True)
            - 1.0 * np.einsum('BcEF,EAFIJK->ABcIJK', X.ab.vvvv[Va, vb, Va, Vb], T.aab.VVVOOO, optimize=True)
    )
    dR.aab.VVvOOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('AmIe,BecmJK->ABcIJK', X.aa.voov[Va, oa, Oa, va], T.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('AmIE,BEcmJK->ABcIJK', X.aa.voov[Va, oa, Oa, Va], T.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('AMIe,BecMJK->ABcIJK', X.aa.voov[Va, Oa, Oa, va], T.aab.VvvOOO, optimize=True)
            - 1.0 * np.einsum('AMIE,BEcMJK->ABcIJK', X.aa.voov[Va, Oa, Oa, Va], T.aab.VVvOOO, optimize=True)
    )
    dR.aab.VVvOOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('AmIe,BceJmK->ABcIJK', X.ab.voov[Va, ob, Oa, vb], T.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('AmIE,BEcJmK->ABcIJK', X.ab.voov[Va, ob, Oa, Vb], T.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('AMIe,BceJMK->ABcIJK', X.ab.voov[Va, Ob, Oa, vb], T.abb.VvvOOO, optimize=True)
            + 1.0 * np.einsum('AMIE,BEcJMK->ABcIJK', X.ab.voov[Va, Ob, Oa, Vb], T.abb.VVvOOO, optimize=True)
    )
    dR.aab.VVvOOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('mceK,BAemIJ->ABcIJK', X.ab.ovvo[oa, vb, va, Ob], T.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('mcEK,EBAmIJ->ABcIJK', X.ab.ovvo[oa, vb, Va, Ob], T.aaa.VVVoOO, optimize=True)
            - 1.0 * np.einsum('MceK,BAeIJM->ABcIJK', X.ab.ovvo[Oa, vb, va, Ob], T.aaa.VVvOOO, optimize=True)
            - 1.0 * np.einsum('McEK,EBAIJM->ABcIJK', X.ab.ovvo[Oa, vb, Va, Ob], T.aaa.VVVOOO, optimize=True)
    )
    dR.aab.VVvOOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('cmKe,BAeIJm->ABcIJK', X.bb.voov[vb, ob, Ob, vb], T.aab.VVvOOo, optimize=True)
            - 1.0 * np.einsum('cmKE,BAEIJm->ABcIJK', X.bb.voov[vb, ob, Ob, Vb], T.aab.VVVOOo, optimize=True)
            - 1.0 * np.einsum('cMKe,BAeIJM->ABcIJK', X.bb.voov[vb, Ob, Ob, vb], T.aab.VVvOOO, optimize=True)
            - 1.0 * np.einsum('cMKE,BAEIJM->ABcIJK', X.bb.voov[vb, Ob, Ob, Vb], T.aab.VVVOOO, optimize=True)
    )
    dR.aab.VVvOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('AmeK,BecIJm->ABcIJK', X.ab.vovo[Va, ob, va, Ob], T.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('AmEK,BEcIJm->ABcIJK', X.ab.vovo[Va, ob, Va, Ob], T.aab.VVvOOo, optimize=True)
            + 1.0 * np.einsum('AMeK,BecIJM->ABcIJK', X.ab.vovo[Va, Ob, va, Ob], T.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('AMEK,BEcIJM->ABcIJK', X.ab.vovo[Va, Ob, Va, Ob], T.aab.VVvOOO, optimize=True)
    )
    dR.aab.VVvOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mcIe,BAemJK->ABcIJK', X.ab.ovov[oa, vb, Oa, vb], T.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('mcIE,BAEmJK->ABcIJK', X.ab.ovov[oa, vb, Oa, Vb], T.aab.VVVoOO, optimize=True)
            + 1.0 * np.einsum('McIe,BAeMJK->ABcIJK', X.ab.ovov[Oa, vb, Oa, vb], T.aab.VVvOOO, optimize=True)
            + 1.0 * np.einsum('McIE,BAEMJK->ABcIJK', X.ab.ovov[Oa, vb, Oa, Vb], T.aab.VVVOOO, optimize=True)
    )
    dR.aab.VVvOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mI,BAcmJK->ABcIJK', H.a.oo[oa, Oa], R.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('MI,BAcMJK->ABcIJK', H.a.oo[Oa, Oa], R.aab.VVvOOO, optimize=True)
    )
    dR.aab.VVvOOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('mK,BAcIJm->ABcIJK', H.b.oo[ob, Ob], R.aab.VVvOOo, optimize=True)
            + 1.0 * np.einsum('MK,BAcIJM->ABcIJK', H.b.oo[Ob, Ob], R.aab.VVvOOO, optimize=True)
    )
    dR.aab.VVvOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('Ae,BecIJK->ABcIJK', H.a.vv[Va, va], R.aab.VvvOOO, optimize=True)
            - 1.0 * np.einsum('AE,BEcIJK->ABcIJK', H.a.vv[Va, Va], R.aab.VVvOOO, optimize=True)
    )
    dR.aab.VVvOOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('ce,BAeIJK->ABcIJK', H.b.vv[vb, vb], R.aab.VVvOOO, optimize=True)
            - 1.0 * np.einsum('cE,BAEIJK->ABcIJK', H.b.vv[vb, Vb], R.aab.VVVOOO, optimize=True)
    )
    dR.aab.VVvOOO += (1.0 / 4.0) * (
            -0.5 * np.einsum('mnIJ,BAcmnK->ABcIJK', H.aa.oooo[oa, oa, Oa, Oa], R.aab.VVvooO, optimize=True)
            + 1.0 * np.einsum('MnIJ,BAcnMK->ABcIJK', H.aa.oooo[Oa, oa, Oa, Oa], R.aab.VVvoOO, optimize=True)
            - 0.5 * np.einsum('MNIJ,BAcMNK->ABcIJK', H.aa.oooo[Oa, Oa, Oa, Oa], R.aab.VVvOOO, optimize=True)
    )
    dR.aab.VVvOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mnJK,BAcmIn->ABcIJK', H.ab.oooo[oa, ob, Oa, Ob], R.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('mNJK,BAcmIN->ABcIJK', H.ab.oooo[oa, Ob, Oa, Ob], R.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('MnJK,BAcIMn->ABcIJK', H.ab.oooo[Oa, ob, Oa, Ob], R.aab.VVvOOo, optimize=True)
            - 1.0 * np.einsum('MNJK,BAcIMN->ABcIJK', H.ab.oooo[Oa, Ob, Oa, Ob], R.aab.VVvOOO, optimize=True)
    )
    dR.aab.VVvOOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('ABeF,FecIJK->ABcIJK', H.aa.vvvv[Va, Va, va, Va], R.aab.VvvOOO, optimize=True)
            - 0.5 * np.einsum('ABEF,FEcIJK->ABcIJK', H.aa.vvvv[Va, Va, Va, Va], R.aab.VVvOOO, optimize=True)
    )
    dR.aab.VVvOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('Bcef,AefIJK->ABcIJK', H.ab.vvvv[Va, vb, va, vb], R.aab.VvvOOO, optimize=True)
            - 1.0 * np.einsum('BcEf,EAfIJK->ABcIJK', H.ab.vvvv[Va, vb, Va, vb], R.aab.VVvOOO, optimize=True)
            + 1.0 * np.einsum('BceF,AeFIJK->ABcIJK', H.ab.vvvv[Va, vb, va, Vb], R.aab.VvVOOO, optimize=True)
            - 1.0 * np.einsum('BcEF,EAFIJK->ABcIJK', H.ab.vvvv[Va, vb, Va, Vb], R.aab.VVVOOO, optimize=True)
    )
    dR.aab.VVvOOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('AmIe,BecmJK->ABcIJK', H.aa.voov[Va, oa, Oa, va], R.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('AmIE,BEcmJK->ABcIJK', H.aa.voov[Va, oa, Oa, Va], R.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('AMIe,BecMJK->ABcIJK', H.aa.voov[Va, Oa, Oa, va], R.aab.VvvOOO, optimize=True)
            - 1.0 * np.einsum('AMIE,BEcMJK->ABcIJK', H.aa.voov[Va, Oa, Oa, Va], R.aab.VVvOOO, optimize=True)
    )
    dR.aab.VVvOOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('AmIe,BceJmK->ABcIJK', H.ab.voov[Va, ob, Oa, vb], R.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('AmIE,BEcJmK->ABcIJK', H.ab.voov[Va, ob, Oa, Vb], R.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('AMIe,BceJMK->ABcIJK', H.ab.voov[Va, Ob, Oa, vb], R.abb.VvvOOO, optimize=True)
            + 1.0 * np.einsum('AMIE,BEcJMK->ABcIJK', H.ab.voov[Va, Ob, Oa, Vb], R.abb.VVvOOO, optimize=True)
    )
    dR.aab.VVvOOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('mceK,BAemIJ->ABcIJK', H.ab.ovvo[oa, vb, va, Ob], R.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('mcEK,EBAmIJ->ABcIJK', H.ab.ovvo[oa, vb, Va, Ob], R.aaa.VVVoOO, optimize=True)
            - 1.0 * np.einsum('MceK,BAeIJM->ABcIJK', H.ab.ovvo[Oa, vb, va, Ob], R.aaa.VVvOOO, optimize=True)
            - 1.0 * np.einsum('McEK,EBAIJM->ABcIJK', H.ab.ovvo[Oa, vb, Va, Ob], R.aaa.VVVOOO, optimize=True)
    )
    dR.aab.VVvOOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('cmKe,BAeIJm->ABcIJK', H.bb.voov[vb, ob, Ob, vb], R.aab.VVvOOo, optimize=True)
            - 1.0 * np.einsum('cmKE,BAEIJm->ABcIJK', H.bb.voov[vb, ob, Ob, Vb], R.aab.VVVOOo, optimize=True)
            - 1.0 * np.einsum('cMKe,BAeIJM->ABcIJK', H.bb.voov[vb, Ob, Ob, vb], R.aab.VVvOOO, optimize=True)
            - 1.0 * np.einsum('cMKE,BAEIJM->ABcIJK', H.bb.voov[vb, Ob, Ob, Vb], R.aab.VVVOOO, optimize=True)
    )
    dR.aab.VVvOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('AmeK,BecIJm->ABcIJK', H.ab.vovo[Va, ob, va, Ob], R.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('AmEK,BEcIJm->ABcIJK', H.ab.vovo[Va, ob, Va, Ob], R.aab.VVvOOo, optimize=True)
            + 1.0 * np.einsum('AMeK,BecIJM->ABcIJK', H.ab.vovo[Va, Ob, va, Ob], R.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('AMEK,BEcIJM->ABcIJK', H.ab.vovo[Va, Ob, Va, Ob], R.aab.VVvOOO, optimize=True)
    )
    dR.aab.VVvOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mcIe,BAemJK->ABcIJK', H.ab.ovov[oa, vb, Oa, vb], R.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('mcIE,BAEmJK->ABcIJK', H.ab.ovov[oa, vb, Oa, Vb], R.aab.VVVoOO, optimize=True)
            + 1.0 * np.einsum('McIe,BAeMJK->ABcIJK', H.ab.ovov[Oa, vb, Oa, vb], R.aab.VVvOOO, optimize=True)
            + 1.0 * np.einsum('McIE,BAEMJK->ABcIJK', H.ab.ovov[Oa, vb, Oa, Vb], R.aab.VVVOOO, optimize=True)
    )

    dR.aab.VVvOOO -= np.transpose(dR.aab.VVvOOO, (1, 0, 2, 3, 4, 5))
    dR.aab.VVvOOO -= np.transpose(dR.aab.VVvOOO, (0, 1, 2, 4, 3, 5))

    return dR

def update(R, omega, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    R.aab.VVvOOO = eomcc_active_loops.update_r3b_110111(
        R.aab.VVvOOO,
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
