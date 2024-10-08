import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops

def build(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.aab.VVVOOo = (2.0 / 4.0) * (
            +1.0 * np.einsum('BCek,AeIJ->ABCIJk', X.ab.vvvo[Va, Vb, :, ob], T.aa[Va, :, Oa, Oa], optimize=True)
    )
    dR.aab.VVVOOo += (2.0 / 4.0) * (
            -1.0 * np.einsum('mCJk,ABIm->ABCIJk', X.ab.ovoo[:, Vb, Oa, ob], T.aa[Va, Va, Oa, :], optimize=True)
    )
    dR.aab.VVVOOo += (4.0 / 4.0) * (
            +1.0 * np.einsum('ACIe,BeJk->ABCIJk', X.ab.vvov[Va, Vb, Oa, :], T.ab[Va, :, Oa, ob], optimize=True)
    )
    dR.aab.VVVOOo += (4.0 / 4.0) * (
            -1.0 * np.einsum('AmIk,BCJm->ABCIJk', X.ab.vooo[Va, :, Oa, ob], T.ab[Va, Vb, Oa, :], optimize=True)
    )
    dR.aab.VVVOOo += (2.0 / 4.0) * (
            +1.0 * np.einsum('ABIe,eCJk->ABCIJk', X.aa.vvov[Va, Va, Oa, :], T.ab[:, Vb, Oa, ob], optimize=True)
    )
    dR.aab.VVVOOo += (2.0 / 4.0) * (
            -1.0 * np.einsum('AmIJ,BCmk->ABCIJk', X.aa.vooo[Va, :, Oa, Oa], T.ab[Va, Vb, :, ob], optimize=True)
    )
    dR.aab.VVVOOo += (2.0 / 4.0) * (
            +1.0 * np.einsum('BCek,AeIJ->ABCIJk', H.ab.vvvo[Va, Vb, :, ob], R.aa[Va, :, Oa, Oa], optimize=True)
    )
    dR.aab.VVVOOo += (2.0 / 4.0) * (
            -1.0 * np.einsum('mCJk,ABIm->ABCIJk', H.ab.ovoo[:, Vb, Oa, ob], R.aa[Va, Va, Oa, :], optimize=True)
    )
    dR.aab.VVVOOo += (4.0 / 4.0) * (
            +1.0 * np.einsum('ACIe,BeJk->ABCIJk', H.ab.vvov[Va, Vb, Oa, :], R.ab[Va, :, Oa, ob], optimize=True)
    )
    dR.aab.VVVOOo += (4.0 / 4.0) * (
            -1.0 * np.einsum('AmIk,BCJm->ABCIJk', H.ab.vooo[Va, :, Oa, ob], R.ab[Va, Vb, Oa, :], optimize=True)
    )
    dR.aab.VVVOOo += (2.0 / 4.0) * (
            +1.0 * np.einsum('ABIe,eCJk->ABCIJk', H.aa.vvov[Va, Va, Oa, :], R.ab[:, Vb, Oa, ob], optimize=True)
    )
    dR.aab.VVVOOo += (2.0 / 4.0) * (
            -1.0 * np.einsum('AmIJ,BCmk->ABCIJk', H.aa.vooo[Va, :, Oa, Oa], R.ab[Va, Vb, :, ob], optimize=True)
    )

    dR.aab.VVVOOo += (2.0 / 4.0) * (
            +1.0 * np.einsum('mI,BACmJk->ABCIJk', X.a.oo[oa, Oa], T.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('MI,BACMJk->ABCIJk', X.a.oo[Oa, Oa], T.aab.VVVOOo, optimize=True)
    )
    dR.aab.VVVOOo += (1.0 / 4.0) * (
            +1.0 * np.einsum('mk,BACIJm->ABCIJk', X.b.oo[ob, ob], T.aab.VVVOOo, optimize=True)
            + 1.0 * np.einsum('Mk,BACIJM->ABCIJk', X.b.oo[Ob, ob], T.aab.VVVOOO, optimize=True)
    )
    dR.aab.VVVOOo += (2.0 / 4.0) * (
            -1.0 * np.einsum('Ae,BeCIJk->ABCIJk', X.a.vv[Va, va], T.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('AE,BECIJk->ABCIJk', X.a.vv[Va, Va], T.aab.VVVOOo, optimize=True)
    )
    dR.aab.VVVOOo += (1.0 / 4.0) * (
            -1.0 * np.einsum('Ce,BAeIJk->ABCIJk', X.b.vv[Vb, vb], T.aab.VVvOOo, optimize=True)
            - 1.0 * np.einsum('CE,BAEIJk->ABCIJk', X.b.vv[Vb, Vb], T.aab.VVVOOo, optimize=True)
    )
    dR.aab.VVVOOo += (1.0 / 4.0) * (
            +1.0 * np.einsum('MnIJ,BACnMk->ABCIJk', X.aa.oooo[Oa, oa, Oa, Oa], T.aab.VVVoOo, optimize=True)
            - 0.5 * np.einsum('MNIJ,BACMNk->ABCIJk', X.aa.oooo[Oa, Oa, Oa, Oa], T.aab.VVVOOo, optimize=True)
    )
    dR.aab.VVVOOo += (2.0 / 4.0) * (
            +1.0 * np.einsum('mnJk,BACmIn->ABCIJk', X.ab.oooo[oa, ob, Oa, ob], T.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('mNJk,BACmIN->ABCIJk', X.ab.oooo[oa, Ob, Oa, ob], T.aab.VVVoOO, optimize=True)
            - 1.0 * np.einsum('MnJk,BACIMn->ABCIJk', X.ab.oooo[Oa, ob, Oa, ob], T.aab.VVVOOo, optimize=True)
            - 1.0 * np.einsum('MNJk,BACIMN->ABCIJk', X.ab.oooo[Oa, Ob, Oa, ob], T.aab.VVVOOO, optimize=True)
    )
    dR.aab.VVVOOo += (1.0 / 4.0) * (
            -0.5 * np.einsum('ABef,feCIJk->ABCIJk', X.aa.vvvv[Va, Va, va, va], T.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('ABEf,EfCIJk->ABCIJk', X.aa.vvvv[Va, Va, Va, va], T.aab.VvVOOo, optimize=True)
            - 0.5 * np.einsum('ABEF,FECIJk->ABCIJk', X.aa.vvvv[Va, Va, Va, Va], T.aab.VVVOOo, optimize=True)
    )
    dR.aab.VVVOOo += (2.0 / 4.0) * (
            +1.0 * np.einsum('BCef,AefIJk->ABCIJk', X.ab.vvvv[Va, Vb, va, vb], T.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('BCeF,AeFIJk->ABCIJk', X.ab.vvvv[Va, Vb, va, Vb], T.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('BCEf,EAfIJk->ABCIJk', X.ab.vvvv[Va, Vb, Va, vb], T.aab.VVvOOo, optimize=True)
            - 1.0 * np.einsum('BCEF,EAFIJk->ABCIJk', X.ab.vvvv[Va, Vb, Va, Vb], T.aab.VVVOOo, optimize=True)
    )
    dR.aab.VVVOOo += (4.0 / 4.0) * (
            -1.0 * np.einsum('AmIe,BeCmJk->ABCIJk', X.aa.voov[Va, oa, Oa, va], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('AMIe,BeCMJk->ABCIJk', X.aa.voov[Va, Oa, Oa, va], T.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('AmIE,BECmJk->ABCIJk', X.aa.voov[Va, oa, Oa, Va], T.aab.VVVoOo, optimize=True)
            - 1.0 * np.einsum('AMIE,BECMJk->ABCIJk', X.aa.voov[Va, Oa, Oa, Va], T.aab.VVVOOo, optimize=True)
    )
    dR.aab.VVVOOo += (4.0 / 4.0) * (
            -1.0 * np.einsum('AmIe,BCeJmk->ABCIJk', X.ab.voov[Va, ob, Oa, vb], T.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('AMIe,BCeJkM->ABCIJk', X.ab.voov[Va, Ob, Oa, vb], T.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('AmIE,BCEJmk->ABCIJk', X.ab.voov[Va, ob, Oa, Vb], T.abb.VVVOoo, optimize=True)
            + 1.0 * np.einsum('AMIE,BCEJkM->ABCIJk', X.ab.voov[Va, Ob, Oa, Vb], T.abb.VVVOoO, optimize=True)
    )
    dR.aab.VVVOOo += (1.0 / 4.0) * (
            -1.0 * np.einsum('mCek,BAemIJ->ABCIJk', X.ab.ovvo[oa, Vb, va, ob], T.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('MCek,BAeIJM->ABCIJk', X.ab.ovvo[Oa, Vb, va, ob], T.aaa.VVvOOO, optimize=True)
            - 1.0 * np.einsum('mCEk,EBAmIJ->ABCIJk', X.ab.ovvo[oa, Vb, Va, ob], T.aaa.VVVoOO, optimize=True)
            - 1.0 * np.einsum('MCEk,EBAIJM->ABCIJk', X.ab.ovvo[Oa, Vb, Va, ob], T.aaa.VVVOOO, optimize=True)
    )
    dR.aab.VVVOOo += (1.0 / 4.0) * (
            -1.0 * np.einsum('Cmke,BAeIJm->ABCIJk', X.bb.voov[Vb, ob, ob, vb], T.aab.VVvOOo, optimize=True)
            - 1.0 * np.einsum('CMke,BAeIJM->ABCIJk', X.bb.voov[Vb, Ob, ob, vb], T.aab.VVvOOO, optimize=True)
            - 1.0 * np.einsum('CmkE,BAEIJm->ABCIJk', X.bb.voov[Vb, ob, ob, Vb], T.aab.VVVOOo, optimize=True)
            - 1.0 * np.einsum('CMkE,BAEIJM->ABCIJk', X.bb.voov[Vb, Ob, ob, Vb], T.aab.VVVOOO, optimize=True)
    )
    dR.aab.VVVOOo += (2.0 / 4.0) * (
            +1.0 * np.einsum('Amek,BeCIJm->ABCIJk', X.ab.vovo[Va, ob, va, ob], T.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('AMek,BeCIJM->ABCIJk', X.ab.vovo[Va, Ob, va, ob], T.aab.VvVOOO, optimize=True)
            + 1.0 * np.einsum('AmEk,BECIJm->ABCIJk', X.ab.vovo[Va, ob, Va, ob], T.aab.VVVOOo, optimize=True)
            + 1.0 * np.einsum('AMEk,BECIJM->ABCIJk', X.ab.vovo[Va, Ob, Va, ob], T.aab.VVVOOO, optimize=True)
    )
    dR.aab.VVVOOo += (2.0 / 4.0) * (
            +1.0 * np.einsum('mCIe,BAemJk->ABCIJk', X.ab.ovov[oa, Vb, Oa, vb], T.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('MCIe,BAeMJk->ABCIJk', X.ab.ovov[Oa, Vb, Oa, vb], T.aab.VVvOOo, optimize=True)
            + 1.0 * np.einsum('mCIE,BAEmJk->ABCIJk', X.ab.ovov[oa, Vb, Oa, Vb], T.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('MCIE,BAEMJk->ABCIJk', X.ab.ovov[Oa, Vb, Oa, Vb], T.aab.VVVOOo, optimize=True)
    )
    dR.aab.VVVOOo += (2.0 / 4.0) * (
            +1.0 * np.einsum('mI,BACmJk->ABCIJk', H.a.oo[oa, Oa], R.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('MI,BACMJk->ABCIJk', H.a.oo[Oa, Oa], R.aab.VVVOOo, optimize=True)
    )
    dR.aab.VVVOOo += (1.0 / 4.0) * (
            +1.0 * np.einsum('mk,BACIJm->ABCIJk', H.b.oo[ob, ob], R.aab.VVVOOo, optimize=True)
            + 1.0 * np.einsum('Mk,BACIJM->ABCIJk', H.b.oo[Ob, ob], R.aab.VVVOOO, optimize=True)
    )
    dR.aab.VVVOOo += (2.0 / 4.0) * (
            -1.0 * np.einsum('Ae,BeCIJk->ABCIJk', H.a.vv[Va, va], R.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('AE,BECIJk->ABCIJk', H.a.vv[Va, Va], R.aab.VVVOOo, optimize=True)
    )
    dR.aab.VVVOOo += (1.0 / 4.0) * (
            -1.0 * np.einsum('Ce,BAeIJk->ABCIJk', H.b.vv[Vb, vb], R.aab.VVvOOo, optimize=True)
            - 1.0 * np.einsum('CE,BAEIJk->ABCIJk', H.b.vv[Vb, Vb], R.aab.VVVOOo, optimize=True)
    )
    dR.aab.VVVOOo += (1.0 / 4.0) * (
            +1.0 * np.einsum('MnIJ,BACnMk->ABCIJk', H.aa.oooo[Oa, oa, Oa, Oa], R.aab.VVVoOo, optimize=True)
            - 0.5 * np.einsum('MNIJ,BACMNk->ABCIJk', H.aa.oooo[Oa, Oa, Oa, Oa], R.aab.VVVOOo, optimize=True)
    )
    dR.aab.VVVOOo += (2.0 / 4.0) * (
            +1.0 * np.einsum('mnJk,BACmIn->ABCIJk', H.ab.oooo[oa, ob, Oa, ob], R.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('mNJk,BACmIN->ABCIJk', H.ab.oooo[oa, Ob, Oa, ob], R.aab.VVVoOO, optimize=True)
            - 1.0 * np.einsum('MnJk,BACIMn->ABCIJk', H.ab.oooo[Oa, ob, Oa, ob], R.aab.VVVOOo, optimize=True)
            - 1.0 * np.einsum('MNJk,BACIMN->ABCIJk', H.ab.oooo[Oa, Ob, Oa, ob], R.aab.VVVOOO, optimize=True)
    )
    dR.aab.VVVOOo += (1.0 / 4.0) * (
            -0.5 * np.einsum('ABef,feCIJk->ABCIJk', H.aa.vvvv[Va, Va, va, va], R.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('ABEf,EfCIJk->ABCIJk', H.aa.vvvv[Va, Va, Va, va], R.aab.VvVOOo, optimize=True)
            - 0.5 * np.einsum('ABEF,FECIJk->ABCIJk', H.aa.vvvv[Va, Va, Va, Va], R.aab.VVVOOo, optimize=True)
    )
    dR.aab.VVVOOo += (2.0 / 4.0) * (
            +1.0 * np.einsum('BCef,AefIJk->ABCIJk', H.ab.vvvv[Va, Vb, va, vb], R.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('BCeF,AeFIJk->ABCIJk', H.ab.vvvv[Va, Vb, va, Vb], R.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('BCEf,EAfIJk->ABCIJk', H.ab.vvvv[Va, Vb, Va, vb], R.aab.VVvOOo, optimize=True)
            - 1.0 * np.einsum('BCEF,EAFIJk->ABCIJk', H.ab.vvvv[Va, Vb, Va, Vb], R.aab.VVVOOo, optimize=True)
    )
    dR.aab.VVVOOo += (4.0 / 4.0) * (
            -1.0 * np.einsum('AmIe,BeCmJk->ABCIJk', H.aa.voov[Va, oa, Oa, va], R.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('AMIe,BeCMJk->ABCIJk', H.aa.voov[Va, Oa, Oa, va], R.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('AmIE,BECmJk->ABCIJk', H.aa.voov[Va, oa, Oa, Va], R.aab.VVVoOo, optimize=True)
            - 1.0 * np.einsum('AMIE,BECMJk->ABCIJk', H.aa.voov[Va, Oa, Oa, Va], R.aab.VVVOOo, optimize=True)
    )
    dR.aab.VVVOOo += (4.0 / 4.0) * (
            -1.0 * np.einsum('AmIe,BCeJmk->ABCIJk', H.ab.voov[Va, ob, Oa, vb], R.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('AMIe,BCeJkM->ABCIJk', H.ab.voov[Va, Ob, Oa, vb], R.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('AmIE,BCEJmk->ABCIJk', H.ab.voov[Va, ob, Oa, Vb], R.abb.VVVOoo, optimize=True)
            + 1.0 * np.einsum('AMIE,BCEJkM->ABCIJk', H.ab.voov[Va, Ob, Oa, Vb], R.abb.VVVOoO, optimize=True)
    )
    dR.aab.VVVOOo += (1.0 / 4.0) * (
            -1.0 * np.einsum('mCek,BAemIJ->ABCIJk', H.ab.ovvo[oa, Vb, va, ob], R.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('MCek,BAeIJM->ABCIJk', H.ab.ovvo[Oa, Vb, va, ob], R.aaa.VVvOOO, optimize=True)
            - 1.0 * np.einsum('mCEk,EBAmIJ->ABCIJk', H.ab.ovvo[oa, Vb, Va, ob], R.aaa.VVVoOO, optimize=True)
            - 1.0 * np.einsum('MCEk,EBAIJM->ABCIJk', H.ab.ovvo[Oa, Vb, Va, ob], R.aaa.VVVOOO, optimize=True)
    )
    dR.aab.VVVOOo += (1.0 / 4.0) * (
            -1.0 * np.einsum('Cmke,BAeIJm->ABCIJk', H.bb.voov[Vb, ob, ob, vb], R.aab.VVvOOo, optimize=True)
            - 1.0 * np.einsum('CMke,BAeIJM->ABCIJk', H.bb.voov[Vb, Ob, ob, vb], R.aab.VVvOOO, optimize=True)
            - 1.0 * np.einsum('CmkE,BAEIJm->ABCIJk', H.bb.voov[Vb, ob, ob, Vb], R.aab.VVVOOo, optimize=True)
            - 1.0 * np.einsum('CMkE,BAEIJM->ABCIJk', H.bb.voov[Vb, Ob, ob, Vb], R.aab.VVVOOO, optimize=True)
    )
    dR.aab.VVVOOo += (2.0 / 4.0) * (
            +1.0 * np.einsum('Amek,BeCIJm->ABCIJk', H.ab.vovo[Va, ob, va, ob], R.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('AMek,BeCIJM->ABCIJk', H.ab.vovo[Va, Ob, va, ob], R.aab.VvVOOO, optimize=True)
            + 1.0 * np.einsum('AmEk,BECIJm->ABCIJk', H.ab.vovo[Va, ob, Va, ob], R.aab.VVVOOo, optimize=True)
            + 1.0 * np.einsum('AMEk,BECIJM->ABCIJk', H.ab.vovo[Va, Ob, Va, ob], R.aab.VVVOOO, optimize=True)
    )
    dR.aab.VVVOOo += (2.0 / 4.0) * (
            +1.0 * np.einsum('mCIe,BAemJk->ABCIJk', H.ab.ovov[oa, Vb, Oa, vb], R.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('MCIe,BAeMJk->ABCIJk', H.ab.ovov[Oa, Vb, Oa, vb], R.aab.VVvOOo, optimize=True)
            + 1.0 * np.einsum('mCIE,BAEmJk->ABCIJk', H.ab.ovov[oa, Vb, Oa, Vb], R.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('MCIE,BAEMJk->ABCIJk', H.ab.ovov[Oa, Vb, Oa, Vb], R.aab.VVVOOo, optimize=True)
    )

    dR.aab.VVVOOo -= np.transpose(dR.aab.VVVOOo, (1, 0, 2, 3, 4, 5))
    dR.aab.VVVOOo -= np.transpose(dR.aab.VVVOOo, (0, 1, 2, 4, 3, 5))

    return dR

def update(R, omega, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    R.aab.VVVOOo = eomcc_active_loops.update_r3b_111110(
        R.aab.VVVOOo,
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
