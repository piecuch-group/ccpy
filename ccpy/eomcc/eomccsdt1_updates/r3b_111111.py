import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops

def build(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.aab.VVVOOO = (2.0 / 4.0) * (
            +1.0 * np.einsum('BCeK,AeIJ->ABCIJK', X.ab.vvvo[Va, Vb, :, Ob], T.aa[Va, :, Oa, Oa], optimize=True)
    )
    dR.aab.VVVOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('mCJK,ABIm->ABCIJK', X.ab.ovoo[:, Vb, Oa, Ob], T.aa[Va, Va, Oa, :], optimize=True)
    )
    dR.aab.VVVOOO += (4.0 / 4.0) * (
            +1.0 * np.einsum('ACIe,BeJK->ABCIJK', X.ab.vvov[Va, Vb, Oa, :], T.ab[Va, :, Oa, Ob], optimize=True)
    )
    dR.aab.VVVOOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('AmIK,BCJm->ABCIJK', X.ab.vooo[Va, :, Oa, Ob], T.ab[Va, Vb, Oa, :], optimize=True)
    )
    dR.aab.VVVOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('ABIe,eCJK->ABCIJK', X.aa.vvov[Va, Va, Oa, :], T.ab[:, Vb, Oa, Ob], optimize=True)
    )
    dR.aab.VVVOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('AmIJ,BCmK->ABCIJK', X.aa.vooo[Va, :, Oa, Oa], T.ab[Va, Vb, :, Ob], optimize=True)
    )
    dR.aab.VVVOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('BCeK,AeIJ->ABCIJK', H.ab.vvvo[Va, Vb, :, Ob], R.aa[Va, :, Oa, Oa], optimize=True)
    )
    dR.aab.VVVOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('mCJK,ABIm->ABCIJK', H.ab.ovoo[:, Vb, Oa, Ob], R.aa[Va, Va, Oa, :], optimize=True)
    )
    dR.aab.VVVOOO += (4.0 / 4.0) * (
            +1.0 * np.einsum('ACIe,BeJK->ABCIJK', H.ab.vvov[Va, Vb, Oa, :], R.ab[Va, :, Oa, Ob], optimize=True)
    )
    dR.aab.VVVOOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('AmIK,BCJm->ABCIJK', H.ab.vooo[Va, :, Oa, Ob], R.ab[Va, Vb, Oa, :], optimize=True)
    )
    dR.aab.VVVOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('ABIe,eCJK->ABCIJK', H.aa.vvov[Va, Va, Oa, :], R.ab[:, Vb, Oa, Ob], optimize=True)
    )
    dR.aab.VVVOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('AmIJ,BCmK->ABCIJK', H.aa.vooo[Va, :, Oa, Oa], R.ab[Va, Vb, :, Ob], optimize=True)
    )
    dR.aab.VVVOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mI,BACmJK->ABCIJK', X.a.oo[oa, Oa], T.aab.VVVoOO, optimize=True)
            + 1.0 * np.einsum('MI,BACMJK->ABCIJK', X.a.oo[Oa, Oa], T.aab.VVVOOO, optimize=True)
    )
    dR.aab.VVVOOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('mK,BACIJm->ABCIJK', X.b.oo[ob, Ob], T.aab.VVVOOo, optimize=True)
            + 1.0 * np.einsum('MK,BACIJM->ABCIJK', X.b.oo[Ob, Ob], T.aab.VVVOOO, optimize=True)
    )
    dR.aab.VVVOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('Ae,BeCIJK->ABCIJK', X.a.vv[Va, va], T.aab.VvVOOO, optimize=True)
            - 1.0 * np.einsum('AE,BECIJK->ABCIJK', X.a.vv[Va, Va], T.aab.VVVOOO, optimize=True)
    )
    dR.aab.VVVOOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('Ce,BAeIJK->ABCIJK', X.b.vv[Vb, vb], T.aab.VVvOOO, optimize=True)
            - 1.0 * np.einsum('CE,BAEIJK->ABCIJK', X.b.vv[Vb, Vb], T.aab.VVVOOO, optimize=True)
    )
    dR.aab.VVVOOO += (1.0 / 4.0) * (
            -0.5 * np.einsum('mnIJ,BACmnK->ABCIJK', X.aa.oooo[oa, oa, Oa, Oa], T.aab.VVVooO, optimize=True)
            + 1.0 * np.einsum('MnIJ,BACnMK->ABCIJK', X.aa.oooo[Oa, oa, Oa, Oa], T.aab.VVVoOO, optimize=True)
            - 0.5 * np.einsum('MNIJ,BACMNK->ABCIJK', X.aa.oooo[Oa, Oa, Oa, Oa], T.aab.VVVOOO, optimize=True)
    )
    dR.aab.VVVOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mnJK,BACmIn->ABCIJK', X.ab.oooo[oa, ob, Oa, Ob], T.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('mNJK,BACmIN->ABCIJK', X.ab.oooo[oa, Ob, Oa, Ob], T.aab.VVVoOO, optimize=True)
            - 1.0 * np.einsum('MnJK,BACIMn->ABCIJK', X.ab.oooo[Oa, ob, Oa, Ob], T.aab.VVVOOo, optimize=True)
            - 1.0 * np.einsum('MNJK,BACIMN->ABCIJK', X.ab.oooo[Oa, Ob, Oa, Ob], T.aab.VVVOOO, optimize=True)
    )
    dR.aab.VVVOOO += (1.0 / 4.0) * (
            -0.5 * np.einsum('ABef,feCIJK->ABCIJK', X.aa.vvvv[Va, Va, va, va], T.aab.vvVOOO, optimize=True)
            + 1.0 * np.einsum('ABEf,EfCIJK->ABCIJK', X.aa.vvvv[Va, Va, Va, va], T.aab.VvVOOO, optimize=True)
            - 0.5 * np.einsum('ABEF,FECIJK->ABCIJK', X.aa.vvvv[Va, Va, Va, Va], T.aab.VVVOOO, optimize=True)
    )
    dR.aab.VVVOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('BCef,AefIJK->ABCIJK', X.ab.vvvv[Va, Vb, va, vb], T.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('BCeF,AeFIJK->ABCIJK', X.ab.vvvv[Va, Vb, va, Vb], T.aab.VvVOOO, optimize=True)
            - 1.0 * np.einsum('BCEf,EAfIJK->ABCIJK', X.ab.vvvv[Va, Vb, Va, vb], T.aab.VVvOOO, optimize=True)
            - 1.0 * np.einsum('BCEF,EAFIJK->ABCIJK', X.ab.vvvv[Va, Vb, Va, Vb], T.aab.VVVOOO, optimize=True)
    )
    dR.aab.VVVOOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('AmIe,BeCmJK->ABCIJK', X.aa.voov[Va, oa, Oa, va], T.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('AMIe,BeCMJK->ABCIJK', X.aa.voov[Va, Oa, Oa, va], T.aab.VvVOOO, optimize=True)
            - 1.0 * np.einsum('AmIE,BECmJK->ABCIJK', X.aa.voov[Va, oa, Oa, Va], T.aab.VVVoOO, optimize=True)
            - 1.0 * np.einsum('AMIE,BECMJK->ABCIJK', X.aa.voov[Va, Oa, Oa, Va], T.aab.VVVOOO, optimize=True)
    )
    dR.aab.VVVOOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('AmIe,BCeJmK->ABCIJK', X.ab.voov[Va, ob, Oa, vb], T.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('AMIe,BCeJMK->ABCIJK', X.ab.voov[Va, Ob, Oa, vb], T.abb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('AmIE,BCEJmK->ABCIJK', X.ab.voov[Va, ob, Oa, Vb], T.abb.VVVOoO, optimize=True)
            - 1.0 * np.einsum('AMIE,BCEJMK->ABCIJK', X.ab.voov[Va, Ob, Oa, Vb], T.abb.VVVOOO, optimize=True)
    )
    dR.aab.VVVOOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('mCeK,BAemIJ->ABCIJK', X.ab.ovvo[oa, Vb, va, Ob], T.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('MCeK,BAeIJM->ABCIJK', X.ab.ovvo[Oa, Vb, va, Ob], T.aaa.VVvOOO, optimize=True)
            - 1.0 * np.einsum('mCEK,EBAmIJ->ABCIJK', X.ab.ovvo[oa, Vb, Va, Ob], T.aaa.VVVoOO, optimize=True)
            - 1.0 * np.einsum('MCEK,EBAIJM->ABCIJK', X.ab.ovvo[Oa, Vb, Va, Ob], T.aaa.VVVOOO, optimize=True)
    )
    dR.aab.VVVOOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('CmKe,BAeIJm->ABCIJK', X.bb.voov[Vb, ob, Ob, vb], T.aab.VVvOOo, optimize=True)
            - 1.0 * np.einsum('CMKe,BAeIJM->ABCIJK', X.bb.voov[Vb, Ob, Ob, vb], T.aab.VVvOOO, optimize=True)
            - 1.0 * np.einsum('CmKE,BAEIJm->ABCIJK', X.bb.voov[Vb, ob, Ob, Vb], T.aab.VVVOOo, optimize=True)
            - 1.0 * np.einsum('CMKE,BAEIJM->ABCIJK', X.bb.voov[Vb, Ob, Ob, Vb], T.aab.VVVOOO, optimize=True)
    )
    dR.aab.VVVOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('AmeK,BeCIJm->ABCIJK', X.ab.vovo[Va, ob, va, Ob], T.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('AMeK,BeCIJM->ABCIJK', X.ab.vovo[Va, Ob, va, Ob], T.aab.VvVOOO, optimize=True)
            + 1.0 * np.einsum('AmEK,BECIJm->ABCIJK', X.ab.vovo[Va, ob, Va, Ob], T.aab.VVVOOo, optimize=True)
            + 1.0 * np.einsum('AMEK,BECIJM->ABCIJK', X.ab.vovo[Va, Ob, Va, Ob], T.aab.VVVOOO, optimize=True)
    )
    dR.aab.VVVOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mCIe,BAemJK->ABCIJK', X.ab.ovov[oa, Vb, Oa, vb], T.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('MCIe,BAeMJK->ABCIJK', X.ab.ovov[Oa, Vb, Oa, vb], T.aab.VVvOOO, optimize=True)
            + 1.0 * np.einsum('mCIE,BAEmJK->ABCIJK', X.ab.ovov[oa, Vb, Oa, Vb], T.aab.VVVoOO, optimize=True)
            + 1.0 * np.einsum('MCIE,BAEMJK->ABCIJK', X.ab.ovov[Oa, Vb, Oa, Vb], T.aab.VVVOOO, optimize=True)
    )
    dR.aab.VVVOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mI,BACmJK->ABCIJK', H.a.oo[oa, Oa], R.aab.VVVoOO, optimize=True)
            + 1.0 * np.einsum('MI,BACMJK->ABCIJK', H.a.oo[Oa, Oa], R.aab.VVVOOO, optimize=True)
    )
    dR.aab.VVVOOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('mK,BACIJm->ABCIJK', H.b.oo[ob, Ob], R.aab.VVVOOo, optimize=True)
            + 1.0 * np.einsum('MK,BACIJM->ABCIJK', H.b.oo[Ob, Ob], R.aab.VVVOOO, optimize=True)
    )
    dR.aab.VVVOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('Ae,BeCIJK->ABCIJK', H.a.vv[Va, va], R.aab.VvVOOO, optimize=True)
            - 1.0 * np.einsum('AE,BECIJK->ABCIJK', H.a.vv[Va, Va], R.aab.VVVOOO, optimize=True)
    )
    dR.aab.VVVOOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('Ce,BAeIJK->ABCIJK', H.b.vv[Vb, vb], R.aab.VVvOOO, optimize=True)
            - 1.0 * np.einsum('CE,BAEIJK->ABCIJK', H.b.vv[Vb, Vb], R.aab.VVVOOO, optimize=True)
    )
    dR.aab.VVVOOO += (1.0 / 4.0) * (
            -0.5 * np.einsum('mnIJ,BACmnK->ABCIJK', H.aa.oooo[oa, oa, Oa, Oa], R.aab.VVVooO, optimize=True)
            + 1.0 * np.einsum('MnIJ,BACnMK->ABCIJK', H.aa.oooo[Oa, oa, Oa, Oa], R.aab.VVVoOO, optimize=True)
            - 0.5 * np.einsum('MNIJ,BACMNK->ABCIJK', H.aa.oooo[Oa, Oa, Oa, Oa], R.aab.VVVOOO, optimize=True)
    )
    dR.aab.VVVOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mnJK,BACmIn->ABCIJK', H.ab.oooo[oa, ob, Oa, Ob], R.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('mNJK,BACmIN->ABCIJK', H.ab.oooo[oa, Ob, Oa, Ob], R.aab.VVVoOO, optimize=True)
            - 1.0 * np.einsum('MnJK,BACIMn->ABCIJK', H.ab.oooo[Oa, ob, Oa, Ob], R.aab.VVVOOo, optimize=True)
            - 1.0 * np.einsum('MNJK,BACIMN->ABCIJK', H.ab.oooo[Oa, Ob, Oa, Ob], R.aab.VVVOOO, optimize=True)
    )
    dR.aab.VVVOOO += (1.0 / 4.0) * (
            -0.5 * np.einsum('ABef,feCIJK->ABCIJK', H.aa.vvvv[Va, Va, va, va], R.aab.vvVOOO, optimize=True)
            + 1.0 * np.einsum('ABEf,EfCIJK->ABCIJK', H.aa.vvvv[Va, Va, Va, va], R.aab.VvVOOO, optimize=True)
            - 0.5 * np.einsum('ABEF,FECIJK->ABCIJK', H.aa.vvvv[Va, Va, Va, Va], R.aab.VVVOOO, optimize=True)
    )
    dR.aab.VVVOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('BCef,AefIJK->ABCIJK', H.ab.vvvv[Va, Vb, va, vb], R.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('BCeF,AeFIJK->ABCIJK', H.ab.vvvv[Va, Vb, va, Vb], R.aab.VvVOOO, optimize=True)
            - 1.0 * np.einsum('BCEf,EAfIJK->ABCIJK', H.ab.vvvv[Va, Vb, Va, vb], R.aab.VVvOOO, optimize=True)
            - 1.0 * np.einsum('BCEF,EAFIJK->ABCIJK', H.ab.vvvv[Va, Vb, Va, Vb], R.aab.VVVOOO, optimize=True)
    )
    dR.aab.VVVOOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('AmIe,BeCmJK->ABCIJK', H.aa.voov[Va, oa, Oa, va], R.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('AMIe,BeCMJK->ABCIJK', H.aa.voov[Va, Oa, Oa, va], R.aab.VvVOOO, optimize=True)
            - 1.0 * np.einsum('AmIE,BECmJK->ABCIJK', H.aa.voov[Va, oa, Oa, Va], R.aab.VVVoOO, optimize=True)
            - 1.0 * np.einsum('AMIE,BECMJK->ABCIJK', H.aa.voov[Va, Oa, Oa, Va], R.aab.VVVOOO, optimize=True)
    )
    dR.aab.VVVOOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('AmIe,BCeJmK->ABCIJK', H.ab.voov[Va, ob, Oa, vb], R.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('AMIe,BCeJMK->ABCIJK', H.ab.voov[Va, Ob, Oa, vb], R.abb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('AmIE,BCEJmK->ABCIJK', H.ab.voov[Va, ob, Oa, Vb], R.abb.VVVOoO, optimize=True)
            - 1.0 * np.einsum('AMIE,BCEJMK->ABCIJK', H.ab.voov[Va, Ob, Oa, Vb], R.abb.VVVOOO, optimize=True)
    )
    dR.aab.VVVOOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('mCeK,BAemIJ->ABCIJK', H.ab.ovvo[oa, Vb, va, Ob], R.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('MCeK,BAeIJM->ABCIJK', H.ab.ovvo[Oa, Vb, va, Ob], R.aaa.VVvOOO, optimize=True)
            - 1.0 * np.einsum('mCEK,EBAmIJ->ABCIJK', H.ab.ovvo[oa, Vb, Va, Ob], R.aaa.VVVoOO, optimize=True)
            - 1.0 * np.einsum('MCEK,EBAIJM->ABCIJK', H.ab.ovvo[Oa, Vb, Va, Ob], R.aaa.VVVOOO, optimize=True)
    )
    dR.aab.VVVOOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('CmKe,BAeIJm->ABCIJK', H.bb.voov[Vb, ob, Ob, vb], R.aab.VVvOOo, optimize=True)
            - 1.0 * np.einsum('CMKe,BAeIJM->ABCIJK', H.bb.voov[Vb, Ob, Ob, vb], R.aab.VVvOOO, optimize=True)
            - 1.0 * np.einsum('CmKE,BAEIJm->ABCIJK', H.bb.voov[Vb, ob, Ob, Vb], R.aab.VVVOOo, optimize=True)
            - 1.0 * np.einsum('CMKE,BAEIJM->ABCIJK', H.bb.voov[Vb, Ob, Ob, Vb], R.aab.VVVOOO, optimize=True)
    )
    dR.aab.VVVOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('AmeK,BeCIJm->ABCIJK', H.ab.vovo[Va, ob, va, Ob], R.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('AMeK,BeCIJM->ABCIJK', H.ab.vovo[Va, Ob, va, Ob], R.aab.VvVOOO, optimize=True)
            + 1.0 * np.einsum('AmEK,BECIJm->ABCIJK', H.ab.vovo[Va, ob, Va, Ob], R.aab.VVVOOo, optimize=True)
            + 1.0 * np.einsum('AMEK,BECIJM->ABCIJK', H.ab.vovo[Va, Ob, Va, Ob], R.aab.VVVOOO, optimize=True)
    )
    dR.aab.VVVOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mCIe,BAemJK->ABCIJK', H.ab.ovov[oa, Vb, Oa, vb], R.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('MCIe,BAeMJK->ABCIJK', H.ab.ovov[Oa, Vb, Oa, vb], R.aab.VVvOOO, optimize=True)
            + 1.0 * np.einsum('mCIE,BAEmJK->ABCIJK', H.ab.ovov[oa, Vb, Oa, Vb], R.aab.VVVoOO, optimize=True)
            + 1.0 * np.einsum('MCIE,BAEMJK->ABCIJK', H.ab.ovov[Oa, Vb, Oa, Vb], R.aab.VVVOOO, optimize=True)
    )

    dR.aab.VVVOOO -= np.transpose(dR.aab.VVVOOO, (1, 0, 2, 3, 4, 5))
    dR.aab.VVVOOO -= np.transpose(dR.aab.VVVOOO, (0, 1, 2, 4, 3, 5))

    return dR

def update(R, omega, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    R.aab.VVVOOO = eomcc_active_loops.update_r3b_111111(
        R.aab.VVVOOO,
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
