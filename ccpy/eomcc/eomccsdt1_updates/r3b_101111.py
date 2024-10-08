import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops

def build(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    dR.aab.VvVOOO = (1.0 / 2.0) * (
            +1.0 * np.einsum('bCeK,AeIJ->AbCIJK', X.ab.vvvo[va, Vb, :, Ob], T.aa[Va, :, Oa, Oa], optimize=True)
    )
    dR.aab.VvVOOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('ACeK,beIJ->AbCIJK', X.ab.vvvo[Va, Vb, :, Ob], T.aa[va, :, Oa, Oa], optimize=True)
    )
    dR.aab.VvVOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mCJK,AbIm->AbCIJK', X.ab.ovoo[:, Vb, Oa, Ob], T.aa[Va, va, Oa, :], optimize=True)
    )
    dR.aab.VvVOOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('ACIe,beJK->AbCIJK', X.ab.vvov[Va, Vb, Oa, :], T.ab[va, :, Oa, Ob], optimize=True)
    )
    dR.aab.VvVOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('bCIe,AeJK->AbCIJK', X.ab.vvov[va, Vb, Oa, :], T.ab[Va, :, Oa, Ob], optimize=True)
    )
    dR.aab.VvVOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('AmIK,bCJm->AbCIJK', X.ab.vooo[Va, :, Oa, Ob], T.ab[va, Vb, Oa, :], optimize=True)
    )
    dR.aab.VvVOOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('bmIK,ACJm->AbCIJK', X.ab.vooo[va, :, Oa, Ob], T.ab[Va, Vb, Oa, :], optimize=True)
    )
    dR.aab.VvVOOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('AbIe,eCJK->AbCIJK', X.aa.vvov[Va, va, Oa, :], T.ab[:, Vb, Oa, Ob], optimize=True)
    )
    dR.aab.VvVOOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('AmIJ,bCmK->AbCIJK', X.aa.vooo[Va, :, Oa, Oa], T.ab[va, Vb, :, Ob], optimize=True)
    )
    dR.aab.VvVOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('bmIJ,ACmK->AbCIJK', X.aa.vooo[va, :, Oa, Oa], T.ab[Va, Vb, :, Ob], optimize=True)
    )
    dR.aab.VvVOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('bCeK,AeIJ->AbCIJK', H.ab.vvvo[va, Vb, :, Ob], R.aa[Va, :, Oa, Oa], optimize=True)
    )
    dR.aab.VvVOOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('ACeK,beIJ->AbCIJK', H.ab.vvvo[Va, Vb, :, Ob], R.aa[va, :, Oa, Oa], optimize=True)
    )
    dR.aab.VvVOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mCJK,AbIm->AbCIJK', H.ab.ovoo[:, Vb, Oa, Ob], R.aa[Va, va, Oa, :], optimize=True)
    )
    dR.aab.VvVOOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('ACIe,beJK->AbCIJK', H.ab.vvov[Va, Vb, Oa, :], R.ab[va, :, Oa, Ob], optimize=True)
    )
    dR.aab.VvVOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('bCIe,AeJK->AbCIJK', H.ab.vvov[va, Vb, Oa, :], R.ab[Va, :, Oa, Ob], optimize=True)
    )
    dR.aab.VvVOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('AmIK,bCJm->AbCIJK', H.ab.vooo[Va, :, Oa, Ob], R.ab[va, Vb, Oa, :], optimize=True)
    )
    dR.aab.VvVOOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('bmIK,ACJm->AbCIJK', H.ab.vooo[va, :, Oa, Ob], R.ab[Va, Vb, Oa, :], optimize=True)
    )
    dR.aab.VvVOOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('AbIe,eCJK->AbCIJK', H.aa.vvov[Va, va, Oa, :], R.ab[:, Vb, Oa, Ob], optimize=True)
    )
    dR.aab.VvVOOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('AmIJ,bCmK->AbCIJK', H.aa.vooo[Va, :, Oa, Oa], R.ab[va, Vb, :, Ob], optimize=True)
    )
    dR.aab.VvVOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('bmIJ,ACmK->AbCIJK', H.aa.vooo[va, :, Oa, Oa], R.ab[Va, Vb, :, Ob], optimize=True)
    )
    dR.aab.VvVOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mI,AbCmJK->AbCIJK', X.a.oo[oa, Oa], T.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('MI,AbCMJK->AbCIJK', X.a.oo[Oa, Oa], T.aab.VvVOOO, optimize=True)
    )
    dR.aab.VvVOOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mK,AbCIJm->AbCIJK', X.b.oo[ob, Ob], T.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('MK,AbCIJM->AbCIJK', X.b.oo[Ob, Ob], T.aab.VvVOOO, optimize=True)
    )
    dR.aab.VvVOOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('Ae,beCIJK->AbCIJK', X.a.vv[Va, va], T.aab.vvVOOO, optimize=True)
            + 1.0 * np.einsum('AE,EbCIJK->AbCIJK', X.a.vv[Va, Va], T.aab.VvVOOO, optimize=True)
    )
    dR.aab.VvVOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('be,AeCIJK->AbCIJK', X.a.vv[va, va], T.aab.VvVOOO, optimize=True)
            + 1.0 * np.einsum('bE,AECIJK->AbCIJK', X.a.vv[va, Va], T.aab.VVVOOO, optimize=True)
    )
    dR.aab.VvVOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('Ce,AbeIJK->AbCIJK', X.b.vv[Vb, vb], T.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('CE,AbEIJK->AbCIJK', X.b.vv[Vb, Vb], T.aab.VvVOOO, optimize=True)
    )
    dR.aab.VvVOOO += (1.0 / 2.0) * (
            +0.5 * np.einsum('mnIJ,AbCmnK->AbCIJK', X.aa.oooo[oa, oa, Oa, Oa], T.aab.VvVooO, optimize=True)
            - 1.0 * np.einsum('MnIJ,AbCnMK->AbCIJK', X.aa.oooo[Oa, oa, Oa, Oa], T.aab.VvVoOO, optimize=True)
            + 0.5 * np.einsum('MNIJ,AbCMNK->AbCIJK', X.aa.oooo[Oa, Oa, Oa, Oa], T.aab.VvVOOO, optimize=True)
    )
    dR.aab.VvVOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mnJK,AbCmIn->AbCIJK', X.ab.oooo[oa, ob, Oa, Ob], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('mNJK,AbCmIN->AbCIJK', X.ab.oooo[oa, Ob, Oa, Ob], T.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('MnJK,AbCIMn->AbCIJK', X.ab.oooo[Oa, ob, Oa, Ob], T.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('MNJK,AbCIMN->AbCIJK', X.ab.oooo[Oa, Ob, Oa, Ob], T.aab.VvVOOO, optimize=True)
    )
    dR.aab.VvVOOO += (1.0 / 2.0) * (
            -0.5 * np.einsum('Abef,feCIJK->AbCIJK', X.aa.vvvv[Va, va, va, va], T.aab.vvVOOO, optimize=True)
            - 1.0 * np.einsum('AbeF,FeCIJK->AbCIJK', X.aa.vvvv[Va, va, va, Va], T.aab.VvVOOO, optimize=True)
            - 0.5 * np.einsum('AbEF,FECIJK->AbCIJK', X.aa.vvvv[Va, va, Va, Va], T.aab.VVVOOO, optimize=True)
    )
    dR.aab.VvVOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('bCef,AefIJK->AbCIJK', X.ab.vvvv[va, Vb, va, vb], T.aab.VvvOOO, optimize=True)
            - 1.0 * np.einsum('bCEf,EAfIJK->AbCIJK', X.ab.vvvv[va, Vb, Va, vb], T.aab.VVvOOO, optimize=True)
            + 1.0 * np.einsum('bCeF,AeFIJK->AbCIJK', X.ab.vvvv[va, Vb, va, Vb], T.aab.VvVOOO, optimize=True)
            - 1.0 * np.einsum('bCEF,EAFIJK->AbCIJK', X.ab.vvvv[va, Vb, Va, Vb], T.aab.VVVOOO, optimize=True)
    )
    dR.aab.VvVOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('ACEf,EbfIJK->AbCIJK', X.ab.vvvv[Va, Vb, Va, vb], T.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('ACeF,ebFIJK->AbCIJK', X.ab.vvvv[Va, Vb, va, Vb], T.aab.vvVOOO, optimize=True)
            + 1.0 * np.einsum('ACEF,EbFIJK->AbCIJK', X.ab.vvvv[Va, Vb, Va, Vb], T.aab.VvVOOO, optimize=True)
    )
    dR.aab.VvVOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('AmIe,beCmJK->AbCIJK', X.aa.voov[Va, oa, Oa, va], T.aab.vvVoOO, optimize=True)
            + 1.0 * np.einsum('AmIE,EbCmJK->AbCIJK', X.aa.voov[Va, oa, Oa, Va], T.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('AMIe,beCMJK->AbCIJK', X.aa.voov[Va, Oa, Oa, va], T.aab.vvVOOO, optimize=True)
            + 1.0 * np.einsum('AMIE,EbCMJK->AbCIJK', X.aa.voov[Va, Oa, Oa, Va], T.aab.VvVOOO, optimize=True)
    )
    dR.aab.VvVOOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('bmIe,AeCmJK->AbCIJK', X.aa.voov[va, oa, Oa, va], T.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('bmIE,AECmJK->AbCIJK', X.aa.voov[va, oa, Oa, Va], T.aab.VVVoOO, optimize=True)
            + 1.0 * np.einsum('bMIe,AeCMJK->AbCIJK', X.aa.voov[va, Oa, Oa, va], T.aab.VvVOOO, optimize=True)
            + 1.0 * np.einsum('bMIE,AECMJK->AbCIJK', X.aa.voov[va, Oa, Oa, Va], T.aab.VVVOOO, optimize=True)
    )
    dR.aab.VvVOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('AmIe,bCeJmK->AbCIJK', X.ab.voov[Va, ob, Oa, vb], T.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('AmIE,bCEJmK->AbCIJK', X.ab.voov[Va, ob, Oa, Vb], T.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('AMIe,bCeJMK->AbCIJK', X.ab.voov[Va, Ob, Oa, vb], T.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('AMIE,bCEJMK->AbCIJK', X.ab.voov[Va, Ob, Oa, Vb], T.abb.vVVOOO, optimize=True)
    )
    dR.aab.VvVOOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('bmIe,ACeJmK->AbCIJK', X.ab.voov[va, ob, Oa, vb], T.abb.VVvOoO, optimize=True)
            + 1.0 * np.einsum('bmIE,ACEJmK->AbCIJK', X.ab.voov[va, ob, Oa, Vb], T.abb.VVVOoO, optimize=True)
            + 1.0 * np.einsum('bMIe,ACeJMK->AbCIJK', X.ab.voov[va, Ob, Oa, vb], T.abb.VVvOOO, optimize=True)
            + 1.0 * np.einsum('bMIE,ACEJMK->AbCIJK', X.ab.voov[va, Ob, Oa, Vb], T.abb.VVVOOO, optimize=True)
    )
    dR.aab.VvVOOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mCeK,AebmIJ->AbCIJK', X.ab.ovvo[oa, Vb, va, Ob], T.aaa.VvvoOO, optimize=True)
            + 1.0 * np.einsum('mCEK,EAbmIJ->AbCIJK', X.ab.ovvo[oa, Vb, Va, Ob], T.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('MCeK,AebIJM->AbCIJK', X.ab.ovvo[Oa, Vb, va, Ob], T.aaa.VvvOOO, optimize=True)
            + 1.0 * np.einsum('MCEK,EAbIJM->AbCIJK', X.ab.ovvo[Oa, Vb, Va, Ob], T.aaa.VVvOOO, optimize=True)
    )
    dR.aab.VvVOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('CmKe,AbeIJm->AbCIJK', X.bb.voov[Vb, ob, Ob, vb], T.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('CmKE,AbEIJm->AbCIJK', X.bb.voov[Vb, ob, Ob, Vb], T.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('CMKe,AbeIJM->AbCIJK', X.bb.voov[Vb, Ob, Ob, vb], T.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('CMKE,AbEIJM->AbCIJK', X.bb.voov[Vb, Ob, Ob, Vb], T.aab.VvVOOO, optimize=True)
    )
    dR.aab.VvVOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('AmeK,beCIJm->AbCIJK', X.ab.vovo[Va, ob, va, Ob], T.aab.vvVOOo, optimize=True)
            - 1.0 * np.einsum('AmEK,EbCIJm->AbCIJK', X.ab.vovo[Va, ob, Va, Ob], T.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('AMeK,beCIJM->AbCIJK', X.ab.vovo[Va, Ob, va, Ob], T.aab.vvVOOO, optimize=True)
            - 1.0 * np.einsum('AMEK,EbCIJM->AbCIJK', X.ab.vovo[Va, Ob, Va, Ob], T.aab.VvVOOO, optimize=True)
    )
    dR.aab.VvVOOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('bmeK,AeCIJm->AbCIJK', X.ab.vovo[va, ob, va, Ob], T.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('bmEK,AECIJm->AbCIJK', X.ab.vovo[va, ob, Va, Ob], T.aab.VVVOOo, optimize=True)
            - 1.0 * np.einsum('bMeK,AeCIJM->AbCIJK', X.ab.vovo[va, Ob, va, Ob], T.aab.VvVOOO, optimize=True)
            - 1.0 * np.einsum('bMEK,AECIJM->AbCIJK', X.ab.vovo[va, Ob, Va, Ob], T.aab.VVVOOO, optimize=True)
    )
    dR.aab.VvVOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mCIe,AbemJK->AbCIJK', X.ab.ovov[oa, Vb, Oa, vb], T.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('mCIE,AbEmJK->AbCIJK', X.ab.ovov[oa, Vb, Oa, Vb], T.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('MCIe,AbeMJK->AbCIJK', X.ab.ovov[Oa, Vb, Oa, vb], T.aab.VvvOOO, optimize=True)
            - 1.0 * np.einsum('MCIE,AbEMJK->AbCIJK', X.ab.ovov[Oa, Vb, Oa, Vb], T.aab.VvVOOO, optimize=True)
    )
    dR.aab.VvVOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mI,AbCmJK->AbCIJK', H.a.oo[oa, Oa], R.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('MI,AbCMJK->AbCIJK', H.a.oo[Oa, Oa], R.aab.VvVOOO, optimize=True)
    )
    dR.aab.VvVOOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mK,AbCIJm->AbCIJK', H.b.oo[ob, Ob], R.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('MK,AbCIJM->AbCIJK', H.b.oo[Ob, Ob], R.aab.VvVOOO, optimize=True)
    )
    dR.aab.VvVOOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('Ae,beCIJK->AbCIJK', H.a.vv[Va, va], R.aab.vvVOOO, optimize=True)
            + 1.0 * np.einsum('AE,EbCIJK->AbCIJK', H.a.vv[Va, Va], R.aab.VvVOOO, optimize=True)
    )
    dR.aab.VvVOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('be,AeCIJK->AbCIJK', H.a.vv[va, va], R.aab.VvVOOO, optimize=True)
            + 1.0 * np.einsum('bE,AECIJK->AbCIJK', H.a.vv[va, Va], R.aab.VVVOOO, optimize=True)
    )
    dR.aab.VvVOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('Ce,AbeIJK->AbCIJK', H.b.vv[Vb, vb], R.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('CE,AbEIJK->AbCIJK', H.b.vv[Vb, Vb], R.aab.VvVOOO, optimize=True)
    )
    dR.aab.VvVOOO += (1.0 / 2.0) * (
            +0.5 * np.einsum('mnIJ,AbCmnK->AbCIJK', H.aa.oooo[oa, oa, Oa, Oa], R.aab.VvVooO, optimize=True)
            - 1.0 * np.einsum('MnIJ,AbCnMK->AbCIJK', H.aa.oooo[Oa, oa, Oa, Oa], R.aab.VvVoOO, optimize=True)
            + 0.5 * np.einsum('MNIJ,AbCMNK->AbCIJK', H.aa.oooo[Oa, Oa, Oa, Oa], R.aab.VvVOOO, optimize=True)
    )
    dR.aab.VvVOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mnJK,AbCmIn->AbCIJK', H.ab.oooo[oa, ob, Oa, Ob], R.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('mNJK,AbCmIN->AbCIJK', H.ab.oooo[oa, Ob, Oa, Ob], R.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('MnJK,AbCIMn->AbCIJK', H.ab.oooo[Oa, ob, Oa, Ob], R.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('MNJK,AbCIMN->AbCIJK', H.ab.oooo[Oa, Ob, Oa, Ob], R.aab.VvVOOO, optimize=True)
    )
    dR.aab.VvVOOO += (1.0 / 2.0) * (
            -0.5 * np.einsum('Abef,feCIJK->AbCIJK', H.aa.vvvv[Va, va, va, va], R.aab.vvVOOO, optimize=True)
            - 1.0 * np.einsum('AbeF,FeCIJK->AbCIJK', H.aa.vvvv[Va, va, va, Va], R.aab.VvVOOO, optimize=True)
            - 0.5 * np.einsum('AbEF,FECIJK->AbCIJK', H.aa.vvvv[Va, va, Va, Va], R.aab.VVVOOO, optimize=True)
    )
    dR.aab.VvVOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('bCef,AefIJK->AbCIJK', H.ab.vvvv[va, Vb, va, vb], R.aab.VvvOOO, optimize=True)
            - 1.0 * np.einsum('bCEf,EAfIJK->AbCIJK', H.ab.vvvv[va, Vb, Va, vb], R.aab.VVvOOO, optimize=True)
            + 1.0 * np.einsum('bCeF,AeFIJK->AbCIJK', H.ab.vvvv[va, Vb, va, Vb], R.aab.VvVOOO, optimize=True)
            - 1.0 * np.einsum('bCEF,EAFIJK->AbCIJK', H.ab.vvvv[va, Vb, Va, Vb], R.aab.VVVOOO, optimize=True)
    )
    dR.aab.VvVOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('ACEf,EbfIJK->AbCIJK', H.ab.vvvv[Va, Vb, Va, vb], R.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('ACeF,ebFIJK->AbCIJK', H.ab.vvvv[Va, Vb, va, Vb], R.aab.vvVOOO, optimize=True)
            + 1.0 * np.einsum('ACEF,EbFIJK->AbCIJK', H.ab.vvvv[Va, Vb, Va, Vb], R.aab.VvVOOO, optimize=True)
    )
    dR.aab.VvVOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('AmIe,beCmJK->AbCIJK', H.aa.voov[Va, oa, Oa, va], R.aab.vvVoOO, optimize=True)
            + 1.0 * np.einsum('AmIE,EbCmJK->AbCIJK', H.aa.voov[Va, oa, Oa, Va], R.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('AMIe,beCMJK->AbCIJK', H.aa.voov[Va, Oa, Oa, va], R.aab.vvVOOO, optimize=True)
            + 1.0 * np.einsum('AMIE,EbCMJK->AbCIJK', H.aa.voov[Va, Oa, Oa, Va], R.aab.VvVOOO, optimize=True)
    )
    dR.aab.VvVOOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('bmIe,AeCmJK->AbCIJK', H.aa.voov[va, oa, Oa, va], R.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('bmIE,AECmJK->AbCIJK', H.aa.voov[va, oa, Oa, Va], R.aab.VVVoOO, optimize=True)
            + 1.0 * np.einsum('bMIe,AeCMJK->AbCIJK', H.aa.voov[va, Oa, Oa, va], R.aab.VvVOOO, optimize=True)
            + 1.0 * np.einsum('bMIE,AECMJK->AbCIJK', H.aa.voov[va, Oa, Oa, Va], R.aab.VVVOOO, optimize=True)
    )
    dR.aab.VvVOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('AmIe,bCeJmK->AbCIJK', H.ab.voov[Va, ob, Oa, vb], R.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('AmIE,bCEJmK->AbCIJK', H.ab.voov[Va, ob, Oa, Vb], R.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('AMIe,bCeJMK->AbCIJK', H.ab.voov[Va, Ob, Oa, vb], R.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('AMIE,bCEJMK->AbCIJK', H.ab.voov[Va, Ob, Oa, Vb], R.abb.vVVOOO, optimize=True)
    )
    dR.aab.VvVOOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('bmIe,ACeJmK->AbCIJK', H.ab.voov[va, ob, Oa, vb], R.abb.VVvOoO, optimize=True)
            + 1.0 * np.einsum('bmIE,ACEJmK->AbCIJK', H.ab.voov[va, ob, Oa, Vb], R.abb.VVVOoO, optimize=True)
            + 1.0 * np.einsum('bMIe,ACeJMK->AbCIJK', H.ab.voov[va, Ob, Oa, vb], R.abb.VVvOOO, optimize=True)
            + 1.0 * np.einsum('bMIE,ACEJMK->AbCIJK', H.ab.voov[va, Ob, Oa, Vb], R.abb.VVVOOO, optimize=True)
    )
    dR.aab.VvVOOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mCeK,AebmIJ->AbCIJK', H.ab.ovvo[oa, Vb, va, Ob], R.aaa.VvvoOO, optimize=True)
            + 1.0 * np.einsum('mCEK,EAbmIJ->AbCIJK', H.ab.ovvo[oa, Vb, Va, Ob], R.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('MCeK,AebIJM->AbCIJK', H.ab.ovvo[Oa, Vb, va, Ob], R.aaa.VvvOOO, optimize=True)
            + 1.0 * np.einsum('MCEK,EAbIJM->AbCIJK', H.ab.ovvo[Oa, Vb, Va, Ob], R.aaa.VVvOOO, optimize=True)
    )
    dR.aab.VvVOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('CmKe,AbeIJm->AbCIJK', H.bb.voov[Vb, ob, Ob, vb], R.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('CmKE,AbEIJm->AbCIJK', H.bb.voov[Vb, ob, Ob, Vb], R.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('CMKe,AbeIJM->AbCIJK', H.bb.voov[Vb, Ob, Ob, vb], R.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('CMKE,AbEIJM->AbCIJK', H.bb.voov[Vb, Ob, Ob, Vb], R.aab.VvVOOO, optimize=True)
    )
    dR.aab.VvVOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('AmeK,beCIJm->AbCIJK', H.ab.vovo[Va, ob, va, Ob], R.aab.vvVOOo, optimize=True)
            - 1.0 * np.einsum('AmEK,EbCIJm->AbCIJK', H.ab.vovo[Va, ob, Va, Ob], R.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('AMeK,beCIJM->AbCIJK', H.ab.vovo[Va, Ob, va, Ob], R.aab.vvVOOO, optimize=True)
            - 1.0 * np.einsum('AMEK,EbCIJM->AbCIJK', H.ab.vovo[Va, Ob, Va, Ob], R.aab.VvVOOO, optimize=True)
    )
    dR.aab.VvVOOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('bmeK,AeCIJm->AbCIJK', H.ab.vovo[va, ob, va, Ob], R.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('bmEK,AECIJm->AbCIJK', H.ab.vovo[va, ob, Va, Ob], R.aab.VVVOOo, optimize=True)
            - 1.0 * np.einsum('bMeK,AeCIJM->AbCIJK', H.ab.vovo[va, Ob, va, Ob], R.aab.VvVOOO, optimize=True)
            - 1.0 * np.einsum('bMEK,AECIJM->AbCIJK', H.ab.vovo[va, Ob, Va, Ob], R.aab.VVVOOO, optimize=True)
    )
    dR.aab.VvVOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mCIe,AbemJK->AbCIJK', H.ab.ovov[oa, Vb, Oa, vb], R.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('mCIE,AbEmJK->AbCIJK', H.ab.ovov[oa, Vb, Oa, Vb], R.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('MCIe,AbeMJK->AbCIJK', H.ab.ovov[Oa, Vb, Oa, vb], R.aab.VvvOOO, optimize=True)
            - 1.0 * np.einsum('MCIE,AbEMJK->AbCIJK', H.ab.ovov[Oa, Vb, Oa, Vb], R.aab.VvVOOO, optimize=True)
    )

    dR.aab.VvVOOO -= np.transpose(dR.aab.VvVOOO, (0, 1, 2, 4, 3, 5))

    return dR

def update(R, omega, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    R.aab.VvVOOO = eomcc_active_loops.update_r3b_101111(
        R.aab.VvVOOO,
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
