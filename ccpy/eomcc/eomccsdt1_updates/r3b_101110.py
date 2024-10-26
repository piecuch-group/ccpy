import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops

def build(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.aab.VvVOOo = (1.0 / 2.0) * (
            +1.0 * np.einsum('bCek,AeIJ->AbCIJk', X.ab.vvvo[va, Vb, :, ob], T.aa[Va, :, Oa, Oa], optimize=True)
    )
    dR.aab.VvVOOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('ACek,beIJ->AbCIJk', X.ab.vvvo[Va, Vb, :, ob], T.aa[va, :, Oa, Oa], optimize=True)
    )
    dR.aab.VvVOOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('mCJk,AbIm->AbCIJk', X.ab.ovoo[:, Vb, Oa, ob], T.aa[Va, va, Oa, :], optimize=True)
    )
    dR.aab.VvVOOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('ACIe,beJk->AbCIJk', X.ab.vvov[Va, Vb, Oa, :], T.ab[va, :, Oa, ob], optimize=True)
    )
    dR.aab.VvVOOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('bCIe,AeJk->AbCIJk', X.ab.vvov[va, Vb, Oa, :], T.ab[Va, :, Oa, ob], optimize=True)
    )
    dR.aab.VvVOOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('AmIk,bCJm->AbCIJk', X.ab.vooo[Va, :, Oa, ob], T.ab[va, Vb, Oa, :], optimize=True)
    )
    dR.aab.VvVOOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('bmIk,ACJm->AbCIJk', X.ab.vooo[va, :, Oa, ob], T.ab[Va, Vb, Oa, :], optimize=True)
    )
    dR.aab.VvVOOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('AbIe,eCJk->AbCIJk', X.aa.vvov[Va, va, Oa, :], T.ab[:, Vb, Oa, ob], optimize=True)
    )
    dR.aab.VvVOOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('AmIJ,bCmk->AbCIJk', X.aa.vooo[Va, :, Oa, Oa], T.ab[va, Vb, :, ob], optimize=True)
    )
    dR.aab.VvVOOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('bmIJ,ACmk->AbCIJk', X.aa.vooo[va, :, Oa, Oa], T.ab[Va, Vb, :, ob], optimize=True)
    )
    dR.aab.VvVOOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('bCek,AeIJ->AbCIJk', H.ab.vvvo[va, Vb, :, ob], R.aa[Va, :, Oa, Oa], optimize=True)
    )
    dR.aab.VvVOOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('ACek,beIJ->AbCIJk', H.ab.vvvo[Va, Vb, :, ob], R.aa[va, :, Oa, Oa], optimize=True)
    )
    dR.aab.VvVOOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('mCJk,AbIm->AbCIJk', H.ab.ovoo[:, Vb, Oa, ob], R.aa[Va, va, Oa, :], optimize=True)
    )
    dR.aab.VvVOOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('ACIe,beJk->AbCIJk', H.ab.vvov[Va, Vb, Oa, :], R.ab[va, :, Oa, ob], optimize=True)
    )
    dR.aab.VvVOOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('bCIe,AeJk->AbCIJk', H.ab.vvov[va, Vb, Oa, :], R.ab[Va, :, Oa, ob], optimize=True)
    )
    dR.aab.VvVOOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('AmIk,bCJm->AbCIJk', H.ab.vooo[Va, :, Oa, ob], R.ab[va, Vb, Oa, :], optimize=True)
    )
    dR.aab.VvVOOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('bmIk,ACJm->AbCIJk', H.ab.vooo[va, :, Oa, ob], R.ab[Va, Vb, Oa, :], optimize=True)
    )
    dR.aab.VvVOOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('AbIe,eCJk->AbCIJk', H.aa.vvov[Va, va, Oa, :], R.ab[:, Vb, Oa, ob], optimize=True)
    )
    dR.aab.VvVOOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('AmIJ,bCmk->AbCIJk', H.aa.vooo[Va, :, Oa, Oa], R.ab[va, Vb, :, ob], optimize=True)
    )
    dR.aab.VvVOOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('bmIJ,ACmk->AbCIJk', H.aa.vooo[va, :, Oa, Oa], R.ab[Va, Vb, :, ob], optimize=True)
    )

    dR.aab.VvVOOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('mI,AbCmJk->AbCIJk', X.a.oo[oa, Oa], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('MI,AbCMJk->AbCIJk', X.a.oo[Oa, Oa], T.aab.VvVOOo, optimize=True)
    )
    dR.aab.VvVOOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('mk,AbCIJm->AbCIJk', X.b.oo[ob, ob], T.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('Mk,AbCIJM->AbCIJk', X.b.oo[Ob, ob], T.aab.VvVOOO, optimize=True)
    )
    dR.aab.VvVOOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('Ae,beCIJk->AbCIJk', X.a.vv[Va, va], T.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('AE,EbCIJk->AbCIJk', X.a.vv[Va, Va], T.aab.VvVOOo, optimize=True)
    )
    dR.aab.VvVOOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('be,AeCIJk->AbCIJk', X.a.vv[va, va], T.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('bE,AECIJk->AbCIJk', X.a.vv[va, Va], T.aab.VVVOOo, optimize=True)
    )
    dR.aab.VvVOOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('Ce,AbeIJk->AbCIJk', X.b.vv[Vb, vb], T.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('CE,AbEIJk->AbCIJk', X.b.vv[Vb, Vb], T.aab.VvVOOo, optimize=True)
    )
    dR.aab.VvVOOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('mNIJ,AbCmNk->AbCIJk', X.aa.oooo[oa, Oa, Oa, Oa], T.aab.VvVoOo, optimize=True)
            + 0.5 * np.einsum('MNIJ,AbCMNk->AbCIJk', X.aa.oooo[Oa, Oa, Oa, Oa], T.aab.VvVOOo, optimize=True)
    )
    dR.aab.VvVOOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('mnJk,AbCmIn->AbCIJk', X.ab.oooo[oa, ob, Oa, ob], T.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('MnJk,AbCIMn->AbCIJk', X.ab.oooo[Oa, ob, Oa, ob], T.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('mNJk,AbCmIN->AbCIJk', X.ab.oooo[oa, Ob, Oa, ob], T.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('MNJk,AbCIMN->AbCIJk', X.ab.oooo[Oa, Ob, Oa, ob], T.aab.VvVOOO, optimize=True)
    )
    dR.aab.VvVOOo += (1.0 / 2.0) * (
            -0.5 * np.einsum('Abef,feCIJk->AbCIJk', X.aa.vvvv[Va, va, va, va], T.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('AbEf,EfCIJk->AbCIJk', X.aa.vvvv[Va, va, Va, va], T.aab.VvVOOo, optimize=True)
            - 0.5 * np.einsum('AbEF,FECIJk->AbCIJk', X.aa.vvvv[Va, va, Va, Va], T.aab.VVVOOo, optimize=True)
    )
    dR.aab.VvVOOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('bCef,AefIJk->AbCIJk', X.ab.vvvv[va, Vb, va, vb], T.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('bCeF,AeFIJk->AbCIJk', X.ab.vvvv[va, Vb, va, Vb], T.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('bCEf,EAfIJk->AbCIJk', X.ab.vvvv[va, Vb, Va, vb], T.aab.VVvOOo, optimize=True)
            - 1.0 * np.einsum('bCEF,EAFIJk->AbCIJk', X.ab.vvvv[va, Vb, Va, Vb], T.aab.VVVOOo, optimize=True)
    )
    dR.aab.VvVOOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('ACeF,ebFIJk->AbCIJk', X.ab.vvvv[Va, Vb, va, Vb], T.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('ACEf,EbfIJk->AbCIJk', X.ab.vvvv[Va, Vb, Va, vb], T.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('ACEF,EbFIJk->AbCIJk', X.ab.vvvv[Va, Vb, Va, Vb], T.aab.VvVOOo, optimize=True)
    )
    dR.aab.VvVOOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('AmIe,beCmJk->AbCIJk', X.aa.voov[Va, oa, Oa, va], T.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('AMIe,beCMJk->AbCIJk', X.aa.voov[Va, Oa, Oa, va], T.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('AmIE,EbCmJk->AbCIJk', X.aa.voov[Va, oa, Oa, Va], T.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('AMIE,EbCMJk->AbCIJk', X.aa.voov[Va, Oa, Oa, Va], T.aab.VvVOOo, optimize=True)
    )
    dR.aab.VvVOOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('bmIe,AeCmJk->AbCIJk', X.aa.voov[va, oa, Oa, va], T.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('bMIe,AeCMJk->AbCIJk', X.aa.voov[va, Oa, Oa, va], T.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('bmIE,AECmJk->AbCIJk', X.aa.voov[va, oa, Oa, Va], T.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('bMIE,AECMJk->AbCIJk', X.aa.voov[va, Oa, Oa, Va], T.aab.VVVOOo, optimize=True)
    )
    dR.aab.VvVOOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('AmIe,bCeJmk->AbCIJk', X.ab.voov[Va, ob, Oa, vb], T.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('AMIe,bCeJkM->AbCIJk', X.ab.voov[Va, Ob, Oa, vb], T.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('AmIE,bCEJmk->AbCIJk', X.ab.voov[Va, ob, Oa, Vb], T.abb.vVVOoo, optimize=True)
            + 1.0 * np.einsum('AMIE,bCEJkM->AbCIJk', X.ab.voov[Va, Ob, Oa, Vb], T.abb.vVVOoO, optimize=True)
    )
    dR.aab.VvVOOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('bmIe,ACeJmk->AbCIJk', X.ab.voov[va, ob, Oa, vb], T.abb.VVvOoo, optimize=True)
            - 1.0 * np.einsum('bMIe,ACeJkM->AbCIJk', X.ab.voov[va, Ob, Oa, vb], T.abb.VVvOoO, optimize=True)
            + 1.0 * np.einsum('bmIE,ACEJmk->AbCIJk', X.ab.voov[va, ob, Oa, Vb], T.abb.VVVOoo, optimize=True)
            - 1.0 * np.einsum('bMIE,ACEJkM->AbCIJk', X.ab.voov[va, Ob, Oa, Vb], T.abb.VVVOoO, optimize=True)
    )
    dR.aab.VvVOOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('mCek,AebmIJ->AbCIJk', X.ab.ovvo[oa, Vb, va, ob], T.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('MCek,AebIJM->AbCIJk', X.ab.ovvo[Oa, Vb, va, ob], T.aaa.VvvOOO, optimize=True)
            + 1.0 * np.einsum('mCEk,EAbmIJ->AbCIJk', X.ab.ovvo[oa, Vb, Va, ob], T.aaa.VVvoOO, optimize=True)
            + 1.0 * np.einsum('MCEk,EAbIJM->AbCIJk', X.ab.ovvo[Oa, Vb, Va, ob], T.aaa.VVvOOO, optimize=True)
    )
    dR.aab.VvVOOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('Cmke,AbeIJm->AbCIJk', X.bb.voov[Vb, ob, ob, vb], T.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('CMke,AbeIJM->AbCIJk', X.bb.voov[Vb, Ob, ob, vb], T.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('CmkE,AbEIJm->AbCIJk', X.bb.voov[Vb, ob, ob, Vb], T.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('CMkE,AbEIJM->AbCIJk', X.bb.voov[Vb, Ob, ob, Vb], T.aab.VvVOOO, optimize=True)
    )
    dR.aab.VvVOOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('Amek,beCIJm->AbCIJk', X.ab.vovo[Va, ob, va, ob], T.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('AMek,beCIJM->AbCIJk', X.ab.vovo[Va, Ob, va, ob], T.aab.vvVOOO, optimize=True)
            - 1.0 * np.einsum('AmEk,EbCIJm->AbCIJk', X.ab.vovo[Va, ob, Va, ob], T.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('AMEk,EbCIJM->AbCIJk', X.ab.vovo[Va, Ob, Va, ob], T.aab.VvVOOO, optimize=True)
    )
    dR.aab.VvVOOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('bmek,AeCIJm->AbCIJk', X.ab.vovo[va, ob, va, ob], T.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('bMek,AeCIJM->AbCIJk', X.ab.vovo[va, Ob, va, ob], T.aab.VvVOOO, optimize=True)
            - 1.0 * np.einsum('bmEk,AECIJm->AbCIJk', X.ab.vovo[va, ob, Va, ob], T.aab.VVVOOo, optimize=True)
            - 1.0 * np.einsum('bMEk,AECIJM->AbCIJk', X.ab.vovo[va, Ob, Va, ob], T.aab.VVVOOO, optimize=True)
    )
    dR.aab.VvVOOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('mCIe,AbemJk->AbCIJk', X.ab.ovov[oa, Vb, Oa, vb], T.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('MCIe,AbeMJk->AbCIJk', X.ab.ovov[Oa, Vb, Oa, vb], T.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('mCIE,AbEmJk->AbCIJk', X.ab.ovov[oa, Vb, Oa, Vb], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('MCIE,AbEMJk->AbCIJk', X.ab.ovov[Oa, Vb, Oa, Vb], T.aab.VvVOOo, optimize=True)
    )
    dR.aab.VvVOOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('mI,AbCmJk->AbCIJk', H.a.oo[oa, Oa], R.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('MI,AbCMJk->AbCIJk', H.a.oo[Oa, Oa], R.aab.VvVOOo, optimize=True)
    )
    dR.aab.VvVOOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('mk,AbCIJm->AbCIJk', H.b.oo[ob, ob], R.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('Mk,AbCIJM->AbCIJk', H.b.oo[Ob, ob], R.aab.VvVOOO, optimize=True)
    )
    dR.aab.VvVOOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('Ae,beCIJk->AbCIJk', H.a.vv[Va, va], R.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('AE,EbCIJk->AbCIJk', H.a.vv[Va, Va], R.aab.VvVOOo, optimize=True)
    )
    dR.aab.VvVOOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('be,AeCIJk->AbCIJk', H.a.vv[va, va], R.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('bE,AECIJk->AbCIJk', H.a.vv[va, Va], R.aab.VVVOOo, optimize=True)
    )
    dR.aab.VvVOOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('Ce,AbeIJk->AbCIJk', H.b.vv[Vb, vb], R.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('CE,AbEIJk->AbCIJk', H.b.vv[Vb, Vb], R.aab.VvVOOo, optimize=True)
    )
    dR.aab.VvVOOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('mNIJ,AbCmNk->AbCIJk', H.aa.oooo[oa, Oa, Oa, Oa], R.aab.VvVoOo, optimize=True)
            + 0.5 * np.einsum('MNIJ,AbCMNk->AbCIJk', H.aa.oooo[Oa, Oa, Oa, Oa], R.aab.VvVOOo, optimize=True)
    )
    dR.aab.VvVOOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('mnJk,AbCmIn->AbCIJk', H.ab.oooo[oa, ob, Oa, ob], R.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('MnJk,AbCIMn->AbCIJk', H.ab.oooo[Oa, ob, Oa, ob], R.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('mNJk,AbCmIN->AbCIJk', H.ab.oooo[oa, Ob, Oa, ob], R.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('MNJk,AbCIMN->AbCIJk', H.ab.oooo[Oa, Ob, Oa, ob], R.aab.VvVOOO, optimize=True)
    )
    dR.aab.VvVOOo += (1.0 / 2.0) * (
            -0.5 * np.einsum('Abef,feCIJk->AbCIJk', H.aa.vvvv[Va, va, va, va], R.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('AbEf,EfCIJk->AbCIJk', H.aa.vvvv[Va, va, Va, va], R.aab.VvVOOo, optimize=True)
            - 0.5 * np.einsum('AbEF,FECIJk->AbCIJk', H.aa.vvvv[Va, va, Va, Va], R.aab.VVVOOo, optimize=True)
    )
    dR.aab.VvVOOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('bCef,AefIJk->AbCIJk', H.ab.vvvv[va, Vb, va, vb], R.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('bCeF,AeFIJk->AbCIJk', H.ab.vvvv[va, Vb, va, Vb], R.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('bCEf,EAfIJk->AbCIJk', H.ab.vvvv[va, Vb, Va, vb], R.aab.VVvOOo, optimize=True)
            - 1.0 * np.einsum('bCEF,EAFIJk->AbCIJk', H.ab.vvvv[va, Vb, Va, Vb], R.aab.VVVOOo, optimize=True)
    )
    dR.aab.VvVOOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('ACeF,ebFIJk->AbCIJk', H.ab.vvvv[Va, Vb, va, Vb], R.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('ACEf,EbfIJk->AbCIJk', H.ab.vvvv[Va, Vb, Va, vb], R.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('ACEF,EbFIJk->AbCIJk', H.ab.vvvv[Va, Vb, Va, Vb], R.aab.VvVOOo, optimize=True)
    )
    dR.aab.VvVOOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('AmIe,beCmJk->AbCIJk', H.aa.voov[Va, oa, Oa, va], R.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('AMIe,beCMJk->AbCIJk', H.aa.voov[Va, Oa, Oa, va], R.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('AmIE,EbCmJk->AbCIJk', H.aa.voov[Va, oa, Oa, Va], R.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('AMIE,EbCMJk->AbCIJk', H.aa.voov[Va, Oa, Oa, Va], R.aab.VvVOOo, optimize=True)
    )
    dR.aab.VvVOOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('bmIe,AeCmJk->AbCIJk', H.aa.voov[va, oa, Oa, va], R.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('bMIe,AeCMJk->AbCIJk', H.aa.voov[va, Oa, Oa, va], R.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('bmIE,AECmJk->AbCIJk', H.aa.voov[va, oa, Oa, Va], R.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('bMIE,AECMJk->AbCIJk', H.aa.voov[va, Oa, Oa, Va], R.aab.VVVOOo, optimize=True)
    )
    dR.aab.VvVOOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('AmIe,bCeJmk->AbCIJk', H.ab.voov[Va, ob, Oa, vb], R.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('AMIe,bCeJkM->AbCIJk', H.ab.voov[Va, Ob, Oa, vb], R.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('AmIE,bCEJmk->AbCIJk', H.ab.voov[Va, ob, Oa, Vb], R.abb.vVVOoo, optimize=True)
            + 1.0 * np.einsum('AMIE,bCEJkM->AbCIJk', H.ab.voov[Va, Ob, Oa, Vb], R.abb.vVVOoO, optimize=True)
    )
    dR.aab.VvVOOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('bmIe,ACeJmk->AbCIJk', H.ab.voov[va, ob, Oa, vb], R.abb.VVvOoo, optimize=True)
            - 1.0 * np.einsum('bMIe,ACeJkM->AbCIJk', H.ab.voov[va, Ob, Oa, vb], R.abb.VVvOoO, optimize=True)
            + 1.0 * np.einsum('bmIE,ACEJmk->AbCIJk', H.ab.voov[va, ob, Oa, Vb], R.abb.VVVOoo, optimize=True)
            - 1.0 * np.einsum('bMIE,ACEJkM->AbCIJk', H.ab.voov[va, Ob, Oa, Vb], R.abb.VVVOoO, optimize=True)
    )
    dR.aab.VvVOOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('mCek,AebmIJ->AbCIJk', H.ab.ovvo[oa, Vb, va, ob], R.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('MCek,AebIJM->AbCIJk', H.ab.ovvo[Oa, Vb, va, ob], R.aaa.VvvOOO, optimize=True)
            + 1.0 * np.einsum('mCEk,EAbmIJ->AbCIJk', H.ab.ovvo[oa, Vb, Va, ob], R.aaa.VVvoOO, optimize=True)
            + 1.0 * np.einsum('MCEk,EAbIJM->AbCIJk', H.ab.ovvo[Oa, Vb, Va, ob], R.aaa.VVvOOO, optimize=True)
    )
    dR.aab.VvVOOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('Cmke,AbeIJm->AbCIJk', H.bb.voov[Vb, ob, ob, vb], R.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('CMke,AbeIJM->AbCIJk', H.bb.voov[Vb, Ob, ob, vb], R.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('CmkE,AbEIJm->AbCIJk', H.bb.voov[Vb, ob, ob, Vb], R.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('CMkE,AbEIJM->AbCIJk', H.bb.voov[Vb, Ob, ob, Vb], R.aab.VvVOOO, optimize=True)
    )
    dR.aab.VvVOOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('Amek,beCIJm->AbCIJk', H.ab.vovo[Va, ob, va, ob], R.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('AMek,beCIJM->AbCIJk', H.ab.vovo[Va, Ob, va, ob], R.aab.vvVOOO, optimize=True)
            - 1.0 * np.einsum('AmEk,EbCIJm->AbCIJk', H.ab.vovo[Va, ob, Va, ob], R.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('AMEk,EbCIJM->AbCIJk', H.ab.vovo[Va, Ob, Va, ob], R.aab.VvVOOO, optimize=True)
    )
    dR.aab.VvVOOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('bmek,AeCIJm->AbCIJk', H.ab.vovo[va, ob, va, ob], R.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('bMek,AeCIJM->AbCIJk', H.ab.vovo[va, Ob, va, ob], R.aab.VvVOOO, optimize=True)
            - 1.0 * np.einsum('bmEk,AECIJm->AbCIJk', H.ab.vovo[va, ob, Va, ob], R.aab.VVVOOo, optimize=True)
            - 1.0 * np.einsum('bMEk,AECIJM->AbCIJk', H.ab.vovo[va, Ob, Va, ob], R.aab.VVVOOO, optimize=True)
    )
    dR.aab.VvVOOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('mCIe,AbemJk->AbCIJk', H.ab.ovov[oa, Vb, Oa, vb], R.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('MCIe,AbeMJk->AbCIJk', H.ab.ovov[Oa, Vb, Oa, vb], R.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('mCIE,AbEmJk->AbCIJk', H.ab.ovov[oa, Vb, Oa, Vb], R.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('MCIE,AbEMJk->AbCIJk', H.ab.ovov[Oa, Vb, Oa, Vb], R.aab.VvVOOo, optimize=True)
    )

    dR.aab.VvVOOo -= np.transpose(dR.aab.VvVOOo, (0, 1, 2, 4, 3, 5))

    return dR

def update(R, omega, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    R.aab.VvVOOo = eomcc_active_loops.update_r3b_101110(
        R.aab.VvVOOo,
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
