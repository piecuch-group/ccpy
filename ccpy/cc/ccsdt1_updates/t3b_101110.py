import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import cc_active_loops

#@profile
def build(T, dT, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    # MM(2,3)
    dT.aab.VvVOOo = (1.0 / 2.0) * (
            +1.0 * np.einsum('bCek,AeIJ->AbCIJk', H.ab.vvvo[va, Vb, :, ob], T.aa[Va, :, Oa, Oa], optimize=True)
    )
    dT.aab.VvVOOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('ACek,beIJ->AbCIJk', H.ab.vvvo[Va, Vb, :, ob], T.aa[va, :, Oa, Oa], optimize=True)
    )
    dT.aab.VvVOOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('mCJk,AbIm->AbCIJk', H.ab.ovoo[:, Vb, Oa, ob], T.aa[Va, va, Oa, :], optimize=True)
    )
    dT.aab.VvVOOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('ACIe,beJk->AbCIJk', H.ab.vvov[Va, Vb, Oa, :], T.ab[va, :, Oa, ob], optimize=True)
    )
    dT.aab.VvVOOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('bCIe,AeJk->AbCIJk', H.ab.vvov[va, Vb, Oa, :], T.ab[Va, :, Oa, ob], optimize=True)
    )
    dT.aab.VvVOOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('AmIk,bCJm->AbCIJk', H.ab.vooo[Va, :, Oa, ob], T.ab[va, Vb, Oa, :], optimize=True)
    )
    dT.aab.VvVOOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('bmIk,ACJm->AbCIJk', H.ab.vooo[va, :, Oa, ob], T.ab[Va, Vb, Oa, :], optimize=True)
    )
    dT.aab.VvVOOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('AbIe,eCJk->AbCIJk', H.aa.vvov[Va, va, Oa, :], T.ab[:, Vb, Oa, ob], optimize=True)
    )
    dT.aab.VvVOOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('AmIJ,bCmk->AbCIJk', H.aa.vooo[Va, :, Oa, Oa], T.ab[va, Vb, :, ob], optimize=True)
    )
    dT.aab.VvVOOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('bmIJ,ACmk->AbCIJk', H.aa.vooo[va, :, Oa, Oa], T.ab[Va, Vb, :, ob], optimize=True)
    )
    # (H(2) * T3)_C
    dT.aab.VvVOOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('mI,AbCmJk->AbCIJk', H.a.oo[oa, Oa], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('MI,AbCMJk->AbCIJk', H.a.oo[Oa, Oa], T.aab.VvVOOo, optimize=True)
    )
    dT.aab.VvVOOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('mk,AbCIJm->AbCIJk', H.b.oo[ob, ob], T.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('Mk,AbCIJM->AbCIJk', H.b.oo[Ob, ob], T.aab.VvVOOO, optimize=True)
    )
    dT.aab.VvVOOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('Ae,beCIJk->AbCIJk', H.a.vv[Va, va], T.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('AE,EbCIJk->AbCIJk', H.a.vv[Va, Va], T.aab.VvVOOo, optimize=True)
    )
    dT.aab.VvVOOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('be,AeCIJk->AbCIJk', H.a.vv[va, va], T.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('bE,AECIJk->AbCIJk', H.a.vv[va, Va], T.aab.VVVOOo, optimize=True)
    )
    dT.aab.VvVOOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('Ce,AbeIJk->AbCIJk', H.b.vv[Vb, vb], T.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('CE,AbEIJk->AbCIJk', H.b.vv[Vb, Vb], T.aab.VvVOOo, optimize=True)
    )
    dT.aab.VvVOOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('mNIJ,AbCmNk->AbCIJk', H.aa.oooo[oa, Oa, Oa, Oa], T.aab.VvVoOo, optimize=True)
            + 0.5 * np.einsum('MNIJ,AbCMNk->AbCIJk', H.aa.oooo[Oa, Oa, Oa, Oa], T.aab.VvVOOo, optimize=True)
    )
    dT.aab.VvVOOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('mnJk,AbCmIn->AbCIJk', H.ab.oooo[oa, ob, Oa, ob], T.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('MnJk,AbCIMn->AbCIJk', H.ab.oooo[Oa, ob, Oa, ob], T.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('mNJk,AbCmIN->AbCIJk', H.ab.oooo[oa, Ob, Oa, ob], T.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('MNJk,AbCIMN->AbCIJk', H.ab.oooo[Oa, Ob, Oa, ob], T.aab.VvVOOO, optimize=True)
    )
    dT.aab.VvVOOo += (1.0 / 2.0) * (
            -0.5 * np.einsum('Abef,feCIJk->AbCIJk', H.aa.vvvv[Va, va, va, va], T.aab.vvVOOo, optimize=True)
            - 1.0 * np.einsum('AbeF,FeCIJk->AbCIJk', H.aa.vvvv[Va, va, va, Va], T.aab.VvVOOo, optimize=True)
            - 0.5 * np.einsum('AbEF,FECIJk->AbCIJk', H.aa.vvvv[Va, va, Va, Va], T.aab.VVVOOo, optimize=True)
    )
    dT.aab.VvVOOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('bCef,AefIJk->AbCIJk', H.ab.vvvv[va, Vb, va, vb], T.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('bCEf,EAfIJk->AbCIJk', H.ab.vvvv[va, Vb, Va, vb], T.aab.VVvOOo, optimize=True)
            + 1.0 * np.einsum('bCeF,AeFIJk->AbCIJk', H.ab.vvvv[va, Vb, va, Vb], T.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('bCEF,EAFIJk->AbCIJk', H.ab.vvvv[va, Vb, Va, Vb], T.aab.VVVOOo, optimize=True)
    )
    dT.aab.VvVOOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('ACEf,EbfIJk->AbCIJk', H.ab.vvvv[Va, Vb, Va, vb], T.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('ACeF,ebFIJk->AbCIJk', H.ab.vvvv[Va, Vb, va, Vb], T.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('ACEF,EbFIJk->AbCIJk', H.ab.vvvv[Va, Vb, Va, Vb], T.aab.VvVOOo, optimize=True)
    )
    dT.aab.VvVOOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('AmIe,beCmJk->AbCIJk', H.aa.voov[Va, oa, Oa, va], T.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('AmIE,EbCmJk->AbCIJk', H.aa.voov[Va, oa, Oa, Va], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('AMIe,beCMJk->AbCIJk', H.aa.voov[Va, Oa, Oa, va], T.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('AMIE,EbCMJk->AbCIJk', H.aa.voov[Va, Oa, Oa, Va], T.aab.VvVOOo, optimize=True)
    )
    dT.aab.VvVOOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('bmIe,AeCmJk->AbCIJk', H.aa.voov[va, oa, Oa, va], T.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('bmIE,AECmJk->AbCIJk', H.aa.voov[va, oa, Oa, Va], T.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('bMIe,AeCMJk->AbCIJk', H.aa.voov[va, Oa, Oa, va], T.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('bMIE,AECMJk->AbCIJk', H.aa.voov[va, Oa, Oa, Va], T.aab.VVVOOo, optimize=True)
    )
    dT.aab.VvVOOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('AmIe,bCeJmk->AbCIJk', H.ab.voov[Va, ob, Oa, vb], T.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('AmIE,bCEJmk->AbCIJk', H.ab.voov[Va, ob, Oa, Vb], T.abb.vVVOoo, optimize=True)
            + 1.0 * np.einsum('AMIe,bCeJkM->AbCIJk', H.ab.voov[Va, Ob, Oa, vb], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('AMIE,bCEJkM->AbCIJk', H.ab.voov[Va, Ob, Oa, Vb], T.abb.vVVOoO, optimize=True)
    )
    dT.aab.VvVOOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('bmIe,ACeJmk->AbCIJk', H.ab.voov[va, ob, Oa, vb], T.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('bmIE,ACEJmk->AbCIJk', H.ab.voov[va, ob, Oa, Vb], T.abb.VVVOoo, optimize=True)
            - 1.0 * np.einsum('bMIe,ACeJkM->AbCIJk', H.ab.voov[va, Ob, Oa, vb], T.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('bMIE,ACEJkM->AbCIJk', H.ab.voov[va, Ob, Oa, Vb], T.abb.VVVOoO, optimize=True)
    )
    dT.aab.VvVOOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('mCek,AebmIJ->AbCIJk', H.ab.ovvo[oa, Vb, va, ob], T.aaa.VvvoOO, optimize=True)
            + 1.0 * np.einsum('mCEk,EAbmIJ->AbCIJk', H.ab.ovvo[oa, Vb, Va, ob], T.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('MCek,AebIJM->AbCIJk', H.ab.ovvo[Oa, Vb, va, ob], T.aaa.VvvOOO, optimize=True)
            + 1.0 * np.einsum('MCEk,EAbIJM->AbCIJk', H.ab.ovvo[Oa, Vb, Va, ob], T.aaa.VVvOOO, optimize=True)
    )
    dT.aab.VvVOOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('Cmke,AbeIJm->AbCIJk', H.bb.voov[Vb, ob, ob, vb], T.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('CmkE,AbEIJm->AbCIJk', H.bb.voov[Vb, ob, ob, Vb], T.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('CMke,AbeIJM->AbCIJk', H.bb.voov[Vb, Ob, ob, vb], T.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('CMkE,AbEIJM->AbCIJk', H.bb.voov[Vb, Ob, ob, Vb], T.aab.VvVOOO, optimize=True)
    )
    dT.aab.VvVOOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('Amek,beCIJm->AbCIJk', H.ab.vovo[Va, ob, va, ob], T.aab.vvVOOo, optimize=True)
            - 1.0 * np.einsum('AmEk,EbCIJm->AbCIJk', H.ab.vovo[Va, ob, Va, ob], T.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('AMek,beCIJM->AbCIJk', H.ab.vovo[Va, Ob, va, ob], T.aab.vvVOOO, optimize=True)
            - 1.0 * np.einsum('AMEk,EbCIJM->AbCIJk', H.ab.vovo[Va, Ob, Va, ob], T.aab.VvVOOO, optimize=True)
    )
    dT.aab.VvVOOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('bmek,AeCIJm->AbCIJk', H.ab.vovo[va, ob, va, ob], T.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('bmEk,AECIJm->AbCIJk', H.ab.vovo[va, ob, Va, ob], T.aab.VVVOOo, optimize=True)
            - 1.0 * np.einsum('bMek,AeCIJM->AbCIJk', H.ab.vovo[va, Ob, va, ob], T.aab.VvVOOO, optimize=True)
            - 1.0 * np.einsum('bMEk,AECIJM->AbCIJk', H.ab.vovo[va, Ob, Va, ob], T.aab.VVVOOO, optimize=True)
    )
    dT.aab.VvVOOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('mCIe,AbemJk->AbCIJk', H.ab.ovov[oa, Vb, Oa, vb], T.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('mCIE,AbEmJk->AbCIJk', H.ab.ovov[oa, Vb, Oa, Vb], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('MCIe,AbeMJk->AbCIJk', H.ab.ovov[Oa, Vb, Oa, vb], T.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('MCIE,AbEMJk->AbCIJk', H.ab.ovov[Oa, Vb, Oa, Vb], T.aab.VvVOOo, optimize=True)
    )

    dT.aab.VvVOOo -= np.transpose(dT.aab.VvVOOo, (0, 1, 2, 4, 3, 5))

    return dT

def update(T, dT, H, shift, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.aab.VvVOOo, dT.aab.VvVOOo = cc_active_loops.update_t3b_101110(
        T.aab.VvVOOo,
        dT.aab.VvVOOo,
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