import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops

def build(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.aab.vvVOOo = (2.0 / 4.0) * (
            +1.0 * np.einsum('bCek,aeIJ->abCIJk', X.ab.vvvo[va, Vb, :, ob], T.aa[va, :, Oa, Oa], optimize=True)
    )
    dR.aab.vvVOOo += (2.0 / 4.0) * (
            -1.0 * np.einsum('mCJk,abIm->abCIJk', X.ab.ovoo[:, Vb, Oa, ob], T.aa[va, va, Oa, :], optimize=True)
    )
    dR.aab.vvVOOo += (4.0 / 4.0) * (
            +1.0 * np.einsum('aCIe,beJk->abCIJk', X.ab.vvov[va, Vb, Oa, :], T.ab[va, :, Oa, ob], optimize=True)
    )
    dR.aab.vvVOOo += (4.0 / 4.0) * (
            -1.0 * np.einsum('amIk,bCJm->abCIJk', X.ab.vooo[va, :, Oa, ob], T.ab[va, Vb, Oa, :], optimize=True)
    )
    dR.aab.vvVOOo += (2.0 / 4.0) * (
            +1.0 * np.einsum('abIe,eCJk->abCIJk', X.aa.vvov[va, va, Oa, :], T.ab[:, Vb, Oa, ob], optimize=True)
    )
    dR.aab.vvVOOo += (2.0 / 4.0) * (
            -1.0 * np.einsum('amIJ,bCmk->abCIJk', X.aa.vooo[va, :, Oa, Oa], T.ab[va, Vb, :, ob], optimize=True)
    )
    dR.aab.vvVOOo += (2.0 / 4.0) * (
            +1.0 * np.einsum('bCek,aeIJ->abCIJk', H.ab.vvvo[va, Vb, :, ob], R.aa[va, :, Oa, Oa], optimize=True)
    )
    dR.aab.vvVOOo += (2.0 / 4.0) * (
            -1.0 * np.einsum('mCJk,abIm->abCIJk', H.ab.ovoo[:, Vb, Oa, ob], R.aa[va, va, Oa, :], optimize=True)
    )
    dR.aab.vvVOOo += (4.0 / 4.0) * (
            +1.0 * np.einsum('aCIe,beJk->abCIJk', H.ab.vvov[va, Vb, Oa, :], R.ab[va, :, Oa, ob], optimize=True)
    )
    dR.aab.vvVOOo += (4.0 / 4.0) * (
            -1.0 * np.einsum('amIk,bCJm->abCIJk', H.ab.vooo[va, :, Oa, ob], R.ab[va, Vb, Oa, :], optimize=True)
    )
    dR.aab.vvVOOo += (2.0 / 4.0) * (
            +1.0 * np.einsum('abIe,eCJk->abCIJk', H.aa.vvov[va, va, Oa, :], R.ab[:, Vb, Oa, ob], optimize=True)
    )
    dR.aab.vvVOOo += (2.0 / 4.0) * (
            -1.0 * np.einsum('amIJ,bCmk->abCIJk', H.aa.vooo[va, :, Oa, Oa], R.ab[va, Vb, :, ob], optimize=True)
    )

    dR.aab.vvVOOo += (2.0 / 4.0) * (
            +1.0 * np.einsum('mI,baCmJk->abCIJk', X.a.oo[oa, Oa], T.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('MI,baCMJk->abCIJk', X.a.oo[Oa, Oa], T.aab.vvVOOo, optimize=True)
    )
    dR.aab.vvVOOo += (1.0 / 4.0) * (
            +1.0 * np.einsum('mk,baCIJm->abCIJk', X.b.oo[ob, ob], T.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('Mk,baCIJM->abCIJk', X.b.oo[Ob, ob], T.aab.vvVOOO, optimize=True)
    )
    dR.aab.vvVOOo += (2.0 / 4.0) * (
            -1.0 * np.einsum('ae,beCIJk->abCIJk', X.a.vv[va, va], T.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('aE,EbCIJk->abCIJk', X.a.vv[va, Va], T.aab.VvVOOo, optimize=True)
    )
    dR.aab.vvVOOo += (1.0 / 4.0) * (
            -1.0 * np.einsum('CE,baEIJk->abCIJk', X.b.vv[Vb, Vb], T.aab.vvVOOo, optimize=True)
    )
    dR.aab.vvVOOo += (1.0 / 4.0) * (
            +1.0 * np.einsum('MnIJ,baCnMk->abCIJk', X.aa.oooo[Oa, oa, Oa, Oa], T.aab.vvVoOo, optimize=True)
            - 0.5 * np.einsum('MNIJ,baCMNk->abCIJk', X.aa.oooo[Oa, Oa, Oa, Oa], T.aab.vvVOOo, optimize=True)
    )
    dR.aab.vvVOOo += (2.0 / 4.0) * (
            +1.0 * np.einsum('mnJk,baCmIn->abCIJk', X.ab.oooo[oa, ob, Oa, ob], T.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('mNJk,baCmIN->abCIJk', X.ab.oooo[oa, Ob, Oa, ob], T.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('MnJk,baCIMn->abCIJk', X.ab.oooo[Oa, ob, Oa, ob], T.aab.vvVOOo, optimize=True)
            - 1.0 * np.einsum('MNJk,baCIMN->abCIJk', X.ab.oooo[Oa, Ob, Oa, ob], T.aab.vvVOOO, optimize=True)
    )
    dR.aab.vvVOOo += (1.0 / 4.0) * (
            -0.5 * np.einsum('abef,feCIJk->abCIJk', X.aa.vvvv[va, va, va, va], T.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('abEf,EfCIJk->abCIJk', X.aa.vvvv[va, va, Va, va], T.aab.VvVOOo, optimize=True)
            - 0.5 * np.einsum('abEF,FECIJk->abCIJk', X.aa.vvvv[va, va, Va, Va], T.aab.VVVOOo, optimize=True)
    )
    dR.aab.vvVOOo += (2.0 / 4.0) * (
            -1.0 * np.einsum('bCeF,eaFIJk->abCIJk', X.ab.vvvv[va, Vb, va, Vb], T.aab.vvVOOo, optimize=True)
            - 1.0 * np.einsum('bCEf,EafIJk->abCIJk', X.ab.vvvv[va, Vb, Va, vb], T.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('bCEF,EaFIJk->abCIJk', X.ab.vvvv[va, Vb, Va, Vb], T.aab.VvVOOo, optimize=True)
    )
    dR.aab.vvVOOo += (4.0 / 4.0) * (
            -1.0 * np.einsum('amIe,beCmJk->abCIJk', X.aa.voov[va, oa, Oa, va], T.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('amIE,EbCmJk->abCIJk', X.aa.voov[va, oa, Oa, Va], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('aMIe,beCMJk->abCIJk', X.aa.voov[va, Oa, Oa, va], T.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('aMIE,EbCMJk->abCIJk', X.aa.voov[va, Oa, Oa, Va], T.aab.VvVOOo, optimize=True)
    )
    dR.aab.vvVOOo += (4.0 / 4.0) * (
            -1.0 * np.einsum('amIe,bCeJmk->abCIJk', X.ab.voov[va, ob, Oa, vb], T.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('amIE,bCEJmk->abCIJk', X.ab.voov[va, ob, Oa, Vb], T.abb.vVVOoo, optimize=True)
            + 1.0 * np.einsum('aMIe,bCeJkM->abCIJk', X.ab.voov[va, Ob, Oa, vb], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('aMIE,bCEJkM->abCIJk', X.ab.voov[va, Ob, Oa, Vb], T.abb.vVVOoO, optimize=True)
    )
    dR.aab.vvVOOo += (1.0 / 4.0) * (
            -1.0 * np.einsum('mCEk,EbamIJ->abCIJk', X.ab.ovvo[oa, Vb, Va, ob], T.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('MCEk,EbaIJM->abCIJk', X.ab.ovvo[Oa, Vb, Va, ob], T.aaa.VvvOOO, optimize=True)
    )
    dR.aab.vvVOOo += (1.0 / 4.0) * (
            -1.0 * np.einsum('CmkE,baEIJm->abCIJk', X.bb.voov[Vb, ob, ob, Vb], T.aab.vvVOOo, optimize=True)
            - 1.0 * np.einsum('CMkE,baEIJM->abCIJk', X.bb.voov[Vb, Ob, ob, Vb], T.aab.vvVOOO, optimize=True)
    )
    dR.aab.vvVOOo += (2.0 / 4.0) * (
            +1.0 * np.einsum('amek,beCIJm->abCIJk', X.ab.vovo[va, ob, va, ob], T.aab.vvVOOo, optimize=True)
            - 1.0 * np.einsum('amEk,EbCIJm->abCIJk', X.ab.vovo[va, ob, Va, ob], T.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('aMek,beCIJM->abCIJk', X.ab.vovo[va, Ob, va, ob], T.aab.vvVOOO, optimize=True)
            - 1.0 * np.einsum('aMEk,EbCIJM->abCIJk', X.ab.vovo[va, Ob, Va, ob], T.aab.VvVOOO, optimize=True)
    )
    dR.aab.vvVOOo += (2.0 / 4.0) * (
            +1.0 * np.einsum('mCIE,baEmJk->abCIJk', X.ab.ovov[oa, Vb, Oa, Vb], T.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('MCIE,baEMJk->abCIJk', X.ab.ovov[Oa, Vb, Oa, Vb], T.aab.vvVOOo, optimize=True)
    )
    dR.aab.vvVOOo += (2.0 / 4.0) * (
            +1.0 * np.einsum('mI,baCmJk->abCIJk', H.a.oo[oa, Oa], R.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('MI,baCMJk->abCIJk', H.a.oo[Oa, Oa], R.aab.vvVOOo, optimize=True)
    )
    dR.aab.vvVOOo += (1.0 / 4.0) * (
            +1.0 * np.einsum('mk,baCIJm->abCIJk', H.b.oo[ob, ob], R.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('Mk,baCIJM->abCIJk', H.b.oo[Ob, ob], R.aab.vvVOOO, optimize=True)
    )
    dR.aab.vvVOOo += (2.0 / 4.0) * (
            -1.0 * np.einsum('ae,beCIJk->abCIJk', H.a.vv[va, va], R.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('aE,EbCIJk->abCIJk', H.a.vv[va, Va], R.aab.VvVOOo, optimize=True)
    )
    dR.aab.vvVOOo += (1.0 / 4.0) * (
            -1.0 * np.einsum('CE,baEIJk->abCIJk', H.b.vv[Vb, Vb], R.aab.vvVOOo, optimize=True)
    )
    dR.aab.vvVOOo += (1.0 / 4.0) * (
            +1.0 * np.einsum('MnIJ,baCnMk->abCIJk', H.aa.oooo[Oa, oa, Oa, Oa], R.aab.vvVoOo, optimize=True)
            - 0.5 * np.einsum('MNIJ,baCMNk->abCIJk', H.aa.oooo[Oa, Oa, Oa, Oa], R.aab.vvVOOo, optimize=True)
    )
    dR.aab.vvVOOo += (2.0 / 4.0) * (
            +1.0 * np.einsum('mnJk,baCmIn->abCIJk', H.ab.oooo[oa, ob, Oa, ob], R.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('mNJk,baCmIN->abCIJk', H.ab.oooo[oa, Ob, Oa, ob], R.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('MnJk,baCIMn->abCIJk', H.ab.oooo[Oa, ob, Oa, ob], R.aab.vvVOOo, optimize=True)
            - 1.0 * np.einsum('MNJk,baCIMN->abCIJk', H.ab.oooo[Oa, Ob, Oa, ob], R.aab.vvVOOO, optimize=True)
    )
    dR.aab.vvVOOo += (1.0 / 4.0) * (
            -0.5 * np.einsum('abef,feCIJk->abCIJk', H.aa.vvvv[va, va, va, va], R.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('abEf,EfCIJk->abCIJk', H.aa.vvvv[va, va, Va, va], R.aab.VvVOOo, optimize=True)
            - 0.5 * np.einsum('abEF,FECIJk->abCIJk', H.aa.vvvv[va, va, Va, Va], R.aab.VVVOOo, optimize=True)
    )
    dR.aab.vvVOOo += (2.0 / 4.0) * (
            -1.0 * np.einsum('bCeF,eaFIJk->abCIJk', H.ab.vvvv[va, Vb, va, Vb], R.aab.vvVOOo, optimize=True)
            - 1.0 * np.einsum('bCEf,EafIJk->abCIJk', H.ab.vvvv[va, Vb, Va, vb], R.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('bCEF,EaFIJk->abCIJk', H.ab.vvvv[va, Vb, Va, Vb], R.aab.VvVOOo, optimize=True)
    )
    dR.aab.vvVOOo += (4.0 / 4.0) * (
            -1.0 * np.einsum('amIe,beCmJk->abCIJk', H.aa.voov[va, oa, Oa, va], R.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('amIE,EbCmJk->abCIJk', H.aa.voov[va, oa, Oa, Va], R.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('aMIe,beCMJk->abCIJk', H.aa.voov[va, Oa, Oa, va], R.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('aMIE,EbCMJk->abCIJk', H.aa.voov[va, Oa, Oa, Va], R.aab.VvVOOo, optimize=True)
    )
    dR.aab.vvVOOo += (4.0 / 4.0) * (
            -1.0 * np.einsum('amIe,bCeJmk->abCIJk', H.ab.voov[va, ob, Oa, vb], R.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('amIE,bCEJmk->abCIJk', H.ab.voov[va, ob, Oa, Vb], R.abb.vVVOoo, optimize=True)
            + 1.0 * np.einsum('aMIe,bCeJkM->abCIJk', H.ab.voov[va, Ob, Oa, vb], R.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('aMIE,bCEJkM->abCIJk', H.ab.voov[va, Ob, Oa, Vb], R.abb.vVVOoO, optimize=True)
    )
    dR.aab.vvVOOo += (1.0 / 4.0) * (
            -1.0 * np.einsum('mCEk,EbamIJ->abCIJk', H.ab.ovvo[oa, Vb, Va, ob], R.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('MCEk,EbaIJM->abCIJk', H.ab.ovvo[Oa, Vb, Va, ob], R.aaa.VvvOOO, optimize=True)
    )
    dR.aab.vvVOOo += (1.0 / 4.0) * (
            -1.0 * np.einsum('CmkE,baEIJm->abCIJk', H.bb.voov[Vb, ob, ob, Vb], R.aab.vvVOOo, optimize=True)
            - 1.0 * np.einsum('CMkE,baEIJM->abCIJk', H.bb.voov[Vb, Ob, ob, Vb], R.aab.vvVOOO, optimize=True)
    )
    dR.aab.vvVOOo += (2.0 / 4.0) * (
            +1.0 * np.einsum('amek,beCIJm->abCIJk', H.ab.vovo[va, ob, va, ob], R.aab.vvVOOo, optimize=True)
            - 1.0 * np.einsum('amEk,EbCIJm->abCIJk', H.ab.vovo[va, ob, Va, ob], R.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('aMek,beCIJM->abCIJk', H.ab.vovo[va, Ob, va, ob], R.aab.vvVOOO, optimize=True)
            - 1.0 * np.einsum('aMEk,EbCIJM->abCIJk', H.ab.vovo[va, Ob, Va, ob], R.aab.VvVOOO, optimize=True)
    )
    dR.aab.vvVOOo += (2.0 / 4.0) * (
            +1.0 * np.einsum('mCIE,baEmJk->abCIJk', H.ab.ovov[oa, Vb, Oa, Vb], R.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('MCIE,baEMJk->abCIJk', H.ab.ovov[Oa, Vb, Oa, Vb], R.aab.vvVOOo, optimize=True)
    )

    dR.aab.vvVOOo -= np.transpose(dR.aab.vvVOOo, (1, 0, 2, 3, 4, 5))
    dR.aab.vvVOOo -= np.transpose(dR.aab.vvVOOo, (0, 1, 2, 4, 3, 5))

    return dR

def update(R, omega, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    R.aab.vvVOOo = eomcc_active_loops.update_r3b_001110(
        R.aab.vvVOOo,
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
