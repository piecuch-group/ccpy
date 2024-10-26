import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops

def build(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.aab.vvVOOO = (2.0 / 4.0) * (
            +1.0 * np.einsum('bCeK,aeIJ->abCIJK', X.ab.vvvo[va, Vb, :, Ob], T.aa[va, :, Oa, Oa], optimize=True)
    )
    dR.aab.vvVOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('mCJK,abIm->abCIJK', X.ab.ovoo[:, Vb, Oa, Ob], T.aa[va, va, Oa, :], optimize=True)
    )
    dR.aab.vvVOOO += (4.0 / 4.0) * (
            +1.0 * np.einsum('aCIe,beJK->abCIJK', X.ab.vvov[va, Vb, Oa, :], T.ab[va, :, Oa, Ob], optimize=True)
    )
    dR.aab.vvVOOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('amIK,bCJm->abCIJK', X.ab.vooo[va, :, Oa, Ob], T.ab[va, Vb, Oa, :], optimize=True)
    )
    dR.aab.vvVOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('abIe,eCJK->abCIJK', X.aa.vvov[va, va, Oa, :], T.ab[:, Vb, Oa, Ob], optimize=True)
    )
    dR.aab.vvVOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('amIJ,bCmK->abCIJK', X.aa.vooo[va, :, Oa, Oa], T.ab[va, Vb, :, Ob], optimize=True)
    )
    dR.aab.vvVOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('bCeK,aeIJ->abCIJK', H.ab.vvvo[va, Vb, :, Ob], R.aa[va, :, Oa, Oa], optimize=True)
    )
    dR.aab.vvVOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('mCJK,abIm->abCIJK', H.ab.ovoo[:, Vb, Oa, Ob], R.aa[va, va, Oa, :], optimize=True)
    )
    dR.aab.vvVOOO += (4.0 / 4.0) * (
            +1.0 * np.einsum('aCIe,beJK->abCIJK', H.ab.vvov[va, Vb, Oa, :], R.ab[va, :, Oa, Ob], optimize=True)
    )
    dR.aab.vvVOOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('amIK,bCJm->abCIJK', H.ab.vooo[va, :, Oa, Ob], R.ab[va, Vb, Oa, :], optimize=True)
    )
    dR.aab.vvVOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('abIe,eCJK->abCIJK', H.aa.vvov[va, va, Oa, :], R.ab[:, Vb, Oa, Ob], optimize=True)
    )
    dR.aab.vvVOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('amIJ,bCmK->abCIJK', H.aa.vooo[va, :, Oa, Oa], R.ab[va, Vb, :, Ob], optimize=True)
    )

    dR.aab.vvVOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mI,baCmJK->abCIJK', X.a.oo[oa, Oa], T.aab.vvVoOO, optimize=True)
            + 1.0 * np.einsum('MI,baCMJK->abCIJK', X.a.oo[Oa, Oa], T.aab.vvVOOO, optimize=True)
    )
    dR.aab.vvVOOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('mK,baCIJm->abCIJK', X.b.oo[ob, Ob], T.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('MK,baCIJM->abCIJK', X.b.oo[Ob, Ob], T.aab.vvVOOO, optimize=True)
    )
    dR.aab.vvVOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('ae,beCIJK->abCIJK', X.a.vv[va, va], T.aab.vvVOOO, optimize=True)
            + 1.0 * np.einsum('aE,EbCIJK->abCIJK', X.a.vv[va, Va], T.aab.VvVOOO, optimize=True)
    )
    dR.aab.vvVOOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('CE,baEIJK->abCIJK', X.b.vv[Vb, Vb], T.aab.vvVOOO, optimize=True)
    )
    dR.aab.vvVOOO += (1.0 / 4.0) * (
            -0.5 * np.einsum('mnIJ,baCmnK->abCIJK', X.aa.oooo[oa, oa, Oa, Oa], T.aab.vvVooO, optimize=True)
            - 1.0 * np.einsum('mNIJ,baCmNK->abCIJK', X.aa.oooo[oa, Oa, Oa, Oa], T.aab.vvVoOO, optimize=True)
            - 0.5 * np.einsum('MNIJ,baCMNK->abCIJK', X.aa.oooo[Oa, Oa, Oa, Oa], T.aab.vvVOOO, optimize=True)
    )
    dR.aab.vvVOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mnJK,baCmIn->abCIJK', X.ab.oooo[oa, ob, Oa, Ob], T.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('MnJK,baCIMn->abCIJK', X.ab.oooo[Oa, ob, Oa, Ob], T.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('mNJK,baCmIN->abCIJK', X.ab.oooo[oa, Ob, Oa, Ob], T.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('MNJK,baCIMN->abCIJK', X.ab.oooo[Oa, Ob, Oa, Ob], T.aab.vvVOOO, optimize=True)
    )
    dR.aab.vvVOOO += (1.0 / 4.0) * (
            -0.5 * np.einsum('abef,feCIJK->abCIJK', X.aa.vvvv[va, va, va, va], T.aab.vvVOOO, optimize=True)
            - 1.0 * np.einsum('abeF,FeCIJK->abCIJK', X.aa.vvvv[va, va, va, Va], T.aab.VvVOOO, optimize=True)
            - 0.5 * np.einsum('abEF,FECIJK->abCIJK', X.aa.vvvv[va, va, Va, Va], T.aab.VVVOOO, optimize=True)
    )
    dR.aab.vvVOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('bCEf,EafIJK->abCIJK', X.ab.vvvv[va, Vb, Va, vb], T.aab.VvvOOO, optimize=True)
            - 1.0 * np.einsum('bCeF,eaFIJK->abCIJK', X.ab.vvvv[va, Vb, va, Vb], T.aab.vvVOOO, optimize=True)
            - 1.0 * np.einsum('bCEF,EaFIJK->abCIJK', X.ab.vvvv[va, Vb, Va, Vb], T.aab.VvVOOO, optimize=True)
    )
    dR.aab.vvVOOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('amIe,beCmJK->abCIJK', X.aa.voov[va, oa, Oa, va], T.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('aMIe,beCMJK->abCIJK', X.aa.voov[va, Oa, Oa, va], T.aab.vvVOOO, optimize=True)
            + 1.0 * np.einsum('amIE,EbCmJK->abCIJK', X.aa.voov[va, oa, Oa, Va], T.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('aMIE,EbCMJK->abCIJK', X.aa.voov[va, Oa, Oa, Va], T.aab.VvVOOO, optimize=True)
    )
    dR.aab.vvVOOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('amIe,bCeJmK->abCIJK', X.ab.voov[va, ob, Oa, vb], T.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('aMIe,bCeJMK->abCIJK', X.ab.voov[va, Ob, Oa, vb], T.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('amIE,bCEJmK->abCIJK', X.ab.voov[va, ob, Oa, Vb], T.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('aMIE,bCEJMK->abCIJK', X.ab.voov[va, Ob, Oa, Vb], T.abb.vVVOOO, optimize=True)
    )
    dR.aab.vvVOOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('mCEK,EbamIJ->abCIJK', X.ab.ovvo[oa, Vb, Va, Ob], T.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('MCEK,EbaIJM->abCIJK', X.ab.ovvo[Oa, Vb, Va, Ob], T.aaa.VvvOOO, optimize=True)
    )
    dR.aab.vvVOOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('CmKE,baEIJm->abCIJK', X.bb.voov[Vb, ob, Ob, Vb], T.aab.vvVOOo, optimize=True)
            - 1.0 * np.einsum('CMKE,baEIJM->abCIJK', X.bb.voov[Vb, Ob, Ob, Vb], T.aab.vvVOOO, optimize=True)
    )
    dR.aab.vvVOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('ameK,beCIJm->abCIJK', X.ab.vovo[va, ob, va, Ob], T.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('aMeK,beCIJM->abCIJK', X.ab.vovo[va, Ob, va, Ob], T.aab.vvVOOO, optimize=True)
            - 1.0 * np.einsum('amEK,EbCIJm->abCIJK', X.ab.vovo[va, ob, Va, Ob], T.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('aMEK,EbCIJM->abCIJK', X.ab.vovo[va, Ob, Va, Ob], T.aab.VvVOOO, optimize=True)
    )
    dR.aab.vvVOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mCIE,baEmJK->abCIJK', X.ab.ovov[oa, Vb, Oa, Vb], T.aab.vvVoOO, optimize=True)
            + 1.0 * np.einsum('MCIE,baEMJK->abCIJK', X.ab.ovov[Oa, Vb, Oa, Vb], T.aab.vvVOOO, optimize=True)
    )
    dR.aab.vvVOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mI,baCmJK->abCIJK', H.a.oo[oa, Oa], R.aab.vvVoOO, optimize=True)
            + 1.0 * np.einsum('MI,baCMJK->abCIJK', H.a.oo[Oa, Oa], R.aab.vvVOOO, optimize=True)
    )
    dR.aab.vvVOOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('mK,baCIJm->abCIJK', H.b.oo[ob, Ob], R.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('MK,baCIJM->abCIJK', H.b.oo[Ob, Ob], R.aab.vvVOOO, optimize=True)
    )
    dR.aab.vvVOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('ae,beCIJK->abCIJK', H.a.vv[va, va], R.aab.vvVOOO, optimize=True)
            + 1.0 * np.einsum('aE,EbCIJK->abCIJK', H.a.vv[va, Va], R.aab.VvVOOO, optimize=True)
    )
    dR.aab.vvVOOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('CE,baEIJK->abCIJK', H.b.vv[Vb, Vb], R.aab.vvVOOO, optimize=True)
    )
    dR.aab.vvVOOO += (1.0 / 4.0) * (
            -0.5 * np.einsum('mnIJ,baCmnK->abCIJK', H.aa.oooo[oa, oa, Oa, Oa], R.aab.vvVooO, optimize=True)
            - 1.0 * np.einsum('mNIJ,baCmNK->abCIJK', H.aa.oooo[oa, Oa, Oa, Oa], R.aab.vvVoOO, optimize=True)
            - 0.5 * np.einsum('MNIJ,baCMNK->abCIJK', H.aa.oooo[Oa, Oa, Oa, Oa], R.aab.vvVOOO, optimize=True)
    )
    dR.aab.vvVOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mnJK,baCmIn->abCIJK', H.ab.oooo[oa, ob, Oa, Ob], R.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('MnJK,baCIMn->abCIJK', H.ab.oooo[Oa, ob, Oa, Ob], R.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('mNJK,baCmIN->abCIJK', H.ab.oooo[oa, Ob, Oa, Ob], R.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('MNJK,baCIMN->abCIJK', H.ab.oooo[Oa, Ob, Oa, Ob], R.aab.vvVOOO, optimize=True)
    )
    dR.aab.vvVOOO += (1.0 / 4.0) * (
            -0.5 * np.einsum('abef,feCIJK->abCIJK', H.aa.vvvv[va, va, va, va], R.aab.vvVOOO, optimize=True)
            - 1.0 * np.einsum('abeF,FeCIJK->abCIJK', H.aa.vvvv[va, va, va, Va], R.aab.VvVOOO, optimize=True)
            - 0.5 * np.einsum('abEF,FECIJK->abCIJK', H.aa.vvvv[va, va, Va, Va], R.aab.VVVOOO, optimize=True)
    )
    dR.aab.vvVOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('bCEf,EafIJK->abCIJK', H.ab.vvvv[va, Vb, Va, vb], R.aab.VvvOOO, optimize=True)
            - 1.0 * np.einsum('bCeF,eaFIJK->abCIJK', H.ab.vvvv[va, Vb, va, Vb], R.aab.vvVOOO, optimize=True)
            - 1.0 * np.einsum('bCEF,EaFIJK->abCIJK', H.ab.vvvv[va, Vb, Va, Vb], R.aab.VvVOOO, optimize=True)
    )
    dR.aab.vvVOOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('amIe,beCmJK->abCIJK', H.aa.voov[va, oa, Oa, va], R.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('aMIe,beCMJK->abCIJK', H.aa.voov[va, Oa, Oa, va], R.aab.vvVOOO, optimize=True)
            + 1.0 * np.einsum('amIE,EbCmJK->abCIJK', H.aa.voov[va, oa, Oa, Va], R.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('aMIE,EbCMJK->abCIJK', H.aa.voov[va, Oa, Oa, Va], R.aab.VvVOOO, optimize=True)
    )
    dR.aab.vvVOOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('amIe,bCeJmK->abCIJK', H.ab.voov[va, ob, Oa, vb], R.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('aMIe,bCeJMK->abCIJK', H.ab.voov[va, Ob, Oa, vb], R.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('amIE,bCEJmK->abCIJK', H.ab.voov[va, ob, Oa, Vb], R.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('aMIE,bCEJMK->abCIJK', H.ab.voov[va, Ob, Oa, Vb], R.abb.vVVOOO, optimize=True)
    )
    dR.aab.vvVOOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('mCEK,EbamIJ->abCIJK', H.ab.ovvo[oa, Vb, Va, Ob], R.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('MCEK,EbaIJM->abCIJK', H.ab.ovvo[Oa, Vb, Va, Ob], R.aaa.VvvOOO, optimize=True)
    )
    dR.aab.vvVOOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('CmKE,baEIJm->abCIJK', H.bb.voov[Vb, ob, Ob, Vb], R.aab.vvVOOo, optimize=True)
            - 1.0 * np.einsum('CMKE,baEIJM->abCIJK', H.bb.voov[Vb, Ob, Ob, Vb], R.aab.vvVOOO, optimize=True)
    )
    dR.aab.vvVOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('ameK,beCIJm->abCIJK', H.ab.vovo[va, ob, va, Ob], R.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('aMeK,beCIJM->abCIJK', H.ab.vovo[va, Ob, va, Ob], R.aab.vvVOOO, optimize=True)
            - 1.0 * np.einsum('amEK,EbCIJm->abCIJK', H.ab.vovo[va, ob, Va, Ob], R.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('aMEK,EbCIJM->abCIJK', H.ab.vovo[va, Ob, Va, Ob], R.aab.VvVOOO, optimize=True)
    )
    dR.aab.vvVOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mCIE,baEmJK->abCIJK', H.ab.ovov[oa, Vb, Oa, Vb], R.aab.vvVoOO, optimize=True)
            + 1.0 * np.einsum('MCIE,baEMJK->abCIJK', H.ab.ovov[Oa, Vb, Oa, Vb], R.aab.vvVOOO, optimize=True)
    )

    dR.aab.vvVOOO -= np.transpose(dR.aab.vvVOOO, (1, 0, 2, 3, 4, 5))
    dR.aab.vvVOOO -= np.transpose(dR.aab.vvVOOO, (0, 1, 2, 4, 3, 5))

    return dR

def update(R, omega, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    R.aab.vvVOOO = eomcc_active_loops.update_r3b_001111(
        R.aab.vvVOOO,
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
