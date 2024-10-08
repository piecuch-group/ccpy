import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import cc_active_loops

#@profile
def build(T, dT, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    # MM(2,3)
    dT.aab.vvVOOO = (2.0 / 4.0) * (
            +1.0 * np.einsum('bCeK,aeIJ->abCIJK', H.ab.vvvo[va, Vb, :, Ob], T.aa[va, :, Oa, Oa], optimize=True)
    )
    dT.aab.vvVOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('mCJK,abIm->abCIJK', H.ab.ovoo[:, Vb, Oa, Ob], T.aa[va, va, Oa, :], optimize=True)
    )
    dT.aab.vvVOOO += (4.0 / 4.0) * (
            +1.0 * np.einsum('aCIe,beJK->abCIJK', H.ab.vvov[va, Vb, Oa, :], T.ab[va, :, Oa, Ob], optimize=True)
    )
    dT.aab.vvVOOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('amIK,bCJm->abCIJK', H.ab.vooo[va, :, Oa, Ob], T.ab[va, Vb, Oa, :], optimize=True)
    )
    dT.aab.vvVOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('abIe,eCJK->abCIJK', H.aa.vvov[va, va, Oa, :], T.ab[:, Vb, Oa, Ob], optimize=True)
    )
    dT.aab.vvVOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('amIJ,bCmK->abCIJK', H.aa.vooo[va, :, Oa, Oa], T.ab[va, Vb, :, Ob], optimize=True)
    )
    # (H(2) * T3)_C
    dT.aab.vvVOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mI,baCmJK->abCIJK', H.a.oo[oa, Oa], T.aab.vvVoOO, optimize=True)
            + 1.0 * np.einsum('MI,baCMJK->abCIJK', H.a.oo[Oa, Oa], T.aab.vvVOOO, optimize=True)
    )
    dT.aab.vvVOOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('mK,baCIJm->abCIJK', H.b.oo[ob, Ob], T.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('MK,baCIJM->abCIJK', H.b.oo[Ob, Ob], T.aab.vvVOOO, optimize=True)
    )
    dT.aab.vvVOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('ae,beCIJK->abCIJK', H.a.vv[va, va], T.aab.vvVOOO, optimize=True)
            + 1.0 * np.einsum('aE,EbCIJK->abCIJK', H.a.vv[va, Va], T.aab.VvVOOO, optimize=True)
    )
    dT.aab.vvVOOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('CE,baEIJK->abCIJK', H.b.vv[Vb, Vb], T.aab.vvVOOO, optimize=True)
    )
    dT.aab.vvVOOO += (1.0 / 4.0) * (
            -0.5 * np.einsum('mnIJ,baCmnK->abCIJK', H.aa.oooo[oa, oa, Oa, Oa], T.aab.vvVooO, optimize=True)
            - 1.0 * np.einsum('mNIJ,baCmNK->abCIJK', H.aa.oooo[oa, Oa, Oa, Oa], T.aab.vvVoOO, optimize=True)
            - 0.5 * np.einsum('MNIJ,baCMNK->abCIJK', H.aa.oooo[Oa, Oa, Oa, Oa], T.aab.vvVOOO, optimize=True)
    )
    dT.aab.vvVOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mnJK,baCmIn->abCIJK', H.ab.oooo[oa, ob, Oa, Ob], T.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('MnJK,baCIMn->abCIJK', H.ab.oooo[Oa, ob, Oa, Ob], T.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('mNJK,baCmIN->abCIJK', H.ab.oooo[oa, Ob, Oa, Ob], T.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('MNJK,baCIMN->abCIJK', H.ab.oooo[Oa, Ob, Oa, Ob], T.aab.vvVOOO, optimize=True)
    )
    dT.aab.vvVOOO += (1.0 / 4.0) * (
            -0.5 * np.einsum('abef,feCIJK->abCIJK', H.aa.vvvv[va, va, va, va], T.aab.vvVOOO, optimize=True) ###
            - 1.0 * np.einsum('abeF,FeCIJK->abCIJK', H.aa.vvvv[va, va, va, Va], T.aab.VvVOOO, optimize=True)
            - 0.5 * np.einsum('abEF,FECIJK->abCIJK', H.aa.vvvv[va, va, Va, Va], T.aab.VVVOOO, optimize=True)
    )
    dT.aab.vvVOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('bCEf,EafIJK->abCIJK', H.ab.vvvv[va, Vb, Va, vb], T.aab.VvvOOO, optimize=True)
            - 1.0 * np.einsum('bCeF,eaFIJK->abCIJK', H.ab.vvvv[va, Vb, va, Vb], T.aab.vvVOOO, optimize=True)
            - 1.0 * np.einsum('bCEF,EaFIJK->abCIJK', H.ab.vvvv[va, Vb, Va, Vb], T.aab.VvVOOO, optimize=True)
    )
    dT.aab.vvVOOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('amIe,beCmJK->abCIJK', H.aa.voov[va, oa, Oa, va], T.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('aMIe,beCMJK->abCIJK', H.aa.voov[va, Oa, Oa, va], T.aab.vvVOOO, optimize=True)
            + 1.0 * np.einsum('amIE,EbCmJK->abCIJK', H.aa.voov[va, oa, Oa, Va], T.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('aMIE,EbCMJK->abCIJK', H.aa.voov[va, Oa, Oa, Va], T.aab.VvVOOO, optimize=True)
    )
    dT.aab.vvVOOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('amIe,bCeJmK->abCIJK', H.ab.voov[va, ob, Oa, vb], T.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('aMIe,bCeJMK->abCIJK', H.ab.voov[va, Ob, Oa, vb], T.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('amIE,bCEJmK->abCIJK', H.ab.voov[va, ob, Oa, Vb], T.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('aMIE,bCEJMK->abCIJK', H.ab.voov[va, Ob, Oa, Vb], T.abb.vVVOOO, optimize=True)
    )
    dT.aab.vvVOOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('mCEK,EbamIJ->abCIJK', H.ab.ovvo[oa, Vb, Va, Ob], T.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('MCEK,EbaIJM->abCIJK', H.ab.ovvo[Oa, Vb, Va, Ob], T.aaa.VvvOOO, optimize=True)
    )
    dT.aab.vvVOOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('CmKE,baEIJm->abCIJK', H.bb.voov[Vb, ob, Ob, Vb], T.aab.vvVOOo, optimize=True)
            - 1.0 * np.einsum('CMKE,baEIJM->abCIJK', H.bb.voov[Vb, Ob, Ob, Vb], T.aab.vvVOOO, optimize=True)
    )
    dT.aab.vvVOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('ameK,beCIJm->abCIJK', H.ab.vovo[va, ob, va, Ob], T.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('aMeK,beCIJM->abCIJK', H.ab.vovo[va, Ob, va, Ob], T.aab.vvVOOO, optimize=True)
            - 1.0 * np.einsum('amEK,EbCIJm->abCIJK', H.ab.vovo[va, ob, Va, Ob], T.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('aMEK,EbCIJM->abCIJK', H.ab.vovo[va, Ob, Va, Ob], T.aab.VvVOOO, optimize=True)
    )
    dT.aab.vvVOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mCIE,baEmJK->abCIJK', H.ab.ovov[oa, Vb, Oa, Vb], T.aab.vvVoOO, optimize=True)
            + 1.0 * np.einsum('MCIE,baEMJK->abCIJK', H.ab.ovov[Oa, Vb, Oa, Vb], T.aab.vvVOOO, optimize=True)
    )

    dT.aab.vvVOOO -= np.transpose(dT.aab.vvVOOO, (0, 1, 2, 4, 3, 5))
    dT.aab.vvVOOO -= np.transpose(dT.aab.vvVOOO, (1, 0, 2, 3, 4, 5))

    return dT

def update(T, dT, H, shift, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.aab.vvVOOO, dT.aab.vvVOOO = cc_active_loops.update_t3b_001111(
        T.aab.vvVOOO,
        dT.aab.vvVOOO,
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