import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import cc_active_loops

#@profile
def build(T, dT, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    # MM(2,3)
    dT.aab.VvVOOO = (1.0 / 2.0) * (
            +1.0 * np.einsum('bCeK,AeIJ->AbCIJK', H.ab.vvvo[va, Vb, :, Ob], T.aa[Va, :, Oa, Oa], optimize=True)
    )
    dT.aab.VvVOOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('ACeK,beIJ->AbCIJK', H.ab.vvvo[Va, Vb, :, Ob], T.aa[va, :, Oa, Oa], optimize=True)
    )
    dT.aab.VvVOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mCJK,AbIm->AbCIJK', H.ab.ovoo[:, Vb, Oa, Ob], T.aa[Va, va, Oa, :], optimize=True)
    )
    dT.aab.VvVOOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('ACIe,beJK->AbCIJK', H.ab.vvov[Va, Vb, Oa, :], T.ab[va, :, Oa, Ob], optimize=True)
    )
    dT.aab.VvVOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('bCIe,AeJK->AbCIJK', H.ab.vvov[va, Vb, Oa, :], T.ab[Va, :, Oa, Ob], optimize=True)
    )
    dT.aab.VvVOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('AmIK,bCJm->AbCIJK', H.ab.vooo[Va, :, Oa, Ob], T.ab[va, Vb, Oa, :], optimize=True)
    )
    dT.aab.VvVOOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('bmIK,ACJm->AbCIJK', H.ab.vooo[va, :, Oa, Ob], T.ab[Va, Vb, Oa, :], optimize=True)
    )
    dT.aab.VvVOOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('AbIe,eCJK->AbCIJK', H.aa.vvov[Va, va, Oa, :], T.ab[:, Vb, Oa, Ob], optimize=True)
    )
    dT.aab.VvVOOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('AmIJ,bCmK->AbCIJK', H.aa.vooo[Va, :, Oa, Oa], T.ab[va, Vb, :, Ob], optimize=True)
    )
    dT.aab.VvVOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('bmIJ,ACmK->AbCIJK', H.aa.vooo[va, :, Oa, Oa], T.ab[Va, Vb, :, Ob], optimize=True)
    )
    # (H(2) * T3)_C
    dT.aab.VvVOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mI,AbCmJK->AbCIJK', H.a.oo[oa, Oa], T.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('MI,AbCMJK->AbCIJK', H.a.oo[Oa, Oa], T.aab.VvVOOO, optimize=True)
    )
    dT.aab.VvVOOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mK,AbCIJm->AbCIJK', H.b.oo[ob, Ob], T.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('MK,AbCIJM->AbCIJK', H.b.oo[Ob, Ob], T.aab.VvVOOO, optimize=True)
    )
    dT.aab.VvVOOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('Ae,beCIJK->AbCIJK', H.a.vv[Va, va], T.aab.vvVOOO, optimize=True)
            + 1.0 * np.einsum('AE,EbCIJK->AbCIJK', H.a.vv[Va, Va], T.aab.VvVOOO, optimize=True)
    )
    dT.aab.VvVOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('be,AeCIJK->AbCIJK', H.a.vv[va, va], T.aab.VvVOOO, optimize=True)
            + 1.0 * np.einsum('bE,AECIJK->AbCIJK', H.a.vv[va, Va], T.aab.VVVOOO, optimize=True)
    )
    dT.aab.VvVOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('Ce,AbeIJK->AbCIJK', H.b.vv[Vb, vb], T.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('CE,AbEIJK->AbCIJK', H.b.vv[Vb, Vb], T.aab.VvVOOO, optimize=True)
    )
    dT.aab.VvVOOO += (1.0 / 2.0) * (
            +0.5 * np.einsum('mnIJ,AbCmnK->AbCIJK', H.aa.oooo[oa, oa, Oa, Oa], T.aab.VvVooO, optimize=True)
            + 1.0 * np.einsum('mNIJ,AbCmNK->AbCIJK', H.aa.oooo[oa, Oa, Oa, Oa], T.aab.VvVoOO, optimize=True)
            + 0.5 * np.einsum('MNIJ,AbCMNK->AbCIJK', H.aa.oooo[Oa, Oa, Oa, Oa], T.aab.VvVOOO, optimize=True)
    )
    dT.aab.VvVOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mnJK,AbCmIn->AbCIJK', H.ab.oooo[oa, ob, Oa, Ob], T.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('MnJK,AbCIMn->AbCIJK', H.ab.oooo[Oa, ob, Oa, Ob], T.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('mNJK,AbCmIN->AbCIJK', H.ab.oooo[oa, Ob, Oa, Ob], T.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('MNJK,AbCIMN->AbCIJK', H.ab.oooo[Oa, Ob, Oa, Ob], T.aab.VvVOOO, optimize=True)
    )
    dT.aab.VvVOOO += (1.0 / 2.0) * (
            -0.5 * np.einsum('Abef,feCIJK->AbCIJK', H.aa.vvvv[Va, va, va, va], T.aab.vvVOOO, optimize=True)
            + 1.0 * np.einsum('AbEf,EfCIJK->AbCIJK', H.aa.vvvv[Va, va, Va, va], T.aab.VvVOOO, optimize=True)
            - 0.5 * np.einsum('AbEF,FECIJK->AbCIJK', H.aa.vvvv[Va, va, Va, Va], T.aab.VVVOOO, optimize=True)
    )
    dT.aab.VvVOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('bCef,AefIJK->AbCIJK', H.ab.vvvv[va, Vb, va, vb], T.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('bCeF,AeFIJK->AbCIJK', H.ab.vvvv[va, Vb, va, Vb], T.aab.VvVOOO, optimize=True)
            - 1.0 * np.einsum('bCEf,EAfIJK->AbCIJK', H.ab.vvvv[va, Vb, Va, vb], T.aab.VVvOOO, optimize=True)
            - 1.0 * np.einsum('bCEF,EAFIJK->AbCIJK', H.ab.vvvv[va, Vb, Va, Vb], T.aab.VVVOOO, optimize=True)
    )
    dT.aab.VvVOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('ACeF,ebFIJK->AbCIJK', H.ab.vvvv[Va, Vb, va, Vb], T.aab.vvVOOO, optimize=True)
            + 1.0 * np.einsum('ACEf,EbfIJK->AbCIJK', H.ab.vvvv[Va, Vb, Va, vb], T.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('ACEF,EbFIJK->AbCIJK', H.ab.vvvv[Va, Vb, Va, Vb], T.aab.VvVOOO, optimize=True)
    )
    dT.aab.VvVOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('AmIe,beCmJK->AbCIJK', H.aa.voov[Va, oa, Oa, va], T.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('AMIe,beCMJK->AbCIJK', H.aa.voov[Va, Oa, Oa, va], T.aab.vvVOOO, optimize=True)
            + 1.0 * np.einsum('AmIE,EbCmJK->AbCIJK', H.aa.voov[Va, oa, Oa, Va], T.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('AMIE,EbCMJK->AbCIJK', H.aa.voov[Va, Oa, Oa, Va], T.aab.VvVOOO, optimize=True)
    )
    dT.aab.VvVOOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('bmIe,AeCmJK->AbCIJK', H.aa.voov[va, oa, Oa, va], T.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('bMIe,AeCMJK->AbCIJK', H.aa.voov[va, Oa, Oa, va], T.aab.VvVOOO, optimize=True)
            + 1.0 * np.einsum('bmIE,AECmJK->AbCIJK', H.aa.voov[va, oa, Oa, Va], T.aab.VVVoOO, optimize=True)
            + 1.0 * np.einsum('bMIE,AECMJK->AbCIJK', H.aa.voov[va, Oa, Oa, Va], T.aab.VVVOOO, optimize=True)
    )
    dT.aab.VvVOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('AmIe,bCeJmK->AbCIJK', H.ab.voov[Va, ob, Oa, vb], T.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('AMIe,bCeJMK->AbCIJK', H.ab.voov[Va, Ob, Oa, vb], T.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('AmIE,bCEJmK->AbCIJK', H.ab.voov[Va, ob, Oa, Vb], T.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('AMIE,bCEJMK->AbCIJK', H.ab.voov[Va, Ob, Oa, Vb], T.abb.vVVOOO, optimize=True)
    )
    dT.aab.VvVOOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('bmIe,ACeJmK->AbCIJK', H.ab.voov[va, ob, Oa, vb], T.abb.VVvOoO, optimize=True)
            + 1.0 * np.einsum('bMIe,ACeJMK->AbCIJK', H.ab.voov[va, Ob, Oa, vb], T.abb.VVvOOO, optimize=True)
            + 1.0 * np.einsum('bmIE,ACEJmK->AbCIJK', H.ab.voov[va, ob, Oa, Vb], T.abb.VVVOoO, optimize=True)
            + 1.0 * np.einsum('bMIE,ACEJMK->AbCIJK', H.ab.voov[va, Ob, Oa, Vb], T.abb.VVVOOO, optimize=True)
    )
    dT.aab.VvVOOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mCeK,AebmIJ->AbCIJK', H.ab.ovvo[oa, Vb, va, Ob], T.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('MCeK,AebIJM->AbCIJK', H.ab.ovvo[Oa, Vb, va, Ob], T.aaa.VvvOOO, optimize=True)
            + 1.0 * np.einsum('mCEK,EAbmIJ->AbCIJK', H.ab.ovvo[oa, Vb, Va, Ob], T.aaa.VVvoOO, optimize=True)
            + 1.0 * np.einsum('MCEK,EAbIJM->AbCIJK', H.ab.ovvo[Oa, Vb, Va, Ob], T.aaa.VVvOOO, optimize=True)
    )
    dT.aab.VvVOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('CmKe,AbeIJm->AbCIJK', H.bb.voov[Vb, ob, Ob, vb], T.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('CMKe,AbeIJM->AbCIJK', H.bb.voov[Vb, Ob, Ob, vb], T.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('CmKE,AbEIJm->AbCIJK', H.bb.voov[Vb, ob, Ob, Vb], T.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('CMKE,AbEIJM->AbCIJK', H.bb.voov[Vb, Ob, Ob, Vb], T.aab.VvVOOO, optimize=True)
    )
    dT.aab.VvVOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('AmeK,beCIJm->AbCIJK', H.ab.vovo[Va, ob, va, Ob], T.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('AMeK,beCIJM->AbCIJK', H.ab.vovo[Va, Ob, va, Ob], T.aab.vvVOOO, optimize=True)
            - 1.0 * np.einsum('AmEK,EbCIJm->AbCIJK', H.ab.vovo[Va, ob, Va, Ob], T.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('AMEK,EbCIJM->AbCIJK', H.ab.vovo[Va, Ob, Va, Ob], T.aab.VvVOOO, optimize=True)
    )
    dT.aab.VvVOOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('bmeK,AeCIJm->AbCIJK', H.ab.vovo[va, ob, va, Ob], T.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('bMeK,AeCIJM->AbCIJK', H.ab.vovo[va, Ob, va, Ob], T.aab.VvVOOO, optimize=True)
            - 1.0 * np.einsum('bmEK,AECIJm->AbCIJK', H.ab.vovo[va, ob, Va, Ob], T.aab.VVVOOo, optimize=True)
            - 1.0 * np.einsum('bMEK,AECIJM->AbCIJK', H.ab.vovo[va, Ob, Va, Ob], T.aab.VVVOOO, optimize=True)
    )
    dT.aab.VvVOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mCIe,AbemJK->AbCIJK', H.ab.ovov[oa, Vb, Oa, vb], T.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('MCIe,AbeMJK->AbCIJK', H.ab.ovov[Oa, Vb, Oa, vb], T.aab.VvvOOO, optimize=True)
            - 1.0 * np.einsum('mCIE,AbEmJK->AbCIJK', H.ab.ovov[oa, Vb, Oa, Vb], T.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('MCIE,AbEMJK->AbCIJK', H.ab.ovov[Oa, Vb, Oa, Vb], T.aab.VvVOOO, optimize=True)
    )

    dT.aab.VvVOOO -= np.transpose(dT.aab.VvVOOO, (0, 1, 2, 4, 3, 5))

    return dT

def update(T, dT, H, shift, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.aab.VvVOOO, dT.aab.VvVOOO = cc_active_loops.update_t3b_101111(
        T.aab.VvVOOO,
        dT.aab.VvVOOO,
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