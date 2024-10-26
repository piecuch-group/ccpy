import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import cc_active_loops

#@profile
def build(T, dT, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    # MM(2,3)
    dT.aab.vvVoOO = (2.0 / 2.0) * (
            +1.0 * np.einsum('bCeK,aeiJ->abCiJK', H.ab.vvvo[va, Vb, :, Ob], T.aa[va, :, oa, Oa], optimize=True)
    )
    dT.aab.vvVoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mCJK,abim->abCiJK', H.ab.ovoo[:, Vb, Oa, Ob], T.aa[va, va, oa, :], optimize=True)
    )
    dT.aab.vvVoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mCiK,abJm->abCiJK', H.ab.ovoo[:, Vb, oa, Ob], T.aa[va, va, Oa, :], optimize=True)
    )
    dT.aab.vvVoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('aCie,beJK->abCiJK', H.ab.vvov[va, Vb, oa, :], T.ab[va, :, Oa, Ob], optimize=True)
    )
    dT.aab.vvVoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('aCJe,beiK->abCiJK', H.ab.vvov[va, Vb, Oa, :], T.ab[va, :, oa, Ob], optimize=True)
    )
    dT.aab.vvVoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('amiK,bCJm->abCiJK', H.ab.vooo[va, :, oa, Ob], T.ab[va, Vb, Oa, :], optimize=True)
    )
    dT.aab.vvVoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('amJK,bCim->abCiJK', H.ab.vooo[va, :, Oa, Ob], T.ab[va, Vb, oa, :], optimize=True)
    )
    dT.aab.vvVoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('abie,eCJK->abCiJK', H.aa.vvov[va, va, oa, :], T.ab[:, Vb, Oa, Ob], optimize=True)
    )
    dT.aab.vvVoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('abJe,eCiK->abCiJK', H.aa.vvov[va, va, Oa, :], T.ab[:, Vb, oa, Ob], optimize=True)
    )
    dT.aab.vvVoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('amiJ,bCmK->abCiJK', H.aa.vooo[va, :, oa, Oa], T.ab[va, Vb, :, Ob], optimize=True)
    )
    # (H(2) * T3)_C
    dT.aab.vvVoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mi,baCmJK->abCiJK', H.a.oo[oa, oa], T.aab.vvVoOO, optimize=True)
            + 1.0 * np.einsum('Mi,baCMJK->abCiJK', H.a.oo[Oa, oa], T.aab.vvVOOO, optimize=True)
    )
    dT.aab.vvVoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mJ,baCmiK->abCiJK', H.a.oo[oa, Oa], T.aab.vvVooO, optimize=True)
            + 1.0 * np.einsum('MJ,baCiMK->abCiJK', H.a.oo[Oa, Oa], T.aab.vvVoOO, optimize=True)
    )
    dT.aab.vvVoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mK,baCiJm->abCiJK', H.b.oo[ob, Ob], T.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('MK,baCiJM->abCiJK', H.b.oo[Ob, Ob], T.aab.vvVoOO, optimize=True)
    )
    dT.aab.vvVoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('ae,beCiJK->abCiJK', H.a.vv[va, va], T.aab.vvVoOO, optimize=True)
            + 1.0 * np.einsum('aE,EbCiJK->abCiJK', H.a.vv[va, Va], T.aab.VvVoOO, optimize=True)
    )
    dT.aab.vvVoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('CE,baEiJK->abCiJK', H.b.vv[Vb, Vb], T.aab.vvVoOO, optimize=True)
    )
    dT.aab.vvVoOO += (1.0 / 2.0) * (
            -0.5 * np.einsum('mniJ,baCmnK->abCiJK', H.aa.oooo[oa, oa, oa, Oa], T.aab.vvVooO, optimize=True)
            - 1.0 * np.einsum('mNiJ,baCmNK->abCiJK', H.aa.oooo[oa, Oa, oa, Oa], T.aab.vvVoOO, optimize=True)
            - 0.5 * np.einsum('MNiJ,baCMNK->abCiJK', H.aa.oooo[Oa, Oa, oa, Oa], T.aab.vvVOOO, optimize=True)
    )
    dT.aab.vvVoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('MnJK,baCiMn->abCiJK', H.ab.oooo[Oa, ob, Oa, Ob], T.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('mNJK,baCimN->abCiJK', H.ab.oooo[oa, Ob, Oa, Ob], T.aab.vvVooO, optimize=True)
            - 1.0 * np.einsum('MNJK,baCiMN->abCiJK', H.ab.oooo[Oa, Ob, Oa, Ob], T.aab.vvVoOO, optimize=True)
    )
    dT.aab.vvVoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mniK,baCmJn->abCiJK', H.ab.oooo[oa, ob, oa, Ob], T.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('MniK,baCJMn->abCiJK', H.ab.oooo[Oa, ob, oa, Ob], T.aab.vvVOOo, optimize=True)
            - 1.0 * np.einsum('mNiK,baCmJN->abCiJK', H.ab.oooo[oa, Ob, oa, Ob], T.aab.vvVoOO, optimize=True)
            + 1.0 * np.einsum('MNiK,baCJMN->abCiJK', H.ab.oooo[Oa, Ob, oa, Ob], T.aab.vvVOOO, optimize=True)
    )
    dT.aab.vvVoOO += (1.0 / 2.0) * (
            -0.5 * np.einsum('abef,feCiJK->abCiJK', H.aa.vvvv[va, va, va, va], T.aab.vvVoOO, optimize=True) ###
            - 1.0 * np.einsum('abeF,FeCiJK->abCiJK', H.aa.vvvv[va, va, va, Va], T.aab.VvVoOO, optimize=True)
            - 0.5 * np.einsum('abEF,FECiJK->abCiJK', H.aa.vvvv[va, va, Va, Va], T.aab.VVVoOO, optimize=True)
    )
    dT.aab.vvVoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('bCEf,EafiJK->abCiJK', H.ab.vvvv[va, Vb, Va, vb], T.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('bCeF,eaFiJK->abCiJK', H.ab.vvvv[va, Vb, va, Vb], T.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('bCEF,EaFiJK->abCiJK', H.ab.vvvv[va, Vb, Va, Vb], T.aab.VvVoOO, optimize=True)
    )
    dT.aab.vvVoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('amie,beCmJK->abCiJK', H.aa.voov[va, oa, oa, va], T.aab.vvVoOO, optimize=True)
            + 1.0 * np.einsum('amiE,EbCmJK->abCiJK', H.aa.voov[va, oa, oa, Va], T.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('aMie,beCMJK->abCiJK', H.aa.voov[va, Oa, oa, va], T.aab.vvVOOO, optimize=True)
            + 1.0 * np.einsum('aMiE,EbCMJK->abCiJK', H.aa.voov[va, Oa, oa, Va], T.aab.VvVOOO, optimize=True)
    )
    dT.aab.vvVoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('amJe,beCmiK->abCiJK', H.aa.voov[va, oa, Oa, va], T.aab.vvVooO, optimize=True)
            - 1.0 * np.einsum('amJE,EbCmiK->abCiJK', H.aa.voov[va, oa, Oa, Va], T.aab.VvVooO, optimize=True)
            - 1.0 * np.einsum('aMJe,beCiMK->abCiJK', H.aa.voov[va, Oa, Oa, va], T.aab.vvVoOO, optimize=True)
            + 1.0 * np.einsum('aMJE,EbCiMK->abCiJK', H.aa.voov[va, Oa, Oa, Va], T.aab.VvVoOO, optimize=True)
    )
    dT.aab.vvVoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('amie,bCeJmK->abCiJK', H.ab.voov[va, ob, oa, vb], T.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('amiE,bCEJmK->abCiJK', H.ab.voov[va, ob, oa, Vb], T.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('aMie,bCeJMK->abCiJK', H.ab.voov[va, Ob, oa, vb], T.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('aMiE,bCEJMK->abCiJK', H.ab.voov[va, Ob, oa, Vb], T.abb.vVVOOO, optimize=True)
    )
    dT.aab.vvVoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('amJe,bCeimK->abCiJK', H.ab.voov[va, ob, Oa, vb], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('amJE,bCEimK->abCiJK', H.ab.voov[va, ob, Oa, Vb], T.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('aMJe,bCeiMK->abCiJK', H.ab.voov[va, Ob, Oa, vb], T.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('aMJE,bCEiMK->abCiJK', H.ab.voov[va, Ob, Oa, Vb], T.abb.vVVoOO, optimize=True)
    )
    dT.aab.vvVoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mCEK,EbaimJ->abCiJK', H.ab.ovvo[oa, Vb, Va, Ob], T.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('MCEK,EbaiJM->abCiJK', H.ab.ovvo[Oa, Vb, Va, Ob], T.aaa.VvvoOO, optimize=True)
    )
    dT.aab.vvVoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('CmKE,baEiJm->abCiJK', H.bb.voov[Vb, ob, Ob, Vb], T.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('CMKE,baEiJM->abCiJK', H.bb.voov[Vb, Ob, Ob, Vb], T.aab.vvVoOO, optimize=True)
    )
    dT.aab.vvVoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('ameK,beCiJm->abCiJK', H.ab.vovo[va, ob, va, Ob], T.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('amEK,EbCiJm->abCiJK', H.ab.vovo[va, ob, Va, Ob], T.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('aMeK,beCiJM->abCiJK', H.ab.vovo[va, Ob, va, Ob], T.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('aMEK,EbCiJM->abCiJK', H.ab.vovo[va, Ob, Va, Ob], T.aab.VvVoOO, optimize=True)
    )
    dT.aab.vvVoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mCiE,baEmJK->abCiJK', H.ab.ovov[oa, Vb, oa, Vb], T.aab.vvVoOO, optimize=True)
            + 1.0 * np.einsum('MCiE,baEMJK->abCiJK', H.ab.ovov[Oa, Vb, oa, Vb], T.aab.vvVOOO, optimize=True)
    )
    dT.aab.vvVoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mCJE,baEmiK->abCiJK', H.ab.ovov[oa, Vb, Oa, Vb], T.aab.vvVooO, optimize=True)
            + 1.0 * np.einsum('MCJE,baEiMK->abCiJK', H.ab.ovov[Oa, Vb, Oa, Vb], T.aab.vvVoOO, optimize=True)
    )

    dT.aab.vvVoOO -= np.transpose(dT.aab.vvVoOO, (1, 0, 2, 3, 4, 5))

    return dT

def update(T, dT, H, shift, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.aab.vvVoOO, dT.aab.vvVoOO = cc_active_loops.update_t3b_001011(
        T.aab.vvVoOO,
        dT.aab.vvVoOO,
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