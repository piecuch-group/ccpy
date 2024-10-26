import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops

def build(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.aab.vvVoOO = (2.0 / 2.0) * (
            +1.0 * np.einsum('bCeK,aeiJ->abCiJK', X.ab.vvvo[va, Vb, :, Ob], T.aa[va, :, oa, Oa], optimize=True)
    )
    dR.aab.vvVoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mCJK,abim->abCiJK', X.ab.ovoo[:, Vb, Oa, Ob], T.aa[va, va, oa, :], optimize=True)
    )
    dR.aab.vvVoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mCiK,abJm->abCiJK', X.ab.ovoo[:, Vb, oa, Ob], T.aa[va, va, Oa, :], optimize=True)
    )
    dR.aab.vvVoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('aCie,beJK->abCiJK', X.ab.vvov[va, Vb, oa, :], T.ab[va, :, Oa, Ob], optimize=True)
    )
    dR.aab.vvVoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('aCJe,beiK->abCiJK', X.ab.vvov[va, Vb, Oa, :], T.ab[va, :, oa, Ob], optimize=True)
    )
    dR.aab.vvVoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('amiK,bCJm->abCiJK', X.ab.vooo[va, :, oa, Ob], T.ab[va, Vb, Oa, :], optimize=True)
    )
    dR.aab.vvVoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('amJK,bCim->abCiJK', X.ab.vooo[va, :, Oa, Ob], T.ab[va, Vb, oa, :], optimize=True)
    )
    dR.aab.vvVoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('abie,eCJK->abCiJK', X.aa.vvov[va, va, oa, :], T.ab[:, Vb, Oa, Ob], optimize=True)
    )
    dR.aab.vvVoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('abJe,eCiK->abCiJK', X.aa.vvov[va, va, Oa, :], T.ab[:, Vb, oa, Ob], optimize=True)
    )
    dR.aab.vvVoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('amiJ,bCmK->abCiJK', X.aa.vooo[va, :, oa, Oa], T.ab[va, Vb, :, Ob], optimize=True)
    )
    dR.aab.vvVoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('bCeK,aeiJ->abCiJK', H.ab.vvvo[va, Vb, :, Ob], R.aa[va, :, oa, Oa], optimize=True)
    )
    dR.aab.vvVoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mCJK,abim->abCiJK', H.ab.ovoo[:, Vb, Oa, Ob], R.aa[va, va, oa, :], optimize=True)
    )
    dR.aab.vvVoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mCiK,abJm->abCiJK', H.ab.ovoo[:, Vb, oa, Ob], R.aa[va, va, Oa, :], optimize=True)
    )
    dR.aab.vvVoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('aCie,beJK->abCiJK', H.ab.vvov[va, Vb, oa, :], R.ab[va, :, Oa, Ob], optimize=True)
    )
    dR.aab.vvVoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('aCJe,beiK->abCiJK', H.ab.vvov[va, Vb, Oa, :], R.ab[va, :, oa, Ob], optimize=True)
    )
    dR.aab.vvVoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('amiK,bCJm->abCiJK', H.ab.vooo[va, :, oa, Ob], R.ab[va, Vb, Oa, :], optimize=True)
    )
    dR.aab.vvVoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('amJK,bCim->abCiJK', H.ab.vooo[va, :, Oa, Ob], R.ab[va, Vb, oa, :], optimize=True)
    )
    dR.aab.vvVoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('abie,eCJK->abCiJK', H.aa.vvov[va, va, oa, :], R.ab[:, Vb, Oa, Ob], optimize=True)
    )
    dR.aab.vvVoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('abJe,eCiK->abCiJK', H.aa.vvov[va, va, Oa, :], R.ab[:, Vb, oa, Ob], optimize=True)
    )
    dR.aab.vvVoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('amiJ,bCmK->abCiJK', H.aa.vooo[va, :, oa, Oa], R.ab[va, Vb, :, Ob], optimize=True)
    )

    dR.aab.vvVoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mi,baCmJK->abCiJK', X.a.oo[oa, oa], T.aab.vvVoOO, optimize=True)
            + 1.0 * np.einsum('Mi,baCMJK->abCiJK', X.a.oo[Oa, oa], T.aab.vvVOOO, optimize=True)
    )
    dR.aab.vvVoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mJ,baCmiK->abCiJK', X.a.oo[oa, Oa], T.aab.vvVooO, optimize=True)
            + 1.0 * np.einsum('MJ,baCiMK->abCiJK', X.a.oo[Oa, Oa], T.aab.vvVoOO, optimize=True)
    )
    dR.aab.vvVoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mK,baCiJm->abCiJK', X.b.oo[ob, Ob], T.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('MK,baCiJM->abCiJK', X.b.oo[Ob, Ob], T.aab.vvVoOO, optimize=True)
    )
    dR.aab.vvVoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('ae,beCiJK->abCiJK', X.a.vv[va, va], T.aab.vvVoOO, optimize=True)
            + 1.0 * np.einsum('aE,EbCiJK->abCiJK', X.a.vv[va, Va], T.aab.VvVoOO, optimize=True)
    )
    dR.aab.vvVoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('CE,baEiJK->abCiJK', X.b.vv[Vb, Vb], T.aab.vvVoOO, optimize=True)
    )
    dR.aab.vvVoOO += (1.0 / 2.0) * (
            -0.5 * np.einsum('mniJ,baCmnK->abCiJK', X.aa.oooo[oa, oa, oa, Oa], T.aab.vvVooO, optimize=True)
            - 1.0 * np.einsum('mNiJ,baCmNK->abCiJK', X.aa.oooo[oa, Oa, oa, Oa], T.aab.vvVoOO, optimize=True)
            - 0.5 * np.einsum('MNiJ,baCMNK->abCiJK', X.aa.oooo[Oa, Oa, oa, Oa], T.aab.vvVOOO, optimize=True)
    )
    dR.aab.vvVoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('MnJK,baCiMn->abCiJK', X.ab.oooo[Oa, ob, Oa, Ob], T.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('mNJK,baCimN->abCiJK', X.ab.oooo[oa, Ob, Oa, Ob], T.aab.vvVooO, optimize=True)
            - 1.0 * np.einsum('MNJK,baCiMN->abCiJK', X.ab.oooo[Oa, Ob, Oa, Ob], T.aab.vvVoOO, optimize=True)
    )
    dR.aab.vvVoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mniK,baCmJn->abCiJK', X.ab.oooo[oa, ob, oa, Ob], T.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('MniK,baCJMn->abCiJK', X.ab.oooo[Oa, ob, oa, Ob], T.aab.vvVOOo, optimize=True)
            - 1.0 * np.einsum('mNiK,baCmJN->abCiJK', X.ab.oooo[oa, Ob, oa, Ob], T.aab.vvVoOO, optimize=True)
            + 1.0 * np.einsum('MNiK,baCJMN->abCiJK', X.ab.oooo[Oa, Ob, oa, Ob], T.aab.vvVOOO, optimize=True)
    )
    dR.aab.vvVoOO += (1.0 / 2.0) * (
            -0.5 * np.einsum('abef,feCiJK->abCiJK', X.aa.vvvv[va, va, va, va], T.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('abeF,FeCiJK->abCiJK', X.aa.vvvv[va, va, va, Va], T.aab.VvVoOO, optimize=True)
            - 0.5 * np.einsum('abEF,FECiJK->abCiJK', X.aa.vvvv[va, va, Va, Va], T.aab.VVVoOO, optimize=True)
    )
    dR.aab.vvVoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('bCEf,EafiJK->abCiJK', X.ab.vvvv[va, Vb, Va, vb], T.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('bCeF,eaFiJK->abCiJK', X.ab.vvvv[va, Vb, va, Vb], T.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('bCEF,EaFiJK->abCiJK', X.ab.vvvv[va, Vb, Va, Vb], T.aab.VvVoOO, optimize=True)
    )
    dR.aab.vvVoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('amie,beCmJK->abCiJK', X.aa.voov[va, oa, oa, va], T.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('aMie,beCMJK->abCiJK', X.aa.voov[va, Oa, oa, va], T.aab.vvVOOO, optimize=True)
            + 1.0 * np.einsum('amiE,EbCmJK->abCiJK', X.aa.voov[va, oa, oa, Va], T.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('aMiE,EbCMJK->abCiJK', X.aa.voov[va, Oa, oa, Va], T.aab.VvVOOO, optimize=True)
    )
    dR.aab.vvVoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('amJe,beCmiK->abCiJK', X.aa.voov[va, oa, Oa, va], T.aab.vvVooO, optimize=True)
            - 1.0 * np.einsum('aMJe,beCiMK->abCiJK', X.aa.voov[va, Oa, Oa, va], T.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('amJE,EbCmiK->abCiJK', X.aa.voov[va, oa, Oa, Va], T.aab.VvVooO, optimize=True)
            + 1.0 * np.einsum('aMJE,EbCiMK->abCiJK', X.aa.voov[va, Oa, Oa, Va], T.aab.VvVoOO, optimize=True)
    )
    dR.aab.vvVoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('amie,bCeJmK->abCiJK', X.ab.voov[va, ob, oa, vb], T.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('aMie,bCeJMK->abCiJK', X.ab.voov[va, Ob, oa, vb], T.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('amiE,bCEJmK->abCiJK', X.ab.voov[va, ob, oa, Vb], T.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('aMiE,bCEJMK->abCiJK', X.ab.voov[va, Ob, oa, Vb], T.abb.vVVOOO, optimize=True)
    )
    dR.aab.vvVoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('amJe,bCeimK->abCiJK', X.ab.voov[va, ob, Oa, vb], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('aMJe,bCeiMK->abCiJK', X.ab.voov[va, Ob, Oa, vb], T.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('amJE,bCEimK->abCiJK', X.ab.voov[va, ob, Oa, Vb], T.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('aMJE,bCEiMK->abCiJK', X.ab.voov[va, Ob, Oa, Vb], T.abb.vVVoOO, optimize=True)
    )
    dR.aab.vvVoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mCEK,EbaimJ->abCiJK', X.ab.ovvo[oa, Vb, Va, Ob], T.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('MCEK,EbaiJM->abCiJK', X.ab.ovvo[Oa, Vb, Va, Ob], T.aaa.VvvoOO, optimize=True)
    )
    dR.aab.vvVoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('CmKE,baEiJm->abCiJK', X.bb.voov[Vb, ob, Ob, Vb], T.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('CMKE,baEiJM->abCiJK', X.bb.voov[Vb, Ob, Ob, Vb], T.aab.vvVoOO, optimize=True)
    )
    dR.aab.vvVoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('ameK,beCiJm->abCiJK', X.ab.vovo[va, ob, va, Ob], T.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('aMeK,beCiJM->abCiJK', X.ab.vovo[va, Ob, va, Ob], T.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('amEK,EbCiJm->abCiJK', X.ab.vovo[va, ob, Va, Ob], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('aMEK,EbCiJM->abCiJK', X.ab.vovo[va, Ob, Va, Ob], T.aab.VvVoOO, optimize=True)
    )
    dR.aab.vvVoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mCiE,baEmJK->abCiJK', X.ab.ovov[oa, Vb, oa, Vb], T.aab.vvVoOO, optimize=True)
            + 1.0 * np.einsum('MCiE,baEMJK->abCiJK', X.ab.ovov[Oa, Vb, oa, Vb], T.aab.vvVOOO, optimize=True)
    )
    dR.aab.vvVoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mCJE,baEmiK->abCiJK', X.ab.ovov[oa, Vb, Oa, Vb], T.aab.vvVooO, optimize=True)
            + 1.0 * np.einsum('MCJE,baEiMK->abCiJK', X.ab.ovov[Oa, Vb, Oa, Vb], T.aab.vvVoOO, optimize=True)
    )
    dR.aab.vvVoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mi,baCmJK->abCiJK', H.a.oo[oa, oa], R.aab.vvVoOO, optimize=True)
            + 1.0 * np.einsum('Mi,baCMJK->abCiJK', H.a.oo[Oa, oa], R.aab.vvVOOO, optimize=True)
    )
    dR.aab.vvVoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mJ,baCmiK->abCiJK', H.a.oo[oa, Oa], R.aab.vvVooO, optimize=True)
            + 1.0 * np.einsum('MJ,baCiMK->abCiJK', H.a.oo[Oa, Oa], R.aab.vvVoOO, optimize=True)
    )
    dR.aab.vvVoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mK,baCiJm->abCiJK', H.b.oo[ob, Ob], R.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('MK,baCiJM->abCiJK', H.b.oo[Ob, Ob], R.aab.vvVoOO, optimize=True)
    )
    dR.aab.vvVoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('ae,beCiJK->abCiJK', H.a.vv[va, va], R.aab.vvVoOO, optimize=True)
            + 1.0 * np.einsum('aE,EbCiJK->abCiJK', H.a.vv[va, Va], R.aab.VvVoOO, optimize=True)
    )
    dR.aab.vvVoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('CE,baEiJK->abCiJK', H.b.vv[Vb, Vb], R.aab.vvVoOO, optimize=True)
    )
    dR.aab.vvVoOO += (1.0 / 2.0) * (
            -0.5 * np.einsum('mniJ,baCmnK->abCiJK', H.aa.oooo[oa, oa, oa, Oa], R.aab.vvVooO, optimize=True)
            - 1.0 * np.einsum('mNiJ,baCmNK->abCiJK', H.aa.oooo[oa, Oa, oa, Oa], R.aab.vvVoOO, optimize=True)
            - 0.5 * np.einsum('MNiJ,baCMNK->abCiJK', H.aa.oooo[Oa, Oa, oa, Oa], R.aab.vvVOOO, optimize=True)
    )
    dR.aab.vvVoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('MnJK,baCiMn->abCiJK', H.ab.oooo[Oa, ob, Oa, Ob], R.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('mNJK,baCimN->abCiJK', H.ab.oooo[oa, Ob, Oa, Ob], R.aab.vvVooO, optimize=True)
            - 1.0 * np.einsum('MNJK,baCiMN->abCiJK', H.ab.oooo[Oa, Ob, Oa, Ob], R.aab.vvVoOO, optimize=True)
    )
    dR.aab.vvVoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mniK,baCmJn->abCiJK', H.ab.oooo[oa, ob, oa, Ob], R.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('MniK,baCJMn->abCiJK', H.ab.oooo[Oa, ob, oa, Ob], R.aab.vvVOOo, optimize=True)
            - 1.0 * np.einsum('mNiK,baCmJN->abCiJK', H.ab.oooo[oa, Ob, oa, Ob], R.aab.vvVoOO, optimize=True)
            + 1.0 * np.einsum('MNiK,baCJMN->abCiJK', H.ab.oooo[Oa, Ob, oa, Ob], R.aab.vvVOOO, optimize=True)
    )
    dR.aab.vvVoOO += (1.0 / 2.0) * (
            -0.5 * np.einsum('abef,feCiJK->abCiJK', H.aa.vvvv[va, va, va, va], R.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('abeF,FeCiJK->abCiJK', H.aa.vvvv[va, va, va, Va], R.aab.VvVoOO, optimize=True)
            - 0.5 * np.einsum('abEF,FECiJK->abCiJK', H.aa.vvvv[va, va, Va, Va], R.aab.VVVoOO, optimize=True)
    )
    dR.aab.vvVoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('bCEf,EafiJK->abCiJK', H.ab.vvvv[va, Vb, Va, vb], R.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('bCeF,eaFiJK->abCiJK', H.ab.vvvv[va, Vb, va, Vb], R.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('bCEF,EaFiJK->abCiJK', H.ab.vvvv[va, Vb, Va, Vb], R.aab.VvVoOO, optimize=True)
    )
    dR.aab.vvVoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('amie,beCmJK->abCiJK', H.aa.voov[va, oa, oa, va], R.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('aMie,beCMJK->abCiJK', H.aa.voov[va, Oa, oa, va], R.aab.vvVOOO, optimize=True)
            + 1.0 * np.einsum('amiE,EbCmJK->abCiJK', H.aa.voov[va, oa, oa, Va], R.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('aMiE,EbCMJK->abCiJK', H.aa.voov[va, Oa, oa, Va], R.aab.VvVOOO, optimize=True)
    )
    dR.aab.vvVoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('amJe,beCmiK->abCiJK', H.aa.voov[va, oa, Oa, va], R.aab.vvVooO, optimize=True)
            - 1.0 * np.einsum('aMJe,beCiMK->abCiJK', H.aa.voov[va, Oa, Oa, va], R.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('amJE,EbCmiK->abCiJK', H.aa.voov[va, oa, Oa, Va], R.aab.VvVooO, optimize=True)
            + 1.0 * np.einsum('aMJE,EbCiMK->abCiJK', H.aa.voov[va, Oa, Oa, Va], R.aab.VvVoOO, optimize=True)
    )
    dR.aab.vvVoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('amie,bCeJmK->abCiJK', H.ab.voov[va, ob, oa, vb], R.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('aMie,bCeJMK->abCiJK', H.ab.voov[va, Ob, oa, vb], R.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('amiE,bCEJmK->abCiJK', H.ab.voov[va, ob, oa, Vb], R.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('aMiE,bCEJMK->abCiJK', H.ab.voov[va, Ob, oa, Vb], R.abb.vVVOOO, optimize=True)
    )
    dR.aab.vvVoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('amJe,bCeimK->abCiJK', H.ab.voov[va, ob, Oa, vb], R.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('aMJe,bCeiMK->abCiJK', H.ab.voov[va, Ob, Oa, vb], R.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('amJE,bCEimK->abCiJK', H.ab.voov[va, ob, Oa, Vb], R.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('aMJE,bCEiMK->abCiJK', H.ab.voov[va, Ob, Oa, Vb], R.abb.vVVoOO, optimize=True)
    )
    dR.aab.vvVoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mCEK,EbaimJ->abCiJK', H.ab.ovvo[oa, Vb, Va, Ob], R.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('MCEK,EbaiJM->abCiJK', H.ab.ovvo[Oa, Vb, Va, Ob], R.aaa.VvvoOO, optimize=True)
    )
    dR.aab.vvVoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('CmKE,baEiJm->abCiJK', H.bb.voov[Vb, ob, Ob, Vb], R.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('CMKE,baEiJM->abCiJK', H.bb.voov[Vb, Ob, Ob, Vb], R.aab.vvVoOO, optimize=True)
    )
    dR.aab.vvVoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('ameK,beCiJm->abCiJK', H.ab.vovo[va, ob, va, Ob], R.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('aMeK,beCiJM->abCiJK', H.ab.vovo[va, Ob, va, Ob], R.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('amEK,EbCiJm->abCiJK', H.ab.vovo[va, ob, Va, Ob], R.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('aMEK,EbCiJM->abCiJK', H.ab.vovo[va, Ob, Va, Ob], R.aab.VvVoOO, optimize=True)
    )
    dR.aab.vvVoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mCiE,baEmJK->abCiJK', H.ab.ovov[oa, Vb, oa, Vb], R.aab.vvVoOO, optimize=True)
            + 1.0 * np.einsum('MCiE,baEMJK->abCiJK', H.ab.ovov[Oa, Vb, oa, Vb], R.aab.vvVOOO, optimize=True)
    )
    dR.aab.vvVoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mCJE,baEmiK->abCiJK', H.ab.ovov[oa, Vb, Oa, Vb], R.aab.vvVooO, optimize=True)
            + 1.0 * np.einsum('MCJE,baEiMK->abCiJK', H.ab.ovov[Oa, Vb, Oa, Vb], R.aab.vvVoOO, optimize=True)
    )

    dR.aab.vvVoOO -= np.transpose(dR.aab.vvVoOO, (1, 0, 2, 3, 4, 5))

    return dR

def update(R, omega, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    R.aab.vvVoOO = eomcc_active_loops.update_r3b_001011(
        R.aab.vvVoOO,
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
