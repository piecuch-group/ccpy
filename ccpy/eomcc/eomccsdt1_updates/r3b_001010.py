import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops

def build(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.aab.vvVoOo = (2.0 / 2.0) * (
            +1.0 * np.einsum('bCek,aeiJ->abCiJk', X.ab.vvvo[va, Vb, :, ob], T.aa[va, :, oa, Oa], optimize=True)
    )
    dR.aab.vvVoOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('mCJk,abim->abCiJk', X.ab.ovoo[:, Vb, Oa, ob], T.aa[va, va, oa, :], optimize=True)
    )
    dR.aab.vvVoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('mCik,abJm->abCiJk', X.ab.ovoo[:, Vb, oa, ob], T.aa[va, va, Oa, :], optimize=True)
    )
    dR.aab.vvVoOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('aCie,beJk->abCiJk', X.ab.vvov[va, Vb, oa, :], T.ab[va, :, Oa, ob], optimize=True)
    )
    dR.aab.vvVoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('aCJe,beik->abCiJk', X.ab.vvov[va, Vb, Oa, :], T.ab[va, :, oa, ob], optimize=True)
    )
    dR.aab.vvVoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('amik,bCJm->abCiJk', X.ab.vooo[va, :, oa, ob], T.ab[va, Vb, Oa, :], optimize=True)
    )
    dR.aab.vvVoOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('amJk,bCim->abCiJk', X.ab.vooo[va, :, Oa, ob], T.ab[va, Vb, oa, :], optimize=True)
    )
    dR.aab.vvVoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('abie,eCJk->abCiJk', X.aa.vvov[va, va, oa, :], T.ab[:, Vb, Oa, ob], optimize=True)
    )
    dR.aab.vvVoOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('abJe,eCik->abCiJk', X.aa.vvov[va, va, Oa, :], T.ab[:, Vb, oa, ob], optimize=True)
    )
    dR.aab.vvVoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('amiJ,bCmk->abCiJk', X.aa.vooo[va, :, oa, Oa], T.ab[va, Vb, :, ob], optimize=True)
    )
    dR.aab.vvVoOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('bCek,aeiJ->abCiJk', H.ab.vvvo[va, Vb, :, ob], R.aa[va, :, oa, Oa], optimize=True)
    )
    dR.aab.vvVoOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('mCJk,abim->abCiJk', H.ab.ovoo[:, Vb, Oa, ob], R.aa[va, va, oa, :], optimize=True)
    )
    dR.aab.vvVoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('mCik,abJm->abCiJk', H.ab.ovoo[:, Vb, oa, ob], R.aa[va, va, Oa, :], optimize=True)
    )
    dR.aab.vvVoOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('aCie,beJk->abCiJk', H.ab.vvov[va, Vb, oa, :], R.ab[va, :, Oa, ob], optimize=True)
    )
    dR.aab.vvVoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('aCJe,beik->abCiJk', H.ab.vvov[va, Vb, Oa, :], R.ab[va, :, oa, ob], optimize=True)
    )
    dR.aab.vvVoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('amik,bCJm->abCiJk', H.ab.vooo[va, :, oa, ob], R.ab[va, Vb, Oa, :], optimize=True)
    )
    dR.aab.vvVoOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('amJk,bCim->abCiJk', H.ab.vooo[va, :, Oa, ob], R.ab[va, Vb, oa, :], optimize=True)
    )
    dR.aab.vvVoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('abie,eCJk->abCiJk', H.aa.vvov[va, va, oa, :], R.ab[:, Vb, Oa, ob], optimize=True)
    )
    dR.aab.vvVoOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('abJe,eCik->abCiJk', H.aa.vvov[va, va, Oa, :], R.ab[:, Vb, oa, ob], optimize=True)
    )
    dR.aab.vvVoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('amiJ,bCmk->abCiJk', H.aa.vooo[va, :, oa, Oa], R.ab[va, Vb, :, ob], optimize=True)
    )

    dR.aab.vvVoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('mi,baCmJk->abCiJk', X.a.oo[oa, oa], T.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('Mi,baCMJk->abCiJk', X.a.oo[Oa, oa], T.aab.vvVOOo, optimize=True)
    )
    dR.aab.vvVoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('MJ,baCiMk->abCiJk', X.a.oo[Oa, Oa], T.aab.vvVoOo, optimize=True)
    )
    dR.aab.vvVoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('mk,baCiJm->abCiJk', X.b.oo[ob, ob], T.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('Mk,baCiJM->abCiJk', X.b.oo[Ob, ob], T.aab.vvVoOO, optimize=True)
    )
    dR.aab.vvVoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('ae,beCiJk->abCiJk', X.a.vv[va, va], T.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('aE,EbCiJk->abCiJk', X.a.vv[va, Va], T.aab.VvVoOo, optimize=True)
    )
    dR.aab.vvVoOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('CE,baEiJk->abCiJk', X.b.vv[Vb, Vb], T.aab.vvVoOo, optimize=True)
    )
    dR.aab.vvVoOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('mNiJ,baCmNk->abCiJk', X.aa.oooo[oa, Oa, oa, Oa], T.aab.vvVoOo, optimize=True)
            - 0.5 * np.einsum('MNiJ,baCMNk->abCiJk', X.aa.oooo[Oa, Oa, oa, Oa], T.aab.vvVOOo, optimize=True)
    )
    dR.aab.vvVoOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('MnJk,baCiMn->abCiJk', X.ab.oooo[Oa, ob, Oa, ob], T.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('mNJk,baCimN->abCiJk', X.ab.oooo[oa, Ob, Oa, ob], T.aab.vvVooO, optimize=True)
            - 1.0 * np.einsum('MNJk,baCiMN->abCiJk', X.ab.oooo[Oa, Ob, Oa, ob], T.aab.vvVoOO, optimize=True)
    )
    dR.aab.vvVoOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('mnik,baCmJn->abCiJk', X.ab.oooo[oa, ob, oa, ob], T.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('Mnik,baCJMn->abCiJk', X.ab.oooo[Oa, ob, oa, ob], T.aab.vvVOOo, optimize=True)
            - 1.0 * np.einsum('mNik,baCmJN->abCiJk', X.ab.oooo[oa, Ob, oa, ob], T.aab.vvVoOO, optimize=True)
            + 1.0 * np.einsum('MNik,baCJMN->abCiJk', X.ab.oooo[Oa, Ob, oa, ob], T.aab.vvVOOO, optimize=True)
    )
    dR.aab.vvVoOo += (1.0 / 2.0) * (
            -0.5 * np.einsum('abef,feCiJk->abCiJk', X.aa.vvvv[va, va, va, va], T.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('abEf,EfCiJk->abCiJk', X.aa.vvvv[va, va, Va, va], T.aab.VvVoOo, optimize=True)
            - 0.5 * np.einsum('abEF,FECiJk->abCiJk', X.aa.vvvv[va, va, Va, Va], T.aab.VVVoOo, optimize=True)
    )
    dR.aab.vvVoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('bCeF,eaFiJk->abCiJk', X.ab.vvvv[va, Vb, va, Vb], T.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('bCEf,EafiJk->abCiJk', X.ab.vvvv[va, Vb, Va, vb], T.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('bCEF,EaFiJk->abCiJk', X.ab.vvvv[va, Vb, Va, Vb], T.aab.VvVoOo, optimize=True)
    )
    dR.aab.vvVoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('amie,beCmJk->abCiJk', X.aa.voov[va, oa, oa, va], T.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('aMie,beCMJk->abCiJk', X.aa.voov[va, Oa, oa, va], T.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('amiE,EbCmJk->abCiJk', X.aa.voov[va, oa, oa, Va], T.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('aMiE,EbCMJk->abCiJk', X.aa.voov[va, Oa, oa, Va], T.aab.VvVOOo, optimize=True)
    )
    dR.aab.vvVoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('aMJe,beCiMk->abCiJk', X.aa.voov[va, Oa, Oa, va], T.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('aMJE,EbCiMk->abCiJk', X.aa.voov[va, Oa, Oa, Va], T.aab.VvVoOo, optimize=True)
    )
    dR.aab.vvVoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('amie,bCeJmk->abCiJk', X.ab.voov[va, ob, oa, vb], T.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('aMie,bCeJkM->abCiJk', X.ab.voov[va, Ob, oa, vb], T.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('amiE,bCEJmk->abCiJk', X.ab.voov[va, ob, oa, Vb], T.abb.vVVOoo, optimize=True)
            + 1.0 * np.einsum('aMiE,bCEJkM->abCiJk', X.ab.voov[va, Ob, oa, Vb], T.abb.vVVOoO, optimize=True)
    )
    dR.aab.vvVoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('aMJe,bCeikM->abCiJk', X.ab.voov[va, Ob, Oa, vb], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('aMJE,bCEikM->abCiJk', X.ab.voov[va, Ob, Oa, Vb], T.abb.vVVooO, optimize=True)
    )
    dR.aab.vvVoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('mCEk,EbaimJ->abCiJk', X.ab.ovvo[oa, Vb, Va, ob], T.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('MCEk,EbaiJM->abCiJk', X.ab.ovvo[Oa, Vb, Va, ob], T.aaa.VvvoOO, optimize=True)
    )
    dR.aab.vvVoOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('CmkE,baEiJm->abCiJk', X.bb.voov[Vb, ob, ob, Vb], T.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('CMkE,baEiJM->abCiJk', X.bb.voov[Vb, Ob, ob, Vb], T.aab.vvVoOO, optimize=True)
    )
    dR.aab.vvVoOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('amek,beCiJm->abCiJk', X.ab.vovo[va, ob, va, ob], T.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('aMek,beCiJM->abCiJk', X.ab.vovo[va, Ob, va, ob], T.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('amEk,EbCiJm->abCiJk', X.ab.vovo[va, ob, Va, ob], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('aMEk,EbCiJM->abCiJk', X.ab.vovo[va, Ob, Va, ob], T.aab.VvVoOO, optimize=True)
    )
    dR.aab.vvVoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('mCiE,baEmJk->abCiJk', X.ab.ovov[oa, Vb, oa, Vb], T.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('MCiE,baEMJk->abCiJk', X.ab.ovov[Oa, Vb, oa, Vb], T.aab.vvVOOo, optimize=True)
    )
    dR.aab.vvVoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('MCJE,baEiMk->abCiJk', X.ab.ovov[Oa, Vb, Oa, Vb], T.aab.vvVoOo, optimize=True)
    )
    dR.aab.vvVoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('mi,baCmJk->abCiJk', H.a.oo[oa, oa], R.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('Mi,baCMJk->abCiJk', H.a.oo[Oa, oa], R.aab.vvVOOo, optimize=True)
    )
    dR.aab.vvVoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('MJ,baCiMk->abCiJk', H.a.oo[Oa, Oa], R.aab.vvVoOo, optimize=True)
    )
    dR.aab.vvVoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('mk,baCiJm->abCiJk', H.b.oo[ob, ob], R.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('Mk,baCiJM->abCiJk', H.b.oo[Ob, ob], R.aab.vvVoOO, optimize=True)
    )
    dR.aab.vvVoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('ae,beCiJk->abCiJk', H.a.vv[va, va], R.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('aE,EbCiJk->abCiJk', H.a.vv[va, Va], R.aab.VvVoOo, optimize=True)
    )
    dR.aab.vvVoOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('CE,baEiJk->abCiJk', H.b.vv[Vb, Vb], R.aab.vvVoOo, optimize=True)
    )
    dR.aab.vvVoOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('mNiJ,baCmNk->abCiJk', H.aa.oooo[oa, Oa, oa, Oa], R.aab.vvVoOo, optimize=True)
            - 0.5 * np.einsum('MNiJ,baCMNk->abCiJk', H.aa.oooo[Oa, Oa, oa, Oa], R.aab.vvVOOo, optimize=True)
    )
    dR.aab.vvVoOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('MnJk,baCiMn->abCiJk', H.ab.oooo[Oa, ob, Oa, ob], R.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('mNJk,baCimN->abCiJk', H.ab.oooo[oa, Ob, Oa, ob], R.aab.vvVooO, optimize=True)
            - 1.0 * np.einsum('MNJk,baCiMN->abCiJk', H.ab.oooo[Oa, Ob, Oa, ob], R.aab.vvVoOO, optimize=True)
    )
    dR.aab.vvVoOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('mnik,baCmJn->abCiJk', H.ab.oooo[oa, ob, oa, ob], R.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('Mnik,baCJMn->abCiJk', H.ab.oooo[Oa, ob, oa, ob], R.aab.vvVOOo, optimize=True)
            - 1.0 * np.einsum('mNik,baCmJN->abCiJk', H.ab.oooo[oa, Ob, oa, ob], R.aab.vvVoOO, optimize=True)
            + 1.0 * np.einsum('MNik,baCJMN->abCiJk', H.ab.oooo[Oa, Ob, oa, ob], R.aab.vvVOOO, optimize=True)
    )
    dR.aab.vvVoOo += (1.0 / 2.0) * (
            -0.5 * np.einsum('abef,feCiJk->abCiJk', H.aa.vvvv[va, va, va, va], R.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('abEf,EfCiJk->abCiJk', H.aa.vvvv[va, va, Va, va], R.aab.VvVoOo, optimize=True)
            - 0.5 * np.einsum('abEF,FECiJk->abCiJk', H.aa.vvvv[va, va, Va, Va], R.aab.VVVoOo, optimize=True)
    )
    dR.aab.vvVoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('bCeF,eaFiJk->abCiJk', H.ab.vvvv[va, Vb, va, Vb], R.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('bCEf,EafiJk->abCiJk', H.ab.vvvv[va, Vb, Va, vb], R.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('bCEF,EaFiJk->abCiJk', H.ab.vvvv[va, Vb, Va, Vb], R.aab.VvVoOo, optimize=True)
    )
    dR.aab.vvVoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('amie,beCmJk->abCiJk', H.aa.voov[va, oa, oa, va], R.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('aMie,beCMJk->abCiJk', H.aa.voov[va, Oa, oa, va], R.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('amiE,EbCmJk->abCiJk', H.aa.voov[va, oa, oa, Va], R.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('aMiE,EbCMJk->abCiJk', H.aa.voov[va, Oa, oa, Va], R.aab.VvVOOo, optimize=True)
    )
    dR.aab.vvVoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('aMJe,beCiMk->abCiJk', H.aa.voov[va, Oa, Oa, va], R.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('aMJE,EbCiMk->abCiJk', H.aa.voov[va, Oa, Oa, Va], R.aab.VvVoOo, optimize=True)
    )
    dR.aab.vvVoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('amie,bCeJmk->abCiJk', H.ab.voov[va, ob, oa, vb], R.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('aMie,bCeJkM->abCiJk', H.ab.voov[va, Ob, oa, vb], R.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('amiE,bCEJmk->abCiJk', H.ab.voov[va, ob, oa, Vb], R.abb.vVVOoo, optimize=True)
            + 1.0 * np.einsum('aMiE,bCEJkM->abCiJk', H.ab.voov[va, Ob, oa, Vb], R.abb.vVVOoO, optimize=True)
    )
    dR.aab.vvVoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('aMJe,bCeikM->abCiJk', H.ab.voov[va, Ob, Oa, vb], R.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('aMJE,bCEikM->abCiJk', H.ab.voov[va, Ob, Oa, Vb], R.abb.vVVooO, optimize=True)
    )
    dR.aab.vvVoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('mCEk,EbaimJ->abCiJk', H.ab.ovvo[oa, Vb, Va, ob], R.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('MCEk,EbaiJM->abCiJk', H.ab.ovvo[Oa, Vb, Va, ob], R.aaa.VvvoOO, optimize=True)
    )
    dR.aab.vvVoOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('CmkE,baEiJm->abCiJk', H.bb.voov[Vb, ob, ob, Vb], R.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('CMkE,baEiJM->abCiJk', H.bb.voov[Vb, Ob, ob, Vb], R.aab.vvVoOO, optimize=True)
    )
    dR.aab.vvVoOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('amek,beCiJm->abCiJk', H.ab.vovo[va, ob, va, ob], R.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('aMek,beCiJM->abCiJk', H.ab.vovo[va, Ob, va, ob], R.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('amEk,EbCiJm->abCiJk', H.ab.vovo[va, ob, Va, ob], R.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('aMEk,EbCiJM->abCiJk', H.ab.vovo[va, Ob, Va, ob], R.aab.VvVoOO, optimize=True)
    )
    dR.aab.vvVoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('mCiE,baEmJk->abCiJk', H.ab.ovov[oa, Vb, oa, Vb], R.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('MCiE,baEMJk->abCiJk', H.ab.ovov[Oa, Vb, oa, Vb], R.aab.vvVOOo, optimize=True)
    )
    dR.aab.vvVoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('MCJE,baEiMk->abCiJk', H.ab.ovov[Oa, Vb, Oa, Vb], R.aab.vvVoOo, optimize=True)
    )

    dR.aab.vvVoOo -= np.transpose(dR.aab.vvVoOo, (1, 0, 2, 3, 4, 5))

    return dR

def update(R, omega, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    R.aab.vvVoOo = eomcc_active_loops.update_r3b_001010(
        R.aab.vvVoOo,
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
