import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops

def build(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.aab.VvVoOo = (1.0 / 1.0) * (
            +1.0 * np.einsum('bCek,AeiJ->AbCiJk', X.ab.vvvo[va, Vb, :, ob], T.aa[Va, :, oa, Oa], optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('ACek,beiJ->AbCiJk', X.ab.vvvo[Va, Vb, :, ob], T.aa[va, :, oa, Oa], optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('mCJk,Abim->AbCiJk', X.ab.ovoo[:, Vb, Oa, ob], T.aa[Va, va, oa, :], optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('mCik,AbJm->AbCiJk', X.ab.ovoo[:, Vb, oa, ob], T.aa[Va, va, Oa, :], optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('ACie,beJk->AbCiJk', X.ab.vvov[Va, Vb, oa, :], T.ab[va, :, Oa, ob], optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('bCie,AeJk->AbCiJk', X.ab.vvov[va, Vb, oa, :], T.ab[Va, :, Oa, ob], optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('ACJe,beik->AbCiJk', X.ab.vvov[Va, Vb, Oa, :], T.ab[va, :, oa, ob], optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('bCJe,Aeik->AbCiJk', X.ab.vvov[va, Vb, Oa, :], T.ab[Va, :, oa, ob], optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('Amik,bCJm->AbCiJk', X.ab.vooo[Va, :, oa, ob], T.ab[va, Vb, Oa, :], optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('bmik,ACJm->AbCiJk', X.ab.vooo[va, :, oa, ob], T.ab[Va, Vb, Oa, :], optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('AmJk,bCim->AbCiJk', X.ab.vooo[Va, :, Oa, ob], T.ab[va, Vb, oa, :], optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('bmJk,ACim->AbCiJk', X.ab.vooo[va, :, Oa, ob], T.ab[Va, Vb, oa, :], optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('Abie,eCJk->AbCiJk', X.aa.vvov[Va, va, oa, :], T.ab[:, Vb, Oa, ob], optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('AbJe,eCik->AbCiJk', X.aa.vvov[Va, va, Oa, :], T.ab[:, Vb, oa, ob], optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('AmiJ,bCmk->AbCiJk', X.aa.vooo[Va, :, oa, Oa], T.ab[va, Vb, :, ob], optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('bmiJ,ACmk->AbCiJk', X.aa.vooo[va, :, oa, Oa], T.ab[Va, Vb, :, ob], optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('bCek,AeiJ->AbCiJk', H.ab.vvvo[va, Vb, :, ob], R.aa[Va, :, oa, Oa], optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('ACek,beiJ->AbCiJk', H.ab.vvvo[Va, Vb, :, ob], R.aa[va, :, oa, Oa], optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('mCJk,Abim->AbCiJk', H.ab.ovoo[:, Vb, Oa, ob], R.aa[Va, va, oa, :], optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('mCik,AbJm->AbCiJk', H.ab.ovoo[:, Vb, oa, ob], R.aa[Va, va, Oa, :], optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('ACie,beJk->AbCiJk', H.ab.vvov[Va, Vb, oa, :], R.ab[va, :, Oa, ob], optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('bCie,AeJk->AbCiJk', H.ab.vvov[va, Vb, oa, :], R.ab[Va, :, Oa, ob], optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('ACJe,beik->AbCiJk', H.ab.vvov[Va, Vb, Oa, :], R.ab[va, :, oa, ob], optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('bCJe,Aeik->AbCiJk', H.ab.vvov[va, Vb, Oa, :], R.ab[Va, :, oa, ob], optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('Amik,bCJm->AbCiJk', H.ab.vooo[Va, :, oa, ob], R.ab[va, Vb, Oa, :], optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('bmik,ACJm->AbCiJk', H.ab.vooo[va, :, oa, ob], R.ab[Va, Vb, Oa, :], optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('AmJk,bCim->AbCiJk', H.ab.vooo[Va, :, Oa, ob], R.ab[va, Vb, oa, :], optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('bmJk,ACim->AbCiJk', H.ab.vooo[va, :, Oa, ob], R.ab[Va, Vb, oa, :], optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('Abie,eCJk->AbCiJk', H.aa.vvov[Va, va, oa, :], R.ab[:, Vb, Oa, ob], optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('AbJe,eCik->AbCiJk', H.aa.vvov[Va, va, Oa, :], R.ab[:, Vb, oa, ob], optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('AmiJ,bCmk->AbCiJk', H.aa.vooo[Va, :, oa, Oa], R.ab[va, Vb, :, ob], optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('bmiJ,ACmk->AbCiJk', H.aa.vooo[va, :, oa, Oa], R.ab[Va, Vb, :, ob], optimize=True)
    )

    dR.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('mi,AbCmJk->AbCiJk', X.a.oo[oa, oa], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('Mi,AbCMJk->AbCiJk', X.a.oo[Oa, oa], T.aab.VvVOOo, optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('MJ,AbCiMk->AbCiJk', X.a.oo[Oa, Oa], T.aab.VvVoOo, optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('mk,AbCiJm->AbCiJk', X.b.oo[ob, ob], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('Mk,AbCiJM->AbCiJk', X.b.oo[Ob, ob], T.aab.VvVoOO, optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('Ae,beCiJk->AbCiJk', X.a.vv[Va, va], T.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('AE,EbCiJk->AbCiJk', X.a.vv[Va, Va], T.aab.VvVoOo, optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('be,AeCiJk->AbCiJk', X.a.vv[va, va], T.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('bE,AECiJk->AbCiJk', X.a.vv[va, Va], T.aab.VVVoOo, optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('Ce,AbeiJk->AbCiJk', X.b.vv[Vb, vb], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('CE,AbEiJk->AbCiJk', X.b.vv[Vb, Vb], T.aab.VvVoOo, optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('mNiJ,AbCmNk->AbCiJk', X.aa.oooo[oa, Oa, oa, Oa], T.aab.VvVoOo, optimize=True)
            + 0.5 * np.einsum('MNiJ,AbCMNk->AbCiJk', X.aa.oooo[Oa, Oa, oa, Oa], T.aab.VvVOOo, optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('MnJk,AbCiMn->AbCiJk', X.ab.oooo[Oa, ob, Oa, ob], T.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('mNJk,AbCimN->AbCiJk', X.ab.oooo[oa, Ob, Oa, ob], T.aab.VvVooO, optimize=True)
            + 1.0 * np.einsum('MNJk,AbCiMN->AbCiJk', X.ab.oooo[Oa, Ob, Oa, ob], T.aab.VvVoOO, optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('mnik,AbCmJn->AbCiJk', X.ab.oooo[oa, ob, oa, ob], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('Mnik,AbCJMn->AbCiJk', X.ab.oooo[Oa, ob, oa, ob], T.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('mNik,AbCmJN->AbCiJk', X.ab.oooo[oa, Ob, oa, ob], T.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('MNik,AbCJMN->AbCiJk', X.ab.oooo[Oa, Ob, oa, ob], T.aab.VvVOOO, optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            -0.5 * np.einsum('Abef,feCiJk->AbCiJk', X.aa.vvvv[Va, va, va, va], T.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('AbEf,EfCiJk->AbCiJk', X.aa.vvvv[Va, va, Va, va], T.aab.VvVoOo, optimize=True)
            - 0.5 * np.einsum('AbEF,FECiJk->AbCiJk', X.aa.vvvv[Va, va, Va, Va], T.aab.VVVoOo, optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('bCef,AefiJk->AbCiJk', X.ab.vvvv[va, Vb, va, vb], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('bCeF,AeFiJk->AbCiJk', X.ab.vvvv[va, Vb, va, Vb], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('bCEf,EAfiJk->AbCiJk', X.ab.vvvv[va, Vb, Va, vb], T.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('bCEF,EAFiJk->AbCiJk', X.ab.vvvv[va, Vb, Va, Vb], T.aab.VVVoOo, optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('ACeF,ebFiJk->AbCiJk', X.ab.vvvv[Va, Vb, va, Vb], T.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('ACEf,EbfiJk->AbCiJk', X.ab.vvvv[Va, Vb, Va, vb], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('ACEF,EbFiJk->AbCiJk', X.ab.vvvv[Va, Vb, Va, Vb], T.aab.VvVoOo, optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('Amie,beCmJk->AbCiJk', X.aa.voov[Va, oa, oa, va], T.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('AMie,beCMJk->AbCiJk', X.aa.voov[Va, Oa, oa, va], T.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('AmiE,EbCmJk->AbCiJk', X.aa.voov[Va, oa, oa, Va], T.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('AMiE,EbCMJk->AbCiJk', X.aa.voov[Va, Oa, oa, Va], T.aab.VvVOOo, optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('bmie,AeCmJk->AbCiJk', X.aa.voov[va, oa, oa, va], T.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('bMie,AeCMJk->AbCiJk', X.aa.voov[va, Oa, oa, va], T.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('bmiE,AECmJk->AbCiJk', X.aa.voov[va, oa, oa, Va], T.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('bMiE,AECMJk->AbCiJk', X.aa.voov[va, Oa, oa, Va], T.aab.VVVOOo, optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('AMJe,beCiMk->AbCiJk', X.aa.voov[Va, Oa, Oa, va], T.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('AMJE,EbCiMk->AbCiJk', X.aa.voov[Va, Oa, Oa, Va], T.aab.VvVoOo, optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('bMJe,AeCiMk->AbCiJk', X.aa.voov[va, Oa, Oa, va], T.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('bMJE,AECiMk->AbCiJk', X.aa.voov[va, Oa, Oa, Va], T.aab.VVVoOo, optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('Amie,bCeJmk->AbCiJk', X.ab.voov[Va, ob, oa, vb], T.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('AMie,bCeJkM->AbCiJk', X.ab.voov[Va, Ob, oa, vb], T.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('AmiE,bCEJmk->AbCiJk', X.ab.voov[Va, ob, oa, Vb], T.abb.vVVOoo, optimize=True)
            + 1.0 * np.einsum('AMiE,bCEJkM->AbCiJk', X.ab.voov[Va, Ob, oa, Vb], T.abb.vVVOoO, optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('bmie,ACeJmk->AbCiJk', X.ab.voov[va, ob, oa, vb], T.abb.VVvOoo, optimize=True)
            - 1.0 * np.einsum('bMie,ACeJkM->AbCiJk', X.ab.voov[va, Ob, oa, vb], T.abb.VVvOoO, optimize=True)
            + 1.0 * np.einsum('bmiE,ACEJmk->AbCiJk', X.ab.voov[va, ob, oa, Vb], T.abb.VVVOoo, optimize=True)
            - 1.0 * np.einsum('bMiE,ACEJkM->AbCiJk', X.ab.voov[va, Ob, oa, Vb], T.abb.VVVOoO, optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('AMJe,bCeikM->AbCiJk', X.ab.voov[Va, Ob, Oa, vb], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('AMJE,bCEikM->AbCiJk', X.ab.voov[Va, Ob, Oa, Vb], T.abb.vVVooO, optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('bMJe,ACeikM->AbCiJk', X.ab.voov[va, Ob, Oa, vb], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('bMJE,ACEikM->AbCiJk', X.ab.voov[va, Ob, Oa, Vb], T.abb.VVVooO, optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('mCek,AebimJ->AbCiJk', X.ab.ovvo[oa, Vb, va, ob], T.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('MCek,AebiJM->AbCiJk', X.ab.ovvo[Oa, Vb, va, ob], T.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('mCEk,EAbimJ->AbCiJk', X.ab.ovvo[oa, Vb, Va, ob], T.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('MCEk,EAbiJM->AbCiJk', X.ab.ovvo[Oa, Vb, Va, ob], T.aaa.VVvoOO, optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('Cmke,AbeiJm->AbCiJk', X.bb.voov[Vb, ob, ob, vb], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('CMke,AbeiJM->AbCiJk', X.bb.voov[Vb, Ob, ob, vb], T.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('CmkE,AbEiJm->AbCiJk', X.bb.voov[Vb, ob, ob, Vb], T.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('CMkE,AbEiJM->AbCiJk', X.bb.voov[Vb, Ob, ob, Vb], T.aab.VvVoOO, optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('Amek,beCiJm->AbCiJk', X.ab.vovo[Va, ob, va, ob], T.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('AMek,beCiJM->AbCiJk', X.ab.vovo[Va, Ob, va, ob], T.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('AmEk,EbCiJm->AbCiJk', X.ab.vovo[Va, ob, Va, ob], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('AMEk,EbCiJM->AbCiJk', X.ab.vovo[Va, Ob, Va, ob], T.aab.VvVoOO, optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('bmek,AeCiJm->AbCiJk', X.ab.vovo[va, ob, va, ob], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('bMek,AeCiJM->AbCiJk', X.ab.vovo[va, Ob, va, ob], T.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('bmEk,AECiJm->AbCiJk', X.ab.vovo[va, ob, Va, ob], T.aab.VVVoOo, optimize=True)
            - 1.0 * np.einsum('bMEk,AECiJM->AbCiJk', X.ab.vovo[va, Ob, Va, ob], T.aab.VVVoOO, optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('mCie,AbemJk->AbCiJk', X.ab.ovov[oa, Vb, oa, vb], T.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('MCie,AbeMJk->AbCiJk', X.ab.ovov[Oa, Vb, oa, vb], T.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('mCiE,AbEmJk->AbCiJk', X.ab.ovov[oa, Vb, oa, Vb], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('MCiE,AbEMJk->AbCiJk', X.ab.ovov[Oa, Vb, oa, Vb], T.aab.VvVOOo, optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('MCJe,AbeiMk->AbCiJk', X.ab.ovov[Oa, Vb, Oa, vb], T.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('MCJE,AbEiMk->AbCiJk', X.ab.ovov[Oa, Vb, Oa, Vb], T.aab.VvVoOo, optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('mi,AbCmJk->AbCiJk', H.a.oo[oa, oa], R.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('Mi,AbCMJk->AbCiJk', H.a.oo[Oa, oa], R.aab.VvVOOo, optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('MJ,AbCiMk->AbCiJk', H.a.oo[Oa, Oa], R.aab.VvVoOo, optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('mk,AbCiJm->AbCiJk', H.b.oo[ob, ob], R.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('Mk,AbCiJM->AbCiJk', H.b.oo[Ob, ob], R.aab.VvVoOO, optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('Ae,beCiJk->AbCiJk', H.a.vv[Va, va], R.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('AE,EbCiJk->AbCiJk', H.a.vv[Va, Va], R.aab.VvVoOo, optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('be,AeCiJk->AbCiJk', H.a.vv[va, va], R.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('bE,AECiJk->AbCiJk', H.a.vv[va, Va], R.aab.VVVoOo, optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('Ce,AbeiJk->AbCiJk', H.b.vv[Vb, vb], R.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('CE,AbEiJk->AbCiJk', H.b.vv[Vb, Vb], R.aab.VvVoOo, optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('mNiJ,AbCmNk->AbCiJk', H.aa.oooo[oa, Oa, oa, Oa], R.aab.VvVoOo, optimize=True)
            + 0.5 * np.einsum('MNiJ,AbCMNk->AbCiJk', H.aa.oooo[Oa, Oa, oa, Oa], R.aab.VvVOOo, optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('MnJk,AbCiMn->AbCiJk', H.ab.oooo[Oa, ob, Oa, ob], R.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('mNJk,AbCimN->AbCiJk', H.ab.oooo[oa, Ob, Oa, ob], R.aab.VvVooO, optimize=True)
            + 1.0 * np.einsum('MNJk,AbCiMN->AbCiJk', H.ab.oooo[Oa, Ob, Oa, ob], R.aab.VvVoOO, optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('mnik,AbCmJn->AbCiJk', H.ab.oooo[oa, ob, oa, ob], R.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('Mnik,AbCJMn->AbCiJk', H.ab.oooo[Oa, ob, oa, ob], R.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('mNik,AbCmJN->AbCiJk', H.ab.oooo[oa, Ob, oa, ob], R.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('MNik,AbCJMN->AbCiJk', H.ab.oooo[Oa, Ob, oa, ob], R.aab.VvVOOO, optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            -0.5 * np.einsum('Abef,feCiJk->AbCiJk', H.aa.vvvv[Va, va, va, va], R.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('AbEf,EfCiJk->AbCiJk', H.aa.vvvv[Va, va, Va, va], R.aab.VvVoOo, optimize=True)
            - 0.5 * np.einsum('AbEF,FECiJk->AbCiJk', H.aa.vvvv[Va, va, Va, Va], R.aab.VVVoOo, optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('bCef,AefiJk->AbCiJk', H.ab.vvvv[va, Vb, va, vb], R.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('bCeF,AeFiJk->AbCiJk', H.ab.vvvv[va, Vb, va, Vb], R.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('bCEf,EAfiJk->AbCiJk', H.ab.vvvv[va, Vb, Va, vb], R.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('bCEF,EAFiJk->AbCiJk', H.ab.vvvv[va, Vb, Va, Vb], R.aab.VVVoOo, optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('ACeF,ebFiJk->AbCiJk', H.ab.vvvv[Va, Vb, va, Vb], R.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('ACEf,EbfiJk->AbCiJk', H.ab.vvvv[Va, Vb, Va, vb], R.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('ACEF,EbFiJk->AbCiJk', H.ab.vvvv[Va, Vb, Va, Vb], R.aab.VvVoOo, optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('Amie,beCmJk->AbCiJk', H.aa.voov[Va, oa, oa, va], R.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('AMie,beCMJk->AbCiJk', H.aa.voov[Va, Oa, oa, va], R.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('AmiE,EbCmJk->AbCiJk', H.aa.voov[Va, oa, oa, Va], R.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('AMiE,EbCMJk->AbCiJk', H.aa.voov[Va, Oa, oa, Va], R.aab.VvVOOo, optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('bmie,AeCmJk->AbCiJk', H.aa.voov[va, oa, oa, va], R.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('bMie,AeCMJk->AbCiJk', H.aa.voov[va, Oa, oa, va], R.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('bmiE,AECmJk->AbCiJk', H.aa.voov[va, oa, oa, Va], R.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('bMiE,AECMJk->AbCiJk', H.aa.voov[va, Oa, oa, Va], R.aab.VVVOOo, optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('AMJe,beCiMk->AbCiJk', H.aa.voov[Va, Oa, Oa, va], R.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('AMJE,EbCiMk->AbCiJk', H.aa.voov[Va, Oa, Oa, Va], R.aab.VvVoOo, optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('bMJe,AeCiMk->AbCiJk', H.aa.voov[va, Oa, Oa, va], R.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('bMJE,AECiMk->AbCiJk', H.aa.voov[va, Oa, Oa, Va], R.aab.VVVoOo, optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('Amie,bCeJmk->AbCiJk', H.ab.voov[Va, ob, oa, vb], R.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('AMie,bCeJkM->AbCiJk', H.ab.voov[Va, Ob, oa, vb], R.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('AmiE,bCEJmk->AbCiJk', H.ab.voov[Va, ob, oa, Vb], R.abb.vVVOoo, optimize=True)
            + 1.0 * np.einsum('AMiE,bCEJkM->AbCiJk', H.ab.voov[Va, Ob, oa, Vb], R.abb.vVVOoO, optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('bmie,ACeJmk->AbCiJk', H.ab.voov[va, ob, oa, vb], R.abb.VVvOoo, optimize=True)
            - 1.0 * np.einsum('bMie,ACeJkM->AbCiJk', H.ab.voov[va, Ob, oa, vb], R.abb.VVvOoO, optimize=True)
            + 1.0 * np.einsum('bmiE,ACEJmk->AbCiJk', H.ab.voov[va, ob, oa, Vb], R.abb.VVVOoo, optimize=True)
            - 1.0 * np.einsum('bMiE,ACEJkM->AbCiJk', H.ab.voov[va, Ob, oa, Vb], R.abb.VVVOoO, optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('AMJe,bCeikM->AbCiJk', H.ab.voov[Va, Ob, Oa, vb], R.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('AMJE,bCEikM->AbCiJk', H.ab.voov[Va, Ob, Oa, Vb], R.abb.vVVooO, optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('bMJe,ACeikM->AbCiJk', H.ab.voov[va, Ob, Oa, vb], R.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('bMJE,ACEikM->AbCiJk', H.ab.voov[va, Ob, Oa, Vb], R.abb.VVVooO, optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('mCek,AebimJ->AbCiJk', H.ab.ovvo[oa, Vb, va, ob], R.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('MCek,AebiJM->AbCiJk', H.ab.ovvo[Oa, Vb, va, ob], R.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('mCEk,EAbimJ->AbCiJk', H.ab.ovvo[oa, Vb, Va, ob], R.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('MCEk,EAbiJM->AbCiJk', H.ab.ovvo[Oa, Vb, Va, ob], R.aaa.VVvoOO, optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('Cmke,AbeiJm->AbCiJk', H.bb.voov[Vb, ob, ob, vb], R.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('CMke,AbeiJM->AbCiJk', H.bb.voov[Vb, Ob, ob, vb], R.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('CmkE,AbEiJm->AbCiJk', H.bb.voov[Vb, ob, ob, Vb], R.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('CMkE,AbEiJM->AbCiJk', H.bb.voov[Vb, Ob, ob, Vb], R.aab.VvVoOO, optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('Amek,beCiJm->AbCiJk', H.ab.vovo[Va, ob, va, ob], R.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('AMek,beCiJM->AbCiJk', H.ab.vovo[Va, Ob, va, ob], R.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('AmEk,EbCiJm->AbCiJk', H.ab.vovo[Va, ob, Va, ob], R.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('AMEk,EbCiJM->AbCiJk', H.ab.vovo[Va, Ob, Va, ob], R.aab.VvVoOO, optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('bmek,AeCiJm->AbCiJk', H.ab.vovo[va, ob, va, ob], R.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('bMek,AeCiJM->AbCiJk', H.ab.vovo[va, Ob, va, ob], R.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('bmEk,AECiJm->AbCiJk', H.ab.vovo[va, ob, Va, ob], R.aab.VVVoOo, optimize=True)
            - 1.0 * np.einsum('bMEk,AECiJM->AbCiJk', H.ab.vovo[va, Ob, Va, ob], R.aab.VVVoOO, optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('mCie,AbemJk->AbCiJk', H.ab.ovov[oa, Vb, oa, vb], R.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('MCie,AbeMJk->AbCiJk', H.ab.ovov[Oa, Vb, oa, vb], R.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('mCiE,AbEmJk->AbCiJk', H.ab.ovov[oa, Vb, oa, Vb], R.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('MCiE,AbEMJk->AbCiJk', H.ab.ovov[Oa, Vb, oa, Vb], R.aab.VvVOOo, optimize=True)
    )
    dR.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('MCJe,AbeiMk->AbCiJk', H.ab.ovov[Oa, Vb, Oa, vb], R.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('MCJE,AbEiMk->AbCiJk', H.ab.ovov[Oa, Vb, Oa, Vb], R.aab.VvVoOo, optimize=True)
    )

    return dR

def update(R, omega, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    R.aab.VvVoOo = eomcc_active_loops.update_r3b_101010(
        R.aab.VvVoOo,
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
