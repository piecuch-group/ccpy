import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops

def build(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.aab.VvvoOo = (1.0 / 1.0) * (
            +1.0 * np.einsum('bcek,AeiJ->AbciJk', X.ab.vvvo[va, vb, :, ob], T.aa[Va, :, oa, Oa], optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('Acek,beiJ->AbciJk', X.ab.vvvo[Va, vb, :, ob], T.aa[va, :, oa, Oa], optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('mcJk,Abim->AbciJk', X.ab.ovoo[:, vb, Oa, ob], T.aa[Va, va, oa, :], optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('mcik,AbJm->AbciJk', X.ab.ovoo[:, vb, oa, ob], T.aa[Va, va, Oa, :], optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('Acie,beJk->AbciJk', X.ab.vvov[Va, vb, oa, :], T.ab[va, :, Oa, ob], optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('bcie,AeJk->AbciJk', X.ab.vvov[va, vb, oa, :], T.ab[Va, :, Oa, ob], optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('AcJe,beik->AbciJk', X.ab.vvov[Va, vb, Oa, :], T.ab[va, :, oa, ob], optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('bcJe,Aeik->AbciJk', X.ab.vvov[va, vb, Oa, :], T.ab[Va, :, oa, ob], optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('Amik,bcJm->AbciJk', X.ab.vooo[Va, :, oa, ob], T.ab[va, vb, Oa, :], optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('bmik,AcJm->AbciJk', X.ab.vooo[va, :, oa, ob], T.ab[Va, vb, Oa, :], optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('AmJk,bcim->AbciJk', X.ab.vooo[Va, :, Oa, ob], T.ab[va, vb, oa, :], optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('bmJk,Acim->AbciJk', X.ab.vooo[va, :, Oa, ob], T.ab[Va, vb, oa, :], optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('Abie,ecJk->AbciJk', X.aa.vvov[Va, va, oa, :], T.ab[:, vb, Oa, ob], optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('AbJe,ecik->AbciJk', X.aa.vvov[Va, va, Oa, :], T.ab[:, vb, oa, ob], optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('AmiJ,bcmk->AbciJk', X.aa.vooo[Va, :, oa, Oa], T.ab[va, vb, :, ob], optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('bmiJ,Acmk->AbciJk', X.aa.vooo[va, :, oa, Oa], T.ab[Va, vb, :, ob], optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('bcek,AeiJ->AbciJk', H.ab.vvvo[va, vb, :, ob], R.aa[Va, :, oa, Oa], optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('Acek,beiJ->AbciJk', H.ab.vvvo[Va, vb, :, ob], R.aa[va, :, oa, Oa], optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('mcJk,Abim->AbciJk', H.ab.ovoo[:, vb, Oa, ob], R.aa[Va, va, oa, :], optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('mcik,AbJm->AbciJk', H.ab.ovoo[:, vb, oa, ob], R.aa[Va, va, Oa, :], optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('Acie,beJk->AbciJk', H.ab.vvov[Va, vb, oa, :], R.ab[va, :, Oa, ob], optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('bcie,AeJk->AbciJk', H.ab.vvov[va, vb, oa, :], R.ab[Va, :, Oa, ob], optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('AcJe,beik->AbciJk', H.ab.vvov[Va, vb, Oa, :], R.ab[va, :, oa, ob], optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('bcJe,Aeik->AbciJk', H.ab.vvov[va, vb, Oa, :], R.ab[Va, :, oa, ob], optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('Amik,bcJm->AbciJk', H.ab.vooo[Va, :, oa, ob], R.ab[va, vb, Oa, :], optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('bmik,AcJm->AbciJk', H.ab.vooo[va, :, oa, ob], R.ab[Va, vb, Oa, :], optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('AmJk,bcim->AbciJk', H.ab.vooo[Va, :, Oa, ob], R.ab[va, vb, oa, :], optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('bmJk,Acim->AbciJk', H.ab.vooo[va, :, Oa, ob], R.ab[Va, vb, oa, :], optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('Abie,ecJk->AbciJk', H.aa.vvov[Va, va, oa, :], R.ab[:, vb, Oa, ob], optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('AbJe,ecik->AbciJk', H.aa.vvov[Va, va, Oa, :], R.ab[:, vb, oa, ob], optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('AmiJ,bcmk->AbciJk', H.aa.vooo[Va, :, oa, Oa], R.ab[va, vb, :, ob], optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('bmiJ,Acmk->AbciJk', H.aa.vooo[va, :, oa, Oa], R.ab[Va, vb, :, ob], optimize=True)
    )

    dR.aab.VvvoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('mi,AbcmJk->AbciJk', X.a.oo[oa, oa], T.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('Mi,AbcMJk->AbciJk', X.a.oo[Oa, oa], T.aab.VvvOOo, optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('MJ,AbciMk->AbciJk', X.a.oo[Oa, Oa], T.aab.VvvoOo, optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('mk,AbciJm->AbciJk', X.b.oo[ob, ob], T.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('Mk,AbciJM->AbciJk', X.b.oo[Ob, ob], T.aab.VvvoOO, optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('AE,EbciJk->AbciJk', X.a.vv[Va, Va], T.aab.VvvoOo, optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('be,AeciJk->AbciJk', X.a.vv[va, va], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('bE,AEciJk->AbciJk', X.a.vv[va, Va], T.aab.VVvoOo, optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('ce,AbeiJk->AbciJk', X.b.vv[vb, vb], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('cE,AbEiJk->AbciJk', X.b.vv[vb, Vb], T.aab.VvVoOo, optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('mNiJ,AbcmNk->AbciJk', X.aa.oooo[oa, Oa, oa, Oa], T.aab.VvvoOo, optimize=True)
            + 0.5 * np.einsum('MNiJ,AbcMNk->AbciJk', X.aa.oooo[Oa, Oa, oa, Oa], T.aab.VvvOOo, optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('MnJk,AbciMn->AbciJk', X.ab.oooo[Oa, ob, Oa, ob], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('mNJk,AbcimN->AbciJk', X.ab.oooo[oa, Ob, Oa, ob], T.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('MNJk,AbciMN->AbciJk', X.ab.oooo[Oa, Ob, Oa, ob], T.aab.VvvoOO, optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('mnik,AbcmJn->AbciJk', X.ab.oooo[oa, ob, oa, ob], T.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('Mnik,AbcJMn->AbciJk', X.ab.oooo[Oa, ob, oa, ob], T.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('mNik,AbcmJN->AbciJk', X.ab.oooo[oa, Ob, oa, ob], T.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('MNik,AbcJMN->AbciJk', X.ab.oooo[Oa, Ob, oa, ob], T.aab.VvvOOO, optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('AbeF,FeciJk->AbciJk', X.aa.vvvv[Va, va, va, Va], T.aab.VvvoOo, optimize=True)
            - 0.5 * np.einsum('AbEF,FEciJk->AbciJk', X.aa.vvvv[Va, va, Va, Va], T.aab.VVvoOo, optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('bcef,AefiJk->AbciJk', X.ab.vvvv[va, vb, va, vb], T.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('bcEf,EAfiJk->AbciJk', X.ab.vvvv[va, vb, Va, vb], T.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('bceF,AeFiJk->AbciJk', X.ab.vvvv[va, vb, va, Vb], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('bcEF,EAFiJk->AbciJk', X.ab.vvvv[va, vb, Va, Vb], T.aab.VVVoOo, optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('AcEf,EbfiJk->AbciJk', X.ab.vvvv[Va, vb, Va, vb], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('AceF,ebFiJk->AbciJk', X.ab.vvvv[Va, vb, va, Vb], T.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('AcEF,EbFiJk->AbciJk', X.ab.vvvv[Va, vb, Va, Vb], T.aab.VvVoOo, optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('AmiE,EbcmJk->AbciJk', X.aa.voov[Va, oa, oa, Va], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('AMiE,EbcMJk->AbciJk', X.aa.voov[Va, Oa, oa, Va], T.aab.VvvOOo, optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('bmie,AecmJk->AbciJk', X.aa.voov[va, oa, oa, va], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('bMie,AecMJk->AbciJk', X.aa.voov[va, Oa, oa, va], T.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('bmiE,AEcmJk->AbciJk', X.aa.voov[va, oa, oa, Va], T.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('bMiE,AEcMJk->AbciJk', X.aa.voov[va, Oa, oa, Va], T.aab.VVvOOo, optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('AMJE,EbciMk->AbciJk', X.aa.voov[Va, Oa, Oa, Va], T.aab.VvvoOo, optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('bMJe,AeciMk->AbciJk', X.aa.voov[va, Oa, Oa, va], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('bMJE,AEciMk->AbciJk', X.aa.voov[va, Oa, Oa, Va], T.aab.VVvoOo, optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('AmiE,bEcJmk->AbciJk', X.ab.voov[Va, ob, oa, Vb], T.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('AMiE,bEcJkM->AbciJk', X.ab.voov[Va, Ob, oa, Vb], T.abb.vVvOoO, optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('bmie,AceJmk->AbciJk', X.ab.voov[va, ob, oa, vb], T.abb.VvvOoo, optimize=True)
            - 1.0 * np.einsum('bMie,AceJkM->AbciJk', X.ab.voov[va, Ob, oa, vb], T.abb.VvvOoO, optimize=True)
            - 1.0 * np.einsum('bmiE,AEcJmk->AbciJk', X.ab.voov[va, ob, oa, Vb], T.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('bMiE,AEcJkM->AbciJk', X.ab.voov[va, Ob, oa, Vb], T.abb.VVvOoO, optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('AMJE,bEcikM->AbciJk', X.ab.voov[Va, Ob, Oa, Vb], T.abb.vVvooO, optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('bMJe,AceikM->AbciJk', X.ab.voov[va, Ob, Oa, vb], T.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('bMJE,AEcikM->AbciJk', X.ab.voov[va, Ob, Oa, Vb], T.abb.VVvooO, optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('mcek,AebimJ->AbciJk', X.ab.ovvo[oa, vb, va, ob], T.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('Mcek,AebiJM->AbciJk', X.ab.ovvo[Oa, vb, va, ob], T.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('mcEk,EAbimJ->AbciJk', X.ab.ovvo[oa, vb, Va, ob], T.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('McEk,EAbiJM->AbciJk', X.ab.ovvo[Oa, vb, Va, ob], T.aaa.VVvoOO, optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('cmke,AbeiJm->AbciJk', X.bb.voov[vb, ob, ob, vb], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('cMke,AbeiJM->AbciJk', X.bb.voov[vb, Ob, ob, vb], T.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('cmkE,AbEiJm->AbciJk', X.bb.voov[vb, ob, ob, Vb], T.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('cMkE,AbEiJM->AbciJk', X.bb.voov[vb, Ob, ob, Vb], T.aab.VvVoOO, optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('AmEk,EbciJm->AbciJk', X.ab.vovo[Va, ob, Va, ob], T.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('AMEk,EbciJM->AbciJk', X.ab.vovo[Va, Ob, Va, ob], T.aab.VvvoOO, optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('bmek,AeciJm->AbciJk', X.ab.vovo[va, ob, va, ob], T.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('bMek,AeciJM->AbciJk', X.ab.vovo[va, Ob, va, ob], T.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('bmEk,AEciJm->AbciJk', X.ab.vovo[va, ob, Va, ob], T.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('bMEk,AEciJM->AbciJk', X.ab.vovo[va, Ob, Va, ob], T.aab.VVvoOO, optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('mcie,AbemJk->AbciJk', X.ab.ovov[oa, vb, oa, vb], T.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('Mcie,AbeMJk->AbciJk', X.ab.ovov[Oa, vb, oa, vb], T.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('mciE,AbEmJk->AbciJk', X.ab.ovov[oa, vb, oa, Vb], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('MciE,AbEMJk->AbciJk', X.ab.ovov[Oa, vb, oa, Vb], T.aab.VvVOOo, optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('McJe,AbeiMk->AbciJk', X.ab.ovov[Oa, vb, Oa, vb], T.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('McJE,AbEiMk->AbciJk', X.ab.ovov[Oa, vb, Oa, Vb], T.aab.VvVoOo, optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('mi,AbcmJk->AbciJk', H.a.oo[oa, oa], R.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('Mi,AbcMJk->AbciJk', H.a.oo[Oa, oa], R.aab.VvvOOo, optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('MJ,AbciMk->AbciJk', H.a.oo[Oa, Oa], R.aab.VvvoOo, optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('mk,AbciJm->AbciJk', H.b.oo[ob, ob], R.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('Mk,AbciJM->AbciJk', H.b.oo[Ob, ob], R.aab.VvvoOO, optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('AE,EbciJk->AbciJk', H.a.vv[Va, Va], R.aab.VvvoOo, optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('be,AeciJk->AbciJk', H.a.vv[va, va], R.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('bE,AEciJk->AbciJk', H.a.vv[va, Va], R.aab.VVvoOo, optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('ce,AbeiJk->AbciJk', H.b.vv[vb, vb], R.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('cE,AbEiJk->AbciJk', H.b.vv[vb, Vb], R.aab.VvVoOo, optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('mNiJ,AbcmNk->AbciJk', H.aa.oooo[oa, Oa, oa, Oa], R.aab.VvvoOo, optimize=True)
            + 0.5 * np.einsum('MNiJ,AbcMNk->AbciJk', H.aa.oooo[Oa, Oa, oa, Oa], R.aab.VvvOOo, optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('MnJk,AbciMn->AbciJk', H.ab.oooo[Oa, ob, Oa, ob], R.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('mNJk,AbcimN->AbciJk', H.ab.oooo[oa, Ob, Oa, ob], R.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('MNJk,AbciMN->AbciJk', H.ab.oooo[Oa, Ob, Oa, ob], R.aab.VvvoOO, optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('mnik,AbcmJn->AbciJk', H.ab.oooo[oa, ob, oa, ob], R.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('Mnik,AbcJMn->AbciJk', H.ab.oooo[Oa, ob, oa, ob], R.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('mNik,AbcmJN->AbciJk', H.ab.oooo[oa, Ob, oa, ob], R.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('MNik,AbcJMN->AbciJk', H.ab.oooo[Oa, Ob, oa, ob], R.aab.VvvOOO, optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('AbeF,FeciJk->AbciJk', H.aa.vvvv[Va, va, va, Va], R.aab.VvvoOo, optimize=True)
            - 0.5 * np.einsum('AbEF,FEciJk->AbciJk', H.aa.vvvv[Va, va, Va, Va], R.aab.VVvoOo, optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('bcef,AefiJk->AbciJk', H.ab.vvvv[va, vb, va, vb], R.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('bcEf,EAfiJk->AbciJk', H.ab.vvvv[va, vb, Va, vb], R.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('bceF,AeFiJk->AbciJk', H.ab.vvvv[va, vb, va, Vb], R.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('bcEF,EAFiJk->AbciJk', H.ab.vvvv[va, vb, Va, Vb], R.aab.VVVoOo, optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('AcEf,EbfiJk->AbciJk', H.ab.vvvv[Va, vb, Va, vb], R.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('AceF,ebFiJk->AbciJk', H.ab.vvvv[Va, vb, va, Vb], R.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('AcEF,EbFiJk->AbciJk', H.ab.vvvv[Va, vb, Va, Vb], R.aab.VvVoOo, optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('AmiE,EbcmJk->AbciJk', H.aa.voov[Va, oa, oa, Va], R.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('AMiE,EbcMJk->AbciJk', H.aa.voov[Va, Oa, oa, Va], R.aab.VvvOOo, optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('bmie,AecmJk->AbciJk', H.aa.voov[va, oa, oa, va], R.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('bMie,AecMJk->AbciJk', H.aa.voov[va, Oa, oa, va], R.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('bmiE,AEcmJk->AbciJk', H.aa.voov[va, oa, oa, Va], R.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('bMiE,AEcMJk->AbciJk', H.aa.voov[va, Oa, oa, Va], R.aab.VVvOOo, optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('AMJE,EbciMk->AbciJk', H.aa.voov[Va, Oa, Oa, Va], R.aab.VvvoOo, optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('bMJe,AeciMk->AbciJk', H.aa.voov[va, Oa, Oa, va], R.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('bMJE,AEciMk->AbciJk', H.aa.voov[va, Oa, Oa, Va], R.aab.VVvoOo, optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('AmiE,bEcJmk->AbciJk', H.ab.voov[Va, ob, oa, Vb], R.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('AMiE,bEcJkM->AbciJk', H.ab.voov[Va, Ob, oa, Vb], R.abb.vVvOoO, optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('bmie,AceJmk->AbciJk', H.ab.voov[va, ob, oa, vb], R.abb.VvvOoo, optimize=True)
            - 1.0 * np.einsum('bMie,AceJkM->AbciJk', H.ab.voov[va, Ob, oa, vb], R.abb.VvvOoO, optimize=True)
            - 1.0 * np.einsum('bmiE,AEcJmk->AbciJk', H.ab.voov[va, ob, oa, Vb], R.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('bMiE,AEcJkM->AbciJk', H.ab.voov[va, Ob, oa, Vb], R.abb.VVvOoO, optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('AMJE,bEcikM->AbciJk', H.ab.voov[Va, Ob, Oa, Vb], R.abb.vVvooO, optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('bMJe,AceikM->AbciJk', H.ab.voov[va, Ob, Oa, vb], R.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('bMJE,AEcikM->AbciJk', H.ab.voov[va, Ob, Oa, Vb], R.abb.VVvooO, optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('mcek,AebimJ->AbciJk', H.ab.ovvo[oa, vb, va, ob], R.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('Mcek,AebiJM->AbciJk', H.ab.ovvo[Oa, vb, va, ob], R.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('mcEk,EAbimJ->AbciJk', H.ab.ovvo[oa, vb, Va, ob], R.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('McEk,EAbiJM->AbciJk', H.ab.ovvo[Oa, vb, Va, ob], R.aaa.VVvoOO, optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('cmke,AbeiJm->AbciJk', H.bb.voov[vb, ob, ob, vb], R.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('cMke,AbeiJM->AbciJk', H.bb.voov[vb, Ob, ob, vb], R.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('cmkE,AbEiJm->AbciJk', H.bb.voov[vb, ob, ob, Vb], R.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('cMkE,AbEiJM->AbciJk', H.bb.voov[vb, Ob, ob, Vb], R.aab.VvVoOO, optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('AmEk,EbciJm->AbciJk', H.ab.vovo[Va, ob, Va, ob], R.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('AMEk,EbciJM->AbciJk', H.ab.vovo[Va, Ob, Va, ob], R.aab.VvvoOO, optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('bmek,AeciJm->AbciJk', H.ab.vovo[va, ob, va, ob], R.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('bMek,AeciJM->AbciJk', H.ab.vovo[va, Ob, va, ob], R.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('bmEk,AEciJm->AbciJk', H.ab.vovo[va, ob, Va, ob], R.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('bMEk,AEciJM->AbciJk', H.ab.vovo[va, Ob, Va, ob], R.aab.VVvoOO, optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('mcie,AbemJk->AbciJk', H.ab.ovov[oa, vb, oa, vb], R.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('Mcie,AbeMJk->AbciJk', H.ab.ovov[Oa, vb, oa, vb], R.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('mciE,AbEmJk->AbciJk', H.ab.ovov[oa, vb, oa, Vb], R.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('MciE,AbEMJk->AbciJk', H.ab.ovov[Oa, vb, oa, Vb], R.aab.VvVOOo, optimize=True)
    )
    dR.aab.VvvoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('McJe,AbeiMk->AbciJk', H.ab.ovov[Oa, vb, Oa, vb], R.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('McJE,AbEiMk->AbciJk', H.ab.ovov[Oa, vb, Oa, Vb], R.aab.VvVoOo, optimize=True)
    )

    return dR

def update(R, omega, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    R.aab.VvvoOo = eomcc_active_loops.update_r3b_100010(
        R.aab.VvvoOo,
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
