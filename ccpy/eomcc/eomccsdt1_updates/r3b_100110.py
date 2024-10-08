import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops

def build(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.aab.VvvOOo = (1.0 / 2.0) * (
            +1.0 * np.einsum('bcek,AeIJ->AbcIJk', X.ab.vvvo[va, vb, :, ob], T.aa[Va, :, Oa, Oa], optimize=True)
    )
    dR.aab.VvvOOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('Acek,beIJ->AbcIJk', X.ab.vvvo[Va, vb, :, ob], T.aa[va, :, Oa, Oa], optimize=True)
    )
    dR.aab.VvvOOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('mcJk,AbIm->AbcIJk', X.ab.ovoo[:, vb, Oa, ob], T.aa[Va, va, Oa, :], optimize=True)
    )
    dR.aab.VvvOOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('AcIe,beJk->AbcIJk', X.ab.vvov[Va, vb, Oa, :], T.ab[va, :, Oa, ob], optimize=True)
    )
    dR.aab.VvvOOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('bcIe,AeJk->AbcIJk', X.ab.vvov[va, vb, Oa, :], T.ab[Va, :, Oa, ob], optimize=True)
    )
    dR.aab.VvvOOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('AmIk,bcJm->AbcIJk', X.ab.vooo[Va, :, Oa, ob], T.ab[va, vb, Oa, :], optimize=True)
    )
    dR.aab.VvvOOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('bmIk,AcJm->AbcIJk', X.ab.vooo[va, :, Oa, ob], T.ab[Va, vb, Oa, :], optimize=True)
    )
    dR.aab.VvvOOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('AbIe,ecJk->AbcIJk', X.aa.vvov[Va, va, Oa, :], T.ab[:, vb, Oa, ob], optimize=True)
    )
    dR.aab.VvvOOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('AmIJ,bcmk->AbcIJk', X.aa.vooo[Va, :, Oa, Oa], T.ab[va, vb, :, ob], optimize=True)
    )
    dR.aab.VvvOOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('bmIJ,Acmk->AbcIJk', X.aa.vooo[va, :, Oa, Oa], T.ab[Va, vb, :, ob], optimize=True)
    )
    dR.aab.VvvOOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('bcek,AeIJ->AbcIJk', H.ab.vvvo[va, vb, :, ob], R.aa[Va, :, Oa, Oa], optimize=True)
    )
    dR.aab.VvvOOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('Acek,beIJ->AbcIJk', H.ab.vvvo[Va, vb, :, ob], R.aa[va, :, Oa, Oa], optimize=True)
    )
    dR.aab.VvvOOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('mcJk,AbIm->AbcIJk', H.ab.ovoo[:, vb, Oa, ob], R.aa[Va, va, Oa, :], optimize=True)
    )
    dR.aab.VvvOOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('AcIe,beJk->AbcIJk', H.ab.vvov[Va, vb, Oa, :], R.ab[va, :, Oa, ob], optimize=True)
    )
    dR.aab.VvvOOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('bcIe,AeJk->AbcIJk', H.ab.vvov[va, vb, Oa, :], R.ab[Va, :, Oa, ob], optimize=True)
    )
    dR.aab.VvvOOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('AmIk,bcJm->AbcIJk', H.ab.vooo[Va, :, Oa, ob], R.ab[va, vb, Oa, :], optimize=True)
    )
    dR.aab.VvvOOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('bmIk,AcJm->AbcIJk', H.ab.vooo[va, :, Oa, ob], R.ab[Va, vb, Oa, :], optimize=True)
    )
    dR.aab.VvvOOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('AbIe,ecJk->AbcIJk', H.aa.vvov[Va, va, Oa, :], R.ab[:, vb, Oa, ob], optimize=True)
    )
    dR.aab.VvvOOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('AmIJ,bcmk->AbcIJk', H.aa.vooo[Va, :, Oa, Oa], R.ab[va, vb, :, ob], optimize=True)
    )
    dR.aab.VvvOOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('bmIJ,Acmk->AbcIJk', H.aa.vooo[va, :, Oa, Oa], R.ab[Va, vb, :, ob], optimize=True)
    )

    dR.aab.VvvOOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('mI,AbcmJk->AbcIJk', X.a.oo[oa, Oa], T.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('MI,AbcMJk->AbcIJk', X.a.oo[Oa, Oa], T.aab.VvvOOo, optimize=True)
    )
    dR.aab.VvvOOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('mk,AbcIJm->AbcIJk', X.b.oo[ob, ob], T.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('Mk,AbcIJM->AbcIJk', X.b.oo[Ob, ob], T.aab.VvvOOO, optimize=True)
    )
    dR.aab.VvvOOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('AE,EbcIJk->AbcIJk', X.a.vv[Va, Va], T.aab.VvvOOo, optimize=True)
    )
    dR.aab.VvvOOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('be,AecIJk->AbcIJk', X.a.vv[va, va], T.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('bE,AEcIJk->AbcIJk', X.a.vv[va, Va], T.aab.VVvOOo, optimize=True)
    )
    dR.aab.VvvOOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('ce,AbeIJk->AbcIJk', X.b.vv[vb, vb], T.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('cE,AbEIJk->AbcIJk', X.b.vv[vb, Vb], T.aab.VvVOOo, optimize=True)
    )
    dR.aab.VvvOOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('MnIJ,AbcnMk->AbcIJk', X.aa.oooo[Oa, oa, Oa, Oa], T.aab.VvvoOo, optimize=True)
            + 0.5 * np.einsum('MNIJ,AbcMNk->AbcIJk', X.aa.oooo[Oa, Oa, Oa, Oa], T.aab.VvvOOo, optimize=True)
    )
    dR.aab.VvvOOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('mnJk,AbcmIn->AbcIJk', X.ab.oooo[oa, ob, Oa, ob], T.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('mNJk,AbcmIN->AbcIJk', X.ab.oooo[oa, Ob, Oa, ob], T.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('MnJk,AbcIMn->AbcIJk', X.ab.oooo[Oa, ob, Oa, ob], T.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('MNJk,AbcIMN->AbcIJk', X.ab.oooo[Oa, Ob, Oa, ob], T.aab.VvvOOO, optimize=True)
    )
    dR.aab.VvvOOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('AbEf,EfcIJk->AbcIJk', X.aa.vvvv[Va, va, Va, va], T.aab.VvvOOo, optimize=True)
            - 0.5 * np.einsum('AbEF,FEcIJk->AbcIJk', X.aa.vvvv[Va, va, Va, Va], T.aab.VVvOOo, optimize=True)
    )
    dR.aab.VvvOOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('bcef,AefIJk->AbcIJk', X.ab.vvvv[va, vb, va, vb], T.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('bceF,AeFIJk->AbcIJk', X.ab.vvvv[va, vb, va, Vb], T.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('bcEf,EAfIJk->AbcIJk', X.ab.vvvv[va, vb, Va, vb], T.aab.VVvOOo, optimize=True)
            - 1.0 * np.einsum('bcEF,EAFIJk->AbcIJk', X.ab.vvvv[va, vb, Va, Vb], T.aab.VVVOOo, optimize=True)
    )
    dR.aab.VvvOOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('AceF,ebFIJk->AbcIJk', X.ab.vvvv[Va, vb, va, Vb], T.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('AcEf,EbfIJk->AbcIJk', X.ab.vvvv[Va, vb, Va, vb], T.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('AcEF,EbFIJk->AbcIJk', X.ab.vvvv[Va, vb, Va, Vb], T.aab.VvVOOo, optimize=True)
    )
    dR.aab.VvvOOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('AmIE,EbcmJk->AbcIJk', X.aa.voov[Va, oa, Oa, Va], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('AMIE,EbcMJk->AbcIJk', X.aa.voov[Va, Oa, Oa, Va], T.aab.VvvOOo, optimize=True)
    )
    dR.aab.VvvOOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('bmIe,AecmJk->AbcIJk', X.aa.voov[va, oa, Oa, va], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('bMIe,AecMJk->AbcIJk', X.aa.voov[va, Oa, Oa, va], T.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('bmIE,AEcmJk->AbcIJk', X.aa.voov[va, oa, Oa, Va], T.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('bMIE,AEcMJk->AbcIJk', X.aa.voov[va, Oa, Oa, Va], T.aab.VVvOOo, optimize=True)
    )
    dR.aab.VvvOOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('AmIE,bEcJmk->AbcIJk', X.ab.voov[Va, ob, Oa, Vb], T.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('AMIE,bEcJkM->AbcIJk', X.ab.voov[Va, Ob, Oa, Vb], T.abb.vVvOoO, optimize=True)
    )
    dR.aab.VvvOOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('bmIe,AceJmk->AbcIJk', X.ab.voov[va, ob, Oa, vb], T.abb.VvvOoo, optimize=True)
            - 1.0 * np.einsum('bMIe,AceJkM->AbcIJk', X.ab.voov[va, Ob, Oa, vb], T.abb.VvvOoO, optimize=True)
            - 1.0 * np.einsum('bmIE,AEcJmk->AbcIJk', X.ab.voov[va, ob, Oa, Vb], T.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('bMIE,AEcJkM->AbcIJk', X.ab.voov[va, Ob, Oa, Vb], T.abb.VVvOoO, optimize=True)
    )
    dR.aab.VvvOOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('mcek,AebmIJ->AbcIJk', X.ab.ovvo[oa, vb, va, ob], T.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('Mcek,AebIJM->AbcIJk', X.ab.ovvo[Oa, vb, va, ob], T.aaa.VvvOOO, optimize=True)
            + 1.0 * np.einsum('mcEk,EAbmIJ->AbcIJk', X.ab.ovvo[oa, vb, Va, ob], T.aaa.VVvoOO, optimize=True)
            + 1.0 * np.einsum('McEk,EAbIJM->AbcIJk', X.ab.ovvo[Oa, vb, Va, ob], T.aaa.VVvOOO, optimize=True)
    )
    dR.aab.VvvOOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('cmke,AbeIJm->AbcIJk', X.bb.voov[vb, ob, ob, vb], T.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('cMke,AbeIJM->AbcIJk', X.bb.voov[vb, Ob, ob, vb], T.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('cmkE,AbEIJm->AbcIJk', X.bb.voov[vb, ob, ob, Vb], T.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('cMkE,AbEIJM->AbcIJk', X.bb.voov[vb, Ob, ob, Vb], T.aab.VvVOOO, optimize=True)
    )
    dR.aab.VvvOOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('AmEk,EbcIJm->AbcIJk', X.ab.vovo[Va, ob, Va, ob], T.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('AMEk,EbcIJM->AbcIJk', X.ab.vovo[Va, Ob, Va, ob], T.aab.VvvOOO, optimize=True)
    )
    dR.aab.VvvOOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('bmek,AecIJm->AbcIJk', X.ab.vovo[va, ob, va, ob], T.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('bMek,AecIJM->AbcIJk', X.ab.vovo[va, Ob, va, ob], T.aab.VvvOOO, optimize=True)
            - 1.0 * np.einsum('bmEk,AEcIJm->AbcIJk', X.ab.vovo[va, ob, Va, ob], T.aab.VVvOOo, optimize=True)
            - 1.0 * np.einsum('bMEk,AEcIJM->AbcIJk', X.ab.vovo[va, Ob, Va, ob], T.aab.VVvOOO, optimize=True)
    )
    dR.aab.VvvOOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('mcIe,AbemJk->AbcIJk', X.ab.ovov[oa, vb, Oa, vb], T.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('McIe,AbeMJk->AbcIJk', X.ab.ovov[Oa, vb, Oa, vb], T.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('mcIE,AbEmJk->AbcIJk', X.ab.ovov[oa, vb, Oa, Vb], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('McIE,AbEMJk->AbcIJk', X.ab.ovov[Oa, vb, Oa, Vb], T.aab.VvVOOo, optimize=True)
    )
    dR.aab.VvvOOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('mI,AbcmJk->AbcIJk', H.a.oo[oa, Oa], R.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('MI,AbcMJk->AbcIJk', H.a.oo[Oa, Oa], R.aab.VvvOOo, optimize=True)
    )
    dR.aab.VvvOOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('mk,AbcIJm->AbcIJk', H.b.oo[ob, ob], R.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('Mk,AbcIJM->AbcIJk', H.b.oo[Ob, ob], R.aab.VvvOOO, optimize=True)
    )
    dR.aab.VvvOOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('AE,EbcIJk->AbcIJk', H.a.vv[Va, Va], R.aab.VvvOOo, optimize=True)
    )
    dR.aab.VvvOOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('be,AecIJk->AbcIJk', H.a.vv[va, va], R.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('bE,AEcIJk->AbcIJk', H.a.vv[va, Va], R.aab.VVvOOo, optimize=True)
    )
    dR.aab.VvvOOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('ce,AbeIJk->AbcIJk', H.b.vv[vb, vb], R.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('cE,AbEIJk->AbcIJk', H.b.vv[vb, Vb], R.aab.VvVOOo, optimize=True)
    )
    dR.aab.VvvOOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('MnIJ,AbcnMk->AbcIJk', H.aa.oooo[Oa, oa, Oa, Oa], R.aab.VvvoOo, optimize=True)
            + 0.5 * np.einsum('MNIJ,AbcMNk->AbcIJk', H.aa.oooo[Oa, Oa, Oa, Oa], R.aab.VvvOOo, optimize=True)
    )
    dR.aab.VvvOOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('mnJk,AbcmIn->AbcIJk', H.ab.oooo[oa, ob, Oa, ob], R.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('mNJk,AbcmIN->AbcIJk', H.ab.oooo[oa, Ob, Oa, ob], R.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('MnJk,AbcIMn->AbcIJk', H.ab.oooo[Oa, ob, Oa, ob], R.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('MNJk,AbcIMN->AbcIJk', H.ab.oooo[Oa, Ob, Oa, ob], R.aab.VvvOOO, optimize=True)
    )
    dR.aab.VvvOOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('AbEf,EfcIJk->AbcIJk', H.aa.vvvv[Va, va, Va, va], R.aab.VvvOOo, optimize=True)
            - 0.5 * np.einsum('AbEF,FEcIJk->AbcIJk', H.aa.vvvv[Va, va, Va, Va], R.aab.VVvOOo, optimize=True)
    )
    dR.aab.VvvOOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('bcef,AefIJk->AbcIJk', H.ab.vvvv[va, vb, va, vb], R.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('bceF,AeFIJk->AbcIJk', H.ab.vvvv[va, vb, va, Vb], R.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('bcEf,EAfIJk->AbcIJk', H.ab.vvvv[va, vb, Va, vb], R.aab.VVvOOo, optimize=True)
            - 1.0 * np.einsum('bcEF,EAFIJk->AbcIJk', H.ab.vvvv[va, vb, Va, Vb], R.aab.VVVOOo, optimize=True)
    )
    dR.aab.VvvOOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('AceF,ebFIJk->AbcIJk', H.ab.vvvv[Va, vb, va, Vb], R.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('AcEf,EbfIJk->AbcIJk', H.ab.vvvv[Va, vb, Va, vb], R.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('AcEF,EbFIJk->AbcIJk', H.ab.vvvv[Va, vb, Va, Vb], R.aab.VvVOOo, optimize=True)
    )
    dR.aab.VvvOOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('AmIE,EbcmJk->AbcIJk', H.aa.voov[Va, oa, Oa, Va], R.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('AMIE,EbcMJk->AbcIJk', H.aa.voov[Va, Oa, Oa, Va], R.aab.VvvOOo, optimize=True)
    )
    dR.aab.VvvOOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('bmIe,AecmJk->AbcIJk', H.aa.voov[va, oa, Oa, va], R.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('bMIe,AecMJk->AbcIJk', H.aa.voov[va, Oa, Oa, va], R.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('bmIE,AEcmJk->AbcIJk', H.aa.voov[va, oa, Oa, Va], R.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('bMIE,AEcMJk->AbcIJk', H.aa.voov[va, Oa, Oa, Va], R.aab.VVvOOo, optimize=True)
    )
    dR.aab.VvvOOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('AmIE,bEcJmk->AbcIJk', H.ab.voov[Va, ob, Oa, Vb], R.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('AMIE,bEcJkM->AbcIJk', H.ab.voov[Va, Ob, Oa, Vb], R.abb.vVvOoO, optimize=True)
    )
    dR.aab.VvvOOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('bmIe,AceJmk->AbcIJk', H.ab.voov[va, ob, Oa, vb], R.abb.VvvOoo, optimize=True)
            - 1.0 * np.einsum('bMIe,AceJkM->AbcIJk', H.ab.voov[va, Ob, Oa, vb], R.abb.VvvOoO, optimize=True)
            - 1.0 * np.einsum('bmIE,AEcJmk->AbcIJk', H.ab.voov[va, ob, Oa, Vb], R.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('bMIE,AEcJkM->AbcIJk', H.ab.voov[va, Ob, Oa, Vb], R.abb.VVvOoO, optimize=True)
    )
    dR.aab.VvvOOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('mcek,AebmIJ->AbcIJk', H.ab.ovvo[oa, vb, va, ob], R.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('Mcek,AebIJM->AbcIJk', H.ab.ovvo[Oa, vb, va, ob], R.aaa.VvvOOO, optimize=True)
            + 1.0 * np.einsum('mcEk,EAbmIJ->AbcIJk', H.ab.ovvo[oa, vb, Va, ob], R.aaa.VVvoOO, optimize=True)
            + 1.0 * np.einsum('McEk,EAbIJM->AbcIJk', H.ab.ovvo[Oa, vb, Va, ob], R.aaa.VVvOOO, optimize=True)
    )
    dR.aab.VvvOOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('cmke,AbeIJm->AbcIJk', H.bb.voov[vb, ob, ob, vb], R.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('cMke,AbeIJM->AbcIJk', H.bb.voov[vb, Ob, ob, vb], R.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('cmkE,AbEIJm->AbcIJk', H.bb.voov[vb, ob, ob, Vb], R.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('cMkE,AbEIJM->AbcIJk', H.bb.voov[vb, Ob, ob, Vb], R.aab.VvVOOO, optimize=True)
    )
    dR.aab.VvvOOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('AmEk,EbcIJm->AbcIJk', H.ab.vovo[Va, ob, Va, ob], R.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('AMEk,EbcIJM->AbcIJk', H.ab.vovo[Va, Ob, Va, ob], R.aab.VvvOOO, optimize=True)
    )
    dR.aab.VvvOOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('bmek,AecIJm->AbcIJk', H.ab.vovo[va, ob, va, ob], R.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('bMek,AecIJM->AbcIJk', H.ab.vovo[va, Ob, va, ob], R.aab.VvvOOO, optimize=True)
            - 1.0 * np.einsum('bmEk,AEcIJm->AbcIJk', H.ab.vovo[va, ob, Va, ob], R.aab.VVvOOo, optimize=True)
            - 1.0 * np.einsum('bMEk,AEcIJM->AbcIJk', H.ab.vovo[va, Ob, Va, ob], R.aab.VVvOOO, optimize=True)
    )
    dR.aab.VvvOOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('mcIe,AbemJk->AbcIJk', H.ab.ovov[oa, vb, Oa, vb], R.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('McIe,AbeMJk->AbcIJk', H.ab.ovov[Oa, vb, Oa, vb], R.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('mcIE,AbEmJk->AbcIJk', H.ab.ovov[oa, vb, Oa, Vb], R.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('McIE,AbEMJk->AbcIJk', H.ab.ovov[Oa, vb, Oa, Vb], R.aab.VvVOOo, optimize=True)
    )

    dR.aab.VvvOOo -= np.transpose(dR.aab.VvvOOo, (0, 1, 2, 4, 3, 5))

    return dR

def update(R, omega, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    R.aab.VvvOOo = eomcc_active_loops.update_r3b_100110(
        R.aab.VvvOOo,
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
