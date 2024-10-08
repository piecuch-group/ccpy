import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops

def build(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.aab.VvvOOO = (1.0 / 2.0) * (
            +1.0 * np.einsum('bceK,AeIJ->AbcIJK', X.ab.vvvo[va, vb, :, Ob], T.aa[Va, :, Oa, Oa], optimize=True)
    )
    dR.aab.VvvOOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('AceK,beIJ->AbcIJK', X.ab.vvvo[Va, vb, :, Ob], T.aa[va, :, Oa, Oa], optimize=True)
    )
    dR.aab.VvvOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mcJK,AbIm->AbcIJK', X.ab.ovoo[:, vb, Oa, Ob], T.aa[Va, va, Oa, :], optimize=True)
    )
    dR.aab.VvvOOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('AcIe,beJK->AbcIJK', X.ab.vvov[Va, vb, Oa, :], T.ab[va, :, Oa, Ob], optimize=True)
    )
    dR.aab.VvvOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('bcIe,AeJK->AbcIJK', X.ab.vvov[va, vb, Oa, :], T.ab[Va, :, Oa, Ob], optimize=True)
    )
    dR.aab.VvvOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('AmIK,bcJm->AbcIJK', X.ab.vooo[Va, :, Oa, Ob], T.ab[va, vb, Oa, :], optimize=True)
    )
    dR.aab.VvvOOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('bmIK,AcJm->AbcIJK', X.ab.vooo[va, :, Oa, Ob], T.ab[Va, vb, Oa, :], optimize=True)
    )
    dR.aab.VvvOOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('AbIe,ecJK->AbcIJK', X.aa.vvov[Va, va, Oa, :], T.ab[:, vb, Oa, Ob], optimize=True)
    )
    dR.aab.VvvOOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('AmIJ,bcmK->AbcIJK', X.aa.vooo[Va, :, Oa, Oa], T.ab[va, vb, :, Ob], optimize=True)
    )
    dR.aab.VvvOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('bmIJ,AcmK->AbcIJK', X.aa.vooo[va, :, Oa, Oa], T.ab[Va, vb, :, Ob], optimize=True)
    )
    dR.aab.VvvOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('bceK,AeIJ->AbcIJK', H.ab.vvvo[va, vb, :, Ob], R.aa[Va, :, Oa, Oa], optimize=True)
    )
    dR.aab.VvvOOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('AceK,beIJ->AbcIJK', H.ab.vvvo[Va, vb, :, Ob], R.aa[va, :, Oa, Oa], optimize=True)
    )
    dR.aab.VvvOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mcJK,AbIm->AbcIJK', H.ab.ovoo[:, vb, Oa, Ob], R.aa[Va, va, Oa, :], optimize=True)
    )
    dR.aab.VvvOOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('AcIe,beJK->AbcIJK', H.ab.vvov[Va, vb, Oa, :], R.ab[va, :, Oa, Ob], optimize=True)
    )
    dR.aab.VvvOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('bcIe,AeJK->AbcIJK', H.ab.vvov[va, vb, Oa, :], R.ab[Va, :, Oa, Ob], optimize=True)
    )
    dR.aab.VvvOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('AmIK,bcJm->AbcIJK', H.ab.vooo[Va, :, Oa, Ob], R.ab[va, vb, Oa, :], optimize=True)
    )
    dR.aab.VvvOOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('bmIK,AcJm->AbcIJK', H.ab.vooo[va, :, Oa, Ob], R.ab[Va, vb, Oa, :], optimize=True)
    )
    dR.aab.VvvOOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('AbIe,ecJK->AbcIJK', H.aa.vvov[Va, va, Oa, :], R.ab[:, vb, Oa, Ob], optimize=True)
    )
    dR.aab.VvvOOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('AmIJ,bcmK->AbcIJK', H.aa.vooo[Va, :, Oa, Oa], R.ab[va, vb, :, Ob], optimize=True)
    )
    dR.aab.VvvOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('bmIJ,AcmK->AbcIJK', H.aa.vooo[va, :, Oa, Oa], R.ab[Va, vb, :, Ob], optimize=True)
    )

    dR.aab.VvvOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mI,AbcmJK->AbcIJK', X.a.oo[oa, Oa], T.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('MI,AbcMJK->AbcIJK', X.a.oo[Oa, Oa], T.aab.VvvOOO, optimize=True)
    )
    dR.aab.VvvOOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mK,AbcIJm->AbcIJK', X.b.oo[ob, Ob], T.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('MK,AbcIJM->AbcIJK', X.b.oo[Ob, Ob], T.aab.VvvOOO, optimize=True)
    )
    dR.aab.VvvOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('AE,EbcIJK->AbcIJK', X.a.vv[Va, Va], T.aab.VvvOOO, optimize=True)
    )
    dR.aab.VvvOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('be,AecIJK->AbcIJK', X.a.vv[va, va], T.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('bE,AEcIJK->AbcIJK', X.a.vv[va, Va], T.aab.VVvOOO, optimize=True)
    )
    dR.aab.VvvOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('ce,AbeIJK->AbcIJK', X.b.vv[vb, vb], T.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('cE,AbEIJK->AbcIJK', X.b.vv[vb, Vb], T.aab.VvVOOO, optimize=True)
    )
    dR.aab.VvvOOO += (1.0 / 2.0) * (
            +0.5 * np.einsum('mnIJ,AbcmnK->AbcIJK', X.aa.oooo[oa, oa, Oa, Oa], T.aab.VvvooO, optimize=True)
            - 1.0 * np.einsum('MnIJ,AbcnMK->AbcIJK', X.aa.oooo[Oa, oa, Oa, Oa], T.aab.VvvoOO, optimize=True)
            + 0.5 * np.einsum('MNIJ,AbcMNK->AbcIJK', X.aa.oooo[Oa, Oa, Oa, Oa], T.aab.VvvOOO, optimize=True)
    )
    dR.aab.VvvOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mnJK,AbcmIn->AbcIJK', X.ab.oooo[oa, ob, Oa, Ob], T.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('mNJK,AbcmIN->AbcIJK', X.ab.oooo[oa, Ob, Oa, Ob], T.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('MnJK,AbcIMn->AbcIJK', X.ab.oooo[Oa, ob, Oa, Ob], T.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('MNJK,AbcIMN->AbcIJK', X.ab.oooo[Oa, Ob, Oa, Ob], T.aab.VvvOOO, optimize=True)
    )
    dR.aab.VvvOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('AbEf,EfcIJK->AbcIJK', X.aa.vvvv[Va, va, Va, va], T.aab.VvvOOO, optimize=True)
            - 0.5 * np.einsum('AbEF,FEcIJK->AbcIJK', X.aa.vvvv[Va, va, Va, Va], T.aab.VVvOOO, optimize=True)
    )
    dR.aab.VvvOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('bcef,AefIJK->AbcIJK', X.ab.vvvv[va, vb, va, vb], T.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('bceF,AeFIJK->AbcIJK', X.ab.vvvv[va, vb, va, Vb], T.aab.VvVOOO, optimize=True)
            - 1.0 * np.einsum('bcEf,EAfIJK->AbcIJK', X.ab.vvvv[va, vb, Va, vb], T.aab.VVvOOO, optimize=True)
            - 1.0 * np.einsum('bcEF,EAFIJK->AbcIJK', X.ab.vvvv[va, vb, Va, Vb], T.aab.VVVOOO, optimize=True)
    )
    dR.aab.VvvOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('AceF,ebFIJK->AbcIJK', X.ab.vvvv[Va, vb, va, Vb], T.aab.vvVOOO, optimize=True)
            + 1.0 * np.einsum('AcEf,EbfIJK->AbcIJK', X.ab.vvvv[Va, vb, Va, vb], T.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('AcEF,EbFIJK->AbcIJK', X.ab.vvvv[Va, vb, Va, Vb], T.aab.VvVOOO, optimize=True)
    )
    dR.aab.VvvOOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('AmIE,EbcmJK->AbcIJK', X.aa.voov[Va, oa, Oa, Va], T.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('AMIE,EbcMJK->AbcIJK', X.aa.voov[Va, Oa, Oa, Va], T.aab.VvvOOO, optimize=True)
    )
    dR.aab.VvvOOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('bmIe,AecmJK->AbcIJK', X.aa.voov[va, oa, Oa, va], T.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('bMIe,AecMJK->AbcIJK', X.aa.voov[va, Oa, Oa, va], T.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('bmIE,AEcmJK->AbcIJK', X.aa.voov[va, oa, Oa, Va], T.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('bMIE,AEcMJK->AbcIJK', X.aa.voov[va, Oa, Oa, Va], T.aab.VVvOOO, optimize=True)
    )
    dR.aab.VvvOOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('AmIE,bEcJmK->AbcIJK', X.ab.voov[Va, ob, Oa, Vb], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('AMIE,bEcJMK->AbcIJK', X.ab.voov[Va, Ob, Oa, Vb], T.abb.vVvOOO, optimize=True)
    )
    dR.aab.VvvOOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('bmIe,AceJmK->AbcIJK', X.ab.voov[va, ob, Oa, vb], T.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('bMIe,AceJMK->AbcIJK', X.ab.voov[va, Ob, Oa, vb], T.abb.VvvOOO, optimize=True)
            - 1.0 * np.einsum('bmIE,AEcJmK->AbcIJK', X.ab.voov[va, ob, Oa, Vb], T.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('bMIE,AEcJMK->AbcIJK', X.ab.voov[va, Ob, Oa, Vb], T.abb.VVvOOO, optimize=True)
    )
    dR.aab.VvvOOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mceK,AebmIJ->AbcIJK', X.ab.ovvo[oa, vb, va, Ob], T.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('MceK,AebIJM->AbcIJK', X.ab.ovvo[Oa, vb, va, Ob], T.aaa.VvvOOO, optimize=True)
            + 1.0 * np.einsum('mcEK,EAbmIJ->AbcIJK', X.ab.ovvo[oa, vb, Va, Ob], T.aaa.VVvoOO, optimize=True)
            + 1.0 * np.einsum('McEK,EAbIJM->AbcIJK', X.ab.ovvo[Oa, vb, Va, Ob], T.aaa.VVvOOO, optimize=True)
    )
    dR.aab.VvvOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('cmKe,AbeIJm->AbcIJK', X.bb.voov[vb, ob, Ob, vb], T.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('cMKe,AbeIJM->AbcIJK', X.bb.voov[vb, Ob, Ob, vb], T.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('cmKE,AbEIJm->AbcIJK', X.bb.voov[vb, ob, Ob, Vb], T.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('cMKE,AbEIJM->AbcIJK', X.bb.voov[vb, Ob, Ob, Vb], T.aab.VvVOOO, optimize=True)
    )
    dR.aab.VvvOOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('AmEK,EbcIJm->AbcIJK', X.ab.vovo[Va, ob, Va, Ob], T.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('AMEK,EbcIJM->AbcIJK', X.ab.vovo[Va, Ob, Va, Ob], T.aab.VvvOOO, optimize=True)
    )
    dR.aab.VvvOOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('bmeK,AecIJm->AbcIJK', X.ab.vovo[va, ob, va, Ob], T.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('bMeK,AecIJM->AbcIJK', X.ab.vovo[va, Ob, va, Ob], T.aab.VvvOOO, optimize=True)
            - 1.0 * np.einsum('bmEK,AEcIJm->AbcIJK', X.ab.vovo[va, ob, Va, Ob], T.aab.VVvOOo, optimize=True)
            - 1.0 * np.einsum('bMEK,AEcIJM->AbcIJK', X.ab.vovo[va, Ob, Va, Ob], T.aab.VVvOOO, optimize=True)
    )
    dR.aab.VvvOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mcIe,AbemJK->AbcIJK', X.ab.ovov[oa, vb, Oa, vb], T.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('McIe,AbeMJK->AbcIJK', X.ab.ovov[Oa, vb, Oa, vb], T.aab.VvvOOO, optimize=True)
            - 1.0 * np.einsum('mcIE,AbEmJK->AbcIJK', X.ab.ovov[oa, vb, Oa, Vb], T.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('McIE,AbEMJK->AbcIJK', X.ab.ovov[Oa, vb, Oa, Vb], T.aab.VvVOOO, optimize=True)
    )
    dR.aab.VvvOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mI,AbcmJK->AbcIJK', H.a.oo[oa, Oa], R.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('MI,AbcMJK->AbcIJK', H.a.oo[Oa, Oa], R.aab.VvvOOO, optimize=True)
    )
    dR.aab.VvvOOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mK,AbcIJm->AbcIJK', H.b.oo[ob, Ob], R.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('MK,AbcIJM->AbcIJK', H.b.oo[Ob, Ob], R.aab.VvvOOO, optimize=True)
    )
    dR.aab.VvvOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('AE,EbcIJK->AbcIJK', H.a.vv[Va, Va], R.aab.VvvOOO, optimize=True)
    )
    dR.aab.VvvOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('be,AecIJK->AbcIJK', H.a.vv[va, va], R.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('bE,AEcIJK->AbcIJK', H.a.vv[va, Va], R.aab.VVvOOO, optimize=True)
    )
    dR.aab.VvvOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('ce,AbeIJK->AbcIJK', H.b.vv[vb, vb], R.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('cE,AbEIJK->AbcIJK', H.b.vv[vb, Vb], R.aab.VvVOOO, optimize=True)
    )
    dR.aab.VvvOOO += (1.0 / 2.0) * (
            +0.5 * np.einsum('mnIJ,AbcmnK->AbcIJK', H.aa.oooo[oa, oa, Oa, Oa], R.aab.VvvooO, optimize=True)
            - 1.0 * np.einsum('MnIJ,AbcnMK->AbcIJK', H.aa.oooo[Oa, oa, Oa, Oa], R.aab.VvvoOO, optimize=True)
            + 0.5 * np.einsum('MNIJ,AbcMNK->AbcIJK', H.aa.oooo[Oa, Oa, Oa, Oa], R.aab.VvvOOO, optimize=True)
    )
    dR.aab.VvvOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mnJK,AbcmIn->AbcIJK', H.ab.oooo[oa, ob, Oa, Ob], R.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('mNJK,AbcmIN->AbcIJK', H.ab.oooo[oa, Ob, Oa, Ob], R.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('MnJK,AbcIMn->AbcIJK', H.ab.oooo[Oa, ob, Oa, Ob], R.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('MNJK,AbcIMN->AbcIJK', H.ab.oooo[Oa, Ob, Oa, Ob], R.aab.VvvOOO, optimize=True)
    )
    dR.aab.VvvOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('AbEf,EfcIJK->AbcIJK', H.aa.vvvv[Va, va, Va, va], R.aab.VvvOOO, optimize=True)
            - 0.5 * np.einsum('AbEF,FEcIJK->AbcIJK', H.aa.vvvv[Va, va, Va, Va], R.aab.VVvOOO, optimize=True)
    )
    dR.aab.VvvOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('bcef,AefIJK->AbcIJK', H.ab.vvvv[va, vb, va, vb], R.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('bceF,AeFIJK->AbcIJK', H.ab.vvvv[va, vb, va, Vb], R.aab.VvVOOO, optimize=True)
            - 1.0 * np.einsum('bcEf,EAfIJK->AbcIJK', H.ab.vvvv[va, vb, Va, vb], R.aab.VVvOOO, optimize=True)
            - 1.0 * np.einsum('bcEF,EAFIJK->AbcIJK', H.ab.vvvv[va, vb, Va, Vb], R.aab.VVVOOO, optimize=True)
    )
    dR.aab.VvvOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('AceF,ebFIJK->AbcIJK', H.ab.vvvv[Va, vb, va, Vb], R.aab.vvVOOO, optimize=True)
            + 1.0 * np.einsum('AcEf,EbfIJK->AbcIJK', H.ab.vvvv[Va, vb, Va, vb], R.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('AcEF,EbFIJK->AbcIJK', H.ab.vvvv[Va, vb, Va, Vb], R.aab.VvVOOO, optimize=True)
    )
    dR.aab.VvvOOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('AmIE,EbcmJK->AbcIJK', H.aa.voov[Va, oa, Oa, Va], R.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('AMIE,EbcMJK->AbcIJK', H.aa.voov[Va, Oa, Oa, Va], R.aab.VvvOOO, optimize=True)
    )
    dR.aab.VvvOOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('bmIe,AecmJK->AbcIJK', H.aa.voov[va, oa, Oa, va], R.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('bMIe,AecMJK->AbcIJK', H.aa.voov[va, Oa, Oa, va], R.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('bmIE,AEcmJK->AbcIJK', H.aa.voov[va, oa, Oa, Va], R.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('bMIE,AEcMJK->AbcIJK', H.aa.voov[va, Oa, Oa, Va], R.aab.VVvOOO, optimize=True)
    )
    dR.aab.VvvOOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('AmIE,bEcJmK->AbcIJK', H.ab.voov[Va, ob, Oa, Vb], R.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('AMIE,bEcJMK->AbcIJK', H.ab.voov[Va, Ob, Oa, Vb], R.abb.vVvOOO, optimize=True)
    )
    dR.aab.VvvOOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('bmIe,AceJmK->AbcIJK', H.ab.voov[va, ob, Oa, vb], R.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('bMIe,AceJMK->AbcIJK', H.ab.voov[va, Ob, Oa, vb], R.abb.VvvOOO, optimize=True)
            - 1.0 * np.einsum('bmIE,AEcJmK->AbcIJK', H.ab.voov[va, ob, Oa, Vb], R.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('bMIE,AEcJMK->AbcIJK', H.ab.voov[va, Ob, Oa, Vb], R.abb.VVvOOO, optimize=True)
    )
    dR.aab.VvvOOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mceK,AebmIJ->AbcIJK', H.ab.ovvo[oa, vb, va, Ob], R.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('MceK,AebIJM->AbcIJK', H.ab.ovvo[Oa, vb, va, Ob], R.aaa.VvvOOO, optimize=True)
            + 1.0 * np.einsum('mcEK,EAbmIJ->AbcIJK', H.ab.ovvo[oa, vb, Va, Ob], R.aaa.VVvoOO, optimize=True)
            + 1.0 * np.einsum('McEK,EAbIJM->AbcIJK', H.ab.ovvo[Oa, vb, Va, Ob], R.aaa.VVvOOO, optimize=True)
    )
    dR.aab.VvvOOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('cmKe,AbeIJm->AbcIJK', H.bb.voov[vb, ob, Ob, vb], R.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('cMKe,AbeIJM->AbcIJK', H.bb.voov[vb, Ob, Ob, vb], R.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('cmKE,AbEIJm->AbcIJK', H.bb.voov[vb, ob, Ob, Vb], R.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('cMKE,AbEIJM->AbcIJK', H.bb.voov[vb, Ob, Ob, Vb], R.aab.VvVOOO, optimize=True)
    )
    dR.aab.VvvOOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('AmEK,EbcIJm->AbcIJK', H.ab.vovo[Va, ob, Va, Ob], R.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('AMEK,EbcIJM->AbcIJK', H.ab.vovo[Va, Ob, Va, Ob], R.aab.VvvOOO, optimize=True)
    )
    dR.aab.VvvOOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('bmeK,AecIJm->AbcIJK', H.ab.vovo[va, ob, va, Ob], R.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('bMeK,AecIJM->AbcIJK', H.ab.vovo[va, Ob, va, Ob], R.aab.VvvOOO, optimize=True)
            - 1.0 * np.einsum('bmEK,AEcIJm->AbcIJK', H.ab.vovo[va, ob, Va, Ob], R.aab.VVvOOo, optimize=True)
            - 1.0 * np.einsum('bMEK,AEcIJM->AbcIJK', H.ab.vovo[va, Ob, Va, Ob], R.aab.VVvOOO, optimize=True)
    )
    dR.aab.VvvOOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mcIe,AbemJK->AbcIJK', H.ab.ovov[oa, vb, Oa, vb], R.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('McIe,AbeMJK->AbcIJK', H.ab.ovov[Oa, vb, Oa, vb], R.aab.VvvOOO, optimize=True)
            - 1.0 * np.einsum('mcIE,AbEmJK->AbcIJK', H.ab.ovov[oa, vb, Oa, Vb], R.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('McIE,AbEMJK->AbcIJK', H.ab.ovov[Oa, vb, Oa, Vb], R.aab.VvVOOO, optimize=True)
    )

    dR.aab.VvvOOO -= np.transpose(dR.aab.VvvOOO, (0, 1, 2, 4, 3, 5))

    return dR

def update(R, omega, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    R.aab.VvvOOO = eomcc_active_loops.update_r3b_100111(
        R.aab.VvvOOO,
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
