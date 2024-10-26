import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops

def build(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.aab.VvvooO = (1.0 / 2.0) * (
            +1.0 * np.einsum('bceK,Aeij->AbcijK', X.ab.vvvo[va, vb, :, Ob], T.aa[Va, :, oa, oa], optimize=True)
    )
    dR.aab.VvvooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('AceK,beij->AbcijK', X.ab.vvvo[Va, vb, :, Ob], T.aa[va, :, oa, oa], optimize=True)
    )
    dR.aab.VvvooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mcjK,Abim->AbcijK', X.ab.ovoo[:, vb, oa, Ob], T.aa[Va, va, oa, :], optimize=True)
    )
    dR.aab.VvvooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('Acie,bejK->AbcijK', X.ab.vvov[Va, vb, oa, :], T.ab[va, :, oa, Ob], optimize=True)
    )
    dR.aab.VvvooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('bcie,AejK->AbcijK', X.ab.vvov[va, vb, oa, :], T.ab[Va, :, oa, Ob], optimize=True)
    )
    dR.aab.VvvooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('AmiK,bcjm->AbcijK', X.ab.vooo[Va, :, oa, Ob], T.ab[va, vb, oa, :], optimize=True)
    )
    dR.aab.VvvooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('bmiK,Acjm->AbcijK', X.ab.vooo[va, :, oa, Ob], T.ab[Va, vb, oa, :], optimize=True)
    )
    dR.aab.VvvooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('Abie,ecjK->AbcijK', X.aa.vvov[Va, va, oa, :], T.ab[:, vb, oa, Ob], optimize=True)
    )
    dR.aab.VvvooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('Amij,bcmK->AbcijK', X.aa.vooo[Va, :, oa, oa], T.ab[va, vb, :, Ob], optimize=True)
    )
    dR.aab.VvvooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('bmij,AcmK->AbcijK', X.aa.vooo[va, :, oa, oa], T.ab[Va, vb, :, Ob], optimize=True)
    )
    dR.aab.VvvooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('bceK,Aeij->AbcijK', H.ab.vvvo[va, vb, :, Ob], R.aa[Va, :, oa, oa], optimize=True)
    )
    dR.aab.VvvooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('AceK,beij->AbcijK', H.ab.vvvo[Va, vb, :, Ob], R.aa[va, :, oa, oa], optimize=True)
    )
    dR.aab.VvvooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mcjK,Abim->AbcijK', H.ab.ovoo[:, vb, oa, Ob], R.aa[Va, va, oa, :], optimize=True)
    )
    dR.aab.VvvooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('Acie,bejK->AbcijK', H.ab.vvov[Va, vb, oa, :], R.ab[va, :, oa, Ob], optimize=True)
    )
    dR.aab.VvvooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('bcie,AejK->AbcijK', H.ab.vvov[va, vb, oa, :], R.ab[Va, :, oa, Ob], optimize=True)
    )
    dR.aab.VvvooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('AmiK,bcjm->AbcijK', H.ab.vooo[Va, :, oa, Ob], R.ab[va, vb, oa, :], optimize=True)
    )
    dR.aab.VvvooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('bmiK,Acjm->AbcijK', H.ab.vooo[va, :, oa, Ob], R.ab[Va, vb, oa, :], optimize=True)
    )
    dR.aab.VvvooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('Abie,ecjK->AbcijK', H.aa.vvov[Va, va, oa, :], R.ab[:, vb, oa, Ob], optimize=True)
    )
    dR.aab.VvvooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('Amij,bcmK->AbcijK', H.aa.vooo[Va, :, oa, oa], R.ab[va, vb, :, Ob], optimize=True)
    )
    dR.aab.VvvooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('bmij,AcmK->AbcijK', H.aa.vooo[va, :, oa, oa], R.ab[Va, vb, :, Ob], optimize=True)
    )

    dR.aab.VvvooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mi,AbcmjK->AbcijK', X.a.oo[oa, oa], T.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('Mi,AbcjMK->AbcijK', X.a.oo[Oa, oa], T.aab.VvvoOO, optimize=True)
    )
    dR.aab.VvvooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('MK,AbcijM->AbcijK', X.b.oo[Ob, Ob], T.aab.VvvooO, optimize=True)
    )
    dR.aab.VvvooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('AE,EbcijK->AbcijK', X.a.vv[Va, Va], T.aab.VvvooO, optimize=True)
    )
    dR.aab.VvvooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('be,AecijK->AbcijK', X.a.vv[va, va], T.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('bE,AEcijK->AbcijK', X.a.vv[va, Va], T.aab.VVvooO, optimize=True)
    )
    dR.aab.VvvooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('ce,AbeijK->AbcijK', X.b.vv[vb, vb], T.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('cE,AbEijK->AbcijK', X.b.vv[vb, Vb], T.aab.VvVooO, optimize=True)
    )
    dR.aab.VvvooO += (1.0 / 2.0) * (
            +0.5 * np.einsum('mnij,AbcmnK->AbcijK', X.aa.oooo[oa, oa, oa, oa], T.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('mNij,AbcmNK->AbcijK', X.aa.oooo[oa, Oa, oa, oa], T.aab.VvvoOO, optimize=True)
            + 0.5 * np.einsum('MNij,AbcMNK->AbcijK', X.aa.oooo[Oa, Oa, oa, oa], T.aab.VvvOOO, optimize=True)
    )
    dR.aab.VvvooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('MnjK,AbciMn->AbcijK', X.ab.oooo[Oa, ob, oa, Ob], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('mNjK,AbcimN->AbcijK', X.ab.oooo[oa, Ob, oa, Ob], T.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('MNjK,AbciMN->AbcijK', X.ab.oooo[Oa, Ob, oa, Ob], T.aab.VvvoOO, optimize=True)
    )
    dR.aab.VvvooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('AbEf,EfcijK->AbcijK', X.aa.vvvv[Va, va, Va, va], T.aab.VvvooO, optimize=True)
            - 0.5 * np.einsum('AbEF,FEcijK->AbcijK', X.aa.vvvv[Va, va, Va, Va], T.aab.VVvooO, optimize=True)
    )
    dR.aab.VvvooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('bcef,AefijK->AbcijK', X.ab.vvvv[va, vb, va, vb], T.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('bceF,AeFijK->AbcijK', X.ab.vvvv[va, vb, va, Vb], T.aab.VvVooO, optimize=True)
            - 1.0 * np.einsum('bcEf,EAfijK->AbcijK', X.ab.vvvv[va, vb, Va, vb], T.aab.VVvooO, optimize=True)
            - 1.0 * np.einsum('bcEF,EAFijK->AbcijK', X.ab.vvvv[va, vb, Va, Vb], T.aab.VVVooO, optimize=True)
    )
    dR.aab.VvvooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('AceF,ebFijK->AbcijK', X.ab.vvvv[Va, vb, va, Vb], T.aab.vvVooO, optimize=True)
            + 1.0 * np.einsum('AcEf,EbfijK->AbcijK', X.ab.vvvv[Va, vb, Va, vb], T.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('AcEF,EbFijK->AbcijK', X.ab.vvvv[Va, vb, Va, Vb], T.aab.VvVooO, optimize=True)
    )
    dR.aab.VvvooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('AmiE,EbcmjK->AbcijK', X.aa.voov[Va, oa, oa, Va], T.aab.VvvooO, optimize=True)
            - 1.0 * np.einsum('AMiE,EbcjMK->AbcijK', X.aa.voov[Va, Oa, oa, Va], T.aab.VvvoOO, optimize=True)
    )
    dR.aab.VvvooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('bmie,AecmjK->AbcijK', X.aa.voov[va, oa, oa, va], T.aab.VvvooO, optimize=True)
            - 1.0 * np.einsum('bMie,AecjMK->AbcijK', X.aa.voov[va, Oa, oa, va], T.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('bmiE,AEcmjK->AbcijK', X.aa.voov[va, oa, oa, Va], T.aab.VVvooO, optimize=True)
            - 1.0 * np.einsum('bMiE,AEcjMK->AbcijK', X.aa.voov[va, Oa, oa, Va], T.aab.VVvoOO, optimize=True)
    )
    dR.aab.VvvooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('AmiE,bEcjmK->AbcijK', X.ab.voov[Va, ob, oa, Vb], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('AMiE,bEcjMK->AbcijK', X.ab.voov[Va, Ob, oa, Vb], T.abb.vVvoOO, optimize=True)
    )
    dR.aab.VvvooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('bmie,AcejmK->AbcijK', X.ab.voov[va, ob, oa, vb], T.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('bMie,AcejMK->AbcijK', X.ab.voov[va, Ob, oa, vb], T.abb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('bmiE,AEcjmK->AbcijK', X.ab.voov[va, ob, oa, Vb], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('bMiE,AEcjMK->AbcijK', X.ab.voov[va, Ob, oa, Vb], T.abb.VVvoOO, optimize=True)
    )
    dR.aab.VvvooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('MceK,AebijM->AbcijK', X.ab.ovvo[Oa, vb, va, Ob], T.aaa.VvvooO, optimize=True)
            + 1.0 * np.einsum('McEK,EAbijM->AbcijK', X.ab.ovvo[Oa, vb, Va, Ob], T.aaa.VVvooO, optimize=True)
    )
    dR.aab.VvvooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('cMKe,AbeijM->AbcijK', X.bb.voov[vb, Ob, Ob, vb], T.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('cMKE,AbEijM->AbcijK', X.bb.voov[vb, Ob, Ob, Vb], T.aab.VvVooO, optimize=True)
    )
    dR.aab.VvvooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('AMEK,EbcijM->AbcijK', X.ab.vovo[Va, Ob, Va, Ob], T.aab.VvvooO, optimize=True)
    )
    dR.aab.VvvooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('bMeK,AecijM->AbcijK', X.ab.vovo[va, Ob, va, Ob], T.aab.VvvooO, optimize=True)
            - 1.0 * np.einsum('bMEK,AEcijM->AbcijK', X.ab.vovo[va, Ob, Va, Ob], T.aab.VVvooO, optimize=True)
    )
    dR.aab.VvvooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mcie,AbemjK->AbcijK', X.ab.ovov[oa, vb, oa, vb], T.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('Mcie,AbejMK->AbcijK', X.ab.ovov[Oa, vb, oa, vb], T.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('mciE,AbEmjK->AbcijK', X.ab.ovov[oa, vb, oa, Vb], T.aab.VvVooO, optimize=True)
            + 1.0 * np.einsum('MciE,AbEjMK->AbcijK', X.ab.ovov[Oa, vb, oa, Vb], T.aab.VvVoOO, optimize=True)
    )
    dR.aab.VvvooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mi,AbcmjK->AbcijK', H.a.oo[oa, oa], R.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('Mi,AbcjMK->AbcijK', H.a.oo[Oa, oa], R.aab.VvvoOO, optimize=True)
    )
    dR.aab.VvvooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('MK,AbcijM->AbcijK', H.b.oo[Ob, Ob], R.aab.VvvooO, optimize=True)
    )
    dR.aab.VvvooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('AE,EbcijK->AbcijK', H.a.vv[Va, Va], R.aab.VvvooO, optimize=True)
    )
    dR.aab.VvvooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('be,AecijK->AbcijK', H.a.vv[va, va], R.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('bE,AEcijK->AbcijK', H.a.vv[va, Va], R.aab.VVvooO, optimize=True)
    )
    dR.aab.VvvooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('ce,AbeijK->AbcijK', H.b.vv[vb, vb], R.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('cE,AbEijK->AbcijK', H.b.vv[vb, Vb], R.aab.VvVooO, optimize=True)
    )
    dR.aab.VvvooO += (1.0 / 2.0) * (
            +0.5 * np.einsum('mnij,AbcmnK->AbcijK', H.aa.oooo[oa, oa, oa, oa], R.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('mNij,AbcmNK->AbcijK', H.aa.oooo[oa, Oa, oa, oa], R.aab.VvvoOO, optimize=True)
            + 0.5 * np.einsum('MNij,AbcMNK->AbcijK', H.aa.oooo[Oa, Oa, oa, oa], R.aab.VvvOOO, optimize=True)
    )
    dR.aab.VvvooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('MnjK,AbciMn->AbcijK', H.ab.oooo[Oa, ob, oa, Ob], R.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('mNjK,AbcimN->AbcijK', H.ab.oooo[oa, Ob, oa, Ob], R.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('MNjK,AbciMN->AbcijK', H.ab.oooo[Oa, Ob, oa, Ob], R.aab.VvvoOO, optimize=True)
    )
    dR.aab.VvvooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('AbEf,EfcijK->AbcijK', H.aa.vvvv[Va, va, Va, va], R.aab.VvvooO, optimize=True)
            - 0.5 * np.einsum('AbEF,FEcijK->AbcijK', H.aa.vvvv[Va, va, Va, Va], R.aab.VVvooO, optimize=True)
    )
    dR.aab.VvvooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('bcef,AefijK->AbcijK', H.ab.vvvv[va, vb, va, vb], R.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('bceF,AeFijK->AbcijK', H.ab.vvvv[va, vb, va, Vb], R.aab.VvVooO, optimize=True)
            - 1.0 * np.einsum('bcEf,EAfijK->AbcijK', H.ab.vvvv[va, vb, Va, vb], R.aab.VVvooO, optimize=True)
            - 1.0 * np.einsum('bcEF,EAFijK->AbcijK', H.ab.vvvv[va, vb, Va, Vb], R.aab.VVVooO, optimize=True)
    )
    dR.aab.VvvooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('AceF,ebFijK->AbcijK', H.ab.vvvv[Va, vb, va, Vb], R.aab.vvVooO, optimize=True)
            + 1.0 * np.einsum('AcEf,EbfijK->AbcijK', H.ab.vvvv[Va, vb, Va, vb], R.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('AcEF,EbFijK->AbcijK', H.ab.vvvv[Va, vb, Va, Vb], R.aab.VvVooO, optimize=True)
    )
    dR.aab.VvvooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('AmiE,EbcmjK->AbcijK', H.aa.voov[Va, oa, oa, Va], R.aab.VvvooO, optimize=True)
            - 1.0 * np.einsum('AMiE,EbcjMK->AbcijK', H.aa.voov[Va, Oa, oa, Va], R.aab.VvvoOO, optimize=True)
    )
    dR.aab.VvvooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('bmie,AecmjK->AbcijK', H.aa.voov[va, oa, oa, va], R.aab.VvvooO, optimize=True)
            - 1.0 * np.einsum('bMie,AecjMK->AbcijK', H.aa.voov[va, Oa, oa, va], R.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('bmiE,AEcmjK->AbcijK', H.aa.voov[va, oa, oa, Va], R.aab.VVvooO, optimize=True)
            - 1.0 * np.einsum('bMiE,AEcjMK->AbcijK', H.aa.voov[va, Oa, oa, Va], R.aab.VVvoOO, optimize=True)
    )
    dR.aab.VvvooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('AmiE,bEcjmK->AbcijK', H.ab.voov[Va, ob, oa, Vb], R.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('AMiE,bEcjMK->AbcijK', H.ab.voov[Va, Ob, oa, Vb], R.abb.vVvoOO, optimize=True)
    )
    dR.aab.VvvooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('bmie,AcejmK->AbcijK', H.ab.voov[va, ob, oa, vb], R.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('bMie,AcejMK->AbcijK', H.ab.voov[va, Ob, oa, vb], R.abb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('bmiE,AEcjmK->AbcijK', H.ab.voov[va, ob, oa, Vb], R.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('bMiE,AEcjMK->AbcijK', H.ab.voov[va, Ob, oa, Vb], R.abb.VVvoOO, optimize=True)
    )
    dR.aab.VvvooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('MceK,AebijM->AbcijK', H.ab.ovvo[Oa, vb, va, Ob], R.aaa.VvvooO, optimize=True)
            + 1.0 * np.einsum('McEK,EAbijM->AbcijK', H.ab.ovvo[Oa, vb, Va, Ob], R.aaa.VVvooO, optimize=True)
    )
    dR.aab.VvvooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('cMKe,AbeijM->AbcijK', H.bb.voov[vb, Ob, Ob, vb], R.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('cMKE,AbEijM->AbcijK', H.bb.voov[vb, Ob, Ob, Vb], R.aab.VvVooO, optimize=True)
    )
    dR.aab.VvvooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('AMEK,EbcijM->AbcijK', H.ab.vovo[Va, Ob, Va, Ob], R.aab.VvvooO, optimize=True)
    )
    dR.aab.VvvooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('bMeK,AecijM->AbcijK', H.ab.vovo[va, Ob, va, Ob], R.aab.VvvooO, optimize=True)
            - 1.0 * np.einsum('bMEK,AEcijM->AbcijK', H.ab.vovo[va, Ob, Va, Ob], R.aab.VVvooO, optimize=True)
    )
    dR.aab.VvvooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mcie,AbemjK->AbcijK', H.ab.ovov[oa, vb, oa, vb], R.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('Mcie,AbejMK->AbcijK', H.ab.ovov[Oa, vb, oa, vb], R.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('mciE,AbEmjK->AbcijK', H.ab.ovov[oa, vb, oa, Vb], R.aab.VvVooO, optimize=True)
            + 1.0 * np.einsum('MciE,AbEjMK->AbcijK', H.ab.ovov[Oa, vb, oa, Vb], R.aab.VvVoOO, optimize=True)
    )

    dR.aab.VvvooO -= np.transpose(dR.aab.VvvooO, (0, 1, 2, 4, 3, 5))

    return dR

def update(R, omega, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    R.aab.VvvooO = eomcc_active_loops.update_r3b_100001(
        R.aab.VvvooO,
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
