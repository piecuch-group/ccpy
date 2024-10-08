import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops

def build(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.abb.VvvooO = (2.0 / 2.0) * (
            +1.0 * np.einsum('Abie,ecjK->AbcijK', X.ab.vvov[Va, vb, oa, :], T.bb[:, vb, ob, Ob], optimize=True)
    )
    dR.abb.VvvooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('Amij,bcmK->AbcijK', X.ab.vooo[Va, :, oa, ob], T.bb[vb, vb, :, Ob], optimize=True)
    )
    dR.abb.VvvooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('AmiK,bcmj->AbcijK', X.ab.vooo[Va, :, oa, Ob], T.bb[vb, vb, :, ob], optimize=True)
    )
    dR.abb.VvvooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('cbKe,Aeij->AbcijK', X.bb.vvov[vb, vb, Ob, :], T.ab[Va, :, oa, ob], optimize=True)
    )
    dR.abb.VvvooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('cbje,AeiK->AbcijK', X.bb.vvov[vb, vb, ob, :], T.ab[Va, :, oa, Ob], optimize=True)
    )
    dR.abb.VvvooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('cmKj,Abim->AbcijK', X.bb.vooo[vb, :, Ob, ob], T.ab[Va, vb, oa, :], optimize=True)
    )
    dR.abb.VvvooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('Abej,eciK->AbcijK', X.ab.vvvo[Va, vb, :, ob], T.ab[:, vb, oa, Ob], optimize=True)
    )
    dR.abb.VvvooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('AbeK,ecij->AbcijK', X.ab.vvvo[Va, vb, :, Ob], T.ab[:, vb, oa, ob], optimize=True)
    )
    dR.abb.VvvooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mbij,AcmK->AbcijK', X.ab.ovoo[:, vb, oa, ob], T.ab[Va, vb, :, Ob], optimize=True)
    )
    dR.abb.VvvooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mbiK,Acmj->AbcijK', X.ab.ovoo[:, vb, oa, Ob], T.ab[Va, vb, :, ob], optimize=True)
    )
    dR.abb.VvvooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('Abie,ecjK->AbcijK', H.ab.vvov[Va, vb, oa, :], R.bb[:, vb, ob, Ob], optimize=True)
    )
    dR.abb.VvvooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('Amij,bcmK->AbcijK', H.ab.vooo[Va, :, oa, ob], R.bb[vb, vb, :, Ob], optimize=True)
    )
    dR.abb.VvvooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('AmiK,bcmj->AbcijK', H.ab.vooo[Va, :, oa, Ob], R.bb[vb, vb, :, ob], optimize=True)
    )
    dR.abb.VvvooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('cbKe,Aeij->AbcijK', H.bb.vvov[vb, vb, Ob, :], R.ab[Va, :, oa, ob], optimize=True)
    )
    dR.abb.VvvooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('cbje,AeiK->AbcijK', H.bb.vvov[vb, vb, ob, :], R.ab[Va, :, oa, Ob], optimize=True)
    )
    dR.abb.VvvooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('cmKj,Abim->AbcijK', H.bb.vooo[vb, :, Ob, ob], R.ab[Va, vb, oa, :], optimize=True)
    )
    dR.abb.VvvooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('Abej,eciK->AbcijK', H.ab.vvvo[Va, vb, :, ob], R.ab[:, vb, oa, Ob], optimize=True)
    )
    dR.abb.VvvooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('AbeK,ecij->AbcijK', H.ab.vvvo[Va, vb, :, Ob], R.ab[:, vb, oa, ob], optimize=True)
    )
    dR.abb.VvvooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mbij,AcmK->AbcijK', H.ab.ovoo[:, vb, oa, ob], R.ab[Va, vb, :, Ob], optimize=True)
    )
    dR.abb.VvvooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mbiK,Acmj->AbcijK', H.ab.ovoo[:, vb, oa, Ob], R.ab[Va, vb, :, ob], optimize=True)
    )
    # of terms =  20
    dR.abb.VvvooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mi,AcbmjK->AbcijK', X.a.oo[oa, oa], T.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('Mi,AcbMjK->AbcijK', X.a.oo[Oa, oa], T.abb.VvvOoO, optimize=True)
    )
    dR.abb.VvvooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mj,AcbimK->AbcijK', X.b.oo[ob, ob], T.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('Mj,AcbiMK->AbcijK', X.b.oo[Ob, ob], T.abb.VvvoOO, optimize=True)
    )
    dR.abb.VvvooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('MK,AcbijM->AbcijK', X.b.oo[Ob, Ob], T.abb.VvvooO, optimize=True)
    )
    dR.abb.VvvooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('AE,EcbijK->AbcijK', X.a.vv[Va, Va], T.abb.VvvooO, optimize=True)
    )
    dR.abb.VvvooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('be,AceijK->AbcijK', X.b.vv[vb, vb], T.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('bE,AEcijK->AbcijK', X.b.vv[vb, Vb], T.abb.VVvooO, optimize=True)
    )
    dR.abb.VvvooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('MnjK,AcbinM->AbcijK', X.bb.oooo[Ob, ob, ob, Ob], T.abb.VvvooO, optimize=True)
            - 0.5 * np.einsum('MNjK,AcbiMN->AbcijK', X.bb.oooo[Ob, Ob, ob, Ob], T.abb.VvvoOO, optimize=True)
    )
    dR.abb.VvvooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mnij,AcbmnK->AbcijK', X.ab.oooo[oa, ob, oa, ob], T.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('mNij,AcbmNK->AbcijK', X.ab.oooo[oa, Ob, oa, ob], T.abb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('Mnij,AcbMnK->AbcijK', X.ab.oooo[Oa, ob, oa, ob], T.abb.VvvOoO, optimize=True)
            - 1.0 * np.einsum('MNij,AcbMNK->AbcijK', X.ab.oooo[Oa, Ob, oa, ob], T.abb.VvvOOO, optimize=True)
    )
    dR.abb.VvvooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mNiK,AcbmjN->AbcijK', X.ab.oooo[oa, Ob, oa, Ob], T.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('MniK,AcbMnj->AbcijK', X.ab.oooo[Oa, ob, oa, Ob], T.abb.VvvOoo, optimize=True)
            - 1.0 * np.einsum('MNiK,AcbMjN->AbcijK', X.ab.oooo[Oa, Ob, oa, Ob], T.abb.VvvOoO, optimize=True)
    )
    dR.abb.VvvooO += (1.0 / 2.0) * (
            -0.5 * np.einsum('bcef,AfeijK->AbcijK', X.bb.vvvv[vb, vb, vb, vb], T.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('bceF,AFeijK->AbcijK', X.bb.vvvv[vb, vb, vb, Vb], T.abb.VVvooO, optimize=True)
            - 0.5 * np.einsum('bcEF,AFEijK->AbcijK', X.bb.vvvv[vb, vb, Vb, Vb], T.abb.VVVooO, optimize=True)
    )
    dR.abb.VvvooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('AbEf,EcfijK->AbcijK', X.ab.vvvv[Va, vb, Va, vb], T.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('AbeF,eFcijK->AbcijK', X.ab.vvvv[Va, vb, va, Vb], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('AbEF,EFcijK->AbcijK', X.ab.vvvv[Va, vb, Va, Vb], T.abb.VVvooO, optimize=True)
    )
    dR.abb.VvvooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('AmiE,EcbmjK->AbcijK', X.aa.voov[Va, oa, oa, Va], T.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('AMiE,EcbMjK->AbcijK', X.aa.voov[Va, Oa, oa, Va], T.abb.VvvOoO, optimize=True)
    )
    dR.abb.VvvooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('AmiE,EcbmjK->AbcijK', X.ab.voov[Va, ob, oa, Vb], T.bbb.VvvooO, optimize=True)
            + 1.0 * np.einsum('AMiE,EcbjMK->AbcijK', X.ab.voov[Va, Ob, oa, Vb], T.bbb.VvvoOO, optimize=True)
    )
    dR.abb.VvvooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mbej,AecimK->AbcijK', X.ab.ovvo[oa, vb, va, ob], T.aab.VvvooO, optimize=True)
            - 1.0 * np.einsum('mbEj,EAcimK->AbcijK', X.ab.ovvo[oa, vb, Va, ob], T.aab.VVvooO, optimize=True)
            + 1.0 * np.einsum('Mbej,AeciMK->AbcijK', X.ab.ovvo[Oa, vb, va, ob], T.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('MbEj,EAciMK->AbcijK', X.ab.ovvo[Oa, vb, Va, ob], T.aab.VVvoOO, optimize=True)
    )
    dR.abb.VvvooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('MbeK,AeciMj->AbcijK', X.ab.ovvo[Oa, vb, va, Ob], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('MbEK,EAciMj->AbcijK', X.ab.ovvo[Oa, vb, Va, Ob], T.aab.VVvoOo, optimize=True)
    )
    dR.abb.VvvooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('bmje,AceimK->AbcijK', X.bb.voov[vb, ob, ob, vb], T.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('bmjE,AEcimK->AbcijK', X.bb.voov[vb, ob, ob, Vb], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('bMje,AceiMK->AbcijK', X.bb.voov[vb, Ob, ob, vb], T.abb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('bMjE,AEciMK->AbcijK', X.bb.voov[vb, Ob, ob, Vb], T.abb.VVvoOO, optimize=True)
    )
    dR.abb.VvvooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('bMKe,AceijM->AbcijK', X.bb.voov[vb, Ob, Ob, vb], T.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('bMKE,AEcijM->AbcijK', X.bb.voov[vb, Ob, Ob, Vb], T.abb.VVvooO, optimize=True)
    )
    dR.abb.VvvooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mbie,AcemjK->AbcijK', X.ab.ovov[oa, vb, oa, vb], T.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('mbiE,AEcmjK->AbcijK', X.ab.ovov[oa, vb, oa, Vb], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('Mbie,AceMjK->AbcijK', X.ab.ovov[Oa, vb, oa, vb], T.abb.VvvOoO, optimize=True)
            - 1.0 * np.einsum('MbiE,AEcMjK->AbcijK', X.ab.ovov[Oa, vb, oa, Vb], T.abb.VVvOoO, optimize=True)
    )
    dR.abb.VvvooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('AmEj,EcbimK->AbcijK', X.ab.vovo[Va, ob, Va, ob], T.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('AMEj,EcbiMK->AbcijK', X.ab.vovo[Va, Ob, Va, ob], T.abb.VvvoOO, optimize=True)
    )
    dR.abb.VvvooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('AMEK,EcbijM->AbcijK', X.ab.vovo[Va, Ob, Va, Ob], T.abb.VvvooO, optimize=True)
    )
    dR.abb.VvvooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mi,AcbmjK->AbcijK', H.a.oo[oa, oa], R.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('Mi,AcbMjK->AbcijK', H.a.oo[Oa, oa], R.abb.VvvOoO, optimize=True)
    )
    dR.abb.VvvooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mj,AcbimK->AbcijK', H.b.oo[ob, ob], R.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('Mj,AcbiMK->AbcijK', H.b.oo[Ob, ob], R.abb.VvvoOO, optimize=True)
    )
    dR.abb.VvvooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('MK,AcbijM->AbcijK', H.b.oo[Ob, Ob], R.abb.VvvooO, optimize=True)
    )
    dR.abb.VvvooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('AE,EcbijK->AbcijK', H.a.vv[Va, Va], R.abb.VvvooO, optimize=True)
    )
    dR.abb.VvvooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('be,AceijK->AbcijK', H.b.vv[vb, vb], R.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('bE,AEcijK->AbcijK', H.b.vv[vb, Vb], R.abb.VVvooO, optimize=True)
    )
    dR.abb.VvvooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('MnjK,AcbinM->AbcijK', H.bb.oooo[Ob, ob, ob, Ob], R.abb.VvvooO, optimize=True)
            - 0.5 * np.einsum('MNjK,AcbiMN->AbcijK', H.bb.oooo[Ob, Ob, ob, Ob], R.abb.VvvoOO, optimize=True)
    )
    dR.abb.VvvooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mnij,AcbmnK->AbcijK', H.ab.oooo[oa, ob, oa, ob], R.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('mNij,AcbmNK->AbcijK', H.ab.oooo[oa, Ob, oa, ob], R.abb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('Mnij,AcbMnK->AbcijK', H.ab.oooo[Oa, ob, oa, ob], R.abb.VvvOoO, optimize=True)
            - 1.0 * np.einsum('MNij,AcbMNK->AbcijK', H.ab.oooo[Oa, Ob, oa, ob], R.abb.VvvOOO, optimize=True)
    )
    dR.abb.VvvooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mNiK,AcbmjN->AbcijK', H.ab.oooo[oa, Ob, oa, Ob], R.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('MniK,AcbMnj->AbcijK', H.ab.oooo[Oa, ob, oa, Ob], R.abb.VvvOoo, optimize=True)
            - 1.0 * np.einsum('MNiK,AcbMjN->AbcijK', H.ab.oooo[Oa, Ob, oa, Ob], R.abb.VvvOoO, optimize=True)
    )
    dR.abb.VvvooO += (1.0 / 2.0) * (
            -0.5 * np.einsum('bcef,AfeijK->AbcijK', H.bb.vvvv[vb, vb, vb, vb], R.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('bceF,AFeijK->AbcijK', H.bb.vvvv[vb, vb, vb, Vb], R.abb.VVvooO, optimize=True)
            - 0.5 * np.einsum('bcEF,AFEijK->AbcijK', H.bb.vvvv[vb, vb, Vb, Vb], R.abb.VVVooO, optimize=True)
    )
    dR.abb.VvvooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('AbEf,EcfijK->AbcijK', H.ab.vvvv[Va, vb, Va, vb], R.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('AbeF,eFcijK->AbcijK', H.ab.vvvv[Va, vb, va, Vb], R.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('AbEF,EFcijK->AbcijK', H.ab.vvvv[Va, vb, Va, Vb], R.abb.VVvooO, optimize=True)
    )
    dR.abb.VvvooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('AmiE,EcbmjK->AbcijK', H.aa.voov[Va, oa, oa, Va], R.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('AMiE,EcbMjK->AbcijK', H.aa.voov[Va, Oa, oa, Va], R.abb.VvvOoO, optimize=True)
    )
    dR.abb.VvvooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('AmiE,EcbmjK->AbcijK', H.ab.voov[Va, ob, oa, Vb], R.bbb.VvvooO, optimize=True)
            + 1.0 * np.einsum('AMiE,EcbjMK->AbcijK', H.ab.voov[Va, Ob, oa, Vb], R.bbb.VvvoOO, optimize=True)
    )
    dR.abb.VvvooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mbej,AecimK->AbcijK', H.ab.ovvo[oa, vb, va, ob], R.aab.VvvooO, optimize=True)
            - 1.0 * np.einsum('mbEj,EAcimK->AbcijK', H.ab.ovvo[oa, vb, Va, ob], R.aab.VVvooO, optimize=True)
            + 1.0 * np.einsum('Mbej,AeciMK->AbcijK', H.ab.ovvo[Oa, vb, va, ob], R.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('MbEj,EAciMK->AbcijK', H.ab.ovvo[Oa, vb, Va, ob], R.aab.VVvoOO, optimize=True)
    )
    dR.abb.VvvooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('MbeK,AeciMj->AbcijK', H.ab.ovvo[Oa, vb, va, Ob], R.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('MbEK,EAciMj->AbcijK', H.ab.ovvo[Oa, vb, Va, Ob], R.aab.VVvoOo, optimize=True)
    )
    dR.abb.VvvooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('bmje,AceimK->AbcijK', H.bb.voov[vb, ob, ob, vb], R.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('bmjE,AEcimK->AbcijK', H.bb.voov[vb, ob, ob, Vb], R.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('bMje,AceiMK->AbcijK', H.bb.voov[vb, Ob, ob, vb], R.abb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('bMjE,AEciMK->AbcijK', H.bb.voov[vb, Ob, ob, Vb], R.abb.VVvoOO, optimize=True)
    )
    dR.abb.VvvooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('bMKe,AceijM->AbcijK', H.bb.voov[vb, Ob, Ob, vb], R.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('bMKE,AEcijM->AbcijK', H.bb.voov[vb, Ob, Ob, Vb], R.abb.VVvooO, optimize=True)
    )
    dR.abb.VvvooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mbie,AcemjK->AbcijK', H.ab.ovov[oa, vb, oa, vb], R.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('mbiE,AEcmjK->AbcijK', H.ab.ovov[oa, vb, oa, Vb], R.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('Mbie,AceMjK->AbcijK', H.ab.ovov[Oa, vb, oa, vb], R.abb.VvvOoO, optimize=True)
            - 1.0 * np.einsum('MbiE,AEcMjK->AbcijK', H.ab.ovov[Oa, vb, oa, Vb], R.abb.VVvOoO, optimize=True)
    )
    dR.abb.VvvooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('AmEj,EcbimK->AbcijK', H.ab.vovo[Va, ob, Va, ob], R.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('AMEj,EcbiMK->AbcijK', H.ab.vovo[Va, Ob, Va, ob], R.abb.VvvoOO, optimize=True)
    )
    dR.abb.VvvooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('AMEK,EcbijM->AbcijK', H.ab.vovo[Va, Ob, Va, Ob], R.abb.VvvooO, optimize=True)
    )
    # of terms =  38

    dR.abb.VvvooO -= np.transpose(dR.abb.VvvooO, (0, 2, 1, 3, 4, 5))

    return dR

def update(R, omega, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    R.abb.VvvooO = eomcc_active_loops.update_r3c_100001(
        R.abb.VvvooO,
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
