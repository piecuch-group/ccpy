import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops

def build(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.abb.VvvOoO = (2.0 / 2.0) * (
            +1.0 * np.einsum('AbIe,ecjK->AbcIjK', X.ab.vvov[Va, vb, Oa, :], T.bb[:, vb, ob, Ob], optimize=True)
    )
    dR.abb.VvvOoO += (1.0 / 2.0) * (
            -1.0 * np.einsum('AmIj,bcmK->AbcIjK', X.ab.vooo[Va, :, Oa, ob], T.bb[vb, vb, :, Ob], optimize=True)
    )
    dR.abb.VvvOoO += (1.0 / 2.0) * (
            +1.0 * np.einsum('AmIK,bcmj->AbcIjK', X.ab.vooo[Va, :, Oa, Ob], T.bb[vb, vb, :, ob], optimize=True)
    )
    dR.abb.VvvOoO += (1.0 / 2.0) * (
            +1.0 * np.einsum('cbKe,AeIj->AbcIjK', X.bb.vvov[vb, vb, Ob, :], T.ab[Va, :, Oa, ob], optimize=True)
    )
    dR.abb.VvvOoO += (1.0 / 2.0) * (
            -1.0 * np.einsum('cbje,AeIK->AbcIjK', X.bb.vvov[vb, vb, ob, :], T.ab[Va, :, Oa, Ob], optimize=True)
    )
    dR.abb.VvvOoO += (2.0 / 2.0) * (
            -1.0 * np.einsum('cmKj,AbIm->AbcIjK', X.bb.vooo[vb, :, Ob, ob], T.ab[Va, vb, Oa, :], optimize=True)
    )
    dR.abb.VvvOoO += (2.0 / 2.0) * (
            +1.0 * np.einsum('Abej,ecIK->AbcIjK', X.ab.vvvo[Va, vb, :, ob], T.ab[:, vb, Oa, Ob], optimize=True)
    )
    dR.abb.VvvOoO += (2.0 / 2.0) * (
            -1.0 * np.einsum('AbeK,ecIj->AbcIjK', X.ab.vvvo[Va, vb, :, Ob], T.ab[:, vb, Oa, ob], optimize=True)
    )
    dR.abb.VvvOoO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mbIj,AcmK->AbcIjK', X.ab.ovoo[:, vb, Oa, ob], T.ab[Va, vb, :, Ob], optimize=True)
    )
    dR.abb.VvvOoO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mbIK,Acmj->AbcIjK', X.ab.ovoo[:, vb, Oa, Ob], T.ab[Va, vb, :, ob], optimize=True)
    )
    dR.abb.VvvOoO += (2.0 / 2.0) * (
            +1.0 * np.einsum('AbIe,ecjK->AbcIjK', H.ab.vvov[Va, vb, Oa, :], R.bb[:, vb, ob, Ob], optimize=True)
    )
    dR.abb.VvvOoO += (1.0 / 2.0) * (
            -1.0 * np.einsum('AmIj,bcmK->AbcIjK', H.ab.vooo[Va, :, Oa, ob], R.bb[vb, vb, :, Ob], optimize=True)
    )
    dR.abb.VvvOoO += (1.0 / 2.0) * (
            +1.0 * np.einsum('AmIK,bcmj->AbcIjK', H.ab.vooo[Va, :, Oa, Ob], R.bb[vb, vb, :, ob], optimize=True)
    )
    dR.abb.VvvOoO += (1.0 / 2.0) * (
            +1.0 * np.einsum('cbKe,AeIj->AbcIjK', H.bb.vvov[vb, vb, Ob, :], R.ab[Va, :, Oa, ob], optimize=True)
    )
    dR.abb.VvvOoO += (1.0 / 2.0) * (
            -1.0 * np.einsum('cbje,AeIK->AbcIjK', H.bb.vvov[vb, vb, ob, :], R.ab[Va, :, Oa, Ob], optimize=True)
    )
    dR.abb.VvvOoO += (2.0 / 2.0) * (
            -1.0 * np.einsum('cmKj,AbIm->AbcIjK', H.bb.vooo[vb, :, Ob, ob], R.ab[Va, vb, Oa, :], optimize=True)
    )
    dR.abb.VvvOoO += (2.0 / 2.0) * (
            +1.0 * np.einsum('Abej,ecIK->AbcIjK', H.ab.vvvo[Va, vb, :, ob], R.ab[:, vb, Oa, Ob], optimize=True)
    )
    dR.abb.VvvOoO += (2.0 / 2.0) * (
            -1.0 * np.einsum('AbeK,ecIj->AbcIjK', H.ab.vvvo[Va, vb, :, Ob], R.ab[:, vb, Oa, ob], optimize=True)
    )
    dR.abb.VvvOoO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mbIj,AcmK->AbcIjK', H.ab.ovoo[:, vb, Oa, ob], R.ab[Va, vb, :, Ob], optimize=True)
    )
    dR.abb.VvvOoO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mbIK,Acmj->AbcIjK', H.ab.ovoo[:, vb, Oa, Ob], R.ab[Va, vb, :, ob], optimize=True)
    )
    # of terms =  20
    dR.abb.VvvOoO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mI,AcbmjK->AbcIjK', X.a.oo[oa, Oa], T.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('MI,AcbMjK->AbcIjK', X.a.oo[Oa, Oa], T.abb.VvvOoO, optimize=True)
    )
    dR.abb.VvvOoO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mj,AcbImK->AbcIjK', X.b.oo[ob, ob], T.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('Mj,AcbIMK->AbcIjK', X.b.oo[Ob, ob], T.abb.VvvOOO, optimize=True)
    )
    dR.abb.VvvOoO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mK,AcbImj->AbcIjK', X.b.oo[ob, Ob], T.abb.VvvOoo, optimize=True)
            + 1.0 * np.einsum('MK,AcbIjM->AbcIjK', X.b.oo[Ob, Ob], T.abb.VvvOoO, optimize=True)
    )
    dR.abb.VvvOoO += (1.0 / 2.0) * (
            -1.0 * np.einsum('AE,EcbIjK->AbcIjK', X.a.vv[Va, Va], T.abb.VvvOoO, optimize=True)
    )
    dR.abb.VvvOoO += (2.0 / 2.0) * (
            -1.0 * np.einsum('be,AceIjK->AbcIjK', X.b.vv[vb, vb], T.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('bE,AEcIjK->AbcIjK', X.b.vv[vb, Vb], T.abb.VVvOoO, optimize=True)
    )
    dR.abb.VvvOoO += (1.0 / 2.0) * (
            -0.5 * np.einsum('mnjK,AcbImn->AbcIjK', X.bb.oooo[ob, ob, ob, Ob], T.abb.VvvOoo, optimize=True)
            + 1.0 * np.einsum('MnjK,AcbInM->AbcIjK', X.bb.oooo[Ob, ob, ob, Ob], T.abb.VvvOoO, optimize=True)
            - 0.5 * np.einsum('MNjK,AcbIMN->AbcIjK', X.bb.oooo[Ob, Ob, ob, Ob], T.abb.VvvOOO, optimize=True)
    )
    dR.abb.VvvOoO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mnIj,AcbmnK->AbcIjK', X.ab.oooo[oa, ob, Oa, ob], T.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('mNIj,AcbmNK->AbcIjK', X.ab.oooo[oa, Ob, Oa, ob], T.abb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('MnIj,AcbMnK->AbcIjK', X.ab.oooo[Oa, ob, Oa, ob], T.abb.VvvOoO, optimize=True)
            - 1.0 * np.einsum('MNIj,AcbMNK->AbcIjK', X.ab.oooo[Oa, Ob, Oa, ob], T.abb.VvvOOO, optimize=True)
    )
    dR.abb.VvvOoO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mNIK,AcbmjN->AbcIjK', X.ab.oooo[oa, Ob, Oa, Ob], T.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('MnIK,AcbMnj->AbcIjK', X.ab.oooo[Oa, ob, Oa, Ob], T.abb.VvvOoo, optimize=True)
            - 1.0 * np.einsum('MNIK,AcbMjN->AbcIjK', X.ab.oooo[Oa, Ob, Oa, Ob], T.abb.VvvOoO, optimize=True)
    )
    dR.abb.VvvOoO += (1.0 / 2.0) * (
            -0.5 * np.einsum('bcef,AfeIjK->AbcIjK', X.bb.vvvv[vb, vb, vb, vb], T.abb.VvvOoO, optimize=True)
            - 1.0 * np.einsum('bceF,AFeIjK->AbcIjK', X.bb.vvvv[vb, vb, vb, Vb], T.abb.VVvOoO, optimize=True)
            - 0.5 * np.einsum('bcEF,AFEIjK->AbcIjK', X.bb.vvvv[vb, vb, Vb, Vb], T.abb.VVVOoO, optimize=True)
    )
    dR.abb.VvvOoO += (2.0 / 2.0) * (
            -1.0 * np.einsum('AbEf,EcfIjK->AbcIjK', X.ab.vvvv[Va, vb, Va, vb], T.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('AbeF,eFcIjK->AbcIjK', X.ab.vvvv[Va, vb, va, Vb], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('AbEF,EFcIjK->AbcIjK', X.ab.vvvv[Va, vb, Va, Vb], T.abb.VVvOoO, optimize=True)
    )
    dR.abb.VvvOoO += (1.0 / 2.0) * (
            -1.0 * np.einsum('AmIE,EcbmjK->AbcIjK', X.aa.voov[Va, oa, Oa, Va], T.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('AMIE,EcbMjK->AbcIjK', X.aa.voov[Va, Oa, Oa, Va], T.abb.VvvOoO, optimize=True)
    )
    dR.abb.VvvOoO += (1.0 / 2.0) * (
            -1.0 * np.einsum('AmIE,EcbmjK->AbcIjK', X.ab.voov[Va, ob, Oa, Vb], T.bbb.VvvooO, optimize=True)
            + 1.0 * np.einsum('AMIE,EcbjMK->AbcIjK', X.ab.voov[Va, Ob, Oa, Vb], T.bbb.VvvoOO, optimize=True)
    )
    dR.abb.VvvOoO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mbej,AecmIK->AbcIjK', X.ab.ovvo[oa, vb, va, ob], T.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('mbEj,EAcmIK->AbcIjK', X.ab.ovvo[oa, vb, Va, ob], T.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('Mbej,AecIMK->AbcIjK', X.ab.ovvo[Oa, vb, va, ob], T.aab.VvvOOO, optimize=True)
            - 1.0 * np.einsum('MbEj,EAcIMK->AbcIjK', X.ab.ovvo[Oa, vb, Va, ob], T.aab.VVvOOO, optimize=True)
    )
    dR.abb.VvvOoO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mbeK,AecmIj->AbcIjK', X.ab.ovvo[oa, vb, va, Ob], T.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('mbEK,EAcmIj->AbcIjK', X.ab.ovvo[oa, vb, Va, Ob], T.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('MbeK,AecIMj->AbcIjK', X.ab.ovvo[Oa, vb, va, Ob], T.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('MbEK,EAcIMj->AbcIjK', X.ab.ovvo[Oa, vb, Va, Ob], T.aab.VVvOOo, optimize=True)
    )
    dR.abb.VvvOoO += (2.0 / 2.0) * (
            -1.0 * np.einsum('bmje,AceImK->AbcIjK', X.bb.voov[vb, ob, ob, vb], T.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('bmjE,AEcImK->AbcIjK', X.bb.voov[vb, ob, ob, Vb], T.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('bMje,AceIMK->AbcIjK', X.bb.voov[vb, Ob, ob, vb], T.abb.VvvOOO, optimize=True)
            + 1.0 * np.einsum('bMjE,AEcIMK->AbcIjK', X.bb.voov[vb, Ob, ob, Vb], T.abb.VVvOOO, optimize=True)
    )
    dR.abb.VvvOoO += (2.0 / 2.0) * (
            +1.0 * np.einsum('bmKe,AceImj->AbcIjK', X.bb.voov[vb, ob, Ob, vb], T.abb.VvvOoo, optimize=True)
            - 1.0 * np.einsum('bmKE,AEcImj->AbcIjK', X.bb.voov[vb, ob, Ob, Vb], T.abb.VVvOoo, optimize=True)
            - 1.0 * np.einsum('bMKe,AceIjM->AbcIjK', X.bb.voov[vb, Ob, Ob, vb], T.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('bMKE,AEcIjM->AbcIjK', X.bb.voov[vb, Ob, Ob, Vb], T.abb.VVvOoO, optimize=True)
    )
    dR.abb.VvvOoO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mbIe,AcemjK->AbcIjK', X.ab.ovov[oa, vb, Oa, vb], T.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('mbIE,AEcmjK->AbcIjK', X.ab.ovov[oa, vb, Oa, Vb], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('MbIe,AceMjK->AbcIjK', X.ab.ovov[Oa, vb, Oa, vb], T.abb.VvvOoO, optimize=True)
            - 1.0 * np.einsum('MbIE,AEcMjK->AbcIjK', X.ab.ovov[Oa, vb, Oa, Vb], T.abb.VVvOoO, optimize=True)
    )
    dR.abb.VvvOoO += (1.0 / 2.0) * (
            +1.0 * np.einsum('AmEj,EcbImK->AbcIjK', X.ab.vovo[Va, ob, Va, ob], T.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('AMEj,EcbIMK->AbcIjK', X.ab.vovo[Va, Ob, Va, ob], T.abb.VvvOOO, optimize=True)
    )
    dR.abb.VvvOoO += (1.0 / 2.0) * (
            -1.0 * np.einsum('AmEK,EcbImj->AbcIjK', X.ab.vovo[Va, ob, Va, Ob], T.abb.VvvOoo, optimize=True)
            + 1.0 * np.einsum('AMEK,EcbIjM->AbcIjK', X.ab.vovo[Va, Ob, Va, Ob], T.abb.VvvOoO, optimize=True)
    )
    dR.abb.VvvOoO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mI,AcbmjK->AbcIjK', H.a.oo[oa, Oa], R.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('MI,AcbMjK->AbcIjK', H.a.oo[Oa, Oa], R.abb.VvvOoO, optimize=True)
    )
    dR.abb.VvvOoO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mj,AcbImK->AbcIjK', H.b.oo[ob, ob], R.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('Mj,AcbIMK->AbcIjK', H.b.oo[Ob, ob], R.abb.VvvOOO, optimize=True)
    )
    dR.abb.VvvOoO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mK,AcbImj->AbcIjK', H.b.oo[ob, Ob], R.abb.VvvOoo, optimize=True)
            + 1.0 * np.einsum('MK,AcbIjM->AbcIjK', H.b.oo[Ob, Ob], R.abb.VvvOoO, optimize=True)
    )
    dR.abb.VvvOoO += (1.0 / 2.0) * (
            -1.0 * np.einsum('AE,EcbIjK->AbcIjK', H.a.vv[Va, Va], R.abb.VvvOoO, optimize=True)
    )
    dR.abb.VvvOoO += (2.0 / 2.0) * (
            -1.0 * np.einsum('be,AceIjK->AbcIjK', H.b.vv[vb, vb], R.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('bE,AEcIjK->AbcIjK', H.b.vv[vb, Vb], R.abb.VVvOoO, optimize=True)
    )
    dR.abb.VvvOoO += (1.0 / 2.0) * (
            -0.5 * np.einsum('mnjK,AcbImn->AbcIjK', H.bb.oooo[ob, ob, ob, Ob], R.abb.VvvOoo, optimize=True)
            + 1.0 * np.einsum('MnjK,AcbInM->AbcIjK', H.bb.oooo[Ob, ob, ob, Ob], R.abb.VvvOoO, optimize=True)
            - 0.5 * np.einsum('MNjK,AcbIMN->AbcIjK', H.bb.oooo[Ob, Ob, ob, Ob], R.abb.VvvOOO, optimize=True)
    )
    dR.abb.VvvOoO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mnIj,AcbmnK->AbcIjK', H.ab.oooo[oa, ob, Oa, ob], R.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('mNIj,AcbmNK->AbcIjK', H.ab.oooo[oa, Ob, Oa, ob], R.abb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('MnIj,AcbMnK->AbcIjK', H.ab.oooo[Oa, ob, Oa, ob], R.abb.VvvOoO, optimize=True)
            - 1.0 * np.einsum('MNIj,AcbMNK->AbcIjK', H.ab.oooo[Oa, Ob, Oa, ob], R.abb.VvvOOO, optimize=True)
    )
    dR.abb.VvvOoO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mNIK,AcbmjN->AbcIjK', H.ab.oooo[oa, Ob, Oa, Ob], R.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('MnIK,AcbMnj->AbcIjK', H.ab.oooo[Oa, ob, Oa, Ob], R.abb.VvvOoo, optimize=True)
            - 1.0 * np.einsum('MNIK,AcbMjN->AbcIjK', H.ab.oooo[Oa, Ob, Oa, Ob], R.abb.VvvOoO, optimize=True)
    )
    dR.abb.VvvOoO += (1.0 / 2.0) * (
            -0.5 * np.einsum('bcef,AfeIjK->AbcIjK', H.bb.vvvv[vb, vb, vb, vb], R.abb.VvvOoO, optimize=True)
            - 1.0 * np.einsum('bceF,AFeIjK->AbcIjK', H.bb.vvvv[vb, vb, vb, Vb], R.abb.VVvOoO, optimize=True)
            - 0.5 * np.einsum('bcEF,AFEIjK->AbcIjK', H.bb.vvvv[vb, vb, Vb, Vb], R.abb.VVVOoO, optimize=True)
    )
    dR.abb.VvvOoO += (2.0 / 2.0) * (
            -1.0 * np.einsum('AbEf,EcfIjK->AbcIjK', H.ab.vvvv[Va, vb, Va, vb], R.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('AbeF,eFcIjK->AbcIjK', H.ab.vvvv[Va, vb, va, Vb], R.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('AbEF,EFcIjK->AbcIjK', H.ab.vvvv[Va, vb, Va, Vb], R.abb.VVvOoO, optimize=True)
    )
    dR.abb.VvvOoO += (1.0 / 2.0) * (
            -1.0 * np.einsum('AmIE,EcbmjK->AbcIjK', H.aa.voov[Va, oa, Oa, Va], R.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('AMIE,EcbMjK->AbcIjK', H.aa.voov[Va, Oa, Oa, Va], R.abb.VvvOoO, optimize=True)
    )
    dR.abb.VvvOoO += (1.0 / 2.0) * (
            -1.0 * np.einsum('AmIE,EcbmjK->AbcIjK', H.ab.voov[Va, ob, Oa, Vb], R.bbb.VvvooO, optimize=True)
            + 1.0 * np.einsum('AMIE,EcbjMK->AbcIjK', H.ab.voov[Va, Ob, Oa, Vb], R.bbb.VvvoOO, optimize=True)
    )
    dR.abb.VvvOoO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mbej,AecmIK->AbcIjK', H.ab.ovvo[oa, vb, va, ob], R.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('mbEj,EAcmIK->AbcIjK', H.ab.ovvo[oa, vb, Va, ob], R.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('Mbej,AecIMK->AbcIjK', H.ab.ovvo[Oa, vb, va, ob], R.aab.VvvOOO, optimize=True)
            - 1.0 * np.einsum('MbEj,EAcIMK->AbcIjK', H.ab.ovvo[Oa, vb, Va, ob], R.aab.VVvOOO, optimize=True)
    )
    dR.abb.VvvOoO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mbeK,AecmIj->AbcIjK', H.ab.ovvo[oa, vb, va, Ob], R.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('mbEK,EAcmIj->AbcIjK', H.ab.ovvo[oa, vb, Va, Ob], R.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('MbeK,AecIMj->AbcIjK', H.ab.ovvo[Oa, vb, va, Ob], R.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('MbEK,EAcIMj->AbcIjK', H.ab.ovvo[Oa, vb, Va, Ob], R.aab.VVvOOo, optimize=True)
    )
    dR.abb.VvvOoO += (2.0 / 2.0) * (
            -1.0 * np.einsum('bmje,AceImK->AbcIjK', H.bb.voov[vb, ob, ob, vb], R.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('bmjE,AEcImK->AbcIjK', H.bb.voov[vb, ob, ob, Vb], R.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('bMje,AceIMK->AbcIjK', H.bb.voov[vb, Ob, ob, vb], R.abb.VvvOOO, optimize=True)
            + 1.0 * np.einsum('bMjE,AEcIMK->AbcIjK', H.bb.voov[vb, Ob, ob, Vb], R.abb.VVvOOO, optimize=True)
    )
    dR.abb.VvvOoO += (2.0 / 2.0) * (
            +1.0 * np.einsum('bmKe,AceImj->AbcIjK', H.bb.voov[vb, ob, Ob, vb], R.abb.VvvOoo, optimize=True)
            - 1.0 * np.einsum('bmKE,AEcImj->AbcIjK', H.bb.voov[vb, ob, Ob, Vb], R.abb.VVvOoo, optimize=True)
            - 1.0 * np.einsum('bMKe,AceIjM->AbcIjK', H.bb.voov[vb, Ob, Ob, vb], R.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('bMKE,AEcIjM->AbcIjK', H.bb.voov[vb, Ob, Ob, Vb], R.abb.VVvOoO, optimize=True)
    )
    dR.abb.VvvOoO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mbIe,AcemjK->AbcIjK', H.ab.ovov[oa, vb, Oa, vb], R.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('mbIE,AEcmjK->AbcIjK', H.ab.ovov[oa, vb, Oa, Vb], R.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('MbIe,AceMjK->AbcIjK', H.ab.ovov[Oa, vb, Oa, vb], R.abb.VvvOoO, optimize=True)
            - 1.0 * np.einsum('MbIE,AEcMjK->AbcIjK', H.ab.ovov[Oa, vb, Oa, Vb], R.abb.VVvOoO, optimize=True)
    )
    dR.abb.VvvOoO += (1.0 / 2.0) * (
            +1.0 * np.einsum('AmEj,EcbImK->AbcIjK', H.ab.vovo[Va, ob, Va, ob], R.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('AMEj,EcbIMK->AbcIjK', H.ab.vovo[Va, Ob, Va, ob], R.abb.VvvOOO, optimize=True)
    )
    dR.abb.VvvOoO += (1.0 / 2.0) * (
            -1.0 * np.einsum('AmEK,EcbImj->AbcIjK', H.ab.vovo[Va, ob, Va, Ob], R.abb.VvvOoo, optimize=True)
            + 1.0 * np.einsum('AMEK,EcbIjM->AbcIjK', H.ab.vovo[Va, Ob, Va, Ob], R.abb.VvvOoO, optimize=True)
    )
    # of terms =  38

    dR.abb.VvvOoO -= np.transpose(dR.abb.VvvOoO, (0, 2, 1, 3, 4, 5))

    return dR

def update(R, omega, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    R.abb.VvvOoO = eomcc_active_loops.update_r3c_100101(
        R.abb.VvvOoO,
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
