import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops

def build(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.abb.VvvOoo = (2.0 / 4.0) * (
            +1.0 * np.einsum('AbIe,ecjk->AbcIjk', X.ab.vvov[Va, vb, Oa, :], T.bb[:, vb, ob, ob], optimize=True)
    )
    dR.abb.VvvOoo += (2.0 / 4.0) * (
            -1.0 * np.einsum('AmIj,bcmk->AbcIjk', X.ab.vooo[Va, :, Oa, ob], T.bb[vb, vb, :, ob], optimize=True)
    )
    dR.abb.VvvOoo += (2.0 / 4.0) * (
            +1.0 * np.einsum('cbke,AeIj->AbcIjk', X.bb.vvov[vb, vb, ob, :], T.ab[Va, :, Oa, ob], optimize=True)
    )
    dR.abb.VvvOoo += (2.0 / 4.0) * (
            -1.0 * np.einsum('cmkj,AbIm->AbcIjk', X.bb.vooo[vb, :, ob, ob], T.ab[Va, vb, Oa, :], optimize=True)
    )
    dR.abb.VvvOoo += (4.0 / 4.0) * (
            +1.0 * np.einsum('Abej,ecIk->AbcIjk', X.ab.vvvo[Va, vb, :, ob], T.ab[:, vb, Oa, ob], optimize=True)
    )
    dR.abb.VvvOoo += (4.0 / 4.0) * (
            -1.0 * np.einsum('mbIj,Acmk->AbcIjk', X.ab.ovoo[:, vb, Oa, ob], T.ab[Va, vb, :, ob], optimize=True)
    )
    dR.abb.VvvOoo += (2.0 / 4.0) * (
            +1.0 * np.einsum('AbIe,ecjk->AbcIjk', H.ab.vvov[Va, vb, Oa, :], R.bb[:, vb, ob, ob], optimize=True)
    )
    dR.abb.VvvOoo += (2.0 / 4.0) * (
            -1.0 * np.einsum('AmIj,bcmk->AbcIjk', H.ab.vooo[Va, :, Oa, ob], R.bb[vb, vb, :, ob], optimize=True)
    )
    dR.abb.VvvOoo += (2.0 / 4.0) * (
            +1.0 * np.einsum('cbke,AeIj->AbcIjk', H.bb.vvov[vb, vb, ob, :], R.ab[Va, :, Oa, ob], optimize=True)
    )
    dR.abb.VvvOoo += (2.0 / 4.0) * (
            -1.0 * np.einsum('cmkj,AbIm->AbcIjk', H.bb.vooo[vb, :, ob, ob], R.ab[Va, vb, Oa, :], optimize=True)
    )
    dR.abb.VvvOoo += (4.0 / 4.0) * (
            +1.0 * np.einsum('Abej,ecIk->AbcIjk', H.ab.vvvo[Va, vb, :, ob], R.ab[:, vb, Oa, ob], optimize=True)
    )
    dR.abb.VvvOoo += (4.0 / 4.0) * (
            -1.0 * np.einsum('mbIj,Acmk->AbcIjk', H.ab.ovoo[:, vb, Oa, ob], R.ab[Va, vb, :, ob], optimize=True)
    )
    # of terms =  12
    dR.abb.VvvOoo += (1.0 / 4.0) * (
            +1.0 * np.einsum('MI,AcbMjk->AbcIjk', X.a.oo[Oa, Oa], T.abb.VvvOoo, optimize=True)
    )
    dR.abb.VvvOoo += (2.0 / 4.0) * (
            +1.0 * np.einsum('mj,AcbImk->AbcIjk', X.b.oo[ob, ob], T.abb.VvvOoo, optimize=True)
            - 1.0 * np.einsum('Mj,AcbIkM->AbcIjk', X.b.oo[Ob, ob], T.abb.VvvOoO, optimize=True)
    )
    dR.abb.VvvOoo += (1.0 / 4.0) * (
            -1.0 * np.einsum('AE,EcbIjk->AbcIjk', X.a.vv[Va, Va], T.abb.VvvOoo, optimize=True)
    )
    dR.abb.VvvOoo += (2.0 / 4.0) * (
            -1.0 * np.einsum('be,AceIjk->AbcIjk', X.b.vv[vb, vb], T.abb.VvvOoo, optimize=True)
            + 1.0 * np.einsum('bE,AEcIjk->AbcIjk', X.b.vv[vb, Vb], T.abb.VVvOoo, optimize=True)
    )
    dR.abb.VvvOoo += (1.0 / 4.0) * (
            -0.5 * np.einsum('mnjk,AcbImn->AbcIjk', X.bb.oooo[ob, ob, ob, ob], T.abb.VvvOoo, optimize=True)
            + 1.0 * np.einsum('Mnjk,AcbInM->AbcIjk', X.bb.oooo[Ob, ob, ob, ob], T.abb.VvvOoO, optimize=True)
            - 0.5 * np.einsum('MNjk,AcbIMN->AbcIjk', X.bb.oooo[Ob, Ob, ob, ob], T.abb.VvvOOO, optimize=True)
    )
    dR.abb.VvvOoo += (2.0 / 4.0) * (
            +1.0 * np.einsum('mNIj,AcbmkN->AbcIjk', X.ab.oooo[oa, Ob, Oa, ob], T.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('MnIj,AcbMnk->AbcIjk', X.ab.oooo[Oa, ob, Oa, ob], T.abb.VvvOoo, optimize=True)
            + 1.0 * np.einsum('MNIj,AcbMkN->AbcIjk', X.ab.oooo[Oa, Ob, Oa, ob], T.abb.VvvOoO, optimize=True)
    )
    dR.abb.VvvOoo += (1.0 / 4.0) * (
            -0.5 * np.einsum('bcef,AfeIjk->AbcIjk', X.bb.vvvv[vb, vb, vb, vb], T.abb.VvvOoo, optimize=True)
            + 1.0 * np.einsum('bcEf,AEfIjk->AbcIjk', X.bb.vvvv[vb, vb, Vb, vb], T.abb.VVvOoo, optimize=True)
            - 0.5 * np.einsum('bcEF,AFEIjk->AbcIjk', X.bb.vvvv[vb, vb, Vb, Vb], T.abb.VVVOoo, optimize=True)
    )
    dR.abb.VvvOoo += (2.0 / 4.0) * (
            +1.0 * np.einsum('AbeF,eFcIjk->AbcIjk', X.ab.vvvv[Va, vb, va, Vb], T.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('AbEf,EcfIjk->AbcIjk', X.ab.vvvv[Va, vb, Va, vb], T.abb.VvvOoo, optimize=True)
            + 1.0 * np.einsum('AbEF,EFcIjk->AbcIjk', X.ab.vvvv[Va, vb, Va, Vb], T.abb.VVvOoo, optimize=True)
    )
    dR.abb.VvvOoo += (1.0 / 4.0) * (
            -1.0 * np.einsum('AMIE,EcbMjk->AbcIjk', X.aa.voov[Va, Oa, Oa, Va], T.abb.VvvOoo, optimize=True)
    )
    dR.abb.VvvOoo += (1.0 / 4.0) * (
            -1.0 * np.einsum('AMIE,EcbjkM->AbcIjk', X.ab.voov[Va, Ob, Oa, Vb], T.bbb.VvvooO, optimize=True)
    )
    dR.abb.VvvOoo += (4.0 / 4.0) * (
            -1.0 * np.einsum('mbej,AecmIk->AbcIjk', X.ab.ovvo[oa, vb, va, ob], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('mbEj,EAcmIk->AbcIjk', X.ab.ovvo[oa, vb, Va, ob], T.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('Mbej,AecIMk->AbcIjk', X.ab.ovvo[Oa, vb, va, ob], T.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('MbEj,EAcIMk->AbcIjk', X.ab.ovvo[Oa, vb, Va, ob], T.aab.VVvOOo, optimize=True)
    )
    dR.abb.VvvOoo += (4.0 / 4.0) * (
            -1.0 * np.einsum('bmje,AceImk->AbcIjk', X.bb.voov[vb, ob, ob, vb], T.abb.VvvOoo, optimize=True)
            + 1.0 * np.einsum('bmjE,AEcImk->AbcIjk', X.bb.voov[vb, ob, ob, Vb], T.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('bMje,AceIkM->AbcIjk', X.bb.voov[vb, Ob, ob, vb], T.abb.VvvOoO, optimize=True)
            - 1.0 * np.einsum('bMjE,AEcIkM->AbcIjk', X.bb.voov[vb, Ob, ob, Vb], T.abb.VVvOoO, optimize=True)
    )
    dR.abb.VvvOoo += (2.0 / 4.0) * (
            +1.0 * np.einsum('MbIe,AceMjk->AbcIjk', X.ab.ovov[Oa, vb, Oa, vb], T.abb.VvvOoo, optimize=True)
            - 1.0 * np.einsum('MbIE,AEcMjk->AbcIjk', X.ab.ovov[Oa, vb, Oa, Vb], T.abb.VVvOoo, optimize=True)
    )
    dR.abb.VvvOoo += (2.0 / 4.0) * (
            +1.0 * np.einsum('AmEj,EcbImk->AbcIjk', X.ab.vovo[Va, ob, Va, ob], T.abb.VvvOoo, optimize=True)
            - 1.0 * np.einsum('AMEj,EcbIkM->AbcIjk', X.ab.vovo[Va, Ob, Va, ob], T.abb.VvvOoO, optimize=True)
    )
    dR.abb.VvvOoo += (1.0 / 4.0) * (
            +1.0 * np.einsum('MI,AcbMjk->AbcIjk', H.a.oo[Oa, Oa], R.abb.VvvOoo, optimize=True)
    )
    dR.abb.VvvOoo += (2.0 / 4.0) * (
            +1.0 * np.einsum('mj,AcbImk->AbcIjk', H.b.oo[ob, ob], R.abb.VvvOoo, optimize=True)
            - 1.0 * np.einsum('Mj,AcbIkM->AbcIjk', H.b.oo[Ob, ob], R.abb.VvvOoO, optimize=True)
    )
    dR.abb.VvvOoo += (1.0 / 4.0) * (
            -1.0 * np.einsum('AE,EcbIjk->AbcIjk', H.a.vv[Va, Va], R.abb.VvvOoo, optimize=True)
    )
    dR.abb.VvvOoo += (2.0 / 4.0) * (
            -1.0 * np.einsum('be,AceIjk->AbcIjk', H.b.vv[vb, vb], R.abb.VvvOoo, optimize=True)
            + 1.0 * np.einsum('bE,AEcIjk->AbcIjk', H.b.vv[vb, Vb], R.abb.VVvOoo, optimize=True)
    )
    dR.abb.VvvOoo += (1.0 / 4.0) * (
            -0.5 * np.einsum('mnjk,AcbImn->AbcIjk', H.bb.oooo[ob, ob, ob, ob], R.abb.VvvOoo, optimize=True)
            + 1.0 * np.einsum('Mnjk,AcbInM->AbcIjk', H.bb.oooo[Ob, ob, ob, ob], R.abb.VvvOoO, optimize=True)
            - 0.5 * np.einsum('MNjk,AcbIMN->AbcIjk', H.bb.oooo[Ob, Ob, ob, ob], R.abb.VvvOOO, optimize=True)
    )
    dR.abb.VvvOoo += (2.0 / 4.0) * (
            +1.0 * np.einsum('mNIj,AcbmkN->AbcIjk', H.ab.oooo[oa, Ob, Oa, ob], R.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('MnIj,AcbMnk->AbcIjk', H.ab.oooo[Oa, ob, Oa, ob], R.abb.VvvOoo, optimize=True)
            + 1.0 * np.einsum('MNIj,AcbMkN->AbcIjk', H.ab.oooo[Oa, Ob, Oa, ob], R.abb.VvvOoO, optimize=True)
    )
    dR.abb.VvvOoo += (1.0 / 4.0) * (
            -0.5 * np.einsum('bcef,AfeIjk->AbcIjk', H.bb.vvvv[vb, vb, vb, vb], R.abb.VvvOoo, optimize=True)
            + 1.0 * np.einsum('bcEf,AEfIjk->AbcIjk', H.bb.vvvv[vb, vb, Vb, vb], R.abb.VVvOoo, optimize=True)
            - 0.5 * np.einsum('bcEF,AFEIjk->AbcIjk', H.bb.vvvv[vb, vb, Vb, Vb], R.abb.VVVOoo, optimize=True)
    )
    dR.abb.VvvOoo += (2.0 / 4.0) * (
            +1.0 * np.einsum('AbeF,eFcIjk->AbcIjk', H.ab.vvvv[Va, vb, va, Vb], R.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('AbEf,EcfIjk->AbcIjk', H.ab.vvvv[Va, vb, Va, vb], R.abb.VvvOoo, optimize=True)
            + 1.0 * np.einsum('AbEF,EFcIjk->AbcIjk', H.ab.vvvv[Va, vb, Va, Vb], R.abb.VVvOoo, optimize=True)
    )
    dR.abb.VvvOoo += (1.0 / 4.0) * (
            -1.0 * np.einsum('AMIE,EcbMjk->AbcIjk', H.aa.voov[Va, Oa, Oa, Va], R.abb.VvvOoo, optimize=True)
    )
    dR.abb.VvvOoo += (1.0 / 4.0) * (
            -1.0 * np.einsum('AMIE,EcbjkM->AbcIjk', H.ab.voov[Va, Ob, Oa, Vb], R.bbb.VvvooO, optimize=True)
    )
    dR.abb.VvvOoo += (4.0 / 4.0) * (
            -1.0 * np.einsum('mbej,AecmIk->AbcIjk', H.ab.ovvo[oa, vb, va, ob], R.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('mbEj,EAcmIk->AbcIjk', H.ab.ovvo[oa, vb, Va, ob], R.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('Mbej,AecIMk->AbcIjk', H.ab.ovvo[Oa, vb, va, ob], R.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('MbEj,EAcIMk->AbcIjk', H.ab.ovvo[Oa, vb, Va, ob], R.aab.VVvOOo, optimize=True)
    )
    dR.abb.VvvOoo += (4.0 / 4.0) * (
            -1.0 * np.einsum('bmje,AceImk->AbcIjk', H.bb.voov[vb, ob, ob, vb], R.abb.VvvOoo, optimize=True)
            + 1.0 * np.einsum('bmjE,AEcImk->AbcIjk', H.bb.voov[vb, ob, ob, Vb], R.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('bMje,AceIkM->AbcIjk', H.bb.voov[vb, Ob, ob, vb], R.abb.VvvOoO, optimize=True)
            - 1.0 * np.einsum('bMjE,AEcIkM->AbcIjk', H.bb.voov[vb, Ob, ob, Vb], R.abb.VVvOoO, optimize=True)
    )
    dR.abb.VvvOoo += (2.0 / 4.0) * (
            +1.0 * np.einsum('MbIe,AceMjk->AbcIjk', H.ab.ovov[Oa, vb, Oa, vb], R.abb.VvvOoo, optimize=True)
            - 1.0 * np.einsum('MbIE,AEcMjk->AbcIjk', H.ab.ovov[Oa, vb, Oa, Vb], R.abb.VVvOoo, optimize=True)
    )
    dR.abb.VvvOoo += (2.0 / 4.0) * (
            +1.0 * np.einsum('AmEj,EcbImk->AbcIjk', H.ab.vovo[Va, ob, Va, ob], R.abb.VvvOoo, optimize=True)
            - 1.0 * np.einsum('AMEj,EcbIkM->AbcIjk', H.ab.vovo[Va, Ob, Va, ob], R.abb.VvvOoO, optimize=True)
    )
    # of terms =  28

    dR.abb.VvvOoo -= np.transpose(dR.abb.VvvOoo, (0, 2, 1, 3, 4, 5))
    dR.abb.VvvOoo -= np.transpose(dR.abb.VvvOoo, (0, 1, 2, 3, 5, 4))

    return dR

def update(R, omega, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    R.abb.VvvOoo = eomcc_active_loops.update_r3c_100100(
        R.abb.VvvOoo,
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
