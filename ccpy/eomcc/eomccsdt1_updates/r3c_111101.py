import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops

def build(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.abb.VVVOoO = (2.0 / 2.0) * (
            +1.0 * np.einsum('ABIe,eCjK->ABCIjK', X.ab.vvov[Va, Vb, Oa, :], T.bb[:, Vb, ob, Ob], optimize=True)
    )
    dR.abb.VVVOoO += (1.0 / 2.0) * (
            -1.0 * np.einsum('AmIj,BCmK->ABCIjK', X.ab.vooo[Va, :, Oa, ob], T.bb[Vb, Vb, :, Ob], optimize=True)
    )
    dR.abb.VVVOoO += (1.0 / 2.0) * (
            +1.0 * np.einsum('AmIK,BCmj->ABCIjK', X.ab.vooo[Va, :, Oa, Ob], T.bb[Vb, Vb, :, ob], optimize=True)
    )
    dR.abb.VVVOoO += (1.0 / 2.0) * (
            +1.0 * np.einsum('CBKe,AeIj->ABCIjK', X.bb.vvov[Vb, Vb, Ob, :], T.ab[Va, :, Oa, ob], optimize=True)
    )
    dR.abb.VVVOoO += (1.0 / 2.0) * (
            -1.0 * np.einsum('CBje,AeIK->ABCIjK', X.bb.vvov[Vb, Vb, ob, :], T.ab[Va, :, Oa, Ob], optimize=True)
    )
    dR.abb.VVVOoO += (2.0 / 2.0) * (
            -1.0 * np.einsum('CmKj,ABIm->ABCIjK', X.bb.vooo[Vb, :, Ob, ob], T.ab[Va, Vb, Oa, :], optimize=True)
    )
    dR.abb.VVVOoO += (2.0 / 2.0) * (
            +1.0 * np.einsum('ABej,eCIK->ABCIjK', X.ab.vvvo[Va, Vb, :, ob], T.ab[:, Vb, Oa, Ob], optimize=True)
    )
    dR.abb.VVVOoO += (2.0 / 2.0) * (
            -1.0 * np.einsum('ABeK,eCIj->ABCIjK', X.ab.vvvo[Va, Vb, :, Ob], T.ab[:, Vb, Oa, ob], optimize=True)
    )
    dR.abb.VVVOoO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mBIj,ACmK->ABCIjK', X.ab.ovoo[:, Vb, Oa, ob], T.ab[Va, Vb, :, Ob], optimize=True)
    )
    dR.abb.VVVOoO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mBIK,ACmj->ABCIjK', X.ab.ovoo[:, Vb, Oa, Ob], T.ab[Va, Vb, :, ob], optimize=True)
    )
    dR.abb.VVVOoO += (2.0 / 2.0) * (
            +1.0 * np.einsum('ABIe,eCjK->ABCIjK', H.ab.vvov[Va, Vb, Oa, :], R.bb[:, Vb, ob, Ob], optimize=True)
    )
    dR.abb.VVVOoO += (1.0 / 2.0) * (
            -1.0 * np.einsum('AmIj,BCmK->ABCIjK', H.ab.vooo[Va, :, Oa, ob], R.bb[Vb, Vb, :, Ob], optimize=True)
    )
    dR.abb.VVVOoO += (1.0 / 2.0) * (
            +1.0 * np.einsum('AmIK,BCmj->ABCIjK', H.ab.vooo[Va, :, Oa, Ob], R.bb[Vb, Vb, :, ob], optimize=True)
    )
    dR.abb.VVVOoO += (1.0 / 2.0) * (
            +1.0 * np.einsum('CBKe,AeIj->ABCIjK', H.bb.vvov[Vb, Vb, Ob, :], R.ab[Va, :, Oa, ob], optimize=True)
    )
    dR.abb.VVVOoO += (1.0 / 2.0) * (
            -1.0 * np.einsum('CBje,AeIK->ABCIjK', H.bb.vvov[Vb, Vb, ob, :], R.ab[Va, :, Oa, Ob], optimize=True)
    )
    dR.abb.VVVOoO += (2.0 / 2.0) * (
            -1.0 * np.einsum('CmKj,ABIm->ABCIjK', H.bb.vooo[Vb, :, Ob, ob], R.ab[Va, Vb, Oa, :], optimize=True)
    )
    dR.abb.VVVOoO += (2.0 / 2.0) * (
            +1.0 * np.einsum('ABej,eCIK->ABCIjK', H.ab.vvvo[Va, Vb, :, ob], R.ab[:, Vb, Oa, Ob], optimize=True)
    )
    dR.abb.VVVOoO += (2.0 / 2.0) * (
            -1.0 * np.einsum('ABeK,eCIj->ABCIjK', H.ab.vvvo[Va, Vb, :, Ob], R.ab[:, Vb, Oa, ob], optimize=True)
    )
    dR.abb.VVVOoO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mBIj,ACmK->ABCIjK', H.ab.ovoo[:, Vb, Oa, ob], R.ab[Va, Vb, :, Ob], optimize=True)
    )
    dR.abb.VVVOoO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mBIK,ACmj->ABCIjK', H.ab.ovoo[:, Vb, Oa, Ob], R.ab[Va, Vb, :, ob], optimize=True)
    )

    dR.abb.VVVOoO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mI,ACBmjK->ABCIjK', X.a.oo[oa, Oa], T.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('MI,ACBMjK->ABCIjK', X.a.oo[Oa, Oa], T.abb.VVVOoO, optimize=True)
    )
    dR.abb.VVVOoO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mj,ACBImK->ABCIjK', X.b.oo[ob, ob], T.abb.VVVOoO, optimize=True)
            + 1.0 * np.einsum('Mj,ACBIMK->ABCIjK', X.b.oo[Ob, ob], T.abb.VVVOOO, optimize=True)
    )
    dR.abb.VVVOoO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mK,ACBImj->ABCIjK', X.b.oo[ob, Ob], T.abb.VVVOoo, optimize=True)
            + 1.0 * np.einsum('MK,ACBIjM->ABCIjK', X.b.oo[Ob, Ob], T.abb.VVVOoO, optimize=True)
    )
    dR.abb.VVVOoO += (1.0 / 2.0) * (
            -1.0 * np.einsum('Ae,eCBIjK->ABCIjK', X.a.vv[Va, va], T.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('AE,ECBIjK->ABCIjK', X.a.vv[Va, Va], T.abb.VVVOoO, optimize=True)
    )
    dR.abb.VVVOoO += (2.0 / 2.0) * (
            -1.0 * np.einsum('Be,ACeIjK->ABCIjK', X.b.vv[Vb, vb], T.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('BE,ACEIjK->ABCIjK', X.b.vv[Vb, Vb], T.abb.VVVOoO, optimize=True)
    )
    dR.abb.VVVOoO += (1.0 / 2.0) * (
            -0.5 * np.einsum('mnjK,ACBImn->ABCIjK', X.bb.oooo[ob, ob, ob, Ob], T.abb.VVVOoo, optimize=True)
            - 1.0 * np.einsum('mNjK,ACBImN->ABCIjK', X.bb.oooo[ob, Ob, ob, Ob], T.abb.VVVOoO, optimize=True)
            - 0.5 * np.einsum('MNjK,ACBIMN->ABCIjK', X.bb.oooo[Ob, Ob, ob, Ob], T.abb.VVVOOO, optimize=True)
    )
    dR.abb.VVVOoO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mnIj,ACBmnK->ABCIjK', X.ab.oooo[oa, ob, Oa, ob], T.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('MnIj,ACBMnK->ABCIjK', X.ab.oooo[Oa, ob, Oa, ob], T.abb.VVVOoO, optimize=True)
            - 1.0 * np.einsum('mNIj,ACBmNK->ABCIjK', X.ab.oooo[oa, Ob, Oa, ob], T.abb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('MNIj,ACBMNK->ABCIjK', X.ab.oooo[Oa, Ob, Oa, ob], T.abb.VVVOOO, optimize=True)
    )
    dR.abb.VVVOoO += (1.0 / 2.0) * (
            +1.0 * np.einsum('MnIK,ACBMnj->ABCIjK', X.ab.oooo[Oa, ob, Oa, Ob], T.abb.VVVOoo, optimize=True)
            - 1.0 * np.einsum('mNIK,ACBmjN->ABCIjK', X.ab.oooo[oa, Ob, Oa, Ob], T.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('MNIK,ACBMjN->ABCIjK', X.ab.oooo[Oa, Ob, Oa, Ob], T.abb.VVVOoO, optimize=True)
    )
    dR.abb.VVVOoO += (1.0 / 2.0) * (
            -0.5 * np.einsum('BCef,AfeIjK->ABCIjK', X.bb.vvvv[Vb, Vb, vb, vb], T.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('BCEf,AEfIjK->ABCIjK', X.bb.vvvv[Vb, Vb, Vb, vb], T.abb.VVvOoO, optimize=True)
            - 0.5 * np.einsum('BCEF,AFEIjK->ABCIjK', X.bb.vvvv[Vb, Vb, Vb, Vb], T.abb.VVVOoO, optimize=True)
    )
    dR.abb.VVVOoO += (2.0 / 2.0) * (
            -1.0 * np.einsum('ABef,eCfIjK->ABCIjK', X.ab.vvvv[Va, Vb, va, vb], T.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('ABeF,eCFIjK->ABCIjK', X.ab.vvvv[Va, Vb, va, Vb], T.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('ABEf,ECfIjK->ABCIjK', X.ab.vvvv[Va, Vb, Va, vb], T.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('ABEF,ECFIjK->ABCIjK', X.ab.vvvv[Va, Vb, Va, Vb], T.abb.VVVOoO, optimize=True)
    )
    dR.abb.VVVOoO += (1.0 / 2.0) * (
            -1.0 * np.einsum('AmIe,eCBmjK->ABCIjK', X.aa.voov[Va, oa, Oa, va], T.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('AMIe,eCBMjK->ABCIjK', X.aa.voov[Va, Oa, Oa, va], T.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('AmIE,ECBmjK->ABCIjK', X.aa.voov[Va, oa, Oa, Va], T.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('AMIE,ECBMjK->ABCIjK', X.aa.voov[Va, Oa, Oa, Va], T.abb.VVVOoO, optimize=True)
    )
    dR.abb.VVVOoO += (1.0 / 2.0) * (
            -1.0 * np.einsum('AmIe,CBemjK->ABCIjK', X.ab.voov[Va, ob, Oa, vb], T.bbb.VVvooO, optimize=True)
            + 1.0 * np.einsum('AMIe,CBejMK->ABCIjK', X.ab.voov[Va, Ob, Oa, vb], T.bbb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('AmIE,CBEmjK->ABCIjK', X.ab.voov[Va, ob, Oa, Vb], T.bbb.VVVooO, optimize=True)
            + 1.0 * np.einsum('AMIE,CBEjMK->ABCIjK', X.ab.voov[Va, Ob, Oa, Vb], T.bbb.VVVoOO, optimize=True)
    )
    dR.abb.VVVOoO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mBej,AeCmIK->ABCIjK', X.ab.ovvo[oa, Vb, va, ob], T.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('MBej,AeCIMK->ABCIjK', X.ab.ovvo[Oa, Vb, va, ob], T.aab.VvVOOO, optimize=True)
            + 1.0 * np.einsum('mBEj,EACmIK->ABCIjK', X.ab.ovvo[oa, Vb, Va, ob], T.aab.VVVoOO, optimize=True)
            - 1.0 * np.einsum('MBEj,EACIMK->ABCIjK', X.ab.ovvo[Oa, Vb, Va, ob], T.aab.VVVOOO, optimize=True)
    )
    dR.abb.VVVOoO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mBeK,AeCmIj->ABCIjK', X.ab.ovvo[oa, Vb, va, Ob], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('MBeK,AeCIMj->ABCIjK', X.ab.ovvo[Oa, Vb, va, Ob], T.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('mBEK,EACmIj->ABCIjK', X.ab.ovvo[oa, Vb, Va, Ob], T.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('MBEK,EACIMj->ABCIjK', X.ab.ovvo[Oa, Vb, Va, Ob], T.aab.VVVOOo, optimize=True)
    )
    dR.abb.VVVOoO += (2.0 / 2.0) * (
            -1.0 * np.einsum('Bmje,ACeImK->ABCIjK', X.bb.voov[Vb, ob, ob, vb], T.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('BMje,ACeIMK->ABCIjK', X.bb.voov[Vb, Ob, ob, vb], T.abb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('BmjE,ACEImK->ABCIjK', X.bb.voov[Vb, ob, ob, Vb], T.abb.VVVOoO, optimize=True)
            - 1.0 * np.einsum('BMjE,ACEIMK->ABCIjK', X.bb.voov[Vb, Ob, ob, Vb], T.abb.VVVOOO, optimize=True)
    )
    dR.abb.VVVOoO += (2.0 / 2.0) * (
            +1.0 * np.einsum('BmKe,ACeImj->ABCIjK', X.bb.voov[Vb, ob, Ob, vb], T.abb.VVvOoo, optimize=True)
            - 1.0 * np.einsum('BMKe,ACeIjM->ABCIjK', X.bb.voov[Vb, Ob, Ob, vb], T.abb.VVvOoO, optimize=True)
            + 1.0 * np.einsum('BmKE,ACEImj->ABCIjK', X.bb.voov[Vb, ob, Ob, Vb], T.abb.VVVOoo, optimize=True)
            - 1.0 * np.einsum('BMKE,ACEIjM->ABCIjK', X.bb.voov[Vb, Ob, Ob, Vb], T.abb.VVVOoO, optimize=True)
    )
    dR.abb.VVVOoO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mBIe,ACemjK->ABCIjK', X.ab.ovov[oa, Vb, Oa, vb], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('MBIe,ACeMjK->ABCIjK', X.ab.ovov[Oa, Vb, Oa, vb], T.abb.VVvOoO, optimize=True)
            + 1.0 * np.einsum('mBIE,ACEmjK->ABCIjK', X.ab.ovov[oa, Vb, Oa, Vb], T.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('MBIE,ACEMjK->ABCIjK', X.ab.ovov[Oa, Vb, Oa, Vb], T.abb.VVVOoO, optimize=True)
    )
    dR.abb.VVVOoO += (1.0 / 2.0) * (
            +1.0 * np.einsum('Amej,eCBImK->ABCIjK', X.ab.vovo[Va, ob, va, ob], T.abb.vVVOoO, optimize=True)
            + 1.0 * np.einsum('AMej,eCBIMK->ABCIjK', X.ab.vovo[Va, Ob, va, ob], T.abb.vVVOOO, optimize=True)
            + 1.0 * np.einsum('AmEj,ECBImK->ABCIjK', X.ab.vovo[Va, ob, Va, ob], T.abb.VVVOoO, optimize=True)
            + 1.0 * np.einsum('AMEj,ECBIMK->ABCIjK', X.ab.vovo[Va, Ob, Va, ob], T.abb.VVVOOO, optimize=True)
    )
    dR.abb.VVVOoO += (1.0 / 2.0) * (
            -1.0 * np.einsum('AmeK,eCBImj->ABCIjK', X.ab.vovo[Va, ob, va, Ob], T.abb.vVVOoo, optimize=True)
            + 1.0 * np.einsum('AMeK,eCBIjM->ABCIjK', X.ab.vovo[Va, Ob, va, Ob], T.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('AmEK,ECBImj->ABCIjK', X.ab.vovo[Va, ob, Va, Ob], T.abb.VVVOoo, optimize=True)
            + 1.0 * np.einsum('AMEK,ECBIjM->ABCIjK', X.ab.vovo[Va, Ob, Va, Ob], T.abb.VVVOoO, optimize=True)
    )
    dR.abb.VVVOoO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mI,ACBmjK->ABCIjK', H.a.oo[oa, Oa], R.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('MI,ACBMjK->ABCIjK', H.a.oo[Oa, Oa], R.abb.VVVOoO, optimize=True)
    )
    dR.abb.VVVOoO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mj,ACBImK->ABCIjK', H.b.oo[ob, ob], R.abb.VVVOoO, optimize=True)
            + 1.0 * np.einsum('Mj,ACBIMK->ABCIjK', H.b.oo[Ob, ob], R.abb.VVVOOO, optimize=True)
    )
    dR.abb.VVVOoO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mK,ACBImj->ABCIjK', H.b.oo[ob, Ob], R.abb.VVVOoo, optimize=True)
            + 1.0 * np.einsum('MK,ACBIjM->ABCIjK', H.b.oo[Ob, Ob], R.abb.VVVOoO, optimize=True)
    )
    dR.abb.VVVOoO += (1.0 / 2.0) * (
            -1.0 * np.einsum('Ae,eCBIjK->ABCIjK', H.a.vv[Va, va], R.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('AE,ECBIjK->ABCIjK', H.a.vv[Va, Va], R.abb.VVVOoO, optimize=True)
    )
    dR.abb.VVVOoO += (2.0 / 2.0) * (
            -1.0 * np.einsum('Be,ACeIjK->ABCIjK', H.b.vv[Vb, vb], R.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('BE,ACEIjK->ABCIjK', H.b.vv[Vb, Vb], R.abb.VVVOoO, optimize=True)
    )
    dR.abb.VVVOoO += (1.0 / 2.0) * (
            -0.5 * np.einsum('mnjK,ACBImn->ABCIjK', H.bb.oooo[ob, ob, ob, Ob], R.abb.VVVOoo, optimize=True)
            - 1.0 * np.einsum('mNjK,ACBImN->ABCIjK', H.bb.oooo[ob, Ob, ob, Ob], R.abb.VVVOoO, optimize=True)
            - 0.5 * np.einsum('MNjK,ACBIMN->ABCIjK', H.bb.oooo[Ob, Ob, ob, Ob], R.abb.VVVOOO, optimize=True)
    )
    dR.abb.VVVOoO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mnIj,ACBmnK->ABCIjK', H.ab.oooo[oa, ob, Oa, ob], R.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('MnIj,ACBMnK->ABCIjK', H.ab.oooo[Oa, ob, Oa, ob], R.abb.VVVOoO, optimize=True)
            - 1.0 * np.einsum('mNIj,ACBmNK->ABCIjK', H.ab.oooo[oa, Ob, Oa, ob], R.abb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('MNIj,ACBMNK->ABCIjK', H.ab.oooo[Oa, Ob, Oa, ob], R.abb.VVVOOO, optimize=True)
    )
    dR.abb.VVVOoO += (1.0 / 2.0) * (
            +1.0 * np.einsum('MnIK,ACBMnj->ABCIjK', H.ab.oooo[Oa, ob, Oa, Ob], R.abb.VVVOoo, optimize=True)
            - 1.0 * np.einsum('mNIK,ACBmjN->ABCIjK', H.ab.oooo[oa, Ob, Oa, Ob], R.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('MNIK,ACBMjN->ABCIjK', H.ab.oooo[Oa, Ob, Oa, Ob], R.abb.VVVOoO, optimize=True)
    )
    dR.abb.VVVOoO += (1.0 / 2.0) * (
            -0.5 * np.einsum('BCef,AfeIjK->ABCIjK', H.bb.vvvv[Vb, Vb, vb, vb], R.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('BCEf,AEfIjK->ABCIjK', H.bb.vvvv[Vb, Vb, Vb, vb], R.abb.VVvOoO, optimize=True)
            - 0.5 * np.einsum('BCEF,AFEIjK->ABCIjK', H.bb.vvvv[Vb, Vb, Vb, Vb], R.abb.VVVOoO, optimize=True)
    )
    dR.abb.VVVOoO += (2.0 / 2.0) * (
            -1.0 * np.einsum('ABef,eCfIjK->ABCIjK', H.ab.vvvv[Va, Vb, va, vb], R.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('ABeF,eCFIjK->ABCIjK', H.ab.vvvv[Va, Vb, va, Vb], R.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('ABEf,ECfIjK->ABCIjK', H.ab.vvvv[Va, Vb, Va, vb], R.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('ABEF,ECFIjK->ABCIjK', H.ab.vvvv[Va, Vb, Va, Vb], R.abb.VVVOoO, optimize=True)
    )
    dR.abb.VVVOoO += (1.0 / 2.0) * (
            -1.0 * np.einsum('AmIe,eCBmjK->ABCIjK', H.aa.voov[Va, oa, Oa, va], R.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('AMIe,eCBMjK->ABCIjK', H.aa.voov[Va, Oa, Oa, va], R.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('AmIE,ECBmjK->ABCIjK', H.aa.voov[Va, oa, Oa, Va], R.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('AMIE,ECBMjK->ABCIjK', H.aa.voov[Va, Oa, Oa, Va], R.abb.VVVOoO, optimize=True)
    )
    dR.abb.VVVOoO += (1.0 / 2.0) * (
            -1.0 * np.einsum('AmIe,CBemjK->ABCIjK', H.ab.voov[Va, ob, Oa, vb], R.bbb.VVvooO, optimize=True)
            + 1.0 * np.einsum('AMIe,CBejMK->ABCIjK', H.ab.voov[Va, Ob, Oa, vb], R.bbb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('AmIE,CBEmjK->ABCIjK', H.ab.voov[Va, ob, Oa, Vb], R.bbb.VVVooO, optimize=True)
            + 1.0 * np.einsum('AMIE,CBEjMK->ABCIjK', H.ab.voov[Va, Ob, Oa, Vb], R.bbb.VVVoOO, optimize=True)
    )
    dR.abb.VVVOoO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mBej,AeCmIK->ABCIjK', H.ab.ovvo[oa, Vb, va, ob], R.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('MBej,AeCIMK->ABCIjK', H.ab.ovvo[Oa, Vb, va, ob], R.aab.VvVOOO, optimize=True)
            + 1.0 * np.einsum('mBEj,EACmIK->ABCIjK', H.ab.ovvo[oa, Vb, Va, ob], R.aab.VVVoOO, optimize=True)
            - 1.0 * np.einsum('MBEj,EACIMK->ABCIjK', H.ab.ovvo[Oa, Vb, Va, ob], R.aab.VVVOOO, optimize=True)
    )
    dR.abb.VVVOoO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mBeK,AeCmIj->ABCIjK', H.ab.ovvo[oa, Vb, va, Ob], R.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('MBeK,AeCIMj->ABCIjK', H.ab.ovvo[Oa, Vb, va, Ob], R.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('mBEK,EACmIj->ABCIjK', H.ab.ovvo[oa, Vb, Va, Ob], R.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('MBEK,EACIMj->ABCIjK', H.ab.ovvo[Oa, Vb, Va, Ob], R.aab.VVVOOo, optimize=True)
    )
    dR.abb.VVVOoO += (2.0 / 2.0) * (
            -1.0 * np.einsum('Bmje,ACeImK->ABCIjK', H.bb.voov[Vb, ob, ob, vb], R.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('BMje,ACeIMK->ABCIjK', H.bb.voov[Vb, Ob, ob, vb], R.abb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('BmjE,ACEImK->ABCIjK', H.bb.voov[Vb, ob, ob, Vb], R.abb.VVVOoO, optimize=True)
            - 1.0 * np.einsum('BMjE,ACEIMK->ABCIjK', H.bb.voov[Vb, Ob, ob, Vb], R.abb.VVVOOO, optimize=True)
    )
    dR.abb.VVVOoO += (2.0 / 2.0) * (
            +1.0 * np.einsum('BmKe,ACeImj->ABCIjK', H.bb.voov[Vb, ob, Ob, vb], R.abb.VVvOoo, optimize=True)
            - 1.0 * np.einsum('BMKe,ACeIjM->ABCIjK', H.bb.voov[Vb, Ob, Ob, vb], R.abb.VVvOoO, optimize=True)
            + 1.0 * np.einsum('BmKE,ACEImj->ABCIjK', H.bb.voov[Vb, ob, Ob, Vb], R.abb.VVVOoo, optimize=True)
            - 1.0 * np.einsum('BMKE,ACEIjM->ABCIjK', H.bb.voov[Vb, Ob, Ob, Vb], R.abb.VVVOoO, optimize=True)
    )
    dR.abb.VVVOoO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mBIe,ACemjK->ABCIjK', H.ab.ovov[oa, Vb, Oa, vb], R.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('MBIe,ACeMjK->ABCIjK', H.ab.ovov[Oa, Vb, Oa, vb], R.abb.VVvOoO, optimize=True)
            + 1.0 * np.einsum('mBIE,ACEmjK->ABCIjK', H.ab.ovov[oa, Vb, Oa, Vb], R.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('MBIE,ACEMjK->ABCIjK', H.ab.ovov[Oa, Vb, Oa, Vb], R.abb.VVVOoO, optimize=True)
    )
    dR.abb.VVVOoO += (1.0 / 2.0) * (
            +1.0 * np.einsum('Amej,eCBImK->ABCIjK', H.ab.vovo[Va, ob, va, ob], R.abb.vVVOoO, optimize=True)
            + 1.0 * np.einsum('AMej,eCBIMK->ABCIjK', H.ab.vovo[Va, Ob, va, ob], R.abb.vVVOOO, optimize=True)
            + 1.0 * np.einsum('AmEj,ECBImK->ABCIjK', H.ab.vovo[Va, ob, Va, ob], R.abb.VVVOoO, optimize=True)
            + 1.0 * np.einsum('AMEj,ECBIMK->ABCIjK', H.ab.vovo[Va, Ob, Va, ob], R.abb.VVVOOO, optimize=True)
    )
    dR.abb.VVVOoO += (1.0 / 2.0) * (
            -1.0 * np.einsum('AmeK,eCBImj->ABCIjK', H.ab.vovo[Va, ob, va, Ob], R.abb.vVVOoo, optimize=True)
            + 1.0 * np.einsum('AMeK,eCBIjM->ABCIjK', H.ab.vovo[Va, Ob, va, Ob], R.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('AmEK,ECBImj->ABCIjK', H.ab.vovo[Va, ob, Va, Ob], R.abb.VVVOoo, optimize=True)
            + 1.0 * np.einsum('AMEK,ECBIjM->ABCIjK', H.ab.vovo[Va, Ob, Va, Ob], R.abb.VVVOoO, optimize=True)
    )

    dR.abb.VVVOoO -= np.transpose(dR.abb.VVVOoO, (0, 2, 1, 3, 4, 5))

    return dR

def update(R, omega, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    R.abb.VVVOoO = eomcc_active_loops.update_r3c_111101(
        R.abb.VVVOoO,
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
