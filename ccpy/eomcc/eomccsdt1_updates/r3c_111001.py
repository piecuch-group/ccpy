import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops

def build(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.abb.VVVooO = (2.0 / 2.0) * (
            +1.0 * np.einsum('ABie,eCjK->ABCijK', X.ab.vvov[Va, Vb, oa, :], T.bb[:, Vb, ob, Ob], optimize=True)
    )
    dR.abb.VVVooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('Amij,BCmK->ABCijK', X.ab.vooo[Va, :, oa, ob], T.bb[Vb, Vb, :, Ob], optimize=True)
    )
    dR.abb.VVVooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('AmiK,BCmj->ABCijK', X.ab.vooo[Va, :, oa, Ob], T.bb[Vb, Vb, :, ob], optimize=True)
    )
    dR.abb.VVVooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('CBKe,Aeij->ABCijK', X.bb.vvov[Vb, Vb, Ob, :], T.ab[Va, :, oa, ob], optimize=True)
    )
    dR.abb.VVVooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('CBje,AeiK->ABCijK', X.bb.vvov[Vb, Vb, ob, :], T.ab[Va, :, oa, Ob], optimize=True)
    )
    dR.abb.VVVooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('CmKj,ABim->ABCijK', X.bb.vooo[Vb, :, Ob, ob], T.ab[Va, Vb, oa, :], optimize=True)
    )
    dR.abb.VVVooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('ABej,eCiK->ABCijK', X.ab.vvvo[Va, Vb, :, ob], T.ab[:, Vb, oa, Ob], optimize=True)
    )
    dR.abb.VVVooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('ABeK,eCij->ABCijK', X.ab.vvvo[Va, Vb, :, Ob], T.ab[:, Vb, oa, ob], optimize=True)
    )
    dR.abb.VVVooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mBij,ACmK->ABCijK', X.ab.ovoo[:, Vb, oa, ob], T.ab[Va, Vb, :, Ob], optimize=True)
    )
    dR.abb.VVVooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mBiK,ACmj->ABCijK', X.ab.ovoo[:, Vb, oa, Ob], T.ab[Va, Vb, :, ob], optimize=True)
    )
    dR.abb.VVVooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('ABie,eCjK->ABCijK', H.ab.vvov[Va, Vb, oa, :], R.bb[:, Vb, ob, Ob], optimize=True)
    )
    dR.abb.VVVooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('Amij,BCmK->ABCijK', H.ab.vooo[Va, :, oa, ob], R.bb[Vb, Vb, :, Ob], optimize=True)
    )
    dR.abb.VVVooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('AmiK,BCmj->ABCijK', H.ab.vooo[Va, :, oa, Ob], R.bb[Vb, Vb, :, ob], optimize=True)
    )
    dR.abb.VVVooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('CBKe,Aeij->ABCijK', H.bb.vvov[Vb, Vb, Ob, :], R.ab[Va, :, oa, ob], optimize=True)
    )
    dR.abb.VVVooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('CBje,AeiK->ABCijK', H.bb.vvov[Vb, Vb, ob, :], R.ab[Va, :, oa, Ob], optimize=True)
    )
    dR.abb.VVVooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('CmKj,ABim->ABCijK', H.bb.vooo[Vb, :, Ob, ob], R.ab[Va, Vb, oa, :], optimize=True)
    )
    dR.abb.VVVooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('ABej,eCiK->ABCijK', H.ab.vvvo[Va, Vb, :, ob], R.ab[:, Vb, oa, Ob], optimize=True)
    )
    dR.abb.VVVooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('ABeK,eCij->ABCijK', H.ab.vvvo[Va, Vb, :, Ob], R.ab[:, Vb, oa, ob], optimize=True)
    )
    dR.abb.VVVooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mBij,ACmK->ABCijK', H.ab.ovoo[:, Vb, oa, ob], R.ab[Va, Vb, :, Ob], optimize=True)
    )
    dR.abb.VVVooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mBiK,ACmj->ABCijK', H.ab.ovoo[:, Vb, oa, Ob], R.ab[Va, Vb, :, ob], optimize=True)
    )
    # of terms =  20
    dR.abb.VVVooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mi,ACBmjK->ABCijK', X.a.oo[oa, oa], T.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('Mi,ACBMjK->ABCijK', X.a.oo[Oa, oa], T.abb.VVVOoO, optimize=True)
    )
    dR.abb.VVVooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mj,ACBimK->ABCijK', X.b.oo[ob, ob], T.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('Mj,ACBiMK->ABCijK', X.b.oo[Ob, ob], T.abb.VVVoOO, optimize=True)
    )
    dR.abb.VVVooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('MK,ACBijM->ABCijK', X.b.oo[Ob, Ob], T.abb.VVVooO, optimize=True)
    )
    dR.abb.VVVooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('Ae,eCBijK->ABCijK', X.a.vv[Va, va], T.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('AE,ECBijK->ABCijK', X.a.vv[Va, Va], T.abb.VVVooO, optimize=True)
    )
    dR.abb.VVVooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('Be,ACeijK->ABCijK', X.b.vv[Vb, vb], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('BE,ACEijK->ABCijK', X.b.vv[Vb, Vb], T.abb.VVVooO, optimize=True)
    )
    dR.abb.VVVooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('MnjK,ACBinM->ABCijK', X.bb.oooo[Ob, ob, ob, Ob], T.abb.VVVooO, optimize=True)
            - 0.5 * np.einsum('MNjK,ACBiMN->ABCijK', X.bb.oooo[Ob, Ob, ob, Ob], T.abb.VVVoOO, optimize=True)
    )
    dR.abb.VVVooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mnij,ACBmnK->ABCijK', X.ab.oooo[oa, ob, oa, ob], T.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('mNij,ACBmNK->ABCijK', X.ab.oooo[oa, Ob, oa, ob], T.abb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('Mnij,ACBMnK->ABCijK', X.ab.oooo[Oa, ob, oa, ob], T.abb.VVVOoO, optimize=True)
            - 1.0 * np.einsum('MNij,ACBMNK->ABCijK', X.ab.oooo[Oa, Ob, oa, ob], T.abb.VVVOOO, optimize=True)
    )
    dR.abb.VVVooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mNiK,ACBmjN->ABCijK', X.ab.oooo[oa, Ob, oa, Ob], T.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('MniK,ACBMnj->ABCijK', X.ab.oooo[Oa, ob, oa, Ob], T.abb.VVVOoo, optimize=True)
            - 1.0 * np.einsum('MNiK,ACBMjN->ABCijK', X.ab.oooo[Oa, Ob, oa, Ob], T.abb.VVVOoO, optimize=True)
    )
    dR.abb.VVVooO += (1.0 / 2.0) * (
            -0.5 * np.einsum('BCef,AfeijK->ABCijK', X.bb.vvvv[Vb, Vb, vb, vb], T.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('BCeF,AFeijK->ABCijK', X.bb.vvvv[Vb, Vb, vb, Vb], T.abb.VVvooO, optimize=True)
            - 0.5 * np.einsum('BCEF,AFEijK->ABCijK', X.bb.vvvv[Vb, Vb, Vb, Vb], T.abb.VVVooO, optimize=True)
    )
    dR.abb.VVVooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('ABef,eCfijK->ABCijK', X.ab.vvvv[Va, Vb, va, vb], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('ABEf,ECfijK->ABCijK', X.ab.vvvv[Va, Vb, Va, vb], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('ABeF,eCFijK->ABCijK', X.ab.vvvv[Va, Vb, va, Vb], T.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('ABEF,ECFijK->ABCijK', X.ab.vvvv[Va, Vb, Va, Vb], T.abb.VVVooO, optimize=True)
    )
    dR.abb.VVVooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('Amie,eCBmjK->ABCijK', X.aa.voov[Va, oa, oa, va], T.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('AmiE,ECBmjK->ABCijK', X.aa.voov[Va, oa, oa, Va], T.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('AMie,eCBMjK->ABCijK', X.aa.voov[Va, Oa, oa, va], T.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('AMiE,ECBMjK->ABCijK', X.aa.voov[Va, Oa, oa, Va], T.abb.VVVOoO, optimize=True)
    )
    dR.abb.VVVooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('Amie,CBemjK->ABCijK', X.ab.voov[Va, ob, oa, vb], T.bbb.VVvooO, optimize=True)
            - 1.0 * np.einsum('AmiE,CBEmjK->ABCijK', X.ab.voov[Va, ob, oa, Vb], T.bbb.VVVooO, optimize=True)
            + 1.0 * np.einsum('AMie,CBejMK->ABCijK', X.ab.voov[Va, Ob, oa, vb], T.bbb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('AMiE,CBEjMK->ABCijK', X.ab.voov[Va, Ob, oa, Vb], T.bbb.VVVoOO, optimize=True)
    )
    dR.abb.VVVooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mBej,AeCimK->ABCijK', X.ab.ovvo[oa, Vb, va, ob], T.aab.VvVooO, optimize=True)
            - 1.0 * np.einsum('mBEj,EACimK->ABCijK', X.ab.ovvo[oa, Vb, Va, ob], T.aab.VVVooO, optimize=True)
            + 1.0 * np.einsum('MBej,AeCiMK->ABCijK', X.ab.ovvo[Oa, Vb, va, ob], T.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('MBEj,EACiMK->ABCijK', X.ab.ovvo[Oa, Vb, Va, ob], T.aab.VVVoOO, optimize=True)
    )
    dR.abb.VVVooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('MBeK,AeCiMj->ABCijK', X.ab.ovvo[Oa, Vb, va, Ob], T.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('MBEK,EACiMj->ABCijK', X.ab.ovvo[Oa, Vb, Va, Ob], T.aab.VVVoOo, optimize=True)
    )
    dR.abb.VVVooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('Bmje,ACeimK->ABCijK', X.bb.voov[Vb, ob, ob, vb], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('BmjE,ACEimK->ABCijK', X.bb.voov[Vb, ob, ob, Vb], T.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('BMje,ACeiMK->ABCijK', X.bb.voov[Vb, Ob, ob, vb], T.abb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('BMjE,ACEiMK->ABCijK', X.bb.voov[Vb, Ob, ob, Vb], T.abb.VVVoOO, optimize=True)
    )
    dR.abb.VVVooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('BMKe,ACeijM->ABCijK', X.bb.voov[Vb, Ob, Ob, vb], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('BMKE,ACEijM->ABCijK', X.bb.voov[Vb, Ob, Ob, Vb], T.abb.VVVooO, optimize=True)
    )
    dR.abb.VVVooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mBie,ACemjK->ABCijK', X.ab.ovov[oa, Vb, oa, vb], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('mBiE,ACEmjK->ABCijK', X.ab.ovov[oa, Vb, oa, Vb], T.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('MBie,ACeMjK->ABCijK', X.ab.ovov[Oa, Vb, oa, vb], T.abb.VVvOoO, optimize=True)
            + 1.0 * np.einsum('MBiE,ACEMjK->ABCijK', X.ab.ovov[Oa, Vb, oa, Vb], T.abb.VVVOoO, optimize=True)
    )
    dR.abb.VVVooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('Amej,eCBimK->ABCijK', X.ab.vovo[Va, ob, va, ob], T.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('AmEj,ECBimK->ABCijK', X.ab.vovo[Va, ob, Va, ob], T.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('AMej,eCBiMK->ABCijK', X.ab.vovo[Va, Ob, va, ob], T.abb.vVVoOO, optimize=True)
            + 1.0 * np.einsum('AMEj,ECBiMK->ABCijK', X.ab.vovo[Va, Ob, Va, ob], T.abb.VVVoOO, optimize=True)
    )
    dR.abb.VVVooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('AMeK,eCBijM->ABCijK', X.ab.vovo[Va, Ob, va, Ob], T.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('AMEK,ECBijM->ABCijK', X.ab.vovo[Va, Ob, Va, Ob], T.abb.VVVooO, optimize=True)
    )
    dR.abb.VVVooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mi,ACBmjK->ABCijK', H.a.oo[oa, oa], R.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('Mi,ACBMjK->ABCijK', H.a.oo[Oa, oa], R.abb.VVVOoO, optimize=True)
    )
    dR.abb.VVVooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mj,ACBimK->ABCijK', H.b.oo[ob, ob], R.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('Mj,ACBiMK->ABCijK', H.b.oo[Ob, ob], R.abb.VVVoOO, optimize=True)
    )
    dR.abb.VVVooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('MK,ACBijM->ABCijK', H.b.oo[Ob, Ob], R.abb.VVVooO, optimize=True)
    )
    dR.abb.VVVooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('Ae,eCBijK->ABCijK', H.a.vv[Va, va], R.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('AE,ECBijK->ABCijK', H.a.vv[Va, Va], R.abb.VVVooO, optimize=True)
    )
    dR.abb.VVVooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('Be,ACeijK->ABCijK', H.b.vv[Vb, vb], R.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('BE,ACEijK->ABCijK', H.b.vv[Vb, Vb], R.abb.VVVooO, optimize=True)
    )
    dR.abb.VVVooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('MnjK,ACBinM->ABCijK', H.bb.oooo[Ob, ob, ob, Ob], R.abb.VVVooO, optimize=True)
            - 0.5 * np.einsum('MNjK,ACBiMN->ABCijK', H.bb.oooo[Ob, Ob, ob, Ob], R.abb.VVVoOO, optimize=True)
    )
    dR.abb.VVVooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mnij,ACBmnK->ABCijK', H.ab.oooo[oa, ob, oa, ob], R.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('mNij,ACBmNK->ABCijK', H.ab.oooo[oa, Ob, oa, ob], R.abb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('Mnij,ACBMnK->ABCijK', H.ab.oooo[Oa, ob, oa, ob], R.abb.VVVOoO, optimize=True)
            - 1.0 * np.einsum('MNij,ACBMNK->ABCijK', H.ab.oooo[Oa, Ob, oa, ob], R.abb.VVVOOO, optimize=True)
    )
    dR.abb.VVVooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mNiK,ACBmjN->ABCijK', H.ab.oooo[oa, Ob, oa, Ob], R.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('MniK,ACBMnj->ABCijK', H.ab.oooo[Oa, ob, oa, Ob], R.abb.VVVOoo, optimize=True)
            - 1.0 * np.einsum('MNiK,ACBMjN->ABCijK', H.ab.oooo[Oa, Ob, oa, Ob], R.abb.VVVOoO, optimize=True)
    )
    dR.abb.VVVooO += (1.0 / 2.0) * (
            -0.5 * np.einsum('BCef,AfeijK->ABCijK', H.bb.vvvv[Vb, Vb, vb, vb], R.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('BCeF,AFeijK->ABCijK', H.bb.vvvv[Vb, Vb, vb, Vb], R.abb.VVvooO, optimize=True)
            - 0.5 * np.einsum('BCEF,AFEijK->ABCijK', H.bb.vvvv[Vb, Vb, Vb, Vb], R.abb.VVVooO, optimize=True)
    )
    dR.abb.VVVooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('ABef,eCfijK->ABCijK', H.ab.vvvv[Va, Vb, va, vb], R.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('ABEf,ECfijK->ABCijK', H.ab.vvvv[Va, Vb, Va, vb], R.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('ABeF,eCFijK->ABCijK', H.ab.vvvv[Va, Vb, va, Vb], R.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('ABEF,ECFijK->ABCijK', H.ab.vvvv[Va, Vb, Va, Vb], R.abb.VVVooO, optimize=True)
    )
    dR.abb.VVVooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('Amie,eCBmjK->ABCijK', H.aa.voov[Va, oa, oa, va], R.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('AmiE,ECBmjK->ABCijK', H.aa.voov[Va, oa, oa, Va], R.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('AMie,eCBMjK->ABCijK', H.aa.voov[Va, Oa, oa, va], R.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('AMiE,ECBMjK->ABCijK', H.aa.voov[Va, Oa, oa, Va], R.abb.VVVOoO, optimize=True)
    )
    dR.abb.VVVooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('Amie,CBemjK->ABCijK', H.ab.voov[Va, ob, oa, vb], R.bbb.VVvooO, optimize=True)
            - 1.0 * np.einsum('AmiE,CBEmjK->ABCijK', H.ab.voov[Va, ob, oa, Vb], R.bbb.VVVooO, optimize=True)
            + 1.0 * np.einsum('AMie,CBejMK->ABCijK', H.ab.voov[Va, Ob, oa, vb], R.bbb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('AMiE,CBEjMK->ABCijK', H.ab.voov[Va, Ob, oa, Vb], R.bbb.VVVoOO, optimize=True)
    )
    dR.abb.VVVooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mBej,AeCimK->ABCijK', H.ab.ovvo[oa, Vb, va, ob], R.aab.VvVooO, optimize=True)
            - 1.0 * np.einsum('mBEj,EACimK->ABCijK', H.ab.ovvo[oa, Vb, Va, ob], R.aab.VVVooO, optimize=True)
            + 1.0 * np.einsum('MBej,AeCiMK->ABCijK', H.ab.ovvo[Oa, Vb, va, ob], R.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('MBEj,EACiMK->ABCijK', H.ab.ovvo[Oa, Vb, Va, ob], R.aab.VVVoOO, optimize=True)
    )
    dR.abb.VVVooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('MBeK,AeCiMj->ABCijK', H.ab.ovvo[Oa, Vb, va, Ob], R.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('MBEK,EACiMj->ABCijK', H.ab.ovvo[Oa, Vb, Va, Ob], R.aab.VVVoOo, optimize=True)
    )
    dR.abb.VVVooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('Bmje,ACeimK->ABCijK', H.bb.voov[Vb, ob, ob, vb], R.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('BmjE,ACEimK->ABCijK', H.bb.voov[Vb, ob, ob, Vb], R.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('BMje,ACeiMK->ABCijK', H.bb.voov[Vb, Ob, ob, vb], R.abb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('BMjE,ACEiMK->ABCijK', H.bb.voov[Vb, Ob, ob, Vb], R.abb.VVVoOO, optimize=True)
    )
    dR.abb.VVVooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('BMKe,ACeijM->ABCijK', H.bb.voov[Vb, Ob, Ob, vb], R.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('BMKE,ACEijM->ABCijK', H.bb.voov[Vb, Ob, Ob, Vb], R.abb.VVVooO, optimize=True)
    )
    dR.abb.VVVooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mBie,ACemjK->ABCijK', H.ab.ovov[oa, Vb, oa, vb], R.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('mBiE,ACEmjK->ABCijK', H.ab.ovov[oa, Vb, oa, Vb], R.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('MBie,ACeMjK->ABCijK', H.ab.ovov[Oa, Vb, oa, vb], R.abb.VVvOoO, optimize=True)
            + 1.0 * np.einsum('MBiE,ACEMjK->ABCijK', H.ab.ovov[Oa, Vb, oa, Vb], R.abb.VVVOoO, optimize=True)
    )
    dR.abb.VVVooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('Amej,eCBimK->ABCijK', H.ab.vovo[Va, ob, va, ob], R.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('AmEj,ECBimK->ABCijK', H.ab.vovo[Va, ob, Va, ob], R.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('AMej,eCBiMK->ABCijK', H.ab.vovo[Va, Ob, va, ob], R.abb.vVVoOO, optimize=True)
            + 1.0 * np.einsum('AMEj,ECBiMK->ABCijK', H.ab.vovo[Va, Ob, Va, ob], R.abb.VVVoOO, optimize=True)
    )
    dR.abb.VVVooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('AMeK,eCBijM->ABCijK', H.ab.vovo[Va, Ob, va, Ob], R.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('AMEK,ECBijM->ABCijK', H.ab.vovo[Va, Ob, Va, Ob], R.abb.VVVooO, optimize=True)
    )
    # of terms =  38

    dR.abb.VVVooO -= np.transpose(dR.abb.VVVooO, (0, 2, 1, 3, 4, 5))

    return dR

def update(R, omega, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    R.abb.VVVooO = eomcc_active_loops.update_r3c_111001(
        R.abb.VVVooO,
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
