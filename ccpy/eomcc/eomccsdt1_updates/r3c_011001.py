import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops

def build(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.abb.vVVooO = (2.0 / 2.0) * (
            +1.0 * np.einsum('aBie,eCjK->aBCijK', X.ab.vvov[va, Vb, oa, :], T.bb[:, Vb, ob, Ob], optimize=True)
    )
    dR.abb.vVVooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('amij,BCmK->aBCijK', X.ab.vooo[va, :, oa, ob], T.bb[Vb, Vb, :, Ob], optimize=True)
    )
    dR.abb.vVVooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('amiK,BCmj->aBCijK', X.ab.vooo[va, :, oa, Ob], T.bb[Vb, Vb, :, ob], optimize=True)
    )
    dR.abb.vVVooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('CBKe,aeij->aBCijK', X.bb.vvov[Vb, Vb, Ob, :], T.ab[va, :, oa, ob], optimize=True)
    )
    dR.abb.vVVooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('CBje,aeiK->aBCijK', X.bb.vvov[Vb, Vb, ob, :], T.ab[va, :, oa, Ob], optimize=True)
    )
    dR.abb.vVVooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('CmKj,aBim->aBCijK', X.bb.vooo[Vb, :, Ob, ob], T.ab[va, Vb, oa, :], optimize=True)
    )
    dR.abb.vVVooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('aBej,eCiK->aBCijK', X.ab.vvvo[va, Vb, :, ob], T.ab[:, Vb, oa, Ob], optimize=True)
    )
    dR.abb.vVVooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('aBeK,eCij->aBCijK', X.ab.vvvo[va, Vb, :, Ob], T.ab[:, Vb, oa, ob], optimize=True)
    )
    dR.abb.vVVooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mBij,aCmK->aBCijK', X.ab.ovoo[:, Vb, oa, ob], T.ab[va, Vb, :, Ob], optimize=True)
    )
    dR.abb.vVVooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mBiK,aCmj->aBCijK', X.ab.ovoo[:, Vb, oa, Ob], T.ab[va, Vb, :, ob], optimize=True)
    )
    dR.abb.vVVooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('aBie,eCjK->aBCijK', H.ab.vvov[va, Vb, oa, :], R.bb[:, Vb, ob, Ob], optimize=True)
    )
    dR.abb.vVVooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('amij,BCmK->aBCijK', H.ab.vooo[va, :, oa, ob], R.bb[Vb, Vb, :, Ob], optimize=True)
    )
    dR.abb.vVVooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('amiK,BCmj->aBCijK', H.ab.vooo[va, :, oa, Ob], R.bb[Vb, Vb, :, ob], optimize=True)
    )
    dR.abb.vVVooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('CBKe,aeij->aBCijK', H.bb.vvov[Vb, Vb, Ob, :], R.ab[va, :, oa, ob], optimize=True)
    )
    dR.abb.vVVooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('CBje,aeiK->aBCijK', H.bb.vvov[Vb, Vb, ob, :], R.ab[va, :, oa, Ob], optimize=True)
    )
    dR.abb.vVVooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('CmKj,aBim->aBCijK', H.bb.vooo[Vb, :, Ob, ob], R.ab[va, Vb, oa, :], optimize=True)
    )
    dR.abb.vVVooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('aBej,eCiK->aBCijK', H.ab.vvvo[va, Vb, :, ob], R.ab[:, Vb, oa, Ob], optimize=True)
    )
    dR.abb.vVVooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('aBeK,eCij->aBCijK', H.ab.vvvo[va, Vb, :, Ob], R.ab[:, Vb, oa, ob], optimize=True)
    )
    dR.abb.vVVooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mBij,aCmK->aBCijK', H.ab.ovoo[:, Vb, oa, ob], R.ab[va, Vb, :, Ob], optimize=True)
    )
    dR.abb.vVVooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mBiK,aCmj->aBCijK', H.ab.ovoo[:, Vb, oa, Ob], R.ab[va, Vb, :, ob], optimize=True)
    )
    # of terms =  20
    dR.abb.vVVooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mi,aCBmjK->aBCijK', X.a.oo[oa, oa], T.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('Mi,aCBMjK->aBCijK', X.a.oo[Oa, oa], T.abb.vVVOoO, optimize=True)
    )
    dR.abb.vVVooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mj,aCBimK->aBCijK', X.b.oo[ob, ob], T.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('Mj,aCBiMK->aBCijK', X.b.oo[Ob, ob], T.abb.vVVoOO, optimize=True)
    )
    dR.abb.vVVooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('MK,aCBijM->aBCijK', X.b.oo[Ob, Ob], T.abb.vVVooO, optimize=True)
    )
    dR.abb.vVVooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('ae,eCBijK->aBCijK', X.a.vv[va, va], T.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('aE,ECBijK->aBCijK', X.a.vv[va, Va], T.abb.VVVooO, optimize=True)
    )
    dR.abb.vVVooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('Be,aCeijK->aBCijK', X.b.vv[Vb, vb], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('BE,aCEijK->aBCijK', X.b.vv[Vb, Vb], T.abb.vVVooO, optimize=True)
    )
    dR.abb.vVVooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('MnjK,aCBinM->aBCijK', X.bb.oooo[Ob, ob, ob, Ob], T.abb.vVVooO, optimize=True)
            - 0.5 * np.einsum('MNjK,aCBiMN->aBCijK', X.bb.oooo[Ob, Ob, ob, Ob], T.abb.vVVoOO, optimize=True)
    )
    dR.abb.vVVooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mnij,aCBmnK->aBCijK', X.ab.oooo[oa, ob, oa, ob], T.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('mNij,aCBmNK->aBCijK', X.ab.oooo[oa, Ob, oa, ob], T.abb.vVVoOO, optimize=True)
            - 1.0 * np.einsum('Mnij,aCBMnK->aBCijK', X.ab.oooo[Oa, ob, oa, ob], T.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('MNij,aCBMNK->aBCijK', X.ab.oooo[Oa, Ob, oa, ob], T.abb.vVVOOO, optimize=True)
    )
    dR.abb.vVVooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mNiK,aCBmjN->aBCijK', X.ab.oooo[oa, Ob, oa, Ob], T.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('MniK,aCBMnj->aBCijK', X.ab.oooo[Oa, ob, oa, Ob], T.abb.vVVOoo, optimize=True)
            - 1.0 * np.einsum('MNiK,aCBMjN->aBCijK', X.ab.oooo[Oa, Ob, oa, Ob], T.abb.vVVOoO, optimize=True)
    )
    dR.abb.vVVooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('BCeF,aFeijK->aBCijK', X.bb.vvvv[Vb, Vb, vb, Vb], T.abb.vVvooO, optimize=True)
            - 0.5 * np.einsum('BCEF,aFEijK->aBCijK', X.bb.vvvv[Vb, Vb, Vb, Vb], T.abb.vVVooO, optimize=True)
    )
    dR.abb.vVVooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('aBef,eCfijK->aBCijK', X.ab.vvvv[va, Vb, va, vb], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('aBEf,ECfijK->aBCijK', X.ab.vvvv[va, Vb, Va, vb], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('aBeF,eCFijK->aBCijK', X.ab.vvvv[va, Vb, va, Vb], T.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('aBEF,ECFijK->aBCijK', X.ab.vvvv[va, Vb, Va, Vb], T.abb.VVVooO, optimize=True)
    )
    dR.abb.vVVooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('amie,eCBmjK->aBCijK', X.aa.voov[va, oa, oa, va], T.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('amiE,ECBmjK->aBCijK', X.aa.voov[va, oa, oa, Va], T.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('aMie,eCBMjK->aBCijK', X.aa.voov[va, Oa, oa, va], T.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('aMiE,ECBMjK->aBCijK', X.aa.voov[va, Oa, oa, Va], T.abb.VVVOoO, optimize=True)
    )
    dR.abb.vVVooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('amie,CBemjK->aBCijK', X.ab.voov[va, ob, oa, vb], T.bbb.VVvooO, optimize=True)
            - 1.0 * np.einsum('amiE,CBEmjK->aBCijK', X.ab.voov[va, ob, oa, Vb], T.bbb.VVVooO, optimize=True)
            + 1.0 * np.einsum('aMie,CBejMK->aBCijK', X.ab.voov[va, Ob, oa, vb], T.bbb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('aMiE,CBEjMK->aBCijK', X.ab.voov[va, Ob, oa, Vb], T.bbb.VVVoOO, optimize=True)
    )
    dR.abb.vVVooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mBej,eaCimK->aBCijK', X.ab.ovvo[oa, Vb, va, ob], T.aab.vvVooO, optimize=True)
            - 1.0 * np.einsum('mBEj,EaCimK->aBCijK', X.ab.ovvo[oa, Vb, Va, ob], T.aab.VvVooO, optimize=True)
            - 1.0 * np.einsum('MBej,eaCiMK->aBCijK', X.ab.ovvo[Oa, Vb, va, ob], T.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('MBEj,EaCiMK->aBCijK', X.ab.ovvo[Oa, Vb, Va, ob], T.aab.VvVoOO, optimize=True)
    )
    dR.abb.vVVooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('MBeK,eaCiMj->aBCijK', X.ab.ovvo[Oa, Vb, va, Ob], T.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('MBEK,EaCiMj->aBCijK', X.ab.ovvo[Oa, Vb, Va, Ob], T.aab.VvVoOo, optimize=True)
    )
    dR.abb.vVVooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('Bmje,aCeimK->aBCijK', X.bb.voov[Vb, ob, ob, vb], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('BmjE,aCEimK->aBCijK', X.bb.voov[Vb, ob, ob, Vb], T.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('BMje,aCeiMK->aBCijK', X.bb.voov[Vb, Ob, ob, vb], T.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('BMjE,aCEiMK->aBCijK', X.bb.voov[Vb, Ob, ob, Vb], T.abb.vVVoOO, optimize=True)
    )
    dR.abb.vVVooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('BMKe,aCeijM->aBCijK', X.bb.voov[Vb, Ob, Ob, vb], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('BMKE,aCEijM->aBCijK', X.bb.voov[Vb, Ob, Ob, Vb], T.abb.vVVooO, optimize=True)
    )
    dR.abb.vVVooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mBie,aCemjK->aBCijK', X.ab.ovov[oa, Vb, oa, vb], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('mBiE,aCEmjK->aBCijK', X.ab.ovov[oa, Vb, oa, Vb], T.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('MBie,aCeMjK->aBCijK', X.ab.ovov[Oa, Vb, oa, vb], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('MBiE,aCEMjK->aBCijK', X.ab.ovov[Oa, Vb, oa, Vb], T.abb.vVVOoO, optimize=True)
    )
    dR.abb.vVVooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('amej,eCBimK->aBCijK', X.ab.vovo[va, ob, va, ob], T.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('amEj,ECBimK->aBCijK', X.ab.vovo[va, ob, Va, ob], T.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('aMej,eCBiMK->aBCijK', X.ab.vovo[va, Ob, va, ob], T.abb.vVVoOO, optimize=True)
            + 1.0 * np.einsum('aMEj,ECBiMK->aBCijK', X.ab.vovo[va, Ob, Va, ob], T.abb.VVVoOO, optimize=True)
    )
    dR.abb.vVVooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('aMeK,eCBijM->aBCijK', X.ab.vovo[va, Ob, va, Ob], T.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('aMEK,ECBijM->aBCijK', X.ab.vovo[va, Ob, Va, Ob], T.abb.VVVooO, optimize=True)
    )
    dR.abb.vVVooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mi,aCBmjK->aBCijK', H.a.oo[oa, oa], R.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('Mi,aCBMjK->aBCijK', H.a.oo[Oa, oa], R.abb.vVVOoO, optimize=True)
    )
    dR.abb.vVVooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mj,aCBimK->aBCijK', H.b.oo[ob, ob], R.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('Mj,aCBiMK->aBCijK', H.b.oo[Ob, ob], R.abb.vVVoOO, optimize=True)
    )
    dR.abb.vVVooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('MK,aCBijM->aBCijK', H.b.oo[Ob, Ob], R.abb.vVVooO, optimize=True)
    )
    dR.abb.vVVooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('ae,eCBijK->aBCijK', H.a.vv[va, va], R.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('aE,ECBijK->aBCijK', H.a.vv[va, Va], R.abb.VVVooO, optimize=True)
    )
    dR.abb.vVVooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('Be,aCeijK->aBCijK', H.b.vv[Vb, vb], R.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('BE,aCEijK->aBCijK', H.b.vv[Vb, Vb], R.abb.vVVooO, optimize=True)
    )
    dR.abb.vVVooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('MnjK,aCBinM->aBCijK', H.bb.oooo[Ob, ob, ob, Ob], R.abb.vVVooO, optimize=True)
            - 0.5 * np.einsum('MNjK,aCBiMN->aBCijK', H.bb.oooo[Ob, Ob, ob, Ob], R.abb.vVVoOO, optimize=True)
    )
    dR.abb.vVVooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mnij,aCBmnK->aBCijK', H.ab.oooo[oa, ob, oa, ob], R.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('mNij,aCBmNK->aBCijK', H.ab.oooo[oa, Ob, oa, ob], R.abb.vVVoOO, optimize=True)
            - 1.0 * np.einsum('Mnij,aCBMnK->aBCijK', H.ab.oooo[Oa, ob, oa, ob], R.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('MNij,aCBMNK->aBCijK', H.ab.oooo[Oa, Ob, oa, ob], R.abb.vVVOOO, optimize=True)
    )
    dR.abb.vVVooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mNiK,aCBmjN->aBCijK', H.ab.oooo[oa, Ob, oa, Ob], R.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('MniK,aCBMnj->aBCijK', H.ab.oooo[Oa, ob, oa, Ob], R.abb.vVVOoo, optimize=True)
            - 1.0 * np.einsum('MNiK,aCBMjN->aBCijK', H.ab.oooo[Oa, Ob, oa, Ob], R.abb.vVVOoO, optimize=True)
    )
    dR.abb.vVVooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('BCeF,aFeijK->aBCijK', H.bb.vvvv[Vb, Vb, vb, Vb], R.abb.vVvooO, optimize=True)
            - 0.5 * np.einsum('BCEF,aFEijK->aBCijK', H.bb.vvvv[Vb, Vb, Vb, Vb], R.abb.vVVooO, optimize=True)
    )
    dR.abb.vVVooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('aBef,eCfijK->aBCijK', H.ab.vvvv[va, Vb, va, vb], R.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('aBEf,ECfijK->aBCijK', H.ab.vvvv[va, Vb, Va, vb], R.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('aBeF,eCFijK->aBCijK', H.ab.vvvv[va, Vb, va, Vb], R.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('aBEF,ECFijK->aBCijK', H.ab.vvvv[va, Vb, Va, Vb], R.abb.VVVooO, optimize=True)
    )
    dR.abb.vVVooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('amie,eCBmjK->aBCijK', H.aa.voov[va, oa, oa, va], R.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('amiE,ECBmjK->aBCijK', H.aa.voov[va, oa, oa, Va], R.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('aMie,eCBMjK->aBCijK', H.aa.voov[va, Oa, oa, va], R.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('aMiE,ECBMjK->aBCijK', H.aa.voov[va, Oa, oa, Va], R.abb.VVVOoO, optimize=True)
    )
    dR.abb.vVVooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('amie,CBemjK->aBCijK', H.ab.voov[va, ob, oa, vb], R.bbb.VVvooO, optimize=True)
            - 1.0 * np.einsum('amiE,CBEmjK->aBCijK', H.ab.voov[va, ob, oa, Vb], R.bbb.VVVooO, optimize=True)
            + 1.0 * np.einsum('aMie,CBejMK->aBCijK', H.ab.voov[va, Ob, oa, vb], R.bbb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('aMiE,CBEjMK->aBCijK', H.ab.voov[va, Ob, oa, Vb], R.bbb.VVVoOO, optimize=True)
    )
    dR.abb.vVVooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mBej,eaCimK->aBCijK', H.ab.ovvo[oa, Vb, va, ob], R.aab.vvVooO, optimize=True)
            - 1.0 * np.einsum('mBEj,EaCimK->aBCijK', H.ab.ovvo[oa, Vb, Va, ob], R.aab.VvVooO, optimize=True)
            - 1.0 * np.einsum('MBej,eaCiMK->aBCijK', H.ab.ovvo[Oa, Vb, va, ob], R.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('MBEj,EaCiMK->aBCijK', H.ab.ovvo[Oa, Vb, Va, ob], R.aab.VvVoOO, optimize=True)
    )
    dR.abb.vVVooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('MBeK,eaCiMj->aBCijK', H.ab.ovvo[Oa, Vb, va, Ob], R.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('MBEK,EaCiMj->aBCijK', H.ab.ovvo[Oa, Vb, Va, Ob], R.aab.VvVoOo, optimize=True)
    )
    dR.abb.vVVooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('Bmje,aCeimK->aBCijK', H.bb.voov[Vb, ob, ob, vb], R.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('BmjE,aCEimK->aBCijK', H.bb.voov[Vb, ob, ob, Vb], R.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('BMje,aCeiMK->aBCijK', H.bb.voov[Vb, Ob, ob, vb], R.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('BMjE,aCEiMK->aBCijK', H.bb.voov[Vb, Ob, ob, Vb], R.abb.vVVoOO, optimize=True)
    )
    dR.abb.vVVooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('BMKe,aCeijM->aBCijK', H.bb.voov[Vb, Ob, Ob, vb], R.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('BMKE,aCEijM->aBCijK', H.bb.voov[Vb, Ob, Ob, Vb], R.abb.vVVooO, optimize=True)
    )
    dR.abb.vVVooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mBie,aCemjK->aBCijK', H.ab.ovov[oa, Vb, oa, vb], R.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('mBiE,aCEmjK->aBCijK', H.ab.ovov[oa, Vb, oa, Vb], R.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('MBie,aCeMjK->aBCijK', H.ab.ovov[Oa, Vb, oa, vb], R.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('MBiE,aCEMjK->aBCijK', H.ab.ovov[Oa, Vb, oa, Vb], R.abb.vVVOoO, optimize=True)
    )
    dR.abb.vVVooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('amej,eCBimK->aBCijK', H.ab.vovo[va, ob, va, ob], R.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('amEj,ECBimK->aBCijK', H.ab.vovo[va, ob, Va, ob], R.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('aMej,eCBiMK->aBCijK', H.ab.vovo[va, Ob, va, ob], R.abb.vVVoOO, optimize=True)
            + 1.0 * np.einsum('aMEj,ECBiMK->aBCijK', H.ab.vovo[va, Ob, Va, ob], R.abb.VVVoOO, optimize=True)
    )
    dR.abb.vVVooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('aMeK,eCBijM->aBCijK', H.ab.vovo[va, Ob, va, Ob], R.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('aMEK,ECBijM->aBCijK', H.ab.vovo[va, Ob, Va, Ob], R.abb.VVVooO, optimize=True)
    )
    # of terms =  38

    dR.abb.vVVooO -= np.transpose(dR.abb.vVVooO, (0, 2, 1, 3, 4, 5))

    return dR

def update(R, omega, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    R.abb.vVVooO = eomcc_active_loops.update_r3c_011001(
        R.abb.vVVooO,
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
