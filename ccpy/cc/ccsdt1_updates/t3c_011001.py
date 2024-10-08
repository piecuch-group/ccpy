import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import cc_active_loops

def build(T, dT, H, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    # MM(2,3)
    dT.abb.vVVooO = (2.0 / 2.0) * (
            +1.0 * np.einsum('aBie,eCjK->aBCijK', H.ab.vvov[va, Vb, oa, :], T.bb[:, Vb, ob, Ob], optimize=True)
    )
    dT.abb.vVVooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('amij,BCmK->aBCijK', H.ab.vooo[va, :, oa, ob], T.bb[Vb, Vb, :, Ob], optimize=True)
    )
    dT.abb.vVVooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('amiK,BCmj->aBCijK', H.ab.vooo[va, :, oa, Ob], T.bb[Vb, Vb, :, ob], optimize=True)
    )
    dT.abb.vVVooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('CBKe,aeij->aBCijK', H.bb.vvov[Vb, Vb, Ob, :], T.ab[va, :, oa, ob], optimize=True)
    )
    dT.abb.vVVooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('CBje,aeiK->aBCijK', H.bb.vvov[Vb, Vb, ob, :], T.ab[va, :, oa, Ob], optimize=True)
    )
    dT.abb.vVVooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('CmKj,aBim->aBCijK', H.bb.vooo[Vb, :, Ob, ob], T.ab[va, Vb, oa, :], optimize=True)
    )
    dT.abb.vVVooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('aBej,eCiK->aBCijK', H.ab.vvvo[va, Vb, :, ob], T.ab[:, Vb, oa, Ob], optimize=True)
    )
    dT.abb.vVVooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('aBeK,eCij->aBCijK', H.ab.vvvo[va, Vb, :, Ob], T.ab[:, Vb, oa, ob], optimize=True)
    )
    dT.abb.vVVooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mBij,aCmK->aBCijK', H.ab.ovoo[:, Vb, oa, ob], T.ab[va, Vb, :, Ob], optimize=True)
    )
    dT.abb.vVVooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mBiK,aCmj->aBCijK', H.ab.ovoo[:, Vb, oa, Ob], T.ab[va, Vb, :, ob], optimize=True)
    )
    # (H(2) * T3)_C
    dT.abb.vVVooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mi,aCBmjK->aBCijK', H.a.oo[oa, oa], T.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('Mi,aCBMjK->aBCijK', H.a.oo[Oa, oa], T.abb.vVVOoO, optimize=True)
    )
    dT.abb.vVVooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mj,aCBimK->aBCijK', H.b.oo[ob, ob], T.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('Mj,aCBiMK->aBCijK', H.b.oo[Ob, ob], T.abb.vVVoOO, optimize=True)
    )
    dT.abb.vVVooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('MK,aCBijM->aBCijK', H.b.oo[Ob, Ob], T.abb.vVVooO, optimize=True)
    )
    dT.abb.vVVooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('ae,eCBijK->aBCijK', H.a.vv[va, va], T.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('aE,ECBijK->aBCijK', H.a.vv[va, Va], T.abb.VVVooO, optimize=True)
    )
    dT.abb.vVVooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('Be,aCeijK->aBCijK', H.b.vv[Vb, vb], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('BE,aCEijK->aBCijK', H.b.vv[Vb, Vb], T.abb.vVVooO, optimize=True)
    )
    dT.abb.vVVooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('MnjK,aCBinM->aBCijK', H.bb.oooo[Ob, ob, ob, Ob], T.abb.vVVooO, optimize=True)
            - 0.5 * np.einsum('MNjK,aCBiMN->aBCijK', H.bb.oooo[Ob, Ob, ob, Ob], T.abb.vVVoOO, optimize=True)
    )
    dT.abb.vVVooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mnij,aCBmnK->aBCijK', H.ab.oooo[oa, ob, oa, ob], T.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('mNij,aCBmNK->aBCijK', H.ab.oooo[oa, Ob, oa, ob], T.abb.vVVoOO, optimize=True)
            - 1.0 * np.einsum('Mnij,aCBMnK->aBCijK', H.ab.oooo[Oa, ob, oa, ob], T.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('MNij,aCBMNK->aBCijK', H.ab.oooo[Oa, Ob, oa, ob], T.abb.vVVOOO, optimize=True)
    )
    dT.abb.vVVooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mNiK,aCBmjN->aBCijK', H.ab.oooo[oa, Ob, oa, Ob], T.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('MniK,aCBMnj->aBCijK', H.ab.oooo[Oa, ob, oa, Ob], T.abb.vVVOoo, optimize=True)
            - 1.0 * np.einsum('MNiK,aCBMjN->aBCijK', H.ab.oooo[Oa, Ob, oa, Ob], T.abb.vVVOoO, optimize=True)
    )
    dT.abb.vVVooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('BCEf,aEfijK->aBCijK', H.bb.vvvv[Vb, Vb, Vb, vb], T.abb.vVvooO, optimize=True)
            - 0.5 * np.einsum('BCEF,aFEijK->aBCijK', H.bb.vvvv[Vb, Vb, Vb, Vb], T.abb.vVVooO, optimize=True)
    )
    dT.abb.vVVooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('aBef,eCfijK->aBCijK', H.ab.vvvv[va, Vb, va, vb], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('aBeF,eCFijK->aBCijK', H.ab.vvvv[va, Vb, va, Vb], T.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('aBEf,ECfijK->aBCijK', H.ab.vvvv[va, Vb, Va, vb], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('aBEF,ECFijK->aBCijK', H.ab.vvvv[va, Vb, Va, Vb], T.abb.VVVooO, optimize=True)
    )
    dT.abb.vVVooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('amie,eCBmjK->aBCijK', H.aa.voov[va, oa, oa, va], T.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('aMie,eCBMjK->aBCijK', H.aa.voov[va, Oa, oa, va], T.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('amiE,ECBmjK->aBCijK', H.aa.voov[va, oa, oa, Va], T.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('aMiE,ECBMjK->aBCijK', H.aa.voov[va, Oa, oa, Va], T.abb.VVVOoO, optimize=True)
    )
    dT.abb.vVVooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('amie,CBemjK->aBCijK', H.ab.voov[va, ob, oa, vb], T.bbb.VVvooO, optimize=True)
            + 1.0 * np.einsum('aMie,CBejMK->aBCijK', H.ab.voov[va, Ob, oa, vb], T.bbb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('amiE,CBEmjK->aBCijK', H.ab.voov[va, ob, oa, Vb], T.bbb.VVVooO, optimize=True)
            + 1.0 * np.einsum('aMiE,CBEjMK->aBCijK', H.ab.voov[va, Ob, oa, Vb], T.bbb.VVVoOO, optimize=True)
    )
    dT.abb.vVVooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mBej,eaCimK->aBCijK', H.ab.ovvo[oa, Vb, va, ob], T.aab.vvVooO, optimize=True)
            - 1.0 * np.einsum('MBej,eaCiMK->aBCijK', H.ab.ovvo[Oa, Vb, va, ob], T.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('mBEj,EaCimK->aBCijK', H.ab.ovvo[oa, Vb, Va, ob], T.aab.VvVooO, optimize=True)
            - 1.0 * np.einsum('MBEj,EaCiMK->aBCijK', H.ab.ovvo[Oa, Vb, Va, ob], T.aab.VvVoOO, optimize=True)
    )
    dT.abb.vVVooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('MBeK,eaCiMj->aBCijK', H.ab.ovvo[Oa, Vb, va, Ob], T.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('MBEK,EaCiMj->aBCijK', H.ab.ovvo[Oa, Vb, Va, Ob], T.aab.VvVoOo, optimize=True)
    )
    dT.abb.vVVooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('Bmje,aCeimK->aBCijK', H.bb.voov[Vb, ob, ob, vb], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('BMje,aCeiMK->aBCijK', H.bb.voov[Vb, Ob, ob, vb], T.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('BmjE,aCEimK->aBCijK', H.bb.voov[Vb, ob, ob, Vb], T.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('BMjE,aCEiMK->aBCijK', H.bb.voov[Vb, Ob, ob, Vb], T.abb.vVVoOO, optimize=True)
    )
    dT.abb.vVVooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('BMKe,aCeijM->aBCijK', H.bb.voov[Vb, Ob, Ob, vb], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('BMKE,aCEijM->aBCijK', H.bb.voov[Vb, Ob, Ob, Vb], T.abb.vVVooO, optimize=True)
    )
    dT.abb.vVVooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mBie,aCemjK->aBCijK', H.ab.ovov[oa, Vb, oa, vb], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('MBie,aCeMjK->aBCijK', H.ab.ovov[Oa, Vb, oa, vb], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('mBiE,aCEmjK->aBCijK', H.ab.ovov[oa, Vb, oa, Vb], T.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('MBiE,aCEMjK->aBCijK', H.ab.ovov[Oa, Vb, oa, Vb], T.abb.vVVOoO, optimize=True)
    )
    dT.abb.vVVooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('amej,eCBimK->aBCijK', H.ab.vovo[va, ob, va, ob], T.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('aMej,eCBiMK->aBCijK', H.ab.vovo[va, Ob, va, ob], T.abb.vVVoOO, optimize=True)
            + 1.0 * np.einsum('amEj,ECBimK->aBCijK', H.ab.vovo[va, ob, Va, ob], T.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('aMEj,ECBiMK->aBCijK', H.ab.vovo[va, Ob, Va, ob], T.abb.VVVoOO, optimize=True)
    )
    dT.abb.vVVooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('aMeK,eCBijM->aBCijK', H.ab.vovo[va, Ob, va, Ob], T.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('aMEK,ECBijM->aBCijK', H.ab.vovo[va, Ob, Va, Ob], T.abb.VVVooO, optimize=True)
    )

    dT.abb.vVVooO -= np.transpose(dT.abb.vVVooO, (0, 2, 1, 3, 4, 5))

    return dT


def update(T, dT, H, shift, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.abb.vVVooO, dT.abb.vVVooO = cc_active_loops.update_t3c_011001(
        T.abb.vVVooO,
        dT.abb.vVVooO,
        H.a.oo[Oa, Oa],
        H.a.vv[Va, Va],
        H.a.oo[oa, oa],
        H.a.vv[va, va],
        H.b.oo[Ob, Ob],
        H.b.vv[Vb, Vb],
        H.b.oo[ob, ob],
        H.b.vv[vb, vb],
        shift,
    )

    return T, dT
