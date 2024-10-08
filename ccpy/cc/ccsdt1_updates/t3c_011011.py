import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import cc_active_loops

def build(T, dT, H, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    # MM(2,3)
    dT.abb.vVVoOO = (2.0 / 4.0) * (
            +1.0 * np.einsum('aBie,eCJK->aBCiJK', H.ab.vvov[va, Vb, oa, :], T.bb[:, Vb, Ob, Ob], optimize=True)
    )
    dT.abb.vVVoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('amiJ,BCmK->aBCiJK', H.ab.vooo[va, :, oa, Ob], T.bb[Vb, Vb, :, Ob], optimize=True)
    )
    dT.abb.vVVoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('CBKe,aeiJ->aBCiJK', H.bb.vvov[Vb, Vb, Ob, :], T.ab[va, :, oa, Ob], optimize=True)
    )
    dT.abb.vVVoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('CmKJ,aBim->aBCiJK', H.bb.vooo[Vb, :, Ob, Ob], T.ab[va, Vb, oa, :], optimize=True)
    )
    dT.abb.vVVoOO += (4.0 / 4.0) * (
            +1.0 * np.einsum('aBeJ,eCiK->aBCiJK', H.ab.vvvo[va, Vb, :, Ob], T.ab[:, Vb, oa, Ob], optimize=True)
    )
    dT.abb.vVVoOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('mBiJ,aCmK->aBCiJK', H.ab.ovoo[:, Vb, oa, Ob], T.ab[va, Vb, :, Ob], optimize=True)
    )
    # (H(2) * T3)_C
    dT.abb.vVVoOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('mi,aCBmJK->aBCiJK', H.a.oo[oa, oa], T.abb.vVVoOO, optimize=True)
            + 1.0 * np.einsum('Mi,aCBMJK->aBCiJK', H.a.oo[Oa, oa], T.abb.vVVOOO, optimize=True)
    )
    dT.abb.vVVoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mJ,aCBimK->aBCiJK', H.b.oo[ob, Ob], T.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('MJ,aCBiMK->aBCiJK', H.b.oo[Ob, Ob], T.abb.vVVoOO, optimize=True)
    )
    dT.abb.vVVoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('ae,eCBiJK->aBCiJK', H.a.vv[va, va], T.abb.vVVoOO, optimize=True)
            - 1.0 * np.einsum('aE,ECBiJK->aBCiJK', H.a.vv[va, Va], T.abb.VVVoOO, optimize=True)
    )
    dT.abb.vVVoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('Be,aCeiJK->aBCiJK', H.b.vv[Vb, vb], T.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('BE,aCEiJK->aBCiJK', H.b.vv[Vb, Vb], T.abb.vVVoOO, optimize=True)
    )
    dT.abb.vVVoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('mNJK,aCBimN->aBCiJK', H.bb.oooo[ob, Ob, Ob, Ob], T.abb.vVVooO, optimize=True)
            - 0.5 * np.einsum('MNJK,aCBiMN->aBCiJK', H.bb.oooo[Ob, Ob, Ob, Ob], T.abb.vVVoOO, optimize=True)
    )
    dT.abb.vVVoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('mniJ,aCBmnK->aBCiJK', H.ab.oooo[oa, ob, oa, Ob], T.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('MniJ,aCBMnK->aBCiJK', H.ab.oooo[Oa, ob, oa, Ob], T.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('mNiJ,aCBmNK->aBCiJK', H.ab.oooo[oa, Ob, oa, Ob], T.abb.vVVoOO, optimize=True)
            - 1.0 * np.einsum('MNiJ,aCBMNK->aBCiJK', H.ab.oooo[Oa, Ob, oa, Ob], T.abb.vVVOOO, optimize=True)
    )
    dT.abb.vVVoOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('BCEf,aEfiJK->aBCiJK', H.bb.vvvv[Vb, Vb, Vb, vb], T.abb.vVvoOO, optimize=True)
            - 0.5 * np.einsum('BCEF,aFEiJK->aBCiJK', H.bb.vvvv[Vb, Vb, Vb, Vb], T.abb.vVVoOO, optimize=True)
    )
    dT.abb.vVVoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('aBef,eCfiJK->aBCiJK', H.ab.vvvv[va, Vb, va, vb], T.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('aBeF,eCFiJK->aBCiJK', H.ab.vvvv[va, Vb, va, Vb], T.abb.vVVoOO, optimize=True)
            - 1.0 * np.einsum('aBEf,ECfiJK->aBCiJK', H.ab.vvvv[va, Vb, Va, vb], T.abb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('aBEF,ECFiJK->aBCiJK', H.ab.vvvv[va, Vb, Va, Vb], T.abb.VVVoOO, optimize=True)
    )
    dT.abb.vVVoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('amie,eCBmJK->aBCiJK', H.aa.voov[va, oa, oa, va], T.abb.vVVoOO, optimize=True)
            - 1.0 * np.einsum('aMie,eCBMJK->aBCiJK', H.aa.voov[va, Oa, oa, va], T.abb.vVVOOO, optimize=True)
            - 1.0 * np.einsum('amiE,ECBmJK->aBCiJK', H.aa.voov[va, oa, oa, Va], T.abb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('aMiE,ECBMJK->aBCiJK', H.aa.voov[va, Oa, oa, Va], T.abb.VVVOOO, optimize=True)
    )
    dT.abb.vVVoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('amie,CBemJK->aBCiJK', H.ab.voov[va, ob, oa, vb], T.bbb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('aMie,CBeMJK->aBCiJK', H.ab.voov[va, Ob, oa, vb], T.bbb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('amiE,CBEmJK->aBCiJK', H.ab.voov[va, ob, oa, Vb], T.bbb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('aMiE,CBEMJK->aBCiJK', H.ab.voov[va, Ob, oa, Vb], T.bbb.VVVOOO, optimize=True)
    )
    dT.abb.vVVoOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('mBeJ,eaCimK->aBCiJK', H.ab.ovvo[oa, Vb, va, Ob], T.aab.vvVooO, optimize=True)
            - 1.0 * np.einsum('MBeJ,eaCiMK->aBCiJK', H.ab.ovvo[Oa, Vb, va, Ob], T.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('mBEJ,EaCimK->aBCiJK', H.ab.ovvo[oa, Vb, Va, Ob], T.aab.VvVooO, optimize=True)
            - 1.0 * np.einsum('MBEJ,EaCiMK->aBCiJK', H.ab.ovvo[Oa, Vb, Va, Ob], T.aab.VvVoOO, optimize=True)
    )
    dT.abb.vVVoOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('BmJe,aCeimK->aBCiJK', H.bb.voov[Vb, ob, Ob, vb], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('BMJe,aCeiMK->aBCiJK', H.bb.voov[Vb, Ob, Ob, vb], T.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('BmJE,aCEimK->aBCiJK', H.bb.voov[Vb, ob, Ob, Vb], T.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('BMJE,aCEiMK->aBCiJK', H.bb.voov[Vb, Ob, Ob, Vb], T.abb.vVVoOO, optimize=True)
    )
    dT.abb.vVVoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mBie,aCemJK->aBCiJK', H.ab.ovov[oa, Vb, oa, vb], T.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('MBie,aCeMJK->aBCiJK', H.ab.ovov[Oa, Vb, oa, vb], T.abb.vVvOOO, optimize=True)
            + 1.0 * np.einsum('mBiE,aCEmJK->aBCiJK', H.ab.ovov[oa, Vb, oa, Vb], T.abb.vVVoOO, optimize=True)
            + 1.0 * np.einsum('MBiE,aCEMJK->aBCiJK', H.ab.ovov[Oa, Vb, oa, Vb], T.abb.vVVOOO, optimize=True)
    )
    dT.abb.vVVoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('ameJ,eCBimK->aBCiJK', H.ab.vovo[va, ob, va, Ob], T.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('aMeJ,eCBiMK->aBCiJK', H.ab.vovo[va, Ob, va, Ob], T.abb.vVVoOO, optimize=True)
            + 1.0 * np.einsum('amEJ,ECBimK->aBCiJK', H.ab.vovo[va, ob, Va, Ob], T.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('aMEJ,ECBiMK->aBCiJK', H.ab.vovo[va, Ob, Va, Ob], T.abb.VVVoOO, optimize=True)
    )

    dT.abb.vVVoOO -= np.transpose(dT.abb.vVVoOO, (0, 2, 1, 3, 4, 5))
    dT.abb.vVVoOO -= np.transpose(dT.abb.vVVoOO, (0, 1, 2, 3, 5, 4))

    return dT


def update(T, dT, H, shift, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.abb.vVVoOO, dT.abb.vVVoOO = cc_active_loops.update_t3c_011011(
        T.abb.vVVoOO,
        dT.abb.vVVoOO,
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
