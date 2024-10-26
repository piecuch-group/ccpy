import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops

def build(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.abb.VVVoOO = (2.0 / 4.0) * (
            +1.0 * np.einsum('ABie,eCJK->ABCiJK', X.ab.vvov[Va, Vb, oa, :], T.bb[:, Vb, Ob, Ob], optimize=True)
    )
    dR.abb.VVVoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('AmiJ,BCmK->ABCiJK', X.ab.vooo[Va, :, oa, Ob], T.bb[Vb, Vb, :, Ob], optimize=True)
    )
    dR.abb.VVVoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('CBKe,AeiJ->ABCiJK', X.bb.vvov[Vb, Vb, Ob, :], T.ab[Va, :, oa, Ob], optimize=True)
    )
    dR.abb.VVVoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('CmKJ,ABim->ABCiJK', X.bb.vooo[Vb, :, Ob, Ob], T.ab[Va, Vb, oa, :], optimize=True)
    )
    dR.abb.VVVoOO += (4.0 / 4.0) * (
            +1.0 * np.einsum('ABeJ,eCiK->ABCiJK', X.ab.vvvo[Va, Vb, :, Ob], T.ab[:, Vb, oa, Ob], optimize=True)
    )
    dR.abb.VVVoOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('mBiJ,ACmK->ABCiJK', X.ab.ovoo[:, Vb, oa, Ob], T.ab[Va, Vb, :, Ob], optimize=True)
    )
    dR.abb.VVVoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('ABie,eCJK->ABCiJK', H.ab.vvov[Va, Vb, oa, :], R.bb[:, Vb, Ob, Ob], optimize=True)
    )
    dR.abb.VVVoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('AmiJ,BCmK->ABCiJK', H.ab.vooo[Va, :, oa, Ob], R.bb[Vb, Vb, :, Ob], optimize=True)
    )
    dR.abb.VVVoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('CBKe,AeiJ->ABCiJK', H.bb.vvov[Vb, Vb, Ob, :], R.ab[Va, :, oa, Ob], optimize=True)
    )
    dR.abb.VVVoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('CmKJ,ABim->ABCiJK', H.bb.vooo[Vb, :, Ob, Ob], R.ab[Va, Vb, oa, :], optimize=True)
    )
    dR.abb.VVVoOO += (4.0 / 4.0) * (
            +1.0 * np.einsum('ABeJ,eCiK->ABCiJK', H.ab.vvvo[Va, Vb, :, Ob], R.ab[:, Vb, oa, Ob], optimize=True)
    )
    dR.abb.VVVoOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('mBiJ,ACmK->ABCiJK', H.ab.ovoo[:, Vb, oa, Ob], R.ab[Va, Vb, :, Ob], optimize=True)
    )

    dR.abb.VVVoOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('mi,ACBmJK->ABCiJK', X.a.oo[oa, oa], T.abb.VVVoOO, optimize=True)
            + 1.0 * np.einsum('Mi,ACBMJK->ABCiJK', X.a.oo[Oa, oa], T.abb.VVVOOO, optimize=True)
    )
    dR.abb.VVVoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mJ,ACBimK->ABCiJK', X.b.oo[ob, Ob], T.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('MJ,ACBiMK->ABCiJK', X.b.oo[Ob, Ob], T.abb.VVVoOO, optimize=True)
    )
    dR.abb.VVVoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('Ae,eCBiJK->ABCiJK', X.a.vv[Va, va], T.abb.vVVoOO, optimize=True)
            - 1.0 * np.einsum('AE,ECBiJK->ABCiJK', X.a.vv[Va, Va], T.abb.VVVoOO, optimize=True)
    )
    dR.abb.VVVoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('Be,ACeiJK->ABCiJK', X.b.vv[Vb, vb], T.abb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('BE,ACEiJK->ABCiJK', X.b.vv[Vb, Vb], T.abb.VVVoOO, optimize=True)
    )
    dR.abb.VVVoOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('MnJK,ACBinM->ABCiJK', X.bb.oooo[Ob, ob, Ob, Ob], T.abb.VVVooO, optimize=True)
            - 0.5 * np.einsum('MNJK,ACBiMN->ABCiJK', X.bb.oooo[Ob, Ob, Ob, Ob], T.abb.VVVoOO, optimize=True)
    )
    dR.abb.VVVoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('mniJ,ACBmnK->ABCiJK', X.ab.oooo[oa, ob, oa, Ob], T.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('mNiJ,ACBmNK->ABCiJK', X.ab.oooo[oa, Ob, oa, Ob], T.abb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('MniJ,ACBMnK->ABCiJK', X.ab.oooo[Oa, ob, oa, Ob], T.abb.VVVOoO, optimize=True)
            - 1.0 * np.einsum('MNiJ,ACBMNK->ABCiJK', X.ab.oooo[Oa, Ob, oa, Ob], T.abb.VVVOOO, optimize=True)
    )
    dR.abb.VVVoOO += (1.0 / 4.0) * (
            -0.5 * np.einsum('BCef,AfeiJK->ABCiJK', X.bb.vvvv[Vb, Vb, vb, vb], T.abb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('BCEf,AEfiJK->ABCiJK', X.bb.vvvv[Vb, Vb, Vb, vb], T.abb.VVvoOO, optimize=True)
            - 0.5 * np.einsum('BCEF,AFEiJK->ABCiJK', X.bb.vvvv[Vb, Vb, Vb, Vb], T.abb.VVVoOO, optimize=True)
    )
    dR.abb.VVVoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('ABef,eCfiJK->ABCiJK', X.ab.vvvv[Va, Vb, va, vb], T.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('ABeF,eCFiJK->ABCiJK', X.ab.vvvv[Va, Vb, va, Vb], T.abb.vVVoOO, optimize=True)
            - 1.0 * np.einsum('ABEf,ECfiJK->ABCiJK', X.ab.vvvv[Va, Vb, Va, vb], T.abb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('ABEF,ECFiJK->ABCiJK', X.ab.vvvv[Va, Vb, Va, Vb], T.abb.VVVoOO, optimize=True)
    )
    dR.abb.VVVoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('Amie,eCBmJK->ABCiJK', X.aa.voov[Va, oa, oa, va], T.abb.vVVoOO, optimize=True)
            - 1.0 * np.einsum('AMie,eCBMJK->ABCiJK', X.aa.voov[Va, Oa, oa, va], T.abb.vVVOOO, optimize=True)
            - 1.0 * np.einsum('AmiE,ECBmJK->ABCiJK', X.aa.voov[Va, oa, oa, Va], T.abb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('AMiE,ECBMJK->ABCiJK', X.aa.voov[Va, Oa, oa, Va], T.abb.VVVOOO, optimize=True)
    )
    dR.abb.VVVoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('Amie,CBemJK->ABCiJK', X.ab.voov[Va, ob, oa, vb], T.bbb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('AMie,CBeMJK->ABCiJK', X.ab.voov[Va, Ob, oa, vb], T.bbb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('AmiE,CBEmJK->ABCiJK', X.ab.voov[Va, ob, oa, Vb], T.bbb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('AMiE,CBEMJK->ABCiJK', X.ab.voov[Va, Ob, oa, Vb], T.bbb.VVVOOO, optimize=True)
    )
    dR.abb.VVVoOO += (4.0 / 4.0) * (
            +1.0 * np.einsum('mBeJ,AeCimK->ABCiJK', X.ab.ovvo[oa, Vb, va, Ob], T.aab.VvVooO, optimize=True)
            + 1.0 * np.einsum('MBeJ,AeCiMK->ABCiJK', X.ab.ovvo[Oa, Vb, va, Ob], T.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('mBEJ,EACimK->ABCiJK', X.ab.ovvo[oa, Vb, Va, Ob], T.aab.VVVooO, optimize=True)
            - 1.0 * np.einsum('MBEJ,EACiMK->ABCiJK', X.ab.ovvo[Oa, Vb, Va, Ob], T.aab.VVVoOO, optimize=True)
    )
    dR.abb.VVVoOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('BmJe,ACeimK->ABCiJK', X.bb.voov[Vb, ob, Ob, vb], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('BMJe,ACeiMK->ABCiJK', X.bb.voov[Vb, Ob, Ob, vb], T.abb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('BmJE,ACEimK->ABCiJK', X.bb.voov[Vb, ob, Ob, Vb], T.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('BMJE,ACEiMK->ABCiJK', X.bb.voov[Vb, Ob, Ob, Vb], T.abb.VVVoOO, optimize=True)
    )
    dR.abb.VVVoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mBie,ACemJK->ABCiJK', X.ab.ovov[oa, Vb, oa, vb], T.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('MBie,ACeMJK->ABCiJK', X.ab.ovov[Oa, Vb, oa, vb], T.abb.VVvOOO, optimize=True)
            + 1.0 * np.einsum('mBiE,ACEmJK->ABCiJK', X.ab.ovov[oa, Vb, oa, Vb], T.abb.VVVoOO, optimize=True)
            + 1.0 * np.einsum('MBiE,ACEMJK->ABCiJK', X.ab.ovov[Oa, Vb, oa, Vb], T.abb.VVVOOO, optimize=True)
    )
    dR.abb.VVVoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('AmeJ,eCBimK->ABCiJK', X.ab.vovo[Va, ob, va, Ob], T.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('AMeJ,eCBiMK->ABCiJK', X.ab.vovo[Va, Ob, va, Ob], T.abb.vVVoOO, optimize=True)
            + 1.0 * np.einsum('AmEJ,ECBimK->ABCiJK', X.ab.vovo[Va, ob, Va, Ob], T.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('AMEJ,ECBiMK->ABCiJK', X.ab.vovo[Va, Ob, Va, Ob], T.abb.VVVoOO, optimize=True)
    )
    dR.abb.VVVoOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('mi,ACBmJK->ABCiJK', H.a.oo[oa, oa], R.abb.VVVoOO, optimize=True)
            + 1.0 * np.einsum('Mi,ACBMJK->ABCiJK', H.a.oo[Oa, oa], R.abb.VVVOOO, optimize=True)
    )
    dR.abb.VVVoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mJ,ACBimK->ABCiJK', H.b.oo[ob, Ob], R.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('MJ,ACBiMK->ABCiJK', H.b.oo[Ob, Ob], R.abb.VVVoOO, optimize=True)
    )
    dR.abb.VVVoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('Ae,eCBiJK->ABCiJK', H.a.vv[Va, va], R.abb.vVVoOO, optimize=True)
            - 1.0 * np.einsum('AE,ECBiJK->ABCiJK', H.a.vv[Va, Va], R.abb.VVVoOO, optimize=True)
    )
    dR.abb.VVVoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('Be,ACeiJK->ABCiJK', H.b.vv[Vb, vb], R.abb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('BE,ACEiJK->ABCiJK', H.b.vv[Vb, Vb], R.abb.VVVoOO, optimize=True)
    )
    dR.abb.VVVoOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('MnJK,ACBinM->ABCiJK', H.bb.oooo[Ob, ob, Ob, Ob], R.abb.VVVooO, optimize=True)
            - 0.5 * np.einsum('MNJK,ACBiMN->ABCiJK', H.bb.oooo[Ob, Ob, Ob, Ob], R.abb.VVVoOO, optimize=True)
    )
    dR.abb.VVVoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('mniJ,ACBmnK->ABCiJK', H.ab.oooo[oa, ob, oa, Ob], R.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('mNiJ,ACBmNK->ABCiJK', H.ab.oooo[oa, Ob, oa, Ob], R.abb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('MniJ,ACBMnK->ABCiJK', H.ab.oooo[Oa, ob, oa, Ob], R.abb.VVVOoO, optimize=True)
            - 1.0 * np.einsum('MNiJ,ACBMNK->ABCiJK', H.ab.oooo[Oa, Ob, oa, Ob], R.abb.VVVOOO, optimize=True)
    )
    dR.abb.VVVoOO += (1.0 / 4.0) * (
            -0.5 * np.einsum('BCef,AfeiJK->ABCiJK', H.bb.vvvv[Vb, Vb, vb, vb], R.abb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('BCEf,AEfiJK->ABCiJK', H.bb.vvvv[Vb, Vb, Vb, vb], R.abb.VVvoOO, optimize=True)
            - 0.5 * np.einsum('BCEF,AFEiJK->ABCiJK', H.bb.vvvv[Vb, Vb, Vb, Vb], R.abb.VVVoOO, optimize=True)
    )
    dR.abb.VVVoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('ABef,eCfiJK->ABCiJK', H.ab.vvvv[Va, Vb, va, vb], R.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('ABeF,eCFiJK->ABCiJK', H.ab.vvvv[Va, Vb, va, Vb], R.abb.vVVoOO, optimize=True)
            - 1.0 * np.einsum('ABEf,ECfiJK->ABCiJK', H.ab.vvvv[Va, Vb, Va, vb], R.abb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('ABEF,ECFiJK->ABCiJK', H.ab.vvvv[Va, Vb, Va, Vb], R.abb.VVVoOO, optimize=True)
    )
    dR.abb.VVVoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('Amie,eCBmJK->ABCiJK', H.aa.voov[Va, oa, oa, va], R.abb.vVVoOO, optimize=True)
            - 1.0 * np.einsum('AMie,eCBMJK->ABCiJK', H.aa.voov[Va, Oa, oa, va], R.abb.vVVOOO, optimize=True)
            - 1.0 * np.einsum('AmiE,ECBmJK->ABCiJK', H.aa.voov[Va, oa, oa, Va], R.abb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('AMiE,ECBMJK->ABCiJK', H.aa.voov[Va, Oa, oa, Va], R.abb.VVVOOO, optimize=True)
    )
    dR.abb.VVVoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('Amie,CBemJK->ABCiJK', H.ab.voov[Va, ob, oa, vb], R.bbb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('AMie,CBeMJK->ABCiJK', H.ab.voov[Va, Ob, oa, vb], R.bbb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('AmiE,CBEmJK->ABCiJK', H.ab.voov[Va, ob, oa, Vb], R.bbb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('AMiE,CBEMJK->ABCiJK', H.ab.voov[Va, Ob, oa, Vb], R.bbb.VVVOOO, optimize=True)
    )
    dR.abb.VVVoOO += (4.0 / 4.0) * (
            +1.0 * np.einsum('mBeJ,AeCimK->ABCiJK', H.ab.ovvo[oa, Vb, va, Ob], R.aab.VvVooO, optimize=True)
            + 1.0 * np.einsum('MBeJ,AeCiMK->ABCiJK', H.ab.ovvo[Oa, Vb, va, Ob], R.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('mBEJ,EACimK->ABCiJK', H.ab.ovvo[oa, Vb, Va, Ob], R.aab.VVVooO, optimize=True)
            - 1.0 * np.einsum('MBEJ,EACiMK->ABCiJK', H.ab.ovvo[Oa, Vb, Va, Ob], R.aab.VVVoOO, optimize=True)
    )
    dR.abb.VVVoOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('BmJe,ACeimK->ABCiJK', H.bb.voov[Vb, ob, Ob, vb], R.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('BMJe,ACeiMK->ABCiJK', H.bb.voov[Vb, Ob, Ob, vb], R.abb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('BmJE,ACEimK->ABCiJK', H.bb.voov[Vb, ob, Ob, Vb], R.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('BMJE,ACEiMK->ABCiJK', H.bb.voov[Vb, Ob, Ob, Vb], R.abb.VVVoOO, optimize=True)
    )
    dR.abb.VVVoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mBie,ACemJK->ABCiJK', H.ab.ovov[oa, Vb, oa, vb], R.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('MBie,ACeMJK->ABCiJK', H.ab.ovov[Oa, Vb, oa, vb], R.abb.VVvOOO, optimize=True)
            + 1.0 * np.einsum('mBiE,ACEmJK->ABCiJK', H.ab.ovov[oa, Vb, oa, Vb], R.abb.VVVoOO, optimize=True)
            + 1.0 * np.einsum('MBiE,ACEMJK->ABCiJK', H.ab.ovov[Oa, Vb, oa, Vb], R.abb.VVVOOO, optimize=True)
    )
    dR.abb.VVVoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('AmeJ,eCBimK->ABCiJK', H.ab.vovo[Va, ob, va, Ob], R.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('AMeJ,eCBiMK->ABCiJK', H.ab.vovo[Va, Ob, va, Ob], R.abb.vVVoOO, optimize=True)
            + 1.0 * np.einsum('AmEJ,ECBimK->ABCiJK', H.ab.vovo[Va, ob, Va, Ob], R.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('AMEJ,ECBiMK->ABCiJK', H.ab.vovo[Va, Ob, Va, Ob], R.abb.VVVoOO, optimize=True)
    )

    dR.abb.VVVoOO -= np.transpose(dR.abb.VVVoOO, (0, 2, 1, 3, 4, 5))
    dR.abb.VVVoOO -= np.transpose(dR.abb.VVVoOO, (0, 1, 2, 3, 5, 4))

    return dR

def update(R, omega, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    R.abb.VVVoOO = eomcc_active_loops.update_r3c_111011(
        R.abb.VVVoOO,
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

