import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops

def build(dR, R, T, H, X, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.aaa.VVVoOO = (6.0 / 12.0) * (
            -1.0 * np.einsum('BmJi,ACmK->ABCiJK', X.aa.vooo[Va, :, Oa, oa], T.aa[Va, Va, :, Oa], optimize=True)
    )
    dR.aaa.VVVoOO += (3.0 / 12.0) * (
            +1.0 * np.einsum('BmJK,ACmi->ABCiJK', X.aa.vooo[Va, :, Oa, Oa], T.aa[Va, Va, :, oa], optimize=True)
    )
    dR.aaa.VVVoOO += (6.0 / 12.0) * (
            -1.0 * np.einsum('BmJi,ACmK->ABCiJK', H.aa.vooo[Va, :, Oa, oa], R.aa[Va, Va, :, Oa], optimize=True)
    )
    dR.aaa.VVVoOO += (3.0 / 12.0) * (
            +1.0 * np.einsum('BmJK,ACmi->ABCiJK', H.aa.vooo[Va, :, Oa, Oa], R.aa[Va, Va, :, oa], optimize=True)
    )
    dR.aaa.VVVoOO += (6.0 / 12.0) * (
            +1.0 * np.einsum('BAJe,eCiK->ABCiJK', X.aa.vvov[Va, Va, Oa, :], T.aa[:, Va, oa, Oa], optimize=True)
    )
    dR.aaa.VVVoOO += (3.0 / 12.0) * (
            -1.0 * np.einsum('BAie,eCJK->ABCiJK', X.aa.vvov[Va, Va, oa, :], T.aa[:, Va, Oa, Oa], optimize=True)
    )
    dR.aaa.VVVoOO += (6.0 / 12.0) * (
            +1.0 * np.einsum('BAJe,eCiK->ABCiJK', H.aa.vvov[Va, Va, Oa, :], R.aa[:, Va, oa, Oa], optimize=True)
    )
    dR.aaa.VVVoOO += (3.0 / 12.0) * (
            -1.0 * np.einsum('BAie,eCJK->ABCiJK', H.aa.vvov[Va, Va, oa, :], R.aa[:, Va, Oa, Oa], optimize=True)
    )

    dR.aaa.VVVoOO += (3.0 / 12.0) * (
            +1.0 * np.einsum('Be,CAeiJK->ABCiJK', X.a.vv[Va, va], T.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('BE,CEAiJK->ABCiJK', X.a.vv[Va, Va], T.aaa.VVVoOO, optimize=True)
    )
    dR.aaa.VVVoOO += (2.0 / 12.0) * (
            +1.0 * np.einsum('mJ,CBAimK->ABCiJK', X.a.oo[oa, Oa], T.aaa.VVVooO, optimize=True)
            + 1.0 * np.einsum('MJ,CBAiMK->ABCiJK', X.a.oo[Oa, Oa], T.aaa.VVVoOO, optimize=True)
    )
    dR.aaa.VVVoOO += (1.0 / 12.0) * (
            +1.0 * np.einsum('mi,CBAmJK->ABCiJK', X.a.oo[oa, oa], T.aaa.VVVoOO, optimize=True)
            - 1.0 * np.einsum('Mi,CBAJMK->ABCiJK', X.a.oo[Oa, oa], T.aaa.VVVOOO, optimize=True)
    )
    dR.aaa.VVVoOO += (2.0 / 12.0) * (
            -0.5 * np.einsum('mniJ,CBAmnK->ABCiJK', X.aa.oooo[oa, oa, oa, Oa], T.aaa.VVVooO, optimize=True)
            + 1.0 * np.einsum('MniJ,CBAnMK->ABCiJK', X.aa.oooo[Oa, oa, oa, Oa], T.aaa.VVVoOO, optimize=True)
            - 0.5 * np.einsum('MNiJ,CBAMNK->ABCiJK', X.aa.oooo[Oa, Oa, oa, Oa], T.aaa.VVVOOO, optimize=True)
    )
    dR.aaa.VVVoOO += (1.0 / 12.0) * (
            +1.0 * np.einsum('MnKJ,CBAniM->ABCiJK', X.aa.oooo[Oa, oa, Oa, Oa], T.aaa.VVVooO, optimize=True)
            + 0.5 * np.einsum('MNKJ,CBAiMN->ABCiJK', X.aa.oooo[Oa, Oa, Oa, Oa], T.aaa.VVVoOO, optimize=True)
    )
    dR.aaa.VVVoOO += (3.0 / 12.0) * (
            -0.5 * np.einsum('ABef,CfeiJK->ABCiJK', X.aa.vvvv[Va, Va, va, va], T.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('ABeF,CFeiJK->ABCiJK', X.aa.vvvv[Va, Va, va, Va], T.aaa.VVvoOO, optimize=True)
            - 0.5 * np.einsum('ABEF,CFEiJK->ABCiJK', X.aa.vvvv[Va, Va, Va, Va], T.aaa.VVVoOO, optimize=True)
    )
    dR.aaa.VVVoOO += (6.0 / 12.0) * (
            +1.0 * np.einsum('BmJe,CAeimK->ABCiJK', X.aa.voov[Va, oa, Oa, va], T.aaa.VVvooO, optimize=True)
            - 1.0 * np.einsum('BmJE,CEAimK->ABCiJK', X.aa.voov[Va, oa, Oa, Va], T.aaa.VVVooO, optimize=True)
            + 1.0 * np.einsum('BMJe,CAeiMK->ABCiJK', X.aa.voov[Va, Oa, Oa, va], T.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('BMJE,CEAiMK->ABCiJK', X.aa.voov[Va, Oa, Oa, Va], T.aaa.VVVoOO, optimize=True)
    )
    dR.aaa.VVVoOO += (3.0 / 12.0) * (
            +1.0 * np.einsum('Bmie,CAemJK->ABCiJK', X.aa.voov[Va, oa, oa, va], T.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('BmiE,CEAmJK->ABCiJK', X.aa.voov[Va, oa, oa, Va], T.aaa.VVVoOO, optimize=True)
            - 1.0 * np.einsum('BMie,CAeJMK->ABCiJK', X.aa.voov[Va, Oa, oa, va], T.aaa.VVvOOO, optimize=True)
            + 1.0 * np.einsum('BMiE,CEAJMK->ABCiJK', X.aa.voov[Va, Oa, oa, Va], T.aaa.VVVOOO, optimize=True)
    )
    dR.aaa.VVVoOO += (6.0 / 12.0) * (
            -1.0 * np.einsum('BmJe,CAeiKm->ABCiJK', X.ab.voov[Va, ob, Oa, vb], T.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('BmJE,CAEiKm->ABCiJK', X.ab.voov[Va, ob, Oa, Vb], T.aab.VVVoOo, optimize=True)
            - 1.0 * np.einsum('BMJe,CAeiKM->ABCiJK', X.ab.voov[Va, Ob, Oa, vb], T.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('BMJE,CAEiKM->ABCiJK', X.ab.voov[Va, Ob, Oa, Vb], T.aab.VVVoOO, optimize=True)
    )
    dR.aaa.VVVoOO += (3.0 / 12.0) * (
            +1.0 * np.einsum('Bmie,CAeJKm->ABCiJK', X.ab.voov[Va, ob, oa, vb], T.aab.VVvOOo, optimize=True)
            + 1.0 * np.einsum('BmiE,CAEJKm->ABCiJK', X.ab.voov[Va, ob, oa, Vb], T.aab.VVVOOo, optimize=True)
            + 1.0 * np.einsum('BMie,CAeJKM->ABCiJK', X.ab.voov[Va, Ob, oa, vb], T.aab.VVvOOO, optimize=True)
            + 1.0 * np.einsum('BMiE,CAEJKM->ABCiJK', X.ab.voov[Va, Ob, oa, Vb], T.aab.VVVOOO, optimize=True)
    )
    dR.aaa.VVVoOO += (2.0 / 12.0) * (
            +1.0 * np.einsum('mJ,CBAimK->ABCiJK', H.a.oo[oa, Oa], R.aaa.VVVooO, optimize=True)
            + 1.0 * np.einsum('MJ,CBAiMK->ABCiJK', H.a.oo[Oa, Oa], R.aaa.VVVoOO, optimize=True)
    )
    dR.aaa.VVVoOO += (1.0 / 12.0) * (
            +1.0 * np.einsum('mi,CBAmJK->ABCiJK', H.a.oo[oa, oa], R.aaa.VVVoOO, optimize=True)
            - 1.0 * np.einsum('Mi,CBAJMK->ABCiJK', H.a.oo[Oa, oa], R.aaa.VVVOOO, optimize=True)
    )
    dR.aaa.VVVoOO += (3.0 / 12.0) * (
            +1.0 * np.einsum('Be,CAeiJK->ABCiJK', H.a.vv[Va, va], R.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('BE,CEAiJK->ABCiJK', H.a.vv[Va, Va], R.aaa.VVVoOO, optimize=True)
    )
    dR.aaa.VVVoOO += (2.0 / 12.0) * (
            -0.5 * np.einsum('mniJ,CBAmnK->ABCiJK', H.aa.oooo[oa, oa, oa, Oa], R.aaa.VVVooO, optimize=True)
            + 1.0 * np.einsum('MniJ,CBAnMK->ABCiJK', H.aa.oooo[Oa, oa, oa, Oa], R.aaa.VVVoOO, optimize=True)
            - 0.5 * np.einsum('MNiJ,CBAMNK->ABCiJK', H.aa.oooo[Oa, Oa, oa, Oa], R.aaa.VVVOOO, optimize=True)
    )
    dR.aaa.VVVoOO += (1.0 / 12.0) * (
            +1.0 * np.einsum('MnKJ,CBAniM->ABCiJK', H.aa.oooo[Oa, oa, Oa, Oa], R.aaa.VVVooO, optimize=True)
            + 0.5 * np.einsum('MNKJ,CBAiMN->ABCiJK', H.aa.oooo[Oa, Oa, Oa, Oa], R.aaa.VVVoOO, optimize=True)
    )
    dR.aaa.VVVoOO += (3.0 / 12.0) * (
            -0.5 * np.einsum('ABef,CfeiJK->ABCiJK', H.aa.vvvv[Va, Va, va, va], R.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('ABeF,CFeiJK->ABCiJK', H.aa.vvvv[Va, Va, va, Va], R.aaa.VVvoOO, optimize=True)
            - 0.5 * np.einsum('ABEF,CFEiJK->ABCiJK', H.aa.vvvv[Va, Va, Va, Va], R.aaa.VVVoOO, optimize=True)
    )
    dR.aaa.VVVoOO += (3.0 / 12.0) * (
            -1.0 * np.einsum('Amie,CBemJK->ABCiJK', H.aa.voov[Va, oa, oa, va], R.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('AmiE,CBEmJK->ABCiJK', H.aa.voov[Va, oa, oa, Va], R.aaa.VVVoOO, optimize=True)
            - 1.0 * np.einsum('AMie,CBeMJK->ABCiJK', H.aa.voov[Va, Oa, oa, va], R.aaa.VVvOOO, optimize=True)
            - 1.0 * np.einsum('AMiE,CBEMJK->ABCiJK', H.aa.voov[Va, Oa, oa, Va], R.aaa.VVVOOO, optimize=True)
    )
    dR.aaa.VVVoOO += (6.0 / 12.0) * (
            +1.0 * np.einsum('AmJe,CBemiK->ABCiJK', H.aa.voov[Va, oa, Oa, va], R.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('AmJE,CBEmiK->ABCiJK', H.aa.voov[Va, oa, Oa, Va], R.aaa.VVVooO, optimize=True)
            - 1.0 * np.einsum('AMJe,CBeiMK->ABCiJK', H.aa.voov[Va, Oa, Oa, va], R.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('AMJE,CBEiMK->ABCiJK', H.aa.voov[Va, Oa, Oa, Va], R.aaa.VVVoOO, optimize=True)
    )
    dR.aaa.VVVoOO += (3.0 / 12.0) * (
            -1.0 * np.einsum('Amie,CBeJKm->ABCiJK', H.ab.voov[Va, ob, oa, vb], R.aab.VVvOOo, optimize=True)
            - 1.0 * np.einsum('AmiE,CBEJKm->ABCiJK', H.ab.voov[Va, ob, oa, Vb], R.aab.VVVOOo, optimize=True)
            - 1.0 * np.einsum('AMie,CBeJKM->ABCiJK', H.ab.voov[Va, Ob, oa, vb], R.aab.VVvOOO, optimize=True)
            - 1.0 * np.einsum('AMiE,CBEJKM->ABCiJK', H.ab.voov[Va, Ob, oa, Vb], R.aab.VVVOOO, optimize=True)
    )
    dR.aaa.VVVoOO += (6.0 / 12.0) * (
            +1.0 * np.einsum('AmJe,CBeiKm->ABCiJK', H.ab.voov[Va, ob, Oa, vb], R.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('AmJE,CBEiKm->ABCiJK', H.ab.voov[Va, ob, Oa, Vb], R.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('AMJe,CBeiKM->ABCiJK', H.ab.voov[Va, Ob, Oa, vb], R.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('AMJE,CBEiKM->ABCiJK', H.ab.voov[Va, Ob, Oa, Vb], R.aab.VVVoOO, optimize=True)
    )

    dR.aaa.VVVoOO -= np.transpose(dR.aaa.VVVoOO, (1, 0, 2, 3, 4, 5)) + np.transpose(dR.aaa.VVVoOO, (0, 2, 1, 3, 4, 5)) \
           + np.transpose(dR.aaa.VVVoOO, (2, 1, 0, 3, 4, 5)) - np.transpose(dR.aaa.VVVoOO, (1, 2, 0, 3, 4, 5)) \
           - np.transpose(dR.aaa.VVVoOO, (2, 0, 1, 3, 4, 5))

    dR.aaa.VVVoOO -= np.transpose(dR.aaa.VVVoOO, (0, 1, 2, 3, 5, 4))

    return dR

def update(R, omega, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    R.aaa.VVVoOO = eomcc_active_loops.update_r3a_111011(
        R.aaa.VVVoOO,
        omega,
        H.a.oo[Oa, Oa],
        H.a.vv[Va, Va],
        H.a.oo[oa, oa],
        H.a.vv[va, va],
        0.0,
    )

    return R