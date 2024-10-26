import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops

def build(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)


    dR.aaa.VVVOOO = (9.0 / 36.0) * (
            -1.0 * np.einsum('BmJI,ACmK->ABCIJK', X.aa.vooo[Va, :, Oa, Oa], T.aa[Va, Va, :, Oa], optimize=True)
    )
    dR.aaa.VVVOOO += (9.0 / 36.0) * (
            -1.0 * np.einsum('BmJI,ACmK->ABCIJK', H.aa.vooo[Va, :, Oa, Oa], R.aa[Va, Va, :, Oa], optimize=True)
    )
    dR.aaa.VVVOOO += (9.0 / 36.0) * (
            +1.0 * np.einsum('BAJe,eCIK->ABCIJK', X.aa.vvov[Va, Va, Oa, :], T.aa[:, Va, Oa, Oa], optimize=True)
    )
    dR.aaa.VVVOOO += (9.0 / 36.0) * (
            +1.0 * np.einsum('BAJe,eCIK->ABCIJK', H.aa.vvov[Va, Va, Oa, :], R.aa[:, Va, Oa, Oa], optimize=True)
    )

    dR.aaa.VVVOOO += (3.0 / 36.0) * (
            +1.0 * np.einsum('Be,CAeIJK->ABCIJK', X.a.vv[Va, va], T.aaa.VVvOOO, optimize=True)
            - 1.0 * np.einsum('BE,CEAIJK->ABCIJK', X.a.vv[Va, Va], T.aaa.VVVOOO, optimize=True)
    )
    dR.aaa.VVVOOO += (3.0 / 36.0) * (
            -1.0 * np.einsum('mJ,CBAmIK->ABCIJK', X.a.oo[oa, Oa], T.aaa.VVVoOO, optimize=True)
            + 1.0 * np.einsum('MJ,CBAIMK->ABCIJK', X.a.oo[Oa, Oa], T.aaa.VVVOOO, optimize=True)
    )
    dR.aaa.VVVOOO += (3.0 / 36.0) * (
            -0.5 * np.einsum('mnIJ,CBAmnK->ABCIJK', X.aa.oooo[oa, oa, Oa, Oa], T.aaa.VVVooO, optimize=True)
            + 1.0 * np.einsum('MnIJ,CBAnMK->ABCIJK', X.aa.oooo[Oa, oa, Oa, Oa], T.aaa.VVVoOO, optimize=True)
            - 0.5 * np.einsum('MNIJ,CBAMNK->ABCIJK', X.aa.oooo[Oa, Oa, Oa, Oa], T.aaa.VVVOOO, optimize=True)
    )
    dR.aaa.VVVOOO += (3.0 / 36.0) * (
            -0.5 * np.einsum('ABef,CfeIJK->ABCIJK', X.aa.vvvv[Va, Va, va, va], T.aaa.VvvOOO, optimize=True)
            + 1.0 * np.einsum('ABEf,CEfIJK->ABCIJK', X.aa.vvvv[Va, Va, Va, va], T.aaa.VVvOOO, optimize=True)
            - 0.5 * np.einsum('ABEF,CFEIJK->ABCIJK', X.aa.vvvv[Va, Va, Va, Va], T.aaa.VVVOOO, optimize=True)
    )
    dR.aaa.VVVOOO += (9.0 / 36.0) * (
            -1.0 * np.einsum('BmJe,CAemIK->ABCIJK', X.aa.voov[Va, oa, Oa, va], T.aaa.VVvoOO, optimize=True)
            + 1.0 * np.einsum('BmJE,CEAmIK->ABCIJK', X.aa.voov[Va, oa, Oa, Va], T.aaa.VVVoOO, optimize=True)
            + 1.0 * np.einsum('BMJe,CAeIMK->ABCIJK', X.aa.voov[Va, Oa, Oa, va], T.aaa.VVvOOO, optimize=True)
            - 1.0 * np.einsum('BMJE,CEAIMK->ABCIJK', X.aa.voov[Va, Oa, Oa, Va], T.aaa.VVVOOO, optimize=True)
    )
    dR.aaa.VVVOOO += (9.0 / 36.0) * (
            -1.0 * np.einsum('BmJe,CAeIKm->ABCIJK', X.ab.voov[Va, ob, Oa, vb], T.aab.VVvOOo, optimize=True)
            - 1.0 * np.einsum('BmJE,CAEIKm->ABCIJK', X.ab.voov[Va, ob, Oa, Vb], T.aab.VVVOOo, optimize=True)
            - 1.0 * np.einsum('BMJe,CAeIKM->ABCIJK', X.ab.voov[Va, Ob, Oa, vb], T.aab.VVvOOO, optimize=True)
            - 1.0 * np.einsum('BMJE,CAEIKM->ABCIJK', X.ab.voov[Va, Ob, Oa, Vb], T.aab.VVVOOO, optimize=True)
    )
    dR.aaa.VVVOOO += (3.0 / 36.0) * (
            -1.0 * np.einsum('mJ,CBAmIK->ABCIJK', H.a.oo[oa, Oa], R.aaa.VVVoOO, optimize=True)
            + 1.0 * np.einsum('MJ,CBAIMK->ABCIJK', H.a.oo[Oa, Oa], R.aaa.VVVOOO, optimize=True)
    )
    dR.aaa.VVVOOO += (3.0 / 36.0) * (
            +1.0 * np.einsum('Be,CAeIJK->ABCIJK', H.a.vv[Va, va], R.aaa.VVvOOO, optimize=True)
            - 1.0 * np.einsum('BE,CEAIJK->ABCIJK', H.a.vv[Va, Va], R.aaa.VVVOOO, optimize=True)
    )
    dR.aaa.VVVOOO += (3.0 / 36.0) * (
            -0.5 * np.einsum('mnIJ,CBAmnK->ABCIJK', H.aa.oooo[oa, oa, Oa, Oa], R.aaa.VVVooO, optimize=True)
            + 1.0 * np.einsum('MnIJ,CBAnMK->ABCIJK', H.aa.oooo[Oa, oa, Oa, Oa], R.aaa.VVVoOO, optimize=True)
            - 0.5 * np.einsum('MNIJ,CBAMNK->ABCIJK', H.aa.oooo[Oa, Oa, Oa, Oa], R.aaa.VVVOOO, optimize=True)
    )
    dR.aaa.VVVOOO += (3.0 / 36.0) * (
            -0.5 * np.einsum('ABef,CfeIJK->ABCIJK', H.aa.vvvv[Va, Va, va, va], R.aaa.VvvOOO, optimize=True)
            + 1.0 * np.einsum('ABEf,CEfIJK->ABCIJK', H.aa.vvvv[Va, Va, Va, va], R.aaa.VVvOOO, optimize=True)
            - 0.5 * np.einsum('ABEF,CFEIJK->ABCIJK', H.aa.vvvv[Va, Va, Va, Va], R.aaa.VVVOOO, optimize=True)
    )
    dR.aaa.VVVOOO += (9.0 / 36.0) * (
            -1.0 * np.einsum('AmIe,CBemJK->ABCIJK', H.aa.voov[Va, oa, Oa, va], R.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('AmIE,CBEmJK->ABCIJK', H.aa.voov[Va, oa, Oa, Va], R.aaa.VVVoOO, optimize=True)
            - 1.0 * np.einsum('AMIe,CBeMJK->ABCIJK', H.aa.voov[Va, Oa, Oa, va], R.aaa.VVvOOO, optimize=True)
            - 1.0 * np.einsum('AMIE,CBEMJK->ABCIJK', H.aa.voov[Va, Oa, Oa, Va], R.aaa.VVVOOO, optimize=True)
    )
    dR.aaa.VVVOOO += (9.0 / 36.0) * (
            -1.0 * np.einsum('AmIe,CBeJKm->ABCIJK', H.ab.voov[Va, ob, Oa, vb], R.aab.VVvOOo, optimize=True)
            - 1.0 * np.einsum('AmIE,CBEJKm->ABCIJK', H.ab.voov[Va, ob, Oa, Vb], R.aab.VVVOOo, optimize=True)
            - 1.0 * np.einsum('AMIe,CBeJKM->ABCIJK', H.ab.voov[Va, Ob, Oa, vb], R.aab.VVvOOO, optimize=True)
            - 1.0 * np.einsum('AMIE,CBEJKM->ABCIJK', H.ab.voov[Va, Ob, Oa, Vb], R.aab.VVVOOO, optimize=True)
    )

    dR.aaa.VVVOOO -= np.transpose(dR.aaa.VVVOOO, (0, 1, 2, 3, 5, 4))
    dR.aaa.VVVOOO -= np.transpose(dR.aaa.VVVOOO, (0, 1, 2, 4, 3, 5)) + np.transpose(dR.aaa.VVVOOO, (0, 1, 2, 5, 4, 3))
    dR.aaa.VVVOOO -= np.transpose(dR.aaa.VVVOOO, (0, 2, 1, 3, 4, 5))
    dR.aaa.VVVOOO -= np.transpose(dR.aaa.VVVOOO, (1, 0, 2, 3, 4, 5)) + np.transpose(dR.aaa.VVVOOO, (2, 1, 0, 3, 4, 5))

    return dR

def update(R, omega, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    R.aaa.VVVOOO = eomcc_active_loops.update_r3a_111111(
        R.aaa.VVVOOO,
        omega,
        H.a.oo[Oa, Oa],
        H.a.vv[Va, Va],
        H.a.oo[oa, oa],
        H.a.vv[va, va],
        0.0,
    )

    return R
