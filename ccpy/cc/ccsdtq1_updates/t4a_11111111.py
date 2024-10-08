import numpy as np
from ccpy.utilities.active_space import get_active_slices
#from ccpy.lib.core import ccsdtq_active_loops

#@profile
def build(T, dT, H, X, VT4, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    # MM(2,4)A
    dT.aaaa.VVVVOOOO = -(144.0 / 576.0) * np.einsum("AmIe,BCmK,eDJL->ABCDIJKL", H.aa.voov[Va, :, Oa, :], T.aa[Va, Va, :, Oa], T.aa[:, Va, Oa, Oa], optimize=True)  # (jl/i/k)(bc/a/d) = 12 * 12 = 144
    dT.aaaa.VVVVOOOO += (36.0 / 576.0) * np.einsum("mnIJ,ADmL,BCnK->ABCDIJKL", H.aa.oooo[:, :, Oa, Oa], T.aa[Va, Va, :, Oa], T.aa[Va, Va, :, Oa], optimize=True)   # (ij/kl)(bc/ad) = 6 * 6 = 36
    dT.aaaa.VVVVOOOO += (36.0 / 576.0) * np.einsum("ABef,fCJK,eDIL->ABCDIJKL", H.aa.vvvv[Va, Va, :, :], T.aa[:, Va, Oa, Oa], T.aa[:, Va, Oa, Oa], optimize=True)   # (jk/il)(ab/cd) = 6 * 6 = 36

    # (H(2) * T3)_C + (H(2) * 1/2 T3**2)_C
    dT.aaaa.VVVVOOOO += (24.0 / 576.0) * (
            +1.0 * np.einsum('CDKe,ABeIJL->ABCDIJKL', H.aa.vvov[Va, Va, Oa, :], T.aaa[Va, Va, :, Oa, Oa, Oa], optimize=True)
    )
    dT.aaaa.VVVVOOOO += (24.0 / 576.0) * (
            -1.0 * np.einsum('CmKL,ABDIJm->ABCDIJKL', H.aa.vooo[Va, :, Oa, Oa], T.aaa[Va, Va, Va, Oa, Oa, :], optimize=True)
    )
    dT.aaaa.VVVVOOOO += (16.0 / 576.0) * (
            +1.0 * np.einsum('BmnJKL,ACDImn->ABCDIJKL', X.aaa.vooooo[Va, :, :, Oa, Oa, Oa], T.aaa[Va, Va, Va, Oa, :, :], optimize=True)
    )
    dT.aaaa.VVVVOOOO += (24.0 / 576.0) * (
            +1.0 * np.einsum('ABmIJL,CDKm->ABCDIJKL', X.aaa.vvoooo[Va, Va, :, Oa, Oa, Oa], T.aa[Va, Va, Oa, :], optimize=True)
    )
    dT.aaaa.VVVVOOOO += (36.0 / 576.0) * (
            +1.0 * np.einsum('CDmKLe,ABeIJm->ABCDIJKL', X.aaa.vvooov[Va, Va, :, Oa, Oa, :], T.aaa[Va, Va, :, Oa, Oa, :], optimize=True)
    )
    dT.aaaa.VVVVOOOO += (36.0 / 576.0) * (
            +1.0 * np.einsum('CDmKLe,ABeIJm->ABCDIJKL', X.aab.vvooov[Va, Va, :, Oa, Oa, :], T.aab[Va, Va, :, Oa, Oa, :], optimize=True)
    )

    # (H(2) * T4)_C
    dT.aaaa.VVVVOOOO += (4.0 / 576.0) * (
            -1.0 * np.einsum('mI,DCBAmJKL->ABCDIJKL', H.a.oo[oa, Oa], T.aaaa.VVVVoOOO, optimize=True)
            - 1.0 * np.einsum('MI,DCBAMJKL->ABCDIJKL', H.a.oo[Oa, Oa], T.aaaa.VVVVOOOO, optimize=True)
    )
    dT.aaaa.VVVVOOOO += (4.0 / 576.0) * (
            +1.0 * np.einsum('Ae,DCBeIJKL->ABCDIJKL', H.a.vv[Va, va], T.aaaa.VVVvOOOO, optimize=True)
            + 1.0 * np.einsum('AE,DCBEIJKL->ABCDIJKL', H.a.vv[Va, Va], T.aaaa.VVVVOOOO, optimize=True)
    )
    dT.aaaa.VVVVOOOO += (6.0 / 576.0) * (
            +0.5 * np.einsum('mnIJ,DCBAmnKL->ABCDIJKL', H.aa.oooo[oa, oa, Oa, Oa], T.aaaa.VVVVooOO, optimize=True)
            - 1.0 * np.einsum('MnIJ,DCBAnMKL->ABCDIJKL', H.aa.oooo[Oa, oa, Oa, Oa], T.aaaa.VVVVoOOO, optimize=True)
            + 0.5 * np.einsum('MNIJ,DCBAMNKL->ABCDIJKL', H.aa.oooo[Oa, Oa, Oa, Oa], T.aaaa.VVVVOOOO, optimize=True)
    )
    dT.aaaa.VVVVOOOO += (6.0 / 576.0) * (
            +0.5 * np.einsum('ABef,DCfeIJKL->ABCDIJKL', H.aa.vvvv[Va, Va, va, va], T.aaaa.VVvvOOOO, optimize=True)
            + 1.0 * np.einsum('ABeF,DCFeIJKL->ABCDIJKL', H.aa.vvvv[Va, Va, va, Va], T.aaaa.VVVvOOOO, optimize=True)
            + 0.5 * np.einsum('ABEF,DCFEIJKL->ABCDIJKL', H.aa.vvvv[Va, Va, Va, Va], T.aaaa.VVVVOOOO, optimize=True)
    )
    dT.aaaa.VVVVOOOO += (16.0 / 576.0) * (
            +1.0 * np.einsum('AmIe,DCBemJKL->ABCDIJKL', H.aa.voov[Va, oa, Oa, va], T.aaaa.VVVvoOOO, optimize=True)
            + 1.0 * np.einsum('AmIE,DCBEmJKL->ABCDIJKL', H.aa.voov[Va, oa, Oa, Va], T.aaaa.VVVVoOOO, optimize=True)
            + 1.0 * np.einsum('AMIe,DCBeMJKL->ABCDIJKL', H.aa.voov[Va, Oa, Oa, va], T.aaaa.VVVvOOOO, optimize=True)
            + 1.0 * np.einsum('AMIE,DCBEMJKL->ABCDIJKL', H.aa.voov[Va, Oa, Oa, Va], T.aaaa.VVVVOOOO, optimize=True)
    )
    dT.aaaa.VVVVOOOO += (16.0 / 576.0) * (
            -1.0 * np.einsum('AmIe,DCBeJKLm->ABCDIJKL', H.ab.voov[Va, ob, Oa, vb], T.aaab.VVVvOOOo, optimize=True)
            - 1.0 * np.einsum('AmIE,DCBEJKLm->ABCDIJKL', H.ab.voov[Va, ob, Oa, Vb], T.aaab.VVVVOOOo, optimize=True)
            - 1.0 * np.einsum('AMIe,DCBeJKLM->ABCDIJKL', H.ab.voov[Va, Ob, Oa, vb], T.aaab.VVVvOOOO, optimize=True)
            - 1.0 * np.einsum('AMIE,DCBEJKLM->ABCDIJKL', H.ab.voov[Va, Ob, Oa, Vb], T.aaab.VVVVOOOO, optimize=True)
    )

    dT.aaaa.VVVVOOOO += (24.0 / 576.0) * (
            +1.0 * np.einsum('BCDJKe,AeIL->ABCDIJKL', VT4.aaa.vvvoov[Va, Va, Va, Oa, Oa, :], T.aa[Va, :, Oa, Oa], optimize=True)
    )
    dT.aaaa.VVVVOOOO += (24.0 / 576.0) * (
            -1.0 * np.einsum('BCmJKL,ADIm->ABCDIJKL', VT4.aaa.vvoooo[Va, Va, :, Oa, Oa, Oa], T.aa[Va, Va, Oa, :], optimize=True)
    )

    return dT

# def update(T, dT, H, shift, system):
#
#     oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
#
#     T.aaa.VVVOOO, dT.aaa.VVVOOO = cc_active_loops.update_t3a_111111(
#         T.aaa.VVVOOO,
#         dT.aaa.VVVOOO,
#         H.a.oo[Oa, Oa],
#         H.a.vv[Va, Va],
#         H.a.oo[oa, oa],
#         H.a.vv[va, va],
#         shift,
#     )
#
#     return T, dT