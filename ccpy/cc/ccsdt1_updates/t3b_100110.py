import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import cc_active_loops

#@profile
def build(T, dT, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    # MM(2,3)
    dT.aab.VvvOOo = (1.0 / 2.0) * (
            +1.0 * np.einsum('bcek,AeIJ->AbcIJk', H.ab.vvvo[va, vb, :, ob], T.aa[Va, :, Oa, Oa], optimize=True)
    )
    dT.aab.VvvOOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('Acek,beIJ->AbcIJk', H.ab.vvvo[Va, vb, :, ob], T.aa[va, :, Oa, Oa], optimize=True)
    )
    dT.aab.VvvOOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('mcJk,AbIm->AbcIJk', H.ab.ovoo[:, vb, Oa, ob], T.aa[Va, va, Oa, :], optimize=True)
    )
    dT.aab.VvvOOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('AcIe,beJk->AbcIJk', H.ab.vvov[Va, vb, Oa, :], T.ab[va, :, Oa, ob], optimize=True)
    )
    dT.aab.VvvOOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('bcIe,AeJk->AbcIJk', H.ab.vvov[va, vb, Oa, :], T.ab[Va, :, Oa, ob], optimize=True)
    )
    dT.aab.VvvOOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('AmIk,bcJm->AbcIJk', H.ab.vooo[Va, :, Oa, ob], T.ab[va, vb, Oa, :], optimize=True)
    )
    dT.aab.VvvOOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('bmIk,AcJm->AbcIJk', H.ab.vooo[va, :, Oa, ob], T.ab[Va, vb, Oa, :], optimize=True)
    )
    dT.aab.VvvOOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('AbIe,ecJk->AbcIJk', H.aa.vvov[Va, va, Oa, :], T.ab[:, vb, Oa, ob], optimize=True)
    )
    dT.aab.VvvOOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('AmIJ,bcmk->AbcIJk', H.aa.vooo[Va, :, Oa, Oa], T.ab[va, vb, :, ob], optimize=True)
    )
    dT.aab.VvvOOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('bmIJ,Acmk->AbcIJk', H.aa.vooo[va, :, Oa, Oa], T.ab[Va, vb, :, ob], optimize=True)
    )
    # (H(2) * T3)_C
    dT.aab.VvvOOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('mI,AbcmJk->AbcIJk', H.a.oo[oa, Oa], T.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('MI,AbcMJk->AbcIJk', H.a.oo[Oa, Oa], T.aab.VvvOOo, optimize=True)
    )
    dT.aab.VvvOOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('mk,AbcIJm->AbcIJk', H.b.oo[ob, ob], T.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('Mk,AbcIJM->AbcIJk', H.b.oo[Ob, ob], T.aab.VvvOOO, optimize=True)
    )
    dT.aab.VvvOOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('AE,EbcIJk->AbcIJk', H.a.vv[Va, Va], T.aab.VvvOOo, optimize=True)
    )
    dT.aab.VvvOOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('be,AecIJk->AbcIJk', H.a.vv[va, va], T.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('bE,AEcIJk->AbcIJk', H.a.vv[va, Va], T.aab.VVvOOo, optimize=True)
    )
    dT.aab.VvvOOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('ce,AbeIJk->AbcIJk', H.b.vv[vb, vb], T.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('cE,AbEIJk->AbcIJk', H.b.vv[vb, Vb], T.aab.VvVOOo, optimize=True)
    )
    dT.aab.VvvOOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('MnIJ,AbcnMk->AbcIJk', H.aa.oooo[Oa, oa, Oa, Oa], T.aab.VvvoOo, optimize=True)
            + 0.5 * np.einsum('MNIJ,AbcMNk->AbcIJk', H.aa.oooo[Oa, Oa, Oa, Oa], T.aab.VvvOOo, optimize=True)
    )
    dT.aab.VvvOOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('mnJk,AbcmIn->AbcIJk', H.ab.oooo[oa, ob, Oa, ob], T.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('mNJk,AbcmIN->AbcIJk', H.ab.oooo[oa, Ob, Oa, ob], T.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('MnJk,AbcIMn->AbcIJk', H.ab.oooo[Oa, ob, Oa, ob], T.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('MNJk,AbcIMN->AbcIJk', H.ab.oooo[Oa, Ob, Oa, ob], T.aab.VvvOOO, optimize=True)
    )
    dT.aab.VvvOOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('AbEf,EfcIJk->AbcIJk', H.aa.vvvv[Va, va, Va, va], T.aab.VvvOOo, optimize=True)
            - 0.5 * np.einsum('AbEF,FEcIJk->AbcIJk', H.aa.vvvv[Va, va, Va, Va], T.aab.VVvOOo, optimize=True)
    )
    dT.aab.VvvOOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('bcef,AefIJk->AbcIJk', H.ab.vvvv[va, vb, va, vb], T.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('bcEf,EAfIJk->AbcIJk', H.ab.vvvv[va, vb, Va, vb], T.aab.VVvOOo, optimize=True)
            + 1.0 * np.einsum('bceF,AeFIJk->AbcIJk', H.ab.vvvv[va, vb, va, Vb], T.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('bcEF,EAFIJk->AbcIJk', H.ab.vvvv[va, vb, Va, Vb], T.aab.VVVOOo, optimize=True)
    )
    dT.aab.VvvOOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('AceF,ebFIJk->AbcIJk', H.ab.vvvv[Va, vb, va, Vb], T.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('AcEf,EbfIJk->AbcIJk', H.ab.vvvv[Va, vb, Va, vb], T.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('AcEF,EbFIJk->AbcIJk', H.ab.vvvv[Va, vb, Va, Vb], T.aab.VvVOOo, optimize=True)
    )
    dT.aab.VvvOOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('AmIE,EbcmJk->AbcIJk', H.aa.voov[Va, oa, Oa, Va], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('AMIE,EbcMJk->AbcIJk', H.aa.voov[Va, Oa, Oa, Va], T.aab.VvvOOo, optimize=True)
    )
    dT.aab.VvvOOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('bmIe,AecmJk->AbcIJk', H.aa.voov[va, oa, Oa, va], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('bmIE,AEcmJk->AbcIJk', H.aa.voov[va, oa, Oa, Va], T.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('bMIe,AecMJk->AbcIJk', H.aa.voov[va, Oa, Oa, va], T.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('bMIE,AEcMJk->AbcIJk', H.aa.voov[va, Oa, Oa, Va], T.aab.VVvOOo, optimize=True)
    )
    dT.aab.VvvOOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('AmIE,bEcJmk->AbcIJk', H.ab.voov[Va, ob, Oa, Vb], T.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('AMIE,bEcJkM->AbcIJk', H.ab.voov[Va, Ob, Oa, Vb], T.abb.vVvOoO, optimize=True)
    )
    dT.aab.VvvOOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('bmIe,AceJmk->AbcIJk', H.ab.voov[va, ob, Oa, vb], T.abb.VvvOoo, optimize=True)
            - 1.0 * np.einsum('bmIE,AEcJmk->AbcIJk', H.ab.voov[va, ob, Oa, Vb], T.abb.VVvOoo, optimize=True)
            - 1.0 * np.einsum('bMIe,AceJkM->AbcIJk', H.ab.voov[va, Ob, Oa, vb], T.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('bMIE,AEcJkM->AbcIJk', H.ab.voov[va, Ob, Oa, Vb], T.abb.VVvOoO, optimize=True)
    )
    dT.aab.VvvOOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('mcek,AebmIJ->AbcIJk', H.ab.ovvo[oa, vb, va, ob], T.aaa.VvvoOO, optimize=True)
            + 1.0 * np.einsum('mcEk,EAbmIJ->AbcIJk', H.ab.ovvo[oa, vb, Va, ob], T.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('Mcek,AebIJM->AbcIJk', H.ab.ovvo[Oa, vb, va, ob], T.aaa.VvvOOO, optimize=True)
            + 1.0 * np.einsum('McEk,EAbIJM->AbcIJk', H.ab.ovvo[Oa, vb, Va, ob], T.aaa.VVvOOO, optimize=True)
    )
    dT.aab.VvvOOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('cmke,AbeIJm->AbcIJk', H.bb.voov[vb, ob, ob, vb], T.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('cMke,AbeIJM->AbcIJk', H.bb.voov[vb, Ob, ob, vb], T.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('cmkE,AbEIJm->AbcIJk', H.bb.voov[vb, ob, ob, Vb], T.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('cMkE,AbEIJM->AbcIJk', H.bb.voov[vb, Ob, ob, Vb], T.aab.VvVOOO, optimize=True)
    )
    dT.aab.VvvOOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('AmEk,EbcIJm->AbcIJk', H.ab.vovo[Va, ob, Va, ob], T.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('AMEk,EbcIJM->AbcIJk', H.ab.vovo[Va, Ob, Va, ob], T.aab.VvvOOO, optimize=True)
    )
    dT.aab.VvvOOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('bmek,AecIJm->AbcIJk', H.ab.vovo[va, ob, va, ob], T.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('bmEk,AEcIJm->AbcIJk', H.ab.vovo[va, ob, Va, ob], T.aab.VVvOOo, optimize=True)
            - 1.0 * np.einsum('bMek,AecIJM->AbcIJk', H.ab.vovo[va, Ob, va, ob], T.aab.VvvOOO, optimize=True)
            - 1.0 * np.einsum('bMEk,AEcIJM->AbcIJk', H.ab.vovo[va, Ob, Va, ob], T.aab.VVvOOO, optimize=True)
    )
    dT.aab.VvvOOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('mcIe,AbemJk->AbcIJk', H.ab.ovov[oa, vb, Oa, vb], T.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('mcIE,AbEmJk->AbcIJk', H.ab.ovov[oa, vb, Oa, Vb], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('McIe,AbeMJk->AbcIJk', H.ab.ovov[Oa, vb, Oa, vb], T.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('McIE,AbEMJk->AbcIJk', H.ab.ovov[Oa, vb, Oa, Vb], T.aab.VvVOOo, optimize=True)
    )

    dT.aab.VvvOOo -= np.transpose(dT.aab.VvvOOo, (0, 1, 2, 4, 3, 5))

    return dT

def update(T, dT, H, shift, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.aab.VvvOOo, dT.aab.VvvOOo = cc_active_loops.update_t3b_100110(
        T.aab.VvvOOo,
        dT.aab.VvvOOo,
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