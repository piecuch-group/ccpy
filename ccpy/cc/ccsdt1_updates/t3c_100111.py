import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import cc_active_loops

def build(T, dT, H, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    # MM(2,3)
    dT.abb.VvvOOO = (2.0 / 4.0) * (
            +1.0 * np.einsum('AbIe,ecJK->AbcIJK', H.ab.vvov[Va, vb, Oa, :], T.bb[:, vb, Ob, Ob], optimize=True)
    )
    dT.abb.VvvOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('AmIJ,bcmK->AbcIJK', H.ab.vooo[Va, :, Oa, Ob], T.bb[vb, vb, :, Ob], optimize=True)
    )
    dT.abb.VvvOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('cbKe,AeIJ->AbcIJK', H.bb.vvov[vb, vb, Ob, :], T.ab[Va, :, Oa, Ob], optimize=True)
    )
    dT.abb.VvvOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('cmKJ,AbIm->AbcIJK', H.bb.vooo[vb, :, Ob, Ob], T.ab[Va, vb, Oa, :], optimize=True)
    )
    dT.abb.VvvOOO += (4.0 / 4.0) * (
            +1.0 * np.einsum('AbeJ,ecIK->AbcIJK', H.ab.vvvo[Va, vb, :, Ob], T.ab[:, vb, Oa, Ob], optimize=True)
    )
    dT.abb.VvvOOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('mbIJ,AcmK->AbcIJK', H.ab.ovoo[:, vb, Oa, Ob], T.ab[Va, vb, :, Ob], optimize=True)
    )
    # (H(2) * T3)_C
    dT.abb.VvvOOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('mI,AcbmJK->AbcIJK', H.a.oo[oa, Oa], T.abb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('MI,AcbMJK->AbcIJK', H.a.oo[Oa, Oa], T.abb.VvvOOO, optimize=True)
    )
    dT.abb.VvvOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mJ,AcbImK->AbcIJK', H.b.oo[ob, Ob], T.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('MJ,AcbIMK->AbcIJK', H.b.oo[Ob, Ob], T.abb.VvvOOO, optimize=True)
    )
    dT.abb.VvvOOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('AE,EcbIJK->AbcIJK', H.a.vv[Va, Va], T.abb.VvvOOO, optimize=True)
    )
    dT.abb.VvvOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('be,AceIJK->AbcIJK', H.b.vv[vb, vb], T.abb.VvvOOO, optimize=True)
            + 1.0 * np.einsum('bE,AEcIJK->AbcIJK', H.b.vv[vb, Vb], T.abb.VVvOOO, optimize=True)
    )
    dT.abb.VvvOOO += (1.0 / 4.0) * (
            -0.5 * np.einsum('mnJK,AcbImn->AbcIJK', H.bb.oooo[ob, ob, Ob, Ob], T.abb.VvvOoo, optimize=True)
            + 1.0 * np.einsum('MnJK,AcbInM->AbcIJK', H.bb.oooo[Ob, ob, Ob, Ob], T.abb.VvvOoO, optimize=True)
            - 0.5 * np.einsum('MNJK,AcbIMN->AbcIJK', H.bb.oooo[Ob, Ob, Ob, Ob], T.abb.VvvOOO, optimize=True)
    )
    dT.abb.VvvOOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('mnIJ,AcbmnK->AbcIJK', H.ab.oooo[oa, ob, Oa, Ob], T.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('mNIJ,AcbmNK->AbcIJK', H.ab.oooo[oa, Ob, Oa, Ob], T.abb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('MnIJ,AcbMnK->AbcIJK', H.ab.oooo[Oa, ob, Oa, Ob], T.abb.VvvOoO, optimize=True)
            - 1.0 * np.einsum('MNIJ,AcbMNK->AbcIJK', H.ab.oooo[Oa, Ob, Oa, Ob], T.abb.VvvOOO, optimize=True)
    )
    dT.abb.VvvOOO += (1.0 / 4.0) * (
            -0.5 * np.einsum('bcef,AfeIJK->AbcIJK', H.bb.vvvv[vb, vb, vb, vb], T.abb.VvvOOO, optimize=True)
            + 1.0 * np.einsum('bcEf,AEfIJK->AbcIJK', H.bb.vvvv[vb, vb, Vb, vb], T.abb.VVvOOO, optimize=True)
            - 0.5 * np.einsum('bcEF,AFEIJK->AbcIJK', H.bb.vvvv[vb, vb, Vb, Vb], T.abb.VVVOOO, optimize=True)
    )
    dT.abb.VvvOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('AbeF,eFcIJK->AbcIJK', H.ab.vvvv[Va, vb, va, Vb], T.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('AbEf,EcfIJK->AbcIJK', H.ab.vvvv[Va, vb, Va, vb], T.abb.VvvOOO, optimize=True)
            + 1.0 * np.einsum('AbEF,EFcIJK->AbcIJK', H.ab.vvvv[Va, vb, Va, Vb], T.abb.VVvOOO, optimize=True)
    )
    dT.abb.VvvOOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('AmIE,EcbmJK->AbcIJK', H.aa.voov[Va, oa, Oa, Va], T.abb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('AMIE,EcbMJK->AbcIJK', H.aa.voov[Va, Oa, Oa, Va], T.abb.VvvOOO, optimize=True)
    )
    dT.abb.VvvOOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('AmIE,EcbmJK->AbcIJK', H.ab.voov[Va, ob, Oa, Vb], T.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('AMIE,EcbMJK->AbcIJK', H.ab.voov[Va, Ob, Oa, Vb], T.bbb.VvvOOO, optimize=True)
    )
    dT.abb.VvvOOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('mbeJ,AecmIK->AbcIJK', H.ab.ovvo[oa, vb, va, Ob], T.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('MbeJ,AecIMK->AbcIJK', H.ab.ovvo[Oa, vb, va, Ob], T.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('mbEJ,EAcmIK->AbcIJK', H.ab.ovvo[oa, vb, Va, Ob], T.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('MbEJ,EAcIMK->AbcIJK', H.ab.ovvo[Oa, vb, Va, Ob], T.aab.VVvOOO, optimize=True)
    )
    dT.abb.VvvOOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('bmJe,AceImK->AbcIJK', H.bb.voov[vb, ob, Ob, vb], T.abb.VvvOoO, optimize=True)
            - 1.0 * np.einsum('bMJe,AceIMK->AbcIJK', H.bb.voov[vb, Ob, Ob, vb], T.abb.VvvOOO, optimize=True)
            + 1.0 * np.einsum('bmJE,AEcImK->AbcIJK', H.bb.voov[vb, ob, Ob, Vb], T.abb.VVvOoO, optimize=True)
            + 1.0 * np.einsum('bMJE,AEcIMK->AbcIJK', H.bb.voov[vb, Ob, Ob, Vb], T.abb.VVvOOO, optimize=True)
    )
    dT.abb.VvvOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mbIe,AcemJK->AbcIJK', H.ab.ovov[oa, vb, Oa, vb], T.abb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('MbIe,AceMJK->AbcIJK', H.ab.ovov[Oa, vb, Oa, vb], T.abb.VvvOOO, optimize=True)
            - 1.0 * np.einsum('mbIE,AEcmJK->AbcIJK', H.ab.ovov[oa, vb, Oa, Vb], T.abb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('MbIE,AEcMJK->AbcIJK', H.ab.ovov[Oa, vb, Oa, Vb], T.abb.VVvOOO, optimize=True)
    )
    dT.abb.VvvOOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('AmEJ,EcbImK->AbcIJK', H.ab.vovo[Va, ob, Va, Ob], T.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('AMEJ,EcbIMK->AbcIJK', H.ab.vovo[Va, Ob, Va, Ob], T.abb.VvvOOO, optimize=True)
    )

    dT.abb.VvvOOO -= np.transpose(dT.abb.VvvOOO, (0, 2, 1, 3, 4, 5))
    dT.abb.VvvOOO -= np.transpose(dT.abb.VvvOOO, (0, 1, 2, 3, 5, 4))

    return dT


def update(T, dT, H, shift, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.abb.VvvOOO, dT.abb.VvvOOO = cc_active_loops.update_t3c_100111(
        T.abb.VvvOOO,
        dT.abb.VvvOOO,
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
