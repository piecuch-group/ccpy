import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import cc_active_loops

def build(T, dT, H, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    # MM(2,3)
    dT.abb.VvvoOO = (2.0 / 4.0) * (
            +1.0 * np.einsum('Abie,ecJK->AbciJK', H.ab.vvov[Va, vb, oa, :], T.bb[:, vb, Ob, Ob], optimize=True)
    )
    dT.abb.VvvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('AmiJ,bcmK->AbciJK', H.ab.vooo[Va, :, oa, Ob], T.bb[vb, vb, :, Ob], optimize=True)
    )
    dT.abb.VvvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('cbKe,AeiJ->AbciJK', H.bb.vvov[vb, vb, Ob, :], T.ab[Va, :, oa, Ob], optimize=True)
    )
    dT.abb.VvvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('cmKJ,Abim->AbciJK', H.bb.vooo[vb, :, Ob, Ob], T.ab[Va, vb, oa, :], optimize=True)
    )
    dT.abb.VvvoOO += (4.0 / 4.0) * (
            +1.0 * np.einsum('AbeJ,eciK->AbciJK', H.ab.vvvo[Va, vb, :, Ob], T.ab[:, vb, oa, Ob], optimize=True)
    )
    dT.abb.VvvoOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('mbiJ,AcmK->AbciJK', H.ab.ovoo[:, vb, oa, Ob], T.ab[Va, vb, :, Ob], optimize=True)
    )
    # (H(2) * T3)_C
    dT.abb.VvvoOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('mi,AcbmJK->AbciJK', H.a.oo[oa, oa], T.abb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('Mi,AcbMJK->AbciJK', H.a.oo[Oa, oa], T.abb.VvvOOO, optimize=True)
    )
    dT.abb.VvvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mJ,AcbimK->AbciJK', H.b.oo[ob, Ob], T.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('MJ,AcbiMK->AbciJK', H.b.oo[Ob, Ob], T.abb.VvvoOO, optimize=True)
    )
    dT.abb.VvvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('AE,EcbiJK->AbciJK', H.a.vv[Va, Va], T.abb.VvvoOO, optimize=True)
    )
    dT.abb.VvvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('be,AceiJK->AbciJK', H.b.vv[vb, vb], T.abb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('bE,AEciJK->AbciJK', H.b.vv[vb, Vb], T.abb.VVvoOO, optimize=True)
    )
    dT.abb.VvvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('mNJK,AcbimN->AbciJK', H.bb.oooo[ob, Ob, Ob, Ob], T.abb.VvvooO, optimize=True)
            - 0.5 * np.einsum('MNJK,AcbiMN->AbciJK', H.bb.oooo[Ob, Ob, Ob, Ob], T.abb.VvvoOO, optimize=True)
    )
    dT.abb.VvvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('mniJ,AcbmnK->AbciJK', H.ab.oooo[oa, ob, oa, Ob], T.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('MniJ,AcbMnK->AbciJK', H.ab.oooo[Oa, ob, oa, Ob], T.abb.VvvOoO, optimize=True)
            - 1.0 * np.einsum('mNiJ,AcbmNK->AbciJK', H.ab.oooo[oa, Ob, oa, Ob], T.abb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('MNiJ,AcbMNK->AbciJK', H.ab.oooo[Oa, Ob, oa, Ob], T.abb.VvvOOO, optimize=True)
    )
    dT.abb.VvvoOO += (1.0 / 4.0) * (
            -0.5 * np.einsum('bcef,AfeiJK->AbciJK', H.bb.vvvv[vb, vb, vb, vb], T.abb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('bceF,AFeiJK->AbciJK', H.bb.vvvv[vb, vb, vb, Vb], T.abb.VVvoOO, optimize=True)
            - 0.5 * np.einsum('bcEF,AFEiJK->AbciJK', H.bb.vvvv[vb, vb, Vb, Vb], T.abb.VVVoOO, optimize=True)
    )
    dT.abb.VvvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('AbEf,EcfiJK->AbciJK', H.ab.vvvv[Va, vb, Va, vb], T.abb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('AbeF,eFciJK->AbciJK', H.ab.vvvv[Va, vb, va, Vb], T.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('AbEF,EFciJK->AbciJK', H.ab.vvvv[Va, vb, Va, Vb], T.abb.VVvoOO, optimize=True)
    )
    dT.abb.VvvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('AmiE,EcbmJK->AbciJK', H.aa.voov[Va, oa, oa, Va], T.abb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('AMiE,EcbMJK->AbciJK', H.aa.voov[Va, Oa, oa, Va], T.abb.VvvOOO, optimize=True)
    )
    dT.abb.VvvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('AmiE,EcbmJK->AbciJK', H.ab.voov[Va, ob, oa, Vb], T.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('AMiE,EcbMJK->AbciJK', H.ab.voov[Va, Ob, oa, Vb], T.bbb.VvvOOO, optimize=True)
    )
    dT.abb.VvvoOO += (4.0 / 4.0) * (
            +1.0 * np.einsum('mbeJ,AecimK->AbciJK', H.ab.ovvo[oa, vb, va, Ob], T.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('MbeJ,AeciMK->AbciJK', H.ab.ovvo[Oa, vb, va, Ob], T.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('mbEJ,EAcimK->AbciJK', H.ab.ovvo[oa, vb, Va, Ob], T.aab.VVvooO, optimize=True)
            - 1.0 * np.einsum('MbEJ,EAciMK->AbciJK', H.ab.ovvo[Oa, vb, Va, Ob], T.aab.VVvoOO, optimize=True)
    )
    dT.abb.VvvoOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('bmJe,AceimK->AbciJK', H.bb.voov[vb, ob, Ob, vb], T.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('bMJe,AceiMK->AbciJK', H.bb.voov[vb, Ob, Ob, vb], T.abb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('bmJE,AEcimK->AbciJK', H.bb.voov[vb, ob, Ob, Vb], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('bMJE,AEciMK->AbciJK', H.bb.voov[vb, Ob, Ob, Vb], T.abb.VVvoOO, optimize=True)
    )
    dT.abb.VvvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mbie,AcemJK->AbciJK', H.ab.ovov[oa, vb, oa, vb], T.abb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('Mbie,AceMJK->AbciJK', H.ab.ovov[Oa, vb, oa, vb], T.abb.VvvOOO, optimize=True)
            - 1.0 * np.einsum('mbiE,AEcmJK->AbciJK', H.ab.ovov[oa, vb, oa, Vb], T.abb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('MbiE,AEcMJK->AbciJK', H.ab.ovov[Oa, vb, oa, Vb], T.abb.VVvOOO, optimize=True)
    )
    dT.abb.VvvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('AmEJ,EcbimK->AbciJK', H.ab.vovo[Va, ob, Va, Ob], T.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('AMEJ,EcbiMK->AbciJK', H.ab.vovo[Va, Ob, Va, Ob], T.abb.VvvoOO, optimize=True)
    )

    dT.abb.VvvoOO -= np.transpose(dT.abb.VvvoOO, (0, 2, 1, 3, 4, 5))
    dT.abb.VvvoOO -= np.transpose(dT.abb.VvvoOO, (0, 1, 2, 3, 5, 4))

    return dT


def update(T, dT, H, shift, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.abb.VvvoOO, dT.abb.VvvoOO = cc_active_loops.update_t3c_100011(
        T.abb.VvvoOO,
        dT.abb.VvvoOO,
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