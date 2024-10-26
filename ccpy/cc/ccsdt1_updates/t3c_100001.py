import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import cc_active_loops

def build(T, dT, H, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    # MM(2,3)
    dT.abb.VvvooO = (2.0 / 2.0) * (
            +1.0 * np.einsum('Abie,ecjK->AbcijK', H.ab.vvov[Va, vb, oa, :], T.bb[:, vb, ob, Ob], optimize=True)
    )
    dT.abb.VvvooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('Amij,bcmK->AbcijK', H.ab.vooo[Va, :, oa, ob], T.bb[vb, vb, :, Ob], optimize=True)
    )
    dT.abb.VvvooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('AmiK,bcmj->AbcijK', H.ab.vooo[Va, :, oa, Ob], T.bb[vb, vb, :, ob], optimize=True)
    )
    dT.abb.VvvooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('cbKe,Aeij->AbcijK', H.bb.vvov[vb, vb, Ob, :], T.ab[Va, :, oa, ob], optimize=True)
    )
    dT.abb.VvvooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('cbje,AeiK->AbcijK', H.bb.vvov[vb, vb, ob, :], T.ab[Va, :, oa, Ob], optimize=True)
    )
    dT.abb.VvvooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('cmKj,Abim->AbcijK', H.bb.vooo[vb, :, Ob, ob], T.ab[Va, vb, oa, :], optimize=True)
    )
    dT.abb.VvvooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('Abej,eciK->AbcijK', H.ab.vvvo[Va, vb, :, ob], T.ab[:, vb, oa, Ob], optimize=True)
    )
    dT.abb.VvvooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('AbeK,ecij->AbcijK', H.ab.vvvo[Va, vb, :, Ob], T.ab[:, vb, oa, ob], optimize=True)
    )
    dT.abb.VvvooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mbij,AcmK->AbcijK', H.ab.ovoo[:, vb, oa, ob], T.ab[Va, vb, :, Ob], optimize=True)
    )
    dT.abb.VvvooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mbiK,Acmj->AbcijK', H.ab.ovoo[:, vb, oa, Ob], T.ab[Va, vb, :, ob], optimize=True)
    )
    # (H(2) * T3)_C
    dT.abb.VvvooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mi,AcbmjK->AbcijK', H.a.oo[oa, oa], T.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('Mi,AcbMjK->AbcijK', H.a.oo[Oa, oa], T.abb.VvvOoO, optimize=True)
    )
    dT.abb.VvvooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mj,AcbimK->AbcijK', H.b.oo[ob, ob], T.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('Mj,AcbiMK->AbcijK', H.b.oo[Ob, ob], T.abb.VvvoOO, optimize=True)
    )
    dT.abb.VvvooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('MK,AcbijM->AbcijK', H.b.oo[Ob, Ob], T.abb.VvvooO, optimize=True)
    )
    dT.abb.VvvooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('AE,EcbijK->AbcijK', H.a.vv[Va, Va], T.abb.VvvooO, optimize=True)
    )
    dT.abb.VvvooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('be,AceijK->AbcijK', H.b.vv[vb, vb], T.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('bE,AEcijK->AbcijK', H.b.vv[vb, Vb], T.abb.VVvooO, optimize=True)
    )
    dT.abb.VvvooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('MnjK,AcbinM->AbcijK', H.bb.oooo[Ob, ob, ob, Ob], T.abb.VvvooO, optimize=True)
            - 0.5 * np.einsum('MNjK,AcbiMN->AbcijK', H.bb.oooo[Ob, Ob, ob, Ob], T.abb.VvvoOO, optimize=True)
    )
    dT.abb.VvvooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mnij,AcbmnK->AbcijK', H.ab.oooo[oa, ob, oa, ob], T.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('mNij,AcbmNK->AbcijK', H.ab.oooo[oa, Ob, oa, ob], T.abb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('Mnij,AcbMnK->AbcijK', H.ab.oooo[Oa, ob, oa, ob], T.abb.VvvOoO, optimize=True)
            - 1.0 * np.einsum('MNij,AcbMNK->AbcijK', H.ab.oooo[Oa, Ob, oa, ob], T.abb.VvvOOO, optimize=True)
    )
    dT.abb.VvvooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mNiK,AcbmjN->AbcijK', H.ab.oooo[oa, Ob, oa, Ob], T.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('MniK,AcbMnj->AbcijK', H.ab.oooo[Oa, ob, oa, Ob], T.abb.VvvOoo, optimize=True)
            - 1.0 * np.einsum('MNiK,AcbMjN->AbcijK', H.ab.oooo[Oa, Ob, oa, Ob], T.abb.VvvOoO, optimize=True)
    )
    dT.abb.VvvooO += (1.0 / 2.0) * (
            -0.5 * np.einsum('bcef,AfeijK->AbcijK', H.bb.vvvv[vb, vb, vb, vb], T.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('bceF,AFeijK->AbcijK', H.bb.vvvv[vb, vb, vb, Vb], T.abb.VVvooO, optimize=True)
            - 0.5 * np.einsum('bcEF,AFEijK->AbcijK', H.bb.vvvv[vb, vb, Vb, Vb], T.abb.VVVooO, optimize=True)
    )
    dT.abb.VvvooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('AbEf,EcfijK->AbcijK', H.ab.vvvv[Va, vb, Va, vb], T.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('AbeF,eFcijK->AbcijK', H.ab.vvvv[Va, vb, va, Vb], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('AbEF,EFcijK->AbcijK', H.ab.vvvv[Va, vb, Va, Vb], T.abb.VVvooO, optimize=True)
    )
    dT.abb.VvvooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('AmiE,EcbmjK->AbcijK', H.aa.voov[Va, oa, oa, Va], T.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('AMiE,EcbMjK->AbcijK', H.aa.voov[Va, Oa, oa, Va], T.abb.VvvOoO, optimize=True)
    )
    dT.abb.VvvooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('AmiE,EcbmjK->AbcijK', H.ab.voov[Va, ob, oa, Vb], T.bbb.VvvooO, optimize=True)
            + 1.0 * np.einsum('AMiE,EcbjMK->AbcijK', H.ab.voov[Va, Ob, oa, Vb], T.bbb.VvvoOO, optimize=True)
    )
    dT.abb.VvvooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mbej,AecimK->AbcijK', H.ab.ovvo[oa, vb, va, ob], T.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('Mbej,AeciMK->AbcijK', H.ab.ovvo[Oa, vb, va, ob], T.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('mbEj,EAcimK->AbcijK', H.ab.ovvo[oa, vb, Va, ob], T.aab.VVvooO, optimize=True)
            - 1.0 * np.einsum('MbEj,EAciMK->AbcijK', H.ab.ovvo[Oa, vb, Va, ob], T.aab.VVvoOO, optimize=True)
    )
    dT.abb.VvvooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('MbeK,AeciMj->AbcijK', H.ab.ovvo[Oa, vb, va, Ob], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('MbEK,EAciMj->AbcijK', H.ab.ovvo[Oa, vb, Va, Ob], T.aab.VVvoOo, optimize=True)
    )
    dT.abb.VvvooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('bmje,AceimK->AbcijK', H.bb.voov[vb, ob, ob, vb], T.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('bMje,AceiMK->AbcijK', H.bb.voov[vb, Ob, ob, vb], T.abb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('bmjE,AEcimK->AbcijK', H.bb.voov[vb, ob, ob, Vb], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('bMjE,AEciMK->AbcijK', H.bb.voov[vb, Ob, ob, Vb], T.abb.VVvoOO, optimize=True)
    )
    dT.abb.VvvooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('bMKe,AceijM->AbcijK', H.bb.voov[vb, Ob, Ob, vb], T.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('bMKE,AEcijM->AbcijK', H.bb.voov[vb, Ob, Ob, Vb], T.abb.VVvooO, optimize=True)
    )
    dT.abb.VvvooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mbie,AcemjK->AbcijK', H.ab.ovov[oa, vb, oa, vb], T.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('Mbie,AceMjK->AbcijK', H.ab.ovov[Oa, vb, oa, vb], T.abb.VvvOoO, optimize=True)
            - 1.0 * np.einsum('mbiE,AEcmjK->AbcijK', H.ab.ovov[oa, vb, oa, Vb], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('MbiE,AEcMjK->AbcijK', H.ab.ovov[Oa, vb, oa, Vb], T.abb.VVvOoO, optimize=True)
    )
    dT.abb.VvvooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('AmEj,EcbimK->AbcijK', H.ab.vovo[Va, ob, Va, ob], T.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('AMEj,EcbiMK->AbcijK', H.ab.vovo[Va, Ob, Va, ob], T.abb.VvvoOO, optimize=True)
    )
    dT.abb.VvvooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('AMEK,EcbijM->AbcijK', H.ab.vovo[Va, Ob, Va, Ob], T.abb.VvvooO, optimize=True)
    )

    dT.abb.VvvooO -= np.transpose(dT.abb.VvvooO, (0, 2, 1, 3, 4, 5))

    return dT


def update(T, dT, H, shift, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.abb.VvvooO, dT.abb.VvvooO = cc_active_loops.update_t3c_100001(
        T.abb.VvvooO,
        dT.abb.VvvooO,
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