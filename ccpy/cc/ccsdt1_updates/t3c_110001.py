import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import cc_active_loops

def build(T, dT, H, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    # MM(2,3)
    dT.abb.VVvooO = (1.0 / 1.0) * (
            +1.0 * np.einsum('ABie,ecjK->ABcijK', H.ab.vvov[Va, Vb, oa, :], T.bb[:, vb, ob, Ob], optimize=True)
    )
    dT.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('Acie,eBjK->ABcijK', H.ab.vvov[Va, vb, oa, :], T.bb[:, Vb, ob, Ob], optimize=True)
    )
    dT.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('Amij,BcmK->ABcijK', H.ab.vooo[Va, :, oa, ob], T.bb[Vb, vb, :, Ob], optimize=True)
    )
    dT.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('AmiK,Bcmj->ABcijK', H.ab.vooo[Va, :, oa, Ob], T.bb[Vb, vb, :, ob], optimize=True)
    )
    dT.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('cBKe,Aeij->ABcijK', H.bb.vvov[vb, Vb, Ob, :], T.ab[Va, :, oa, ob], optimize=True)
    )
    dT.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('cBje,AeiK->ABcijK', H.bb.vvov[vb, Vb, ob, :], T.ab[Va, :, oa, Ob], optimize=True)
    )
    dT.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('cmKj,ABim->ABcijK', H.bb.vooo[vb, :, Ob, ob], T.ab[Va, Vb, oa, :], optimize=True)
    )
    dT.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('BmKj,Acim->ABcijK', H.bb.vooo[Vb, :, Ob, ob], T.ab[Va, vb, oa, :], optimize=True)
    )
    dT.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('ABej,eciK->ABcijK', H.ab.vvvo[Va, Vb, :, ob], T.ab[:, vb, oa, Ob], optimize=True)
    )
    dT.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('Acej,eBiK->ABcijK', H.ab.vvvo[Va, vb, :, ob], T.ab[:, Vb, oa, Ob], optimize=True)
    )
    dT.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('ABeK,ecij->ABcijK', H.ab.vvvo[Va, Vb, :, Ob], T.ab[:, vb, oa, ob], optimize=True)
    )
    dT.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('AceK,eBij->ABcijK', H.ab.vvvo[Va, vb, :, Ob], T.ab[:, Vb, oa, ob], optimize=True)
    )
    dT.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mBij,AcmK->ABcijK', H.ab.ovoo[:, Vb, oa, ob], T.ab[Va, vb, :, Ob], optimize=True)
    )
    dT.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mcij,ABmK->ABcijK', H.ab.ovoo[:, vb, oa, ob], T.ab[Va, Vb, :, Ob], optimize=True)
    )
    dT.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mBiK,Acmj->ABcijK', H.ab.ovoo[:, Vb, oa, Ob], T.ab[Va, vb, :, ob], optimize=True)
    )
    dT.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mciK,ABmj->ABcijK', H.ab.ovoo[:, vb, oa, Ob], T.ab[Va, Vb, :, ob], optimize=True)
    )
    # (H(2) * T3)_C
    dT.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mi,ABcmjK->ABcijK', H.a.oo[oa, oa], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('Mi,ABcMjK->ABcijK', H.a.oo[Oa, oa], T.abb.VVvOoO, optimize=True)
    )
    dT.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mj,ABcimK->ABcijK', H.b.oo[ob, ob], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('Mj,ABciMK->ABcijK', H.b.oo[Ob, ob], T.abb.VVvoOO, optimize=True)
    )
    dT.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('MK,ABcijM->ABcijK', H.b.oo[Ob, Ob], T.abb.VVvooO, optimize=True)
    )
    dT.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('Ae,eBcijK->ABcijK', H.a.vv[Va, va], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('AE,EBcijK->ABcijK', H.a.vv[Va, Va], T.abb.VVvooO, optimize=True)
    )
    dT.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('Be,AceijK->ABcijK', H.b.vv[Vb, vb], T.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('BE,AEcijK->ABcijK', H.b.vv[Vb, Vb], T.abb.VVvooO, optimize=True)
    )
    dT.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('ce,ABeijK->ABcijK', H.b.vv[vb, vb], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('cE,ABEijK->ABcijK', H.b.vv[vb, Vb], T.abb.VVVooO, optimize=True)
    )
    dT.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mNjK,ABcimN->ABcijK', H.bb.oooo[ob, Ob, ob, Ob], T.abb.VVvooO, optimize=True)
            + 0.5 * np.einsum('MNjK,ABciMN->ABcijK', H.bb.oooo[Ob, Ob, ob, Ob], T.abb.VVvoOO, optimize=True)
    )
    dT.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mnij,ABcmnK->ABcijK', H.ab.oooo[oa, ob, oa, ob], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('Mnij,ABcMnK->ABcijK', H.ab.oooo[Oa, ob, oa, ob], T.abb.VVvOoO, optimize=True)
            + 1.0 * np.einsum('mNij,ABcmNK->ABcijK', H.ab.oooo[oa, Ob, oa, ob], T.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('MNij,ABcMNK->ABcijK', H.ab.oooo[Oa, Ob, oa, ob], T.abb.VVvOOO, optimize=True)
    )
    dT.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('MniK,ABcMnj->ABcijK', H.ab.oooo[Oa, ob, oa, Ob], T.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('mNiK,ABcmjN->ABcijK', H.ab.oooo[oa, Ob, oa, Ob], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('MNiK,ABcMjN->ABcijK', H.ab.oooo[Oa, Ob, oa, Ob], T.abb.VVvOoO, optimize=True)
    )
    dT.abb.VVvooO += (1.0 / 1.0) * (
            -0.5 * np.einsum('Bcef,AfeijK->ABcijK', H.bb.vvvv[Vb, vb, vb, vb], T.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('BceF,AFeijK->ABcijK', H.bb.vvvv[Vb, vb, vb, Vb], T.abb.VVvooO, optimize=True)
            - 0.5 * np.einsum('BcEF,AFEijK->ABcijK', H.bb.vvvv[Vb, vb, Vb, Vb], T.abb.VVVooO, optimize=True)
    )
    dT.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('ABEf,EcfijK->ABcijK', H.ab.vvvv[Va, Vb, Va, vb], T.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('ABeF,eFcijK->ABcijK', H.ab.vvvv[Va, Vb, va, Vb], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('ABEF,EFcijK->ABcijK', H.ab.vvvv[Va, Vb, Va, Vb], T.abb.VVvooO, optimize=True)
    )
    dT.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('Acef,eBfijK->ABcijK', H.ab.vvvv[Va, vb, va, vb], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('AcEf,EBfijK->ABcijK', H.ab.vvvv[Va, vb, Va, vb], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('AceF,eBFijK->ABcijK', H.ab.vvvv[Va, vb, va, Vb], T.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('AcEF,EBFijK->ABcijK', H.ab.vvvv[Va, vb, Va, Vb], T.abb.VVVooO, optimize=True)
    )
    dT.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('Amie,eBcmjK->ABcijK', H.aa.voov[Va, oa, oa, va], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('AMie,eBcMjK->ABcijK', H.aa.voov[Va, Oa, oa, va], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('AmiE,EBcmjK->ABcijK', H.aa.voov[Va, oa, oa, Va], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('AMiE,EBcMjK->ABcijK', H.aa.voov[Va, Oa, oa, Va], T.abb.VVvOoO, optimize=True)
    )
    dT.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('Amie,BcemjK->ABcijK', H.ab.voov[Va, ob, oa, vb], T.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('AMie,BcejMK->ABcijK', H.ab.voov[Va, Ob, oa, vb], T.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('AmiE,BEcmjK->ABcijK', H.ab.voov[Va, ob, oa, Vb], T.bbb.VVvooO, optimize=True)
            + 1.0 * np.einsum('AMiE,BEcjMK->ABcijK', H.ab.voov[Va, Ob, oa, Vb], T.bbb.VVvoOO, optimize=True)
    )
    dT.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mBej,AecimK->ABcijK', H.ab.ovvo[oa, Vb, va, ob], T.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('MBej,AeciMK->ABcijK', H.ab.ovvo[Oa, Vb, va, ob], T.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('mBEj,EAcimK->ABcijK', H.ab.ovvo[oa, Vb, Va, ob], T.aab.VVvooO, optimize=True)
            - 1.0 * np.einsum('MBEj,EAciMK->ABcijK', H.ab.ovvo[Oa, Vb, Va, ob], T.aab.VVvoOO, optimize=True)
    )
    dT.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mcej,AeBimK->ABcijK', H.ab.ovvo[oa, vb, va, ob], T.aab.VvVooO, optimize=True)
            - 1.0 * np.einsum('Mcej,AeBiMK->ABcijK', H.ab.ovvo[Oa, vb, va, ob], T.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('mcEj,EABimK->ABcijK', H.ab.ovvo[oa, vb, Va, ob], T.aab.VVVooO, optimize=True)
            + 1.0 * np.einsum('McEj,EABiMK->ABcijK', H.ab.ovvo[Oa, vb, Va, ob], T.aab.VVVoOO, optimize=True)
    )
    dT.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('MBeK,AeciMj->ABcijK', H.ab.ovvo[Oa, Vb, va, Ob], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('MBEK,EAciMj->ABcijK', H.ab.ovvo[Oa, Vb, Va, Ob], T.aab.VVvoOo, optimize=True)
    )
    dT.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('MceK,AeBiMj->ABcijK', H.ab.ovvo[Oa, vb, va, Ob], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('McEK,EABiMj->ABcijK', H.ab.ovvo[Oa, vb, Va, Ob], T.aab.VVVoOo, optimize=True)
    )
    dT.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('Bmje,AceimK->ABcijK', H.bb.voov[Vb, ob, ob, vb], T.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('BMje,AceiMK->ABcijK', H.bb.voov[Vb, Ob, ob, vb], T.abb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('BmjE,AEcimK->ABcijK', H.bb.voov[Vb, ob, ob, Vb], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('BMjE,AEciMK->ABcijK', H.bb.voov[Vb, Ob, ob, Vb], T.abb.VVvoOO, optimize=True)
    )
    dT.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('cmje,ABeimK->ABcijK', H.bb.voov[vb, ob, ob, vb], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('cMje,ABeiMK->ABcijK', H.bb.voov[vb, Ob, ob, vb], T.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('cmjE,ABEimK->ABcijK', H.bb.voov[vb, ob, ob, Vb], T.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('cMjE,ABEiMK->ABcijK', H.bb.voov[vb, Ob, ob, Vb], T.abb.VVVoOO, optimize=True)
    )
    dT.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('BMKe,AceijM->ABcijK', H.bb.voov[Vb, Ob, Ob, vb], T.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('BMKE,AEcijM->ABcijK', H.bb.voov[Vb, Ob, Ob, Vb], T.abb.VVvooO, optimize=True)
    )
    dT.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('cMKe,ABeijM->ABcijK', H.bb.voov[vb, Ob, Ob, vb], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('cMKE,ABEijM->ABcijK', H.bb.voov[vb, Ob, Ob, Vb], T.abb.VVVooO, optimize=True)
    )
    dT.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mBie,AcemjK->ABcijK', H.ab.ovov[oa, Vb, oa, vb], T.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('MBie,AceMjK->ABcijK', H.ab.ovov[Oa, Vb, oa, vb], T.abb.VvvOoO, optimize=True)
            - 1.0 * np.einsum('mBiE,AEcmjK->ABcijK', H.ab.ovov[oa, Vb, oa, Vb], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('MBiE,AEcMjK->ABcijK', H.ab.ovov[Oa, Vb, oa, Vb], T.abb.VVvOoO, optimize=True)
    )
    dT.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mcie,ABemjK->ABcijK', H.ab.ovov[oa, vb, oa, vb], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('Mcie,ABeMjK->ABcijK', H.ab.ovov[Oa, vb, oa, vb], T.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('mciE,ABEmjK->ABcijK', H.ab.ovov[oa, vb, oa, Vb], T.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('MciE,ABEMjK->ABcijK', H.ab.ovov[Oa, vb, oa, Vb], T.abb.VVVOoO, optimize=True)
    )
    dT.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('Amej,eBcimK->ABcijK', H.ab.vovo[Va, ob, va, ob], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('AMej,eBciMK->ABcijK', H.ab.vovo[Va, Ob, va, ob], T.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('AmEj,EBcimK->ABcijK', H.ab.vovo[Va, ob, Va, ob], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('AMEj,EBciMK->ABcijK', H.ab.vovo[Va, Ob, Va, ob], T.abb.VVvoOO, optimize=True)
    )
    dT.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('AMeK,eBcijM->ABcijK', H.ab.vovo[Va, Ob, va, Ob], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('AMEK,EBcijM->ABcijK', H.ab.vovo[Va, Ob, Va, Ob], T.abb.VVvooO, optimize=True)
    )

    return dT


def update(T, dT, H, shift, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.abb.VVvooO, dT.abb.VVvooO = cc_active_loops.update_t3c_110001(
        T.abb.VVvooO,
        dT.abb.VVvooO,
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