import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import cc_active_loops

def build(T, dT, H, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    # MM(2,3)
    dT.abb.VVvoOO = (1.0 / 2.0) * (
            +1.0 * np.einsum('ABie,ecJK->ABciJK', H.ab.vvov[Va, Vb, oa, :], T.bb[:, vb, Ob, Ob], optimize=True)
    )
    dT.abb.VVvoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('Acie,eBJK->ABciJK', H.ab.vvov[Va, vb, oa, :], T.bb[:, Vb, Ob, Ob], optimize=True)
    )
    dT.abb.VVvoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('AmiJ,BcmK->ABciJK', H.ab.vooo[Va, :, oa, Ob], T.bb[Vb, vb, :, Ob], optimize=True)
    )
    dT.abb.VVvoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('cBKe,AeiJ->ABciJK', H.bb.vvov[vb, Vb, Ob, :], T.ab[Va, :, oa, Ob], optimize=True)
    )
    dT.abb.VVvoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('cmKJ,ABim->ABciJK', H.bb.vooo[vb, :, Ob, Ob], T.ab[Va, Vb, oa, :], optimize=True)
    )
    dT.abb.VVvoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('BmKJ,Acim->ABciJK', H.bb.vooo[Vb, :, Ob, Ob], T.ab[Va, vb, oa, :], optimize=True)
    )
    dT.abb.VVvoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('ABeJ,eciK->ABciJK', H.ab.vvvo[Va, Vb, :, Ob], T.ab[:, vb, oa, Ob], optimize=True)
    )
    dT.abb.VVvoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('AceJ,eBiK->ABciJK', H.ab.vvvo[Va, vb, :, Ob], T.ab[:, Vb, oa, Ob], optimize=True)
    )
    dT.abb.VVvoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mBiJ,AcmK->ABciJK', H.ab.ovoo[:, Vb, oa, Ob], T.ab[Va, vb, :, Ob], optimize=True)
    )
    dT.abb.VVvoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mciJ,ABmK->ABciJK', H.ab.ovoo[:, vb, oa, Ob], T.ab[Va, Vb, :, Ob], optimize=True)
    )
    # (H(2) * T3)_C
    dT.abb.VVvoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mi,ABcmJK->ABciJK', H.a.oo[oa, oa], T.abb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('Mi,ABcMJK->ABciJK', H.a.oo[Oa, oa], T.abb.VVvOOO, optimize=True)
    )
    dT.abb.VVvoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mJ,ABcimK->ABciJK', H.b.oo[ob, Ob], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('MJ,ABciMK->ABciJK', H.b.oo[Ob, Ob], T.abb.VVvoOO, optimize=True)
    )
    dT.abb.VVvoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('Ae,eBciJK->ABciJK', H.a.vv[Va, va], T.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('AE,EBciJK->ABciJK', H.a.vv[Va, Va], T.abb.VVvoOO, optimize=True)
    )
    dT.abb.VVvoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('Be,AceiJK->ABciJK', H.b.vv[Vb, vb], T.abb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('BE,AEciJK->ABciJK', H.b.vv[Vb, Vb], T.abb.VVvoOO, optimize=True)
    )
    dT.abb.VVvoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('ce,ABeiJK->ABciJK', H.b.vv[vb, vb], T.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('cE,ABEiJK->ABciJK', H.b.vv[vb, Vb], T.abb.VVVoOO, optimize=True)
    )
    dT.abb.VVvoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mNJK,ABcimN->ABciJK', H.bb.oooo[ob, Ob, Ob, Ob], T.abb.VVvooO, optimize=True)
            + 0.5 * np.einsum('MNJK,ABciMN->ABciJK', H.bb.oooo[Ob, Ob, Ob, Ob], T.abb.VVvoOO, optimize=True)
    )
    dT.abb.VVvoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mniJ,ABcmnK->ABciJK', H.ab.oooo[oa, ob, oa, Ob], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('MniJ,ABcMnK->ABciJK', H.ab.oooo[Oa, ob, oa, Ob], T.abb.VVvOoO, optimize=True)
            + 1.0 * np.einsum('mNiJ,ABcmNK->ABciJK', H.ab.oooo[oa, Ob, oa, Ob], T.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('MNiJ,ABcMNK->ABciJK', H.ab.oooo[Oa, Ob, oa, Ob], T.abb.VVvOOO, optimize=True)
    )
    dT.abb.VVvoOO += (1.0 / 2.0) * (
            -0.5 * np.einsum('Bcef,AfeiJK->ABciJK', H.bb.vvvv[Vb, vb, vb, vb], T.abb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('BcEf,AEfiJK->ABciJK', H.bb.vvvv[Vb, vb, Vb, vb], T.abb.VVvoOO, optimize=True)
            - 0.5 * np.einsum('BcEF,AFEiJK->ABciJK', H.bb.vvvv[Vb, vb, Vb, Vb], T.abb.VVVoOO, optimize=True)
    )
    dT.abb.VVvoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('ABeF,eFciJK->ABciJK', H.ab.vvvv[Va, Vb, va, Vb], T.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('ABEf,EcfiJK->ABciJK', H.ab.vvvv[Va, Vb, Va, vb], T.abb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('ABEF,EFciJK->ABciJK', H.ab.vvvv[Va, Vb, Va, Vb], T.abb.VVvoOO, optimize=True)
    )
    dT.abb.VVvoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('Acef,eBfiJK->ABciJK', H.ab.vvvv[Va, vb, va, vb], T.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('AceF,eBFiJK->ABciJK', H.ab.vvvv[Va, vb, va, Vb], T.abb.vVVoOO, optimize=True)
            + 1.0 * np.einsum('AcEf,EBfiJK->ABciJK', H.ab.vvvv[Va, vb, Va, vb], T.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('AcEF,EBFiJK->ABciJK', H.ab.vvvv[Va, vb, Va, Vb], T.abb.VVVoOO, optimize=True)
    )
    dT.abb.VVvoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('Amie,eBcmJK->ABciJK', H.aa.voov[Va, oa, oa, va], T.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('AMie,eBcMJK->ABciJK', H.aa.voov[Va, Oa, oa, va], T.abb.vVvOOO, optimize=True)
            + 1.0 * np.einsum('AmiE,EBcmJK->ABciJK', H.aa.voov[Va, oa, oa, Va], T.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('AMiE,EBcMJK->ABciJK', H.aa.voov[Va, Oa, oa, Va], T.abb.VVvOOO, optimize=True)
    )
    dT.abb.VVvoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('Amie,BcemJK->ABciJK', H.ab.voov[Va, ob, oa, vb], T.bbb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('AMie,BceMJK->ABciJK', H.ab.voov[Va, Ob, oa, vb], T.bbb.VvvOOO, optimize=True)
            - 1.0 * np.einsum('AmiE,BEcmJK->ABciJK', H.ab.voov[Va, ob, oa, Vb], T.bbb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('AMiE,BEcMJK->ABciJK', H.ab.voov[Va, Ob, oa, Vb], T.bbb.VVvOOO, optimize=True)
    )
    dT.abb.VVvoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mBeJ,AecimK->ABciJK', H.ab.ovvo[oa, Vb, va, Ob], T.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('MBeJ,AeciMK->ABciJK', H.ab.ovvo[Oa, Vb, va, Ob], T.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('mBEJ,EAcimK->ABciJK', H.ab.ovvo[oa, Vb, Va, Ob], T.aab.VVvooO, optimize=True)
            - 1.0 * np.einsum('MBEJ,EAciMK->ABciJK', H.ab.ovvo[Oa, Vb, Va, Ob], T.aab.VVvoOO, optimize=True)
    )
    dT.abb.VVvoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mceJ,AeBimK->ABciJK', H.ab.ovvo[oa, vb, va, Ob], T.aab.VvVooO, optimize=True)
            - 1.0 * np.einsum('MceJ,AeBiMK->ABciJK', H.ab.ovvo[Oa, vb, va, Ob], T.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('mcEJ,EABimK->ABciJK', H.ab.ovvo[oa, vb, Va, Ob], T.aab.VVVooO, optimize=True)
            + 1.0 * np.einsum('McEJ,EABiMK->ABciJK', H.ab.ovvo[Oa, vb, Va, Ob], T.aab.VVVoOO, optimize=True)
    )
    dT.abb.VVvoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('BmJe,AceimK->ABciJK', H.bb.voov[Vb, ob, Ob, vb], T.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('BMJe,AceiMK->ABciJK', H.bb.voov[Vb, Ob, Ob, vb], T.abb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('BmJE,AEcimK->ABciJK', H.bb.voov[Vb, ob, Ob, Vb], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('BMJE,AEciMK->ABciJK', H.bb.voov[Vb, Ob, Ob, Vb], T.abb.VVvoOO, optimize=True)
    )
    dT.abb.VVvoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('cmJe,ABeimK->ABciJK', H.bb.voov[vb, ob, Ob, vb], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('cMJe,ABeiMK->ABciJK', H.bb.voov[vb, Ob, Ob, vb], T.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('cmJE,ABEimK->ABciJK', H.bb.voov[vb, ob, Ob, Vb], T.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('cMJE,ABEiMK->ABciJK', H.bb.voov[vb, Ob, Ob, Vb], T.abb.VVVoOO, optimize=True)
    )
    dT.abb.VVvoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mBie,AcemJK->ABciJK', H.ab.ovov[oa, Vb, oa, vb], T.abb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('MBie,AceMJK->ABciJK', H.ab.ovov[Oa, Vb, oa, vb], T.abb.VvvOOO, optimize=True)
            - 1.0 * np.einsum('mBiE,AEcmJK->ABciJK', H.ab.ovov[oa, Vb, oa, Vb], T.abb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('MBiE,AEcMJK->ABciJK', H.ab.ovov[Oa, Vb, oa, Vb], T.abb.VVvOOO, optimize=True)
    )
    dT.abb.VVvoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mcie,ABemJK->ABciJK', H.ab.ovov[oa, vb, oa, vb], T.abb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('Mcie,ABeMJK->ABciJK', H.ab.ovov[Oa, vb, oa, vb], T.abb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('mciE,ABEmJK->ABciJK', H.ab.ovov[oa, vb, oa, Vb], T.abb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('MciE,ABEMJK->ABciJK', H.ab.ovov[Oa, vb, oa, Vb], T.abb.VVVOOO, optimize=True)
    )
    dT.abb.VVvoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('AmeJ,eBcimK->ABciJK', H.ab.vovo[Va, ob, va, Ob], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('AMeJ,eBciMK->ABciJK', H.ab.vovo[Va, Ob, va, Ob], T.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('AmEJ,EBcimK->ABciJK', H.ab.vovo[Va, ob, Va, Ob], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('AMEJ,EBciMK->ABciJK', H.ab.vovo[Va, Ob, Va, Ob], T.abb.VVvoOO, optimize=True)
    )

    dT.abb.VVvoOO -= np.transpose(dT.abb.VVvoOO, (0, 1, 2, 3, 5, 4))

    return dT


def update(T, dT, H, shift, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.abb.VVvoOO, dT.abb.VVvoOO = cc_active_loops.update_t3c_110011(
        T.abb.VVvoOO,
        dT.abb.VVvoOO,
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
