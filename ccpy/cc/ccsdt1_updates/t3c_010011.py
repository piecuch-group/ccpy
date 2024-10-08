import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import cc_active_loops

def build(T, dT, H, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    # MM(2,3)
    dT.abb.vVvoOO = (1.0 / 2.0) * (
            +1.0 * np.einsum('aBie,ecJK->aBciJK', H.ab.vvov[va, Vb, oa, :], T.bb[:, vb, Ob, Ob], optimize=True)
    )
    dT.abb.vVvoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('acie,eBJK->aBciJK', H.ab.vvov[va, vb, oa, :], T.bb[:, Vb, Ob, Ob], optimize=True)
    )
    dT.abb.vVvoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('amiJ,BcmK->aBciJK', H.ab.vooo[va, :, oa, Ob], T.bb[Vb, vb, :, Ob], optimize=True)
    )
    dT.abb.vVvoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('cBKe,aeiJ->aBciJK', H.bb.vvov[vb, Vb, Ob, :], T.ab[va, :, oa, Ob], optimize=True)
    )
    dT.abb.vVvoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('cmKJ,aBim->aBciJK', H.bb.vooo[vb, :, Ob, Ob], T.ab[va, Vb, oa, :], optimize=True)
    )
    dT.abb.vVvoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('BmKJ,acim->aBciJK', H.bb.vooo[Vb, :, Ob, Ob], T.ab[va, vb, oa, :], optimize=True)
    )
    dT.abb.vVvoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('aBeJ,eciK->aBciJK', H.ab.vvvo[va, Vb, :, Ob], T.ab[:, vb, oa, Ob], optimize=True)
    )
    dT.abb.vVvoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('aceJ,eBiK->aBciJK', H.ab.vvvo[va, vb, :, Ob], T.ab[:, Vb, oa, Ob], optimize=True)
    )
    dT.abb.vVvoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mBiJ,acmK->aBciJK', H.ab.ovoo[:, Vb, oa, Ob], T.ab[va, vb, :, Ob], optimize=True)
    )
    dT.abb.vVvoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mciJ,aBmK->aBciJK', H.ab.ovoo[:, vb, oa, Ob], T.ab[va, Vb, :, Ob], optimize=True)
    )
    # (H(2) * T3)_C
    dT.abb.vVvoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mi,aBcmJK->aBciJK', H.a.oo[oa, oa], T.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('Mi,aBcMJK->aBciJK', H.a.oo[Oa, oa], T.abb.vVvOOO, optimize=True)
    )
    dT.abb.vVvoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mJ,aBcimK->aBciJK', H.b.oo[ob, Ob], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('MJ,aBciMK->aBciJK', H.b.oo[Ob, Ob], T.abb.vVvoOO, optimize=True)
    )
    dT.abb.vVvoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('ae,eBciJK->aBciJK', H.a.vv[va, va], T.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('aE,EBciJK->aBciJK', H.a.vv[va, Va], T.abb.VVvoOO, optimize=True)
    )
    dT.abb.vVvoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('BE,aEciJK->aBciJK', H.b.vv[Vb, Vb], T.abb.vVvoOO, optimize=True)
    )
    dT.abb.vVvoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('ce,aBeiJK->aBciJK', H.b.vv[vb, vb], T.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('cE,aBEiJK->aBciJK', H.b.vv[vb, Vb], T.abb.vVVoOO, optimize=True)
    )
    dT.abb.vVvoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mNJK,aBcimN->aBciJK', H.bb.oooo[ob, Ob, Ob, Ob], T.abb.vVvooO, optimize=True)
            + 0.5 * np.einsum('MNJK,aBciMN->aBciJK', H.bb.oooo[Ob, Ob, Ob, Ob], T.abb.vVvoOO, optimize=True)
    )
    dT.abb.vVvoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mniJ,aBcmnK->aBciJK', H.ab.oooo[oa, ob, oa, Ob], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('MniJ,aBcMnK->aBciJK', H.ab.oooo[Oa, ob, oa, Ob], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('mNiJ,aBcmNK->aBciJK', H.ab.oooo[oa, Ob, oa, Ob], T.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('MNiJ,aBcMNK->aBciJK', H.ab.oooo[Oa, Ob, oa, Ob], T.abb.vVvOOO, optimize=True)
    )
    dT.abb.vVvoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('BcEf,aEfiJK->aBciJK', H.bb.vvvv[Vb, vb, Vb, vb], T.abb.vVvoOO, optimize=True)
            - 0.5 * np.einsum('BcEF,aFEiJK->aBciJK', H.bb.vvvv[Vb, vb, Vb, Vb], T.abb.vVVoOO, optimize=True)
    )
    dT.abb.vVvoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('aBeF,eFciJK->aBciJK', H.ab.vvvv[va, Vb, va, Vb], T.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('aBEf,EcfiJK->aBciJK', H.ab.vvvv[va, Vb, Va, vb], T.abb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('aBEF,EFciJK->aBciJK', H.ab.vvvv[va, Vb, Va, Vb], T.abb.VVvoOO, optimize=True)
    )
    dT.abb.vVvoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('acef,eBfiJK->aBciJK', H.ab.vvvv[va, vb, va, vb], T.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('aceF,eBFiJK->aBciJK', H.ab.vvvv[va, vb, va, Vb], T.abb.vVVoOO, optimize=True)
            + 1.0 * np.einsum('acEf,EBfiJK->aBciJK', H.ab.vvvv[va, vb, Va, vb], T.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('acEF,EBFiJK->aBciJK', H.ab.vvvv[va, vb, Va, Vb], T.abb.VVVoOO, optimize=True)
    )
    dT.abb.vVvoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('amie,eBcmJK->aBciJK', H.aa.voov[va, oa, oa, va], T.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('aMie,eBcMJK->aBciJK', H.aa.voov[va, Oa, oa, va], T.abb.vVvOOO, optimize=True)
            + 1.0 * np.einsum('amiE,EBcmJK->aBciJK', H.aa.voov[va, oa, oa, Va], T.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('aMiE,EBcMJK->aBciJK', H.aa.voov[va, Oa, oa, Va], T.abb.VVvOOO, optimize=True)
    )
    dT.abb.vVvoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('amie,BcemJK->aBciJK', H.ab.voov[va, ob, oa, vb], T.bbb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('aMie,BceMJK->aBciJK', H.ab.voov[va, Ob, oa, vb], T.bbb.VvvOOO, optimize=True)
            - 1.0 * np.einsum('amiE,BEcmJK->aBciJK', H.ab.voov[va, ob, oa, Vb], T.bbb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('aMiE,BEcMJK->aBciJK', H.ab.voov[va, Ob, oa, Vb], T.bbb.VVvOOO, optimize=True)
    )
    dT.abb.vVvoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mBEJ,EacimK->aBciJK', H.ab.ovvo[oa, Vb, Va, Ob], T.aab.VvvooO, optimize=True)
            - 1.0 * np.einsum('MBEJ,EaciMK->aBciJK', H.ab.ovvo[Oa, Vb, Va, Ob], T.aab.VvvoOO, optimize=True)
    )
    dT.abb.vVvoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mceJ,eaBimK->aBciJK', H.ab.ovvo[oa, vb, va, Ob], T.aab.vvVooO, optimize=True)
            + 1.0 * np.einsum('MceJ,eaBiMK->aBciJK', H.ab.ovvo[Oa, vb, va, Ob], T.aab.vvVoOO, optimize=True)
            + 1.0 * np.einsum('mcEJ,EaBimK->aBciJK', H.ab.ovvo[oa, vb, Va, Ob], T.aab.VvVooO, optimize=True)
            + 1.0 * np.einsum('McEJ,EaBiMK->aBciJK', H.ab.ovvo[Oa, vb, Va, Ob], T.aab.VvVoOO, optimize=True)
    )
    dT.abb.vVvoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('BmJE,aEcimK->aBciJK', H.bb.voov[Vb, ob, Ob, Vb], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('BMJE,aEciMK->aBciJK', H.bb.voov[Vb, Ob, Ob, Vb], T.abb.vVvoOO, optimize=True)
    )
    dT.abb.vVvoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('cmJe,aBeimK->aBciJK', H.bb.voov[vb, ob, Ob, vb], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('cMJe,aBeiMK->aBciJK', H.bb.voov[vb, Ob, Ob, vb], T.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('cmJE,aBEimK->aBciJK', H.bb.voov[vb, ob, Ob, Vb], T.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('cMJE,aBEiMK->aBciJK', H.bb.voov[vb, Ob, Ob, Vb], T.abb.vVVoOO, optimize=True)
    )
    dT.abb.vVvoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mBiE,aEcmJK->aBciJK', H.ab.ovov[oa, Vb, oa, Vb], T.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('MBiE,aEcMJK->aBciJK', H.ab.ovov[Oa, Vb, oa, Vb], T.abb.vVvOOO, optimize=True)
    )
    dT.abb.vVvoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mcie,aBemJK->aBciJK', H.ab.ovov[oa, vb, oa, vb], T.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('Mcie,aBeMJK->aBciJK', H.ab.ovov[Oa, vb, oa, vb], T.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('mciE,aBEmJK->aBciJK', H.ab.ovov[oa, vb, oa, Vb], T.abb.vVVoOO, optimize=True)
            - 1.0 * np.einsum('MciE,aBEMJK->aBciJK', H.ab.ovov[Oa, vb, oa, Vb], T.abb.vVVOOO, optimize=True)
    )
    dT.abb.vVvoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('ameJ,eBcimK->aBciJK', H.ab.vovo[va, ob, va, Ob], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('aMeJ,eBciMK->aBciJK', H.ab.vovo[va, Ob, va, Ob], T.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('amEJ,EBcimK->aBciJK', H.ab.vovo[va, ob, Va, Ob], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('aMEJ,EBciMK->aBciJK', H.ab.vovo[va, Ob, Va, Ob], T.abb.VVvoOO, optimize=True)
    )

    dT.abb.vVvoOO -= np.transpose(dT.abb.vVvoOO, (0, 1, 2, 3, 5, 4))

    return dT


def update(T, dT, H, shift, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.abb.vVvoOO, dT.abb.vVvoOO = cc_active_loops.update_t3c_010011(
        T.abb.vVvoOO,
        dT.abb.vVvoOO,
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