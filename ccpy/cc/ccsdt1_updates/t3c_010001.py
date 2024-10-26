import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import cc_active_loops

def build(T, dT, H, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    # MM(2,3)
    dT.abb.vVvooO = (1.0 / 1.0) * (
            +1.0 * np.einsum('aBie,ecjK->aBcijK', H.ab.vvov[va, Vb, oa, :], T.bb[:, vb, ob, Ob], optimize=True)
    )
    dT.abb.vVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('acie,eBjK->aBcijK', H.ab.vvov[va, vb, oa, :], T.bb[:, Vb, ob, Ob], optimize=True)
    )
    dT.abb.vVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('amij,BcmK->aBcijK', H.ab.vooo[va, :, oa, ob], T.bb[Vb, vb, :, Ob], optimize=True)
    )
    dT.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('amiK,Bcmj->aBcijK', H.ab.vooo[va, :, oa, Ob], T.bb[Vb, vb, :, ob], optimize=True)
    )
    dT.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('cBKe,aeij->aBcijK', H.bb.vvov[vb, Vb, Ob, :], T.ab[va, :, oa, ob], optimize=True)
    )
    dT.abb.vVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('cBje,aeiK->aBcijK', H.bb.vvov[vb, Vb, ob, :], T.ab[va, :, oa, Ob], optimize=True)
    )
    dT.abb.vVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('cmKj,aBim->aBcijK', H.bb.vooo[vb, :, Ob, ob], T.ab[va, Vb, oa, :], optimize=True)
    )
    dT.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('BmKj,acim->aBcijK', H.bb.vooo[Vb, :, Ob, ob], T.ab[va, vb, oa, :], optimize=True)
    )
    dT.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('aBej,eciK->aBcijK', H.ab.vvvo[va, Vb, :, ob], T.ab[:, vb, oa, Ob], optimize=True)
    )
    dT.abb.vVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('acej,eBiK->aBcijK', H.ab.vvvo[va, vb, :, ob], T.ab[:, Vb, oa, Ob], optimize=True)
    )
    dT.abb.vVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('aBeK,ecij->aBcijK', H.ab.vvvo[va, Vb, :, Ob], T.ab[:, vb, oa, ob], optimize=True)
    )
    dT.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('aceK,eBij->aBcijK', H.ab.vvvo[va, vb, :, Ob], T.ab[:, Vb, oa, ob], optimize=True)
    )
    dT.abb.vVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mBij,acmK->aBcijK', H.ab.ovoo[:, Vb, oa, ob], T.ab[va, vb, :, Ob], optimize=True)
    )
    dT.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mcij,aBmK->aBcijK', H.ab.ovoo[:, vb, oa, ob], T.ab[va, Vb, :, Ob], optimize=True)
    )
    dT.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mBiK,acmj->aBcijK', H.ab.ovoo[:, Vb, oa, Ob], T.ab[va, vb, :, ob], optimize=True)
    )
    dT.abb.vVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mciK,aBmj->aBcijK', H.ab.ovoo[:, vb, oa, Ob], T.ab[va, Vb, :, ob], optimize=True)
    )
    # (H(2) * T3)_C
    dT.abb.vVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mi,aBcmjK->aBcijK', H.a.oo[oa, oa], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('Mi,aBcMjK->aBcijK', H.a.oo[Oa, oa], T.abb.vVvOoO, optimize=True)
    )
    dT.abb.vVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mj,aBcimK->aBcijK', H.b.oo[ob, ob], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('Mj,aBciMK->aBcijK', H.b.oo[Ob, ob], T.abb.vVvoOO, optimize=True)
    )
    dT.abb.vVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('MK,aBcijM->aBcijK', H.b.oo[Ob, Ob], T.abb.vVvooO, optimize=True)
    )
    dT.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('ae,eBcijK->aBcijK', H.a.vv[va, va], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('aE,EBcijK->aBcijK', H.a.vv[va, Va], T.abb.VVvooO, optimize=True)
    )
    dT.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('BE,aEcijK->aBcijK', H.b.vv[Vb, Vb], T.abb.vVvooO, optimize=True)
    )
    dT.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('ce,aBeijK->aBcijK', H.b.vv[vb, vb], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('cE,aBEijK->aBcijK', H.b.vv[vb, Vb], T.abb.vVVooO, optimize=True)
    )
    dT.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mNjK,aBcimN->aBcijK', H.bb.oooo[ob, Ob, ob, Ob], T.abb.vVvooO, optimize=True)
            + 0.5 * np.einsum('MNjK,aBciMN->aBcijK', H.bb.oooo[Ob, Ob, ob, Ob], T.abb.vVvoOO, optimize=True)
    )
    dT.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mnij,aBcmnK->aBcijK', H.ab.oooo[oa, ob, oa, ob], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('Mnij,aBcMnK->aBcijK', H.ab.oooo[Oa, ob, oa, ob], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('mNij,aBcmNK->aBcijK', H.ab.oooo[oa, Ob, oa, ob], T.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('MNij,aBcMNK->aBcijK', H.ab.oooo[Oa, Ob, oa, ob], T.abb.vVvOOO, optimize=True)
    )
    dT.abb.vVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('MniK,aBcMnj->aBcijK', H.ab.oooo[Oa, ob, oa, Ob], T.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('mNiK,aBcmjN->aBcijK', H.ab.oooo[oa, Ob, oa, Ob], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('MNiK,aBcMjN->aBcijK', H.ab.oooo[Oa, Ob, oa, Ob], T.abb.vVvOoO, optimize=True)
    )
    dT.abb.vVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('BceF,aFeijK->aBcijK', H.bb.vvvv[Vb, vb, vb, Vb], T.abb.vVvooO, optimize=True)
            - 0.5 * np.einsum('BcEF,aFEijK->aBcijK', H.bb.vvvv[Vb, vb, Vb, Vb], T.abb.vVVooO, optimize=True)
    )
    dT.abb.vVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('aBEf,EcfijK->aBcijK', H.ab.vvvv[va, Vb, Va, vb], T.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('aBeF,eFcijK->aBcijK', H.ab.vvvv[va, Vb, va, Vb], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('aBEF,EFcijK->aBcijK', H.ab.vvvv[va, Vb, Va, Vb], T.abb.VVvooO, optimize=True)
    )
    dT.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('acef,eBfijK->aBcijK', H.ab.vvvv[va, vb, va, vb], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('acEf,EBfijK->aBcijK', H.ab.vvvv[va, vb, Va, vb], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('aceF,eBFijK->aBcijK', H.ab.vvvv[va, vb, va, Vb], T.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('acEF,EBFijK->aBcijK', H.ab.vvvv[va, vb, Va, Vb], T.abb.VVVooO, optimize=True)
    )
    dT.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('amie,eBcmjK->aBcijK', H.aa.voov[va, oa, oa, va], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('aMie,eBcMjK->aBcijK', H.aa.voov[va, Oa, oa, va], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('amiE,EBcmjK->aBcijK', H.aa.voov[va, oa, oa, Va], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('aMiE,EBcMjK->aBcijK', H.aa.voov[va, Oa, oa, Va], T.abb.VVvOoO, optimize=True)
    )
    dT.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('amie,BcemjK->aBcijK', H.ab.voov[va, ob, oa, vb], T.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('aMie,BcejMK->aBcijK', H.ab.voov[va, Ob, oa, vb], T.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('amiE,BEcmjK->aBcijK', H.ab.voov[va, ob, oa, Vb], T.bbb.VVvooO, optimize=True)
            + 1.0 * np.einsum('aMiE,BEcjMK->aBcijK', H.ab.voov[va, Ob, oa, Vb], T.bbb.VVvoOO, optimize=True)
    )
    dT.abb.vVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mBEj,EacimK->aBcijK', H.ab.ovvo[oa, Vb, Va, ob], T.aab.VvvooO, optimize=True)
            - 1.0 * np.einsum('MBEj,EaciMK->aBcijK', H.ab.ovvo[Oa, Vb, Va, ob], T.aab.VvvoOO, optimize=True)
    )
    dT.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mcej,eaBimK->aBcijK', H.ab.ovvo[oa, vb, va, ob], T.aab.vvVooO, optimize=True)
            + 1.0 * np.einsum('Mcej,eaBiMK->aBcijK', H.ab.ovvo[Oa, vb, va, ob], T.aab.vvVoOO, optimize=True)
            + 1.0 * np.einsum('mcEj,EaBimK->aBcijK', H.ab.ovvo[oa, vb, Va, ob], T.aab.VvVooO, optimize=True)
            + 1.0 * np.einsum('McEj,EaBiMK->aBcijK', H.ab.ovvo[Oa, vb, Va, ob], T.aab.VvVoOO, optimize=True)
    )
    dT.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('MBEK,EaciMj->aBcijK', H.ab.ovvo[Oa, Vb, Va, Ob], T.aab.VvvoOo, optimize=True)
    )
    dT.abb.vVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('MceK,eaBiMj->aBcijK', H.ab.ovvo[Oa, vb, va, Ob], T.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('McEK,EaBiMj->aBcijK', H.ab.ovvo[Oa, vb, Va, Ob], T.aab.VvVoOo, optimize=True)
    )
    dT.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('BmjE,aEcimK->aBcijK', H.bb.voov[Vb, ob, ob, Vb], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('BMjE,aEciMK->aBcijK', H.bb.voov[Vb, Ob, ob, Vb], T.abb.vVvoOO, optimize=True)
    )
    dT.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('cmje,aBeimK->aBcijK', H.bb.voov[vb, ob, ob, vb], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('cMje,aBeiMK->aBcijK', H.bb.voov[vb, Ob, ob, vb], T.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('cmjE,aBEimK->aBcijK', H.bb.voov[vb, ob, ob, Vb], T.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('cMjE,aBEiMK->aBcijK', H.bb.voov[vb, Ob, ob, Vb], T.abb.vVVoOO, optimize=True)
    )
    dT.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('BMKE,aEcijM->aBcijK', H.bb.voov[Vb, Ob, Ob, Vb], T.abb.vVvooO, optimize=True)
    )
    dT.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('cMKe,aBeijM->aBcijK', H.bb.voov[vb, Ob, Ob, vb], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('cMKE,aBEijM->aBcijK', H.bb.voov[vb, Ob, Ob, Vb], T.abb.vVVooO, optimize=True)
    )
    dT.abb.vVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mBiE,aEcmjK->aBcijK', H.ab.ovov[oa, Vb, oa, Vb], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('MBiE,aEcMjK->aBcijK', H.ab.ovov[Oa, Vb, oa, Vb], T.abb.vVvOoO, optimize=True)
    )
    dT.abb.vVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mcie,aBemjK->aBcijK', H.ab.ovov[oa, vb, oa, vb], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('Mcie,aBeMjK->aBcijK', H.ab.ovov[Oa, vb, oa, vb], T.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('mciE,aBEmjK->aBcijK', H.ab.ovov[oa, vb, oa, Vb], T.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('MciE,aBEMjK->aBcijK', H.ab.ovov[Oa, vb, oa, Vb], T.abb.vVVOoO, optimize=True)
    )
    dT.abb.vVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('amej,eBcimK->aBcijK', H.ab.vovo[va, ob, va, ob], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('aMej,eBciMK->aBcijK', H.ab.vovo[va, Ob, va, ob], T.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('amEj,EBcimK->aBcijK', H.ab.vovo[va, ob, Va, ob], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('aMEj,EBciMK->aBcijK', H.ab.vovo[va, Ob, Va, ob], T.abb.VVvoOO, optimize=True)
    )
    dT.abb.vVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('aMeK,eBcijM->aBcijK', H.ab.vovo[va, Ob, va, Ob], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('aMEK,EBcijM->aBcijK', H.ab.vovo[va, Ob, Va, Ob], T.abb.VVvooO, optimize=True)
    )

    return dT


def update(T, dT, H, shift, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.abb.vVvooO, dT.abb.vVvooO = cc_active_loops.update_t3c_010001(
        T.abb.vVvooO,
        dT.abb.vVvooO,
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