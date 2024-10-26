import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import cc_active_loops

#@profile
def build(T, dT, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    # MM(2,3)
    dT.aab.VvVoOo = (1.0 / 1.0) * (
            +1.0 * np.einsum('bCek,AeiJ->AbCiJk', H.ab.vvvo[va, Vb, :, ob], T.aa[Va, :, oa, Oa], optimize=True)
    )
    dT.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('ACek,beiJ->AbCiJk', H.ab.vvvo[Va, Vb, :, ob], T.aa[va, :, oa, Oa], optimize=True)
    )
    dT.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('mCJk,Abim->AbCiJk', H.ab.ovoo[:, Vb, Oa, ob], T.aa[Va, va, oa, :], optimize=True)
    )
    dT.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('mCik,AbJm->AbCiJk', H.ab.ovoo[:, Vb, oa, ob], T.aa[Va, va, Oa, :], optimize=True)
    )
    dT.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('ACie,beJk->AbCiJk', H.ab.vvov[Va, Vb, oa, :], T.ab[va, :, Oa, ob], optimize=True)
    )
    dT.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('bCie,AeJk->AbCiJk', H.ab.vvov[va, Vb, oa, :], T.ab[Va, :, Oa, ob], optimize=True)
    )
    dT.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('ACJe,beik->AbCiJk', H.ab.vvov[Va, Vb, Oa, :], T.ab[va, :, oa, ob], optimize=True)
    )
    dT.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('bCJe,Aeik->AbCiJk', H.ab.vvov[va, Vb, Oa, :], T.ab[Va, :, oa, ob], optimize=True)
    )
    dT.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('Amik,bCJm->AbCiJk', H.ab.vooo[Va, :, oa, ob], T.ab[va, Vb, Oa, :], optimize=True)
    )
    dT.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('bmik,ACJm->AbCiJk', H.ab.vooo[va, :, oa, ob], T.ab[Va, Vb, Oa, :], optimize=True)
    )
    dT.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('AmJk,bCim->AbCiJk', H.ab.vooo[Va, :, Oa, ob], T.ab[va, Vb, oa, :], optimize=True)
    )
    dT.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('bmJk,ACim->AbCiJk', H.ab.vooo[va, :, Oa, ob], T.ab[Va, Vb, oa, :], optimize=True)
    )
    dT.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('Abie,eCJk->AbCiJk', H.aa.vvov[Va, va, oa, :], T.ab[:, Vb, Oa, ob], optimize=True)
    )
    dT.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('AbJe,eCik->AbCiJk', H.aa.vvov[Va, va, Oa, :], T.ab[:, Vb, oa, ob], optimize=True)
    )
    dT.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('AmiJ,bCmk->AbCiJk', H.aa.vooo[Va, :, oa, Oa], T.ab[va, Vb, :, ob], optimize=True)
    )
    dT.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('bmiJ,ACmk->AbCiJk', H.aa.vooo[va, :, oa, Oa], T.ab[Va, Vb, :, ob], optimize=True)
    )
    # (H(2) * T3)_C
    dT.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('mi,AbCmJk->AbCiJk', H.a.oo[oa, oa], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('Mi,AbCMJk->AbCiJk', H.a.oo[Oa, oa], T.aab.VvVOOo, optimize=True)
    )
    dT.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('MJ,AbCiMk->AbCiJk', H.a.oo[Oa, Oa], T.aab.VvVoOo, optimize=True)
    )
    dT.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('mk,AbCiJm->AbCiJk', H.b.oo[ob, ob], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('Mk,AbCiJM->AbCiJk', H.b.oo[Ob, ob], T.aab.VvVoOO, optimize=True)
    )
    dT.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('Ae,beCiJk->AbCiJk', H.a.vv[Va, va], T.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('AE,EbCiJk->AbCiJk', H.a.vv[Va, Va], T.aab.VvVoOo, optimize=True)
    )
    dT.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('be,AeCiJk->AbCiJk', H.a.vv[va, va], T.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('bE,AECiJk->AbCiJk', H.a.vv[va, Va], T.aab.VVVoOo, optimize=True)
    )
    dT.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('Ce,AbeiJk->AbCiJk', H.b.vv[Vb, vb], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('CE,AbEiJk->AbCiJk', H.b.vv[Vb, Vb], T.aab.VvVoOo, optimize=True)
    )
    dT.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('mNiJ,AbCmNk->AbCiJk', H.aa.oooo[oa, Oa, oa, Oa], T.aab.VvVoOo, optimize=True)
            + 0.5 * np.einsum('MNiJ,AbCMNk->AbCiJk', H.aa.oooo[Oa, Oa, oa, Oa], T.aab.VvVOOo, optimize=True)
    )
    dT.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('MnJk,AbCiMn->AbCiJk', H.ab.oooo[Oa, ob, Oa, ob], T.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('mNJk,AbCimN->AbCiJk', H.ab.oooo[oa, Ob, Oa, ob], T.aab.VvVooO, optimize=True)
            + 1.0 * np.einsum('MNJk,AbCiMN->AbCiJk', H.ab.oooo[Oa, Ob, Oa, ob], T.aab.VvVoOO, optimize=True)
    )
    dT.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('mnik,AbCmJn->AbCiJk', H.ab.oooo[oa, ob, oa, ob], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('Mnik,AbCJMn->AbCiJk', H.ab.oooo[Oa, ob, oa, ob], T.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('mNik,AbCmJN->AbCiJk', H.ab.oooo[oa, Ob, oa, ob], T.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('MNik,AbCJMN->AbCiJk', H.ab.oooo[Oa, Ob, oa, ob], T.aab.VvVOOO, optimize=True)
    )
    dT.aab.VvVoOo += (1.0 / 1.0) * (
            -0.5 * np.einsum('Abef,feCiJk->AbCiJk', H.aa.vvvv[Va, va, va, va], T.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('AbEf,EfCiJk->AbCiJk', H.aa.vvvv[Va, va, Va, va], T.aab.VvVoOo, optimize=True)
            - 0.5 * np.einsum('AbEF,FECiJk->AbCiJk', H.aa.vvvv[Va, va, Va, Va], T.aab.VVVoOo, optimize=True)
    )
    dT.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('bCef,AefiJk->AbCiJk', H.ab.vvvv[va, Vb, va, vb], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('bCeF,AeFiJk->AbCiJk', H.ab.vvvv[va, Vb, va, Vb], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('bCEf,EAfiJk->AbCiJk', H.ab.vvvv[va, Vb, Va, vb], T.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('bCEF,EAFiJk->AbCiJk', H.ab.vvvv[va, Vb, Va, Vb], T.aab.VVVoOo, optimize=True)
    )
    dT.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('ACeF,ebFiJk->AbCiJk', H.ab.vvvv[Va, Vb, va, Vb], T.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('ACEf,EbfiJk->AbCiJk', H.ab.vvvv[Va, Vb, Va, vb], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('ACEF,EbFiJk->AbCiJk', H.ab.vvvv[Va, Vb, Va, Vb], T.aab.VvVoOo, optimize=True)
    )
    dT.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('Amie,beCmJk->AbCiJk', H.aa.voov[Va, oa, oa, va], T.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('AMie,beCMJk->AbCiJk', H.aa.voov[Va, Oa, oa, va], T.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('AmiE,EbCmJk->AbCiJk', H.aa.voov[Va, oa, oa, Va], T.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('AMiE,EbCMJk->AbCiJk', H.aa.voov[Va, Oa, oa, Va], T.aab.VvVOOo, optimize=True)
    )
    dT.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('bmie,AeCmJk->AbCiJk', H.aa.voov[va, oa, oa, va], T.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('bMie,AeCMJk->AbCiJk', H.aa.voov[va, Oa, oa, va], T.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('bmiE,AECmJk->AbCiJk', H.aa.voov[va, oa, oa, Va], T.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('bMiE,AECMJk->AbCiJk', H.aa.voov[va, Oa, oa, Va], T.aab.VVVOOo, optimize=True)
    )
    dT.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('AMJe,beCiMk->AbCiJk', H.aa.voov[Va, Oa, Oa, va], T.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('AMJE,EbCiMk->AbCiJk', H.aa.voov[Va, Oa, Oa, Va], T.aab.VvVoOo, optimize=True)
    )
    dT.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('bMJe,AeCiMk->AbCiJk', H.aa.voov[va, Oa, Oa, va], T.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('bMJE,AECiMk->AbCiJk', H.aa.voov[va, Oa, Oa, Va], T.aab.VVVoOo, optimize=True)
    )
    dT.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('Amie,bCeJmk->AbCiJk', H.ab.voov[Va, ob, oa, vb], T.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('AMie,bCeJkM->AbCiJk', H.ab.voov[Va, Ob, oa, vb], T.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('AmiE,bCEJmk->AbCiJk', H.ab.voov[Va, ob, oa, Vb], T.abb.vVVOoo, optimize=True)
            + 1.0 * np.einsum('AMiE,bCEJkM->AbCiJk', H.ab.voov[Va, Ob, oa, Vb], T.abb.vVVOoO, optimize=True)
    )
    dT.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('bmie,ACeJmk->AbCiJk', H.ab.voov[va, ob, oa, vb], T.abb.VVvOoo, optimize=True)
            - 1.0 * np.einsum('bMie,ACeJkM->AbCiJk', H.ab.voov[va, Ob, oa, vb], T.abb.VVvOoO, optimize=True)
            + 1.0 * np.einsum('bmiE,ACEJmk->AbCiJk', H.ab.voov[va, ob, oa, Vb], T.abb.VVVOoo, optimize=True)
            - 1.0 * np.einsum('bMiE,ACEJkM->AbCiJk', H.ab.voov[va, Ob, oa, Vb], T.abb.VVVOoO, optimize=True)
    )
    dT.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('AMJe,bCeikM->AbCiJk', H.ab.voov[Va, Ob, Oa, vb], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('AMJE,bCEikM->AbCiJk', H.ab.voov[Va, Ob, Oa, Vb], T.abb.vVVooO, optimize=True)
    )
    dT.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('bMJe,ACeikM->AbCiJk', H.ab.voov[va, Ob, Oa, vb], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('bMJE,ACEikM->AbCiJk', H.ab.voov[va, Ob, Oa, Vb], T.abb.VVVooO, optimize=True)
    )
    dT.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('mCek,AebimJ->AbCiJk', H.ab.ovvo[oa, Vb, va, ob], T.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('MCek,AebiJM->AbCiJk', H.ab.ovvo[Oa, Vb, va, ob], T.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('mCEk,EAbimJ->AbCiJk', H.ab.ovvo[oa, Vb, Va, ob], T.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('MCEk,EAbiJM->AbCiJk', H.ab.ovvo[Oa, Vb, Va, ob], T.aaa.VVvoOO, optimize=True)
    )
    dT.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('Cmke,AbeiJm->AbCiJk', H.bb.voov[Vb, ob, ob, vb], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('CMke,AbeiJM->AbCiJk', H.bb.voov[Vb, Ob, ob, vb], T.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('CmkE,AbEiJm->AbCiJk', H.bb.voov[Vb, ob, ob, Vb], T.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('CMkE,AbEiJM->AbCiJk', H.bb.voov[Vb, Ob, ob, Vb], T.aab.VvVoOO, optimize=True)
    )
    dT.aab.VvVoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('Amek,beCiJm->AbCiJk', H.ab.vovo[Va, ob, va, ob], T.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('AMek,beCiJM->AbCiJk', H.ab.vovo[Va, Ob, va, ob], T.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('AmEk,EbCiJm->AbCiJk', H.ab.vovo[Va, ob, Va, ob], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('AMEk,EbCiJM->AbCiJk', H.ab.vovo[Va, Ob, Va, ob], T.aab.VvVoOO, optimize=True)
    )
    dT.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('bmek,AeCiJm->AbCiJk', H.ab.vovo[va, ob, va, ob], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('bMek,AeCiJM->AbCiJk', H.ab.vovo[va, Ob, va, ob], T.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('bmEk,AECiJm->AbCiJk', H.ab.vovo[va, ob, Va, ob], T.aab.VVVoOo, optimize=True)
            - 1.0 * np.einsum('bMEk,AECiJM->AbCiJk', H.ab.vovo[va, Ob, Va, ob], T.aab.VVVoOO, optimize=True)
    )
    dT.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('mCie,AbemJk->AbCiJk', H.ab.ovov[oa, Vb, oa, vb], T.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('MCie,AbeMJk->AbCiJk', H.ab.ovov[Oa, Vb, oa, vb], T.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('mCiE,AbEmJk->AbCiJk', H.ab.ovov[oa, Vb, oa, Vb], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('MCiE,AbEMJk->AbCiJk', H.ab.ovov[Oa, Vb, oa, Vb], T.aab.VvVOOo, optimize=True)
    )
    dT.aab.VvVoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('MCJe,AbeiMk->AbCiJk', H.ab.ovov[Oa, Vb, Oa, vb], T.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('MCJE,AbEiMk->AbCiJk', H.ab.ovov[Oa, Vb, Oa, Vb], T.aab.VvVoOo, optimize=True)
    )

    return dT

def update(T, dT, H, shift, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.aab.VvVoOo, dT.aab.VvVoOo = cc_active_loops.update_t3b_101010(
        T.aab.VvVoOo,
        dT.aab.VvVoOo,
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