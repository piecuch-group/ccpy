import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import cc_active_loops

#@profile
def build(T, dT, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    # MM(2,3)
    dT.aab.VvVoOO = (1.0 / 1.0) * (
            +1.0 * np.einsum('bCeK,AeiJ->AbCiJK', H.ab.vvvo[va, Vb, :, Ob], T.aa[Va, :, oa, Oa], optimize=True)
    )
    dT.aab.VvVoOO += (1.0 / 1.0) * (
            -1.0 * np.einsum('ACeK,beiJ->AbCiJK', H.ab.vvvo[Va, Vb, :, Ob], T.aa[va, :, oa, Oa], optimize=True)
    )
    dT.aab.VvVoOO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mCJK,Abim->AbCiJK', H.ab.ovoo[:, Vb, Oa, Ob], T.aa[Va, va, oa, :], optimize=True)
    )
    dT.aab.VvVoOO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mCiK,AbJm->AbCiJK', H.ab.ovoo[:, Vb, oa, Ob], T.aa[Va, va, Oa, :], optimize=True)
    )
    dT.aab.VvVoOO += (1.0 / 1.0) * (
            +1.0 * np.einsum('ACie,beJK->AbCiJK', H.ab.vvov[Va, Vb, oa, :], T.ab[va, :, Oa, Ob], optimize=True)
    )
    dT.aab.VvVoOO += (1.0 / 1.0) * (
            -1.0 * np.einsum('bCie,AeJK->AbCiJK', H.ab.vvov[va, Vb, oa, :], T.ab[Va, :, Oa, Ob], optimize=True)
    )
    dT.aab.VvVoOO += (1.0 / 1.0) * (
            -1.0 * np.einsum('ACJe,beiK->AbCiJK', H.ab.vvov[Va, Vb, Oa, :], T.ab[va, :, oa, Ob], optimize=True)
    )
    dT.aab.VvVoOO += (1.0 / 1.0) * (
            +1.0 * np.einsum('bCJe,AeiK->AbCiJK', H.ab.vvov[va, Vb, Oa, :], T.ab[Va, :, oa, Ob], optimize=True)
    )
    dT.aab.VvVoOO += (1.0 / 1.0) * (
            -1.0 * np.einsum('AmiK,bCJm->AbCiJK', H.ab.vooo[Va, :, oa, Ob], T.ab[va, Vb, Oa, :], optimize=True)
    )
    dT.aab.VvVoOO += (1.0 / 1.0) * (
            +1.0 * np.einsum('bmiK,ACJm->AbCiJK', H.ab.vooo[va, :, oa, Ob], T.ab[Va, Vb, Oa, :], optimize=True)
    )
    dT.aab.VvVoOO += (1.0 / 1.0) * (
            +1.0 * np.einsum('AmJK,bCim->AbCiJK', H.ab.vooo[Va, :, Oa, Ob], T.ab[va, Vb, oa, :], optimize=True)
    )
    dT.aab.VvVoOO += (1.0 / 1.0) * (
            -1.0 * np.einsum('bmJK,ACim->AbCiJK', H.ab.vooo[va, :, Oa, Ob], T.ab[Va, Vb, oa, :], optimize=True)
    )
    dT.aab.VvVoOO += (1.0 / 1.0) * (
            +1.0 * np.einsum('Abie,eCJK->AbCiJK', H.aa.vvov[Va, va, oa, :], T.ab[:, Vb, Oa, Ob], optimize=True)
    )
    dT.aab.VvVoOO += (1.0 / 1.0) * (
            -1.0 * np.einsum('AbJe,eCiK->AbCiJK', H.aa.vvov[Va, va, Oa, :], T.ab[:, Vb, oa, Ob], optimize=True)
    )
    dT.aab.VvVoOO += (1.0 / 1.0) * (
            -1.0 * np.einsum('AmiJ,bCmK->AbCiJK', H.aa.vooo[Va, :, oa, Oa], T.ab[va, Vb, :, Ob], optimize=True)
    )
    dT.aab.VvVoOO += (1.0 / 1.0) * (
            +1.0 * np.einsum('bmiJ,ACmK->AbCiJK', H.aa.vooo[va, :, oa, Oa], T.ab[Va, Vb, :, Ob], optimize=True)
    )
    # (H(2) * T3)_C
    dT.aab.VvVoOO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mi,AbCmJK->AbCiJK', H.a.oo[oa, oa], T.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('Mi,AbCMJK->AbCiJK', H.a.oo[Oa, oa], T.aab.VvVOOO, optimize=True)
    )
    dT.aab.VvVoOO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mJ,AbCmiK->AbCiJK', H.a.oo[oa, Oa], T.aab.VvVooO, optimize=True)
            - 1.0 * np.einsum('MJ,AbCiMK->AbCiJK', H.a.oo[Oa, Oa], T.aab.VvVoOO, optimize=True)
    )
    dT.aab.VvVoOO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mK,AbCiJm->AbCiJK', H.b.oo[ob, Ob], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('MK,AbCiJM->AbCiJK', H.b.oo[Ob, Ob], T.aab.VvVoOO, optimize=True)
    )
    dT.aab.VvVoOO += (1.0 / 1.0) * (
            -1.0 * np.einsum('Ae,beCiJK->AbCiJK', H.a.vv[Va, va], T.aab.vvVoOO, optimize=True)
            + 1.0 * np.einsum('AE,EbCiJK->AbCiJK', H.a.vv[Va, Va], T.aab.VvVoOO, optimize=True)
    )
    dT.aab.VvVoOO += (1.0 / 1.0) * (
            +1.0 * np.einsum('be,AeCiJK->AbCiJK', H.a.vv[va, va], T.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('bE,AECiJK->AbCiJK', H.a.vv[va, Va], T.aab.VVVoOO, optimize=True)
    )
    dT.aab.VvVoOO += (1.0 / 1.0) * (
            +1.0 * np.einsum('Ce,AbeiJK->AbCiJK', H.b.vv[Vb, vb], T.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('CE,AbEiJK->AbCiJK', H.b.vv[Vb, Vb], T.aab.VvVoOO, optimize=True)
    )
    dT.aab.VvVoOO += (1.0 / 1.0) * (
            +0.5 * np.einsum('mniJ,AbCmnK->AbCiJK', H.aa.oooo[oa, oa, oa, Oa], T.aab.VvVooO, optimize=True)
            - 1.0 * np.einsum('MniJ,AbCnMK->AbCiJK', H.aa.oooo[Oa, oa, oa, Oa], T.aab.VvVoOO, optimize=True)
            + 0.5 * np.einsum('MNiJ,AbCMNK->AbCiJK', H.aa.oooo[Oa, Oa, oa, Oa], T.aab.VvVOOO, optimize=True)
    )
    dT.aab.VvVoOO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mNJK,AbCimN->AbCiJK', H.ab.oooo[oa, Ob, Oa, Ob], T.aab.VvVooO, optimize=True)
            + 1.0 * np.einsum('MnJK,AbCiMn->AbCiJK', H.ab.oooo[Oa, ob, Oa, Ob], T.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('MNJK,AbCiMN->AbCiJK', H.ab.oooo[Oa, Ob, Oa, Ob], T.aab.VvVoOO, optimize=True)
    )
    dT.aab.VvVoOO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mniK,AbCmJn->AbCiJK', H.ab.oooo[oa, ob, oa, Ob], T.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('mNiK,AbCmJN->AbCiJK', H.ab.oooo[oa, Ob, oa, Ob], T.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('MniK,AbCJMn->AbCiJK', H.ab.oooo[Oa, ob, oa, Ob], T.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('MNiK,AbCJMN->AbCiJK', H.ab.oooo[Oa, Ob, oa, Ob], T.aab.VvVOOO, optimize=True)
    )
    dT.aab.VvVoOO += (1.0 / 1.0) * (
            -0.5 * np.einsum('Abef,feCiJK->AbCiJK', H.aa.vvvv[Va, va, va, va], T.aab.vvVoOO, optimize=True)
            + 1.0 * np.einsum('AbEf,EfCiJK->AbCiJK', H.aa.vvvv[Va, va, Va, va], T.aab.VvVoOO, optimize=True)
            - 0.5 * np.einsum('AbEF,FECiJK->AbCiJK', H.aa.vvvv[Va, va, Va, Va], T.aab.VVVoOO, optimize=True)
    )
    dT.aab.VvVoOO += (1.0 / 1.0) * (
            +1.0 * np.einsum('bCef,AefiJK->AbCiJK', H.ab.vvvv[va, Vb, va, vb], T.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('bCeF,AeFiJK->AbCiJK', H.ab.vvvv[va, Vb, va, Vb], T.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('bCEf,EAfiJK->AbCiJK', H.ab.vvvv[va, Vb, Va, vb], T.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('bCEF,EAFiJK->AbCiJK', H.ab.vvvv[va, Vb, Va, Vb], T.aab.VVVoOO, optimize=True)
    )
    dT.aab.VvVoOO += (1.0 / 1.0) * (
            +1.0 * np.einsum('ACeF,ebFiJK->AbCiJK', H.ab.vvvv[Va, Vb, va, Vb], T.aab.vvVoOO, optimize=True)
            + 1.0 * np.einsum('ACEf,EbfiJK->AbCiJK', H.ab.vvvv[Va, Vb, Va, vb], T.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('ACEF,EbFiJK->AbCiJK', H.ab.vvvv[Va, Vb, Va, Vb], T.aab.VvVoOO, optimize=True)
    )
    dT.aab.VvVoOO += (1.0 / 1.0) * (
            -1.0 * np.einsum('Amie,beCmJK->AbCiJK', H.aa.voov[Va, oa, oa, va], T.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('AMie,beCMJK->AbCiJK', H.aa.voov[Va, Oa, oa, va], T.aab.vvVOOO, optimize=True)
            + 1.0 * np.einsum('AmiE,EbCmJK->AbCiJK', H.aa.voov[Va, oa, oa, Va], T.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('AMiE,EbCMJK->AbCiJK', H.aa.voov[Va, Oa, oa, Va], T.aab.VvVOOO, optimize=True)
    )
    dT.aab.VvVoOO += (1.0 / 1.0) * (
            +1.0 * np.einsum('bmie,AeCmJK->AbCiJK', H.aa.voov[va, oa, oa, va], T.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('bMie,AeCMJK->AbCiJK', H.aa.voov[va, Oa, oa, va], T.aab.VvVOOO, optimize=True)
            + 1.0 * np.einsum('bmiE,AECmJK->AbCiJK', H.aa.voov[va, oa, oa, Va], T.aab.VVVoOO, optimize=True)
            + 1.0 * np.einsum('bMiE,AECMJK->AbCiJK', H.aa.voov[va, Oa, oa, Va], T.aab.VVVOOO, optimize=True)
    )
    dT.aab.VvVoOO += (1.0 / 1.0) * (
            +1.0 * np.einsum('AmJe,beCmiK->AbCiJK', H.aa.voov[Va, oa, Oa, va], T.aab.vvVooO, optimize=True)
            - 1.0 * np.einsum('AMJe,beCiMK->AbCiJK', H.aa.voov[Va, Oa, Oa, va], T.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('AmJE,EbCmiK->AbCiJK', H.aa.voov[Va, oa, Oa, Va], T.aab.VvVooO, optimize=True)
            + 1.0 * np.einsum('AMJE,EbCiMK->AbCiJK', H.aa.voov[Va, Oa, Oa, Va], T.aab.VvVoOO, optimize=True)
    )
    dT.aab.VvVoOO += (1.0 / 1.0) * (
            -1.0 * np.einsum('bmJe,AeCmiK->AbCiJK', H.aa.voov[va, oa, Oa, va], T.aab.VvVooO, optimize=True)
            + 1.0 * np.einsum('bMJe,AeCiMK->AbCiJK', H.aa.voov[va, Oa, Oa, va], T.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('bmJE,AECmiK->AbCiJK', H.aa.voov[va, oa, Oa, Va], T.aab.VVVooO, optimize=True)
            + 1.0 * np.einsum('bMJE,AECiMK->AbCiJK', H.aa.voov[va, Oa, Oa, Va], T.aab.VVVoOO, optimize=True)
    )
    dT.aab.VvVoOO += (1.0 / 1.0) * (
            -1.0 * np.einsum('Amie,bCeJmK->AbCiJK', H.ab.voov[Va, ob, oa, vb], T.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('AMie,bCeJMK->AbCiJK', H.ab.voov[Va, Ob, oa, vb], T.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('AmiE,bCEJmK->AbCiJK', H.ab.voov[Va, ob, oa, Vb], T.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('AMiE,bCEJMK->AbCiJK', H.ab.voov[Va, Ob, oa, Vb], T.abb.vVVOOO, optimize=True)
    )
    dT.aab.VvVoOO += (1.0 / 1.0) * (
            +1.0 * np.einsum('bmie,ACeJmK->AbCiJK', H.ab.voov[va, ob, oa, vb], T.abb.VVvOoO, optimize=True)
            + 1.0 * np.einsum('bMie,ACeJMK->AbCiJK', H.ab.voov[va, Ob, oa, vb], T.abb.VVvOOO, optimize=True)
            + 1.0 * np.einsum('bmiE,ACEJmK->AbCiJK', H.ab.voov[va, ob, oa, Vb], T.abb.VVVOoO, optimize=True)
            + 1.0 * np.einsum('bMiE,ACEJMK->AbCiJK', H.ab.voov[va, Ob, oa, Vb], T.abb.VVVOOO, optimize=True)
    )
    dT.aab.VvVoOO += (1.0 / 1.0) * (
            +1.0 * np.einsum('AmJe,bCeimK->AbCiJK', H.ab.voov[Va, ob, Oa, vb], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('AMJe,bCeiMK->AbCiJK', H.ab.voov[Va, Ob, Oa, vb], T.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('AmJE,bCEimK->AbCiJK', H.ab.voov[Va, ob, Oa, Vb], T.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('AMJE,bCEiMK->AbCiJK', H.ab.voov[Va, Ob, Oa, Vb], T.abb.vVVoOO, optimize=True)
    )
    dT.aab.VvVoOO += (1.0 / 1.0) * (
            -1.0 * np.einsum('bmJe,ACeimK->AbCiJK', H.ab.voov[va, ob, Oa, vb], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('bMJe,ACeiMK->AbCiJK', H.ab.voov[va, Ob, Oa, vb], T.abb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('bmJE,ACEimK->AbCiJK', H.ab.voov[va, ob, Oa, Vb], T.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('bMJE,ACEiMK->AbCiJK', H.ab.voov[va, Ob, Oa, Vb], T.abb.VVVoOO, optimize=True)
    )
    dT.aab.VvVoOO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mCeK,AebimJ->AbCiJK', H.ab.ovvo[oa, Vb, va, Ob], T.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('MCeK,AebiJM->AbCiJK', H.ab.ovvo[Oa, Vb, va, Ob], T.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('mCEK,EAbimJ->AbCiJK', H.ab.ovvo[oa, Vb, Va, Ob], T.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('MCEK,EAbiJM->AbCiJK', H.ab.ovvo[Oa, Vb, Va, Ob], T.aaa.VVvoOO, optimize=True)
    )
    dT.aab.VvVoOO += (1.0 / 1.0) * (
            +1.0 * np.einsum('CmKe,AbeiJm->AbCiJK', H.bb.voov[Vb, ob, Ob, vb], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('CMKe,AbeiJM->AbCiJK', H.bb.voov[Vb, Ob, Ob, vb], T.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('CmKE,AbEiJm->AbCiJK', H.bb.voov[Vb, ob, Ob, Vb], T.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('CMKE,AbEiJM->AbCiJK', H.bb.voov[Vb, Ob, Ob, Vb], T.aab.VvVoOO, optimize=True)
    )
    dT.aab.VvVoOO += (1.0 / 1.0) * (
            +1.0 * np.einsum('AmeK,beCiJm->AbCiJK', H.ab.vovo[Va, ob, va, Ob], T.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('AMeK,beCiJM->AbCiJK', H.ab.vovo[Va, Ob, va, Ob], T.aab.vvVoOO, optimize=True)
            - 1.0 * np.einsum('AmEK,EbCiJm->AbCiJK', H.ab.vovo[Va, ob, Va, Ob], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('AMEK,EbCiJM->AbCiJK', H.ab.vovo[Va, Ob, Va, Ob], T.aab.VvVoOO, optimize=True)
    )
    dT.aab.VvVoOO += (1.0 / 1.0) * (
            -1.0 * np.einsum('bmeK,AeCiJm->AbCiJK', H.ab.vovo[va, ob, va, Ob], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('bMeK,AeCiJM->AbCiJK', H.ab.vovo[va, Ob, va, Ob], T.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('bmEK,AECiJm->AbCiJK', H.ab.vovo[va, ob, Va, Ob], T.aab.VVVoOo, optimize=True)
            - 1.0 * np.einsum('bMEK,AECiJM->AbCiJK', H.ab.vovo[va, Ob, Va, Ob], T.aab.VVVoOO, optimize=True)
    )
    dT.aab.VvVoOO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mCie,AbemJK->AbCiJK', H.ab.ovov[oa, Vb, oa, vb], T.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('MCie,AbeMJK->AbCiJK', H.ab.ovov[Oa, Vb, oa, vb], T.aab.VvvOOO, optimize=True)
            - 1.0 * np.einsum('mCiE,AbEmJK->AbCiJK', H.ab.ovov[oa, Vb, oa, Vb], T.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('MCiE,AbEMJK->AbCiJK', H.ab.ovov[Oa, Vb, oa, Vb], T.aab.VvVOOO, optimize=True)
    )
    dT.aab.VvVoOO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mCJe,AbemiK->AbCiJK', H.ab.ovov[oa, Vb, Oa, vb], T.aab.VvvooO, optimize=True)
            - 1.0 * np.einsum('MCJe,AbeiMK->AbCiJK', H.ab.ovov[Oa, Vb, Oa, vb], T.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('mCJE,AbEmiK->AbCiJK', H.ab.ovov[oa, Vb, Oa, Vb], T.aab.VvVooO, optimize=True)
            - 1.0 * np.einsum('MCJE,AbEiMK->AbCiJK', H.ab.ovov[Oa, Vb, Oa, Vb], T.aab.VvVoOO, optimize=True)
    )

    return dT

def update(T, dT, H, shift, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.aab.VvVoOO, dT.aab.VvVoOO = cc_active_loops.update_t3b_101011(
        T.aab.VvVoOO,
        dT.aab.VvVoOO,
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