import numpy as np
from ccpy.utilities.active_space import get_active_slices

def update(T, dT, H, H0, shift, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dT.aab.VvvoOO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mi,AbcmJK->AbciJK', H.a.oo[oa, oa], T.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('Mi,AbcMJK->AbciJK', H.a.oo[Oa, oa], T.aab.VvvOOO, optimize=True)
    )
    dT.aab.VvvoOO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mJ,AbcmiK->AbciJK', H.a.oo[oa, Oa], T.aab.VvvooO, optimize=True)
            - 1.0 * np.einsum('MJ,AbciMK->AbciJK', H.a.oo[Oa, Oa], T.aab.VvvoOO, optimize=True)
    )
    dT.aab.VvvoOO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mK,AbciJm->AbciJK', H.b.oo[ob, Ob], T.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('MK,AbciJM->AbciJK', H.b.oo[Ob, Ob], T.aab.VvvoOO, optimize=True)
    )
    dT.aab.VvvoOO += (1.0 / 1.0) * (
            +1.0 * np.einsum('AE,EbciJK->AbciJK', H.a.vv[Va, Va], T.aab.VvvoOO, optimize=True)
    )
    dT.aab.VvvoOO += (1.0 / 1.0) * (
            +1.0 * np.einsum('be,AeciJK->AbciJK', H.a.vv[va, va], T.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('bE,AEciJK->AbciJK', H.a.vv[va, Va], T.aab.VVvoOO, optimize=True)
    )
    dT.aab.VvvoOO += (1.0 / 1.0) * (
            +1.0 * np.einsum('ce,AbeiJK->AbciJK', H.b.vv[vb, vb], T.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('cE,AbEiJK->AbciJK', H.b.vv[vb, Vb], T.aab.VvVoOO, optimize=True)
    )
    dT.aab.VvvoOO += (1.0 / 1.0) * (
            +0.5 * np.einsum('mniJ,AbcmnK->AbciJK', H.aa.oooo[oa, oa, oa, Oa], T.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('mNiJ,AbcmNK->AbciJK', H.aa.oooo[oa, Oa, oa, Oa], T.aab.VvvoOO, optimize=True)
            + 0.5 * np.einsum('MNiJ,AbcMNK->AbciJK', H.aa.oooo[Oa, Oa, oa, Oa], T.aab.VvvOOO, optimize=True)
    )
    dT.aab.VvvoOO += (1.0 / 1.0) * (
            +1.0 * np.einsum('MnJK,AbciMn->AbciJK', H.ab.oooo[Oa, ob, Oa, Ob], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('mNJK,AbcimN->AbciJK', H.ab.oooo[oa, Ob, Oa, Ob], T.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('MNJK,AbciMN->AbciJK', H.ab.oooo[Oa, Ob, Oa, Ob], T.aab.VvvoOO, optimize=True)
    )
    dT.aab.VvvoOO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mniK,AbcmJn->AbciJK', H.ab.oooo[oa, ob, oa, Ob], T.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('MniK,AbcJMn->AbciJK', H.ab.oooo[Oa, ob, oa, Ob], T.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('mNiK,AbcmJN->AbciJK', H.ab.oooo[oa, Ob, oa, Ob], T.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('MNiK,AbcJMN->AbciJK', H.ab.oooo[Oa, Ob, oa, Ob], T.aab.VvvOOO, optimize=True)
    )
    dT.aab.VvvoOO += (1.0 / 1.0) * (
            -1.0 * np.einsum('AbeF,FeciJK->AbciJK', H.aa.vvvv[Va, va, va, Va], T.aab.VvvoOO, optimize=True)
            - 0.5 * np.einsum('AbEF,FEciJK->AbciJK', H.aa.vvvv[Va, va, Va, Va], T.aab.VVvoOO, optimize=True)
    )
    dT.aab.VvvoOO += (1.0 / 1.0) * (
            +1.0 * np.einsum('bcef,AefiJK->AbciJK', H.ab.vvvv[va, vb, va, vb], T.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('bcEf,EAfiJK->AbciJK', H.ab.vvvv[va, vb, Va, vb], T.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('bceF,AeFiJK->AbciJK', H.ab.vvvv[va, vb, va, Vb], T.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('bcEF,EAFiJK->AbciJK', H.ab.vvvv[va, vb, Va, Vb], T.aab.VVVoOO, optimize=True)
    )
    dT.aab.VvvoOO += (1.0 / 1.0) * (
            +1.0 * np.einsum('AcEf,EbfiJK->AbciJK', H.ab.vvvv[Va, vb, Va, vb], T.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('AceF,ebFiJK->AbciJK', H.ab.vvvv[Va, vb, va, Vb], T.aab.vvVoOO, optimize=True)
            + 1.0 * np.einsum('AcEF,EbFiJK->AbciJK', H.ab.vvvv[Va, vb, Va, Vb], T.aab.VvVoOO, optimize=True)
    )
    dT.aab.VvvoOO += (1.0 / 1.0) * (
            +1.0 * np.einsum('AmiE,EbcmJK->AbciJK', H.aa.voov[Va, oa, oa, Va], T.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('AMiE,EbcMJK->AbciJK', H.aa.voov[Va, Oa, oa, Va], T.aab.VvvOOO, optimize=True)
    )
    dT.aab.VvvoOO += (1.0 / 1.0) * (
            +1.0 * np.einsum('bmie,AecmJK->AbciJK', H.aa.voov[va, oa, oa, va], T.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('bmiE,AEcmJK->AbciJK', H.aa.voov[va, oa, oa, Va], T.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('bMie,AecMJK->AbciJK', H.aa.voov[va, Oa, oa, va], T.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('bMiE,AEcMJK->AbciJK', H.aa.voov[va, Oa, oa, Va], T.aab.VVvOOO, optimize=True)
    )
    dT.aab.VvvoOO += (1.0 / 1.0) * (
            -1.0 * np.einsum('AmJE,EbcmiK->AbciJK', H.aa.voov[Va, oa, Oa, Va], T.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('AMJE,EbciMK->AbciJK', H.aa.voov[Va, Oa, Oa, Va], T.aab.VvvoOO, optimize=True)
    )
    dT.aab.VvvoOO += (1.0 / 1.0) * (
            -1.0 * np.einsum('bmJe,AecmiK->AbciJK', H.aa.voov[va, oa, Oa, va], T.aab.VvvooO, optimize=True)
            - 1.0 * np.einsum('bmJE,AEcmiK->AbciJK', H.aa.voov[va, oa, Oa, Va], T.aab.VVvooO, optimize=True)
            + 1.0 * np.einsum('bMJe,AeciMK->AbciJK', H.aa.voov[va, Oa, Oa, va], T.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('bMJE,AEciMK->AbciJK', H.aa.voov[va, Oa, Oa, Va], T.aab.VVvoOO, optimize=True)
    )
    dT.aab.VvvoOO += (1.0 / 1.0) * (
            +1.0 * np.einsum('AmiE,bEcJmK->AbciJK', H.ab.voov[Va, ob, oa, Vb], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('AMiE,bEcJMK->AbciJK', H.ab.voov[Va, Ob, oa, Vb], T.abb.vVvOOO, optimize=True)
    )
    dT.aab.VvvoOO += (1.0 / 1.0) * (
            +1.0 * np.einsum('bmie,AceJmK->AbciJK', H.ab.voov[va, ob, oa, vb], T.abb.VvvOoO, optimize=True)
            - 1.0 * np.einsum('bmiE,AEcJmK->AbciJK', H.ab.voov[va, ob, oa, Vb], T.abb.VVvOoO, optimize=True)
            + 1.0 * np.einsum('bMie,AceJMK->AbciJK', H.ab.voov[va, Ob, oa, vb], T.abb.VvvOOO, optimize=True)
            - 1.0 * np.einsum('bMiE,AEcJMK->AbciJK', H.ab.voov[va, Ob, oa, Vb], T.abb.VVvOOO, optimize=True)
    )
    dT.aab.VvvoOO += (1.0 / 1.0) * (
            -1.0 * np.einsum('AmJE,bEcimK->AbciJK', H.ab.voov[Va, ob, Oa, Vb], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('AMJE,bEciMK->AbciJK', H.ab.voov[Va, Ob, Oa, Vb], T.abb.vVvoOO, optimize=True)
    )
    dT.aab.VvvoOO += (1.0 / 1.0) * (
            -1.0 * np.einsum('bmJe,AceimK->AbciJK', H.ab.voov[va, ob, Oa, vb], T.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('bmJE,AEcimK->AbciJK', H.ab.voov[va, ob, Oa, Vb], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('bMJe,AceiMK->AbciJK', H.ab.voov[va, Ob, Oa, vb], T.abb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('bMJE,AEciMK->AbciJK', H.ab.voov[va, Ob, Oa, Vb], T.abb.VVvoOO, optimize=True)
    )
    dT.aab.VvvoOO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mceK,AebimJ->AbciJK', H.ab.ovvo[oa, vb, va, Ob], T.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('mcEK,EAbimJ->AbciJK', H.ab.ovvo[oa, vb, Va, Ob], T.aaa.VVvooO, optimize=True)
            - 1.0 * np.einsum('MceK,AebiJM->AbciJK', H.ab.ovvo[Oa, vb, va, Ob], T.aaa.VvvoOO, optimize=True)
            + 1.0 * np.einsum('McEK,EAbiJM->AbciJK', H.ab.ovvo[Oa, vb, Va, Ob], T.aaa.VVvoOO, optimize=True)
    )
    dT.aab.VvvoOO += (1.0 / 1.0) * (
            +1.0 * np.einsum('cmKe,AbeiJm->AbciJK', H.bb.voov[vb, ob, Ob, vb], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('cmKE,AbEiJm->AbciJK', H.bb.voov[vb, ob, Ob, Vb], T.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('cMKe,AbeiJM->AbciJK', H.bb.voov[vb, Ob, Ob, vb], T.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('cMKE,AbEiJM->AbciJK', H.bb.voov[vb, Ob, Ob, Vb], T.aab.VvVoOO, optimize=True)
    )
    dT.aab.VvvoOO += (1.0 / 1.0) * (
            -1.0 * np.einsum('AmEK,EbciJm->AbciJK', H.ab.vovo[Va, ob, Va, Ob], T.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('AMEK,EbciJM->AbciJK', H.ab.vovo[Va, Ob, Va, Ob], T.aab.VvvoOO, optimize=True)
    )
    dT.aab.VvvoOO += (1.0 / 1.0) * (
            -1.0 * np.einsum('bmeK,AeciJm->AbciJK', H.ab.vovo[va, ob, va, Ob], T.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('bmEK,AEciJm->AbciJK', H.ab.vovo[va, ob, Va, Ob], T.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('bMeK,AeciJM->AbciJK', H.ab.vovo[va, Ob, va, Ob], T.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('bMEK,AEciJM->AbciJK', H.ab.vovo[va, Ob, Va, Ob], T.aab.VVvoOO, optimize=True)
    )
    dT.aab.VvvoOO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mcie,AbemJK->AbciJK', H.ab.ovov[oa, vb, oa, vb], T.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('mciE,AbEmJK->AbciJK', H.ab.ovov[oa, vb, oa, Vb], T.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('Mcie,AbeMJK->AbciJK', H.ab.ovov[Oa, vb, oa, vb], T.aab.VvvOOO, optimize=True)
            - 1.0 * np.einsum('MciE,AbEMJK->AbciJK', H.ab.ovov[Oa, vb, oa, Vb], T.aab.VvVOOO, optimize=True)
    )
    dT.aab.VvvoOO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mcJe,AbemiK->AbciJK', H.ab.ovov[oa, vb, Oa, vb], T.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('mcJE,AbEmiK->AbciJK', H.ab.ovov[oa, vb, Oa, Vb], T.aab.VvVooO, optimize=True)
            - 1.0 * np.einsum('McJe,AbeiMK->AbciJK', H.ab.ovov[Oa, vb, Oa, vb], T.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('McJE,AbEiMK->AbciJK', H.ab.ovov[Oa, vb, Oa, Vb], T.aab.VvVoOO, optimize=True)
    )

    return T, dT