import numpy as np
from ccpy.utilities.active_space import get_active_slices

def update(T, dT, H, H0, shift, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dT.aab.VvvoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('mi,AbcmJk->AbciJk', H.a.oo[oa, oa], T.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('Mi,AbcMJk->AbciJk', H.a.oo[Oa, oa], T.aab.VvvOOo, optimize=True)
    )
    dT.aab.VvvoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('MJ,AbciMk->AbciJk', H.a.oo[Oa, Oa], T.aab.VvvoOo, optimize=True)
    )
    dT.aab.VvvoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('mk,AbciJm->AbciJk', H.b.oo[ob, ob], T.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('Mk,AbciJM->AbciJk', H.b.oo[Ob, ob], T.aab.VvvoOO, optimize=True)
    )
    dT.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('AE,EbciJk->AbciJk', H.a.vv[Va, Va], T.aab.VvvoOo, optimize=True)
    )
    dT.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('be,AeciJk->AbciJk', H.a.vv[va, va], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('bE,AEciJk->AbciJk', H.a.vv[va, Va], T.aab.VVvoOo, optimize=True)
    )
    dT.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('ce,AbeiJk->AbciJk', H.b.vv[vb, vb], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('cE,AbEiJk->AbciJk', H.b.vv[vb, Vb], T.aab.VvVoOo, optimize=True)
    )
    dT.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('mNiJ,AbcmNk->AbciJk', H.aa.oooo[oa, Oa, oa, Oa], T.aab.VvvoOo, optimize=True)
            + 0.5 * np.einsum('MNiJ,AbcMNk->AbciJk', H.aa.oooo[Oa, Oa, oa, Oa], T.aab.VvvOOo, optimize=True)
    )
    dT.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('MnJk,AbciMn->AbciJk', H.ab.oooo[Oa, ob, Oa, ob], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('mNJk,AbcimN->AbciJk', H.ab.oooo[oa, Ob, Oa, ob], T.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('MNJk,AbciMN->AbciJk', H.ab.oooo[Oa, Ob, Oa, ob], T.aab.VvvoOO, optimize=True)
    )
    dT.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('mnik,AbcmJn->AbciJk', H.ab.oooo[oa, ob, oa, ob], T.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('Mnik,AbcJMn->AbciJk', H.ab.oooo[Oa, ob, oa, ob], T.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('mNik,AbcmJN->AbciJk', H.ab.oooo[oa, Ob, oa, ob], T.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('MNik,AbcJMN->AbciJk', H.ab.oooo[Oa, Ob, oa, ob], T.aab.VvvOOO, optimize=True)
    )
    dT.aab.VvvoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('AbeF,FeciJk->AbciJk', H.aa.vvvv[Va, va, va, Va], T.aab.VvvoOo, optimize=True)
            - 0.5 * np.einsum('AbEF,FEciJk->AbciJk', H.aa.vvvv[Va, va, Va, Va], T.aab.VVvoOo, optimize=True)
    )
    dT.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('bcef,AefiJk->AbciJk', H.ab.vvvv[va, vb, va, vb], T.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('bcEf,EAfiJk->AbciJk', H.ab.vvvv[va, vb, Va, vb], T.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('bceF,AeFiJk->AbciJk', H.ab.vvvv[va, vb, va, Vb], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('bcEF,EAFiJk->AbciJk', H.ab.vvvv[va, vb, Va, Vb], T.aab.VVVoOo, optimize=True)
    )
    dT.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('AcEf,EbfiJk->AbciJk', H.ab.vvvv[Va, vb, Va, vb], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('AceF,ebFiJk->AbciJk', H.ab.vvvv[Va, vb, va, Vb], T.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('AcEF,EbFiJk->AbciJk', H.ab.vvvv[Va, vb, Va, Vb], T.aab.VvVoOo, optimize=True)
    )
    dT.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('AmiE,EbcmJk->AbciJk', H.aa.voov[Va, oa, oa, Va], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('AMiE,EbcMJk->AbciJk', H.aa.voov[Va, Oa, oa, Va], T.aab.VvvOOo, optimize=True)
    )
    dT.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('bmie,AecmJk->AbciJk', H.aa.voov[va, oa, oa, va], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('bMie,AecMJk->AbciJk', H.aa.voov[va, Oa, oa, va], T.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('bmiE,AEcmJk->AbciJk', H.aa.voov[va, oa, oa, Va], T.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('bMiE,AEcMJk->AbciJk', H.aa.voov[va, Oa, oa, Va], T.aab.VVvOOo, optimize=True)
    )
    dT.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('AMJE,EbciMk->AbciJk', H.aa.voov[Va, Oa, Oa, Va], T.aab.VvvoOo, optimize=True)
    )
    dT.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('bMJe,AeciMk->AbciJk', H.aa.voov[va, Oa, Oa, va], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('bMJE,AEciMk->AbciJk', H.aa.voov[va, Oa, Oa, Va], T.aab.VVvoOo, optimize=True)
    )
    dT.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('AmiE,bEcJmk->AbciJk', H.ab.voov[Va, ob, oa, Vb], T.abb.vVvOoo, optimize=True)
            - 1.0 * np.einsum('AMiE,bEcJkM->AbciJk', H.ab.voov[Va, Ob, oa, Vb], T.abb.vVvOoO, optimize=True)
    )
    dT.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('bmie,AceJmk->AbciJk', H.ab.voov[va, ob, oa, vb], T.abb.VvvOoo, optimize=True)
            - 1.0 * np.einsum('bMie,AceJkM->AbciJk', H.ab.voov[va, Ob, oa, vb], T.abb.VvvOoO, optimize=True)
            - 1.0 * np.einsum('bmiE,AEcJmk->AbciJk', H.ab.voov[va, ob, oa, Vb], T.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('bMiE,AEcJkM->AbciJk', H.ab.voov[va, Ob, oa, Vb], T.abb.VVvOoO, optimize=True)
    )
    dT.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('AMJE,bEcikM->AbciJk', H.ab.voov[Va, Ob, Oa, Vb], T.abb.vVvooO, optimize=True)
    )
    dT.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('bMJe,AceikM->AbciJk', H.ab.voov[va, Ob, Oa, vb], T.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('bMJE,AEcikM->AbciJk', H.ab.voov[va, Ob, Oa, Vb], T.abb.VVvooO, optimize=True)
    )
    dT.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('mcek,AebimJ->AbciJk', H.ab.ovvo[oa, vb, va, ob], T.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('Mcek,AebiJM->AbciJk', H.ab.ovvo[Oa, vb, va, ob], T.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('mcEk,EAbimJ->AbciJk', H.ab.ovvo[oa, vb, Va, ob], T.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('McEk,EAbiJM->AbciJk', H.ab.ovvo[Oa, vb, Va, ob], T.aaa.VVvoOO, optimize=True)
    )
    dT.aab.VvvoOo += (1.0 / 1.0) * (
            +1.0 * np.einsum('cmke,AbeiJm->AbciJk', H.bb.voov[vb, ob, ob, vb], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('cMke,AbeiJM->AbciJk', H.bb.voov[vb, Ob, ob, vb], T.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('cmkE,AbEiJm->AbciJk', H.bb.voov[vb, ob, ob, Vb], T.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('cMkE,AbEiJM->AbciJk', H.bb.voov[vb, Ob, ob, Vb], T.aab.VvVoOO, optimize=True)
    )
    dT.aab.VvvoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('AmEk,EbciJm->AbciJk', H.ab.vovo[Va, ob, Va, ob], T.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('AMEk,EbciJM->AbciJk', H.ab.vovo[Va, Ob, Va, ob], T.aab.VvvoOO, optimize=True)
    )
    dT.aab.VvvoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('bmek,AeciJm->AbciJk', H.ab.vovo[va, ob, va, ob], T.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('bMek,AeciJM->AbciJk', H.ab.vovo[va, Ob, va, ob], T.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('bmEk,AEciJm->AbciJk', H.ab.vovo[va, ob, Va, ob], T.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('bMEk,AEciJM->AbciJk', H.ab.vovo[va, Ob, Va, ob], T.aab.VVvoOO, optimize=True)
    )
    dT.aab.VvvoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('mcie,AbemJk->AbciJk', H.ab.ovov[oa, vb, oa, vb], T.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('Mcie,AbeMJk->AbciJk', H.ab.ovov[Oa, vb, oa, vb], T.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('mciE,AbEmJk->AbciJk', H.ab.ovov[oa, vb, oa, Vb], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('MciE,AbEMJk->AbciJk', H.ab.ovov[Oa, vb, oa, Vb], T.aab.VvVOOo, optimize=True)
    )
    dT.aab.VvvoOo += (1.0 / 1.0) * (
            -1.0 * np.einsum('McJe,AbeiMk->AbciJk', H.ab.ovov[Oa, vb, Oa, vb], T.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('McJE,AbEiMk->AbciJk', H.ab.ovov[Oa, vb, Oa, Vb], T.aab.VvVoOo, optimize=True)
    )

    return T, dT