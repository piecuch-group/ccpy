import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops

def build(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.aab.VVvoOo = (2.0 / 2.0) * (
            +1.0 * np.einsum('Bcek,AeiJ->ABciJk', X.ab.vvvo[Va, vb, :, ob], T.aa[Va, :, oa, Oa], optimize=True)
    )
    dR.aab.VVvoOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('mcJk,ABim->ABciJk', X.ab.ovoo[:, vb, Oa, ob], T.aa[Va, Va, oa, :], optimize=True)
    )
    dR.aab.VVvoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('mcik,ABJm->ABciJk', X.ab.ovoo[:, vb, oa, ob], T.aa[Va, Va, Oa, :], optimize=True)
    )
    dR.aab.VVvoOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('Acie,BeJk->ABciJk', X.ab.vvov[Va, vb, oa, :], T.ab[Va, :, Oa, ob], optimize=True)
    )
    dR.aab.VVvoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('AcJe,Beik->ABciJk', X.ab.vvov[Va, vb, Oa, :], T.ab[Va, :, oa, ob], optimize=True)
    )
    dR.aab.VVvoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('Amik,BcJm->ABciJk', X.ab.vooo[Va, :, oa, ob], T.ab[Va, vb, Oa, :], optimize=True)
    )
    dR.aab.VVvoOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('AmJk,Bcim->ABciJk', X.ab.vooo[Va, :, Oa, ob], T.ab[Va, vb, oa, :], optimize=True)
    )
    dR.aab.VVvoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('ABie,ecJk->ABciJk', X.aa.vvov[Va, Va, oa, :], T.ab[:, vb, Oa, ob], optimize=True)
    )
    dR.aab.VVvoOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('ABJe,ecik->ABciJk', X.aa.vvov[Va, Va, Oa, :], T.ab[:, vb, oa, ob], optimize=True)
    )
    dR.aab.VVvoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('AmiJ,Bcmk->ABciJk', X.aa.vooo[Va, :, oa, Oa], T.ab[Va, vb, :, ob], optimize=True)
    )
    dR.aab.VVvoOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('Bcek,AeiJ->ABciJk', H.ab.vvvo[Va, vb, :, ob], R.aa[Va, :, oa, Oa], optimize=True)
    )
    dR.aab.VVvoOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('mcJk,ABim->ABciJk', H.ab.ovoo[:, vb, Oa, ob], R.aa[Va, Va, oa, :], optimize=True)
    )
    dR.aab.VVvoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('mcik,ABJm->ABciJk', H.ab.ovoo[:, vb, oa, ob], R.aa[Va, Va, Oa, :], optimize=True)
    )
    dR.aab.VVvoOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('Acie,BeJk->ABciJk', H.ab.vvov[Va, vb, oa, :], R.ab[Va, :, Oa, ob], optimize=True)
    )
    dR.aab.VVvoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('AcJe,Beik->ABciJk', H.ab.vvov[Va, vb, Oa, :], R.ab[Va, :, oa, ob], optimize=True)
    )
    dR.aab.VVvoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('Amik,BcJm->ABciJk', H.ab.vooo[Va, :, oa, ob], R.ab[Va, vb, Oa, :], optimize=True)
    )
    dR.aab.VVvoOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('AmJk,Bcim->ABciJk', H.ab.vooo[Va, :, Oa, ob], R.ab[Va, vb, oa, :], optimize=True)
    )
    dR.aab.VVvoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('ABie,ecJk->ABciJk', H.aa.vvov[Va, Va, oa, :], R.ab[:, vb, Oa, ob], optimize=True)
    )
    dR.aab.VVvoOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('ABJe,ecik->ABciJk', H.aa.vvov[Va, Va, Oa, :], R.ab[:, vb, oa, ob], optimize=True)
    )
    dR.aab.VVvoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('AmiJ,Bcmk->ABciJk', H.aa.vooo[Va, :, oa, Oa], R.ab[Va, vb, :, ob], optimize=True)
    )

    dR.aab.VVvoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('mi,BAcmJk->ABciJk', X.a.oo[oa, oa], T.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('Mi,BAcMJk->ABciJk', X.a.oo[Oa, oa], T.aab.VVvOOo, optimize=True)
    )
    dR.aab.VVvoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('MJ,BAciMk->ABciJk', X.a.oo[Oa, Oa], T.aab.VVvoOo, optimize=True)
    )
    dR.aab.VVvoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('mk,BAciJm->ABciJk', X.b.oo[ob, ob], T.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('Mk,BAciJM->ABciJk', X.b.oo[Ob, ob], T.aab.VVvoOO, optimize=True)
    )
    dR.aab.VVvoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('Ae,BeciJk->ABciJk', X.a.vv[Va, va], T.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('AE,BEciJk->ABciJk', X.a.vv[Va, Va], T.aab.VVvoOo, optimize=True)
    )
    dR.aab.VVvoOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('ce,BAeiJk->ABciJk', X.b.vv[vb, vb], T.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('cE,BAEiJk->ABciJk', X.b.vv[vb, Vb], T.aab.VVVoOo, optimize=True)
    )
    dR.aab.VVvoOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('mNiJ,BAcmNk->ABciJk', X.aa.oooo[oa, Oa, oa, Oa], T.aab.VVvoOo, optimize=True)
            - 0.5 * np.einsum('MNiJ,BAcMNk->ABciJk', X.aa.oooo[Oa, Oa, oa, Oa], T.aab.VVvOOo, optimize=True)
    )
    dR.aab.VVvoOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('MnJk,BAciMn->ABciJk', X.ab.oooo[Oa, ob, Oa, ob], T.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('mNJk,BAcimN->ABciJk', X.ab.oooo[oa, Ob, Oa, ob], T.aab.VVvooO, optimize=True)
            - 1.0 * np.einsum('MNJk,BAciMN->ABciJk', X.ab.oooo[Oa, Ob, Oa, ob], T.aab.VVvoOO, optimize=True)
    )
    dR.aab.VVvoOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('mnik,BAcmJn->ABciJk', X.ab.oooo[oa, ob, oa, ob], T.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('Mnik,BAcJMn->ABciJk', X.ab.oooo[Oa, ob, oa, ob], T.aab.VVvOOo, optimize=True)
            - 1.0 * np.einsum('mNik,BAcmJN->ABciJk', X.ab.oooo[oa, Ob, oa, ob], T.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('MNik,BAcJMN->ABciJk', X.ab.oooo[Oa, Ob, oa, ob], T.aab.VVvOOO, optimize=True)
    )
    dR.aab.VVvoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('ABEf,EfciJk->ABciJk', X.aa.vvvv[Va, Va, Va, va], T.aab.VvvoOo, optimize=True)
            - 0.5 * np.einsum('ABEF,FEciJk->ABciJk', X.aa.vvvv[Va, Va, Va, Va], T.aab.VVvoOo, optimize=True)
    )
    dR.aab.VVvoOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('Bcef,AefiJk->ABciJk', X.ab.vvvv[Va, vb, va, vb], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('BceF,AeFiJk->ABciJk', X.ab.vvvv[Va, vb, va, Vb], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('BcEf,EAfiJk->ABciJk', X.ab.vvvv[Va, vb, Va, vb], T.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('BcEF,EAFiJk->ABciJk', X.ab.vvvv[Va, vb, Va, Vb], T.aab.VVVoOo, optimize=True)
    )
    dR.aab.VVvoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('Amie,BecmJk->ABciJk', X.aa.voov[Va, oa, oa, va], T.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('AMie,BecMJk->ABciJk', X.aa.voov[Va, Oa, oa, va], T.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('AmiE,BEcmJk->ABciJk', X.aa.voov[Va, oa, oa, Va], T.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('AMiE,BEcMJk->ABciJk', X.aa.voov[Va, Oa, oa, Va], T.aab.VVvOOo, optimize=True)
    )
    dR.aab.VVvoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('AMJe,BeciMk->ABciJk', X.aa.voov[Va, Oa, Oa, va], T.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('AMJE,BEciMk->ABciJk', X.aa.voov[Va, Oa, Oa, Va], T.aab.VVvoOo, optimize=True)
    )
    dR.aab.VVvoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('Amie,BceJmk->ABciJk', X.ab.voov[Va, ob, oa, vb], T.abb.VvvOoo, optimize=True)
            + 1.0 * np.einsum('AMie,BceJkM->ABciJk', X.ab.voov[Va, Ob, oa, vb], T.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('AmiE,BEcJmk->ABciJk', X.ab.voov[Va, ob, oa, Vb], T.abb.VVvOoo, optimize=True)
            - 1.0 * np.einsum('AMiE,BEcJkM->ABciJk', X.ab.voov[Va, Ob, oa, Vb], T.abb.VVvOoO, optimize=True)
    )
    dR.aab.VVvoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('AMJe,BceikM->ABciJk', X.ab.voov[Va, Ob, Oa, vb], T.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('AMJE,BEcikM->ABciJk', X.ab.voov[Va, Ob, Oa, Vb], T.abb.VVvooO, optimize=True)
    )
    dR.aab.VVvoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('mcek,BAeimJ->ABciJk', X.ab.ovvo[oa, vb, va, ob], T.aaa.VVvooO, optimize=True)
            - 1.0 * np.einsum('Mcek,BAeiJM->ABciJk', X.ab.ovvo[Oa, vb, va, ob], T.aaa.VVvoOO, optimize=True)
            + 1.0 * np.einsum('mcEk,EBAimJ->ABciJk', X.ab.ovvo[oa, vb, Va, ob], T.aaa.VVVooO, optimize=True)
            - 1.0 * np.einsum('McEk,EBAiJM->ABciJk', X.ab.ovvo[Oa, vb, Va, ob], T.aaa.VVVoOO, optimize=True)
    )
    dR.aab.VVvoOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('cmke,BAeiJm->ABciJk', X.bb.voov[vb, ob, ob, vb], T.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('cMke,BAeiJM->ABciJk', X.bb.voov[vb, Ob, ob, vb], T.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('cmkE,BAEiJm->ABciJk', X.bb.voov[vb, ob, ob, Vb], T.aab.VVVoOo, optimize=True)
            - 1.0 * np.einsum('cMkE,BAEiJM->ABciJk', X.bb.voov[vb, Ob, ob, Vb], T.aab.VVVoOO, optimize=True)
    )
    dR.aab.VVvoOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('Amek,BeciJm->ABciJk', X.ab.vovo[Va, ob, va, ob], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('AMek,BeciJM->ABciJk', X.ab.vovo[Va, Ob, va, ob], T.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('AmEk,BEciJm->ABciJk', X.ab.vovo[Va, ob, Va, ob], T.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('AMEk,BEciJM->ABciJk', X.ab.vovo[Va, Ob, Va, ob], T.aab.VVvoOO, optimize=True)
    )
    dR.aab.VVvoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('mcie,BAemJk->ABciJk', X.ab.ovov[oa, vb, oa, vb], T.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('Mcie,BAeMJk->ABciJk', X.ab.ovov[Oa, vb, oa, vb], T.aab.VVvOOo, optimize=True)
            + 1.0 * np.einsum('mciE,BAEmJk->ABciJk', X.ab.ovov[oa, vb, oa, Vb], T.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('MciE,BAEMJk->ABciJk', X.ab.ovov[Oa, vb, oa, Vb], T.aab.VVVOOo, optimize=True)
    )
    dR.aab.VVvoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('McJe,BAeiMk->ABciJk', X.ab.ovov[Oa, vb, Oa, vb], T.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('McJE,BAEiMk->ABciJk', X.ab.ovov[Oa, vb, Oa, Vb], T.aab.VVVoOo, optimize=True)
    )
    dR.aab.VVvoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('mi,BAcmJk->ABciJk', H.a.oo[oa, oa], R.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('Mi,BAcMJk->ABciJk', H.a.oo[Oa, oa], R.aab.VVvOOo, optimize=True)
    )
    dR.aab.VVvoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('MJ,BAciMk->ABciJk', H.a.oo[Oa, Oa], R.aab.VVvoOo, optimize=True)
    )
    dR.aab.VVvoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('mk,BAciJm->ABciJk', H.b.oo[ob, ob], R.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('Mk,BAciJM->ABciJk', H.b.oo[Ob, ob], R.aab.VVvoOO, optimize=True)
    )
    dR.aab.VVvoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('Ae,BeciJk->ABciJk', H.a.vv[Va, va], R.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('AE,BEciJk->ABciJk', H.a.vv[Va, Va], R.aab.VVvoOo, optimize=True)
    )
    dR.aab.VVvoOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('ce,BAeiJk->ABciJk', H.b.vv[vb, vb], R.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('cE,BAEiJk->ABciJk', H.b.vv[vb, Vb], R.aab.VVVoOo, optimize=True)
    )
    dR.aab.VVvoOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('mNiJ,BAcmNk->ABciJk', H.aa.oooo[oa, Oa, oa, Oa], R.aab.VVvoOo, optimize=True)
            - 0.5 * np.einsum('MNiJ,BAcMNk->ABciJk', H.aa.oooo[Oa, Oa, oa, Oa], R.aab.VVvOOo, optimize=True)
    )
    dR.aab.VVvoOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('MnJk,BAciMn->ABciJk', H.ab.oooo[Oa, ob, Oa, ob], R.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('mNJk,BAcimN->ABciJk', H.ab.oooo[oa, Ob, Oa, ob], R.aab.VVvooO, optimize=True)
            - 1.0 * np.einsum('MNJk,BAciMN->ABciJk', H.ab.oooo[Oa, Ob, Oa, ob], R.aab.VVvoOO, optimize=True)
    )
    dR.aab.VVvoOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('mnik,BAcmJn->ABciJk', H.ab.oooo[oa, ob, oa, ob], R.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('Mnik,BAcJMn->ABciJk', H.ab.oooo[Oa, ob, oa, ob], R.aab.VVvOOo, optimize=True)
            - 1.0 * np.einsum('mNik,BAcmJN->ABciJk', H.ab.oooo[oa, Ob, oa, ob], R.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('MNik,BAcJMN->ABciJk', H.ab.oooo[Oa, Ob, oa, ob], R.aab.VVvOOO, optimize=True)
    )
    dR.aab.VVvoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('ABEf,EfciJk->ABciJk', H.aa.vvvv[Va, Va, Va, va], R.aab.VvvoOo, optimize=True)
            - 0.5 * np.einsum('ABEF,FEciJk->ABciJk', H.aa.vvvv[Va, Va, Va, Va], R.aab.VVvoOo, optimize=True)
    )
    dR.aab.VVvoOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('Bcef,AefiJk->ABciJk', H.ab.vvvv[Va, vb, va, vb], R.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('BceF,AeFiJk->ABciJk', H.ab.vvvv[Va, vb, va, Vb], R.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('BcEf,EAfiJk->ABciJk', H.ab.vvvv[Va, vb, Va, vb], R.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('BcEF,EAFiJk->ABciJk', H.ab.vvvv[Va, vb, Va, Vb], R.aab.VVVoOo, optimize=True)
    )
    dR.aab.VVvoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('Amie,BecmJk->ABciJk', H.aa.voov[Va, oa, oa, va], R.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('AMie,BecMJk->ABciJk', H.aa.voov[Va, Oa, oa, va], R.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('AmiE,BEcmJk->ABciJk', H.aa.voov[Va, oa, oa, Va], R.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('AMiE,BEcMJk->ABciJk', H.aa.voov[Va, Oa, oa, Va], R.aab.VVvOOo, optimize=True)
    )
    dR.aab.VVvoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('AMJe,BeciMk->ABciJk', H.aa.voov[Va, Oa, Oa, va], R.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('AMJE,BEciMk->ABciJk', H.aa.voov[Va, Oa, Oa, Va], R.aab.VVvoOo, optimize=True)
    )
    dR.aab.VVvoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('Amie,BceJmk->ABciJk', H.ab.voov[Va, ob, oa, vb], R.abb.VvvOoo, optimize=True)
            + 1.0 * np.einsum('AMie,BceJkM->ABciJk', H.ab.voov[Va, Ob, oa, vb], R.abb.VvvOoO, optimize=True)
            + 1.0 * np.einsum('AmiE,BEcJmk->ABciJk', H.ab.voov[Va, ob, oa, Vb], R.abb.VVvOoo, optimize=True)
            - 1.0 * np.einsum('AMiE,BEcJkM->ABciJk', H.ab.voov[Va, Ob, oa, Vb], R.abb.VVvOoO, optimize=True)
    )
    dR.aab.VVvoOo += (2.0 / 2.0) * (
            -1.0 * np.einsum('AMJe,BceikM->ABciJk', H.ab.voov[Va, Ob, Oa, vb], R.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('AMJE,BEcikM->ABciJk', H.ab.voov[Va, Ob, Oa, Vb], R.abb.VVvooO, optimize=True)
    )
    dR.aab.VVvoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('mcek,BAeimJ->ABciJk', H.ab.ovvo[oa, vb, va, ob], R.aaa.VVvooO, optimize=True)
            - 1.0 * np.einsum('Mcek,BAeiJM->ABciJk', H.ab.ovvo[Oa, vb, va, ob], R.aaa.VVvoOO, optimize=True)
            + 1.0 * np.einsum('mcEk,EBAimJ->ABciJk', H.ab.ovvo[oa, vb, Va, ob], R.aaa.VVVooO, optimize=True)
            - 1.0 * np.einsum('McEk,EBAiJM->ABciJk', H.ab.ovvo[Oa, vb, Va, ob], R.aaa.VVVoOO, optimize=True)
    )
    dR.aab.VVvoOo += (1.0 / 2.0) * (
            -1.0 * np.einsum('cmke,BAeiJm->ABciJk', H.bb.voov[vb, ob, ob, vb], R.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('cMke,BAeiJM->ABciJk', H.bb.voov[vb, Ob, ob, vb], R.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('cmkE,BAEiJm->ABciJk', H.bb.voov[vb, ob, ob, Vb], R.aab.VVVoOo, optimize=True)
            - 1.0 * np.einsum('cMkE,BAEiJM->ABciJk', H.bb.voov[vb, Ob, ob, Vb], R.aab.VVVoOO, optimize=True)
    )
    dR.aab.VVvoOo += (2.0 / 2.0) * (
            +1.0 * np.einsum('Amek,BeciJm->ABciJk', H.ab.vovo[Va, ob, va, ob], R.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('AMek,BeciJM->ABciJk', H.ab.vovo[Va, Ob, va, ob], R.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('AmEk,BEciJm->ABciJk', H.ab.vovo[Va, ob, Va, ob], R.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('AMEk,BEciJM->ABciJk', H.ab.vovo[Va, Ob, Va, ob], R.aab.VVvoOO, optimize=True)
    )
    dR.aab.VVvoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('mcie,BAemJk->ABciJk', H.ab.ovov[oa, vb, oa, vb], R.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('Mcie,BAeMJk->ABciJk', H.ab.ovov[Oa, vb, oa, vb], R.aab.VVvOOo, optimize=True)
            + 1.0 * np.einsum('mciE,BAEmJk->ABciJk', H.ab.ovov[oa, vb, oa, Vb], R.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('MciE,BAEMJk->ABciJk', H.ab.ovov[Oa, vb, oa, Vb], R.aab.VVVOOo, optimize=True)
    )
    dR.aab.VVvoOo += (1.0 / 2.0) * (
            +1.0 * np.einsum('McJe,BAeiMk->ABciJk', H.ab.ovov[Oa, vb, Oa, vb], R.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('McJE,BAEiMk->ABciJk', H.ab.ovov[Oa, vb, Oa, Vb], R.aab.VVVoOo, optimize=True)
    )

    dR.aab.VVvoOo -= np.transpose(dR.aab.VVvoOo, (1, 0, 2, 3, 4, 5))

    return dR

def update(R, omega, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    R.aab.VVvoOo = eomcc_active_loops.update_r3b_110010(
        R.aab.VVvoOo,
        omega,
        H.a.oo[Oa, Oa],
        H.a.vv[Va, Va],
        H.a.oo[oa, oa],
        H.a.vv[va, va],
        H.b.oo[Ob, Ob],
        H.b.vv[Vb, Vb],
        H.b.oo[ob, ob],
        H.b.vv[vb, vb],
        0.0,
    )
    return R

