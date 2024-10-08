import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops

def build(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.aab.VVvoOO = (2.0 / 2.0) * (
            +1.0 * np.einsum('BceK,AeiJ->ABciJK', X.ab.vvvo[Va, vb, :, Ob], T.aa[Va, :, oa, Oa], optimize=True)
    )
    dR.aab.VVvoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mcJK,ABim->ABciJK', X.ab.ovoo[:, vb, Oa, Ob], T.aa[Va, Va, oa, :], optimize=True)
    )
    dR.aab.VVvoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mciK,ABJm->ABciJK', X.ab.ovoo[:, vb, oa, Ob], T.aa[Va, Va, Oa, :], optimize=True)
    )
    dR.aab.VVvoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('Acie,BeJK->ABciJK', X.ab.vvov[Va, vb, oa, :], T.ab[Va, :, Oa, Ob], optimize=True)
    )
    dR.aab.VVvoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('AcJe,BeiK->ABciJK', X.ab.vvov[Va, vb, Oa, :], T.ab[Va, :, oa, Ob], optimize=True)
    )
    dR.aab.VVvoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('AmiK,BcJm->ABciJK', X.ab.vooo[Va, :, oa, Ob], T.ab[Va, vb, Oa, :], optimize=True)
    )
    dR.aab.VVvoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('AmJK,Bcim->ABciJK', X.ab.vooo[Va, :, Oa, Ob], T.ab[Va, vb, oa, :], optimize=True)
    )
    dR.aab.VVvoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('ABie,ecJK->ABciJK', X.aa.vvov[Va, Va, oa, :], T.ab[:, vb, Oa, Ob], optimize=True)
    )
    dR.aab.VVvoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('ABJe,eciK->ABciJK', X.aa.vvov[Va, Va, Oa, :], T.ab[:, vb, oa, Ob], optimize=True)
    )
    dR.aab.VVvoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('AmiJ,BcmK->ABciJK', X.aa.vooo[Va, :, oa, Oa], T.ab[Va, vb, :, Ob], optimize=True)
    )
    dR.aab.VVvoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('BceK,AeiJ->ABciJK', H.ab.vvvo[Va, vb, :, Ob], R.aa[Va, :, oa, Oa], optimize=True)
    )
    dR.aab.VVvoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mcJK,ABim->ABciJK', H.ab.ovoo[:, vb, Oa, Ob], R.aa[Va, Va, oa, :], optimize=True)
    )
    dR.aab.VVvoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mciK,ABJm->ABciJK', H.ab.ovoo[:, vb, oa, Ob], R.aa[Va, Va, Oa, :], optimize=True)
    )
    dR.aab.VVvoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('Acie,BeJK->ABciJK', H.ab.vvov[Va, vb, oa, :], R.ab[Va, :, Oa, Ob], optimize=True)
    )
    dR.aab.VVvoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('AcJe,BeiK->ABciJK', H.ab.vvov[Va, vb, Oa, :], R.ab[Va, :, oa, Ob], optimize=True)
    )
    dR.aab.VVvoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('AmiK,BcJm->ABciJK', H.ab.vooo[Va, :, oa, Ob], R.ab[Va, vb, Oa, :], optimize=True)
    )
    dR.aab.VVvoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('AmJK,Bcim->ABciJK', H.ab.vooo[Va, :, Oa, Ob], R.ab[Va, vb, oa, :], optimize=True)
    )
    dR.aab.VVvoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('ABie,ecJK->ABciJK', H.aa.vvov[Va, Va, oa, :], R.ab[:, vb, Oa, Ob], optimize=True)
    )
    dR.aab.VVvoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('ABJe,eciK->ABciJK', H.aa.vvov[Va, Va, Oa, :], R.ab[:, vb, oa, Ob], optimize=True)
    )
    dR.aab.VVvoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('AmiJ,BcmK->ABciJK', H.aa.vooo[Va, :, oa, Oa], R.ab[Va, vb, :, Ob], optimize=True)
    )

    dR.aab.VVvoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mi,BAcmJK->ABciJK', X.a.oo[oa, oa], T.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('Mi,BAcMJK->ABciJK', X.a.oo[Oa, oa], T.aab.VVvOOO, optimize=True)
    )
    dR.aab.VVvoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mJ,BAcmiK->ABciJK', X.a.oo[oa, Oa], T.aab.VVvooO, optimize=True)
            + 1.0 * np.einsum('MJ,BAciMK->ABciJK', X.a.oo[Oa, Oa], T.aab.VVvoOO, optimize=True)
    )
    dR.aab.VVvoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mK,BAciJm->ABciJK', X.b.oo[ob, Ob], T.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('MK,BAciJM->ABciJK', X.b.oo[Ob, Ob], T.aab.VVvoOO, optimize=True)
    )
    dR.aab.VVvoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('Ae,BeciJK->ABciJK', X.a.vv[Va, va], T.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('AE,BEciJK->ABciJK', X.a.vv[Va, Va], T.aab.VVvoOO, optimize=True)
    )
    dR.aab.VVvoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('ce,BAeiJK->ABciJK', X.b.vv[vb, vb], T.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('cE,BAEiJK->ABciJK', X.b.vv[vb, Vb], T.aab.VVVoOO, optimize=True)
    )
    dR.aab.VVvoOO += (1.0 / 2.0) * (
            -0.5 * np.einsum('mniJ,BAcmnK->ABciJK', X.aa.oooo[oa, oa, oa, Oa], T.aab.VVvooO, optimize=True)
            + 1.0 * np.einsum('MniJ,BAcnMK->ABciJK', X.aa.oooo[Oa, oa, oa, Oa], T.aab.VVvoOO, optimize=True)
            - 0.5 * np.einsum('MNiJ,BAcMNK->ABciJK', X.aa.oooo[Oa, Oa, oa, Oa], T.aab.VVvOOO, optimize=True)
    )
    dR.aab.VVvoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mNJK,BAcimN->ABciJK', X.ab.oooo[oa, Ob, Oa, Ob], T.aab.VVvooO, optimize=True)
            - 1.0 * np.einsum('MnJK,BAciMn->ABciJK', X.ab.oooo[Oa, ob, Oa, Ob], T.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('MNJK,BAciMN->ABciJK', X.ab.oooo[Oa, Ob, Oa, Ob], T.aab.VVvoOO, optimize=True)
    )
    dR.aab.VVvoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mniK,BAcmJn->ABciJK', X.ab.oooo[oa, ob, oa, Ob], T.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('mNiK,BAcmJN->ABciJK', X.ab.oooo[oa, Ob, oa, Ob], T.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('MniK,BAcJMn->ABciJK', X.ab.oooo[Oa, ob, oa, Ob], T.aab.VVvOOo, optimize=True)
            + 1.0 * np.einsum('MNiK,BAcJMN->ABciJK', X.ab.oooo[Oa, Ob, oa, Ob], T.aab.VVvOOO, optimize=True)
    )
    dR.aab.VVvoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('ABEf,EfciJK->ABciJK', X.aa.vvvv[Va, Va, Va, va], T.aab.VvvoOO, optimize=True)
            - 0.5 * np.einsum('ABEF,FEciJK->ABciJK', X.aa.vvvv[Va, Va, Va, Va], T.aab.VVvoOO, optimize=True)
    )
    dR.aab.VVvoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('Bcef,AefiJK->ABciJK', X.ab.vvvv[Va, vb, va, vb], T.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('BceF,AeFiJK->ABciJK', X.ab.vvvv[Va, vb, va, Vb], T.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('BcEf,EAfiJK->ABciJK', X.ab.vvvv[Va, vb, Va, vb], T.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('BcEF,EAFiJK->ABciJK', X.ab.vvvv[Va, vb, Va, Vb], T.aab.VVVoOO, optimize=True)
    )
    dR.aab.VVvoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('Amie,BecmJK->ABciJK', X.aa.voov[Va, oa, oa, va], T.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('AMie,BecMJK->ABciJK', X.aa.voov[Va, Oa, oa, va], T.aab.VvvOOO, optimize=True)
            - 1.0 * np.einsum('AmiE,BEcmJK->ABciJK', X.aa.voov[Va, oa, oa, Va], T.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('AMiE,BEcMJK->ABciJK', X.aa.voov[Va, Oa, oa, Va], T.aab.VVvOOO, optimize=True)
    )
    dR.aab.VVvoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('AmJe,BecmiK->ABciJK', X.aa.voov[Va, oa, Oa, va], T.aab.VvvooO, optimize=True)
            - 1.0 * np.einsum('AMJe,BeciMK->ABciJK', X.aa.voov[Va, Oa, Oa, va], T.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('AmJE,BEcmiK->ABciJK', X.aa.voov[Va, oa, Oa, Va], T.aab.VVvooO, optimize=True)
            - 1.0 * np.einsum('AMJE,BEciMK->ABciJK', X.aa.voov[Va, Oa, Oa, Va], T.aab.VVvoOO, optimize=True)
    )
    dR.aab.VVvoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('Amie,BceJmK->ABciJK', X.ab.voov[Va, ob, oa, vb], T.abb.VvvOoO, optimize=True)
            - 1.0 * np.einsum('AMie,BceJMK->ABciJK', X.ab.voov[Va, Ob, oa, vb], T.abb.VvvOOO, optimize=True)
            + 1.0 * np.einsum('AmiE,BEcJmK->ABciJK', X.ab.voov[Va, ob, oa, Vb], T.abb.VVvOoO, optimize=True)
            + 1.0 * np.einsum('AMiE,BEcJMK->ABciJK', X.ab.voov[Va, Ob, oa, Vb], T.abb.VVvOOO, optimize=True)
    )
    dR.aab.VVvoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('AmJe,BceimK->ABciJK', X.ab.voov[Va, ob, Oa, vb], T.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('AMJe,BceiMK->ABciJK', X.ab.voov[Va, Ob, Oa, vb], T.abb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('AmJE,BEcimK->ABciJK', X.ab.voov[Va, ob, Oa, Vb], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('AMJE,BEciMK->ABciJK', X.ab.voov[Va, Ob, Oa, Vb], T.abb.VVvoOO, optimize=True)
    )
    dR.aab.VVvoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mceK,BAeimJ->ABciJK', X.ab.ovvo[oa, vb, va, Ob], T.aaa.VVvooO, optimize=True)
            - 1.0 * np.einsum('MceK,BAeiJM->ABciJK', X.ab.ovvo[Oa, vb, va, Ob], T.aaa.VVvoOO, optimize=True)
            + 1.0 * np.einsum('mcEK,EBAimJ->ABciJK', X.ab.ovvo[oa, vb, Va, Ob], T.aaa.VVVooO, optimize=True)
            - 1.0 * np.einsum('McEK,EBAiJM->ABciJK', X.ab.ovvo[Oa, vb, Va, Ob], T.aaa.VVVoOO, optimize=True)
    )
    dR.aab.VVvoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('cmKe,BAeiJm->ABciJK', X.bb.voov[vb, ob, Ob, vb], T.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('cMKe,BAeiJM->ABciJK', X.bb.voov[vb, Ob, Ob, vb], T.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('cmKE,BAEiJm->ABciJK', X.bb.voov[vb, ob, Ob, Vb], T.aab.VVVoOo, optimize=True)
            - 1.0 * np.einsum('cMKE,BAEiJM->ABciJK', X.bb.voov[vb, Ob, Ob, Vb], T.aab.VVVoOO, optimize=True)
    )
    dR.aab.VVvoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('AmeK,BeciJm->ABciJK', X.ab.vovo[Va, ob, va, Ob], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('AMeK,BeciJM->ABciJK', X.ab.vovo[Va, Ob, va, Ob], T.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('AmEK,BEciJm->ABciJK', X.ab.vovo[Va, ob, Va, Ob], T.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('AMEK,BEciJM->ABciJK', X.ab.vovo[Va, Ob, Va, Ob], T.aab.VVvoOO, optimize=True)
    )
    dR.aab.VVvoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mcie,BAemJK->ABciJK', X.ab.ovov[oa, vb, oa, vb], T.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('Mcie,BAeMJK->ABciJK', X.ab.ovov[Oa, vb, oa, vb], T.aab.VVvOOO, optimize=True)
            + 1.0 * np.einsum('mciE,BAEmJK->ABciJK', X.ab.ovov[oa, vb, oa, Vb], T.aab.VVVoOO, optimize=True)
            + 1.0 * np.einsum('MciE,BAEMJK->ABciJK', X.ab.ovov[Oa, vb, oa, Vb], T.aab.VVVOOO, optimize=True)
    )
    dR.aab.VVvoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mcJe,BAemiK->ABciJK', X.ab.ovov[oa, vb, Oa, vb], T.aab.VVvooO, optimize=True)
            + 1.0 * np.einsum('McJe,BAeiMK->ABciJK', X.ab.ovov[Oa, vb, Oa, vb], T.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('mcJE,BAEmiK->ABciJK', X.ab.ovov[oa, vb, Oa, Vb], T.aab.VVVooO, optimize=True)
            + 1.0 * np.einsum('McJE,BAEiMK->ABciJK', X.ab.ovov[Oa, vb, Oa, Vb], T.aab.VVVoOO, optimize=True)
    )
    dR.aab.VVvoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mi,BAcmJK->ABciJK', H.a.oo[oa, oa], R.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('Mi,BAcMJK->ABciJK', H.a.oo[Oa, oa], R.aab.VVvOOO, optimize=True)
    )
    dR.aab.VVvoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mJ,BAcmiK->ABciJK', H.a.oo[oa, Oa], R.aab.VVvooO, optimize=True)
            + 1.0 * np.einsum('MJ,BAciMK->ABciJK', H.a.oo[Oa, Oa], R.aab.VVvoOO, optimize=True)
    )
    dR.aab.VVvoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mK,BAciJm->ABciJK', H.b.oo[ob, Ob], R.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('MK,BAciJM->ABciJK', H.b.oo[Ob, Ob], R.aab.VVvoOO, optimize=True)
    )
    dR.aab.VVvoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('Ae,BeciJK->ABciJK', H.a.vv[Va, va], R.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('AE,BEciJK->ABciJK', H.a.vv[Va, Va], R.aab.VVvoOO, optimize=True)
    )
    dR.aab.VVvoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('ce,BAeiJK->ABciJK', H.b.vv[vb, vb], R.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('cE,BAEiJK->ABciJK', H.b.vv[vb, Vb], R.aab.VVVoOO, optimize=True)
    )
    dR.aab.VVvoOO += (1.0 / 2.0) * (
            -0.5 * np.einsum('mniJ,BAcmnK->ABciJK', H.aa.oooo[oa, oa, oa, Oa], R.aab.VVvooO, optimize=True)
            + 1.0 * np.einsum('MniJ,BAcnMK->ABciJK', H.aa.oooo[Oa, oa, oa, Oa], R.aab.VVvoOO, optimize=True)
            - 0.5 * np.einsum('MNiJ,BAcMNK->ABciJK', H.aa.oooo[Oa, Oa, oa, Oa], R.aab.VVvOOO, optimize=True)
    )
    dR.aab.VVvoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mNJK,BAcimN->ABciJK', H.ab.oooo[oa, Ob, Oa, Ob], R.aab.VVvooO, optimize=True)
            - 1.0 * np.einsum('MnJK,BAciMn->ABciJK', H.ab.oooo[Oa, ob, Oa, Ob], R.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('MNJK,BAciMN->ABciJK', H.ab.oooo[Oa, Ob, Oa, Ob], R.aab.VVvoOO, optimize=True)
    )
    dR.aab.VVvoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mniK,BAcmJn->ABciJK', H.ab.oooo[oa, ob, oa, Ob], R.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('mNiK,BAcmJN->ABciJK', H.ab.oooo[oa, Ob, oa, Ob], R.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('MniK,BAcJMn->ABciJK', H.ab.oooo[Oa, ob, oa, Ob], R.aab.VVvOOo, optimize=True)
            + 1.0 * np.einsum('MNiK,BAcJMN->ABciJK', H.ab.oooo[Oa, Ob, oa, Ob], R.aab.VVvOOO, optimize=True)
    )
    dR.aab.VVvoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('ABEf,EfciJK->ABciJK', H.aa.vvvv[Va, Va, Va, va], R.aab.VvvoOO, optimize=True)
            - 0.5 * np.einsum('ABEF,FEciJK->ABciJK', H.aa.vvvv[Va, Va, Va, Va], R.aab.VVvoOO, optimize=True)
    )
    dR.aab.VVvoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('Bcef,AefiJK->ABciJK', H.ab.vvvv[Va, vb, va, vb], R.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('BceF,AeFiJK->ABciJK', H.ab.vvvv[Va, vb, va, Vb], R.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('BcEf,EAfiJK->ABciJK', H.ab.vvvv[Va, vb, Va, vb], R.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('BcEF,EAFiJK->ABciJK', H.ab.vvvv[Va, vb, Va, Vb], R.aab.VVVoOO, optimize=True)
    )
    dR.aab.VVvoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('Amie,BecmJK->ABciJK', H.aa.voov[Va, oa, oa, va], R.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('AMie,BecMJK->ABciJK', H.aa.voov[Va, Oa, oa, va], R.aab.VvvOOO, optimize=True)
            - 1.0 * np.einsum('AmiE,BEcmJK->ABciJK', H.aa.voov[Va, oa, oa, Va], R.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('AMiE,BEcMJK->ABciJK', H.aa.voov[Va, Oa, oa, Va], R.aab.VVvOOO, optimize=True)
    )
    dR.aab.VVvoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('AmJe,BecmiK->ABciJK', H.aa.voov[Va, oa, Oa, va], R.aab.VvvooO, optimize=True)
            - 1.0 * np.einsum('AMJe,BeciMK->ABciJK', H.aa.voov[Va, Oa, Oa, va], R.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('AmJE,BEcmiK->ABciJK', H.aa.voov[Va, oa, Oa, Va], R.aab.VVvooO, optimize=True)
            - 1.0 * np.einsum('AMJE,BEciMK->ABciJK', H.aa.voov[Va, Oa, Oa, Va], R.aab.VVvoOO, optimize=True)
    )
    dR.aab.VVvoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('Amie,BceJmK->ABciJK', H.ab.voov[Va, ob, oa, vb], R.abb.VvvOoO, optimize=True)
            - 1.0 * np.einsum('AMie,BceJMK->ABciJK', H.ab.voov[Va, Ob, oa, vb], R.abb.VvvOOO, optimize=True)
            + 1.0 * np.einsum('AmiE,BEcJmK->ABciJK', H.ab.voov[Va, ob, oa, Vb], R.abb.VVvOoO, optimize=True)
            + 1.0 * np.einsum('AMiE,BEcJMK->ABciJK', H.ab.voov[Va, Ob, oa, Vb], R.abb.VVvOOO, optimize=True)
    )
    dR.aab.VVvoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('AmJe,BceimK->ABciJK', H.ab.voov[Va, ob, Oa, vb], R.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('AMJe,BceiMK->ABciJK', H.ab.voov[Va, Ob, Oa, vb], R.abb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('AmJE,BEcimK->ABciJK', H.ab.voov[Va, ob, Oa, Vb], R.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('AMJE,BEciMK->ABciJK', H.ab.voov[Va, Ob, Oa, Vb], R.abb.VVvoOO, optimize=True)
    )
    dR.aab.VVvoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mceK,BAeimJ->ABciJK', H.ab.ovvo[oa, vb, va, Ob], R.aaa.VVvooO, optimize=True)
            - 1.0 * np.einsum('MceK,BAeiJM->ABciJK', H.ab.ovvo[Oa, vb, va, Ob], R.aaa.VVvoOO, optimize=True)
            + 1.0 * np.einsum('mcEK,EBAimJ->ABciJK', H.ab.ovvo[oa, vb, Va, Ob], R.aaa.VVVooO, optimize=True)
            - 1.0 * np.einsum('McEK,EBAiJM->ABciJK', H.ab.ovvo[Oa, vb, Va, Ob], R.aaa.VVVoOO, optimize=True)
    )
    dR.aab.VVvoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('cmKe,BAeiJm->ABciJK', H.bb.voov[vb, ob, Ob, vb], R.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('cMKe,BAeiJM->ABciJK', H.bb.voov[vb, Ob, Ob, vb], R.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('cmKE,BAEiJm->ABciJK', H.bb.voov[vb, ob, Ob, Vb], R.aab.VVVoOo, optimize=True)
            - 1.0 * np.einsum('cMKE,BAEiJM->ABciJK', H.bb.voov[vb, Ob, Ob, Vb], R.aab.VVVoOO, optimize=True)
    )
    dR.aab.VVvoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('AmeK,BeciJm->ABciJK', H.ab.vovo[Va, ob, va, Ob], R.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('AMeK,BeciJM->ABciJK', H.ab.vovo[Va, Ob, va, Ob], R.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('AmEK,BEciJm->ABciJK', H.ab.vovo[Va, ob, Va, Ob], R.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('AMEK,BEciJM->ABciJK', H.ab.vovo[Va, Ob, Va, Ob], R.aab.VVvoOO, optimize=True)
    )
    dR.aab.VVvoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mcie,BAemJK->ABciJK', H.ab.ovov[oa, vb, oa, vb], R.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('Mcie,BAeMJK->ABciJK', H.ab.ovov[Oa, vb, oa, vb], R.aab.VVvOOO, optimize=True)
            + 1.0 * np.einsum('mciE,BAEmJK->ABciJK', H.ab.ovov[oa, vb, oa, Vb], R.aab.VVVoOO, optimize=True)
            + 1.0 * np.einsum('MciE,BAEMJK->ABciJK', H.ab.ovov[Oa, vb, oa, Vb], R.aab.VVVOOO, optimize=True)
    )
    dR.aab.VVvoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mcJe,BAemiK->ABciJK', H.ab.ovov[oa, vb, Oa, vb], R.aab.VVvooO, optimize=True)
            + 1.0 * np.einsum('McJe,BAeiMK->ABciJK', H.ab.ovov[Oa, vb, Oa, vb], R.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('mcJE,BAEmiK->ABciJK', H.ab.ovov[oa, vb, Oa, Vb], R.aab.VVVooO, optimize=True)
            + 1.0 * np.einsum('McJE,BAEiMK->ABciJK', H.ab.ovov[Oa, vb, Oa, Vb], R.aab.VVVoOO, optimize=True)
    )
    # of terms =  38

    dR.aab.VVvoOO -= np.transpose(dR.aab.VVvoOO, (1, 0, 2, 3, 4, 5))

    return dR

def update(R, omega, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    R.aab.VVvoOO = eomcc_active_loops.update_r3b_110011(
        R.aab.VVvoOO,
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
