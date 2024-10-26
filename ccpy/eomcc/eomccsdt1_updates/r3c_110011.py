import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops

def build(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.abb.VVvoOO = (1.0 / 2.0) * (
            +1.0 * np.einsum('ABie,ecJK->ABciJK', X.ab.vvov[Va, Vb, oa, :], T.bb[:, vb, Ob, Ob], optimize=True)
    )
    dR.abb.VVvoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('Acie,eBJK->ABciJK', X.ab.vvov[Va, vb, oa, :], T.bb[:, Vb, Ob, Ob], optimize=True)
    )
    dR.abb.VVvoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('AmiJ,BcmK->ABciJK', X.ab.vooo[Va, :, oa, Ob], T.bb[Vb, vb, :, Ob], optimize=True)
    )
    dR.abb.VVvoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('cBKe,AeiJ->ABciJK', X.bb.vvov[vb, Vb, Ob, :], T.ab[Va, :, oa, Ob], optimize=True)
    )
    dR.abb.VVvoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('cmKJ,ABim->ABciJK', X.bb.vooo[vb, :, Ob, Ob], T.ab[Va, Vb, oa, :], optimize=True)
    )
    dR.abb.VVvoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('BmKJ,Acim->ABciJK', X.bb.vooo[Vb, :, Ob, Ob], T.ab[Va, vb, oa, :], optimize=True)
    )
    dR.abb.VVvoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('ABeJ,eciK->ABciJK', X.ab.vvvo[Va, Vb, :, Ob], T.ab[:, vb, oa, Ob], optimize=True)
    )
    dR.abb.VVvoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('AceJ,eBiK->ABciJK', X.ab.vvvo[Va, vb, :, Ob], T.ab[:, Vb, oa, Ob], optimize=True)
    )
    dR.abb.VVvoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mBiJ,AcmK->ABciJK', X.ab.ovoo[:, Vb, oa, Ob], T.ab[Va, vb, :, Ob], optimize=True)
    )
    dR.abb.VVvoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mciJ,ABmK->ABciJK', X.ab.ovoo[:, vb, oa, Ob], T.ab[Va, Vb, :, Ob], optimize=True)
    )
    dR.abb.VVvoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('ABie,ecJK->ABciJK', H.ab.vvov[Va, Vb, oa, :], R.bb[:, vb, Ob, Ob], optimize=True)
    )
    dR.abb.VVvoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('Acie,eBJK->ABciJK', H.ab.vvov[Va, vb, oa, :], R.bb[:, Vb, Ob, Ob], optimize=True)
    )
    dR.abb.VVvoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('AmiJ,BcmK->ABciJK', H.ab.vooo[Va, :, oa, Ob], R.bb[Vb, vb, :, Ob], optimize=True)
    )
    dR.abb.VVvoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('cBKe,AeiJ->ABciJK', H.bb.vvov[vb, Vb, Ob, :], R.ab[Va, :, oa, Ob], optimize=True)
    )
    dR.abb.VVvoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('cmKJ,ABim->ABciJK', H.bb.vooo[vb, :, Ob, Ob], R.ab[Va, Vb, oa, :], optimize=True)
    )
    dR.abb.VVvoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('BmKJ,Acim->ABciJK', H.bb.vooo[Vb, :, Ob, Ob], R.ab[Va, vb, oa, :], optimize=True)
    )
    dR.abb.VVvoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('ABeJ,eciK->ABciJK', H.ab.vvvo[Va, Vb, :, Ob], R.ab[:, vb, oa, Ob], optimize=True)
    )
    dR.abb.VVvoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('AceJ,eBiK->ABciJK', H.ab.vvvo[Va, vb, :, Ob], R.ab[:, Vb, oa, Ob], optimize=True)
    )
    dR.abb.VVvoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mBiJ,AcmK->ABciJK', H.ab.ovoo[:, Vb, oa, Ob], R.ab[Va, vb, :, Ob], optimize=True)
    )
    dR.abb.VVvoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mciJ,ABmK->ABciJK', H.ab.ovoo[:, vb, oa, Ob], R.ab[Va, Vb, :, Ob], optimize=True)
    )
    # of terms =  20
    dR.abb.VVvoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mi,ABcmJK->ABciJK', X.a.oo[oa, oa], T.abb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('Mi,ABcMJK->ABciJK', X.a.oo[Oa, oa], T.abb.VVvOOO, optimize=True)
    )
    dR.abb.VVvoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mJ,ABcimK->ABciJK', X.b.oo[ob, Ob], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('MJ,ABciMK->ABciJK', X.b.oo[Ob, Ob], T.abb.VVvoOO, optimize=True)
    )
    dR.abb.VVvoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('Ae,eBciJK->ABciJK', X.a.vv[Va, va], T.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('AE,EBciJK->ABciJK', X.a.vv[Va, Va], T.abb.VVvoOO, optimize=True)
    )
    dR.abb.VVvoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('Be,AceiJK->ABciJK', X.b.vv[Vb, vb], T.abb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('BE,AEciJK->ABciJK', X.b.vv[Vb, Vb], T.abb.VVvoOO, optimize=True)
    )
    dR.abb.VVvoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('ce,ABeiJK->ABciJK', X.b.vv[vb, vb], T.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('cE,ABEiJK->ABciJK', X.b.vv[vb, Vb], T.abb.VVVoOO, optimize=True)
    )
    dR.abb.VVvoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('MnJK,ABcinM->ABciJK', X.bb.oooo[Ob, ob, Ob, Ob], T.abb.VVvooO, optimize=True)
            + 0.5 * np.einsum('MNJK,ABciMN->ABciJK', X.bb.oooo[Ob, Ob, Ob, Ob], T.abb.VVvoOO, optimize=True)
    )
    dR.abb.VVvoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mniJ,ABcmnK->ABciJK', X.ab.oooo[oa, ob, oa, Ob], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('mNiJ,ABcmNK->ABciJK', X.ab.oooo[oa, Ob, oa, Ob], T.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('MniJ,ABcMnK->ABciJK', X.ab.oooo[Oa, ob, oa, Ob], T.abb.VVvOoO, optimize=True)
            + 1.0 * np.einsum('MNiJ,ABcMNK->ABciJK', X.ab.oooo[Oa, Ob, oa, Ob], T.abb.VVvOOO, optimize=True)
    )
    dR.abb.VVvoOO += (1.0 / 2.0) * (
            -0.5 * np.einsum('Bcef,AfeiJK->ABciJK', X.bb.vvvv[Vb, vb, vb, vb], T.abb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('BcEf,AEfiJK->ABciJK', X.bb.vvvv[Vb, vb, Vb, vb], T.abb.VVvoOO, optimize=True)
            - 0.5 * np.einsum('BcEF,AFEiJK->ABciJK', X.bb.vvvv[Vb, vb, Vb, Vb], T.abb.VVVoOO, optimize=True)
    )
    dR.abb.VVvoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('ABeF,eFciJK->ABciJK', X.ab.vvvv[Va, Vb, va, Vb], T.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('ABEf,EcfiJK->ABciJK', X.ab.vvvv[Va, Vb, Va, vb], T.abb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('ABEF,EFciJK->ABciJK', X.ab.vvvv[Va, Vb, Va, Vb], T.abb.VVvoOO, optimize=True)
    )
    dR.abb.VVvoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('Acef,eBfiJK->ABciJK', X.ab.vvvv[Va, vb, va, vb], T.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('AceF,eBFiJK->ABciJK', X.ab.vvvv[Va, vb, va, Vb], T.abb.vVVoOO, optimize=True)
            + 1.0 * np.einsum('AcEf,EBfiJK->ABciJK', X.ab.vvvv[Va, vb, Va, vb], T.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('AcEF,EBFiJK->ABciJK', X.ab.vvvv[Va, vb, Va, Vb], T.abb.VVVoOO, optimize=True)
    )
    dR.abb.VVvoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('Amie,eBcmJK->ABciJK', X.aa.voov[Va, oa, oa, va], T.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('AMie,eBcMJK->ABciJK', X.aa.voov[Va, Oa, oa, va], T.abb.vVvOOO, optimize=True)
            + 1.0 * np.einsum('AmiE,EBcmJK->ABciJK', X.aa.voov[Va, oa, oa, Va], T.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('AMiE,EBcMJK->ABciJK', X.aa.voov[Va, Oa, oa, Va], T.abb.VVvOOO, optimize=True)
    )
    dR.abb.VVvoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('Amie,BcemJK->ABciJK', X.ab.voov[Va, ob, oa, vb], T.bbb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('AMie,BceMJK->ABciJK', X.ab.voov[Va, Ob, oa, vb], T.bbb.VvvOOO, optimize=True)
            - 1.0 * np.einsum('AmiE,BEcmJK->ABciJK', X.ab.voov[Va, ob, oa, Vb], T.bbb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('AMiE,BEcMJK->ABciJK', X.ab.voov[Va, Ob, oa, Vb], T.bbb.VVvOOO, optimize=True)
    )
    dR.abb.VVvoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mBeJ,AecimK->ABciJK', X.ab.ovvo[oa, Vb, va, Ob], T.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('MBeJ,AeciMK->ABciJK', X.ab.ovvo[Oa, Vb, va, Ob], T.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('mBEJ,EAcimK->ABciJK', X.ab.ovvo[oa, Vb, Va, Ob], T.aab.VVvooO, optimize=True)
            - 1.0 * np.einsum('MBEJ,EAciMK->ABciJK', X.ab.ovvo[Oa, Vb, Va, Ob], T.aab.VVvoOO, optimize=True)
    )
    dR.abb.VVvoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mceJ,AeBimK->ABciJK', X.ab.ovvo[oa, vb, va, Ob], T.aab.VvVooO, optimize=True)
            - 1.0 * np.einsum('MceJ,AeBiMK->ABciJK', X.ab.ovvo[Oa, vb, va, Ob], T.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('mcEJ,EABimK->ABciJK', X.ab.ovvo[oa, vb, Va, Ob], T.aab.VVVooO, optimize=True)
            + 1.0 * np.einsum('McEJ,EABiMK->ABciJK', X.ab.ovvo[Oa, vb, Va, Ob], T.aab.VVVoOO, optimize=True)
    )
    dR.abb.VVvoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('BmJe,AceimK->ABciJK', X.bb.voov[Vb, ob, Ob, vb], T.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('BMJe,AceiMK->ABciJK', X.bb.voov[Vb, Ob, Ob, vb], T.abb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('BmJE,AEcimK->ABciJK', X.bb.voov[Vb, ob, Ob, Vb], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('BMJE,AEciMK->ABciJK', X.bb.voov[Vb, Ob, Ob, Vb], T.abb.VVvoOO, optimize=True)
    )
    dR.abb.VVvoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('cmJe,ABeimK->ABciJK', X.bb.voov[vb, ob, Ob, vb], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('cMJe,ABeiMK->ABciJK', X.bb.voov[vb, Ob, Ob, vb], T.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('cmJE,ABEimK->ABciJK', X.bb.voov[vb, ob, Ob, Vb], T.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('cMJE,ABEiMK->ABciJK', X.bb.voov[vb, Ob, Ob, Vb], T.abb.VVVoOO, optimize=True)
    )
    dR.abb.VVvoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mBie,AcemJK->ABciJK', X.ab.ovov[oa, Vb, oa, vb], T.abb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('MBie,AceMJK->ABciJK', X.ab.ovov[Oa, Vb, oa, vb], T.abb.VvvOOO, optimize=True)
            - 1.0 * np.einsum('mBiE,AEcmJK->ABciJK', X.ab.ovov[oa, Vb, oa, Vb], T.abb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('MBiE,AEcMJK->ABciJK', X.ab.ovov[Oa, Vb, oa, Vb], T.abb.VVvOOO, optimize=True)
    )
    dR.abb.VVvoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mcie,ABemJK->ABciJK', X.ab.ovov[oa, vb, oa, vb], T.abb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('Mcie,ABeMJK->ABciJK', X.ab.ovov[Oa, vb, oa, vb], T.abb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('mciE,ABEmJK->ABciJK', X.ab.ovov[oa, vb, oa, Vb], T.abb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('MciE,ABEMJK->ABciJK', X.ab.ovov[Oa, vb, oa, Vb], T.abb.VVVOOO, optimize=True)
    )
    dR.abb.VVvoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('AmeJ,eBcimK->ABciJK', X.ab.vovo[Va, ob, va, Ob], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('AMeJ,eBciMK->ABciJK', X.ab.vovo[Va, Ob, va, Ob], T.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('AmEJ,EBcimK->ABciJK', X.ab.vovo[Va, ob, Va, Ob], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('AMEJ,EBciMK->ABciJK', X.ab.vovo[Va, Ob, Va, Ob], T.abb.VVvoOO, optimize=True)
    )
    dR.abb.VVvoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mi,ABcmJK->ABciJK', H.a.oo[oa, oa], R.abb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('Mi,ABcMJK->ABciJK', H.a.oo[Oa, oa], R.abb.VVvOOO, optimize=True)
    )
    dR.abb.VVvoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mJ,ABcimK->ABciJK', H.b.oo[ob, Ob], R.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('MJ,ABciMK->ABciJK', H.b.oo[Ob, Ob], R.abb.VVvoOO, optimize=True)
    )
    dR.abb.VVvoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('Ae,eBciJK->ABciJK', H.a.vv[Va, va], R.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('AE,EBciJK->ABciJK', H.a.vv[Va, Va], R.abb.VVvoOO, optimize=True)
    )
    dR.abb.VVvoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('Be,AceiJK->ABciJK', H.b.vv[Vb, vb], R.abb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('BE,AEciJK->ABciJK', H.b.vv[Vb, Vb], R.abb.VVvoOO, optimize=True)
    )
    dR.abb.VVvoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('ce,ABeiJK->ABciJK', H.b.vv[vb, vb], R.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('cE,ABEiJK->ABciJK', H.b.vv[vb, Vb], R.abb.VVVoOO, optimize=True)
    )
    dR.abb.VVvoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('MnJK,ABcinM->ABciJK', H.bb.oooo[Ob, ob, Ob, Ob], R.abb.VVvooO, optimize=True)
            + 0.5 * np.einsum('MNJK,ABciMN->ABciJK', H.bb.oooo[Ob, Ob, Ob, Ob], R.abb.VVvoOO, optimize=True)
    )
    dR.abb.VVvoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mniJ,ABcmnK->ABciJK', H.ab.oooo[oa, ob, oa, Ob], R.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('mNiJ,ABcmNK->ABciJK', H.ab.oooo[oa, Ob, oa, Ob], R.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('MniJ,ABcMnK->ABciJK', H.ab.oooo[Oa, ob, oa, Ob], R.abb.VVvOoO, optimize=True)
            + 1.0 * np.einsum('MNiJ,ABcMNK->ABciJK', H.ab.oooo[Oa, Ob, oa, Ob], R.abb.VVvOOO, optimize=True)
    )
    dR.abb.VVvoOO += (1.0 / 2.0) * (
            -0.5 * np.einsum('Bcef,AfeiJK->ABciJK', H.bb.vvvv[Vb, vb, vb, vb], R.abb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('BcEf,AEfiJK->ABciJK', H.bb.vvvv[Vb, vb, Vb, vb], R.abb.VVvoOO, optimize=True)
            - 0.5 * np.einsum('BcEF,AFEiJK->ABciJK', H.bb.vvvv[Vb, vb, Vb, Vb], R.abb.VVVoOO, optimize=True)
    )
    dR.abb.VVvoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('ABeF,eFciJK->ABciJK', H.ab.vvvv[Va, Vb, va, Vb], R.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('ABEf,EcfiJK->ABciJK', H.ab.vvvv[Va, Vb, Va, vb], R.abb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('ABEF,EFciJK->ABciJK', H.ab.vvvv[Va, Vb, Va, Vb], R.abb.VVvoOO, optimize=True)
    )
    dR.abb.VVvoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('Acef,eBfiJK->ABciJK', H.ab.vvvv[Va, vb, va, vb], R.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('AceF,eBFiJK->ABciJK', H.ab.vvvv[Va, vb, va, Vb], R.abb.vVVoOO, optimize=True)
            + 1.0 * np.einsum('AcEf,EBfiJK->ABciJK', H.ab.vvvv[Va, vb, Va, vb], R.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('AcEF,EBFiJK->ABciJK', H.ab.vvvv[Va, vb, Va, Vb], R.abb.VVVoOO, optimize=True)
    )
    dR.abb.VVvoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('Amie,eBcmJK->ABciJK', H.aa.voov[Va, oa, oa, va], R.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('AMie,eBcMJK->ABciJK', H.aa.voov[Va, Oa, oa, va], R.abb.vVvOOO, optimize=True)
            + 1.0 * np.einsum('AmiE,EBcmJK->ABciJK', H.aa.voov[Va, oa, oa, Va], R.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('AMiE,EBcMJK->ABciJK', H.aa.voov[Va, Oa, oa, Va], R.abb.VVvOOO, optimize=True)
    )
    dR.abb.VVvoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('Amie,BcemJK->ABciJK', H.ab.voov[Va, ob, oa, vb], R.bbb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('AMie,BceMJK->ABciJK', H.ab.voov[Va, Ob, oa, vb], R.bbb.VvvOOO, optimize=True)
            - 1.0 * np.einsum('AmiE,BEcmJK->ABciJK', H.ab.voov[Va, ob, oa, Vb], R.bbb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('AMiE,BEcMJK->ABciJK', H.ab.voov[Va, Ob, oa, Vb], R.bbb.VVvOOO, optimize=True)
    )
    dR.abb.VVvoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mBeJ,AecimK->ABciJK', H.ab.ovvo[oa, Vb, va, Ob], R.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('MBeJ,AeciMK->ABciJK', H.ab.ovvo[Oa, Vb, va, Ob], R.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('mBEJ,EAcimK->ABciJK', H.ab.ovvo[oa, Vb, Va, Ob], R.aab.VVvooO, optimize=True)
            - 1.0 * np.einsum('MBEJ,EAciMK->ABciJK', H.ab.ovvo[Oa, Vb, Va, Ob], R.aab.VVvoOO, optimize=True)
    )
    dR.abb.VVvoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mceJ,AeBimK->ABciJK', H.ab.ovvo[oa, vb, va, Ob], R.aab.VvVooO, optimize=True)
            - 1.0 * np.einsum('MceJ,AeBiMK->ABciJK', H.ab.ovvo[Oa, vb, va, Ob], R.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('mcEJ,EABimK->ABciJK', H.ab.ovvo[oa, vb, Va, Ob], R.aab.VVVooO, optimize=True)
            + 1.0 * np.einsum('McEJ,EABiMK->ABciJK', H.ab.ovvo[Oa, vb, Va, Ob], R.aab.VVVoOO, optimize=True)
    )
    dR.abb.VVvoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('BmJe,AceimK->ABciJK', H.bb.voov[Vb, ob, Ob, vb], R.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('BMJe,AceiMK->ABciJK', H.bb.voov[Vb, Ob, Ob, vb], R.abb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('BmJE,AEcimK->ABciJK', H.bb.voov[Vb, ob, Ob, Vb], R.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('BMJE,AEciMK->ABciJK', H.bb.voov[Vb, Ob, Ob, Vb], R.abb.VVvoOO, optimize=True)
    )
    dR.abb.VVvoOO += (2.0 / 2.0) * (
            +1.0 * np.einsum('cmJe,ABeimK->ABciJK', H.bb.voov[vb, ob, Ob, vb], R.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('cMJe,ABeiMK->ABciJK', H.bb.voov[vb, Ob, Ob, vb], R.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('cmJE,ABEimK->ABciJK', H.bb.voov[vb, ob, Ob, Vb], R.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('cMJE,ABEiMK->ABciJK', H.bb.voov[vb, Ob, Ob, Vb], R.abb.VVVoOO, optimize=True)
    )
    dR.abb.VVvoOO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mBie,AcemJK->ABciJK', H.ab.ovov[oa, Vb, oa, vb], R.abb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('MBie,AceMJK->ABciJK', H.ab.ovov[Oa, Vb, oa, vb], R.abb.VvvOOO, optimize=True)
            - 1.0 * np.einsum('mBiE,AEcmJK->ABciJK', H.ab.ovov[oa, Vb, oa, Vb], R.abb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('MBiE,AEcMJK->ABciJK', H.ab.ovov[Oa, Vb, oa, Vb], R.abb.VVvOOO, optimize=True)
    )
    dR.abb.VVvoOO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mcie,ABemJK->ABciJK', H.ab.ovov[oa, vb, oa, vb], R.abb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('Mcie,ABeMJK->ABciJK', H.ab.ovov[Oa, vb, oa, vb], R.abb.VVvOOO, optimize=True)
            - 1.0 * np.einsum('mciE,ABEmJK->ABciJK', H.ab.ovov[oa, vb, oa, Vb], R.abb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('MciE,ABEMJK->ABciJK', H.ab.ovov[Oa, vb, oa, Vb], R.abb.VVVOOO, optimize=True)
    )
    dR.abb.VVvoOO += (2.0 / 2.0) * (
            -1.0 * np.einsum('AmeJ,eBcimK->ABciJK', H.ab.vovo[Va, ob, va, Ob], R.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('AMeJ,eBciMK->ABciJK', H.ab.vovo[Va, Ob, va, Ob], R.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('AmEJ,EBcimK->ABciJK', H.ab.vovo[Va, ob, Va, Ob], R.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('AMEJ,EBciMK->ABciJK', H.ab.vovo[Va, Ob, Va, Ob], R.abb.VVvoOO, optimize=True)
    )
    # of terms =  38

    dR.abb.VVvoOO -= np.transpose(dR.abb.VVvoOO, (0, 1, 2, 3, 5, 4))

    return dR

def update(R, omega, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    R.abb.VVvoOO = eomcc_active_loops.update_r3c_110011(
        R.abb.VVvoOO,
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
