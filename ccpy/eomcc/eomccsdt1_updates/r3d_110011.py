import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops

def build(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.bbb.VVvoOO = (4.0 / 4.0) * (
            -1.0 * np.einsum('BmJi,AcmK->ABciJK', X.bb.vooo[Vb, :, Ob, ob], T.bb[Vb, vb, :, Ob], optimize=True)
    )
    dR.bbb.VVvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('cmJi,ABmK->ABciJK', X.bb.vooo[vb, :, Ob, ob], T.bb[Vb, Vb, :, Ob], optimize=True)
    )
    dR.bbb.VVvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('BmJK,Acmi->ABciJK', X.bb.vooo[Vb, :, Ob, Ob], T.bb[Vb, vb, :, ob], optimize=True)
    )
    dR.bbb.VVvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('cmJK,ABmi->ABciJK', X.bb.vooo[vb, :, Ob, Ob], T.bb[Vb, Vb, :, ob], optimize=True)
    )
    dR.bbb.VVvoOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('BmJi,AcmK->ABciJK', H.bb.vooo[Vb, :, Ob, ob], R.bb[Vb, vb, :, Ob], optimize=True)
    )
    dR.bbb.VVvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('cmJi,ABmK->ABciJK', H.bb.vooo[vb, :, Ob, ob], R.bb[Vb, Vb, :, Ob], optimize=True)
    )
    dR.bbb.VVvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('BmJK,Acmi->ABciJK', H.bb.vooo[Vb, :, Ob, Ob], R.bb[Vb, vb, :, ob], optimize=True)
    )
    dR.bbb.VVvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('cmJK,ABmi->ABciJK', H.bb.vooo[vb, :, Ob, Ob], R.bb[Vb, Vb, :, ob], optimize=True)
    )
    dR.bbb.VVvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('BAJe,eciK->ABciJK', X.bb.vvov[Vb, Vb, Ob, :], T.bb[:, vb, ob, Ob], optimize=True)
    )
    dR.bbb.VVvoOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('BcJe,eAiK->ABciJK', X.bb.vvov[Vb, vb, Ob, :], T.bb[:, Vb, ob, Ob], optimize=True)
    )
    dR.bbb.VVvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('BAie,ecJK->ABciJK', X.bb.vvov[Vb, Vb, ob, :], T.bb[:, vb, Ob, Ob], optimize=True)
    )
    dR.bbb.VVvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('Bcie,eAJK->ABciJK', X.bb.vvov[Vb, vb, ob, :], T.bb[:, Vb, Ob, Ob], optimize=True)
    )
    dR.bbb.VVvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('BAJe,eciK->ABciJK', H.bb.vvov[Vb, Vb, Ob, :], R.bb[:, vb, ob, Ob], optimize=True)
    )
    dR.bbb.VVvoOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('BcJe,eAiK->ABciJK', H.bb.vvov[Vb, vb, Ob, :], R.bb[:, Vb, ob, Ob], optimize=True)
    )
    dR.bbb.VVvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('BAie,ecJK->ABciJK', H.bb.vvov[Vb, Vb, ob, :], R.bb[:, vb, Ob, Ob], optimize=True)
    )
    dR.bbb.VVvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('Bcie,eAJK->ABciJK', H.bb.vvov[Vb, vb, ob, :], R.bb[:, Vb, Ob, Ob], optimize=True)
    )
    # of terms =  16
    dR.bbb.VVvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('Be,AceiJK->ABciJK', X.b.vv[Vb, vb], T.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('BE,EAciJK->ABciJK', X.b.vv[Vb, Vb], T.bbb.VVvoOO, optimize=True)
    )
    dR.bbb.VVvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('ce,BAeiJK->ABciJK', X.b.vv[vb, vb], T.bbb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('cE,BEAiJK->ABciJK', X.b.vv[vb, Vb], T.bbb.VVVoOO, optimize=True)
    )
    dR.bbb.VVvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mJ,BAcimK->ABciJK', X.b.oo[ob, Ob], T.bbb.VVvooO, optimize=True)
            + 1.0 * np.einsum('MJ,BAciMK->ABciJK', X.b.oo[Ob, Ob], T.bbb.VVvoOO, optimize=True)
    )
    dR.bbb.VVvoOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('mi,BAcmJK->ABciJK', X.b.oo[ob, ob], T.bbb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('Mi,BAcJMK->ABciJK', X.b.oo[Ob, ob], T.bbb.VVvOOO, optimize=True)
    )
    dR.bbb.VVvoOO += (2.0 / 4.0) * (
            -0.5 * np.einsum('mniJ,BAcmnK->ABciJK', X.bb.oooo[ob, ob, ob, Ob], T.bbb.VVvooO, optimize=True)
            - 1.0 * np.einsum('mNiJ,BAcmNK->ABciJK', X.bb.oooo[ob, Ob, ob, Ob], T.bbb.VVvoOO, optimize=True)
            - 0.5 * np.einsum('MNiJ,BAcMNK->ABciJK', X.bb.oooo[Ob, Ob, ob, Ob], T.bbb.VVvOOO, optimize=True)
    )
    dR.bbb.VVvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('mNKJ,BAcmiN->ABciJK', X.bb.oooo[ob, Ob, Ob, Ob], T.bbb.VVvooO, optimize=True)
            + 0.5 * np.einsum('MNKJ,BAciMN->ABciJK', X.bb.oooo[Ob, Ob, Ob, Ob], T.bbb.VVvoOO, optimize=True)
    )
    dR.bbb.VVvoOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('ABeF,FceiJK->ABciJK', X.bb.vvvv[Vb, Vb, vb, Vb], T.bbb.VvvoOO, optimize=True)
            - 0.5 * np.einsum('ABEF,FEciJK->ABciJK', X.bb.vvvv[Vb, Vb, Vb, Vb], T.bbb.VVvoOO, optimize=True)
    )
    dR.bbb.VVvoOO += (2.0 / 4.0) * (
            +0.5 * np.einsum('cBef,AfeiJK->ABciJK', X.bb.vvvv[vb, Vb, vb, vb], T.bbb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('cBeF,AFeiJK->ABciJK', X.bb.vvvv[vb, Vb, vb, Vb], T.bbb.VVvoOO, optimize=True)
            + 0.5 * np.einsum('cBEF,AFEiJK->ABciJK', X.bb.vvvv[vb, Vb, Vb, Vb], T.bbb.VVVoOO, optimize=True)
    )
    dR.bbb.VVvoOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('BmJe,AceimK->ABciJK', X.bb.voov[Vb, ob, Ob, vb], T.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('BMJe,AceiMK->ABciJK', X.bb.voov[Vb, Ob, Ob, vb], T.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('BmJE,EAcimK->ABciJK', X.bb.voov[Vb, ob, Ob, Vb], T.bbb.VVvooO, optimize=True)
            - 1.0 * np.einsum('BMJE,EAciMK->ABciJK', X.bb.voov[Vb, Ob, Ob, Vb], T.bbb.VVvoOO, optimize=True)
    )
    dR.bbb.VVvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('cmJe,BAeimK->ABciJK', X.bb.voov[vb, ob, Ob, vb], T.bbb.VVvooO, optimize=True)
            - 1.0 * np.einsum('cMJe,BAeiMK->ABciJK', X.bb.voov[vb, Ob, Ob, vb], T.bbb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('cmJE,BEAimK->ABciJK', X.bb.voov[vb, ob, Ob, Vb], T.bbb.VVVooO, optimize=True)
            + 1.0 * np.einsum('cMJE,BEAiMK->ABciJK', X.bb.voov[vb, Ob, Ob, Vb], T.bbb.VVVoOO, optimize=True)
    )
    dR.bbb.VVvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('Bmie,AcemJK->ABciJK', X.bb.voov[Vb, ob, ob, vb], T.bbb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('BMie,AceJMK->ABciJK', X.bb.voov[Vb, Ob, ob, vb], T.bbb.VvvOOO, optimize=True)
            - 1.0 * np.einsum('BmiE,EAcmJK->ABciJK', X.bb.voov[Vb, ob, ob, Vb], T.bbb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('BMiE,EAcJMK->ABciJK', X.bb.voov[Vb, Ob, ob, Vb], T.bbb.VVvOOO, optimize=True)
    )
    dR.bbb.VVvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('cmie,BAemJK->ABciJK', X.bb.voov[vb, ob, ob, vb], T.bbb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('cMie,BAeJMK->ABciJK', X.bb.voov[vb, Ob, ob, vb], T.bbb.VVvOOO, optimize=True)
            + 1.0 * np.einsum('cmiE,BEAmJK->ABciJK', X.bb.voov[vb, ob, ob, Vb], T.bbb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('cMiE,BEAJMK->ABciJK', X.bb.voov[vb, Ob, ob, Vb], T.bbb.VVVOOO, optimize=True)
    )
    dR.bbb.VVvoOO += (4.0 / 4.0) * (
            +1.0 * np.einsum('mBeJ,eAcmiK->ABciJK', X.ab.ovvo[oa, Vb, va, Ob], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('MBeJ,eAcMiK->ABciJK', X.ab.ovvo[Oa, Vb, va, Ob], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('mBEJ,EAcmiK->ABciJK', X.ab.ovvo[oa, Vb, Va, Ob], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('MBEJ,EAcMiK->ABciJK', X.ab.ovvo[Oa, Vb, Va, Ob], T.abb.VVvOoO, optimize=True)
    )
    dR.bbb.VVvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('mceJ,eABmiK->ABciJK', X.ab.ovvo[oa, vb, va, Ob], T.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('MceJ,eABMiK->ABciJK', X.ab.ovvo[Oa, vb, va, Ob], T.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('mcEJ,EABmiK->ABciJK', X.ab.ovvo[oa, vb, Va, Ob], T.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('McEJ,EABMiK->ABciJK', X.ab.ovvo[Oa, vb, Va, Ob], T.abb.VVVOoO, optimize=True)
    )
    dR.bbb.VVvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mBei,eAcmKJ->ABciJK', X.ab.ovvo[oa, Vb, va, ob], T.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('MBei,eAcMKJ->ABciJK', X.ab.ovvo[Oa, Vb, va, ob], T.abb.vVvOOO, optimize=True)
            + 1.0 * np.einsum('mBEi,EAcmKJ->ABciJK', X.ab.ovvo[oa, Vb, Va, ob], T.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('MBEi,EAcMKJ->ABciJK', X.ab.ovvo[Oa, Vb, Va, ob], T.abb.VVvOOO, optimize=True)
    )
    dR.bbb.VVvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('mcei,eABmKJ->ABciJK', X.ab.ovvo[oa, vb, va, ob], T.abb.vVVoOO, optimize=True)
            - 1.0 * np.einsum('Mcei,eABMKJ->ABciJK', X.ab.ovvo[Oa, vb, va, ob], T.abb.vVVOOO, optimize=True)
            - 1.0 * np.einsum('mcEi,EABmKJ->ABciJK', X.ab.ovvo[oa, vb, Va, ob], T.abb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('McEi,EABMKJ->ABciJK', X.ab.ovvo[Oa, vb, Va, ob], T.abb.VVVOOO, optimize=True)
    )
    dR.bbb.VVvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mJ,BAcimK->ABciJK', H.b.oo[ob, Ob], R.bbb.VVvooO, optimize=True)
            + 1.0 * np.einsum('MJ,BAciMK->ABciJK', H.b.oo[Ob, Ob], R.bbb.VVvoOO, optimize=True)
    )
    dR.bbb.VVvoOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('mi,BAcmJK->ABciJK', H.b.oo[ob, ob], R.bbb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('Mi,BAcJMK->ABciJK', H.b.oo[Ob, ob], R.bbb.VVvOOO, optimize=True)
    )
    dR.bbb.VVvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('Be,AceiJK->ABciJK', H.b.vv[Vb, vb], R.bbb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('BE,EAciJK->ABciJK', H.b.vv[Vb, Vb], R.bbb.VVvoOO, optimize=True)
    )
    dR.bbb.VVvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('ce,BAeiJK->ABciJK', H.b.vv[vb, vb], R.bbb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('cE,BEAiJK->ABciJK', H.b.vv[vb, Vb], R.bbb.VVVoOO, optimize=True)
    )
    dR.bbb.VVvoOO += (2.0 / 4.0) * (
            -0.5 * np.einsum('mniJ,BAcmnK->ABciJK', H.bb.oooo[ob, ob, ob, Ob], R.bbb.VVvooO, optimize=True)
            - 1.0 * np.einsum('mNiJ,BAcmNK->ABciJK', H.bb.oooo[ob, Ob, ob, Ob], R.bbb.VVvoOO, optimize=True)
            - 0.5 * np.einsum('MNiJ,BAcMNK->ABciJK', H.bb.oooo[Ob, Ob, ob, Ob], R.bbb.VVvOOO, optimize=True)
    )
    dR.bbb.VVvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('mNKJ,BAcmiN->ABciJK', H.bb.oooo[ob, Ob, Ob, Ob], R.bbb.VVvooO, optimize=True)
            + 0.5 * np.einsum('MNKJ,BAciMN->ABciJK', H.bb.oooo[Ob, Ob, Ob, Ob], R.bbb.VVvoOO, optimize=True)
    )
    dR.bbb.VVvoOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('ABeF,FceiJK->ABciJK', H.bb.vvvv[Vb, Vb, vb, Vb], R.bbb.VvvoOO, optimize=True)
            - 0.5 * np.einsum('ABEF,FEciJK->ABciJK', H.bb.vvvv[Vb, Vb, Vb, Vb], R.bbb.VVvoOO, optimize=True)
    )
    dR.bbb.VVvoOO += (2.0 / 4.0) * (
            +0.5 * np.einsum('cBef,AfeiJK->ABciJK', H.bb.vvvv[vb, Vb, vb, vb], R.bbb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('cBeF,AFeiJK->ABciJK', H.bb.vvvv[vb, Vb, vb, Vb], R.bbb.VVvoOO, optimize=True)
            + 0.5 * np.einsum('cBEF,AFEiJK->ABciJK', H.bb.vvvv[vb, Vb, Vb, Vb], R.bbb.VVVoOO, optimize=True)
    )
    dR.bbb.VVvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('Amie,BcemJK->ABciJK', H.bb.voov[Vb, ob, ob, vb], R.bbb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('AMie,BceMJK->ABciJK', H.bb.voov[Vb, Ob, ob, vb], R.bbb.VvvOOO, optimize=True)
            - 1.0 * np.einsum('AmiE,BEcmJK->ABciJK', H.bb.voov[Vb, ob, ob, Vb], R.bbb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('AMiE,BEcMJK->ABciJK', H.bb.voov[Vb, Ob, ob, Vb], R.bbb.VVvOOO, optimize=True)
    )
    dR.bbb.VVvoOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('cmie,ABemJK->ABciJK', H.bb.voov[vb, ob, ob, vb], R.bbb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('cMie,ABeMJK->ABciJK', H.bb.voov[vb, Ob, ob, vb], R.bbb.VVvOOO, optimize=True)
            + 1.0 * np.einsum('cmiE,ABEmJK->ABciJK', H.bb.voov[vb, ob, ob, Vb], R.bbb.VVVoOO, optimize=True)
            + 1.0 * np.einsum('cMiE,ABEMJK->ABciJK', H.bb.voov[vb, Ob, ob, Vb], R.bbb.VVVOOO, optimize=True)
    )
    dR.bbb.VVvoOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('AmJe,BcemiK->ABciJK', H.bb.voov[Vb, ob, Ob, vb], R.bbb.VvvooO, optimize=True)
            + 1.0 * np.einsum('AMJe,BceiMK->ABciJK', H.bb.voov[Vb, Ob, Ob, vb], R.bbb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('AmJE,BEcmiK->ABciJK', H.bb.voov[Vb, ob, Ob, Vb], R.bbb.VVvooO, optimize=True)
            - 1.0 * np.einsum('AMJE,BEciMK->ABciJK', H.bb.voov[Vb, Ob, Ob, Vb], R.bbb.VVvoOO, optimize=True)
    )
    dR.bbb.VVvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('cmJe,ABemiK->ABciJK', H.bb.voov[vb, ob, Ob, vb], R.bbb.VVvooO, optimize=True)
            + 1.0 * np.einsum('cMJe,ABeiMK->ABciJK', H.bb.voov[vb, Ob, Ob, vb], R.bbb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('cmJE,ABEmiK->ABciJK', H.bb.voov[vb, ob, Ob, Vb], R.bbb.VVVooO, optimize=True)
            + 1.0 * np.einsum('cMJE,ABEiMK->ABciJK', H.bb.voov[vb, Ob, Ob, Vb], R.bbb.VVVoOO, optimize=True)
    )
    dR.bbb.VVvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('mAei,eBcmKJ->ABciJK', H.ab.ovvo[oa, Vb, va, ob], R.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('MAei,eBcMKJ->ABciJK', H.ab.ovvo[Oa, Vb, va, ob], R.abb.vVvOOO, optimize=True)
            - 1.0 * np.einsum('mAEi,EBcmKJ->ABciJK', H.ab.ovvo[oa, Vb, Va, ob], R.abb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('MAEi,EBcMKJ->ABciJK', H.ab.ovvo[Oa, Vb, Va, ob], R.abb.VVvOOO, optimize=True)
    )
    dR.bbb.VVvoOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('mcei,eBAmKJ->ABciJK', H.ab.ovvo[oa, vb, va, ob], R.abb.vVVoOO, optimize=True)
            + 1.0 * np.einsum('Mcei,eBAMKJ->ABciJK', H.ab.ovvo[Oa, vb, va, ob], R.abb.vVVOOO, optimize=True)
            + 1.0 * np.einsum('mcEi,EBAmKJ->ABciJK', H.ab.ovvo[oa, vb, Va, ob], R.abb.VVVoOO, optimize=True)
            + 1.0 * np.einsum('McEi,EBAMKJ->ABciJK', H.ab.ovvo[Oa, vb, Va, ob], R.abb.VVVOOO, optimize=True)
    )
    dR.bbb.VVvoOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('mAeJ,eBcmiK->ABciJK', H.ab.ovvo[oa, Vb, va, Ob], R.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('MAeJ,eBcMiK->ABciJK', H.ab.ovvo[Oa, Vb, va, Ob], R.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('mAEJ,EBcmiK->ABciJK', H.ab.ovvo[oa, Vb, Va, Ob], R.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('MAEJ,EBcMiK->ABciJK', H.ab.ovvo[Oa, Vb, Va, Ob], R.abb.VVvOoO, optimize=True)
    )
    dR.bbb.VVvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mceJ,eBAmiK->ABciJK', H.ab.ovvo[oa, vb, va, Ob], R.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('MceJ,eBAMiK->ABciJK', H.ab.ovvo[Oa, vb, va, Ob], R.abb.vVVOoO, optimize=True)
            + 1.0 * np.einsum('mcEJ,EBAmiK->ABciJK', H.ab.ovvo[oa, vb, Va, Ob], R.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('McEJ,EBAMiK->ABciJK', H.ab.ovvo[Oa, vb, Va, Ob], R.abb.VVVOoO, optimize=True)
    )
    # of terms =  32

    dR.bbb.VVvoOO -= np.transpose(dR.bbb.VVvoOO, (0, 1, 2, 3, 5, 4))
    dR.bbb.VVvoOO -= np.transpose(dR.bbb.VVvoOO, (1, 0, 2, 3, 4, 5))

    return dR

def update(R, omega, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    R.bbb.VVvoOO = eomcc_active_loops.update_r3d_110011(
        R.bbb.VVvoOO,
        omega,
        H.b.oo[Ob, Ob],
        H.b.vv[Vb, Vb],
        H.b.oo[ob, ob],
        H.b.vv[vb, vb],
        0.0,
    )
    return R
