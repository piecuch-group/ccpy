import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops

def build(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.abb.VVvooO = (1.0 / 1.0) * (
            +1.0 * np.einsum('ABie,ecjK->ABcijK', X.ab.vvov[Va, Vb, oa, :], T.bb[:, vb, ob, Ob], optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('Acie,eBjK->ABcijK', X.ab.vvov[Va, vb, oa, :], T.bb[:, Vb, ob, Ob], optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('Amij,BcmK->ABcijK', X.ab.vooo[Va, :, oa, ob], T.bb[Vb, vb, :, Ob], optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('AmiK,Bcmj->ABcijK', X.ab.vooo[Va, :, oa, Ob], T.bb[Vb, vb, :, ob], optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('cBKe,Aeij->ABcijK', X.bb.vvov[vb, Vb, Ob, :], T.ab[Va, :, oa, ob], optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('cBje,AeiK->ABcijK', X.bb.vvov[vb, Vb, ob, :], T.ab[Va, :, oa, Ob], optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('cmKj,ABim->ABcijK', X.bb.vooo[vb, :, Ob, ob], T.ab[Va, Vb, oa, :], optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('BmKj,Acim->ABcijK', X.bb.vooo[Vb, :, Ob, ob], T.ab[Va, vb, oa, :], optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('ABej,eciK->ABcijK', X.ab.vvvo[Va, Vb, :, ob], T.ab[:, vb, oa, Ob], optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('Acej,eBiK->ABcijK', X.ab.vvvo[Va, vb, :, ob], T.ab[:, Vb, oa, Ob], optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('ABeK,ecij->ABcijK', X.ab.vvvo[Va, Vb, :, Ob], T.ab[:, vb, oa, ob], optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('AceK,eBij->ABcijK', X.ab.vvvo[Va, vb, :, Ob], T.ab[:, Vb, oa, ob], optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mBij,AcmK->ABcijK', X.ab.ovoo[:, Vb, oa, ob], T.ab[Va, vb, :, Ob], optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mcij,ABmK->ABcijK', X.ab.ovoo[:, vb, oa, ob], T.ab[Va, Vb, :, Ob], optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mBiK,Acmj->ABcijK', X.ab.ovoo[:, Vb, oa, Ob], T.ab[Va, vb, :, ob], optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mciK,ABmj->ABcijK', X.ab.ovoo[:, vb, oa, Ob], T.ab[Va, Vb, :, ob], optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('ABie,ecjK->ABcijK', H.ab.vvov[Va, Vb, oa, :], R.bb[:, vb, ob, Ob], optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('Acie,eBjK->ABcijK', H.ab.vvov[Va, vb, oa, :], R.bb[:, Vb, ob, Ob], optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('Amij,BcmK->ABcijK', H.ab.vooo[Va, :, oa, ob], R.bb[Vb, vb, :, Ob], optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('AmiK,Bcmj->ABcijK', H.ab.vooo[Va, :, oa, Ob], R.bb[Vb, vb, :, ob], optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('cBKe,Aeij->ABcijK', H.bb.vvov[vb, Vb, Ob, :], R.ab[Va, :, oa, ob], optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('cBje,AeiK->ABcijK', H.bb.vvov[vb, Vb, ob, :], R.ab[Va, :, oa, Ob], optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('cmKj,ABim->ABcijK', H.bb.vooo[vb, :, Ob, ob], R.ab[Va, Vb, oa, :], optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('BmKj,Acim->ABcijK', H.bb.vooo[Vb, :, Ob, ob], R.ab[Va, vb, oa, :], optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('ABej,eciK->ABcijK', H.ab.vvvo[Va, Vb, :, ob], R.ab[:, vb, oa, Ob], optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('Acej,eBiK->ABcijK', H.ab.vvvo[Va, vb, :, ob], R.ab[:, Vb, oa, Ob], optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('ABeK,ecij->ABcijK', H.ab.vvvo[Va, Vb, :, Ob], R.ab[:, vb, oa, ob], optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('AceK,eBij->ABcijK', H.ab.vvvo[Va, vb, :, Ob], R.ab[:, Vb, oa, ob], optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mBij,AcmK->ABcijK', H.ab.ovoo[:, Vb, oa, ob], R.ab[Va, vb, :, Ob], optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mcij,ABmK->ABcijK', H.ab.ovoo[:, vb, oa, ob], R.ab[Va, Vb, :, Ob], optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mBiK,Acmj->ABcijK', H.ab.ovoo[:, Vb, oa, Ob], R.ab[Va, vb, :, ob], optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mciK,ABmj->ABcijK', H.ab.ovoo[:, vb, oa, Ob], R.ab[Va, Vb, :, ob], optimize=True)
    )
    # of terms =  32
    dR.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mi,ABcmjK->ABcijK', X.a.oo[oa, oa], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('Mi,ABcMjK->ABcijK', X.a.oo[Oa, oa], T.abb.VVvOoO, optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mj,ABcimK->ABcijK', X.b.oo[ob, ob], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('Mj,ABciMK->ABcijK', X.b.oo[Ob, ob], T.abb.VVvoOO, optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('MK,ABcijM->ABcijK', X.b.oo[Ob, Ob], T.abb.VVvooO, optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('Ae,eBcijK->ABcijK', X.a.vv[Va, va], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('AE,EBcijK->ABcijK', X.a.vv[Va, Va], T.abb.VVvooO, optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('Be,AceijK->ABcijK', X.b.vv[Vb, vb], T.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('BE,AEcijK->ABcijK', X.b.vv[Vb, Vb], T.abb.VVvooO, optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('ce,ABeijK->ABcijK', X.b.vv[vb, vb], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('cE,ABEijK->ABcijK', X.b.vv[vb, Vb], T.abb.VVVooO, optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('MnjK,ABcinM->ABcijK', X.bb.oooo[Ob, ob, ob, Ob], T.abb.VVvooO, optimize=True)
            + 0.5 * np.einsum('MNjK,ABciMN->ABcijK', X.bb.oooo[Ob, Ob, ob, Ob], T.abb.VVvoOO, optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mnij,ABcmnK->ABcijK', X.ab.oooo[oa, ob, oa, ob], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('mNij,ABcmNK->ABcijK', X.ab.oooo[oa, Ob, oa, ob], T.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('Mnij,ABcMnK->ABcijK', X.ab.oooo[Oa, ob, oa, ob], T.abb.VVvOoO, optimize=True)
            + 1.0 * np.einsum('MNij,ABcMNK->ABcijK', X.ab.oooo[Oa, Ob, oa, ob], T.abb.VVvOOO, optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mNiK,ABcmjN->ABcijK', X.ab.oooo[oa, Ob, oa, Ob], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('MniK,ABcMnj->ABcijK', X.ab.oooo[Oa, ob, oa, Ob], T.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('MNiK,ABcMjN->ABcijK', X.ab.oooo[Oa, Ob, oa, Ob], T.abb.VVvOoO, optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            -0.5 * np.einsum('Bcef,AfeijK->ABcijK', X.bb.vvvv[Vb, vb, vb, vb], T.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('BcEf,AEfijK->ABcijK', X.bb.vvvv[Vb, vb, Vb, vb], T.abb.VVvooO, optimize=True)
            - 0.5 * np.einsum('BcEF,AFEijK->ABcijK', X.bb.vvvv[Vb, vb, Vb, Vb], T.abb.VVVooO, optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('ABeF,eFcijK->ABcijK', X.ab.vvvv[Va, Vb, va, Vb], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('ABEf,EcfijK->ABcijK', X.ab.vvvv[Va, Vb, Va, vb], T.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('ABEF,EFcijK->ABcijK', X.ab.vvvv[Va, Vb, Va, Vb], T.abb.VVvooO, optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('Acef,eBfijK->ABcijK', X.ab.vvvv[Va, vb, va, vb], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('AceF,eBFijK->ABcijK', X.ab.vvvv[Va, vb, va, Vb], T.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('AcEf,EBfijK->ABcijK', X.ab.vvvv[Va, vb, Va, vb], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('AcEF,EBFijK->ABcijK', X.ab.vvvv[Va, vb, Va, Vb], T.abb.VVVooO, optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('Amie,eBcmjK->ABcijK', X.aa.voov[Va, oa, oa, va], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('AmiE,EBcmjK->ABcijK', X.aa.voov[Va, oa, oa, Va], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('AMie,eBcMjK->ABcijK', X.aa.voov[Va, Oa, oa, va], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('AMiE,EBcMjK->ABcijK', X.aa.voov[Va, Oa, oa, Va], T.abb.VVvOoO, optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('Amie,BcemjK->ABcijK', X.ab.voov[Va, ob, oa, vb], T.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('AmiE,BEcmjK->ABcijK', X.ab.voov[Va, ob, oa, Vb], T.bbb.VVvooO, optimize=True)
            - 1.0 * np.einsum('AMie,BcejMK->ABcijK', X.ab.voov[Va, Ob, oa, vb], T.bbb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('AMiE,BEcjMK->ABcijK', X.ab.voov[Va, Ob, oa, Vb], T.bbb.VVvoOO, optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mBej,AecimK->ABcijK', X.ab.ovvo[oa, Vb, va, ob], T.aab.VvvooO, optimize=True)
            - 1.0 * np.einsum('mBEj,EAcimK->ABcijK', X.ab.ovvo[oa, Vb, Va, ob], T.aab.VVvooO, optimize=True)
            + 1.0 * np.einsum('MBej,AeciMK->ABcijK', X.ab.ovvo[Oa, Vb, va, ob], T.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('MBEj,EAciMK->ABcijK', X.ab.ovvo[Oa, Vb, Va, ob], T.aab.VVvoOO, optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mcej,AeBimK->ABcijK', X.ab.ovvo[oa, vb, va, ob], T.aab.VvVooO, optimize=True)
            + 1.0 * np.einsum('mcEj,EABimK->ABcijK', X.ab.ovvo[oa, vb, Va, ob], T.aab.VVVooO, optimize=True)
            - 1.0 * np.einsum('Mcej,AeBiMK->ABcijK', X.ab.ovvo[Oa, vb, va, ob], T.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('McEj,EABiMK->ABcijK', X.ab.ovvo[Oa, vb, Va, ob], T.aab.VVVoOO, optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('MBeK,AeciMj->ABcijK', X.ab.ovvo[Oa, Vb, va, Ob], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('MBEK,EAciMj->ABcijK', X.ab.ovvo[Oa, Vb, Va, Ob], T.aab.VVvoOo, optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('MceK,AeBiMj->ABcijK', X.ab.ovvo[Oa, vb, va, Ob], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('McEK,EABiMj->ABcijK', X.ab.ovvo[Oa, vb, Va, Ob], T.aab.VVVoOo, optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('Bmje,AceimK->ABcijK', X.bb.voov[Vb, ob, ob, vb], T.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('BmjE,AEcimK->ABcijK', X.bb.voov[Vb, ob, ob, Vb], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('BMje,AceiMK->ABcijK', X.bb.voov[Vb, Ob, ob, vb], T.abb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('BMjE,AEciMK->ABcijK', X.bb.voov[Vb, Ob, ob, Vb], T.abb.VVvoOO, optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('cmje,ABeimK->ABcijK', X.bb.voov[vb, ob, ob, vb], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('cmjE,ABEimK->ABcijK', X.bb.voov[vb, ob, ob, Vb], T.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('cMje,ABeiMK->ABcijK', X.bb.voov[vb, Ob, ob, vb], T.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('cMjE,ABEiMK->ABcijK', X.bb.voov[vb, Ob, ob, Vb], T.abb.VVVoOO, optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('BMKe,AceijM->ABcijK', X.bb.voov[Vb, Ob, Ob, vb], T.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('BMKE,AEcijM->ABcijK', X.bb.voov[Vb, Ob, Ob, Vb], T.abb.VVvooO, optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('cMKe,ABeijM->ABcijK', X.bb.voov[vb, Ob, Ob, vb], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('cMKE,ABEijM->ABcijK', X.bb.voov[vb, Ob, Ob, Vb], T.abb.VVVooO, optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mBie,AcemjK->ABcijK', X.ab.ovov[oa, Vb, oa, vb], T.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('mBiE,AEcmjK->ABcijK', X.ab.ovov[oa, Vb, oa, Vb], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('MBie,AceMjK->ABcijK', X.ab.ovov[Oa, Vb, oa, vb], T.abb.VvvOoO, optimize=True)
            - 1.0 * np.einsum('MBiE,AEcMjK->ABcijK', X.ab.ovov[Oa, Vb, oa, Vb], T.abb.VVvOoO, optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mcie,ABemjK->ABcijK', X.ab.ovov[oa, vb, oa, vb], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('mciE,ABEmjK->ABcijK', X.ab.ovov[oa, vb, oa, Vb], T.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('Mcie,ABeMjK->ABcijK', X.ab.ovov[Oa, vb, oa, vb], T.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('MciE,ABEMjK->ABcijK', X.ab.ovov[Oa, vb, oa, Vb], T.abb.VVVOoO, optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('Amej,eBcimK->ABcijK', X.ab.vovo[Va, ob, va, ob], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('AmEj,EBcimK->ABcijK', X.ab.vovo[Va, ob, Va, ob], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('AMej,eBciMK->ABcijK', X.ab.vovo[Va, Ob, va, ob], T.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('AMEj,EBciMK->ABcijK', X.ab.vovo[Va, Ob, Va, ob], T.abb.VVvoOO, optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('AMeK,eBcijM->ABcijK', X.ab.vovo[Va, Ob, va, Ob], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('AMEK,EBcijM->ABcijK', X.ab.vovo[Va, Ob, Va, Ob], T.abb.VVvooO, optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mi,ABcmjK->ABcijK', H.a.oo[oa, oa], R.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('Mi,ABcMjK->ABcijK', H.a.oo[Oa, oa], R.abb.VVvOoO, optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mj,ABcimK->ABcijK', H.b.oo[ob, ob], R.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('Mj,ABciMK->ABcijK', H.b.oo[Ob, ob], R.abb.VVvoOO, optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('MK,ABcijM->ABcijK', H.b.oo[Ob, Ob], R.abb.VVvooO, optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('Ae,eBcijK->ABcijK', H.a.vv[Va, va], R.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('AE,EBcijK->ABcijK', H.a.vv[Va, Va], R.abb.VVvooO, optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('Be,AceijK->ABcijK', H.b.vv[Vb, vb], R.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('BE,AEcijK->ABcijK', H.b.vv[Vb, Vb], R.abb.VVvooO, optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('ce,ABeijK->ABcijK', H.b.vv[vb, vb], R.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('cE,ABEijK->ABcijK', H.b.vv[vb, Vb], R.abb.VVVooO, optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('MnjK,ABcinM->ABcijK', H.bb.oooo[Ob, ob, ob, Ob], R.abb.VVvooO, optimize=True)
            + 0.5 * np.einsum('MNjK,ABciMN->ABcijK', H.bb.oooo[Ob, Ob, ob, Ob], R.abb.VVvoOO, optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mnij,ABcmnK->ABcijK', H.ab.oooo[oa, ob, oa, ob], R.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('mNij,ABcmNK->ABcijK', H.ab.oooo[oa, Ob, oa, ob], R.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('Mnij,ABcMnK->ABcijK', H.ab.oooo[Oa, ob, oa, ob], R.abb.VVvOoO, optimize=True)
            + 1.0 * np.einsum('MNij,ABcMNK->ABcijK', H.ab.oooo[Oa, Ob, oa, ob], R.abb.VVvOOO, optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mNiK,ABcmjN->ABcijK', H.ab.oooo[oa, Ob, oa, Ob], R.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('MniK,ABcMnj->ABcijK', H.ab.oooo[Oa, ob, oa, Ob], R.abb.VVvOoo, optimize=True)
            + 1.0 * np.einsum('MNiK,ABcMjN->ABcijK', H.ab.oooo[Oa, Ob, oa, Ob], R.abb.VVvOoO, optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            -0.5 * np.einsum('Bcef,AfeijK->ABcijK', H.bb.vvvv[Vb, vb, vb, vb], R.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('BcEf,AEfijK->ABcijK', H.bb.vvvv[Vb, vb, Vb, vb], R.abb.VVvooO, optimize=True)
            - 0.5 * np.einsum('BcEF,AFEijK->ABcijK', H.bb.vvvv[Vb, vb, Vb, Vb], R.abb.VVVooO, optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('ABeF,eFcijK->ABcijK', H.ab.vvvv[Va, Vb, va, Vb], R.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('ABEf,EcfijK->ABcijK', H.ab.vvvv[Va, Vb, Va, vb], R.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('ABEF,EFcijK->ABcijK', H.ab.vvvv[Va, Vb, Va, Vb], R.abb.VVvooO, optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('Acef,eBfijK->ABcijK', H.ab.vvvv[Va, vb, va, vb], R.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('AceF,eBFijK->ABcijK', H.ab.vvvv[Va, vb, va, Vb], R.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('AcEf,EBfijK->ABcijK', H.ab.vvvv[Va, vb, Va, vb], R.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('AcEF,EBFijK->ABcijK', H.ab.vvvv[Va, vb, Va, Vb], R.abb.VVVooO, optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('Amie,eBcmjK->ABcijK', H.aa.voov[Va, oa, oa, va], R.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('AmiE,EBcmjK->ABcijK', H.aa.voov[Va, oa, oa, Va], R.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('AMie,eBcMjK->ABcijK', H.aa.voov[Va, Oa, oa, va], R.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('AMiE,EBcMjK->ABcijK', H.aa.voov[Va, Oa, oa, Va], R.abb.VVvOoO, optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('Amie,BcemjK->ABcijK', H.ab.voov[Va, ob, oa, vb], R.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('AmiE,BEcmjK->ABcijK', H.ab.voov[Va, ob, oa, Vb], R.bbb.VVvooO, optimize=True)
            - 1.0 * np.einsum('AMie,BcejMK->ABcijK', H.ab.voov[Va, Ob, oa, vb], R.bbb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('AMiE,BEcjMK->ABcijK', H.ab.voov[Va, Ob, oa, Vb], R.bbb.VVvoOO, optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mBej,AecimK->ABcijK', H.ab.ovvo[oa, Vb, va, ob], R.aab.VvvooO, optimize=True)
            - 1.0 * np.einsum('mBEj,EAcimK->ABcijK', H.ab.ovvo[oa, Vb, Va, ob], R.aab.VVvooO, optimize=True)
            + 1.0 * np.einsum('MBej,AeciMK->ABcijK', H.ab.ovvo[Oa, Vb, va, ob], R.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('MBEj,EAciMK->ABcijK', H.ab.ovvo[Oa, Vb, Va, ob], R.aab.VVvoOO, optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mcej,AeBimK->ABcijK', H.ab.ovvo[oa, vb, va, ob], R.aab.VvVooO, optimize=True)
            + 1.0 * np.einsum('mcEj,EABimK->ABcijK', H.ab.ovvo[oa, vb, Va, ob], R.aab.VVVooO, optimize=True)
            - 1.0 * np.einsum('Mcej,AeBiMK->ABcijK', H.ab.ovvo[Oa, vb, va, ob], R.aab.VvVoOO, optimize=True)
            + 1.0 * np.einsum('McEj,EABiMK->ABcijK', H.ab.ovvo[Oa, vb, Va, ob], R.aab.VVVoOO, optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('MBeK,AeciMj->ABcijK', H.ab.ovvo[Oa, Vb, va, Ob], R.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('MBEK,EAciMj->ABcijK', H.ab.ovvo[Oa, Vb, Va, Ob], R.aab.VVvoOo, optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('MceK,AeBiMj->ABcijK', H.ab.ovvo[Oa, vb, va, Ob], R.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('McEK,EABiMj->ABcijK', H.ab.ovvo[Oa, vb, Va, Ob], R.aab.VVVoOo, optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('Bmje,AceimK->ABcijK', H.bb.voov[Vb, ob, ob, vb], R.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('BmjE,AEcimK->ABcijK', H.bb.voov[Vb, ob, ob, Vb], R.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('BMje,AceiMK->ABcijK', H.bb.voov[Vb, Ob, ob, vb], R.abb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('BMjE,AEciMK->ABcijK', H.bb.voov[Vb, Ob, ob, Vb], R.abb.VVvoOO, optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('cmje,ABeimK->ABcijK', H.bb.voov[vb, ob, ob, vb], R.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('cmjE,ABEimK->ABcijK', H.bb.voov[vb, ob, ob, Vb], R.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('cMje,ABeiMK->ABcijK', H.bb.voov[vb, Ob, ob, vb], R.abb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('cMjE,ABEiMK->ABcijK', H.bb.voov[vb, Ob, ob, Vb], R.abb.VVVoOO, optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('BMKe,AceijM->ABcijK', H.bb.voov[Vb, Ob, Ob, vb], R.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('BMKE,AEcijM->ABcijK', H.bb.voov[Vb, Ob, Ob, Vb], R.abb.VVvooO, optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('cMKe,ABeijM->ABcijK', H.bb.voov[vb, Ob, Ob, vb], R.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('cMKE,ABEijM->ABcijK', H.bb.voov[vb, Ob, Ob, Vb], R.abb.VVVooO, optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mBie,AcemjK->ABcijK', H.ab.ovov[oa, Vb, oa, vb], R.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('mBiE,AEcmjK->ABcijK', H.ab.ovov[oa, Vb, oa, Vb], R.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('MBie,AceMjK->ABcijK', H.ab.ovov[Oa, Vb, oa, vb], R.abb.VvvOoO, optimize=True)
            - 1.0 * np.einsum('MBiE,AEcMjK->ABcijK', H.ab.ovov[Oa, Vb, oa, Vb], R.abb.VVvOoO, optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mcie,ABemjK->ABcijK', H.ab.ovov[oa, vb, oa, vb], R.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('mciE,ABEmjK->ABcijK', H.ab.ovov[oa, vb, oa, Vb], R.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('Mcie,ABeMjK->ABcijK', H.ab.ovov[Oa, vb, oa, vb], R.abb.VVvOoO, optimize=True)
            - 1.0 * np.einsum('MciE,ABEMjK->ABcijK', H.ab.ovov[Oa, vb, oa, Vb], R.abb.VVVOoO, optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('Amej,eBcimK->ABcijK', H.ab.vovo[Va, ob, va, ob], R.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('AmEj,EBcimK->ABcijK', H.ab.vovo[Va, ob, Va, ob], R.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('AMej,eBciMK->ABcijK', H.ab.vovo[Va, Ob, va, ob], R.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('AMEj,EBciMK->ABcijK', H.ab.vovo[Va, Ob, Va, ob], R.abb.VVvoOO, optimize=True)
    )
    dR.abb.VVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('AMeK,eBcijM->ABcijK', H.ab.vovo[Va, Ob, va, Ob], R.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('AMEK,EBcijM->ABcijK', H.ab.vovo[Va, Ob, Va, Ob], R.abb.VVvooO, optimize=True)
    )
    # of terms =  52

    return dR

def update(R, omega, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    R.abb.VVvooO = eomcc_active_loops.update_r3c_110001(
        R.abb.VVvooO,
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
