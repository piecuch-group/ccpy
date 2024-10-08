import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import eomcc_active_loops

def build(dR, R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dR.abb.vVvooO = (1.0 / 1.0) * (
            +1.0 * np.einsum('aBie,ecjK->aBcijK', X.ab.vvov[va, Vb, oa, :], T.bb[:, vb, ob, Ob], optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('acie,eBjK->aBcijK', X.ab.vvov[va, vb, oa, :], T.bb[:, Vb, ob, Ob], optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('amij,BcmK->aBcijK', X.ab.vooo[va, :, oa, ob], T.bb[Vb, vb, :, Ob], optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('amiK,Bcmj->aBcijK', X.ab.vooo[va, :, oa, Ob], T.bb[Vb, vb, :, ob], optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('cBKe,aeij->aBcijK', X.bb.vvov[vb, Vb, Ob, :], T.ab[va, :, oa, ob], optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('cBje,aeiK->aBcijK', X.bb.vvov[vb, Vb, ob, :], T.ab[va, :, oa, Ob], optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('cmKj,aBim->aBcijK', X.bb.vooo[vb, :, Ob, ob], T.ab[va, Vb, oa, :], optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('BmKj,acim->aBcijK', X.bb.vooo[Vb, :, Ob, ob], T.ab[va, vb, oa, :], optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('aBej,eciK->aBcijK', X.ab.vvvo[va, Vb, :, ob], T.ab[:, vb, oa, Ob], optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('acej,eBiK->aBcijK', X.ab.vvvo[va, vb, :, ob], T.ab[:, Vb, oa, Ob], optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('aBeK,ecij->aBcijK', X.ab.vvvo[va, Vb, :, Ob], T.ab[:, vb, oa, ob], optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('aceK,eBij->aBcijK', X.ab.vvvo[va, vb, :, Ob], T.ab[:, Vb, oa, ob], optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mBij,acmK->aBcijK', X.ab.ovoo[:, Vb, oa, ob], T.ab[va, vb, :, Ob], optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mcij,aBmK->aBcijK', X.ab.ovoo[:, vb, oa, ob], T.ab[va, Vb, :, Ob], optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mBiK,acmj->aBcijK', X.ab.ovoo[:, Vb, oa, Ob], T.ab[va, vb, :, ob], optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mciK,aBmj->aBcijK', X.ab.ovoo[:, vb, oa, Ob], T.ab[va, Vb, :, ob], optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('aBie,ecjK->aBcijK', H.ab.vvov[va, Vb, oa, :], R.bb[:, vb, ob, Ob], optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('acie,eBjK->aBcijK', H.ab.vvov[va, vb, oa, :], R.bb[:, Vb, ob, Ob], optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('amij,BcmK->aBcijK', H.ab.vooo[va, :, oa, ob], R.bb[Vb, vb, :, Ob], optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('amiK,Bcmj->aBcijK', H.ab.vooo[va, :, oa, Ob], R.bb[Vb, vb, :, ob], optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('cBKe,aeij->aBcijK', H.bb.vvov[vb, Vb, Ob, :], R.ab[va, :, oa, ob], optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('cBje,aeiK->aBcijK', H.bb.vvov[vb, Vb, ob, :], R.ab[va, :, oa, Ob], optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('cmKj,aBim->aBcijK', H.bb.vooo[vb, :, Ob, ob], R.ab[va, Vb, oa, :], optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('BmKj,acim->aBcijK', H.bb.vooo[Vb, :, Ob, ob], R.ab[va, vb, oa, :], optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('aBej,eciK->aBcijK', H.ab.vvvo[va, Vb, :, ob], R.ab[:, vb, oa, Ob], optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('acej,eBiK->aBcijK', H.ab.vvvo[va, vb, :, ob], R.ab[:, Vb, oa, Ob], optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('aBeK,ecij->aBcijK', H.ab.vvvo[va, Vb, :, Ob], R.ab[:, vb, oa, ob], optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('aceK,eBij->aBcijK', H.ab.vvvo[va, vb, :, Ob], R.ab[:, Vb, oa, ob], optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mBij,acmK->aBcijK', H.ab.ovoo[:, Vb, oa, ob], R.ab[va, vb, :, Ob], optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mcij,aBmK->aBcijK', H.ab.ovoo[:, vb, oa, ob], R.ab[va, Vb, :, Ob], optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mBiK,acmj->aBcijK', H.ab.ovoo[:, Vb, oa, Ob], R.ab[va, vb, :, ob], optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mciK,aBmj->aBcijK', H.ab.ovoo[:, vb, oa, Ob], R.ab[va, Vb, :, ob], optimize=True)
    )
    # of terms =  32
    dR.abb.vVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mi,aBcmjK->aBcijK', X.a.oo[oa, oa], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('Mi,aBcMjK->aBcijK', X.a.oo[Oa, oa], T.abb.vVvOoO, optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mj,aBcimK->aBcijK', X.b.oo[ob, ob], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('Mj,aBciMK->aBcijK', X.b.oo[Ob, ob], T.abb.vVvoOO, optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('MK,aBcijM->aBcijK', X.b.oo[Ob, Ob], T.abb.vVvooO, optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('ae,eBcijK->aBcijK', X.a.vv[va, va], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('aE,EBcijK->aBcijK', X.a.vv[va, Va], T.abb.VVvooO, optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('BE,aEcijK->aBcijK', X.b.vv[Vb, Vb], T.abb.vVvooO, optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('ce,aBeijK->aBcijK', X.b.vv[vb, vb], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('cE,aBEijK->aBcijK', X.b.vv[vb, Vb], T.abb.vVVooO, optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('MnjK,aBcinM->aBcijK', X.bb.oooo[Ob, ob, ob, Ob], T.abb.vVvooO, optimize=True)
            + 0.5 * np.einsum('MNjK,aBciMN->aBcijK', X.bb.oooo[Ob, Ob, ob, Ob], T.abb.vVvoOO, optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mnij,aBcmnK->aBcijK', X.ab.oooo[oa, ob, oa, ob], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('mNij,aBcmNK->aBcijK', X.ab.oooo[oa, Ob, oa, ob], T.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('Mnij,aBcMnK->aBcijK', X.ab.oooo[Oa, ob, oa, ob], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('MNij,aBcMNK->aBcijK', X.ab.oooo[Oa, Ob, oa, ob], T.abb.vVvOOO, optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mNiK,aBcmjN->aBcijK', X.ab.oooo[oa, Ob, oa, Ob], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('MniK,aBcMnj->aBcijK', X.ab.oooo[Oa, ob, oa, Ob], T.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('MNiK,aBcMjN->aBcijK', X.ab.oooo[Oa, Ob, oa, Ob], T.abb.vVvOoO, optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('BcEf,aEfijK->aBcijK', X.bb.vvvv[Vb, vb, Vb, vb], T.abb.vVvooO, optimize=True)
            - 0.5 * np.einsum('BcEF,aFEijK->aBcijK', X.bb.vvvv[Vb, vb, Vb, Vb], T.abb.vVVooO, optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('aBeF,eFcijK->aBcijK', X.ab.vvvv[va, Vb, va, Vb], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('aBEf,EcfijK->aBcijK', X.ab.vvvv[va, Vb, Va, vb], T.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('aBEF,EFcijK->aBcijK', X.ab.vvvv[va, Vb, Va, Vb], T.abb.VVvooO, optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('acef,eBfijK->aBcijK', X.ab.vvvv[va, vb, va, vb], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('aceF,eBFijK->aBcijK', X.ab.vvvv[va, vb, va, Vb], T.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('acEf,EBfijK->aBcijK', X.ab.vvvv[va, vb, Va, vb], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('acEF,EBFijK->aBcijK', X.ab.vvvv[va, vb, Va, Vb], T.abb.VVVooO, optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('amie,eBcmjK->aBcijK', X.aa.voov[va, oa, oa, va], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('amiE,EBcmjK->aBcijK', X.aa.voov[va, oa, oa, Va], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('aMie,eBcMjK->aBcijK', X.aa.voov[va, Oa, oa, va], T.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('aMiE,EBcMjK->aBcijK', X.aa.voov[va, Oa, oa, Va], T.abb.VVvOoO, optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('amie,BcemjK->aBcijK', X.ab.voov[va, ob, oa, vb], T.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('amiE,BEcmjK->aBcijK', X.ab.voov[va, ob, oa, Vb], T.bbb.VVvooO, optimize=True)
            - 1.0 * np.einsum('aMie,BcejMK->aBcijK', X.ab.voov[va, Ob, oa, vb], T.bbb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('aMiE,BEcjMK->aBcijK', X.ab.voov[va, Ob, oa, Vb], T.bbb.VVvoOO, optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mBEj,EacimK->aBcijK', X.ab.ovvo[oa, Vb, Va, ob], T.aab.VvvooO, optimize=True)
            - 1.0 * np.einsum('MBEj,EaciMK->aBcijK', X.ab.ovvo[Oa, Vb, Va, ob], T.aab.VvvoOO, optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mcej,eaBimK->aBcijK', X.ab.ovvo[oa, vb, va, ob], T.aab.vvVooO, optimize=True)
            + 1.0 * np.einsum('mcEj,EaBimK->aBcijK', X.ab.ovvo[oa, vb, Va, ob], T.aab.VvVooO, optimize=True)
            + 1.0 * np.einsum('Mcej,eaBiMK->aBcijK', X.ab.ovvo[Oa, vb, va, ob], T.aab.vvVoOO, optimize=True)
            + 1.0 * np.einsum('McEj,EaBiMK->aBcijK', X.ab.ovvo[Oa, vb, Va, ob], T.aab.VvVoOO, optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('MBEK,EaciMj->aBcijK', X.ab.ovvo[Oa, Vb, Va, Ob], T.aab.VvvoOo, optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('MceK,eaBiMj->aBcijK', X.ab.ovvo[Oa, vb, va, Ob], T.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('McEK,EaBiMj->aBcijK', X.ab.ovvo[Oa, vb, Va, Ob], T.aab.VvVoOo, optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('BmjE,aEcimK->aBcijK', X.bb.voov[Vb, ob, ob, Vb], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('BMjE,aEciMK->aBcijK', X.bb.voov[Vb, Ob, ob, Vb], T.abb.vVvoOO, optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('cmje,aBeimK->aBcijK', X.bb.voov[vb, ob, ob, vb], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('cmjE,aBEimK->aBcijK', X.bb.voov[vb, ob, ob, Vb], T.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('cMje,aBeiMK->aBcijK', X.bb.voov[vb, Ob, ob, vb], T.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('cMjE,aBEiMK->aBcijK', X.bb.voov[vb, Ob, ob, Vb], T.abb.vVVoOO, optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('BMKE,aEcijM->aBcijK', X.bb.voov[Vb, Ob, Ob, Vb], T.abb.vVvooO, optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('cMKe,aBeijM->aBcijK', X.bb.voov[vb, Ob, Ob, vb], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('cMKE,aBEijM->aBcijK', X.bb.voov[vb, Ob, Ob, Vb], T.abb.vVVooO, optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mBiE,aEcmjK->aBcijK', X.ab.ovov[oa, Vb, oa, Vb], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('MBiE,aEcMjK->aBcijK', X.ab.ovov[Oa, Vb, oa, Vb], T.abb.vVvOoO, optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mcie,aBemjK->aBcijK', X.ab.ovov[oa, vb, oa, vb], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('mciE,aBEmjK->aBcijK', X.ab.ovov[oa, vb, oa, Vb], T.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('Mcie,aBeMjK->aBcijK', X.ab.ovov[Oa, vb, oa, vb], T.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('MciE,aBEMjK->aBcijK', X.ab.ovov[Oa, vb, oa, Vb], T.abb.vVVOoO, optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('amej,eBcimK->aBcijK', X.ab.vovo[va, ob, va, ob], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('amEj,EBcimK->aBcijK', X.ab.vovo[va, ob, Va, ob], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('aMej,eBciMK->aBcijK', X.ab.vovo[va, Ob, va, ob], T.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('aMEj,EBciMK->aBcijK', X.ab.vovo[va, Ob, Va, ob], T.abb.VVvoOO, optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('aMeK,eBcijM->aBcijK', X.ab.vovo[va, Ob, va, Ob], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('aMEK,EBcijM->aBcijK', X.ab.vovo[va, Ob, Va, Ob], T.abb.VVvooO, optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mi,aBcmjK->aBcijK', H.a.oo[oa, oa], R.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('Mi,aBcMjK->aBcijK', H.a.oo[Oa, oa], R.abb.vVvOoO, optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mj,aBcimK->aBcijK', H.b.oo[ob, ob], R.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('Mj,aBciMK->aBcijK', H.b.oo[Ob, ob], R.abb.vVvoOO, optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('MK,aBcijM->aBcijK', H.b.oo[Ob, Ob], R.abb.vVvooO, optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('ae,eBcijK->aBcijK', H.a.vv[va, va], R.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('aE,EBcijK->aBcijK', H.a.vv[va, Va], R.abb.VVvooO, optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('BE,aEcijK->aBcijK', H.b.vv[Vb, Vb], R.abb.vVvooO, optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('ce,aBeijK->aBcijK', H.b.vv[vb, vb], R.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('cE,aBEijK->aBcijK', H.b.vv[vb, Vb], R.abb.vVVooO, optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('MnjK,aBcinM->aBcijK', H.bb.oooo[Ob, ob, ob, Ob], R.abb.vVvooO, optimize=True)
            + 0.5 * np.einsum('MNjK,aBciMN->aBcijK', H.bb.oooo[Ob, Ob, ob, Ob], R.abb.vVvoOO, optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mnij,aBcmnK->aBcijK', H.ab.oooo[oa, ob, oa, ob], R.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('mNij,aBcmNK->aBcijK', H.ab.oooo[oa, Ob, oa, ob], R.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('Mnij,aBcMnK->aBcijK', H.ab.oooo[Oa, ob, oa, ob], R.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('MNij,aBcMNK->aBcijK', H.ab.oooo[Oa, Ob, oa, ob], R.abb.vVvOOO, optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mNiK,aBcmjN->aBcijK', H.ab.oooo[oa, Ob, oa, Ob], R.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('MniK,aBcMnj->aBcijK', H.ab.oooo[Oa, ob, oa, Ob], R.abb.vVvOoo, optimize=True)
            + 1.0 * np.einsum('MNiK,aBcMjN->aBcijK', H.ab.oooo[Oa, Ob, oa, Ob], R.abb.vVvOoO, optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('BcEf,aEfijK->aBcijK', H.bb.vvvv[Vb, vb, Vb, vb], R.abb.vVvooO, optimize=True)
            - 0.5 * np.einsum('BcEF,aFEijK->aBcijK', H.bb.vvvv[Vb, vb, Vb, Vb], R.abb.vVVooO, optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('aBeF,eFcijK->aBcijK', H.ab.vvvv[va, Vb, va, Vb], R.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('aBEf,EcfijK->aBcijK', H.ab.vvvv[va, Vb, Va, vb], R.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('aBEF,EFcijK->aBcijK', H.ab.vvvv[va, Vb, Va, Vb], R.abb.VVvooO, optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('acef,eBfijK->aBcijK', H.ab.vvvv[va, vb, va, vb], R.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('aceF,eBFijK->aBcijK', H.ab.vvvv[va, vb, va, Vb], R.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('acEf,EBfijK->aBcijK', H.ab.vvvv[va, vb, Va, vb], R.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('acEF,EBFijK->aBcijK', H.ab.vvvv[va, vb, Va, Vb], R.abb.VVVooO, optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('amie,eBcmjK->aBcijK', H.aa.voov[va, oa, oa, va], R.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('amiE,EBcmjK->aBcijK', H.aa.voov[va, oa, oa, Va], R.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('aMie,eBcMjK->aBcijK', H.aa.voov[va, Oa, oa, va], R.abb.vVvOoO, optimize=True)
            + 1.0 * np.einsum('aMiE,EBcMjK->aBcijK', H.aa.voov[va, Oa, oa, Va], R.abb.VVvOoO, optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('amie,BcemjK->aBcijK', H.ab.voov[va, ob, oa, vb], R.bbb.VvvooO, optimize=True)
            - 1.0 * np.einsum('amiE,BEcmjK->aBcijK', H.ab.voov[va, ob, oa, Vb], R.bbb.VVvooO, optimize=True)
            - 1.0 * np.einsum('aMie,BcejMK->aBcijK', H.ab.voov[va, Ob, oa, vb], R.bbb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('aMiE,BEcjMK->aBcijK', H.ab.voov[va, Ob, oa, Vb], R.bbb.VVvoOO, optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mBEj,EacimK->aBcijK', H.ab.ovvo[oa, Vb, Va, ob], R.aab.VvvooO, optimize=True)
            - 1.0 * np.einsum('MBEj,EaciMK->aBcijK', H.ab.ovvo[Oa, Vb, Va, ob], R.aab.VvvoOO, optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('mcej,eaBimK->aBcijK', H.ab.ovvo[oa, vb, va, ob], R.aab.vvVooO, optimize=True)
            + 1.0 * np.einsum('mcEj,EaBimK->aBcijK', H.ab.ovvo[oa, vb, Va, ob], R.aab.VvVooO, optimize=True)
            + 1.0 * np.einsum('Mcej,eaBiMK->aBcijK', H.ab.ovvo[Oa, vb, va, ob], R.aab.vvVoOO, optimize=True)
            + 1.0 * np.einsum('McEj,EaBiMK->aBcijK', H.ab.ovvo[Oa, vb, Va, ob], R.aab.VvVoOO, optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('MBEK,EaciMj->aBcijK', H.ab.ovvo[Oa, Vb, Va, Ob], R.aab.VvvoOo, optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('MceK,eaBiMj->aBcijK', H.ab.ovvo[Oa, vb, va, Ob], R.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('McEK,EaBiMj->aBcijK', H.ab.ovvo[Oa, vb, Va, Ob], R.aab.VvVoOo, optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('BmjE,aEcimK->aBcijK', H.bb.voov[Vb, ob, ob, Vb], R.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('BMjE,aEciMK->aBcijK', H.bb.voov[Vb, Ob, ob, Vb], R.abb.vVvoOO, optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('cmje,aBeimK->aBcijK', H.bb.voov[vb, ob, ob, vb], R.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('cmjE,aBEimK->aBcijK', H.bb.voov[vb, ob, ob, Vb], R.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('cMje,aBeiMK->aBcijK', H.bb.voov[vb, Ob, ob, vb], R.abb.vVvoOO, optimize=True)
            + 1.0 * np.einsum('cMjE,aBEiMK->aBcijK', H.bb.voov[vb, Ob, ob, Vb], R.abb.vVVoOO, optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('BMKE,aEcijM->aBcijK', H.bb.voov[Vb, Ob, Ob, Vb], R.abb.vVvooO, optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            +1.0 * np.einsum('cMKe,aBeijM->aBcijK', H.bb.voov[vb, Ob, Ob, vb], R.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('cMKE,aBEijM->aBcijK', H.bb.voov[vb, Ob, Ob, Vb], R.abb.vVVooO, optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mBiE,aEcmjK->aBcijK', H.ab.ovov[oa, Vb, oa, Vb], R.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('MBiE,aEcMjK->aBcijK', H.ab.ovov[Oa, Vb, oa, Vb], R.abb.vVvOoO, optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('mcie,aBemjK->aBcijK', H.ab.ovov[oa, vb, oa, vb], R.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('mciE,aBEmjK->aBcijK', H.ab.ovov[oa, vb, oa, Vb], R.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('Mcie,aBeMjK->aBcijK', H.ab.ovov[Oa, vb, oa, vb], R.abb.vVvOoO, optimize=True)
            - 1.0 * np.einsum('MciE,aBEMjK->aBcijK', H.ab.ovov[Oa, vb, oa, Vb], R.abb.vVVOoO, optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('amej,eBcimK->aBcijK', H.ab.vovo[va, ob, va, ob], R.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('amEj,EBcimK->aBcijK', H.ab.vovo[va, ob, Va, ob], R.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('aMej,eBciMK->aBcijK', H.ab.vovo[va, Ob, va, ob], R.abb.vVvoOO, optimize=True)
            - 1.0 * np.einsum('aMEj,EBciMK->aBcijK', H.ab.vovo[va, Ob, Va, ob], R.abb.VVvoOO, optimize=True)
    )
    dR.abb.vVvooO += (1.0 / 1.0) * (
            -1.0 * np.einsum('aMeK,eBcijM->aBcijK', H.ab.vovo[va, Ob, va, Ob], R.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('aMEK,EBcijM->aBcijK', H.ab.vovo[va, Ob, Va, Ob], R.abb.VVvooO, optimize=True)
    )
    # of terms =  52

    return dR

def update(R, omega, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    R.abb.vVvooO = eomcc_active_loops.update_r3c_010001(
        R.abb.vVvooO,
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
