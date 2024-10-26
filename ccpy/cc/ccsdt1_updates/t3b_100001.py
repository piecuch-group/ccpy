import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import cc_active_loops

import time as time

#@profile
def build(T, dT, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    # MM(2,3)
    dT.aab.VvvooO = (1.0 / 2.0) * (
            +1.0 * np.einsum('bceK,Aeij->AbcijK', H.ab.vvvo[va, vb, :, Ob], T.aa[Va, :, oa, oa], optimize=True)
    )
    dT.aab.VvvooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('AceK,beij->AbcijK', H.ab.vvvo[Va, vb, :, Ob], T.aa[va, :, oa, oa], optimize=True)
    )
    dT.aab.VvvooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mcjK,Abim->AbcijK', H.ab.ovoo[:, vb, oa, Ob], T.aa[Va, va, oa, :], optimize=True)
    )
    dT.aab.VvvooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('Acie,bejK->AbcijK', H.ab.vvov[Va, vb, oa, :], T.ab[va, :, oa, Ob], optimize=True)
    )
    dT.aab.VvvooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('bcie,AejK->AbcijK', H.ab.vvov[va, vb, oa, :], T.ab[Va, :, oa, Ob], optimize=True)
    )
    dT.aab.VvvooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('AmiK,bcjm->AbcijK', H.ab.vooo[Va, :, oa, Ob], T.ab[va, vb, oa, :], optimize=True)
    )
    dT.aab.VvvooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('bmiK,Acjm->AbcijK', H.ab.vooo[va, :, oa, Ob], T.ab[Va, vb, oa, :], optimize=True)
    )
    dT.aab.VvvooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('Abie,ecjK->AbcijK', H.aa.vvov[Va, va, oa, :], T.ab[:, vb, oa, Ob], optimize=True)
    )
    dT.aab.VvvooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('Amij,bcmK->AbcijK', H.aa.vooo[Va, :, oa, oa], T.ab[va, vb, :, Ob], optimize=True)
    )
    dT.aab.VvvooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('bmij,AcmK->AbcijK', H.aa.vooo[va, :, oa, oa], T.ab[Va, vb, :, Ob], optimize=True)
    )
    # (H(2) * T3)_C
    dT.aab.VvvooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mi,AbcmjK->AbcijK', H.a.oo[oa, oa], T.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('Mi,AbcjMK->AbcijK', H.a.oo[Oa, oa], T.aab.VvvoOO, optimize=True)
    )
    dT.aab.VvvooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('MK,AbcijM->AbcijK', H.b.oo[Ob, Ob], T.aab.VvvooO, optimize=True)
    )
    dT.aab.VvvooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('AE,EbcijK->AbcijK', H.a.vv[Va, Va], T.aab.VvvooO, optimize=True)
    )
    dT.aab.VvvooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('be,AecijK->AbcijK', H.a.vv[va, va], T.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('bE,AEcijK->AbcijK', H.a.vv[va, Va], T.aab.VVvooO, optimize=True)
    )
    dT.aab.VvvooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('ce,AbeijK->AbcijK', H.b.vv[vb, vb], T.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('cE,AbEijK->AbcijK', H.b.vv[vb, Vb], T.aab.VvVooO, optimize=True)
    )
    dT.aab.VvvooO += (1.0 / 2.0) * (
            +0.5 * np.einsum('mnij,AbcmnK->AbcijK', H.aa.oooo[oa, oa, oa, oa], T.aab.VvvooO, optimize=True)
            - 1.0 * np.einsum('Mnij,AbcnMK->AbcijK', H.aa.oooo[Oa, oa, oa, oa], T.aab.VvvoOO, optimize=True)
            + 0.5 * np.einsum('MNij,AbcMNK->AbcijK', H.aa.oooo[Oa, Oa, oa, oa], T.aab.VvvOOO, optimize=True)
    )
    dT.aab.VvvooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mNjK,AbcimN->AbcijK', H.ab.oooo[oa, Ob, oa, Ob], T.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('MnjK,AbciMn->AbcijK', H.ab.oooo[Oa, ob, oa, Ob], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('MNjK,AbciMN->AbcijK', H.ab.oooo[Oa, Ob, oa, Ob], T.aab.VvvoOO, optimize=True)
    )
    #t1 = time.time()
    dT.aab.VvvooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('AbeF,FecijK->AbcijK', H.aa.vvvv[Va, va, va, Va], T.aab.VvvooO, optimize=True)
            - 0.5 * np.einsum('AbEF,FEcijK->AbcijK', H.aa.vvvv[Va, va, Va, Va], T.aab.VVvooO, optimize=True)
    )
    dT.aab.VvvooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('bcef,AefijK->AbcijK', H.ab.vvvv[va, vb, va, vb], T.aab.VvvooO, optimize=True)
            - 1.0 * np.einsum('bcEf,EAfijK->AbcijK', H.ab.vvvv[va, vb, Va, vb], T.aab.VVvooO, optimize=True)
            + 1.0 * np.einsum('bceF,AeFijK->AbcijK', H.ab.vvvv[va, vb, va, Vb], T.aab.VvVooO, optimize=True)
            - 1.0 * np.einsum('bcEF,EAFijK->AbcijK', H.ab.vvvv[va, vb, Va, Vb], T.aab.VVVooO, optimize=True)
    )
    dT.aab.VvvooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('AcEf,EbfijK->AbcijK', H.ab.vvvv[Va, vb, Va, vb], T.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('AceF,ebFijK->AbcijK', H.ab.vvvv[Va, vb, va, Vb], T.aab.vvVooO, optimize=True)
            + 1.0 * np.einsum('AcEF,EbFijK->AbcijK', H.ab.vvvv[Va, vb, Va, Vb], T.aab.VvVooO, optimize=True)
    )
    #print("Time for t3b VvvooO = ", time.time() - t1)
    dT.aab.VvvooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('AmiE,EbcmjK->AbcijK', H.aa.voov[Va, oa, oa, Va], T.aab.VvvooO, optimize=True)
            - 1.0 * np.einsum('AMiE,EbcjMK->AbcijK', H.aa.voov[Va, Oa, oa, Va], T.aab.VvvoOO, optimize=True)
    )
    dT.aab.VvvooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('bmie,AecmjK->AbcijK', H.aa.voov[va, oa, oa, va], T.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('bmiE,AEcmjK->AbcijK', H.aa.voov[va, oa, oa, Va], T.aab.VVvooO, optimize=True)
            - 1.0 * np.einsum('bMie,AecjMK->AbcijK', H.aa.voov[va, Oa, oa, va], T.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('bMiE,AEcjMK->AbcijK', H.aa.voov[va, Oa, oa, Va], T.aab.VVvoOO, optimize=True)
    )
    dT.aab.VvvooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('AmiE,bEcjmK->AbcijK', H.ab.voov[Va, ob, oa, Vb], T.abb.vVvooO, optimize=True)
            + 1.0 * np.einsum('AMiE,bEcjMK->AbcijK', H.ab.voov[Va, Ob, oa, Vb], T.abb.vVvoOO, optimize=True)
    )
    dT.aab.VvvooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('bmie,AcejmK->AbcijK', H.ab.voov[va, ob, oa, vb], T.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('bmiE,AEcjmK->AbcijK', H.ab.voov[va, ob, oa, Vb], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('bMie,AcejMK->AbcijK', H.ab.voov[va, Ob, oa, vb], T.abb.VvvoOO, optimize=True)
            - 1.0 * np.einsum('bMiE,AEcjMK->AbcijK', H.ab.voov[va, Ob, oa, Vb], T.abb.VVvoOO, optimize=True)
    )
    dT.aab.VvvooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('MceK,AebijM->AbcijK', H.ab.ovvo[Oa, vb, va, Ob], T.aaa.VvvooO, optimize=True)
            + 1.0 * np.einsum('McEK,EAbijM->AbcijK', H.ab.ovvo[Oa, vb, Va, Ob], T.aaa.VVvooO, optimize=True)
    )
    dT.aab.VvvooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('cMKe,AbeijM->AbcijK', H.bb.voov[vb, Ob, Ob, vb], T.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('cMKE,AbEijM->AbcijK', H.bb.voov[vb, Ob, Ob, Vb], T.aab.VvVooO, optimize=True)
    )
    dT.aab.VvvooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('AMEK,EbcijM->AbcijK', H.ab.vovo[Va, Ob, Va, Ob], T.aab.VvvooO, optimize=True)
    )
    dT.aab.VvvooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('bMeK,AecijM->AbcijK', H.ab.vovo[va, Ob, va, Ob], T.aab.VvvooO, optimize=True)
            - 1.0 * np.einsum('bMEK,AEcijM->AbcijK', H.ab.vovo[va, Ob, Va, Ob], T.aab.VVvooO, optimize=True)
    )
    dT.aab.VvvooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('mcie,AbemjK->AbcijK', H.ab.ovov[oa, vb, oa, vb], T.aab.VvvooO, optimize=True)
            - 1.0 * np.einsum('mciE,AbEmjK->AbcijK', H.ab.ovov[oa, vb, oa, Vb], T.aab.VvVooO, optimize=True)
            + 1.0 * np.einsum('Mcie,AbejMK->AbcijK', H.ab.ovov[Oa, vb, oa, vb], T.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('MciE,AbEjMK->AbcijK', H.ab.ovov[Oa, vb, oa, Vb], T.aab.VvVoOO, optimize=True)
    )

    dT.aab.VvvooO -= np.transpose(dT.aab.VvvooO, (0, 1, 2, 4, 3, 5))

    return dT

def update(T, dT, H, shift, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.aab.VvvooO, dT.aab.VvvooO = cc_active_loops.update_t3b_100001(
        T.aab.VvvooO,
        dT.aab.VvvooO,
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