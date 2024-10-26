import numpy as np
from ccpy.utilities.active_space import get_active_slices
from ccpy.lib.core import cc_active_loops

#@profile
def build(T, dT, H, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    # MM(2,3)
    dT.aab.VVvooO = (2.0 / 4.0) * (
            +1.0 * np.einsum('BceK,Aeij->ABcijK', H.ab.vvvo[Va, vb, :, Ob], T.aa[Va, :, oa, oa], optimize=True)
    )
    dT.aab.VVvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('mcjK,ABim->ABcijK', H.ab.ovoo[:, vb, oa, Ob], T.aa[Va, Va, oa, :], optimize=True)
    )
    dT.aab.VVvooO += (4.0 / 4.0) * (
            +1.0 * np.einsum('Acie,BejK->ABcijK', H.ab.vvov[Va, vb, oa, :], T.ab[Va, :, oa, Ob], optimize=True)
    )
    dT.aab.VVvooO += (4.0 / 4.0) * (
            -1.0 * np.einsum('AmiK,Bcjm->ABcijK', H.ab.vooo[Va, :, oa, Ob], T.ab[Va, vb, oa, :], optimize=True)
    )
    dT.aab.VVvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('ABie,ecjK->ABcijK', H.aa.vvov[Va, Va, oa, :], T.ab[:, vb, oa, Ob], optimize=True)
    )
    dT.aab.VVvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('Amij,BcmK->ABcijK', H.aa.vooo[Va, :, oa, oa], T.ab[Va, vb, :, Ob], optimize=True)
    )
    # (H(2) * T3)_C
    dT.aab.VVvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mi,BAcmjK->ABcijK', H.a.oo[oa, oa], T.aab.VVvooO, optimize=True)
            - 1.0 * np.einsum('Mi,BAcjMK->ABcijK', H.a.oo[Oa, oa], T.aab.VVvoOO, optimize=True)
    )
    dT.aab.VVvooO += (1.0 / 4.0) * (
            +1.0 * np.einsum('MK,BAcijM->ABcijK', H.b.oo[Ob, Ob], T.aab.VVvooO, optimize=True)
    )
    dT.aab.VVvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('Ae,BecijK->ABcijK', H.a.vv[Va, va], T.aab.VvvooO, optimize=True)
            - 1.0 * np.einsum('AE,BEcijK->ABcijK', H.a.vv[Va, Va], T.aab.VVvooO, optimize=True)
    )
    dT.aab.VVvooO += (1.0 / 4.0) * (
            -1.0 * np.einsum('ce,BAeijK->ABcijK', H.b.vv[vb, vb], T.aab.VVvooO, optimize=True)
            - 1.0 * np.einsum('cE,BAEijK->ABcijK', H.b.vv[vb, Vb], T.aab.VVVooO, optimize=True)
    )
    dT.aab.VVvooO += (1.0 / 4.0) * (
            -0.5 * np.einsum('mnij,BAcmnK->ABcijK', H.aa.oooo[oa, oa, oa, oa], T.aab.VVvooO, optimize=True)
            - 1.0 * np.einsum('mNij,BAcmNK->ABcijK', H.aa.oooo[oa, Oa, oa, oa], T.aab.VVvoOO, optimize=True)
            - 0.5 * np.einsum('MNij,BAcMNK->ABcijK', H.aa.oooo[Oa, Oa, oa, oa], T.aab.VVvOOO, optimize=True)
    )
    dT.aab.VVvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('MnjK,BAciMn->ABcijK', H.ab.oooo[Oa, ob, oa, Ob], T.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('mNjK,BAcimN->ABcijK', H.ab.oooo[oa, Ob, oa, Ob], T.aab.VVvooO, optimize=True)
            - 1.0 * np.einsum('MNjK,BAciMN->ABcijK', H.ab.oooo[Oa, Ob, oa, Ob], T.aab.VVvoOO, optimize=True)
    )
    dT.aab.VVvooO += (1.0 / 4.0) * (
            +1.0 * np.einsum('ABEf,EfcijK->ABcijK', H.aa.vvvv[Va, Va, Va, va], T.aab.VvvooO, optimize=True)
            - 0.5 * np.einsum('ABEF,FEcijK->ABcijK', H.aa.vvvv[Va, Va, Va, Va], T.aab.VVvooO, optimize=True)
    )
    dT.aab.VVvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('Bcef,AefijK->ABcijK', H.ab.vvvv[Va, vb, va, vb], T.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('BceF,AeFijK->ABcijK', H.ab.vvvv[Va, vb, va, Vb], T.aab.VvVooO, optimize=True)
            - 1.0 * np.einsum('BcEf,EAfijK->ABcijK', H.ab.vvvv[Va, vb, Va, vb], T.aab.VVvooO, optimize=True)
            - 1.0 * np.einsum('BcEF,EAFijK->ABcijK', H.ab.vvvv[Va, vb, Va, Vb], T.aab.VVVooO, optimize=True)
    )
    dT.aab.VVvooO += (4.0 / 4.0) * (
            -1.0 * np.einsum('Amie,BecmjK->ABcijK', H.aa.voov[Va, oa, oa, va], T.aab.VvvooO, optimize=True)
            - 1.0 * np.einsum('AmiE,BEcmjK->ABcijK', H.aa.voov[Va, oa, oa, Va], T.aab.VVvooO, optimize=True)
            + 1.0 * np.einsum('AMie,BecjMK->ABcijK', H.aa.voov[Va, Oa, oa, va], T.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('AMiE,BEcjMK->ABcijK', H.aa.voov[Va, Oa, oa, Va], T.aab.VVvoOO, optimize=True)
    )
    dT.aab.VVvooO += (4.0 / 4.0) * (
            -1.0 * np.einsum('Amie,BcejmK->ABcijK', H.ab.voov[Va, ob, oa, vb], T.abb.VvvooO, optimize=True)
            + 1.0 * np.einsum('AmiE,BEcjmK->ABcijK', H.ab.voov[Va, ob, oa, Vb], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('AMie,BcejMK->ABcijK', H.ab.voov[Va, Ob, oa, vb], T.abb.VvvoOO, optimize=True)
            + 1.0 * np.einsum('AMiE,BEcjMK->ABcijK', H.ab.voov[Va, Ob, oa, Vb], T.abb.VVvoOO, optimize=True)
    )
    dT.aab.VVvooO += (1.0 / 4.0) * (
            -1.0 * np.einsum('MceK,BAeijM->ABcijK', H.ab.ovvo[Oa, vb, va, Ob], T.aaa.VVvooO, optimize=True)
            - 1.0 * np.einsum('McEK,EBAijM->ABcijK', H.ab.ovvo[Oa, vb, Va, Ob], T.aaa.VVVooO, optimize=True)
    )
    dT.aab.VVvooO += (1.0 / 4.0) * (
            -1.0 * np.einsum('cMKe,BAeijM->ABcijK', H.bb.voov[vb, Ob, Ob, vb], T.aab.VVvooO, optimize=True)
            - 1.0 * np.einsum('cMKE,BAEijM->ABcijK', H.bb.voov[vb, Ob, Ob, Vb], T.aab.VVVooO, optimize=True)
    )
    dT.aab.VVvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('AMeK,BecijM->ABcijK', H.ab.vovo[Va, Ob, va, Ob], T.aab.VvvooO, optimize=True)
            + 1.0 * np.einsum('AMEK,BEcijM->ABcijK', H.ab.vovo[Va, Ob, Va, Ob], T.aab.VVvooO, optimize=True)
    )
    dT.aab.VVvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mcie,BAemjK->ABcijK', H.ab.ovov[oa, vb, oa, vb], T.aab.VVvooO, optimize=True)
            + 1.0 * np.einsum('mciE,BAEmjK->ABcijK', H.ab.ovov[oa, vb, oa, Vb], T.aab.VVVooO, optimize=True)
            - 1.0 * np.einsum('Mcie,BAejMK->ABcijK', H.ab.ovov[Oa, vb, oa, vb], T.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('MciE,BAEjMK->ABcijK', H.ab.ovov[Oa, vb, oa, Vb], T.aab.VVVoOO, optimize=True)
    )

    dT.aab.VVvooO -= np.transpose(dT.aab.VVvooO, (1, 0, 2, 3, 4, 5))
    dT.aab.VVvooO -= np.transpose(dT.aab.VVvooO, (0, 1, 2, 4, 3, 5))

    return dT

def update(T, dT, H, shift, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    T.aab.VVvooO, dT.aab.VVvooO = cc_active_loops.update_t3b_110001(
        T.aab.VVvooO,
        dT.aab.VVvooO,
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