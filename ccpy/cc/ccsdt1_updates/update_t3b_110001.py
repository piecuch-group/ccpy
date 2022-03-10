import numpy as np
from ccpy.utilities.active_space import get_active_slices

def update(T, dT, H, H0, shift, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

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

    return T, dT