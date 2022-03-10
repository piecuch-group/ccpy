import numpy as np
from ccpy.utilities.active_space import get_active_slices

def update(T, dT, H, H0, shift, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    dT.abb.VVVooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mi,ACBmjK->ABCijK', H.a.oo[oa, oa], T.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('Mi,ACBMjK->ABCijK', H.a.oo[Oa, oa], T.abb.VVVOoO, optimize=True)
    )
    dT.abb.VVVooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('mj,ACBimK->ABCijK', H.b.oo[ob, ob], T.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('Mj,ACBiMK->ABCijK', H.b.oo[Ob, ob], T.abb.VVVoOO, optimize=True)
    )
    dT.abb.VVVooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('MK,ACBijM->ABCijK', H.b.oo[Ob, Ob], T.abb.VVVooO, optimize=True)
    )
    dT.abb.VVVooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('Ae,eCBijK->ABCijK', H.a.vv[Va, va], T.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('AE,ECBijK->ABCijK', H.a.vv[Va, Va], T.abb.VVVooO, optimize=True)
    )
    dT.abb.VVVooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('Be,ACeijK->ABCijK', H.b.vv[Vb, vb], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('BE,ACEijK->ABCijK', H.b.vv[Vb, Vb], T.abb.VVVooO, optimize=True)
    )
    dT.abb.VVVooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mNjK,ACBimN->ABCijK', H.bb.oooo[ob, Ob, ob, Ob], T.abb.VVVooO, optimize=True)
            - 0.5 * np.einsum('MNjK,ACBiMN->ABCijK', H.bb.oooo[Ob, Ob, ob, Ob], T.abb.VVVoOO, optimize=True)
    )
    dT.abb.VVVooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('mnij,ACBmnK->ABCijK', H.ab.oooo[oa, ob, oa, ob], T.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('Mnij,ACBMnK->ABCijK', H.ab.oooo[Oa, ob, oa, ob], T.abb.VVVOoO, optimize=True)
            - 1.0 * np.einsum('mNij,ACBmNK->ABCijK', H.ab.oooo[oa, Ob, oa, ob], T.abb.VVVoOO, optimize=True)
            - 1.0 * np.einsum('MNij,ACBMNK->ABCijK', H.ab.oooo[Oa, Ob, oa, ob], T.abb.VVVOOO, optimize=True)
    )
    dT.abb.VVVooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('MniK,ACBMnj->ABCijK', H.ab.oooo[Oa, ob, oa, Ob], T.abb.VVVOoo, optimize=True)
            - 1.0 * np.einsum('mNiK,ACBmjN->ABCijK', H.ab.oooo[oa, Ob, oa, Ob], T.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('MNiK,ACBMjN->ABCijK', H.ab.oooo[Oa, Ob, oa, Ob], T.abb.VVVOoO, optimize=True)
    )
    dT.abb.VVVooO += (1.0 / 2.0) * (
            -0.5 * np.einsum('BCef,AfeijK->ABCijK', H.bb.vvvv[Vb, Vb, vb, vb], T.abb.VvvooO, optimize=True)
            - 1.0 * np.einsum('BCeF,AFeijK->ABCijK', H.bb.vvvv[Vb, Vb, vb, Vb], T.abb.VVvooO, optimize=True)
            - 0.5 * np.einsum('BCEF,AFEijK->ABCijK', H.bb.vvvv[Vb, Vb, Vb, Vb], T.abb.VVVooO, optimize=True)
    )
    dT.abb.VVVooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('ABef,eCfijK->ABCijK', H.ab.vvvv[Va, Vb, va, vb], T.abb.vVvooO, optimize=True)
            - 1.0 * np.einsum('ABEf,ECfijK->ABCijK', H.ab.vvvv[Va, Vb, Va, vb], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('ABeF,eCFijK->ABCijK', H.ab.vvvv[Va, Vb, va, Vb], T.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('ABEF,ECFijK->ABCijK', H.ab.vvvv[Va, Vb, Va, Vb], T.abb.VVVooO, optimize=True)
    )
    dT.abb.VVVooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('Amie,eCBmjK->ABCijK', H.aa.voov[Va, oa, oa, va], T.abb.vVVooO, optimize=True)
            - 1.0 * np.einsum('AmiE,ECBmjK->ABCijK', H.aa.voov[Va, oa, oa, Va], T.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('AMie,eCBMjK->ABCijK', H.aa.voov[Va, Oa, oa, va], T.abb.vVVOoO, optimize=True)
            - 1.0 * np.einsum('AMiE,ECBMjK->ABCijK', H.aa.voov[Va, Oa, oa, Va], T.abb.VVVOoO, optimize=True)
    )
    dT.abb.VVVooO += (1.0 / 2.0) * (
            -1.0 * np.einsum('Amie,CBemjK->ABCijK', H.ab.voov[Va, ob, oa, vb], T.bbb.VVvooO, optimize=True)
            - 1.0 * np.einsum('AmiE,CBEmjK->ABCijK', H.ab.voov[Va, ob, oa, Vb], T.bbb.VVVooO, optimize=True)
            + 1.0 * np.einsum('AMie,CBejMK->ABCijK', H.ab.voov[Va, Ob, oa, vb], T.bbb.VVvoOO, optimize=True)
            + 1.0 * np.einsum('AMiE,CBEjMK->ABCijK', H.ab.voov[Va, Ob, oa, Vb], T.bbb.VVVoOO, optimize=True)
    )
    dT.abb.VVVooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mBej,AeCimK->ABCijK', H.ab.ovvo[oa, Vb, va, ob], T.aab.VvVooO, optimize=True)
            - 1.0 * np.einsum('mBEj,EACimK->ABCijK', H.ab.ovvo[oa, Vb, Va, ob], T.aab.VVVooO, optimize=True)
            + 1.0 * np.einsum('MBej,AeCiMK->ABCijK', H.ab.ovvo[Oa, Vb, va, ob], T.aab.VvVoOO, optimize=True)
            - 1.0 * np.einsum('MBEj,EACiMK->ABCijK', H.ab.ovvo[Oa, Vb, Va, ob], T.aab.VVVoOO, optimize=True)
    )
    dT.abb.VVVooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('MBeK,AeCiMj->ABCijK', H.ab.ovvo[Oa, Vb, va, Ob], T.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('MBEK,EACiMj->ABCijK', H.ab.ovvo[Oa, Vb, Va, Ob], T.aab.VVVoOo, optimize=True)
    )
    dT.abb.VVVooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('Bmje,ACeimK->ABCijK', H.bb.voov[Vb, ob, ob, vb], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('BmjE,ACEimK->ABCijK', H.bb.voov[Vb, ob, ob, Vb], T.abb.VVVooO, optimize=True)
            - 1.0 * np.einsum('BMje,ACeiMK->ABCijK', H.bb.voov[Vb, Ob, ob, vb], T.abb.VVvoOO, optimize=True)
            - 1.0 * np.einsum('BMjE,ACEiMK->ABCijK', H.bb.voov[Vb, Ob, ob, Vb], T.abb.VVVoOO, optimize=True)
    )
    dT.abb.VVVooO += (2.0 / 2.0) * (
            -1.0 * np.einsum('BMKe,ACeijM->ABCijK', H.bb.voov[Vb, Ob, Ob, vb], T.abb.VVvooO, optimize=True)
            - 1.0 * np.einsum('BMKE,ACEijM->ABCijK', H.bb.voov[Vb, Ob, Ob, Vb], T.abb.VVVooO, optimize=True)
    )
    dT.abb.VVVooO += (2.0 / 2.0) * (
            +1.0 * np.einsum('mBie,ACemjK->ABCijK', H.ab.ovov[oa, Vb, oa, vb], T.abb.VVvooO, optimize=True)
            + 1.0 * np.einsum('mBiE,ACEmjK->ABCijK', H.ab.ovov[oa, Vb, oa, Vb], T.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('MBie,ACeMjK->ABCijK', H.ab.ovov[Oa, Vb, oa, vb], T.abb.VVvOoO, optimize=True)
            + 1.0 * np.einsum('MBiE,ACEMjK->ABCijK', H.ab.ovov[Oa, Vb, oa, Vb], T.abb.VVVOoO, optimize=True)
    )
    dT.abb.VVVooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('Amej,eCBimK->ABCijK', H.ab.vovo[Va, ob, va, ob], T.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('AmEj,ECBimK->ABCijK', H.ab.vovo[Va, ob, Va, ob], T.abb.VVVooO, optimize=True)
            + 1.0 * np.einsum('AMej,eCBiMK->ABCijK', H.ab.vovo[Va, Ob, va, ob], T.abb.vVVoOO, optimize=True)
            + 1.0 * np.einsum('AMEj,ECBiMK->ABCijK', H.ab.vovo[Va, Ob, Va, ob], T.abb.VVVoOO, optimize=True)
    )
    dT.abb.VVVooO += (1.0 / 2.0) * (
            +1.0 * np.einsum('AMeK,eCBijM->ABCijK', H.ab.vovo[Va, Ob, va, Ob], T.abb.vVVooO, optimize=True)
            + 1.0 * np.einsum('AMEK,ECBijM->ABCijK', H.ab.vovo[Va, Ob, Va, Ob], T.abb.VVVooO, optimize=True)
    )

    dT.abb.VVVooO -= np.transpose(dT.abb.VVVooO, (0, 2, 1, 3, 4, 5))

    return T, dT
