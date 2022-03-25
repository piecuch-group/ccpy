import numpy as np
from ccpy.utilities.active_space import get_active_slices

def build(R, T, H, X, system):

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    x3a = (6.0 / 12.0) * (
            -1.0 * np.einsum('bmJI,AcmK->AbcIJK', X.aa.vooo[va, :, Oa, Oa], T.aa[Va, va, :, Oa], optimize=True)
    )
    x3a += (3.0 / 12.0) * (
            +1.0 * np.einsum('AmJI,bcmK->AbcIJK', X.aa.vooo[Va, :, Oa, Oa], T.aa[va, va, :, Oa], optimize=True)
    )
    x3a += (6.0 / 12.0) * (
            -1.0 * np.einsum('bmJI,AcmK->AbcIJK', H.aa.vooo[va, :, Oa, Oa], R.aa[Va, va, :, Oa], optimize=True)
    )
    x3a += (3.0 / 12.0) * (
            +1.0 * np.einsum('AmJI,bcmK->AbcIJK', H.aa.vooo[Va, :, Oa, Oa], R.aa[va, va, :, Oa], optimize=True)
    )
    x3a += (6.0 / 12.0) * (
            +1.0 * np.einsum('bAJe,ecIK->AbcIJK', X.aa.vvov[va, Va, Oa, :], T.aa[:, va, Oa, Oa], optimize=True)
    )
    x3a += (3.0 / 12.0) * (
            -1.0 * np.einsum('bcJe,eAIK->AbcIJK', X.aa.vvov[va, va, Oa, :], T.aa[:, Va, Oa, Oa], optimize=True)
    )
    x3a += (6.0 / 12.0) * (
            +1.0 * np.einsum('bAJe,ecIK->AbcIJK', H.aa.vvov[va, Va, Oa, :], R.aa[:, va, Oa, Oa], optimize=True)
    )
    x3a += (3.0 / 12.0) * (
            -1.0 * np.einsum('bcJe,eAIK->AbcIJK', H.aa.vvov[va, va, Oa, :], R.aa[:, Va, Oa, Oa], optimize=True)
    )

    x3a += (2.0 / 12.0) * (
            -1.0 * np.einsum('be,AceIJK->AbcIJK', X.a.vv[va, va], T.aaa.VvvOOO, optimize=True)
            - 1.0 * np.einsum('bE,EAcIJK->AbcIJK', X.a.vv[va, Va], T.aaa.VVvOOO, optimize=True)
    )
    x3a += (1.0 / 12.0) * (
            -1.0 * np.einsum('AE,EcbIJK->AbcIJK', X.a.vv[Va, Va], T.aaa.VvvOOO, optimize=True)
    )
    x3a += (3.0 / 12.0) * (
            -1.0 * np.einsum('mJ,AcbmIK->AbcIJK', X.a.oo[oa, Oa], T.aaa.VvvoOO, optimize=True)
            + 1.0 * np.einsum('MJ,AcbIMK->AbcIJK', X.a.oo[Oa, Oa], T.aaa.VvvOOO, optimize=True)
    )
    x3a += (3.0 / 12.0) * (
            -0.5 * np.einsum('mnIJ,AcbmnK->AbcIJK', X.aa.oooo[oa, oa, Oa, Oa], T.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('mNIJ,AcbmNK->AbcIJK', X.aa.oooo[oa, Oa, Oa, Oa], T.aaa.VvvoOO, optimize=True)
            - 0.5 * np.einsum('MNIJ,AcbMNK->AbcIJK', X.aa.oooo[Oa, Oa, Oa, Oa], T.aaa.VvvOOO, optimize=True)
    )
    x3a += (2.0 / 12.0) * (
            +1.0 * np.einsum('AbeF,FceIJK->AbcIJK', X.aa.vvvv[Va, va, va, Va], T.aaa.VvvOOO, optimize=True)
            - 0.5 * np.einsum('AbEF,FEcIJK->AbcIJK', X.aa.vvvv[Va, va, Va, Va], T.aaa.VVvOOO, optimize=True)
    )
    x3a += (1.0 / 12.0) * (
            +0.5 * np.einsum('cbef,AfeIJK->AbcIJK', X.aa.vvvv[va, va, va, va], T.aaa.VvvOOO, optimize=True)
            + 1.0 * np.einsum('cbeF,AFeIJK->AbcIJK', X.aa.vvvv[va, va, va, Va], T.aaa.VVvOOO, optimize=True)
            + 0.5 * np.einsum('cbEF,AFEIJK->AbcIJK', X.aa.vvvv[va, va, Va, Va], T.aaa.VVVOOO, optimize=True)
    )
    x3a += (6.0 / 12.0) * (
            +1.0 * np.einsum('bmJe,AcemIK->AbcIJK', X.aa.voov[va, oa, Oa, va], T.aaa.VvvoOO, optimize=True)
            + 1.0 * np.einsum('bmJE,EAcmIK->AbcIJK', X.aa.voov[va, oa, Oa, Va], T.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('bMJe,AceIMK->AbcIJK', X.aa.voov[va, Oa, Oa, va], T.aaa.VvvOOO, optimize=True)
            - 1.0 * np.einsum('bMJE,EAcIMK->AbcIJK', X.aa.voov[va, Oa, Oa, Va], T.aaa.VVvOOO, optimize=True)
    )
    x3a += (3.0 / 12.0) * (
            +1.0 * np.einsum('AmJE,EcbmIK->AbcIJK', X.aa.voov[Va, oa, Oa, Va], T.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('AMJE,EcbIMK->AbcIJK', X.aa.voov[Va, Oa, Oa, Va], T.aaa.VvvOOO, optimize=True)
    )
    x3a += (6.0 / 12.0) * (
            +1.0 * np.einsum('bmJe,AceIKm->AbcIJK', X.ab.voov[va, ob, Oa, vb], T.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('bmJE,AcEIKm->AbcIJK', X.ab.voov[va, ob, Oa, Vb], T.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('bMJe,AceIKM->AbcIJK', X.ab.voov[va, Ob, Oa, vb], T.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('bMJE,AcEIKM->AbcIJK', X.ab.voov[va, Ob, Oa, Vb], T.aab.VvVOOO, optimize=True)
    )
    x3a += (3.0 / 12.0) * (
            +1.0 * np.einsum('AmJE,cbEIKm->AbcIJK', X.ab.voov[Va, ob, Oa, Vb], T.aab.vvVOOo, optimize=True)
            + 1.0 * np.einsum('AMJE,cbEIKM->AbcIJK', X.ab.voov[Va, Ob, Oa, Vb], T.aab.vvVOOO, optimize=True)
    )
    x3a += (3.0 / 12.0) * (
            -1.0 * np.einsum('mJ,AcbmIK->AbcIJK', H.a.oo[oa, Oa], R.aaa.VvvoOO, optimize=True)
            + 1.0 * np.einsum('MJ,AcbIMK->AbcIJK', H.a.oo[Oa, Oa], R.aaa.VvvOOO, optimize=True)
    )
    x3a += (2.0 / 12.0) * (
            -1.0 * np.einsum('be,AceIJK->AbcIJK', H.a.vv[va, va], R.aaa.VvvOOO, optimize=True)
            - 1.0 * np.einsum('bE,EAcIJK->AbcIJK', H.a.vv[va, Va], R.aaa.VVvOOO, optimize=True)
    )
    x3a += (1.0 / 12.0) * (
            -1.0 * np.einsum('AE,EcbIJK->AbcIJK', H.a.vv[Va, Va], R.aaa.VvvOOO, optimize=True)
    )
    x3a += (3.0 / 12.0) * (
            -0.5 * np.einsum('mnIJ,AcbmnK->AbcIJK', H.aa.oooo[oa, oa, Oa, Oa], R.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('mNIJ,AcbmNK->AbcIJK', H.aa.oooo[oa, Oa, Oa, Oa], R.aaa.VvvoOO, optimize=True)
            - 0.5 * np.einsum('MNIJ,AcbMNK->AbcIJK', H.aa.oooo[Oa, Oa, Oa, Oa], R.aaa.VvvOOO, optimize=True)
    )
    x3a += (2.0 / 12.0) * (
            +1.0 * np.einsum('AbeF,FceIJK->AbcIJK', H.aa.vvvv[Va, va, va, Va], R.aaa.VvvOOO, optimize=True)
            - 0.5 * np.einsum('AbEF,FEcIJK->AbcIJK', H.aa.vvvv[Va, va, Va, Va], R.aaa.VVvOOO, optimize=True)
    )
    x3a += (1.0 / 12.0) * (
            +0.5 * np.einsum('cbef,AfeIJK->AbcIJK', H.aa.vvvv[va, va, va, va], R.aaa.VvvOOO, optimize=True)
            + 1.0 * np.einsum('cbeF,AFeIJK->AbcIJK', H.aa.vvvv[va, va, va, Va], R.aaa.VVvOOO, optimize=True)
            + 0.5 * np.einsum('cbEF,AFEIJK->AbcIJK', H.aa.vvvv[va, va, Va, Va], R.aaa.VVVOOO, optimize=True)
    )
    x3a += (3.0 / 12.0) * (
            -1.0 * np.einsum('AmIE,EcbmJK->AbcIJK', H.aa.voov[Va, oa, Oa, Va], R.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('AMIE,EcbMJK->AbcIJK', H.aa.voov[Va, Oa, Oa, Va], R.aaa.VvvOOO, optimize=True)
    )
    x3a += (6.0 / 12.0) * (
            -1.0 * np.einsum('bmIe,AcemJK->AbcIJK', H.aa.voov[va, oa, Oa, va], R.aaa.VvvoOO, optimize=True)
            + 1.0 * np.einsum('bmIE,AEcmJK->AbcIJK', H.aa.voov[va, oa, Oa, Va], R.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('bMIe,AceMJK->AbcIJK', H.aa.voov[va, Oa, Oa, va], R.aaa.VvvOOO, optimize=True)
            + 1.0 * np.einsum('bMIE,AEcMJK->AbcIJK', H.aa.voov[va, Oa, Oa, Va], R.aaa.VVvOOO, optimize=True)
    )
    x3a += (3.0 / 12.0) * (
            -1.0 * np.einsum('AmIE,cbEJKm->AbcIJK', H.ab.voov[Va, ob, Oa, Vb], R.aab.vvVOOo, optimize=True)
            - 1.0 * np.einsum('AMIE,cbEJKM->AbcIJK', H.ab.voov[Va, Ob, Oa, Vb], R.aab.vvVOOO, optimize=True)
    )
    x3a += (6.0 / 12.0) * (
            -1.0 * np.einsum('bmIe,AceJKm->AbcIJK', H.ab.voov[va, ob, Oa, vb], R.aab.VvvOOo, optimize=True)
            - 1.0 * np.einsum('bmIE,AcEJKm->AbcIJK', H.ab.voov[va, ob, Oa, Vb], R.aab.VvVOOo, optimize=True)
            - 1.0 * np.einsum('bMIe,AceJKM->AbcIJK', H.ab.voov[va, Ob, Oa, vb], R.aab.VvvOOO, optimize=True)
            - 1.0 * np.einsum('bMIE,AcEJKM->AbcIJK', H.ab.voov[va, Ob, Oa, Vb], R.aab.VvVOOO, optimize=True)
    )

    x3a -= np.transpose(x3a, (0, 2, 1, 3, 4, 5))

    x3a -= np.transpose(x3a, (0, 1, 2, 4, 3, 5)) + np.transpose(x3a, (0, 1, 2, 3, 5, 4)) \
           + np.transpose(x3a, (0, 1, 2, 5, 4, 3)) - np.transpose(x3a, (0, 1, 2, 4, 5, 3)) \
           - np.transpose(x3a, (0, 1, 2, 5, 3, 4))

    return x3a