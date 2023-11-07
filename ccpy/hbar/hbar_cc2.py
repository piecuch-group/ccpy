import numpy as np
from ccpy.models.integrals import Integral

def build_hbar_cc2(T, H0, RHF_symmetry, system, *args):
    """Calculate the one- and two-body components of the CC2 similarity-transformed
     Hamiltonian."""
    from copy import deepcopy

    # Copy the Bare Hamiltonian object for T1/T2-similarity transformed HBar
    H = deepcopy(H0)

    # 1-body components
    H.a.ov += (
        np.einsum("mnef,fn->me", H0.aa.oovv, T.a, optimize=True)
        + np.einsum("mnef,fn->me", H0.ab.oovv, T.b, optimize=True)
    )

    H.b.ov += (
        np.einsum("nmfe,fn->me", H0.ab.oovv, T.a, optimize=True)
        + np.einsum("mnef,fn->me", H0.bb.oovv, T.b, optimize=True)
    )

    H.a.vv += (
        np.einsum("anef,fn->ae", H0.aa.vovv, T.a, optimize=True)
        + np.einsum("anef,fn->ae", H0.ab.vovv, T.b, optimize=True)
        - np.einsum("me,am->ae", H.a.ov, T.a, optimize=True)
    )

    H.a.oo += (
        np.einsum("mnif,fn->mi", H0.aa.ooov, T.a, optimize=True)
        + np.einsum("mnif,fn->mi", H0.ab.ooov, T.b, optimize=True)
        + np.einsum("me,ei->mi", H.a.ov, T.a, optimize=True)
    )

    H.b.vv += (
        np.einsum("anef,fn->ae", H0.bb.vovv, T.b, optimize=True)
        + np.einsum("nafe,fn->ae", H0.ab.ovvv, T.a, optimize=True)
        - np.einsum("me,am->ae", H.b.ov, T.b, optimize=True)
    )

    H.b.oo += (
        np.einsum("mnif,fn->mi", H0.bb.ooov, T.b, optimize=True)
        + np.einsum("nmfi,fn->mi", H0.ab.oovo, T.a, optimize=True)
        + np.einsum("me,ei->mi", H.b.ov, T.b, optimize=True)
    )
    # 2-body components
    H.aa.oooo = (
        0.5 * H0.aa.oooo
        + np.einsum("mnej,ei->mnij", H0.aa.oovo, T.a, optimize=True)
        + 0.5 * np.einsum("mnef,ei,fj->mnij", H0.aa.oovv, T.a, T.a, optimize=True)
    )
    H.aa.oooo -= np.transpose(H.aa.oooo, (0, 1, 3, 2))

    H.aa.vvvv = (
        0.5 * H0.aa.vvvv
        - np.einsum("mbef,am->abef", H0.aa.ovvv, T.a, optimize=True)
        + 0.5 * np.einsum("mnef,bn,am->abef", H0.aa.oovv, T.a, T.a, optimize=True)
    )
    H.aa.vvvv -= np.transpose(H.aa.vvvv, (1, 0, 2, 3))

    H.aa.vooo += (
        - 0.5 * np.einsum("nmij,an->amij", H0.aa.oooo, T.a, optimize=True)
        + np.einsum("amef,ei,fj->amij", H0.aa.vovv, T.a, T.a, optimize=True)
        + np.einsum("amie,ej->amij", H0.aa.voov, T.a, optimize=True)
        - np.einsum("amje,ei->amij", H0.aa.voov, T.a, optimize=True)
        - 0.5 * np.einsum("nmef,fj,an,ei->amij", H0.aa.oovv, T.a, T.a, T.a, optimize=True)
    )

    H.aa.vvov += (
         0.5 * np.einsum("abfe,fi->abie", H0.aa.vvvv, T.a, optimize=True)
         + np.einsum("mnie,am,bn->abie", H0.aa.ooov, T.a, T.a, optimize=True)
    )

    H.aa.voov += (
        - np.einsum("nmie,an->amie", H0.aa.ooov, T.a, optimize=True)
        + np.einsum("amfe,fi->amie", H0.aa.vovv, T.a, optimize=True)
        - np.einsum("nmfe,fi,an->amie", H0.aa.oovv, T.a, T.a, optimize=True)
    )

    H.aa.ooov += np.einsum("mnfe,fi->mnie", H0.aa.oovv, T.a, optimize=True)

    H.aa.vovv -= np.einsum("mnfe,an->amef", H0.aa.oovv, T.a, optimize=True)

    H.ab.oooo += (
        + np.einsum("mnej,ei->mnij", H0.ab.oovo, T.a, optimize=True)
        + np.einsum("mnif,fj->mnij", H0.ab.ooov, T.b, optimize=True)
        + np.einsum("mnef,ei,fj->mnij", H0.ab.oovv, T.a, T.b, optimize=True)
    )

    H.ab.vvvv += (
        - np.einsum("mbef,am->abef", H0.ab.ovvv, T.a, optimize=True)
        - np.einsum("anef,bn->abef", H0.ab.vovv, T.b, optimize=True)
        + np.einsum("mnef,am,bn->abef", H0.ab.oovv, T.a, T.b, optimize=True)
    )

    H.ab.voov += (
        - np.einsum("nmie,an->amie", H0.ab.ooov, T.a, optimize=True)
        + np.einsum("amfe,fi->amie", H0.ab.vovv, T.a, optimize=True)
        - np.einsum("nmfe,fi,an->amie", H0.ab.oovv, T.a, T.a, optimize=True)
    )

    H.ab.ovov += (
        + np.einsum("mafe,fi->maie", H0.ab.ovvv, T.a, optimize=True)
        - np.einsum("mnie,an->maie", H0.ab.ooov, T.b, optimize=True)
        - np.einsum("mnfe,an,fi->maie", H0.ab.oovv, T.b, T.a, optimize=True)
    )

    H.ab.vovo += (
        - np.einsum("nmei,an->amei", H0.ab.oovo, T.a, optimize=True)
        + np.einsum("amef,fi->amei", H0.ab.vovv, T.b, optimize=True)
        - np.einsum("nmef,fi,an->amei", H0.ab.oovv, T.b, T.a, optimize=True)
    )

    H.ab.ovvo += (
        + np.einsum("maef,fi->maei", H0.ab.ovvv, T.b, optimize=True)
        - np.einsum("mnei,an->maei", H0.ab.oovo, T.b, optimize=True)
        - np.einsum("mnef,fi,an->maei", H0.ab.oovv, T.b, T.b, optimize=True)
    )

    H.ab.ovoo += (
        + np.einsum("mbej,ei->mbij", H0.ab.ovvo, T.a, optimize=True)
        - np.einsum("mnij,bn->mbij", H0.ab.oooo, T.b, optimize=True)
        - np.einsum("mnif,bn,fj->mbij", H0.ab.ooov, T.b, T.b, optimize=True)
        - np.einsum("mnej,bn,ei->mbij", H0.ab.oovo, T.b, T.a, optimize=True)
        + np.einsum("mbef,fj,ei->mbij", H0.ab.ovvv, T.b, T.a, optimize=True)
    )

    H.ab.vooo += (
        + np.einsum("amif,fj->amij", H0.ab.voov, T.b, optimize=True)
        - np.einsum("nmef,an,ei,fj->amij", H0.ab.oovv, T.a, T.a, T.b, optimize=True)
        + np.einsum("amef,fj,ei->amij", H0.ab.vovv, T.b, T.a, optimize=True)
    )

    H.ab.vvvo += (
        + np.einsum("abef,fj->abej", H0.ab.vvvv, T.b, optimize=True)
        - np.einsum("anej,bn->abej", H0.ab.vovo, T.b, optimize=True)
    )

    H.ab.vvov -= np.einsum("mbie,am->abie", H0.ab.ovov, T.a, optimize=True)

    H.ab.ooov += np.einsum("mnfe,fi->mnie", H0.ab.oovv, T.a, optimize=True)

    H.ab.oovo += np.einsum("nmef,fi->nmei", H0.ab.oovv, T.b, optimize=True)

    H.ab.vovv -= np.einsum("nmef,an->amef", H0.ab.oovv, T.a, optimize=True)

    H.ab.ovvv -= np.einsum("mnfe,an->mafe", H0.ab.oovv, T.b, optimize=True)

    H.bb.oooo = (
        0.5 * H0.bb.oooo
        + np.einsum("mnie,ej->mnij", H0.bb.ooov, T.b, optimize=True)
        + 0.5 * np.einsum("mnef,ei,fj->mnij", H0.bb.oovv, T.b, T.b, optimize=True)
    )
    H.bb.oooo -= np.transpose(H.bb.oooo, (0, 1, 3, 2))

    H.bb.vvvv = (
        0.5 * H0.bb.vvvv
        - np.einsum("mbef,am->abef", H0.bb.ovvv, T.b, optimize=True)
        + 0.5 * np.einsum("mnef,bn,am->abef", H0.bb.oovv, T.b, T.b, optimize=True)
    )
    H.bb.vvvv -= np.transpose(H.bb.vvvv, (1, 0, 2, 3))

    H.bb.voov += (
        - np.einsum("mnei,an->amie", H0.bb.oovo, T.b, optimize=True)
        + np.einsum("amfe,fi->amie", H0.bb.vovv, T.b, optimize=True)
        - np.einsum("mnef,fi,an->amie", H0.bb.oovv, T.b, T.b, optimize=True)
    )

    H.bb.vooo += (
        - 0.5 * np.einsum("mnij,bn->bmji", H0.bb.oooo, T.b, optimize=True)
        + np.einsum("mbef,ei,fj->bmji", H0.bb.ovvv, T.b, T.b, optimize=True)
        - 0.5 * np.einsum("mnef,fj,ei,bn->bmji", H0.bb.oovv, T.b, T.b, T.b, optimize=True)
        + np.einsum("mbif,fj->bmji", H0.bb.ovov, T.b, optimize=True)
        - np.einsum("mbjf,fi->bmji", H0.bb.ovov, T.b, optimize=True)
    )

    H.bb.vvov +=(
        0.5 * np.einsum("abef,fj->baje", H0.bb.vvvv, T.b, optimize=True)
        + np.einsum("mnej,am,bn->baje", H0.bb.oovo, T.b, T.b, optimize=True)
    )

    # For RHF symmetry, copy a parts to b and aa parts to bb
    if RHF_symmetry:
        H.b.ov = H.a.ov.copy()
        H.b.oo = H.a.oo.copy()
        H.b.vv = H.a.vv.copy()
        H.bb.oooo = H.aa.oooo.copy()
        H.bb.ooov = H.aa.ooov.copy()
        H.bb.vooo = H.aa.vooo.copy()
        H.bb.oovv = H.aa.oovv.copy()
        H.bb.voov = H.aa.voov.copy()
        H.bb.vovv = H.aa.vovv.copy()
        H.bb.vvov = H.aa.vvov.copy()
        H.bb.vvvv = H.aa.vvvv.copy()

    return H
