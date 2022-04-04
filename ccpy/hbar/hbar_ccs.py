import numpy as np

def get_ccs_intermediates(T, H0):
    """
    Calculate the CCS-like similarity-transformed HBar intermediates (H_N e^T1)_C.
    Copied as-is from original CCpy.
    """
    from copy import deepcopy

    # Copy the Bare Hamiltonian object for T1-transforemd HBar
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

    H.bb.ooov += np.einsum("mnfe,fi->mnie", H0.bb.oovv, T.b, optimize=True)

    H.bb.vovv -= np.einsum("mnfe,an->amef", H0.bb.oovv, T.b, optimize=True)

    return H


def get_ccs_intermediates_opt(T, H0):
    """
    Calculate the CCS-like similarity-transformed HBar intermediates (H_N e^T1)_C.
    """

    # [TODO]: Copying large arrays is slow! We should pass in Hbar and simply update its elements.
    from copy import deepcopy
    # Copy the Bare Hamiltonian object for T1-transforemd HBar
    H = deepcopy(H0)

    # 1-body components
    # -------------------#
    H.a.ov = H0.a.ov + (
        np.einsum("mnef,fn->me", H0.aa.oovv, T.a, optimize=True)
        + np.einsum("mnef,fn->me", H0.ab.oovv, T.b, optimize=True)
    ) # no(2)nu(2)

    H.b.ov = H0.b.ov + (
            np.einsum("nmfe,fn->me", H0.ab.oovv, T.a, optimize=True)
            + np.einsum("mnef,fn->me", H0.bb.oovv, T.b, optimize=True)
    ) # no(2)nu(2)

    H.a.vv = H0.a.vv + (
        np.einsum("anef,fn->ae", H0.aa.vovv, T.a, optimize=True)
        + np.einsum("anef,fn->ae", H0.ab.vovv, T.b, optimize=True)
        - np.einsum("me,am->ae", H.a.ov, T.a, optimize=True)
    ) # no(1)nu(3)

    H.a.oo = H0.a.oo + (
        np.einsum("mnif,fn->mi", H0.aa.ooov, T.a, optimize=True)
        + np.einsum("mnif,fn->mi", H0.ab.ooov, T.b, optimize=True)
        + np.einsum("me,ei->mi", H.a.ov, T.a, optimize=True)
    ) # no(3)nu(1)

    H.b.vv = H0.b.vv + (
        np.einsum("anef,fn->ae", H0.bb.vovv, T.b, optimize=True)
        + np.einsum("nafe,fn->ae", H0.ab.ovvv, T.a, optimize=True)
        - np.einsum("me,am->ae", H.b.ov, T.b, optimize=True)
    ) # no(1)nu(3)

    H.b.oo = H0.b.oo + (
        np.einsum("mnif,fn->mi", H0.bb.ooov, T.b, optimize=True)
        + np.einsum("nmfi,fn->mi", H0.ab.oovo, T.a, optimize=True)
        + np.einsum("me,ei->mi", H.b.ov, T.b, optimize=True)
    ) # no(3)nu(1)

    # 2-body components
    # -------------------#
    # AA parts
    # -------------------#
    Q_ooov = np.einsum("mnfe,fi->mnie", H0.aa.oovv, T.a, optimize=True) # no(3)nu(2)
    H.aa.ooov = H0.aa.ooov + 0.5 * Q_ooov

    H.aa.oooo = 0.5 * H0.aa.oooo + np.einsum("nmje,ei->mnij", H.aa.ooov, T.a, optimize=True) # no(4)nu(1)
    H.aa.oooo -= np.transpose(H.aa.oooo, (0, 1, 3, 2))

    Q_vovv = -np.einsum("mnfe,an->amef", H0.aa.oovv, T.a, optimize=True) # no(2)nu(3)
    H.aa.vovv = H0.aa.vovv + 0.5 * Q_vovv

    H.aa.voov = H0.aa.voov + (
            np.einsum("amfe,fi->amie", H.aa.vovv, T.a, optimize=True)
            - np.einsum("nmie,an->amie", H.aa.ooov, T.a, optimize=True)
    ) # no(2)nu(3)

    L_amie = H0.aa.voov + 0.5 * np.einsum('amef,ei->amif', H0.aa.vovv, T.a, optimize=True) # no(2)nu(3)
    X_mnij = H0.aa.oooo + np.einsum('mnie,ej->mnij', Q_ooov, T.a, optimize=True) # no(4)nu(1)
    H.aa.vooo = 0.5 * H0.aa.vooo + (
        np.einsum('amie,ej->amij', L_amie, T.a, optimize=True)
       -0.25 * np.einsum('mnij,am->anij', X_mnij, T.a, optimize=True)
    ) # no(3)nu(2)
    H.aa.vooo -= np.transpose(H.aa.vooo, (0, 1, 3, 2))

    L_amie = np.einsum('mnie,am->anie', H0.aa.ooov, T.a, optimize=True)
    H.aa.vvov = H0.aa.vvov + + np.einsum("anie,bn->abie", L_amie, T.a, optimize=True) # no(1)nu(4)
    #H.aa.vvov -= np.transpose(H.aa.vvov, (1, 0, 2, 3)) # WHY IS THIS NOT NEEDED???

    # -------------------#
    # AB parts
    # -------------------#
    Q1 = np.einsum("mnfe,fi->mnie", H0.ab.oovv, T.a, optimize=True)
    H.ab.ooov = H0.ab.ooov + 0.5 * Q1

    Q1 = np.einsum("nmef,fi->nmei", H0.ab.oovv, T.b, optimize=True)
    H.ab.oovo = H0.ab.oovo + 0.5 * Q1

    H.ab.oooo = H0.ab.oooo + (
        np.einsum("mnej,ei->mnij", H.ab.oovo, T.a, optimize=True)
        + np.einsum("mnie,ej->mnij", H.ab.ooov, T.b, optimize=True)
    )

    Q_vovv = -np.einsum("nmef,an->amef", H0.ab.oovv, T.a, optimize=True)
    H.ab.vovv = H0.ab.vovv + 0.5 * Q_vovv

    Q1 = -np.einsum("mnef,an->maef", H0.ab.oovv, T.b, optimize=True)
    H.ab.ovvv = H0.ab.ovvv + 0.5 * Q1

    H.ab.voov = H0.ab.voov + (
        np.einsum("amfe,fi->amie", H.ab.vovv, T.a, optimize=True)
        - np.einsum("nmie,an->amie", H.ab.ooov, T.a, optimize=True)
    )

    H.ab.ovvo = H0.ab.ovvo + (
        np.einsum("maef,fi->maei", H.ab.ovvv, T.b, optimize=True)
        - np.einsum("mnei,an->maei", H.ab.oovo, T.b, optimize=True)
    )

    H.ab.ovov = H0.ab.ovov + (
        np.einsum("mafe,fi->maie", H.ab.ovvv, T.a, optimize=True)
        - np.einsum("mnie,an->maie", H.ab.ooov, T.b, optimize=True)
    )

    H.ab.vovo = H0.ab.vovo - (
        np.einsum("nmei,an->amei", H.ab.oovo, T.a, optimize=True)
        - np.einsum("amef,fi->amei", H.ab.vovv, T.b, optimize=True)
    )

    X_mnij = H0.ab.oooo + (
        np.einsum("mnif,fj->mnij", H0.ab.ooov, T.b, optimize=True)
        +np.einsum("mnej,ei->mnij", H0.ab.oovo, T.a, optimize=True)
    )
    L_mbej = H0.ab.ovvo + np.einsum("mbef,fj->mbej", H0.ab.ovvv, T.b, optimize=True)
    H.ab.ovoo = H0.ab.ovoo + (
        np.einsum("mbej,ei->mbij", L_mbej, T.a, optimize=True)
        -np.einsum("mnij,bn->mbij", X_mnij, T.b, optimize=True)
    )

    L_amie = np.einsum("amef,ei->amif", H.ab.vovv + 0.5 * Q_vovv, T.a, optimize=True)
    H.ab.vooo =H0.ab.vooo + np.einsum("amif,fj->amij", H0.ab.voov + L_amie, T.b, optimize=True)

    H.ab.vvvo = H0.ab.vvvo - np.einsum("anej,bn->abej", H0.ab.vovo, T.b, optimize=True)

    H.ab.vvov = H0.ab.vvov - np.einsum("mbie,am->abie", H0.ab.ovov, T.a, optimize=True)
    # -------------------#
    # BB parts
    # -------------------#
    Q_ooov = np.einsum("mnfe,fi->mnie", H0.bb.oovv, T.b, optimize=True)
    H.bb.ooov = H0.bb.ooov + 0.5 * Q_ooov

    H.bb.oooo = 0.5 * H0.bb.oooo + np.einsum("nmje,ei->mnij", H.bb.ooov, T.b, optimize=True)
    H.bb.oooo -= np.transpose(H.bb.oooo, (0, 1, 3, 2))

    Q_vovv = -np.einsum("mnfe,an->amef", H0.bb.oovv, T.b, optimize=True)
    H.bb.vovv = H0.bb.vovv + 0.5 * Q_vovv

    H.bb.voov = H0.bb.voov + (
        np.einsum("amfe,fi->amie", H.bb.vovv, T.b, optimize=True)
        - np.einsum("nmie,an->amie", H.bb.ooov, T.b, optimize=True)
    )

    L_amie = H0.bb.voov + 0.5 * np.einsum('amef,ei->amif', H0.bb.vovv, T.b, optimize=True)
    X_mnij = H0.bb.oooo + np.einsum('mnie,ej->mnij', Q_ooov, T.b, optimize=True)
    H.bb.vooo = 0.5 * H0.bb.vooo + (
        np.einsum('amie,ej->amij', L_amie, T.b, optimize=True)
       -0.25 * np.einsum('mnij,am->anij', X_mnij, T.b, optimize=True)
    )
    H.bb.vooo -= np.transpose(H.bb.vooo, (0, 1, 3, 2))

    L_amie = np.einsum('mnie,am->anie', H0.bb.ooov, T.b, optimize=True)
    H.bb.vvov = H0.bb.vvov + + np.einsum("anie,bn->abie", L_amie, T.b, optimize=True)
    #H.bb.vvov -= np.transpose(H.bb.vvov, (1, 0, 2, 3))


    return H
