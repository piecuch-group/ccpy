import numpy as np

def get_ccs_intermediates_opt(X, T, H, system, flag_RHF):
    """
    Calculate the CCS-like similarity-transformed HBar intermediates (H_N e^T1)_C.
    """
    # 1-body components
    # -------------------#
    X.a.vv -= np.einsum("me,am->ae", X.a.ov, T.a, optimize=True)
    if flag_RHF:
        X.b.vv = X.a.vv
    else:
        X.b.vv -= np.einsum("me,am->ae", X.b.ov, T.b, optimize=True)
    # 2-body components
    # -------------------#
    X.aa.ooov = np.einsum("mnfe,fi->mnie", H.aa.oovv, T.a, optimize=True) # no(3)nu(2)
    X.aa.vovv = -np.einsum("mnfe,an->amef", H.aa.oovv, T.a, optimize=True)  # no(2)nu(3)
    X.aa.oooo = 0.5 * H.aa.oooo + np.einsum("nmje,ei->mnij", H.aa.ooov + 0.5 * X.aa.ooov, T.a, optimize=True) # no(4)nu(1)
    X.aa.oooo -= np.transpose(X.aa.oooo, (0, 1, 3, 2))
    X.aa.voov = H.aa.voov + (
            np.einsum("amfe,fi->amie", H.aa.vovv + 0.5 * X.aa.vovv, T.a, optimize=True)
            - np.einsum("nmie,an->amie", H.aa.ooov + 0.5 * X.aa.ooov, T.a, optimize=True)
    ) # no(2)nu(3)
    L_amie = H.aa.voov + 0.5 * np.einsum('amef,ei->amif', H.aa.vovv, T.a, optimize=True) # no(2)nu(3)
    X_mnij = H.aa.oooo + np.einsum('mnie,ej->mnij', X.aa.ooov, T.a, optimize=True) # no(4)nu(1)
    X.aa.vooo = 0.5 * H.aa.vooo + (
        np.einsum('amie,ej->amij', L_amie, T.a, optimize=True)
       - 0.25 * np.einsum('mnij,am->anij', X_mnij, T.a, optimize=True)
    ) # no(3)nu(2)
    X.aa.vooo -= np.transpose(X.aa.vooo, (0, 1, 3, 2))
    L_amie = np.einsum('mnie,am->anie', H.aa.ooov, T.a, optimize=True)
    X.aa.vvov = H.aa.vvov + np.einsum("anie,bn->abie", L_amie, T.a, optimize=True) # no(1)nu(4)
    # You would expect to need this antisymmetrizer A(ab), but in the CCSD term H2A(abie)*T1A(ej),
    # the A(ab) term on the second term in this expression disappears because it's a V*1/2 T1^2
    # situation.
    #H.aa.vvov -= np.transpose(H.aa.vvov, (1, 0, 2, 3)) # WHY IS THIS NOT NEEDED???
    if flag_RHF:
        X.bb.ooov = X.aa.ooov
        X.bb.oooo = X.aa.oooo
        X.bb.vovv = X.aa.vovv
        X.bb.voov = X.aa.voov
        X.bb.vooo = X.aa.vooo
        X.bb.vvov = X.aa.vvov
    else:
        X.bb.ooov = np.einsum("mnfe,fi->mnie", H.bb.oovv, T.b, optimize=True)
        X.bb.oooo = 0.5 * H.bb.oooo + np.einsum("nmje,ei->mnij", H.bb.ooov + 0.5 * X.bb.ooov, T.b, optimize=True)
        X.bb.oooo -= np.transpose(X.bb.oooo, (0, 1, 3, 2))
        X.bb.vovv = -np.einsum("mnfe,an->amef", H.bb.oovv, T.b, optimize=True)
        X.bb.voov = H.bb.voov + (
            np.einsum("amfe,fi->amie", H.bb.vovv + 0.5 * X.bb.vovv, T.b, optimize=True)
            - np.einsum("nmie,an->amie", H.bb.ooov + 0.5 * X.bb.ooov, T.b, optimize=True)
        )
        L_amie = H.bb.voov + 0.5 * np.einsum('amef,ei->amif', H.bb.vovv, T.b, optimize=True)
        X_mnij = H.bb.oooo + np.einsum('mnie,ej->mnij', X.bb.ooov, T.b, optimize=True)
        X.bb.vooo = 0.5 * H.bb.vooo + (
            np.einsum('amie,ej->amij', L_amie, T.b, optimize=True)
           -0.25 * np.einsum('mnij,am->anij', X_mnij, T.b, optimize=True)
        )
        X.bb.vooo -= np.transpose(X.bb.vooo, (0, 1, 3, 2))
        L_amie = np.einsum('mnie,am->anie', H.bb.ooov, T.b, optimize=True)
        X.bb.vvov = H.bb.vvov + np.einsum("anie,bn->abie", L_amie, T.b, optimize=True)
        # You would expect to need this antisymmetrizer A(ab), but in the CCSD term H2C(abie)*T1B(ej),
        # the A(ab) term on the second term in this expression disappears because it's a V*1/2 T1^2
        # situation.
        #H.bb.vvov -= np.transpose(H.bb.vvov, (1, 0, 2, 3))

    X.ab.ooov = np.einsum("mnfe,fi->mnie", H.ab.oovv, T.a, optimize=True)
    X.ab.oovo = np.einsum("nmef,fi->nmei", H.ab.oovv, T.b, optimize=True)
    X.ab.oooo = H.ab.oooo + (
        np.einsum("mnej,ei->mnij", H.ab.oovo + 0.5 * X.ab.oovo, T.a, optimize=True)
        + np.einsum("mnie,ej->mnij", H.ab.ooov + 0.5 * X.ab.ooov, T.b, optimize=True)
    )
    X.ab.vovv = -np.einsum("nmef,an->amef", H.ab.oovv, T.a, optimize=True)
    X.ab.ovvv = -np.einsum("mnef,an->maef", H.ab.oovv, T.b, optimize=True)
    X.ab.voov = H.ab.voov + (
        np.einsum("amfe,fi->amie", H.ab.vovv + 0.5 * X.ab.vovv, T.a, optimize=True)
        - np.einsum("nmie,an->amie", H.ab.ooov + 0.5 * X.ab.ooov, T.a, optimize=True)
    )
    X.ab.ovvo = H.ab.ovvo + (
        np.einsum("maef,fi->maei", H.ab.ovvv + 0.5 * X.ab.ovvv, T.b, optimize=True)
        - np.einsum("mnei,an->maei", H.ab.oovo + 0.5 * X.ab.oovo, T.b, optimize=True)
    )
    X.ab.ovov = H.ab.ovov + (
        np.einsum("mafe,fi->maie", H.ab.ovvv + 0.5 * X.ab.ovvv, T.a, optimize=True)
        - np.einsum("mnie,an->maie", H.ab.ooov + 0.5 * X.ab.ooov, T.b, optimize=True)
    )
    X.ab.vovo = H.ab.vovo - (
        np.einsum("nmei,an->amei", H.ab.oovo + 0.5 * X.ab.oovo, T.a, optimize=True)
        - np.einsum("amef,fi->amei", H.ab.vovv + 0.5 * X.ab.vovv, T.b, optimize=True)
    )
    X_mnij = H.ab.oooo + (
        np.einsum("mnif,fj->mnij", H.ab.ooov, T.b, optimize=True)
        +np.einsum("mnej,ei->mnij", H.ab.oovo, T.a, optimize=True)
    )
    L_mbej = H.ab.ovvo + np.einsum("mbef,fj->mbej", H.ab.ovvv, T.b, optimize=True)
    X.ab.ovoo = H.ab.ovoo + (
        np.einsum("mbej,ei->mbij", L_mbej, T.a, optimize=True)
        -np.einsum("mnij,bn->mbij", X_mnij, T.b, optimize=True)
    )
    L_amie = np.einsum("amef,ei->amif", H.ab.vovv + X.ab.vovv, T.a, optimize=True)
    X.ab.vooo = H.ab.vooo + np.einsum("amif,fj->amij", H.ab.voov + L_amie, T.b, optimize=True)
    X.ab.vvvo = H.ab.vvvo - np.einsum("anej,bn->abej", H.ab.vovo, T.b, optimize=True)
    X.ab.vvov = H.ab.vvov - np.einsum("mbie,am->abie", H.ab.ovov, T.a, optimize=True)

    return X

def get_pre_ccs_intermediates(X, T, H, system, flag_RHF):
    X.a.ov = H.a.ov + (
            np.einsum("mnef,fn->me", H.aa.oovv, T.a, optimize=True)
            + np.einsum("mnef,fn->me", H.ab.oovv, T.b, optimize=True)
    )
    if flag_RHF:
        X.b.ov = X.a.ov
    else:
        X.b.ov = H.b.ov + (
            np.einsum("nmfe,fn->me", H.ab.oovv, T.a, optimize=True)
            + np.einsum("mnef,fn->me", H.bb.oovv, T.b, optimize=True)
        )
    X.a.vv = H.a.vv + (
            np.einsum("anef,fn->ae", H.aa.vovv, T.a, optimize=True)
            + np.einsum("anef,fn->ae", H.ab.vovv, T.b, optimize=True)
            - 0.5 * np.einsum("mnef,afmn->ae", H.aa.oovv, T.aa, optimize=True) #
            - np.einsum("mnef,afmn->ae", H.ab.oovv, T.ab, optimize=True) #
    )
    X.a.oo = H.a.oo + (
            np.einsum("mnif,fn->mi", H.aa.ooov, T.a, optimize=True)
            + np.einsum("mnif,fn->mi", H.ab.ooov, T.b, optimize=True)
            + np.einsum("me,ei->mi", X.a.ov, T.a, optimize=True)
            + 0.5 * np.einsum("mnef,efin->mi", H.aa.oovv, T.aa, optimize=True) # 
            + np.einsum("mnef,efin->mi", H.ab.oovv, T.ab, optimize=True) #
    )
    if flag_RHF:
        X.b.vv = X.a.vv
        X.b.oo = X.a.oo
    else:
        X.b.vv = H.b.vv + (
                    + np.einsum("anef,fn->ae", H.bb.vovv, T.b, optimize=True)
                    + np.einsum("nafe,fn->ae", H.ab.ovvv, T.a, optimize=True)
                    - 0.5 * np.einsum("mnef,afmn->ae", H.bb.oovv, T.bb, optimize=True) #
                    - np.einsum("nmfe,fanm->ae", H.ab.oovv, T.ab, optimize=True) #
        )
        X.b.oo = H.b.oo + (
                    + np.einsum("mnif,fn->mi", H.bb.ooov, T.b, optimize=True)
                    + np.einsum("nmfi,fn->mi", H.ab.oovo, T.a, optimize=True)
                    + np.einsum("me,ei->mi", X.b.ov, T.b, optimize=True)
                    + 0.5 * np.einsum("mnef,efin->mi", H.bb.oovv, T.bb, optimize=True) # 
                    + np.einsum("nmfe,feni->mi", H.ab.oovv, T.ab, optimize=True) #
        )
    return X

def get_ccs_intermediates_slow(T, H0):
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
