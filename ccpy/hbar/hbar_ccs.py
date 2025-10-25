import numpy as np
from ccpy.utilities.linear_algebra import ccpy_einsum

def get_ccs_intermediates_opt(X, T, H, flag_RHF):
    """
    Calculate the CCS-like similarity-transformed HBar intermediates (H_N e^T1)_C.
    """
    # 1-body components
    # -------------------#
    X.a.vv -= ccpy_einsum("me,am->ae", X.a.ov, T.a)
    if flag_RHF:
        X.b.vv = X.a.vv
    else:
        X.b.vv -= ccpy_einsum("me,am->ae", X.b.ov, T.b)
    # 2-body components
    # -------------------#
    X.aa.ooov = ccpy_einsum("mnfe,fi->mnie", H.aa.oovv, T.a) # no(3)nu(2)
    X.aa.vovv = -ccpy_einsum("mnfe,an->amef", H.aa.oovv, T.a)  # no(2)nu(3)
    X.aa.oooo = 0.5 * H.aa.oooo + ccpy_einsum("nmje,ei->mnij", H.aa.ooov + 0.5 * X.aa.ooov, T.a) # no(4)nu(1)
    X.aa.oooo -= np.transpose(X.aa.oooo, (0, 1, 3, 2))
    X.aa.voov = H.aa.voov + (
            ccpy_einsum("amfe,fi->amie", H.aa.vovv + 0.5 * X.aa.vovv, T.a)
            - ccpy_einsum("nmie,an->amie", H.aa.ooov + 0.5 * X.aa.ooov, T.a)
    ) # no(2)nu(3)
    L_amie = H.aa.voov + 0.5 * ccpy_einsum('amef,ei->amif', H.aa.vovv, T.a) # no(2)nu(3)
    X_mnij = H.aa.oooo + ccpy_einsum('mnie,ej->mnij', X.aa.ooov, T.a) # no(4)nu(1)
    X.aa.vooo = 0.5 * H.aa.vooo + (
        ccpy_einsum('amie,ej->amij', L_amie, T.a)
       - 0.25 * ccpy_einsum('mnij,am->anij', X_mnij, T.a)
    ) # no(3)nu(2)
    X.aa.vooo -= np.transpose(X.aa.vooo, (0, 1, 3, 2))
    L_amie = ccpy_einsum('mnie,am->anie', H.aa.ooov, T.a)
    X.aa.vvov = H.aa.vvov + ccpy_einsum("anie,bn->abie", L_amie, T.a) # no(1)nu(4)
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
        X.bb.ooov = ccpy_einsum("mnfe,fi->mnie", H.bb.oovv, T.b)
        X.bb.oooo = 0.5 * H.bb.oooo + ccpy_einsum("nmje,ei->mnij", H.bb.ooov + 0.5 * X.bb.ooov, T.b)
        X.bb.oooo -= np.transpose(X.bb.oooo, (0, 1, 3, 2))
        X.bb.vovv = -ccpy_einsum("mnfe,an->amef", H.bb.oovv, T.b)
        X.bb.voov = H.bb.voov + (
            ccpy_einsum("amfe,fi->amie", H.bb.vovv + 0.5 * X.bb.vovv, T.b)
            - ccpy_einsum("nmie,an->amie", H.bb.ooov + 0.5 * X.bb.ooov, T.b)
        )
        L_amie = H.bb.voov + 0.5 * ccpy_einsum('amef,ei->amif', H.bb.vovv, T.b)
        X_mnij = H.bb.oooo + ccpy_einsum('mnie,ej->mnij', X.bb.ooov, T.b)
        X.bb.vooo = 0.5 * H.bb.vooo + (
            ccpy_einsum('amie,ej->amij', L_amie, T.b)
           -0.25 * ccpy_einsum('mnij,am->anij', X_mnij, T.b)
        )
        X.bb.vooo -= np.transpose(X.bb.vooo, (0, 1, 3, 2))
        L_amie = ccpy_einsum('mnie,am->anie', H.bb.ooov, T.b)
        X.bb.vvov = H.bb.vvov + ccpy_einsum("anie,bn->abie", L_amie, T.b)
        # You would expect to need this antisymmetrizer A(ab), but in the CCSD term H2C(abie)*T1B(ej),
        # the A(ab) term on the second term in this expression disappears because it's a V*1/2 T1^2
        # situation.
        #H.bb.vvov -= np.transpose(H.bb.vvov, (1, 0, 2, 3))

    X.ab.ooov = ccpy_einsum("mnfe,fi->mnie", H.ab.oovv, T.a)
    X.ab.oovo = ccpy_einsum("nmef,fi->nmei", H.ab.oovv, T.b)
    X.ab.oooo = H.ab.oooo + (
        ccpy_einsum("mnej,ei->mnij", H.ab.oovo + 0.5 * X.ab.oovo, T.a)
        + ccpy_einsum("mnie,ej->mnij", H.ab.ooov + 0.5 * X.ab.ooov, T.b)
    )
    X.ab.vovv = -ccpy_einsum("nmef,an->amef", H.ab.oovv, T.a)
    X.ab.ovvv = -ccpy_einsum("mnef,an->maef", H.ab.oovv, T.b)
    X.ab.voov = H.ab.voov + (
        ccpy_einsum("amfe,fi->amie", H.ab.vovv + 0.5 * X.ab.vovv, T.a)
        - ccpy_einsum("nmie,an->amie", H.ab.ooov + 0.5 * X.ab.ooov, T.a)
    )
    X.ab.ovvo = H.ab.ovvo + (
        ccpy_einsum("maef,fi->maei", H.ab.ovvv + 0.5 * X.ab.ovvv, T.b)
        - ccpy_einsum("mnei,an->maei", H.ab.oovo + 0.5 * X.ab.oovo, T.b)
    )
    X.ab.ovov = H.ab.ovov + (
        ccpy_einsum("mafe,fi->maie", H.ab.ovvv + 0.5 * X.ab.ovvv, T.a)
        - ccpy_einsum("mnie,an->maie", H.ab.ooov + 0.5 * X.ab.ooov, T.b)
    )
    X.ab.vovo = H.ab.vovo - (
        ccpy_einsum("nmei,an->amei", H.ab.oovo + 0.5 * X.ab.oovo, T.a)
        - ccpy_einsum("amef,fi->amei", H.ab.vovv + 0.5 * X.ab.vovv, T.b)
    )
    X_mnij = H.ab.oooo + (
        ccpy_einsum("mnif,fj->mnij", H.ab.ooov, T.b)
        +ccpy_einsum("mnej,ei->mnij", H.ab.oovo, T.a)
    )
    L_mbej = H.ab.ovvo + ccpy_einsum("mbef,fj->mbej", H.ab.ovvv, T.b)
    X.ab.ovoo = H.ab.ovoo + (
        ccpy_einsum("mbej,ei->mbij", L_mbej, T.a)
        -ccpy_einsum("mnij,bn->mbij", X_mnij, T.b)
    )
    L_amie = ccpy_einsum("amef,ei->amif", H.ab.vovv + X.ab.vovv, T.a)
    X.ab.vooo = H.ab.vooo + ccpy_einsum("amif,fj->amij", H.ab.voov + L_amie, T.b)
    X.ab.vvvo = H.ab.vvvo - ccpy_einsum("anej,bn->abej", H.ab.vovo, T.b)
    X.ab.vvov = H.ab.vvov - ccpy_einsum("mbie,am->abie", H.ab.ovov, T.a)

    return X

def get_pre_ccs_intermediates(X, T, H, flag_RHF):
    X.a.ov = H.a.ov + (
            ccpy_einsum("mnef,fn->me", H.aa.oovv, T.a)
            + ccpy_einsum("mnef,fn->me", H.ab.oovv, T.b)
    )
    if flag_RHF:
        X.b.ov = X.a.ov
    else:
        X.b.ov = H.b.ov + (
            ccpy_einsum("nmfe,fn->me", H.ab.oovv, T.a)
            + ccpy_einsum("mnef,fn->me", H.bb.oovv, T.b)
        )
    X.a.vv = H.a.vv + (
            ccpy_einsum("anef,fn->ae", H.aa.vovv, T.a)
            + ccpy_einsum("anef,fn->ae", H.ab.vovv, T.b)
            - 0.5 * ccpy_einsum("mnef,afmn->ae", H.aa.oovv, T.aa) #
            - ccpy_einsum("mnef,afmn->ae", H.ab.oovv, T.ab) #
    )
    X.a.oo = H.a.oo + (
            ccpy_einsum("mnif,fn->mi", H.aa.ooov, T.a)
            + ccpy_einsum("mnif,fn->mi", H.ab.ooov, T.b)
            + ccpy_einsum("me,ei->mi", X.a.ov, T.a)
            + 0.5 * ccpy_einsum("mnef,efin->mi", H.aa.oovv, T.aa) # 
            + ccpy_einsum("mnef,efin->mi", H.ab.oovv, T.ab) #
    )
    if flag_RHF:
        X.b.vv = X.a.vv
        X.b.oo = X.a.oo
    else:
        X.b.vv = H.b.vv + (
                    + ccpy_einsum("anef,fn->ae", H.bb.vovv, T.b)
                    + ccpy_einsum("nafe,fn->ae", H.ab.ovvv, T.a)
                    - 0.5 * ccpy_einsum("mnef,afmn->ae", H.bb.oovv, T.bb) #
                    - ccpy_einsum("nmfe,fanm->ae", H.ab.oovv, T.ab) #
        )
        X.b.oo = H.b.oo + (
                    + ccpy_einsum("mnif,fn->mi", H.bb.ooov, T.b)
                    + ccpy_einsum("nmfi,fn->mi", H.ab.oovo, T.a)
                    + ccpy_einsum("me,ei->mi", X.b.ov, T.b)
                    + 0.5 * ccpy_einsum("mnef,efin->mi", H.bb.oovv, T.bb) # 
                    + ccpy_einsum("nmfe,feni->mi", H.ab.oovv, T.ab) #
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
        ccpy_einsum("mnef,fn->me", H0.aa.oovv, T.a)
        + ccpy_einsum("mnef,fn->me", H0.ab.oovv, T.b)
    )

    H.b.ov += (
        ccpy_einsum("nmfe,fn->me", H0.ab.oovv, T.a)
        + ccpy_einsum("mnef,fn->me", H0.bb.oovv, T.b)
    )

    H.a.vv += (
        ccpy_einsum("anef,fn->ae", H0.aa.vovv, T.a)
        + ccpy_einsum("anef,fn->ae", H0.ab.vovv, T.b)
        - ccpy_einsum("me,am->ae", H.a.ov, T.a)
    )

    H.a.oo += (
        ccpy_einsum("mnif,fn->mi", H0.aa.ooov, T.a)
        + ccpy_einsum("mnif,fn->mi", H0.ab.ooov, T.b)
        + ccpy_einsum("me,ei->mi", H.a.ov, T.a)
    )

    H.b.vv += (
        ccpy_einsum("anef,fn->ae", H0.bb.vovv, T.b)
        + ccpy_einsum("nafe,fn->ae", H0.ab.ovvv, T.a)
        - ccpy_einsum("me,am->ae", H.b.ov, T.b)
    )

    H.b.oo += (
        ccpy_einsum("mnif,fn->mi", H0.bb.ooov, T.b)
        + ccpy_einsum("nmfi,fn->mi", H0.ab.oovo, T.a)
        + ccpy_einsum("me,ei->mi", H.b.ov, T.b)
    )
    # 2-body components
    H.aa.oooo = (
        0.5 * H0.aa.oooo
        + ccpy_einsum("mnej,ei->mnij", H0.aa.oovo, T.a)
        + 0.5 * ccpy_einsum("mnef,ei,fj->mnij", H0.aa.oovv, T.a, T.a)
    )
    H.aa.oooo -= np.transpose(H.aa.oooo, (0, 1, 3, 2))

    H.aa.vvvv = (
        0.5 * H0.aa.vvvv
        - ccpy_einsum("mbef,am->abef", H0.aa.ovvv, T.a)
        + 0.5 * ccpy_einsum("mnef,bn,am->abef", H0.aa.oovv, T.a, T.a)
    )
    H.aa.vvvv -= np.transpose(H.aa.vvvv, (1, 0, 2, 3))

    H.aa.vooo += (
        - 0.5 * ccpy_einsum("nmij,an->amij", H0.aa.oooo, T.a)
        + ccpy_einsum("amef,ei,fj->amij", H0.aa.vovv, T.a, T.a)
        + ccpy_einsum("amie,ej->amij", H0.aa.voov, T.a)
        - ccpy_einsum("amje,ei->amij", H0.aa.voov, T.a)
        - 0.5 * ccpy_einsum("nmef,fj,an,ei->amij", H0.aa.oovv, T.a, T.a, T.a)
    )

    H.aa.vvov += (
         0.5 * ccpy_einsum("abfe,fi->abie", H0.aa.vvvv, T.a)
         + ccpy_einsum("mnie,am,bn->abie", H0.aa.ooov, T.a, T.a)
    )

    H.aa.voov += (
        - ccpy_einsum("nmie,an->amie", H0.aa.ooov, T.a)
        + ccpy_einsum("amfe,fi->amie", H0.aa.vovv, T.a)
        - ccpy_einsum("nmfe,fi,an->amie", H0.aa.oovv, T.a, T.a)
    )

    H.aa.ooov += ccpy_einsum("mnfe,fi->mnie", H0.aa.oovv, T.a)

    H.aa.vovv -= ccpy_einsum("mnfe,an->amef", H0.aa.oovv, T.a)

    H.ab.oooo += (
        + ccpy_einsum("mnej,ei->mnij", H0.ab.oovo, T.a)
        + ccpy_einsum("mnif,fj->mnij", H0.ab.ooov, T.b)
        + ccpy_einsum("mnef,ei,fj->mnij", H0.ab.oovv, T.a, T.b)
    )

    H.ab.vvvv += (
        - ccpy_einsum("mbef,am->abef", H0.ab.ovvv, T.a)
        - ccpy_einsum("anef,bn->abef", H0.ab.vovv, T.b)
        + ccpy_einsum("mnef,am,bn->abef", H0.ab.oovv, T.a, T.b)
    )

    H.ab.voov += (
        - ccpy_einsum("nmie,an->amie", H0.ab.ooov, T.a)
        + ccpy_einsum("amfe,fi->amie", H0.ab.vovv, T.a)
        - ccpy_einsum("nmfe,fi,an->amie", H0.ab.oovv, T.a, T.a)
    )

    H.ab.ovov += (
        + ccpy_einsum("mafe,fi->maie", H0.ab.ovvv, T.a)
        - ccpy_einsum("mnie,an->maie", H0.ab.ooov, T.b)
        - ccpy_einsum("mnfe,an,fi->maie", H0.ab.oovv, T.b, T.a)
    )

    H.ab.vovo += (
        - ccpy_einsum("nmei,an->amei", H0.ab.oovo, T.a)
        + ccpy_einsum("amef,fi->amei", H0.ab.vovv, T.b)
        - ccpy_einsum("nmef,fi,an->amei", H0.ab.oovv, T.b, T.a)
    )

    H.ab.ovvo += (
        + ccpy_einsum("maef,fi->maei", H0.ab.ovvv, T.b)
        - ccpy_einsum("mnei,an->maei", H0.ab.oovo, T.b)
        - ccpy_einsum("mnef,fi,an->maei", H0.ab.oovv, T.b, T.b)
    )

    H.ab.ovoo += (
        + ccpy_einsum("mbej,ei->mbij", H0.ab.ovvo, T.a)
        - ccpy_einsum("mnij,bn->mbij", H0.ab.oooo, T.b)
        - ccpy_einsum("mnif,bn,fj->mbij", H0.ab.ooov, T.b, T.b)
        - ccpy_einsum("mnej,bn,ei->mbij", H0.ab.oovo, T.b, T.a)
        + ccpy_einsum("mbef,fj,ei->mbij", H0.ab.ovvv, T.b, T.a)
    )

    H.ab.vooo += (
        + ccpy_einsum("amif,fj->amij", H0.ab.voov, T.b)
        - ccpy_einsum("nmef,an,ei,fj->amij", H0.ab.oovv, T.a, T.a, T.b)
        + ccpy_einsum("amef,fj,ei->amij", H0.ab.vovv, T.b, T.a)
    )

    H.ab.vvvo += (
        + ccpy_einsum("abef,fj->abej", H0.ab.vvvv, T.b)
        - ccpy_einsum("anej,bn->abej", H0.ab.vovo, T.b)
    )

    H.ab.vvov -= ccpy_einsum("mbie,am->abie", H0.ab.ovov, T.a)

    H.ab.ooov += ccpy_einsum("mnfe,fi->mnie", H0.ab.oovv, T.a)

    H.ab.oovo += ccpy_einsum("nmef,fi->nmei", H0.ab.oovv, T.b)

    H.ab.vovv -= ccpy_einsum("nmef,an->amef", H0.ab.oovv, T.a)

    H.ab.ovvv -= ccpy_einsum("mnfe,an->mafe", H0.ab.oovv, T.b)

    H.bb.oooo = (
        0.5 * H0.bb.oooo
        + ccpy_einsum("mnie,ej->mnij", H0.bb.ooov, T.b)
        + 0.5 * ccpy_einsum("mnef,ei,fj->mnij", H0.bb.oovv, T.b, T.b)
    )
    H.bb.oooo -= np.transpose(H.bb.oooo, (0, 1, 3, 2))

    H.bb.vvvv = (
        0.5 * H0.bb.vvvv
        - ccpy_einsum("mbef,am->abef", H0.bb.ovvv, T.b)
        + 0.5 * ccpy_einsum("mnef,bn,am->abef", H0.bb.oovv, T.b, T.b)
    )
    H.bb.vvvv -= np.transpose(H.bb.vvvv, (1, 0, 2, 3))

    H.bb.voov += (
        - ccpy_einsum("mnei,an->amie", H0.bb.oovo, T.b)
        + ccpy_einsum("amfe,fi->amie", H0.bb.vovv, T.b)
        - ccpy_einsum("mnef,fi,an->amie", H0.bb.oovv, T.b, T.b)
    )

    H.bb.vooo += (
        - 0.5 * ccpy_einsum("mnij,bn->bmji", H0.bb.oooo, T.b)
        + ccpy_einsum("mbef,ei,fj->bmji", H0.bb.ovvv, T.b, T.b)
        - 0.5 * ccpy_einsum("mnef,fj,ei,bn->bmji", H0.bb.oovv, T.b, T.b, T.b)
        + ccpy_einsum("mbif,fj->bmji", H0.bb.ovov, T.b)
        - ccpy_einsum("mbjf,fi->bmji", H0.bb.ovov, T.b)
    )

    H.bb.vvov +=(
        0.5 * ccpy_einsum("abef,fj->baje", H0.bb.vvvv, T.b)
        + ccpy_einsum("mnej,am,bn->baje", H0.bb.oovo, T.b, T.b)
    )

    H.bb.ooov += ccpy_einsum("mnfe,fi->mnie", H0.bb.oovv, T.b)

    H.bb.vovv -= ccpy_einsum("mnfe,an->amef", H0.bb.oovv, T.b)

    return H
