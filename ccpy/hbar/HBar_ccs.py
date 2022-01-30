import numpy as np

def get_ccs_intermediates_opt(T, H0):
    """Calculate the CCS-like similarity-transformed HBar intermediates (H_N e^T1)_C.

    Parameters
    ----------
    cc_t : dict
        Cluster amplitudes T1
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary

    Returns
    -------
    H1* : dict
        One-body HBar similarity-transformed intermediates. Sorted by occ/unocc blocks.
    H2* : dict
        Two-body HBar similarity-transformed intermediates. Sorted by occ/unocc blocks.
    """
    # H = Integral.fromEmpty(system, 2, data_type=np.type)

    from copy import deepcopy
    # Copy the Bare Hamiltonian object for T1-transforemd HBar
    H = deepcopy(H0)

    # 1-body components
    # -------------------#
    H.a.ov += np.einsum('mnef,fn->me', H0.aa.oovv, T.a, optimize=True) \
              + np.einsum('mnef,fn->me', H0.ab.oovv, T.b, optimize=True)

    H.b.ov += np.einsum('nmfe,fn->me', H0.ab.oovv, T.a, optimize=True) \
              + np.einsum('mnef,fn->me', H0.bb.oovv, T.b, optimize=True)

    H.a.vv += np.einsum('anef,fn->ae', H0.aa.vovv, T.a, optimize=True) \
              + np.einsum('anef,fn->ae', H0.ab.vovv, T.b, optimize=True) \
              - np.einsum('me,am->ae', H.a.ov, T.a, optimize=True)

    H.a.oo += np.einsum('mnif,fn->mi', H0.aa.ooov, T.a, optimize=True) \
              + np.einsum('mnif,fn->mi', H0.ab.ooov, T.b, optimize=True) \
              + np.einsum('me,ei->mi', H.a.ov, T.a, optimize=True)

    H.b.vv += np.einsum('anef,fn->ae', H0.bb.vovv, T.b, optimize=True) \
              + np.einsum('nafe,fn->ae', H0.ab.ovvv, T.a, optimize=True) \
              - np.einsum('me,am->ae', H.b.ov, T.b, optimize=True)

    H.b.oo += np.einsum('mnif,fn->mi', H0.bb.ooov, T.b, optimize=True) \
              + np.einsum('nmfi,fn->mi', H0.ab.oovo, T.a, optimize=True) \
              + np.einsum('me,ei->mi', H.b.ov, T.b, optimize=True)
    # 2-body components
    # -------------------#
    # AA parts
    # -------------------#
    Q1 = np.einsum('mnfe,fi->mnie', H0.aa.oovv, T.a, optimize=True)
    I2A_ooov = H0.aa.ooov + 0.5 * Q1
    H.aa.ooov = I2A_ooov + 0.5 * Q1

    H.aa.oooo = 0.5 * H0.aa.oooo + np.einsum('nmje,ei->mnij', I2A_ooov, T.a, optimize=True)
    H.aa.oooo -= np.transpose(H.aa.oooo, (0, 1, 3, 2))

    Q1 = -np.einsum('mnfe,an->amef', H0.aa.oovv, T.a, optimize=True)
    I2A_vovv = H0.aa.vovv + 0.5 * Q1
    H.aa.vovv = I2A_vovv + 0.5 * Q1

    H.aa.vvvv = 0.5 * H0.aa.vvvv - np.einsum('amef,bm->abef', I2A_vovv, T.a, optimize=True)
    H.aa.vvvv -= np.transpose(H.aa.vvvv, (1, 0, 2, 3))

    H.aa.voov += np.einsum('amfe,fi->amie', I2A_vovv, T.a, optimize=True) \
                 - np.einsum('nmie,an->amie', I2A_ooov, T.a, optimize=True)

    H.aa.vooo = 0.5 * H0.aa.vooo - 0.25 * np.einsum('mnij,am->amij', H.aa.oooo, T.a, optimize=True) \
                + np.einsum('amie,ej->amij', H0.aa.voov, T.a, optimize=True) \
                + 0.5 * np.einsum('amef,ei,fj->amij', H0.aa.vovv, T.a, T.a, optimize=True)
    H.aa.vooo -= np.transpose(H.aa.vooo, (0, 1, 3, 2))

    H.aa.vvov = H0.aa.vvov + 0.5 * np.einsum('abfe,fi->abie', H0.aa.vvvv, T.a, optimize=True)
    # -------------------#
    # AB parts
    # -------------------#
    Q1 = np.einsum('mnfe,fi->mnie', H0.ab.oovv, T.a, optimize=True)
    I2B_ooov = H0.ab.ooov + 0.5 * Q1
    H.ab.ooov = I2B_ooov + 0.5 * Q1

    Q1 = np.einsum('nmef,fi->nmei', H0.ab.oovv, T.b, optimize=True)
    I2B_oovo = H0.ab.oovo + 0.5 * Q1
    H.ab.oovo = I2B_oovo + 0.5 * Q1

    H.ab.oooo += np.einsum('mnej,ei->mnij', I2B_oovo, T.a, optimize=True) \
                 + np.einsum('mnie,ej->mnij', I2B_ooov, T.b, optimize=True)

    Q1 = -np.einsum('nmef,an->amef', H0.ab.oovv, T.a, optimize=True)
    I2B_vovv = H0.ab.vovv + 0.5 * Q1
    H.ab.vovv = I2B_vovv + 0.5 * Q1

    Q1 = -np.einsum('mnef,an->maef', H0.ab.oovv, T.b, optimize=True)
    I2B_ovvv = H0.ab.ovvv + 0.5 * Q1
    H.ab.ovvv = I2B_ovvv + 0.5 * Q1

    H.ab.vvvv -= np.einsum('mbef,am->abef', I2B_ovvv, T.a, optimize=True) \
                 + np.einsum('amef,bm->abef', I2B_vovv, T.b, optimize=True)

    H.ab.voov += np.einsum('amfe,fi->amie', I2B_vovv, T.a, optimize=True) \
                 - np.einsum('nmie,an->amie', I2B_ooov, T.a, optimize=True)

    H.ab.ovvo += np.einsum('maef,fi->maei', I2B_ovvv, T.b, optimize=True) \
                 - np.einsum('mnei,an->maei', I2B_oovo, T.b, optimize=True)

    H.ab.ovov += np.einsum('mafe,fi->maie', I2B_ovvv, T.a, optimize=True) \
                 - np.einsum('mnie,an->maie', I2B_ooov, T.b, optimize=True)

    H.ab.vovo -= np.einsum('nmei,an->amei', I2B_oovo, T.a, optimize=True) \
                 - np.einsum('amef,fi->amei', I2B_vovv, T.b, optimize=True)

    H.ab.ovoo = H0.ab.ovoo - np.einsum('mnij,bn->mbij', H.ab.oooo, T.b, optimize=True) \
                + np.einsum('mbej,ei->mbij', H0.ab.ovvo, T.a, optimize=True) \
                + np.einsum('mbef,fj,ei->mbij', H0.ab.ovvv, T.b, T.a, optimize=True)

    H.ab.vooo = H0.ab.vooo + np.einsum('amif,fj->amij', H0.ab.voov, T.b, optimize=True) \
                + np.einsum('amef,fj,ei->amij', H.ab.vovv, T.b, T.a, optimize=True)

    H.ab.vvvo = H0.ab.vvvo + np.einsum('abef,fj->abej', H0.ab.vvvv, T.b, optimize=True) \
                - np.einsum('anej,bn->abej', H0.ab.vovo, T.b, optimize=True)

    H.ab.vvov = H0.ab.vvov - np.einsum('mbie,am->abie', H0.ab.ovov, T.a, optimize=True)
    # -------------------#
    # BB parts
    # -------------------#
    Q1 = np.einsum('mnfe,fi->mnie', H0.bb.oovv, T.b, optimize=True)
    I2C_ooov = H0.bb.ooov + 0.5 * Q1
    H.bb.ooov = I2C_ooov + 0.5 * Q1

    H.bb.oooo = 0.5 * H0.bb.oooo + np.einsum('nmje,ei->mnij', I2C_ooov, T.b, optimize=True)
    H.bb.oooo -= np.transpose(H.bb.oooo, (0, 1, 3, 2))

    Q1 = -np.einsum('mnfe,an->amef', H0.bb.oovv, T.b, optimize=True)
    I2C_vovv = H0.bb.vovv + 0.5 * Q1
    H.bb.vovv = I2C_vovv + 0.5 * Q1

    H.bb.vvvv = 0.5 * H0.bb.vvvv - np.einsum('amef,bm->abef', I2C_vovv, T.b, optimize=True)
    H.bb.vvvv -= np.transpose(H.bb.vvvv, (1, 0, 2, 3))

    H.bb.voov += np.einsum('amfe,fi->amie', I2C_vovv, T.b, optimize=True) \
                 - np.einsum('nmie,an->amie', I2C_ooov, T.b, optimize=True)

    H.bb.vooo = 0.5 * H0.bb.vooo - 0.25 * np.einsum('mnij,am->amij', H.bb.oooo, T.b, optimize=True) \
                + np.einsum('amie,ej->amij', H0.bb.voov, T.b, optimize=True) \
                + 0.5 * np.einsum('amef,ei,fj->amij', H0.bb.vovv, T.b, T.b, optimize=True)
    H.bb.vooo -= np.transpose(H.bb.vooo, (0, 1, 3, 2))

    H.bb.vvov = H0.bb.vvov + 0.5 * np.einsum('abef,fj->baje', H0.bb.vvvv, T.b, optimize=True)

    return H


def get_ccs_intermediates(T, H0):
    """Calculate the CCS-like similarity-transformed HBar intermediates (H_N e^T1)_C.

    Parameters
    ----------
    cc_t : dict
        Cluster amplitudes T1
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary

    Returns
    -------
    H1* : dict
        One-body HBar similarity-transformed intermediates. Sorted by occ/unocc blocks.
    H2* : dict
        Two-body HBar similarity-transformed intermediates. Sorted by occ/unocc blocks.
    """
    # H = Integral.fromEmpty(system, 2, data_type=np.type)

    from copy import deepcopy
    # Copy the Bare Hamiltonian object for T1-transforemd HBar
    H = deepcopy(H0)

    # 1-body components
    H.a.ov = H0.a.ov + np.einsum('mnef,fn->me', H0.aa.oovv, T.a, optimize=True) \
             + np.einsum('mnef,fn->me', H0.ab.oovv, T.b, optimize=True)

    H.b.ov = H0.b.ov + np.einsum('nmfe,fn->me', H0.ab.oovv, T.a, optimize=True) \
             + np.einsum('mnef,fn->me', H0.bb.oovv, T.b, optimize=True)

    H.a.vv = H0.a.vv + np.einsum('anef,fn->ae', H0.aa.vovv, T.a, optimize=True) \
             + np.einsum('anef,fn->ae', H0.ab.vovv, T.b, optimize=True) \
             - np.einsum('me,am->ae', H.a.ov, T.a, optimize=True)

    H.a.oo = H0.a.oo + np.einsum('mnif,fn->mi', H0.aa.ooov, T.a, optimize=True) \
             + np.einsum('mnif,fn->mi', H0.ab.ooov, T.b, optimize=True) \
             + np.einsum('me,ei->mi', H.a.ov, T.a, optimize=True)

    H.b.vv = H0.b.vv + np.einsum('anef,fn->ae', H0.bb.vovv, T.b, optimize=True) \
             + np.einsum('nafe,fn->ae', H0.ab.ovvv, T.a, optimize=True) \
             - np.einsum('me,am->ae', H.b.ov, T.b, optimize=True)

    H.b.oo = H0.b.oo + np.einsum('mnif,fn->mi', H0.bb.ooov, T.b, optimize=True) \
             + np.einsum('nmfi,fn->mi', H0.ab.oovo, T.a, optimize=True) \
             + np.einsum('me,ei->mi', H.b.ov, T.b, optimize=True)
    # 2-body components
    H.aa.oooo = 0.5 * H0.aa.oooo + np.einsum('mnej,ei->mnij', H0.aa.oovo, T.a, optimize=True) \
                + 0.5 * np.einsum('mnef,ei,fj->mnij', H0.aa.oovv, T.a, T.a, optimize=True)
    H.aa.oooo -= np.transpose(H.aa.oooo, (0, 1, 3, 2))

    H.aa.vvvv = 0.5 * H0.aa.vvvv - np.einsum('mbef,am->abef', H0.aa.ovvv, T.a, optimize=True) \
                + 0.5 * np.einsum('mnef,bn,am->abef', H0.aa.oovv, T.a, T.a, optimize=True)
    H.aa.vvvv -= np.transpose(H.aa.vvvv, (1, 0, 2, 3))

    H.aa.vooo = 0.5 * H0.aa.vooo - 0.25 * np.einsum('nmij,an->amij', H0.aa.oooo, T.a, optimize=True) \
                + 0.5 * np.einsum('amef,ei,fj->amij', H0.aa.vovv, T.a, T.a, optimize=True) \
                + np.einsum('amie,ej->amij', H0.aa.voov, T.a, optimize=True) \
                - 0.25 * np.einsum('nmef,fj,an,ei->amij', H0.aa.oovv, T.a, T.a, T.a, optimize=True) \
                - np.einsum('mnjf,fi,an->amij', H0.aa.ooov, T.a, T.a, optimize=True)
    H.aa.vooo -= np.transpose(H.aa.vooo, (0, 1, 3, 2))

    H.aa.vvov = H0.aa.vvov + 0.5 * np.einsum('abfe,fi->abie', H0.aa.vvvv, T.a, optimize=True)

    H.aa.voov = H0.aa.voov - np.einsum('nmie,an->amie', H0.aa.ooov, T.a, optimize=True) \
                + np.einsum('amfe,fi->amie', H0.aa.vovv, T.a, optimize=True) \
                - np.einsum('nmfe,fi,an->amie', H0.aa.oovv, T.a, T.a, optimize=True)

    H.aa.ooov = H0.aa.ooov + np.einsum('mnfe,fi->mnie', H0.aa.oovv, T.a, optimize=True)

    H.aa.vovv = H0.aa.vovv - np.einsum('mnfe,an->amef', H0.aa.oovv, T.a, optimize=True)

    H.ab.oooo = H0.ab.oooo + np.einsum('mnej,ei->mnij', H0.ab.oovo, T.a, optimize=True) \
                + np.einsum('mnif,fj->mnij', H0.ab.ooov, T.b, optimize=True) \
                + np.einsum('mnef,ei,fj->mnij', H0.ab.oovv, T.a, T.b, optimize=True)

    H.ab.vvvv = H0.ab.vvvv - np.einsum('mbef,am->abef', H0.ab.ovvv, T.a, optimize=True) \
                - np.einsum('anef,bn->abef', H0.ab.vovv, T.b, optimize=True) \
                + np.einsum('mnef,am,bn->abef', H0.ab.oovv, T.a, T.b, optimize=True)

    H.ab.voov = H0.ab.voov - np.einsum('nmie,an->amie', H0.ab.ooov, T.a, optimize=True) \
                + np.einsum('amfe,fi->amie', H0.ab.vovv, T.a, optimize=True) \
                - np.einsum('nmfe,fi,an->amie', H0.ab.oovv, T.a, T.a, optimize=True)

    H.ab.ovov = H0.ab.ovov + np.einsum('mafe,fi->maie', H0.ab.ovvv, T.a, optimize=True) \
                - np.einsum('mnie,an->maie', H0.ab.ooov, T.b, optimize=True) \
                - np.einsum('mnfe,an,fi->maie', H0.ab.oovv, T.b, T.a, optimize=True)

    H.ab.vovo = H0.ab.vovo - np.einsum('nmei,an->amei', H0.ab.oovo, T.a, optimize=True) \
                + np.einsum('amef,fi->amei', H0.ab.vovv, T.b, optimize=True) \
                - np.einsum('nmef,fi,an->amei', H0.ab.oovv, T.b, T.a, optimize=True)

    H.ab.ovvo = H0.ab.ovvo + np.einsum('maef,fi->maei', H0.ab.ovvv, T.b, optimize=True) \
                - np.einsum('mnei,an->maei', H0.ab.oovo, T.b, optimize=True) \
                - np.einsum('mnef,fi,an->maei', H0.ab.oovv, T.b, T.b, optimize=True)

    H.ab.ovoo = H0.ab.ovoo + np.einsum('mbej,ei->mbij', H0.ab.ovvo, T.a, optimize=True) \
                - np.einsum('mnij,bn->mbij', H0.ab.oooo, T.b, optimize=True) \
                - np.einsum('mnif,bn,fj->mbij', H0.ab.ooov, T.b, T.b, optimize=True) \
                - np.einsum('mnej,bn,ei->mbij', H0.ab.oovo, T.b, T.a, optimize=True) \
                + np.einsum('mbef,fj,ei->mbij', H0.ab.ovvv, T.b, T.a, optimize=True)

    H.ab.vooo = H0.ab.vooo + np.einsum('amif,fj->amij', H0.ab.voov, T.b, optimize=True) \
                - np.einsum('nmef,an,ei,fj->amij', H0.ab.oovv, T.a, T.a, T.b, optimize=True) \
                + np.einsum('amef,fj,ei->amij', H0.ab.vovv, T.b, T.a, optimize=True)

    H.ab.vvvo = H0.ab.vvvo + np.einsum('abef,fj->abej', H0.ab.vvvv, T.b, optimize=True) \
                - np.einsum('anej,bn->abej', H0.ab.vovo, T.b, optimize=True)

    H.ab.vvov = H0.ab.vvov - np.einsum('mbie,am->abie', H0.ab.ovov, T.a, optimize=True)

    H.ab.ooov = H0.ab.ooov + np.einsum('mnfe,fi->mnie', H0.ab.oovv, T.a, optimize=True)

    H.ab.oovo = H0.ab.oovo + np.einsum('nmef,fi->nmei', H0.ab.oovv, T.b, optimize=True)

    H.ab.vovv = H0.ab.vovv - np.einsum('nmef,an->amef', H0.ab.oovv, T.a, optimize=True)

    H.ab.ovvv = H0.ab.ovvv - np.einsum('mnfe,an->mafe', H0.ab.oovv, T.b, optimize=True)

    H.bb.oooo = 0.5 * H0.bb.oooo + np.einsum('mnie,ej->mnij', H0.bb.ooov, T.b, optimize=True) \
                + 0.5 * np.einsum('mnef,ei,fj->mnij', H0.bb.oovv, T.b, T.b, optimize=True)
    H.bb.oooo -= np.transpose(H.bb.oooo, (0, 1, 3, 2))

    H.bb.vvvv = 0.5 * H0.bb.vvvv - np.einsum('mbef,am->abef', H0.bb.ovvv, T.b, optimize=True) \
                + 0.5 * np.einsum('mnef,bn,am->abef', H0.bb.oovv, T.b, T.b, optimize=True)
    H.bb.vvvv -= np.transpose(H.bb.vvvv, (1, 0, 2, 3))

    H.bb.voov = H0.bb.voov - np.einsum('mnei,an->amie', H0.bb.oovo, T.b, optimize=True) \
                + np.einsum('amfe,fi->amie', H0.bb.vovv, T.b, optimize=True) \
                - np.einsum('mnef,fi,an->amie', H0.bb.oovv, T.b, T.b, optimize=True)

    H.bb.vooo = 0.5 * H0.bb.vooo - 0.25 * np.einsum('mnij,bn->bmji', H0.bb.oooo, T.b, optimize=True) \
                + 0.5 * np.einsum('mbef,ei,fj->bmji', H0.bb.ovvv, T.b, T.b, optimize=True) \
                - 0.25 * np.einsum('mnef,fj,ei,bn->bmji', H0.bb.oovv, T.b, T.b, T.b, optimize=True) \
                + np.einsum('mbif,fj->bmji', H0.bb.ovov, T.b, optimize=True) \
                - np.einsum('mnjf,fi,an->amij', H0.bb.ooov, T.b, T.b, optimize=True)
    H.bb.vooo -= np.transpose(H.bb.vooo, (0, 1, 3, 2))

    H.bb.vvov = H0.bb.vvov + 0.5 * np.einsum('abef,fj->baje', H0.bb.vvvv, T.b, optimize=True)

    H.bb.ooov = H0.bb.ooov + np.einsum('mnfe,fi->mnie', H0.bb.oovv, T.b, optimize=True)

    H.bb.vovv = H0.bb.vovv - np.einsum('mnfe,an->amef', H0.bb.oovv, T.b, optimize=True)

    return H
