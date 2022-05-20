import numpy as np

def get_ccs_intermediates(t1, f, g, o, v):
    """
    Calculate the CCS-like similarity-transformed HBar intermediates (H_N e^T1)_C.
    """

    norbitals = f.shape[0]

    H1 = np.zeros((norbitals, norbitals))
    H2 = np.zeros((norbitals, norbitals, norbitals, norbitals))

    # 1-body components
    H1[o, v] = f[o, v] + np.einsum("mnef,fn->me", g[o, o, v, v], t1, optimize=True)
    H1[v, v] = f[v, v] + (
        np.einsum("anef,fn->ae", g[v, o, v, v], t1, optimize=True)
        - np.einsum("me,am->ae", H1[o, v], t1, optimize=True)
    ) 
    H1[o, o] = f[o, o] + (
        np.einsum("mnif,fn->mi", g[o, o, o, v], t1, optimize=True)
        + np.einsum("me,ei->mi", H1[o, v], t1, optimize=True)
    ) 
    # 2-body components
    H2[o, o, o, v] = np.einsum("mnfe,fi->mnie", g[o, o, v, v], t1, optimize=True) 

    H2[o, o, o, o] = 0.5 * g[o, o, o, o] + np.einsum("nmje,ei->mnij", g[o, o, o, v] + 0.5 * H2[o, o, o, v], t1, optimize=True) # no(4)nu(1)
    H2[o, o, o, o] -= np.transpose(H2[o, o, o, o], (0, 1, 3, 2))

    H2[v, o, v, v] = -np.einsum("mnfe,an->amef", g[o, o, v, v], t1, optimize=True) # no(2)nu(3)

    H2[v, o, o, v] = g[v, o, o, v] + (
              np.einsum("amfe,fi->amie", g[v, o, v, v] + 0.5 * H2[v, o, v, v], t1, optimize=True)
            - np.einsum("nmie,an->amie", g[o, o, o, v] + 0.5 * H2[o, o, o, v], t1, optimize=True)
    )

    L_amie = g[v, o, o, v] + 0.5 * np.einsum('amef,ei->amif', g[v, o, v, v], t1, optimize=True) # no(2)nu(3)
    X_mnij = g[o, o, o, o] + np.einsum('mnie,ej->mnij', H2[o, o, o, v], t1, optimize=True) # no(4)nu(1)
    H2[v, o, o, o] = 0.5 * g[v, o, o, o] + (
        np.einsum('amie,ej->amij', L_amie, t1, optimize=True)
       -0.25 * np.einsum('mnij,am->anij', X_mnij, t1, optimize=True)
    ) 
    H2[v, o, o, o] -= np.transpose(H2[v, o, o, o], (0, 1, 3, 2))

    L_amie = np.einsum('mnie,am->anie', g[o, o, o, v], t1, optimize=True)
    H2[v, v, o, v] = g[v, v, o, v] + np.einsum("anie,bn->abie", L_amie, t1, optimize=True) # no(1)nu(4)

    return H1, H2


def get_ccsd_intermediates(t1, t2, f, g, o, v):
    """Calculate the CCSD-like intermediates for CCSDT. This routine
    should only calculate terms with T2 and any remaining terms outside of the CCS intermediate
    routine."""

    norbitals = f.shape[0]

    H1 = np.zeros((norbitals, norbitals))
    H2 = np.zeros((norbitals, norbitals, norbitals, norbitals))

    # 1-body components
    H1[o, v] = f[o, v] + np.einsum("imae,em->ia", g[o, o, v, v], t1, optimize=True)

    H1[o, o] = f[o, o] + (
            np.einsum("je,ei->ji", H1[o, v], t1, optimize=True)
            + np.einsum("jmie,em->ji", g[o, o, o, v], t1, optimize=True)
            + 0.5 * np.einsum("jnef,efin->ji", g[o, o, v, v], t2, optimize=True)
    )

    H1[v, v] = f[v, v] + (
            - np.einsum("mb,am->ab", H1[o, v], t1, optimize=True)
            + np.einsum("ambe,em->ab", g[v, o, v, v], t1, optimize=True)
            - 0.5 * np.einsum("mnbf,afmn->ab", g[o, o, v, v], t2, optimize=True)
    )

    # 2-body components
    Q1 = -np.einsum("mnfe,an->amef", g[o, o, v, v], t1, optimize=True)
    I_vovv = g[v, o, v, v] + 0.5 * Q1
    H2[v, o, v, v] = I_vovv + 0.5 * Q1

    Q1 = np.einsum("mnfe,fi->mnie", g[o, o, v, v], t1, optimize=True)
    I_ooov = g[o, o, o, v] + 0.5 * Q1
    H2[o, o, o, v] = I_ooov + 0.5 * Q1

    Q1 = -np.einsum("bmfe,am->abef", I_vovv, t1, optimize=True)
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    H2[v, v, v, v] = g[v, v, v, v] + 0.5 * np.einsum("mnef,abmn->abef", g[o, o, v, v], t2, optimize=True) + Q1

    Q1 = +np.einsum("nmje,ei->mnij", I_ooov, t1, optimize=True)
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    H2[o, o, o, o] = g[o, o, o, o] + 0.5 * np.einsum("mnef,efij->mnij", g[o, o, v, v], t2, optimize=True) + Q1

    H2[v, o, o, v] = g[v, o, o, v] + (
            np.einsum("amfe,fi->amie", I_vovv, t1, optimize=True)
            - np.einsum("nmie,an->amie", I_ooov, t1, optimize=True)
            + np.einsum("nmfe,afin->amie", g[o, o, v, v], t2, optimize=True)
    )

    Q1 = np.einsum("mnjf,afin->amij", H2[o, o, o, v], t2, optimize=True)
    Q2 = g[v, o, o, v] + 0.5 * np.einsum("amef,ei->amif", g[v, o, v, v], t1, optimize=True)
    Q2 = np.einsum("amif,fj->amij", Q2, t1, optimize=True)
    Q1 += Q2
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    H2[v, o, o, o] = g[v, o, o, o] + Q1 + (
            np.einsum("me,aeij->amij", H1[o, v], t2, optimize=True)
            - np.einsum("nmij,an->amij", H2[o, o, o, o], t1, optimize=True)
            + 0.5 * np.einsum("amef,efij->amij", g[v, o, v, v], t2, optimize=True)
    )

    Q1 = np.einsum("bnef,afin->abie", H2[v, o, v, v], t2, optimize=True)
    Q2 = g[o, v, o, v] - 0.5 * np.einsum("mnie,bn->mbie", g[o, o, o, v], t1, optimize=True)
    Q2 = -np.einsum("mbie,am->abie", Q2, t1, optimize=True)
    Q1 += Q2
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    H2[v, v, o, v] = g[v, v, o, v] + Q1 + (
            - np.einsum("me,abim->abie", H1[o, v], t2, optimize=True)
            + np.einsum("abfe,fi->abie", H2[v, v, v, v], t1, optimize=True)
            + 0.5 * np.einsum("mnie,abmn->abie", g[o, o, o, v], t2, optimize=True)
    )

    return H1, H2

def get_ccsdt_intermediates(t1, t2, t3, f, g, o, v):
    """Calculate the CCSDT-like intermediates for CCSDTQ. This routine
    should only calculate terms with T2 and any remaining terms outside of the CCS intermediate
    routine."""

    norbitals = f.shape[0]

    H1 = np.zeros((norbitals, norbitals))
    H2 = np.zeros((norbitals, norbitals, norbitals, norbitals))

    # 1-body components
    H1[o, v] = f[o, v] + np.einsum("imae,em->ia", g[o, o, v, v], t1, optimize=True)

    H1[o, o] = f[o, o] + (
            np.einsum("je,ei->ji", H1[o, v], t1, optimize=True)
            + np.einsum("jmie,em->ji", g[o, o, o, v], t1, optimize=True)
            + 0.5 * np.einsum("jnef,efin->ji", g[o, o, v, v], t2, optimize=True)
    )

    H1[v, v] = f[v, v] + (
            - np.einsum("mb,am->ab", H1[o, v], t1, optimize=True)
            + np.einsum("ambe,em->ab", g[v, o, v, v], t1, optimize=True)
            - 0.5 * np.einsum("mnbf,afmn->ab", g[o, o, v, v], t2, optimize=True)
    )

    # 2-body components
    Q1 = -np.einsum("mnfe,an->amef", g[o, o, v, v], t1, optimize=True)
    I_vovv = g[v, o, v, v] + 0.5 * Q1
    H2[v, o, v, v] = I_vovv + 0.5 * Q1

    Q1 = np.einsum("mnfe,fi->mnie", g[o, o, v, v], t1, optimize=True)
    I_ooov = g[o, o, o, v] + 0.5 * Q1
    H2[o, o, o, v] = I_ooov + 0.5 * Q1

    Q1 = -np.einsum("bmfe,am->abef", I_vovv, t1, optimize=True)
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    H2[v, v, v, v] = g[v, v, v, v] + 0.5 * np.einsum("mnef,abmn->abef", g[o, o, v, v], t2, optimize=True) + Q1

    Q1 = +np.einsum("nmje,ei->mnij", I_ooov, t1, optimize=True)
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    H2[o, o, o, o] = g[o, o, o, o] + 0.5 * np.einsum("mnef,efij->mnij", g[o, o, v, v], t2, optimize=True) + Q1

    H2[v, o, o, v] = g[v, o, o, v] + (
            np.einsum("amfe,fi->amie", I_vovv, t1, optimize=True)
            - np.einsum("nmie,an->amie", I_ooov, t1, optimize=True)
            + np.einsum("nmfe,afin->amie", g[o, o, v, v], t2, optimize=True)
    )

    Q1 = np.einsum("mnjf,afin->amij", H2[o, o, o, v], t2, optimize=True)
    Q2 = g[v, o, o, v] + 0.5 * np.einsum("amef,ei->amif", g[v, o, v, v], t1, optimize=True)
    Q2 = np.einsum("amif,fj->amij", Q2, t1, optimize=True)
    Q1 += Q2
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    H2[v, o, o, o] = g[v, o, o, o] + Q1 + (
            np.einsum("me,aeij->amij", H1[o, v], t2, optimize=True)
            - np.einsum("nmij,an->amij", H2[o, o, o, o], t1, optimize=True)
            + 0.5 * np.einsum("amef,efij->amij", g[v, o, v, v], t2, optimize=True)
    )

    Q1 = np.einsum("bnef,afin->abie", H2[v, o, v, v], t2, optimize=True)
    Q2 = g[o, v, o, v] - 0.5 * np.einsum("mnie,bn->mbie", g[o, o, o, v], t1, optimize=True)
    Q2 = -np.einsum("mbie,am->abie", Q2, t1, optimize=True)
    Q1 += Q2
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    H2[v, v, o, v] = g[v, v, o, v] + Q1 + (
            - np.einsum("me,abim->abie", H1[o, v], t2, optimize=True)
            + np.einsum("abfe,fi->abie", H2[v, v, v, v], t1, optimize=True)
            + 0.5 * np.einsum("mnie,abmn->abie", g[o, o, o, v], t2, optimize=True)
    )

    # Add the T3 terms
    H2[v, o, o, o] += 0.5 * np.einsum("mnef,aefijn->amij", g[o, o, v, v], t3, optimize=True)
    H2[v, v, o, v] -= 0.5 * np.einsum("mnef,abfimn->abie", g[o, o, v, v], t3, optimize=True)

    return H1, H2
