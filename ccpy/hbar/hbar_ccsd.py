import time

import numpy as np


def build_hbar_ccsd(T, H0):

    from copy import deepcopy

    # Copy the Bare Hamiltonian object for T1/T2-similarity transformed HBar
    H = deepcopy(H0)
    
    H.a.ov += (
                np.einsum("imae,em->ia", H0.aa.oovv, T.a, optimize=True)
                + np.einsum("imae,em->ia", H0.ab.oovv, T.b, optimize=True)
    )

    H.a.oo += (
                np.einsum("je,ei->ji", H.a.ov, T.a, optimize=True)
                + np.einsum("jmie,em->ji", H0.aa.ooov, T.a, optimize=True)
                + np.einsum("jmie,em->ji", H0.ab.ooov, T.b, optimize=True)
                + 0.5 * np.einsum("jnef,efin->ji", H0.aa.oovv, T.aa, optimize=True)
                + np.einsum("jnef,efin->ji", H0.ab.oovv, T.ab, optimize=True)
    )

    H.a.vv += (
                - np.einsum("mb,am->ab", H.a.ov, T.a, optimize=True)
                + np.einsum("ambe,em->ab", H0.aa.vovv, T.a, optimize=True)
                + np.einsum("ambe,em->ab", H0.ab.vovv, T.b, optimize=True)
                - 0.5 * np.einsum("mnbf,afmn->ab", H0.aa.oovv, T.aa, optimize=True)
                - np.einsum("mnbf,afmn->ab", H0.ab.oovv, T.ab, optimize=True)
    )
    
    H.b.ov += (
                np.einsum("imae,em->ia", H0.bb.oovv, T.b, optimize=True)
                + np.einsum("miea,em->ia", H0.ab.oovv, T.a, optimize=True)
    )

    H.b.oo += (
                np.einsum("je,ei->ji", H.b.ov, T.b, optimize=True)
                + np.einsum("jmie,em->ji", H0.bb.ooov, T.b, optimize=True)
                + np.einsum("mjei,em->ji", H0.ab.oovo, T.a, optimize=True)
                + 0.5 * np.einsum("jnef,efin->ji", H0.bb.oovv, T.bb, optimize=True)
                + np.einsum("njfe,feni->ji", H0.ab.oovv, T.ab, optimize=True)
    )
    
    H.b.vv += (
                - np.einsum("mb,am->ab", H.b.ov, T.b, optimize=True)
                + np.einsum("ambe,em->ab", H0.bb.vovv, T.b, optimize=True)
                + np.einsum("maeb,em->ab", H0.bb.ovvv, T.a, optimize=True)
                - 0.5 * np.einsum("mnbf,afmn->ab", H0.bb.oovv, T.bb, optimize=True)
                - np.einsum("nmfb,fanm->ab", H0.ab.oovv, T.ab, optimize=True)
    )
    
    Q1 = -np.einsum("mnfe,an->amef", H0.aa.oovv, T.a, optimize=True)
    I2A_vovv = H0.aa.vovv + 0.5 * Q1
    H.aa.vovv = I2A_vovv + 0.5 * Q1

    Q1 = np.einsum("mnfe,fi->mnie", H0.aa.oovv, T.a, optimize=True)
    I2A_ooov = H0.aa.ooov + 0.5 * Q1
    H.aa.ooov = I2A_ooov + 0.5 * Q1

    Q1 = -np.einsum("nmef,an->amef", H0.ab.oovv, T.a, optimize=True)
    I2B_vovv = H0.ab.vovv + 0.5 * Q1
    H.ab.vovv = I2B_vovv + 0.5 * Q1

    Q1 = np.einsum("mnfe,fi->mnie", H0.ab.oovv, T.a, optimize=True)
    I2B_ooov = H0.ab.ooov + 0.5 * Q1
    H.ab.ooov = I2B_ooov + 0.5 * Q1

    Q1 = -np.einsum("mnef,an->maef", H0.ab.oovv, T.b, optimize=True)
    I2B_ovvv = H0.bb.ovvv + 0.5 * Q1
    H.ab.ovvv = I2B_ovvv + 0.5 * Q1

    Q1 = np.einsum("nmef,fi->nmei", H0.ab.oovv, T.b, optimize=True)
    I2B_oovo = H0.ab.oovo + 0.5 * Q1
    H.ab.oovo = I2B_oovo + 0.5 * Q1

    Q1 = -np.einsum("nmef,an->amef", H0.bb.oovv, T.b, optimize=True)
    I2C_vovv = H0.bb.vovv + 0.5 * Q1
    H.bb.vovv = I2C_vovv + 0.5 * Q1

    Q1 = np.einsum("mnfe,fi->mnie", H0.bb.oovv, T.b, optimize=True)
    I2C_ooov = H0.bb.ooov + 0.5 * Q1
    H.bb.ooov = I2C_ooov + 0.5 * Q1

    Q1 = -np.einsum("bmfe,am->abef", I2A_vovv, T.a, optimize=True)
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    H.aa.vvvv += 0.5 * np.einsum("mnef,abmn->abef", H0.aa.oovv, T.aa, optimize=True) + Q1

    H.ab.vvvv += (
                - np.einsum("mbef,am->abef", I2B_ovvv, T.a, optimize=True)
                - np.einsum("amef,bm->abef", I2B_vovv, T.b, optimize=True)
                + np.einsum("mnef,abmn->abef", H0.ab.oovv, T.ab, optimize=True)
    )

    Q1 = -np.einsum("bmfe,am->abef", I2C_vovv, T.b, optimize=True)
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    H.bb.vvvv += 0.5 * np.einsum("mnef,abmn->abef", H0.bb.oovv, T.bb, optimize=True) + Q1

    Q1 = +np.einsum("nmje,ei->mnij", I2A_ooov, T.a, optimize=True)
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    H.aa.oooo += 0.5 * np.einsum("mnef,efij->mnij", H0.aa.oovv, T.aa, optimize=True) + Q1

    H.ab.oooo += (
                np.einsum("mnej,ei->mnij", I2B_oovo, T.a, optimize=True)
                + np.einsum("mnie,ej->mnij", I2B_ooov, T.b, optimize=True)
                + np.einsum("mnef,efij->mnij", H0.ab.oovv, T.ab, optimize=True)
    )

    Q1 = +np.einsum("nmje,ei->mnij", I2C_ooov, T.b, optimize=True)
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    H.bb.oooo += 0.5 * np.einsum("mnef,efij->mnij", H0.bb.oovv, T.bb, optimize=True) + Q1

    H.aa.voov += (
                np.einsum("amfe,fi->amie", I2A_vovv, T.a, optimize=True)
                - np.einsum("nmie,an->amie", I2A_ooov, T.a, optimize=True)
                + np.einsum("nmfe,afin->amie", H0.aa.oovv, T.aa, optimize=True)
                + np.einsum("mnef,afin->amie", H0.ab.oovv, T.ab, optimize=True)
    )

    H.ab.voov += (
                np.einsum("amfe,fi->amie", I2B_vovv, T.a, optimize=True)
                - np.einsum("nmie,an->amie", I2B_ooov, T.a, optimize=True)
                + np.einsum("nmfe,afin->amie", H0.ab.oovv, T.aa, optimize=True)
                + np.einsum("nmfe,afin->amie", H0.bb.oovv, T.ab, optimize=True)
    )

    H.ab.ovvo += (
                np.einsum("maef,fi->maei", I2B_ovvv, T.b, optimize=True)
                - np.einsum("mnei,an->maei", I2B_oovo, T.b, optimize=True)
                + np.einsum("mnef,afin->maei", H0.ab.oovv, T.bb, optimize=True)
                + np.einsum("mnef,fani->maei", H0.aa.oovv, T.ab, optimize=True)
    )

    H.ab.ovov += (
                np.einsum("mafe,fi->maie", I2B_ovvv, T.a, optimize=True)
                - np.einsum("mnie,an->maie", I2B_ooov, T.b, optimize=True)
                - np.einsum("mnfe,fain->maie", H0.ab.oovv, T.ab, optimize=True)
    )

    H.ab.vovo += (
                - np.einsum("nmei,an->amei", I2B_oovo, T.a, optimize=True)
                + np.einsum("amef,fi->amei", I2B_vovv, T.b, optimize=True)
                - np.einsum("nmef,afni->amei", H0.ab.oovv, T.ab, optimize=True)
    )

    H.bb.voov += (
                np.einsum("amfe,fi->amie", I2C_vovv, T.b, optimize=True)
                - np.einsum("nmie,an->amie", I2C_ooov, T.b, optimize=True)
                + np.einsum("nmfe,afin->amie", H0.bb.oovv, T.bb, optimize=True)
                + np.einsum("nmfe,fani->amie", H0.ab.oovv, T.ab, optimize=True)
    )

    Q1 = (
        np.einsum("mnjf,afin->amij", H.aa.ooov, T.aa, optimize=True)
        + np.einsum("mnjf,afin->amij", H.ab.ooov, T.ab, optimize=True)
    )
    Q2 = H0.aa.voov + 0.5 * np.einsum("amef,ei->amif", H0.aa.vovv, T.a, optimize=True)
    Q2 = np.einsum("amif,fj->amij", Q2, T.a, optimize=True)
    Q1 += Q2
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    H.aa.vooo += Q1 + (
                np.einsum("me,aeij->amij", H.a.ov, T.aa, optimize=True)
                - np.einsum("nmij,an->amij", H.aa.oooo, T.a, optimize=True)
                + 0.5 * np.einsum("amef,efij->amij", H0.aa.vovv, T.aa, optimize=True)
    )

    Q1 = H0.ab.voov + np.einsum("amfe,fi->amie", H0.ab.vovv, T.a, optimize=True)
    H.ab.vooo += (
                np.einsum("me,aeij->amij", H.b.ov, T.ab, optimize=True)
                - np.einsum("nmij,an->amij", H.ab.oooo, T.a, optimize=True)
                + np.einsum("mnjf,afin->amij", H.bb.ooov, T.ab, optimize=True)
                + np.einsum("nmfj,afin->amij", H.ab.oovo, T.aa, optimize=True)
                - np.einsum("nmif,afnj->amij", H.ab.ooov, T.ab, optimize=True)
                + np.einsum("amej,ei->amij", H0.ab.vovo, T.a, optimize=True)
                + np.einsum("amie,ej->amij", Q1, T.b, optimize=True)
                + np.einsum("amef,efij->amij", H0.ab.vovv, T.ab, optimize=True)
    )

    Q1 = H0.ab.ovov + np.einsum("mafe,fj->maje", H0.bb.ovvv, T.a, optimize=True)
    H.ab.ovoo += (
                np.einsum("me,eaji->maji", H.a.ov, T.ab, optimize=True)
                - np.einsum("mnji,an->maji", H.ab.oooo, T.b, optimize=True)
                + np.einsum("mnjf,fani->maji", H.aa.ooov, T.ab, optimize=True)
                + np.einsum("mnjf,fani->maji", H.ab.ooov, T.bb, optimize=True)
                - np.einsum("mnfi,fajn->maji", H.ab.oovo, T.ab, optimize=True)
                + np.einsum("maje,ei->maji", Q1, T.b, optimize=True)
                + np.einsum("maei,ej->maji", H0.ab.ovvo, T.a, optimize=True)
                + np.einsum("mafe,feji->maji", H0.bb.ovvv, T.ab, optimize=True)
    )

    Q1 = (
        np.einsum("mnjf,afin->amij", H.bb.ooov, T.bb, optimize=True) 
        + np.einsum("nmfj,fani->amij", H.ab.oovo, T.ab, optimize=True)
    )
    Q2 = H0.bb.voov + 0.5 * np.einsum("amef,ei->amif", H0.bb.vovv, T.b, optimize=True)
    Q2 = np.einsum("amif,fj->amij", Q2, T.b, optimize=True)
    Q1 += Q2
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    H.bb.vooo += Q1 + (
                + np.einsum("me,aeij->amij", H.b.ov, T.bb, optimize=True)
                - np.einsum("nmij,an->amij", H.bb.oooo, T.b, optimize=True)
                + 0.5 * np.einsum("amef,efij->amij", H0.bb.vovv, T.bb, optimize=True)
    )

    Q1 = (
        np.einsum("bnef,afin->abie", H.aa.vovv, T.aa, optimize=True)
        + np.einsum("bnef,afin->abie", H.ab.vovv, T.ab, optimize=True)
    )
    Q2 = H0.aa.ovov - 0.5 * np.einsum("mnie,bn->mbie", H0.aa.ooov, T.a, optimize=True)
    Q2 = -np.einsum("mbie,am->abie", Q2, T.a, optimize=True)
    Q1 += Q2
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    H.aa.vvov += Q1 + (
                - np.einsum("me,abim->abie", H.a.ov, T.aa, optimize=True)
                + np.einsum("abfe,fi->abie", H.aa.vvvv, T.a, optimize=True)
                + 0.5 * np.einsum("mnie,abmn->abie", H0.aa.ooov, T.aa, optimize=True)
    )

    Q1 = H0.ab.ovov - np.einsum("mnie,bn->mbie", H0.ab.ooov, T.b, optimize=True)
    Q1 = -np.einsum("mbie,am->abie", Q1, T.a, optimize=True)
    H.ab.vvov += Q1 + (
                - np.einsum("me,abim->abie", H.b.ov, T.ab, optimize=True)
                + np.einsum("abfe,fi->abie", H.ab.vvvv, T.a, optimize=True)
                + np.einsum("nbfe,afin->abie", H.ab.ovvv, T.aa, optimize=True)
                + np.einsum("bnef,afin->abie", H.bb.vovv, T.ab, optimize=True)
                - np.einsum("amfe,fbim->abie", H.ab.vovv, T.ab, optimize=True)
                - np.einsum("amie,bm->abie", H0.ab.voov, T.b, optimize=True)
                + np.einsum("nmie,abnm->abie", H0.ab.ooov, T.ab, optimize=True)
    )

    Q1 = H0.ab.vovo - np.einsum("nmei,bn->bmei", H0.ab.oovo, T.a, optimize=True)
    Q1 = -np.einsum("bmei,am->baei", Q1, T.b, optimize=True)
    H.ab.vvvo += Q1 + (
                - np.einsum("me,bami->baei", H.a.ov, T.ab, optimize=True)
                + np.einsum("baef,fi->baei", H.ab.vvvv, T.b, optimize=True)
                + np.einsum("bnef,fani->baei", H.aa.vovv, T.ab, optimize=True)
                + np.einsum("bnef,fani->baei", H.ab.vovv, T.bb, optimize=True)
                - np.einsum("maef,bfmi->baei", H.ab.ovvv, T.ab, optimize=True)
                - np.einsum("naei,bn->baei", H0.ab.ovvo, T.a, optimize=True)
                + np.einsum("nmei,banm->baei", H0.ab.oovo, T.ab, optimize=True)
    )

    Q1 = (
          np.einsum("bnef,afin->abie", H.bb.vovv, T.bb, optimize=True)
         + np.einsum("nbfe,fani->abie", H.ab.ovvv, T.ab, optimize=True)
    )
    Q2 = H.bb.ovov - 0.5 * np.einsum("mnie,bn->mbie", H0.bb.ooov, T.b, optimize=True)
    Q2 = -np.einsum("mbie,am->abie", Q2, T.b, optimize=True)
    Q1 += Q2
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    H.bb.vvov += Q1 + (
                - np.einsum("me,abim->abie", H.b.ov, T.bb, optimize=True)
                + np.einsum("abfe,fi->abie", H.bb.vvvv, T.b, optimize=True)
                + 0.5 * np.einsum("mnie,abmn->abie", H0.bb.ooov, T.bb, optimize=True)
    )

    return H

if __name__ == "__main__":
    from pyscf import gto, scf

    from ccpy.models.calculation import Calculation
    from ccpy.interfaces.pyscf_tools import load_pyscf_integrals
    from ccpy.drivers.cc import driver

    mol = gto.Mole()
    mol.build(
        atom="""F 0.0 0.0 -2.66816
                F 0.0 0.0  2.66816""",
        basis="ccpvdz",
        charge=0,
        spin=0,
        symmetry="D2H",
        cart=True,
        unit='Bohr',
    )
    mf = scf.ROHF(mol)
    mf.kernel()

    system, H0 = load_pyscf_integrals(mf, nfrozen=2)
    system.print_info()

    calculation = Calculation(
        order=2,
        calculation_type="ccsd",
        convergence_tolerance=1.0e-08,
    )

    T, total_energy, is_converged = driver(calculation, system, H0)

    H = build_hbar_ccsd(T, H0)