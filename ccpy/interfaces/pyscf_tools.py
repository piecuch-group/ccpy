import numpy as np
from pyscf import ao2mo, symm

from ccpy.models.integrals import getHamiltonian
from ccpy.models.system import System
from ccpy.utilities.dumping import dumpIntegralstoPGFiles

from ccpy.drivers.hf_energy import calc_hf_frozen_core_energy


def load_pyscf_integrals(
        meanfield, nfrozen=0,
        num_act_holes_alpha=0, num_act_particles_alpha=0,
        num_act_holes_beta=0, num_act_particles_beta=0,
        normal_ordered=True, dump_integrals=False, sorted=True
):
    """Builds the System and Integral objects using the information contained within a PySCF
    mean-field object for a molecular system.

    Arguments:
    ----------
    meanFieldObj : Object -> PySCF SCF/mean-field object
    nfrozen : int -> number of frozen electrons
    Returns:
    ----------
    system: System object
    integrals: Integral object
    """
    molecule = meanfield.mol
    nelectrons = molecule.nelectron
    mo_coeff = meanfield.mo_coeff
    norbitals = mo_coeff.shape[1]
    nuclear_repulsion = molecule.energy_nuc()

    system = System(
        nelectrons,
        norbitals,
        molecule.spin + 1,  # PySCF mol.spin returns 2S, not S
        nfrozen,
        point_group=molecule.symmetry,
        orbital_symmetries = [x.upper() for x in symm.label_orb_symm(molecule, molecule.irrep_name, molecule.symm_orb, mo_coeff)],
        charge=molecule.charge,
        nuclear_repulsion=nuclear_repulsion,
        mo_energies=meanfield.mo_energy,
        mo_occupation=meanfield.mo_occ,
    )

    kinetic_aoints = molecule.intor_symmetric("int1e_kin")
    nuclear_aoints = molecule.intor_symmetric("int1e_nuc")
    e1int = np.einsum(
        "pi,pq,qj->ij", mo_coeff, kinetic_aoints + nuclear_aoints, mo_coeff
    )
    e2int = np.transpose(
        np.reshape(ao2mo.kernel(molecule, mo_coeff, compact=False), 4 * (norbitals,)),
        (0, 2, 1, 3),
    )

    # Check that the HF energy calculated using the integrals matches the PySCF result
    hf_energy = get_hf_energy(e1int, e2int, system, notation="physics")
    hf_energy += nuclear_repulsion

    if not np.allclose(hf_energy, meanfield.energy_tot(), atol=1.0e-06, rtol=0.0):
        raise RuntimeError("Integrals don't match mean field energy")

    system.reference_energy = hf_energy
    system.frozen_energy = calc_hf_frozen_core_energy(e1int, e2int, system)

    if dump_integrals:
        dumpIntegralstoPGFiles(e1int, e2int, system)

    return system, getHamiltonian(e1int, e2int, system, normal_ordered, sorted)



def get_kconserv1(a, kpts, thresh=1.0e-07):
    nkpts = len(kpts)
    kconserv = np.zeros(nkpts, dtype=np.int32)
    for p, kp in enumerate(kpts):
        for q, kq in enumerate(kpts):
            dk = kp - kq
            svec = np.einsum("i,xi->x", dk, a / (2.0 * np.pi))
            if np.linalg.norm(svec - np.rint(svec)) < thresh:
                kconserv[p] = q
    return kconserv


def get_kconserv2(a, kpts, thresh=1.0e-07):
    nkpts = len(kpts)
    kconserv = np.zeros((nkpts, nkpts, nkpts), dtype=np.int32)
    for p, kp in enumerate(kpts):
        for q, kq in enumerate(kpts):
            for r, kr in enumerate(kpts):
                for s, ks in enumerate(kpts):
                    dk = kp + kq - kr - ks
                    svec = np.einsum("i,xi->x", dk, a / (2.0 * np.pi))
                    if np.linalg.norm(svec - np.rint(svec)) < thresh:
                        kconserv[p, q, r] = s
    return kconserv


def get_kpoints(cell, nk, G):
    kpts = cell.make_kpts(nk)
    nkpts = len(kpts)
    for i in range(nkpts):
        kpts[i, :] += G

    return kpts


def get_pbc_mo_integrals(cell, kmf, kpts, notation="chemist"):
    from pyscf.pbc import df, tools

    if kmf.exxdiv == "ewald":
        madelung = tools.pbc.madelung(cell, kpts)
        nkpts = len(kpts)
        nelectron = float(cell.tot_electrons(nkpts)) / nkpts
        e_ewald = -0.5 * madelung * nelectron
    else:
        e_ewald = 0.0

    nmo = kmf.mo_coeff[0].shape[1]
    nkpts = len(kpts)
    e_nuc = cell.energy_nuc()

    # k-point conservation array
    kconserv1 = get_kconserv1(cell.lattice_vectors(), kpts)
    kconserv2 = get_kconserv2(cell.lattice_vectors(), kpts)

    kinetic = cell.pbc_intor("cint1e_kin_sph", kpts=kpts)
    nuclear = df.FFTDF(cell).get_pp(kpts)
    hcore_ao = [x + y for x, y in zip(kinetic, nuclear)]
    Z = np.zeros((nkpts, nkpts, nmo, nmo), dtype=np.complex128)
    for kp in range(nkpts):
        kq = kconserv1[kp]
        z0 = np.einsum("pj,pi->ij", hcore_ao[kp], kmf.mo_coeff[kp].conj())
        z0 = np.einsum("ip,pj->ij", z0, kmf.mo_coeff[kp])
        # 1-electron integrals do not scale with Nkpt
        Z[kp, kq, :, :] = z0

    V = np.zeros((nkpts, nkpts, nkpts, nkpts, nmo, nmo, nmo, nmo), dtype=np.complex128)
    for kp in range(nkpts):
        for kq in range(nkpts):
            for kr in range(nkpts):

                if notation == "chemist":  # chemist notation
                    ks = kconserv2[kp, kq, kr]
                    # Get the ERI integrals from kmf object and convert to MO basis
                    eri_kpt = kmf.with_df.ao2mo(
                        [kmf.mo_coeff[i] for i in (kp, kq, kr, ks)],
                        [kpts[i] for i in (kp, kq, kr, ks)],
                        compact=False,
                    )
                    # Store the integrals. Don't forget the 1/Nkpt scaling of 2-body ints!
                    V[kp, kq, kr, ks, :, :, :, :] = (
                            1.0 / nkpts * np.reshape(eri_kpt, (nmo, nmo, nmo, nmo))
                    )
                else:  # physics notation
                    ks = kconserv2[kp, kr, kq]
                    # Get the ERI integrals from kmf object and convert to MO basis
                    eri_kpt = kmf.with_df.ao2mo(
                        [kmf.mo_coeff[i] for i in (kp, kq, kr, ks)],
                        [kpts[i] for i in (kp, kq, kr, ks)],
                        compact=False,
                    )
                    # Store the integrals. Don't forget the 1/Nkpt scaling of 2-body ints!
                    V[kp, kr, kq, ks, :, :, :, :] = (
                            1.0
                            / nkpts
                            * np.transpose(
                        np.reshape(eri_kpt, (nmo, nmo, nmo, nmo)), (0, 2, 1, 3)
                    )
                    )

    # Calculate reference energy using extracted MO integrals
    e_hf = calc_khf_energy(Z, V, cell.nelectron, nkpts, notation)
    e_calc = e_hf + e_nuc + e_ewald
    print("")
    print("PBC MO Integral Conversion Summary:")
    print("-----------------------------------")
    print("HF electronic energy = ", np.real(e_calc))
    print("Nuclear repulsion energy = ", np.real(e_nuc))
    print("Ewald energy shift (exxdiv = {}) = {}".format(kmf.exxdiv, np.real(e_ewald)))
    print("Total energy = ", np.real(e_calc))
    print("-----------------------------------")
    # Verify that the result is the same as that produced by PySCF
    assert np.allclose(e_calc, kmf.energy_tot(), atol=1.0e-06, rtol=0.0)

    return Z, V, e_nuc


def calc_khf_energy(e1int, e2int, Nelec, Nkpts, notation):
    # Note that any V must have a factor of 1/Nkpts!
    e1a = 0.0
    e1b = 0.0
    e2a = 0.0
    e2b = 0.0
    e2c = 0.0

    if Nelec % 2 == 0:
        Nocc_a = int(Nelec / 2)
        Nocc_b = int(Nelec / 2)
    else:
        Nocc_a = int((Nelec + 1) / 2)
        Nocc_b = int((Nelec - 1) / 2)

    # slices
    oa = slice(0, Nocc_a)
    ob = slice(0, Nocc_b)

    if notation == "chemist":
        e1a = np.einsum("uuii->", e1int[:, :, oa, oa])
        e1b = np.einsum("uuii->", e1int[:, :, ob, ob])
        e2a = 0.5 * (
                np.einsum("uuvviijj->", e2int[:, :, :, :, oa, oa, oa, oa])
                - np.einsum("uvvuijji->", e2int[:, :, :, :, oa, oa, oa, oa])
        )
        e2b = 1.0 * (np.einsum("uuvviijj->", e2int[:, :, :, :, oa, ob, oa, ob]))
        e2c = 0.5 * (
                np.einsum("uuvviijj->", e2int[:, :, :, :, ob, ob, ob, ob])
                - np.einsum("uvvuijji->", e2int[:, :, :, :, ob, ob, ob, ob])
        )
    else:  # physicist notation
        e1a = np.einsum("uuii->", e1int[:, :, oa, oa])
        e1b = np.einsum("uuii->", e1int[:, :, ob, ob])
        e2a = 0.5 * (
                np.einsum("uvuvijij->", e2int[:, :, :, :, oa, oa, oa, oa])
                - np.einsum("uvvuijji->", e2int[:, :, :, :, oa, oa, oa, oa])
        )
        e2b = 1.0 * (np.einsum("uvuvijij->", e2int[:, :, :, :, oa, ob, oa, ob]))
        e2c = 0.5 * (
                np.einsum("uvuvijij->", e2int[:, :, :, :, ob, ob, ob, ob])
                - np.einsum("uvvuijji->", e2int[:, :, :, :, ob, ob, ob, ob])
        )

    Escf = e1a + e1b + e2a + e2b + e2c

    return np.real(Escf) / Nkpts


def get_sc_mo_integrals(supcell, kmf, G, notation="chemist"):
    from pyscf.pbc import df, tools

    if kmf.exxdiv == "ewald":
        madelung = tools.pbc.madelung(supcell, G)
        nkpts = 1
        nelectron = float(supcell.tot_electrons(nkpts)) / nkpts
        e_ewald = -0.5 * madelung * nelectron
    else:
        e_ewald = 0.0

    if len(kmf.mo_coeff) > 0:
        mo_coeff = kmf.mo_coeff[0]
    else:
        mo_coeff = kmf.mo_coeff
    nmo = mo_coeff.shape[1]
    e_nuc = supcell.energy_nuc()

    kinetic = supcell.pbc_intor("cint1e_kin_sph", kpts=G)
    nuclear = df.FFTDF(supcell).get_pp(G)
    z0 = np.einsum("pj,pi->ij", kinetic + nuclear, mo_coeff.conj())
    Z = np.einsum("ip,pj->ij", z0, mo_coeff)

    eri_kpt = kmf.with_df.ao2mo(mo_coeff, G, compact=False)
    V = np.reshape(eri_kpt, (nmo, nmo, nmo, nmo))
    if notation != "chemist":
        V = np.transpose(V, (0, 2, 1, 3))

    e_hf = get_hf_energy(Z, V, supcell.nelectron, notation)
    e_calc = e_hf + e_nuc + e_ewald
    print("")
    print("PBC MO Integral Conversion Summary:")
    print("-----------------------------------")
    print("HF electronic energy = ", np.real(e_calc))
    print("Nuclear repulsion energy = ", np.real(e_nuc))
    print("Ewald energy shift (exxdiv = {}) = {}".format(kmf.exxdiv, np.real(e_ewald)))
    print("Total energy = ", np.real(e_calc))
    print("-----------------------------------")

    assert np.allclose(e_calc, kmf.energy_tot(), atol=1.0e-06, rtol=0.0)

    return Z, V, e_nuc


def get_mo_integrals(mol, mf, notation="chemist"):
    from pyscf import ao2mo

    nmo = mf.mo_coeff.shape[1]
    e_nuc = mol.energy_nuc()

    kinetic_ao = mol.intor_symmetric("int1e_kin")
    nuclear_ao = mol.intor_symmetric("int1e_nuc")
    Z = np.einsum("pi,pq,qj->ij", mf.mo_coeff, kinetic_ao + nuclear_ao, mf.mo_coeff)

    V = np.reshape(ao2mo.kernel(mol, mf.mo_coeff, compact=False), (nmo, nmo, nmo, nmo))
    if notation != "chemist":  # physics notation
        V = np.transpose(V, (0, 2, 1, 3))

    e_calc = get_hf_energy(Z, V, mol.nelectron, notation)
    e_calc += e_nuc
    assert np.allclose(e_calc, mf.energy_tot(), atol=1.0e-06, rtol=0.0)

    return Z, V, e_nuc


def get_hf_energy(e1int, e2int, system, notation="chemist"):
    oa = slice(0, system.nfrozen + system.noccupied_alpha)
    ob = slice(0, system.nfrozen + system.noccupied_beta)

    if notation == "chemist":
        e1a = np.einsum("ii->", e1int[oa, oa])
        e1b = np.einsum("ii->", e1int[ob, ob])
        e2a = 0.5 * (
                np.einsum("iijj->", e2int[oa, oa, oa, oa])
                - np.einsum("ijji->", e2int[oa, oa, oa, oa])
        )
        e2b = 1.0 * (np.einsum("iijj->", e2int[oa, ob, oa, ob]))
        e2c = 0.5 * (
                np.einsum("iijj->", e2int[ob, ob, ob, ob])
                - np.einsum("ijji->", e2int[ob, ob, ob, ob])
        )

    elif notation == "physics":
        e1a = np.einsum("ii->", e1int[oa, oa])
        e1b = np.einsum("ii->", e1int[ob, ob])
        e2a = 0.5 * (
                np.einsum("ijij->", e2int[oa, oa, oa, oa])
                - np.einsum("ijji->", e2int[oa, oa, oa, oa])
        )
        e2b = 1.0 * (np.einsum("ijij->", e2int[oa, ob, oa, ob]))
        e2c = 0.5 * (
                np.einsum("ijij->", e2int[ob, ob, ob, ob])
                - np.einsum("ijji->", e2int[ob, ob, ob, ob])
        )

    else:
        raise RuntimeError("Integral notation not recognized")

    E_scf = e1a + e1b + e2a + e2b + e2c

    return E_scf


def write_onebody_pbc_integrals(Z):
    Nkpts = Z.shape[0]
    Norb = Z.shape[2]
    with open("onebody.inp", "w") as f:
        for kp in range(Nkpts):
            for kq in range(Nkpts):
                ct = 1
                for i in range(Norb):
                    for j in range(i + 1):
                        f.write("     {}    {:.11f}\n".format(Z[kp, kq, i, j], ct))
                        ct += 1


def write_twobody_pbc_integrals(V, e_nuc):
    # inefficient. we should be saving only those V(kp,kq,kr,ks) that
    # obey symmetry constraint
    Nkpts = V.shape[0]
    Norb = V.shape[4]
    with open("twobody.inp", "w") as f:
        for kp in range(Nkpts):
            for kq in range(Nkpts):
                for kr in range(Nkpts):
                    for ks in range(Nkpts):
                        for i in range(Norb):
                            for j in range(Norb):
                                for k in range(Norb):
                                    for l in range(Norb):
                                        f.write(
                                            "    {}    {}    {}    {}       {:.11f}\n".format(
                                                i + 1,
                                                j + 1,
                                                k + 1,
                                                l + 1,
                                                V[kp, kq, kr, ks, i, j, k, l],
                                            )
                                        )
        f.write("    {}    {}    {}    {}        {:.11f}\n".format(0, 0, 0, 0, e_nuc))


def write_onebody_integrals(Z):
    Norb = Z.shape[0]
    with open("onebody.inp", "w") as f:
        ct = 1
        for i in range(Norb):
            for j in range(i + 1):
                f.write("     {}    {:.11f}\n".format(Z[i, j], ct))
                ct += 1


def write_twobody_integrals(V, e_nuc):
    Norb = V.shape[0]
    with open("twobody.inp", "w") as f:
        for i in range(Norb):
            for j in range(Norb):
                for k in range(Norb):
                    for l in range(Norb):
                        f.write(
                            "    {}    {}    {}    {}       {:.11f}\n".format(
                                i + 1, j + 1, k + 1, l + 1, V[i, j, k, l]
                            )
                        )
        f.write("    {}    {}    {}    {}        {:.11f}\n".format(0, 0, 0, 0, e_nuc))


def get_dipole_integrals(mol, mf):

    charges = mol.atom_charges()
    coords = mol.atom_coords()
    nuc_charge_center = np.einsum('z,zx->x', charges, coords) / charges.sum()
    mol.set_common_orig_(nuc_charge_center)
    dip_ints_ao = mol.intor_symmetric('cint1e_r_sph', comp=3)

    dip_ints_mo = np.einsum("xij,ip,jq->xpq", dip_ints_ao, mf.mo_coeff, mf.mo_coeff, optimize=True)

    return dip_ints_mo