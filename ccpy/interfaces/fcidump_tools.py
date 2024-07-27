import numpy as np

def load_fcidump_integrals(
    fcidump_file,
    nfrozen=0,
    ndelete=0,
    charge=0,
    rohf_canonicalization="Guest-Saunders",
    normal_ordered=True,
    sorted=True,
    data_type=np.float64,
):
    from ccpy.energy.hf_energy import calc_hf_energy, calc_hf_frozen_core_energy
    from ccpy.models.integrals import getHamiltonian
    from ccpy.models.system import System

    # Read nelectrons & norbitals directly from fcidump in order to accomodate ECP GAMESS runs
    norbitals, nelectrons, ms2 = load_system_params_from_fcidump(fcidump_file)

    system = System(
        nelectrons,
        norbitals,
        ms2 + 1,
        nfrozen,
        ndelete=ndelete,
        point_group="C1",
        orbital_symmetries=["A" for _ in range(norbitals)],
        charge=charge,
        nuclear_repulsion=0.0,
        mo_energies=[0.0 for _ in range(norbitals)],
    )

    # Load using onebody and twobody direct integral files
    e1int, e2int, nuclear_repulsion = load_integrals_from_fcidump(fcidump_file, system)
    system.nuclear_repulsion = nuclear_repulsion
    # obtain the MO energies via Fock matrix
    system.mo_energies, _ = build_rohf_fock(e1int, e2int, system, rohf_canonicalization)

    # Check that the HF energy calculated using the integrals matches the GAMESS result
    hf_energy = calc_hf_energy(e1int, e2int, system)
    hf_energy += system.nuclear_repulsion

    system.reference_energy = hf_energy
    system.frozen_energy = calc_hf_frozen_core_energy(e1int, e2int, system)
    return system, getHamiltonian(e1int, e2int, system, normal_ordered, sorted)

def load_system_params_from_fcidump(fcidump):
    """This function parses and returns the values printed in the top line of
    the FCIDUMP file, which lists the number of orbitals (NORB), the number of
    electrons (NELEC), and 2*S_z (MS2) for the system."""
    with open(fcidump, "r") as f:
        # Holds ['NORB=#', 'NELEC=#', 'MS2=#']
        firstline = f.readline().strip("\n").split(',')
        for entry in firstline:
            if "NORB" in entry:
                # Get norbitals
                norbitals = int(entry.split("=")[1])
            if "NELEC" in entry:
                # Get nelectrons
                nelectrons = int(entry.split("=")[1])
            if "MS2" in entry:
                # Get MS2 (note: this is not going to be correct for open shells in GAMESS)
                ms2 = int(entry.split("=")[1])
    return norbitals, nelectrons, ms2

def load_integrals_from_fcidump(fcidump, system):
    """This function reads the FCIDUMP file to obtain the onebody and twobody
    integrals as well as nuclear repulsion energy.

    Parameters
    ----------
    fcidump : str
        Path to FCIDUMP file
    system : System object
        System object

    Returns
    -------
    e1int : ndarray(dtype=float, shape=(norb,norb))
        Onebody part of the bare Hamiltonian in the MO basis (Z)
    e2int : ndarray(dtype=float, shape=(norb,norb,norb,norb))
        Twobody part of the bare Hamiltonian in the MO basis (V)
    e_nn : float
        Nuclear repulsion energy (in hartree)
    """
    norb = system.norbitals + system.nfrozen - system.ndelete
    e1int = np.zeros((norb, norb), order="F")
    e2int = np.zeros((norb, norb, norb, norb), order="F")

    with open(fcidump) as f:
        for ct, line in enumerate(f.readlines()):
            if ct < 4: continue
            L = line.split()
            Cf = float(L[0].replace("D", "E")) # GAMESS FCIDUMP uses old-school D instead of E for scientific notation
            p = int(L[1]) - 1
            q = int(L[3]) - 1
            r = int(L[2]) - 1
            s = int(L[4]) - 1
            if q != -1 and s != -1: # twobody term
                e2int[p, q, r, s] = Cf
                e2int[r, q, p, s] = Cf
                e2int[p, s, r, q] = Cf
                e2int[r, s, p, q] = Cf
                e2int[q, p, s, r] = Cf
                e2int[q, r, s, p] = Cf
                e2int[s, p, q, r] = Cf
                e2int[s, r, q, p] = Cf
            elif q == -1 and s == -1 and p != -1: # onebody term
                e1int[p, r] = Cf
                e1int[r, p] = Cf
            else: # nuclear repulsion
                e_nn = Cf

    return e1int, e2int, e_nn

def build_rohf_fock(e1int, e2int, system, canonicalization):

    print(f"   Computing MO Energies from FCIDUMP: {canonicalization.upper()} Canonicalization")

    # Remember, default ROHF canonicalization in GAMESS is Roothaan
    A_ROHF = {"Davidson": [0.5, 1, 1],
              "Roothaan": [-0.5, 0.5, 1.5],
              "Guest-Saunders": [0.5, 0.5, 0.5]}
    B_ROHF = {"Davidson": [0.5, 0, 0],
              "Roothaan": [1.5, 0.5, -0.5],
              "Guest-Saunders": [0.5, 0.5, 0.5]}

    oa = slice(system.noccupied_alpha + system.nfrozen)
    ob = slice(system.noccupied_beta + system.nfrozen)

    fock_a = (e1int
              + np.einsum("pjqj->pq", e2int[:, oa, :, oa], optimize=True)
              - np.einsum("pjjq->pq", e2int[:, oa, oa, :], optimize=True)
              + np.einsum("pjqj->pq", e2int[:, ob, :, ob], optimize=True)
    )
    fock_b = (e1int
              + np.einsum("pjqj->pq", e2int[:, ob, :, ob], optimize=True)
              - np.einsum("pjjq->pq", e2int[:, ob, ob, :], optimize=True)
              + np.einsum("jpjq->pq", e2int[oa, :, oa, :], optimize=True)
    )

    # ROHF Canonicalization
    docc = slice(system.nfrozen + system.noccupied_beta) # doubly occupied orbitals (core)
    socc = slice(system.nfrozen + system.noccupied_beta, system.nfrozen + system.noccupied_alpha) # singly occupied orbitals
    virt = slice(system.nfrozen + system.noccupied_alpha, system.nfrozen + system.noccupied_alpha + system.nunoccupied_alpha) # unoccupied orbitals

    A = A_ROHF[canonicalization]
    B = B_ROHF[canonicalization]

    # diagonal docc, socc, and virt blocks
    F2 = A[0] * fock_a[docc, docc] + B[0] * fock_b[docc, docc]
    F1 = A[1] * fock_a[socc, socc] + B[1] * fock_b[socc, socc]
    F0 = A[2] * fock_a[virt, virt] + B[2] * fock_b[virt, virt]
    # the other blocks (not dependent on canonicalization)
    f_ds = fock_b[docc, socc]
    f_dv = 0.5 * (fock_a[docc, virt] + fock_b[docc, virt])
    f_sd = fock_b[socc, docc]
    f_sv = fock_a[socc, virt]
    f_vd = 0.5 * (fock_a[virt, docc] + fock_b[virt, docc])
    f_vs = fock_a[virt, socc]
    # Assemble ROHF Fock matrix
    fock_rohf = np.vstack((np.hstack((F2, f_ds, f_dv)),
                            np.hstack((f_sd, F1, f_sv)),
                            np.hstack((f_vd, f_vs, F0))))
    # Diagonalize to get MO energies and coefficients
    eps_mo, mo_coeff = np.linalg.eigh(fock_rohf)
    return eps_mo, mo_coeff