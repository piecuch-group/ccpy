import numpy as np

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
