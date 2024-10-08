import numpy as np
from ccpy.interfaces.fcidump_tools import load_integrals_from_fcidump, load_system_params_from_fcidump

def load_gamess_integrals(
    gamess_logfile,
    fcidump_file=None,
    onebody_file=None,
    twobody_file=None,
    nfrozen=0,
    ndelete=0,
    multiplicity=None,
    normal_ordered=True,
    sorted=True,
    data_type=np.float64,
):
    from cclib.io import ccread

    from ccpy.constants import constants
    from ccpy.energy.hf_energy import calc_hf_energy, calc_hf_frozen_core_energy
    from ccpy.models.integrals import getHamiltonian
    from ccpy.models.system import System

    data = ccread(gamess_logfile)
    # unable to change nelectrons attribute
    nelectrons = data.nelectrons
    # if multiplicity is provided, use that; otherwise, read from GAMESS logfile
    if multiplicity is None:
        multiplicity = data.mult
    # Read nelectrons & norbitals directly from fcidump in order to accomodate ECP GAMESS runs
    if fcidump_file:
        nmo_fcidump, nelectron_fcidump, ms2_fcidump = load_system_params_from_fcidump(fcidump_file)
        data.nmo = nmo_fcidump
        if data.nelectrons != nelectron_fcidump:
            print("\n   =============================================================================================")
            print("      WARNING: NELEC in GAMESS logfile and FCIDUMP do not match! Using NELEC listed in FCIDUMP.")
            print("      (Beware, GAMESS FCIDUMP prints wrong NELEC and MS2 for open-shell systems!)")
            print("   =============================================================================================\n")
        nelectrons = nelectron_fcidump

    system = System(
        nelectrons,
        data.nmo,
        multiplicity,
        nfrozen,
        ndelete=ndelete,
        point_group=get_point_group(gamess_logfile),
        orbital_symmetries=[x.upper() for x in data.mosyms[0]],
        charge=data.charge,
        nuclear_repulsion=get_nuclear_repulsion(gamess_logfile),
        mo_energies=[x * constants.eVtohartree for x in data.moenergies[0]],
    )

    # Load using onebody and twobody direct integral files
    if fcidump_file is None and onebody_file is not None and twobody_file is not None:
        e1int = load_onebody_integrals(onebody_file, system, data_type)
        nuclear_repulsion, e2int = load_twobody_integrals(twobody_file, system, data_type)
    # Load from FCIDUMP file
    elif fcidump_file is not None:
        e1int, e2int, nuclear_repulsion = load_integrals_from_fcidump(fcidump_file, system)

    assert np.allclose(
        nuclear_repulsion, system.nuclear_repulsion, atol=1.0e-06, rtol=0.0
    )
    system.nuclear_repulsion = nuclear_repulsion

    # Check that the HF energy calculated using the integrals matches the GAMESS result
    hf_energy = calc_hf_energy(e1int, e2int, system)
    hf_energy += system.nuclear_repulsion
    assert np.allclose(
        hf_energy, get_reference_energy(gamess_logfile), atol=1.0e-06, rtol=0.0
    )
    system.reference_energy = hf_energy
    system.frozen_energy = calc_hf_frozen_core_energy(e1int, e2int, system)
    return system, getHamiltonian(e1int, e2int, system, normal_ordered, sorted)

# def load_gamess_bare_integrals(
#     logfile=None,
#     fcidump=None,
#     onebody=None,
#     twobody=None,
#     nfrozen=0,
#     ndelete=0,
#     multiplicity=None,
#     data_type=np.float64,
# ):
#     from cclib.io import ccread
#
#     from ccpy.constants import constants
#     from ccpy.energy.hf_energy import calc_hf_energy, calc_hf_frozen_core_energy
#     from ccpy.models.integrals import getHamiltonian
#     from ccpy.models.system import System
#
#     data = ccread(logfile)
#     # unable to change nelectrons attribute
#     nelectrons = data.nelectrons
#     # if multiplicity is provided, use that; otherwise, read from GAMESS logfile
#     if multiplicity is None:
#         multiplicity = data.mult
#     # Read nelectrons & norbitals directly from fcidump in order to accomodate ECP GAMESS runs
#     if fcidump:
#         nmo_fcidump, nelectron_fcidump, ms2_fcidump = load_system_params_from_fcidump(fcidump)
#         data.nmo = nmo_fcidump
#         if data.nelectrons != nelectron_fcidump:
#             print("\n   =============================================================================================")
#             print("      WARNING: NELEC in GAMESS logfile and FCIDUMP do not match! Using NELEC listed in FCIDUMP.")
#             print("      (Beware, GAMESS FCIDUMP prints wrong NELEC and MS2 for open-shell systems!)")
#             print("   =============================================================================================\n")
#         nelectrons = nelectron_fcidump
#
#     system = System(
#         nelectrons,
#         data.nmo,
#         multiplicity,
#         nfrozen,
#         ndelete=ndelete,
#         point_group=get_point_group(logfile),
#         orbital_symmetries=[x.upper() for x in data.mosyms[0]],
#         charge=data.charge,
#         nuclear_repulsion=get_nuclear_repulsion(logfile),
#         mo_energies=[x * constants.eVtohartree for x in data.moenergies[0]],
#     )
#
#     # Load using onebody and twobody direct integral files
#     if fcidump is None and onebody is not None and twobody is not None:
#         e1int = load_onebody_integrals(onebody, system, data_type)
#         nuclear_repulsion, e2int = load_twobody_integrals(twobody, system, data_type)
#     # Load from FCIDUMP file
#     elif fcidump is not None:
#         e1int, e2int, nuclear_repulsion = load_integrals_from_fcidump(fcidump, system)
#
#     assert np.allclose(
#         nuclear_repulsion, system.nuclear_repulsion, atol=1.0e-06, rtol=0.0
#     )
#     system.nuclear_repulsion = nuclear_repulsion
#
#     # Check that the HF energy calculated using the integrals matches the GAMESS result
#     hf_energy = calc_hf_energy(e1int, e2int, system)
#     hf_energy += system.nuclear_repulsion
#     assert np.allclose(
#         hf_energy, get_reference_energy(logfile), atol=1.0e-06, rtol=0.0
#     )
#     system.reference_energy = hf_energy
#     system.frozen_energy = calc_hf_frozen_core_energy(e1int, e2int, system)
#     return system, e1int, e2int

def get_reference_energy(gamess_logfile):

    with open(gamess_logfile, "r") as f:
        for line in f.readlines():
            if all(s in line.split() for s in ["FINAL", "ROHF", "ENERGY", "IS"]) or all(
                s in line.split() for s in ["FINAL", "RHF", "ENERGY", "IS"]
            ):
                hf_energy = float(line.split()[4])
                break
    return hf_energy

def get_nuclear_repulsion(gamess_logfile):

    with open(gamess_logfile, "r") as f:
        for line in f.readlines():
            if all(
                s in line.split()
                for s in ["THE", "NUCLEAR", "REPULSION", "ENERGY", "IS"]
            ):
                e_nuclear = float(line.split()[-1])
                break
    return e_nuclear

def get_point_group(gamess_logfile):
    """Dumb way of getting the point group from GAMESS log files.

    Arguments:
    ----------
    gamessFile : str -> Path to GAMESS log file
    Returns:
    ----------
    point_group : str -> Molecular point group"""
    point_group = "C1"
    flag_found = False
    with open(gamess_logfile, "r") as f:
        for line in f.readlines():
            if flag_found and point_group != "CI" and point_group != "CS":
                order = line.split()[-1]
                if len(point_group) == 3:
                    point_group = point_group[0] + order + point_group[2]
                if len(point_group) == 2:
                    point_group = point_group[0] + order
                if len(point_group) == 1:
                    point_group = point_group[0] + order
                break
            if "THE POINT GROUP OF THE MOLECULE IS" in line:
                point_group = line.split()[-1]
                flag_found = True
    if point_group == 'C0':
        point_group = 'C1'
    return point_group

def load_onebody_integrals(onebody_file, system, data_type):
    """This function reads the onebody.inp file from GAMESS
    and returns a numpy matrix.

    Parameters
    ----------
    filename : str
        Path to onebody integral file
    sys : dict
        System information dict

    Returns
    -------
    e1int : ndarray(dtype=float, shape=(norb,norb))
        Onebody part of the bare Hamiltonian in the MO basis (Z)
    """
    norb = system.norbitals + system.nfrozen - system.ndelete
    e1int = np.zeros((norb, norb), dtype=data_type, order="F")
    try:
        with open(onebody_file) as f_in:
            lines = f_in.readlines()
            ct = 0
            for i in range(norb):
                for j in range(i + 1):
                    val = float(lines[ct].split()[0])
                    e1int[i, j] = val
                    e1int[j, i] = val
                    ct += 1
    except IOError:
        print("Error: {} does not appear to exist.".format(onebody_file))
    return e1int


def load_twobody_integrals(twobody_file, system, data_type):
    """This function reads the twobody.inp file from GAMESS
    and returns a numpy matrix.

    Parameters
    ----------
    filename : str
        Path to twobody integral file
    sys : dict
        System information dict

    Returns
    -------
    e_nn : float
        Nuclear repulsion energy (in hartree)
    e2int : ndarray(dtype=float, shape=(norb,norb,norb,norb))
        Twobody part of the bare Hamiltonian in the MO basis (V)
    """
    try:
        norb = system.norbitals + system.nfrozen - system.ndelete
        # initialize numpy array
        e2int = np.zeros((norb, norb, norb, norb), dtype=data_type, order="F")
        # open file
        with open(twobody_file) as f_in:
            # loop over lines
            for line in f_in:
                # split fields and parse
                fields = line.split()
                indices = tuple(map(int, fields[:4]))
                val = float(fields[4])
                # check whether value is nuclear repulsion
                # fill matrix otherwise
                if sum(indices) == 0:
                    e_nn = val
                else:
                    indices = tuple(i - 1 for i in indices)
                    e2int[indices] = val
        # convert e2int from chemist notation (ia|jb) to
        # physicist notation <ij|ab>
        e2int = np.einsum("iajb->ijab", e2int)
    except IOError:
        print("Error: {} does not appear to exist.".format(twobody_file))
    return e_nn, e2int

