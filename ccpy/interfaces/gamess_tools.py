import numpy as np

def loadFromGamess(gamess_logfile, onebody_file, twobody_file, nfrozen, normal_ordered=True, data_type=np.float64):

    from cclib.io import ccread
    from ccpy.models.system import System
    from ccpy.models.integrals import getHamiltonian
    from ccpy.drivers.hf_energy import calc_hf_energy

    data = ccread(gamess_logfile)

    system = System(data.nelectrons,
               data.nmo,
               data.mult,
               nfrozen,
               point_group = getGamessPointGroup(gamess_logfile),
               orbital_symmetries = data.mosyms[0],
               charge = data.charge,
               nuclear_repulsion = getGamessNuclearRepulsion(gamess_logfile))

    e1int = loadOnebodyIntegralFile(onebody_file, system, data_type)
    nuclear_repulsion, e2int = loadTwobodyIntegralFile(twobody_file, system, data_type)

    assert (np.allclose(nuclear_repulsion, system.nuclear_repulsion, atol=1.0e-06, rtol=0.0))
    system.nuclear_repulsion = nuclear_repulsion

    # Check that the HF energy calculated using the integrals matches the GAMESS result
    hf_energy = calc_hf_energy(e1int, e2int, system)
    hf_energy += system.nuclear_repulsion
    assert (np.allclose(hf_energy, getGamessSCFEnergy(gamess_logfile), atol=1.0e-06, rtol=0.0))
    system.reference_energy = hf_energy

    return system, getHamiltonian(e1int, e2int, system, normal_ordered)

def getGamessSCFEnergy(gamess_logfile):

    with open(gamess_logfile, 'r') as f:
        for line in f.readlines():
            if all( s in line.split() for s in ['FINAL', 'ROHF', 'ENERGY', 'IS']) or\
                    all( s in line.split() for s in ['FINAL', 'RHF', 'ENERGY', 'IS']):
                print(line.split())
                hf_energy = float(line.split()[4])
                break
    return hf_energy

def getGamessNuclearRepulsion(gamess_logfile):

    with open(gamess_logfile, 'r') as f:
        for line in f.readlines():
            if all( s in line.split() for s in ['THE', 'NUCLEAR', 'REPULSION', 'ENERGY', 'IS']):
                e_nuclear = float(line.split()[-1])
                break
    return e_nuclear

def getGamessPointGroup(gamess_logfile):
    """Dumb way of getting the point group from GAMESS log files.

    Arguments:
    ----------
    gamessFile : str -> Path to GAMESS log file
    Returns:
    ----------
    point_group : str -> Molecular point group"""
    point_group = 'C1'
    flag_found = False
    with open(gamess_logfile, 'r') as f:
        for line in f.readlines():
            if flag_found:
                order = line.split()[-1]
                if len(point_group) == 3:
                    point_group = point_group[0] + order + point_group[2]
                if len(point_group) == 2:
                    point_group = point_group[0] + order
                if len(point_group) == 1:
                    point_group = point_group[0] + order
                break
            if 'THE POINT GROUP OF THE MOLECULE IS' in line:
                point_group = line.split()[-1]
                flag_found = True
    return point_group


def getNumberTotalOrbitals(onebody_file):
    with open(onebody_file) as f_in:
        lines = f_in.readlines()
        ct = 0
        for line in lines:
            ct += 1
    return int(-0.5 + np.sqrt(0.25 + 2 * x))


def loadOnebodyIntegralFile(onebody_file, system, data_type):
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
    norb = system.norbitals + system.nfrozen
    e1int = np.zeros((norb, norb), dtype=data_type)
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
        print('Error: {} does not appear to exist.'.format(onebody_file))
    return e1int


def loadTwobodyIntegralFile(twobody_file, system, data_type):
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
        norb = system.norbitals + system.nfrozen
        # initialize numpy array
        e2int = np.zeros((norb, norb, norb, norb), dtype=data_type)
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
        e2int = np.einsum('iajb->ijab', e2int)
    except IOError:
        print('Error: {} does not appear to exist.'.format(twobody_file))
    return e_nn, e2int
