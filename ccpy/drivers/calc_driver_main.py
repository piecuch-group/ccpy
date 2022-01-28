"""Main calculation driver module of CCpy."""


#from ccpy import cc
from ccpy.drivers.solvers import solve_cc_jacobi
from ccpy.models.operators import ClusterOperator

def calc_driver_main(calculation, system, H, T_init=None):
    """Performs the calculation specified by the user in the input.

    Parameters
    ----------
    inputs : dict
        Contains all input keyword flags obtained from parsing the user-supplied input
    sys : dict
        System information dictionary
    ints : dict
        Sliced F_N and V_N integrals that define the bare Hamiltonian H_N

    Returns
    -------
    None
    """

    #if calculation_type not in cc.MODULES:
    #    raise NotImplementedError("Calculation type {calculation_type} not implemented")

    # CCSD Calculation
    order = 2
    if T_init is None:
        T = ClusterOperator(system, order)
        dT = ClusterOperator(system, order)

    from ccpy.cc.ccsd import update_t
    T, cc_energy = solve_cc_jacobi(update_t, T, dT, H, calculation)


    return

if __name__ == "__main__":

    from ccpy.interfaces.pyscf_tools import loadFromPyscfMolecular
    from pyscf import gto, scf

    from ccpy.models.calculation import Calculation

    # Testing from PySCF
    mol = gto.Mole()
    mol.build(
        atom='''F 0.0 0.0 -2.66816
                F 0.0 0.0  2.66816''',
        basis='ccpvdz',
        charge=0,
        spin=0,
        symmetry='D2H',
        cart=True,
        unit='Bohr',
    )
    mf = scf.ROHF(mol)
    mf.kernel()

    nfrozen = 2
    system, H = loadFromPyscfMolecular(mf, nfrozen, dumpIntegrals=False)

    print(system)

    calculation = Calculation('CCSD')
    calc_driver_main(calculation, system, H)