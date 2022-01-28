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

    print('   ===========================================')
    print('               ',calculation.calculation_type.upper(),'Calculation')
    print('   ===========================================\n')

    # CCSD Calculation
    order = 2
    if T_init is None:
        T = ClusterOperator(system, order)
        dT = ClusterOperator(system, order)

    from ccpy.cc.ccsd import update_t
    T, cc_energy = solve_cc_jacobi(update_t, T, dT, H, calculation)
    total_energy = system.reference_energy + cc_energy

    print('')
    print('  CC Calculation Summary')
    print('  -------------------------------------')
    print('    Reference energy = ', system.reference_energy)
    print('    CC correlation energy = ', cc_energy)
    print('    Total CC energy = ', total_energy)

    return T, total_energy
