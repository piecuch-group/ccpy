"""Main calculation driver module of CCpy."""

from ccpy import cc
from ccpy.drivers.solvers import solve_cc_jacobi
from ccpy.models.operators import ClusterOperator

from ccpy.utilities.printing import *

def calc_driver_main(calculation, system, hamiltonian, T_init=None):
    """Performs the calculation specified by the user in the input."""

    ccpy_header()
    sys_printer = SystemPrinter(system)
    sys_printer.header()

    if calculation.calculation_type not in cc.MODULES:
        raise NotImplementedError("{} not implemented".format(calculation.calculation_type))

    cc_printer = CCPrinter(calculation)

    cc_printer.header()
    # CCSD Calculation
    order = 2
    if T_init is None:
        T = ClusterOperator(system, order)
        dT = ClusterOperator(system, order)

    from ccpy.cc.ccsd import update
    T, cc_energy = solve_cc_jacobi(update, T, dT, hamiltonian, calculation, diis_out_of_core=True)
    total_energy = system.reference_energy + cc_energy

    cc_printer.calculation_summary(system.reference_energy, cc_energy)

    return T, total_energy
