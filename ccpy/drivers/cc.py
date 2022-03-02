"""Ground-state CC calculation driver."""

from importlib import import_module

import ccpy.cc
from ccpy.drivers.solvers import cc_jacobi
from ccpy.models.operators import ClusterOperator
from ccpy.utilities.printing import ccpy_header, SystemPrinter, CCPrinter

#[TODO]: Think whether this driver will be used for CC(P) as well as normal CC.
# The form of the update function would change in CC(P) as lists of triples will need to be passed in
def driver(calculation, system, hamiltonian, T=None):
    """Performs the calculation specified by the user in the input."""

    ccpy_header()
    sys_printer = SystemPrinter(system)
    sys_printer.header()

    # check if requested CC calculation is implemented in modules
    if calculation.calculation_type not in ccpy.cc.MODULES:
        raise NotImplementedError(
            "{} not implemented".format(calculation.calculation_type)
        )

    # import the specific CC method module and get its update function
    cc_mod = import_module("ccpy.cc." + calculation.calculation_type.lower())
    update_function = getattr(cc_mod, 'update')

    cc_printer = CCPrinter(calculation)
    cc_printer.header()

    # initialize the cluster operator anew, or use restart
    if T is None:
        T = ClusterOperator(system, calculation.order)

    # regardless of restart status, initialize residual anew
    dT = ClusterOperator(system, calculation.order)

    T, cc_energy, is_converged = cc_jacobi(
                                           update_function,
                                           T,
                                           dT,
                                           hamiltonian,
                                           calculation,
                                           )
    total_energy = system.reference_energy + cc_energy

    cc_printer.calculation_summary(system.reference_energy, cc_energy)

    return T, total_energy, is_converged

#[TODO]: Think whether this driver will be used for CC(P) as well as normal CC.
# The form of the update function would change in CC(P) as lists of triples will need to be passed in
def ccpdriver(calculation, system, hamiltonian, pspace, T=None):
    """Performs the calculation specified by the user in the input."""

    ccpy_header()
    sys_printer = SystemPrinter(system)
    sys_printer.header()

    # check if requested CC calculation is implemented in modules
    if calculation.calculation_type not in ccpy.cc.MODULES:
        raise NotImplementedError(
            "{} not implemented".format(calculation.calculation_type)
        )

    # import the specific CC method module and get its update function
    cc_mod = import_module("ccpy.cc." + calculation.calculation_type.lower())
    update_function = getattr(cc_mod, 'update')

    cc_printer = CCPrinter(calculation)
    cc_printer.header()

    # initialize the cluster operator anew, or use restart
    if T is None:
        T = ClusterOperator(system, calculation.order)

    # regardless of restart status, initialize residual anew
    dT = ClusterOperator(system, calculation.order)

    T, cc_energy, is_converged = cc_jacobi(
                                           update_function,
                                           T,
                                           dT,
                                           hamiltonian,
                                           calculation,
                                           )
    total_energy = system.reference_energy + cc_energy

    cc_printer.calculation_summary(system.reference_energy, cc_energy)

    return T, total_energy, is_converged
