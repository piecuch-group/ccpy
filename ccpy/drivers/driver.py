"""Main calculation driver module of CCpy."""

from importlib import import_module

import ccpy.cc
import ccpy.left
import ccpy.eomcc

from ccpy.drivers.solvers import cc_jacobi, ccp_jacobi, left_cc_jacobi, eomcc_davidson, eomcc_davidson_lowmem
from ccpy.models.operators import ClusterOperator
from ccpy.utilities.printing import ccpy_header, SystemPrinter, CCPrinter

from ccpy.eomcc.initial_guess import get_initial_guess


def cc_driver(calculation, system, hamiltonian, T=None, pspace=None):
    """Performs the calculation specified by the user in the input."""

    # check if requested CC calculation is implemented in modules
    if calculation.calculation_type not in ccpy.cc.MODULES:
        raise NotImplementedError(
            "{} not implemented".format(calculation.calculation_type)
        )

    # [TODO]: Check if calculation parameters (e.g, active orbitals) make sense

    # import the specific CC method module and get its update function
    cc_mod = import_module("ccpy.cc." + calculation.calculation_type.lower())
    update_function = getattr(cc_mod, 'update')

    cc_printer = CCPrinter(calculation)
    cc_printer.cc_header()

    # initialize the cluster operator anew, or use restart
    if T is None:
        T = ClusterOperator(system,
                            order=calculation.order,
                            active_orders=calculation.active_orders,
                            num_active=calculation.num_active)

    # regardless of restart status, initialize residual anew
    dT = ClusterOperator(system,
                        order=calculation.order,
                        active_orders=calculation.active_orders,
                        num_active=calculation.num_active)


    if pspace is None:  # Run the standard CC solver if no explicit P space is used
        T, corr_energy, is_converged = cc_jacobi(
                                               update_function,
                                               T,
                                               dT,
                                               hamiltonian,
                                               calculation,
                                               system,
                                               )
    else: # Run the dedicated CC(P) solver
        T, corr_energy, is_converged = ccp_jacobi(
            update_function,
            T,
            dT,
            hamiltonian,
            calculation,
            system,
            pspace
        )

    total_energy = system.reference_energy + corr_energy

    cc_printer.cc_calculation_summary(system.reference_energy, corr_energy)

    return T, total_energy, is_converged


# [TODO]: Pass in cc_energy or somehow fix issue that output prints 0 for correlation energy
def lcc_driver(calculation, system, T, hamiltonian, omega=0.0, L=None, R=None):
    """Performs the calculation specified by the user in the input."""

    # check if requested CC calculation is implemented in modules
    if calculation.calculation_type not in ccpy.left.MODULES:
        raise NotImplementedError(
            "{} not implemented".format(calculation.calculation_type)
        )

    # import the specific CC method module and get its update function
    lcc_mod = import_module("ccpy.left." + calculation.calculation_type.lower())
    update_function = getattr(lcc_mod, 'update')

    cc_printer = CCPrinter(calculation)
    cc_printer.cc_header()

    # decide whether this is a ground-state calculation
    is_ground = True
    if R is not None:
        is_ground = False

    # initialize the cluster operator anew, or use restart
    if L is None:
        L = ClusterOperator(system,
                            calculation.order,
                            calculation.active_orders,
                            calculation.num_active,
                            )
        if is_ground:
            L.unflatten(T.flatten()[:L.ndim])
        else:
            L.unflatten(R.flatten()[:L.ndim])

    # regardless of restart status, initialize residual anew
    LH = ClusterOperator(system, calculation.order)

    L, omega, LR, is_converged = left_cc_jacobi(update_function,
                                         L,
                                         LH,
                                         T,
                                         R,
                                         hamiltonian,
                                         omega,
                                         calculation,
                                         is_ground,
                                         system
                                         )
    total_energy = system.reference_energy + omega

    cc_printer.leftcc_calculation_summary(omega, LR, is_converged)

    return L, total_energy, is_converged

def eomcc_driver(calculation, system, hamiltonian, T, R, omega):
    """Performs the EOMCC calculation specified by the user in the input."""

    # check if requested CC calculation is implemented in modules
    if calculation.calculation_type not in ccpy.eomcc.MODULES:
        raise NotImplementedError(
            "{} not implemented".format(calculation.calculation_type)
        )

    # import the specific CC method module and get its update function
    module = import_module("ccpy.eomcc." + calculation.calculation_type.lower())
    HR_function = getattr(module, 'HR')
    update_function = getattr(module, 'update')

    cc_printer = CCPrinter(calculation)
    cc_printer.eomcc_header()

    if calculation.low_memory:
        R, omega, r0, is_converged = eomcc_davidson_lowmem(
                                           HR_function,
                                           update_function,
                                           R,
                                           omega,
                                           T,
                                           hamiltonian,
                                           calculation,
                                           system,
                                           )
    else:
        R, omega, r0, is_converged = eomcc_davidson(
                                           HR_function,
                                           update_function,
                                           R,
                                           omega,
                                           T,
                                           hamiltonian,
                                           calculation,
                                           system,
                                           )

    cc_printer.eomcc_calculation_summary(omega, r0, is_converged)

    return R, omega, r0, is_converged
