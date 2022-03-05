"""Main calculation driver module of CCpy."""

import copy
from importlib import import_module
from copy import deepcopy

import ccpy.cc
import ccpy.left
from ccpy.drivers.solvers import cc_jacobi, left_cc_jacobi
from ccpy.models.operators import ClusterOperator
from ccpy.utilities.printing import ccpy_header, SystemPrinter, CCPrinter


def cc_driver(calculation, system, hamiltonian, T=None):
    """Performs the calculation specified by the user in the input."""

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

# [TODO]: The form of the update function would change in CC(P) as lists of triples will need to be passed in
def ccp_driver(calculation, system, hamiltonian, pspace, T=None):
    """Performs the calculation specified by the user in the input."""

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
    cc_printer.header()

    # decide whether this is a ground-state calculation
    is_ground = True
    if R is not None:
        if omega == 0.0:
            is_ground = False
        else:
            print('WARNING: omega for ground-state left CC calculation is not identicall 0!')

    # initialize the cluster operator anew, or use restart
    if is_ground:
        if L is None:
            L = copy.deepcopy(T)
    else:
        if L is None:
            L = copy.deepcopy(R)

    # regardless of restart status, initialize residual anew
    LH = ClusterOperator(system, calculation.order)

    L, lcc_energy, is_converged = left_cc_jacobi(update_function,
                                         L,
                                         LH,
                                         T,
                                         R,
                                         hamiltonian,
                                         omega,
                                         calculation,
                                         )
    total_energy = system.reference_energy + lcc_energy

    #cc_printer.calculation_summary(system.reference_energy, lcc_energy)

    return L, total_energy, is_converged