"""Main calculation driver module of CCpy."""

from importlib import import_module

import ccpy.cc
import ccpy.left
import ccpy.eomcc

from ccpy.drivers.solvers import cc_jacobi, ccp_jacobi, left_cc_jacobi, left_ccp_jacobi, eomcc_davidson, eomcc_davidson_lowmem, mrcc_jacobi
from ccpy.drivers.solvers import eccc_jacobi

from ccpy.models.operators import ClusterOperator, FockOperator
from ccpy.utilities.printing import CCPrinter
from ccpy.utilities.pspace import count_excitations_in_pspace

from copy import deepcopy


def cc_driver(calculation, system, hamiltonian, T=None, pspace=None, excitation_count=None):
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
    #[TODO]: This is not compatible if the initial T is a lower order than the
    # one used in the calculation. For example, we could not start a CCSDT
    # calculation using the CCSD cluster amplitudes.


    if pspace is None:  # Run the standard CC solver if no explicit P space is used
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

        T, corr_energy, is_converged = cc_jacobi(
                                               update_function,
                                               T,
                                               dT,
                                               hamiltonian,
                                               calculation,
                                               system,
                                               )
    else: # Run the dedicated CC(P) solver
        p_orders = [3 + i for i in range(len(pspace))]
        pspace_sizes = count_excitations_in_pspace(pspace, system, ordered_index=True)
        if T is None:
            T = ClusterOperator(system,
                                order=calculation.order,
                                active_orders=calculation.active_orders,
                                p_orders=p_orders,
                                pspace_sizes=pspace_sizes,
                                num_active=calculation.num_active)

        # regardless of restart status, initialize residual anew
        dT = ClusterOperator(system,
                             order=calculation.order,
                             active_orders=calculation.active_orders,
                             p_orders=p_orders,
                             pspace_sizes=pspace_sizes,
                             num_active=calculation.num_active)

        T, corr_energy, is_converged = ccp_jacobi(
            update_function,
            T,
            dT,
            hamiltonian,
            calculation,
            system,
            pspace,
        )

    total_energy = system.reference_energy + corr_energy

    cc_printer.cc_calculation_summary(system.reference_energy, corr_energy)

    return T, total_energy, is_converged

def eccc_driver(calculation, system, hamiltonian, external_wavefunction, T=None):
    """Performs the calculation specified by the user in the input."""
    from ccpy.utilities.external_correction import cluster_analysis

    # Get the external T vector corresponding to the cluster analysis
    T_ext, VT_ext = cluster_analysis(external_wavefunction, hamiltonian, system, debug=True)

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
    #[TODO]: This is not compatible if the initial T is a lower order than the
    # one used in the calculation. For example, we could not start a CCSDT
    # calculation using the CCSD cluster amplitudes.

    if T is None:
        T = ClusterOperator(system,
                            order=calculation.order,
                            active_orders=calculation.active_orders,
                            num_active=calculation.num_active)

        # Set the initial T1 and T2 components to that from cluster analysis
        setattr(T, 'a', T_ext.a)
        setattr(T, 'b', T_ext.b)
        setattr(T, 'aa', T_ext.aa)
        setattr(T, 'ab', T_ext.ab)
        setattr(T, 'bb', T_ext.bb)

    # regardless of restart status, initialize residual anew
    dT = ClusterOperator(system,
                         order=calculation.order,
                         active_orders=calculation.active_orders,
                         num_active=calculation.num_active)

    T, corr_energy, is_converged = eccc_jacobi(
                                           update_function,
                                           T,
                                           dT,
                                           hamiltonian,
                                           calculation,
                                           system,
                                           T_ext,
                                           VT_ext)

    total_energy = system.reference_energy + corr_energy

    cc_printer.cc_calculation_summary(system.reference_energy, corr_energy)

    return T, total_energy, is_converged



def lcc_driver(calculation, system, T, hamiltonian, omega=0.0, L=None, R=None, pspace=None):
    """Performs the calculation specified by the user in the input."""

    # check if requested CC calculation is implemented in modules
    if calculation.calculation_type not in ccpy.left.MODULES:
        raise NotImplementedError(
            "{} not implemented".format(calculation.calculation_type)
        )

    # import the specific CC method module and get its update function
    lcc_mod = import_module("ccpy.left." + calculation.calculation_type.lower())
    update_function = getattr(lcc_mod, 'update')
    #LR_function = getattr(lcc_mod, 'LR')

    cc_printer = CCPrinter(calculation)
    cc_printer.cc_header()

    # decide whether this is a ground-state calculation
    is_ground = True
    if R is not None:
        is_ground = False

    # initialize the cluster operator anew, or use restart
    if L is None:

        if is_ground:
            L = ClusterOperator(system,
                                calculation.order,
                                calculation.active_orders,
                                calculation.num_active,
                                )

            L.unflatten(T.flatten()[:L.ndim])
        else:
            if isinstance(R, ClusterOperator):
                L = ClusterOperator(system,
                                    calculation.order,
                                    calculation.active_orders,
                                    calculation.num_active,
                                    )
            elif isinstance(R, FockOperator):
                L = FockOperator(system,
                                 calculation.num_particles,
                                 calculation.num_holes)

            L.unflatten(R.flatten()[:L.ndim])

    # regardless of restart status, initialize residual anew
    LH = deepcopy(L)
    LH.unflatten(0.0 * L.flatten())

    if pspace is None:

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
    else:

        L, omega, LR, is_converged = left_ccp_jacobi(update_function,
                                             L,
                                             LH,
                                             T,
                                             R,
                                             hamiltonian,
                                             omega,
                                             calculation,
                                             is_ground,
                                             system,
                                             pspace,
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

def mrcc_driver(calculation, system, hamiltonian, model_space, two_body_approximation=True):
    """Performs the calculation specified by the user in the input."""

    # check if requested CC calculation is implemented in modules
    if calculation.calculation_type not in ccpy.mrcc.MODULES:
        raise NotImplementedError(
            "{} not implemented".format(calculation.calculation_type)
        )

    # [TODO]: Check if calculation parameters (e.g, active orbitals) make sense

    # import the specific CC method module and get its update function
    cc_mod = import_module("ccpy.mrcc." + calculation.calculation_type.lower())
    heff_mod = import_module("ccpy.mrcc.effective_hamiltonian")
    update_function = getattr(cc_mod, 'update')

    if two_body_approximation: # use the Heff matrix elements computed using T(p) = T1(p) + T2(p)
        compute_Heff_function = getattr(heff_mod, 'compute_Heff_mkmrccsd')
    else:
        compute_Heff_function = getattr(heff_mod, 'compute_Heff_' + calculation.calculation_type.lower())

    cc_printer = CCPrinter(calculation)
    cc_printer.cc_header()

    # initialize the cluster operator anew, or use restart
    #[TODO]: This is not compatible if the initial T is a lower order than the
    # one used in the calculation. For example, we could not start a CCSDT
    # calculation using the CCSD cluster amplitudes.
    T = [None for i in range(len(model_space))]
    dT = [None for i in range(len(model_space))]
    for p in range(len(model_space)):
        if T[p] is None:
            T[p] = ClusterOperator(system,
                                order=calculation.order,
                                active_orders=calculation.active_orders,
                                num_active=calculation.num_active)

        # regardless of restart status, initialize residual anew
        dT[p] = ClusterOperator(system,
                            order=calculation.order,
                            active_orders=calculation.active_orders,
                            num_active=calculation.num_active)

    T, total_energy, is_converged = mrcc_jacobi(
                                           update_function,
                                           compute_Heff_function,
                                           T,
                                           dT,
                                           model_space,
                                           hamiltonian,
                                           calculation,
                                           system,
                                           )


    #total_energy = system.reference_energy + corr_energy

    #cc_printer.cc_calculation_summary(system.reference_energy, corr_energy)

    return T, total_energy, is_converged
