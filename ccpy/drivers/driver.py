"""Main calculation driver module of CCpy."""
import numpy as np
from importlib import import_module
from functools import partial

import ccpy.cc
import ccpy.hbar
import ccpy.left
import ccpy.eomcc

from ccpy.drivers.solvers import cc_jacobi, ccp_jacobi, left_cc_jacobi, left_ccp_jacobi, eomcc_davidson, eccc_jacobi
from ccpy.drivers.cc_energy import get_LR, get_r0

from ccpy.models.operators import ClusterOperator, FockOperator
from ccpy.utilities.printing import get_timestamp, cc_calculation_summary, eomcc_calculation_summary, leftcc_calculation_summary

# [TODO]: - make CC(P) solver able to read in previous T vector of smaller P space as initial guess
#         - clean up CC(P) solvers to remove options for previous linear and quadratic solvers

class Driver:

    def __init__(self, system, hamiltonian):
        self.system = system
        self.hamiltonian = hamiltonian
        self.options = {"method" : None,
                        "maximum_iterations": 80,
                        "convergence_tolerance" : 1.0e-07,
                        "energy_shift" : 0.0,
                        "diis_size" : 6,
                        "RHF_symmetry" : False,
                        "diis_out_of_core" : False}
        self.operator_params = {"order" : 0,
                                "number_particles" : 0,
                                "number_holes" : 0,
                                "active_orders" : [None],
                                "number_active_indices" : [None],
                                "pspace_orders" : [None]}
        self.T = None
        self.L = [None] * 100
        self.R = [None] * 100
        self.correlation_energy = 0.0
        self.vertical_excitation_energy = np.zeros(100)
        self.r0 = np.zeros(100)

    def set_operator_params(self, method):
        if method in ["ccd", "ccsd", "eomccsd", "left_ccsd", "eccc2"]:
            self.operator_params["order"] = 2
            self.operator_params["number_particles"] = 2
            self.operator_params["number_holes"] = 2
        elif method in ["ccsdt", "eomccsdt", "left_ccsdt", "left_ccsdt_p_slow"]:
            self.operator_params["order"] = 3
            self.operator_params["number_particles"] = 3
            self.operator_params["number_holes"] = 3
        elif method in ["ccsdtq"]:
            self.operator_params["order"] = 4
            self.operator_params["number_particles"] = 4
            self.operator_params["number_holes"] = 4
        elif method in ["ccsdt1", "eomccsdt1"]:
            self.operator_params["order"] = 3
            self.operator_params["number_particles"] = 3
            self.operator_params["number_holes"] = 3
            self.operator_params["active_orders"] = [3]
            self.operator_params["number_active_indices"] = [1]
        elif method in ["ccsdt_p"]:
            self.operator_params["order"] = 3
            self.operator_params["number_particles"] = 3
            self.operator_params["number_holes"] = 3
            self.operator_params["pspace_orders"] = [3]

    def print_options(self):
        print("   ------------------------------------------")
        for option_key, option_value in self.options.items():
            print("  ", option_key, "=", option_value)
        print("   ------------------------------------------\n")

    def run_cc(self, method, t3_excitations=None):
        # check if requested CC calculation is implemented in modules
        if method.lower() not in ccpy.cc.MODULES:
            raise NotImplementedError(
                "{} not implemented".format(method.lower())
            )
        # Set operator parameters needed to build T
        self.set_operator_params(method)
        self.options["method"] = method.upper()

        # import the specific CC method module and get its update function
        cc_mod = import_module("ccpy.cc." + method.lower())
        update_function = getattr(cc_mod, 'update')

        # Print the options as a header
        print("   CC calculation started on", get_timestamp())
        self.print_options()

        if t3_excitations is None:  # Run the standard CC solver if no explicit P space is used
            if self.T is None:
                self.T = ClusterOperator(self.system,
                                         order=self.operator_params["order"],
                                         active_orders=self.operator_params["active_orders"],
                                         num_active=self.operator_params["number_active_indices"])
            # regardless of restart status, initialize residual anew
            dT = ClusterOperator(self.system,
                                 order=self.operator_params["order"],
                                 active_orders=self.operator_params["active_orders"],
                                 num_active=self.operator_params["number_active_indices"])
            # Run the CC calculation
            self.T, self.corr_energy, _ = cc_jacobi(update_function,
                                                    self.T,
                                                    dT,
                                                    self.hamiltonian,
                                                    self.system,
                                                    self.options,
                                                   )
        else: # CC(P) method
    
            # Get dimensions of T3 spincases in P space
            n3aaa = t3_excitations["aaa"].shape[0]
            n3aab = t3_excitations["aab"].shape[0]
            n3abb = t3_excitations["abb"].shape[0]
            n3bbb = t3_excitations["bbb"].shape[0]
            excitation_count = [[n3aaa, n3aab, n3abb, n3bbb]]

            # If RHF, copy aab into abb and aaa in bbb
            if calculation.RHF_symmetry:
                assert (n3aaa == n3bbb)
                assert (n3aab == n3abb)
                t3_excitations["bbb"] = t3_excitations["aaa"].copy()
                t3_excitations["abb"] = t3_excitations["aab"][:, [2, 0, 1, 5, 3, 4]]  # want abb excitations as a b~<c~ i j~<k~; MUST be this order!

            if self.T is None:
                T = ClusterOperator(self.system,
                                    order=self.operator_params["order"],
                                    p_orders=self.operator_params["pspace_orders"],
                                    pspace_sizes=excitation_count)
            # regardless of restart status, initialize residual anew
            dT = ClusterOperator(self.system,
                                 order=self.operator_params["order"],
                                 p_orders=self.operator_params["pspace_orders"],
                                 pspace_sizes=excitation_count)
            # Run the CC calculation
            self.T, self.correlation_energy, _ = ccp_jacobi(update_function,
                                                            self.T,
                                                            dT,
                                                            self.hamiltonian,
                                                            self.system,
                                                            t3_excitations
                                                           )

        cc_calculation_summary(self.system.reference_energy, self.correlation_energy)
        print("   CC calculation ended on", get_timestamp())

    def run_hbar(self, method):
        # check if requested CC calculation is implemented in modules
        if "hbar_" + method.lower() not in ccpy.hbar.MODULES:
            raise NotImplementedError(
                "HBar for {} not implemented".format(method.lower())
            )

        # import the specific CC method module and get its update function
        hbar_mod = import_module("ccpy.hbar." + "hbar_" + method.lower())
        hbar_build_function = getattr(hbar_mod, 'build_hbar_' + method.lower())

        # Replace the driver hamiltonian with the Hbar
        self.hamiltonian = hbar_build_function(self.T, self.hamiltonian)

    def run_eomcc(self, method, state_index, t3_excitations=None, r3_excitations=None, guess_method="cis", multiplicity=None):
        """Performs the EOMCC calculation specified by the user in the input."""

        # check if requested CC calculation is implemented in modules
        if method.lower() not in ccpy.eomcc.MODULES:
            raise NotImplementedError(
                "{} not implemented".format(method.lower())
            )
        # Set operator parameters needed to build R
        self.set_operator_params(method)
        self.options["method"] = method.upper()

        # import the specific EOMCC method module and get its update function
        eom_module = import_module("ccpy.eomcc." + method.lower())
        HR_function = getattr(eom_module, 'HR')
        update_function = getattr(eom_module, 'update')
        # import the specific guess function
        guess_module = import_module("ccpy.eomcc." + guess_method.lower() + "_guess")
        guess_function = getattr(guess_module, "run_diagonalization")
        if multiplicity is None:
            multiplicity = self.system.multiplicity

        # Run the initial guess function
        omega, V = guess_function(self.system, self.hamiltonian, multiplicity)
        V, _ = np.linalg.qr(V[:, state_index])

        # Print the options as a header
        self.print_options()

        # Create the residual R that is re-used for each root
        dR = ClusterOperator(self.system,
                             order=self.operator_params["order"],
                             active_orders=self.operator_params["active_orders"],
                             num_active=self.operator_params["number_active_indices"])
        
        for i in state_index:
            print("   EOMCC calculation started on", get_timestamp())
            # if R[i] doesn't exist, then set it equal to initial guess
            if self.R[i] is None:
                self.R[i] = ClusterOperator(self.system,
                                            order=self.operator_params["order"],
                                            active_orders=self.operator_params["active_orders"],
                                            num_active=self.operator_params["number_active_indices"])
                self.R[i].unflatten(V[:, i - 1], order=1)
                self.vertical_excitation_energy[i] = omega[i - 1]

            self.R[i], self.vertical_excitation_energy[i], is_converged = eomcc_davidson(HR_function, update_function,
                                                                              self.R[i], dR, self.vertical_excitation_energy[i],
                                                                              self.T, self.hamiltonian, self.system, self.options)
            # Compute r0 a posteriori
            self.r0[i] = get_r0(self.R[i], self.hamiltonian, self.vertical_excitation_energy[i])
            eomcc_calculation_summary(self.vertical_excitation_energy[i], self.r0[i], is_converged)
            print("   EOMCC calculation ended on", get_timestamp())

    def run_leftcc(self, method, state_index=[0], t3_excitations=None, l3_excitations=None, pspace=None):
        # check if requested CC calculation is implemented in modules
        if method.lower() not in ccpy.left.MODULES:
            raise NotImplementedError(
                "{} not implemented".format(method.lower())
            )
        # Set operator parameters needed to build L
        self.set_operator_params(method)
        self.options["method"] = method.upper()

        # import the specific CC method module and get its update function
        lcc_mod = import_module("ccpy.left." + method.lower())
        update_function = getattr(lcc_mod, 'update')

        LR_function = None

        # regardless of restart status, initialize residual anew
        LH = ClusterOperator(self.system,
                             order=self.operator_params["order"],
                             active_orders=self.operator_params["active_orders"],
                             num_active=self.operator_params["number_active_indices"])
        for i in state_index:
            print("   Left CC alculation started on", get_timestamp())
            # decide whether this is a ground-state calculation
            if i == 0: 
                ground_state = True
            else:
                ground_state = False
                LR_function = partial(get_LR, self.R[i])
                #LR_function = lambda x: np.dot(x.flatten().T, self.R[i].flatten())

            # initialize the left CC operator anew, or use restart
            if self.L[i] is None:
                self.L[i] = ClusterOperator(self.system,
                                            order=self.operator_params["order"],
                                            active_orders=self.operator_params["active_orders"],
                                            num_active=self.operator_params["number_active_indices"])
                # set initial value based on ground- or excited-state
                if ground_state:
                    self.L[i].unflatten(self.T.flatten()[:self.L[i].ndim])
                else:
                    self.L[i].unflatten(self.R[i].flatten())

            # Zero out the residual
            LH.unflatten(0.0 * LH.flatten())

            #if pspace is None:
            self.L[i], _, LR, is_converged = left_cc_jacobi(update_function, self.L[i], LH, self.T, self.hamiltonian, 
                                                            LR_function, self.vertical_excitation_energy[i],
                                                            ground_state, self.system, self.options)
            #else:
            #    self.L[i], self.vertical_excitation_energy[i], LR, is_converged = left_ccp_jacobi(update_function,
            #                                                                                      self.L[i], LH, self.T, self.hamiltonian, 
            #                                                                                      LR_function, self.vertical_excitation_energy[i],
            #                                                                                      ground_state, self.system, self.options, pspace)
            if not ground_state:
                self.L[i].unflatten(1.0 / LR_function(self.L[i]) * self.L[i].flatten())

            leftcc_calculation_summary(self.vertical_excitation_energy[i], LR, is_converged)
            print("   Left CC calculation ended on", get_timestamp())


def eccc_driver(calculation, system, hamiltonian, external_wavefunction, T=None):
    """Performs the calculation specified by the user in the input."""
    from ccpy.extcorr.external_correction import cluster_analysis

    # Get the external T vector corresponding to the cluster analysis
    T_ext, VT_ext = cluster_analysis(external_wavefunction, hamiltonian, system)

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

    return T, total_energy, is_converged, T_ext



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
