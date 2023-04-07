"""Main calculation driver module of CCpy."""
import numpy as np
from importlib import import_module
from functools import partial

import ccpy.cc
import ccpy.hbar
import ccpy.left
import ccpy.eomcc

from ccpy.drivers.solvers import cc_jacobi, left_cc_jacobi, left_ccp_jacobi, eomcc_davidson, eccc_jacobi
from ccpy.drivers.cc_energy import get_LR, get_r0

from ccpy.models.integrals import Integral
from ccpy.models.operators import ClusterOperator, FockOperator
from ccpy.utilities.printing import get_timestamp, cc_calculation_summary, eomcc_calculation_summary, leftcc_calculation_summary

from ccpy.interfaces.pyscf_tools import load_pyscf_integrals
from ccpy.interfaces.gamess_tools import load_gamess_integrals


class Driver:

    @classmethod
    def from_pyscf(cls, meanfield, nfrozen, normal_ordered=True, dump_integrals=False, sorted=True):
        return cls(
            *load_pyscf_integrals(meanfield, nfrozen, normal_ordered=normal_ordered, dump_integrals=dump_integrals,
                                  sorted=sorted))

    @classmethod
    def from_gamess(cls, logfile, nfrozen, fcidump=None, onebody=None, twobody=None, normal_ordered=True, sorted=True, data_type=np.float64):
        return cls(*load_gamess_integrals(logfile, fcidump, onebody, twobody, nfrozen, normal_ordered=normal_ordered, sorted=sorted,
                                   data_type=data_type))

    def __init__(self, system, hamiltonian, max_number_states=100):
        self.system = system
        self.hamiltonian = hamiltonian
        self.flag_hbar = False
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
        self.L = [None] * max_number_states
        self.R = [None] * max_number_states
        self.correlation_energy = 0.0
        self.vertical_excitation_energy = np.zeros(max_number_states)
        self.r0 = np.zeros(max_number_states)
        self.deltapq = [None] * max_number_states
        self.ddeltapq = [None] * max_number_states

        # Store alpha and beta fock matrices for later usage before HBar overwrites bare Hamiltonian
        self.fock = Integral.from_empty(system, 1, data_type=self.hamiltonian.a.oo.dtype)
        self.fock.a.oo = self.hamiltonian.a.oo.copy()
        self.fock.b.oo = self.hamiltonian.b.oo.copy()
        self.fock.a.vv = self.hamiltonian.a.vv.copy()
        self.fock.b.vv = self.hamiltonian.b.vv.copy()

    def set_operator_params(self, method):
        if method.lower() in ["ccd", "ccsd", "eomccsd", "left_ccsd", "eccc2"]:
            self.operator_params["order"] = 2
            self.operator_params["number_particles"] = 2
            self.operator_params["number_holes"] = 2
        elif method.lower() in ["ccsdt", "eomccsdt", "left_ccsdt", "left_ccsdt_p_slow"]:
            self.operator_params["order"] = 3
            self.operator_params["number_particles"] = 3
            self.operator_params["number_holes"] = 3
        elif method.lower() in ["ccsdtq"]:
            self.operator_params["order"] = 4
            self.operator_params["number_particles"] = 4
            self.operator_params["number_holes"] = 4
        elif method.lower() in ["ccsdt1", "eomccsdt1"]:
            self.operator_params["order"] = 3
            self.operator_params["number_particles"] = 3
            self.operator_params["number_holes"] = 3
            self.operator_params["active_orders"] = [3]
            self.operator_params["number_active_indices"] = [1]
        elif method.lower() in ["ccsdt_p"]:
            self.operator_params["order"] = 3
            self.operator_params["number_particles"] = 3
            self.operator_params["number_holes"] = 3
            self.operator_params["pspace_orders"] = [3]
        elif method.lower() in ["ipeom2", "left_ipeom2"]:
            self.order = 2
            self.num_particles = 1
            self.num_holes = 2
        elif method.lower() in ["ipeom3", "left_ipeom3"]:
            self.order = 3
            self.num_particles = 2
            self.num_holes = 3
        elif method.lower() in ["eaeom2", "left_eaeom2"]:
            self.order = 2
            self.num_particles = 2
            self.num_holes = 1
        elif method.lower() in ["eaeom3", "left_eaeom3"]:
            self.order = 3
            self.num_particles = 3
            self.num_holes = 2
        elif method.lower() in ["dipeom3", "left_dipeom3"]:
            self.order = 3
            self.num_particles = 1
            self.num_holes = 3
        elif method.lower() in ["dipeom4", "left_dipeom4"]:
            self.order = 4
            self.num_particles = 2
            self.num_holes = 4
        elif method.lower() in ["deaeom3", "left_deaeom3"]:
            self.order = 3
            self.num_particles = 3
            self.num_holes = 1
        elif method.lower() in ["deaeom4", "left_deaeom4"]:
            self.order = 4
            self.num_particles = 4
            self.num_holes = 2
            
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

        # Create either the standard CC or CC(P) cluster operator
        if t3_excitations is None:
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
        else:
            # Get dimensions of T3 spincases in P space
            n3aaa = t3_excitations["aaa"].shape[0]
            n3aab = t3_excitations["aab"].shape[0]
            n3abb = t3_excitations["abb"].shape[0]
            n3bbb = t3_excitations["bbb"].shape[0]
            excitation_count = [[n3aaa, n3aab, n3abb, n3bbb]]

            # If RHF, copy aab into abb and aaa in bbb
            if self.options["RHF_symmetry"]:
                assert (n3aaa == n3bbb)
                assert (n3aab == n3abb)
                t3_excitations["bbb"] = t3_excitations["aaa"].copy()
                t3_excitations["abb"] = t3_excitations["aab"][:, [2, 0, 1, 5, 3, 4]]  # want abb excitations as a b~<c~ i j~<k~; MUST be this order!

            if self.T is None:
                self.T = ClusterOperator(self.system,
                                         order=self.operator_params["order"],
                                         p_orders=self.operator_params["pspace_orders"],
                                         pspace_sizes=excitation_count)
            # regardless of restart status, initialize residual anew
            dT = ClusterOperator(self.system,
                                 order=self.operator_params["order"],
                                 p_orders=self.operator_params["pspace_orders"],
                                 pspace_sizes=excitation_count)
        # Run the CC calculation
        # NOTE: It may not look like it, but t3_excitations is permuted and matches T at this point. It changes from its value at input!
        self.T, self.correlation_energy, _ = cc_jacobi(update_function,
                                                self.T,
                                                dT,
                                                self.hamiltonian,
                                                self.system,
                                                self.options,
                                                t3_excitations,
                                               )
        cc_calculation_summary(self.T, self.system.reference_energy, self.correlation_energy, self.system)
        print("   CC calculation ended on", get_timestamp())

    def run_hbar(self, method, t3_excitations=None):
        # check if requested CC calculation is implemented in modules
        if "hbar_" + method.lower() not in ccpy.hbar.MODULES:
            raise NotImplementedError(
                "HBar for {} not implemented".format(method.lower())
            )

        # import the specific CC method module and get its update function
        hbar_mod = import_module("ccpy.hbar." + "hbar_" + method.lower())
        hbar_build_function = getattr(hbar_mod, 'build_hbar_' + method.lower())

        # Replace the driver hamiltonian with the Hbar
        print("")
        print("   HBar construction began on", get_timestamp(), end="")
        self.hamiltonian = hbar_build_function(self.T, self.hamiltonian, self.system)
        print("... completed on", get_timestamp(), "\n")
        # Set flag indicating that hamiltonian is set to Hbar is now true
        self.flag_hbar = True

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

        # Ensure that Hbar is set upon entry
        assert(self.flag_hbar)

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

        for i in state_index:
            if self.R[i] is None:
                self.R[i] = ClusterOperator(self.system,
                                            order=self.operator_params["order"],
                                            active_orders=self.operator_params["active_orders"],
                                            num_active=self.operator_params["number_active_indices"])
                self.R[i].unflatten(V[:, i - 1], order=1)
                self.vertical_excitation_energy[i] = omega[i - 1]

        # Form the initial subspace vectors
        B0, _ = np.linalg.qr(np.asarray([self.R[i].flatten() for i in state_index]).T)

        # Print the options as a header
        self.print_options()

        # Create the residual R that is re-used for each root
        dR = ClusterOperator(self.system,
                             order=self.operator_params["order"],
                             active_orders=self.operator_params["active_orders"],
                             num_active=self.operator_params["number_active_indices"])
        
        ct = 0
        for i in state_index:
            print("   EOMCC calculation for root %d started on" % i, get_timestamp())
            self.R[i], self.vertical_excitation_energy[i], is_converged = eomcc_davidson(HR_function, update_function, B0[:, ct],
                                                                              self.R[i], dR, self.vertical_excitation_energy[i],
                                                                              self.T, self.hamiltonian, self.system, self.options)
            # Compute r0 a posteriori
            self.r0[i] = get_r0(self.R[i], self.hamiltonian, self.vertical_excitation_energy[i])
            eomcc_calculation_summary(self.R[i], self.vertical_excitation_energy[i], self.r0[i], is_converged, self.system)
            print("   EOMCC calculation for root %d ended on" % i, get_timestamp(), "\n")
            ct += 1

    def run_leftcc(self, method, state_index=[0], t3_excitations=None, l3_excitations=None, pspace=None):
        # check if requested CC calculation is implemented in modules
        if method.lower() not in ccpy.left.MODULES:
            raise NotImplementedError(
                "{} not implemented".format(method.lower())
            )
        # Set operator parameters needed to build L
        self.set_operator_params(method)
        self.options["method"] = method.upper()

        # Ensure that Hbar is set upon entry
        assert(self.flag_hbar)

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
            print("   Left CC calculation for root %d started on" % i, get_timestamp())
            # decide whether this is a ground-state calculation
            if i == 0: 
                ground_state = True
            else:
                ground_state = False
                LR_function = partial(get_LR, self.R[i])

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

            if pspace is None:
                self.L[i], _, LR, is_converged = left_cc_jacobi(update_function, self.L[i], LH, self.T, self.hamiltonian, 
                                                                LR_function, self.vertical_excitation_energy[i],
                                                                ground_state, self.system, self.options)
            else:
                self.L[i], _, LR, is_converged = left_ccp_jacobi(update_function, self.L[i], LH, self.T, self.hamiltonian, 
                                                                 LR_function, self.vertical_excitation_energy[i],
                                                                 ground_state, self.system, self.options, pspace)
            if not ground_state:
                self.L[i].unflatten(1.0 / LR_function(self.L[i]) * self.L[i].flatten())

            leftcc_calculation_summary(self.L[i], self.vertical_excitation_energy[i], LR, is_converged, self.system)
            print("   Left CC calculation for root %d ended on" % i, get_timestamp(), "\n")

    def run_eccc(self, method, external_wavefunction, t3_excitations=None):
        from ccpy.extcorr.external_correction import cluster_analysis

        # Get the external T vector corresponding to the cluster analysis
        T_ext, VT_ext = cluster_analysis(external_wavefunction, self.hamiltonian, self.system)

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
        print("   ec-CC calculation started on", get_timestamp())
        self.print_options()

        if self.T is None:
            self.T = ClusterOperator(self.system,
                                     order=self.operator_params["order"],
                                     active_orders=self.operator_params["active_orders"],
                                     num_active=self.operator_params["number_active_indices"])
            # Set the initial T1 and T2 components to that from cluster analysis
            setattr(self.T, 'a', T_ext.a)
            setattr(self.T, 'b', T_ext.b)
            setattr(self.T, 'aa', T_ext.aa)
            setattr(self.T, 'ab', T_ext.ab)
            setattr(self.T, 'bb', T_ext.bb)

        # regardless of restart status, initialize residual anew
        dT = ClusterOperator(self.system,
                             order=self.operator_params["order"],
                             active_orders=self.operator_params["active_orders"],
                             num_active=self.operator_params["number_active_indices"])
        # Run the CC calculation
        self.T, self.correlation_energy, _ = eccc_jacobi(update_function,
                                                  self.T,
                                                  dT,
                                                  self.hamiltonian,
                                                  T_ext, VT_ext,
                                                  self.system,
                                                  self.options,
                                               )

        cc_calculation_summary(self.T, self.system.reference_energy, self.correlation_energy, self.system)
        print("   ec-CC calculation ended on", get_timestamp())

    def run_ccp3(self, method, state_index=[0], two_body_approx=True, t3_excitations=None, l3_excitations=None, r3_excitations=None, pspace=None):

        if method.lower() == "crcc23":
            from ccpy.moments.crcc23 import calc_crcc23
            from ccpy.moments.creomcc23 import calc_creomcc23
            # Ensure that HBar is set upon entry
            assert(self.flag_hbar)
            for i in state_index:
                # Perform ground-state correction
                if i == 0:
                    _, self.deltapq[i] = calc_crcc23(self.T, self.L[i], self.correlation_energy, self.hamiltonian, self.fock, self.system, self.options["RHF_symmetry"])
                else:
                    # Perform excited-state corrections
                    _, self.deltapq[i], self.ddeltapq[i] = calc_creomcc23(self.T, self.R[i], self.L[i], self.r0[i],
                                                                          self.vertical_excitation_energy[i], self.correlation_energy, self.hamiltonian, self.fock,
                                                                          self.system, self.options["RHF_symmetry"])
        elif method.lower() == "ccsd(t)":
            from ccpy.moments.crcc23 import calc_ccsdpt
            # Warn the user if they run using HBar instead of H; we will not disallow it, however
            if self.flag_hbar:
                print("WARNING: CCSD(T) is using similarity-transformed Hamiltonian! Results will not match conventional CCSD(T)!")
            _, self.deltapq[0] = calc_ccsdpt(self.T, self.correlation_energy, self.hamiltonian, self.system, self.options["RHF_symmetry"])

        elif method.lower() == "crcc24":
            from ccpy.moments.crcc24 import calc_crcc24
            # Ensure that HBar is set
            assert(self.flag_hbar)
            # Perform ground-state correction
            _, self.deltapq[0] = calc_crcc24(self.T, self.L[0], self.correlation_energy, self.hamiltonian, self.fock, self.system, self.options["RHF_symmetry"])

        elif method.lower() == "cct3":
            from ccpy.moments.cct3 import calc_cct3
            # Ensure that HBar is set
            assert(self.flag_hbar)
            # Perform ground-state correction
            _, self.deltapq[0] = calc_cct3(self.T, self.L[0], self.correlation_energy, self.hamiltonian, self.fock, self.system,
                                           self.options["RHF_symmetry"], num_active=self.operator_params["number_active_indices"])

        elif method.lower() == "ccp3":
            from ccpy.moments.ccp3 import calc_ccp3_2ba, calc_ccp3_full
            # Ensure that both HBar and pspace are set
            assert(self.flag_hbar)
            assert(pspace)
            # Perform ground-state correction
            if two_body_approx: # Use the 2BA (requires only L1, L2 and HBar of CCSD)
                _, self.deltapq[0] = calc_ccp3_2ba(self.T, self.L[0], self.correlation_energy, self.hamiltonian, self.fock, self.system, pspace, self.options["RHF_symmetry"])
            else: # full correction (requires L1, L2, and L3 as well as HBar of CCSDt)
                _, self.delta_pq[0] = calc_ccp3_full(self.T, self.L[0], self.correlation_energy, self.hamiltonian, self.fock, self.system, pspace, self.options["RHF_symmetry"])
        else:
            raise NotImplementedError("Triples correction {} not implemented".format(method.lower()))




# def mrcc_driver(calculation, system, hamiltonian, model_space, two_body_approximation=True):
#     """Performs the calculation specified by the user in the input."""
#
#     # check if requested CC calculation is implemented in modules
#     if calculation.calculation_type not in ccpy.mrcc.MODULES:
#         raise NotImplementedError(
#             "{} not implemented".format(calculation.calculation_type)
#         )
#
#     # import the specific CC method module and get its update function
#     cc_mod = import_module("ccpy.mrcc." + calculation.calculation_type.lower())
#     heff_mod = import_module("ccpy.mrcc.effective_hamiltonian")
#     update_function = getattr(cc_mod, 'update')
#
#     if two_body_approximation: # use the Heff matrix elements computed using T(p) = T1(p) + T2(p)
#         compute_Heff_function = getattr(heff_mod, 'compute_Heff_mkmrccsd')
#     else:
#         compute_Heff_function = getattr(heff_mod, 'compute_Heff_' + calculation.calculation_type.lower())
#
#     cc_printer = CCPrinter(calculation)
#     cc_printer.cc_header()
#
#     # initialize the cluster operator anew, or use restart
#     # one used in the calculation. For example, we could not start a CCSDT
#     # calculation using the CCSD cluster amplitudes.
#     T = [None for i in range(len(model_space))]
#     dT = [None for i in range(len(model_space))]
#     for p in range(len(model_space)):
#         if T[p] is None:
#             T[p] = ClusterOperator(system,
#                                 order=calculation.order,
#                                 active_orders=calculation.active_orders,
#                                 num_active=calculation.num_active)
#
#         # regardless of restart status, initialize residual anew
#         dT[p] = ClusterOperator(system,
#                             order=calculation.order,
#                             active_orders=calculation.active_orders,
#                             num_active=calculation.num_active)
#
#     T, total_energy, is_converged = mrcc_jacobi(
#                                            update_function,
#                                            compute_Heff_function,
#                                            T,
#                                            dT,
#                                            model_space,
#                                            hamiltonian,
#                                            calculation,
#                                            system,
#                                            )
#
#
#     #total_energy = system.reference_energy + corr_energy
#
#     #cc_printer.cc_calculation_summary(system.reference_energy, corr_energy)
#
#     return T, total_energy, is_converged
