import numpy as np
import time
from copy import deepcopy

from ccpy.extrapolation.goodson_extrapolation import goodson_extrapolation
from ccpy.utilities.printing import get_timestamp
from ccpy.constants.constants import hartreetoeV

class AdaptDriver:

    def __init__(self, driver, percentage=None):
        self.driver = driver
        self.percentage = percentage
        self.options = {"two_body_approx": True,
                        "reset_amplitudes": False,
                        "energy_tolerance": 1.0e-04,
                        "maximum_iterations": 10,
                        "n_det_max": 100000000,
                        "selection_factor": 1.0,
                        "base_growth": "ccsd",
                        "buffer_factor": 2,
                        "minimum_threshold": 0.0}
        if percentage is None:
            self.nmacro = self.options["maximum_iterations"] + 1
        else:
            self.nmacro = len(self.percentage)
        self.energy_tolerance = self.options["energy_tolerance"]
        # energy containers
        self.ccp_energy = np.zeros(self.nmacro)
        self.ccpq_energy = np.zeros(self.nmacro)
        # extrapolated energy containers
        self.ex_ccq = np.zeros(self.nmacro)
        self.ex_ccr = np.zeros(self.nmacro)
        self.ex_cccf = np.zeros(self.nmacro)
        # t3 excitations
        self.t3_excitations = {"aaa": np.ones((1, 6), order="F"),
                               "aab": np.ones((1, 6), order="F"),
                               "abb": np.ones((1, 6), order="F"),
                               "bbb": np.ones((1, 6), order="F")}
        self.excitation_count_by_symmetry = [{'aaa': 0, 'aab': 0, 'abb': 0, 'bbb': 0} for _ in range(len(self.driver.system.point_group_irrep_to_number))]
        self.n_det = 0
        # Save the bare Hamiltonian for later iterations if using CR-CC(2,3)
        self.bare_hamiltonian = deepcopy(self.driver.hamiltonian)

    def print_options(self):
        print("   ------------------------------------------")
        for option_key, option_value in self.options.items():
            print("  ", option_key, "=", option_value)
        print("   ------------------------------------------\n")

    def excitation_count(self):
        """Performs an initial symmetry-adapted count of the relevant excitation
           space to determine the growth increment for each iteration of the
           calculation."""
        from ccpy.utilities.symmetry import count_singles, count_doubles, count_triples
        # Use fixed growth size equal to number of determinants in CCS or CCSD P space
        if self.percentage is None:
            self.num_excitations_symmetry, _ = count_singles(self.driver.system)
            if self.options["base_growth"].lower() == "ccsd":
                num_doubles_symmetry, _ = count_doubles(self.driver.system)
                self.num_excitations_symmetry = [x + y for x, y in zip(self.num_excitations_symmetry, num_doubles_symmetry)]
            # Get the increment of either CCS or CCSD adding 1 for reference determinant
            self.one_increment = self.num_excitations_symmetry[self.driver.system.point_group_irrep_to_number[self.driver.system.reference_symmetry]]
            self.num_dets_to_add = [int(self.one_increment * self.options["selection_factor"]) for _ in range(1, self.nmacro)]
            # Store the base value of n_det as the number of singles + doubles plus the reference determinant
            self.base_pspace_size = self.one_increment + 1
            self.n_det = self.base_pspace_size
        # Use the original %T scheme
        else:
            # Count the triples using symmetry
            self.num_excitations_symmetry, _ = count_triples(self.driver.system)
            self.num_total_excitations = self.num_excitations_symmetry[self.driver.system.point_group_irrep_to_number[self.driver.system.reference_symmetry]]
            self.one_increment = int(0.01 * self.num_total_excitations)
            # Setting up the number of added determinants in each iteration
            self.num_dets_to_add = np.zeros(len(self.percentage))
            for i in range(len(self.percentage) - 1):
                if self.percentage[i + 1] == 100.0:
                    self.num_dets_to_add[i] = self.num_total_excitations - self.one_increment * self.percentage[i]
                else:
                    self.num_dets_to_add[i] = self.one_increment * (self.percentage[i + 1] - self.percentage[i])
            self.num_dets_to_add[-1] = 1
            # Set the base P space size to 0 (technically it should be all singles and doubles here too)
            self.base_pspace_size = 0
        # Adjust for RHF symmetry
        if self.driver.options["RHF_symmetry"]:
            for i in range(len(self.num_dets_to_add)):
                self.num_dets_to_add[i] = int(self.num_dets_to_add[i] / 2)

    def print_pspace(self):
        """Counts and analyzes the P space in terms of spatial and Sz-spin symmetry."""
        print("   Total number of determinants in P space =", self.n_det)
        for isym, excitation_count_irrep in enumerate(self.excitation_count_by_symmetry):
            tot_excitation_count_irrep = excitation_count_irrep['aaa'] + excitation_count_irrep['aab'] + \
                                         excitation_count_irrep['abb'] + excitation_count_irrep['bbb']
            print("   Symmetry", self.driver.system.point_group_number_to_irrep[isym], "-", "Total number of triples in P space = ", tot_excitation_count_irrep)
            print("      Number of aaa = ", excitation_count_irrep['aaa'])
            print("      Number of aab = ", excitation_count_irrep['aab'])
            print("      Number of abb = ", excitation_count_irrep['abb'])
            print("      Number of bbb = ", excitation_count_irrep['bbb'])
        print("")

    def run_ccp(self, imacro):
        """Runs iterative CC(P), and if needed, HBar and iterative left-CC calculations."""
        self.driver.run_ccp(method="ccsdt_p", t3_excitations=self.t3_excitations)
        if self.options["two_body_approx"]:
            self.driver.run_hbar(method="ccsd")
            self.driver.run_leftcc(method="left_ccsd")
        else:
            self.driver.run_hbar(method="ccsdt_p", t3_excitations=self.t3_excitations)
            self.driver.run_leftccp(method="left_ccsdt_p", state_index=[0], t3_excitations=self.t3_excitations)
        self.ccp_energy[imacro] = self.driver.system.reference_energy + self.driver.correlation_energy

    def run_ccp_chol(self, imacro):
        """Runs iterative CC(P), and if needed, HBar and iterative left-CC calculations."""
        self.driver.run_ccp(method="ccsdt_p_chol", t3_excitations=self.t3_excitations)
        if self.options["two_body_approx"]:
            self.driver.run_hbar(method="ccsd_chol")
            self.driver.run_leftcc(method="left_ccsd_chol")
        else:
            self.driver.run_hbar(method="ccsdt_p_chol", t3_excitations=self.t3_excitations)
            self.driver.run_leftccp(method="left_ccsdt_p_chol", state_index=[0], t3_excitations=self.t3_excitations)
        self.ccp_energy[imacro] = self.driver.system.reference_energy + self.driver.correlation_energy

    def run_ccp3(self, imacro):
        """Runs the CC(P;3) correction using either the CR-CC(2,3)- or CCSD(T)-like approach,
           while simultaneously selecting the leading triply excited determinants and returning
           the result in an array. For the last calculation, this should not perform the
           selection steps."""
        from ccpy.moments.ccp3 import calc_ccp3_2ba_with_selection, calc_ccp3_with_selection

        if imacro < self.nmacro - 1:
            if self.options["two_body_approx"]:
                self.ccpq_energy[imacro], triples_list = calc_ccp3_2ba_with_selection(self.driver.T,
                                                                                      self.driver.L[0],
                                                                                      self.t3_excitations,
                                                                                      self.driver.correlation_energy,
                                                                                      self.driver.hamiltonian,
                                                                                      self.bare_hamiltonian,
                                                                                      self.driver.system,
                                                                                      self.num_dets_to_add[imacro],
                                                                                      use_RHF=self.driver.options["RHF_symmetry"],
                                                                                      min_thresh=self.options["minimum_threshold"],
                                                                                      buffer_factor=self.options["buffer_factor"])
            else:
                self.ccpq_energy[imacro], triples_list = calc_ccp3_with_selection(self.driver.T,
                                                                                      self.driver.L[0],
                                                                                      self.t3_excitations,
                                                                                      self.driver.correlation_energy,
                                                                                      self.driver.hamiltonian,
                                                                                      self.bare_hamiltonian,
                                                                                      self.driver.system,
                                                                                      self.num_dets_to_add[imacro],
                                                                                      use_RHF=self.driver.options["RHF_symmetry"],
                                                                                      min_thresh=self.options["minimum_threshold"],
                                                                                      buffer_factor=self.options["buffer_factor"])
        else:
            triples_list = []
            self.driver.run_ccp3(method="ccp3", state_index=0, two_body_approx=self.options["two_body_approx"], t3_excitations=self.t3_excitations)
            self.ccpq_energy[imacro] = self.driver.system.reference_energy + self.driver.correlation_energy + self.driver.deltap3[0]["D"]

        return triples_list

    def run_ccp3_chol(self, imacro):
        """Runs the CC(P;3) correction using either the CR-CC(2,3)- or CCSD(T)-like approach,
           while simultaneously selecting the leading triply excited determinants and returning
           the result in an array. For the last calculation, this should not perform the
           selection steps."""
        from ccpy.moments.ccp3_chol import calc_ccp3_2ba_with_selection

        if imacro < self.nmacro - 1:
            if self.options["two_body_approx"]:
                self.ccpq_energy[imacro], triples_list = calc_ccp3_2ba_with_selection(self.driver.T,
                                                                                      self.driver.L[0],
                                                                                      self.t3_excitations,
                                                                                      self.driver.correlation_energy,
                                                                                      self.driver.hamiltonian,
                                                                                      self.bare_hamiltonian,
                                                                                      self.driver.system,
                                                                                      self.num_dets_to_add[imacro],
                                                                                      use_RHF=self.driver.options[
                                                                                          "RHF_symmetry"],
                                                                                      min_thresh=self.options[
                                                                                          "minimum_threshold"],
                                                                                      buffer_factor=self.options[
                                                                                          "buffer_factor"])
            else:
                pass
        else:
            triples_list = []
            self.driver.run_ccp3(method="ccp3_chol", state_index=0, two_body_approx=self.options["two_body_approx"],
                                 t3_excitations=self.t3_excitations)
            self.ccpq_energy[imacro] = self.driver.system.reference_energy + self.driver.correlation_energy + \
                                       self.driver.deltap3[0]["D"]

        return triples_list

    def run_expand_pspace(self, triples_list):
        """This will expand the P space using the list of triply excited determinants identified
           using the CC(P;Q) moment expansions, above."""
        from ccpy.utilities.selection import add_spinorbital_triples_to_pspace
        self.t3_excitations, self.excitation_count_by_symmetry = add_spinorbital_triples_to_pspace(triples_list,
                                                                                                   self.t3_excitations,
                                                                                                   self.excitation_count_by_symmetry,
                                                                                                   self.driver.system,
                                                                                                   self.driver.options["RHF_symmetry"])

    def run(self):
        """This is the main driver for the entire adaptive CC(P;Q) calculation. It will call the above
           methods in the correct sequence and handle logic accordingly."""
        # Print the options as a header
        print("   Adaptive CC(P;Q) calculation started on", get_timestamp())
        self.print_options()

        # Step 0a: Perform the preliminary excitation counting
        print("   Preliminary excitation count...", end=" ")
        t1 = time.perf_counter()
        self.excitation_count()
        print("completed in", time.perf_counter() - t1, "seconds")
        # Step 0b: Print the results of excitation count as well as determinant addition plan
        print("   Excitation Count Summary:")
        spin_fact = 1
        if self.driver.options["RHF_symmetry"]:
            spin_fact = 2
        for i, count in enumerate(self.num_excitations_symmetry):
            symmetry = self.driver.system.point_group_number_to_irrep[i]
            print("      Symmetry", symmetry, " = ", count)
        #print("      Using", self.num_total_excitations, "as total for ground state.")
        #print("      Determinant addition plan:", [int("{0:.0f}".format(v, i)) for i, v in enumerate(self.num_dets_to_add[:-1] * spin_fact)])
        print("")

        # Begin adaptive loop iterations
        for imacro in range(self.nmacro):
            print("")
            print("   Adaptive CC(P;Q) Macroiteration - ", imacro)
            print("   ===========================================")
            # Offset needed to count number of determinants in P space correctly
            # Offset needed to count number of determinants in P space correctly
            offset = 0
            if np.array_equal(self.t3_excitations["aaa"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
                offset += 1
            if np.array_equal(self.t3_excitations["aab"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
                offset += 1
            if np.array_equal(self.t3_excitations["abb"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
                offset += 1
            if np.array_equal(self.t3_excitations["bbb"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
                offset += 1

            # Update n_det
            self.n_det = self.base_pspace_size + (
                            self.t3_excitations["aaa"].shape[0]
                            + self.t3_excitations["aab"].shape[0]
                            + self.t3_excitations["abb"].shape[0]
                            + self.t3_excitations["bbb"].shape[0]
                            - offset
            )

            # Step 1: Analyze the P space (optional)
            x1 = time.perf_counter()
            self.print_pspace()
            x2 = time.perf_counter()
            t_pspace_printing = x2 - x1

            # Step 2: Run CC(P) on this P space
            x1 = time.perf_counter()
            if self.driver.system.cholesky:
                self.run_ccp_chol(imacro)
            else:
                self.run_ccp(imacro)
            x2 = time.perf_counter()
            t_ccp = x2 - x1

            # Step 3: Moment correction + adaptive selection
            x1 = time.perf_counter()
            if self.driver.system.cholesky:
                selection_arr = self.run_ccp3_chol(imacro)
            else:
                selection_arr = self.run_ccp3(imacro)
            x2 = time.perf_counter()
            t_selection_and_ccp3 = x2 - x1

            # Step 4: Perform Goodson extrapolation
            x1 = time.perf_counter()
            print("   Goodson FCI Extrapolation")
            print("   -------------------------")
            self.ex_ccq[imacro] = goodson_extrapolation(self.driver.system.reference_energy,
                                                        self.ccp_energy[imacro],
                                                        self.ccpq_energy[imacro],
                                                        approximant="ccq")
            self.ex_ccr[imacro] = goodson_extrapolation(self.driver.system.reference_energy,
                                                        self.ccp_energy[imacro],
                                                        self.ccpq_energy[imacro],
                                                        approximant="ccr")
            self.ex_cccf[imacro] = goodson_extrapolation(self.driver.system.reference_energy,
                                                         self.ccp_energy[imacro],
                                                         self.ccpq_energy[imacro],
                                                         approximant="cccf")
            print("   ex-CCq Total Energy =", self.ex_ccq[imacro])
            print("   ex-CCr Total Energy =", self.ex_ccr[imacro])
            print("   ex-CCcf Total Energy =", self.ex_cccf[imacro])
            print("")
            x2 = time.perf_counter()
            t_extrap = x2 - x1

            # Check convergence conditions
            if imacro > 0:
                delta_e_ccp = self.ccp_energy[imacro] - self.ccp_energy[imacro - 1]
                delta_e_ccpq = self.ccpq_energy[imacro] - self.ccpq_energy[imacro - 1]
                print("   Change in CC(P) energy = ", delta_e_ccp)
                print("   Change in CC(P;Q) energy = ", delta_e_ccpq, "\n")
                # Energy condition
                if abs(delta_e_ccpq) < self.options["energy_tolerance"]:
                    print("   Adaptive CC(P;Q) calculation converged to within energy tolerance!")
                    break
                # N_det condition
                if self.n_det >= self.options["n_det_max"]:
                    print(f"   Adaptive CC(P;Q) calculation reached maximum dimension of P space (n_det_max = {self.options['n_det_max']})")
                    break
            if imacro == self.nmacro - 1:
                print(f"   Adaptive CC(P;Q) reached maximum number of iterations (maximum_iterations = {self.options['maximum_iterations']})")
                break

            # Step 5: Expand the P space
            x1 = time.perf_counter()
            self.run_expand_pspace(selection_arr)
            x2 = time.perf_counter()
            t_pspace_expand = x2 - x1

            # Step 6: Reset variables in driver
            if self.options["reset_amplitudes"]:
                self.driver.T = None
                self.driver.L[0] = None

            #setattr(self.driver, "hamiltonian", self.bare_hamiltonian)
            # Since the HBar routines are now using H = H0, so there is no deepcopy of the hamiltonian there,
            # we need to use deepcopy every imacro iteration to reset the driver Hamiltonian
            self.driver.hamiltonian = deepcopy(self.bare_hamiltonian)

            # Step 7: Report wall and CPU timings of each step
            print(f"   Timing breakdown for macrostep {imacro}")
            print("   ---------------------------------")
            print(f"   - P space printing took {t_pspace_printing:.2f} seconds")
            print(f"   - CC(P) took {t_ccp:.2f} seconds")
            print(f"   - CC(P;Q) + selection took {t_selection_and_ccp3:.2f} seconds")
            print(f"   - Goodson extrapolation took {t_extrap:.2f} seconds")
            print(f"   - Expanding P space took {t_pspace_expand:.2f} seconds")
            print(f"   - Total time: {t_extrap + t_pspace_printing + t_pspace_expand + t_ccp + t_selection_and_ccp3 + t_pspace_expand:.2f} seconds")

        # Final printout of the results including energy extrapolations
        print("   Adaptive CC(P;Q) calculation ended on", get_timestamp(), "\n")
        print("   Summary of results:")
        print("    Iteration           E(P)             E(P;Q)             ex-CCq             ex-CCr             ex-CCcf")
        print("   -------------------------------------------------------------------------------------------------------")
        for i in range(imacro + 1):
            print("   %8d    %.10f     %.10f     %.10f     %.10f     %.10f" %
                  (i,
                   self.ccp_energy[i],
                   self.ccpq_energy[i],
                   self.ex_ccq[i],
                   self.ex_ccr[i],
                   self.ex_cccf[i]
                   )
            )

class AdaptEOMDriverSS:

    def __init__(self, driver, state_index, roots_per_irrep, multiplicity, nacto=0, nactu=0, percentage=None):
        self.driver = driver
        # inputs used to form the one-time EOMCCSd initial guess
        self.state_index = state_index
        self.roots_per_irrep = roots_per_irrep
        self.multiplicity = multiplicity
        self.nacto = nacto
        self.nactu = nactu
        # Get the state irrep from the roots_per_irrep input
        irrep_list = []
        for irrep, nguess in self.roots_per_irrep.items():
            for k in range(nguess):
                irrep_list.append(irrep)
        self.state_irrep = irrep_list[self.state_index - 1]
        #
        self.percentage = percentage
        #
        self.options = {"reset_amplitudes": False,
                        "energy_tolerance": 1.0e-04,
                        "maximum_iterations": 10,
                        "n_det_max": 100000000,
                        "selection_factor": 1.0,
                        "base_growth": "ccsd",
                        "buffer_factor": 2,
                        "minimum_threshold": 0.0,
                        "p_space_selection": 3}
        # options["p_space_selection"] = 1, 2, or 3 controls how the P spaces for the states considered are constructed
        # 1 - Uses the P space for the lowest space of a given symmetry
        # 2 - Uses a unified P space for all states being considered
        # 3 (default) - Uses an individual P space adapted to each state; this option can formally break size-intensivity!
        #
        self.nmacro = len(percentage)
        self.energy_tolerance = self.options["energy_tolerance"]
        # energy containers
        self.ccp_energy = np.zeros(self.nmacro)
        self.ccpq_energy = np.zeros(self.nmacro)
        self.eomccp_energy = np.zeros(self.nmacro)
        self.eomccpq_energy = np.zeros(self.nmacro)
        # t3 excitations and r3_excitations
        self.t3_excitations = {"aaa": np.ones((1, 6), order="F"),
                               "aab": np.ones((1, 6), order="F"),
                               "abb": np.ones((1, 6), order="F"),
                               "bbb": np.ones((1, 6), order="F")}
        self.r3_excitations = {"aaa": np.ones((1, 6), order="F"),
                               "aab": np.ones((1, 6), order="F"),
                               "abb": np.ones((1, 6), order="F"),
                               "bbb": np.ones((1, 6), order="F")}
        self.excitation_count_by_symmetry_t = [{'aaa': 0, 'aab': 0, 'abb': 0, 'bbb': 0} for _ in range(len(self.driver.system.point_group_irrep_to_number))]
        self.excitation_count_by_symmetry_r = [{'aaa': 0, 'aab': 0, 'abb': 0, 'bbb': 0} for _ in range(len(self.driver.system.point_group_irrep_to_number))]
        self.n_det_t = 0
        self.n_det_r = 0
        # RHF flags for ground and excited states
        self.RHF_ground = self.driver.options["RHF_symmetry"]
        self.RHF_excited = True if self.multiplicity == 1 else False
        # Save the bare Hamiltonian for later iterations if using CR-CC(2,3)
        self.bare_hamiltonian = deepcopy(self.driver.hamiltonian)

    def print_options(self):
        print("   ------------------------------------------")
        for option_key, option_value in self.options.items():
            print("  ", option_key, "=", option_value)
        print("   ------------------------------------------\n")

    def excitation_count(self):
        """Performs an initial symmetry-adapted count of the relevant excitation
           space to determine the growth increment for each iteration of the
           calculation."""
        from ccpy.utilities.symmetry import count_singles, count_doubles, count_triples
        # Count the triples using symmetry
        self.num_excitations_symmetry, _ = count_triples(self.driver.system)
        self.num_total_excitations_t = self.num_excitations_symmetry[self.driver.system.point_group_irrep_to_number[self.driver.system.reference_symmetry]]
        self.num_total_excitations_r = self.num_excitations_symmetry[self.driver.system.point_group_irrep_to_number[self.state_irrep]]
        # 1% increment
        self.one_increment_t = int(0.01 * self.num_total_excitations_t)
        self.one_increment_r = int(0.01 * self.num_total_excitations_r)
        # Setting up the number of added determinants in each iteration
        self.num_dets_to_add_t = np.zeros(len(self.percentage))
        self.num_dets_to_add_r = np.zeros(len(self.percentage))
        for i in range(len(self.percentage) - 1):
            if self.percentage[i + 1] == 100.0:
                self.num_dets_to_add_t[i] = self.num_total_excitations_t - self.one_increment_t * self.percentage[i]
                self.num_dets_to_add_r[i] = self.num_total_excitations_r - self.one_increment_r * self.percentage[i]
            else:
                self.num_dets_to_add_t[i] = self.one_increment_t * (self.percentage[i + 1] - self.percentage[i])
                self.num_dets_to_add_r[i] = self.one_increment_r * (self.percentage[i + 1] - self.percentage[i])
        self.num_dets_to_add_t[-1] = 1
        self.num_dets_to_add_r[-1] = 1
        # Set the base P space size to 0 (technically it should be all singles and doubles here too)
        self.base_pspace_size_t = 0
        self.base_pspace_size_r = 0

        # Adjust for RHF symmetry
        for i in range(len(self.num_dets_to_add_t)):
            if self.RHF_ground:
                self.num_dets_to_add_t[i] = int(self.num_dets_to_add_t[i] / 2)
            if self.RHF_excited:
                self.num_dets_to_add_r[i] = int(self.num_dets_to_add_r[i] / 2)

    def print_pspace(self):
        """Counts and analyzes the P space in terms of spatial and Sz-spin symmetry."""
        print("   Total number of determinants in P spaces:", self.n_det_t, self.n_det_r)
        print("   Symmetries of states:", self.driver.system.reference_symmetry, self.state_irrep)
        for isym, (counts_t, counts_r) in enumerate(zip(self.excitation_count_by_symmetry_t, self.excitation_count_by_symmetry_r)):
            print("   Symmetry", self.driver.system.point_group_number_to_irrep[isym])
            print("      Number of aaa = ", counts_t['aaa'], counts_r['aaa'])
            print("      Number of aab = ", counts_t['aab'], counts_r['aab'])
            print("      Number of abb = ", counts_t['abb'], counts_r['abb'])
            print("      Number of bbb = ", counts_t['bbb'], counts_r['bbb'])
        print("")

    def run_ccp(self, imacro):
        """Runs iterative CC(P), and if needed, HBar and iterative left-CC calculations."""

        ### Run the ground-state calculation
        self.driver.options["RHF_symmetry"] = self.RHF_ground
        self.driver.run_ccp(method="ccsdt_p", t3_excitations=self.t3_excitations)
        self.driver.run_hbar(method="ccsdt_p", t3_excitations=self.t3_excitations)
        self.driver.run_leftccp(method="left_ccsdt_p", state_index=[0], t3_excitations=self.t3_excitations)

        ### Run the excited-state calculations
        # Compute initial guess once and save it in order to initiate all subsequent EOMCC iterations
        self.driver.options["RHF_symmetry"] = self.RHF_excited
        if imacro == 0:
            self.driver.run_guess(method="cisd", multiplicity=self.multiplicity, roots_per_irrep=self.roots_per_irrep, nact_occupied=self.nacto, nact_unoccupied=self.nactu)
        self.driver.run_eomccp(method="eomccsdt_p", state_index=self.state_index, t3_excitations=self.t3_excitations, r3_excitations=self.r3_excitations)
        self.driver.run_lefteomccp(method="left_ccsdt_p", state_index=self.state_index, t3_excitations=self.t3_excitations, r3_excitations=self.r3_excitations)

        # reset the driver symmetry option
        self.driver.options["RHF_symmetry"] = self.RHF_ground

        # record energies
        self.ccp_energy[imacro] = self.driver.system.reference_energy + self.driver.correlation_energy
        self.eomccp_energy[imacro] = self.driver.system.reference_energy + self.driver.correlation_energy + self.driver.vertical_excitation_energy[self.state_index]

    def run_ccp3(self, imacro):
        """Runs the CC(P;3) correction using either the CR-CC(2,3)- or CCSD(T)-like approach,
           while simultaneously selecting the leading triply excited determinants and returning
           the result in an array. For the last calculation, this should not perform the
           selection steps."""
        from ccpy.moments.ccp3 import calc_ccp3_with_selection, calc_eomccp3_with_selection

        if imacro < self.nmacro - 1:
            self.ccpq_energy[imacro], triples_list_t = calc_ccp3_with_selection(self.driver.T,
                                                                                     self.driver.L[0],
                                                                                     self.t3_excitations,
                                                                                     self.driver.correlation_energy,
                                                                                     self.driver.hamiltonian,
                                                                                     self.bare_hamiltonian,
                                                                                     self.driver.system,
                                                                                     self.num_dets_to_add_t[imacro],
                                                                                     use_RHF=self.RHF_ground,
                                                                                     min_thresh=self.options["minimum_threshold"],
                                                                                     buffer_factor=self.options["buffer_factor"])

            self.eomccpq_energy[imacro], triples_list_r = calc_eomccp3_with_selection(self.driver.T,
                                                                                           self.driver.R[self.state_index],
                                                                                           self.driver.L[self.state_index],
                                                                                           self.t3_excitations,
                                                                                           self.r3_excitations,
                                                                                           self.driver.r0[self.state_index],
                                                                                           self.driver.vertical_excitation_energy[self.state_index],
                                                                                           self.driver.correlation_energy,
                                                                                           self.driver.hamiltonian,
                                                                                           self.bare_hamiltonian,
                                                                                           self.driver.system,
                                                                                           self.num_dets_to_add_r[imacro],
                                                                                           use_RHF=self.RHF_excited,
                                                                                           min_thresh=self.options["minimum_threshold"],
                                                                                           buffer_factor=self.options["buffer_factor"])

            # # Apply P space selection schemes here
            # if self.options["p_space_selection"] == 1: # use P space of the lowest state of a given symmetry
            #     # if excited states shares ground-state symmetry, then copy over T list to R list
            #     if self.driver.system.reference_symmetry == self.state_irrep and self.driver.system.multiplicity == self.multiplicity:
            #         triples_list_r = triples_list_t.copy()
            # elif self.options["p_space_selection"] == 2: # use unified P space for all states
            #     pass

        else:
            triples_list_t = []
            triples_list_r = []
            self.driver.run_ccp3(method="ccp3", state_index=0, two_body_approx=False, t3_excitations=self.t3_excitations)
            self.driver.run_ccp3(method="ccp3", state_index=self.state_index, two_body_approx=False, t3_excitations=self.t3_excitations, r3_excitations=self.r3_excitations)
            self.ccpq_energy[imacro] = self.driver.system.reference_energy + self.driver.correlation_energy + self.driver.deltap3[0]["D"]
            self.eomccpq_energy[imacro] = (self.driver.system.reference_energy
                                           + self.driver.correlation_energy
                                           + self.driver.vertical_excitation_energy[self.state_index]
                                           + self.driver.deltap3[self.state_index]["D"])

        return triples_list_t, triples_list_r

    def run_expand_pspace(self, triples_list_t, triples_list_r):
        """This will expand the P space using the list of triply excited determinants identified
           using the CC(P;Q) moment expansions, above."""
        from ccpy.utilities.selection import add_spinorbital_triples_to_pspace
        self.t3_excitations, self.excitation_count_by_symmetry_t = add_spinorbital_triples_to_pspace(triples_list_t,
                                                                                                      self.t3_excitations,
                                                                                                      self.excitation_count_by_symmetry_t,
                                                                                                      self.driver.system,
                                                                                                      self.RHF_ground)
        self.r3_excitations, self.excitation_count_by_symmetry_r = add_spinorbital_triples_to_pspace(triples_list_r,
                                                                                                      self.r3_excitations,
                                                                                                      self.excitation_count_by_symmetry_r,
                                                                                                      self.driver.system,
                                                                                                      self.RHF_excited)

    def run(self):
        """This is the main driver for the entire adaptive CC(P;Q) calculation. It will call the above
           methods in the correct sequence and handle logic accordingly."""
        # Print the options as a header
        print("   Adaptive CC(P;Q) calculation started on", get_timestamp())
        self.print_options()

        # Step 0a: Perform the preliminary excitation counting
        print("   Preliminary excitation count...", end=" ")
        t1 = time.perf_counter()
        self.excitation_count()
        print("completed in", time.perf_counter() - t1, "seconds")
        # Step 0b: Print the results of excitation count as well as determinant addition plan
        print("   Excitation Count Summary:")
        for i, count in enumerate(self.num_excitations_symmetry):
            symmetry = self.driver.system.point_group_number_to_irrep[i]
            print("      Symmetry", symmetry, " = ", count)
        print("")
        print("   Determinant Addition Plan:")
        print("   Ground State:", self.num_dets_to_add_t)
        print("   Excited State:", self.num_dets_to_add_r)

        # Begin adaptive loop iterations over P-space steps
        for imacro in range(self.nmacro):
            print("")
            print("   Adaptive CC(P;Q) Macroiteration - ", imacro)
            print("   ===========================================")
            # Offset needed to count number of determinants in P space correctly
            offset_t = 0
            offset_r = 0
            if np.array_equal(self.t3_excitations["aaa"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
                offset_t += 1
            if np.array_equal(self.t3_excitations["aab"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
                offset_t += 1
            if np.array_equal(self.t3_excitations["abb"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
                offset_t += 1
            if np.array_equal(self.t3_excitations["bbb"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
                offset_t += 1
            if np.array_equal(self.r3_excitations["aaa"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
                offset_r += 1
            if np.array_equal(self.r3_excitations["aab"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
                offset_r += 1
            if np.array_equal(self.r3_excitations["abb"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
                offset_r += 1
            if np.array_equal(self.r3_excitations["bbb"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
                offset_r += 1

            # Update n_det
            self.n_det_t = self.base_pspace_size_t + (
                            self.t3_excitations["aaa"].shape[0]
                            + self.t3_excitations["aab"].shape[0]
                            + self.t3_excitations["abb"].shape[0]
                            + self.t3_excitations["bbb"].shape[0]
                            - offset_t
            )
            self.n_det_r = self.base_pspace_size_r + (
                            self.r3_excitations["aaa"].shape[0]
                            + self.r3_excitations["aab"].shape[0]
                            + self.r3_excitations["abb"].shape[0]
                            + self.r3_excitations["bbb"].shape[0]
                            - offset_r
            )

            # Step 1: Analyze the P space (optional)
            x1 = time.perf_counter()
            self.print_pspace()
            x2 = time.perf_counter()
            t_pspace_printing = x2 - x1

            # Step 2: Run CC(P) on this P space
            x1 = time.perf_counter()
            self.run_ccp(imacro)
            x2 = time.perf_counter()
            t_ccp = x2 - x1

            # Step 3: Moment correction + adaptive selection
            x1 = time.perf_counter()
            selection_arr_t, selection_arr_r = self.run_ccp3(imacro)
            x2 = time.perf_counter()
            t_selection_and_ccp3 = x2 - x1

            # Check convergence conditions
            if imacro > 0:
                delta_e_ccp = self.ccp_energy[imacro] - self.ccp_energy[imacro - 1]
                delta_e_ccpq = self.ccpq_energy[imacro] - self.ccpq_energy[imacro - 1]
                delta_e_ccp_mu = self.eomccp_energy[imacro] - self.eomccp_energy[imacro - 1]
                delta_e_ccpq_mu = self.eomccpq_energy[imacro] - self.eomccpq_energy[imacro - 1]
                print("   Change in CC(P) energy = ", delta_e_ccp)
                print("   Change in CC(P;Q) energy = ", delta_e_ccpq)
                print("   Change in EOMCC(P) energy = ", delta_e_ccp_mu)
                print("   Change in EOMCC(P;Q) energy = ", delta_e_ccpq_mu, "\n")
                # Energy condition
                if abs(delta_e_ccpq) < self.options["energy_tolerance"] and abs(delta_e_ccpq_mu) < self.options["energy_tolerance"]:
                    print("   Adaptive CC(P;Q) calculation converged to within energy tolerance!")
                    break
                # N_det condition
                if self.n_det_t >= self.options["n_det_max"] or self.n_det_r > self.options["n_det_max"]:
                    print(f"   Adaptive CC(P;Q) calculation reached maximum dimension of P space (n_det_max = {self.options['n_det_max']})")
                    break
            if imacro == self.nmacro - 1:
                print(f"   Adaptive CC(P;Q) reached maximum number of iterations (maximum_iterations = {self.options['maximum_iterations']})")
                break

            # Step 5: Expand the P space
            x1 = time.perf_counter()
            self.run_expand_pspace(selection_arr_t, selection_arr_r)
            x2 = time.perf_counter()
            t_pspace_expand = x2 - x1

            # [TODO]: FIX P SPACE REORDERING BUG WHEN PREVIOUS R VECTOR IS USED
            # Step 6: Reset variables in driver. Right now, R is messed up if it is extended.
            if self.options["reset_amplitudes"]:
                self.driver.R[self.state_index] = None
                self.driver.T = None
                self.driver.L[0] = None

            # ALWAYS reset the excited-state left amplitude. This forces L to begin with converged R as a guess,
            # which generally ensures that R and L converge the same root.
            self.driver.L[self.state_index] = None

            #setattr(self.driver, "hamiltonian", self.bare_hamiltonian)
            # Since the HBar routines are now using H = H0, so there is no deepcopy of the hamiltonian there,
            # we need to use deepcopy every imacro iteration to reset the driver Hamiltonian
            self.driver.hamiltonian = deepcopy(self.bare_hamiltonian)

            # Step 7: Report timings of each step
            print(f"   Timing breakdown for macrostep {imacro}")
            print("   ---------------------------------")
            print(f"   - P space printing took {t_pspace_printing:.2f} seconds")
            print(f"   - CC(P) took {t_ccp:.2f} seconds")
            print(f"   - CC(P;Q) + selection took {t_selection_and_ccp3:.2f} seconds")
            print(f"   - Expanding P space took {t_pspace_expand:.2f} seconds")
            print(f"   - Total time: {t_pspace_printing + t_pspace_expand + t_ccp + t_selection_and_ccp3 + t_pspace_expand:.2f} seconds")

        # Final printout of the results including energy extrapolations
        print("   Adaptive CC(P;Q) calculation ended on", get_timestamp(), "\n")
        print("   Summary of results:")
        print(f"    Iteration       E0(P)             E0(P;Q)             E{self.state_index}(P)             E{self.state_index}(P;Q)         VEE(P)      VEE(P;Q)")
        print("   --------------------------------------------------------------------------------------------------------------")
        for i in range(imacro + 1):
            print("   %8d    %.10f     %.10f     %.10f     %.10f     %.4f eV    %.4f eV" %
                (i,
                 self.ccp_energy[i],
                 self.ccpq_energy[i],
                 self.eomccp_energy[i],
                 self.eomccpq_energy[i],
                 (self.eomccp_energy[i] - self.ccp_energy[i]) * hartreetoeV,
                 (self.eomccpq_energy[i] - self.ccpq_energy[i]) * hartreetoeV,
                 )
            )

class AdaptEOMDriver:

    def __init__(self, driver, state_index, roots_per_irrep, multiplicity, nacto=0, nactu=0, percentage=None, unify=True):
        self.driver = driver
        self.state_index = state_index # list of guess roots to solve for
        self.nstates = len(self.state_index) + 1
        # option to unify the excited-state P spaces or not
        self.unify = unify
        # inputs used to form the one-time EOMCCSd initial guess
        self.roots_per_irrep = roots_per_irrep
        self.multiplicity = multiplicity
        self.nacto = nacto
        self.nactu = nactu
        # Get the state irrep from the roots_per_irrep input
        irrep_list = []
        for irrep, nguess in self.roots_per_irrep.items():
            for k in range(nguess):
                irrep_list.append(irrep)
        self.state_irrep = irrep_list[self.state_index[0] - 1]
        #
        self.percentage = percentage
        #
        self.options = {"reset_amplitudes": False,
                        "energy_tolerance": 1.0e-04,
                        "maximum_iterations": 10,
                        "n_det_max": 100000000,
                        "selection_factor": 1.0,
                        "base_growth": "ccsd",
                        "buffer_factor": 2,
                        "minimum_threshold": 0.0}
        #
        self.nmacro = len(percentage)
        self.energy_tolerance = self.options["energy_tolerance"]
        # energy containers
        self.ccp_energy = np.zeros((self.nstates, self.nmacro))
        self.ccpq_energy = np.zeros((self.nstates, self.nmacro))
        # t3 excitations and r3_excitations
        self.t3_excitations = {"aaa": np.ones((1, 6), order="F"),
                               "aab": np.ones((1, 6), order="F"),
                               "abb": np.ones((1, 6), order="F"),
                               "bbb": np.ones((1, 6), order="F")}
        self.r3_excitations = {"aaa": np.ones((1, 6), order="F"),
                               "aab": np.ones((1, 6), order="F"),
                               "abb": np.ones((1, 6), order="F"),
                               "bbb": np.ones((1, 6), order="F")}
        self.excitation_count_by_symmetry_t = [{'aaa': 0, 'aab': 0, 'abb': 0, 'bbb': 0} for _ in range(len(self.driver.system.point_group_irrep_to_number))]
        self.excitation_count_by_symmetry_r = [{'aaa': 0, 'aab': 0, 'abb': 0, 'bbb': 0} for _ in range(len(self.driver.system.point_group_irrep_to_number))]
        self.n_det_t = 0
        self.n_det_r = 0
        # RHF flags for ground and excited states
        self.RHF_ground = self.driver.options["RHF_symmetry"]
        self.RHF_excited = True if self.multiplicity == 1 else False
        # Save the bare Hamiltonian for later iterations if using CR-CC(2,3)
        self.bare_hamiltonian = deepcopy(self.driver.hamiltonian)

    def print_options(self):
        print("   ------------------------------------------")
        for option_key, option_value in self.options.items():
            print("  ", option_key, "=", option_value)
        print("   ------------------------------------------\n")

    def excitation_count(self):
        """Performs an initial symmetry-adapted count of the relevant excitation
           space to determine the growth increment for each iteration of the
           calculation."""
        from ccpy.utilities.symmetry import count_singles, count_doubles, count_triples
        # Count the triples using symmetry
        self.num_excitations_symmetry, _ = count_triples(self.driver.system)
        self.num_total_excitations_t = self.num_excitations_symmetry[self.driver.system.point_group_irrep_to_number[self.driver.system.reference_symmetry]]
        self.num_total_excitations_r = self.num_excitations_symmetry[self.driver.system.point_group_irrep_to_number[self.state_irrep]]
        # 1% increment
        self.one_increment_t = int(0.01 * self.num_total_excitations_t)
        self.one_increment_r = int(0.01 * self.num_total_excitations_r)
        # Setting up the number of added determinants in each iteration
        self.num_dets_to_add_t = np.zeros(len(self.percentage))
        self.num_dets_to_add_r = np.zeros(len(self.percentage))
        for i in range(len(self.percentage) - 1):
            if self.percentage[i + 1] == 100.0:
                self.num_dets_to_add_t[i] = self.num_total_excitations_t - self.one_increment_t * self.percentage[i]
                self.num_dets_to_add_r[i] = self.num_total_excitations_r - self.one_increment_r * self.percentage[i]
            else:
                self.num_dets_to_add_t[i] = self.one_increment_t * (self.percentage[i + 1] - self.percentage[i])
                self.num_dets_to_add_r[i] = self.one_increment_r * (self.percentage[i + 1] - self.percentage[i])
        self.num_dets_to_add_t[-1] = 1
        self.num_dets_to_add_r[-1] = 1
        # Set the base P space size to 0 (technically it should be all singles and doubles here too)
        self.base_pspace_size_t = 0
        self.base_pspace_size_r = 0

        # Adjust for RHF symmetry
        for i in range(len(self.num_dets_to_add_t)):
            if self.RHF_ground:
                self.num_dets_to_add_t[i] = int(self.num_dets_to_add_t[i] / 2)
            if self.RHF_excited:
                self.num_dets_to_add_r[i] = int(self.num_dets_to_add_r[i] / 2)

    def print_pspace(self):
        """Counts and analyzes the P space in terms of spatial and Sz-spin symmetry."""
        print("   Total number of determinants in P spaces:", self.n_det_t, self.n_det_r)
        print("   Symmetries of states:", self.driver.system.reference_symmetry, self.state_irrep)
        for isym, (counts_t, counts_r) in enumerate(zip(self.excitation_count_by_symmetry_t, self.excitation_count_by_symmetry_r)):
            print("   Symmetry", self.driver.system.point_group_number_to_irrep[isym])
            print("      Number of aaa = ", counts_t['aaa'], counts_r['aaa'])
            print("      Number of aab = ", counts_t['aab'], counts_r['aab'])
            print("      Number of abb = ", counts_t['abb'], counts_r['abb'])
            print("      Number of bbb = ", counts_t['bbb'], counts_r['bbb'])
        print("")

    def run_ccp(self, imacro):
        """Runs iterative CC(P), and if needed, HBar and iterative left-CC calculations."""

        ### Run the ground-state calculation
        self.driver.options["RHF_symmetry"] = self.RHF_ground
        self.driver.run_ccp(method="ccsdt_p", t3_excitations=self.t3_excitations)
        self.driver.run_hbar(method="ccsdt_p", t3_excitations=self.t3_excitations)
        self.driver.run_leftccp(method="left_ccsdt_p", state_index=[0], t3_excitations=self.t3_excitations)
        # record energies
        self.ccp_energy[0, imacro] = self.driver.system.reference_energy + self.driver.correlation_energy

        ### Run the excited-state calculations
        # Compute initial guess once and save it in order to initiate all subsequent EOMCC iterations
        self.driver.options["RHF_symmetry"] = self.RHF_excited
        if imacro == 0:
            self.driver.run_guess(method="cisd", multiplicity=self.multiplicity, roots_per_irrep=self.roots_per_irrep, nact_occupied=self.nacto, nact_unoccupied=self.nactu)

        if self.driver.options["davidson_solver"] == "multiroot":
            self.driver.run_eomccp(method="eomccsdt_p", state_index=self.state_index, t3_excitations=self.t3_excitations, r3_excitations=self.r3_excitations)
        else:
            for i, istate in enumerate(self.state_index):
                self.driver.run_eomccp(method="eomccsdt_p", state_index=istate, t3_excitations=self.t3_excitations, r3_excitations=self.r3_excitations)

        for i, istate in enumerate(self.state_index):
            self.driver.run_lefteomccp(method="left_ccsdt_p", state_index=istate, t3_excitations=self.t3_excitations, r3_excitations=self.r3_excitations)
            # record energies
            self.ccp_energy[i + 1, imacro] = self.driver.system.reference_energy + self.driver.correlation_energy + self.driver.vertical_excitation_energy[istate]

        # reset the driver symmetry option
        self.driver.options["RHF_symmetry"] = self.RHF_ground

    def run_ccp3(self, imacro):
        """Runs the CC(P;3) correction using either the CR-CC(2,3)- or CCSD(T)-like approach,
           while simultaneously selecting the leading triply excited determinants and returning
           the result in an array. For the last calculation, this should not perform the
           selection steps."""
        from ccpy.moments.ccp3 import calc_ccp3_with_selection, calc_eomccp3_with_selection

        if imacro < self.nmacro - 1:
            # Perform ground-state correction + selection
            self.ccpq_energy[0, imacro], triples_list_t = calc_ccp3_with_selection(self.driver.T,
                                                                                        self.driver.L[0],
                                                                                        self.t3_excitations,
                                                                                        self.driver.correlation_energy,
                                                                                        self.driver.hamiltonian,
                                                                                        self.bare_hamiltonian,
                                                                                        self.driver.system,
                                                                                        self.num_dets_to_add_t[imacro],
                                                                                        use_RHF=self.RHF_ground,
                                                                                        min_thresh=self.options["minimum_threshold"],
                                                                                        buffer_factor=self.options["buffer_factor"])

            for i, istate in enumerate(self.state_index):
                # Perform excited-state corrections
                if i == 0:
                    # Obtain list only for the lowest excited state
                    self.ccpq_energy[i + 1, imacro], triples_list_r = calc_eomccp3_with_selection(self.driver.T,
                                                                                                       self.driver.R[istate],
                                                                                                       self.driver.L[istate],
                                                                                                       self.t3_excitations,
                                                                                                       self.r3_excitations,
                                                                                                       self.driver.r0[istate],
                                                                                                       self.driver.vertical_excitation_energy[istate],
                                                                                                       self.driver.correlation_energy,
                                                                                                       self.driver.hamiltonian,
                                                                                                       self.bare_hamiltonian,
                                                                                                       self.driver.system,
                                                                                                       self.num_dets_to_add_r[imacro],
                                                                                                       use_RHF=self.RHF_excited,
                                                                                                       min_thresh=self.options["minimum_threshold"],
                                                                                                       buffer_factor=self.options["buffer_factor"])
                else:
                    # For all other excited states, only perform correction, no selection
                    self.driver.run_ccp3(method="ccp3",
                                         state_index=istate,
                                         two_body_approx=False,
                                         t3_excitations=self.t3_excitations,
                                         r3_excitations=self.r3_excitations)
                    self.ccpq_energy[i + 1, imacro] = (self.driver.system.reference_energy
                                                       +self.driver.correlation_energy
                                                       +self.driver.vertical_excitation_energy[istate]
                                                       +self.driver.deltap3[istate]["D"])
        else:
            triples_list_t = []
            triples_list_r = []
            self.driver.run_ccp3(method="ccp3", state_index=0, two_body_approx=False, t3_excitations=self.t3_excitations)
            self.ccpq_energy[0, imacro] = self.driver.system.reference_energy + self.driver.correlation_energy + self.driver.deltap3[0]["D"]
            for i, istate in enumerate(self.state_index):
                self.driver.run_ccp3(method="ccp3", state_index=istate, two_body_approx=False, t3_excitations=self.t3_excitations, r3_excitations=self.r3_excitations)
                self.ccpq_energy[i + 1, imacro] = (self.driver.system.reference_energy
                                                   +self.driver.correlation_energy
                                                   +self.driver.vertical_excitation_energy[istate]
                                                   +self.driver.deltap3[istate]["D"])

        # if excited states shares ground-state symmetry, then copy over T list to R list
        if self.driver.system.reference_symmetry == self.state_irrep and self.driver.system.multiplicity == self.multiplicity:
            triples_list_r = triples_list_t.copy()

        return triples_list_t, triples_list_r

    def run_ccp3_with_overlap(self, imacro):
        """Runs the CC(P;3) correction using either the CR-CC(2,3)- or CCSD(T)-like approach,
           while simultaneously selecting the leading triply excited determinants and returning
           the result in an array. For the last calculation, this should not perform the
           selection steps."""
        from ccpy.moments.ccp3 import calc_ccp3_with_selection, calc_eomccp3_with_selection
        from ccpy.utilities.pspace import overlap_p_spaces

        if imacro < self.nmacro - 1:
            # Perform ground-state correction + selection
            self.ccpq_energy[0, imacro], triples_list_t = calc_ccp3_with_selection(self.driver.T,
                                                                                        self.driver.L[0],
                                                                                        self.t3_excitations,
                                                                                        self.driver.correlation_energy,
                                                                                        self.driver.hamiltonian,
                                                                                        self.bare_hamiltonian,
                                                                                        self.driver.system,
                                                                                        self.num_dets_to_add_t[imacro],
                                                                                        use_RHF=self.RHF_ground,
                                                                                        min_thresh=self.options["minimum_threshold"],
                                                                                        buffer_factor=self.options["buffer_factor"])
            # Create global overlapped excited-state P space excitation list
            triples_list_r = []
            # Perform excited-state corrections and selection
            for i, istate in enumerate(self.state_index):
                self.ccpq_energy[i + 1, imacro], triples_list_r_istate = calc_eomccp3_with_selection(self.driver.T,
                                                                                                     self.driver.R[istate],
                                                                                                     self.driver.L[istate],
                                                                                                     self.t3_excitations,
                                                                                                     self.r3_excitations,
                                                                                                     self.driver.r0[istate],
                                                                                                     self.driver.vertical_excitation_energy[istate],
                                                                                                     self.driver.correlation_energy,
                                                                                                     self.driver.hamiltonian,
                                                                                                     self.bare_hamiltonian,
                                                                                                     self.driver.system,
                                                                                                     self.num_dets_to_add_r[imacro],
                                                                                                     use_RHF=self.RHF_excited,
                                                                                                     min_thresh=self.options["minimum_threshold"],
                                                                                                     buffer_factor=self.options["buffer_factor"])
                triples_list_r = overlap_p_spaces(triples_list_r, triples_list_r_istate)

        else:
            triples_list_t = []
            triples_list_r = []
            self.driver.run_ccp3(method="ccp3", state_index=0, two_body_approx=False, t3_excitations=self.t3_excitations)
            self.ccpq_energy[0, imacro] = self.driver.system.reference_energy + self.driver.correlation_energy + self.driver.deltap3[0]["D"]
            for i, istate in enumerate(self.state_index):
                self.driver.run_ccp3(method="ccp3", state_index=istate, two_body_approx=False, t3_excitations=self.t3_excitations, r3_excitations=self.r3_excitations)
                self.ccpq_energy[i + 1, imacro] = (self.driver.system.reference_energy
                                                   +self.driver.correlation_energy
                                                   +self.driver.vertical_excitation_energy[istate]
                                                   +self.driver.deltap3[istate]["D"])

        # if excited states shares ground-state symmetry, then overlap R with T
        if self.driver.system.reference_symmetry == self.state_irrep and self.driver.system.multiplicity == self.multiplicity:
            triples_list_t = overlap_p_spaces(triples_list_t, triples_list_r)
            triples_list_r = triples_list_t.copy()

        return triples_list_t, triples_list_r

    def run_expand_pspace(self, triples_list_t, triples_list_r):
        """This will expand the P space using the list of triply excited determinants identified
           using the CC(P;Q) moment expansions, above."""
        from ccpy.utilities.selection import add_spinorbital_triples_to_pspace
        self.t3_excitations, self.excitation_count_by_symmetry_t = add_spinorbital_triples_to_pspace(triples_list_t,
                                                                                                     self.t3_excitations,
                                                                                                     self.excitation_count_by_symmetry_t,
                                                                                                     self.driver.system,
                                                                                                     self.RHF_ground)

        self.r3_excitations, self.excitation_count_by_symmetry_r = add_spinorbital_triples_to_pspace(triples_list_r,
                                                                                                     self.r3_excitations,
                                                                                                     self.excitation_count_by_symmetry_r,
                                                                                                     self.driver.system,
                                                                                                     self.RHF_excited)

    def run(self):
        """This is the main driver for the entire adaptive CC(P;Q) calculation. It will call the above
           methods in the correct sequence and handle logic accordingly."""
        # Print the options as a header
        print("   Adaptive CC(P;Q) calculation started on", get_timestamp())
        self.print_options()

        # Step 0a: Perform the preliminary excitation counting
        print("   Preliminary excitation count...", end=" ")
        t1 = time.perf_counter()
        self.excitation_count()
        print("completed in", time.perf_counter() - t1, "seconds")
        # Step 0b: Print the results of excitation count as well as determinant addition plan
        print("   Excitation Count Summary:")
        for i, count in enumerate(self.num_excitations_symmetry):
            symmetry = self.driver.system.point_group_number_to_irrep[i]
            print("      Symmetry", symmetry, " = ", count)
        print("")
        print("   Determinant Addition Plan:")
        print("   Ground State:", self.num_dets_to_add_t)
        print("   Excited State:", self.num_dets_to_add_r)

        # Begin adaptive loop iterations over P-space steps
        for imacro in range(self.nmacro):
            print("")
            print("   Adaptive CC(P;Q) Macroiteration - ", imacro)
            print("   ===========================================")
            # Offset needed to count number of determinants in P space correctly
            offset_t = 0
            offset_r = 0
            if np.array_equal(self.t3_excitations["aaa"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
                offset_t += 1
            if np.array_equal(self.t3_excitations["aab"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
                offset_t += 1
            if np.array_equal(self.t3_excitations["abb"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
                offset_t += 1
            if np.array_equal(self.t3_excitations["bbb"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
                offset_t += 1
            if np.array_equal(self.r3_excitations["aaa"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
                offset_r += 1
            if np.array_equal(self.r3_excitations["aab"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
                offset_r += 1
            if np.array_equal(self.r3_excitations["abb"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
                offset_r += 1
            if np.array_equal(self.r3_excitations["bbb"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
                offset_r += 1

            # Update n_det
            self.n_det_t = self.base_pspace_size_t + (
                            self.t3_excitations["aaa"].shape[0]
                            + self.t3_excitations["aab"].shape[0]
                            + self.t3_excitations["abb"].shape[0]
                            + self.t3_excitations["bbb"].shape[0]
                            - offset_t
            )
            self.n_det_r = self.base_pspace_size_r + (
                            self.r3_excitations["aaa"].shape[0]
                            + self.r3_excitations["aab"].shape[0]
                            + self.r3_excitations["abb"].shape[0]
                            + self.r3_excitations["bbb"].shape[0]
                            - offset_r
            )

            # Step 1: Analyze the P space (optional)
            x1 = time.perf_counter()
            self.print_pspace()
            x2 = time.perf_counter()
            t_pspace_printing = x2 - x1

            # Step 2: Run CC(P) on this P space
            x1 = time.perf_counter()
            self.run_ccp(imacro)
            x2 = time.perf_counter()
            t_ccp = x2 - x1

            # Step 3: Moment correction + adaptive selection
            x1 = time.perf_counter()
            if self.unify:
                selection_arr_t, selection_arr_r = self.run_ccp3_with_overlap(imacro)
            else:
                selection_arr_t, selection_arr_r = self.run_ccp3(imacro)
            x2 = time.perf_counter()
            t_selection_and_ccp3 = x2 - x1

            # Check convergence conditions
            if imacro > 0:
                delta_e_ccp = self.ccp_energy[:, imacro] - self.ccp_energy[:, imacro - 1]
                delta_e_ccpq = self.ccpq_energy[:, imacro] - self.ccpq_energy[:, imacro - 1]
                print("   Change in CC(P) energy = ", delta_e_ccp)
                print("   Change in CC(P;Q) energy = ", delta_e_ccpq, "\n")
                # Energy condition
                if all(abs(delta_e_ccpq) < self.options["energy_tolerance"]):
                    print("   Adaptive CC(P;Q) calculation converged to within energy tolerance!")
                    break
                # N_det condition
                if self.n_det_t >= self.options["n_det_max"] or self.n_det_r > self.options["n_det_max"]:
                    print(f"   Adaptive CC(P;Q) calculation reached maximum dimension of P space (n_det_max = {self.options['n_det_max']})")
                    break
            if imacro == self.nmacro - 1:
                print(f"   Adaptive CC(P;Q) reached maximum number of iterations (maximum_iterations = {self.options['maximum_iterations']})")
                break

            # Step 5: Expand the P space
            x1 = time.perf_counter()
            self.run_expand_pspace(selection_arr_t, selection_arr_r)
            x2 = time.perf_counter()
            t_pspace_expand = x2 - x1

            # [TODO]: FIX P SPACE REORDERING BUG WHEN PREVIOUS R VECTOR IS USED
            # Step 6: Reset variables in driver. Right now, R is messed up if it is extended.
            if self.options["reset_amplitudes"]:
                for istate in self.state_index:
                    self.driver.R[istate] = None
                self.driver.T = None
                self.driver.L[0] = None
            # ALWAYS reset the excited-state left amplitude. This forces L to begin with converged R as a guess,
            # which generally ensures that R and L converge the same root.
            for istate in self.state_index:
                self.driver.L[istate] = None

            #setattr(self.driver, "hamiltonian", self.bare_hamiltonian)
            # Since the HBar routines are now using H = H0, so there is no deepcopy of the hamiltonian there,
            # we need to use deepcopy every imacro iteration to reset the driver Hamiltonian
            self.driver.hamiltonian = deepcopy(self.bare_hamiltonian)

            # Step 7: Report timings of each step
            print(f"   Timing breakdown for macrostep {imacro}")
            print("   ---------------------------------")
            print(f"   - P space printing took {t_pspace_printing:.2f} seconds")
            print(f"   - CC(P) took {t_ccp:.2f} seconds")
            print(f"   - CC(P;Q) + selection took {t_selection_and_ccp3:.2f} seconds")
            print(f"   - Expanding P space took {t_pspace_expand:.2f} seconds")
            print(f"   - Total time: {t_pspace_printing + t_pspace_expand + t_ccp + t_selection_and_ccp3 + t_pspace_expand:.2f} seconds")

        # Final printout of the results including energy extrapolations
        print("   Adaptive CC(P;Q) calculation ended on", get_timestamp(), "\n")
        print("   Summary of results:")
        for j in range(self.nstates):
            print("")
            print(f"    Iteration       E{j}(P)             E{j}(P;Q)         VEE(P)      VEE(P;Q)")
            print("   --------------------------------------------------------------------------------------------------------------")
            for i in range(imacro + 1):
                print("   %8d    %.10f     %.10f     %.4f eV    %.4f eV" %
                      (i,
                       self.ccp_energy[j, i],
                       self.ccpq_energy[j, i],
                       (self.ccp_energy[j, i] - self.ccp_energy[0, i]) * hartreetoeV,
                       (self.ccpq_energy[j, i] - self.ccpq_energy[0, i]) * hartreetoeV,
                       )
                )

class AdaptEAEOMDriverSS:

    def __init__(self, driver, state_index, roots_per_irrep, multiplicity, nacto=0, nactu=0, percentage=None):
        self.driver = driver
        # inputs used to form the one-time EA-EOMCCSd initial guess
        self.state_index = state_index
        self.roots_per_irrep = roots_per_irrep
        self.multiplicity = multiplicity
        self.nacto = nacto
        self.nactu = nactu
        # Get the state irrep from the roots_per_irrep input
        irrep_list = []
        for irrep, nguess in self.roots_per_irrep.items():
            for k in range(nguess):
                irrep_list.append(irrep)
        self.state_irrep = irrep_list[self.state_index - 1]
        #
        self.percentage = percentage
        #
        self.options = {"reset_amplitudes": False,
                        "energy_tolerance": 1.0e-04,
                        "maximum_iterations": 10,
                        "n_det_max": 100000000,
                        "selection_factor": 1.0,
                        "base_growth": "ccsd",
                        "buffer_factor": 2,
                        "minimum_threshold": 0.0,
                        "p_space_selection": 3}
        # options["p_space_selection"] = 1, 2, or 3 controls how the P spaces for the states considered are constructed
        # 1 - Uses the P space for the lowest space of a given symmetry
        # 2 - Uses a unified P space for all states being considered
        # 3 (default) - Uses an individual P space adapted to each state; this option can formally break size-intensivity!
        #
        self.nmacro = len(percentage)
        self.energy_tolerance = self.options["energy_tolerance"]
        # energy containers
        self.ccp_energy = np.zeros(self.nmacro)
        self.ccpq_energy = np.zeros(self.nmacro)
        self.eomccp_energy = np.zeros(self.nmacro)
        self.eomccpq_energy = np.zeros(self.nmacro)
        # r3_excitations
        self.r3_excitations = {"aaa": np.ones((1, 6), order="F"),
                               "aab": np.ones((1, 6), order="F"),
                               "abb": np.ones((1, 6), order="F")}
        self.excitation_count_by_symmetry_r = [{'aaa': 0, 'aab': 0, 'abb': 0} for _ in range(len(self.driver.system.point_group_irrep_to_number))]
        self.n_det_r = 0
        # RHF flags for ground and excited states
        self.RHF_ground = self.driver.options["RHF_symmetry"]
        self.RHF_excited = True if self.multiplicity == 1 else False
        # Save the bare Hamiltonian for later iterations if using CR-CC(2,3)
        self.bare_hamiltonian = deepcopy(self.driver.hamiltonian)

    def print_options(self):
        print("   ------------------------------------------")
        for option_key, option_value in self.options.items():
            print("  ", option_key, "=", option_value)
        print("   ------------------------------------------\n")

    def excitation_count(self):
        """Performs an initial symmetry-adapted count of the relevant excitation
           space to determine the growth increment for each iteration of the
           calculation."""
        from ccpy.utilities.symmetry import count_1p, count_2p1h, count_3p2h
        # Count the triples using symmetry
        self.num_excitations_symmetry, _ = count_3p2h(self.driver.system)
        self.num_total_excitations_r = self.num_excitations_symmetry[self.driver.system.point_group_irrep_to_number[self.state_irrep]]
        # 1% increment
        self.one_increment_r = int(0.01 * self.num_total_excitations_r)
        # Setting up the number of added determinants in each iteration
        self.num_dets_to_add_r = np.zeros(len(self.percentage))
        for i in range(len(self.percentage) - 1):
            if self.percentage[i + 1] == 100.0:
                self.num_dets_to_add_r[i] = self.num_total_excitations_r - self.one_increment_r * self.percentage[i]
            else:
                self.num_dets_to_add_r[i] = self.one_increment_r * (self.percentage[i + 1] - self.percentage[i])
        self.num_dets_to_add_r[-1] = 1
        # Set the base P space size to 0 (technically it should be all singles and doubles here too)
        self.base_pspace_size_r = 0

        # Adjust for RHF symmetry
        for i in range(len(self.num_dets_to_add_r)):
            if self.RHF_excited:
                self.num_dets_to_add_r[i] = int(self.num_dets_to_add_r[i] / 2)

    def print_pspace(self):
        """Counts and analyzes the P space in terms of spatial and Sz-spin symmetry."""
        print("   Total number of 3p2h determinants in P space:", self.n_det_r)
        print("   Symmetry of target state:", self.state_irrep)
        for isym, counts_r in enumerate(self.excitation_count_by_symmetry_r):
            print("   Symmetry", self.driver.system.point_group_number_to_irrep[isym])
            print("      Number of aaa = ", counts_r['aaa'])
            print("      Number of aab = ", counts_r['aab'])
            print("      Number of abb = ", counts_r['abb'])
            print("      Number of bbb = ", counts_r['bbb'])
        print("")

    def run_ccsd(self, imacro):
        """Runs iterative CCSD, and if needed, HBar and iterative left-CC calculations."""

        ### Run the ground-state calculation
        if imacro == 0:
            self.driver.options["RHF_symmetry"] = self.RHF_ground
            self.driver.run_cc(method="ccsd")
            self.driver.run_hbar(method="ccsd")

        ### Run the excited-state calculations
        # Compute initial guess once and save it in order to initiate all subsequent EOMCC iterations
        self.driver.options["RHF_symmetry"] = self.RHF_excited
        if imacro == 0:
            self.driver.run_guess(method="eacisd", multiplicity=self.multiplicity, roots_per_irrep=self.roots_per_irrep, nact_occupied=self.nacto, nact_unoccupied=self.nactu)
        self.driver.run_eaeomccp(method="eaeom3_p", state_index=self.state_index, r3_excitations=self.r3_excitations)
        self.driver.run_lefteomccp(method="left_ccsdt_p", state_index=self.state_index, t3_excitations=self.t3_excitations, r3_excitations=self.r3_excitations)

        # reset the driver symmetry option
        self.driver.options["RHF_symmetry"] = self.RHF_ground

        # record energies
        self.ccp_energy[imacro] = self.driver.system.reference_energy + self.driver.correlation_energy
        self.eomccp_energy[imacro] = self.driver.system.reference_energy + self.driver.correlation_energy + self.driver.vertical_excitation_energy[self.state_index]

    def run_ccp3(self, imacro):
        """Runs the CC(P;3) correction using either the CR-CC(2,3)- or CCSD(T)-like approach,
           while simultaneously selecting the leading triply excited determinants and returning
           the result in an array. For the last calculation, this should not perform the
           selection steps."""
        from ccpy.moments.ccp3 import calc_eomccp3_full_with_selection

        if imacro < self.nmacro - 1:
            self.eomccpq_energy[imacro], triples_list_r = calc_eomccp3_full_with_selection(self.driver.T,
                                                                                           self.driver.R[self.state_index],
                                                                                           self.driver.L[self.state_index],
                                                                                           self.r3_excitations,
                                                                                           self.driver.r0[self.state_index],
                                                                                           self.driver.vertical_excitation_energy[self.state_index],
                                                                                           self.driver.correlation_energy,
                                                                                           self.driver.hamiltonian,
                                                                                           self.bare_hamiltonian,
                                                                                           self.driver.system,
                                                                                           self.num_dets_to_add_r[imacro],
                                                                                           use_RHF=self.RHF_excited,
                                                                                           min_thresh=self.options["minimum_threshold"],
                                                                                           buffer_factor=self.options["buffer_factor"])

            # # Apply P space selection schemes here
            # if self.options["p_space_selection"] == 1: # use P space of the lowest state of a given symmetry
            #     # if excited states shares ground-state symmetry, then copy over T list to R list
            #     if self.driver.system.reference_symmetry == self.state_irrep and self.driver.system.multiplicity == self.multiplicity:
            #         triples_list_r = triples_list_t.copy()
            # elif self.options["p_space_selection"] == 2: # use unified P space for all states
            #     pass

        else:
            triples_list_r = []
            self.driver.run_ccp3(method="crcc23")
            self.driver.run_eaccp3(method="eaccp3", state_index=self.state_index, two_body_approx=False, r3_excitations=self.r3_excitations)
            self.ccpq_energy[imacro] = self.driver.system.reference_energy + self.driver.correlation_energy + self.driver.deltap3[0]["D"]
            self.eomccpq_energy[imacro] = (self.driver.system.reference_energy
                                           + self.driver.correlation_energy
                                           + self.driver.vertical_excitation_energy[self.state_index]
                                           + self.driver.deltap3[self.state_index]["D"])

        return triples_list_r

    def run_expand_pspace(self, triples_list_t, triples_list_r):
        """This will expand the P space using the list of triply excited determinants identified
           using the CC(P;Q) moment expansions, above."""
        from ccpy.utilities.selection import add_spinorbital_triples_to_pspace
        self.r3_excitations, self.excitation_count_by_symmetry_r = add_spinorbital_triples_to_pspace(triples_list_r,
                                                                                                      self.r3_excitations,
                                                                                                      self.excitation_count_by_symmetry_r,
                                                                                                      self.driver.system,
                                                                                                      self.RHF_excited)

    def run(self):
        """This is the main driver for the entire adaptive CC(P;Q) calculation. It will call the above
           methods in the correct sequence and handle logic accordingly."""
        # Print the options as a header
        print("   Adaptive CC(P;Q) calculation started on", get_timestamp())
        self.print_options()

        # Step 0a: Perform the preliminary excitation counting
        print("   Preliminary excitation count...", end=" ")
        t1 = time.perf_counter()
        self.excitation_count()
        print("completed in", time.perf_counter() - t1, "seconds")
        # Step 0b: Print the results of excitation count as well as determinant addition plan
        print("   Excitation Count Summary:")
        for i, count in enumerate(self.num_excitations_symmetry):
            symmetry = self.driver.system.point_group_number_to_irrep[i]
            print("      Symmetry", symmetry, " = ", count)
        print("")
        print("   Determinant Addition Plan:")
        print("   Ground State:", self.num_dets_to_add_t)
        print("   Excited State:", self.num_dets_to_add_r)

        # Begin adaptive loop iterations over P-space steps
        for imacro in range(self.nmacro):
            print("")
            print("   Adaptive CC(P;Q) Macroiteration - ", imacro)
            print("   ===========================================")
            # Offset needed to count number of determinants in P space correctly
            offset_r = 0
            if np.array_equal(self.r3_excitations["aaa"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
                offset_r += 1
            if np.array_equal(self.r3_excitations["aab"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
                offset_r += 1
            if np.array_equal(self.r3_excitations["abb"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
                offset_r += 1
            if np.array_equal(self.r3_excitations["bbb"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
                offset_r += 1

            # Update n_det
            self.n_det_r = self.base_pspace_size_r + (
                            self.r3_excitations["aaa"].shape[0]
                            + self.r3_excitations["aab"].shape[0]
                            + self.r3_excitations["abb"].shape[0]
                            + self.r3_excitations["bbb"].shape[0]
                            - offset_r
            )

            # Step 1: Analyze the P space (optional)
            x1 = time.perf_counter()
            self.print_pspace()
            x2 = time.perf_counter()
            t_pspace_printing = x2 - x1

            # Step 2: Run CC(P) on this P space
            x1 = time.perf_counter()
            self.run_ccp(imacro)
            x2 = time.perf_counter()
            t_ccp = x2 - x1

            # Step 3: Moment correction + adaptive selection
            x1 = time.perf_counter()
            selection_arr_t, selection_arr_r = self.run_ccp3(imacro)
            x2 = time.perf_counter()
            t_selection_and_ccp3 = x2 - x1

            # Check convergence conditions
            if imacro > 0:
                delta_e_ccp = self.ccp_energy[imacro] - self.ccp_energy[imacro - 1]
                delta_e_ccpq = self.ccpq_energy[imacro] - self.ccpq_energy[imacro - 1]
                delta_e_ccp_mu = self.eomccp_energy[imacro] - self.eomccp_energy[imacro - 1]
                delta_e_ccpq_mu = self.eomccpq_energy[imacro] - self.eomccpq_energy[imacro - 1]
                print("   Change in CC(P) energy = ", delta_e_ccp)
                print("   Change in CC(P;Q) energy = ", delta_e_ccpq)
                print("   Change in EOMCC(P) energy = ", delta_e_ccp_mu)
                print("   Change in EOMCC(P;Q) energy = ", delta_e_ccpq_mu, "\n")
                # Energy condition
                if abs(delta_e_ccpq) < self.options["energy_tolerance"] and abs(delta_e_ccpq_mu) < self.options["energy_tolerance"]:
                    print("   Adaptive CC(P;Q) calculation converged to within energy tolerance!")
                    break
                # N_det condition
                if self.n_det_t >= self.options["n_det_max"] or self.n_det_r > self.options["n_det_max"]:
                    print(f"   Adaptive CC(P;Q) calculation reached maximum dimension of P space (n_det_max = {self.options['n_det_max']})")
                    break
            if imacro == self.nmacro - 1:
                print(f"   Adaptive CC(P;Q) reached maximum number of iterations (maximum_iterations = {self.options['maximum_iterations']})")
                break

            # Step 5: Expand the P space
            x1 = time.perf_counter()
            self.run_expand_pspace(selection_arr_t, selection_arr_r)
            x2 = time.perf_counter()
            t_pspace_expand = x2 - x1

            # [TODO]: FIX P SPACE REORDERING BUG WHEN PREVIOUS R VECTOR IS USED
            # Step 6: Reset variables in driver. Right now, R is messed up if it is extended.
            if self.options["reset_amplitudes"]:
                self.driver.R[self.state_index] = None
                self.driver.T = None
                self.driver.L[0] = None
            # ALWAYS reset the excited-state left amplitude. This forces L to begin with converged R as a guess,
            # which generally ensures that R and L converge the same root.
            self.driver.L[self.state_index] = None

            #setattr(self.driver, "hamiltonian", self.bare_hamiltonian)
            # Since the HBar routines are now using H = H0, so there is no deepcopy of the hamiltonian there,
            # we need to use deepcopy every imacro iteration to reset the driver Hamiltonian
            self.driver.hamiltonian = deepcopy(self.bare_hamiltonian)

            # Step 7: Report timings of each step
            print(f"   Timing breakdown for macrostep {imacro}")
            print("   ---------------------------------")
            print(f"   - P space printing took {t_pspace_printing:.2f} seconds")
            print(f"   - CC(P) took {t_ccp:.2f} seconds")
            print(f"   - CC(P;Q) + selection took {t_selection_and_ccp3:.2f} seconds")
            print(f"   - Expanding P space took {t_pspace_expand:.2f} seconds")
            print(f"   - Total time: {t_pspace_printing + t_pspace_expand + t_ccp + t_selection_and_ccp3 + t_pspace_expand:.2f} seconds")

        # Final printout of the results including energy extrapolations
        print("   Adaptive CC(P;Q) calculation ended on", get_timestamp(), "\n")
        print("   Summary of results:")
        print(f"    Iteration       E0(P)             E0(P;Q)             E{self.state_index}(P)             E{self.state_index}(P;Q)         VEE(P)      VEE(P;Q)")
        print("   --------------------------------------------------------------------------------------------------------------")
        for i in range(imacro + 1):
            print("   %8d    %.10f     %.10f     %.10f     %.10f     %.4f eV    %.4f eV" %
                (i,
                 self.ccp_energy[i],
                 self.ccpq_energy[i],
                 self.eomccp_energy[i],
                 self.eomccpq_energy[i],
                 (self.eomccp_energy[i] - self.ccp_energy[i]) * hartreetoeV,
                 (self.eomccpq_energy[i] - self.ccpq_energy[i]) * hartreetoeV,
                 )
            )



# Legacy version of adaptive CC(P;Q) that includes CCSD(T) corrections and full moment selection options
# class AdaptDriver:
#
#     def __init__(self, driver, percentage, full_storage=False, perturbative=False, pspace_analysis=True, two_body_left=False):
#         from ccpy.utilities.pspace import get_empty_pspace
#         self.driver = driver
#         self.percentage = percentage
#         self.options = {"full storage": full_storage,
#                         "perturbative": perturbative,
#                         "P space analysis": pspace_analysis,
#                         "two_body_left": two_body_left}
#
#         self.nmacro = len(self.percentage)
#         self.ccp_energy = np.zeros(self.nmacro)
#         self.ccpq_energy = np.zeros(self.nmacro)
#         self.pspace = get_empty_pspace(self.driver.system, 3, use_bool=True)
#         self.t3_excitations = {"aaa": np.ones((1, 6), order="F"),
#                                "aab": np.ones((1, 6), order="F"),
#                                "abb": np.ones((1, 6), order="F"),
#                                "bbb": np.ones((1, 6), order="F")}
#
#         # Save the bare Hamiltonian for later iterations if using CR-CC(2,3)
#         # since H will get overwritten to HBar. When using CCSD(T), H never
#         # changes, so you don't need to do this.
#         if not self.options["perturbative"]:
#             self.bare_hamiltonian = deepcopy(self.driver.hamiltonian)
#
#     def print_options(self):
#         print("   ------------------------------------------")
#         for option_key, option_value in self.options.items():
#             print("  ", option_key, "=", option_value)
#         print("   ------------------------------------------\n")
#
#     def excitation_count(self):
#         """Performs an initial symmetry-adapted count of the triples excitation
#            space to determine the total number of triples and the relevant incremental
#            additions used throughout the calculation."""
#         from ccpy.utilities.symmetry import count_triples
#         # Count the triples using symmetry
#         self.num_excitations_symmetry, _ = count_triples(self.driver.system)
#         self.num_total_excitations = self.num_excitations_symmetry[self.driver.system.point_group_irrep_to_number[self.driver.system.reference_symmetry]]
#         self.one_increment = int(0.01 * self.num_total_excitations)
#         # Setting up the number of added determinants in each iteration
#         self.num_dets_to_add = np.zeros(len(self.percentage))
#         for i in range(len(self.percentage) - 1):
#             if self.percentage[i + 1] == 100.0:
#                 self.num_dets_to_add[i] = self.num_total_excitations - self.one_increment * self.percentage[i]
#             else:
#                 self.num_dets_to_add[i] = self.one_increment * (self.percentage[i + 1] - self.percentage[i])
#         self.num_dets_to_add[-1] = 1
#         # Adjust for RHF symmetry
#         if self.driver.options["RHF_symmetry"]:
#             for i in range(len(self.percentage) - 1):
#                 self.num_dets_to_add[i] = int(self.num_dets_to_add[i] / 2)
#
#     def analyze_pspace(self):
#         """Counts and analyzes the P space in terms of spatial and Sz-spin symmetry."""
#         from ccpy.utilities.pspace import count_excitations_in_pspace
#         # Count the excitations in the current P space
#         excitation_count = count_excitations_in_pspace(self.pspace, self.driver.system)
#
#         # The adaptive P spaces do not follow the same pattern of the CIPSI-generated P spaces
#         # is this just a difference in how CI vs. CC gets spin adapted or is this a problem?
#         # E.g., if the aaa and aab do not completely overlap, is S2 symmetry broken for RHF?
#         # Clearly at the CCSDT limit, this condition is filled, so spin symmetry is restored.
#         # noa, nob, nua, nub = self.bare_hamiltonian.ab.oovv.shape
#         # for a in range(nua):
#         #     for b in range(a + 1, nua):
#         #         for c in range(b + 1, nua):
#         #             for i in range(noa):
#         #                 for j in range(i + 1, noa):
#         #                     for k in range(j + 1, noa):
#         #                         if self.pspace["aaa"][a, b, c, i, j, k] == 1:
#         #                             if self.pspace["aab"][a, b, c, i, j, k] != 1:
#         #                                 print("aab VIOLATION")
#         #                             if self.pspace["bbb"][a, b, c, i, j, k] != 1:
#         #                                 print("bbb VIOLATION")
#         # noa, nob, nua, nub = self.bare_hamiltonian.ab.oovv.shape
#         # for a in range(nua):
#         #     for b in range(a + 1, nua):
#         #         for c in range(nub):
#         #             for i in range(noa):
#         #                 for j in range(i + 1, noa):
#         #                     for k in range(nob):
#         #                         if self.pspace["aab"][a, b, c, i, j, k] == 1:
#         #                             if self.pspace["abb"][c, a, b, k, i, j] != 1:
#         #                                 print("VIOLATION")
#
#     def run_ccp(self, imacro):
#         """Runs iterative CC(P), and if needed, HBar and iterative left-CC calculations."""
#         self.driver.run_ccp(method="ccsdt_p", t3_excitations=self.t3_excitations)
#         if not self.options["perturbative"]:
#             if self.options["two_body_left"]:
#                 self.driver.run_hbar(method="ccsd")
#                 self.driver.run_leftcc(method="left_ccsd")
#             else:
#                 self.driver.run_hbar(method="ccsdt_p", t3_excitations=self.t3_excitations)
#                 self.driver.run_leftccp(method="left_ccsdt_p", t3_excitations=self.t3_excitations)
#         self.ccp_energy[imacro] = self.driver.system.reference_energy + self.driver.correlation_energy
#
#     def run_ccp3(self, imacro):
#         """Runs the CC(P;3) correction using either the CR-CC(2,3)- or CCSD(T)-like approach,
#            while simultaneously selecting the leading triply excited determinants and returning
#            the result in an array. For the last calculation, this should not perform the
#            selection steps."""
#         from ccpy.moments.ccp3 import calc_ccp3_with_selection, calc_ccpert3_with_selection
#
#         if not self.options["perturbative"]: # CR-CC(2,3) method
#             if imacro < self.nmacro - 1:
#                 self.ccpq_energy[imacro], triples_list = calc_ccp3_with_selection(self.driver.T,
#                                                                                   self.driver.L[0],
#                                                                                   self.driver.correlation_energy,
#                                                                                   self.driver.hamiltonian,
#                                                                                   self.bare_hamiltonian,
#                                                                                   self.driver.system,
#                                                                                   self.pspace,
#                                                                                   self.num_dets_to_add[imacro],
#                                                                                   use_RHF=self.driver.options["RHF_symmetry"])
#             else:
#                 triples_list = []
#                 self.driver.run_ccp3(method="ccp3", state_index=[0], two_body_approx=True, pspace=self.pspace, t3_excitations=self.t3_excitations)
#                 self.ccpq_energy[imacro] = self.driver.system.reference_energy + self.driver.correlation_energy + self.driver.deltap3[0]["D"]
#         else: # CCSD(T) method
#             if imacro < self.nmacro - 1:
#                 self.ccpq_energy[imacro], triples_list = calc_ccpert3_with_selection(self.driver.T,
#                                                                                      self.driver.correlation_energy,
#                                                                                      self.driver.hamiltonian,
#                                                                                      self.driver.system,
#                                                                                      self.pspace,
#                                                                                      self.num_dets_to_add[imacro],
#                                                                                      use_RHF=self.driver.options["RHF_symmetry"])
#             else:
#                 triples_list = []
#                 self.driver.run_ccp3(method="ccp3(t)", state_index=[0], pspace=self.pspace)
#                 self.ccpq_energy[imacro] = self.driver.system.reference_energy + self.driver.correlation_energy + self.driver.deltap3[0]["A"]
#
#         return triples_list
#
#     def run_ccp3_fullstorage(self, imacro):
#         """Runs the CC(P;3) correction using either the CR-CC(2,3)- or CCSD(T)-like
#            approach and stores all moment corrections in one array in memory, which is later used
#            to identify the top determinants for inclusion in the P space.
#            For the last calculation, this should not perform the selection steps."""
#         from ccpy.moments.ccp3 import calc_ccp3_with_moments, calc_ccpert3_with_moments
#
#         if not self.options["perturbative"]: # CR-CC(2,3) method
#             if imacro < self.nmacro - 1:
#                 self.ccpq_energy[imacro], moments = calc_ccp3_with_moments(self.driver.T,
#                                                                            self.driver.L[0],
#                                                                            self.driver.correlation_energy,
#                                                                            self.driver.hamiltonian,
#                                                                            self.bare_hamiltonian,
#                                                                            self.driver.system,
#                                                                            self.pspace,
#                                                                            use_RHF=self.driver.options["RHF_symmetry"])
#             else:
#                 moments = []
#                 self.driver.run_ccp3(method="ccp3", state_index=[0], two_body_approx=True, pspace=self.pspace, t3_excitations=self.t3_excitations)
#                 self.ccpq_energy[imacro] = self.driver.system.reference_energy + self.driver.correlation_energy + self.driver.deltap3[0]["D"]
#         else: # CCSD(T) method
#             if imacro < self.nmacro - 1:
#                 self.ccpq_energy[imacro], moments = calc_ccpert3_with_moments(self.driver.T,
#                                                                               self.driver.correlation_energy,
#                                                                               self.driver.hamiltonian,
#                                                                               self.driver.system,
#                                                                               self.pspace,
#                                                                               use_RHF=self.driver.options["RHF_symmetry"])
#             else:
#                 moments = []
#                 self.driver.run_ccp3(method="ccp3(t)", state_index=[0], pspace=self.pspace)
#                 self.ccpq_energy[imacro] = self.driver.system.reference_energy + self.driver.correlation_energy + self.driver.deltap3[0]["A"]
#
#         return moments
#
#     def run_expand_pspace(self, imacro, moments_or_triples):
#         """This will expand the P space using the list of triply excited determinants identified
#            using the CC(P;Q) moment expansions, above."""
#         if self.options["full storage"]:
#             from ccpy.utilities.pspace import adaptive_triples_selection_from_moments
#             self.pspace, self.t3_excitations = adaptive_triples_selection_from_moments(moments_or_triples, self.pspace, self.t3_excitations, self.num_dets_to_add[imacro], self.driver.system, self.driver.options["RHF_symmetry"])
#         else:
#             from ccpy.utilities.pspace import add_spinorbital_triples_to_pspace
#             self.pspace, self.t3_excitations = add_spinorbital_triples_to_pspace(moments_or_triples, self.pspace, self.t3_excitations, self.driver.options["RHF_symmetry"])
#
#     def run(self):
#         """This is the main driver for the entire adaptive CC(P;Q) calculation. It will call the above
#            methods in the correct sequence and handle logic accordingly."""
#
#         # Print the options as a header
#         print("   Adaptive CC(P;Q) calculation started on", get_timestamp())
#         self.print_options()
#
#         # Step 0: Perform the preliminary excitation counting
#         print("   Preliminary excitation count...", end=" ")
#         t1 = time.perf_counter()
#         self.excitation_count()
#         print("completed in", time.perf_counter() - t1, "seconds")
#         print("   Excitation Count Summary:")
#         spin_fact = 1.0
#         if self.driver.options["RHF_symmetry"]:
#             spin_fact = 2.0
#         for i, count in enumerate(self.num_excitations_symmetry):
#             symmetry = self.driver.system.point_group_number_to_irrep[i]
#             print("      Symmetry", symmetry, " = ", count)
#         print("      Using", self.num_total_excitations, "as total for ground state.")
#         print("      Determinant addition plan:", [int("{0:.0f}".format(v, i)) for i, v in enumerate(self.num_dets_to_add[:-1] * spin_fact)])
#         print("")
#         for imacro in range(self.nmacro):
#             print("   Adaptive CC(P;Q) Macroiteration - ", imacro, "Fraction of triples = ", self.percentage[imacro], "%")
#             # Step 1: Analyze the P space (optional)
#             if self.options["P space analysis"]:
#                 self.analyze_pspace()
#             # Step 2: Run CC(P) on this P space
#             self.run_ccp(imacro)
#             # Step 3: Moment correction + adaptive selection
#             if self.options["full storage"]:
#                 selection_arr = self.run_ccp3_fullstorage(imacro)
#             else:
#                 selection_arr = self.run_ccp3(imacro)
#
#             # Print the change in CC(P) and CC(P;Q) energies
#             if imacro > 0:
#                 print("   Change in CC(P) energy = ", self.ccp_energy[imacro] - self.ccp_energy[imacro - 1])
#                 print("   Change in CC(P;Q) energy = ", self.ccpq_energy[imacro] - self.ccpq_energy[imacro - 1], "\n")
#             if imacro == self.nmacro - 1:
#                 break
#
#             # Step 4: Expand the P space
#             self.run_expand_pspace(imacro, selection_arr)
#             # Step 5: Reset variables in driver (CAN WE AVOID DOING THIS, PLEASE?)
#             self.driver.T = None
#             if not self.options["perturbative"]:
#                 self.driver.L[0] = None
#                 setattr(self.driver, "hamiltonian", self.bare_hamiltonian)
#
#         # Print results
#         print("   Adaptive CC(P;Q) calculation ended on", get_timestamp(), "\n")
#         print("   Summary of results:")
#         print("    %T           E(P)              E(P;Q)")
#         print("   ------------------------------------------")
#         for i in range(self.nmacro):
#             print("   %3.2f    %.10f     %.10f" % (self.percentage[i], self.ccp_energy[i], self.ccpq_energy[i]))

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
