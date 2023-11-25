import numpy as np
import time
from copy import deepcopy
from ccpy.extrapolation.goodson_extrapolation import goodson_extrapolation
from ccpy.utilities.printing import get_timestamp

# [TODO]: Fixed-schedule tempered growth by addition of CCSD problem size each round

class AdaptDriver:

    def __init__(self, driver, percentage):
        self.driver = driver
        self.percentage = percentage
        self.options = {"two_body_left": True,
                        "reset_amplitudes": False,
                        "buffer_factor": 2,
                        "minimum_threshold": 0.0}
        self.nmacro = len(self.percentage)
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
        # Save the bare Hamiltonian for later iterations if using CR-CC(2,3)
        self.bare_hamiltonian = deepcopy(self.driver.hamiltonian)

    def print_options(self):
        print("   ------------------------------------------")
        for option_key, option_value in self.options.items():
            print("  ", option_key, "=", option_value)
        print("   ------------------------------------------\n")

    def excitation_count(self):
        """Performs an initial symmetry-adapted count of the triples excitation
           space to determine the total number of triples and the relevant incremental
           additions used throughout the calculation."""
        from ccpy.utilities.symmetry import count_triples
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
        # Adjust for RHF symmetry
        if self.driver.options["RHF_symmetry"]:
            for i in range(len(self.percentage) - 1):
                self.num_dets_to_add[i] = int(self.num_dets_to_add[i] / 2)

    def print_pspace(self):
        """Counts and analyzes the P space in terms of spatial and Sz-spin symmetry."""
        t_start = time.perf_counter()
        for isym, excitation_count_irrep in enumerate(self.excitation_count_by_symmetry):
            tot_excitation_count_irrep = excitation_count_irrep['aaa'] + excitation_count_irrep['aab'] + \
                                         excitation_count_irrep['abb'] + excitation_count_irrep['bbb']
            print("   Symmetry", self.driver.system.point_group_number_to_irrep[isym], "-", "Total number of triples in P space = ", tot_excitation_count_irrep)
            print("      Number of aaa = ", excitation_count_irrep['aaa'])
            print("      Number of aab = ", excitation_count_irrep['aab'])
            print("      Number of abb = ", excitation_count_irrep['abb'])
            print("      Number of bbb = ", excitation_count_irrep['bbb'])
        minutes, seconds = divmod(time.perf_counter() - t_start, 60)
        print(f"   P space analyzed in {minutes:.1f}m {seconds:.1f}s\n")

    def run_ccp(self, imacro):
        """Runs iterative CC(P), and if needed, HBar and iterative left-CC calculations."""
        self.driver.run_ccp(method="ccsdt_p", t3_excitations=self.t3_excitations)
        if self.options["two_body_left"]:
            self.driver.run_hbar(method="ccsd")
            self.driver.run_leftcc(method="left_ccsd")
        else:
            self.driver.run_hbar(method="ccsdt_p", t3_excitations=self.t3_excitations)
            self.driver.run_leftccp(method="left_ccsdt_p", t3_excitations=self.t3_excitations)
        self.ccp_energy[imacro] = self.driver.system.reference_energy + self.driver.correlation_energy

    def run_ccp3(self, imacro):
        """Runs the CC(P;3) correction using either the CR-CC(2,3)- or CCSD(T)-like approach,
           while simultaneously selecting the leading triply excited determinants and returning
           the result in an array. For the last calculation, this should not perform the
           selection steps."""
        from ccpy.moments.ccp3 import calc_ccp3_with_selection
        from ccpy.hbar.hbar_ccsdt_p import remove_VT3_intermediates

        if imacro < self.nmacro - 1:
            if self.driver.L[0].order > 2: # remove V*T3 intermediates from HBar before proceeding if left-CC(P) is used
                self.driver.hamiltonian = remove_VT3_intermediates(self.driver.T, self.t3_excitations, self.driver.hamiltonian)
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
            self.driver.run_ccp3(method="ccp3", state_index=[0], two_body_approx=True, t3_excitations=self.t3_excitations)
            self.ccpq_energy[imacro] = self.driver.system.reference_energy + self.driver.correlation_energy + self.driver.deltap3[0]["D"]

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
        spin_fact = 1.0
        if self.driver.options["RHF_symmetry"]:
            spin_fact = 2.0
        for i, count in enumerate(self.num_excitations_symmetry):
            symmetry = self.driver.system.point_group_number_to_irrep[i]
            print("      Symmetry", symmetry, " = ", count)
        print("      Using", self.num_total_excitations, "as total for ground state.")
        print("      Determinant addition plan:", [int("{0:.0f}".format(v, i)) for i, v in enumerate(self.num_dets_to_add[:-1] * spin_fact)])
        print("")

        # Begin adaptive loop iterations
        for imacro in range(self.nmacro):
            print("")
            print("   Adaptive CC(P;Q) Macroiteration - ", imacro, "Fraction of triples = ", self.percentage[imacro], "%")
            print("   ==================================================================")

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
            selection_arr = self.run_ccp3(imacro)
            x2 = time.perf_counter()
            t_selection_and_ccp3 = x2 - x1

            # Step 4: Perform Goodson extrapolation
            x1 = time.perf_counter()
            print("   Goodson FCI Extrapolation")
            print("   -------------------------")
            self.ex_ccq[imacro] = goodson_extrapolation(self.driver.system.reference_energy,
                                                        #self.ccp_energy[0],
                                                        self.ccp_energy[imacro],
                                                        self.ccpq_energy[imacro],
                                                        approximant="ccq")
            self.ex_ccr[imacro] = goodson_extrapolation(self.driver.system.reference_energy,
                                                        #self.ccp_energy[0],
                                                        self.ccp_energy[imacro],
                                                        self.ccpq_energy[imacro],
                                                        approximant="ccr")
            self.ex_cccf[imacro] = goodson_extrapolation(self.driver.system.reference_energy,
                                                         #self.ccp_energy[0],
                                                         self.ccp_energy[imacro],
                                                         self.ccpq_energy[imacro],
                                                         approximant="cccf")
            print("   ex-CCq Total Energy =", self.ex_ccq[imacro])
            print("   ex-CCr Total Energy =", self.ex_ccr[imacro])
            print("   ex-CCcf Total Energy =", self.ex_cccf[imacro])
            print("")
            x2 = time.perf_counter()
            t_extrap = x2 - x1

            # Print the change in CC(P) and CC(P;Q) energies
            if imacro > 0:
                print("   Change in CC(P) energy = ", self.ccp_energy[imacro] - self.ccp_energy[imacro - 1])
                print("   Change in CC(P;Q) energy = ", self.ccpq_energy[imacro] - self.ccpq_energy[imacro - 1], "\n")
            if imacro == self.nmacro - 1:
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
            setattr(self.driver, "hamiltonian", self.bare_hamiltonian)

            # Step 7: Report timings of each step
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
        print("    %T           E(P)             E(P;Q)             ex-CCq             ex-CCr             ex-CCcf")
        print("   -------------------------------------------------------------------------------------------------------")
        for i in range(self.nmacro):
            print("   %3.2f    %.10f     %.10f     %.10f     %.10f     %.10f" % (self.percentage[i],
                                                                                 self.ccp_energy[i],
                                                                                 self.ccpq_energy[i],
                                                                                 self.ex_ccq[i],
                                                                                 self.ex_ccr[i],
                                                                                 self.ex_cccf[i])
            )
