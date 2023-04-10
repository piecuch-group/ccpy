import numpy as np

from ccpy.models.calculation import Calculation
from ccpy.utilities.pspace import get_empty_pspace, count_excitations_in_pspace_with_symmetry, add_spinorbital_triples_to_pspace
from ccpy.utilities.symmetry import count_triples
from ccpy.drivers.driver import cc_driver, lcc_driver
from ccpy.hbar.hbar_ccsd import build_hbar_ccsd
from ccpy.moments.ccp3 import calc_ccp3, calc_ccp3_with_selection, calc_ccp3_with_moments, calc_ccpert3_with_selection, calc_ccpert3_with_moments
from ccpy.utilities.pspace import adaptive_triples_selection_from_moments

def adapt_ccsdt(calculation, system, hamiltonian, T=None, pert_corr=False, on_the_fly=True, relaxed=True):

    if relaxed:
        T, ccp_energy, ccpq_energy = adapt_ccsdt_relaxed(calculation, system, hamiltonian, pert_corr, on_the_fly, T=None)


    return T, ccp_energy, ccpq_energy

#[TODO]: Generalize this function to at least handling singles through quadruples, not just triples
def adapt_ccsdt_relaxed(calculation, system, hamiltonian, pert_corr, on_the_fly, T=None):
    """Performs the adaptive CC(P;Q) calculation specified by the user in the input."""

    # check if requested CC(P) calculation is implemented in modules
    # assuming the underlying CC(P) follows in the * in the calculation_type
    # input adapt_*, as in adapt_ccsdt -> "ccsdt_p:
    # setattr(calculation, "calculation_type", calculation.calculation_type.split('_')[1] + "_p_slow")
    # setattr(calculation, "calculation_type", calculation.calculation_type.split('_')[1] + "_p_quadratic_omp")
    setattr(calculation, "calculation_type", calculation.calculation_type.split('_')[1] + "_p_quadratic_direct_opt")

    # make the left-CC calculation using the CC(P) parameters (maybe this isn't the best way)
    calculation_left = Calculation(
                            order=2,
                            calculation_type='left_ccsd',
                            maximum_iterations=calculation.maximum_iterations,
                            convergence_tolerance=calculation.convergence_tolerance,
                            energy_shift=calculation.energy_shift,
                            low_memory=False
                            )

    # start with an empty P space
    pspace = get_empty_pspace(system, calculation.order, use_bool=True)

    # Start with empty t3_excitations; remember, a single entry of [1., 1., 1., 1., 1., 1.]
    # in t3_excitations[spincase] is the default to signal that there are 0 triples in that 
    # category
    t3_excitations = {"aaa" : np.ones((1, 6)),
                      "aab" : np.ones((1, 6)),
                      "abb" : np.ones((1, 6)),
                      "bbb" : np.ones((1, 6))}

    # get total number of external determinants in the problem (e.g., triples)
    count_sym, _ = count_triples(system)
    num_total = count_sym[system.point_group_irrep_to_number[system.reference_symmetry]]

    # Get the increment of 1% triples
    n1 = int(0.01 * num_total)
    print("   The total number of triples of ground-state symmetry ({}) is {}".format(system.reference_symmetry, num_total))
    print("   The increment of 1% is {}".format(n1))

    calculation.adaptive_percentages.insert(0, 0.0)
    num_calcs = len(calculation.adaptive_percentages)
    num_dets_to_add = np.zeros(num_calcs)
    for i in range(num_calcs - 1):
        if calculation.adaptive_percentages[i + 1] == 100.0:
            num_dets_to_add[i] = num_total - n1 * calculation.adaptive_percentages[i]
        else:
            num_dets_to_add[i] = n1 * (calculation.adaptive_percentages[i + 1] - calculation.adaptive_percentages[i])
    num_dets_to_add[-1] = 1

    # half the number of determinants to add for RHF symmetry case
    if calculation.RHF_symmetry:
        for i in range(num_calcs - 1):
            num_dets_to_add[i] = int(num_dets_to_add[i] / 2)

    ccp_energy = np.zeros(num_calcs)
    ccpq_energy = np.zeros(num_calcs)
    for n in range(num_calcs):

        percentage = calculation.adaptive_percentages[n]
        print("\n   ===========================================================================================")
        print("        Performing CC(P;Q) calculation with", percentage, "% triples (", n1 * percentage, "triples )")
        print("   ===========================================================================================\n")

        # Count the excitations in the current P space
        excitation_count = count_excitations_in_pspace_with_symmetry(pspace, system)
        for ind, excitation_count_irrep in enumerate(excitation_count):
            tot_p_space = excitation_count_irrep[0]['aaa'] + excitation_count_irrep[0]['aab'] + excitation_count_irrep[0]['abb'] + excitation_count_irrep[0]['bbb']
            print("   Symmetry", system.point_group_number_to_irrep[ind], "-", "Total number of triples in P space = ", tot_p_space)
            print("      Number of aaa = ", excitation_count_irrep[0]['aaa'])
            print("      Number of aab = ", excitation_count_irrep[0]['aab'])
            print("      Number of abb = ", excitation_count_irrep[0]['abb'])
            print("      Number of bbb = ", excitation_count_irrep[0]['bbb'])
        
        # Perform CC(P) calculation using previous T vector as initial guess
        if n > 0:
            # T, ccp_energy[n], is_converged = cc_driver(calculation, system, hamiltonian, pspace=pspace, T_init=T)
            T, ccp_energy[n], is_converged = cc_driver(calculation, system, hamiltonian, t3_excitations=t3_excitations, pspace=pspace[0])
        else:
            # T, ccp_energy[n], is_converged = cc_driver(calculation, system, hamiltonian, pspace=pspace)
            T, ccp_energy[n], is_converged = cc_driver(calculation, system, hamiltonian, t3_excitations=t3_excitations, pspace=pspace[0])

        if pert_corr:
            # Compute the CCSD(T)-like correction and return the moments as well as selected triples
            if on_the_fly:
                Eccp3, deltap3, moments, triples_list = calc_ccpert3_with_selection(T,
                                                                                    hamiltonian,
                                                                                    system,
                                                                                    pspace,
                                                                                    num_dets_to_add[n],
                                                                                    use_RHF=calculation.RHF_symmetry)
            else:
                Eccp3, deltap3, moments = calc_ccpert3_with_moments(T,
                                                                    hamiltonian,
                                                                    system,
                                                                    pspace,
                                                                    use_RHF=calculation.RHF_symmetry)
            ccpq_energy[n] = Eccp3
        else:
            # Build CCSD-like Hbar from CC(P)
            Hbar = build_hbar_ccsd(T, hamiltonian)
            # Perform left-CCSD calculation
            L, _, is_converged = lcc_driver(calculation_left, system, T, Hbar)
            # If last calculation, just perform a simple CC(P;3) and move on; no need to select triples
            if n == num_calcs - 1:
                Eccp3, deltap3 = calc_ccp3(T, L, Hbar, hamiltonian, system, pspace, use_RHF=calculation.RHF_symmetry)
            else: # Compute the CR-CC(2,3)-like correction and return the moments as well as selected triples
                if on_the_fly:
                    Eccp3, deltap3, moments, triples_list = calc_ccp3_with_selection(T, L,
                                                                                     Hbar, hamiltonian,
                                                                                     system,
                                                                                     pspace,
                                                                                     num_dets_to_add[n],
                                                                                     use_RHF=calculation.RHF_symmetry)
                else:
                    Eccp3, deltap3, moments = calc_ccp3_with_moments(T, L, Hbar, hamiltonian, system, pspace, use_RHF=calculation.RHF_symmetry)
            ccpq_energy[n] = Eccp3["D"]

        # add the triples
        if n < num_calcs - 1:
            if on_the_fly:
                pspace[0], t3_excitations = add_spinorbital_triples_to_pspace(triples_list, pspace[0], t3_excitations, calculation.RHF_symmetry)
            else:
                pspace[0], t3_excitations = adaptive_triples_selection_from_moments(moments, pspace[0], t3_excitations, num_dets_to_add[n], system)

    return T, ccp_energy, ccpq_energy
