import numpy as np

#[TODO]: Generalize this function to at least handling singles through quadruples, not just triples
def adapt_ccsdt(calculation, system, hamiltonian, T=None, relaxed=True):
    """Performs the adaptive CC(P;Q) calculation specified by the user in the input."""
    from ccpy.models.calculation import Calculation
    from ccpy.utilities.pspace import get_empty_pspace, count_excitations_in_pspace, add_spinorbital_triples_to_pspace
    from ccpy.utilities.symmetry_count import count_triples, count_quadruples
    from ccpy.drivers.driver import cc_driver, lcc_driver
    from ccpy.hbar.hbar_ccsd import build_hbar_ccsd
    from ccpy.moments.ccp3 import calc_ccp3_with_selection

    # check if requested CC(P) calculation is implemented in modules
    # assuming the underlying CC(P) follows in the * in the calculation_type
    # input adapt_*, as in adapt_ccsdt -> "ccsdt_p:
    setattr(calculation, "calculation_type", calculation.calculation_type.split('_')[1] + "_p_v2")

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
    pspace = get_empty_pspace(system, calculation.order)

    # get total number of external determinants in the problem (e.g., triples)
    if calculation.order == 3:
        count_sym, _ = count_triples(system)
        num_total = count_sym[system.point_group_irrep_to_number[system.reference_symmetry]]
        n1 = int(0.01 * num_total)
        print("   The total number of triples of ground-state symmetry ({}) is {}".format(system.reference_symmetry, num_total))
        print("   The increment of 1% is {}".format(n1))
    elif calculation.order == 4:
        count_sym_t3, _ = count_triples(system)
        count_sym_t4, _ = count_quadruples(system)
        num_t3 = count_sym_t3[system.point_group_irrep_to_number[system.reference_symmetry]]
        num_t4 = count_sym_t4[system.point_group_irrep_to_number[system.reference_symmetry]]
        num_total = num_t3 + num_t4
        n1 = int(0.01 * num_total)
        print("   The total number of triples of ground-state symmetry ({}) is {}".format(system.reference_symmetry, num_t3))
        print("   The total number of quadruples of ground-state symmetry ({}) is {}".format(system.reference_symmetry, num_t4))
        print("   The increment of 1% is {}".format(n1))

    calculation.adaptive_percentages.insert(0, 0.0)
    num_calcs = len(calculation.adaptive_percentages)
    num_dets_to_add = np.zeros(num_calcs - 1)
    for i in range(num_calcs - 1):
        num_dets_to_add[i] = n1 * (calculation.adaptive_percentages[i + 1] - calculation.adaptive_percentages[i])  

    ccp_energy = np.zeros(num_calcs)
    ccpq_energy = np.zeros(num_calcs)
    for n in range(num_calcs):

        percentage = calculation.adaptive_percentages[n]
        print("   \nPerforming CC(P;Q) calculation with", percentage, "% triples (", n1 * percentage, "triples )")
        print("   ===========================================================================================\n")

        # Count the excitations in the current P space
        excitation_count = count_excitations_in_pspace(pspace, system)
        tot_p_space = excitation_count[0]['aaa'] + excitation_count[0]['aab'] + excitation_count[0]['abb'] + excitation_count[0]['bbb']
        print("   Total number of triples in P space = ", tot_p_space)
        print("   Number of aaa = ", excitation_count[0]['aaa'])
        print("   Number of aab = ", excitation_count[0]['aab'])
        print("   Number of abb = ", excitation_count[0]['abb'])
        print("   Number of bbb = ", excitation_count[0]['bbb'])
        
        # Perform CC(P) calculation using previous T vector as initial guess
        if n > 0:
            T, ccp_energy[n], is_converged = cc_driver(calculation, system, hamiltonian, pspace=pspace, T=T)
        else:
            T, ccp_energy[n], is_converged = cc_driver(calculation, system, hamiltonian, pspace=pspace)
        assert(is_converged)

        # Build CCSD-like Hbar from CC(P) 
        Hbar = build_hbar_ccsd(T, hamiltonian)

        # Perform left-CCSD calculation
        L, _, is_converged = lcc_driver(calculation_left, system, T, Hbar)
        assert(is_converged)
    
        # Compute the CC(P;3) moment correction used the 2BA and return the moments
        Eccp3, deltap3, moments, triples_list = calc_ccp3_with_selection(T, L,
                                                                         Hbar, hamiltonian,
                                                                         system,
                                                                         pspace,
                                                                         num_dets_to_add[n],
                                                                         use_RHF=calculation.RHF_symmetry)
        ccpq_energy[n] = Eccp3["D"]

        # Select the leading triples from the moment correction
        if n < num_calcs - 1:
            pspace[0] = add_spinorbital_triples_to_pspace(triples_list, pspace[0])

    return T, ccp_energy, ccpq_energy


