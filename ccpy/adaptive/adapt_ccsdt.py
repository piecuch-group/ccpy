import numpy as np

def adapt_ccsdt(calculation_cc, calculation_lcc, system, hamiltonian, T=None, relaxed=True):
    """Performs the adaptive CC(P;Q) calculation specified by the user in the input."""
    from ccpy.utilities.pspace import get_empty_pspace
    from ccpy.adaptive.selection import select_triples_by_moments 
    from ccpy.drivers.driver import ccp_driver, lcc_driver
    from ccpy.hbar.hbar_ccsd import build_hbar_ccsd
    from ccpy.moments.ccp3 import calc_ccp3

    # check if requested CC(P) calculation is implemented in modules
    # assuming the underlying CC(P) follows in the * in the calculation_type
    # input adapt_*, as in adapt_ccsdt -> "ccsdt_p:
    setattr(calculation_cc, "calculation_type", calculation_cc.calculation_type.split('_')[1] + "_p")

    # start with an empty P space
    pspace = get_empty_pspace(system, calculation.order)

    for n in 

    for n in calculation.adaptive_triples_percentages:

        T, ccp_energy[n], is_converged = ccp_driver(calculation_cc, system. hamiltonian, pspace)
        assert(is_converged)

        Hbar = build_hbar_ccsd(T, hamiltonian)
    
        L, _, is_converged = lcc_driver(calculation, system, T, Hbar)
        assert(is_converged)
    
        ccpq_energy[n], _, moments_aaa, moments_aab, moments_abb, moments_bbb = calc_ccp3(T, L, 
                                                                                          Hbar, hamiltonian, 
                                                                                          system, 
                                                                                          pspace, 
                                                                                          use_RHF=False, 
                                                                                          return_moment=True)

        pspace, num_added_dets = select_triples_by_moments(moments_aaa, moments_aab, moments_abb, moments_bbb, 
                                                           pspace, 
                                                           num_add, 
                                                           system)
        assert(num_added_dets == num_add)


    return T, ccp_energy, ccpq_energy


