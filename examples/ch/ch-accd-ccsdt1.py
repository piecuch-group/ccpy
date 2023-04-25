from ccpy.models.calculation import Calculation
from ccpy.interfaces.gamess_tools import load_from_gamess
from ccpy.drivers.driver import cc_driver, lcc_driver

from ccpy.hbar.hbar_ccsd import build_hbar_ccsd

from ccpy.moments.cct3 import calc_cct3

from ccpy.utilities.pspace import get_active_pspace
from ccpy.utilities.active_space import get_active_slices


if __name__ == "__main__":



    system, H = load_from_gamess("cc-cct3-ch.log", "onebody.inp", "twobody.inp", nfrozen=1)
    system.set_active_space(nact_occupied=1, nact_unoccupied=4)
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    print("Active alpha occupied = ", Oa)
    print("Active beta occupied = ", Ob)
    print("Active alpha unoccupied = ", Va)
    print("Active beta unoccupied = ", Vb)

    calculation = Calculation(
        order=3,
        active_orders=[3],
        num_active=[1],
        calculation_type="ccsdt1",
        convergence_tolerance=1.0e-08
    )
    system.print_info()
    #pspace = get_active_pspace(system, nact_o_alpha=2, nact_u_alpha=2, nact_o_beta=1, nact_u_beta=3)
    T, total_energy, is_converged = cc_driver(calculation, system, H)


    #T, total_energy, is_converged = cc_driver(calculation, system, H)

    # calculation = Calculation(
    #    order=2,
    #    calculation_type="left_ccsd",
    #    convergence_tolerance=1.0e-08
    # )
    #
    # Hbar = build_hbar_ccsd(T, H)
    #
    # L, total_energy, is_converged = lcc_driver(calculation, system, T, Hbar, omega=0.0, L=None, R=None)
    #
    # Ecct3, delta23 = calc_cct3(T, L, Hbar, H, system, use_RHF=False)
