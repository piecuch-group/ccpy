from pyscf import gto, scf

from ccpy.models.calculation import Calculation
from ccpy.interfaces.pyscf_tools import load_pyscf_integrals, get_mo_integrals
from ccpy.drivers.driver import cc_driver, lcc_driver

from ccpy.hbar.hbar_ccsd import build_hbar_ccsd

from ccpy.density.rdm1 import calc_rdm1
from ccpy.density.ccsd_no import convert_to_ccsd_no

def test_ccsdt1(nact_occ_canonical, nact_unocc_canonical, nact_occ_natorb, nact_unocc_natorb):

    mol = gto.Mole()

    mol.build(
        atom=[['F', (0, 0, -2.66816)], ['F', (0, 0, 2.66816)]],
        basis="augccpvdz",
        charge=0,
        spin=0,
        symmetry="D2H",
        unit='Bohr',
        cart=True,
    )
    mf = scf.ROHF(mol)
    mf.kernel()

    system, H = load_pyscf_integrals(mf, nfrozen=2)
    system.print_info()

    ### Initial CCSDt calculation ###
    system.set_active_space(nact_occupied=nact_occ_canonical, nact_unoccupied=nact_unocc_canonical)
    calculation = Calculation(
        order=3,
        calculation_type="ccsdt1",
        active_orders=[3],
        num_active=[1],
        convergence_tolerance=1.0e-08,
        diis_size=6,
        maximum_iterations=300,
    )
    T, total_energy_canonical, is_converged = cc_driver(calculation, system, H)

    ### CCSD Natural Orbital Calculation ###
    calculation = Calculation(
        order=2,
        calculation_type="ccsd",
        convergence_tolerance=1.0e-08,
        diis_size=6,
        maximum_iterations=80,
    )

    T, total_energy_ccsd, _ = cc_driver(calculation, system, H)

    calculation = Calculation(
        order=2,
        calculation_type="left_ccsd",
        convergence_tolerance=1.0e-08
    )

    Hbar = build_hbar_ccsd(T, H)

    L, _, _ = lcc_driver(calculation, system, T, Hbar, omega=0.0, L=None, R=None)

    rdm1 = calc_rdm1(T, L, system)
    H, system = convert_to_ccsd_no(rdm1, H, system, print_diagnostics=True)

    ### Check that the NO transformation was successful ###
    calculation = Calculation(
        order=2,
        calculation_type="ccsd",
        convergence_tolerance=1.0e-08,
        diis_size=6,
        maximum_iterations=80,
    )
    T, e_check, _ = cc_driver(calculation, system, H)

    if abs(e_check - total_energy_ccsd) < 1.0e-06:
        flag_pass = True
    else:
        flag_pass = False

    # Stop the program if the NO transformation is not successful
    assert(flag_pass)

    ### CCSD NO-based CCSDt Calculation ###
    system.set_active_space(nact_occupied=nact_occ_natorb, nact_unoccupied=nact_unocc_natorb)
    calculation = Calculation(
        order=3,
        calculation_type="ccsdt1",
        active_orders=[3],
        num_active=[1],
        convergence_tolerance=1.0e-08,
        diis_size=6,
        maximum_iterations=300,
    )
    T, total_energy_natorb, is_converged = cc_driver(calculation, system, H)

    ### Print summary ###
    print('Summary:')
    print('=========================================')
    print('HF-CCSDt (No={}, Nu={}) energy = {}'.format(nact_occ_canonical, nact_unocc_canonical, total_energy_canonical))
    print('NO-CCSDt (No={}, Nu={}) energy = {}'.format(nact_occ_natorb, nact_unocc_natorb, total_energy_natorb))
    print('=========================================')



if __name__ == "__main__":

    test_ccsdt1(nact_occ_canonical = 1,
                 nact_unocc_canonical = 1,
                 nact_occ_natorb = 1,
                 nact_unocc_natorb = 1)

