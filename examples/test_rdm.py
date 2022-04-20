import numpy as np

def main():

    from pyscf import gto, scf

    from ccpy.models.calculation import Calculation
    from ccpy.interfaces.pyscf_tools import load_pyscf_integrals, get_mo_integrals
    from ccpy.drivers.driver import cc_driver, lcc_driver

    from ccpy.hbar.hbar_ccsd import build_hbar_ccsd

    from ccpy.density.rdm1 import calc_rdm1
    from ccpy.density.ccsd_no import convert_to_ccsd_no

    mol = gto.Mole()

    mol.build(
        atom=[['H', (0, 0, -2.66816/2.0)], ['F', (0, 0, 2.66816/2.0)]],
        basis="ccpvdz",
        charge=0,
        spin=0,
        symmetry="C2V",
        unit='Bohr',
        cart=True
    )
    mf = scf.ROHF(mol)
    mf.kernel()

    system, H = load_pyscf_integrals(mf, nfrozen=0)
    system.print_info()

    calculation = Calculation(
        order=2,
        calculation_type="ccsd",
        convergence_tolerance=1.0e-08,
        diis_size=6,
        maximum_iterations=80,
    )

    T, total_energy_canonical, is_converged = cc_driver(calculation, system, H)

    calculation = Calculation(
        order=2,
        calculation_type="left_ccsd",
        convergence_tolerance=1.0e-08
    )

    Hbar = build_hbar_ccsd(T, H)

    L, total_energy, is_converged = lcc_driver(calculation, system, T, Hbar, omega=0.0, L=None, R=None)

    rdm1 = calc_rdm1(T, L, system)
    H, system = convert_to_ccsd_no(rdm1, H, system)

    calculation = Calculation(
        order=2,
        calculation_type="ccsd",
        convergence_tolerance=1.0e-08,
        diis_size=6,
        maximum_iterations=80,
    )
    T, total_energy_natorb, is_converged = cc_driver(calculation, system, H)

    if abs(total_energy_natorb - total_energy_canonical) < 1.0e-06:
        print('Passed!')
        print('Canonical CC energy = ', total_energy_canonical)
        print('Natural Orbital CC energy = ', total_energy_natorb)
    else:
        print('Failed!')
        print('Canonical CC energy = ', total_energy_canonical)
        print('Natural Orbital CC energy = ', total_energy_natorb)



if __name__ == "__main__":

    main()

