import numpy as np

def main():

    from pyscf import gto, scf

    from ccpy.models.calculation import Calculation
    from ccpy.interfaces.pyscf_tools import load_pyscf_integrals
    from ccpy.drivers.driver import cc_driver, lcc_driver

    from ccpy.hbar.hbar_ccsd import build_hbar_ccsd

    from ccpy.density.rdm1 import calc_rdm1
    from ccpy.density.ccsd_no import convert_to_ccsd_no

    mol = gto.Mole()

    mol.build(
        atom=[['H', (0, 0, -0.5)], ['H', (0, 0, 0.5)]],
        basis="ccpvtz",
        charge=0,
        spin=0,
        symmetry="D2H",
        unit='Bohr',
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

    T, total_energy, is_converged = cc_driver(calculation, system, H)

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
    T, total_energy, is_converged = cc_driver(calculation, system, H)


    #Ecrcc23, delta23 = calc_crcc23(T, L, Hbar, H, system, use_RHF=False)



if __name__ == "__main__":


    main()

