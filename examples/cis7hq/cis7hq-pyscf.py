
def main():

    from pyscf import gto, scf

    from ccpy.models.calculation import Calculation
    from ccpy.interfaces.pyscf_tools import load_pyscf_integrals
    from ccpy.drivers.driver import cc_driver, lcc_driver

    from ccpy.hbar.hbar_ccsd import build_hbar_ccsd

    from ccpy.moments.crcc23 import calc_crcc23

    mol = gto.Mole()


    cis7hq=[['N', ( 0.343271, 0.0, -2.068562)],
            ['C', ( 0.354869, 0.0, -0.704676)],
            ['C', ( 1.562696, 0.0,  0.055484)],
            ['C', ( 2.783986, 0.0, -0.648897)],
            ['C', ( 2.760066, 0.0, -2.024315)],
            ['C', ( 1.514293, 0.0, -2.685591)],
            ['C', (-0.888194, 0.0, -0.034322)],
            ['C', (-0.929640, 0.0,  1.341470)],
            ['C', ( 0.261864, 0.0,  2.099571)],
            ['C', ( 1.482223, 0.0,  1.468716)],
            ['O', (-2.092007, 0.0,  2.061693)],
            ['H', (-2.835718, 0.0,  1.447352)],
            ['H', ( 0.186177, 0.0,  3.179039)],
            ['H', ( 2.396159, 0.0,  2.050825)],
            ['H', (-1.790949, 0.0, -0.634527)],
            ['H', ( 3.674601, 0.0, -2.602008)],
            ['H', ( 1.480380, 0.0, -3.769744)],
            ['H', ( 3.719711, 0.0, -0.101760)],
            ]

    mol.build(
        atom=cis7hq,
        basis="ccpvdz",
        charge=0,
        spin=0,
        symmetry="C1",
        cart=False,
        unit='Angstrom',
    )
    mf = scf.RHF(mol)
    mf.kernel()

    system, H = load_pyscf_integrals(mf, nfrozen=11)
    system.print_info()

    calculation = Calculation(
        order=2,
        calculation_type="ccsd",
        convergence_tolerance=1.0e-07,
        diis_size=6,
        RHF_symmetry=True
    )

    T, total_energy, is_converged = cc_driver(calculation, system, H)

    calculation = Calculation(
        order=2,
        calculation_type="left_ccsd",
        convergence_tolerance=1.0e-07,
        diis_size=6,
        RHF_symmetry=True
    )

    Hbar = build_hbar_ccsd(T, H)

    L, total_energy, is_converged = lcc_driver(calculation, system, T, Hbar, omega=0.0, L=None, R=None)

    Ecrcc23, delta23 = calc_crcc23(T, L, Hbar, H, system, use_RHF=True)



if __name__ == "__main__":

    main()



