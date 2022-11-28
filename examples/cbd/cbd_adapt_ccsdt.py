
def test_pyscf(geometry, basis):

    from pyscf import gto, scf

    from ccpy.models.calculation import Calculation
    from ccpy.adaptive.adapt_ccsdt import adapt_ccsdt
    from ccpy.interfaces.pyscf_tools import load_pyscf_integrals

    mol = gto.Mole()


    mol = gto.Mole()
    if geometry == "R":
        mol.build(
            atom="""C   0.68350000  0.78650000  0.00000000
                    C  -0.68350000  0.78650000  0.00000000
                    C   0.68350000 -0.78650000  0.00000000
                    C  -0.68350000 -0.78650000  0.00000000
                    H   1.45771544  1.55801763  0.00000000
                    H   1.45771544 -1.55801763  0.00000000
                    H  -1.45771544  1.55801763  0.00000000
                    H  -1.45771544 -1.55801763  0.00000000""",
            basis=basis,
            charge=0,
            spin=0,
            symmetry="D2H",
            cart=False,
            unit="Angstrom",
        )

    mf = scf.ROHF(mol)
    mf.kernel()

    system, H = load_pyscf_integrals(mf, nfrozen=4)
    system.print_info()

    calculation = Calculation(
            order=3,
            calculation_type="adapt_ccsdt",
            convergence_tolerance=1.0e-08,
            diis_size=6,
            energy_shift=0.0,
            maximum_iterations=5,
            RHF_symmetry=False,
            low_memory=False,
            adaptive_percentages=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    )

    T, total_energy, is_converged = adapt_ccsdt(calculation, system, H, relaxed=True)

if __name__ == "__main__":

    test_pyscf("R", "ccpvdz")


