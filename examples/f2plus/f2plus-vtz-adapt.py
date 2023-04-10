
def test_gamess():

    from ccpy.models.calculation import Calculation
    from ccpy.adaptive.adapt_ccsdt import adapt_ccsdt
    from ccpy.interfaces.gamess_tools import load_from_gamess

    system, H = load_from_gamess(
            "F2+-2.0-VTZ.log",
            "onebody.inp",
            "twobody.inp",
            nfrozen=2,
    )

    system.print_info()

    calculation = Calculation(
            calculation_type="adapt_ccsdt",
            convergence_tolerance=1.0e-08,
            diis_size=6,
            energy_shift=0.0,
            maximum_iterations=500,
            RHF_symmetry=False,
            adaptive_percentages=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
    )
    T, ccp_energy, ccpq_energy = adapt_ccsdt(calculation, system, H, pert_corr=False, relaxed=True, on_the_fly=True)

    print("CC(P) energies")
    print(ccp_energy)

    print("CC(P;Q) energies")
    print(ccpq_energy)

if __name__ == "__main__":

    test_gamess()


