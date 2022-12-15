
def test_gamess():

    from ccpy.models.calculation import Calculation
    from ccpy.adaptive.adapt_ccsdt import adapt_ccsdt
    from ccpy.interfaces.gamess_tools import load_from_gamess

    system, H = load_from_gamess(
            "/scratch/gururang/test_ccpq_2ba/cbd-R/rectangle-d2h.log",
            "/scratch/gururang/test_ccpq_2ba/cbd-R/onebody.inp",
            "/scratch/gururang/test_ccpq_2ba/cbd-R/twobody.inp",
            nfrozen=4,
    )
    system.print_info()

    calculation = Calculation(
            calculation_type="adapt_ccsdt",
            convergence_tolerance=1.0e-08,
            diis_size=6,
            energy_shift=0.0,
            maximum_iterations=500,
            RHF_symmetry=False,
            adaptive_percentages=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    )   

    T, total_energy, is_converged = adapt_ccsdt(calculation, system, H, relaxed=True)

if __name__ == "__main__":

    test_gamess()


