
def test_gamess():

    from ccpy.models.calculation import Calculation
    from ccpy.adaptive.adapt_ccsdt import adapt_ccsdt
    from ccpy.interfaces.gamess_tools import load_from_gamess

    system, H = load_from_gamess(
            "f2-2Re.log",
            "onebody.inp",
            "twobody.inp",
            nfrozen=2,
    )

    system.print_info()

    calculation = Calculation(
            order=3,
            calculation_type="adapt_ccsdt",
            convergence_tolerance=1.0e-08,
            diis_size=6,
            energy_shift=0.0,
            maximum_iterations=500,
            RHF_symmetry=False,
            low_memory=False,
            adaptive_percentages=[100.0]
    )
    T, total_energy, is_converged = adapt_ccsdt(calculation, system, H, relaxed=True)

def test_pyscf(stretch, basis):

    from pyscf import gto, scf

    from ccpy.models.calculation import Calculation
    from ccpy.adaptive.adapt_ccsdt import adapt_ccsdt
    from ccpy.interfaces.pyscf_tools import load_pyscf_integrals

    mol = gto.Mole()

    if basis == "ccpvdz":
        cartesian = True
    else:
        cartesian = False

    Re = 2.66816 # a.u.
    mol.build(
        atom=[['F', (0, 0, -0.5 * Re * stretch)], ['F', (0, 0, 0.5 * Re * stretch)]],
        basis=basis,
        charge=0,
        spin=0,
        symmetry="D2H",
        cart=cartesian,
        unit='Bohr',
    )
    mf = scf.ROHF(mol)
    mf.kernel()

    system, H = load_pyscf_integrals(mf, nfrozen=2)
    system.print_info()

    calculation = Calculation(
            calculation_type="adapt_ccsdt",
            convergence_tolerance=1.0e-08,
            diis_size=6,
            energy_shift=0.0,
            maximum_iterations=500,
            RHF_symmetry=False,
            low_memory=False,
            adaptive_percentages=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
            #adaptive_percentages=[1.0]
    )

    T, total_energy, is_converged = adapt_ccsdt(calculation, system, H, pert_corr=False, relaxed=True)

if __name__ == "__main__":

    #test_gamess()
    test_pyscf(2.0, "ccpvdz")


