from pyscf import gto, scf

from ccpy.models.calculation import Calculation
from ccpy.interfaces.pyscf_tools import load_pyscf_integrals
from ccpy.adaptive.adapt_ccsdt import adapt_ccsdt

if __name__ == "__main__":

    mol = gto.Mole()

    # singlet geometry at 45 degree torsion (biradical state)
    mol.build(
        atom="""C 0.000000000 0.000000000 -1.413312001
                C 0.000000000 0.000000000 1.413312001
                C 0.875012370 2.112466730 -2.712868237
                C -0.875012370 2.112466730 2.712868237
                C -0.875012370 -2.112466730 -2.712868237
                C 0.875012370 -2.112466730 2.712868237
                H 0.868569677 2.152608171 -4.739082085
                H -0.868569677 -2.152608171 -4.739082085
                H 0.868569677 -2.152608171 4.739082085
                H -0.868569677 2.152608171 4.739082085
                H 1.622454529 3.723989797 -1.744591448
                H -1.622454529 -3.723989797 -1.744591448
                H 1.622454529 -3.723989797 1.744591448
                H -1.622454529 3.723989797 1.744591448""",
        basis="ccpvdz",
        charge=0,
        spin=0,
        symmetry="D2",
        cart=False,
        unit="Bohr",
    )
    mf = scf.RHF(mol)
    mf.kernel()

    system, H = load_pyscf_integrals(mf, 6)
    system.print_info()

    calculation = Calculation(
            calculation_type="adapt_ccsdt",
            convergence_tolerance=1.0e-08,
            diis_size=6,
            energy_shift=0.0,
            maximum_iterations=500,
            RHF_symmetry=True,
            adaptive_percentages=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    )

    T, total_energy, is_converged = adapt_ccsdt(calculation, system, H, pert_corr=False, relaxed=True, on_the_fly=True)

