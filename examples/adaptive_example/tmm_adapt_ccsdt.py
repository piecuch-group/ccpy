from pyscf import gto, scf

from ccpy.models.calculation import Calculation
from ccpy.adaptive.adapt_ccsdt import adapt_ccsdt
from ccpy.interfaces.pyscf_tools import load_pyscf_integrals

if __name__ == "__main__":

    tmm_singlet = """
            C  0.0000000000        0.0000000000       -2.5866403780
            C  0.0000000000        0.0000000000       -0.0574311160
            C  0.0000000000       -2.3554018654        1.3522478168
            C  0.0000000000        2.3554018654        1.3522478168
            H  0.0000000000        1.7421982357       -3.6398423076
            H  0.0000000000       -1.7421982357       -3.6398423076
            H  0.0000000000        2.3674499646        3.3825336644
            H  0.0000000000       -2.3674499646        3.3825336644
            H  0.0000000000        4.1366308985        0.3778613882
            H  0.0000000000       -4.1366308985        0.3778613882
    """

    mol = gto.Mole()

    mol.build(
        atom=tmm_singlet,
        basis="cc-pvdz",
        charge=0,
        spin=0,
        symmetry="C2V",
        cart=False,
        unit="Bohr",
    )
    mf = scf.ROHF(mol)
    mf.kernel()

    system, H = load_pyscf_integrals(mf, nfrozen=4)
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



