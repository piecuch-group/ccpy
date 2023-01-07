from pyscf import gto, scf

from ccpy.models.calculation import Calculation
from ccpy.interfaces.pyscf_tools import load_pyscf_integrals
from ccpy.drivers.driver import cc_driver, lcc_driver

from ccpy.hbar.hbar_ccsd import build_hbar_ccsd

from ccpy.moments.crcc23 import calc_crcc23
from ccpy.moments.cct3 import calc_cct3


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

    system.set_active_space(6, 6)
    calculation = Calculation(
        calculation_type="ccsdt1",
        maximum_iterations=200,
        RHF_symmetry=False,
    )
    T, total_energy, is_converged = cc_driver(calculation, system, H)

    calculation = Calculation(
        calculation_type="left_ccsd",
        maximum_iterations=200,
        RHF_symmetry=True,
    )

    Hbar = build_hbar_ccsd(T, H)

    L, total_energy, is_converged = lcc_driver(calculation, system, T, Hbar, omega=0.0, L=None, R=None)

    Ecct3, delta23 = calc_cct3(T, L, Hbar, H, system, use_RHF=True)
