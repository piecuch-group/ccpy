from pyscf import gto, scf

from ccpy.models.calculation import Calculation
from ccpy.interfaces.pyscf_tools import load_pyscf_integrals
from ccpy.drivers.driver import cc_driver, lcc_driver, eomcc_driver

from ccpy.hbar.hbar_ccsd import build_hbar_ccsd

from ccpy.eomcc.initial_guess import get_initial_guess

if __name__ == "__main__":

    mol = gto.Mole()
    mol.build(
        atom="""H 0.0 -1.515263  -1.058898
                H 0.0 1.515263  -1.058898
                O 0.0 0.0 -0.0090""",
        basis="ccpvdz",
        charge=0,
        spin=0,
        symmetry="C2V",
        cart=False,
        unit='Bohr',
    )
    mf = scf.ROHF(mol)
    mf.kernel()

    system, H = load_pyscf_integrals(mf, 0)

    calculation = Calculation(
        order=2,
        calculation_type="ccsd",
        convergence_tolerance=1.0e-08
    )

    T, total_energy, is_converged = cc_driver(calculation, system, H)

    Hbar = build_hbar_ccsd(T, H)

    calculation = Calculation(
        order=2,
        calculation_type="eomccsd",
        convergence_tolerance=1.0e-08,
        multiplicity=1,
    )

    R, _ = get_initial_guess(calculation, system, Hbar, 10, noact=0, nuact=0, guess_order=1)

    R, omega, is_converged = eomcc_driver(calculation, system, Hbar, T, R)