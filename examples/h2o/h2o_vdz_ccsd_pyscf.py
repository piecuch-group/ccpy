from pyscf import gto, scf

from ccpy.models.calculation import Calculation
from ccpy.interfaces.pyscf_tools import load_pyscf_integrals
from ccpy.drivers.driver import cc_driver

if __name__ == "__main__":

    mol = gto.Mole()
    mol.build(
        atom="""H 0.0 -1.515263  -1.058898
                H 0.0 1.515263  -1.058898
                O 0.0 0.0 -0.0090""",
        basis="dz",
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
        calculation_type="ccsdtq",
        convergence_tolerance=1.0e-08,
        RHF_symmetry=True,
        low_memory=True,
        maximum_iterations=80,
    )

    T, total_energy, is_converged = cc_driver(calculation, system, H)
