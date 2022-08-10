from pyscf import gto, scf

from ccpy.models.calculation import Calculation
from ccpy.interfaces.pyscf_tools import load_pyscf_integrals
from ccpy.drivers.driver import cc_driver

if __name__ == "__main__":

    mol = gto.Mole()

    mol.build(
        atom="""F 0.0 0.0 -2.66816
                F 0.0 0.0  2.66816""",
        basis="ccpvtz",
        charge=0,
        spin=0,
        symmetry="D2H",
        cart=False,
        unit='Bohr',
    )
    mf = scf.ROHF(mol)
    mf.kernel()

    system, H = load_pyscf_integrals(mf, nfrozen=2)
    system.set_active_space(nact_unoccupied=9, nact_occupied=5)
    system.print_info()

    calculation = Calculation(
        order=3,
        active_orders=[3],
        num_active=[1],
        calculation_type="ccsdt1",
        convergence_tolerance=1.0e-08
    )

    T, total_energy, is_converged = cc_driver(calculation, system, H)