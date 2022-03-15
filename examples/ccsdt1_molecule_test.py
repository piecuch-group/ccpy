from pyscf import gto, scf

from ccpy.models.calculation import Calculation
from ccpy.interfaces.pyscf_tools import load_pyscf_integrals
from ccpy.drivers.driver import cc_driver, lcc_driver

if __name__ == "__main__":

    mol = gto.Mole()

    case = 'F2'

    if case == 'F2':
        mol.build(
            atom="""F 0.0 0.0 -2.66816
                    F 0.0 0.0  2.66816""",
            basis="ccpvdz",
            charge=0,
            spin=0,
            symmetry="D2H",
            cart=True,
            unit='Bohr',
        )
        mf = scf.ROHF(mol)
        mf.kernel()

    if case == 'H2O':
        mol.build(
            atom="""H 0.0 -1.515263  -1.058898
                    H 0.0 1.515263  -1.058898
                    O 0.0 0.0 -0.0090""",
            basis="ccpvdz",
            charge=0,
            spin=0,
            symmetry="C2V",
            cart=True,
            unit='Bohr',
        )
        mf = scf.RHF(mol)
        mf.kernel()

    system, H = load_pyscf_integrals(mf, nfrozen=2,
                                     num_act_holes_alpha=5, num_act_particles_alpha=9,
                                     num_act_holes_beta=5, num_act_particles_beta=9,
                                     )

    calculation = Calculation(
        order=3,
        active_orders=[3],
        num_active=[1],
        calculation_type="ccsdt1",
        convergence_tolerance=1.0e-08
    )

    T, total_energy, is_converged = cc_driver(calculation, system, H)