from pyscf import gto, scf

from ccpy.models.calculation import Calculation
from ccpy.interfaces.pyscf_tools import load_pyscf_integrals
from ccpy.drivers.driver import cc_driver, lcc_driver
from ccpy.hbar.hbar_ccsd import build_hbar_ccsd
from ccpy.moments.cct3 import calc_cct3
from ccpy.moments.ccp3 import calc_ccp3

from ccpy.utilities.pspace import get_active_pspace

if __name__ == "__main__":

    mol = gto.Mole()

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

    system, H = load_pyscf_integrals(mf, nfrozen=2)
    system.set_active_space(nact_unoccupied=2, nact_occupied=2)
    system.print_info()

    # check that CC(P;3) and CC(t;3) give the same answer
    #pspace = get_active_pspace(system, nact_o=5, nact_u=9)


    calculation = Calculation(
        calculation_type="ccsdt1",
        convergence_tolerance=1.0e-08
    )

    T, total_energy, is_converged = cc_driver(calculation, system, H)

    Hbar = build_hbar_ccsd(T, H)

    calculation = Calculation(
        calculation_type="left_ccsd",
        convergence_tolerance=1.0e-08
    )

    L, _, _ = lcc_driver(calculation, system, T, Hbar)

    Ecct3, deltat3 = calc_cct3(T, L, Hbar, H, system)

    #Ecct3, deltat3 = calc_ccp3(T, L, Hbar, H, system, pspace=pspace)