from pyscf import gto, scf

from ccpy.models.calculation import Calculation
from ccpy.interfaces.pyscf_tools import load_pyscf_integrals
from ccpy.drivers.driver import cc_driver, lcc_driver

from ccpy.hbar.hbar_ccsd import build_hbar_ccsd

from ccpy.moments.cct3 import calc_cct3


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

    system.set_active_space(nact_occupied=5, nact_unoccupied=9)
    calculation = Calculation(
        order=3,
        active_orders=[3],
        num_active=[1],
        calculation_type="ccsdt1",
        convergence_tolerance=1.0e-08
    )

    T, total_energy, is_converged = cc_driver(calculation, system, H)

    calculation = Calculation(
       order=2,
       calculation_type="left_ccsd",
       convergence_tolerance=1.0e-08
    )

    Hbar = build_hbar_ccsd(T, H)

    L, total_energy, is_converged = lcc_driver(calculation, system, T, Hbar, omega=0.0, L=None, R=None)

    Ecct3, delta23 = calc_cct3(T, L, Hbar, H, system, use_RHF=False)
