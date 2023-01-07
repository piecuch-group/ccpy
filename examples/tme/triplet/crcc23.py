from pyscf import gto, scf

from ccpy.models.calculation import Calculation
from ccpy.interfaces.pyscf_tools import load_pyscf_integrals
from ccpy.drivers.driver import cc_driver, lcc_driver

from ccpy.hbar.hbar_ccsd import build_hbar_ccsd

from ccpy.moments.crcc23 import calc_crcc23
from ccpy.moments.cct3 import calc_cct3


if __name__ == "__main__":

    mol = gto.Mole()

    # triplet geometry at 45 degree torsion (biradical state)
    mol.build(
        atom="""
            C  0.000000000  0.000000000 -1.419530145
            C  0.000000000  0.000000000  1.419530145
            C  0.874825472  2.112015519 -2.715168977
            C -0.874825472  2.112015519  2.715168977
            C -0.874825472 -2.112015519 -2.715168977
            C  0.874825472 -2.112015519  2.715168977
            H  0.875059361  2.151729681 -4.741412815
            H -0.875059361 -2.151729681 -4.741412815
            H  0.875059361 -2.151729681  4.741412815
            H -0.875059361  2.151729681  4.741412815
            H  1.618623554  3.724058890 -1.744607378
            H -1.618623554 -3.724058890 -1.744607378
            H  1.618623554 -3.724058890  1.744607378
            H -1.618623554  3.724058890  1.744607378
            """,
        basis="ccpvdz",
        charge=0,
        spin=2,
        symmetry="D2",
        cart=False,
        unit="Bohr",
    )
    mf = scf.RHF(mol)
    mf.kernel()

    system, H = load_pyscf_integrals(mf, 6)
    system.print_info()

    #system.set_active_space(6, 6)
    calculation = Calculation(
        calculation_type="ccsd",
        maximum_iterations=200,
        RHF_symmetry=False,
    )
    T, total_energy, is_converged = cc_driver(calculation, system, H)

    calculation = Calculation(
        calculation_type="left_ccsd",
        maximum_iterations=200,
        RHF_symmetry=False,
    )

    Hbar = build_hbar_ccsd(T, H)

    L, total_energy, is_converged = lcc_driver(calculation, system, T, Hbar, omega=0.0, L=None, R=None)

    Ecrcc23, delta23 = calc_crcc23(T, L, Hbar, H, system, use_RHF=False)
    # Ecct3, delta23 = calc_cct3(T, L, Hbar, H, system, use_RHF=False)
