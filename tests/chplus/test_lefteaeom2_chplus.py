"""EA-EOMCCSD(2h-1p) computation used to describe the spectrum of the
open-shell CH molecule by attaching an electron to closed-shell CH+."""

import pytest
from ccpy import Driver
from pyscf import gto, scf

@pytest.mark.short
def test_eaeom2_chplus():

    mol = gto.M(
        atom='''C 0.0 0.0 0.0
                H 0.0 0.0 1.1197868''',
        basis="aug-cc-pvdz",
        unit="angstrom",
        spin=0,
        charge=1,
        symmetry="c2v",
    )
    mf = scf.RHF(mol)
    mf.kernel()

    driver = Driver.from_pyscf(mf, nfrozen=1)
    driver.system.print_info()

    driver.run_cc(method="ccsd")
    driver.run_hbar(method="ccsd")
    driver.run_guess(method="eacisd", multiplicity=-1, nact_occupied=3, nact_unoccupied=8,
                     roots_per_irrep={"B1": 1, "A1": 4, "A2": 1})
    driver.run_eaeomcc(method="eaeom2", state_index=[0, 1, 2, 3, 4, 5])
    driver.run_lefteaeomcc(method="left_eaeom2", state_index=[0, 1, 2, 3, 4, 5])
    driver.run_eaccp3(method="creacc23", state_index=[0, 1, 2, 3, 4, 5])

if __name__ == "__main__":
    test_eaeom2_chplus()
