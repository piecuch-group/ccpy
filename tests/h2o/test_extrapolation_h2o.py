"""CR-CC(2,4) calculation for the symmetrically stretched
H2O molecule with R(OH) = 2Re, where Re = 1.84345 bohr, described
using the Dunning DZ basis set.
Reference: Mol. Phys, 115, 2860 (2017)."""

from pyscf import scf, gto
from ccpy import Driver
from ccpy.extrapolation.goodson_extrapolation import (rational_pade_approximant,
                                                      quadratic_pade_approximant,
                                                      continued_fraction_approximant)

def test_extrapolation_h2o():
    geometry = [["O", (0.0, 0.0, -0.0180)],
                ["H", (0.0, 3.030526, -2.117796)],
                ["H", (0.0, -3.030526, -2.117796)]]
    #geometry = [["O", (0.0, 0.0, -0.0270)],
    #            ["H", (0.0, 4.545789, -3.176694)],
    #            ["H", (0.0, -4.545789, -3.176694)]]
    mol = gto.M(
        atom=geometry,
        basis="cc-pvdz",
        charge=0,
        spin=0,
        symmetry="C2V",
        cart=False,
        unit="Bohr",
    )
    mf = scf.RHF(mol)
    mf.kernel()

    driver = Driver.from_pyscf(mf, nfrozen=0)
    
    # reference energy
    e_ref = driver.system.reference_energy
    # CCSD energy
    driver.run_cc(method="ccsd")
    e_ccsd = driver.correlation_energy + e_ref
    driver.T = None
    driver.correlation_energy = 0.0
    # CCSDT energy
    driver.run_cc(method="ccsdt")
    e_ccsdt = driver.correlation_energy + e_ref


    delta1 = e_ref
    delta2 = e_ccsd - e_ref
    delta3 = e_ccsdt - e_ccsd

    ex_ccq = quadratic_pade_approximant(delta1, delta2, delta3)
    ex_ccr = rational_pade_approximant(delta1, delta2, delta3)
    ex_cccf = continued_fraction_approximant(delta1, delta2, delta3)

    print("ex-CCq = ", ex_ccq)
    print("ex-CCr = ", ex_ccr)
    print("ex-CCcf = ", ex_cccf)


if __name__ == "__main__":
    test_extrapolation_h2o()
