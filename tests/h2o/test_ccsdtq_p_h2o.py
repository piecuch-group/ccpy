"""CCSDTQ(P) with all triples and quadruples in the P space reproduces CCSDTQ, for the
symmetrically stretched H2O molecule with R(OH) = 2Re (Re = 1.84345 bohr) described using
the 6-31G basis set with the oxygen 1s orbital frozen."""

import itertools

import numpy as np
from pyscf import scf, gto
from ccpy import Driver


def full_excitations(system, spincases):
    """All excitations [a, b, ..., i, j, ...] (1-based) with ascending indices within each spin block."""
    nv = {"a": system.nunoccupied_alpha, "b": system.nunoccupied_beta}
    no = {"a": system.noccupied_alpha, "b": system.noccupied_beta}
    excitations = {}
    for spincase in spincases:
        def block(n):
            out = [()]
            for s in "ab":
                k = spincase.count(s)
                if k:
                    out = [x + y for x in out for y in itertools.combinations(range(1, n[s] + 1), k)]
            return out
        excitations[spincase] = np.asarray([v + o for v in block(nv) for o in block(no)], order="F")
    return excitations


def _driver():
    geometry = [["O", (0.0, 0.0, -0.0180)],
                ["H", (0.0, 3.030526, -2.117796)],
                ["H", (0.0, -3.030526, -2.117796)]]
    mol = gto.M(atom=geometry, basis="6-31g", charge=0, spin=0, symmetry="C2V", cart=False, unit="Bohr")
    mf = scf.RHF(mol)
    mf.kernel()
    driver = Driver.from_pyscf(mf, nfrozen=1)
    driver.options["energy_convergence"] = 1.0e-09
    driver.options["amp_convergence"] = 1.0e-07
    driver.options["maximum_iterations"] = 100
    return driver


def test_ccsdtq_p_h2o():
    driver = _driver()
    driver.run_cc(method="ccsdtq")
    e_ccsdtq = driver.correlation_energy

    for rhf in (False, True):
        driver = _driver()
        driver.options["RHF_symmetry"] = rhf
        t3_excitations = full_excitations(driver.system, ["aaa", "aab", "abb", "bbb"])
        t4_excitations = full_excitations(driver.system, ["aaaa", "aaab", "aabb", "abbb", "bbbb"])
        driver.run_ccp(method="ccsdtq_p", t3_excitations=t3_excitations, t4_excitations=t4_excitations)
        assert np.allclose(driver.correlation_energy, e_ccsdtq, atol=1.0e-08)


if __name__ == "__main__":
    test_ccsdtq_p_h2o()
