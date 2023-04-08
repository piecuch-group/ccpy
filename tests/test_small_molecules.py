import numpy as np
from pyscf import scf, gto
from ccpy.drivers.driver import Driver

def test_cct3_f2():
    """
    F2 / cc-pVDZ at R = 2.0Re, where Re = 2.66816 bohr using RHF.
    Cartesian orbitals are used for the d orbitals in the cc-pVDZ basis.
    Reference: Chem. Phys. Lett. 344, 165 (2001).
    """
    geometry = [["F", (0.0, 0.0, -2.66816)],
                ["F", (0.0, 0.0,  2.66816)]]
    mol = gto.M(atom=geometry,
                basis="cc-pvdz",
                charge=0,
                spin=0,
                symmetry="D2H",
                cart=True,
                unit="Bohr",
    )
    mf = scf.RHF(mol)
    mf.kernel()

    driver = Driver.from_pyscf(mf, nfrozen=2)
    driver.system.set_active_space(nact_occupied=5, nact_unoccupied=1)
    driver.run_cc(method="ccsdt1")
    driver.run_hbar(method="ccsd")
    driver.run_leftcc(method="left_ccsd")
    driver.run_ccp3(method="cct3", state_index=[0])

    # Check reference energy
    assert(np.allclose(driver.system.reference_energy, -198.4200962814))
    # Check CCSDt energy
    assert(np.allclose(driver.correlation_energy, -0.6363154135))
    assert(np.allclose(driver.system.reference_energy + driver.correlation_energy, -199.0564116949))
    # Check CC(t;3)_A energy
    assert(np.allclose(driver.correlation_energy + driver.deltapq[0]["A"], -0.6376818524))
    assert(np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.deltapq[0]["A"], -199.0577781338))
    # Check CC(t;3)_D energy
    assert(np.allclose(driver.correlation_energy + driver.deltapq[0]["D"], -0.6378384699))
    assert(np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.deltapq[0]["D"], -199.0579347513))

def test_cct3_hfhminus_triplet():
    """
    (HFH)- / 6-31g(1d,1p) open-shell triplet in the symmetric D2H geometry
    with R_{HF} = 2.0 angstrom using ROHF and MO integrals from GAMESS.
    Reference: J. Chem. Theory Comput. 8, 4968 (2012)
    """
    driver = Driver.from_gamess(logfile="data/hfhminus-triplet/hfhminus-triplet.log",
                                fcidump="data/hfhminus-triplet/hfhminus-triplet.FCIDUMP",
                                nfrozen=1)
    driver.system.set_active_space(nact_unoccupied=1, nact_occupied=1)
    driver.system.print_info()

    driver.run_cc(method="ccsdt1")
    driver.run_hbar(method="ccsd")
    driver.run_leftcc(method="left_ccsd")
    driver.run_ccp3(method="cct3", state_index=[0])

    # Check reference energy
    assert(np.allclose(driver.system.reference_energy, -100.3591573557))
    # Check CCSDt energy
    assert(np.allclose(driver.correlation_energy, -0.1925359236))
    assert(np.allclose(driver.system.reference_energy + driver.correlation_energy, -100.5516932793))
    # Check CC(t;3)_A energy
    assert(np.allclose(driver.correlation_energy + driver.deltapq[0]["A"], -0.1936455544))
    assert(np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.deltapq[0]["A"], -100.5528029101))
    # Check CC(t;3)_D energy
    assert(np.allclose(driver.correlation_energy + driver.deltapq[0]["D"], -0.1938719549))
    assert(np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.deltapq[0]["D"], -100.5530293106))

def test_crcc23_glycine():
    """
    """
    geometry = [["O", (-2.877091949897, -1.507375565672, -0.398996049903)],
                ["C", (-0.999392972049, -0.222326510867,  0.093940021615)],
                ["C", ( 1.633098051399, -1.126399113321,  0.723677865007)],
                ["O", (-1.316707936360,  2.330484008081,  0.195537896270)],
                ["N", ( 3.588772131647,  0.190046035276, -0.635572324857)],
                ["H", ( 1.738434758147, -3.192291478262,  0.201142047999)],
                ["H", ( 1.805107822402, -0.972547254301,  2.850386782716)],
                ["H", ( 3.367427816470,  2.065392438845, -0.521139962778)],
                ["H", ( 5.288732713155, -0.301105855518,  0.028508872837)],
                ["H", (-3.050135067115,  2.755707159769, -0.234244183166)]]
                
    mol = gto.M(atom=geometry,
                basis="6-31g**",
                charge=0,
                spin=0,
                symmetry="C1",
                unit="Bohr",
                cart=True,
    )
    mf = scf.RHF(mol)
    mf.kernel()

    driver = Driver.from_pyscf(mf, nfrozen=5)
    driver.system.print_info()

    driver.options["RHF_symmetry"] = True
    driver.run_cc(method="ccsd")
    driver.run_hbar(method="ccsd")
    driver.run_leftcc(method="left_ccsd")
    driver.run_ccp3(method="crcc23")

    # Check reference energy
    assert(np.allclose(driver.system.reference_energy, -282.8324072998))
    # Check CCSD energy
    assert(np.allclose(driver.correlation_energy, -0.8319770162))
    assert(np.allclose(driver.system.reference_energy + driver.correlation_energy, -283.6643843160))
    # Check CR-CC(2,3)_A energy
    assert(np.allclose(driver.correlation_energy + driver.deltapq[0]["A"], -0.8546012006))
    assert(np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.deltapq[0]["A"], -283.6870085004))
    # Check CR-CC(2,3)_D energy
    assert(np.allclose(driver.correlation_energy + driver.deltapq[0]["D"], -0.8574520862))
    assert(np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.deltapq[0]["D"], -283.6898593860))

def test_ccsdt_ch():
    """
    """
    driver = Driver.from_gamess(logfile="data/ch/ch.log", fcidump="data/ch/ch.FCIDUMP", nfrozen=1)
    driver.system.print_info()

    driver.run_cc(method="ccsdt")

    # Check reference energy
    assert(np.allclose(driver.system.reference_energy, -38.2713247488))
    # Check CCSDT energy
    assert(np.allclose(driver.correlation_energy, -0.1164237849))
    assert(np.allclose(driver.system.reference_energy + driver.correlation_energy, -38.3877485336))
                

def test_cct3_ch():
    """
    """
    driver = Driver.from_gamess(logfile="data/ch/ch.log", fcidump="data/ch/ch.FCIDUMP", nfrozen=1)
    driver.system.print_info()
    driver.system.set_active_space(nact_occupied=1, nact_unoccupied=1)

    print(driver.system.num_act_occupied_alpha)
    print(driver.system.num_act_occupied_beta)
    print(driver.system.num_act_unoccupied_alpha)
    print(driver.system.num_act_unoccupied_beta)

    driver.run_cc(method="ccsdt1")
    #driver.run_hbar(method="ccsd")
    #driver.run_leftcc(method="left_ccsd")
    #driver.run_ccp3(method="cct3")


    # Check reference energy
    assert(np.allclose(driver.system.reference_energy, -38.2713247488))
                
if __name__ == "__main__":
#     test_ccsdt_ch()
#     test_crcc23_glycine()
    test_cct3_hfhminus_triplet()
#     test_cct3_f2()
    test_cct3_ch()

# Methods
# - CCD
# - CCSD
# - CCSDT
# - CCSDt
# - CC(P) aimed at CCSDT
# - CCSDTQ
# - CCSD(T)
# - CR-CC(2,3)
# - CR-CC(2,4)
# - EOMCCSD
# - EOMCCSDt
# - EOMCCSDT
# - CR-EOMCC(2,3)
# - delta-CR-EOMCC(2,3)
# - left CCSD
# - left CCSDT
# - ec-CC-II
# - ec-CC-II_{3}
#
# - Adaptive CC(P;Q) aimed at CCSDT
# - ec-CC-II_{3,4}

# Molecules
# F2 / cc-pVDZ, cc-pVTZ, aug-cc-pVTZ
# H2O / DZ, cc-pVDZ, cc-pVTZ, cc-pVQZ
# CH2 / 6-31G*, cc-pVTZ, cc-pVQZ
# H4
# H8
#
