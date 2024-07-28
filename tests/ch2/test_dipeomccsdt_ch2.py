import numpy as np
from pyscf import gto, scf
from ccpy.drivers.driver import Driver

def test_dipeomccsdt_ch2():

    mol = gto.M(atom=[["C", (0.0, 0.0, 0.0)],
                      ["H", (0.0, 1.644403, -1.32213)],
                      ["H", (0.0, -1.644403, -1.32213)]],
                basis="6-31g",
                charge=-2,
                symmetry="C2V",
                cart=False,
                spin=0,
                unit="Bohr")
    mf = scf.RHF(mol)
    mf.kernel()

    driver = Driver.from_pyscf(mf, nfrozen=0)
    driver.system.print_info()

    driver.run_cc(method="ccsdt")
    driver.run_hbar(method="ccsdt")
    driver.run_guess(method="dipcis", multiplicity=-1, nact_occupied=driver.system.noccupied_alpha, roots_per_irrep={"A1": 6}, use_symmetry=False)
    driver.run_dipeomcc(method="dipeomccsdt", state_index=[0, 1])
    #driver.run_dipeomcc(method="dipeom3", state_index=[1])

    # H = driver.hamiltonian
    # R = driver.R[1]
    #
    # # Create dictionary to store intermediates, which have spincases that resemble those of the DIP R operator itself
    # X = {"ab": {"vo": np.array([0.0]), "ov": np.array([0.0]), "vv": np.array([0.0])},
    #      "aba": {"oooo": np.array([0.0]), "oovv": np.array([0.0]), "ovvo": np.array([0.0]), "oovo": np.array([0.0]), "ovoo": np.array([0.0])},
    #      "abb": {"oooo": np.array([0.0]), "oovv": np.array([0.0]), "vovo": np.array([0.0]), "oovo": np.array([0.0]), "vooo": np.array([0.0])}}
    #
    # # x(ij~em) [7]
    # X["aba"]["oovo"] = (
    #     -np.einsum("mnej,in->ijem", H.ab.oovo, R.ab, optimize=True)
    #     -np.einsum("nmie,nj->ijem", H.aa.ooov, R.ab, optimize=True)
    #     +np.einsum("mnef,ijfn->ijem", H.aa.oovv, R.aba, optimize=True)
    #     +np.einsum("mnef,ijfn->ijem", H.ab.oovv, R.abb, optimize=True)
    # )
    #
    # # x(ij~e~m~) [8]
    # X["abb"]["oovo"] = (
    #     -np.einsum("nmje,in->ijem", H.bb.ooov, R.ab, optimize=True)
    #     -np.einsum("nmie,nj->ijem", H.ab.ooov, R.ab, optimize=True)
    #     +np.einsum("nmfe,ijfn->ijem", H.ab.oovv, R.aba, optimize=True)
    #     +np.einsum("mnef,ijfn->ijem", H.bb.oovv, R.abb, optimize=True)
    # )
    #
    # # x(ie~mk) [9]; i ->, e~ -> (j~), k -> m
    # X["aba"]["ovoo"] = (
    #     -0.5 * np.einsum("mnfe,infk->iemk", H.ab.oovv, R.aba, optimize=True)
    #     -np.einsum("mnie,kn->iemk", H.ab.ooov, R.ab, optimize=True)
    # )
    # # antisymmetrize (ik)
    # X["aba"]["ovoo"] -= np.transpose(X["aba"]["ovoo"], (3, 1, 2, 0))
    #
    # # x(ej~m~k~) [10]; j~ ->, e -> (i), k~ -> m~
    # X["abb"]["vooo"] = (
    #     -0.5 * np.einsum("nmef,njfk->ejmk", H.ab.oovv, R.abb, optimize=True)
    #     -np.einsum("nmek,nj->ejmk", H.ab.oovo, R.ab, optimize=True)
    # )
    # # antisymmetrize A(j~k~)
    # X["abb"]["vooo"] -= np.transpose(X["abb"]["vooo"], (0, 3, 2, 1))
    #
    # nua, nub, noa, nob = driver.T.ab.shape
    #
    # fact = 0.5829864646726097 / 0.867576
    # for i in range(noa):
    #     for j in range(nob):
    #         for e in range(nua):
    #             for m in range(noa):
    #                 if abs(X["aba"]["oovo"][i, j, e, m] * fact) > 1.0e-03:
    #                     print(i, j, e, m, X["aba"]["oovo"][i, j, e, m] * fact)

    #
    # Check the results
    #
    expected_vee = [-0.467288309238, -0.446416466198]
    assert np.allclose(driver.correlation_energy, -0.105856560769, atol=1.0e-07)
    for i in range(len(expected_vee)):
       assert np.allclose(expected_vee[i], driver.vertical_excitation_energy[i], atol=1.0e-07)

if __name__ == "__main__":
    test_dipeomccsdt_ch2()
