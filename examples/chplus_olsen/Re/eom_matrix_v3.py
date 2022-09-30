import numpy as np

from ccpy.models.calculation import Calculation
from ccpy.interfaces.gamess_tools import load_from_gamess
from ccpy.drivers.driver import cc_driver

from ccpy.hbar.hbar_ccsd import build_hbar_ccsd

if __name__ == "__main__":


    system, H0 = load_from_gamess(
            "chplus_re.log",
            "onebody.inp",
            "twobody.inp",
            nfrozen=0,
    )

    calculation = Calculation(
        order=2,
        calculation_type="ccsd",
        convergence_tolerance=1.0e-08,
        RHF_symmetry=False,
    )

    T, total_energy, _ = cc_driver(calculation, system, H0)

    H = build_hbar_ccsd(T, H0)

    nact_occ_beta = 2
    nact_unocc_alpha = 2
    nact_occ_alpha = (system.multiplicity - 1) + nact_occ_beta
    nact_unocc_beta = (system.multiplicity - 1) + nact_unocc_alpha

    # slicing vectors
    occ_a = slice(0, system.noccupied_alpha)
    occ_b = slice(0, system.noccupied_beta)
    unocc_a = slice(system.noccupied_alpha, system.norbitals)
    unocc_b = slice(system.noccupied_beta, system.norbitals)

    occ_inact_a = slice(0, system.noccupied_alpha - nact_occ_alpha)
    occ_act_a = slice(system.noccupied_alpha - nact_occ_alpha, system.noccupied_alpha)
    occ_inact_b = slice(0, system.noccupied_beta - nact_occ_beta)
    occ_act_b = slice(system.noccupied_beta - nact_occ_beta, system.noccupied_beta)
    unocc_act_a = slice(0, nact_unocc_alpha)
    unocc_inact_a = slice(nact_unocc_alpha, system.norbitals)
    unocc_act_b = slice(0, nact_unocc_beta)
    unocc_inact_b = slice(nact_unocc_beta, system.norbitals)

    # orbital dimensions
    n1a = system.noccupied_alpha * system.nunoccupied_alpha
    n1b = system.noccupied_beta * system.nunoccupied_beta
    n2a = system.noccupied_alpha**2 * system.nunoccupied_alpha**2
    N2A = nact_occ_alpha**2 * nact_unocc_alpha**2
    n2b = system.noccupied_alpha * system.noccupied_beta * system.nunoccupied_alpha * system.nunoccupied_beta
    N2B = nact_occ_alpha * nact_occ_beta * nact_unocc_alpha * nact_unocc_beta
    n2c = system.noccupied_beta**2 * system.nunoccupied_beta**2
    N2C = nact_occ_beta**2 * nact_unocc_beta**2

    # orbital identity matrix
    Iorb = np.eye(system.norbitals)

    # < ia | H | jb >
    H1A1A = -1.0 * np.einsum("ab,ji->iajb", Iorb[unocc_a, unocc_a], H.a.oo, optimize=True)
    H1A1A += np.einsum("ij,ab->iajb", Iorb[occ_a, occ_a], H.a.vv, optimize=True)
    H1A1A += np.einsum("ajib->iajb", H.aa.voov, optimize=True)
    # < ia | H | j~b~ >
    H1A1B = np.einsum("ajib->iajb", H.ab.voov, optimize=True)
    # < i~a~ | H | jb >
    H1B1A = np.einsum("jabi->iajb", H.ab.ovvo, optimize=True)
    # < i~a~ | H | j~b~ >
    H1B1B = -1.0 * np.einsum("ab,ji->iajb", Iorb[unocc_b, unocc_b], H.b.oo, optimize=True)
    H1B1B += np.einsum("ij,ab->iajb", Iorb[occ_b, occ_b], H.b.vv, optimize=True)
    H1B1B += np.einsum("ajib->iajb", H.bb.voov, optimize=True)

    # # < IJAB | H | kc >
    # H2A1A = -1.0 * np.einsum("Ac,BkJI->IJABkc", Iorb[unocc_act_a, unocc_a], H.aa.vooo, optimize=True)
    # H2A1A += np.einsum("Bc,AkJI->IJABkc", Iorb[unocc_act_a, unocc_a], H.aa.vooo, optimize=True)
    # H2A1A += np.einsum("Ik,A")


    # H_SS = np.vstack(( np.hstack((H1A1A.reshape((n1a, n1a)), H1A1B.reshape((n1a, n1b)))),
    #                    np.hstack((H1B1A.reshape((n1b, n1a)), H1B1B.reshape((n1b, n1b)))),
    # ))
    #
    # E, V = np.linalg.eig(H_SS)
    # idx = np.argsort(E)
    # E = E[idx]
    # for i in range(len(E)):
    #     print("Eigval ", i + 1, " = ", E[i])