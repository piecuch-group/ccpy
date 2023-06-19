import numpy as np

from ccpy.eom_guess.s2matrix import build_s2matrix_cisd, spin_adapt_guess
from ccpy.utilities.updates import eomcc_initial_guess

def run_diagonalization(system, H, multiplicity, nroot, nacto, nactu):

    #Hmat = build_cisd_hamiltonian(dets1A, dets1B, dets2A, dets2B, dets2C, H, system)
    ##S2mat = build_s2matrix_cisd(system)
    ##omega, V = spin_adapt_guess(S2mat, Hmat, multiplicity)

    noa, nob, nua, nub = H.ab.oovv.shape
    ndim = noa*nua + nob*nub + noa**2*nua**2 + noa*nob*nua*nub + nob**2*nub**2

    idx1A, idx1B, idx2A, idx2B, idx2C, n1a_act, n1b_act, n2a_act, n2b_act, n2c_act = \
        eomcc_initial_guess.eomcc_initial_guess.get_active_dimensions(
                                                                nacto, nactu,
                                                                system.noccupied_alpha, system.nunoccupied_alpha,
                                                                system.noccupied_beta, system.nunoccupied_beta,
        )
    ndim_act = n1a_act + n1b_act + n2a_act + n2b_act + n2c_act

    V_act, omega, Hmat = eomcc_initial_guess.eomcc_initial_guess.eomccs_d_matrix(
                                            idx1A, idx1B, idx2A, idx2B, idx2C,
                                            H.a.oo, H.a.vv, H.a.ov,
                                            H.b.oo, H.b.vv, H.b.ov,
                                            H.aa.oooo, H.aa.vvvv, H.aa.voov, H.aa.vooo, H.aa.vvov, H.aa.ooov, H.aa.vovv,
                                            H.ab.oooo, H.ab.vvvv, H.ab.voov, H.ab.ovvo, H.ab.vovo, H.ab.ovov, H.ab.vooo,
                                            H.ab.ovoo, H.ab.vvov, H.ab.vvvo, H.ab.ooov, H.ab.oovo, H.ab.vovv, H.ab.ovvv,
                                            H.bb.oooo, H.bb.vvvv, H.bb.voov, H.bb.vooo, H.bb.vvov, H.bb.ooov, H.bb.vovv,
                                            n1a_act, n1b_act, n2a_act, n2b_act, n2c_act, ndim_act)
    idx = np.argsort(omega)
    omega = omega[idx]
    V_act = V_act[:, idx]
    nroot = min(len(omega), nroot)

    omega = omega[:nroot]
    V = np.zeros((ndim, nroot))

    for i in range(nroot):
        V_a, V_b, V_aa, V_ab, V_bb = eomcc_initial_guess.eomcc_initial_guess.unflatten_guess_vector(
                                                            V_act[:, i],
                                                            idx1A, idx1B, idx2A, idx2B, idx2C,
                                                            n1a_act, n1b_act, n2a_act, n2b_act, n2c_act, ndim_act,
        )
        V[:, i] = np.hstack((V_a.flatten(), V_b.flatten(), V_aa.flatten(), V_ab.flatten(), V_bb.flatten()))

    print("Dimension of CISd problem = ", ndim_act)
    for i in range(nroot):
        print("Eigenvalue", i + 1, " = ", omega[i])

    return omega, V

