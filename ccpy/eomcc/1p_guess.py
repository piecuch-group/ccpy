import numpy as np

from ccpy.utilities.printing import print_amplitudes
from ccpy.eomcc.s2matrix import build_s2matrix_cis
from ccpy.models.operators import ClusterOperator, FockOperator

def get_initial_guess(calculation, system, H, nroot, noact=0, nuact=0, guess_order=1, verbose=False):

    calc_type = "ee"
    if "ip" in calculation.calculation_type.lower():
        calc_type = "ip"
    if "ea" in calculation.calculation_type.lower():
        calc_type = "ea"

    if calc_type == "ee":
        Hmat = build_cis_hamiltonian(H, system)
        omega, V = spin_adapt_guess(system, Hmat, calculation.multiplicity)
        idx = np.argsort(omega)
        omega = omega[idx]
        V = V[:, idx]

    elif calc_type == "ip":
        Hmat = build_1h_hamiltonian(H, system)
        omega, V = np.linalg.eig(Hmat)
        idx = np.flip(np.argsort(omega))
        omega = omega[idx]
        V = V[:, idx]

    elif calc_type == "ea":
        Hmat = build_1p_hamiltonian(H, system)
        omega, V = np.linalg.eig(Hmat)
        idx = np.argsort(omega)
        omega = omega[idx]
        V = V[:, idx]

    R = []
    for i in range(nroot):

        if calc_type == "ee":
            R.append(ClusterOperator(system,
                                    order=calculation.order,
                                    active_orders=calculation.active_orders,
                                    num_active=calculation.num_active)
                     )
        else:
            R.append(FockOperator(system, num_particles=calculation.num_particles, num_holes=calculation.num_holes))

        R[i].unflatten(V[:, i], order=guess_order)

        if verbose:
            print("Guess for root:", i + 1, " ω = ", omega[i])
            print_amplitudes(R[i], system, guess_order, nprint=5)

    return R, omega[:nroot]

def spin_adapt_guess(system, H, multiplicity):

    def _get_multiplicity(s2):
        s = -0.5 + np.sqrt(0.25 + s2)
        return 2.0 * s + 1.0

    ndim = H.shape[0]

    S2 = build_s2matrix_cis(system)
    eval_s2, V_s2 = np.linalg.eig(S2)
    idx_s2 = [i for i, s2 in enumerate(eval_s2) if abs(_get_multiplicity(s2) - multiplicity) < 1.0e-07]

    W = np.zeros((ndim, len(idx_s2)))
    for i in range(len(idx_s2)):
        W[:, i] = V_s2[:, idx_s2[i]]

    # Transform from determinantal basis to basis of S2 eigenfunctions
    G = np.einsum("Ku,Nv,Lu,Mv,LM->KN", W, W, W, W, H, optimize=True)

    omega, V = np.linalg.eig(G)
    omega = np.real(omega)
    V = np.real(V)

    idx = np.argsort(omega)

    return omega[idx[len(idx_s2):]], V[:, idx[len(idx_s2):]]


def build_cis_hamiltonian(H, system):

    n1a = system.noccupied_alpha * system.nunoccupied_alpha
    n1b = system.noccupied_beta * system.nunoccupied_beta

    Haa = np.zeros((n1a, n1a))
    ct1 = 0
    for a in range(system.nunoccupied_alpha):
        for i in range(system.noccupied_alpha):
            ct2 = 0
            for b in range(system.nunoccupied_alpha):
                for j in range(system.noccupied_alpha):
                    Haa[ct1, ct2] = (
                          H.a.vv[a, b] * (i == j)
                        - H.a.oo[j, i] * (a == b)
                        + H.aa.voov[a, j, i, b]
                    )
                    ct2 += 1
            ct1 += 1
    Hab = np.zeros((n1a, n1b))
    ct1 = 0
    for a in range(system.nunoccupied_alpha):
        for i in range(system.noccupied_alpha):
            ct2 = 0
            for b in range(system.nunoccupied_beta):
                for j in range(system.noccupied_beta):
                    Hab[ct1, ct2] = H.ab.voov[a, j, i, b]
                    ct2 += 1
            ct1 += 1
    Hba = np.zeros((n1b, n1a))
    ct1 = 0
    for a in range(system.nunoccupied_beta):
        for i in range(system.noccupied_beta):
            ct2 = 0
            for b in range(system.nunoccupied_alpha):
                for j in range(system.noccupied_alpha):
                    Hba[ct1, ct2] = H.ab.ovvo[j, a, b, i]
                    ct2 += 1
            ct1 += 1
    Hbb = np.zeros((n1b, n1b))
    ct1 = 0
    for a in range(system.nunoccupied_beta):
        for i in range(system.noccupied_beta):
            ct2 = 0
            for b in range(system.nunoccupied_beta):
                for j in range(system.noccupied_beta):
                    Hbb[ct1, ct2] = (
                        H.b.vv[a, b] * (i == j)
                        - H.b.oo[j, i] * (a == b)
                        + H.bb.voov[a, j, i, b]
                    )
                    ct2 += 1
            ct1 += 1
    return np.concatenate(
        (np.concatenate((Haa, Hab), axis=1), np.concatenate((Hba, Hbb), axis=1)), axis=0
    )

def build_1h_hamiltonian(H, system):
    """Build and diagonalize the Hamiltonian in the space of 1h excitations."""

    HAA = np.zeros((system.noccupied_alpha, system.noccupied_alpha))
    HAB = np.zeros((system.noccupied_alpha, system.noccupied_beta))
    HBA = np.zeros((system.noccupied_beta, system.noccupied_alpha))
    HBB = np.zeros((system.noccupied_beta, system.noccupied_beta))

    ct1 = 0
    for i in range(system.noccupied_alpha):
        ct2 = 0
        for j in range(system.noccupied_alpha):
            HAA[ct1, ct2] = H.a.oo[j, i]
            ct2 += 1
        ct1 += 1

    ct1 = 0
    for i in range(system.noccupied_beta):
        ct2 = 0
        for j in range(system.noccupied_beta):
            HBB[ct1, ct2] = H.b.oo[j, i]
            ct2 += 1
        ct1 += 1

    Hmat = np.vstack(
                      (np.hstack((HAA, HAB)),
                       np.hstack((HBA, HBB)))
                     )

    return Hmat

def build_1p_hamiltonian(H, system):
    """Build and diagonalize the Hamiltonian in the space of 1p excitations."""

    HAA = np.zeros((system.nunoccupied_alpha, system.nunoccupied_alpha))
    HAB = np.zeros((system.nunoccupied_alpha, system.nunoccupied_beta))
    HBA = np.zeros((system.nunoccupied_beta, system.nunoccupied_alpha))
    HBB = np.zeros((system.nunoccupied_beta, system.nunoccupied_beta))

    ct1 = 0
    for a in range(system.nunoccupied_alpha):
        ct2 = 0
        for b in range(system.nunoccupied_alpha):
            HAA[ct1, ct2] = H.a.vv[a, b]
            ct2 += 1
        ct1 += 1

    ct1 = 0
    for a in range(system.nunoccupied_beta):
        ct2 = 0
        for b in range(system.nunoccupied_beta):
            HBB[ct1, ct2] = H.b.vv[a, b]
            ct2 += 1
        ct1 += 1

    Hmat = np.vstack(
                      (np.hstack((HAA, HAB)),
                       np.hstack((HBA, HBB)))
                     )

    return Hmat


# def build_eomccsd_matrix(dets1A,dets1B,dets2A,dets2B,dets2C,H1A,H1B,H2A,H2B,H2C):
#
#     n1a = len(dets1A['inds'])
#     n1b = len(dets1B['inds'])
#     n2a = len(dets2A['inds'])
#     n2b = len(dets2B['inds'])
#     n2c = len(dets2C['inds'])
#
#     # 1A - 1A block
#     H1A1A = np.zeros((n1a,n1a))
#     for idet in range(n1a):
#         i,a = dets1A['inds'][idet]
#         sym1 = dets1A['sym'][idet]
#         for jdet in range(n1a):
#             j,b = dets1A['inds'][jdet]
#             sym2 = dets1A['sym'][jdet]
#             H1A1A[idet,jdet] = (i==j)*H1A['vv'][a,b] - (a==b)*H1A['oo'][j,i] + H2A['voov'][a,j,i,b]
#
#     # 1A - 1B block
#     H1A1B = np.zeros((n1a,n1b))
#     for idet in range(n1a):
#         i,a = dets1A['inds'][idet]
#         sym1 = dets1A['sym'][idet]
#         for jdet in range(n1b):
#             j,b = dets1B['inds'][jdet]
#             sym2 = dets1B['sym'][jdet]
#             H1A1B[idet,jdet] = H2B['voov'][a,j,i,b]
#
#     # 1B - 1A block
#     H1B1A = np.zeros((n1b,n1a))
#     for idet in range(n1b):
#         i,a = dets1B['inds'][idet]
#         sym1 = dets1B['sym'][idet]
#         for jdet in range(n1a):
#             j,b = dets1A['inds'][jdet]
#             sym2 = dets1A['sym'][jdet]
#             H1B1A[idet,jdet] = H2B['ovvo'][j,a,b,i]
#
#     # 1B - 1B block
#     H1B1B = np.zeros((n1b,n1b))
#     for idet in range(n1b):
#         i,a = dets1B['inds'][idet]
#         sym1 = dets1B['sym'][idet]
#         for jdet in range(n1b):
#             j,b = dets1B['inds'][jdet]
#             sym2 = dets1B['sym'][jdet]
#             H1B1B[idet,jdet] = (i==j)*H1B['vv'][a,b] - (a==b)*H1B['oo'][j,i] + H2C['voov'][a,j,i,b]
#
#     # 1A - 2A block
#     H1A2A = np.zeros((n1a,n2a))
#     for idet in range(n1a):
#         i,a = dets1A['inds'][idet]
#         sym1 = dets1A['sym'][idet]
#         for jdet in range(n2a):
#             j,k,b,c = dets2A['inds'][jdet]
#             sym2 = dets2A['sym'][jdet]
#             H1A2A[idet,jdet] = (
#                                (i==k)*(a==c)*H1A['ov'][j,b]
#                               +(i==j)*(a==b)*H1A['ov'][k,c]
#                               -(i==j)*(a==c)*H1A['ov'][k,b]
#                               -(i==k)*(a==b)*H1A['ov'][j,c]
#                               -(a==b)*H2A['ooov'][j,k,i,c]
#                               -(a==c)*H2A['ooov'][k,j,i,b]
#                               +(i==j)*H2A['vovv'][a,k,b,c]
#                               +(i==k)*H2A['vovv'][a,j,c,b]
#             )
#
#     # 1A - 2B block
#     H1A2B = np.zeros((n1a,n2b))
#     for idet in range(n1a):
#         i,a = dets1A['inds'][idet]
#         sym1 = dets1A['sym'][idet]
#         for jdet in range(n2b):
#             j,k,b,c = dets2B['inds'][jdet]
#             sym2 = dets2B['sym'][jdet]
#             H1A2B[idet,jdet] = (
#                                (i==j)*H2B['vovv'][a,k,b,c]
#                               -(a==b)*H2B['ooov'][j,k,i,c]
#                               +(i==j)*(a==b)*H1B['ov'][k,c]
#             )
#
#     # 1B - 2B block
#     H1B2B = np.zeros((n1b,n2b))
#     for idet in range(n1b):
#         i,a = dets1B['inds'][idet]
#         sym1 = dets1B['sym'][idet]
#         for jdet in range(n2b):
#             j,k,b,c = dets2B['inds'][jdet]
#             sym2 = dets2B['sym'][jdet]
#             H1B2B[idet,jdet] = (
#                                (i==k)*H2B['ovvv'][j,a,b,c]
#                               -(a==c)*H2B['oovo'][j,k,b,i]
#                               +(i==k)*(a==c)*H1A['ov'][j,b]
#             )
#
#     # 1B - 2C block
#     H1B2C = np.zeros((n1b,n2c))
#     for idet in range(n1b):
#         i,a = dets1B['inds'][idet]
#         sym1 = dets1B['sym'][idet]
#         for jdet in range(n2c):
#             j,k,b,c = dets2C['inds'][jdet]
#             sym2 = dets2C['sym'][jdet]
#             H1B2C[idet,jdet] = (
#                                (i==k)*(a==c)*H1B['ov'][j,b]
#                               +(i==j)*(a==b)*H1B['ov'][k,c]
#                               -(i==j)*(a==c)*H1B['ov'][k,b]
#                               -(i==k)*(a==b)*H1B['ov'][j,c]
#                               -(a==b)*H2C['ooov'][j,k,i,c]
#                               -(a==c)*H2C['ooov'][k,j,i,b]
#                               +(i==j)*H2C['vovv'][a,k,b,c]
#                               +(i==k)*H2C['vovv'][a,j,c,b]
#             )
#
#     # 2A - 1A block
#     H2A1A = np.zeros((n2a,n1a))
#     for idet in range(n2a):
#         i,j,a,b = dets2A['inds'][idet]
#         sym1 = dets2A['sym'][idet]
#         for jdet in range(n1a):
#             k,c = dets1A['inds'][jdet]
#             sym2 = dets1A['sym'][jdet]
#             H2A1A[idet,jdet] = (
#                                (j==k)*H2A['vvov'][a,b,i,c]
#                               +(i==k)*H2A['vvov'][b,a,j,c]
#                               -(b==c)*H2A['vooo'][a,k,i,j]
#                               -(a==c)*H2A['vooo'][b,k,j,i]
#             )
#
#     # 2B - 1A block
#     H2B1A = np.zeros((n2b,n1a))
#     for idet in range(n2b):
#         i,j,a,b = dets2B['inds'][idet]
#         sym1 = dets2B['sym'][idet]
#         for jdet in range(n1a):
#             k,c = dets1A['inds'][jdet]
#             sym2 = dets1A['sym'][jdet]
#             H2B1A[idet,jdet] = (
#                                (i==k)*H2B['vvvo'][a,b,c,j]
#                               -(a==c)*H2B['ovoo'][k,b,i,j]
#             )
#
#     # 2B - 1B block
#     H2B1B = np.zeros((n2b,n1b))
#     for idet in range(n2b):
#         i,j,a,b = dets2B['inds'][idet]
#         sym1 = dets2B['sym'][idet]
#         for jdet in range(n1b):
#             k,c = dets1B['inds'][jdet]
#             sym2 = dets1B['sym'][jdet]
#             H2B1B[idet,jdet] = (
#                                (j==k)*H2B['vvov'][a,b,i,c]
#                               -(b==c)*H2B['vooo'][a,k,i,j]
#             )
#
#     # 2C - 1B block
#     H2C1B = np.zeros((n2c,n1b))
#     for idet in range(n2c):
#         i,j,a,b = dets2C['inds'][idet]
#         sym1 = dets2C['sym'][idet]
#         for jdet in range(n1b):
#             k,c = dets1B['inds'][jdet]
#             sym2 = dets1B['sym'][jdet]
#             H2C1B[idet,jdet] = (
#                                (j==k)*H2C['vvov'][a,b,i,c]
#                               +(i==k)*H2C['vvov'][b,a,j,c]
#                               -(b==c)*H2C['vooo'][a,k,i,j]
#                               -(a==c)*H2C['vooo'][b,k,j,i]
#             )
#
#     # 2A - 2A block
#     H2A2A = np.zeros((n2a,n2a))
#     for idet in range(n2a):
#         i,j,a,b = dets2A['inds'][idet]
#         sym1 = dets2A['sym'][idet]
#         for jdet in range(n2a):
#             k,l,c,d = dets2A['inds'][jdet]
#             sym2 = dets2A['sym'][jdet]
#             H2A2A[idet,jdet] =(
#                              (a==c)*(b==d)*(
#                             -(j==l)*H1A['oo'][k,i]
#                             +(i==l)*H1A['oo'][k,j]
#                             +(j==k)*H1A['oo'][l,i]
#                             -(i==k)*H1A['oo'][l,j])
#                             +(j==l)*(i==k)*(
#                             +(b==d)*H1A['vv'][a,c]
#                             -(b==c)*H1A['vv'][a,d]
#                             +(a==c)*H1A['vv'][b,d]
#                             -(a==d)*H1A['vv'][b,c])
#                             +(i==k)*(a==c)*H2A['voov'][b,l,j,d]
#                             -(i==k)*(a==d)*H2A['voov'][b,l,j,c]
#                             -(i==k)*(b==c)*H2A['voov'][a,l,j,d]
#                             +(i==k)*(b==d)*H2A['voov'][a,l,j,c]
#                             -(i==l)*(a==c)*H2A['voov'][b,k,j,d]
#                             +(i==l)*(a==d)*H2A['voov'][b,k,j,c]
#                             +(i==l)*(b==c)*H2A['voov'][a,k,j,d]
#                             -(i==l)*(b==d)*H2A['voov'][a,k,j,c]
#                             -(j==k)*(a==c)*H2A['voov'][b,l,i,d]
#                             +(j==k)*(a==d)*H2A['voov'][b,l,i,c]
#                             +(j==k)*(b==c)*H2A['voov'][a,l,i,d]
#                             -(j==k)*(b==d)*H2A['voov'][a,l,i,c]
#                             +(j==l)*(a==c)*H2A['voov'][b,k,i,d]
#                             -(j==l)*(a==d)*H2A['voov'][b,k,i,c]
#                             -(j==l)*(b==c)*H2A['voov'][a,k,i,d]
#                             +(j==l)*(b==d)*H2A['voov'][a,k,i,c]
#                             +(b==d)*(a==c)*H2A['oooo'][k,l,i,j]
#                             +(i==k)*(j==l)*H2A['vvvv'][a,b,c,d]
#             )
#
#     # 2A - 2B block
#     H2A2B = np.zeros((n2a,n2b))
#     for idet in range(n2a):
#         i,j,a,b = dets2A['inds'][idet]
#         sym1 = dets2A['sym'][idet]
#         for jdet in range(n2b):
#             k,l,c,d = dets2B['inds'][jdet]
#             sym2 = dets2B['sym'][jdet]
#             H2A2B[idet,jdet] = (i==k)*(a==c)*H2B['voov'][b,l,j,d]\
#                               -(i==k)*(b==c)*H2B['voov'][a,l,j,d]\
#                               -(j==k)*(a==c)*H2B['voov'][b,l,i,d]\
#                               +(j==k)*(b==c)*H2B['voov'][a,l,i,d]
#
#     # 2B - 2A block
#     H2B2A = np.zeros((n2b,n2a))
#     for idet in range(n2b):
#         i,j,a,b = dets2B['inds'][idet]
#         sym1 = dets2B['sym'][idet]
#         for jdet in range(n2a):
#             k,l,c,d = dets2A['inds'][jdet]
#             sym2 = dets2A['sym'][jdet]
#             H2B2A[idet,jdet] = (i==k)*(a==c)*H2B['ovvo'][l,b,d,j]
#                               -(i==k)*(a==d)*H2B['ovvo'][l,b,c,j]
#                               -(i==l)*(a==c)*H2B['ovvo'][k,b,d,j]
#                               +(i==l)*(a==d)*H2B['ovvo'][k,b,c,j]
#
#     # 2B - 2B block
#     H2B2B = np.zeros((n2b,n2b))
#     for idet in range(n2b):
#         i,j,a,b = dets2B['inds'][idet]
#         sym1 = dets2B['sym'][idet]
#         for jdet in range(n2b):
#             k,l,c,d = dets2B['inds'][jdet]
#             sym2 = dets2B['sym'][jdet]
#             H2B2B[idet,jdet] = (
#                                (j==l)*(b==d)*H2A['voov'][a,k,i,c]
#                               +(j==l)*(i==k)*H2B['vvvv'][a,b,c,d]
#                               +(a==c)*(i==k)*H2C['voov'][b,l,j,d]
#                               +(a==c)*(b==d)*H2B['oooo'][k,l,i,j]
#                               -(j==l)*(a==c)*H2B['ovov'][k,b,i,d]
#                               -(i==k)*(b==d)*H2B['vovo'][a,l,c,j]
#                               -(j==l)*(a==c)*(b==d)*H1A['oo'][k,i]
#                               -(a==c)*(b==d)*(i==k)*H1B['oo'][l,j]
#                               +(i==k)*(b==d)*(j==l)*H1A['vv'][a,c]
#                               +(j==l)*(i==k)*(a==c)*H1B['vv'][b,d]
#             )
#
#     # 2B - 2C block
#     H2B2C = np.zeros((n2b,n2c))
#     for idet in range(n2b):
#         i,j,a,b = dets2B['inds'][idet]
#         sym1 = dets2B['sym'][idet]
#         for jdet in range(n2c):
#             k,l,c,d = dets2C['inds'][jdet]
#             sym2 = dets2C['sym'][jdet]
#             H2B2C[idet,jdet] = (
#                                (j==k)*(b==c)*H2B['voov'][a,l,i,d]
#                               -(j==k)*(b==d)*H2B['voov'][a,l,i,c]
#                               -(j==l)*(b==c)*H2B['voov'][a,k,i,d]
#                               +(j==l)*(b==d)*H2B['voov'][a,k,i,c]
#             )
#
#     # 2C - 2B block
#     H2C2B = np.zeros((n2c,n2b))
#     for idet in range(n2c):
#         i,j,a,b = dets2C['inds'][idet]
#         sym1 = dets2C['sym'][idet]
#         for jdet in range(n2b):
#             k,l,c,d = dets2B['inds'][jdet]
#             sym2 = dets2B['sym'][jdet]
#             H2C2B[idet,jdet] = (
#                                (i==l)*(a==d)*H2B['ovvo'][k,b,c,j]
#                               -(i==l)*(b==d)*H2B['ovvo'][k,a,c,j]
#                               -(j==l)*(a==d)*H2B['ovvo'][k,b,c,i]
#                               +(j==l)*(b==d)*H2B['ovvo'][k,a,c,i]
#             )
#
#     # 2C - 2C block
#     H2C2C = np.zeros((n2c,n2c))
#     for idet in range(n2c):
#         i,j,a,b = dets2C['inds'][idet]
#         sym1 = dets2C['sym'][idet]
#         for jdet in range(n2c):
#             k,l,c,d = dets2C['inds'][jdet]
#             sym2 = dets2C['sym'][jdet]
#             H2C2C[idet,jdet] = (a==c)*(b==d)*(\
#                             -(j==l)*H1B['oo'][k,i]\
#                             +(i==l)*H1B['oo'][k,j]\
#                             +(j==k)*H1B['oo'][l,i]\
#                             -(i==k)*H1B['oo'][l,j])\
#                                +(j==l)*(i==k)*(\
#                             +(b==d)*H1B['vv'][a,c]\
#                             -(b==c)*H1B['vv'][a,d]\
#                             +(a==c)*H1B['vv'][b,d]\
#                             -(a==d)*H1B['vv'][b,c])\
#                             +(i==k)*(a==c)*H2C['voov'][b,l,j,d]\
#                             -(i==k)*(a==d)*H2C['voov'][b,l,j,c]\
#                             -(i==k)*(b==c)*H2C['voov'][a,l,j,d]\
#                             +(i==k)*(b==d)*H2C['voov'][a,l,j,c]\
#                             -(i==l)*(a==c)*H2C['voov'][b,k,j,d]\
#                             +(i==l)*(a==d)*H2C['voov'][b,k,j,c]\
#                             +(i==l)*(b==c)*H2C['voov'][a,k,j,d]\
#                             -(i==l)*(b==d)*H2C['voov'][a,k,j,c]\
#                             -(j==k)*(a==c)*H2C['voov'][b,l,i,d]\
#                             +(j==k)*(a==d)*H2C['voov'][b,l,i,c]\
#                             +(j==k)*(b==c)*H2C['voov'][a,l,i,d]\
#                             -(j==k)*(b==d)*H2C['voov'][a,l,i,c]\
#                             +(j==l)*(a==c)*H2C['voov'][b,k,i,d]\
#                             -(j==l)*(a==d)*H2C['voov'][b,k,i,c]\
#                             -(j==l)*(b==c)*H2C['voov'][a,k,i,d]\
#                             +(j==l)*(b==d)*H2C['voov'][a,k,i,c]\
#                             +(b==d)*(a==c)*H2C['oooo'][k,l,i,j]\
#                             +(i==k)*(j==l)*H2C['vvvv'][a,b,c,d]
#
#     # Assemble full matrix
#     h1a = np.concatenate( (H1A1A, H1A1B, H1A2A, H1A2B, np.zeros((n1a,n2c))), axis=1)
#     h1b = np.concatenate( (H1B1A, H1B1B, np.zeros((n1b, n2a)), H1B2B, H1B2C), axis=1)
#     h2a = np.concatenate( (H2A1A, np.zeros((n2a,n1b)), H2A2A, H2A2B, np.zeros((n2a,n2c))), axis=1)
#     h2b = np.concatenate( (H2B1A, H2B1B, H2B2A, H2B2B, H2B2C), axis=1)
#     h2c = np.concatenate( (np.zeros((n2c,n1a)), H2C1B, np.zeros((n2c,n2a)), H2C2B, H2C2C), axis=1)
#     Hmat = np.concatenate( (h1a,h1b,h2a,h2b,h2c), axis=0)
#
#     return Hmat
