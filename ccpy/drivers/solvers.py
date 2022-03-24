"""Module containing the driving solvers"""
import time

import numpy as np

from ccpy.utilities.printing import print_iteration, print_iteration_header
from ccpy.utilities.utilities import remove_file
from ccpy.models.operators import ClusterOperator

def eomcc_davidson_lowmem(HR, update_r, R, omega, T, H, calculation, system):
    """
    Diagonalize the similarity-transformed Hamiltonian HBar using the
    non-Hermitian Davidson algorithm. This version has a much lower memory
    consumption based on storing R and HR vectors on disk and reading them
    in one-by-one as needed.
    """
    from ccpy.drivers.cc_energy import get_r0

    nroot = len(R)

    is_converged = [False] * nroot
    r0 = np.zeros(nroot)

    # orthonormalize the initial trial space; this is important when using doubles in EOMCCSd guess
    B0, _ = np.linalg.qr(np.asarray([r.flatten() for r in R]).T)

    # Allocate residual R cluster operator
    dR = ClusterOperator(system,
                         order=R[0].order,
                         active_orders=calculation.active_orders,
                         num_active=calculation.num_active,
                         data_type=R[0].a.dtype)

    # Allocate the B and sigma matrices
    B = np.memmap("B.npy", dtype=R[0].a.dtype, mode="w+", shape=(R[0].ndim, calculation.maximum_iterations))
    del B
    sigma = np.memmap("sigma.npy", dtype=R[0].a.dtype, mode="w+", shape=(R[0].ndim, calculation.maximum_iterations))
    del sigma

    for n in range(nroot):

        print("Solving for root - {}    Energy of initial guess = {}".format(n + 1, omega[n]))
        print(
            " Iter        omega                |r|               dE            Wall Time"
        )
        print(
            "--------------------------------------------------------------------------------"
        )

        # Initial values
        B = np.memmap("B.npy", dtype=R[0].a.dtype, mode="r+", shape=(R[0].ndim, calculation.maximum_iterations))
        B[:, 0] = B0[:, n]
        del B
        dR.unflatten(B0[:, 0])

        sigma = np.memmap("sigma.npy", dtype=R[0].a.dtype, mode="r+", shape=(R[0].ndim, calculation.maximum_iterations))
        sigma[:, 0] = HR(dR, T, H, calculation.RHF_symmetry, system)
        del sigma

        curr_size = 1
        while curr_size < calculation.maximum_iterations:
            t1 = time.time()

            # open the B and sigma files
            B = np.memmap("B.npy", dtype=R[0].a.dtype, mode="r", shape=(R[0].ndim, calculation.maximum_iterations))
            sigma = np.memmap("sigma.npy", dtype=R[0].a.dtype, mode="r", shape=(R[0].ndim, calculation.maximum_iterations))

            # store old energy
            omega_old = omega[n]

            # solve projection subspace eigenproblem
            G = np.zeros((curr_size, curr_size))
            for k1 in range(curr_size):
                for k2 in range(curr_size):
                    G[k1, k2] = np.dot(B[:, k1].T, sigma[:, k2])

            e, alpha = np.linalg.eig(G)

            # select root based on maximum overlap with initial guess
            idx = np.argsort(abs(alpha[0, :]))
            alpha = np.real(alpha[:, idx[-1]])
            omega[n] = np.real(e[idx[-1]])

            # Calculate the eigenvector and the residual
            r = np.zeros(R[n].ndim)
            dr = np.zeros(R[n].ndim)

            for k1 in range(curr_size):
                r += B[:, k1] * alpha[k1]
                dr += sigma[:, k1] * alpha[k1]
            dr -= omega[n] * r
            R[n].unflatten(r)
            dR.unflatten(dr)

            residual = np.linalg.norm(dR.flatten())
            deltaE = omega[n] - omega_old

            t2 = time.time()
            minutes, seconds = divmod(t2 - t1, 60)
            print(
                "   {}      {:.10f}       {:.10f}      {:.10f}      {:.2f}m {:.2f}s".format(
                    curr_size, omega[n], residual, deltaE, minutes, seconds
                )
            )
            if residual < calculation.convergence_tolerance and abs(deltaE) < calculation.convergence_tolerance:
                is_converged[n] = True
                break

            # update residual vector
            dR = update_r(dR, omega[n], H)
            q = dR.flatten()

            for k1 in range(curr_size):
                b = B[:, k1] / np.linalg.norm(B[:, k1])
                q -= np.dot(b.T, q) * b

            q *= 1.0 / np.linalg.norm(q)
            dR.unflatten(q)

            # write new vectors to the B and sigma files
            B = np.memmap("B.npy", dtype=R[0].a.dtype, mode="r+", shape=(R[0].ndim, calculation.maximum_iterations))
            B[:, curr_size] = q
            del B

            sigma = np.memmap("sigma.npy", dtype=R[0].a.dtype, mode="r+", shape=(R[0].ndim, calculation.maximum_iterations))
            sigma[:, curr_size] = HR(dR, T, H, calculation.RHF_symmetry, system)
            del sigma

            curr_size += 1

        if is_converged[n]:
            print("Converged root {}".format(n + 1))
        else:
            print("Failed to converge root {}".format(n + 1))
        print("")

        # Calculate r0 for the root
        r0[n] = get_r0(R[n], H, omega[n])

    # clean up B and sigma npy memory maps
    remove_file("B.npy")
    remove_file("sigma.npy")

    return R, omega, r0, is_converged



def eomcc_davidson(HR, update_r, R, omega, T, H, calculation, system):
    """
    Diagonalize the similarity-transformed Hamiltonian HBar using the
    non-Hermitian Davidson algorithm.
    """
    from ccpy.drivers.cc_energy import get_r0

    nroot = len(R)

    is_converged = [False] * nroot
    r0 = np.zeros(nroot)

    # orthonormalize the initial trial space; this is important when using doubles in EOMCCSd guess
    B0, _ = np.linalg.qr(np.asarray([r.flatten() for r in R]).T)

    # Allocate residual R cluster operator
    dR = ClusterOperator(system,
                         order=R[0].order,
                         active_orders=calculation.active_orders,
                         num_active=calculation.num_active,
                         data_type=R[0].a.dtype)

    for n in range(nroot):

        print("Solving for root - {}    Energy of initial guess = {}".format(n + 1, omega[n]))
        print(
            " Iter        omega                |r|               dE            Wall Time"
        )
        print(
            "--------------------------------------------------------------------------------"
        )

        # Allocate the B and sigma matrices
        sigma = np.zeros((R[n].ndim, calculation.maximum_iterations))
        B = np.zeros((R[n].ndim, calculation.maximum_iterations))

        # Initial values
        B[:, 0] = B0[:, n]
        dR.unflatten(B[:, 0])
        sigma[:, 0] = HR(dR, T, H, calculation.RHF_symmetry, system)

        curr_size = 1
        while curr_size < calculation.maximum_iterations:
            t1 = time.time()
            # store old energy
            omega_old = omega[n]

            # solve projection subspace eigenproblem
            G = np.dot(B[:, :curr_size].T, sigma[:, :curr_size])
            e, alpha = np.linalg.eig(G)

            # select root based on maximum overlap with initial guess
            # < b0 | V_i > = < b0 | \sum_k alpha_{ik} |b_k>
            # = \sum_k alpha_{ik} < b0 | b_k > = \sum_k alpha_{i0}
            idx = np.argsort(abs(alpha[0, :]))
            alpha = np.real(alpha[:, idx[-1]])

            # Get the eigenpair of interest
            omega[n] = np.real(e[idx[-1]])
            R[n].unflatten(np.dot(B[:, :curr_size], alpha))

            # calculate residual vector
            dR.unflatten(np.dot(sigma[:, :curr_size], alpha) - omega[n] * R[n].flatten())
            residual = np.linalg.norm(dR.flatten())
            deltaE = omega[n] - omega_old

            t2 = time.time()
            minutes, seconds = divmod(t2 - t1, 60)
            print(
                "   {}      {:.10f}       {:.10f}      {:.10f}      {:.2f}m {:.2f}s".format(
                    curr_size, omega[n], residual, deltaE, minutes, seconds
                )
            )
            if residual < calculation.convergence_tolerance and abs(deltaE) < calculation.convergence_tolerance:
                is_converged[n] = True
                break

            # update residual vector
            dR = update_r(dR, omega[n], H)
            q = dR.flatten()
            for p in range(curr_size):
                b = B[:, p] / np.linalg.norm(B[:, p])
                q -= np.dot(b.T, q) * b
            q *= 1.0 / np.linalg.norm(q)
            dR.unflatten(q)

            B[:, curr_size] = q
            sigma[:, curr_size] = HR(dR, T, H, calculation.RHF_symmetry, system)
            curr_size += 1

        if is_converged[n]:
            print("Converged root {}".format(n + 1))
        else:
            print("Failed to converge root {}".format(n + 1))
        print("")

        # Calculate r0 for the root
        r0[n] = get_r0(R[n], H, omega[n])

    return R, omega, r0, is_converged



def cc_jacobi(update_t, T, dT, H, calculation, system):
    import time

    from ccpy.drivers.cc_energy import get_cc_energy
    from ccpy.drivers.diis import DIIS

    # instantiate the DIIS accelerator object
    diis_engine = DIIS(T, calculation.diis_size, calculation.diis_out_of_core)

    # Jacobi/DIIS iterations
    ndiis_cycle = 0
    energy = 0.0
    energy_old = get_cc_energy(T, H)
    is_converged = False

    print("   Energy of initial guess = {:>20.10f}".format(energy_old))

    t_start = time.time()
    print_iteration_header()
    for niter in range(calculation.maximum_iterations):
        # get iteration start time
        t1 = time.time()

        # Update the T vector
        T, dT = update_t(T, dT, H, calculation.energy_shift, calculation.RHF_symmetry, system)

        # CC correlation energy
        energy = get_cc_energy(T, H)

        # change in energy
        delta_energy = energy - energy_old

        # check for exit condition
        residuum = np.linalg.norm(dT.flatten())
        if (
            residuum < calculation.convergence_tolerance
            and abs(delta_energy) < calculation.convergence_tolerance
        ):
            # print the iteration of convergence
            elapsed_time = time.time() - t1
            print_iteration(niter, residuum, delta_energy, energy, elapsed_time)

            t_end = time.time()
            minutes, seconds = divmod(t_end - t_start, 60)
            print(
                "   CC calculation successfully converged! ({:0.2f}m  {:0.2f}s)".format(
                    minutes, seconds
                )
            )
            is_converged = True
            break

        # Save T and dT vectors to disk for DIIS
        diis_engine.push(T, dT, niter)

        # Do DIIS extrapolation
        if niter % calculation.diis_size == 0 and niter > 1:
            ndiis_cycle += 1
            print("   DIIS Cycle - {}".format(ndiis_cycle))
            T.unflatten(diis_engine.extrapolate())

        # Update old energy
        energy_old = energy

        elapsed_time = time.time() - t1
        print_iteration(niter, residuum, delta_energy, energy, elapsed_time)
    else:
        print("CC calculation did not converge.")

    return T, energy, is_converged


# [TODO]: extend this routine to excited-state left CC in a natural way
def left_cc_jacobi(update_l, L, LH, T, R, H, omega, calculation):
    import time

    from ccpy.drivers.cc_energy import get_lcc_energy
    from ccpy.drivers.diis import DIIS

    # decide whether this is an excited-state left CC calc based on R
    is_ground = True
    if R is not None:
        is_ground = False

    # instantiate the DIIS accelerator object
    diis_engine = DIIS(L, calculation.diis_size, calculation.diis_out_of_core)

    # Jacobi/DIIS iterations
    ndiis_cycle = 0
    energy = 0.0
    energy_old = get_lcc_energy(L, LH)
    is_converged = False

    print("   Energy of initial guess = {:>20.10f}".format(energy_old))

    t_start = time.time()
    print_iteration_header()
    for niter in range(calculation.maximum_iterations):
        # get iteration start time
        t1 = time.time()

        # Update the L vector
        L, LH = update_l(L,
                         LH,
                         T,
                         H,
                         omega,
                         calculation.energy_shift,
                         is_ground,
                         calculation.RHF_symmetry)

        # left CC correlation energy
        energy = get_lcc_energy(L, LH)

        # change in energy
        delta_energy = energy - energy_old

        # check for exit condition
        residuum = np.linalg.norm(LH.flatten())
        if (
            residuum < calculation.convergence_tolerance
            and abs(delta_energy) < calculation.convergence_tolerance
        ):
            # print the iteration of convergence
            elapsed_time = time.time() - t1
            print_iteration(niter, residuum, delta_energy, energy, elapsed_time)

            t_end = time.time()
            minutes, seconds = divmod(t_end - t_start, 60)
            print(
                "   Left CC calculation successfully converged! ({:0.2f}m  {:0.2f}s)".format(
                    minutes, seconds
                )
            )
            is_converged = True
            break

        # Save T and dT vectors to disk for DIIS
        diis_engine.push(L, LH, niter)

        # Do DIIS extrapolation
        if niter % calculation.diis_size == 0 and niter > 1:
            ndiis_cycle += 1
            print("   DIIS Cycle - {}".format(ndiis_cycle))
            L.unflatten(diis_engine.extrapolate())

        # Update old energy
        energy_old = energy

        # biorthogonalize to R for excited states
        if not is_ground:
            LR = np.dot(L.flatten().T, R.flatten())
            L.unflatten(1.0/LR * L.flatten())

        elapsed_time = time.time() - t1
        print_iteration(niter, residuum, delta_energy, energy, elapsed_time)
    else:
        print("Left CC calculation did not converge.")

    # explicitly enforce biorthonormality
    if not is_ground:
        LR = np.einsum("em,em->", R.a, L.a, optimize=True)
        LR += np.einsum("em,em->", R.b, L.b, optimize=True)
        LR += 0.25 * np.einsum("efmn,efmn->", R.aa, L.aa, optimize=True)
        LR += np.einsum("efmn,efmn->", R.ab, L.ab, optimize=True)
        LR += 0.25 * np.einsum("efmn,efmn->", R.bb, L.bb, optimize=True)
        L.unflatten(1.0/LR * L.flatten())

    return L, energy, is_converged
