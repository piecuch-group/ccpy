"""Module containing the driving solvers"""
import time

import numpy as np

from ccpy.utilities.printing import print_cc_iteration, print_cc_iteration_header,\
                                    print_eomcc_iteration, print_eomcc_iteration_header,\
                                    print_amplitudes
from ccpy.utilities.utilities import remove_file
from ccpy.models.operators import ClusterOperator, FockOperator

# [TODO]: There is an error here. All roots beyond the first are 0 for some reason...
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

    # # Allocate the B and sigma matrices
    # B = np.memmap("B.npy", dtype=R[0].a.dtype, mode="w+", shape=(R[0].ndim, calculation.maximum_iterations))
    # del B
    # sigma = np.memmap("sigma.npy", dtype=R[0].a.dtype, mode="w+", shape=(R[0].ndim, calculation.maximum_iterations))
    # del sigma

    for n in range(nroot):
        t0_root = time.time()

        print("=======================================")
        print("Solving for root - ", n + 1)
        print("Energy of initial guess = {:>10.10f}".format(omega[n]))
        print("=======================================")

        print(
            " Iter        omega                |r|               dE            Wall Time"
        )
        print(
            "--------------------------------------------------------------------------------"
        )

        # Initial values
        B = np.memmap("B.npy", dtype=R[0].a.dtype, mode="w+", shape=(R[0].ndim, calculation.maximum_iterations))
        B[:, 0] = B0[:, n]
        del B
        R.unflatten(B0[:, 0])

        sigma = np.memmap("sigma.npy", dtype=R[0].a.dtype, mode="w+", shape=(R[0].ndim, calculation.maximum_iterations))
        sigma[:, 0] = HR(dR, R, T, H, calculation.RHF_symmetry, system)
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
            R.unflatten(dr)

            residual = np.linalg.norm(dr)
            deltaE = omega[n] - omega_old

            minutes, seconds = divmod(time.time() - t1, 60)
            print(
                "   {}      {:.10f}       {:.10f}      {:.10f}      {:.2f}m {:.2f}s".format(
                    curr_size, omega[n], residual, deltaE, minutes, seconds
                )
            )
            if residual < calculation.convergence_tolerance and abs(deltaE) < calculation.convergence_tolerance:
                is_converged[n] = True
                break

            # update residual vector
            R = update_r(R, omega[n], H, system)
            q = R.flatten()

            for k1 in range(curr_size):
                b = B[:, k1] / np.linalg.norm(B[:, k1])
                q -= np.dot(b.T, q) * b

            q *= 1.0 / np.linalg.norm(q)
            R.unflatten(q)

            # write new vectors to the B and sigma files
            B = np.memmap("B.npy", dtype=R[0].a.dtype, mode="r+", shape=(R[0].ndim, calculation.maximum_iterations))
            B[:, curr_size] = q
            del B

            sigma = np.memmap("sigma.npy", dtype=R[0].a.dtype, mode="r+", shape=(R[0].ndim, calculation.maximum_iterations))
            sigma[:, curr_size] = HR(dR, R, T, H, calculation.RHF_symmetry, system)
            del sigma

            curr_size += 1

        if is_converged[n]:
            print("Converged root {}".format(n + 1))
        else:
            print("Failed to converge root {}".format(n + 1))

        # get the time for the root
        minutes, seconds = divmod(time.time() - t0_root, 60)
        print('Total time: {:.2f}m {:.2f}s'.format(minutes, seconds))
        print("")

        # store the actual root you've solved for
        R[n].unflatten(r)

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
    from copy import deepcopy

    nroot = len(R)

    is_converged = [False] * nroot
    r0 = np.zeros(nroot)

    # orthonormalize the initial trial space; this is important when using doubles in EOMCCSd guess
    B0, _ = np.linalg.qr(np.asarray([r.flatten() for r in R]).T)

    # Allocate residual R cluster operator
    dR = deepcopy(R[0])
    dR.unflatten(np.zeros(shape=dR.ndim))

    for n in range(nroot):
        t0_root = time.time()
        print("   =======================================")
        print("   Solving for root - ", n + 1)
        print("   Energy of initial guess = {:>10.10f}".format(omega[n]))
        print_amplitudes(R[n], system, 2, nprint=5)
        print("   =======================================")
        print_eomcc_iteration_header()

        # Allocate the B (correction/subspace) and sigma (HR) matrices
        sigma = np.zeros((R[n].ndim, calculation.maximum_iterations))
        B = np.zeros((R[n].ndim, calculation.maximum_iterations))

        # Initial values
        B[:, 0] = B0[:, n]
        R[n].unflatten(B[:, 0])
        sigma[:, 0] = HR(dR, R[n], T, H, calculation.RHF_symmetry, system)

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
            r = np.dot(B[:, :curr_size], alpha)

            # calculate residual vector
            R[n].unflatten(np.dot(sigma[:, :curr_size], alpha) - omega[n] * r)
            residual = np.linalg.norm(R[n].flatten())
            deltaE = omega[n] - omega_old

            # print the iteration of convergence
            elapsed_time = time.time() - t1
            print_eomcc_iteration(curr_size, omega[n], residual, deltaE, elapsed_time)

            if residual < calculation.convergence_tolerance and abs(deltaE) < calculation.convergence_tolerance:
                is_converged[n] = True
                break

            # update residual vector
            R[n] = update_r(R[n], omega[n], H, system)
            q = R[n].flatten()
            for p in range(curr_size):
                b = B[:, p] / np.linalg.norm(B[:, p])
                q -= np.dot(b.T, q) * b
            q *= 1.0 / np.linalg.norm(q)
            R[n].unflatten(q)

            B[:, curr_size] = q
            sigma[:, curr_size] = HR(dR, R[n], T, H, calculation.RHF_symmetry, system)
            curr_size += 1

        if is_converged[n]:
            print("Converged root {}".format(n + 1))
        else:
            print("Failed to converge root {}".format(n + 1))
        minutes, seconds = divmod(time.time() - t0_root, 60)
        print('Total time: {:.2f}m {:.2f}s'.format(minutes, seconds))
        print("")

        # store the actual root you've solved for
        R[n].unflatten(r)

        # Calculate r0 for the root
        if isinstance(R[n], ClusterOperator):
            r0[n] = get_r0(R[n], H, omega[n])

    return R, omega, r0, is_converged

def eccc_jacobi(update_t, T, dT, H, calculation, system, T_ext, VT_ext):

    from ccpy.drivers.cc_energy import get_cc_energy
    from ccpy.drivers.diis import DIIS

    # instantiate the DIIS accelerator object
    diis_engine = DIIS(T, calculation.diis_size, calculation.low_memory)

    # Jacobi/DIIS iterations
    num_throw_away = 3
    ndiis_cycle = 0
    energy = 0.0
    energy_old = get_cc_energy(T, H)
    is_converged = False

    print("   Energy of initial guess = {:>20.10f}".format(energy_old))

    t_start = time.time()
    print_cc_iteration_header()
    for niter in range(calculation.maximum_iterations):
        # get iteration start time
        t1 = time.time()

        # Update the T vector
        T, dT = update_t(T, dT, H, calculation.energy_shift, calculation.RHF_symmetry, system, T_ext, VT_ext)

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
            print_cc_iteration(niter, residuum, delta_energy, energy, elapsed_time)

            t_end = time.time()
            minutes, seconds = divmod(t_end - t_start, 60)
            print(
                "   ec-CC calculation successfully converged! ({:0.2f}m  {:0.2f}s)".format(
                    minutes, seconds
                )
            )
            is_converged = True
            break

        # Save T and dT vectors to disk for DIIS
        if niter >= num_throw_away:
            diis_engine.push(T, dT, niter)

        # Do DIIS extrapolation
        if niter >= calculation.diis_size + num_throw_away:
            ndiis_cycle += 1
            T.unflatten(diis_engine.extrapolate())

        # Update old energy
        energy_old = energy

        elapsed_time = time.time() - t1
        print_cc_iteration(niter, residuum, delta_energy, energy, elapsed_time)
    else:
        print("ec-CC calculation did not converge.")

    # Remove the t.npy and dt.npy files if out-of-core DIIS was used
    diis_engine.cleanup()

    return T, energy, is_converged

def cc_jacobi(update_t, T, dT, H, calculation, system):

    from ccpy.drivers.cc_energy import get_cc_energy
    from ccpy.drivers.diis import DIIS

    # instantiate the DIIS accelerator object
    diis_engine = DIIS(T, calculation.diis_size, calculation.low_memory)

    # Jacobi/DIIS iterations
    num_throw_away = 3
    ndiis_cycle = 0
    energy = 0.0
    energy_old = get_cc_energy(T, H)
    is_converged = False

    print("   Energy of initial guess = {:>20.10f}".format(energy_old))

    t_start = time.time()
    print_cc_iteration_header()
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
            print_cc_iteration(niter, residuum, delta_energy, energy, elapsed_time)

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
        if niter >= num_throw_away:
            diis_engine.push(T, dT, niter)

        # Do DIIS extrapolation
        if niter >= calculation.diis_size + num_throw_away:
            ndiis_cycle += 1
            T.unflatten(diis_engine.extrapolate())

        # Update old energy
        energy_old = energy

        elapsed_time = time.time() - t1
        print_cc_iteration(niter, residuum, delta_energy, energy, elapsed_time)
    else:
        print("CC calculation did not converge.")

    # Remove the t.npy and dt.npy files if out-of-core DIIS was used
    diis_engine.cleanup()

    return T, energy, is_converged


def left_cc_jacobi(update_l, L, LH, T, R, H, omega, calculation, is_ground, system):

    from ccpy.drivers.cc_energy import get_lcc_energy
    from ccpy.drivers.diis import DIIS

    # instantiate the DIIS accelerator object
    diis_engine = DIIS(L, calculation.diis_size, calculation.low_memory, vecfile="l.npy", residfile="dl.npy")

    # Jacobi/DIIS iterations
    ndiis_cycle = 0
    energy = 0.0
    energy_old = get_lcc_energy(L, LH)
    is_converged = False

    print("   Energy of initial guess = {:>20.10f}".format(energy_old))

    t_start = time.time()
    print_eomcc_iteration_header()
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
                         calculation.RHF_symmetry,
                         system)

        # left CC correlation energy
        energy = get_lcc_energy(L, LH) + omega

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
            print_eomcc_iteration(niter, omega, residuum, delta_energy, elapsed_time)

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
        if niter >= calculation.diis_size:
            ndiis_cycle += 1
            L.unflatten(diis_engine.extrapolate())

        # Update old energy
        energy_old = energy

        # biorthogonalize to R for excited states
        if not is_ground:
            LR = np.dot(L.flatten().T, R.flatten())
            L.unflatten(1.0 / LR * L.flatten())

        elapsed_time = time.time() - t1
        print_eomcc_iteration(niter, energy, residuum, delta_energy, elapsed_time)
    else:
        print("Left CC calculation did not converge.")

    # explicitly enforce biorthonormality
    if isinstance(L, ClusterOperator):
        if not is_ground:
            LR =  np.einsum("em,em->", R.a, L.a, optimize=True)
            LR += np.einsum("em,em->", R.b, L.b, optimize=True)
            LR += 0.25 * np.einsum("efmn,efmn->", R.aa, L.aa, optimize=True)
            LR += np.einsum("efmn,efmn->", R.ab, L.ab, optimize=True)
            LR += 0.25 * np.einsum("efmn,efmn->", R.bb, L.bb, optimize=True)

            if L.order == 3 and R.order == 3:
                LR += (1.0 / 36.0) * np.einsum("efgmno,efgmno->", R.aaa, L.aaa, optimize=True)
                LR += (1.0 / 4.0) * np.einsum("efgmno,efgmno->", R.aab, L.aab, optimize=True)
                LR += (1.0 / 4.0) * np.einsum("efgmno,efgmno->", R.abb, L.abb, optimize=True)
                LR += (1.0 / 36.0) * np.einsum("efgmno,efgmno->", R.bbb, L.bbb, optimize=True)

            L.unflatten(1.0/LR * L.flatten())
        else:
            LR = 0.0

        if isinstance(L, FockOperator):

            LR = -np.einsum("m,m->", R.a, L.a, optimize=True)
            LR -= np.einsum("m,m->", R.b, L.b, optimize=True)
            LR -= 0.5 * np.einsum("fnm,fnm->", R.aa, L.aa, optimize=True)
            LR -= np.einsum("fnm,fnm->", R.ab, L.ab, optimize=True)
            LR -= np.einsum("fnm,fnm->", R.ba, L.ba, optimize=True)
            LR -= 0.5 * np.einsum("fnm,fnm->", R.bb, L.bb, optimize=True)

            L.unflatten(1.0 / LR * L.flatten())

    # Remove the t.npy and dt.npy files if out-of-core DIIS was used
    diis_engine.cleanup()

    return L, energy, LR, is_converged


def left_ccp_jacobi(update_l, L, LH, T, R, H, omega, calculation, is_ground, system, pspace):

    from ccpy.drivers.cc_energy import get_lcc_energy
    from ccpy.drivers.diis import DIIS

    # instantiate the DIIS accelerator object
    diis_engine = DIIS(L, calculation.diis_size, calculation.low_memory, vecfile="l.npy", residfile="dl.npy")

    # Jacobi/DIIS iterations
    ndiis_cycle = 0
    energy = 0.0
    energy_old = get_lcc_energy(L, LH)
    is_converged = False

    print("   Energy of initial guess = {:>20.10f}".format(energy_old))

    t_start = time.time()
    print_eomcc_iteration_header()
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
                         calculation.RHF_symmetry,
                         system,
                         pspace)

        # left CC correlation energy
        energy = get_lcc_energy(L, LH) + omega

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
            print_eomcc_iteration(niter, omega, residuum, delta_energy, elapsed_time)

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
        if niter >= calculation.diis_size:
            ndiis_cycle += 1
            L.unflatten(diis_engine.extrapolate())

        # Update old energy
        energy_old = energy

        # biorthogonalize to R for excited states
        if not is_ground:
            LR = np.dot(L.flatten().T, R.flatten())
            L.unflatten(1.0 / LR * L.flatten())

        elapsed_time = time.time() - t1
        print_eomcc_iteration(niter, energy, residuum, delta_energy, elapsed_time)
    else:
        print("Left CC calculation did not converge.")

    # explicitly enforce biorthonormality
    if isinstance(L, ClusterOperator):
        if not is_ground:
            LR =  np.einsum("em,em->", R.a, L.a, optimize=True)
            LR += np.einsum("em,em->", R.b, L.b, optimize=True)
            LR += 0.25 * np.einsum("efmn,efmn->", R.aa, L.aa, optimize=True)
            LR += np.einsum("efmn,efmn->", R.ab, L.ab, optimize=True)
            LR += 0.25 * np.einsum("efmn,efmn->", R.bb, L.bb, optimize=True)

            if L.order == 3 and R.order == 3:
                LR += (1.0 / 36.0) * np.einsum("efgmno,efgmno->", R.aaa, L.aaa, optimize=True)
                LR += (1.0 / 4.0) * np.einsum("efgmno,efgmno->", R.aab, L.aab, optimize=True)
                LR += (1.0 / 4.0) * np.einsum("efgmno,efgmno->", R.abb, L.abb, optimize=True)
                LR += (1.0 / 36.0) * np.einsum("efgmno,efgmno->", R.bbb, L.bbb, optimize=True)

            L.unflatten(1.0/LR * L.flatten())
        else:
            LR = 0.0

        if isinstance(L, FockOperator):

            LR = -np.einsum("m,m->", R.a, L.a, optimize=True)
            LR -= np.einsum("m,m->", R.b, L.b, optimize=True)
            LR -= 0.5 * np.einsum("fnm,fnm->", R.aa, L.aa, optimize=True)
            LR -= np.einsum("fnm,fnm->", R.ab, L.ab, optimize=True)
            LR -= np.einsum("fnm,fnm->", R.ba, L.ba, optimize=True)
            LR -= 0.5 * np.einsum("fnm,fnm->", R.bb, L.bb, optimize=True)

            L.unflatten(1.0 / LR * L.flatten())

    # Remove the t.npy and dt.npy files if out-of-core DIIS was used
    diis_engine.cleanup()

    return L, energy, LR, is_converged


def ccp_jacobi(update_t, T, dT, H, calculation, system, pspace):

    from ccpy.drivers.cc_energy import get_cc_energy
    from ccpy.drivers.diis import DIIS

    # instantiate the DIIS accelerator object
    diis_engine = DIIS(T, calculation.diis_size, calculation.low_memory)

    # Jacobi/DIIS iterations
    num_throw_away = 3
    ndiis_cycle = 0
    energy = 0.0
    energy_old = get_cc_energy(T, H)
    is_converged = False

    print("   Energy of initial guess = {:>20.10f}".format(energy_old))

    t_start = time.time()
    print_cc_iteration_header()
    for niter in range(calculation.maximum_iterations):
        # get iteration start time
        t1 = time.time()

        # Update the T vector
        T, dT = update_t(T, dT, H, calculation.energy_shift, calculation.RHF_symmetry, system, pspace)

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
            print_cc_iteration(niter, residuum, delta_energy, energy, elapsed_time)

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
        if niter >= num_throw_away:
            diis_engine.push(T, dT, niter)

        # Do DIIS extrapolation
        if niter >= calculation.diis_size + num_throw_away:
            ndiis_cycle += 1
            T.unflatten(diis_engine.extrapolate())

        # Update old energy
        energy_old = energy

        elapsed_time = time.time() - t1
        print_cc_iteration(niter, residuum, delta_energy, energy, elapsed_time)
    else:
        print("CC calculation did not converge.")

    # Remove the t.npy and dt.npy files if out-of-core DIIS was used
    diis_engine.cleanup()

    return T, energy, is_converged


def ccp_linear_jacobi(update_t, T, dT, H, calculation, system, t3_excitations):

    from ccpy.drivers.cc_energy import get_cc_energy
    from ccpy.drivers.diis import DIIS

    # instantiate the DIIS accelerator object
    diis_engine = DIIS(T, calculation.diis_size, calculation.low_memory)

    # Jacobi/DIIS iterations
    num_throw_away = 3
    ndiis_cycle = 0
    energy = 0.0
    energy_old = get_cc_energy(T, H)
    is_converged = False

    print("   Energy of initial guess = {:>20.10f}".format(energy_old))

    t_start = time.time()
    print_cc_iteration_header()
    for niter in range(calculation.maximum_iterations):
        # get iteration start time
        t1 = time.time()

        # Update the T vector
        T, dT = update_t(T, dT, H, calculation.energy_shift, calculation.RHF_symmetry, system, t3_excitations)

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
            print_cc_iteration(niter, residuum, delta_energy, energy, elapsed_time)

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
        if niter >= num_throw_away:
            diis_engine.push(T, dT, niter)

        # Do DIIS extrapolation
        if niter >= calculation.diis_size + num_throw_away:
            ndiis_cycle += 1
            T.unflatten(diis_engine.extrapolate())

        # Update old energy
        energy_old = energy

        elapsed_time = time.time() - t1
        print_cc_iteration(niter, residuum, delta_energy, energy, elapsed_time)
    else:
        print("CC calculation did not converge.")

    # Remove the t.npy and dt.npy files if out-of-core DIIS was used
    diis_engine.cleanup()

    return T, energy, is_converged

def mrcc_jacobi(update_t, compute_Heff, T, dT, H, model_space, calculation, system):

    from ccpy.drivers.diis import DIIS

    model_space_dim = len(model_space)

    # instantiate the DIIS accelerator object
    diis_engine = []
    for p in range(model_space_dim):
        diis_engine.append(DIIS(T, calculation.diis_size, calculation.low_memory))

    # Jacobi/DIIS iterations
    num_throw_away = 3
    ndiis_cycle = 0
    is_converged = False

    # Form the effective Hamiltonian
    Heff = compute_Heff(H, T, model_space)

    # Diagonalize effective Hamiltonian
    energies, coeffs = np.linalg.eig(Heff)
    idx = np.argsort(energies)
    energy = energies[idx[0]]
    coeff = coeffs[:, idx[0]]

    coeff_guess = coeff.copy()

    print("   Energy of initial guess = {:>20.10f}".format(energy))

    t_start = time.time()
    print_cc_iteration_header()
    for niter in range(calculation.maximum_iterations):
        # get iteration start time
        t1 = time.time()

        # update old eigenpair
        energy_old = energy
        coeff_old = coeff.copy()

        # Update the T vector
        T, dT = update_t(T, dT, H, model_space, Heff, coeff, calculation.energy_shift, calculation.RHF_symmetry, system)

        # Form the effective Hamiltonian
        Heff = compute_Heff(H, T, model_space)

        # Diagonalize effective Hamiltonian and get eigenpair of interest
        energies, coeffs = np.linalg.eig(Heff)
        overlaps = np.zeros(model_space_dim)
        for p in range(model_space_dim):
            overlaps[p] = np.dot(coeff_guess.T, coeffs[:, p])
        idx = np.argsort(overlaps)
        energy = energies[idx[-1]]
        coeff = coeffs[:, idx[-1]]

        # change in energy
        delta_energy = energy - energy_old
        # change in eigenvector
        delta_coeff = np.linalg.norm(coeff - coeff_old)
        # change in T vectors
        residuum = 0.0
        for p in range(model_space_dim):
            residuum += np.linalg.norm(dT[p].flatten())

        # check for exit condition
        if (
            residuum < calculation.convergence_tolerance
            and abs(delta_energy) < calculation.convergence_tolerance
            and abs(delta_coeff) < calculation.convergence_tolerance
        ):
            # print the iteration of convergence
            elapsed_time = time.time() - t1
            print_cc_iteration(niter, residuum, delta_energy, energy, elapsed_time)

            t_end = time.time()
            minutes, seconds = divmod(t_end - t_start, 60)
            print(
                "   MRCC calculation successfully converged! ({:0.2f}m  {:0.2f}s)".format(
                    minutes, seconds
                )
            )
            is_converged = True
            break

        # Save T and dT vectors to disk for DIIS
        if niter >= num_throw_away:
            for p in range(model_space_dim):
                diis_engine[p].push(T[p], dT[p], niter)

        # Do DIIS extrapolation
        if niter >= calculation.diis_size + num_throw_away:
            ndiis_cycle += 1
            for p in range(model_space_dim):
                T[p].unflatten(diis_engine[p].extrapolate())

        elapsed_time = time.time() - t1
        print_cc_iteration(niter, residuum, delta_energy, energy, elapsed_time)
    else:
        print("MRCC calculation did not converge.")

    # Remove the t.npy and dt.npy files if out-of-core DIIS was used
    for p in range(model_space_dim):
        diis_engine[p].cleanup()

    return T, energy, is_converged
