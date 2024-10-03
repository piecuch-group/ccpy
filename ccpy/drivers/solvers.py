"""Module containing the driving solvers"""
import time
import numpy as np
import h5py
import signal
import os
import sys

from ccpy.utilities.printing import (
        print_cc_iteration, print_cc_iteration_header,
        print_eomcc_iteration, print_eomcc_iteration_header,
        print_block_eomcc_iteration,
        print_ee_amplitudes
)
from ccpy.drivers.diis import DIIS
from ccpy.utilities.utilities import remove_file

# Define a signal handler function to handle SIGINT
def signal_handler(sig, frame):
    print("\nCtrl+C detected. Cleaning up...", end=" ")
    remove_file("eomcc-vectors.hdf5")
    remove_file("cc-diis-vectors.hdf5")
    print("Cleanup complete.")
    sys.exit(0)

# Register the signal handler for SIGINT
signal.signal(signal.SIGINT, signal_handler)

# [TODO]: Add biorthogonal L and R single-root Davidson solver (non-Hermitian Hirao-Nakatsuji algorithm)
def eomcc_nonlinear_diis(HR, update_r, B0, R, dR, omega, T, H, X, fock, system, options):
    """
    Diagonalize the nonlinear partitioned (or folded) non-Hermitian eigenvalue 
    problem A(w)*R = w*R, where A(w) is the CC Jacobian. This solver is used 
    in the excited-state CC3 (or CC2) computations, where w-dependence originates 
    from the implicit evaluation of R3 (or R2) in terms of R1 and R2 (or R1).
    [for description of algorithm, see J. Chem. Phys. 113, 5154 (2000)].
    """
    # start clock for the root
    t_root_start = time.perf_counter()
    t_cpu_root_start = time.process_time()
    # print header
    print_eomcc_iteration_header()
    # Instantiate DIIS accelerator (re-used for all roots)
    diis_engine = DIIS(R, options["diis_size"], options["diis_out_of_core"])
    # Initial values
    R.unflatten(B0)
    # begin iteration loop
    is_converged = False
    for niter in range(options["maximum_iterations"]):
        t1 = time.perf_counter()
        # store old energy
        omega_old = omega.copy()
        # normalize the right eigenvector
        R.unflatten(R.flatten() / np.linalg.norm(R.flatten()))
        # compute H*R for a given omega
        sigma = HR(dR, R, T, H, X, fock, omega, options["RHF_symmetry"], system)
        # update the value of omega
        omega = np.dot(sigma.T, R.flatten())
        # compute the residual H(omega)*R - omega*R
        dR.unflatten(sigma - omega * R.flatten())
        residual = np.linalg.norm(dR.flatten())
        delta_energy = omega - omega_old
        # check convergence logicals
        if residual < options["amp_convergence"] and abs(delta_energy) < options["energy_convergence"]:
            is_converged = True
            # print the iteration of convergence
            elapsed_time = time.perf_counter() - t1
            print_eomcc_iteration(niter, omega, residual, delta_energy, elapsed_time)
            break
        # perturbational update step u_K = r_K / (omega - D_K), where D_K = (MP) energy denominator
        dR = update_r(dR, omega, fock, options["RHF_symmetry"], system)
        # add the correction vector to R
        R.unflatten(R.flatten() + dR.flatten())
        # save new R and dR vectors for DIIS
        diis_engine.push(R, dR, niter)
        # Do DIIS extrapolation
        if niter >= options["diis_size"]:
            R.unflatten(diis_engine.extrapolate())
        # print the iteration of convergence
        elapsed_time = time.perf_counter() - t1
        print_eomcc_iteration(niter, omega, residual, delta_energy, elapsed_time)

    # Clean up the HDF5 file used to store DIIS vectors
    diis_engine.cleanup()

    # print the time taken for the root
    minutes, seconds = divmod(time.perf_counter() - t_root_start, 60)
    print(f"   Completed in {minutes:.1f}m {seconds:.1f}s")
    print(f"   Total CPU time is {time.process_time() - t_cpu_root_start} seconds")
    return R, omega, is_converged

def eomcc_davidson(HR, update_r, B0, R, dR, omega, T, H, system, options, t3_excitations=None, r3_excitations=None):
    """
    Diagonalize the similarity-transformed CC Hamiltonian entering the
    EOMCC computations (e.g., of the EE, IP, EA, DEA, or DIP varieties)
    using the non-Hermitian version of the Davidson algorithm.
    """
    t_root_start = time.perf_counter()
    t_cpu_root_start = time.process_time()

    print_eomcc_iteration_header()

    # Create new HDF5 file by first checking if one exists and if so, remove it
    remove_file("eomcc-vectors.hdf5")
    if options["davidson_out_of_core"]:
        f = h5py.File("eomcc-vectors.hdf5", "w")

    # Maximum subspace size
    nrest = 1   # number of previous vectors used to restart (>1 does not work, why?)
    noffset = 0 # flag set to 1 to include B0 in the restart subspace or 0 to exclude it (doesn't seem to help).
    max_size = options["davidson_max_subspace_size"]
    selection_method = options["davidson_selection_method"]

    # Allocate the B (correction/subspace), sigma (HR), and G (interaction) matrices
    if options["davidson_out_of_core"]:
        sigma = f.create_dataset("sigma", (max_size, R.ndim), dtype=np.float64)
        B = f.create_dataset("bmatrix", (max_size, R.ndim), dtype=np.float64)
    else:
        sigma = np.zeros((max_size, R.ndim))
        B = np.zeros((max_size, R.ndim))
    G = np.zeros((max_size, max_size))
    restart_block = np.zeros((R.ndim, nrest + noffset))

    # Initial values
    B[0, :] = B0
    R.unflatten(B0)
    dR.unflatten(dR.flatten() * 0.0)
    if t3_excitations or r3_excitations:
        sigma[0, :] = HR(dR, R, T, H, options["RHF_symmetry"], system, t3_excitations, r3_excitations)
    else:
        sigma[0, :] = HR(dR, R, T, H, options["RHF_symmetry"], system)
    if noffset == 1:
        restart_block[:, 0] = B0

    is_converged = False
    curr_size = 1
    for niter in range(options["maximum_iterations"]):
        t1 = time.perf_counter()
        # store old energy
        omega_old = omega.copy()

        # solve projection subspace eigenproblem: G_{IJ} = sum_K B_{KI} S_{KJ} (vectorized)
        G[curr_size - 1, :curr_size] = np.einsum("k,pk->p", B[curr_size - 1, :], sigma[:curr_size, :])
        G[:curr_size, curr_size - 1] = np.einsum("k,pk->p", sigma[curr_size - 1, :], B[:curr_size, :])
        e, alpha_full = np.linalg.eig(G[:curr_size, :curr_size])

        # select root
        if selection_method == "overlap":  # Option 1: based on overlap
            # < b0 | V_i > = < b0 | \sum_k alpha_{ik} |b_k>
            # = \sum_k alpha_{ik} < b0 | b_k > = \sum_k alpha_{i0}
            idx = np.argsort(abs(alpha_full[0, :]))
            iselect = idx[-1]
        elif selection_method == "energy":  # Option 2: based on energy
            idx = np.argsort([abs(x - omega) for x in e])
            iselect = idx[0]

        alpha = np.real(alpha_full[:, iselect])

        # Get the eigenpair of interest
        omega = np.real(e[iselect])
        r = np.dot(B[:curr_size, :].T, alpha)
        # Uncomment these lines to print R at each iteration
        #R.unflatten(r)
        #print_ee_amplitudes(R, system, R.order, 0.09)
        restart_block[:, niter % nrest + noffset] = r

        # calculate residual vector: r_i = S_{iK}*alpha_{K} - omega * r_i
        R.unflatten(np.dot(sigma[:curr_size, :].T, alpha) - omega * r)
        residual = np.linalg.norm(R.flatten())
        delta_energy = omega - omega_old

        if residual < options["amp_convergence"] and abs(delta_energy) < options["energy_convergence"]:
            is_converged = True
            # print the iteration of convergence
            elapsed_time = time.perf_counter() - t1
            print_eomcc_iteration(niter, omega, residual, delta_energy, elapsed_time)
            break

        # update residual vector using diagonal preconditioning
        if t3_excitations or r3_excitations:
            R = update_r(R, omega, H, options["RHF_symmetry"], system, r3_excitations)
        else:
            R = update_r(R, omega, H, options["RHF_symmetry"], system)
        # orthogonalize residual against subspace vectors (would be nice to vectorize this)
        q = R.flatten()
        q /= np.linalg.norm(q)
        for p in range(curr_size):
            b = B[p, :] / np.linalg.norm(B[p, :])
            q -= np.dot(b, q) * b
        q /= np.linalg.norm(q)
        R.unflatten(q)

        # If below maximum subspace size, expand the subspace
        if curr_size < max_size:
            B[curr_size, :] = q
            if t3_excitations or r3_excitations:
                sigma[curr_size, :] = HR(dR, R, T, H, options["RHF_symmetry"], system, t3_excitations, r3_excitations)
            else:
                sigma[curr_size, :] = HR(dR, R, T, H, options["RHF_symmetry"], system)
        else:
            # Basic restart - use the last approximation to the eigenvector
            print("       **Deflating subspace**")
            restart_block, _ = np.linalg.qr(restart_block)
            for j in range(restart_block.shape[1]):
                R.unflatten(restart_block[:, j])
                B[j, :] = R.flatten()
                if t3_excitations or r3_excitations:
                    sigma[j, :] = HR(dR, R, T, H, options["RHF_symmetry"], system, t3_excitations, r3_excitations)
                else:
                    sigma[j, :] = HR(dR, R, T, H, options["RHF_symmetry"], system)
            curr_size = restart_block.shape[1] - 1

        # print the iteration of convergence
        elapsed_time = time.perf_counter() - t1
        print_eomcc_iteration(niter, omega, residual, delta_energy, elapsed_time)

        curr_size += 1

    # store the actual root you've solved for
    R.unflatten(r)
    # remove HDF5 file
    remove_file("eomcc-vectors.hdf5")
    # print the time taken for the root
    minutes, seconds = divmod(time.perf_counter() - t_root_start, 60)
    print(f"   Completed in {minutes:.1f}m {seconds:.1f}s")
    print(f"   Total CPU time is {time.process_time() - t_cpu_root_start} seconds")
    return R, omega, is_converged

# Trying to get the version with restarts working...
# def eomcc_block_davidson(HR, update_r, B0, R, dR, omega, T, H, system, state_index, options):
#     """
#     Diagonalize the similarity-transformed Hamiltonian HBar using the
#     non-Hermitian block Davidson algorithm.
#     Here, it is assumed that you have a list of R operators, [R1, R2, ..., Rn]
#     and a single residual container dR that is re-used for each root.
#     """
#     print_eomcc_iteration_header()
#
#     # Number of roots
#     nroot = len(state_index)
#     ndim = R[state_index[0]].ndim
#     max_size = nroot * options["davidson_max_subspace_size"]
#     selection_method = options["davidson_selection_method"]
#
#     # Allocate the B (correction/subspace), sigma (HR), and G (interaction) matrices
#     sigma = np.zeros((ndim, max_size))
#     B = np.zeros((ndim, max_size))
#     G = np.zeros((max_size, max_size))
#
#     # Initial values
#     num_add = 0
#     curr_size = 0
#     for j, istate in enumerate(state_index):
#         B[:, j] = B0[:, j]
#         R[istate].unflatten(B[:, j])
#         dR.unflatten(dR.flatten() * 0.0)
#         sigma[:, j] = HR(dR, R[istate], T, H, options["RHF_symmetry"], system)
#         num_add += 1
#         curr_size += 1
#
#     is_converged = [False] * nroot
#     residual = np.zeros(nroot)
#     delta_energy = np.zeros(nroot)
#     for niter in range(options["maximum_iterations"]):
#         t1 = time.perf_counter()
#         # store old energy
#         omega_old = omega.copy()
#
#         # solve projection subspace eigenproblem: G_{IJ} = sum_K B_{KI} S_{KJ}
#         for p in range(curr_size):
#             for j in range(1, num_add + 1):
#                 G[curr_size - j, p] = np.dot(B[:, curr_size - j].T, sigma[:, p])
#                 G[p, curr_size - j] = np.dot(sigma[:, curr_size - j].T, B[:, p])
#         e, alpha_full = np.linalg.eig(G[:curr_size, :curr_size])
#
#         # Compute the approximations
#         n_active = sum([not x for x in is_converged])
#         i_active = [j for j in range(nroot) if is_converged[j]]
#         alpha = np.zeros((curr_size, nroot))
#         for j, istate in enumerate(state_index):
#             # Cycle if root is already converged
#             if is_converged[j]: continue
#             # select root
#             if selection_method == "overlap":  # Option 1: based on overlap
#                 idx = np.argsort(abs(alpha_full[j, :]))
#                 iselect = idx[-1]
#             elif selection_method == "energy": # Option 2: based on energy
#                 idx = np.argsort(e)
#                 iselect = idx[j]
#
#             # Get the expansion coefficients for the j-th root
#             alpha[:, j] = np.real(alpha_full[:, iselect])
#
#             # Get the eigenpair of interest
#             omega[istate] = np.real(e[iselect])
#             r = np.dot(B[:, :curr_size], alpha[:, j])
#
#             # calculate residual vector: r_i = S_{iK}*alpha_{K} - omega * r_i
#             R[istate].unflatten(np.dot(sigma[:, :curr_size], alpha[:, j]) - omega[istate] * r)
#             residual[j] = np.linalg.norm(R[istate].flatten())
#             delta_energy[j] = omega[istate] - omega_old[istate]
#
#             # Check convergence
#             if residual[j] < options["amp_convergence"] and abs(delta_energy[j]) < options["energy_convergence"]:
#                 R[istate].unflatten(r)
#                 is_converged[j] = True
#
#         # Expand the subspace
#         num_add = 0
#         if curr_size + n_active < max_size:
#             for j, istate in enumerate(state_index):
#                 # Cycle if root is already converged
#                 if is_converged[j]: continue
#                 # update the residual vector
#                 R[istate] = update_r(R[istate], omega[istate], H, system)
#                 q = R[istate].flatten()
#                 for p in range(curr_size + num_add):
#                     b = B[:, p] / np.linalg.norm(B[:, p])
#                     q -= np.dot(b.T, q) * b
#                 q /= np.linalg.norm(q)
#                 R[istate].unflatten(q)
#                 B[:, curr_size + num_add] = q
#                 sigma[:, curr_size + num_add] = HR(dR, R[istate], T, H, options["RHF_symmetry"], system)
#                 num_add += 1
#             curr_size += num_add
#         else: # deflate the subspace
#             print("      **Deflating subspace**")
#             btemp = np.zeros((ndim, n_active))
#             ct = 0
#             for j in range(nroot):
#                 if is_converged[j]: continue
#                 btemp[:, ct] = np.dot(B[:, :curr_size], alpha[:, j])
#                 ct += 1
#             B[:, :n_active] = btemp
#             B[:, :n_active], _ = np.linalg.qr(B[:, :n_active])
#             ct = 0
#             for j, istate in enumerate(state_index):
#                 # Cycle if root is already converged
#                 if is_converged[j]: continue
#                 R[istate].unflatten(B[:, ct])
#                 sigma[:, ct] = HR(dR, R[istate], T, H, options["RHF_symmetry"], system)
#                 ct += 1
#             curr_size = n_active
#             num_add = n_active
#
#         # print the iteration
#         elapsed_time = time.perf_counter() - t1
#         print_block_eomcc_iteration(niter + 1, curr_size, omega, residual, delta_energy, elapsed_time, state_index)
#
#         # Check for all roots converged and break
#         if all(is_converged):
#             print("   All roots converged")
#             break
#
#     return R, omega, is_converged

def eomcc_block_davidson(HR, update_r, B0, R, dR, omega, T, H, system, state_index, options, t3_excitations=None, r3_excitations=None):
    """
    Diagonalize the similarity-transformed Hamiltonian HBar using the
    non-Hermitian block Davidson algorithm.
    Here, it is assumed that you have a list of R operators, [R1, R2, ..., Rn]
    and a single residual container dR that is re-used for each root.
    """
    print_eomcc_iteration_header()

    # Number of roots
    nroot = len(state_index)
    ndim = R[state_index[0]].ndim
    max_size = nroot * options["maximum_iterations"]
    selection_method = options["davidson_selection_method"]

    # Allocate the B (correction/subspace), sigma (HR), and G (interaction) matrices
    sigma = np.zeros((ndim, max_size))
    B = np.zeros((ndim, max_size))
    G = np.zeros((max_size, max_size))

    # Initial values
    num_add = 0
    curr_size = 0
    for j, istate in enumerate(state_index):
        B[:, j] = B0[:, j]
        R[istate].unflatten(B[:, j])
        dR.unflatten(dR.flatten() * 0.0)
        if t3_excitations or r3_excitations:
            sigma[:, j] = HR(dR, R[istate], T, H, options["RHF_symmetry"], system, t3_excitations, r3_excitations)
        else:
            sigma[:, j] = HR(dR, R[istate], T, H, options["RHF_symmetry"], system)
        num_add += 1
        curr_size += 1

    is_converged = [False for _ in range(nroot)]
    residual = np.zeros(nroot)
    delta_energy = np.zeros(nroot)
    for niter in range(options["maximum_iterations"]):
        t1 = time.perf_counter()
        # store old energy
        omega_old = omega.copy()

        # solve projection subspace eigenproblem: G_{IJ} = sum_K B_{KI} S_{KJ}
        for p in range(curr_size):
            for j in range(1, num_add + 1):
                G[curr_size - j, p] = np.dot(B[:, curr_size - j].T, sigma[:, p])
                G[p, curr_size - j] = np.dot(sigma[:, curr_size - j].T, B[:, p])
        e, alpha_full = np.linalg.eig(G[:curr_size, :curr_size])

        num_add = 0
        nmax_add = sum([not x for x in is_converged])
        alpha = np.zeros((curr_size, nroot))
        for j, istate in enumerate(state_index):
            # Cycle if root is already converged
            if is_converged[j]: continue

            # select root
            if selection_method == "overlap":  # Option 1: based on overlap
                idx = np.argsort(abs(alpha_full[j, :]))
                iselect = idx[-1]
            elif selection_method == "energy": # Option 2: based on energy
                idx = np.argsort(e)
                iselect = idx[j]

            # Get the expansion coefficients for the j-th root
            alpha[:, j] = np.real(alpha_full[:, iselect])

            # Get the eigenpair of interest
            omega[istate] = np.real(e[iselect])
            r = np.dot(B[:, :curr_size], alpha[:, j])

            # calculate residual vector: r_i = S_{iK}*alpha_{K} - omega * r_i
            R[istate].unflatten(np.dot(sigma[:, :curr_size], alpha[:, j]) - omega[istate] * r)
            residual[j] = np.linalg.norm(R[istate].flatten())
            delta_energy[j] = omega[istate] - omega_old[istate]

            # Check convergence
            if residual[j] < options["amp_convergence"] and abs(delta_energy[j]) < options["energy_convergence"]:
                is_converged[j] = True
            else:
                # update the residual vector
                if t3_excitations or r3_excitations:
                    R[istate] = update_r(R[istate], omega[istate], H, options["RHF_symmetry"], system, r3_excitations)
                else:
                    R[istate] = update_r(R[istate], omega[istate], H, options["RHF_symmetry"], system)
                q = R[istate].flatten()
                for p in range(curr_size + num_add):
                    b = B[:, p] / np.linalg.norm(B[:, p])
                    q -= np.dot(b.T, q) * b
                q /= np.linalg.norm(q)
                R[istate].unflatten(q)
                B[:, curr_size + num_add] = q
                if t3_excitations or r3_excitations:
                    sigma[:, curr_size + num_add] = HR(dR, R[istate], T, H, options["RHF_symmetry"], system, t3_excitations, r3_excitations)
                else:
                    sigma[:, curr_size + num_add] = HR(dR, R[istate], T, H, options["RHF_symmetry"], system)
                num_add += 1

            # Store the root you've solved for
            R[istate].unflatten(r)

        # Check for all roots converged and break
        if all(is_converged):
            print("   All roots converged")
            break

        # print the iteration
        elapsed_time = time.perf_counter() - t1
        print_block_eomcc_iteration(niter + 1, curr_size, omega, residual, delta_energy, elapsed_time, state_index)
        curr_size += num_add

    return R, omega, is_converged

def eccc_jacobi(update_t, T, dT, H, X, T_ext, VT_ext, system, options):
    from ccpy.energy.cc_energy import get_cc_energy

    # check whether DIIS is being used
    do_diis = True
    if options["diis_size"] == -1:
        do_diis = False

    # instantiate the DIIS accelerator object
    if do_diis:
        diis_engine = DIIS(T, options["diis_size"], options["diis_out_of_core"])

    # Jacobi/DIIS iterations
    num_throw_away = 0
    ndiis_cycle = 0
    energy = 0.0
    energy_old = get_cc_energy(T, H)
    is_converged = False

    print("   Energy of initial guess = {:>20.10f}".format(energy_old))

    t_start = time.perf_counter()
    t_cpu_start = time.process_time()
    print_cc_iteration_header()
    for niter in range(options["maximum_iterations"]):
        # get iteration start time
        t1 = time.perf_counter()

        # Update the T vector
        T, dT = update_t(T, dT, H, X, options["energy_shift"], options["RHF_symmetry"], system, T_ext, VT_ext)

        # CC correlation energy
        energy = get_cc_energy(T, H)

        # change in energy
        delta_energy = energy - energy_old

        # check for exit condition
        residuum = np.linalg.norm(dT.flatten())
        if (
            residuum < options["amp_convergence"]
            and abs(delta_energy) < options["energy_convergence"]
        ):
            # print the iteration of convergence
            elapsed_time = time.perf_counter() - t1
            print_cc_iteration(niter, residuum, delta_energy, energy, elapsed_time)

            t_end = time.perf_counter()
            minutes, seconds = divmod(t_end - t_start, 60)
            print("   ec-CC calculation successfully converged! ({:0.2f}m  {:0.2f}s)".format(minutes, seconds))
            print(f"   Total CPU time is {time.process_time() - t_cpu_start} seconds")
            is_converged = True
            break

        # Save T and dT vectors to disk for DIIS
        if niter >= num_throw_away and do_diis:
            diis_engine.push(T, dT, niter)

        # Do DIIS extrapolation
        if niter >= options["diis_size"] + num_throw_away and do_diis:
            ndiis_cycle += 1
            T.unflatten(diis_engine.extrapolate())

        # Update old energy
        energy_old = energy

        elapsed_time = time.perf_counter() - t1
        print_cc_iteration(niter, residuum, delta_energy, energy, elapsed_time)
    else:
        print("ec-CC calculation did not converge.")

    # Remove the t.npy and dt.npy files if out-of-core DIIS was used
    if do_diis:
        diis_engine.cleanup()

    return T, energy, is_converged

def cc_jacobi(update_t, T, dT, H, X, system, options, t3_excitations=None, acparray=None):
    from ccpy.energy.cc_energy import get_cc_energy

    # check whether DIIS is being used
    do_diis = True
    if options["diis_size"] == -1:
        do_diis = False

    # instantiate the DIIS accelerator object
    if do_diis:
        diis_engine = DIIS(T, options["diis_size"], options["diis_out_of_core"])

    # Jacobi/DIIS iterations
    num_throw_away = 0
    ndiis_cycle = 0
    energy = 0.0
    energy_old = get_cc_energy(T, H)
    is_converged = False

    print("   Energy of initial guess = {:>20.10f}".format(energy_old))

    t_start = time.perf_counter()
    t_cpu_start = time.process_time()
    print_cc_iteration_header()
    for niter in range(options["maximum_iterations"]):
        # get iteration start time
        t1 = time.perf_counter()

        # Update the T vector
        if t3_excitations: # CC(P) update
            if acparray:
                T, dT = update_t(T, dT, H, X, options["energy_shift"], options["RHF_symmetry"], system, t3_excitations, acparray)
            else:
                T, dT = update_t(T, dT, H, X, options["energy_shift"], options["RHF_symmetry"], system, t3_excitations)
        else: # regular update
            if acparray:
                T, dT = update_t(T, dT, H, X, options["energy_shift"], options["RHF_symmetry"], system, acparray)
            else:
                T, dT = update_t(T, dT, H, X, options["energy_shift"], options["RHF_symmetry"], system)

        # CC correlation energy
        energy = get_cc_energy(T, H)

        # change in energy
        delta_energy = energy - energy_old

        # check for exit condition
        residuum = np.linalg.norm(dT.flatten())
        if (
            residuum < options["amp_convergence"]
            and abs(delta_energy) < options["energy_convergence"]
        ):
            # print the iteration of convergence
            elapsed_time = time.perf_counter() - t1
            print_cc_iteration(niter, residuum, delta_energy, energy, elapsed_time)

            t_end = time.perf_counter()
            minutes, seconds = divmod(t_end - t_start, 60)
            print(
                "   CC calculation successfully converged! ({:0.2f}m  {:0.2f}s)".format(
                    minutes, seconds
                )
            )
            print(f"   Total CPU time is {time.process_time() - t_cpu_start} seconds")
            is_converged = True
            break

        # Save T and dT vectors to disk for DIIS
        if niter >= num_throw_away and do_diis:
            diis_engine.push(T, dT, niter)

        # Do DIIS extrapolation
        if niter >= options["diis_size"] + num_throw_away and do_diis:
            ndiis_cycle += 1
            T.unflatten(diis_engine.extrapolate())

        # Update old energy
        energy_old = energy

        elapsed_time = time.perf_counter() - t1
        print_cc_iteration(niter, residuum, delta_energy, energy, elapsed_time)
    else:
        print("CC calculation did not converge.")

    # Remove the t.npy and dt.npy files if out-of-core DIIS was used
    if do_diis:
        diis_engine.cleanup()

    return T, energy, is_converged


def left_cc_jacobi(update_l, L, LH, T, H, LR_function, omega, ground_state, system, options, t3_excitations=None, l3_excitations=None):
    from ccpy.energy.cc_energy import get_lcc_energy

    do_diis = True
    if options["diis_size"] == -1:
        do_diis = False

    # instantiate the DIIS accelerator object
    if do_diis:
        diis_engine = DIIS(L, options["diis_size"], options["diis_out_of_core"])

    # Jacobi/DIIS iterations
    num_throw_away = 0 # keep this at 0 for now...
    ndiis_cycle = 0
    energy = 0.0
    energy_old = get_lcc_energy(L, LH) + omega
    is_converged = False
    LR = 0.0

    print("   Energy of initial guess = {:>20.10f}".format(energy_old))

    t_start = time.perf_counter()
    t_cpu_start = time.process_time()
    print_eomcc_iteration_header()
    for niter in range(options["maximum_iterations"]):
        # get iteration start time
        t1 = time.perf_counter()

        # Update the L vector in either a CC(P) or regular fashion
        if t3_excitations or l3_excitations:
            L, LH = update_l(L,
                             LH,
                             T,
                             H,
                             omega,
                             options["energy_shift"],
                             ground_state,
                             options["RHF_symmetry"],
                             system,
                             t3_excitations,
                             l3_excitations)
        else:
            L, LH = update_l(L,
                             LH,
                             T,
                             H,
                             omega,
                             options["energy_shift"],
                             ground_state,
                             options["RHF_symmetry"],
                             system)

        # left CC correlation energy
        energy = get_lcc_energy(L, LH) + omega

        # change in energy
        delta_energy = energy - energy_old

        # check for exit condition
        residuum = np.linalg.norm(LH.flatten())
        if (
            residuum < options["amp_convergence"]
            and abs(delta_energy) < options["energy_convergence"]
        ):
            # print the iteration of convergence
            elapsed_time = time.perf_counter() - t1
            print_eomcc_iteration(niter, omega, residuum, delta_energy, elapsed_time)

            t_end = time.perf_counter()
            minutes, seconds = divmod(t_end - t_start, 60)
            print(
                "   Left CC calculation successfully converged! ({:0.2f}m  {:0.2f}s)".format(
                    minutes, seconds
                )
            )
            print(f"   Total CPU time is {time.process_time() - t_cpu_start} seconds")
            is_converged = True
            break

        # Save T and dT vectors to disk for DIIS
        if niter >= num_throw_away and do_diis:
            diis_engine.push(L, LH, niter)

        # Do DIIS extrapolation
        #if (niter + 1) % options["diis_size"] == 0 and do_diis: # this criterion works better I've found...
        if niter >= options["diis_size"] + num_throw_away and do_diis:
            ndiis_cycle += 1
            L.unflatten(diis_engine.extrapolate())

        # Update old energy
        energy_old = energy

        # biorthogonalize to R for excited states
        if not ground_state:
            LR = LR_function(L, l3_excitations)
            L.unflatten(1.0 / LR * L.flatten())

        elapsed_time = time.perf_counter() - t1
        print_eomcc_iteration(niter, energy, residuum, delta_energy, elapsed_time)
    else:
        print("Left CC calculation did not converge.")

    # Remove the t.npy and dt.npy files if out-of-core DIIS was used
    if do_diis:
        diis_engine.cleanup()

    return L, energy, LR, is_converged

def lrcc_jacobi(update_t, T1, W, dT, T, H, X, system, options, t3_excitations=None):
    from ccpy.energy.lrcc_energy import get_lrcc_energy

    # check whether DIIS is being used
    do_diis = True
    if options["diis_size"] == -1:
        do_diis = False

    # instantiate the DIIS accelerator object
    if do_diis:
        diis_engine = DIIS(T1, options["diis_size"], options["diis_out_of_core"])

    # Jacobi/DIIS iterations
    num_throw_away = 0
    ndiis_cycle = 0
    energy = 0.0
    energy_old = get_lrcc_energy(T1, W, T, H)
    is_converged = False

    print("   Initial property guess value = {:>20.10f}".format(energy_old))

    t_start = time.perf_counter()
    t_cpu_start = time.process_time()
    print_cc_iteration_header()
    for niter in range(options["maximum_iterations"]):
        # get iteration start time
        t1 = time.perf_counter()

        # Update the T vector
        if t3_excitations: # CC(P) update
             T1, dT = update_t(T1, dT, T, W, H, X, options["energy_shift"], options["RHF_symmetry"], system, t3_excitations)
        else: # regular update
             T1, dT = update_t(T1, dT, T, W, H, X, options["energy_shift"], options["RHF_symmetry"], system)

        # CC correlation energy
        energy = get_lrcc_energy(T1, W, T, H)

        # change in energy
        delta_energy = energy - energy_old

        # check for exit condition
        residuum = np.linalg.norm(dT.flatten())
        if (
            residuum < options["amp_convergence"]
            and abs(delta_energy) < options["energy_convergence"]
        ):
            # print the iteration of convergence
            elapsed_time = time.perf_counter() - t1
            print_cc_iteration(niter, residuum, delta_energy, energy, elapsed_time)

            t_end = time.perf_counter()
            minutes, seconds = divmod(t_end - t_start, 60)
            print(
                "   LR-CC calculation successfully converged! ({:0.2f}m  {:0.2f}s)".format(
                    minutes, seconds
                )
            )
            print(f"   Total CPU time is {time.process_time() - t_cpu_start} seconds")
            is_converged = True
            break

        # Save T and dT vectors to disk for DIIS
        if niter >= num_throw_away and do_diis:
            diis_engine.push(T1, dT, niter)

        # Do DIIS extrapolation
        if niter >= options["diis_size"] + num_throw_away and do_diis:
            ndiis_cycle += 1
            T1.unflatten(diis_engine.extrapolate())

        # Update old energy
        energy_old = energy

        elapsed_time = time.perf_counter() - t1
        print_cc_iteration(niter, residuum, delta_energy, energy, elapsed_time)
    else:
        print("LR-CC calculation did not converge.")

    # Remove the t.npy and dt.npy files if out-of-core DIIS was used
    if do_diis:
        diis_engine.cleanup()

    return T1, energy, is_converged


# def mrcc_jacobi(update_t, compute_Heff, T, dT, H, model_space, calculation, system):
#
#     from ccpy.drivers.diis import DIIS
#
#     model_space_dim = len(model_space)
#
#     # instantiate the DIIS accelerator object
#     diis_engine = []
#     for p in range(model_space_dim):
#         diis_engine.append(DIIS(T, calculation.diis_size, calculation.low_memory))
#
#     # Jacobi/DIIS iterations
#     num_throw_away = 3
#     ndiis_cycle = 0
#     is_converged = False
#
#     # Form the effective Hamiltonian
#     Heff = compute_Heff(H, T, model_space)
#
#     # Diagonalize effective Hamiltonian
#     energies, coeffs = np.linalg.eig(Heff)
#     idx = np.argsort(energies)
#     energy = energies[idx[0]]
#     coeff = coeffs[:, idx[0]]
#
#     coeff_guess = coeff.copy()
#
#     print("   Energy of initial guess = {:>20.10f}".format(energy))
#
#     t_start = time.perf_counter()
#     print_cc_iteration_header()
#     for niter in range(calculation.maximum_iterations):
#         # get iteration start time
#         t1 = time.perf_counter()
#
#         # update old eigenpair
#         energy_old = energy
#         coeff_old = coeff.copy()
#
#         # Update the T vector
#         T, dT = update_t(T, dT, H, model_space, Heff, coeff, calculation.energy_shift, calculation.RHF_symmetry, system)
#
#         # Form the effective Hamiltonian
#         Heff = compute_Heff(H, T, model_space)
#
#         # Diagonalize effective Hamiltonian and get eigenpair of interest
#         energies, coeffs = np.linalg.eig(Heff)
#         overlaps = np.zeros(model_space_dim)
#         for p in range(model_space_dim):
#             overlaps[p] = np.dot(coeff_guess.T, coeffs[:, p])
#         idx = np.argsort(overlaps)
#         energy = energies[idx[-1]]
#         coeff = coeffs[:, idx[-1]]
#
#         # change in energy
#         delta_energy = energy - energy_old
#         # change in eigenvector
#         delta_coeff = np.linalg.norm(coeff - coeff_old)
#         # change in T vectors
#         residuum = 0.0
#         for p in range(model_space_dim):
#             residuum += np.linalg.norm(dT[p].flatten())
#
#         # check for exit condition
#         if (
#             residuum < calculation.convergence_tolerance
#             and abs(delta_energy) < calculation.convergence_tolerance
#             and abs(delta_coeff) < calculation.convergence_tolerance
#         ):
#             # print the iteration of convergence
#             elapsed_time = time.perf_counter() - t1
#             print_cc_iteration(niter, residuum, delta_energy, energy, elapsed_time)
#
#             t_end = time.perf_counter()
#             minutes, seconds = divmod(t_end - t_start, 60)
#             print(
#                 "   MRCC calculation successfully converged! ({:0.2f}m  {:0.2f}s)".format(
#                     minutes, seconds
#                 )
#             )
#             is_converged = True
#             break
#
#         # Save T and dT vectors to disk for DIIS
#         if niter >= num_throw_away:
#             for p in range(model_space_dim):
#                 diis_engine[p].push(T[p], dT[p], niter)
#
#         # Do DIIS extrapolation
#         if niter >= calculation.diis_size + num_throw_away:
#             ndiis_cycle += 1
#             for p in range(model_space_dim):
#                 T[p].unflatten(diis_engine[p].extrapolate())
#
#         elapsed_time = time.perf_counter() - t1
#         print_cc_iteration(niter, residuum, delta_energy, energy, elapsed_time)
#     else:
#         print("MRCC calculation did not converge.")
#
#     # Remove the t.npy and dt.npy files if out-of-core DIIS was used
#     for p in range(model_space_dim):
#         diis_engine[p].cleanup()
#
#     return T, energy, is_converged
