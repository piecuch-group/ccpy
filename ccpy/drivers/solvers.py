"""Module containing the driving solvers"""
import numpy as np
import time


def davidson_out_of_core(HR,update_R,B0,E0,maxit,tol,flag_lowmem=True):
    """Diagonalize the similarity-transformed Hamiltonian HBar using the
    non-Hermitian Davidson algorithm. Low memory version where previous
    iteration vectors are stored on disk and accessed one-by-one.

    Parameters
    ----------
    HR : func
        Call type HR(R). Returns the matrix-vector product of HBar acting on an R vector.
    B0 : ndarray(dtype=float, shape=(ndim,nroot))
        Matrix containing the initial guess vectors for the Davidson procedure
    E0 : ndarray(dtype=float, shape=(nroot))
        Vector containing the energies corresponding to the initial guess vectors
    maxit : int, optional
        Maximum number of Davidson iterations in the EOMCC procedure.
    tol : float, optional
        Convergence tolerance for the EOMCC calculation. Default is 1.0e-06.
    flag_lowmem : bool, optional, default = True
        Flag to use low-memory subroutines that do not store more than 1 R or Sigma vector
        in memory at a time.

    Returns
    -------
    Rvec : ndarray(dtype=float, shape=(ndim,nroot))
        Matrix containing the final converged R vectors corresponding to the EOMCC linear excitation amplitudes
    omega : ndarray(dtype=float, shape=(nroot))
        Vector of vertical excitation energies (in hartree) for each root
    is_converged : list
        List of boolean indicating whether each root converged to within the specified tolerance
    """
    ndim = B0.shape[0]
    nroot = B0.shape[1]
    bshape = (ndim,maxit)

    Rvec = np.zeros((ndim,nroot))
    is_converged = [False] * nroot
    omega = np.zeros(nroot)
    residuals = np.zeros(nroot)

    # orthonormalize the initial trial space
    # this is important when using doubles in EOMCCSd guess
    B0,_ = np.linalg.qr(B0)

    for iroot in range(nroot):

        print('Solving for root - {}'.format(iroot+1))
        print(' Iter        omega                |r|               dE            Wall Time')
        print('--------------------------------------------------------------------------------')

        # Initialize the memory map for the B and sigma matrices
        B = np.memmap('Rmat.npy',dtype=np.float64,mode='w+',shape=(ndim,maxit))
        sigma = np.memmap('HRmat.npy',dtype=np.float64,mode='w+',shape=(ndim,maxit))
        # [TODO] add on converged R vectors to prevent collapse onto previous roots
        B[:,0] = B0[:,iroot]
        sigma[:,0] = HR(B[:,0])
        del sigma
        del B
    
        omega[iroot] = E0[iroot]

        for curr_size in range(1,maxit):
            t1 = time.time()

            omega_old = omega[iroot]

            G = calc_Gmat(curr_size,bshape,flag_lowmem)
            e, alpha = np.linalg.eig(G)

            # select root based on maximum overlap with initial guess
            # < b0 | V_i > = < b0 | \sum_k alpha_{ik} |b_k>
            # = \sum_k alpha_{ik} < b0 | b_k > = \sum_k alpha_{i0}
            idx = np.argsort( abs(alpha[0,:]) )
            omega[iroot] = np.real(e[idx[-1]])
            alpha = np.real(alpha[:,idx[-1]])
            Rvec[:,iroot] = calc_Rvec(curr_size,alpha,bshape,flag_lowmem)

            # calculate residual vector
            q = calc_qvec(curr_size,alpha,bshape,flag_lowmem)
            q -= omega[iroot]*Rvec[:,iroot]
            residuals[iroot] = np.linalg.norm(q)
            deltaE = omega[iroot] - omega_old
            
            # update residual vector
            q = update_R(q,omega[iroot])
            q *= 1.0/np.linalg.norm(q)
            q = orthogonalize(q,curr_size,bshape)
            q *= 1.0/np.linalg.norm(q)

            t2 = time.time()
            minutes, seconds = divmod(t2 - t1, 60)
            print('   {}      {:.10f}       {:.10f}      {:.10f}      {:.2f}m {:.2f}s'.\
                            format(curr_size,omega[iroot],residuals[iroot],deltaE,minutes,seconds))

            if residuals[iroot] < tol and abs(deltaE) < tol:
                is_converged[iroot] = True
                break

            update_subspace_vecs(q,curr_size,HR,bshape)

            #print_memory_usage()
        if is_converged[iroot]:
            print('Converged root {}'.format(iroot+1))
        else:
            print('Failed to converge root {}'.format(iroot+1))
        print('')

    return Rvec, omega, is_converged

def update_subspace_vecs(q,curr_size,HR,bshape):

    B = np.memmap('Rmat.npy',dtype=np.float64,shape=bshape,mode='r+')
    sigma = np.memmap('HRmat.npy',dtype=np.float64,shape=bshape,mode='r+')
    B[:,curr_size] = q
    sigma[:,curr_size] = HR(q)
    del B
    del sigma
    return

def orthogonalize(q,curr_size,bshape):

    B = np.memmap('Rmat.npy',dtype=np.float64,shape=bshape,mode='r')
    for i in range(curr_size):
        b = B[:,i]/np.linalg.norm(B[:,i])
        q -= np.dot(b.T,q)*b
    return q

def calc_qvec(curr_size,alpha,bshape,flag_lowmem):

    sigma = np.memmap('HRmat.npy',dtype=np.float64,shape=bshape,mode='r')
    if flag_lowmem:
        q = np.zeros(bshape[0])
        for i in range(curr_size):
            q += sigma[:,i]*alpha[i]
    else:
        q = np.dot(sigma[:,:curr_size],alpha)
    return q

def calc_Rvec(curr_size,alpha,bshape,flag_lowmem):

    B = np.memmap('Rmat.npy',dtype=np.float64,shape=bshape,mode='r')
    if flag_lowmem:
        Rvec = np.zeros(bshape[0])
        for i in range(curr_size):
            Rvec += B[:,i]*alpha[i]
    else:
        Rvec = np.dot(B[:,:curr_size],alpha)
    return Rvec

def calc_Gmat(curr_size,bshape,flag_lowmem):
    
    B = np.memmap('Rmat.npy',dtype=np.float64,shape=bshape,mode='r')
    sigma = np.memmap('HRmat.npy',dtype=np.float64,shape=bshape,mode='r')
    if flag_lowmem:
        Gmat = np.zeros((curr_size,curr_size))
        for i in range(curr_size):
            for j in range(curr_size):
                Gmat[i,j] = np.dot(B[:,i].T,sigma[:,j])
    else:
        Gmat = np.dot(B[:,:curr_size].T,sigma[:,:curr_size])
    return Gmat

def davidson(HR,update_R,B0,E0,maxit,max_dim,tol):
    """Diagonalize the similarity-transformed Hamiltonian HBar using the
    non-Hermitian Davidson algorithm.

    Parameters
    ----------
    HR : func
        Function of the general call structure
            HR(R,CC-t,H1A,H1B,H2A,H2B,H2C,ints,sys,flag_RHF)
    H1*, H2* : dict
        Sliced similarity-transformed HBar integrals
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    cc_t : dict
        Cluster amplitudes T1, T2 of the ground-state
    nroot : int
        Number of excited-states to solve for
    B0 : ndarray(dtype=float, shape=(ndim,nroot))
        Matrix containing the initial guess vectors for the Davidson procedure
    E0 : ndarray(dtype=float, shape=(nroot))
        Vector containing the energies corresponding to the initial guess vectors
    sys : dict
        System information dictionary
    maxit : int, optional
        Maximum number of Davidson iterations in the EOMCC procedure.
    tol : float, optional
        Convergence tolerance for the EOMCC calculation. Default is 1.0e-06.

    Returns
    -------
    Rvec : ndarray(dtype=float, shape=(ndim,nroot))
        Matrix containing the final converged R vectors corresponding to the EOMCC linear excitation amplitudes
    omega : ndarray(dtype=float, shape=(nroot))
        Vector of vertical excitation energies (in hartree) for each root
    is_converged : list
        List of boolean indicating whether each root converged to within the specified tolerance
    """
    ndim = B0.shape[0]
    nroot = B0.shape[1]

    Rvec = np.zeros((ndim,nroot))
    is_converged = [False] * nroot
    omega = np.zeros(nroot)
    residuals = np.zeros(nroot)

    # orthonormalize the initial trial space
    # this is important when using doubles in EOMCCSd guess
    B0,_ = np.linalg.qr(B0)

    for iroot in range(nroot):

        print('Solving for root - {}'.format(iroot+1))
        print(' Iter        omega                |r|               dE            Wall Time')
        print('--------------------------------------------------------------------------------')

        # Initialize the memory map for the B and sigma matrices
        sigma = np.zeros((ndim,max_dim))
        B = np.zeros((ndim,max_dim))
        B[:,0] = B0[:,iroot]
        sigma[:,0] = HR(B[:,0])
        omega[iroot] = E0[iroot]

        curr_size = 1
        for it in range(maxit):
            t1 = time.time()

            omega_old = omega[iroot]
            G = np.dot(B[:,:curr_size].T,sigma[:,:curr_size])
            e, alpha = np.linalg.eig(G)

            # select root based on maximum overlap with initial guess
            # < b0 | V_i > = < b0 | \sum_k alpha_{ik} |b_k>
            # = \sum_k alpha_{ik} < b0 | b_k > = \sum_k alpha_{i0}
            idx = np.argsort( abs(alpha[0,:]) )
            omega[iroot] = np.real(e[idx[-1]])
            alpha = np.real(alpha[:,idx[-1]])
            Rvec[:,iroot] = np.dot(B[:,:curr_size],alpha)

            # calculate residual vector
            q = np.dot(sigma[:,:curr_size],alpha) - omega[iroot]*Rvec[:,iroot]
            residuals[iroot] = np.linalg.norm(q)
            deltaE = omega[iroot] - omega_old

            t2 = time.time()
            minutes, seconds = divmod(t2 - t1, 60)
            print('   {}      {:.10f}       {:.10f}      {:.10f}      {:.2f}m {:.2f}s'.\
                            format(curr_size,omega[iroot],residuals[iroot],deltaE,minutes,seconds))
            if residuals[iroot] < tol and abs(deltaE) < tol:
                is_converged[iroot] = True
                break
            
            # update residual vector
            q = update_R(q,omega[iroot])
            q *= 1.0/np.linalg.norm(q)
            for p in range(curr_size):
                b = B[:,p]/np.linalg.norm(B[:,p])
                q -= np.dot(b.T,q)*b
            q *= 1.0/np.linalg.norm(q)

            if curr_size < max_dim:
                B[:,curr_size] = q
                sigma[:,curr_size] = HR(q)
                curr_size += 1
            else:
                # Subspace collapse... it does not help for difficult roots
                B[:,0] = B0[:,iroot]
                B[:,1] = Rvec[:,iroot]
                B[:,:2],_ = np.linalg.qr(B[:,:2])
                sigma[:,0] = HR(B[:,0])
                sigma[:,1] = HR(B[:,1])
                curr_size = 2

        if is_converged[iroot]:
            print('Converged root {}'.format(iroot+1))
        else:
            print('Failed to converge root {}'.format(iroot+1))
        print('')

    return Rvec, omega, is_converged

def solve_cc_jacobi(update_t, T, dT, H, calculation, diis_out_of_core=False):
    import time
    from ccpy.drivers.cc_energy import calc_cc_energy
    from ccpy.drivers.diis import DIIS

    # instantiate the DIIS accelerator object
    diis_engine = DIIS(T, calculation.diis_size, diis_out_of_core)

    # Jacobi/DIIS iterations
    ndiis_cycle = 0
    energy = 0.0
    energy_old = 0.0

    t_start = time.time()
    print('  Iter        Residuum        deltaE          Ecorr')
    print(' ======================================================')
    for niter in range(calculation.maximum_iterations):
        # get iteration start time
        t1 = time.time()
      
        # Update the T vector
        T, dT = update_t(T, dT, H, calculation.level_shift, calculation.RHF_symmetry)

        # CC correlation energy
        energy = calc_cc_energy(T, H)

        # change in energy
        delta_energy = energy - energy_old

        # check for exit condition
        residuum = np.linalg.norm(dT.flatten())
        if residuum < calculation.convergence_tolerance and \
                abs(delta_energy) < calculation.convergence_tolerance:

            t_end = time.time()
            minutes, seconds = divmod(t_end - t_start, 60)
            print(' CC calculation successfully converged! ({:0.2f}m  {:0.2f}s)'.format(minutes, seconds))

            break

        # Save T and dT vectors to disk for DIIS
        diis_engine.push(T, dT, niter)

        # Do DIIS extrapolation
        if niter % calculation.diis_size == 0 and niter > 1:
            ndiis_cycle += 1
            print(' DIIS Cycle - {}'.format(ndiis_cycle))
            T.unflatten(diis_engine.extrapolate())

        # Update old energy
        energy_old = energy

        elapsed_time = time.time()-t1
        minutes, seconds = divmod(elapsed_time, 60)
        print('   {}       {:.10f}   {:.10f}   {:.10f}   ({:0.2f}m {:0.2f}s)'.format(niter,
                                                                                     residuum,
                                                                                     delta_energy,
                                                                                     energy,
                                                                                     minutes,seconds))
    else:
        print('CC calculation did not converge.')

    return T, energy

