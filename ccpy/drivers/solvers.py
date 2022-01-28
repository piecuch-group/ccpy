"""Module containing the driving solvers"""
import numpy as np
import time
from ccpy.utilities.utilities import print_memory_usage

def diis_out_of_core(cc_t, slices, sizes, vec_dim, diis_dim):
    """Performs DIIS extrapolation for the solution of nonlinear equations. Out of core
    version with residuals and previous vectors stored on disk and accessed one-by-one.
    
    Parameters:
    -----------
    cc_t : dict
        Dictionary of current T vector amplitudes
    slices : dict
        Dictionary containing slices of each T amplitude type and spincase
    sizes : dict
        Dictionary containing the sizes (tuples) for each T amplitude type and spincase
    vec_dim : int
        Total dimension of the T vector
    diis_dim : int
        Number of DIIS vectors used in extrapolation

    Returns:
    --------
    cc_t : dict
        New DIIS-extrapolated T vector amplitudes
    """
    B_dim = diis_dim + 1
    B = -1.0*np.ones((B_dim,B_dim))

    nhalf = int(vec_dim/2)
    diis_resid = np.memmap('dt.npy',dtype=np.float64,shape=(vec_dim,diis_dim),mode='r')
    for i in range(diis_dim):
        for j in range(i,diis_dim):
            B[i,j] = np.dot(diis_resid[:nhalf,i],diis_resid[:nhalf,j])
            B[i,j] += np.dot(diis_resid[nhalf:,i],diis_resid[nhalf:,j])
            B[j,i] = B[i,j]
    B[-1,-1] = 0.0

    rhs = np.zeros(B_dim)
    rhs[-1] = -1.0

    coeff = solve_gauss(B,rhs)

    tvec = np.memmap('t.npy',dtype=np.float64,shape=(vec_dim,diis_dim),mode='r')
    for key in cc_t.keys():
        cc_t[key] = 0.0*cc_t[key]
        for i in range(diis_dim):
            cc_t[key] += coeff[i]*np.reshape(tvec[slices[key],i],sizes[key])

    return cc_t

def diis(x_list, diis_resid):
    """Performs DIIS extrapolation for the solution of nonlinear equations. In core
    version with residuals and previous vectors stored fully as matrices in memory.
    
    Parameters:
    -----------
    x_list : ndarray(dtype=np.float64,shape=(vec_dim,diis_dim))
        Matrix of previous T vectors
    diis_resid : ndarray(dtype=np.float64,shape=(vec_dim,diis_dim))
        Matrix of previous residuals
    Returns:
    --------
    x_xtrap : ndarray(dtype=np.float64,shape=(vec_dim))
        New DIIS-extrapolated T vector amplitudes
    """
    vec_dim, diis_dim = np.shape(x_list)
    B_dim = diis_dim + 1
    B = -1.0*np.ones((B_dim,B_dim))

    for i in range(diis_dim):
        for j in range(diis_dim):
            B[i,j] = np.dot(diis_resid[:,i].T,diis_resid[:,j])
    B[-1,-1] = 0.0

    rhs = np.zeros(B_dim)
    rhs[-1] = -1.0

    coeff = solve_gauss(B,rhs)
    x_xtrap = np.zeros(vec_dim)
    for i in range(diis_dim):
        x_xtrap += coeff[i]*x_list[:,i]

    return x_xtrap

def solve_gauss(A, b):
    """DIIS helper function. Solves the linear system Ax=b using
    Gaussian elimination"""
    n =  A.shape[0]
    for i in range(n-1):
        for j in range(i+1, n):
            m = A[j,i]/A[i,i]
            A[j,:] -= m*A[i,:]
            b[j] -= m*b[i]
    x = np.zeros(n)
    k = n-1
    x[k] = b[k]/A[k,k]
    while k >= 0:
        x[k] = (b[k] - np.dot(A[k,k+1:],x[k+1:]))/A[k,k]
        k = k-1
    return x

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

def solve_cc_jacobi_out_of_core(cc_t,update_t,ints,maxit,tol,ndim,diis_size):
    import time
    from cc_energy import calc_cc_energy

    # Get dimensions of the CC theory
    pos = [0] 
    slices = {}
    sizes = {}
    ct = 0
    ndim = 0
    for key,value in cc_t.items():
        ndim += np.size(value)
        pos.append(pos[ct]+np.size(value))
        slices[key] = slice(pos[ct],pos[ct+1])
        sizes[key] = value.shape
        ct += 1

    # Create the memory maps to the T and dT (residual) vectors for DIIS stored on disk
    tvec_mmap = np.memmap('t.npy', mode="w+", dtype=np.float64, shape=(ndim,diis_size))
    resid_mmap = np.memmap('dt.npy', mode="w+", dtype=np.float64, shape=(ndim,diis_size))
    del tvec_mmap
    del resid_mmap

    # Jacobi/DIIS iterations
    it_micro = 0
    flag_conv = False
    it_macro = 0
    Ecorr_old = 0.0

    t_start = time.time()
    print('Iteration    Residuum               deltaE                 Ecorr                Wall Time')
    print('============================================================================================')
    while it_micro < maxit:
        # get iteration start time
        t1 = time.time()

        # get DIIS counter
        ndiis = it_micro%diis_size
      
        # Update the T vector
        cc_t, T_resid = update_t(cc_t)

        # CC correlation energy
        Ecorr = calc_cc_energy(cc_t,ints)

        # change in Ecorr
        deltaE = Ecorr - Ecorr_old

        # check for exit condition
        resid = np.linalg.norm(T_resid)
        if resid < tol and abs(deltaE) < tol:
            flag_conv = True
            break

        # Save T and dT vectors to disk using memory maps
        tvec_mmap = np.memmap('t.npy', mode="r+", dtype=np.float64, shape=(ndim,diis_size))
        for key, value in cc_t.items():
            tvec_mmap[slices[key],ndiis] = value.flatten()
        del tvec_mmap
        resid_mmap = np.memmap('dt.npy', mode="r+", dtype=np.float64, shape=(ndim,diis_size))
        resid_mmap[:,ndiis] = T_resid
        del resid_mmap

        # Do out-of-core DIIS extrapolation        
        if ndiis == 0 and it_micro > 1:
            it_macro = it_macro + 1
            print('DIIS Cycle - {}'.format(it_macro))
            cc_t = diis_out_of_core(cc_t,slices,sizes,ndim,diis_size)
        
        elapsed_time = time.time()-t1
        minutes, seconds = divmod(elapsed_time, 60)

        print('   {}       {:.10f}          {:.10f}          {:.10f}        {:0.2f}m  {:0.2f}s'.format(it_micro,resid,deltaE,Ecorr,minutes,seconds))
        
        it_micro += 1
        Ecorr_old = Ecorr

    t_end = time.time()
    minutes, seconds = divmod(t_end-t_start, 60)
    if flag_conv:
        print('CC calculation successfully converged! ({:0.2f}m  {:0.2f}s)'.format(minutes,seconds))
        print('')
        print('CC Correlation Energy = {} Eh'.format(Ecorr))
        print('CC Total Energy = {} Eh'.format(ints['Escf']+Ecorr))
    else:
        print('Failed to converge CC in {} iterations'.format(maxit))

    return cc_t, ints['Escf']+Ecorr

def solve_cc_jacobi(update_t, T, dT, H, calculation):
    import time
    from ccpy.drivers.cc_energy import calc_cc_energy
    from ccpy.models.operators import DIIS

    # instantiate the DIIS accelerator object
    diis_engine = DIIS(T, calculation.diis_size)

    # Jacobi/DIIS iterations
    ndiis_cycle = 0
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

