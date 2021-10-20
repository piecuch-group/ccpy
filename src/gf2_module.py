import numpy as np

def calc_mp2_selfenergy(omega,ints,sys):
    """Calculate the self-energy matrix \Sigma_{pq} for all p,q at
    2nd-order MBPT. 

    Parameters
    ----------
    omega : float
        Energy parameter of self-energy
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary

    Returns
    -------
    sigma_mp2 : ndarray(dtype=np.float64, shape=(norb,norb))
        MBPT(2) estimate of self-energy matrix
    """
    fA = ints['fA']; fB = ints['fB'];
    vA = ints['vA']; vB = ints['vB']; vC = ints['vC'];
    # allocate separate onebody matrices for sigma_a and sigma_b (they will be the same!)
    sigma_a = np.zeros((sys['Nocc_a']+sys['Nunocc_a'],sys['Nocc_a']+sys['Nunocc_a']))
    sigma_b = np.zeros((sys['Nocc_b']+sys['Nunocc_b'],sys['Nocc_b']+sys['Nunocc_b']))
    # ooa block
    sigma_ooa = np.zeros((sys['Nocc_a'],sys['Nocc_a']))
    for i in range(sys['Nocc_a']):
        for j in range(sys['Nocc_a']):
            # alpha loop
            for n in range(sys['Nocc_a']):
                for f in range(sys['Nunocc_a']):
                    e_nf = fA['oo'][n,n] - fA['vv'][f,f]
                    # hole diagram
                    for m in range(sys['Nocc_a']):
                        denom = fA['oo'][m,n] + e_nf - omega
                        sigma_ooa[i,j] -= 0.5*(vA['ooov'][m,n,i,f]*vA['ovoo'][j,f,m,n])/denom
                    # particle diagram
                    for e in range(sys['Nunocc_a']):
                        denom = e_nf - fA['vv'][e,e] + omega
                        sigma_ooa[i,j] += 0.5*(vA['oovv'][j,n,e,f]*vA['vvoo'][e,f,i,n])/denom
            # beta loop
            for n in range(sys['Nocc_b']):
                for f in range(sys['Nunocc_b']):
                    e_nf = fB['oo'][n,n] - fB['vv'][f,f]
                    # hole diagram
                    for m in range(sys['Nocc_b']):
                        denom = fA['oo'][m,m] + e_nf - omega
                        sigma_ooa[i,j] -= (vB['ooov'][m,n,i,f]*vB['ovoo'][j,f,m,n])/denom
                    # particle diagram
                    for e in range(sys['Nunocc_a']):
                        denom = e_nf - fA['vv'][e,e] + omega
                        sigma_ooa[i,j] += (vB['oovv'][j,n,e,f]*vB['vvoo'][e,f,i,n])/denom
    # oob block
    sigma_oob = np.zeros((sys['Nocc_b'],sys['Nocc_b']))
    for i in range(sys['Nocc_b']):
        for j in range(sys['Nocc_b']):
            # alpha loop
            for n in range(sys['Nocc_a']):
                for f in range(sys['Nunocc_a']):
                    e_nf = fA['oo'][n,n] - fA['vv'][f,f]
                    # hole diagram
                    for m in range(sys['Nocc_a']):
                        denom = fB['oo'][m,m] + e_nf - omega
                        sigma_oob[i,j] -= (vB['oovo'][n,m,f,i]*vB['vooo'][f,j,n,m])/denom
                    # particle diagram
                    for e in range(sys['Nunocc_a']):
                        denom = e_nf - fB['vv'][e,e] + omega
                        sigma_oob[i,j] += (vB['oovv'][n,j,f,e]*vB['vvoo'][f,e,n,i])/denom
            # beta loop
            for n in range(sys['Nocc_b']):
                for f in range(sys['Nunocc_b']):
                    e_nf = fB['oo'][n,n] - fB['vv'][f,f]
                    # hole diagram
                    for m in range(sys['Nocc_b']):
                        denom = fB['oo'][m,m] + e_nf - omega
                        sigma_oob[i,j] -= 0.5*(vC['ooov'][m,n,i,f]*vC['ovoo'][j,f,m,n])/denom
                    # particle diagram
                    for e in range(sys['Nunocc_b']):
                        denom = e_nf - fB['vv'][e,e] + omega
                        sigma_oob[i,j] += 0.5*(vC['oovv'][j,n,e,f]*vC['vvoo'][e,f,j,n])/denom
    # populate full sigma matrix
    oa = slice(0,sys['Nocc_a'])
    ob = slice(0,sys['Nocc_b'])
    ua = slice(sys['Nocc_a'],sys['Nocc_a']+sys['Nunocc_a'])
    ub = slice(sys['Nocc_b'],sys['Nocc_b']+sys['Nunocc_b'])
    sigma_a[oa,oa] = sigma_ooa
    sigma_b[ob,ob] = sigma_oob

    return sigma_a, sigma_b

def gf2(nroot,ints,sys,maxit=50,tol=1.0e-08):

    fA = ints['fA']; fB = ints['fB'];
    omega = np.zeros(nroot)
    for p in range(sys['Nocc_a']-1,sys['Nocc_a']-nroot-1,-1):
        iroot = sys['Nocc_a'] - p
        print('\nStarting MBGF(2) iterations for root {}'.format(iroot))
        print("Koopman's estimate of IP energy = {:>8f} hartree".format(fA['oo'][p,p]))
        print('Iter        IP Energy       Residuum')
        print('========================================')
        e0 = fA['oo'][p,p]
        for it in range(maxit):
            sigma_a, _ = calc_mp2_selfenergy(e0,ints,sys)
            e1 = fA['oo'][p,p] + sigma_a[p,p]
            resid = e1-e0
            print('  {}          {:>8f}      {:>8f}'.format(it+1,e1,resid))
            if abs(resid) < tol:
                omega[iroot-1] = e1
                print('{}st IP root converged!\n'.format(iroot))
                break
            e0 = e1
        else:
            print('Failed to converge {}st IP root\n'.format(iroot))
    return omega

#def calc_mp2_selfenergy(omega,p,q,ints,sys):
#    """Calculates the matrix element \Sigma_{pq} of the self-energy
#    matrix to 2nd-order in MBPT.
#
#    Parameters
#    ----------
#    omega : float
#        Energy parameter of self-energy
#    p, q : int
#        Index of single-particle functions (MOs) outside the frozen core.
#    ints : dict
#        Collection of MO integrals defining the bare Hamiltonian H_N
#    sys : dict
#        System information dictionary
#
#    Returns
#    -------
#    sigma_mp2 : float
#        MP2 estimate of self-energy
#    """
#    # obtain the FULL integral matrices (these include core, occ, and unocc)
#    fA = ints['Fmat']['A']; fB = ints['Fmat']['B'];
#    vA = ints['Vmat']['A']; vB = ints['Vmat']['B']; vC = ints['Vmat']['C'];
#    sigma_mp2 = 0.0
#    N0 = sys['Nfroz'] # frozen spatial orbs
#    N1 = N0 + sys['Nocc_a'] # extent of occupied alpha orbs
#    N2 = N0 + sys['Nocc_b'] # extent of occupied beta orbs
#    N3 = N1 + sys['Nunocc_a'] # extent of unoccupied alpha orbs
#    N4 = N2 + sys['Nunocc_b'] # extent of unoccupied beta orbs
#    for n in range(N0,N1):
#        for f in range(N1,N3):
#            e_nf = fA[n,n] - fA[f,f]
#            # hole diagram
#            for m in range(N0,N1):
#                denom = fA[m,m] + e_nf - omega
#                sigma_mp2 -= 0.5*(vA[m,n,p+N0,f]*vA[q+N0,f,m,n])/denom
#            # particle diagram
#            for e in range(N1,N3):
#                denom = e_nf - fA[e,e] + omega
#                sigma_mp2 += 0.5*(vA[q+N0,n,e,f]*vA[e,f,p+N0,n])/denom
#    for n in range(N0,N2):
#        for f in range(N2,N4):
#            e_nf = fB[n,n] - fB[f,f]
#            # hole diagram

    
