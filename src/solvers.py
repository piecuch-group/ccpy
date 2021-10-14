import numpy as np

def diis(x_list, diis_resid):

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

def gramschmidt(X):
    d, n = X.shape
    m = min(d,n)
    Q = np.zeros((d,m))
    for i in range(m):
        v = X[:,i]
        for j in range(i):
            m = np.dot(Q[:,j].T,v)
            v -= m*Q[:,j]
        Q[:,i] = v/np.linalg.norm(v)
    return Q

def davidson_eomcc(A,H1A,H1B,vec_dim,nroot,sys,nvec=6,maxit=100,tol=1e-07,thresh_vec=1e-04):

    import scipy.linalg

    def _diagonal_guess(nroot,H1A,H1B,sys):
        for i in range(sys['Nocc_a']):
            for a in range(sys['Nunocc_a']):
                D1A[ct] = H1A['vv'][a,a] - H1A['oo'][i,i]
        for i in range(sys['Nocc_b']):
            for a in range(sys['Nunocc_b']):
                D1B = H1B['vv'][a,a] - H1B['oo'][i,i]
        idx = np.argsort(D)
        I = np.eye(len(D))
        return I[:,idx[:nroot]]

    def _orthogonalize_root(r,B):
        for i in range(B.shape[1]):
            b = B[:,i]/np.linalg.norm(B[:,i])
            r -= np.dot(b.T,r)*b
        return r

    max_size = nroot*nvec
    SIGMA = np.zeros((vec_dim,max_size))
    B = np.zeros((vec_dim,max_size))
    add_B = np.zeros((vec_dim,nroot))
    
    B[:,:nroot] = _diagonal_guess(nroot,H1A,H1B)

    curr_size = nroot
    for it in range(maxit):

        B0 = gramschmidt(B[:,:curr_size])
        B[:,:curr_size] = B0

        nprev = np.max(curr_size-nroot,0)
        sigma = A.matmat(B[:,nprev:curr_size])
        SIGMA[:,nprev:curr_size] = sigma
        G = np.dot(B[:,:curr_size].T,SIGMA[:,:curr_size])
        
        eval_E, alpha = scipy.linalg.eig(G)

        idx = np.argsort(eval_E)
        eigval = eval_E[idx[:nroot]]
        alpha = alpha[:,idx[:nroot]]
        V = np.dot(B[:,:curr_size],alpha)

        ct_add = 0
        resid_norm = np.zeros(nroot)
        print('\nIter - {}    Subspace Dim = {}'.format(it+1,curr_size))
        print('----------------------------------------------------')
        for j in range(nroot):
            r = np.dot(SIGMA[:,:curr_size],alpha[:,j]) - V[:,j]*eigval[j]
            resid_norm[j] = np.linalg.norm(r)
            # HERE
            #r1a = np.reshape(r[])
            q = _update_R(r,eigval[j],D)
            q = q/np.linalg.norm(q)
            if ct_add > 0:
                q_orth = _orthogonalize_root(q,np.concatenate((B[:,:curr_size],add_B[:,:ct_add]),axis=1))
            else:
                q_orth = _orthogonalize_root(q,B[:,:curr_size])
            if np.linalg.norm(q_orth) > thresh_vec:
                add_B[:,ct_add] = q_orth/np.linalg.norm(q_orth)
                ct_add += 1

            print('Root - {}      e = {:.8f}      |r| = {:.8f}'.format(j+1,eigval[j],resid_norm[j]))

        if all(resid_norm <= tol):
            print('\nDavidson successfully converged!')
            break
        else:
            if curr_size >= max_size:
                print('Restarting and collapsing')
                B[:,:nroot] = np.dot(B,alpha)
                curr_size = nroot
            else:
                B[:,curr_size:curr_size+ct_add] = add_B[:,:ct_add]
                curr_size += ct_add

    return eigval, V
