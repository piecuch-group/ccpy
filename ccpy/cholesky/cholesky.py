import numpy as np
import time

def pivoted_chol(get_diag, get_row, rank_max, err_tol = 1e-6):
    """
    A simple python function which computes the Pivoted Cholesky decomposition of a 
    positive semi-definite operator. Only diagonal elements and select rows of the
    operator's matrix represenation are required.

    get_diag - A function which takes no arguments and returns the diagonal of the matrix when called.
    get_row - A function which takes 1 integer argument and returns the desired row (zero indexed).
    M - The maximum rank of the approximate decomposition; an integer. 
    err_tol - The maximum error tolerance, that is difference between the approximate decomposition and true matrix, allowed. 
              Note that this is in the Trace norm, not the spectral or frobenius norm. 

    Returns: R, an upper triangular matrix of column dimension equal to the target matrix. It's row dimension will 
                be at most M, but may be less if the termination condition was acceptably low error rather than max iters reached.
    """

    t_start = time.time()
    print("     Rank         Error")
    print("   ------------------------")


    d = np.copy(get_diag())
    N = len(d)

    piv = np.arange(N)

    R = np.zeros((rank_max, N))

    err = np.sum(np.abs(d))

    m = 0
    while (m < rank_max) and (err > err_tol):

        i = m + np.argmax(d[piv[m:]])

        tmp = piv[m]
        piv[m] = piv[i]
        piv[i] = tmp

        R[m, piv[m]] = np.sqrt(d[piv[m]])
        Apivm = get_row(piv[m])
        for i in range(m + 1, N):
            if m > 0:
                ip = np.dot(R[:m, piv[m]].T, R[:m, piv[i]])
            else:
                ip = 0
            R[m, piv[i]] = (Apivm[piv[i]] - ip) / R[m, piv[m]]
            d[piv[i]] -= R[m, piv[i]]**2

        err = np.sum(d[piv[m + 1:]])
        m += 1
        # print the rank and error every so often
        if m % 20 == 0:
            print("     ", m, "       ", np.round(err, 6))

    # Final cholesky vectors stored as R(s|pq), where s = 1,...,rank
    R = R[:m, :]
    print("   Final rank = ", m, "error = ", err)
    t_end = time.time()
    minutes, seconds = divmod(t_end - t_start, 60)
    print(f"   Completed Cholesky decomposition in {minutes} min {seconds} s")
    return R

def cholesky_eri_from_pyscf(mol, tol=1.0e-09):
    print("   ==========================")
    print("   ERI Cholesky Decomposition")
    print("   Error tolerance = ", tol)
    print("   ==========================")
    # Perform Cholesky decomposition of ERIs
    norb = mol.nao
    eri = mol.intor("int2e", aosym="s1").transpose(0, 2, 1, 3) # in physics notation
    # Use the unique set of 2-electron integrals <pq|rs>, for p<q and r<s
    ndim = int(norb * (norb + 1) / 2)
    # Loop through integrals to extract diagonal elements and array for p<q indexing
    index_pq = np.zeros((ndim, 2), dtype=np.int32)
    eri_diag = np.zeros(ndim)
    kout = 0
    for p in range(norb):
        for q in range(p, norb):
            index_pq[kout, 0] = p
            index_pq[kout, 1] = q
            eri_diag[kout] = eri[p, p, q, q]
            kout += 1
    # Define diagonal function
    get_diag = lambda: eri_diag.copy()
    # Define row function to obtain all <pi|qj> for a given p,q, where p<q and i<j
    def get_row(row):
        p = index_pq[row, 0]
        q = index_pq[row, 1]
        kout = 0
        eri_row = np.zeros(ndim)
        for i in range(norb):
            for j in range(i, norb):
                eri_row[kout] = eri[p, i, q, j]
                kout += 1
        return eri_row 
    # Perform the Cholesky decomposition to obtain R(x|pq), where pq is the composite index for p<q
    Rp = pivoted_chol(get_diag, get_row, rank_max=ndim, err_tol=tol)
    rank_chol = Rp.shape[0]
    # Unflatten Cholesky vectors into R(x|pq) defined for all p,q
    R = np.zeros((rank_chol, norb, norb))
    for m in range(rank_chol):
        for kout in range(ndim):
            p = index_pq[kout, 0]
            q = index_pq[kout, 1]
            R[m, p, q] = Rp[m, kout]
            R[m, q, p] = Rp[m, kout]
    return R

def cholesky_eri_from_file(eri_file, norb, tol=1.0e-09):

    print("   ERI Cholesky Decomposition")
    print("   Error tolerance = ", tol)

    # Perform Cholesky decomposition of ERIs using file saved on disk
    ndim = int(norb * (norb + 1) / 2)
    index_pq = np.zeros((ndim, 2), dtype=np.int32)
    eri_diag = np.zeros(ndim)

    kout = 0
    with open(eri_file, "r") as f:
        for line in f.readlines():
            kout += 1
    print("Found", kout, "integrals")

    # Open the file and read it to find diagonal elements
    kout = 0
    with open(eri_file, "r") as f:
        for line in f.readlines():
            L = line.split()
            p, q, r, s = [int(i) - 1 for i in L[:-1]]
            assert p > q
            erival = float(L[-1])
            # diagonal element <pp|qq> = [pq|pq]
            if p == q and r == s:
                print(kout, p, q, r, s)
                #index_pq[kout, 0] = p
                #index_pq[kout, 1] = q
                #eri_diag[kout] = erival
                kout += 1

    get_diag = lambda: eri_diag.copy()

    def get_row(row):
        p_row = index_pq[row, 0]
        q_row = index_pq[row, 1]
        kout = 0
        eri_row = np.zeros(ndim)

        with open(eri_file, "r") as f:
            L = line.split()
            p, q, r, s = [int(i) - 1 for i in L[:-1]]
            erival = float(L[-1])
            # element of row pq: <pq|rs> = <pi|qj>
            if p == p_row and r == q_row:
                eri_row[kout] = erival
                kout += 1

        return eri_row 

    Rp = pivoted_chol(get_diag, get_row, rank_max=ndim**2, err_tol=1.0e-09)
    rank_chol = Rp.shape[0]

    R = np.zeros((rank_chol, norb, norb))
    for m in range(rank_chol):
        for kout in range(ndim):
            p = index_pq[kout, 0]
            q = index_pq[kout, 1]
            R[m, p, q] = Rp[m, kout]
            R[m, q, p] = Rp[m, kout]

    return R

