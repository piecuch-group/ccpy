import numpy as np
import time
from ccpy.utilities.utilities import get_memory_usage

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

    t_start = time.perf_counter()
    t_cpu_start = time.process_time()
    print("     Rank         Error         Memory")
    print("   -----------------------------------")


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
            print("     ", m, "       ", np.round(err, 10), "       ", get_memory_usage(), "MB")

    # Final cholesky vectors stored as R(s|pq), where s = 1,...,rank
    R = R[:m, :]
    print("   Final rank = ", m, "error = ", err)
    t_end = time.perf_counter()
    minutes, seconds = divmod(t_end - t_start, 60)
    print(f"   Completed Cholesky decomposition in {minutes} min {seconds} s")
    print(f"   Total CPU time is {time.process_time() - t_cpu_start} seconds")
    print(f"   Current memory used {get_memory_usage()} MB")
    return R

# def cholesky_eri_from_pyscf(mol, tol=1.0e-09, cmax=10):
#     print("   ==========================")
#     print("   ERI Cholesky Decomposition")
#     print("   Error tolerance = ", tol)
#     print("   ==========================")
#     # Perform Cholesky decomposition of ERIs
#     norb = mol.nao
#     eri = mol.intor("int2e", aosym="s1").transpose(0, 2, 1, 3) # in physics notation
#     # Use the unique set of 2-electron integrals <pq|rs>, for p<=q and r<=s
#     ndim = int(norb * (norb + 1) / 2)
#     # Loop through integrals to extract diagonal elements and array for p<=q indexing
#     index_pq = np.zeros((ndim, 2), dtype=np.int32)
#     eri_diag = np.zeros(ndim)
#     kout = 0
#     for p in range(norb):
#         for q in range(p, norb):
#             index_pq[kout, 0] = p
#             index_pq[kout, 1] = q
#             eri_diag[kout] = eri[p, p, q, q]
#             kout += 1
#     # Define diagonal function
#     get_diag = lambda: eri_diag.copy()
#     # Define row function to obtain all <pi|qj> for a given p,q, where p<q and i<j
#     def get_row(row):
#         p = index_pq[row, 0]
#         q = index_pq[row, 1]
#         kout = 0
#         eri_row = np.zeros(ndim)
#         for i in range(norb):
#             for j in range(i, norb):
#                 eri_row[kout] = eri[p, i, q, j]
#                 kout += 1
#         return eri_row
#     # Perform the Cholesky decomposition to obtain R(x|pq), where pq is the composite index for p<q
#     Rp = pivoted_chol(get_diag, get_row, rank_max=ndim, err_tol=tol)
#     rank_chol = Rp.shape[0]
#     # Unflatten Cholesky vectors into R(x|pq) defined for all p,q
#     R = np.zeros((rank_chol, norb, norb))
#     for m in range(rank_chol):
#         for kout in range(ndim):
#             p = index_pq[kout, 0]
#             q = index_pq[kout, 1]
#             R[m, p, q] = Rp[m, kout]
#             R[m, q, p] = Rp[m, kout]
#     return R

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

def cholesky_eri_from_pyscf(mol, tol=1e-5, verbose=True, cmax=10):
    """Modified cholesky decomposition from pyscf eris.

    See, e.g. [Motta17]_

    Only works for molecular systems.

    Parameters
    ----------
    orthoAO: :class:`numpy.ndarray`
        Orthogonalising matrix for AOs. (e.g., mo_coeff).
    delta : float
        Accuracy desired.
    verbose : bool
        If true print out convergence progress.
    cmax : int
        nchol = cmax * M, where M is the number of basis functions.
        Controls buffer size for cholesky vectors.

    Returns
    -------
    chol_vecs : :class:`numpy.ndarray`
        Matrix of cholesky vectors in AO basis.
    """
    # This function only works with spherical basis sets, not Cartesian!

    t_start = time.perf_counter()
    t_cpu_start = time.process_time()
    nao = mol.nao_nr()
    diag = np.zeros(nao * nao)
    nchol_max = cmax * nao

    print("   =======================================")
    print("Generating Cholesky decomposition of ERI" % nchol_max)
    print("Max number of Cholesky vectors = %d" % nchol_max)
    print("# current memory usage: %f MB" % (get_memory_usage()))
    print("   =======================================")

    print("     Rank           Error            Memory")
    print("   -----------------------------------------")

    # This shape is more convenient for pauxy.
    chol_vecs = np.zeros((nchol_max, nao * nao))
    ndiag = 0
    dims = [0]
    nao_per_i = 0
    for i in range(0, mol.nbas):
        l = mol.bas_angular(i)
        nc = mol.bas_nctr(i)
        nao_per_i += (2 * l + 1) * nc
        dims.append(nao_per_i)
    for i in range(0, mol.nbas):
        shls = (i, i + 1, 0, mol.nbas, i, i + 1, 0, mol.nbas)
        buf = mol.intor('int2e_sph', shls_slice=shls)
        di, dk, dj, dl = buf.shape
        diag[ndiag:ndiag + di * nao] = buf.reshape(di * nao, di * nao).diagonal()
        ndiag += di * nao
    nu = np.argmax(diag)
    delta_max = diag[nu]
    j = nu // nao
    l = nu % nao
    sj = np.searchsorted(dims, j)
    sl = np.searchsorted(dims, l)
    if dims[sj] != j and j != 0:
        sj -= 1
    if dims[sl] != l and l != 0:
        sl -= 1
    Mapprox = np.zeros(nao * nao)
    # ERI[:,jl]
    eri_col = mol.intor('int2e_sph', shls_slice=(0, mol.nbas, 0, mol.nbas, sj, sj + 1, sl, sl + 1))
    cj, cl = max(j - dims[sj], 0), max(l - dims[sl], 0)
    chol_vecs[0] = np.copy(eri_col[:, :, cj, cl].reshape(nao * nao)) / delta_max ** 0.5

    nchol = 0
    while abs(delta_max) > tol:
        # Update cholesky vector
        # start = time.time()
        # M'_ii = L_i^x L_i^x
        Mapprox += chol_vecs[nchol] * chol_vecs[nchol]
        # D_ii = M_ii - M'_ii
        delta = diag - Mapprox
        nu = np.argmax(np.abs(delta))
        delta_max = np.abs(delta[nu])
        # Compute ERI chunk.
        # shls_slice computes shells of integrals as determined by the angular
        # momentum of the basis function and the number of contraction
        # coefficients. Need to search for AO index within this shell indexing
        # scheme.
        # AO index.
        j = nu // nao
        l = nu % nao
        # Associated shell index.
        sj = np.searchsorted(dims, j)
        sl = np.searchsorted(dims, l)
        if dims[sj] != j and j != 0:
            sj -= 1
        if dims[sl] != l and l != 0:
            sl -= 1
        # Compute ERI chunk.
        eri_col = mol.intor('int2e_sph', shls_slice=(0, mol.nbas, 0, mol.nbas, sj, sj + 1, sl, sl + 1))
        # Select correct ERI chunk from shell.
        cj, cl = max(j - dims[sj], 0), max(l - dims[sl], 0)
        Munu0 = eri_col[:, :, cj, cl].reshape(nao * nao)
        # Updated residual = \sum_x L_i^x L_nu^x
        R = np.dot(chol_vecs[:nchol + 1, nu], chol_vecs[:nchol + 1, :])
        chol_vecs[nchol + 1] = (Munu0 - R) / (delta_max) ** 0.5
        nchol += 1
        # step_time = time.time() - start
        print("     ", nchol, "       ", np.round(delta_max, 10), "       ", np.round(get_memory_usage(), 0), "MB")

    chol_vecs = chol_vecs[:nchol].reshape(nchol, nao, nao)

    print("   Final rank = ", nchol, "error = ", delta_max)
    t_end = time.perf_counter()
    minutes, seconds = divmod(t_end - t_start, 60)
    print(f"   Completed Cholesky decomposition in {minutes} min {seconds} s")
    print(f"   Total CPU time is {time.process_time() - t_cpu_start} seconds")
    print(f"   Current memory used {get_memory_usage()} MB")

    return chol_vecs

