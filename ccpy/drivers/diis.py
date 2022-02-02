import numpy as np


class DIIS:
    def __init__(self, T, diis_size, out_of_core):

        self.diis_size = diis_size
        self.out_of_core = out_of_core
        self.ndim = T.ndim

        if self.out_of_core:
            self.T_list = np.memmap(
                "t.npy", mode="w+", dtype=T.a.dtype, shape=(self.ndim, self.diis_size)
            )
            self.T_residuum_list = np.memmap(
                "dt.npy", mode="w+", dtype=T.a.dtype, shape=(self.ndim, self.diis_size)
            )
            self.flush()
        else:
            self.T_list = np.zeros((self.ndim, diis_size))
            self.T_residuum_list = np.zeros((self.ndim, diis_size))

    def push(self, T, T_residuum, iteration):
        self.T_list[:, iteration % self.diis_size] = T.flatten()
        self.T_residuum_list[:, iteration % self.diis_size] = T_residuum.flatten()
        if self.out_of_core:
            self.flush()

    def flush(self):
        self.T_list.flush()
        self.T_residuum_list.flush()

    def extrapolate(self):

        B_dim = self.diis_size + 1
        B = -1.0 * np.ones((B_dim, B_dim))

        nhalf = int(self.ndim / 2)
        for i in range(self.diis_size):
            for j in range(i, self.diis_size):
                B[i, j] = np.dot(
                    self.T_residuum_list[:nhalf, i], self.T_residuum_list[:nhalf, j]
                )
                B[i, j] += np.dot(
                    self.T_residuum_list[nhalf:, i], self.T_residuum_list[nhalf:, j]
                )
                B[j, i] = B[i, j]
        B[-1, -1] = 0.0

        rhs = np.zeros(B_dim)
        rhs[-1] = -1.0

        # TODO: replace with numpy.linalg.solve
        coeff = solve_gauss(B, rhs)
        x_xtrap = np.zeros(self.ndim)
        for i in range(self.diis_size):
            x_xtrap += coeff[i] * self.T_list[:, i]

        return x_xtrap


# TODO: Is this really needed? scipy/numpy does it better
def solve_gauss(A, b):
    """DIIS helper function. Solves the linear system Ax=b using
    Gaussian elimination"""
    n = A.shape[0]
    for i in range(n - 1):
        for j in range(i + 1, n):
            m = A[j, i] / A[i, i]
            A[j, :] -= m * A[i, :]
            b[j] -= m * b[i]
    x = np.zeros(n)
    k = n - 1
    x[k] = b[k] / A[k, k]
    while k >= 0:
        x[k] = (b[k] - np.dot(A[k, k + 1 :], x[k + 1 :])) / A[k, k]
        k = k - 1

    return x
