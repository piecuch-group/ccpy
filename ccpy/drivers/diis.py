import numpy as np
import h5py
from ccpy.utilities.utilities import remove_file


class DIIS:
    def __init__(self, T, diis_size, out_of_core, data_type=np.float64):

        self.diis_size = diis_size
        self.out_of_core = out_of_core
        self.ndim = T.ndim
        self.dtype = data_type

        if self.out_of_core:
            remove_file("cc-diis-vectors.hdf5")
            f = h5py.File("cc-diis-vectors.hdf5", "w")
            self.T_list = f.create_dataset("t-vectors", (self.diis_size, self.ndim), dtype=self.dtype)
            self.T_residuum_list = f.create_dataset("resid-vectors", (self.diis_size, self.ndim), dtype=self.dtype)
        else:
            self.T_list = np.zeros((self.diis_size, self.ndim), dtype=self.dtype)
            self.T_residuum_list = np.zeros((self.diis_size, self.ndim), dtype=self.dtype)

    def cleanup(self):
        if self.out_of_core:
            remove_file("cc-diis-vectors.hdf5")
            
    def push(self, T, T_residuum, iteration):
            self.T_list[iteration % self.diis_size, :] = T.flatten()
            self.T_residuum_list[iteration % self.diis_size, :] = T_residuum.flatten()

    def extrapolate(self):
        B_dim = self.diis_size + 1
        B = -1.0 * np.ones((B_dim, B_dim), dtype=self.dtype)
        for i in range(self.diis_size):
            for j in range(i, self.diis_size):
                B[i, j] = np.dot(np.conj(self.T_residuum_list[i, :].T), self.T_residuum_list[j, :])
                B[j, i] = B[i, j].conj()
        B[-1, -1] = 0.0

        rhs = np.zeros(B_dim, dtype=self.dtype)
        rhs[-1] = -1.0

        # TODO: replace with numpy.linalg.solve
        # TODO: replace with scipy.linalg.lu
        coeff = self.solve_gauss(B, rhs)
        x_xtrap = np.zeros(self.ndim, dtype=self.dtype)
        for i in range(self.diis_size):
            x_xtrap += coeff[i] * self.T_list[i, :]

        return x_xtrap

    def solve_gauss(self, A, b):
        """DIIS helper function. Solves the linear system Ax=b using
        Gaussian elimination"""
        n = A.shape[0]
        for i in range(n - 1):
            for j in range(i + 1, n):
                m = A[j, i] / A[i, i]
                A[j, :] -= m * A[i, :]
                b[j] -= m * b[i]
        x = np.zeros(n, dtype=self.dtype)
        k = n - 1
        x[k] = b[k] / A[k, k]
        while k >= 0:
            x[k] = (b[k] - np.dot(A[k, k + 1 :], x[k + 1 :])) / A[k, k]
            k = k - 1

        return x
