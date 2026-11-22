import numpy as np
import h5py
import tempfile
from ccpy.utilities.utilities import remove_file

class DIIS:
    def __init__(self, T, diis_size, out_of_core):

        ftmp = tempfile.NamedTemporaryFile()
        self.diis_size = diis_size
        self.out_of_core = out_of_core
        self.ndim = T.ndim
        self.data_type = T.a.dtype
        self.file_name = ftmp.name

        if self.out_of_core:
            remove_file(self.file_name)
            f = h5py.File(self.file_name, "w")
            self.T_list = f.create_dataset("t-vectors", (self.diis_size, self.ndim), dtype=self.data_type)
            self.T_residuum_list = f.create_dataset("resid-vectors", (self.diis_size, self.ndim), dtype=self.data_type)
        else:
            self.T_list = np.zeros((self.diis_size, self.ndim), dtype=self.data_type)
            self.T_residuum_list = np.zeros((self.diis_size, self.ndim), dtype=self.data_type)
        # residual overlaps, updated one row per push so each stored vector is read once per push
        self.B = np.zeros((self.diis_size, self.diis_size), dtype=self.data_type)

    def cleanup(self):
        if self.out_of_core:
            remove_file(self.file_name)
            
    def push(self, T, T_residuum, iteration):
            k = iteration % self.diis_size
            self.T_list[k, :] = T.flatten()
            r = T_residuum.flatten()
            self.T_residuum_list[k, :] = r
            for j in range(self.diis_size):
                rj = r if j == k else self.T_residuum_list[j, :]
                # B[i, j] = <r_i|r_j> for i <= j, mirrored (not conjugated) below the diagonal
                self.B[k, j] = self.B[j, k] = np.vdot(r, rj) if k <= j else np.vdot(rj, r)

    def extrapolate(self):
        B_dim = self.diis_size + 1
        B = -1.0 * np.ones((B_dim, B_dim), self.data_type)
        B[:-1, :-1] = self.B
        B[-1, -1] = 0.0

        rhs = np.zeros(B_dim, dtype=self.data_type)
        rhs[-1] = -1.0

        # TODO: replace with numpy.linalg.solve
        # TODO: replace with scipy.linalg.lu
        coeff = self.solve_gauss(B, rhs)
        x_xtrap = np.zeros(self.ndim, dtype=self.data_type)
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
        x = np.zeros(n, dtype=self.data_type)
        k = n - 1
        x[k] = b[k] / A[k, k]
        while k >= 0:
            x[k] = (b[k] - np.dot(A[k, k + 1 :], x[k + 1 :])) / A[k, k]
            k = k - 1

        return x
