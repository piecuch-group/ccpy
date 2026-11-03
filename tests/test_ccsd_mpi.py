"""
Test script: compare serial CCSD vs MPI-parallel CCSD.

Run with:
    mpirun -np 2 python tests/test_ccsd_mpi.py

Usage example for a real production script::

    from pyscf import gto, scf
    from ccpy import Driver, mpi_suppress_stdout

    mpi_suppress_stdout()   # call once — silences non-root ranks

    mol = gto.M(atom=..., basis="cc-pvdz", symmetry="C2V")
    mf = scf.RHF(mol)
    mf.kernel()

    driver = Driver.from_pyscf(mf, nfrozen=2)
    driver.system.print_info()
    driver.run_cc_mpi(method="ccsd")
"""

import numpy as np
from pyscf import gto, scf
from ccpy import Driver, mpi_suppress_stdout

# Silence non-root ranks for the entire script
mpi_suppress_stdout()

# ================================================================
# RHF test — H2O / STO-3G
# ================================================================
mol = gto.M(
    atom=[["O", (0.0, 0.0, -0.1173)],
          ["H", (0.0, 0.7572, 0.4692)],
          ["H", (0.0, -0.7572, 0.4692)]],
    basis="sto-3g",
    symmetry="C2V",
    unit="Angstrom",
    spin=0,
    verbose=0,
)
mf = scf.RHF(mol)
mf.kernel()

driver_serial = Driver.from_pyscf(mf, nfrozen=0)
driver_serial.run_cc(method="ccsd")
e_serial = driver_serial.correlation_energy

driver_mpi = Driver.from_pyscf(mf, nfrozen=0)
driver_mpi.run_cc_mpi(method="ccsd")
e_mpi = driver_mpi.correlation_energy

diff = abs(e_serial - e_mpi)
assert diff < 1e-10, f"RHF FAIL: energies differ by {diff:.2e}"
t_diff = np.max(np.abs(driver_serial.T.flatten() - driver_mpi.T.flatten()))
assert t_diff < 1e-10, f"RHF FAIL: amplitudes differ by {t_diff:.2e}"

# ================================================================
# UHF test — OH radical / STO-3G  (exercises update_t2c)
# ================================================================
mol_uhf = gto.M(
    atom=[["O", (0.0, 0.0, 0.0)],
          ["H", (0.0, 0.0, 0.97)]],
    basis="sto-3g",
    symmetry="C2V",
    unit="Angstrom",
    spin=1,
    verbose=0,
)
mf_uhf = scf.UHF(mol_uhf)
mf_uhf.kernel()

driver_s2 = Driver.from_pyscf(mf_uhf, nfrozen=0, uhf=True)
driver_s2.run_cc(method="ccsd")
e_s2 = driver_s2.correlation_energy

driver_m2 = Driver.from_pyscf(mf_uhf, nfrozen=0, uhf=True)
driver_m2.run_cc_mpi(method="ccsd")
e_m2 = driver_m2.correlation_energy

diff2 = abs(e_s2 - e_m2)
assert diff2 < 1e-10, f"UHF FAIL: energies differ by {diff2:.2e}"
t_diff2 = np.max(np.abs(driver_s2.T.flatten() - driver_m2.T.flatten()))
assert t_diff2 < 1e-10, f"UHF FAIL: amplitudes differ by {t_diff2:.2e}"

# ================================================================
print("test_ccsd_mpi: ALL PASSED  "
      f"(RHF dE={diff:.1e} dT={t_diff:.1e}  "
      f"UHF dE={diff2:.1e} dT={t_diff2:.1e})")
