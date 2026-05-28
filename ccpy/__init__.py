"""A Python framework for coupled-cluster computations of molecular systems"""

# Import the specific modules that form the CCpy API here
#from . import *
from ccpy.drivers.driver import Driver
from ccpy.drivers.adaptive import AdaptDriver, AdaptEOMDriver, AdaptEOMDriverSS
from ccpy.utilities.pspace import (get_active_triples_pspace,
                                   get_active_3h2p_pspace,
                                   get_active_3p2h_pspace,
                                   get_active_4p2h_pspace,
                                   get_pspace_from_qmc,
                                   get_triples_pspace_from_cipsi,
                                   get_quadruples_pspace_from_cipsi,
)


def mpi_suppress_stdout():
    """Silence ``stdout`` on every MPI rank except rank 0.

    Call this once near the top of a script that is launched with
    ``mpirun`` / ``mpiexec``.  It prevents duplicate output from
    ``Driver.from_pyscf()``, ``system.print_info()``, ``run_cc()``,
    user ``print()`` calls, etc.  ``run_cc_mpi()`` already handles
    its own suppression, but this function covers everything else.

    Example::

        from ccpy import Driver, mpi_suppress_stdout

        mpi_suppress_stdout()          # call once at top of script

        driver = Driver.from_pyscf(mf, nfrozen=0)
        driver.system.print_info()     # printed only on rank 0
        driver.run_cc_mpi()            # MPI-parallel CCSD
    """
    import os, sys
    from mpi4py import MPI
    if MPI.COMM_WORLD.Get_rank() != 0:
        sys.stdout = open(os.devnull, "w")

# Handle versioneer
from ._version import get_versions

versions = get_versions()
__version__ = versions["version"]
__git_revision__ = versions["full-revisionid"]
del get_versions, versions
