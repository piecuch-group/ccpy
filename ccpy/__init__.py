"""A Python framework for coupled-cluster computations of molecular systems"""

# Import the specific modules that form the CCpy API here
#from . import *
from ccpy.drivers.driver import Driver
from ccpy.drivers.adaptive import AdaptDriver, AdaptEOMDriver, AdaptEOMDriverSS
from ccpy.utilities.pspace import (get_active_triples_pspace,
                                   get_active_3h2p_pspace,
                                   get_active_3p2h_pspace,
                                   get_pspace_from_qmc,
                                   get_triples_pspace_from_cipsi,
                                   get_quadruples_pspace_from_cipsi,
)

# Handle versioneer
from ._version import get_versions

versions = get_versions()
__version__ = versions["version"]
__git_revision__ = versions["full-revisionid"]
del get_versions, versions
