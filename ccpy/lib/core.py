try:
    from ._fortran import *
except ImportError:
    from .fortran import *
