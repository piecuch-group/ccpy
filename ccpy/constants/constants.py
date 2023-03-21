from enum import Enum

eVtohartree = 0.036749308136649
hartreetoeV = 1.0/(eVtohartree)

class LengthUnit(Enum):
    BOHR = 1
    ANGSTROM = 2

class Symmetry(Enum):
    C1 = 1
    C2V = 2
    D2H = 3


