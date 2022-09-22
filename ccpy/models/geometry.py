from typing import Any, List, Tuple

from pydantic import BaseModel

from ccpy.constants import LengthUnit, Symmetry


class Atom:
    def __init__(self, name, number=None):
        self.name = name


class Coordinates(BaseModel):
    x: float
    y: float
    z: float

    def __getitem__(self, key):
        point = (self.x, self.y, self.z)
        return point[key]


class Geometry(BaseModel):
    unit: Any
    symmetry: Any
    structure: Any

    @property
    def atoms(self):
        return [a for a, c in self.structure]

    @property
    def xyz(self):
        return [(*c,) for a, c in self.structure]

    @classmethod
    def from_dict(cls, geometry):
        structure = []
        for atom in geometry["structure"]:
            structure.append(
                (Atom(atom[0]), Coordinates(x=atom[1], y=atom[2],
                                            z=atom[3])
                 )
            )

        return cls(unit=geometry["unit"],
                   symmetry=geometry["symmetry"],
                   structure=structure)
