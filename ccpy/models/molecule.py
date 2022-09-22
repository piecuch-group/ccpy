from typing import Optional
from pydantic import BaseModel

from .geometry import Geometry

class Molecule(BaseModel):
    name: str
    basis: str
    geometry: Geometry

    smiles: Optional[str]

    @classmethod
    def from_dict(cls, molecule):
        return cls(
            name=molecule["name"],
            basis=molecule["basis"],
            geometry=Geometry.from_dict(molecule["geometry"]))

