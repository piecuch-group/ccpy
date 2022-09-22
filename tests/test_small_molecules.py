from pytest import mark, approx
from pyscf import scf, gto

import numpy as np

from ccpy.models.molecule import Molecule
from ccpy.models.calculation import Calculation
from ccpy.interfaces.pyscf_tools import load_pyscf_integrals
from ccpy.drivers.driver import cc_driver


def build_pyscf_molecule(molecule):
    strings = [
        f"{element.name} {xyz.x:.6f} {xyz.y:.6f} {xyz.z:.6f}"
        for element, xyz in molecule.geometry.structure
    ]
    string = ";".join(strings)

    return gto.M(atom=string,
                 basis=molecule.basis,
                 symmetry=molecule.geometry.symmetry,
                 unit=molecule.geometry.unit)


MOLECULES = {
    "h2o_olsen": {  # https://doi.org/10.1063/1.471518
        "metadata": {
            "symmetry": "C2V",
            "unit": "bohr"
        },
            1.0: [("O", 0.0, 0.0, -0.0090),
                  ("H", 0.0,  1.515_263, -1.058_898),
                  ("H", 0.0, -1.515_263, -1.058_898)],
            1.5: [("O", 0.0, 0.0, -0.0135),
                  ("H", 0.0,  2.272_894_5, -1.588_347),
                  ("H", 0.0, -2.272_894_5, -1.588_347)],
            2.0: [("O", 0.0, 0.0, -0.0180),
                  ("H", 0.0,  3.030_526, -2.117_796),
                  ("H", 0.0, -3.030_526, -2.117_796)],
    },

    "f2": {
        "metadata": {
            "symmetry": "D2H",
            "unit": "bohr"
        },
            1.0: [("F", 0.0, 0.0, -1.334_08),
                  ("F", 0.0, 0.0,  1.334_08)],
    }
}

REFERENCE = {
    "h2o_olsen": {
#        1.0: {
#            "ccsd": {
#                "huzinaga": 2.2,
#                "dz": 1.1,
#                "cc-pvdz": 1.1,
#                "cc-pvtz": 1.1
#            }
#        },
#
#        1.5: {
#            "ccsd": {
#                "huzinaga": 2.2,
#                "dz": 1.1,
#                "cc-pvdz": 1.1,
#                "cc-pvtz": 1.1
#            }
#        },
#
        2.0: {
            "reference": {
                "cc-pvdz": -75.5877112496
            },
            "ccsd": {
#                "huzinaga": 2.2,
#                "dz": 1.1,
                "cc-pvdz": -75.9296328657,
#                "cc-pvtz": 1.1
            }
        }
    }
}

def get_molecule(molecule_key, geometry_key, basis_set):
    metadata = MOLECULES[molecule_key]["metadata"]
    structure = MOLECULES[molecule_key][geometry_key]
    molecule = {
        "name": molecule_key,
        "basis": basis_set,
        "geometry": {
            "structure": structure,
            "unit": metadata["unit"],
            "symmetry": metadata["symmetry"]
        }
    }

    return Molecule.from_dict(molecule)


def get_tests():
    tests = []
    for molecule_key, geometries in REFERENCE.items():
        for geometry_key, methods in geometries.items():
            for method_key, basis_sets in methods.items():
                if method_key == "reference":
                    continue

                for basis_set, energy in basis_sets.items():
                    molecule = get_molecule(molecule_key,
                                            geometry_key,
                                            basis_set)
                    if "reference" in methods:
                        reference = methods["reference"].get(basis_set, None)
                    else:
                        reference = None
                    tests.append(
                        (molecule, basis_set, reference, energy)
                    )

    return tests


@mark.parametrize(
    "molecule, basis, reference, energy",
    get_tests()
)
def test_molecule(molecule, basis, reference, energy):
    mol = build_pyscf_molecule(molecule)
    mf = scf.RHF(mol)
    e = mf.kernel()

    if reference:
        assert e == approx(reference)

    system, H = load_pyscf_integrals(mf, nfrozen=0)
    calculation = Calculation(
        order=2,
        calculation_type="ccsd",
        convergence_tolerance=1.0e-8,
        diis_size=6
    )

    T, total_energy, is_converged = cc_driver(calculation, system, H)

    assert total_energy == approx(energy)

