from pytest import mark, approx
from pyscf import scf, gto

import numpy as np

def build_pyscf_molecule(geometry):
    strings = [
        f"{element} {x:.6f} {y:.6f} {z:.6f}"
        for element, x, y, z in geometry
    ]
    return ";".join(strings)


# https://doi.org/10.1063/1.471518
H2O_OLSEN = {
    "metadata": {
        "symmetry": "C2V",
        "unit": "bohr?"
    },
    "geometries": {
        1.0: [("O", 0.0, 0.0, -0.0090),
              ("H", 0.0,  1.515_263, -1.058_898),
              ("H", 0.0, -1.515_263, -1.058_898)],
        1.5: [("O", 0.0, 0.0, -0.0135),
              ("H", 0.0,  2.272_894_5, -1.588_347),
              ("H", 0.0, -2.272_894_5, -1.588_347)],
        2.0: [("O", 0.0, 0.0, -0.0180),
              ("H", 0.0,  3.030_526, -2.117_796),
              ("H", 0.0, -3.030_526, -2.117_796)],
    }
}


@mark.parametrize(
    "geometry_indx,basis,energy",
    [
        (1.0, "ccpvdz", np.nan),
        (1.5, "ccpvdz", np.nan)
    ]
)
def test_h2o(geometry_indx, basis, energy):
    atom = build_pyscf_molecule(H2O_OLSEN["geometries"][geometry_indx])
    mol_h2o = gto.M(atom=atom, basis=basis)
    rhf_h2o = scf.RHF(mol_h2o)
    e_h2o = rhf_h2o.kernel()

    assert e_h2o == approx(energy)


F2 = {
    "metadata": {
        "symmetry": "D2H",
        "unit": "bohr"
    },
    1.0: [("F", 0.0, 0.0, -1.334_08),
          ("F", 0.0, 0.0,  1.334_08)],
}


@mark.parametrize(
    "geometry_indx,basis,energy",
    [
        (1.0, "ccpvdz", np.nan),
    ]
)
def test_f2(geometry_indx, basis, energy):
    atom = build_pyscf_molecule(F2["geometry"][geometry_indx])
    mol = gto.M(atom=atom, basis=basis)
    rhf = scf.RHF(mol)
    e = rhf.kernel()

    assert np.allclose(e, energy)
