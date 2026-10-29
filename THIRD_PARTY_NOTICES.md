# Third-party code in CCpy

## forte2

The atomic-orbital integral interface to libint2 and the native SCF solvers are adapted
from [forte2](https://github.com/evangelistalab/forte2) (Evangelista group, Emory
University), which is distributed under the GNU Lesser General Public License v3.0.
CCpy as a whole is distributed under the GNU General Public License v3.0 (see `COPYING`),
which is compatible with the LGPL-3.0.

Files adapted from forte2 carry a note in their header. They are:

- `ccpy/integrals/libint_interface.cc` (C++ integral library and its Python bindings)
- `ccpy/integrals/*.py` and `ccpy/integrals/symmetry/*.py` (molecule, basis sets,
  AutoAux, density fitting, point-group symmetry)
- `ccpy/scf/*.py` (RHF, ROHF, UHF, occupations, SCF gradients)

The files in `ccpy/integrals/basis_data/` come from the
[Basis Set Exchange](https://www.basissetexchange.org) (via forte2), except `6-31g.json`,
`6-31g_st_.json`, `6-31g_st__st_.json` and `dz.json`, which were converted from the [PySCF](https://github.com/pyscf/pyscf)
basis library (Apache-2.0).

## libint2

The integrals are computed with [libint2](https://github.com/evaleev/libint) (E. F.
Valeev, LGPL-3.0), which is linked at build time and not distributed with CCpy.
