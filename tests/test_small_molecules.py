from pyscf import scf, gto

import numpy as np


def test_h2o_ccpvdz():
    mol_h2o = gto.M(atom = 'O 0 0 0; H 0 1 0; H 0 0 1', basis = 'ccpvdz')
    rhf_h2o = scf.RHF(mol_h2o)
    e_h2o = rhf_h2o.kernel()

    assert np.allclose(e_h2o, -76.0167894720692)

