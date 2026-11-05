"""SUMRCCSD through the methods API: the one-reference limit is DF-CCSD, and for two
electrons the roots are FCI eigenvalues of the same DF Hamiltonian."""
import numpy as np
import pytest
from conftest import AUX, OPTS, WATER, converged_scf

from ccpy import CCSD, Molecule, RHF, SUMRCCSD, SUMRCCSD1


def test_single_reference_is_ccsd():
    mf = converged_scf(RHF, Molecule(WATER, "6-31g", auxiliary_basis=AUX, verbose=False))
    e_mr = SUMRCCSD(mf, 0, 0, nfrozen=1, **OPTS).energy()
    assert e_mr.shape == (1,)
    assert abs(e_mr[0] - CCSD(mf, nfrozen=1, **OPTS).energy()) < 1e-8


def test_two_electrons_is_exact():
    pyscf_fci = pytest.importorskip("pyscf.fci")
    mf = converged_scf(RHF, Molecule("H 0 0 0; H 0 0 0.9", "6-31g**", auxiliary_basis=AUX, verbose=False))
    mr = SUMRCCSD(mf, 2, 2, **OPTS)
    roots = mr.energy()
    B = mr.reference.B_mo["a"]
    eri = np.einsum("xpq,xrs->pqrs", B, B)
    n = eri.shape[0]
    fci = pyscf_fci.direct_spin1.FCI().kernel(mr.reference.h_mo["a"], eri, n, (1, 1), nroots=n * n)[0]
    fci = fci + mr.system.nuclear_repulsion
    assert len(roots) == 4 and np.isclose(mr.energy(0), roots.min())
    for e in roots:
        assert np.abs(fci - e).min() < 1e-8


def test_frozen_core_virtual_variant_runs():
    mf = converged_scf(RHF, Molecule("H 0 0 0; H 1.06 0 0; H 0 1.27 0; H 1.06 1.27 0", "6-31g",
                                     auxiliary_basis=AUX, verbose=False))
    full, frozen = (cls(mf, 2, 2, **OPTS).energy() for cls in (SUMRCCSD, SUMRCCSD1))
    # SUMRCCSD(1) only approximates the core-virtual amplitudes: close to, but not equal to, SUMRCCSD
    assert 1e-5 < np.abs(full - frozen).max() < 1e-2


def test_determinant_list():
    """A CAS given as a determinant list is the CAS; one determinant fewer keeps the
    two-electron exactness (C conditions); arguments are validated."""
    pyscf_fci = pytest.importorskip("pyscf.fci")
    mf = converged_scf(RHF, Molecule("H 0 0 0; H 0 0 0.9", "6-31g**", auxiliary_basis=AUX, verbose=False))
    cas = SUMRCCSD(mf, 2, 2, **OPTS)
    dets = [[1, 1], [1, 2], [2, 1], [2, 2]]  # N_alpha = N_beta = 1: [alpha orbital, beta orbital]
    assert np.allclose(SUMRCCSD(mf, determinants=dets, **OPTS).energy(), cas.energy(), atol=1e-9)
    roots = SUMRCCSD(mf, determinants=dets[:3], **OPTS).energy()
    B = cas.reference.B_mo["a"]
    n = B.shape[1]
    fci = pyscf_fci.direct_spin1.FCI().kernel(cas.reference.h_mo["a"], np.einsum("xpq,xrs->pqrs", B, B), n, (1, 1),
                                              nroots=n * n)[0] + cas.system.nuclear_repulsion
    assert len(roots) == 3 and all(np.abs(fci - e).min() < 1e-8 for e in roots)
    with pytest.raises(ValueError):
        SUMRCCSD(mf, 2, 2, determinants=dets)
    with pytest.raises(ValueError):
        SUMRCCSD(mf, determinants=[[1, 2, 3]]).energy()


def test_prune_intruders():
    """CAS(2,4) of a distorted H4 diverges with Jacobi; dropping the intruder determinants
    (a general model space) converges, with fewer roots."""
    mf = converged_scf(RHF, Molecule("H 0 0 0; H 1.058 0.053 0; H 0.106 1.27 0; H 1.111 1.323 0.159", "6-31g",
                                     auxiliary_basis=AUX, verbose=False))
    with pytest.raises(RuntimeError):
        SUMRCCSD(mf, 2, 4, solver="jacobi", maximum_iterations=60).energy()
    mr = SUMRCCSD(mf, 2, 4, solver="jacobi", prune_intruders=0.0, maximum_iterations=200, **OPTS)
    roots = mr.energy()
    assert len(roots) == len(mr.model_space.determinants) < 16


def test_mmsucc23():
    """Two electrons have no triples: the MM-SUCC(2,3) corrections vanish. For H4 they do not,
    and they match the kernel-level routine on the same integrals."""
    from ccpy.mrcc.mmsucc23 import mmsucc23
    from ccpy.mrcc.sumrcc import get_reference_hamiltonian

    mf = converged_scf(RHF, Molecule("H 0 0 0; H 0 0 0.9", "6-31g", auxiliary_basis=AUX, verbose=False))
    mr = SUMRCCSD(mf, 2, 2, **OPTS)
    assert np.allclose(mr.mmsucc23(), mr.energy(), atol=1e-12)
    mf = converged_scf(RHF, Molecule("H 0 0 0; H 1.058 0.053 0; H 0.106 1.27 0; H 1.111 1.323 0.159", "6-31g",
                                     auxiliary_basis=AUX, verbose=False))
    mr = SUMRCCSD(mf, 2, 2, **OPTS)
    e23 = mr.mmsucc23()
    H = [get_reference_hamiltonian(mr.reference.h_mo["a"], mr.reference.B_mo["a"], mr.system, r)[0]
         for r in mr.model_space.references]
    ref = mmsucc23(mr.T, H, mr.heff, mr.roots, mr.coefficients, mr.model_space, mr.system)[0]
    assert np.allclose(e23, ref, atol=1e-12) and np.all(np.abs(mr.mm23_corrections) > 1e-6)
    assert mr.mmsucc23(0) == e23[0]
