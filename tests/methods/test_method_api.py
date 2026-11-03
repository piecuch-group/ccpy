"""Behavior of the ccpy.methods API: restarts, dependency tracking, sharing, options,
integral handling, and re-creation at new geometries."""
import warnings

import numpy as np
import pytest
from conftest import OPTS

from ccpy import CCSD, CCSDT, CRCC23, EOMCCSD, LeftCCSD, Reference, methods_requiring_df_port
from ccpy.methods import DFPortWarning
from ccpy.methods.hamiltonian import VVVVError


def test_energy_solves_once_and_restarts(water_rhf):
    cc = CCSD(water_rhf, nfrozen=1, **OPTS)
    assert cc.T is None and not cc.converged
    e = cc.energy()
    version = cc.version
    assert cc.energy() == e and cc.version == version  # converged: no new solve
    T = cc.T.flatten().copy()
    cc.converged = False
    assert np.isclose(cc.energy(), e, atol=1.0e-10)  # restart from the current T
    assert cc.version == version + 1
    assert np.allclose(cc.T.flatten(), T, atol=1.0e-8)


def test_downstream_recomputed_after_upstream_changes(water_rhf):
    cc = CCSD(water_rhf, nfrozen=1, **OPTS)
    cr = CRCC23(cc)
    e = cr.energy()
    hbar, left_version = cc.hbar, cr.left.version
    assert cr.is_current()
    cc.converged = False
    cc.solve()
    assert not cr.is_current()
    assert np.isclose(cr.energy(), e, atol=1.0e-9)
    assert cc.hbar.version == hbar.version and cr.left.version == left_version + 1


def test_upstream_sharing_and_option_inheritance(water_rhf):
    cc = CCSD(water_rhf, nfrozen=1, amp_convergence=1.0e-9)
    left = LeftCCSD(cc)
    assert left.cc is cc and left.reference is cc.reference
    assert left.options["amp_convergence"] == 1.0e-9
    assert LeftCCSD(cc, amp_convergence=1.0e-6).options["amp_convergence"] == 1.0e-6
    # methods created from the SCF build their own upstream with the same options
    cr = CRCC23(water_rhf, nfrozen=1, amp_convergence=1.0e-9)
    assert isinstance(cr.cc, CCSD) and cr.cc.options["amp_convergence"] == 1.0e-9
    with pytest.raises(ValueError):
        LeftCCSD(cc, nfrozen=0)
    with pytest.raises(TypeError):
        CCSD(water_rhf, not_an_option=1)


def test_eom_requires_initial_guess(water_rhf):
    eom = EOMCCSD(water_rhf, nfrozen=1, **OPTS)
    with pytest.raises(RuntimeError):
        eom.energy()


def test_df_hamiltonian(water_rhf):
    ref = Reference(water_rhf, nfrozen=1)
    assert np.isclose(ref.energy, water_rhf.e_tot, atol=1.0e-10)
    H = ref.H
    B = H.chol.a
    expected = np.einsum("xme,xnf->mnef", B.ov, B.ov) - np.einsum("xmf,xne->mnef", B.ov, B.ov)
    assert np.allclose(H.aa.oovv, expected)
    assert np.allclose(H.ab.vovv, np.einsum("xae,xmf->amef", B.vv, H.chol.b.ov))
    with pytest.raises(VVVVError):
        H.aa.vvvv
    # HBar is built on a copy: the bare Hamiltonian is unchanged
    oo = H.a.oo.copy()
    cc = CCSD(ref, **OPTS)
    cc.energy()
    assert cc.hbar.H is not H and np.array_equal(H.a.oo, oo)
    assert not np.allclose(cc.hbar.chol.a.vo, H.chol.a.vo)  # T1-transformed vectors


def test_df_port_warning_and_registry(water_rhf):
    names = methods_requiring_df_port()
    assert "CCSDT" in names and "CCSD" not in names and "EOMCCSD" not in names
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        CCSDT(water_rhf, nfrozen=1, **OPTS).energy()
    assert any(issubclass(w.category, DFPortWarning) for w in caught)


def test_at_geometry_rebuilds_the_chain(water_rhf):
    cc = CCSD(water_rhf, nfrozen=1, amp_convergence=1.0e-9)
    eom = EOMCCSD(cc)
    eom.initial_guess(method="cisd", multiplicity=1, nacto=2, nactu=2, roots_per_irrep={"A": 1})
    X = np.array([xyz for _, xyz in water_rhf.mol.atoms]) * 1.01
    moved = eom.at_geometry(water_rhf.mol.with_geometry(X))
    assert type(moved) is EOMCCSD and moved.cc is not cc
    assert moved.cc.options["amp_convergence"] == 1.0e-9
    assert moved.guess_settings == eom.guess_settings
    assert moved.reference.nfrozen == 1
    assert type(moved.reference.scf) is type(water_rhf)
