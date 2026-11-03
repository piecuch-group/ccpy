"""Shared fixtures for the ccpy.methods tests: small DF molecules, and legacy Driver runs on
the same SCF solutions used as references."""
import warnings

import pytest

from ccpy import Driver, Molecule, RHF, ROHF, UHF

OPTS = dict(amp_print_threshold=1.0, energy_convergence=1.0e-10, amp_convergence=1.0e-09)
AUX = "def2-universal-jkfit"
WATER = "O 0.0 0.0 0.1173; H 0.0 0.7572 -0.4692; H 0.0 -0.7572 -0.4692"


def converged_scf(cls, molecule):
    mf = cls(molecule, verbose=0, e_tol=1.0e-12, d_tol=1.0e-10)
    mf.kernel()
    return mf


def fresh_driver(mf, nfrozen=1, use_cholesky=False):
    """Legacy Driver on a new, identically converged copy of `mf` (its Hamiltonian is
    overwritten by run_hbar, so every reference chain needs its own)."""
    driver = Driver.from_scf(converged_scf(type(mf), mf.mol), nfrozen=nfrozen, use_cholesky=use_cholesky)
    driver.options.update(energy_convergence=1.0e-10, amp_convergence=1.0e-09, maximum_iterations=100)
    return driver


def total(driver, i=None):
    energy = driver.system.reference_energy + driver.correlation_energy
    return energy if i is None else energy + driver.vertical_excitation_energy[i]


@pytest.fixture(autouse=True)
def _quiet_df_port_warnings():
    from ccpy.methods import DFPortWarning

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DFPortWarning)
        yield


@pytest.fixture(scope="module")
def water_rhf():
    return converged_scf(RHF, Molecule(WATER, "6-31g", auxiliary_basis=AUX, verbose=False))


@pytest.fixture(scope="module")
def oh_doublet():
    return Molecule("O 0 0 0; H 0 0 1.85", "6-31g", auxiliary_basis=AUX, multiplicity=2, unit="bohr",
                    verbose=False)


@pytest.fixture(scope="module")
def oh_rohf(oh_doublet):
    return converged_scf(ROHF, oh_doublet)


@pytest.fixture(scope="module")
def oh_uhf(oh_doublet):
    return converged_scf(UHF, oh_doublet)


@pytest.fixture(scope="module")
def hydroxide_rhf():
    mol = Molecule("O 0 0 -0.48483; H 0 0 0.48483", "6-31g", auxiliary_basis=AUX, charge=-1, verbose=False)
    return converged_scf(RHF, mol)
