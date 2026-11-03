"""roots_per_irrep initial guesses (ccpy.methods.guess) for every EOMCC family.

For each family the roots of a symmetry-targeted guess must reproduce, energy and irrep, the
same states found among the lowest roots of any symmetry (roots_per_irrep=n), whether the guess
is diagonalized irrep by irrep (cisd, ipcisd, eacisd, dipcis) or in full and sorted into irreps
by its leading amplitudes (sfcis, ipcis, eacis).
"""
import contextlib
import io

import numpy as np
import pytest
from conftest import AUX, OPTS, WATER, converged_scf

from ccpy import CCSD, DIPEOM3, EAEOM2, EOMCCSD, IPEOM2, ROHF, SFEOMCCSD, HBarCCSDTA, Molecule, RHF


def _quiet(f, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*args, **kwargs)


@pytest.fixture(scope="module")
def water_cc():
    mf = converged_scf(RHF, Molecule(WATER, "6-31g", auxiliary_basis=AUX, symmetry=True, verbose=False))
    cc = CCSD(mf, nfrozen=1, **OPTS)
    assert cc.system.point_group == "C2V"
    return cc


def _targeted_matches_lowest(cls, upstream, n, pick, **guess):
    """Target the irreps of roots `pick` of the lowest n (any symmetry): the targeted roots must
    be the lowest roots of those irreps among the n."""
    lowest = cls(upstream, **OPTS).initial_guess(roots_per_irrep=n, **guess)
    energies = _quiet(lowest.energy)
    assert lowest.root_irreps == lowest.check_root_symmetry(lowest.R, lowest.root_irreps)
    counts = {}
    for k in pick:
        counts[lowest.root_irreps[k]] = counts.get(lowest.root_irreps[k], 0) + 1
    targeted = cls(upstream, **OPTS).initial_guess(roots_per_irrep=counts, **guess)
    found = _quiet(targeted.energy)
    expected = [(irrep, e) for irrep, c in counts.items()
                for e in [e for label, e in zip(lowest.root_irreps, energies) if label == irrep][:c]]
    assert targeted.root_irreps == [irrep for irrep, _ in expected]
    assert np.allclose(found, [e for _, e in expected], atol=1.0e-8)
    assert targeted.check_root_symmetry(targeted.R, targeted.root_irreps) == targeted.root_irreps
    return lowest, targeted


def test_eomccsd_cisd_by_irrep(water_cc):
    lowest, targeted = _targeted_matches_lowest(EOMCCSD, water_cc, 4, [0, 3], nacto=2, nactu=2)
    assert lowest.guess_settings["method"] == "cisd_chol"                       # CISd by default (DF)
    assert len(set(lowest.root_irreps)) > 1
    assert list(targeted.guess_settings["roots_per_irrep"]) == targeted.root_irreps   # in requested order
    # roots= selects guess roots within the targeted guess
    one = EOMCCSD(water_cc, **OPTS).initial_guess(roots_per_irrep={targeted.root_irreps[1]: 1}, nacto=2, nactu=2,
                                                 roots=[0])
    assert np.isclose(_quiet(one.energy)[0], targeted.energy()[1], atol=1.0e-8)


def test_ip_ea_by_irrep(water_cc):
    _targeted_matches_lowest(IPEOM2, water_cc, 3, [0, 2], nacto=2, nactu=2)        # ipcisd (blocked)
    _targeted_matches_lowest(IPEOM2, water_cc, 3, [1, 2], method="ipcis")          # sorted by leading 1h
    _targeted_matches_lowest(EAEOM2, water_cc, 3, [0, 1], method="eacis")          # sorted by leading 1p
    ea, _ = _targeted_matches_lowest(EAEOM2, water_cc, 2, [1], nacto=2, nactu=3)   # eacisd (blocked)
    assert ea.guess_settings["multiplicity"] == -1


def test_dip_by_irrep(water_cc):
    nocc = water_cc.system.noccupied_alpha
    _targeted_matches_lowest(lambda up, **o: DIPEOM3(up, hbar_class=HBarCCSDTA, **o), water_cc, 3, [0, 2],
                             nacto=nocc)


def test_sf_by_irrep():
    mol = Molecule("C 0 0 0; H 0 1.0 0.6; H 0 -1.0 0.6", "6-31g", auxiliary_basis=AUX, multiplicity=3,
                   symmetry=True, verbose=False)
    mf = converged_scf(ROHF, mol)
    _targeted_matches_lowest(lambda up, **o: SFEOMCCSD(up, nfrozen=1, **o), mf, 3, [1, 2], multiplicity=1)


def test_invalid_guesses(water_cc):
    eom = EOMCCSD(water_cc, **OPTS)
    with pytest.raises(ValueError, match="not in point group"):
        eom.initial_guess(roots_per_irrep={"A'": 1})
    with pytest.raises(ValueError, match="guess method"):
        eom.initial_guess(method="ipcisd2", roots_per_irrep=1)
    with pytest.raises(ValueError, match="needs roots_per_irrep"):
        eom.initial_guess().run_guess()
    with pytest.raises(ValueError, match="exceeds"):
        eom.initial_guess(roots_per_irrep={"B1": 1}, roots=[1])
