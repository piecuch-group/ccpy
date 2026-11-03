"""Folded/full equivalence and explicit satellite targeting."""
import numpy as np
import pytest

from ccpy import ADC2, EAADC2, IPADC2, FNOTarget, FNOOptions, Reference, prepare_fno
from ccpy.adc.folded import FoldedOperator, solve_folded


@pytest.mark.parametrize("fixture", ["water_rhf", "oh_rohf", "oh_uhf"])
@pytest.mark.parametrize("cls", [ADC2, IPADC2, EAADC2])
def test_folded_full_primary_equivalence(fixture, cls, request):
    from conftest import OPTS
    mf = request.getfixturevalue(fixture)
    full, folded = cls(mf, nfrozen=1, formulation="full", **OPTS), cls(mf, nfrozen=1, **OPTS)
    assert folded.formulation == "folded"
    for method in (full, folded):
        method.initial_guess(nacto=2, nactu=2, roots_per_irrep={"A": 1})
    np.testing.assert_allclose(folded.energy(), full.energy(), atol=1.e-8)
    assert abs(np.dot(folded.R[0].flatten(), full.R[0].flatten())) > 1 - 1.e-8
    for spin in ("a", "b"):
        np.testing.assert_allclose(folded.virtual_density()[spin], full.virtual_density()[spin], atol=1.e-8)
    op = FoldedOperator(folded.kernel, folded.omega[0], folded.kernel.symmetry_mask("A"))
    assert op.nprimary < folded.R[0].ndim
    assert len(op.diagonal) == op.nprimary


@pytest.mark.parametrize("cls", [ADC2, IPADC2, EAADC2])
def test_full_satellite_target_and_folded_guards(water_rhf, cls):
    from conftest import OPTS
    method = cls(water_rhf, nfrozen=1, formulation="full", **OPTS)
    kernel = method.kernel
    matrix = np.column_stack([kernel.matvec(e) for e in np.eye(kernel.space.ndim)])
    values, vectors = np.linalg.eigh(matrix)
    nprimary = FoldedOperator(kernel, -10., np.ones(kernel.space.ndim, bool)).nprimary
    index = np.argmin(np.sum(vectors[:nprimary] ** 2, axis=0))
    method.R = [method._new_operator()]
    method.R[0].unflatten(vectors[:, index])
    method.omega, method.root_irreps = np.array([values[index]]), ["A"]
    method.solve()
    assert method.omega[0] == pytest.approx(values[index], abs=1.e-9)
    assert all(np.all(np.isfinite(d)) for d in method.virtual_density().values())
    seed = np.zeros(kernel.space.ndim)
    seed[nprimary] = 1
    with pytest.raises(ValueError, match="primary character"):
        solve_folded(kernel, seed, values[index], np.ones(seed.size, bool), method.system, method.options)
    with pytest.raises(ValueError, match="satellite pole"):
        FoldedOperator(kernel, kernel.diagonal[nprimary], np.ones(seed.size, bool))


@pytest.mark.parametrize("sector", ["EE", "IP", "EA"])
@pytest.mark.parametrize("formulation", ["full", "folded"])
def test_fno_formulation_selection(water_rhf, sector, formulation):
    target = FNOTarget(sector, irrep="A", multiplicity=1 if sector == "EE" else None,
                       formulation=formulation)
    reference = prepare_fno(Reference(water_rhf, nfrozen=1), FNOOptions(
        source="adc2", occupation_cutoff=0., targets=(target,), energy_completion="state_specific"))
    assert abs(reference.fno_result.delta_state(target)) < 1.e-8


def test_invalid_formulation(water_rhf):
    with pytest.raises(ValueError, match="formulation"):
        ADC2(water_rhf, formulation="other")
    with pytest.raises(ValueError, match="formulation"):
        FNOTarget("EA", formulation="other")


def test_full_satellite_fno_rohf(oh_rohf):
    # The second bare EA-CISd root has negligible 1p character on this fixture.
    target = FNOTarget("EA", root=1, irrep="A", formulation="full")
    reference = prepare_fno(Reference(oh_rohf, nfrozen=1), FNOOptions(
        source="adc2", occupation_cutoff=0., targets=(target,), energy_completion="state_specific"))
    result = reference.fno_result
    assert abs(result.delta_state(target)) < 1.e-8
    cluster = result.full.metadata["states"][target]["cluster"]
    assert np.linalg.norm(cluster.a) < .1
