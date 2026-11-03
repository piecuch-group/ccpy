"""Independent PySCF spin-block checks of native DF-ADC(2)."""
import io
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from ccpy import ADC2, CCSD, EAADC2, IPADC2, FNOOptions, FNOTarget, Reference, prepare_fno
from ccpy.adc.eeadc2 import EEADC2


def pyscf_oracle(kernel, sector="EE"):
    pytest.importorskip("pyscf")
    from pyscf.adc import uadc_ee, uadc_ip, uadc_ea

    na, nb, va, vb = kernel.space.sizes
    t = kernel.t
    adc = SimpleNamespace(method="adc(2)", nocc_a=na, nocc_b=nb, nvir_a=va, nvir_b=vb,
                          mo_energy_a=np.r_[kernel.eo[:na], kernel.ev[:va]],
                          mo_energy_b=np.r_[kernel.eo[na:], kernel.ev[va:]],
                          t1=((None, None), None, None),
                          t2=((t[:na, :na, :va, :va], t[:na, na:, :va, va:], t[na:, na:, va:, va:]),),
                          f_ov=None, _scf=None, verbose=0, stdout=io.StringIO())
    if np.max(np.abs(kernel.fov), initial=0) > 1.e-10:
        from pyscf.scf.rohf import ROHF
        adc._scf = object.__new__(ROHF)
        adc.f_ov = kernel.fov[:na, :va], kernel.fov[na:, va:]
        adc.t1 = ((None, None), None, (kernel.t1[:na, :va], kernel.t1[na:, va:]))
    B = {}
    for left, right in product(("o", "v", "O", "V"), repeat=2):
        if left.isupper() != right.isupper():
            continue
        if left.lower() == right.lower() == "o":
            arr = kernel.oo
        elif left.lower() == right.lower() == "v":
            arr = kernel.vv
        else:
            arr = kernel.ov if left.lower() == "o" else kernel.ov.transpose(0, 2, 1)
        indices = {"o": slice(0, na), "O": slice(na, na + nb), "v": slice(0, va), "V": slice(va, va + vb)}
        B[left + right] = arr[:, indices[left], indices[right]]
    eris = SimpleNamespace()
    for left, right in product(B, repeat=2):
        value = np.einsum("Pij,Pkl->ijkl", B[left], B[right])
        if left.lower() == "ov" and right.lower() == "vv":
            a, b = np.tril_indices(value.shape[2])
            value = value[:, :, a, b]
        setattr(eris, left + right, value)
    module = {"EE": uadc_ee, "IP": uadc_ip, "EA": uadc_ea}[sector]
    imds = module.get_imds(adc, eris)
    return module.matvec(adc, imds, eris), adc


def charged_to_pyscf(kernel, vector):
    na, nb, va, vb = kernel.space.sizes
    sections = kernel.space.slices
    if kernel.sector == "IP":
        mixed = -vector[sections[2]].reshape(vb, na, nb).transpose(0, 2, 1).ravel()
        return np.r_[vector[sections[0]], np.zeros(nb), vector[sections[1]], mixed,
                     np.zeros(va * nb * na + vb * nb * (nb - 1) // 2)]
    return np.r_[vector[sections[0]], np.zeros(vb), vector[sections[1]], vector[sections[2]],
                 np.zeros(na * vb * va + nb * vb * (vb - 1) // 2)]


@pytest.mark.parametrize("fixture", ["water_rhf", "oh_uhf"])
@pytest.mark.parametrize("cls,sector", [(IPADC2, "IP"), (EAADC2, "EA")])
def test_charged_against_pyscf(fixture, cls, sector, request):
    pytest.importorskip("pyscf")
    from pyscf.adc import uadc_ip, uadc_ea

    kernel = cls(Reference(request.getfixturevalue(fixture), nfrozen=1)).kernel
    action, oracle = pyscf_oracle(kernel, sector)
    projection = np.column_stack([charged_to_pyscf(kernel, e) for e in np.eye(kernel.space.ndim)])
    rng = np.random.default_rng(412)
    v = rng.normal(size=kernel.space.ndim)
    v /= np.linalg.norm(v)
    np.testing.assert_allclose(kernel.matvec(v), projection.T @ action(projection @ v), atol=1.e-11)
    np.testing.assert_allclose(kernel.space.pack(*kernel.space.unpack(v)), v, atol=1.e-13)
    module = uadc_ip if sector == "IP" else uadc_ea
    Da, Db = module.make_rdm1_eigenvectors(oracle, projection @ v, projection @ v)
    na, nb, _, _ = kernel.space.sizes
    density = kernel.virtual_density(v)
    for s, no, D in (("a", na, Da), ("b", nb, Db)):
        U = kernel.rotations[s]["v"]
        np.testing.assert_allclose(density[s], U @ D[no:, no:] @ U.T, atol=1.e-11)
    matrix = np.column_stack([kernel.matvec(e) for e in np.eye(kernel.space.ndim)])
    np.testing.assert_allclose(matrix, matrix.T, atol=1.e-12)
    np.testing.assert_allclose(np.diag(matrix), kernel.diagonal, atol=1.e-12)


@pytest.mark.parametrize("fixture", ["water_rhf", "oh_rohf", "oh_uhf"])
@pytest.mark.parametrize("cls,guess", [(IPADC2, "ipcisd"), (EAADC2, "eacisd")])
def test_charged_davidson_guesses(fixture, cls, guess, request):
    from conftest import OPTS
    method = cls(request.getfixturevalue(fixture), nfrozen=1, formulation="full", **OPTS)
    method.initial_guess(method=guess, nacto=2, nactu=2, roots_per_irrep={"A": 2})
    energies = method.energy()
    assert method.converged and method.guess_settings["method"] == guess
    np.testing.assert_allclose(energies, method.reference_energy + method.correlation_energy + method.omega)
    for R, omega in zip(method.R, method.omega):
        assert np.linalg.norm(method.kernel.matvec(R.flatten()) - omega * R.flatten()) < 1.e-8
        converted = method._new_operator().from_cluster(R.as_cluster())
        np.testing.assert_allclose(converted.flatten(), R.flatten(), atol=1.e-11)
    assert method.reference.H.aa.__dict__.get("vvvv") is None


@pytest.mark.parametrize("sector", ["IP", "EA"])
@pytest.mark.parametrize("fixture", ["water_rhf", "oh_rohf"])
def test_charged_fno_provider(sector, fixture, request):
    targets = (FNOTarget(sector, root=0, irrep="A"),)
    parent = Reference(request.getfixturevalue(fixture), nfrozen=1)
    ref = prepare_fno(parent, FNOOptions(source="adc2", targets=targets, occupation_cutoff=0,
                                       energy_completion="state_specific"))
    assert abs(ref.fno_result.delta_ground) < 1.e-9
    assert abs(ref.fno_result.delta_state(targets[0])) < 1.e-8


@pytest.mark.parametrize("sector", ["EE", "IP", "EA"])
def test_rohf_truncated_adc_fno(sector, oh_rohf):
    targets = (FNOTarget(sector, root=0, irrep="A"),)
    ref = prepare_fno(Reference(oh_rohf, nfrozen=1),
                      FNOOptions(source="adc2", targets=targets, nvirtuals=3,
                                 energy_completion="state_specific"))
    assert ref.fno_result.discarded["a"] > 0
    assert np.isfinite(ref.fno_result.delta_state(targets[0]))


def test_rohf_ee_davidson(oh_rohf):
    from conftest import OPTS
    method = ADC2(oh_rohf, nfrozen=1, **OPTS)
    method.initial_guess(nacto=2, nactu=2, roots_per_irrep={"A": 2})
    method.solve()
    assert method.converged
    from ccpy.mbpt.dfmp2 import dfmp2
    assert method.correlation_energy == pytest.approx(dfmp2(method.reference, False)[0], abs=1.e-11)


@pytest.mark.parametrize("cls", [IPADC2, EAADC2])
def test_charged_symmetry(cls):
    from conftest import AUX, OPTS, WATER, converged_scf
    from ccpy import Molecule, RHF
    mf = converged_scf(RHF, Molecule(WATER, "6-31g", auxiliary_basis=AUX, symmetry=True, verbose=False))
    method = cls(mf, nfrozen=1, formulation="full", **OPTS)
    method.initial_guess(nacto=2, nactu=2, roots_per_irrep={"A1": 1, "B1": 1})
    method.solve()
    assert method.root_irreps == method.check_root_symmetry(method.R, method.root_irreps) == ["A1", "B1"]


@pytest.mark.parametrize("fixture", ["water_rhf", "oh_uhf", "oh_rohf"])
def test_matrix_action_diagonal_and_density_against_pyscf(fixture, request):
    pytest.importorskip("pyscf")
    from pyscf.adc import uadc_ee

    reference = Reference(request.getfixturevalue(fixture), nfrozen=1)
    kernel = EEADC2(reference)
    action, oracle = pyscf_oracle(kernel)
    rng = np.random.default_rng(2719)
    vector = rng.normal(size=kernel.space.ndim)
    vector /= np.linalg.norm(vector)
    np.testing.assert_allclose(kernel.matvec(vector), action(vector), atol=1.e-11)
    other = rng.normal(size=vector.size)
    assert np.dot(other, kernel.matvec(vector)) == pytest.approx(np.dot(vector, kernel.matvec(other)), abs=1.e-10)
    np.testing.assert_allclose(kernel.space.pack(*kernel.space.unpack(vector)), vector, atol=1.e-13)
    D = kernel.virtual_density(vector)
    Da, Db = uadc_ee.make_rdm1_eigenvectors(oracle, vector, vector)
    na, nb, _, _ = kernel.space.sizes
    Ua, Ub = kernel.rotations["a"]["v"], kernel.rotations["b"]["v"]
    np.testing.assert_allclose(D["a"], Ua @ Da[na:, na:] @ Ua.T, atol=1.e-11)
    np.testing.assert_allclose(D["b"], Ub @ Db[nb:, nb:] @ Ub.T, atol=1.e-11)
    for k in rng.choice(vector.size, 8, replace=False):
        e = np.zeros_like(vector)
        e[k] = 1
        assert kernel.matvec(e)[k] == pytest.approx(kernel.diagonal[k], abs=1.e-11)
    assert reference.H.aa.blocks() == reference.H.ab.blocks() == reference.H.bb.blocks() == []


@pytest.mark.parametrize("multiplicity", [1, 3])
def test_davidson_bare_guesses_and_residuals(water_rhf, multiplicity):
    from conftest import OPTS

    method = ADC2(water_rhf, nfrozen=1, **OPTS)
    method.initial_guess(multiplicity=multiplicity, nacto=2, nactu=2, roots_per_irrep={"A": 2})
    energies = method.energy()
    assert method.converged and not hasattr(method, "hbar") and not hasattr(method, "cc")
    np.testing.assert_allclose(energies, method.reference_energy + method.correlation_energy + method.omega)
    vectors = [r.flatten() for r in method.R]
    assert abs(np.dot(*vectors)) < 1.e-8
    for vector, omega in zip(vectors, method.omega):
        assert np.linalg.norm(method.kernel.matvec(vector) - omega * vector) < 1.e-8
    assert method.version == 1
    method.energy()
    assert method.version == 1


def test_adc_fno_provider(water_rhf):
    targets = (FNOTarget("EE", root=0, irrep="A", multiplicity=1),)
    options = FNOOptions(source="adc2", nvirtuals=6, targets=targets, energy_completion="state_specific")
    reduced = prepare_fno(Reference(water_rhf, nfrozen=1), options)
    result = reduced.fno_result
    assert result.full.state_energies.keys() == result.truncated.state_energies.keys() == set(targets)
    assert np.isfinite(result.delta_state(targets[0]))
    cc = CCSD(reduced)
    assert cc.reference is reduced


def test_adc_fno_double_charged_targets_rejected(water_rhf):
    with pytest.raises(NotImplementedError, match="EE, IP, and EA"):
        prepare_fno(Reference(water_rhf, nfrozen=1),
                    FNOOptions(source="adc2", targets=(FNOTarget("DEA"),)))


def test_dense_spectrum_and_root_selection():
    from conftest import AUX, OPTS, converged_scf
    from ccpy import Molecule, RHF

    mf = converged_scf(RHF, Molecule("Li 0 0 0; H 0 0 1.6", "sto-3g", auxiliary_basis=AUX, verbose=False))
    method = ADC2(mf, **OPTS)
    kernel = method.kernel
    oracle, _ = pyscf_oracle(kernel)
    identity = np.eye(kernel.space.ndim)
    matrix = np.column_stack([oracle(column) for column in identity.T])
    spectrum = np.linalg.eigvalsh(matrix)
    method.initial_guess(multiplicity=1, nacto=2, nactu=2, roots_per_irrep={"A": 3}, roots=[0, 2])
    method.solve()
    assert method.nroots == 2
    for omega in method.omega:
        assert np.min(np.abs(spectrum - omega)) < 1.e-9
    np.testing.assert_allclose(kernel.matvec(identity[:, 5]), matrix[:, 5], atol=1.e-12)
    np.testing.assert_allclose(method.completed_omega, method.omega, atol=1.e-12)


def test_symmetry_targeting_and_geometry(water_rhf):
    from conftest import AUX, OPTS, WATER, converged_scf
    from ccpy import Molecule, RHF

    mf = converged_scf(RHF, Molecule(WATER, "6-31g", auxiliary_basis=AUX, symmetry=True, verbose=False))
    method = ADC2(mf, nfrozen=1, **OPTS)
    method.initial_guess(multiplicity=1, nacto=2, nactu=2, roots_per_irrep={"A1": 1, "B1": 1})
    method.solve()
    assert method.root_irreps == ["A1", "B1"]
    assert method.check_root_symmetry(method.R, method.root_irreps) == method.root_irreps
    # Geometry reconstruction uses C1, as required by Molecule.with_geometry.
    plain = ADC2(water_rhf, nfrozen=1, **OPTS)
    plain.initial_guess(nacto=2, nactu=2, roots_per_irrep={"A": 1})
    coords = np.array([xyz for _, xyz in water_rhf.mol.atoms]) * 1.01
    moved = plain.at_geometry(water_rhf.mol.with_geometry(coords))
    assert moved.reference is not plain.reference
    assert moved.guess_settings == plain.guess_settings
    assert moved.R is None


def test_uhf_davidson(oh_uhf):
    from conftest import OPTS

    method = ADC2(oh_uhf, nfrozen=1, **OPTS)
    method.initial_guess(method="cis", multiplicity=-1, roots_per_irrep={"A": 2})
    method.solve()
    assert method.converged
    for vector, omega in zip(method.R, method.omega):
        assert np.linalg.norm(method.kernel.matvec(vector.flatten()) - omega * vector.flatten()) < 1.e-8


def test_unconverged_errors(water_rhf):
    method = ADC2(water_rhf, nfrozen=1, maximum_iterations=1)
    method.initial_guess(multiplicity=1, nacto=1, nactu=1, roots_per_irrep={"A": 1})
    with pytest.raises(RuntimeError, match="did not converge"):
        method.solve()
    assert not method.converged


def test_builtin_provider_does_not_recurse(water_rhf):
    targets = (FNOTarget("EE", root=0, irrep="A", multiplicity=1),)
    options = FNOOptions(source="adc2", targets=targets, occupation_cutoff=0, energy_completion="state_specific")
    method = ADC2(water_rhf, nfrozen=1, fno=options)
    method.initial_guess(nacto=2, nactu=2, roots_per_irrep={"A": 1})
    assert np.isfinite(method.energy(0))
    assert abs(method.fno_result.delta_ground) < 1.e-10
    assert abs(method.fno_result.delta_state(targets[0])) < 1.e-8


def test_provider_multiple_spin_groups(water_rhf):
    targets = (FNOTarget("EE", root=0, irrep="A", multiplicity=1),
               FNOTarget("EE", root=0, irrep="A", multiplicity=3))
    ref = prepare_fno(Reference(water_rhf, nfrozen=1),
                      FNOOptions(source="adc2", targets=targets, occupation_cutoff=0,
                                 energy_completion="state_specific", weights=(0.5, 0.5)))
    for target in targets:
        assert abs(ref.fno_result.delta_state(target)) < 1.e-8
        assert target in ref.fno_result.full.virtual_densities
