"""DF-MP2 FNO numerics, ordinary working spaces, and method-layer completion."""
import numpy as np
import pytest
from conftest import AUX, OPTS, converged_scf

from ccpy import (CCSD, CCSDT, CCSDT_P, CRCC23, CREOMCC23, DEAEOM2, DIPEOM3, EAEOM2,
                  EOMCCSD, FNOEvaluation, FNOOptions, FNOTarget, IPEOM2, LeftCCSD,
                  MP2, Molecule, RHF, ROHF, Reference, SFEOMCCSD, STEOMCC,
                  register_fno_provider, get_active_triples_pspace)
from ccpy.mbpt.dfmp2 import dfmp2
from ccpy.mbpt.mbpt import calc_mp2
from ccpy.methods.fno import prepare_fno


def dense_spin_orbital_mp2(ref):
    """Small-system oracle: explicit full spin-orbital tensor, unlike the pair path."""
    from scipy.linalg import block_diag

    H, sys = ref.H, ref.system
    no = [sys.noccupied_alpha, sys.noccupied_beta]
    nv = [sys.nunoccupied_alpha, sys.nunoccupied_beta]
    eo, ev, Uo, Uv, B, fov = [], [], [], [], [], []
    for s in ("a", "b"):
        f = getattr(H, s)
        e, U = np.linalg.eigh(f.oo)
        eo.extend(e)
        Uo.append(U)
        e, V = np.linalg.eigh(f.vv)
        ev.extend(e)
        Uv.append(V)
        B.append(np.einsum("Pia,ij,ab->Pjb", getattr(H.chol, s).ov, U, V))
        fov.append(U.T @ f.ov @ V)
    factors = np.zeros((H.naux, sum(no), sum(nv)))
    factors[:, :no[0], :nv[0]] = B[0]
    factors[:, no[0]:, nv[0]:] = B[1]
    g = np.einsum("Pia,Pjb->ijab", factors, factors)
    g -= g.swapaxes(2, 3).copy()
    eo, ev = np.asarray(eo), np.asarray(ev)
    t = g / (eo[:, None, None, None] + eo[None, :, None, None]
             - ev[None, None, :, None] - ev[None, None, None, :])
    singles = block_diag(*fov) / (eo[:, None] - ev[None, :])
    energy = 0.25 * np.sum(g * t) + np.sum(block_diag(*fov) * singles)
    density = 0.5 * np.einsum("ijac,ijbc->ab", t, t) + singles.T @ singles
    return energy, {"a": Uv[0] @ density[:nv[0], :nv[0]] @ Uv[0].T,
                    "b": Uv[1] @ density[nv[0]:, nv[0]:] @ Uv[1].T}


@pytest.mark.parametrize("fixture", ["water_rhf", "oh_uhf", "oh_rohf"])
def test_dfmp2_energy_density_and_no_four_index(fixture, request):
    ref = Reference(request.getfixturevalue(fixture), nfrozen=1)
    energy, density = dfmp2(ref)
    expected_energy, expected_density = dense_spin_orbital_mp2(ref)
    assert energy == pytest.approx(expected_energy, abs=1.0e-11)
    for spin in ("a", "b"):
        np.testing.assert_allclose(density[spin], expected_density[spin], atol=1.0e-11)
        assert np.linalg.eigvalsh(density[spin]).min() > -1.0e-12
    assert ref.H.aa.blocks() == ref.H.ab.blocks() == ref.H.bb.blocks() == []
    assert dfmp2(ref, need_density=False)[0] == pytest.approx(energy, abs=1.0e-12)
    if fixture != "oh_rohf":
        assert energy == pytest.approx(calc_mp2(ref.system, ref.H), abs=1.0e-10)


@pytest.mark.parametrize("fixture", ["water_rhf", "oh_uhf", "oh_rohf"])
def test_reduced_reference(fixture, request):
    parent = Reference(request.getfixturevalue(fixture), nfrozen=1)
    original = {s: C.copy() for s, C in parent.mo_coeff.items()}
    child = prepare_fno(parent, FNOOptions(nvirtuals=3))
    assert child is not parent and parent.fno_result is None
    assert child.system.norbitals < parent.system.norbitals
    assert child.ndelete == child.system.ndelete == 0
    assert child.energy == parent.energy
    assert child.system.frozen_energy == parent.system.frozen_energy
    assert child.system.num_virt_beta + child.system.num_act_unoccupied_beta == child.system.nunoccupied_beta
    for s, spin in (("a", "alpha"), ("b", "beta")):
        nocc = parent.nfrozen + getattr(parent.system, f"noccupied_{spin}")
        np.testing.assert_array_equal(child.mo_coeff[s][:, :nocc], original[s][:, :nocc])
        np.testing.assert_array_equal(parent.mo_coeff[s], original[s])
        Q = child.fno_result.transforms[s]
        np.testing.assert_allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=1.0e-12)
        np.testing.assert_allclose(child.mo_coeff[s].T @ parent.scf.get_ovlp() @ child.mo_coeff[s],
                                   np.eye(Q.shape[1]), atol=1.0e-9)
        assert child.H.chol.__dict__[s].vv.shape[1] == getattr(child.system, f"nunoccupied_{spin}")
        assert len(child.system.orbital_symmetries_all[s] if child.is_uhf
                   else child.system.orbital_symmetries_all) == Q.shape[1]
    assert child.fno_result.discarded["a"] == child.fno_result.discarded["b"]
    legacy = child.legacy_hamiltonian(copy_=False)
    for s in ("aa", "ab", "bb"):
        np.testing.assert_allclose(getattr(legacy, s).oovv, getattr(child.H, s).oovv, atol=1.0e-11)
    assert MP2(child).energy() == pytest.approx(MP2(parent).energy(), abs=1.0e-10)


@pytest.fixture(scope="module")
def reduced_cc(water_rhf):
    return CCSD(water_rhf, nfrozen=1, fno=FNOOptions(nvirtuals=4), **OPTS).solve()


def test_composite_cc_corrections_and_inheritance(reduced_cc):
    cc = reduced_cc
    delta = cc.fno_result.delta_ground
    assert cc.converged
    assert cc.energy() == pytest.approx(cc.raw_energy() + delta, abs=1.0e-12)
    assert cc.total_energy == cc.energy()
    assert not cc.has_analytic_gradient()
    left = LeftCCSD(cc)
    assert left.reference is cc.reference and left.cc is cc
    assert not left.has_analytic_gradient()
    cr = CRCC23(cc)
    assert cr.energy() == pytest.approx(cr.raw_energy() + delta, abs=1.0e-12)
    assert not cr.has_analytic_gradient()
    assert CRCC23(cc, fno=cc.fno_options).reference is cc.reference
    with pytest.raises(ValueError, match="conflict"):
        CCSD(cc, fno=FNOOptions(nvirtuals=2))
    with pytest.raises(ValueError, match="disable"):
        CCSD(cc, fno=False)


@pytest.mark.parametrize("cls", [EOMCCSD, IPEOM2, EAEOM2, DEAEOM2, DIPEOM3])
def test_eom_sectors_in_reduced_space(reduced_cc, cls):
    eom = cls(reduced_cc)
    eom.initial_guess(nacto=2, nactu=2, roots_per_irrep={"A": 1}, use_symmetry=False)
    completed = eom.energy()
    assert eom.converged
    assert eom.reference is reduced_cc.reference
    np.testing.assert_allclose(completed - eom.raw_energy(), reduced_cc.fno_result.delta_ground, atol=1.0e-12)
    np.testing.assert_allclose(eom.completed_omega, eom.omega, atol=1.0e-12)
    assert eom.R[0].flatten().size == eom._new_operator().flatten().size
    if cls is EOMCCSD:
        correction = CREOMCC23(eom)
        np.testing.assert_allclose(correction.energy() - correction.raw_energy(),
                                   reduced_cc.fno_result.delta_ground, atol=1.0e-12)


def test_full_retention_and_high_rank(water_rhf):
    full = CCSD(water_rhf, nfrozen=1, **OPTS)
    retained = CCSD(water_rhf, nfrozen=1, fno=FNOOptions(occupation_cutoff=0), **OPTS)
    assert retained.system.norbitals == full.system.norbitals
    assert retained.fno_result.delta_ground == pytest.approx(0, abs=1.0e-10)
    assert retained.energy() == pytest.approx(full.energy(), abs=1.0e-9)
    # The four-index high-rank path receives only reduced integrals.
    triples = CCSDT(water_rhf, nfrozen=1, fno=FNOOptions(nvirtuals=2), **OPTS)
    assert triples.energy() == pytest.approx(triples.raw_energy() + triples.fno_result.delta_ground)
    assert triples.converged


def test_steom_and_geometry():
    mf = converged_scf(RHF, Molecule("Li 0 0 0; H 0 0 1.6", "sto-3g", auxiliary_basis=AUX, verbose=False))
    cc = CCSD(mf, fno=FNOOptions(nvirtuals=3), maximum_iterations=100, **OPTS)
    method = STEOMCC(cc, nroots=1, nacto=2, nactu=3, ip_threshold=0, ea_threshold=0)
    np.testing.assert_allclose(method.energy() - method.raw_energy(), cc.fno_result.delta_ground, atol=1.0e-12)
    assert method.hbar.reference is cc.reference
    assert method.ip.R[0].a.shape == (cc.system.noccupied_alpha,)
    assert method.ea.R[0].a.shape == (cc.system.nunoccupied_alpha,)
    moved = method.at_geometry(mf.mol.with_geometry(np.array([xyz for _, xyz in mf.mol.atoms]) * 1.01))
    assert moved.fno_options == method.fno_options
    assert moved.reference is not method.reference
    assert moved.reference.fno_parent is not method.reference.fno_parent
    assert not np.array_equal(moved.reference.mo_coeff["a"], method.reference.mo_coeff["a"])


def test_provider_hooks_and_state_completion(water_rhf, monkeypatch):
    from ccpy.methods import fno as module

    calls = []
    targets = (FNOTarget("EE", 0, "A"), FNOTarget("EE", 1, "A"))

    def adc_stub(ref, targets, need_density, full_result):
        calls.append((ref, need_density))
        if need_density:
            assert full_result is None
        else:
            assert full_result is not None and ref.fno_parent is calls[0][0]
            assert ref.orbital_transform["a"].shape[0] == ref.fno_parent.mo_coeff["a"].shape[1]
        ec, d = dfmp2(ref, need_density)
        ground = ref.energy + ec
        energies = {t: ground + 0.2 + t.root * 0.1 + ref.system.nunoccupied_alpha * 0.001
                    for t in targets}
        return FNOEvaluation(ground, {t: d for t in targets} if need_density else {}, energies)

    monkeypatch.setattr(module, "_PROVIDERS", module._PROVIDERS.copy())
    register_fno_provider("adc2", adc_stub)
    options = FNOOptions(source="adc2", nvirtuals=3, targets=targets, energy_completion="state_specific")
    eom = EOMCCSD(water_rhf, nfrozen=1, fno=options, **OPTS)
    eom.initial_guess(nacto=2, nactu=2, roots_per_irrep={"A": 2})
    total = eom.energy()
    result = eom.fno_result
    assert len(calls) == 2 and calls[0][1] and not calls[1][1]
    assert calls[0][0] is eom.reference.fno_parent and calls[1][0] is eom.reference
    np.testing.assert_allclose(total - eom.raw_energy(), [result.delta_state(t) for t in targets], atol=1.0e-12)
    np.testing.assert_allclose(eom.completed_omega - eom.omega,
                               [result.delta_state(t) - result.delta_ground for t in targets], atol=1.0e-12)
    assert np.max(np.abs(eom.completed_omega - eom.omega)) > 1.0e-4
    wrong_sector = IPEOM2(eom.cc)
    with pytest.raises(ValueError, match="matching-sector"):
        wrong_sector._state_shifts(2, "IP", ["A", "A"])


def test_bad_options_and_missing_provider(water_rhf):
    with pytest.raises(ValueError):
        FNOOptions(nvirtuals=2, occupation_cutoff=1.0e-6)
    with pytest.raises(ValueError):
        FNOOptions(source="mp2", targets=(FNOTarget("EA"),))
    with pytest.raises(NotImplementedError, match="provider"):
        CCSD(water_rhf, fno=FNOOptions(source="unregistered_adc2"))
    with pytest.raises(ValueError, match="ndelete"):
        CCSD(water_rhf, ndelete=1, fno=FNOOptions())
    with pytest.raises(ValueError, match="every"):
        CCSD(water_rhf, fno=FNOOptions(occupation_cutoff=3))
    with pytest.raises(ValueError, match="upstream method"):
        EOMCCSD(CCSD(water_rhf), fno=FNOOptions())


def test_nonfinite_and_nonpositive_density(water_rhf, monkeypatch):
    from ccpy.methods import fno as module

    def broken(ref, targets, need_density, full_result):
        n = ref.system.nunoccupied_alpha
        return FNOEvaluation(ref.energy, {None: {"a": -np.eye(n), "b": -np.eye(n)}})

    monkeypatch.setitem(module._PROVIDERS, "bad", broken)
    with pytest.raises(ValueError, match="negative"):
        CCSD(water_rhf, fno=FNOOptions(source="bad"))


def test_sf_protected_open_shell():
    mol = Molecule("C 0 0 0; H 0 1.0 .6; H 0 -1.0 .6", "6-31g", auxiliary_basis=AUX,
                   multiplicity=3, verbose=False)
    mf = converged_scf(ROHF, mol)
    sf = SFEOMCCSD(mf, nfrozen=1, fno=FNOOptions(nvirtuals=3), **OPTS)
    sf.initial_guess("sfcis", multiplicity=1, nacto=2, nactu=2, roots_per_irrep={"A": 1})
    sf.energy()
    assert sf.converged
    na = sf.system.noccupied_alpha + sf.reference.nfrozen
    np.testing.assert_array_equal(sf.reference.mo_coeff["a"][:, :na], mf.mo_coeff[:, :na])
    np.testing.assert_allclose(sf.completed_omega, sf.omega, atol=1.0e-12)


def test_steom_sdt_fno():
    mf = converged_scf(RHF, Molecule("H 0 0 0; H 0 0 .74", "6-31g", auxiliary_basis=AUX, verbose=False))
    cc = CCSD(mf, fno=FNOOptions(nvirtuals=2), **OPTS)
    method = STEOMCC(cc, level="sdt", nroots=1, nacto=1, nactu=1)
    np.testing.assert_allclose(method.energy() - method.raw_energy(), cc.fno_result.delta_ground, atol=1.0e-12)
    assert method.converged


def test_selected_space_and_ao_operator(water_rhf):
    ref = prepare_fno(Reference(water_rhf, nfrozen=1), FNOOptions(nvirtuals=3))
    ref.system.set_active_space(2, 2)
    triples = get_active_triples_pspace(ref.system)
    method = CCSDT_P(ref, triples, **OPTS)
    assert method.energy() == pytest.approx(method.raw_energy() + ref.fno_result.delta_ground)
    assert method.converged
    ao = water_rhf.get_hcore()
    for s, value in ref.one_body_from_ao(ao).items():
        np.testing.assert_allclose(value, ref.h_mo[s][1:, 1:], atol=1.0e-12)
    stacked = ref.one_body_from_ao(np.stack((ao, 2 * ao)))
    np.testing.assert_allclose(stacked["a"][1], 2 * stacked["a"][0], atol=1.0e-12)


def test_symmetry_and_cumulative_selection():
    from ccpy.methods.fno import _select

    labels = {"a": ["A", "B", "A", "B"]}
    density = {"a": np.diag([.1, .1 - 1.0e-10, .001, 0])}
    fock = {"a": np.diag([3., 2., 1., 4.])}
    # Close occupations in different irreps must be retained together.
    rotations, _, population = _select([density], labels, fock, FNOOptions(nvirtuals=1), False)
    W, new_labels = rotations["a"]
    assert W.shape == (4, 2) and new_labels == ["B", "A"]
    assert population["a"][0] > .99
    rotations, _, _ = _select([density], labels, fock, FNOOptions(retained_population=.999), False)
    assert rotations["a"][0].shape[1] == 3


@pytest.mark.parametrize("uhf", [False, True])
def test_union_preserves_each_selected_subspace(uhf):
    from ccpy.methods.fno import _select

    labels = {"a": ["A"] * 4}
    fock = {"a": np.diag(np.arange(1., 5.))}
    densities = [{"a": np.diag([1., 0, 0, 0])}, {"a": np.diag([0, 1., 0, 0])}]
    if uhf:
        labels["b"], fock["b"] = labels["a"], fock["a"]
        # Different beta target union rank forces conservative spin closure.
        densities[0]["b"] = densities[1]["b"] = densities[0]["a"]
    options = FNOOptions(source="adc2", targets=(FNOTarget("EE", 0), FNOTarget("EE", 1)),
                         nvirtuals=1, space_policy="union")
    rotations, _, _ = _select(densities, labels, fock, options, uhf)
    W = rotations["a"][0]
    assert np.linalg.norm((np.eye(4) - W @ W.T) @ np.eye(4)[:, :2]) < 1.0e-12
    if uhf:
        assert rotations["a"][0].shape[1] == rotations["b"][0].shape[1]


def test_no_completion_and_reference_geometry(water_rhf):
    method = MP2(water_rhf, nfrozen=1, fno=FNOOptions(nvirtuals=3, energy_completion="none"))
    assert method.energy() == method.raw_energy()
    mol = water_rhf.mol
    moved = method.reference.with_molecule(mol.with_geometry(np.array([xyz for _, xyz in mol.atoms]) * 1.01))
    assert moved.fno_result.options == method.fno_options
    assert moved is not method.reference
    assert moved.fno_result.delta_ground != method.fno_result.delta_ground


def test_threshold_sweep(water_rhf):
    parent = Reference(water_rhf, nfrozen=1)
    reference_energy = MP2(parent).energy()
    counts = []
    for cutoff in (1.0e-2, 1.0e-3, 1.0e-4, 0):
        child = prepare_fno(parent, FNOOptions(occupation_cutoff=cutoff))
        counts.append(child.system.nunoccupied_alpha)
        assert MP2(child).energy() == pytest.approx(reference_energy, abs=1.0e-10)
        fvv = child.H.a.vv
        np.testing.assert_allclose(fvv, np.diag(np.diag(fvv)), atol=1.0e-9)
    assert counts == sorted(counts)
    assert counts[0] < counts[-1] == parent.system.nunoccupied_alpha


def test_uhf_equal_removals():
    from ccpy.methods.fno import _select

    density = {"a": np.diag([.1, .01, .0001, .00001]),
               "b": np.diag([.2, .1, .02, .000001])}
    labels = {s: ["A"] * 4 for s in density}
    focks = {s: np.diag(np.arange(1., 5.)) for s in density}
    rotations, _, _ = _select([density], labels, focks, FNOOptions(occupation_cutoff=.05), True)
    assert rotations["a"][0].shape[1] == rotations["b"][0].shape[1] == 2


def test_uhf_spin_matching_preserves_cross_irrep_degeneracy():
    from ccpy.methods.fno import _select

    labels = {s: ["A", "A", "B", "B"] for s in ("a", "b")}
    densities = {"a": np.diag([.1, .01, .01, .0001]),
                 "b": np.diag([.2, .1, .005, .0001])}
    focks = {s: np.diag(np.arange(1., 5.)) for s in labels}
    rotations, _, _ = _select([densities], labels, focks, FNOOptions(occupation_cutoff=.05), True)
    for s in labels:
        assert rotations[s][1].count("A") == 2
        assert rotations[s][1].count("B") == 1


def test_protected_canonical_virtuals(water_rhf):
    parent = Reference(water_rhf, nfrozen=1)
    child = prepare_fno(parent, FNOOptions(occupation_cutoff=3, protected_virtuals=(0, 2)))
    assert child.system.nunoccupied_alpha == 2
    no = parent.system.noccupied_alpha + parent.nfrozen
    W = child.orbital_transform["a"][no:, no:]
    protected = np.eye(parent.system.nunoccupied_alpha)[:, [0, 2]]
    np.testing.assert_allclose(W @ W.T @ protected, protected, atol=1.e-12)
    assert MP2(child).energy() == pytest.approx(MP2(parent).energy(), abs=1.e-10)
    with pytest.raises(ValueError, match="protected virtual index"):
        prepare_fno(parent, FNOOptions(protected_virtuals=(1000,)))


@pytest.mark.parametrize("fixture", ["water_rhf", "oh_rohf", "oh_uhf"])
def test_provider_protected_subspaces(fixture, request, monkeypatch):
    from ccpy.methods import fno as module

    def provider(ref, targets, need_density, full_result):
        energy, density = dfmp2(ref, need_density)
        subspaces = {}
        if need_density:
            for s, spin in (("a", "alpha"), ("b", "beta")):
                size = getattr(ref.system, f"nunoccupied_{spin}")
                subspaces[s] = np.arange(1., size + 1)[:, None]
        return FNOEvaluation(ref.energy + energy, {None: density} if need_density else {},
                             protected_subspaces=subspaces)

    monkeypatch.setitem(module._PROVIDERS, "protected", provider)
    parent = Reference(request.getfixturevalue(fixture), nfrozen=1)
    child = prepare_fno(parent, FNOOptions(source="protected", occupation_cutoff=3))
    for s, spin in (("a", "alpha"), ("b", "beta")):
        no = getattr(parent.system, f"noccupied_{spin}") + parent.nfrozen
        size = getattr(parent.system, f"nunoccupied_{spin}")
        vector = np.zeros(parent.mo_coeff[s].shape[1])
        vector[no:] = np.arange(1., size + 1)
        Q = child.orbital_transform[s]
        np.testing.assert_allclose(Q @ Q.T @ vector, vector, atol=1.e-11)
    assert child.fno_result.discarded["a"] == child.fno_result.discarded["b"]


def test_protected_space_symmetry_closure():
    from ccpy.methods.fno import _select

    labels = {"a": ["A", "B", "A", "B"]}
    density = {"a": np.diag([.1, .01, .001, 0])}
    fock = {"a": np.diag([3., 2., 1., 4.])}
    vector = np.array([[1.], [2.], [3.], [4.]]) * 1.e-40
    rotations, _, _ = _select([density], labels, fock, FNOOptions(occupation_cutoff=3), False,
                              protected_subspaces={"a": vector})
    W, irreps = rotations["a"]
    assert W.shape == (4, 2) and sorted(irreps) == ["A", "B"]
    np.testing.assert_allclose(W @ W.T @ (vector / 1.e-40), vector / 1.e-40, atol=1.e-12)


def test_properties_use_working_coefficients(reduced_cc):
    from ccpy import LRCCSD
    from ccpy.interfaces.scf_tools import get_multipole_integral

    ref = reduced_cc.reference
    operators, expectation = get_multipole_integral(1, ref)
    parent_ops, parent_expectation = get_multipole_integral(1, ref.fno_parent)
    np.testing.assert_allclose(expectation, parent_expectation, atol=1.e-12)
    no = ref.system.noccupied_alpha
    nfrozen = ref.nfrozen
    W = ref.orbital_transform["a"][nfrozen:, nfrozen:]
    from ccpy.integrals import integrals

    ao = np.array(integrals.emultipole1(ref.molecule)[1:4])
    for k, operator in enumerate(operators):
        direct = ref.one_body_operator_from_ao(ao[k])
        np.testing.assert_allclose(operator.a.ov, direct.a.ov, atol=1.e-12)
        np.testing.assert_allclose(operator.a.vv, direct.a.vv, atol=1.e-12)
        np.testing.assert_allclose(operator.a.vv, W[no:, no:].T @ parent_ops[k].a.vv @ W[no:, no:], atol=1.e-12)
    lr = LRCCSD(reduced_cc, operators[2])
    assert np.isfinite(lr.property()) and lr.converged
    assert lr.energy() == reduced_cc.energy()
    with pytest.raises(ValueError, match="different orbital reference"):
        LRCCSD(reduced_cc, parent_ops[2])
    # Even identical dimensions cannot bypass an explicit reference binding.
    other = prepare_fno(ref.fno_parent, reduced_cc.fno_options)
    with pytest.raises(ValueError, match="different orbital reference"):
        LRCCSD(reduced_cc, other.one_body_operator_from_ao(ao[2]))


@pytest.mark.parametrize("fixture", ["oh_rohf", "oh_uhf"])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_open_shell_multipoles(fixture, order, request):
    from ccpy.interfaces.scf_tools import get_multipole_integral

    parent = Reference(request.getfixturevalue(fixture), nfrozen=1)
    child = prepare_fno(parent, FNOOptions(nvirtuals=3))
    original, expectation = get_multipole_integral(order, parent)
    reduced, reduced_expectation = get_multipole_integral(order, child)
    np.testing.assert_allclose(reduced_expectation, expectation, atol=1.e-11)
    for full, small in zip(np.asarray(original, dtype=object).flat, np.asarray(reduced, dtype=object).flat):
        for s, spin in (("a", "alpha"), ("b", "beta")):
            no = getattr(parent.system, f"noccupied_{spin}")
            block = getattr(full, s)
            matrix = np.block([[block.oo, block.ov], [block.vo, block.vv]])
            Q = child.orbital_transform[s][1:, 1:]
            projected = Q.T @ matrix @ Q
            np.testing.assert_allclose(getattr(small, s).ov, projected[:no, no:], atol=1.e-11)
            np.testing.assert_allclose(getattr(small, s).vv, projected[no:, no:], atol=1.e-11)


@pytest.mark.parametrize("family", ["CC", "EE", "IP", "EA", "DEA"])
def test_stale_selected_spaces_rejected(reduced_cc, family):
    from ccpy import DEAEOM4_P, EAEOM3_P, EOMCCSDT_P, IPEOM3_P

    if family in ("CC", "EE"):
        spins = ("aaa", "aab", "abb", "bbb")
        rows = {s: np.ones((1, 6), dtype=np.int32) for s in spins}
        cls = CCSDT_P if family == "CC" else EOMCCSDT_P
    else:
        spins = ("abaa", "abab", "abbb") if family == "DEA" else ("aaa", "aab", "abb")
        rows = {s: np.ones((1, 6 if family == "DEA" else 5), dtype=np.int32) for s in spins}
        cls = {"IP": IPEOM3_P, "EA": EAEOM3_P, "DEA": DEAEOM4_P}[family]
    rows[spins[0]][0, 0] = 1000
    with pytest.raises(ValueError, match="working orbital dimensions"):
        cls(reduced_cc, rows)
