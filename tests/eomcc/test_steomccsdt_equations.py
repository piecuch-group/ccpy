"""Independent small Fock-space checks of the enlarged-model transformation."""

from itertools import combinations, product
from types import SimpleNamespace

import numpy as np
import pytest

from ccpy.eomcc.steomccsdt import (
    ExtendedAction,
    NormalOperator,
    SDConfigurations,
    auxiliary_model,
    bits,
    enlarged_transformation,
    normal_product_apply,
    rows,
    state_from_row,
)


def explicit_apply(ops, state):
    phase = 1
    for create, p in reversed(ops):
        if bool(state & (1 << p)) == create:
            return None, 0
        phase *= (-1) ** (state & ((1 << p) - 1)).bit_count()
        state ^= 1 << p
    return state, phase


def normal_order_without_contractions(ops):
    ops = list(ops)
    phase = 1
    for _ in range(len(ops)):
        for k in range(len(ops) - 1):
            if not ops[k][0] and ops[k + 1][0]:
                ops[k], ops[k + 1] = ops[k + 1], ops[k]
                phase *= -1
    for create in (True, False):
        labels = [p for c, p in ops if c == create]
        if len(set(labels)) != len(labels):
            return [], 0
    return ops, phase


def monomial_ops(c, a):
    return [(True, p) for p in bits(c)] + [(False, p) for p in reversed(list(bits(a)))]


def operator_matrix(terms, states):
    index = {s: k for k, s in enumerate(states)}
    out = np.zeros((len(states), len(states)))
    for coefficient, ops in terms:
        for col, state in enumerate(states):
            target, phase = explicit_apply(ops, state)
            if phase and target in index:
                out[index[target], col] += coefficient * phase
    return out


def reference_exponential(S, states):
    # On neutral SD inputs, only 1, S and {SS}/2 can contribute through triples.
    terms = [
        (value, monomial_ops(c, a))
        for operator in (S.one2, S.one3, S.same, S.higher)
        for (c, a), value in operator.terms.items()
        if value
    ]
    second = []
    for (v, left), (w, right) in product(terms, repeat=2):
        ordered, phase = normal_order_without_contractions(left + right)
        if phase:
            second.append((0.5 * v * w * phase, ordered))
    return np.eye(len(states)) + operator_matrix(terms + second, states)


def random_models(no, nv, seed):
    rng = np.random.default_rng(seed)
    models = []
    for sector in ("ip", "ea"):
        states, B = auxiliary_model(no, nv, [no - 1], [0], sector)
        targets = [
            state_from_row(row, s, no, nv, sector)[0]
            for rank in (1, 2, 3)
            for s, row in rows(no, nv, rank, sector)
        ]
        embedding = np.zeros((len(targets), len(states)))
        for k, state in enumerate(states):
            embedding[targets.index(state), k] = 1.0
        projected = embedding @ B.toarray()
        random = rng.normal(size=projected.shape) * 0.015
        random -= projected @ (projected.T @ random)
        omega = (projected + random) @ B.T
        models.append(
            SimpleNamespace(
                states=states,
                projection=(B @ B.T).toarray(),
                omega=dict(zip(targets, omega)),
            )
        )
    return models


def reference_hamiltonian(system, seed=20):
    from ccpy.models.integrals import Integral

    rng = np.random.default_rng(seed)
    no, nv = system.noccupied_alpha, system.nunoccupied_alpha
    n = no + nv
    f = rng.normal(size=(n, n))
    f[no:, :no] = 0
    w = rng.normal(size=(n, n, n, n))
    w = (w + w.transpose(1, 0, 3, 2)) / 2
    w[no:, no:, :no, :no] = 0
    anti = w - w.transpose(0, 1, 3, 2)
    H = Integral(system, 2, {"a": f, "b": f, "aa": anti, "ab": w, "bb": anti})

    def qp(create, p, spin):
        occupied = p < no
        label = p + spin * no if occupied else 2 * no + (p - no) + spin * nv
        return (not create if occupied else create, label)

    terms = []
    for spin, p, q in product(range(2), range(n), range(n)):
        ops, phase = normal_order_without_contractions(
            [qp(True, p, spin), qp(False, q, spin)]
        )
        if phase:
            terms.append((phase * f[p, q], ops))
    for sig, tau, p, q, r, s in product(range(2), range(2), *([range(n)] * 4)):
        ops, phase = normal_order_without_contractions(
            [qp(True, p, sig), qp(True, q, tau), qp(False, s, tau), qp(False, r, sig)]
        )
        if phase:
            terms.append((phase * 0.5 * w[p, q, r, s], ops))
    return H, terms


@pytest.mark.parametrize("seed", [12, 35])
def test_normal_product_against_operator_strings(seed):
    rng = np.random.default_rng(seed)
    left, right = NormalOperator(), NormalOperator()
    for operator in (left, right):
        for _ in range(20):
            c = sum(1 << int(x) for x in rng.choice(8, 3, replace=False))
            a = 1 << int(rng.integers(8))
            operator.add(c, a, rng.normal())
    for labels in combinations(range(8), 4):
        state = sum(1 << p for p in labels)
        actual = normal_product_apply(left, right, {state: 1.0}, 8)
        expected = {}
        for (c, a), v in left.terms.items():
            for (d, b), w in right.terms.items():
                ops, phase = normal_order_without_contractions(
                    monomial_ops(c, a) + monomial_ops(d, b)
                )
                target, sign = explicit_apply(ops, state) if phase else (None, 0)
                if sign:
                    expected[target] = expected.get(target, 0.0) + phase * sign * v * w
        for target in set(actual) | set(expected):
            assert np.isclose(
                actual.get(target, 0), expected.get(target, 0), atol=1.0e-12
            )


@pytest.mark.parametrize("seed", [8, 31])
def test_enlarged_model_wave_operator_and_sd_action(seed):
    from ccpy.models.operators import ClusterOperator

    no = nv = 2
    system = SimpleNamespace(
        noccupied_alpha=no,
        noccupied_beta=no,
        nunoccupied_alpha=nv,
        nunoccupied_beta=nv,
        norbitals=no + nv,
    )
    ip, ea = random_models(no, nv, seed)
    S = enlarged_transformation(ip, ea, no, nv)
    # Auxiliary model columns must reproduce the prescribed normalized Omega
    # through 3h2p/3p2h, including cross-IP/EA subtractions.
    for model in (ip, ea):
        for col, state in enumerate(model.states):
            vector = {
                s: model.projection[k, col]
                for k, s in enumerate(model.states)
                if model.projection[k, col]
            }
            actual = dict(vector)
            for degree in (3, 5):
                for operator in (S.one2, S.one3, S.same, S.higher):
                    for target, value in operator.apply(vector, degree).items():
                        actual[target] = actual.get(target, 0.0) + value
            for target, values in model.omega.items():
                assert np.isclose(actual.get(target, 0), values[col], atol=2.0e-13)
    H, hterms = reference_hamiltonian(system)
    T = ClusterOperator(system, order=2)
    action = ExtendedAction(system, H, T, S, 1)
    cfg = action.configurations
    all_states = [
        s
        for s in range(1 << (2 * (no + nv)))
        if all(
            sum(bool(s & (1 << (spin * no + i))) for i in range(no))
            == sum(bool(s & (1 << (2 * no + spin * nv + a))) for a in range(nv))
            for spin in (0, 1)
        )
    ]
    U = reference_exponential(S, all_states)
    HH = operator_matrix(hterms, all_states)
    indices = [all_states.index(s) for s in cfg.states]
    retained = [k for k, s in enumerate(all_states) if s.bit_count() <= 6]
    # Standard extended STEOM drops all implicit quadruples before Hbar acts.
    lifted = U[:, indices].copy()
    lifted[[k for k in range(len(all_states)) if k not in retained], :] = 0
    metric = U[np.ix_(indices, indices)]
    exact = np.linalg.solve(metric, (HH @ lifted)[indices])
    calculated = np.column_stack([action.raw_matvec(v) for v in np.eye(len(indices))])
    np.testing.assert_allclose(calculated, exact, atol=2.0e-12)
    # This check would fail if the enlarged-model doubles metric were omitted.
    assert np.linalg.norm(metric - np.eye(len(indices))) > 1.0e-4


def test_spin_dimensions_and_model_satellites():
    system = SimpleNamespace(
        noccupied_alpha=1,
        noccupied_beta=1,
        nunoccupied_alpha=2,
        nunoccupied_beta=2,
        norbitals=3,
    )
    assert SDConfigurations(system, 1).basis.shape[1] == 5
    assert SDConfigurations(system, 3).basis.shape[1] == 3
    for sector in ("ip", "ea"):
        states, B = auxiliary_model(2, 2, [1], [0], sector)
        assert B.shape[1] > 2
        np.testing.assert_allclose(
            (B.T @ B).toarray(), np.eye(B.shape[1]), atol=1.0e-13
        )
        assert any(s.bit_count() == 3 for s in states)


@pytest.mark.parametrize("sector", ["ip", "ea"])
def test_enlarged_overlap_normalization_and_root_mixing(sector):
    from ccpy.eomcc.steomccsdt import array_index, normalize_auxiliary
    from ccpy.models.operators import FockOperator

    no = nv = 2
    system = SimpleNamespace(
        noccupied_alpha=no,
        noccupied_beta=no,
        nunoccupied_alpha=nv,
        nunoccupied_beta=nv,
        norbitals=no + nv,
    )
    expected = random_models(no, nv, 78)[0 if sector == "ip" else 1]
    model_states, B = auxiliary_model(no, nv, [1], [0], sector)
    rng = np.random.default_rng(79)
    mix = np.eye(B.shape[1]) + 0.1 * rng.normal(size=(B.shape[1],) * 2)
    amplitudes = {s: values @ B @ mix for s, values in expected.omega.items()}
    roots = []
    for k in range(B.shape[1]):
        R = FockOperator(system, 2, 3) if sector == "ip" else FockOperator(system, 3, 2)
        for rank in (1, 2, 3):
            for spin, row in rows(no, nv, rank, sector):
                state, phase = state_from_row(row, spin, no, nv, sector)
                getattr(R, spin)[array_index(row, sector)] = (
                    phase * amplitudes[state][k]
                )
        roots.append(R)
    result = SimpleNamespace(R=roots, excitations=None)
    actual = normalize_auxiliary(result, model_states, B, no, nv, sector)
    for state, values in expected.omega.items():
        np.testing.assert_allclose(actual.omega[state], values, atol=1.0e-13)
    # The model is recovered as the identity on its spin-adapted subspace.
    np.testing.assert_allclose(
        np.array([actual.omega[s] for s in model_states]),
        (B @ B.T).toarray(),
        atol=1.0e-13,
    )
    roots[1] = roots[0]
    with pytest.raises(ValueError, match="Ill-conditioned"):
        normalize_auxiliary(result, model_states, B, no, nv, sector)


def test_singular_neutral_metric_is_rejected():
    system = SimpleNamespace(
        noccupied_alpha=2,
        noccupied_beta=2,
        nunoccupied_alpha=2,
        nunoccupied_beta=2,
        norbitals=4,
    )
    same = NormalOperator()
    # Project out a satellite component, making a neutral doubles row singular.
    satellite = (1 << 0) | (1 << 2) | (1 << 6)
    same.add(satellite, satellite, -1.0)
    S = SimpleNamespace(
        one2=NormalOperator(), one3=NormalOperator(), same=same, higher=NormalOperator()
    )
    with pytest.raises(ValueError, match="Singular neutral"):
        ExtendedAction(system, None, None, S, 1)
