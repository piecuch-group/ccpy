import itertools
from types import SimpleNamespace as NS
import numpy as np
from ccpy.eomcc.steomccsd import build_g, singles_matrix

import pytest


@pytest.mark.parametrize("seed", [8, 19])
@pytest.mark.parametrize("multiplicity", [1, 3])
@pytest.mark.parametrize("full_active", [False, True])
def test_normal_ordered_determinant_oracle(seed, multiplicity, full_active):
    # Independent quasiparticle Wick algebra. (creation?, spin orbital).
    no = nv = 2
    n = no + nv
    ns = 2 * n
    rng = np.random.default_rng(seed)
    f = rng.normal(size=(n, n))
    f[no:, :no] = 0
    w = rng.normal(size=(n, n, n, n))
    w = (w + w.transpose(1, 0, 3, 2)) / 2
    w[no:, no:, :no, :no] = 0
    sl = {"o": slice(no), "v": slice(no, n)}
    blocks = NS(
        **{
            "".join(b): w[tuple(sl[t] for t in b)]
            for b in itertools.product("ov", repeat=4)
        }
    )
    H = NS(a=NS(oo=f[:no, :no], ov=f[:no, no:], vv=f[no:, no:]), ab=blocks)
    occ = np.array([0, 1]) if full_active else np.array([1])
    vir = np.array([0, 1]) if full_active else np.array([0])
    ip = rng.normal(size=(len(occ), no, no, nv)) * 0.1
    ea = rng.normal(size=(len(vir), nv, nv, no)) * 0.1

    def normal(ops):
        ops = list(ops)
        sign = 1
        for i in range(len(ops)):
            for j in range(len(ops) - i - 1):
                if not ops[j][0] and ops[j + 1][0]:
                    ops[j], ops[j + 1] = ops[j + 1], ops[j]
                    sign = -sign
        for create in (True, False):
            labels = [k for c, k in ops if c == create]
            if len(set(labels)) < len(labels):
                return 0, []
        return sign, ops

    def phys(ops):
        return [(not c if p % n < no else c, p) for c, p in ops]

    # Restrict to N_alpha=N_beta=no physical electrons.
    states = [
        s
        for s in range(1 << ns)
        if all(
            sum((s >> (spin * n + p)) & 1 for p in range(no))
            == sum((s >> (spin * n + p)) & 1 for p in range(no, n))
            for spin in (0, 1)
        )
    ]
    lookup = {s: i for i, s in enumerate(states)}
    dim = len(states)

    def matrix(terms):
        out = np.zeros((dim, dim))
        for coeff, ops in terms:
            if coeff == 0:
                continue
            for col, s in enumerate(states):
                sign = coeff
                for create, p in reversed(ops):
                    bit = (s >> p) & 1
                    if bit == create:
                        break
                    sign *= (-1) ** ((s & ((1 << p) - 1)).bit_count())
                    s ^= 1 << p
                else:
                    if s in lookup:
                        out[lookup[s], col] += sign
        return out

    def term(coeff, ops):
        sign, ordered = normal(phys(ops))
        return coeff * sign, ordered

    ht = []
    for spin, p, q in itertools.product(range(2), range(n), range(n)):
        ht.append(term(f[p, q], [(True, p + spin * n), (False, q + spin * n)]))
    for sig, tau, p, q, r, s in itertools.product(
        range(2), range(2), *([range(n)] * 4)
    ):
        ht.append(
            term(
                0.5 * w[p, q, r, s],
                [
                    (True, p + sig * n),
                    (True, q + tau * n),
                    (False, s + tau * n),
                    (False, r + sig * n),
                ],
            )
        )
    st = []
    for m, sig, tau, i, j, b in itertools.product(
        range(len(occ)), range(2), range(2), range(no), range(no), range(nv)
    ):
        st.append(
            term(
                ip[m, i, j, b],
                [
                    (True, occ[m] + sig * n),
                    (True, no + b + tau * n),
                    (False, j + tau * n),
                    (False, i + sig * n),
                ],
            )
        )
    for e, sig, tau, a, b, j in itertools.product(
        range(len(vir)), range(2), range(2), range(nv), range(nv), range(no)
    ):
        st.append(
            term(
                ea[e, a, b, j],
                [
                    (True, no + a + sig * n),
                    (True, no + b + tau * n),
                    (False, j + tau * n),
                    (False, no + vir[e] + sig * n),
                ],
            )
        )
    st = [(c, o) for c, o in st if c]
    st2 = []
    for (c, o), (d, p) in itertools.product(st, repeat=2):
        sign, q = normal(o + p)
        if sign:
            st2.append((0.5 * c * d * sign, q))
    HM = matrix(ht)
    U = np.eye(dim) + matrix(st) + matrix(st2)
    vac = np.eye(dim)[:, lookup[0]]
    for mult in (multiplicity,):
        B = []
        for a, i in itertools.product(range(nv), range(no)):
            terms = [
                term(1 / np.sqrt(2), [(True, no + a), (False, i)]),
                term(
                    (1 if mult == 1 else -1) / np.sqrt(2),
                    [(True, n + no + a), (False, n + i)],
                ),
            ]
            B.append(matrix(terms) @ vac)
        B = np.array(B).T
        expected = B.T @ HM @ U @ B
        actual = singles_matrix(build_g(H, ip, ea, occ, vir), mult)
        np.testing.assert_allclose(actual, expected, atol=2.0e-12)


def test_model_normalization_and_root_mixing():
    from ccpy.eomcc.steomccsd import amplitudes_from_eom

    rng = np.random.default_rng(25)
    r1 = np.array([[1.0, 0.1], [0.2, 1.0], [0.1, 0.3]])
    doubles = rng.normal(size=(3, 4, 3, 2))
    vectors = [
        NS(a=r1[:, k], aa=np.zeros((3, 4, 3)), ab=doubles[..., k]) for k in range(2)
    ]
    sip, active, _ = amplitudes_from_eom(vectors, [0, 1], "ip")
    expected = -np.einsum("ibjr,rm->mijb", doubles, np.linalg.inv(r1[:2]))
    np.testing.assert_allclose(sip, expected)
    mix = np.array([[2.0, -0.3], [0.5, 1.0]])
    mixed = [
        NS(
            a=(r1 @ mix)[:, k],
            aa=np.zeros((3, 4, 3)),
            ab=np.einsum("ibjr,rs->ibjs", doubles, mix)[..., k],
        )
        for k in range(2)
    ]
    np.testing.assert_allclose(amplitudes_from_eom(mixed, active, "ip")[0], sip)
    vectors[1] = vectors[0]
    with pytest.raises(ValueError, match="Ill-conditioned"):
        amplitudes_from_eom(vectors, active, "ip")


def test_metric_and_filtering():
    from ccpy.eomcc.steomccsd import amplitudes_from_eom, singles_character

    a = NS(
        a=np.array([1.0, 0.0]),
        aa=np.array([[[0.0, 1.0]], [[-1.0, 0.0]]]),
        ab=np.zeros((2, 1, 2)),
    )
    b = NS(a=np.array([0.0, 1.0]), aa=np.zeros((2, 1, 2)), ab=np.zeros((2, 1, 2)))
    assert singles_character(a) == 0.5
    _, labels, keep = amplitudes_from_eom([a, b], [0, 1], "ip", threshold=0.9)
    assert keep == [1]
    np.testing.assert_array_equal(labels, [1])
    with pytest.raises(ValueError, match="No IP roots"):
        amplitudes_from_eom([a], [0], "ip", threshold=0.9)


def test_natural_orbitals_and_semicanonical_blocks():
    from ccpy.eomcc.steomccsd import natural_orbital_space

    vectors = np.zeros((2, 4, 3))
    vectors[0, 0, 0] = 1
    vectors[1, 1, 1] = 1
    rng = np.random.default_rng(35)
    qo = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    qv = np.linalg.qr(rng.normal(size=(4, 4)))[0]
    vectors = np.einsum("ab,rbi,ij->raj", qv, vectors, qo.T)
    fo = np.diag([-2.0, -1.0, -0.5])
    fv = np.diag([0.1, 0.3, 0.6, 1.0])
    uo, uv, occupied, virtual, wo, wv = natural_orbital_space(vectors, fo, fv, 0.7, 0.7)
    # Keep the entire degenerate NO pair, including at a threshold cutting that pair.
    assert len(occupied) == len(virtual) == 2
    np.testing.assert_allclose(wo, [0.5, 0.5, 0.0], atol=1.0e-14)
    np.testing.assert_allclose(wv, [0.5, 0.5, 0.0, 0.0], atol=1.0e-14)
    for u, f, active in [(uo, fo, occupied), (uv, fv, virtual)]:
        np.testing.assert_allclose(u.T @ u, np.eye(len(u)), atol=1.0e-14)
        for block in [active, np.setdiff1d(np.arange(len(u)), active)]:
            ff = (u.T @ f @ u)[np.ix_(block, block)]
            np.testing.assert_allclose(ff, np.diag(np.diag(ff)), atol=1.0e-14)
    with pytest.raises(ValueError, match="threshold"):
        natural_orbital_space(vectors, fo, fv, 0.0, 0.99)


def test_degenerate_numerical_complex_pair():
    from ccpy.eomcc.steomccsd import diagonalize_singles

    matrix = np.array([[1.0, 1.0e-12], [-1.0e-12, 1.0]])
    omega, vectors = diagonalize_singles(matrix, 1.0e-9)
    np.testing.assert_array_equal(omega.imag, 0.0)
    np.testing.assert_array_equal(vectors.imag, 0.0)
    np.testing.assert_allclose(vectors.T @ vectors, np.eye(2), atol=1.0e-14)
    assert np.max(np.linalg.norm(matrix @ vectors - vectors * omega, axis=0)) < 1.0e-9
    # A truly complex pair must remain visible to the caller.
    omega, _ = diagonalize_singles(np.array([[1.0, 0.1], [-0.1, 1.0]]), 1.0e-9)
    np.testing.assert_allclose(abs(omega.imag), 0.1)


def test_cisd_densities_include_doubles():
    from ccpy.eomcc.steomccsd import (
        guess_densities,
        natural_orbital_space_from_densities,
    )
    from ccpy.eomcc.steomccsdt import SDConfigurations, act
    from ccpy.models.operators import ClusterOperator

    system = NS(
        noccupied_alpha=2,
        noccupied_beta=2,
        nunoccupied_alpha=2,
        nunoccupied_beta=2,
        norbitals=4,
    )
    cfg = SDConfigurations(system, 1)
    rng = np.random.default_rng(66)
    coefficients = rng.normal(size=len(cfg.states))
    coefficients /= np.linalg.norm(coefficients)
    R = cfg.to_operator(coefficients)
    do, dv = guess_densities([R])
    values = dict(zip(cfg.states, coefficients))
    expected = []
    for offset in (0, 4):
        density = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                for spin in (0, 1):
                    for state, value in values.items():
                        target, phase = act(
                            1 << (offset + 2 * spin + i),
                            1 << (offset + 2 * spin + j),
                            state,
                        )
                        if phase:
                            density[i, j] += 0.5 * values.get(target, 0) * phase * value
        expected.append(density)
    np.testing.assert_allclose(do, expected[0], atol=1.0e-13)
    np.testing.assert_allclose(dv, expected[1], atol=1.0e-13)
    # A pure double has no CIS projection, but must still select its orbitals.
    R = ClusterOperator(system, order=2)
    R.ab[1, 1, 0, 0] = 1.0
    do, dv = guess_densities([R])
    np.testing.assert_array_equal(np.diag(do), [1.0, 0.0])
    np.testing.assert_array_equal(np.diag(dv), [0.0, 1.0])
    uo, uv, occupied, virtual, *_ = natural_orbital_space_from_densities(
        do, dv, -np.eye(2), np.eye(2)
    )
    np.testing.assert_allclose(abs(uo[:, occupied[0]]), [1.0, 0.0])
    np.testing.assert_allclose(abs(uv[:, virtual[0]]), [0.0, 1.0])
