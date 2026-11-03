"""General-reference ADC checks independent of the canonical PySCF equations."""
from itertools import combinations
from types import SimpleNamespace

import numpy as np
import pytest

from ccpy.adc.charged import ChargedADC2
from ccpy.methods.hamiltonian import DFHamiltonian
from ccpy.models.system import System


def synthetic_reference(electrons):
    rng = np.random.default_rng(95)
    system = System(electrons, 3, 2 if electrons == 3 else 1, 0)
    factors = rng.normal(scale=.1, size=(4, 3, 3))
    factors += factors.transpose(0, 2, 1)
    focks = []
    for no in (system.noccupied_alpha, system.noccupied_beta):
        eps = np.r_[np.linspace(-1.4, -.7, no), np.linspace(.3, 1.1, 3 - no)]
        f = np.diag(eps)
        f[:no, no:] = rng.normal(scale=.03, size=(no, 3 - no))
        f[no:, :no] = f[:no, no:].T
        focks.append(f)
    return SimpleNamespace(system=system, nfrozen=0, is_uhf=False,
                           H=DFHamiltonian(system, *focks, factors, factors))


def determinant_oracle(kernel):
    """Explicit second-order BCH operators on the entire tiny Fock space."""
    o, v = len(kernel.eo), len(kernel.ev)
    n = o + v
    dim = 1 << n
    annihilate = []
    for p in range(n):
        matrix = np.zeros((dim, dim))
        for ket in range(dim):
            if ket & (1 << p):
                matrix[ket ^ (1 << p), ket] = (-1) ** ((ket & ((1 << p) - 1)).bit_count())
        annihilate.append(matrix)
    create = [matrix.T for matrix in annihilate]
    B = np.concatenate((np.concatenate((kernel.oo, kernel.ov), axis=2),
                        np.concatenate((kernel.ov.transpose(0, 2, 1), kernel.vv), axis=2)), axis=1)
    g = np.einsum("Ppr,Pqs->pqrs", B, B) - np.einsum("Pps,Pqr->pqrs", B, B)
    F = np.diag(np.r_[kernel.eo, kernel.ev])
    F[:o, o:], F[o:, :o] = kernel.fov, kernel.fov.T
    h = F - np.einsum("piqi->pq", g[:, :o, :, :o])
    H = sum(h[p, q] * create[p] @ annihilate[q] for p in range(n) for q in range(n))
    pairs = list(combinations(range(n), 2))
    operators = {(p, q): create[p] @ create[q] for p, q in pairs}
    for p, q in pairs:
        for r, s in pairs:
            H += g[p, q, r, s] * operators[p, q] @ operators[r, s].T
    phi = np.zeros(dim)
    phi[(1 << o) - 1] = 1
    H -= (phi @ H @ phi) * np.eye(dim)
    H0 = sum(e * create[p] @ annihilate[p] for p, e in enumerate(np.r_[kernel.eo, kernel.ev]))
    H0 -= np.sum(kernel.eo) * np.eye(dim)
    T = sum(kernel.t1[i, a] * create[o + a] @ annihilate[i] for i in range(o) for a in range(v))
    for i, j in combinations(range(o), 2):
        for a, b in combinations(range(v), 2):
            T += kernel.t[i, j, a, b] * create[o + a] @ create[o + b] @ annihilate[j] @ annihilate[i]
    A = T - T.T
    def comm(X, Y):
        return X @ Y - Y @ X
    first = H + comm(H0, A)
    second = comm(H - H0, A) + .5 * comm(comm(H0, A), A)
    assert phi @ second @ phi == pytest.approx(kernel.correlation_energy, abs=1.e-12)
    return create, annihilate, phi, A, first, second


@pytest.mark.parametrize("electrons", [2, 3])
@pytest.mark.parametrize("sector", ["IP", "EA"])
def test_general_reference_primary_and_density_from_bch(electrons, sector):
    kernel = ChargedADC2(synthetic_reference(electrons), sector)
    create, annihilate, phi, A, first, second = determinant_oracle(kernel)
    o, v = len(kernel.eo), len(kernel.ev)
    leading = [annihilate[i] @ phi for i in range(o)] if sector == "IP" else [create[o + a] @ phi for a in range(v)]
    leading = np.column_stack(leading)
    physical = leading.T @ (first + second) @ leading - kernel.correlation_energy * np.eye(leading.shape[1])
    np.testing.assert_allclose(kernel.primary, physical, atol=1.e-12)
    rng = np.random.default_rng(43)
    vector = rng.normal(size=kernel.space.ndim)
    vector /= np.linalg.norm(vector)
    X, Y = kernel.space.unpack(vector)
    psi1 = leading @ X
    psi2 = np.zeros_like(phi)
    if sector == "IP":
        for a in range(v):
            for i, j in combinations(range(o), 2):
                psi2 += Y[a, i, j] * create[o + a] @ annihilate[j] @ annihilate[i] @ phi
    else:
        for i in range(o):
            for a, b in combinations(range(v), 2):
                psi2 += Y[i, a, b] * create[o + a] @ create[o + b] @ annihilate[i] @ phi
    # Match the documented charged ISR satellite phase to physical determinants.
    phase = -1 if sector == "IP" else 1
    psi2 *= phase
    assert psi1 @ first @ psi2 == pytest.approx(np.dot(X, kernel.coupling_up(Y)), abs=1.e-12)
    density = np.zeros((v, v))
    for a in range(v):
        for b in range(v):
            operator = create[o + a] @ annihilate[o + b]
            O1 = operator @ A - A @ operator
            O2 = .5 * (O1 @ A - A @ O1)
            density[a, b] = ((psi1 + psi2) @ operator @ (psi1 + psi2) + psi1 @ O1 @ psi1
                              + psi1 @ O1 @ psi2 + psi2 @ O1 @ psi1 + psi1 @ O2 @ psi1)
    expected = kernel.density_to_working(density)
    actual = kernel.virtual_density(vector)
    for s in ("a", "b"):
        np.testing.assert_allclose(actual[s], expected[s], atol=1.e-12)


def rotated_reference(parent):
    import copy
    from ccpy.methods.hamiltonian import build_fock

    rng = np.random.default_rng(147)
    child = copy.copy(parent)
    child.system = copy.deepcopy(parent.system)
    child.mo_coeff, child.B_mo, child.h_mo = {}, {}, {}
    Q = {}
    n = parent.mo_coeff["a"].shape[1]
    nf = parent.nfrozen
    na = nf + parent.system.noccupied_alpha
    nb = nf + parent.system.noccupied_beta
    for s, no in (("a", na), ("b", nb)):
        q = np.eye(n)
        # Restricted open-shell rotations preserve the common occupied subspaces.
        occupied = no if parent.is_uhf else nb
        virtual = no if parent.is_uhf else na
        for start, stop in ((nf, occupied), (virtual, n)):
            q[start:stop, start:stop] = np.linalg.qr(rng.normal(size=(stop - start, stop - start)))[0]
        if s == "b" and not parent.is_uhf:
            q = Q["a"]
        Q[s] = q
        child.mo_coeff[s] = parent.mo_coeff[s] @ q
        child.h_mo[s] = q.T @ parent.h_mo[s] @ q
        child.B_mo[s] = np.einsum("Ppq,pi,qj->Pij", parent.B_mo[s], q, q, optimize=True)
    Fa, Fb = build_fock(child.h_mo["a"], child.h_mo["b"], child.B_mo["a"], child.B_mo["b"], na, nb)
    child.H = DFHamiltonian(child.system, Fa[nf:, nf:], Fb[nf:, nf:],
                            child.B_mo["a"][:, nf:, nf:], child.B_mo["b"][:, nf:, nf:])
    child._legacy_H = None
    rotations = {s: {"o": Q[s][nf:no, nf:no], "v": Q[s][no:, no:]}
                 for s, no in (("a", na), ("b", nb))}
    return child, rotations


def rotate_amplitudes(vector, rotations, sector):
    import copy
    result = copy.deepcopy(vector)
    for name in vector.spin_cases:
        if sector == "EE":
            spins, kinds = name * 2, "v" * len(name) + "o" * len(name)
        elif name == "a":
            spins, kinds = "a", "o" if sector == "IP" else "v"
        else:
            spins, kinds = "aaa" if name == "aa" else "abb", "ovo" if sector == "IP" else "vvo"
        value = getattr(vector, name)
        for axis, (spin, kind) in enumerate(zip(spins, kinds)):
            value = np.moveaxis(np.tensordot(rotations[spin][kind].T, value, axes=(1, axis)), 0, axis)
        setattr(result, name, value)
    return result


@pytest.mark.parametrize("fixture", ["water_rhf", "oh_rohf", "oh_uhf"])
@pytest.mark.parametrize("sector", ["EE", "IP", "EA"])
def test_noncanonical_covariance(fixture, sector, request):
    from ccpy import ADC2, EAADC2, IPADC2, Reference

    cls = {"EE": ADC2, "IP": IPADC2, "EA": EAADC2}[sector]
    parent = Reference(request.getfixturevalue(fixture), nfrozen=1)
    child, Q = rotated_reference(parent)
    original, rotated = cls(parent), cls(child)
    first, second = original.kernel, rotated.kernel
    assert first.correlation_energy == pytest.approx(second.correlation_energy, abs=1.e-12)
    rng = np.random.default_rng(38)
    R = original._new_operator()
    R.unflatten(rng.normal(size=R.ndim))
    Rnew = rotated._new_operator().from_cluster(rotate_amplitudes(R.as_cluster(), Q, sector))
    sigma = original._new_operator()
    sigma.unflatten(first.matvec(R.flatten()))
    expected = rotated._new_operator().from_cluster(rotate_amplitudes(sigma.as_cluster(), Q, sector))
    np.testing.assert_allclose(second.matvec(Rnew.flatten()), expected.flatten(), atol=1.e-10)
    D1, D2 = first.virtual_density(R.flatten()), second.virtual_density(Rnew.flatten())
    for s in ("a", "b"):
        np.testing.assert_allclose(D2[s], Q[s]["v"].T @ D1[s] @ Q[s]["v"], atol=1.e-11)


@pytest.mark.parametrize("fixture", ["water_rhf", "oh_rohf", "oh_uhf"])
@pytest.mark.parametrize("sector", ["IP", "EA"])
def test_noncanonical_davidson_energies(fixture, sector, request):
    from ccpy import EAADC2, IPADC2, Reference
    from conftest import OPTS

    cls = IPADC2 if sector == "IP" else EAADC2
    parent = Reference(request.getfixturevalue(fixture), nfrozen=1)
    child, _ = rotated_reference(parent)
    first, second = cls(parent, **OPTS), cls(child, **OPTS)
    settings = dict(nacto=parent.system.noccupied_alpha, nactu=parent.system.nunoccupied_beta,
                    roots_per_irrep={"A": 2})
    first.initial_guess(**settings)
    second.initial_guess(**settings)
    np.testing.assert_allclose(first.energy(), second.energy(), atol=1.e-8)
