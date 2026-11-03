"""Extended STEOM with a 1h+active(2h1p)/1p+active(2p1h) model.

The neutral eigenproblem is SD-sized. The wave operator is represented by sparse,
normal-ordered quasiparticle monomials, not a full Fock-space matrix. Rank-three
neutral vectors are induced, not independently optimized. The projection drops
implicit quadruples, following Nooijen & Lotrich JCP 113, 494 (2000), Eq.18,
and generalizes their Eq.17 to the enlarged auxiliary model.

Quasiparticle bits: occupied alpha, occupied beta, virtual alpha, virtual beta.
A monomial (creators, annihilators) uses ascending creators and descending
annihilators. Both are bit masks. Its coefficient therefore equals its matrix
element between the corresponding quasiparticle determinants.
"""

from collections import defaultdict
from itertools import combinations, product
from types import SimpleNamespace

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import LinearOperator, onenormest, splu


def bits(mask):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def wedge(left, right):
    if left & right:
        return 0
    return (
        -1 if sum((right & ((1 << p) - 1)).bit_count() for p in bits(left)) % 2 else 1
    )


def act(creators, annihilators, state):
    if state & annihilators != annihilators:
        return None, 0
    sign = 1
    for p in bits(annihilators):
        sign *= -1 if (state & ((1 << p) - 1)).bit_count() % 2 else 1
        state ^= 1 << p
    return (
        (state | creators, sign * wedge(creators, state))
        if not state & creators
        else (None, 0)
    )


def rows(no, nv, rank, sector=None):
    """Unique CCpy spin blocks, canonical particles first then holes (0-based)."""
    cases = ["a" * (rank - b) + "b" * b for b in range(rank + 1)]
    if sector:
        cases = cases[:-1]  # IP/EA remove/add alpha only
    for spin in cases:
        na = spin.count("a")
        pa = na - (sector == "ip")
        pb = rank - na
        ha = na - (sector == "ea")
        hb = rank - na
        for p, q, i, j in product(
            combinations(range(nv), pa),
            combinations(range(nv), pb),
            combinations(range(no), ha),
            combinations(range(no), hb),
        ):
            yield spin, tuple(p + q + i + j)


def state_from_row(row, spin, no, nv, sector=None):
    rank = len(spin)
    npart = rank - (sector == "ip")
    na = spin.count("a")
    pa = na - (sector == "ip")
    ha = na - (sector == "ea")
    particle, hole = row[:npart], row[npart:]
    labels = [2 * no + p + (0 if k < pa else nv) for k, p in enumerate(particle)]
    labels += [i + (0 if k < ha else no) for k, i in enumerate(hole)]
    if len(set(labels)) != len(labels):
        raise ValueError("Repeated same-spin orbital in an excitation")
    mask = sum(1 << p for p in labels)
    nh = len(hole)
    phase = (-1) ** (npart * nh + nh * (nh - 1) // 2)
    return mask, phase


def array_index(row, sector=None):
    if sector == "ip":
        npart = (len(row) - 1) // 2
        return (row[npart], *row[:npart], *row[npart + 1 :])
    return tuple(row)


def auxiliary_entries(R, no, nv, sector, rank, excitations=None):
    if rank == 3 and excitations is not None:
        iterator = (
            (s, tuple(int(x) - 1 for x in row), amp)
            for s in ("aaa", "aab", "abb")
            for row, amp in zip(excitations[s], getattr(R, s))
            if not np.all(row == 1)
        )
    else:
        iterator = (
            (s, row, getattr(R, s)[array_index(row, sector)])
            for s, row in rows(no, nv, rank, sector)
        )
    for spin, row, amplitude in iterator:
        if amplitude != 0:
            state, phase = state_from_row(row, spin, no, nv, sector)
            yield state, phase * amplitude


def spin_flip(mask, no, nv):
    labels = []
    for p in bits(mask):
        if p < 2 * no:
            labels.append(p + no if p < no else p - no)
        else:
            labels.append(p + nv if p < 2 * no + nv else p - nv)
    sign = (-1) ** sum(a > b for k, a in enumerate(labels) for b in labels[k + 1 :])
    return sum(1 << p for p in labels), sign


def spin_basis(states, no, nv, eigenvalue, lower=False):
    """Sparse orthonormal spin basis, diagonalizing only small connected S^2 blocks.

    eigenvalue refers to S_-S_+ (or S_+S_-). For neutral Ms=0 use S(S+1);
    for the auxiliary Ms=+/-1/2 doublets use 1.
    """
    rowmap, rr, cc, vv = {}, [], [], []
    for col, state in enumerate(states):
        for p in range(no + nv):
            alpha = p if p < no else 2 * no + p - no
            beta = p + no if p < no else 2 * no + nv + p - no
            dest, source = (beta, alpha) if lower else (alpha, beta)
            # Physical spin flip. Occupied physical creation is qp annihilation.
            occupied = p < no
            ops = [(not occupied, dest), (occupied, source)]
            current, value = state, 1
            for create, label in reversed(ops):
                if bool(current & (1 << label)) == create:
                    value = 0
                    break
                value *= (-1) ** ((current & ((1 << label) - 1)).bit_count())
                current ^= 1 << label
            if value:
                row = rowmap.setdefault(current, len(rowmap))
                rr.append(row)
                cc.append(col)
                vv.append(value)
    raising = sparse.coo_matrix(
        (vv, (rr, cc)), shape=(len(rowmap), len(states))
    ).tocsr()
    spin2 = (raising.T @ raising).tocsr()
    spin2.eliminate_zeros()
    _, labels = connected_components(spin2, directed=False)
    rr, cc, vv, ncol = [], [], [], 0
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        values, vectors = np.linalg.eigh(spin2[indices][:, indices].toarray())
        for k in np.flatnonzero(abs(values - eigenvalue) < 1.0e-8):
            for row, value in zip(indices, vectors[:, k]):
                if abs(value) > 1.0e-14:
                    rr.append(row)
                    cc.append(ncol)
                    vv.append(value)
            ncol += 1
    return sparse.coo_matrix((vv, (rr, cc)), shape=(len(states), ncol)).tocsr()


def auxiliary_model(no, nv, occupied, virtual, sector):
    states = []
    for rank in (1, 2):
        for spin, row in rows(no, nv, rank, sector):
            npart = rank - (sector == "ip")
            if rank == 2 and (
                any(p not in virtual for p in row[:npart])
                or any(i not in occupied for i in row[npart:])
            ):
                continue
            states.append(state_from_row(row, spin, no, nv, sector)[0])
    basis = spin_basis(states, no, nv, 1.0, lower=sector == "ea")
    return states, basis


def active_triples(no, nv, occupied, virtual, sector, minimum_active=1):
    """CCpy's active-hole (IP)/active-particle (EA) triples selection, arbitrary labels."""
    if minimum_active not in (1, 2, 3):
        raise ValueError("minimum_active must be 1, 2, or 3")
    result = {s: [] for s in ("aaa", "aab", "abb")}
    for spin, row in rows(no, nv, 3, sector):
        npart = 2 if sector == "ip" else 3
        count = (
            sum(i in occupied for i in row[npart:])
            if sector == "ip"
            else sum(a in virtual for a in row[:npart])
        )
        if count >= minimum_active:
            result[spin].append(np.array(row) + 1)
    return {
        s: np.asfortranarray(v, dtype=np.int32).reshape(-1, 5)
        if len(v)
        else np.ones((1, 5), dtype=np.int32, order="F")
        for s, v in result.items()
    }


class NormalOperator:
    def __init__(self, terms=None):
        self.terms = defaultdict(float, terms or {})
        self.groups = None

    def add(self, creators, annihilators, value):
        self.terms[creators, annihilators] += value
        self.groups = None

    def complete_spin(self, no, nv):
        for (creators, annihilators), value in list(self.terms.items()):
            c, sc = spin_flip(creators, no, nv)
            a, sa = spin_flip(annihilators, no, nv)
            self.add(c, a, sc * sa * value)
        return self

    def grouped(self):
        if self.groups is None:
            self.groups = defaultdict(list)
            for (c, a), v in self.terms.items():
                if v:
                    self.groups[a].append((c, v))
        return self.groups

    def applicable(self, state):
        labels = list(bits(state))
        groups = self.grouped()
        for count in (1, 3):
            for subset in combinations(labels, count):
                mask = sum(1 << p for p in subset)
                for c, v in groups.get(mask, ()):
                    yield c, mask, v

    def apply(self, vector, output_degree):
        result = defaultdict(float)
        for state, amplitude in vector.items():
            for c, a, value in self.applicable(state):
                if state.bit_count() + c.bit_count() - a.bit_count() != output_degree:
                    continue
                target, phase = act(c, a, state)
                if phase:
                    result[target] += phase * value * amplitude
        return dict(result)


def normal_product_apply(left, right, vector, output_degree):
    """{left right} on a sparse vector, excluding all inter-operator contractions."""
    result = defaultdict(float)
    for state, amplitude in vector.items():
        lterms, rterms = list(left.applicable(state)), list(right.applicable(state))
        for c, a, value in lterms:
            for d, b, other in rterms:
                if (
                    a & b
                    or c & d
                    or state.bit_count()
                    + c.bit_count()
                    + d.bit_count()
                    - a.bit_count()
                    - b.bit_count()
                    != output_degree
                ):
                    continue
                # Move A's annihilators past B's creators, then canonicalize
                # creation and annihilation strings (the latter descending).
                sign = (
                    (-1) ** (a.bit_count() * d.bit_count()) * wedge(c, d) * wedge(b, a)
                )
                target, phase = act(c | d, a | b, state)
                if phase:
                    result[target] += amplitude * value * other * sign * phase
    return dict(result)


def normalize_auxiliary(
    result, model_states, model_basis, no, nv, sector, condition_limit=1.0e10
):
    """Bloch intermediate normalization on the full spin-adapted auxiliary model."""
    nroot = len(result.R)
    if nroot != model_basis.shape[1]:
        raise ValueError(
            "One auxiliary eigenvector per spin-adapted model configuration is required"
        )
    data = defaultdict(lambda: np.zeros(nroot))
    for k, R in enumerate(result.R):
        excitations = result.excitations[k] if result.excitations is not None else None
        for rank in range(1, R.order + 1):
            for state, value in auxiliary_entries(R, no, nv, sector, rank, excitations):
                data[state][k] = value
    model = np.array([data[s] for s in model_states])
    overlap = model_basis.T @ model
    condition = np.linalg.cond(overlap)
    if not np.isfinite(condition) or condition > condition_limit:
        raise ValueError(
            f"Ill-conditioned {sector.upper()} enlarged-model overlap ({condition:.3e})"
        )
    masks = list(data)
    normalized = np.linalg.solve(overlap.T, np.array([data[s] for s in masks]).T).T
    omega = normalized @ model_basis.T
    projection = (model_basis @ model_basis.T).toarray()
    norms = sum(abs(v) ** 2 for v in data.values())
    result.model_weights = np.sum(abs(overlap) ** 2, axis=0) / norms
    result.condition_number = condition
    return SimpleNamespace(
        states=model_states, projection=projection, omega=dict(zip(masks, omega))
    )


def enlarged_transformation(ip, ea, no, nv):
    """Construct one-annihilator and three-annihilator S components.

    Omega P = R (P R)^-1 P. S_(c,1) follows from its 1h/1p columns.
    S_(3,3) = Q Omega P on the active satellite model. S_(5,3) also
    subtracts the already determined S_(3,1) action on that model.
    Both IP and EA one-annihilator components enter this subtraction.
    """
    one2, one3, same, higher = (NormalOperator() for _ in range(4))
    for model in (ip, ea):
        for col, state in enumerate(model.states):
            if state.bit_count() != 1:
                continue
            for target, values in model.omega.items():
                degree = target.bit_count()
                if degree in (3, 5):
                    (one2 if degree == 3 else one3).add(target, state, values[col])
    one2.complete_spin(no, nv)
    one3.complete_spin(no, nv)
    for model in (ip, ea):
        for col, state in enumerate(model.states):
            if state.bit_count() != 3:
                continue
            pcol = {
                s: model.projection[k, col]
                for k, s in enumerate(model.states)
                if model.projection[k, col]
            }
            known = one2.apply(pcol, 5)
            for target in set(model.omega) | set(known) | set(pcol):
                values = model.omega.get(target)
                value = values[col] if values is not None else 0.0
                if target.bit_count() == 3:
                    same.add(target, state, value - pcol.get(target, 0.0))
                elif target.bit_count() == 5:
                    higher.add(target, state, value - known.get(target, 0.0))
    same.complete_spin(no, nv)
    higher.complete_spin(no, nv)
    return SimpleNamespace(one2=one2, one3=one3, same=same, higher=higher)


class SDConfigurations:
    """Unique Ms=0 SD determinants and their sparse spin-adapted basis."""

    def __init__(self, system, multiplicity):
        self.system = system
        self.no, self.nv = system.noccupied_alpha, system.nunoccupied_alpha
        self.entries, self.states = [], []
        for rank in (1, 2):
            for spin, row in rows(self.no, self.nv, rank):
                state, phase = state_from_row(row, spin, self.no, self.nv)
                self.entries.append((spin, row, phase))
                self.states.append(state)
        self.index = {s: k for k, s in enumerate(self.states)}
        self.basis = spin_basis(
            self.states, self.no, self.nv, (multiplicity**2 - 1) / 4
        )

    def to_operator(self, values):
        from ccpy.models.operators import ClusterOperator

        R = ClusterOperator(self.system, order=2)
        for value, (spin, row, phase) in zip(values, self.entries):
            if not value:
                continue
            amplitude = phase * value
            if len(spin) == 1 or spin == "ab":
                getattr(R, spin)[row] = amplitude
            else:
                a, b, i, j = row
                block = getattr(R, spin)
                block[a, b, i, j] = block[b, a, j, i] = amplitude
                block[b, a, i, j] = block[a, b, j, i] = -amplitude
        return R

    def from_operator(self, R):
        return np.array(
            [phase * getattr(R, spin)[row] for spin, row, phase in self.entries]
        )

    def vector(self, values, degree=None):
        return {
            s: v
            for s, v in zip(self.states, values)
            if v and (degree is None or s.bit_count() == degree)
        }

    def array(self, vector):
        out = np.zeros(len(self.states))
        for state, value in vector.items():
            out[self.index[state]] += value
        return out


def triple_operator(vector, no, nv):
    """Induced triples to compact EE-EOMCC(P) arrays, with no dense R3 allocation."""
    entries = {s: [] for s in ("aaa", "aab", "abb", "bbb")}
    amplitudes = {s: [] for s in entries}
    for state, value in vector.items():
        if not value:
            continue
        ha = [p for p in bits(state) if p < no]
        hb = [p - no for p in bits(state) if no <= p < 2 * no]
        pa = [p - 2 * no for p in bits(state) if 2 * no <= p < 2 * no + nv]
        pb = [p - 2 * no - nv for p in bits(state) if p >= 2 * no + nv]
        if len(pa) != len(ha) or len(pb) != len(hb) or len(pa) + len(pb) != 3:
            raise ValueError("Induced triples do not conserve spin and particle number")
        spin = "a" * len(pa) + "b" * len(pb)
        row = tuple(pa + pb + ha + hb)
        _, phase = state_from_row(row, spin, no, nv)
        entries[spin].append(np.array(row) + 1)
        amplitudes[spin].append(phase * value)
    exc, R = {}, SimpleNamespace()
    for spin, data in entries.items():
        exc[spin] = (
            np.asfortranarray(data, dtype=np.int32).reshape(-1, 6)
            if data
            else np.ones((1, 6), dtype=np.int32, order="F")
        )
        setattr(R, spin, np.asfortranarray(amplitudes[spin] if data else [0.0]))
    return R, exc


class ExtendedAction:
    """P_SD U_SD^-1 Hbar U_(SDT) P_SD; no optimized neutral triples amplitudes.

    U is the normal exponential of S. Implicit quadruples are omitted. Because
    the enlarged model admits rank-preserving S_(3,3), U_SD has a nontrivial
    doubles metric. Its inverse replaces the ordinary STEOM (1-S2) factor.
    """

    def __init__(
        self, system, H, T, transformation, multiplicity, condition_limit=1.0e10
    ):
        self.configurations = SDConfigurations(system, multiplicity)
        self.H, self.T, self.S = H, T, transformation
        self.system = system
        cfg = self.configurations
        rr, cc, vv = [], [], []
        for col, state in enumerate(cfg.states):
            for target, value in transformation.same.apply({state: 1.0}, 4).items():
                rr.append(cfg.index[target])
                cc.append(col)
                vv.append(value)
        self.doubles_metric = (
            sparse.eye(len(cfg.states), format="csc")
            + sparse.coo_matrix((vv, (rr, cc)), shape=(len(cfg.states),) * 2)
        ).tocsc()
        try:
            self.factor = splu(self.doubles_metric)
        except RuntimeError as error:
            raise ValueError("Singular neutral SD wave-operator metric") from error
        rr, cc, vv = [], [], []
        for col, state in enumerate(cfg.states):
            for target, value in transformation.one2.apply({state: 1.0}, 4).items():
                rr.append(cfg.index[target])
                cc.append(col)
                vv.append(value)
        self.singles_to_doubles = sparse.coo_matrix(
            (vv, (rr, cc)), shape=(len(cfg.states),) * 2
        ).tocsr()
        self.metric = self.doubles_metric + self.singles_to_doubles
        inverse = LinearOperator(
            self.metric.shape,
            matvec=lambda v: self.factor.solve(v - self.singles_to_doubles @ v),
            rmatvec=lambda v: (lambda w: w - self.singles_to_doubles.T @ w)(
                self.factor.solve(v, trans="T")
            ),
            dtype=float,
        )
        self.metric_condition = onenormest(self.metric) * onenormest(inverse)
        if (
            not np.isfinite(self.metric_condition)
            or self.metric_condition > condition_limit
        ):
            raise ValueError(
                f"Ill-conditioned neutral SD wave-operator metric ({self.metric_condition:.3e})"
            )
        self.maximum_induced_triples = 0
        self.operator = LinearOperator(
            (cfg.basis.shape[1],) * 2, matvec=self.matvec, dtype=float
        )

    def lift(self, values):
        cfg, S = self.configurations, self.S
        c1, c2 = cfg.vector(values, 2), cfg.vector(values, 4)
        r12 = self.metric @ values
        r3 = defaultdict(float)
        contributions = [
            S.one2.apply(c2, 6),
            S.one3.apply(c1, 6),
            S.higher.apply(c2, 6),
            {
                k: 0.5 * v
                for k, v in normal_product_apply(S.one2, S.one2, c1, 6).items()
            },
            normal_product_apply(S.same, S.one2, c2, 6),
        ]
        for term in contributions:
            for key, value in term.items():
                r3[key] += value
        return r12, dict(r3)

    def raw_matvec(self, values):
        from ccpy.eomcc import eomccsdt_p
        from ccpy.eomcc.eomccsdt_intermediates import get_eomccsd_intermediates
        from ccpy.models.operators import ClusterOperator

        cfg = self.configurations
        r12, r3 = self.lift(values)
        self.maximum_induced_triples = max(self.maximum_induced_triples, len(r3))
        R = cfg.to_operator(r12)
        triples, excitations = triple_operator(r3, cfg.no, cfg.nv)
        for spin in ("aaa", "aab", "abb", "bbb"):
            setattr(R, spin, getattr(triples, spin))
        T = SimpleNamespace(
            **{s: getattr(self.T, s) for s in ("a", "b", "aa", "ab", "bb")},
            **{s: np.zeros(1) for s in ("aaa", "aab", "abb", "bbb")},
        )
        empty = {
            s: np.ones((1, 6), dtype=np.int32, order="F")
            for s in ("aaa", "aab", "abb", "bbb")
        }
        sigma = ClusterOperator(self.system, order=2)
        eomccsdt_p.build_HR_1A(sigma, R, excitations, self.H)
        eomccsdt_p.build_HR_1B(sigma, R, excitations, self.H)
        X = get_eomccsd_intermediates(self.H, R, self.system)
        eomccsdt_p.build_HR_2A(sigma, R, excitations, T, empty, self.H, X)
        eomccsdt_p.build_HR_2B(sigma, R, excitations, T, empty, self.H, X)
        eomccsdt_p.build_HR_2C(sigma, R, excitations, T, empty, self.H, X)
        result = cfg.from_operator(sigma)
        # U_SD = [[1,0],[S21,1+S22]]. First eliminate S21 sigma1.
        result -= self.singles_to_doubles @ result
        return self.factor.solve(result)

    def matvec(self, values):
        B = self.configurations.basis
        return np.asarray(B.T @ self.raw_matvec(B @ values)).ravel()
