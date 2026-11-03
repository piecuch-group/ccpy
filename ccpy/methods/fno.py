"""Virtual-only FNO preparation and low-level provider extension points.

Providers implement ``provider(reference, targets=(), need_density=True,
full_result=None)`` and
return FNOEvaluation. Energies are absolute totals (not positive affinities).
Densities are spin-resolved virtual blocks in the reference's working basis.
Providers own root tracking: state keys must identify the same physical states
in full and reduced calculations, not merely energy-sorted root numbers.
"""
from dataclasses import dataclass, field
from numbers import Integral

import numpy as np

from ccpy.mbpt.dfmp2 import dfmp2


@dataclass(frozen=True)
class FNOTarget:
    """Stable provider state identity; root labels are zero based within an irrep."""

    sector: str
    root: int = 0
    irrep: str | None = None
    multiplicity: int | None = None
    formulation: str = "folded"

    def __post_init__(self):
        if self.formulation not in ("full", "folded"):
            raise ValueError("ADC target formulation must be full or folded.")
        if self.sector not in ("EE", "IP", "EA", "DIP", "DEA", "SF"):
            raise ValueError("Unknown FNO target sector.")
        if not isinstance(self.root, Integral) or self.root < 0:
            raise ValueError("FNO target root must be a nonnegative integer.")
        if self.multiplicity is not None and (not isinstance(self.multiplicity, Integral) or self.multiplicity < 1):
            raise ValueError("FNO target multiplicity must be a positive integer.")


@dataclass(frozen=True)
class FNOOptions:
    """Immutable FNO policy. Specify at most one virtual selection criterion.

    ``occupation_cutoff`` uses spatial (0--2) occupations for restricted
    references and spin (0--1) occupations for UHF. Zero retains every orbital.
    ``nvirtuals`` counts external spatial virtuals for restricted references;
    for UHF it is a minimum per-spin virtual count (equal removals are enforced).
    Degeneracy preservation may retain more orbitals than requested.
    ``protected_virtuals`` lists zero-based input external-virtual indices that
    must survive selection (per spin for UHF; common spatial virtuals for ROHF).
    """

    source: str = "mp2"
    occupation_cutoff: float | None = None
    retained_population: float | None = None
    nvirtuals: int | None = None
    targets: tuple = ()
    weights: tuple = ()
    space_policy: str = "average"
    energy_completion: str = "ground"
    degeneracy_tolerance: float = 1.0e-8
    density_tolerance: float = 1.0e-10
    protected_virtuals: tuple = ()

    def __post_init__(self):
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("FNO source must be a nonempty provider name.")
        object.__setattr__(self, "targets", tuple(self.targets))
        object.__setattr__(self, "weights", tuple(self.weights))
        object.__setattr__(self, "protected_virtuals", tuple(self.protected_virtuals))
        if (any(not isinstance(i, Integral) or i < 0 for i in self.protected_virtuals) or
                len(set(self.protected_virtuals)) != len(self.protected_virtuals)):
            raise ValueError("protected_virtuals must contain distinct nonnegative virtual indices.")
        criteria = (self.occupation_cutoff, self.retained_population, self.nvirtuals)
        if sum(x is not None for x in criteria) > 1:
            raise ValueError("Specify only one FNO selection criterion.")
        if all(x is None for x in criteria):
            object.__setattr__(self, "occupation_cutoff", 1.0e-6)
        for value in (self.occupation_cutoff, self.degeneracy_tolerance, self.density_tolerance):
            if value is not None and (not np.isfinite(value) or value < 0):
                raise ValueError("FNO cutoffs and tolerances must be finite and nonnegative.")
        if self.retained_population is not None and not 0 < self.retained_population <= 1:
            raise ValueError("retained_population must be in (0, 1].")
        if self.nvirtuals is not None and (not isinstance(self.nvirtuals, Integral) or self.nvirtuals < 1):
            raise ValueError("nvirtuals must be a positive integer.")
        if self.space_policy not in ("average", "union", "root_specific"):
            raise ValueError("space_policy must be average, union, or root_specific.")
        if self.space_policy == "root_specific" and len(self.targets) != 1:
            raise ValueError("Construct one calculation per root-specific FNO target.")
        if not all(isinstance(t, FNOTarget) for t in self.targets) or len(set(self.targets)) != len(self.targets):
            raise ValueError("targets must contain distinct FNOTarget objects.")
        if self.weights and (len(self.weights) != len(self.targets) or
                             not np.all(np.isfinite(self.weights)) or min(self.weights) <= 0):
            raise ValueError("weights must contain one finite positive weight per target.")
        if self.energy_completion not in ("none", "ground", "state_specific"):
            raise ValueError("Unknown FNO energy_completion policy.")
        if self.source == "mp2" and (self.targets or self.energy_completion == "state_specific"):
            raise ValueError("MP2 supplies a ground-state density/correction, not target-state corrections.")
        if self.energy_completion == "state_specific" and not self.targets:
            raise ValueError("State-specific completion requires targets.")


@dataclass
class FNOEvaluation:
    """Provider output in the input basis.

    ``virtual_densities[target][spin]`` is an unrelaxed/relaxed state 1-RDM
    block under the provider's documented convention; use target None for the
    ground state. A transition or signed difference density is not acceptable.
    For ``need_density=False`` the density mapping may be empty. State energies
    must be root matched by the provider, with missing/lost roots raising errors.
    ``protected_subspaces[spin]`` optionally gives virtual-basis column vectors
    to retain in addition to the selected natural orbitals. Include all beta
    virtual rows for ROHF: its singly occupied directions are already protected.
    """

    ground_energy: float
    virtual_densities: dict = field(default_factory=dict)
    state_energies: dict = field(default_factory=dict)
    converged: bool = True
    metadata: dict = field(default_factory=dict)
    protected_subspaces: dict = field(default_factory=dict)


@dataclass
class FNOResult:
    options: FNOOptions
    full: FNOEvaluation
    truncated: FNOEvaluation
    transforms: dict
    occupations: dict
    retained_population: dict
    discarded: dict

    @property
    def delta_ground(self):
        return self.full.ground_energy - self.truncated.ground_energy

    def delta_state(self, target):
        return self.full.state_energies[target] - self.truncated.state_energies[target]


def _mp2_provider(reference, targets=(), need_density=True, full_result=None):
    correlation, density = dfmp2(reference, need_density=need_density)
    return FNOEvaluation(reference.energy + correlation,
                         {None: density} if need_density else {},
                         metadata={"density": "unrelaxed spin-Fock MP2 including singles"})


_PROVIDERS = {"mp2": _mp2_provider}


def register_fno_provider(source, provider):
    """Register e.g. ``register_fno_provider('adc2', evaluate_adc2)``.

    Registration does not certify sector or density support. The provider must
    raise NotImplementedError for unsupported sectors/reference/density types.
    Register before constructing methods; prepared references retain their data.
    """
    if source == "mp2":
        raise ValueError("The built-in DF-MP2 provider cannot be replaced.")
    if not isinstance(source, str) or not source or not callable(provider):
        raise TypeError("A provider needs a nonempty source name and a callable.")
    _PROVIDERS[source] = provider


def _evaluate(provider, reference, options, need_density, full_result=None):
    result = provider(reference, targets=options.targets, need_density=need_density, full_result=full_result)
    if not isinstance(result, FNOEvaluation):
        raise TypeError("FNO providers must return FNOEvaluation.")
    if not result.converged:
        raise RuntimeError("The FNO low-level calculation did not converge.")
    if (np.ndim(result.ground_energy) != 0 or np.iscomplexobj(result.ground_energy) or
            not np.isfinite(result.ground_energy)):
        raise ValueError("The FNO provider must return a finite real scalar ground energy.")
    if options.energy_completion == "state_specific":
        for target in options.targets:
            value = result.state_energies.get(target)
            if (value is None or np.ndim(value) != 0 or np.iscomplexobj(value) or not np.isfinite(value)):
                raise ValueError(f"FNO provider missing a finite matched energy for {target}.")
    return result


def _validated_density(density, size, tolerance):
    density = np.asarray(density)
    if density.shape != (size, size) or not np.all(np.isfinite(density)):
        raise ValueError("FNO virtual density has an invalid shape or nonfinite entries.")
    if np.iscomplexobj(density) and np.max(np.abs(density.imag), initial=0) > tolerance:
        raise NotImplementedError("Complex FNO densities are not supported by the real CC kernels.")
    density = density.real
    if not np.allclose(density, density.T, atol=tolerance, rtol=0):
        raise ValueError("FNO occupation density must be Hermitian.")
    density = 0.5 * (density + density.T)
    if np.min(np.linalg.eigvalsh(density), initial=0) < -tolerance:
        raise ValueError("FNO occupation density has significant negative eigenvalues.")
    return density


def _spectrum(density, labels, options):
    labels = np.asarray(labels)
    # Reject symmetry-breaking state densities rather than silently dropping blocks.
    if np.max(np.abs(density[labels[:, None] != labels[None, :]]), initial=0) > options.density_tolerance:
        raise ValueError("FNO density mixes irreps; use symmetry-averaged states or a C1 reference.")
    blocks = {}
    for label in dict.fromkeys(labels.tolist()):
        idx = np.flatnonzero(labels == label)
        n, U = np.linalg.eigh(density[np.ix_(idx, idx)])
        order = np.argsort(-n, kind="stable")
        blocks[label] = (idx, np.maximum(n[order], 0), U[:, order])
    return blocks


def _counts(blocks, options, allow_empty=False):
    values = sorted(((n, label, i) for label, (_, ns, _) in blocks.items()
                     for i, n in enumerate(ns)), key=lambda x: -x[0])
    counts = {label: 0 for label in blocks}
    if options.nvirtuals is not None:
        if options.nvirtuals > len(values):
            raise ValueError("FNO nvirtuals exceeds the available virtual space.")
        selected = values[:options.nvirtuals]
    elif options.retained_population is not None:
        total = sum(x[0] for x in values)
        # No information in a zero density: conservatively keep all orbitals.
        number = (len(values) if total == 0 else
                  min(len(values), np.searchsorted(np.cumsum([x[0] for x in values]),
                                                  options.retained_population * total) + 1))
        selected = values[:number]
    else:
        selected = [x for x in values if x[0] >= options.occupation_cutoff]
    for _, label, i in selected:
        counts[label] = max(counts[label], i + 1)
    if not sum(counts.values()):
        if allow_empty:
            return counts
        raise ValueError("FNO threshold removes every external virtual orbital; lower the cutoff.")
    # Preserve occupation-degenerate groups even when their components occur in
    # different Abelian irreps (e.g. components of a higher-symmetry multiplet).
    boundary = min(x[0] for x in selected)
    for n, label, i in values:
        if boundary - n <= options.degeneracy_tolerance:
            counts[label] = max(counts[label], i + 1)
    return counts


def _extend_degeneracy(ns, count, tolerance):
    while 0 < count < len(ns) and ns[count - 1] - ns[count] <= tolerance:
        count += 1
    return count


def _span(matrices):
    """Orthonormal union, independent of the scaling of supplied columns."""
    matrix = np.column_stack(matrices)
    norms = np.linalg.norm(matrix, axis=0)
    matrix = matrix[:, norms > 0]
    if matrix.shape[1] == 0:
        return matrix
    matrix = matrix / norms[norms > 0]
    U, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    return U[:, singular > 1.0e-10]


def _select(densities, labels, focks, options, uhf, protected_subspaces=None):
    """Return per-spin external virtual rotations and density diagnostics."""
    protected = {}
    for s in labels:
        size = len(labels[s])
        if any(i >= size for i in options.protected_virtuals):
            raise ValueError("A protected virtual index exceeds the available external virtual space.")
        basis = np.asarray((protected_subspaces or {}).get(s, np.empty((size, 0))))
        if (basis.ndim != 2 or basis.shape[0] != size or np.iscomplexobj(basis) or
                not np.all(np.isfinite(basis))):
            raise ValueError("Protected virtual subspaces must be finite real matrices in the input virtual basis.")
        protected[s] = _span([basis, np.eye(size)[:, options.protected_virtuals]])
    allow_empty = any(basis.shape[1] for basis in protected.values())
    spectra = {s: [_spectrum(d[s], labels[s], options) for d in densities] for s in labels}
    # Average before diagonalization. A union selects independently then spans both.
    if options.space_policy != "union":
        weights = np.asarray(options.weights or (1.0,) * len(densities))
        spectra = {s: [_spectrum(sum(w * d[s] for w, d in zip(weights / weights.sum(), densities)),
                                 labels[s], options)] for s in labels}
    selected = {s: [_counts(blocks, options, allow_empty=allow_empty) for blocks in spectra[s]] for s in labels}
    for s in labels:
        for blocks, counts in zip(spectra[s], selected[s]):
            for label, (_, ns, _) in blocks.items():
                counts[label] = _extend_degeneracy(ns, counts[label], options.degeneracy_tolerance)
    if uhf and (len(densities) == 1 or options.space_policy != "union"):
        # Equal removals per irrep preserve the common total orbital dimension.
        keys = set(spectra["a"][0]) | set(spectra["b"][0])
        while True:
            before = {s: selected[s][0].copy() for s in labels}
            for label in keys:
                sizes = {s: len(spectra[s][0].get(label, ((), (), ()))[1]) for s in labels}
                drop = min(sizes[s] - selected[s][0].get(label, 0) for s in labels)
                for s in labels:
                    if label in selected[s][0]:
                        selected[s][0][label] = sizes[s] - drop
            # Spin matching can introduce a new occupation boundary. Close its
            # degeneracies across irreps, then repeat spin matching to a fixed point.
            for s in labels:
                blocks, counts = spectra[s][0], selected[s][0]
                boundary = min((ns[counts[label] - 1] for label, (_, ns, _) in blocks.items()
                                if counts[label] > 0), default=np.inf)
                for label, (_, ns, _) in blocks.items():
                    counts[label] = max(counts[label], int(np.count_nonzero(
                        ns >= boundary - options.degeneracy_tolerance)))
            if all(before[s] == selected[s][0] for s in labels):
                break
    transforms, occupations, population = {}, {}, {}
    for s in labels:
        columns = []
        out_labels = []
        energies = []
        for label in dict.fromkeys(labels[s]):
            idx = spectra[s][0][label][0]
            spaces = [blocks[label][2][:, :counts[label]]
                      for blocks, counts in zip(spectra[s], selected[s])]
            if protected[s].shape[1]:
                projected = protected[s][idx]
                spaces.append(projected[:, np.linalg.norm(projected, axis=0) > 1.0e-10])
            U = spaces[0]
            if len(spaces) > 1:
                U = _span(spaces)
            F = focks[s][np.ix_(idx, idx)]
            eps, Z = np.linalg.eigh(U.T @ F @ U)
            block = np.zeros((len(labels[s]), U.shape[1]))
            block[idx] = U @ Z
            columns.append(block)
            out_labels.extend([label] * U.shape[1])
            energies.extend(eps)
        W = np.column_stack(columns)
        order = np.argsort(energies, kind="stable")
        W = W[:, order]
        transforms[s] = (W, [out_labels[i] for i in order])
        occupations[s] = [np.concatenate([b[1] for b in blocks.values()]) for blocks in spectra[s]]
        # Report each input state's actual population in the final shared subspace.
        population[s] = tuple(float(np.trace(W.T @ d[s] @ W) / np.trace(d[s]))
                              if np.trace(d[s]) > 0 else 1.0 for d in densities)
    if uhf:
        # Independent unions/protected spaces can have different spin ranks. Keep
        # the entire irrep in both whenever removals disagree. This conservative
        # closure preserves every selected direction and all degeneracies.
        keys = set(labels["a"]) | set(labels["b"])
        close = set()
        for label in keys:
            removed = [list(labels[s]).count(label) - transforms[s][1].count(label) for s in labels]
            if removed[0] != removed[1]:
                close.add(label)
        for s in labels if close else ():
            W, out_labels = transforms[s]
            keep = [i for i, label in enumerate(out_labels) if label not in close]
            pieces, new_labels = [W[:, keep]], [out_labels[i] for i in keep]
            for label in close:
                idx = np.flatnonzero(np.asarray(labels[s]) == label)
                pieces.append(np.eye(len(labels[s]))[:, idx])
                new_labels.extend([label] * len(idx))
            W = np.column_stack(pieces)
            # Recanonicalize each retained irrep and sort orbital energies.
            new_columns, eps_all, final_labels = [], [], []
            for label in dict.fromkeys(new_labels):
                U = W[:, np.asarray(new_labels) == label]
                eps, Z = np.linalg.eigh(U.T @ focks[s] @ U)
                new_columns.append(U @ Z)
                eps_all.extend(eps)
                final_labels.extend([label] * U.shape[1])
            order = np.argsort(eps_all, kind="stable")
            W = np.column_stack(new_columns)[:, order]
            transforms[s] = (W, [final_labels[i] for i in order])
            population[s] = tuple(float(np.trace(W.T @ d[s] @ W) / np.trace(d[s]))
                                  if np.trace(d[s]) > 0 else 1.0 for d in densities)
    return transforms, occupations, population


def prepare_fno(reference, options):
    """Build a fresh, ordinary reduced Reference. Never mutate the parent or SCF."""
    if not isinstance(options, FNOOptions):
        raise TypeError("fno must be FNOOptions, None, or False.")
    if reference.ndelete:
        raise ValueError("FNO and ndelete cannot be combined: FNO requires the full virtual space.")
    if reference.fno_result is not None:
        if options != reference.fno_result.options:
            raise ValueError("FNO settings conflict with the upstream reference; start from SCF.")
        return reference
    if options.source not in _PROVIDERS:
        if options.source == "adc2":
            from ccpy.methods.adc import evaluate_adc2

            _PROVIDERS["adc2"] = evaluate_adc2
    if options.source not in _PROVIDERS:
        raise NotImplementedError(f"No FNO provider registered for {options.source!r}; "
                                  "use register_fno_provider(source, provider).")
    provider = _PROVIDERS[options.source]
    full = _evaluate(provider, reference, options, True)
    keys = options.targets or (None,)
    densities = []
    protected = {}
    if set(full.protected_subspaces) - {"a", "b"}:
        raise ValueError("Protected subspace keys must be 'a' and/or 'b'.")
    for s, spin in (("a", "alpha"), ("b", "beta")):
        size = getattr(reference.system, f"nunoccupied_{spin}")
        basis = np.asarray(full.protected_subspaces.get(s, np.empty((size, 0))))
        if (basis.ndim != 2 or basis.shape[0] != size or np.iscomplexobj(basis) or
                not np.all(np.isfinite(basis))):
            raise ValueError("Provider protected_subspaces must contain real spin-virtual matrices.")
        protected[s] = basis
    for key in keys:
        if key not in full.virtual_densities:
            raise ValueError(f"FNO provider missing a virtual density for {key}.")
        if any(s not in full.virtual_densities[key] for s in ("a", "b")):
            raise ValueError(f"FNO provider must supply both spin densities for {key}.")
        densities.append({s: _validated_density(full.virtual_densities[key][s],
                          getattr(reference.system, f"nunoccupied_{spin}"), options.density_tolerance)
                          for s, spin in (("a", "alpha"), ("b", "beta"))})
    nocc = {"a": reference.nfrozen + reference.system.noccupied_alpha,
            "b": reference.nfrozen + reference.system.noccupied_beta}
    from ccpy.methods.hamiltonian import build_fock

    Fa, Fb = build_fock(reference.h_mo["a"], reference.h_mo["b"],
                        reference.B_mo["a"], reference.B_mo["b"], nocc["a"], nocc["b"])
    focks = {"a": Fa[nocc["a"]:, nocc["a"]:], "b": Fb[nocc["b"]:, nocc["b"]:]}
    all_labels = reference.system.orbital_symmetries_all
    if reference.is_uhf:
        labels = {s: all_labels[s][nocc[s]:] for s in nocc}
    else:
        # Preserve singly occupied spatial orbitals (beta virtual, alpha occupied).
        offset = nocc["a"] - nocc["b"]
        densities = [{"a": d["a"] + d["b"][offset:, offset:]} for d in densities]
        protected = {"a": np.column_stack((protected["a"], protected["b"][offset:]))}
        labels = {"a": all_labels[nocc["a"]:]}
        from ccpy.scf.rohf import ROHF

        ca, cb = (ROHF.CANONICALIZATIONS[reference.scf.canonicalization][2]
                  if isinstance(reference.scf, ROHF) else (0.5, 0.5))
        focks = {"a": ca * focks["a"] + cb * focks["b"][offset:, offset:]}
    rotations, occupations, population = _select(densities, labels, focks, options, reference.is_uhf,
                                                protected_subspaces=protected)
    if not reference.is_uhf:
        rotations["b"] = rotations["a"]
    # Restricted transforms act only on the common external virtuals.
    starts = nocc if reference.is_uhf else {"a": nocc["a"], "b": nocc["a"]}
    Q, child_labels = {}, {}
    for s in nocc:
        W, ls = rotations[s]
        start = starts[s]
        Q[s] = np.zeros((reference.mo_coeff[s].shape[1], start + W.shape[1]))
        Q[s][:start, :start] = np.eye(start)
        Q[s][start:, start:] = W
        parent_labels = all_labels[s] if reference.is_uhf else all_labels
        child_labels[s] = list(parent_labels[:start]) + ls
    child = reference.from_orbital_transform(Q, child_labels)
    child.fno_parent = reference
    truncated = _evaluate(provider, child, options, False, full_result=full)
    child.fno_result = FNOResult(options, full, truncated, Q, occupations, population,
                                {s: Q[s].shape[0] - Q[s].shape[1] for s in Q})
    return child
