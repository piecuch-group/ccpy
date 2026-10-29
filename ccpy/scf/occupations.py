from numbers import Integral

import numpy as np

from ccpy.integrals.symmetry.sym_utils import COTTON_LABELS, get_irrep_index, irrep_product


def _is_index(value):
    return isinstance(value, Integral) and not isinstance(value, bool)


def _spin_nelec(key, value):
    """(alpha, beta) electrons of one irrep; an integer is split equally between the spins."""
    if isinstance(value, (tuple, list)):
        nelec = tuple(value)
    elif _is_index(value) and value % 2 == 0:
        nelec = (value // 2, value // 2)
    elif _is_index(value):
        raise ValueError(
            f"irrep_occupations give {key!r} an odd number of electrons ({value}). "
            "Give its (alpha, beta) electrons as a pair instead."
        )
    else:
        nelec = (value,)
    if len(nelec) != 2 or not all(_is_index(n) and n >= 0 for n in nelec):
        raise ValueError(
            "Irrep occupations must be non-negative integers or (alpha, beta) pairs "
            "of them."
        )
    return int(nelec[0]), int(nelec[1])


def validate_occupation_options(target_symmetry, irrep_occupations):
    """
    Check the types of the target_symmetry and irrep_occupations options.

    Parameters
    ----------
    target_symmetry : str | int | None
        The irrep of the determinant, by label or index.
    irrep_occupations : dict | None
        The electrons in each irrep, keyed by irrep label or index, with integer or
        (alpha, beta) pair values.

    Raises
    ------
    ValueError
        If either option has the wrong type.
    """
    if target_symmetry is not None and not (
        isinstance(target_symmetry, str) or _is_index(target_symmetry)
    ):
        raise ValueError("target_symmetry must be an irrep label or index.")
    if irrep_occupations is None:
        return
    if not isinstance(irrep_occupations, dict):
        raise ValueError("irrep_occupations must be a dictionary.")
    for key, value in irrep_occupations.items():
        if not (isinstance(key, str) or _is_index(key)):
            raise ValueError("irrep_occupations keys must be irrep labels or indices.")
        _spin_nelec(key, value)


class OccupationConstraints:
    """
    Irrep occupations of a one-component SCF method, either fixed or chosen in each
    iteration as the lowest-energy determinant of a target irrep.

    Parameters
    ----------
    restricted : bool
        Whether both spins occupy one set of orbitals (RHF, ROHF), so that the
        minority-spin occupied orbitals lie within the majority-spin ones. Otherwise
        each spin has its own orbitals (UHF).
    point_group : str
        The point group the irreps belong to.
    nelec : tuple[int, int]
        The numbers of alpha and beta electrons.
    target_symmetry : str | int | None
        The irrep of the determinant.
    irrep_occupations : dict | None
        Electrons per irrep, as described for SCF methods.

    Raises
    ------
    ValueError
        If the options are inconsistent with each other, with the numbers of electrons, or
        with restricted orbitals.
    """

    def __init__(
        self, restricted, point_group, nelec, target_symmetry, irrep_occupations
    ):
        self.restricted = restricted
        self.point_group = point_group
        self.nelec = tuple(nelec)
        self.irrep_labels = COTTON_LABELS[point_group]
        self.nirrep = len(self.irrep_labels)
        # the spin with more electrons
        self.majority_spin = int(self.nelec[1] > self.nelec[0])
        self.target = None
        if target_symmetry is not None:
            self.target = get_irrep_index(point_group, target_symmetry)
        self.irrep_nelec = None
        if irrep_occupations is not None:
            self.irrep_nelec = self._resolve_irrep_nelec(irrep_occupations)

        if restricted:
            if self.nelec[0] == self.nelec[1] and self.target not in (None, 0):
                raise ValueError(
                    "RHF solutions are always totally symmetric, "
                    "so target_symmetry must be the totally symmetric irrep."
                )
            if self.irrep_nelec is not None:
                self._check_restricted_nelec()
        if self.irrep_nelec is not None and self.target is not None:
            if self._determinant_irrep(self.irrep_nelec) != self.target:
                raise ValueError(
                    "irrep_occupations do not give a determinant of target_symmetry."
                )

    def _check_restricted_nelec(self):
        """In restricted orbitals, each minority-spin electron pairs with a majority-spin one."""
        alpha, beta = self.irrep_nelec
        if self.nelec[0] == self.nelec[1]:
            bad = np.flatnonzero(alpha != beta)
            rule = (
                "Restricted closed-shell occupations need equal alpha and beta electrons "
                "in each irrep"
            )
        else:
            major, minor = (
                ("alpha", "beta") if self.majority_spin == 0 else ("beta", "alpha")
            )
            bad = np.flatnonzero(
                self.irrep_nelec[1 - self.majority_spin]
                > self.irrep_nelec[self.majority_spin]
            )
            rule = (
                f"Restricted orbitals with more {major} than {minor} electrons need at "
                f"least as many {major} as {minor} electrons in each irrep"
            )
        if len(bad) > 0:
            names = {index: label for label, index in self.irrep_labels.items()}
            details = ", ".join(
                f"{names[h]} ({alpha[h]} alpha, {beta[h]} beta)" for h in bad
            )
            raise ValueError(f"{rule}, but irrep_occupations give {details}.")

    def _resolve_irrep_nelec(self, irrep_occupations):
        irrep_nelec = np.zeros((2, self.nirrep), dtype=int)
        seen = set()
        for key, value in irrep_occupations.items():
            irrep = get_irrep_index(self.point_group, key)
            if irrep in seen:
                raise ValueError(f"Irrep {key!r} appears more than once.")
            seen.add(irrep)
            irrep_nelec[:, irrep] = _spin_nelec(key, value)
        totals = tuple(int(n) for n in irrep_nelec.sum(axis=1))
        if totals != self.nelec:
            raise ValueError(
                f"irrep_occupations assign {totals} (alpha, beta) electrons, but there "
                f"are {self.nelec}. Irreps that are not listed are empty."
            )
        return irrep_nelec

    def _determinant_irrep(self, irrep_nelec):
        # Only irreps with an odd number of electrons contribute.
        odd = np.flatnonzero(irrep_nelec.sum(axis=0) % 2)
        return irrep_product(odd)

    def check_capacity(self, irreps):
        """
        Check that orbitals with these irreps can hold the requested occupations.

        Parameters
        ----------
        irreps : NDArray
            The irrep index of each orbital that the electrons can occupy.

        Raises
        ------
        ValueError
            If an irrep has fewer orbitals than irrep_occupations assign to it, or if
            no occupation realizes target_symmetry.
        """
        names = {index: label for label, index in self.irrep_labels.items()}
        if self.irrep_nelec is not None:
            capacity = np.bincount(irreps, minlength=self.nirrep)
            alpha, beta = self.irrep_nelec
            over = np.flatnonzero(np.any(self.irrep_nelec > capacity, axis=0))
            if len(over):
                details = ", ".join(
                    f"{names[h]} ({alpha[h]} alpha and {beta[h]} beta electrons, "
                    f"{capacity[h]} orbitals)"
                    for h in over
                )
                raise ValueError(
                    f"irrep_occupations need more orbitals than these irreps have: "
                    f"{details}."
                )
        else:
            nsets = 1 if self.restricted else 2
            try:
                self._lowest_irrep_nelec(
                    [np.zeros(len(irreps))] * nsets, [irreps] * nsets
                )
            except ValueError as e:
                raise ValueError(
                    f"No occupation of {self.nelec[0]} alpha and {self.nelec[1]} beta "
                    f"electrons gives a determinant of {names[self.target]} symmetry."
                ) from e

    def order(self, eps, irreps):
        """
        Return, for each orbital set, the order that puts its occupied orbitals first.

        Parameters
        ----------
        eps : list[NDArray]
            The orbital energies of each set.
        irreps : list[NDArray]
            The irrep of each orbital of each set.

        Returns
        -------
        list[NDArray]
            For each set, its orbitals in order: occupied orbitals (docc, then singly
            occupied for restricted orbitals), then virtual ones, each by energy.
        """
        # irrep_nelec is [[alpha nelec per irrep], [beta nelec per irrep]]
        irrep_nelec = self.irrep_nelec

        # if only target_symmetry is set, we find the occupation pattern
        # that gives the lowest energy while consistent with target_symmetry

        # RHF/ROHF case
        if irrep_nelec is None:
            irrep_nelec = self._lowest_irrep_nelec(eps, irreps)
        if self.restricted:
            docc, socc = [], []
            for irrep in range(self.nirrep):
                orbitals = _irrep_orbitals(eps[0], irreps[0], irrep)
                ndocc = irrep_nelec[1 - self.majority_spin, irrep]
                nsocc = irrep_nelec[self.majority_spin, irrep]
                docc.extend(orbitals[:ndocc])
                socc.extend(orbitals[ndocc:nsocc])
            return [_occupied_first(eps[0], docc, socc)]

        # UHF case
        orders = []
        for ispin in range(2):
            occ = []
            for irrep in range(self.nirrep):
                nocc = irrep_nelec[ispin, irrep]
                orbitals = _irrep_orbitals(eps[ispin], irreps[ispin], irrep)[:nocc]
                occ.extend(orbitals)
            orders.append(_occupied_first(eps[ispin], occ))
        return orders

    def _lowest_irrep_nelec(self, eps, irreps):
        """Electrons per irrep of the lowest-energy determinant of the target irrep."""
        if self.restricted:
            return _restricted_irrep_nelec(
                eps[0], irreps[0], self.nelec, self.nirrep, self.target
            )
        return _unrestricted_irrep_nelec(
            eps, irreps, self.nelec, self.nirrep, self.target
        )


def _irrep_orbitals(eps, irreps, irrep):
    """The orbitals of one irrep, by energy."""
    indices = np.flatnonzero(irreps == irrep)
    return indices[np.argsort(eps[indices], kind="stable")]


def _occupied_first(eps, *occupied):
    """Each group of occupied orbitals by energy, then the remaining orbitals by energy."""
    used = np.zeros(len(eps), dtype=bool)
    groups = []
    for indices in occupied:
        indices = np.array(indices, dtype=int)
        used[indices] = True
        groups.append(indices[np.argsort(eps[indices], kind="stable")])
    rest = np.flatnonzero(~used)
    groups.append(rest[np.argsort(eps[rest], kind="stable")])
    return np.concatenate(groups)


def _restricted_irrep_nelec(eps, irreps, nelec, nirrep, target):
    """
    Given a target symmetry, find the aufbau occupation per irrep
    that gives the lowest sum of orbital energies, for RHF and ROHF.
    """
    major = int(nelec[1] > nelec[0])
    ndocc, nsocc = min(nelec), abs(nelec[0] - nelec[1])
    # dynamic programming algorithm: states is a map as follows
    # (ndocc_curr, nsocc_curr, sym_curr) -> (lowest energy, per-irrep (ndocc, nsocc))
    # initialize from all zeros
    states = {(0, 0, 0): (0.0, [])}
    for irrep in range(nirrep):
        # cumulative energies of the lowest orbitals of this irrep
        prefix = np.concatenate(
            ([0.0], np.cumsum(eps[_irrep_orbitals(eps, irreps, irrep)]))
        )
        capacity = len(prefix) - 1
        next_states = {}
        for (ndocc_curr, nsocc_curr, sym_curr), (energy, choices) in states.items():
            for idocc in range(min(capacity, ndocc - ndocc_curr) + 1):
                for isocc in range(min(capacity - idocc, nsocc - nsocc_curr) + 1):
                    # for RHF this inner loop is only run once per outer iteration
                    key = (
                        ndocc_curr + idocc,
                        nsocc_curr + isocc,
                        sym_curr ^ (irrep if isocc % 2 else 0),
                    )
                    # the majority spin fills idocc + isocc orbitals, the minority idocc
                    value = energy + prefix[idocc + isocc] + prefix[idocc]
                    # overwrite if a lower energy state found
                    if key not in next_states or value < next_states[key][0]:
                        next_states[key] = (value, choices + [(idocc, isocc)])
        states = next_states
    result = states.get((ndocc, nsocc, target))
    if result is None:
        raise ValueError(
            f"No occupation gives a determinant of target symmetry {target}."
        )
    irrep_nelec = np.zeros((2, nirrep), dtype=int)
    for irrep, (idocc, isocc) in enumerate(result[1]):
        irrep_nelec[major, irrep] = idocc + isocc
        irrep_nelec[1 - major, irrep] = idocc
    return irrep_nelec


def _lowest_energy_occupation_per_symmetry(eps, irreps, nocc, nirrep):
    """
    For one spin and each determinant irrep, the lowest sum of nocc orbital energies
    and the number of electrons in each irrep that gives it.
    """
    # (nocc_curr, sym_curr) -> (lowest energy, electrons per irrep)
    states = {(0, 0): (0.0, [])}
    for irrep in range(nirrep):
        # cumulative energies of the lowest orbitals of this irrep
        prefix = np.concatenate(
            ([0.0], np.cumsum(eps[_irrep_orbitals(eps, irreps, irrep)]))
        )
        capacity = len(prefix) - 1
        next_states = {}
        for (nocc_curr, sym_curr), (energy, choices) in states.items():
            for iocc in range(min(capacity, nocc - nocc_curr) + 1):
                key = (nocc_curr + iocc, sym_curr ^ (irrep if iocc % 2 else 0))
                value = energy + prefix[iocc]
                # overwrite if a lower energy state found
                if key not in next_states or value < next_states[key][0]:
                    next_states[key] = (value, choices + [iocc])
        states = next_states
    # unreachable irreps keep an infinite energy (sentinel value for argmin in _unrestricted_irrep_nelec)
    energies = np.full(nirrep, np.inf)
    irrep_nelec = [None] * nirrep
    for (n, sym), (energy, choices) in states.items():
        if n == nocc:
            energies[sym], irrep_nelec[sym] = energy, choices
    return energies, irrep_nelec


def _unrestricted_irrep_nelec(eps, irreps, nelec, nirrep, target):
    """Choose the alpha and beta occupations jointly, for UHF."""
    energy_a, nelec_a = _lowest_energy_occupation_per_symmetry(
        eps[0], irreps[0], nelec[0], nirrep
    )
    energy_b, nelec_b = _lowest_energy_occupation_per_symmetry(
        eps[1], irreps[1], nelec[1], nirrep
    )
    # this makes each entry of "costs" correspond to a determinant with irrep "target"
    costs = energy_a + energy_b[np.arange(nirrep) ^ target]
    alpha = int(np.argmin(costs))
    if not np.isfinite(costs[alpha]):
        raise ValueError(
            f"No occupation gives a determinant of target symmetry {target}."
        )
    return np.array([nelec_a[alpha], nelec_b[alpha ^ target]])
