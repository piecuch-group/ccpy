"""Initial guesses of the excited-state methods, targeted by irreducible representation.

    eom = EOMCCSD(mf, nfrozen=1)
    eom.initial_guess(roots_per_irrep={"B2": 2, "A1": 1}, nacto=3, nactu=3)
    eom.energy()
    eom.root_irreps                   # ["B2", "B2", "A1"]

The guess is a CI-like diagonalization (ccpy.eom_guess) in a space of the operator's leading
excitations: CISd (all singles, active-space doubles) for EE, IP-/EA-CISd for IP/EA, SF-CIS
for SF, and DIP-/DEA-CIS for DIP/DEA. ``roots_per_irrep`` maps irreps of the point group to
numbers of roots; an integer n asks for the n lowest roots of any symmetry.

Guesses with point-group symmetry (cis, cisd, ipcisd, eacisd, dipcis, deacis) are run irrep by
irrep; those without it (sfcis, ipcis, eacis) are diagonalized in full and their roots sorted
into irreps by their leading amplitude. Every guess root keeps its irrep (``guess_irreps``), and
the converged roots are checked against it.
"""
from importlib import import_module

import numpy as np

# guess methods that generate singles (order 1) or singles and doubles (order 2) only
GUESS_ORDER = {"cis": 1, "sfcis": 1, "eacis": 1, "ipcis": 1,
               "cisd": 2, "cisd_chol": 2, "deacis": 2, "dipcis": 2, "eacisd": 2, "ipcisd": 2}
# guess methods that diagonalize one irrep at a time
SYMMETRY_BLOCKED = ("cis", "cisd", "cisd_chol", "ipcisd", "eacisd", "dipcis", "deacis")
# root count asking the guesses without symmetry for all of their (singles) roots
ALL_ROOTS = 10 ** 9


class InitialGuessMixin:
    """roots_per_irrep initial guesses for methods with a ``system`` and a ``_new_operator``."""

    default_guess = "cisd"
    guess_methods = None            # allowed guess methods (None: all of GUESS_ORDER)
    guess_multiplicity = None       # multiplicity of the targeted states (None: the reference's)

    def initial_guess(self, method=None, multiplicity=None, nacto=None, nactu=None, roots_per_irrep=None,
                      use_symmetry=True, roots=None, debug=False):
        """Set up the initial guess, run when the equations are first solved.

        Parameters
        ----------
        method : str, guess method (default: CISd for EE, IP-/EA-CISd, SF-CIS, DIP-/DEA-CIS)
        multiplicity : int, spin multiplicity of the targeted states (-1: no spin adaptation;
            the default for IP, EA, DIP, DEA and SF)
        nacto, nactu : int, active occupied/unoccupied orbitals of the guess doubles (0: none)
        roots_per_irrep : dict {irrep: number of roots}, or an int (lowest roots of any symmetry)
        use_symmetry : bool, target roots by irrep (False: the lowest roots, labeled by irrep)
        roots : list of int, guess roots (0-based, in the order of the guess) to converge
        """
        method = (method or self.default_guess).lower()
        if method == "cisd" and not self.requires_df_port:
            method = "cisd_chol"
        allowed = self.guess_methods or tuple(GUESS_ORDER)
        if method not in allowed:
            raise ValueError(f"{type(self).__name__}: guess method must be one of {allowed}, not {method}.")
        if multiplicity is None:
            multiplicity = self._default_guess_multiplicity()
        if roots is not None:
            roots = [int(k) for k in roots]
            if not roots or min(roots) < 0 or len(set(roots)) != len(roots):
                raise ValueError("Guess roots must be distinct nonnegative integer indices.")
        if roots_per_irrep is not None:
            roots_per_irrep = self._check_roots_per_irrep(roots_per_irrep, use_symmetry)
            if roots is not None and max(roots) >= self._count(roots_per_irrep):
                raise ValueError("Requested guess index exceeds the roots_per_irrep count.")
        settings = dict(method=method, multiplicity=multiplicity, nacto=nacto, nactu=nactu,
                        roots_per_irrep=roots_per_irrep, use_symmetry=use_symmetry, roots=roots, debug=debug)
        self.guess_settings = self._configure_guess(settings)
        self.guess_energy = self.guess_vectors = self.guess_irreps = None
        self.converged = False
        return self

    # ------------------------------------------------------------------ hooks
    def _default_guess_multiplicity(self):
        return self.system.multiplicity if self.guess_multiplicity is None else self.guess_multiplicity

    def _configure_guess(self, settings):
        """Method-specific checks of the settings (returned, possibly completed)."""
        settings["nacto"] = settings["nacto"] or 0
        settings["nactu"] = settings["nactu"] or 0
        return settings

    def _default_root_count(self):
        raise ValueError(f"{type(self).__name__}.initial_guess needs roots_per_irrep, e.g., {{'A1': 2}}.")

    def _guess_arguments(self):
        """Leading arguments of run_diagonalization: (system, H) or (system, H, T)."""
        hbar = self.hbar
        return (self.system, hbar.H) + ((hbar.T,) if self.guess_settings["method"] == "cisd_chol" else ())

    def _new_guess_operator(self):
        return self._new_operator()

    def _leading_orbitals(self, R):
        """Correlated orbitals (occupied i, virtual noccupied + a) of the largest singles or
        doubles amplitude of an excitation operator R."""
        no = {"a": self.system.noccupied_alpha, "b": self.system.noccupied_beta}
        best, orbitals = 0.0, None
        for block in ("a", "b", "aa", "ab", "bb"):
            X = getattr(R, block, None)
            if not isinstance(X, np.ndarray) or X.ndim != 2 * len(block) or not X.size:
                continue
            idx = np.unravel_index(np.argmax(np.abs(X)), X.shape)
            if abs(X[idx]) > best:
                k = len(block)
                best, orbitals = abs(X[idx]), [no[s] + p for s, p in zip(block, idx[:k])] + list(idx[k:])
        return orbitals

    # ------------------------------------------------------------------ guess
    def _check_roots_per_irrep(self, roots_per_irrep, use_symmetry):
        if isinstance(roots_per_irrep, (int, np.integer)):
            if roots_per_irrep < 1:
                raise ValueError("roots_per_irrep needs a positive total root count.")
            return int(roots_per_irrep)
        counts = dict(roots_per_irrep)
        if (not counts or any(not isinstance(n, (int, np.integer)) or n < 0 for n in counts.values())
                or sum(counts.values()) < 1):
            raise ValueError("roots_per_irrep needs a positive total root count.")
        irreps = self.system.point_group_irrep_to_number
        unknown = [irrep for irrep in counts if irrep not in irreps]
        if use_symmetry and unknown:
            raise ValueError(f"Irreps {unknown} are not in point group {self.system.point_group} "
                             f"({', '.join(irreps)}).")
        return {irrep: int(n) for irrep, n in counts.items()}

    @staticmethod
    def _count(roots_per_irrep):
        return roots_per_irrep if isinstance(roots_per_irrep, int) else sum(roots_per_irrep.values())

    def _irrep(self, orbitals):
        """Irrep of an excitation from (or attachment to) the given correlated orbitals."""
        if orbitals is None:
            return None
        number = self.system.point_group_irrep_to_number
        x = number[self.system.reference_symmetry]
        for p in orbitals:
            x ^= number[self.system.orbital_symmetries[p]]
        return self.system.point_group_number_to_irrep[x]

    def _vector_irrep(self, v):
        R = self._new_guess_operator()
        R.unflatten(v, order=GUESS_ORDER[self.guess_settings["method"]])
        return self._irrep(self._leading_orbitals(R))

    def run_guess(self):
        """Diagonalize the guess; sets guess_energy, guess_vectors (columns) and guess_irreps."""
        if self.guess_settings is None:
            raise RuntimeError(f"Call {type(self).__name__}.initial_guess(...) before solving.")
        s = self.guess_settings
        module = import_module("ccpy.eom_guess." + s["method"])
        args = self._guess_arguments()

        def diagonalize(counts, use_symmetry):
            omega, V = module.run_diagonalization(*args, s["multiplicity"], counts, s["nacto"], s["nactu"],
                                                  debug=s["debug"], use_symmetry=use_symmetry)
            found = [k for k in range(V.shape[1]) if np.any(V[:, k])]
            return np.asarray(omega, dtype=float)[found], V[:, found]

        counts = s["roots_per_irrep"]
        if counts is None:
            counts = max(s["roots"]) + 1 if s["roots"] is not None else self._default_root_count()
        by_irrep = s["use_symmetry"] and isinstance(counts, dict)
        if by_irrep and s["method"] in SYMMETRY_BLOCKED:
            energies, vectors, irreps = [], [], []
            for irrep, n in counts.items():
                if n == 0:
                    continue
                omega, V = diagonalize({irrep: n}, True)
                if len(omega) < n:
                    print(f"   WARNING: the {s['method']} guess has {len(omega)} of the {n} {irrep} roots requested.")
                energies.append(omega)
                vectors.append(V)
                irreps += [irrep] * len(omega)
            energy, V = np.concatenate(energies), np.hstack(vectors)
        else:
            # the lowest roots of any symmetry; with symmetry, all (singles) roots are sorted into irreps
            total = self._count(counts)
            request = ALL_ROOTS if by_irrep else total
            omega, V = diagonalize({next(iter(counts)) if isinstance(counts, dict) else None: request}, False)
            labels = [self._vector_irrep(V[:, k]) for k in range(V.shape[1])]
            if by_irrep:
                keep = []
                for irrep, n in counts.items():
                    found = [k for k, label in enumerate(labels) if label == irrep][:n]
                    if len(found) < n:
                        print(f"   WARNING: the {s['method']} guess has {len(found)} of the {n} {irrep} roots requested.")
                    keep += found
            else:
                keep = list(range(min(total, len(omega))))
            energy, V, irreps = omega[keep], V[:, keep], [labels[k] for k in keep]
        if not len(energy):
            raise ValueError(f"The {s['method']} guess has none of the requested roots.")
        self.guess_energy, self.guess_vectors, self.guess_irreps = energy, V, irreps
        print(f"\n   Initial guess ({s['method']}, multiplicity {s['multiplicity']})")
        print("   root   irrep        energy")
        for k, (e, irrep) in enumerate(zip(energy, irreps)):
            print(f"   {k:4d}   {str(irrep):6s}  {e:14.8f}")
        return self.guess_energy

    def selected_guess_roots(self, roots=None):
        """Guess roots to converge: ``roots`` (or the initial_guess roots), all by default."""
        roots = roots if roots is not None else self.guess_settings["roots"]
        if roots is None:
            return list(range(len(self.guess_energy)))
        missing = [k for k in roots if k >= len(self.guess_energy)]
        if missing:
            raise ValueError(f"Requested guess roots {missing} are unavailable (the guess has "
                             f"{len(self.guess_energy)} roots in this spin and active space).")
        return list(roots)

    def check_root_symmetry(self, R, irreps):
        """Irreps of the converged roots R (from their leading amplitudes), with a warning for
        each root whose symmetry differs from that of its guess."""
        found = []
        for k, (Rk, expected) in enumerate(zip(R, irreps)):
            irrep = self._irrep(self._leading_orbitals(Rk))
            if expected is not None and irrep is not None and irrep != expected:
                print(f"   WARNING: root {k} converged to a {irrep} state from a {expected} guess.")
            found.append(irrep)
        return found
