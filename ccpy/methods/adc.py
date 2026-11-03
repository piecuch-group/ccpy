"""DF EE/IP/EA-ADC(2) with bare-Hamiltonian guesses and CCpy Davidson."""
import numpy as np

from ccpy.adc.eeadc2 import ADCVector, EEADC2, update
from ccpy.adc.charged import ChargedADC2, ChargedVector
from ccpy.methods.base import MethodBase
from ccpy.methods.guess import GUESS_ORDER, InitialGuessMixin
from ccpy.models.operators import ClusterOperator, FockOperator


class _VirtualERI:
    """Scalar virtual integrals for active EA-CISd guesses, directly from DF."""
    def __init__(self, first, second, antisymmetric):
        self.first, self.second, self.antisymmetric = first, second, antisymmetric

    def __getitem__(self, indices):
        p, q, r, s = indices
        value = np.dot(self.first[:, p, r], self.second[:, q, s])
        if self.antisymmetric:
            value -= np.dot(self.first[:, p, s], self.second[:, q, r])
        return value


class _GuessBlock:
    def __init__(self, block, first, second, antisymmetric):
        self.block = block
        self.vvvv = _VirtualERI(first, second, antisymmetric)

    def __getattr__(self, key):
        return getattr(self.block, key)


def _bare_guess_hamiltonian(H):
    from types import SimpleNamespace

    return SimpleNamespace(a=H.a, b=H.b, chol=H.chol,
                           aa=_GuessBlock(H.aa, H.chol.a.vv, H.chol.a.vv, True),
                           ab=_GuessBlock(H.ab, H.chol.a.vv, H.chol.b.vv, False),
                           bb=_GuessBlock(H.bb, H.chol.b.vv, H.chol.b.vv, True))


class ADC2(InitialGuessMixin, MethodBase):
    """Strict EE-ADC(2) for real RHF/ROHF/UHF and noncanonical references.

    ``initial_guess`` has the same roots/irreps/active-space API as EOMCC, but
    always diagonalizes the bare Hamiltonian. No CC calculation/HBar is built.
    R uses the independent-determinant metric; ``R[k].as_cluster()`` converts
    to ordinary CCpy spin-block amplitudes. ``virtual_density`` is the
    second-order unrelaxed ISR state density, suitable for FNO selection.
    """

    default_guess = "cisd"
    guess_methods = ("cis", "cisd_chol")
    fno_sector = "EE"
    vector_type = ADCVector

    def __init__(self, upstream, roots=None, nfrozen=None, ndelete=None, formulation="folded", **options):
        if formulation not in ("full", "folded"):
            raise ValueError("ADC formulation must be full or folded.")
        self.formulation = formulation
        if formulation == "folded":
            self.default_guess = {"EE": "cis", "IP": "ipcis", "EA": "eacis"}[self.fno_sector]
        super().__init__(upstream, nfrozen, ndelete, **options)
        self.roots = None if roots is None else list(roots)
        self.R = self.omega = self.root_irreps = self.root_converged = None
        self.guess_settings = self.guess_energy = self.guess_vectors = self.guess_irreps = None
        self._kernel = None

    @property
    def kernel(self):
        if self._kernel is None:
            self._kernel = (EEADC2(self.reference) if self.fno_sector == "EE"
                            else ChargedADC2(self.reference, self.fno_sector))
        return self._kernel

    @property
    def correlation_energy(self):
        return self.kernel.correlation_energy

    @property
    def nroots(self):
        return len(self.R) if self.R is not None else 0

    def _guess_arguments(self):
        if self.guess_settings["method"] == "cisd_chol":
            return self.system, self.reference.H, ClusterOperator(self.system, 2)
        return self.system, _bare_guess_hamiltonian(self.reference.H)

    def _new_guess_operator(self):
        return ClusterOperator(self.system, 2)

    def _new_operator(self):
        return self.vector_type(self.kernel.space)

    def _leading_orbitals(self, R):
        return super()._leading_orbitals(R.as_cluster() if hasattr(R, "as_cluster") else R)

    def _configure_guess(self, settings):
        settings = super()._configure_guess(settings)
        if self.fno_sector == "EE" and self.reference.rhf_symmetry and settings["multiplicity"] not in (1, 3):
            raise ValueError("Closed-shell EE-ADC(2) guesses must target singlets or triplets.")
        self.R = None
        return settings

    def _copy_setup(self, new):
        new.formulation = self.formulation
        if self.guess_settings is not None:
            new.initial_guess(**self.guess_settings)

    def solve(self):
        from ccpy.drivers.solvers import eomcc_davidson

        kernel = self.kernel
        if self.options["davidson_solver"] != "standard":
            raise NotImplementedError("ADC2 currently uses sequential eomcc_davidson; select davidson_solver='standard'.")
        if self.R is None:
            self.run_guess()
            roots = self.selected_guess_roots(None if self.guess_settings["roots"] is not None else self.roots)
            self.R = []
            for k in roots:
                guess = self._new_guess_operator()
                guess.unflatten(self.guess_vectors[:, k], order=GUESS_ORDER[self.guess_settings["method"]])
                self.R.append(self._new_operator().from_cluster(guess))
            self.omega = np.asarray([self.guess_energy[k] for k in roots])
            self.root_irreps = [self.guess_irreps[k] for k in roots]
        self.root_converged = []
        self._print_header()
        locked = []
        for k, R in enumerate(self.R):
            print(f"   DF-{self.fno_sector}-ADC(2) root {k}, irrep {self.root_irreps[k]}")
            Q = np.column_stack(locked) if locked else np.empty((R.ndim, 0))
            mask = kernel.symmetry_mask(self.root_irreps[k])
            seed = R.flatten() * mask
            seed -= Q @ (Q.T @ seed)
            norm = np.linalg.norm(seed)
            if norm < 1.e-10:
                raise ValueError("ADC guess duplicates an already converged root; enlarge/change the CISd guesses.")
            seed /= norm
            shift = 100 + 10 * np.max(np.abs(kernel.diagonal), initial=0)

            def action(dR, trial, T, H, RHF_symmetry, system):
                vector = trial.flatten()
                projected = (vector - Q @ (Q.T @ vector)) * mask
                sigma = H.matvec(projected)
                return (sigma - Q @ (Q.T @ sigma)) * mask + shift * (Q @ (Q.T @ vector) + vector * ~mask)

            if self.formulation == "folded":
                from ccpy.adc.folded import solve_folded
                vector, self.omega[k] = solve_folded(kernel, R.flatten(), self.omega[k], mask,
                                                     self.system, self.options)
                if any(abs(np.dot(vector, q)) > 1 - 1.e-6 for q in locked):
                    raise RuntimeError("Folded ADC guesses converged to the same root; change guesses or use formulation='full'.")
                self.R[k].unflatten(vector)
                ok = True
            else:
                self.R[k], self.omega[k], ok = eomcc_davidson(
                    action, update, seed, R, self._new_operator(), self.omega[k], None,
                    kernel, self.system, self.options)
            vector = self.R[k].flatten()
            norm = np.linalg.norm(vector)
            if norm < 1.e-8:
                raise RuntimeError("ADC Davidson returned a vanishing Ritz vector.")
            vector /= norm
            self.R[k].unflatten(vector)
            residual = np.linalg.norm(kernel.matvec(vector) - self.omega[k] * vector)
            ok = bool(ok and residual < self.options["amp_convergence"] * 10)
            self.root_converged.append(ok)
            if not ok:
                self._finish_solve(False)
                raise RuntimeError(f"ADC(2) root {k} did not converge (residual {residual:.3e}).")
            locked.append(vector / np.linalg.norm(vector))
        self.check_root_symmetry(self.R, self.root_irreps)
        self._finish_solve(True)
        return self

    def raw_energy(self, root=None):
        self.ensure_solved()
        energies = self.reference_energy + self.correlation_energy + self.omega
        return energies if root is None else energies[root]

    def energy(self, root=None):
        energies = self.raw_energy() + self._state_shifts(self.nroots, self.fno_sector, self.root_irreps)
        return energies if root is None else energies[root]

    @property
    def completed_omega(self):
        return self.energy() - self._complete_ground_energy(self.reference_energy + self.correlation_energy)

    def virtual_density(self, root=0):
        self.ensure_solved()
        return self.kernel.virtual_density(self.R[root].flatten())


class IPADC2(ADC2):
    """DF IP-ADC(2), alpha-removal branch, with bare IP-CISd root targeting.

    omega = E(N-1)-E(N), so positive roots are ionization potentials.
    """
    default_guess = "ipcisd"
    guess_methods = ("ipcis", "ipcisd")
    guess_multiplicity = -1
    fno_sector = "IP"
    vector_type = ChargedVector

    def _new_guess_operator(self):
        return FockOperator(self.system, 1, 2)

    def _leading_orbitals(self, R):
        R = R.as_cluster() if hasattr(R, "as_cluster") else R
        return [int(np.argmax(np.abs(R.a)))]


class EAADC2(ADC2):
    """DF EA-ADC(2), alpha-attachment branch, with bare EA-CISd root targeting.

    omega = E(N+1)-E(N); a positive electron affinity equals -omega.
    """
    default_guess = "eacisd"
    guess_methods = ("eacis", "eacisd")
    guess_multiplicity = -1
    fno_sector = "EA"
    vector_type = ChargedVector

    def _new_guess_operator(self):
        return FockOperator(self.system, 2, 1)

    def _leading_orbitals(self, R):
        R = R.as_cluster() if hasattr(R, "as_cluster") else R
        return [self.system.noccupied_alpha + int(np.argmax(np.abs(R.a)))]


def evaluate_adc2(reference, targets=(), need_density=True, full_result=None):
    """Built-in EE/IP/EA FNO provider with projected physical-root following.

    Full-space roots use bare CISd guesses. Reduced roots are seeded by the
    projected full-space ISR vectors and followed by overlap Davidson. A lost
    root (normalized projected overlap below 0.5) is rejected, not relabeled.
    """
    from ccpy.methods.fno import FNOEvaluation
    from ccpy.mbpt.dfmp2 import dfmp2

    if any(t.sector not in ("EE", "IP", "EA") for t in targets):
        raise NotImplementedError("The built-in ADC(2) FNO provider supports EE, IP, and EA targets.")
    if not targets:
        ground_correlation, ground_density = dfmp2(reference, need_density=need_density)
        return FNOEvaluation(reference.energy + ground_correlation,
                             {None: ground_density} if need_density else {})
    energies, densities, states = {}, {}, {}
    groups = {}
    for target in targets:
        default = reference.system.multiplicity if target.sector == "EE" else -1
        groups.setdefault((target.sector, target.multiplicity or default, target.formulation), []).append(target)
    kernels = {}
    for (sector, multiplicity, formulation), group in groups.items():
        cls = {"EE": ADC2, "IP": IPADC2, "EA": EAADC2}[sector]
        method = cls(reference, formulation=formulation, maximum_iterations=150, amp_convergence=1.e-8, energy_convergence=1.e-10,
                     davidson_max_subspace_size=40)
        if sector in kernels:
            method._kernel = kernels[sector]
        else:
            kernels[sector] = method.kernel
        if full_result is None:
            counts = {}
            for target in group:
                if target.irrep is None and reference.system.point_group != "C1":
                    raise ValueError("ADC-FNO targets in a symmetry reference must specify an irrep.")
                irrep = target.irrep or reference.system.reference_symmetry
                counts[irrep] = max(counts.get(irrep, 0), target.root + 1)
            method.initial_guess(multiplicity=multiplicity, nacto=min(2, reference.system.noccupied_beta),
                                 nactu=min(2, reference.system.nunoccupied_alpha), roots_per_irrep=counts)
            method.solve()
            selected = []
            for target in group:
                matches = [k for k, irrep in enumerate(method.root_irreps)
                           if irrep == (target.irrep or reference.system.reference_symmetry)]
                if len(matches) <= target.root:
                    raise ValueError(f"ADC(2) guess did not provide target {target}.")
                selected.append(matches[target.root])
        else:
            method.R, method.omega, method.root_irreps = [], [], []
            for target in group:
                old = full_result.metadata["states"][target]
                cluster = old["cluster"]
                rotations = {}
                for s, spin in (("a", "alpha"), ("b", "beta")):
                    no = getattr(reference.system, f"noccupied_{spin}") + reference.nfrozen
                    rotations[s] = reference.orbital_transform[s][no:, no:]
                projected = method._new_guess_operator()
                if sector == "EE":
                    for s in ("a", "b"):
                        setattr(projected, s, rotations[s].T @ getattr(cluster, s))
                    for s in ("aa", "ab", "bb"):
                        setattr(projected, s, np.einsum("ai,bj,abkl->ijkl", rotations[s[0]], rotations[s[1]],
                                                       getattr(cluster, s), optimize=True))
                elif sector == "IP":
                    projected.a = cluster.a.copy()
                    projected.aa = np.einsum("ba,ibj->iaj", rotations["a"], cluster.aa)
                    projected.ab = np.einsum("ba,ibj->iaj", rotations["b"], cluster.ab)
                else:
                    projected.a = rotations["a"].T @ cluster.a
                    projected.aa = np.einsum("ai,bj,abk->ijk", rotations["a"], rotations["a"], cluster.aa)
                    projected.ab = np.einsum("ai,bj,abk->ijk", rotations["a"], rotations["b"], cluster.ab)
                method.R.append(method._new_operator().from_cluster(projected))
                method.omega.append(old["omega"])
                method.root_irreps.append(target.irrep or reference.system.reference_symmetry)
            seeds = [r.flatten() for r in method.R]
            method.omega = np.asarray(method.omega)
            method.solve()
            selected = list(range(len(group)))
            for target, seed, vector in zip(group, seeds, method.R):
                overlap = abs(np.dot(seed, vector.flatten())) / np.linalg.norm(seed)
                if overlap < 0.5:
                    raise ValueError(f"ADC-FNO lost target {target} (projected overlap {overlap:.3f}); retain more virtuals.")
        for target, k in zip(group, selected):
            energies[target] = float(method.raw_energy(k))
            if need_density:
                densities[target] = method.virtual_density(k)
            states[target] = {"cluster": method.R[k].as_cluster(), "omega": float(method.omega[k])}
    return FNOEvaluation(reference.energy + method.correlation_energy, densities, energies,
                         metadata={"states": states, "density_convention": "second-order unrelaxed ISR virtual block"})
