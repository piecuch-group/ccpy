"""State-universal multireference coupled-cluster methods.

    mol = Molecule(geometry, "cc-pvdz", auxiliary_basis="cc-pvdz-rifit")
    mf = RHF(mol)
    mr = SUMRCCSD(mf, 2, 2, nfrozen=1)    # CAS(2,2) model space
    energies = mr.energy()                # one energy per model-space determinant
    corrected = mr.mmsucc23()             # with the MM-SUCC(2,3) triples corrections
    # general (incomplete) model space: N_alpha then N_beta occupied correlated orbitals (1-based)
    mr = SUMRCCSD(mf, determinants=[[1, 2, 3, 4, 5, 1, 2, 3, 4, 5], [1, 2, 3, 4, 6, 1, 2, 3, 4, 6]], nfrozen=1)

``energy()`` solves the SU-MRCC equations if needed (always from T = 0).
"""
import numpy as np

from ccpy.methods.base import MethodBase
from ccpy.utilities.printing import get_timestamp


class SUMRCCSD(MethodBase):
    """SU-MRCCSD (Jeziorski-Monkhorst) of RHF/ROHF orbitals in the complete model space
    CAS(nact_electrons, nact_orbitals), with the active orbitals right above the core, or in a
    general model space of given determinants (Li-Paldus C conditions on the internal amplitudes).

    The roots are the eigenvalues of the non-Hermitian effective Hamiltonian, one per
    model-space determinant (all spin states with the reference's M_s in a CAS; point-group
    symmetry is not used to restrict it).

    Parameters
    ----------
    nact_electrons, nact_orbitals : int, optional
        Active electrons and orbitals of a complete model space.
    determinants : sequence of int arrays, optional
        Instead of a CAS: the model-space determinants, each listing the N_alpha occupied alpha
        and then the N_beta occupied beta correlated orbitals, numbered from 1 (as in
        ccpy.models.hilbert.Determinant), e.g. a CAS with problematic determinants removed.
    prune_intruders : float, optional
        Gap (Eh): before solving, drop determinants, highest energy first, until every single
        or double of the remaining ones lies at least this far above the highest of them
        (orbital-energy estimate; 0.0 drops what the intruder warning flags). The roots then
        belong to the reduced, general model space. A positive gap is necessary for Jacobi
        convergence but no value guarantees it.
    solver : {"auto", "jacobi", "newton_krylov"}
        "auto" runs Jacobi/DIIS and, if it fails, Newton-Krylov from its best iterate.
        Intruder states (external determinants below the highest reference) are reported.
    **options
        maximum_iterations, amp_convergence (residuum and root changes), energy_shift and
        diis_size are used.

    Attributes
    ----------
    T : list of ClusterOperator
        Cluster operators of the references, each in the orbital order of its reference.
    heff : ndarray
        Effective Hamiltonian, Heff_pq = <Phi_p|H exp(T(q))|Phi_q>.
    roots : ndarray
        Its eigenvalues, ascending real part (total energies).
    coefficients : ndarray
        Right eigenvectors (columns), the model-space coefficients c_p of each root.
    model_space : SimpleNamespace
        Determinants, reference orbital lists and internal excitations (ccpy.mrcc.sumrcc).
    """

    frozen_cv = False

    def __init__(self, upstream, nact_electrons=None, nact_orbitals=None, nfrozen=None, ndelete=None, *,
                 determinants=None, solver="auto", prune_intruders=None, **options):
        super().__init__(upstream, nfrozen, ndelete, **options)
        if self.reference.is_uhf:
            raise NotImplementedError(f"{type(self).__name__} requires RHF or ROHF orbitals.")
        if solver not in ("auto", "jacobi", "newton_krylov"):
            raise ValueError(f"{type(self).__name__}: unknown solver {solver}.")
        if (determinants is None) == (nact_electrons is None or nact_orbitals is None):
            raise ValueError(f"{type(self).__name__}: give either nact_electrons and nact_orbitals or determinants.")
        self.nact_electrons, self.nact_orbitals, self.solver = nact_electrons, nact_orbitals, solver
        self.prune_intruders = prune_intruders
        self.determinants = None if determinants is None else [np.asarray(d) for d in determinants]
        self.T = self.heff = self.roots = self.coefficients = self.model_space = None

    def solve(self):
        from ccpy.mrcc.sumrcc import get_cas_determinants
        from ccpy.mrcc.sumrccsd import kernel

        self._warn_df_port()
        self._print_header()
        print("   SU-MRCC calculation started on", get_timestamp())
        reference = self.reference
        determinants = self.determinants or get_cas_determinants(self.system, self.nact_electrons,
                                                                 self.nact_orbitals)
        self.T, self.heff, self.roots, self.coefficients, self.model_space = kernel(
            reference.h_mo["a"], reference.B_mo["a"], self.system, determinants,
            frozen_cv=self.frozen_cv, solver=self.solver, prune_intruders=self.prune_intruders, maxit=self.options["maximum_iterations"],
            convergence=self.options["amp_convergence"], shift=self.options["energy_shift"],
            diis_size=self.options["diis_size"])
        print("   SU-MRCC calculation ended on", get_timestamp())
        self._finish_solve(True)
        return self

    def mmsucc23(self, root=None):
        """Energies of the roots (or of one root) with the MM-SUCC(2,3) noniterative triples
        correction of Piecuch and Kowalski [IJMS 3, 676 (2002)], using the MRMBPT(2)-like (CR-CCSD(T)
        analog) trial wave function (ccpy.mrcc.mmsucc23). The corrections are kept in
        self.mm23_corrections. Evaluated on the fly over the triples (no triples are stored)."""
        from ccpy.mrcc.mmsucc23 import mmsucc23
        from ccpy.mrcc.sumrcc import get_reference_hamiltonian

        self.ensure_solved()
        H = [get_reference_hamiltonian(self.reference.h_mo["a"], self.reference.B_mo["a"], self.system, ref)[0]
             for ref in self.model_space.references]
        energies, self.mm23_corrections = mmsucc23(self.T, H, self.heff, self.roots, self.coefficients,
                                                   self.model_space, self.system)
        return energies if root is None else energies[root]

    def raw_energy(self, root=None):
        """Total energies of the roots (or of one root), solving the equations if needed."""
        self.ensure_solved()
        roots = self.roots.real
        return roots if root is None else roots[root]

    def energy(self, root=None):
        return self.raw_energy(root)


class SUMRCCSD1(SUMRCCSD):
    """SUMRCCSD(1): SU-MRCCSD with core->virtual t1 = 0 and core-core->virtual-virtual t2
    fixed at their first-order MRMBPT values (Piecuch & Kowalski, IJMS 3, 676 (2002))."""

    frozen_cv = True
