Spin-integrated STEOM-CCSD
=========================

``STEOMCC`` is available from ``ccpy`` and ``ccpy.methods``. It requires a
closed-shell restricted reference and computes singlet (``multiplicity=1``) or
triplet (``multiplicity=3``) excitation energies. The ground-state calculation
is CCSD; an upstream ``CCSD`` object is reused.

.. code-block:: python

   from ccpy import Molecule, RHF, CCSD, STEOMCC

   mol = Molecule("Li 0 0 0; H 0 0 1.6", "sto-3g",
                  auxiliary_basis="def2-universal-jkfit", verbose=False)
   mf = RHF(mol).run()
   cc = CCSD(mf)

   # State-averaged CIS natural orbitals select the active spaces automatically.
   steom = STEOMCC(cc, nroots=2, multiplicity=1).solve()
   print(steom.omega)       # excitation energies in hartree
   print(steom.energy())    # total excited-state energies in hartree
   print(steom.energy(0))   # total energy of root 0

Manual active spaces
--------------------

Specify both ``nacto`` and ``nactu`` to select the highest occupied and lowest
virtual canonical orbitals. Alternatively, give both ``active_occupied`` and
``active_virtual`` as lists of distinct, zero-based indices within their
respective correlated spaces. Frozen occupied and deleted virtual orbitals
are excluded from these indices.

.. code-block:: python

   steom = STEOMCC(cc, nroots=2, nacto=1, nactu=2)
   explicit = STEOMCC(cc, nroots=2, active_occupied=[1], active_virtual=[0, 1])

The active space selects the model orbitals and the targeted principal IP/EA
roots. The auxiliary IP-EOMCCSD(2h-1p) and EA-EOMCCSD(2p-1h) correlation spaces
retain all correlated orbitals. The final diagonalization also retains all
occupied-to-virtual singles, including inactive ones. It is not an active-only
CIS projection.

Automatic selection
-------------------

The automatic algorithm forms the occupied and virtual state-averaged CIS
densities for the lowest ``nroots`` CIS states of the requested multiplicity.
It selects natural orbitals by cumulative population and diagonalizes the
active/active and inactive/inactive Fock blocks separately. Degenerate natural
orbital populations are kept together at the selection boundary.

The defaults follow Dutta et al., J. Chem. Phys. **146**, 074103 (2017):

============================== ========== ==========
Parameter                      Singlets   Triplets
============================== ========== ==========
``occupied_threshold``         0.99       0.999
``virtual_threshold``          0.99       0.999
``ip_threshold``               0.90       0.80
``ea_threshold``               0.90       0.80
============================== ========== ==========

All thresholds are fractions, not percentages. IP/EA thresholds apply to the
one-hole/one-particle weight using the determinant metric (including a factor
of one half for same-spin doubles). Low-character roots are discarded, with
pivoted QR choosing a matching square subset of the original model orbitals.
An empty sector, an unconverged auxiliary solve, or an overlap condition number
above ``condition_limit`` (default ``1e10``) raises an error. Set the IP/EA
thresholds to zero to retain all converged auxiliary states.

CCSD and its HBar can be computed in the original orbitals: rotating their
converged tensors is equivalent to recomputing them in the CIS-NO basis.
The upstream reference, T amplitudes, and cached HBar are preserved. Automatic
STEOM roots are matched one-to-one to the CIS states by maximum total squared
overlap. Manual calculations return the lowest STEOM roots in energy order.
Explicit neutral guesses can use per-irrep counts; the final Hamiltonian is
not diagonalized in separate point-group blocks. See neutral targeting below.

Equations and diagnostics
-------------------------

``ccpy/eomcc/steomccsd.py`` implements the spatial-orbital equations of
Nooijen and Bartlett, J. Chem. Phys. **107**, 6812 (1997). The ``ab`` blocks of
the closed-shell CCpy HBar supply the spin-free two-electron coefficients in
1212 notation. Intermediate normalization uses a matrix solve on the active
one-hole/one-particle overlaps; IP amplitudes carry the extra hole-contraction
minus sign. The G construction includes the mixed IP/EA quadratic terms
that account for implicit triple excitations.

Only G.oo, G.vv, G.vovo and G.voov are formed. With row ``(a,i)`` and column
``(b,j)``, the final matrix is

.. math::

   A_{ai,bj} = \delta_{ij}g_{ab} - \delta_{ab}g_{ji}
                + c\,g_{ajib} - g_{ajbi},

where ``c=2`` for singlets and ``c=0`` for triplets. The S1 transformation is
omitted because it preserves the spectrum of the full singles block.

The first ``H.ov * S(EA)`` contribution to the phph intermediate is positive
in this implementation, differing from the minus printed in Eq. (57).
An independent determinant-space normal-ordering oracle, with nonzero H.ov,
verifies this sign and all linear and quadratic contributions for singlets
and triplets. Eq. (44) also requires attention to the order of its EA indices:
``s_ie^ab = s_ei^ba``.

Useful result attributes include:

* ``ip`` and ``ea``: auxiliary ``R``, ``omega``, ``root_converged``,
  ``singles_character`` and ``retained_roots``;
* ``active_occupied`` and ``active_virtual``: retained model labels;
* ``S_ip`` and ``S_ea``: compact tensors ``[m,i,j,b]`` and ``[e,a,b,j]``;
* ``G`` and ``matrix``: the transformed blocks and singles matrix;
* ``orbital_rotations``: occupied and virtual matrices, with old orbitals on
  rows and working orbitals on columns;
* ``active_weights``: final right-vector squared weight in active singles;
* ``residual_norms`` and ``root_converged``: final eigenpair checks.

G, S and active labels refer to the working basis. ``R`` contains normalized
spin-resolved singles transformed back to the upstream orbital basis. These
are reduced-space eigenvectors, not full STEOM wavefunctions or transition
densities. The IP/EA and final matrices are non-Hermitian; selected complex
final eigenpairs raise an error when they are genuinely complex. Conjugate
pairs split only by numerical noise are converted to a real basis when the
resulting vectors satisfy the requested residual tolerance.

Current scope
-------------

This implementation materializes the legacy four-index HBar and uses dense
CIS and final singles diagonalizations. It supports energies, with the common
method API's finite-difference gradients; no analytical STEOM gradients,
transition properties, open-shell equations, or DF-specific STEOM contractions
are provided. Auxiliary roots are followed individually from active orbital
unit vectors using the existing Davidson solver. If separate guesses converge
to linearly dependent roots, the model-overlap check rejects the calculation.

Validation
----------

``tests/eomcc/test_steomccsd.py`` compares the tensor contractions to independent
normal-ordered fermion algebra, including partial and full active spaces, and
checks normalization, filtering and CIS-NO selection.
``tests/methods/test_steomcc.py`` exercises singlet/triplet molecular calculations,
automatic selection, full-space orbital covariance, upstream cache preservation,
cache invalidation and failure handling. These checks do not replace a broad
comparison against published excitation-energy benchmarks.

Higher-rank auxiliaries and an SD-sized neutral problem
------------------------------------------------------

``STEOMCC(..., level="sdt")`` dispatches to ``STEOMCCSDT``. The ground-state
operator remains CCSD. The new level solves IP-EOMCCSD(3h-2p) and
EA-EOMCCSD(3p-2h) auxiliaries and diagonalizes the final neutral Hamiltonian
over singles **and doubles**.

.. code-block:: python

   extended = STEOMCC(cc, level="sdt", nroots=5,
                      nacto=2, nactu=3,
                      auxiliary_solver="pspace").solve()
   print(extended.omega)
   print(extended.singles_weights)
   print(extended.ip.model_weights)
   print(extended.ea.model_weights)

   # Equivalent explicit class; use dense auxiliary triples for comparison.
   from ccpy import STEOMCCSDT
   full_aux = STEOMCCSDT(cc, nroots=5, nacto=2, nactu=3,
                        auxiliary_solver="dense")

The auxiliary model is enlarged to include all correlated 1h/1p configurations
and all active 2h1p/2p1h **doublets**. Both the dense initial guesses and the
normalization use this enlarged model. The guesses come from ``ipcisd.py`` and
``eacisd.py``. One root is solved for each spin-adapted model configuration,
using block Davidson with a unique assignment of Ritz roots to guesses. This
allows predominantly 2h1p/2p1h roots to be retained rather than rejecting them
with the ordinary STEOM singles-character cutoff.

With auxiliary eigenvectors stored in the columns of R and an orthonormal
model basis P, the wave operator satisfies

.. math::

   \Omega P = R (P^\dagger R)^{-1} P^\dagger.

The number of auxiliary roots is therefore set by the model dimension, not by
the requested number of neutral roots. An unconverged root or an ill-conditioned
model overlap fails the calculation. In the new level ``ip_threshold`` and
``ea_threshold`` apply to the **total enlarged-model weight**, including active
satellites. Both default to zero. No root is silently dropped because doing so
would leave the prescribed model incomplete.

P-space selection
~~~~~~~~~~~~~~~~~

Only the highest-rank auxiliary sector is restricted in P-space mode. Lower
1h+2h1p / 1p+2p1h amplitudes remain unrestricted. By default auxiliary triples
contain at least one active hole (IP) or one active particle (EA), matching
CCpy's active 3h2p/3p2h convention. ``minimum_active=2`` or ``3`` tightens that
condition. Empty spin cases use CCpy's zero-amplitude sentinel convention.

``ip_pspace`` and ``ea_pspace`` can provide custom ``aaa``, ``aab`` and ``abb``
arrays with one-based canonical rows. IP rows are ``[b,c,i,j,k]``; EA rows are
``[a,b,c,j,k]``. These labels refer to the **working basis** after the reported
orbital rotation, not necessarily the upstream canonical basis. Custom spaces
should preserve spin symmetry. Fortran kernels may sort excitation lists;
the wrapper restores a fixed external ordering after every sigma evaluation
so that stored Davidson vectors retain their meaning. Caller-owned arrays
are copied.

``auxiliary_order=2`` selects the lower-rank IP/EA solvers with the same
enlarged model and neutral SD projection. It is useful for isolating the
higher-rank auxiliary contribution, but does not reproduce the 2000 paper's
original one-hole/one-particle normalization model.

Transformation and neutral sigma vectors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The 2000 extended-STEOM paper extends the final diagonalization space while
retaining an S2 transformation obtained from ordinary IP/EA auxiliaries. The
implementation here goes beyond that paper: its enlarged satellite model
requires additional normal-ordered operator components as well as the
higher-rank auxiliary amplitudes.

Write S by the numbers of quasiparticle creation and annihilation operators:

* ``S.one2``: three creators and one annihilator;
* ``S.one3``: five creators and one annihilator;
* ``S.same``: three creators and three annihilators;
* ``S.higher``: five creators and three annihilators.

The three-annihilator terms act on the satellite model. They are obtained
recursively from the normalized auxiliary wave operator, subtracting the
already determined one-annihilator action. Both IP and EA terms enter these
subtractions. In particular, simply inserting improved R2 into the original
STEOM S2 formula does not implement the enlarged-model transformation.

Let U be the normal exponential of S, and let C contain neutral singles and
doubles. The code forms the induced components

.. math::

   R_1 &= C_1, \\
   R_2 &= S_{3,1} C_1 + (1+S_{3,3}) C_2, \\
   R_3 &= S_{3,1} C_2 + S_{5,1} C_1 + S_{5,3} C_2
           + \tfrac12\{S_{3,1}^2\}C_1
           + \{S_{3,3}S_{3,1}\}C_2.

Braces exclude contractions between the S operators. Implicit neutral
quadruples are dropped before HBar acts, following the truncation strategy of
Eq. (18) in the 2000 paper. The SD-projected sigma vector is computed with
the existing EE-EOMCC(P) singles/doubles contractions and sparse induced
triples. No dense R3 array is allocated and no neutral triples are optimized.
The CCSD HBar's factorized three-body contributions are retained by these
existing contractions.

The SD block of U is lower triangular:

.. math::

   M=P_{SD}UP_{SD}=
   \begin{pmatrix}1&0\\S_{3,1}&1+S_{3,3}\end{pmatrix}.

The final action is

.. math::

   G_{SD}C=M^{-1}P_{SD}\bar H P_{SDT}UP_{SD}C.

The new rank-preserving satellite contribution makes the doubles block of M
nontrivial. The ordinary ``1-S2`` left factor is insufficient; a sparse LU
solve applies this inverse. This projected, omitted-quadruples extension is a
research implementation, not a claim of equivalence to the published
S2-only extended-STEOM method or to full EOMCCSDT.

Spin and solver behavior
~~~~~~~~~~~~~~~~~~~~~~~~

The final basis contains unique Ms=0 SD configurations and is projected onto
singlets or triplets by diagonalizing small connected blocks of the spin
matrix. ``nroots`` may exceed the CIS dimension. Without an explicit neutral
guess, roots are returned in energy order, including doubly excited roots.
Configured neutral guesses instead determine both the targeted states and their
returned order. ``singles_weights`` are measured
in the transformed EOM picture, normalized over R1+R2, as in the paper.

``G`` is a spin-adapted ``scipy.sparse.linalg.LinearOperator``. Dimensions up to
``dense_limit`` (default 80) use dense diagonalization; larger problems use
Arnoldi. Asking for almost all roots also uses dense diagonalization. ``matrix``
is populated only for dense solves. The solver options ``maximum_iterations``,
``amp_convergence`` and ``davidson_max_subspace_size`` control the iterative
solves. Upstream CCSD amplitudes and HBar remain unchanged, and neutral R
vectors are returned in the upstream orbital basis.

Automatic orbital selection uses CIS by default. With a configured CISd
guess, it instead includes the selected guesses' singles and doubles contributions
to the occupied-hole and virtual-particle densities. A manual auxiliary active
space remains available. The sparse
normal-product implementation is intended to establish correctness and avoids
full Fock-space matrices, but its Python exterior-algebra loops are not a
replacement for a fully factorized production tensor implementation. Large
satellite models can require many auxiliary roots, substantial storage, and
expensive induced-triples contractions. Published molecular benchmarks and
systematic active-space convergence studies remain necessary.

Additional validation
~~~~~~~~~~~~~~~~~~~~~

``tests/eomcc/test_steomccsdt_equations.py`` checks normal products against
independent fermion strings, verifies the enlarged-model normalization, and
compares the complete SD action and inverse metric against explicit small
Fock-space matrices. ``tests/methods/test_steomccsdt.py`` checks the exact
neutral two-electron limit, higher-rank full/P-space agreement, satellite roots,
Arnoldi diagonalization, and failure handling.

Targeting neutral CIS or CISd roots
----------------------------------

Both STEOM levels support ``initial_guess("cis", ...)`` and
``initial_guess("cisd", ...)``. CISd retains all singles and active doubles and
uses CCpy's existing spin-adapted ``ccpy.eom_guess.cisd`` solver. The guess
multiplicity is the STEOM calculation's multiplicity.

.. code-block:: python

   steom = STEOMCC(cc, level="sdt", nacto=2, nactu=3)
   steom.initial_guess("cisd", nacto=2, nactu=4, roots=[1, 5])
   steom.solve()
   print(steom.omega)                  # targets in guess order: 1, then 5
   print(steom.guess_singles_weights)  # singles fractions of the CISd guesses
   print(steom.guess_overlaps)         # squared final overlaps with the guesses

``roots`` are distinct zero-based indices in the energy-ordered, spin-adapted
guess spectrum. The call above computes enough CISd roots to select indices 1
and 5 and targets two final states, even if they are not the two lowest STEOM
states. Omitting ``roots`` targets the lowest ``nroots`` guess states. Optional
``roots_per_irrep`` supplies guess counts using the actual molecular irrep
labels, with ``use_symmetry=True`` to enable symmetry restriction in the guess.
Explicit indices then refer to the concatenated guess output in dictionary
order. Without explicit indices all requested per-irrep guesses are targeted.

The guess ``nacto`` and ``nactu`` define the highest occupied / lowest virtual
canonical orbitals allowed in the **neutral guess doubles**. They are independent
of the auxiliary active space passed to the STEOM constructor. For manual
auxiliary spaces the guess counts default to the corresponding active counts;
for automatic selection provide explicit CISd guess counts. The guess is solved
on the bare Hamiltonian in the original reference orbital basis, then rotated
into the STEOM working basis.

For the extended SD problem, final C vectors are compared to CIS/CISd guesses
in the transformed EOM picture: ``R_SD = M C_SD``. The normalized overlap uses
unique determinants, equivalently the metric with a 1/4 factor for same-spin
doubles and unit weight for mixed-spin doubles. A one-to-one assignment
maximizes the total squared overlap across targeted states. Dense solves search
the full spectrum. Configured iterative solves use block Davidson with the
same overlap criterion throughout; they do not search only the lowest-energy
Arnoldi roots. Intermediate complex Ritz pairs can occur in non-Hermitian
subspaces; real residual expansions are allowed, but converged targets must
satisfy the real full SD equation.

For automatic active-space selection, state averaging uses the selected
CISd vectors' full hole and particle densities. Thus a pure double with zero
singles amplitude still selects its occupied and virtual orbitals. OO and VV
one-body operators preserve excitation rank, so these densities contain the
singles/singles and doubles/doubles contributions without cross terms.

Use ``level="sdt"`` to represent neutral states dominated by doubles. The
original ``level="sd"`` Hamiltonian has only a singles space: it can accept a
CISd guess's singles projection, but a guess with no singles projection is
rejected. Neither guess selection nor root following enlarges that singles-only
Hamiltonian.

``guess_energy`` and ``guess_vectors`` retain the selected bare CI energies
and normalized ClusterOperator guesses. Changing ``initial_guess`` invalidates
the cached neutral solution; geometry reconstruction preserves the guess setup.
Without ``initial_guess``, the previous defaults remain: automatic ordinary
STEOM follows CIS roots, manual ordinary STEOM selects the lowest energies,
and extended STEOM selects the lowest energies.
