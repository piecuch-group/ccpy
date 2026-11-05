"""SU-MRCCSD: the single-reference (CCSD) limit, exactness for two electrons, and the
SU-MRCC equations (Eq. 15 of IJMS 3, 676; Eq. 35 of JCP 119, 5320) checked in the
determinant space, for complete and general model spaces."""
import numpy as np
import pytest
from scipy.linalg import expm

pyscf = pytest.importorskip("pyscf")
from pyscf import gto, scf, cc, ao2mo

from ccpy.gradients.ccpq.determinants import DeterminantSpace, spin_orbital_integrals, hamiltonian_matrix
from ccpy.models.system import System
from ccpy.mrcc.sumrcc import get_cas_determinants, get_model_space, get_reference_hamiltonian, \
    get_external_estimates, get_lowest_external
from ccpy.mrcc.sumrccsd import kernel, get_core_virtual, check_intruders, get_denominators
from test_sumrcc_projection import excitation_strings, determinant, cc_wavefunction

# slightly distorted rectangle: no symmetry, so active singles (and the C-condition t2) are nonzero
H4 = "H 0 0 0; H 2.0 0.1 0; H 0.2 2.4 0; H 2.1 2.5 0.3"


def molecule(atom, basis):
    mol = gto.M(atom=atom, basis=basis, unit="Bohr", verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    C = mf.mo_coeff
    n = C.shape[1]
    h = C.T @ mf.get_hcore() @ C
    eri = ao2mo.restore(1, ao2mo.kernel(mol, C), n)
    system = System(mol.nelectron, n, 1, 0, nuclear_repulsion=mol.energy_nuc())
    return mol, mf, h, eri, system


def determinant_hamiltonian(mol, h, eri):
    n = h.shape[0]
    space = DeterminantSpace(n, mol.nelectron // 2, mol.nelectron // 2)
    return space, hamiltonian_matrix(space, *spin_orbital_integrals(h, eri)) + mol.energy_nuc() * np.eye(space.ndet)


def t_matrix(space, orbs, T):
    """T1 + T2 of a reference as a dense determinant-space matrix."""
    ops, amps = [], []
    for key, idx_op in excitation_strings(orbs, space.norb).items():
        amp = getattr(T, key)
        for idx, op in idx_op:
            if key in ("aa", "bb") and not (idx[0] < idx[1] and idx[2] < idx[3]):
                continue
            ops.append(op)
            amps.append(amp[idx])
    return space.transitions_of(ops).matrix(np.array(amps), space.ndet).toarray()


def closed_shells(system, *occupations):
    """Closed-shell determinants doubly occupying the given (1-based) orbitals."""
    return [np.array(list(o) + list(o)) for o in occupations]


def test_single_reference_is_ccsd():
    mol, mf, h, eri, system = molecule("O 0 0 -0.124; H 0 1.43 0.98; H 0 -1.43 0.98", "6-31g")
    _, _, energy, _, _ = kernel(h, eri.transpose(0, 2, 1, 3), system, get_cas_determinants(system, 0, 0),
                                convergence=1e-11)
    mycc = cc.CCSD(mf)
    mycc.conv_tol, mycc.conv_tol_normt = 1e-12, 1e-10
    assert abs(energy[0].real - mycc.run().e_tot) < 1e-9


@pytest.mark.parametrize("drop", [None, 3])
def test_two_electrons_is_exact(drop):
    """CAS(2,2), complete or without one determinant: every root is an FCI eigenvalue."""
    mol, mf, h, eri, system = molecule("H 0 0 0; H 0 0 1.4", "6-31g**")
    dets = get_cas_determinants(system, 2, 2)
    if drop is not None:
        dets.pop(drop)
    _, _, energy, _, _ = kernel(h, eri.transpose(0, 2, 1, 3), system, dets, convergence=1e-11)
    _, Hmat = determinant_hamiltonian(mol, h, eri)
    fci = np.linalg.eigvalsh(Hmat)
    assert len(energy) == len(dets)
    for e in energy:
        assert abs(e.imag) < 1e-10
        assert np.abs(fci - e.real).min() < 1e-9


MODEL_SPACES = {
    "cas22": lambda system: get_cas_determinants(system, 2, 2),
    # CAS(4,3) has intruders: Jacobi stalls and the auto solver finishes with Newton-Krylov
    "cas43": lambda system: get_cas_determinants(system, 4, 3),
    # two closed shells: the internal double is -t1 t1 of external active singles
    "two_closed_shells": lambda system: closed_shells(system, (1, 2), (1, 3)),
    # quadruply excited pair: a rank-4 internal amplitude (C conditions) in Heff, coupling and
    # the direct term (V T4)
    "quadruple_pair": lambda system: closed_shells(system, (1, 2), (3, 4)),
    # alpha (1,2)->(3,4), beta 2->3: a rank-3 internal amplitude, entering the direct term (Hbar T3)
    "triple_pair": lambda system: [np.array([1, 2, 1, 2]), np.array([3, 4, 1, 3])],
}


@pytest.mark.parametrize("name, frozen_cv", [("cas22", False), ("cas22", True), ("cas43", False),
                                             ("two_closed_shells", False), ("two_closed_shells", True),
                                             ("quadruple_pair", False), ("triple_pair", False)])
def test_su_equations_in_determinant_space(name, frozen_cv):
    mol, mf, h, eri, system = molecule(H4, "6-31g")
    T, heff, _, _, ms = kernel(h, eri.transpose(0, 2, 1, 3), system, MODEL_SPACES[name](system),
                               frozen_cv=frozen_cv, convergence=1e-11)
    space, Hmat = determinant_hamiltonian(mol, h, eri)
    nref = len(ms.references)
    idx = [determinant(space, ref) for ref in ms.references]
    # exp(T(q))|Phi_q> and T(q) with the rank >= 3 internal amplitudes from the C conditions
    # (and checks that the converged internal t1/t2 satisfy them)
    psi, Tm = zip(*[cc_wavefunction(space, ms, q, T[q], return_t=True) for q in range(nref)])
    Tm = [t.toarray() for t in Tm]
    assert np.allclose(heff, [[Hmat[idx[p]] @ psi[q] for q in range(nref)] for p in range(nref)], atol=1e-10)

    core_virtual = get_core_virtual(system, ms)
    checked = 0
    for p in range(nref):
        # direct term <K|exp(-T(p)) H exp(T(p))|Phi_p> with the internal T3+ included
        U = expm(-Tm[p])
        R = U @ Hmat @ expm(Tm[p])[:, idx[p]] - sum(U @ psi[q] * heff[q, p] for q in range(nref) if q != p)
        for key, entries in excitation_strings(ms.references[p], space.norb).items():
            skip = ms.internal[p][key].copy()
            if frozen_cv:
                skip[core_virtual[key]] = True
            for ix, op in entries:
                _, d, _ = space.apply(op, src=np.array([idx[p]]))
                k, _ = space.lookup(d)
                if skip[ix]:
                    assert k[0] in idx or frozen_cv
                    continue
                assert k[0] not in idx
                assert abs(R[k[0]]) < 1e-8, (p, key, ix, R[k[0]])
                checked += 1
    assert checked > 0


def test_solvers_agree():
    mol, mf, h, eri, system = molecule(H4, "6-31g")
    energies = [kernel(h, eri.transpose(0, 2, 1, 3), system, get_cas_determinants(system, 2, 2), solver=solver,
                       convergence=1e-11)[2] for solver in ("jacobi", "newton_krylov")]
    assert np.allclose(*energies, atol=1e-9)


@pytest.mark.parametrize("cas, intruder", [((2, 2), False), ((4, 4), True)])
def test_intruder_warning(cas, intruder, capsys):
    mol, mf, h, eri, system = molecule(H4, "6-31g")
    estimates = get_external_estimates(h, eri.transpose(0, 2, 1, 3), system, get_cas_determinants(system, *cas))
    assert check_intruders(estimates) == intruder
    assert ("WARNING" in capsys.readouterr().out) == intruder


def test_external_estimates():
    """Reference energies, and the lowest single/double estimate equals the lowest valid
    Jacobi denominator (a < b, i < j for same-spin doubles) of each reference."""
    mol, mf, h, eri, system = molecule(H4, "6-31g")
    e2 = eri.transpose(0, 2, 1, 3)
    dets = get_cas_determinants(system, 4, 4)
    estimates = get_external_estimates(h, e2, system, dets)
    ms = get_model_space(system, dets)
    for p, ref in enumerate(ms.references):
        H, e_ref = get_reference_hamiltonian(h, e2, system, ref)
        assert abs(e_ref - estimates[p][0]) < 1e-12
        lowest = np.inf
        for key, d in get_denominators(H, 0.0).items():
            if key in ("aa", "bb"):
                nu, no = d.shape[0], d.shape[2]
                d = d[np.triu_indices(nu, 1)][:, np.triu_indices(no, 1)[0], np.triu_indices(no, 1)[1]]
            lowest = min(lowest, e_ref - d.max())
        assert abs(lowest - estimates[p][1][0]) < 1e-12
        assert estimates[p][3] not in estimates[p][2]


@pytest.mark.parametrize("cas, gap", [((2, 4), 0.0), ((4, 4), 0.3)])
def test_prune_intruders(cas, gap):
    """Complete CAS(2,4) and CAS(4,4) of the distorted H4 do not converge with Jacobi; after
    dropping the highest determinants (a general model space) they do."""
    mol, mf, h, eri, system = molecule(H4, "6-31g")
    e2 = eri.transpose(0, 2, 1, 3)
    dets = get_cas_determinants(system, *cas)
    with pytest.raises(RuntimeError):
        kernel(h, e2, system, dets, solver="jacobi", maxit=60)
    _, _, energy, _, ms = kernel(h, e2, system, dets, solver="jacobi", prune_intruders=gap)
    estimates = get_external_estimates(h, e2, system, dets)
    kept = {estimates[p][3] for p in range(len(dets))} & {get_external_estimates(h, e2, system, ms.determinants)[p][3]
                                                          for p in range(len(ms.determinants))}
    assert len(energy) == len(ms.determinants) < len(dets)
    # the dropped determinants are the highest in energy
    e_kept = [estimates[p][0] for p in range(len(dets)) if estimates[p][3] in kept]
    e_dropped = [estimates[p][0] for p in range(len(dets)) if estimates[p][3] not in kept]
    assert max(e_kept) <= min(e_dropped)
    gap_left = get_lowest_external(estimates, [p for p in range(len(dets)) if estimates[p][3] in kept]) - max(e_kept)
    assert gap_left >= gap


def test_active_t2_residual_matches_full():
    """SUMRCCSD(1) evaluates the DF doubles residual only on blocks with an active index; on
    those it must equal the full residual, and vanish on the core-virtual block."""
    from ccpy.cc import ccsd_chol
    from ccpy.models.integrals import Integral
    from ccpy.models.operators import ClusterOperator
    from ccpy.mrcc.sumrccsd import get_t2_residual

    mol, mf, h, eri, system = molecule("O 0 0 0; H 0 1.43 -1.1; H 0 -1.43 -1.1", "6-31g")
    ms = get_model_space(system, get_cas_determinants(system, 2, 2))
    H, _ = get_reference_hamiltonian(h, eri.transpose(0, 2, 1, 3), system, ms.references[1])
    rng = np.random.default_rng(7)
    T = ClusterOperator(system, 2)
    for key in ("a", "b", "aa", "ab", "bb"):
        x = 0.05 * rng.standard_normal(getattr(T, key).shape)
        if key in ("aa", "bb"):
            x = x - x.transpose(1, 0, 2, 3)
            x = x - x.transpose(0, 1, 3, 2)
        setattr(T, key, x)
    X = Integral.from_empty(system, 2, use_none=True)
    X.chol = Integral.from_empty(system, 1, use_none=True)
    X = ccsd_chol.get_t1_transformed(ccsd_chol.get_pre_intermediates(X, T, H, False), T, H, False)
    cv = get_core_virtual(system, ms)
    for key in ("aa", "ab", "bb"):
        full = get_t2_residual(T, X, H, key)
        R = get_t2_residual(T, X, H, key, cv)
        free = np.ones(R.shape, bool)
        free[cv[key]] = False
        assert np.abs(R[cv[key]]).max() == 0.0
        assert np.allclose(R[free], full[free], atol=1e-12)
