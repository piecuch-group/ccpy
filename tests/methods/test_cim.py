"""Cluster-in-molecule (CIM) CC: canonical limit, mixed-index partition, SECIM subsystems,
fixed domains, and the multilevel scheme."""
import contextlib
import io
from itertools import product

import numpy as np
import pytest
from conftest import AUX, OPTS, WATER, converged_scf

from ccpy import CCSD, CCSDT, CIMOptions, CRCC23, EOMCCSD, MP2, Molecule, RHF, UHF
from ccpy.methods.cim import CIMDomains, CIMSubsystem, boys_localize, lmo_populations

WATER_TRIMER = """O 0.000 1.470 0.110; H 0.000 1.470 1.070; H 0.900 1.770 -0.200
O 1.270 -0.735 0.110; H 1.270 -0.735 1.070; H 1.080 -1.680 -0.200
O -1.270 -0.735 0.110; H -1.270 -0.735 1.070; H -1.980 -0.090 -0.200"""


def quiet(f, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*args, **kwargs)


@pytest.fixture(scope="module")
def trimer_rhf():
    return converged_scf(RHF, Molecule(WATER_TRIMER, "6-31g", auxiliary_basis=AUX, verbose=False))


@pytest.mark.parametrize("method", [MP2, CCSD, CCSDT, CRCC23])
def test_single_subsystem_is_canonical(water_rhf, method):
    """zeta = 0 puts every orbital in one subsystem, with all LMOs central."""
    expected = quiet(lambda: method(water_rhf, nfrozen=1, **OPTS).energy())
    cim = quiet(lambda: method(water_rhf, nfrozen=1, cim=CIMOptions(zeta=0.0), **OPTS))
    assert quiet(cim.energy) == pytest.approx(expected, abs=1.0e-9)
    assert len(cim.domains.subsystems) == 1


def spin_orbital(H, name, system):
    """Dense spin-orbital block <pq||rs> (or <p|f|q>) of the type `name`, alpha before beta."""
    n = {"o": (system.noccupied_alpha, system.noccupied_beta), "v": (system.nunoccupied_alpha,
                                                                     system.nunoccupied_beta)}
    rng = {(c, s): slice(0, n[c][0]) if s == "a" else slice(n[c][0], sum(n[c])) for c in "ov" for s in "ab"}
    x = np.zeros([sum(n[c]) for c in name])
    if len(name) == 2:
        for s in "ab":
            x[rng[name[0], s], rng[name[1], s]] = getattr(getattr(H, s), name)
        return x
    for s in product("ab", repeat=4):
        if sorted(s[:2]) != sorted(s[2:]):
            continue
        idx = tuple(rng[c, spin] for c, spin in zip(name, s))
        if len(set(s)) == 1:
            x[idx] = getattr(H.aa if s[0] == "a" else H.bb, name)
            continue
        # <pq||rs> = <pq|rs> for (p, r) alpha; move alpha to the first bra/ket slots
        bra, ket = (0, 1) if s[0] == "a" else (1, 0), (2, 3) if s[2] == "a" else (3, 2)
        order = bra + ket
        block = getattr(H.ab, "".join(name[i] for i in order))
        sign = (-1) ** ((bra != (0, 1)) + (ket != (2, 3)))
        x[idx] = sign * block.transpose(np.argsort(order))
    return x


def central_oracle(cr, P):
    """sum_i'k' P_i'k' dE_k'(CCSD and CR-CC(2,3)_A) from dense spin-orbital M and l = N / D over
    all (unrestricted) occupied triples, Eqs. (11), (13) of J. Chem. Phys. 131, 114109."""
    system, hbar, L = cr.system, cr.hbar, cr.left.L
    T, H = hbar.T, hbar.H
    so = lambda name, op=H: spin_orbital(op, name, system)
    noa, nua = system.noccupied_alpha, system.nunoccupied_alpha
    o, v = slice(0, noa), slice(0, nua)
    t1 = np.zeros((2 * nua, 2 * noa)); t1[v, o] = T.a; t1[nua:, noa:] = T.b
    l1 = np.zeros_like(t1); l1[v, o] = L.a; l1[nua:, noa:] = L.b
    t2, l2 = np.zeros((2 * nua,) * 2 + (2 * noa,) * 2), np.zeros((2 * nua,) * 2 + (2 * noa,) * 2)
    for x, op in ((t2, T), (l2, L)):
        x[v, v, o, o] = op.aa; x[nua:, nua:, noa:, noa:] = op.bb
        x[v, nua:, o, noa:] = op.ab; x[nua:, v, noa:, o] = op.ab.transpose(1, 0, 3, 2)
        x[v, nua:, noa:, o] = -op.ab.transpose(0, 1, 3, 2); x[nua:, v, o, noa:] = -op.ab.transpose(1, 0, 2, 3)
    Pso = np.kron(np.eye(2), P)
    # CCSD: e_ki = f_ka t_ai + 1/4 <kj||ab> tau_abij
    v0 = so("oovv", cr.reference.H)
    tau = t2 + np.einsum("ai,bj->abij", t1, t1) - np.einsum("aj,bi->abij", t1, t1)
    e = so("ov", cr.reference.H) @ t1 + 0.25 * np.einsum("kjab,abij->ki", v0, tau)
    energy = np.sum(Pso * e)
    # triples: M_ijk^abc and N_ijk^abc for all i, j, k (antisymmetrized), l = N / D_A
    h_ov, I_vvov = so("ov"), so("vvov") + np.einsum("me,abim->abie", so("ov"), t2)
    A = lambda x, axes: sum(np.sign(p) * x.transpose(perm) for p, perm in axes)
    perms3 = [(1, (0, 1, 2)), (1, (1, 2, 0)), (1, (2, 0, 1)), (-1, (0, 2, 1)), (-1, (1, 0, 2)), (-1, (2, 1, 0))]

    def antisym(x):  # A over (abc) and (ijk) of x[a,b,c,i,j,k]
        y = sum(s * x.transpose(p + (3, 4, 5)) for s, p in perms3)
        return sum(s * y.transpose((0, 1, 2) + tuple(3 + q for q in p)) for s, p in perms3)

    M = antisym(-np.einsum("amij,bcmk->abcijk", so("vooo"), t2) + np.einsum("abie,ecjk->abcijk", I_vvov, t2)) / 4
    N = antisym(np.einsum("eiba,ecjk->abcijk", so("vovv"), l2) - np.einsum("jima,bcmk->abcijk", so("ooov"), l2)
                + np.einsum("ck,ijab->abcijk", l1, v0) + np.einsum("kc,abij->abcijk", h_ov, l2)) / 4
    fo, fv = np.diag(so("oo", cr.reference.H)), np.diag(so("vv", cr.reference.H))
    D = (fo[None, None, None, :, None, None] + fo[None, None, None, None, :, None] + fo[None, None, None, None, None, :]
         - fv[:, None, None, None, None, None] - fv[None, :, None, None, None, None] - fv[None, None, :, None, None, None])
    full = np.sum(M * N / D) / 36
    assert full == pytest.approx(cr.deltas["A"], abs=1.0e-12)  # the oracle is CR-CC(2,3)_A
    return energy + np.einsum("ik,abckjl,abcijl->", Pso, N / D, M) / 36


def test_mixed_index_partition_matches_dense_oracle(water_rhf):
    """Two subsystems with every orbital, central LMOs (0, 1) and (2, 3): each subsystem energy
    is the LMO-resolved canonical energy of its central orbitals."""
    mol = water_rhf.mol
    canonical = quiet(lambda: CRCC23(water_rhf, nfrozen=1, variant="A", **OPTS).ensure_solved())
    C = water_rhf.mo_coeff[:, 1:5]
    U = boys_localize(C, mol)
    subsystems = tuple(CIMSubsystem(central, central + tuple(i for i in range(4) if i not in central),
                                    (0, 1, 2), CRCC23) for central in ((0, 1), (2, 3)))
    cim = CRCC23(water_rhf, nfrozen=1, variant="A", cim=CIMOptions(domains=CIMDomains(
        subsystems, lmo_populations(C @ U, mol))), **OPTS)
    total = quiet(cim.energy)
    assert total == pytest.approx(quiet(canonical.energy), abs=1.0e-9)
    for s, result in zip(subsystems, cim.subsystems):
        P = U[:, list(s.central)] @ U[:, list(s.central)].T   # canonical MO basis
        assert result["energy"] == pytest.approx(central_oracle(canonical, P), abs=1.0e-9)


def test_adaptive_ccpq_canonical_limit_and_partition(water_rhf):
    """Adaptive CC(P;Q): one subsystem with every orbital selects the canonical P spaces, and two
    subsystems with every orbital and central LMOs (0, 1) and (2, 3) add up to the canonical
    energy, so the central CC(P;3) parts (triples sum skipping the P space) partition it."""
    from ccpy.methods.corrections import AdaptiveCCPQ

    options = dict(nfrozen=1, percentage=[0.0, 2.0, 5.0], energy_tolerance=1.0e-10, **OPTS)
    canonical = AdaptiveCCPQ(water_rhf, **options)
    expected = quiet(canonical.energy)
    assert len(canonical.ccpq_energies) == 3
    single = AdaptiveCCPQ(water_rhf, cim=CIMOptions(zeta=0.0), **options)
    assert quiet(single.energy) == pytest.approx(expected, abs=1.0e-9)
    C = water_rhf.mo_coeff[:, 1:5]
    U = boys_localize(C, water_rhf.mol)
    subsystems = tuple(CIMSubsystem(central, central + tuple(i for i in range(4) if i not in central),
                                    (0, 1, 2), AdaptiveCCPQ) for central in ((0, 1), (2, 3)))
    split = AdaptiveCCPQ(water_rhf, cim=CIMOptions(domains=CIMDomains(subsystems, lmo_populations(C @ U, water_rhf.mol))),
                         **options)
    assert quiet(split.energy) == pytest.approx(expected, abs=1.0e-9)


def test_secim_water_trimer(trimer_rhf):
    """One subsystem per water (its four LMOs central, a neighbor as environment), and 99.5% of
    the canonical CCSD correlation energy at zeta = 0.01."""
    expected = quiet(lambda: CCSD(trimer_rhf, nfrozen=3, **OPTS).energy()) - trimer_rhf.e_tot
    cim = CCSD(trimer_rhf, nfrozen=3, cim=CIMOptions(zeta=0.01), **OPTS)
    correlation = quiet(cim.energy) - trimer_rhf.e_tot
    assert sorted(i for s in cim.domains.subsystems for i in s.central) == list(range(12))
    assert [(len(s.central), len(s.occupied), len(s.atoms)) for s in cim.domains.subsystems] == [(4, 8, 6)] * 3
    assert correlation / expected == pytest.approx(0.99508, abs=1.0e-4)
    # tighter zeta: one subsystem
    assert len(quiet(lambda: CCSD(trimer_rhf, nfrozen=3, cim=CIMOptions(zeta=0.003), **OPTS).ensure_solved())
               .domains.subsystems) == 1


def test_local_subsystems_decane():
    """Subsystem orbitals and integrals are built from the AOs of each extended domain only:
    with a 2.5 angstrom buffer the outer subsystems exclude atoms of n-decane, and CIM-MP2 stays
    within 0.3 mEh of the CIM-MP2 whose extended domains are the whole molecule. The parent
    reference holds no MO integrals."""
    xyz = []
    for k in range(10):
        x, y, s = 1.27 * k, (0.42, -0.42)[k % 2 == 0], (1, -1)[k % 2 == 0]
        xyz += [f"C {x} {y} 0", f"H {x} {y + 0.63 * s} 0.89", f"H {x} {y + 0.63 * s} -0.89"]
    xyz += ["H -1.03 -0.10 0", "H 12.46 0.10 0"]
    mf = converged_scf(RHF, Molecule("; ".join(xyz), "6-31g", auxiliary_basis=AUX, verbose=False))
    full = MP2(mf, nfrozen=10, cim=CIMOptions(zeta=0.003), **OPTS)
    local = MP2(mf, nfrozen=10, cim=CIMOptions(zeta=0.003, buffer_radius=2.5), **OPTS)
    assert quiet(local.energy) == pytest.approx(quiet(full.energy), abs=3.0e-4)
    assert all(len(s.extended) == 32 for s in full.domains.subsystems)
    assert min(len(s.extended) for s in local.domains.subsystems) < 32
    assert not hasattr(local.reference, "B_mo")


def test_buffer_too_small(trimer_rhf):
    """Without buffer atoms the lone pairs of the excluded water leak into the unoccupied space,
    which is detected instead of producing a meaningless energy."""
    with pytest.raises(ValueError, match="buffer_radius"):
        quiet(CCSD(trimer_rhf, nfrozen=3, cim=CIMOptions(zeta=0.01, buffer_radius=0.0), **OPTS).energy)


def test_fixed_domains_follow_the_lmos(trimer_rhf):
    """at_geometry reuses the subsystems, matching the LMOs by their populations; for a small
    displacement they are the subsystems the displaced geometry would choose itself."""
    cim = CCSD(trimer_rhf, nfrozen=3, cim=CIMOptions(zeta=0.01), **OPTS)
    quiet(cim.energy)
    mol = trimer_rhf.mol
    xyz = mol.atomic_positions.copy()
    xyz[1, 2] += 0.02
    fixed = cim.at_geometry(mol.with_geometry(xyz))
    adaptive = CCSD(converged_scf(RHF, mol.with_geometry(xyz)), nfrozen=3, cim=CIMOptions(zeta=0.01), **OPTS)
    assert quiet(fixed.energy) == pytest.approx(quiet(adaptive.energy), abs=1.0e-9)
    assert fixed.domains is cim.domains


def test_multilevel(trimer_rhf):
    """CCSD for the subsystem centered on the first water and MP2 for the others; levels that
    cover every atom give single-level CIM."""
    options = dict(nfrozen=3, **OPTS)
    high = quiet(lambda: CCSD(trimer_rhf, cim=CIMOptions(zeta=0.01), **options).energy())
    low = quiet(lambda: MP2(trimer_rhf, cim=CIMOptions(zeta=0.01), **options).energy())
    everything = quiet(lambda: MP2(trimer_rhf, cim=CIMOptions(zeta=0.01, levels={CCSD: range(9)}),
                                   **options).energy())
    assert everything == pytest.approx(high, abs=1.0e-9)
    mixed = MP2(trimer_rhf, cim=CIMOptions(zeta=0.01, levels={CCSD: [0]}), **options)
    energy = quiet(mixed.energy)
    assert [s["method"] for s in mixed.subsystems] == ["CCSD", "MP2", "MP2"]
    assert min(high, low) < energy < max(high, low)


def test_unsupported(water_rhf):
    with pytest.raises(NotImplementedError):
        quiet(EOMCCSD(water_rhf, nfrozen=1, cim=CIMOptions()).energy)
    uhf = converged_scf(UHF, Molecule(WATER, "6-31g", auxiliary_basis=AUX, verbose=False))
    with pytest.raises(NotImplementedError):
        quiet(CCSD(uhf, nfrozen=1, cim=CIMOptions()).energy)


def test_development_warning(water_rhf):
    """CIM is under development: every CIM calculation says so."""
    from ccpy.methods.cim import CIMDevelopmentWarning

    with pytest.warns(CIMDevelopmentWarning):
        MP2(water_rhf, nfrozen=1, cim=CIMOptions())
