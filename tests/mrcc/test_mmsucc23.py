"""MM-SUCC(2,3) corrections: CCSD(T) pieces for one reference, and the full correction
delta_mu = sum_p c_pmu <Psi_mu|Gamma_3(p)|Phi_p> / <Psi_mu|Psi_mu^SU> evaluated by brute force in
the determinant space (Gamma_3 from the SU-MRCC residual on the external triples of each
reference, Psi_mu as a vector), for complete and general model spaces."""
import numpy as np
import pytest
from scipy.linalg import expm

pyscf = pytest.importorskip("pyscf")

from ccpy.moments.crcc23 import calc_ccsdpt
from ccpy.mrcc.mmsucc23 import mmsucc23, mmsucc23_dense, get_z3, T3_WEIGHTS, _dot
from ccpy.mrcc.sumrcc import get_cas_determinants, get_reference_hamiltonian, materialize
from ccpy.mrcc.sumrccsd import kernel
from test_sumrcc_projection import excitation_strings, determinant, cc_wavefunction
from test_sumrccsd import molecule, determinant_hamiltonian, t_matrix, closed_shells, H4, MODEL_SPACES


def test_single_reference_ccsdpt():
    """Sum over triples of Z3 (V_N T2)_C is the CCSD(T) correction."""
    mol, mf, h, eri, system = molecule("O 0 0 -0.124; H 0 1.43 0.98; H 0 -1.43 0.98", "6-31g")
    e2 = eri.transpose(0, 2, 1, 3)
    T, _, energy, _, ms = kernel(h, e2, system, get_cas_determinants(system, 0, 0), convergence=1e-11)
    H, e_ref = get_reference_hamiltonian(h, e2, system, ms.references[0])
    z3, W = get_z3(T[0], materialize(H), system)
    _, correction = calc_ccsdpt(T[0], energy[0].real - e_ref, materialize(H), system)
    assert abs(_dot(z3, W, T3_WEIGHTS) - correction["A"]) < 1e-10


def triples_strings(orbs, norb):
    """Operator strings of the unique triples of a reference, with their array index."""
    occ_a, unocc_a, occ_b, unocc_b = orbs
    spaces = {"a": (occ_a, unocc_a, 0), "b": (occ_b, unocc_b, norb)}
    out = {}
    for key in ("aaa", "aab", "abb", "bbb"):
        o = [spaces[s] for s in key]
        entries = []
        for i in range(len(o[0][0])):
            for j in range(len(o[1][0])):
                for k in range(len(o[2][0])):
                    for a in range(len(o[0][1])):
                        for b in range(len(o[1][1])):
                            for c in range(len(o[2][1])):
                                if key[0] == key[1] and not (a < b and i < j):
                                    continue
                                if key[1] == key[2] and not (b < c and j < k):
                                    continue
                                holes = [o[0][0][i] + o[0][2], o[1][0][j] + o[1][2], o[2][0][k] + o[2][2]]
                                parts = [o[0][1][a] + o[0][2], o[1][1][b] + o[1][2], o[2][1][c] + o[2][2]]
                                ops = [("+", x) for x in parts] + [("-", x) for x in reversed(holes)]
                                entries.append(((a, b, c, i, j, k), ops))
        out[key] = entries
    return out


def brute_force_corrections(space, Hmat, T, heff, coefficients, ms, z3, lin12):
    nref = len(ms.references)
    idx = [determinant(space, r) for r in ms.references]
    psi, Tm = zip(*[cc_wavefunction(space, ms, q, T[q], return_t=True) for q in range(nref)])
    Tm = [t.toarray() for t in Tm]
    # trial vectors (1 + T1 + T2 + Z3)(q)|Phi_q>
    lin = []
    for q in range(nref):
        v = np.zeros(space.ndet)
        v[idx[q]] = 1.0

        class Lin:  # T1 + T2 without internal excitations
            pass
        L = Lin()
        for k, x in lin12[q].items():
            setattr(L, k, x)
        v += t_matrix(space, ms.references[q], L)[:, idx[q]]
        ops, amps = [], []
        for key, entries in triples_strings(ms.references[q], space.norb).items():
            for ix, op in entries:
                ops.append(op)
                amps.append(z3[q][key][ix])
        v += space.transitions_of(ops).matrix(np.array(amps), space.ndet) @ np.eye(space.ndet)[:, idx[q]]
        lin.append(v)
    # external triples of each reference
    triples = []
    for p in range(nref):
        dets = set()
        for entries in triples_strings(ms.references[p], space.norb).values():
            for _, op in entries:
                _, d, _ = space.apply(op, src=np.array([idx[p]]))
                if len(d):
                    k, _ = space.lookup(d)
                    if k[0] not in idx:
                        dets.add(k[0])
        triples.append(np.array(sorted(dets)))
    out = []
    for mu in range(coefficients.shape[1]):
        c = np.real(coefficients[:, mu])
        Psi = sum(c[q] * lin[q] for q in range(nref))
        numerator = 0.0
        for p in range(nref):
            U = expm(-Tm[p])
            R = U @ Hmat @ expm(Tm[p])[:, idx[p]] - sum(U @ psi[q] * heff[q, p] for q in range(nref) if q != p)
            numerator += c[p] * Psi[triples[p]] @ R[triples[p]]
        out.append(numerator / (Psi @ sum(c[p] * psi[p] for p in range(nref))))
    return np.array(out)


@pytest.mark.parametrize("name", ["cas22", "two_closed_shells", "triple_pair", "quadruple_pair"])
def test_corrections_match_determinant_space(name):
    import ccpy.mrcc.mmsucc23 as mm
    mol, mf, h, eri, system = molecule(H4, "6-31g")
    e2 = eri.transpose(0, 2, 1, 3)
    T, heff, roots, coefficients, ms = kernel(h, e2, system, MODEL_SPACES[name](system), convergence=1e-11)
    H = [get_reference_hamiltonian(h, e2, system, r)[0] for r in ms.references]
    energies, corrections, _ = mmsucc23_dense(T, H, heff, roots, coefficients, ms, system)
    # the on-the-fly implementation (no stored triples) agrees with the dense one
    assert np.allclose(mmsucc23(T, H, heff, roots, coefficients, ms, system)[1], corrections, atol=1e-12)
    # the same trial amplitudes, for the brute force
    z3 = []
    for p in range(len(ms.references)):
        z, _ = get_z3(T[p], materialize(H[p]), system)
        mask = mm._internal_triples_mask(ms, p, system)
        z3.append({k: np.where(mask[k], 0.0, z[k]) for k in z})
    lin12 = [{k: np.where(ms.internal[p][k], 0.0, getattr(T[p], k)) for k in ("a", "b", "aa", "ab", "bb")}
             for p in range(len(ms.references))]
    space, Hmat = determinant_hamiltonian(mol, h, eri)
    ref = brute_force_corrections(space, Hmat, T, heff, coefficients, ms, z3, lin12)
    assert np.allclose(corrections, ref, atol=1e-9), (corrections, ref)
    assert np.all(np.abs(corrections) > 1e-6)
