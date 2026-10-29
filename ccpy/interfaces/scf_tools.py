"""Build ccpy System and Integral objects from the native density-fitted SCF solvers
(ccpy.scf.RHF/ROHF/UHF), without PySCF.

The two-electron integrals are the density-fitted ones used in the SCF,

    <pq|rs> = (pr|qs) = \\sum_P B^P_{pr} B^P_{qs},

so the reference energy recomputed from the MO integrals equals the SCF energy to
numerical precision. With ``use_cholesky=True`` the MO-basis B tensor is passed to the
``*_chol`` methods in place of Cholesky vectors and the four-index tensor is never
formed.
"""

import numpy as np

from ccpy.energy.hf_energy import (
    calc_hf_energy,
    calc_hf_energy_chol,
    calc_hf_frozen_core_energy,
)
from ccpy.models.integrals import (
    Integral,
    build_f,
    getCholeskyHamiltonian,
    getHamiltonian,
)
from ccpy.models.system import System, UHFSystem


def _check_scf(meanfield):
    from ccpy.scf.scf_base import SCFBase

    if not isinstance(meanfield, SCFBase):
        raise TypeError("Expected a ccpy.scf RHF, ROHF, or UHF object.")
    if meanfield.e_tot is None:
        meanfield.run()
    if not meanfield.converged:
        print("   WARNING: building integrals from an unconverged SCF solution.")


def load_scf_integrals(
    meanfield,
    nfrozen=0,
    ndelete=0,
    use_cholesky=False,
    normal_ordered=True,
    sorted=True,
):
    """System and Hamiltonian for an RHF or ROHF reference from ``ccpy.scf``.

    Parameters
    ----------
    meanfield : ccpy.scf.RHF | ccpy.scf.ROHF
    nfrozen : int
        Number of frozen (core) spatial orbitals.
    ndelete : int
        Number of deleted (highest) virtual orbitals.
    use_cholesky : bool
        Keep the integrals in factorized (DF) form for the ``*_chol`` methods.
    """
    from ccpy.scf.uhf import UHF

    _check_scf(meanfield)
    if isinstance(meanfield, UHF):
        raise TypeError("Use load_scf_uhf_integrals (or Driver.from_scf) for UHF.")

    mol = meanfield.mol
    mo_coeff = meanfield.mo_coeff
    norbitals = mo_coeff.shape[1]
    nuclear_repulsion = mol.nuclear_repulsion

    system = System(
        mol.nelectrons,
        norbitals,
        mol.multiplicity,
        nfrozen,
        ndelete=ndelete,
        point_group=mol.point_group,
        orbital_symmetries=meanfield.orbital_symmetries(),
        charge=mol.charge,
        nuclear_repulsion=nuclear_repulsion,
        mo_energies=meanfield.mo_energy,
        mo_occupation=meanfield.mo_occ,
    )

    e1int = np.asfortranarray(mo_coeff.T @ meanfield.get_hcore() @ mo_coeff)
    R_mo = mol.df.mo_cholesky_vectors(mo_coeff)

    if use_cholesky:
        hf_energy = calc_hf_energy_chol(e1int, R_mo, system) + nuclear_repulsion
        _check_energy(hf_energy, meanfield)
        system.reference_energy = hf_energy
        system.frozen_energy = _frozen_core_energy_chol(e1int, R_mo, nfrozen)
        system.cholesky = True
        return system, getCholeskyHamiltonian(e1int, R_mo, system, normal_ordered, sorted)

    # <pq|rs> = (pr|qs)
    e2int = np.asfortranarray(np.einsum("xpr,xqs->pqrs", R_mo, R_mo, optimize=True))
    del R_mo
    hf_energy = calc_hf_energy(e1int, e2int, system) + nuclear_repulsion
    _check_energy(hf_energy, meanfield)
    system.reference_energy = hf_energy
    system.frozen_energy = calc_hf_frozen_core_energy(e1int, e2int, system)
    return system, getHamiltonian(e1int, e2int, system, normal_ordered, sorted)


def load_scf_uhf_integrals(
    meanfield,
    nfrozen=0,
    ndelete=0,
    normal_ordered=True,
    sorted=True,
):
    """UHFSystem and Hamiltonian for a UHF reference from ``ccpy.scf``."""
    _check_scf(meanfield)
    mol = meanfield.mol
    coeff_a, coeff_b = meanfield.mo_coeff
    norbitals = coeff_a.shape[1]

    system = UHFSystem(
        mol.nelectrons,
        norbitals,
        mol.multiplicity,
        nfrozen,
        ndelete=ndelete,
        point_group=mol.point_group,
        orbital_symmetries={
            "a": meanfield.orbital_symmetries(0),
            "b": meanfield.orbital_symmetries(1),
        },
        charge=mol.charge,
        nuclear_repulsion=mol.nuclear_repulsion,
        mo_energies={"a": meanfield.mo_energy[0], "b": meanfield.mo_energy[1]},
        mo_occupation={"a": meanfield.mo_occ[0], "b": meanfield.mo_occ[1]},
    )
    system.reference_energy = meanfield.e_tot
    return system, getUHFHamiltonianDF(
        coeff_a, coeff_b, meanfield.get_hcore(), mol.df, system, normal_ordered, sorted
    )


def getUHFHamiltonianDF(coeff_a, coeff_b, h_ao, df, system, normal_ordered, sorted=True):
    """Spin-orbital Hamiltonian of a UHF reference from density-fitted integrals
    (DF counterpart of ccpy.models.integrals.getUHFHamiltonian)."""
    corr_slice = slice(system.nfrozen, system.nfrozen + system.norbitals)
    z = {"a": coeff_a.T @ h_ao @ coeff_a, "b": coeff_b.T @ h_ao @ coeff_b}
    Ra = df.mo_cholesky_vectors(coeff_a)
    Rb = df.mo_cholesky_vectors(coeff_b)
    v_aa = np.einsum("xpr,xqs->pqrs", Ra, Ra, optimize=True)
    v_aa -= np.transpose(v_aa, (0, 1, 3, 2))
    v_ab = np.einsum("xpr,xqs->pqrs", Ra, Rb, optimize=True)
    v_bb = np.einsum("xpr,xqs->pqrs", Rb, Rb, optimize=True)
    v_bb -= np.transpose(v_bb, (0, 1, 3, 2))
    v = {"aa": v_aa, "ab": v_ab, "bb": v_bb}
    fock = build_f(z, v, system) if normal_ordered else z
    onebody = {s: fock[s][corr_slice, corr_slice] for s in ("a", "b")}
    twobody = {s: v[s][corr_slice, corr_slice, corr_slice, corr_slice] for s in ("aa", "ab", "bb")}
    return Integral(system, 2, {**onebody, **twobody}, sorted=sorted)


def get_multipole_integral(l, mf, system):
    """Multipole integrals (l = 1 dipole, 2 quadrupole, 3 octupole) in the correlated MO
    basis, with the reference expectation value, in the format of
    ccpy.interfaces.pyscf_tools.get_multipole_integral. The origin is the coordinate
    origin, as for PySCF's int1e_r/rr/rrr.
    """
    from ccpy.integrals import integrals

    mol = mf.mol
    C = mf.mo_coeff
    noa, nob = system.noccupied_alpha, system.noccupied_beta
    nua, nub = system.nunoccupied_alpha, system.nunoccupied_beta
    corr_slice = slice(system.nfrozen, system.nfrozen + system.norbitals)
    oa, ob = slice(noa), slice(nob)
    va, vb = slice(noa, noa + nua), slice(nob, nob + nub)
    occ_a = slice(noa + system.nfrozen)
    occ_b = slice(nob + system.nfrozen)

    print(f"   Computing L = {l} multipole integrals with libint2")
    if l == 1:
        ao = np.array(integrals.emultipole1(mol)[1:4])
        shape = (3,)
    elif l == 2:
        # libint2 order: xx, xy, xz, yy, yz, zz
        comp = integrals.emultipole2(mol)[4:10]
        idx = {(0, 0): 0, (0, 1): 1, (0, 2): 2, (1, 1): 3, (1, 2): 4, (2, 2): 5}
        ao = np.array([comp[idx[tuple(sorted((x, y)))]] for x in range(3) for y in range(3)])
        shape = (3, 3)
    elif l == 3:
        # libint2 order: xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz
        comp = integrals.emultipole3(mol)[10:20]
        keys = [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 1), (0, 1, 2), (0, 2, 2),
                (1, 1, 1), (1, 1, 2), (1, 2, 2), (2, 2, 2)]
        idx = {k: i for i, k in enumerate(keys)}
        ao = np.array(
            [comp[idx[tuple(sorted((x, y, z)))]] for x in range(3) for y in range(3) for z in range(3)]
        )
        shape = (3, 3, 3)
    else:
        raise NotImplementedError(f"Multipole integrals with L = {l} are not available.")
    # libint2 multipoles are <m|r|n> about the origin, the same sign as PySCF's int1e_r
    mo =np.einsum("xij,ip,jq->xpq", ao, C, C, optimize=True)
    Q_ref = (np.einsum("xii->x", mo[:, occ_a, occ_a]) + np.einsum("xii->x", mo[:, occ_b, occ_b])).reshape(shape)
    mo = mo[:, corr_slice, corr_slice]

    mu = np.empty(int(np.prod(shape)), dtype=object)
    for k in range(mu.size):
        m = Integral.from_empty(system, order=1, data_type=np.float64, use_none=True)
        m.a.oo, m.a.vv, m.a.ov, m.a.vo = mo[k, oa, oa], mo[k, va, va], mo[k, oa, va], mo[k, va, oa]
        m.b.oo, m.b.vv, m.b.ov, m.b.vo = mo[k, ob, ob], mo[k, vb, vb], mo[k, ob, vb], mo[k, vb, ob]
        mu[k] = m
    return mu.reshape(shape).tolist(), Q_ref


def _frozen_core_energy_chol(e1int, R, nfrozen):
    if nfrozen == 0:
        return 0.0
    c = slice(0, nfrozen)
    J = np.einsum("xii->x", R[:, c, c])
    return 2.0 * np.trace(e1int[c, c]) + 2.0 * J @ J - np.einsum("xij,xji->", R[:, c, c], R[:, c, c])


def _check_energy(hf_energy, meanfield):
    if not np.allclose(hf_energy, meanfield.e_tot, atol=1.0e-06, rtol=0.0):
        raise RuntimeError(
            f"Integrals don't match the mean-field energy ({hf_energy:.10f} vs {meanfield.e_tot:.10f})"
        )
