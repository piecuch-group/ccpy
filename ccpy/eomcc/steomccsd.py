"""Closed-shell spin-integrated STEOM-CCSD (Nooijen & Bartlett, JCP 107, 6812).

Spatial two-electron tensors use the CCpy mixed-spin ``ab`` (1212) convention.
Only the G2 blocks needed for a full singles diagonalization are constructed;
S1 is deliberately omitted, as allowed by Eqs. (31)-(32). Active labels are
compact: ``ip[m,i,j,b]`` and ``ea[e,a,b,j]``. All other labels span the full
correlated occupied/virtual spaces, including inactive orbitals.
"""

from types import SimpleNamespace

import numpy as np
from scipy.linalg import qr


def _einsum(subscripts, *args):
    return np.einsum(subscripts, *args, optimize=True)


def singles_character(R):
    """Determinant-metric 1h/1p weight, with the same-spin 1/2 factor."""
    one = np.vdot(R.a, R.a).real
    return one / (one + 0.5 * np.vdot(R.aa, R.aa).real + np.vdot(R.ab, R.ab).real)


def amplitudes_from_eom(vectors, active, sector, threshold=0.0, condition_limit=1.0e10):
    """Intermediate-normalize selected IP/EA roots (Eqs. 29-30).

    Roots below the singles threshold are discarded. If necessary, pivoted QR
    chooses a square subset of the original model orbitals for the retained
    roots. A singular model overlap is an error, never silently pseudoinverted.
    Returns (S2, remaining model labels, retained root indices).
    """
    if sector not in ("ip", "ea"):
        raise ValueError("sector must be 'ip' or 'ea'")
    active = np.asarray(active, dtype=int)
    if len(vectors) != len(active):
        raise ValueError("One auxiliary root is required per active orbital")
    keep = [k for k, R in enumerate(vectors) if singles_character(R) >= threshold]
    if not keep:
        raise ValueError(
            f"No {sector.upper()} roots pass the singles-character threshold"
        )
    r1 = np.column_stack([vectors[k].a for k in keep])
    if len(keep) < len(active):
        _, _, pivot = qr(r1[active].T, pivoting=True, mode="economic")
        active = active[pivot[: len(keep)]]
    model = r1[active]
    condition = np.linalg.cond(model)
    if not np.isfinite(condition) or condition > condition_limit:
        raise ValueError(
            f"Ill-conditioned {sector.upper()} model overlap (condition={condition:.3e})"
        )
    # Solve on the root index; no explicit inverse or root-by-root normalization.
    doubles = np.stack([vectors[k].ab for k in keep], axis=-1)
    normalized = np.linalg.solve(model.T, doubles.reshape(-1, len(keep)).T)
    normalized = normalized.reshape((len(keep),) + doubles.shape[:-1])
    if sector == "ip":
        normalized = -normalized.transpose(0, 1, 3, 2)  # (m,i,b,j) -> (m,i,j,b)
    return normalized, active, keep


def build_g(H, ip, ea, occupied, virtual):
    """Build spatial G.oo, G.vv, G.vovo, G.voov using Eqs. 33-63.

    The hhhp intermediates (42-44) are required by the quadratic terms
    in (58) and (62), which include the implicit triple-excitation effects.
    H is read-only. Both active and inactive singles are kept in G.
    """
    E = _einsum
    w = H.ab
    ov = H.a.ov
    tip = 2 * ip - ip.swapaxes(1, 2)
    tea = 2 * ea - ea.swapaxes(1, 2)
    # (34), (36), (38), (39)
    umi = E("kc,mikc->mi", ov, tip) - E("klid,mkld->mi", w.ooov, tip)
    uae = E("kc,eack->ae", ov, tea) + E("alcd,ecdl->ae", w.vovv, tea)
    uma = -E("klad,mkld->ma", w.oovv, tip)
    uie = E("ilcd,ecdl->ie", w.oovv, tea)
    goo, gvv = H.a.oo.copy(), H.a.vv.copy()
    goo[occupied, :] += umi
    gvv[:, virtual] += uae
    # (42)-(44). Pair interchange supplies u_lmdi and u_klei.
    umlid = E("jlbd,mijb->mlid", w.oovv, tip) - E("ljbd,mijb->mlid", w.oovv, ip)
    ukmid = -E("jkdb,mjib->kmid", w.oovv, ip)
    # Eq. (44) uses s_ie^ab = s_ei^ba, not s_ei^ab.
    uklie = E("klab,ebai->klie", w.oovv, ea)
    # phph: (56)-(59)
    uamci = (
        -E("kc,mika->amci", ov, ip)
        + E("alcd,mild->amci", w.vovv, tip)
        - E("aldc,mild->amci", w.vovv, ip)
        + E("klci,mlka->amci", w.oovo, ip)
    )
    # The ov*S(EA) term has a positive sign in CCpy's 1212 convention.
    # This differs from the printed minus in Eq. (57); the independent
    # normal-ordered determinant oracle checks the sign with nonzero H.ov.
    uakei = (
        E("kc,eaci->akei", ov, ea)
        + E("lkdi,eadl->akei", w.oovo, tea)
        - E("lkid,eadl->akei", w.ooov, ea)
        + E("akcd,ecdi->akei", w.vovv, ea)
    )
    uamei = (
        E("mc,eaci->amei", uma, ea)
        - E("ke,mika->amei", uie, ip)
        + E("mlid,eadl->amei", umlid, tea)
        - E("lmid,eadl->amei", ukmid, ea)
        + E("lkie,mlka->amei", uklie, ip)
    )
    gvovo = w.vovo.copy()
    gvovo[:, occupied, :, :] += uamci
    gvovo[:, :, virtual, :] += uakei
    for m, k in enumerate(occupied):
        gvovo[:, k, virtual, :] += uamei[:, m, :, :]
    # phhp: (60)-(63)
    ubmjc = (
        -E("kc,mkjb->bmjc", ov, ip)
        + E("klcj,mklb->bmjc", w.oovo, ip)
        - E("kbcd,mkjd->bmjc", w.ovvv, ip)
    )
    ubkje = (
        E("kd,edbj->bkje", ov, ea)
        + E("bkdc,ecdj->bkje", w.vovv, ea)
        - E("lkjd,edbl->bkje", w.ooov, ea)
    )
    ubmje = (
        E("md,edbj->bmje", uma, ea)
        - E("ke,mkjb->bmje", uie, ip)
        + E("lkje,mklb->bmje", uklie, ip)
        - E("lmjd,edbl->bmje", ukmid, ea)
    )
    gvoov = w.voov.copy()
    gvoov[:, occupied, :, :] += ubmjc
    gvoov[:, :, :, virtual] += ubkje
    for m, k in enumerate(occupied):
        gvoov[:, k, :, virtual] += ubmje[:, m, :, :].transpose(2, 0, 1)
    return SimpleNamespace(oo=goo, vv=gvv, vovo=gvovo, voov=gvoov)


def singles_matrix(G, multiplicity=1):
    """Spin-adapted, non-Hermitian singles matrix in flattened (a,i) order."""
    if multiplicity not in (1, 3):
        raise ValueError("Closed-shell STEOM supports multiplicity 1 or 3")
    nv, no = G.voov.shape[:2]
    matrix = -G.vovo.transpose(0, 3, 2, 1).copy()  # a i b j
    if multiplicity == 1:
        matrix += 2 * G.voov.transpose(0, 2, 3, 1)
    matrix += _einsum("ab,ij->aibj", G.vv, np.eye(no))
    matrix -= _einsum("ji,ab->aibj", G.oo, np.eye(nv))
    return matrix.reshape(nv * no, nv * no)


def diagonalize_singles(matrix, tolerance=1.0e-7):
    """General eigensolve, resolving numerically split real degenerate pairs.

    A real, non-Hermitian matrix may split a degenerate real eigenvalue into
    a conjugate pair with tiny imaginary parts. Real/imaginary eigenvector
    components span the associated real invariant subspace. Use an orthonormal
    real basis there, subject to the same residual tolerance as other roots.
    Genuinely complex pairs remain complex for the caller to diagnose.
    """
    omega, vectors = np.linalg.eig(matrix)
    for k in range(len(omega)):
        if 0 < omega[k].imag <= tolerance:
            candidates = np.flatnonzero(omega.imag < 0)
            if not len(candidates):
                continue
            partner = candidates[
                np.argmin(np.abs(omega[candidates] - omega[k].conjugate()))
            ]
            if abs(omega[partner] - omega[k].conjugate()) > tolerance:
                continue
            basis, triangular = np.linalg.qr(
                np.column_stack((vectors[:, k].real, vectors[:, k].imag))
            )
            value = omega[k].real
            if (
                abs(triangular[1, 1]) > 1.0e-12
                and np.max(np.linalg.norm(matrix @ basis - value * basis, axis=0))
                < tolerance
            ):
                vectors[:, [k, partner]] = basis
                omega[[k, partner]] = value
    return omega, vectors


def natural_orbital_space(
    cis_vectors,
    fock_oo,
    fock_vv,
    occupied_threshold=0.99,
    virtual_threshold=0.99,
    degeneracy_tolerance=1.0e-10,
):
    """State-averaged CIS NOs and block semicanonicalization (2017, Eqs.24-29).

    Vectors have shape (nroot,nvirtual,noccupied) and unit spatial norm.
    Returns occupied/virtual rotations (old x new), active labels and NO weights.
    Active occupied orbitals are placed last, active virtual orbitals first.
    """
    vectors = np.asarray(cis_vectors)
    if vectors.ndim != 3 or not len(vectors) or not np.all(np.isfinite(vectors)):
        raise ValueError(
            "CIS vectors must be a finite (nroot,nvirtual,noccupied) array"
        )
    norms = np.linalg.norm(vectors.reshape(len(vectors), -1), axis=1)
    if np.any(norms == 0):
        raise ValueError("CIS vectors must have nonzero norm")
    vectors = vectors / norms[:, None, None]
    do = _einsum("rai,raj->ij", vectors.conj(), vectors).real / len(vectors)
    dv = _einsum("rai,rbi->ab", vectors, vectors.conj()).real / len(vectors)

    return natural_orbital_space_from_densities(
        do,
        dv,
        fock_oo,
        fock_vv,
        occupied_threshold,
        virtual_threshold,
        degeneracy_tolerance,
    )


def guess_densities(vectors):
    """State-averaged hole/particle densities of normalized spin-resolved CI SD vectors.

    Same-spin doubles have the 1/4 determinant metric. OO/VV one-body
    operators preserve excitation rank, so no singles/doubles cross terms enter.
    """
    do, dv = [], []
    E = _einsum
    for R in vectors:
        norm = sum(
            np.vdot(getattr(R, s), getattr(R, s)).real * weight
            for s, weight in [("a", 1), ("b", 1), ("aa", 0.25), ("ab", 1), ("bb", 0.25)]
        )
        if norm <= 0:
            raise ValueError("Neutral guess has zero determinant norm")
        hole = E("ai,aj->ij", R.a, R.a) + E("ai,aj->ij", R.b, R.b)
        hole += 0.5 * E("abik,abjk->ij", R.aa, R.aa) + 0.5 * E(
            "abik,abjk->ij", R.bb, R.bb
        )
        hole += E("abik,abjk->ij", R.ab, R.ab) + E("abki,abkj->ij", R.ab, R.ab)
        particle = E("ai,bi->ab", R.a, R.a) + E("ai,bi->ab", R.b, R.b)
        particle += 0.5 * E("acij,bcij->ab", R.aa, R.aa) + 0.5 * E(
            "acij,bcij->ab", R.bb, R.bb
        )
        particle += E("acij,bcij->ab", R.ab, R.ab) + E("caij,cbij->ab", R.ab, R.ab)
        do.append(hole / (2 * norm))
        dv.append(particle / (2 * norm))
    return np.mean(do, axis=0), np.mean(dv, axis=0)


def natural_orbital_space_from_densities(
    do,
    dv,
    fock_oo,
    fock_vv,
    occupied_threshold=0.99,
    virtual_threshold=0.99,
    degeneracy_tolerance=1.0e-10,
):
    """Cumulative density selection and block semicanonicalization for CIS or CISd."""

    def select(density, fock, threshold, holes):
        if not 0 < threshold <= 1:
            raise ValueError("Cumulative NO thresholds must be in (0,1]")
        weights, U = np.linalg.eigh(density)
        weights, U = np.maximum(weights[::-1], 0), U[:, ::-1]
        nact = min(
            len(weights),
            np.searchsorted(np.cumsum(weights), threshold * weights.sum()) + 1,
        )
        while (
            nact < len(weights)
            and abs(weights[nact] - weights[nact - 1]) < degeneracy_tolerance
        ):
            nact += 1
        active, inactive = U[:, :nact], U[:, nact:]

        def canonicalize(block):
            return block @ np.linalg.eigh(block.T @ fock @ block)[1]

        active, inactive = canonicalize(active), canonicalize(inactive)
        U = np.column_stack((inactive, active) if holes else (active, inactive))
        indices = (
            np.arange(len(weights) - nact, len(weights)) if holes else np.arange(nact)
        )
        return U, indices, weights

    Uo, occupied, wo = select(do, fock_oo, occupied_threshold, True)
    Uv, virtual, wv = select(dv, fock_vv, virtual_threshold, False)
    return Uo, Uv, occupied, virtual, wo, wv


def rotate_hbar(H, T, Uo, Uv):
    """Rotate a converged HBar and T within occupied/virtual spaces.

    This is equivalent to solving CCSD in the rotated orbitals; neither the
    upstream reference nor the cached upstream HBar/amplitudes are modified.
    """
    from copy import deepcopy

    out, amplitudes = deepcopy(H), deepcopy(T)
    rotations = {"o": Uo, "v": Uv}
    for spin in ("a", "b", "aa", "ab", "bb"):
        source, target = getattr(H, spin), getattr(out, spin)
        for block in source.slices:
            value = getattr(source, block)
            if value is None:
                continue
            for axis, label in enumerate(block):
                value = np.moveaxis(
                    np.tensordot(value, rotations[label], axes=(axis, 0)), -1, axis
                )
            setattr(target, block, np.asfortranarray(value))
    for spin in ("a", "b", "aa", "ab", "bb"):
        value = getattr(T, spin)
        for axis, rotation in enumerate([Uv] * len(spin) + [Uo] * len(spin)):
            value = np.moveaxis(np.tensordot(value, rotation, axes=(axis, 0)), -1, axis)
        setattr(amplitudes, spin, np.asfortranarray(value))
    return out, amplitudes
