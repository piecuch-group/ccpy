"""
MPI-Parallel Coupled-Cluster Method with Singles and Doubles (CCSD)

Parallelization strategy:
  - T1 updates (O(N^4)) are replicated on all ranks.
  - Intermediates are computed fully on all ranks.
  - T2 residual contractions are distributed over the first virtual index
    dimension.  Each rank computes its slice of the output, then an
    MPI_Allreduce combines the partial results before the Fortran
    denominator-update routine (which runs identically on every rank).
"""

import numpy as np

from ccpy.hbar.hbar_ccs import get_pre_ccs_intermediates, get_ccs_intermediates_opt
from ccpy.lib.core import cc_loops2

# Lazy import: mpi4py is only required when the update functions are actually
# called (i.e. at runtime under ``mpirun``).  This allows the module to be
# registered and introspected without mpi4py being installed.
def _get_MPI():
    from mpi4py import MPI as _MPI
    return _MPI


# ------------------------------------------------------------------ helpers
def _get_distribution(n, rank, size):
    """Return (start, end) for a contiguous partition of range *n* on *rank*."""
    chunk = n // size
    remainder = n % size
    if rank < remainder:
        start = rank * (chunk + 1)
        end = start + chunk + 1
    else:
        start = remainder * (chunk + 1) + (rank - remainder) * chunk
        end = start + chunk
    return start, end


# ------------------------------------------------------------------ driver
def update(T, dT, H, X, shift, flag_RHF, comm):
    """Top-level CCSD update with MPI distribution over the first virtual index.

    Parameters
    ----------
    comm : MPI.Comm
        MPI communicator (e.g. ``MPI.COMM_WORLD``).
    """

    # pre-CCS intermediates (replicated)
    X = get_pre_ccs_intermediates(X, T, H, flag_RHF)

    # T1 updates – cheap, replicated on all ranks
    T, dT = update_t1a(T, dT, X, H, shift)
    if flag_RHF:
        T.b = T.a.copy()
        dT.b = dT.a.copy()
    else:
        T, dT = update_t1b(T, dT, X, H, shift)

    # CCS intermediates (replicated)
    X = get_ccs_intermediates_opt(X, T, H, flag_RHF)

    # T2 updates – distributed over first virtual index, then Allreduced
    T, dT = update_t2a_mpi(T, dT, X, H, shift, comm)
    T, dT = update_t2b_mpi(T, dT, X, H, shift, comm)
    if flag_RHF:
        T.bb = T.aa.copy()
        dT.bb = dT.aa.copy()
    else:
        T, dT = update_t2c_mpi(T, dT, X, H, shift, comm)
    return T, dT


# --------------------------------------------------------------- T1 (replicated)
def update_t1a(T, dT, X, H, shift):
    """Update t1a amplitudes (replicated on every rank)."""
    dT.a = -np.einsum("mi,am->ai", X.a.oo, T.a, optimize=True)
    dT.a += np.einsum("ae,ei->ai", X.a.vv, T.a, optimize=True)
    dT.a += np.einsum("me,aeim->ai", X.a.ov, T.aa, optimize=True)
    dT.a += np.einsum("me,aeim->ai", X.b.ov, T.ab, optimize=True)
    dT.a += np.einsum("anif,fn->ai", H.aa.voov, T.a, optimize=True)
    dT.a += np.einsum("anif,fn->ai", H.ab.voov, T.b, optimize=True)
    dT.a -= 0.5 * np.einsum("mnif,afmn->ai", H.aa.ooov, T.aa, optimize=True)
    dT.a -= np.einsum("mnif,afmn->ai", H.ab.ooov, T.ab, optimize=True)
    dT.a += 0.5 * np.einsum("anef,efin->ai", H.aa.vovv, T.aa, optimize=True)
    dT.a += np.einsum("anef,efin->ai", H.ab.vovv, T.ab, optimize=True)
    T.a, dT.a = cc_loops2.update_t1a(
        T.a, dT.a + H.a.vo, H.a.oo, H.a.vv, shift
    )
    return T, dT


def update_t1b(T, dT, X, H, shift):
    """Update t1b amplitudes (replicated on every rank)."""
    dT.b = -np.einsum("mi,am->ai", X.b.oo, T.b, optimize=True)
    dT.b += np.einsum("ae,ei->ai", X.b.vv, T.b, optimize=True)
    dT.b += np.einsum("anif,fn->ai", H.bb.voov, T.b, optimize=True)
    dT.b += np.einsum("nafi,fn->ai", H.ab.ovvo, T.a, optimize=True)
    dT.b += np.einsum("me,eami->ai", X.a.ov, T.ab, optimize=True)
    dT.b += np.einsum("me,aeim->ai", X.b.ov, T.bb, optimize=True)
    dT.b -= 0.5 * np.einsum("mnif,afmn->ai", H.bb.ooov, T.bb, optimize=True)
    dT.b -= np.einsum("nmfi,fanm->ai", H.ab.oovo, T.ab, optimize=True)
    dT.b += 0.5 * np.einsum("anef,efin->ai", H.bb.vovv, T.bb, optimize=True)
    dT.b += np.einsum("nafe,feni->ai", H.ab.ovvv, T.ab, optimize=True)
    T.b, dT.b = cc_loops2.update_t1b(
        T.b, dT.b + H.b.vo, H.b.oo, H.b.vv, shift
    )
    return T, dT


# --------------------------------------------------------------- T2 (distributed)
def update_t2a_mpi(T, dT, H, H0, shift, comm):
    """Update t2a amplitudes, distributing contractions over the first
    alpha-virtual index and combining with ``Allreduce``."""
    rank = comm.Get_rank()
    size = comm.Get_size()
    nua, noa = T.a.shape
    a0, a1 = _get_distribution(nua, rank, size)

    # intermediates (replicated)
    I2A_voov = H.aa.voov + (
        + 0.5 * np.einsum("mnef,afin->amie", H0.aa.oovv, T.aa, optimize=True)
        + np.einsum("mnef,afin->amie", H0.ab.oovv, T.ab, optimize=True)
    )
    I2A_oooo = H.aa.oooo + 0.5 * np.einsum("mnef,efij->mnij", H0.aa.oovv, T.aa, optimize=True)
    I2B_voov = H.ab.voov + 0.5 * np.einsum("mnef,afin->amie", H0.bb.oovv, T.ab, optimize=True)
    I2A_vooo = H.aa.vooo + 0.5 * np.einsum('anef,efij->anij', H0.aa.vovv + 0.5 * H.aa.vovv, T.aa, optimize=True)

    tau = 0.5 * T.aa + np.einsum('ai,bj->abij', T.a, T.a, optimize=True)

    # each rank fills only its slice; rest stays zero
    dT.aa[:] = 0.0
    dT.aa[a0:a1] = -0.5 * np.einsum("amij,bm->abij", I2A_vooo[a0:a1], T.a, optimize=True)
    dT.aa[a0:a1] += 0.5 * np.einsum("abie,ej->abij", H.aa.vvov[a0:a1], T.a, optimize=True)
    dT.aa[a0:a1] += 0.5 * np.einsum("ae,ebij->abij", H.a.vv[a0:a1, :], T.aa, optimize=True)
    dT.aa[a0:a1] -= 0.5 * np.einsum("mi,abmj->abij", H.a.oo, T.aa[a0:a1], optimize=True)
    dT.aa[a0:a1] += np.einsum("amie,ebmj->abij", I2A_voov[a0:a1], T.aa, optimize=True)
    dT.aa[a0:a1] += np.einsum("amie,bejm->abij", I2B_voov[a0:a1], T.ab, optimize=True)
    dT.aa[a0:a1] += 0.125 * np.einsum("mnij,abmn->abij", I2A_oooo, T.aa[a0:a1], optimize=True)
    dT.aa[a0:a1] += 0.25 * np.einsum("abef,efij->abij", H0.aa.vvvv[a0:a1], tau, optimize=True)

    # combine partial results across ranks
    MPI = _get_MPI()
    comm.Allreduce(MPI.IN_PLACE, dT.aa, op=MPI.SUM)

    # denominator update (replicated)
    T.aa, dT.aa = cc_loops2.update_t2a(
        T.aa, dT.aa + 0.25 * H0.aa.vvoo, H0.a.oo, H0.a.vv, shift
    )
    return T, dT


def update_t2b_mpi(T, dT, H, H0, shift, comm):
    """Update t2b amplitudes, distributing contractions over the first
    alpha-virtual index and combining with ``Allreduce``."""
    rank = comm.Get_rank()
    size = comm.Get_size()
    nua, nub, noa, nob = T.ab.shape
    a0, a1 = _get_distribution(nua, rank, size)

    # intermediates (replicated)
    I2A_voov = H.aa.voov + (
        + np.einsum("mnef,aeim->anif", H0.aa.oovv, T.aa, optimize=True)
        + np.einsum("nmfe,aeim->anif", H0.ab.oovv, T.ab, optimize=True)
    )
    I2B_voov = H.ab.voov + (
        + np.einsum("mnef,aeim->anif", H0.ab.oovv, T.aa, optimize=True)
        + np.einsum("mnef,aeim->anif", H0.bb.oovv, T.ab, optimize=True)
    )
    I2B_oooo = H.ab.oooo + np.einsum("mnef,efij->mnij", H0.ab.oovv, T.ab, optimize=True)
    I2B_vovo = H.ab.vovo - np.einsum("mnef,afmj->anej", H0.ab.oovv, T.ab, optimize=True)
    I2B_ovoo = H.ab.ovoo + np.einsum("maef,efij->maij", H0.ab.ovvv + 0.5 * H.ab.ovvv, T.ab, optimize=True)
    I2B_vooo = H.ab.vooo + np.einsum("amef,efij->amij", H0.ab.vovv + 0.5 * H.ab.vovv, T.ab, optimize=True)

    tau = T.ab + np.einsum('ai,bj->abij', T.a, T.b, optimize=True)

    # each rank fills only its slice; rest stays zero
    dT.ab[:] = 0.0
    dT.ab[a0:a1] = -np.einsum("mbij,am->abij", I2B_ovoo, T.a[a0:a1], optimize=True)
    dT.ab[a0:a1] -= np.einsum("amij,bm->abij", I2B_vooo[a0:a1], T.b, optimize=True)
    dT.ab[a0:a1] += np.einsum("abej,ei->abij", H.ab.vvvo[a0:a1], T.a, optimize=True)
    dT.ab[a0:a1] += np.einsum("abie,ej->abij", H.ab.vvov[a0:a1], T.b, optimize=True)
    dT.ab[a0:a1] += np.einsum("ae,ebij->abij", H.a.vv[a0:a1, :], T.ab, optimize=True)
    dT.ab[a0:a1] += np.einsum("be,aeij->abij", H.b.vv, T.ab[a0:a1], optimize=True)
    dT.ab[a0:a1] -= np.einsum("mi,abmj->abij", H.a.oo, T.ab[a0:a1], optimize=True)
    dT.ab[a0:a1] -= np.einsum("mj,abim->abij", H.b.oo, T.ab[a0:a1], optimize=True)
    dT.ab[a0:a1] += np.einsum("amie,ebmj->abij", I2A_voov[a0:a1], T.ab, optimize=True)
    dT.ab[a0:a1] += np.einsum("amie,ebmj->abij", I2B_voov[a0:a1], T.bb, optimize=True)
    dT.ab[a0:a1] += np.einsum("mbej,aeim->abij", H.ab.ovvo, T.aa[a0:a1], optimize=True)
    dT.ab[a0:a1] += np.einsum("bmje,aeim->abij", H.bb.voov, T.ab[a0:a1], optimize=True)
    dT.ab[a0:a1] -= np.einsum("mbie,aemj->abij", H.ab.ovov, T.ab[a0:a1], optimize=True)
    dT.ab[a0:a1] -= np.einsum("amej,ebim->abij", I2B_vovo[a0:a1], T.ab, optimize=True)
    dT.ab[a0:a1] += np.einsum("mnij,abmn->abij", I2B_oooo, T.ab[a0:a1], optimize=True)
    dT.ab[a0:a1] += np.einsum("abef,efij->abij", H0.ab.vvvv[a0:a1], tau, optimize=True)

    # combine partial results across ranks
    MPI = _get_MPI()
    comm.Allreduce(MPI.IN_PLACE, dT.ab, op=MPI.SUM)

    # denominator update (replicated)
    T.ab, dT.ab = cc_loops2.update_t2b(
        T.ab, dT.ab + H0.ab.vvoo, H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv, shift
    )
    return T, dT


def update_t2c_mpi(T, dT, H, H0, shift, comm):
    """Update t2c amplitudes, distributing contractions over the first
    beta-virtual index and combining with ``Allreduce``."""
    rank = comm.Get_rank()
    size = comm.Get_size()
    nub, nob = T.b.shape
    a0, a1 = _get_distribution(nub, rank, size)

    # intermediates (replicated)
    I2C_oooo = H.bb.oooo + 0.5 * np.einsum("mnef,efij->mnij", H0.bb.oovv, T.bb, optimize=True)
    I2B_ovvo = H.ab.ovvo + (
        + np.einsum("mnef,afin->maei", H0.ab.oovv, T.bb, optimize=True)
        + 0.5 * np.einsum("mnef,fani->maei", H0.aa.oovv, T.ab, optimize=True)
    )
    I2C_voov = H.bb.voov + 0.5 * np.einsum("mnef,afin->amie", H0.bb.oovv, T.bb, optimize=True)
    I2C_vooo = H.bb.vooo + 0.5 * np.einsum('anef,efij->anij', H0.bb.vovv + 0.5 * H.bb.vovv, T.bb, optimize=True)

    tau = 0.5 * T.bb + np.einsum('ai,bj->abij', T.b, T.b, optimize=True)

    # each rank fills only its slice; rest stays zero
    dT.bb[:] = 0.0
    dT.bb[a0:a1] = -0.5 * np.einsum("amij,bm->abij", I2C_vooo[a0:a1], T.b, optimize=True)
    dT.bb[a0:a1] += 0.5 * np.einsum("abie,ej->abij", H.bb.vvov[a0:a1], T.b, optimize=True)
    dT.bb[a0:a1] += 0.5 * np.einsum("ae,ebij->abij", H.b.vv[a0:a1, :], T.bb, optimize=True)
    dT.bb[a0:a1] -= 0.5 * np.einsum("mi,abmj->abij", H.b.oo, T.bb[a0:a1], optimize=True)
    dT.bb[a0:a1] += np.einsum("amie,ebmj->abij", I2C_voov[a0:a1], T.bb, optimize=True)
    dT.bb[a0:a1] += np.einsum("maei,ebmj->abij", I2B_ovvo[:, a0:a1], T.ab, optimize=True)
    dT.bb[a0:a1] += 0.125 * np.einsum("mnij,abmn->abij", I2C_oooo, T.bb[a0:a1], optimize=True)
    dT.bb[a0:a1] += 0.25 * np.einsum("abef,efij->abij", H0.bb.vvvv[a0:a1], tau, optimize=True)

    # combine partial results across ranks
    MPI = _get_MPI()
    comm.Allreduce(MPI.IN_PLACE, dT.bb, op=MPI.SUM)

    # denominator update (replicated)
    T.bb, dT.bb = cc_loops2.update_t2c(
        T.bb, dT.bb + 0.25 * H0.bb.vvoo, H0.b.oo, H0.b.vv, shift
    )
    return T, dT
