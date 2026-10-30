"""Unrelaxed CCSD densities for the factorized analytic gradients.

The densities are those of the T1-dressed frame: with H = e^{-T1} H e^{T1} (one- plus
two-body, see ccpy.gradients.ccsd_gradient.build_t1_dressed_hamiltonian),

    L = E_ref(H) + E(T2; H) + sum_mu lambda_mu R_mu(T2; H),

and every density element is the derivative of the functional with respect to the
corresponding block of H, exactly as the blocks are stored (normal-ordered):

    rdm1.a.ov[m, e] = dF/dH.a.ov[m, e],   rdm2.ab.vovv[a, m, e, f] = dF/dH.ab.vovv[a, m, e, f], ...

so F = sum_blocks rdm . H (the reference energy excluded). The vvvv blocks are never stored;
calc_rdm2_vvvv_slice returns them one first index at a time.
"""
import numpy as np
from ccpy.models.integrals import Integral

def calc_ccsd_rdm(T, L, system, energy=True):
    """Densities of E(T2; H) + sum_mu L_mu R_mu(T2; H) at T1 = 0 (energy=False drops E: the
    density of sum_mu z_mu R_mu for a response vector L = z). Only T2 of T is used."""
    rdm1 = Integral.from_empty(system, 1, use_none=True)
    rdm2 = Integral.from_empty(system, 2, use_none=True)

    rdm1.a.oo = -0.5 * np.einsum("abmj,abij->mi", T.aa, L.aa, optimize=True)
    rdm1.a.oo -= np.einsum("abmj,abij->mi", T.ab, L.ab, optimize=True)

    rdm1.a.ov = np.einsum("aeim,ai->me", T.aa, L.a, optimize=True)
    rdm1.a.ov += np.einsum("eami,ai->me", T.ab, L.b, optimize=True)

    rdm1.a.vo = L.a.copy()

    rdm1.a.vv = 0.5 * np.einsum("ebij,abij->ae", T.aa, L.aa, optimize=True)
    rdm1.a.vv += np.einsum("ebij,abij->ae", T.ab, L.ab, optimize=True)

    rdm1.b.oo = -np.einsum("abim,abij->mj", T.ab, L.ab, optimize=True)
    rdm1.b.oo -= 0.5 * np.einsum("abmj,abij->mi", T.bb, L.bb, optimize=True)

    rdm1.b.ov = np.einsum("aeim,ai->me", T.ab, L.a, optimize=True)
    rdm1.b.ov += np.einsum("aeim,ai->me", T.bb, L.b, optimize=True)

    rdm1.b.vo = L.b.copy()

    rdm1.b.vv = np.einsum("aeij,abij->be", T.ab, L.ab, optimize=True)
    rdm1.b.vv += 0.5 * np.einsum("ebij,abij->ae", T.bb, L.bb, optimize=True)

    rdm2.aa.oooo = 0.125 * np.einsum("abmn,abij->mnij", T.aa, L.aa, optimize=True)

    rdm2.aa.ooov = -0.5 * np.einsum("afmn,ai->mnif", T.aa, L.a, optimize=True)

    if energy:
        rdm2.aa.oovv = 0.25 * T.aa.transpose(2, 3, 0, 1)
    else:
        rdm2.aa.oovv = 0.0
    rdm2.aa.oovv -= 0.25 * np.einsum("afmn,ebij,abij->mnef", T.aa, T.aa, L.aa, optimize=True)
    rdm2.aa.oovv -= 0.25 * np.einsum("efin,abmj,abij->mnef", T.aa, T.aa, L.aa, optimize=True)
    rdm2.aa.oovv += 0.5 * np.einsum("afin,ebmj,abij->mnef", T.aa, T.aa, L.aa, optimize=True)
    rdm2.aa.oovv += 0.0625 * np.einsum("efij,abmn,abij->mnef", T.aa, T.aa, L.aa, optimize=True)
    rdm2.aa.oovv -= 0.5 * np.einsum("afmn,ebij,abij->mnef", T.aa, T.ab, L.ab, optimize=True)
    rdm2.aa.oovv -= 0.5 * np.einsum("efin,abmj,abij->mnef", T.aa, T.ab, L.ab, optimize=True)
    rdm2.aa.oovv += np.einsum("aeim,fbnj,abij->mnef", T.aa, T.ab, L.ab, optimize=True)
    rdm2.aa.oovv += 0.5 * np.einsum("fani,ebmj,abij->mnef", T.ab, T.ab, L.bb, optimize=True)

    rdm2.aa.voov = np.einsum("ebmj,abij->amie", T.aa, L.aa, optimize=True)
    rdm2.aa.voov += np.einsum("ebmj,abij->amie", T.ab, L.ab, optimize=True)

    rdm2.aa.vovv = 0.5 * np.einsum("efin,ai->anef", T.aa, L.a, optimize=True)

    rdm2.aa.vvoo = 0.25 * L.aa

    rdm2.ab.oooo = np.einsum("abmn,abij->mnij", T.ab, L.ab, optimize=True)

    rdm2.ab.ooov = -np.einsum("afmn,ai->mnif", T.ab, L.a, optimize=True)

    rdm2.ab.oovo = -np.einsum("fanm,ai->nmfi", T.ab, L.b, optimize=True)

    if energy:
        rdm2.ab.oovv = T.ab.transpose(2, 3, 0, 1).copy()
    else:
        rdm2.ab.oovv = 0.0
    rdm2.ab.oovv -= 0.5 * np.einsum("afmn,ebij,abij->mnef", T.ab, T.aa, L.aa, optimize=True)
    rdm2.ab.oovv -= 0.5 * np.einsum("efin,abmj,abij->mnef", T.ab, T.aa, L.aa, optimize=True)
    rdm2.ab.oovv += np.einsum("afin,ebmj,abij->mnef", T.ab, T.aa, L.aa, optimize=True)
    rdm2.ab.oovv -= np.einsum("afmn,ebij,abij->mnef", T.ab, T.ab, L.ab, optimize=True)
    rdm2.ab.oovv -= np.einsum("fbnm,aeij,abij->nmfe", T.ab, T.ab, L.ab, optimize=True)
    rdm2.ab.oovv -= np.einsum("efin,abmj,abij->mnef", T.ab, T.ab, L.ab, optimize=True)
    rdm2.ab.oovv -= np.einsum("fenj,abim,abij->nmfe", T.ab, T.ab, L.ab, optimize=True)
    rdm2.ab.oovv += np.einsum("aeim,fbnj,abij->nmfe", T.ab, T.ab, L.ab, optimize=True)
    rdm2.ab.oovv += np.einsum("aeim,fbnj,abij->mnef", T.aa, T.bb, L.ab, optimize=True)
    rdm2.ab.oovv += np.einsum("afmj,ebin,abij->mnef", T.ab, T.ab, L.ab, optimize=True)
    rdm2.ab.oovv += np.einsum("efij,abmn,abij->mnef", T.ab, T.ab, L.ab, optimize=True)
    rdm2.ab.oovv -= 0.5 * np.einsum("fanm,ebij,abij->nmfe", T.ab, T.bb, L.bb, optimize=True)
    rdm2.ab.oovv -= 0.5 * np.einsum("feni,abmj,abij->nmfe", T.ab, T.bb, L.bb, optimize=True)
    rdm2.ab.oovv += np.einsum("afin,ebmj,abij->mnef", T.bb, T.ab, L.bb, optimize=True)

    rdm2.ab.ovov = -np.einsum("aemj,abij->mbie", T.ab, L.ab, optimize=True)

    rdm2.ab.ovvo = np.einsum("aeim,abij->mbej", T.aa, L.ab, optimize=True)
    rdm2.ab.ovvo += np.einsum("ebmj,abij->maei", T.ab, L.bb, optimize=True)

    rdm2.ab.ovvv = np.einsum("feni,ai->nafe", T.ab, L.b, optimize=True)

    rdm2.ab.voov = np.einsum("bejm,abij->amie", T.ab, L.aa, optimize=True)
    rdm2.ab.voov += np.einsum("ebmj,abij->amie", T.bb, L.ab, optimize=True)

    rdm2.ab.vovo = -np.einsum("ebim,abij->amej", T.ab, L.ab, optimize=True)

    rdm2.ab.vovv = np.einsum("efin,ai->anef", T.ab, L.a, optimize=True)

    rdm2.ab.vvoo = L.ab.copy()

    rdm2.bb.oooo = 0.125 * np.einsum("abmn,abij->mnij", T.bb, L.bb, optimize=True)

    rdm2.bb.ooov = -0.5 * np.einsum("afmn,ai->mnif", T.bb, L.b, optimize=True)

    if energy:
        rdm2.bb.oovv = 0.25 * T.bb.transpose(2, 3, 0, 1)
    else:
        rdm2.bb.oovv = 0.0
    rdm2.bb.oovv += 0.5 * np.einsum("afin,bejm,abij->mnef", T.ab, T.ab, L.aa, optimize=True)
    rdm2.bb.oovv -= 0.5 * np.einsum("bfmn,aeij,abij->mnef", T.bb, T.ab, L.ab, optimize=True)
    rdm2.bb.oovv -= 0.5 * np.einsum("efjn,abim,abij->mnef", T.bb, T.ab, L.ab, optimize=True)
    rdm2.bb.oovv += np.einsum("aeim,fbnj,abij->mnef", T.ab, T.bb, L.ab, optimize=True)
    rdm2.bb.oovv -= 0.25 * np.einsum("afmn,ebij,abij->mnef", T.bb, T.bb, L.bb, optimize=True)
    rdm2.bb.oovv -= 0.25 * np.einsum("efin,abmj,abij->mnef", T.bb, T.bb, L.bb, optimize=True)
    rdm2.bb.oovv += 0.5 * np.einsum("afin,ebmj,abij->mnef", T.bb, T.bb, L.bb, optimize=True)
    rdm2.bb.oovv += 0.0625 * np.einsum("efij,abmn,abij->mnef", T.bb, T.bb, L.bb, optimize=True)

    rdm2.bb.voov = np.einsum("aeim,abij->bmje", T.ab, L.ab, optimize=True)
    rdm2.bb.voov += np.einsum("ebmj,abij->amie", T.bb, L.bb, optimize=True)

    rdm2.bb.vovv = 0.5 * np.einsum("efin,ai->anef", T.bb, L.b, optimize=True)

    rdm2.bb.vvoo = 0.25 * L.bb

    return rdm1, rdm2


def calc_ccsd_rdm_derivative(T, W, L, system, energy=True):
    """d/de of calc_ccsd_rdm(T + e W, L) at e = 0 (T2 direction W.aa, W.ab, W.bb)."""
    rdm1 = Integral.from_empty(system, 1, use_none=True)
    rdm2 = Integral.from_empty(system, 2, use_none=True)

    rdm1.a.oo = -0.5 * np.einsum("abmj,abij->mi", W.aa, L.aa, optimize=True)
    rdm1.a.oo -= np.einsum("abmj,abij->mi", W.ab, L.ab, optimize=True)

    rdm1.a.ov = np.einsum("aeim,ai->me", W.aa, L.a, optimize=True)
    rdm1.a.ov += np.einsum("eami,ai->me", W.ab, L.b, optimize=True)

    rdm1.a.vv = 0.5 * np.einsum("ebij,abij->ae", W.aa, L.aa, optimize=True)
    rdm1.a.vv += np.einsum("ebij,abij->ae", W.ab, L.ab, optimize=True)

    rdm1.b.oo = -np.einsum("abim,abij->mj", W.ab, L.ab, optimize=True)
    rdm1.b.oo -= 0.5 * np.einsum("abmj,abij->mi", W.bb, L.bb, optimize=True)

    rdm1.b.ov = np.einsum("aeim,ai->me", W.ab, L.a, optimize=True)
    rdm1.b.ov += np.einsum("aeim,ai->me", W.bb, L.b, optimize=True)

    rdm1.b.vv = np.einsum("aeij,abij->be", W.ab, L.ab, optimize=True)
    rdm1.b.vv += 0.5 * np.einsum("ebij,abij->ae", W.bb, L.bb, optimize=True)

    rdm2.aa.oooo = 0.125 * np.einsum("abmn,abij->mnij", W.aa, L.aa, optimize=True)

    rdm2.aa.ooov = -0.5 * np.einsum("afmn,ai->mnif", W.aa, L.a, optimize=True)

    if energy:
        rdm2.aa.oovv = 0.25 * W.aa.transpose(2, 3, 0, 1)
    else:
        rdm2.aa.oovv = 0.0
    rdm2.aa.oovv -= 0.25 * np.einsum("afmn,ebij,abij->mnef", W.aa, T.aa, L.aa, optimize=True)
    rdm2.aa.oovv -= 0.25 * np.einsum("afmn,ebij,abij->mnef", T.aa, W.aa, L.aa, optimize=True)
    rdm2.aa.oovv -= 0.25 * np.einsum("efin,abmj,abij->mnef", W.aa, T.aa, L.aa, optimize=True)
    rdm2.aa.oovv -= 0.25 * np.einsum("efin,abmj,abij->mnef", T.aa, W.aa, L.aa, optimize=True)
    rdm2.aa.oovv += 0.5 * np.einsum("afin,ebmj,abij->mnef", W.aa, T.aa, L.aa, optimize=True)
    rdm2.aa.oovv += 0.5 * np.einsum("afin,ebmj,abij->mnef", T.aa, W.aa, L.aa, optimize=True)
    rdm2.aa.oovv += 0.0625 * np.einsum("efij,abmn,abij->mnef", W.aa, T.aa, L.aa, optimize=True)
    rdm2.aa.oovv += 0.0625 * np.einsum("efij,abmn,abij->mnef", T.aa, W.aa, L.aa, optimize=True)
    rdm2.aa.oovv -= 0.5 * np.einsum("afmn,ebij,abij->mnef", W.aa, T.ab, L.ab, optimize=True)
    rdm2.aa.oovv -= 0.5 * np.einsum("afmn,ebij,abij->mnef", T.aa, W.ab, L.ab, optimize=True)
    rdm2.aa.oovv -= 0.5 * np.einsum("efin,abmj,abij->mnef", W.aa, T.ab, L.ab, optimize=True)
    rdm2.aa.oovv -= 0.5 * np.einsum("efin,abmj,abij->mnef", T.aa, W.ab, L.ab, optimize=True)
    rdm2.aa.oovv += np.einsum("aeim,fbnj,abij->mnef", W.aa, T.ab, L.ab, optimize=True)
    rdm2.aa.oovv += np.einsum("aeim,fbnj,abij->mnef", T.aa, W.ab, L.ab, optimize=True)
    rdm2.aa.oovv += 0.5 * np.einsum("fani,ebmj,abij->mnef", W.ab, T.ab, L.bb, optimize=True)
    rdm2.aa.oovv += 0.5 * np.einsum("fani,ebmj,abij->mnef", T.ab, W.ab, L.bb, optimize=True)

    rdm2.aa.voov = np.einsum("ebmj,abij->amie", W.aa, L.aa, optimize=True)
    rdm2.aa.voov += np.einsum("ebmj,abij->amie", W.ab, L.ab, optimize=True)

    rdm2.aa.vovv = 0.5 * np.einsum("efin,ai->anef", W.aa, L.a, optimize=True)

    rdm2.ab.oooo = np.einsum("abmn,abij->mnij", W.ab, L.ab, optimize=True)

    rdm2.ab.ooov = -np.einsum("afmn,ai->mnif", W.ab, L.a, optimize=True)

    rdm2.ab.oovo = -np.einsum("fanm,ai->nmfi", W.ab, L.b, optimize=True)

    if energy:
        rdm2.ab.oovv = W.ab.transpose(2, 3, 0, 1).copy()
    else:
        rdm2.ab.oovv = 0.0
    rdm2.ab.oovv -= 0.5 * np.einsum("afmn,ebij,abij->mnef", W.ab, T.aa, L.aa, optimize=True)
    rdm2.ab.oovv -= 0.5 * np.einsum("afmn,ebij,abij->mnef", T.ab, W.aa, L.aa, optimize=True)
    rdm2.ab.oovv -= 0.5 * np.einsum("efin,abmj,abij->mnef", W.ab, T.aa, L.aa, optimize=True)
    rdm2.ab.oovv -= 0.5 * np.einsum("efin,abmj,abij->mnef", T.ab, W.aa, L.aa, optimize=True)
    rdm2.ab.oovv += np.einsum("afin,ebmj,abij->mnef", W.ab, T.aa, L.aa, optimize=True)
    rdm2.ab.oovv += np.einsum("afin,ebmj,abij->mnef", T.ab, W.aa, L.aa, optimize=True)
    rdm2.ab.oovv -= np.einsum("afmn,ebij,abij->mnef", W.ab, T.ab, L.ab, optimize=True)
    rdm2.ab.oovv -= np.einsum("afmn,ebij,abij->mnef", T.ab, W.ab, L.ab, optimize=True)
    rdm2.ab.oovv -= np.einsum("fbnm,aeij,abij->nmfe", W.ab, T.ab, L.ab, optimize=True)
    rdm2.ab.oovv -= np.einsum("fbnm,aeij,abij->nmfe", T.ab, W.ab, L.ab, optimize=True)
    rdm2.ab.oovv -= np.einsum("efin,abmj,abij->mnef", W.ab, T.ab, L.ab, optimize=True)
    rdm2.ab.oovv -= np.einsum("efin,abmj,abij->mnef", T.ab, W.ab, L.ab, optimize=True)
    rdm2.ab.oovv -= np.einsum("fenj,abim,abij->nmfe", W.ab, T.ab, L.ab, optimize=True)
    rdm2.ab.oovv -= np.einsum("fenj,abim,abij->nmfe", T.ab, W.ab, L.ab, optimize=True)
    rdm2.ab.oovv += np.einsum("aeim,fbnj,abij->nmfe", W.ab, T.ab, L.ab, optimize=True)
    rdm2.ab.oovv += np.einsum("aeim,fbnj,abij->nmfe", T.ab, W.ab, L.ab, optimize=True)
    rdm2.ab.oovv += np.einsum("aeim,fbnj,abij->mnef", W.aa, T.bb, L.ab, optimize=True)
    rdm2.ab.oovv += np.einsum("aeim,fbnj,abij->mnef", T.aa, W.bb, L.ab, optimize=True)
    rdm2.ab.oovv += np.einsum("afmj,ebin,abij->mnef", W.ab, T.ab, L.ab, optimize=True)
    rdm2.ab.oovv += np.einsum("afmj,ebin,abij->mnef", T.ab, W.ab, L.ab, optimize=True)
    rdm2.ab.oovv += np.einsum("efij,abmn,abij->mnef", W.ab, T.ab, L.ab, optimize=True)
    rdm2.ab.oovv += np.einsum("efij,abmn,abij->mnef", T.ab, W.ab, L.ab, optimize=True)
    rdm2.ab.oovv -= 0.5 * np.einsum("fanm,ebij,abij->nmfe", W.ab, T.bb, L.bb, optimize=True)
    rdm2.ab.oovv -= 0.5 * np.einsum("fanm,ebij,abij->nmfe", T.ab, W.bb, L.bb, optimize=True)
    rdm2.ab.oovv -= 0.5 * np.einsum("feni,abmj,abij->nmfe", W.ab, T.bb, L.bb, optimize=True)
    rdm2.ab.oovv -= 0.5 * np.einsum("feni,abmj,abij->nmfe", T.ab, W.bb, L.bb, optimize=True)
    rdm2.ab.oovv += np.einsum("afin,ebmj,abij->mnef", W.bb, T.ab, L.bb, optimize=True)
    rdm2.ab.oovv += np.einsum("afin,ebmj,abij->mnef", T.bb, W.ab, L.bb, optimize=True)

    rdm2.ab.ovov = -np.einsum("aemj,abij->mbie", W.ab, L.ab, optimize=True)

    rdm2.ab.ovvo = np.einsum("aeim,abij->mbej", W.aa, L.ab, optimize=True)
    rdm2.ab.ovvo += np.einsum("ebmj,abij->maei", W.ab, L.bb, optimize=True)

    rdm2.ab.ovvv = np.einsum("feni,ai->nafe", W.ab, L.b, optimize=True)

    rdm2.ab.voov = np.einsum("bejm,abij->amie", W.ab, L.aa, optimize=True)
    rdm2.ab.voov += np.einsum("ebmj,abij->amie", W.bb, L.ab, optimize=True)

    rdm2.ab.vovo = -np.einsum("ebim,abij->amej", W.ab, L.ab, optimize=True)

    rdm2.ab.vovv = np.einsum("efin,ai->anef", W.ab, L.a, optimize=True)

    rdm2.bb.oooo = 0.125 * np.einsum("abmn,abij->mnij", W.bb, L.bb, optimize=True)

    rdm2.bb.ooov = -0.5 * np.einsum("afmn,ai->mnif", W.bb, L.b, optimize=True)

    if energy:
        rdm2.bb.oovv = 0.25 * W.bb.transpose(2, 3, 0, 1)
    else:
        rdm2.bb.oovv = 0.0
    rdm2.bb.oovv += 0.5 * np.einsum("afin,bejm,abij->mnef", W.ab, T.ab, L.aa, optimize=True)
    rdm2.bb.oovv += 0.5 * np.einsum("afin,bejm,abij->mnef", T.ab, W.ab, L.aa, optimize=True)
    rdm2.bb.oovv -= 0.5 * np.einsum("bfmn,aeij,abij->mnef", W.bb, T.ab, L.ab, optimize=True)
    rdm2.bb.oovv -= 0.5 * np.einsum("bfmn,aeij,abij->mnef", T.bb, W.ab, L.ab, optimize=True)
    rdm2.bb.oovv -= 0.5 * np.einsum("efjn,abim,abij->mnef", W.bb, T.ab, L.ab, optimize=True)
    rdm2.bb.oovv -= 0.5 * np.einsum("efjn,abim,abij->mnef", T.bb, W.ab, L.ab, optimize=True)
    rdm2.bb.oovv += np.einsum("aeim,fbnj,abij->mnef", W.ab, T.bb, L.ab, optimize=True)
    rdm2.bb.oovv += np.einsum("aeim,fbnj,abij->mnef", T.ab, W.bb, L.ab, optimize=True)
    rdm2.bb.oovv -= 0.25 * np.einsum("afmn,ebij,abij->mnef", W.bb, T.bb, L.bb, optimize=True)
    rdm2.bb.oovv -= 0.25 * np.einsum("afmn,ebij,abij->mnef", T.bb, W.bb, L.bb, optimize=True)
    rdm2.bb.oovv -= 0.25 * np.einsum("efin,abmj,abij->mnef", W.bb, T.bb, L.bb, optimize=True)
    rdm2.bb.oovv -= 0.25 * np.einsum("efin,abmj,abij->mnef", T.bb, W.bb, L.bb, optimize=True)
    rdm2.bb.oovv += 0.5 * np.einsum("afin,ebmj,abij->mnef", W.bb, T.bb, L.bb, optimize=True)
    rdm2.bb.oovv += 0.5 * np.einsum("afin,ebmj,abij->mnef", T.bb, W.bb, L.bb, optimize=True)
    rdm2.bb.oovv += 0.0625 * np.einsum("efij,abmn,abij->mnef", W.bb, T.bb, L.bb, optimize=True)
    rdm2.bb.oovv += 0.0625 * np.einsum("efij,abmn,abij->mnef", T.bb, W.bb, L.bb, optimize=True)

    rdm2.bb.voov = np.einsum("aeim,abij->bmje", W.ab, L.ab, optimize=True)
    rdm2.bb.voov += np.einsum("ebmj,abij->amie", W.bb, L.bb, optimize=True)

    rdm2.bb.vovv = 0.5 * np.einsum("efin,ai->anef", W.bb, L.b, optimize=True)

    return rdm1, rdm2


def calc_rdm2_vvvv_slice(L, T, a, spin):
    """The vvvv densities of calc_ccsd_rdm for one value of the first index: for spin "a",
    (dF/dH.aa.vvvv[a], dF/dH.ab.vvvv[a]); for spin "b", dF/dH.bb.vvvv[a]. The vvvv term of the
    residual is linear in T2, so calc_rdm2_vvvv_slice(L, W, a, spin) is the slice of
    calc_ccsd_rdm_derivative. Pass L = z for the z densities."""
    if spin == "a":
        rdm_aa = 0.125 * np.einsum("bij,cdij->bcd", L.aa[a], T.aa, optimize=True)
        rdm_ab = np.einsum("bij,cdij->bcd", L.ab[a], T.ab, optimize=True)
        return rdm_aa, rdm_ab
    return 0.125 * np.einsum("bij,cdij->bcd", L.bb[a], T.bb, optimize=True)
