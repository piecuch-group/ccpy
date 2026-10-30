"""Unrelaxed CR-CC(2,3)_A densities (the triples part) for the analytic gradient.

With r_K = M_K / D_K and l_K = N_K / D_K held fixed, delta_A = sum_K (l_K M_K + r_K N_K
- l_K r_K D_K). In the T1-dressed frame (ccpy.gradients.ccsd_gradient) M_K = <K|Hbar|0> and
N_K = <0|(L1 + L2) Hbar|K> are linear in the dressed Hamiltonian H, and

    calc_crcc23_rdm         d(sum_K l_K M_K + r_K N_K)/dH, as the stored blocks of H
                            (rdm1.a.ov = d/dH.a.ov, rdm2.ab.vovv = d/dH.ab.vovv, ...)
    calc_crcc23_rdm1_denominator
                            -d(sum_K l_K r_K D_K)/df, f the bare Fock diagonal (undressed)

R3 and L3 hold r_K and l_K at every antisymmetric copy (spin blocks aaa, aab, abb, bbb).
No density has a vvvv block.
"""
import numpy as np
from ccpy.models.integrals import Integral


def calc_crcc23_rdm(R3, L3, T, L, system):
    """Triples densities of delta_A in the T1-dressed frame (only T2 of T is used)."""
    rdm1 = Integral.from_empty(system, 1, use_none=True)
    rdm2 = Integral.from_empty(system, 2, use_none=True)

    rdm1.a.ov = -0.25 * np.einsum("aeij,bcmk,abcijk->me", T.aa, T.aa, L3.aaa, optimize=True)
    rdm1.a.ov += 0.25 * np.einsum("abij,abcijk->kc", L.aa, R3.aaa, optimize=True)
    rdm1.a.ov -= 0.5 * np.einsum("bcmk,aeij,abcijk->me", T.ab, T.aa, L3.aab, optimize=True)
    rdm1.a.ov -= 0.5 * np.einsum("abim,ecjk,abcijk->me", T.aa, T.ab, L3.aab, optimize=True)
    rdm1.a.ov += np.einsum("bcjk,abcijk->ia", L.ab, R3.aab, optimize=True)
    rdm1.a.ov -= np.einsum("cami,ebkj,cabkij->me", T.ab, T.ab, L3.abb, optimize=True)
    rdm1.a.ov += 0.25 * np.einsum("abij,cabkij->kc", L.bb, R3.abb, optimize=True)

    rdm1.b.ov = -np.einsum("acim,bejk,abcijk->me", T.ab, T.ab, L3.aab, optimize=True)
    rdm1.b.ov += 0.25 * np.einsum("abij,abcijk->kc", L.aa, R3.aab, optimize=True)
    rdm1.b.ov -= 0.5 * np.einsum("cbkm,aeij,cabkij->me", T.ab, T.bb, L3.abb, optimize=True)
    rdm1.b.ov -= 0.5 * np.einsum("abim,cekj,cabkij->me", T.bb, T.ab, L3.abb, optimize=True)
    rdm1.b.ov += np.einsum("cbkj,cabkij->ia", L.ab, R3.abb, optimize=True)
    rdm1.b.ov -= 0.25 * np.einsum("aeij,bcmk,abcijk->me", T.bb, T.bb, L3.bbb, optimize=True)
    rdm1.b.ov += 0.25 * np.einsum("abij,abcijk->kc", L.bb, R3.bbb, optimize=True)

    rdm2.aa.ooov = -0.5 * np.einsum("afin,bcmk,abcijk->mnjf", T.aa, T.aa, L3.aaa, optimize=True)
    rdm2.aa.ooov += 0.125 * np.einsum("abmn,ecjk,abcijk->mnie", T.aa, T.aa, L3.aaa, optimize=True)
    rdm2.aa.ooov -= 0.25 * np.einsum("bcmk,abcijk->jima", L.aa, R3.aaa, optimize=True)
    rdm2.aa.ooov -= 0.5 * np.einsum("fcnk,abim,abcijk->mnjf", T.ab, T.aa, L3.aab, optimize=True)
    rdm2.aa.ooov += 0.25 * np.einsum("abmn,ecjk,abcijk->mnie", T.aa, T.ab, L3.aab, optimize=True)
    rdm2.aa.ooov -= np.einsum("afin,bcmk,abcijk->mnjf", T.aa, T.ab, L3.aab, optimize=True)
    rdm2.aa.ooov -= 0.5 * np.einsum("bcmk,abcijk->jima", L.ab, R3.aab, optimize=True)
    rdm2.aa.ooov -= np.einsum("fani,cbmj,cabkij->mnkf", T.ab, T.ab, L3.abb, optimize=True)

    rdm2.aa.oovv = 0.25 * np.einsum("ck,abcijk->ijab", L.a, R3.aaa, optimize=True)
    rdm2.aa.oovv += 0.25 * np.einsum("ck,abcijk->ijab", L.b, R3.aab, optimize=True)

    rdm2.aa.vooo = -0.25 * np.einsum("bcmk,abcijk->amij", T.aa, L3.aaa, optimize=True)
    rdm2.aa.vooo -= 0.5 * np.einsum("bcmk,abcijk->amij", T.ab, L3.aab, optimize=True)

    rdm2.aa.vovv = -0.125 * np.einsum("efij,bcmk,abcijk->amef", T.aa, T.aa, L3.aaa, optimize=True)
    rdm2.aa.vovv += 0.5 * np.einsum("afin,ecjk,abcijk->bnef", T.aa, T.aa, L3.aaa, optimize=True)
    rdm2.aa.vovv += 0.25 * np.einsum("ecjk,abcijk->eiba", L.aa, R3.aaa, optimize=True)
    rdm2.aa.vovv += 0.5 * np.einsum("fcmk,aeij,abcijk->bmef", T.ab, T.aa, L3.aab, optimize=True)
    rdm2.aa.vovv += np.einsum("afin,ecjk,abcijk->bnef", T.aa, T.ab, L3.aab, optimize=True)
    rdm2.aa.vovv -= 0.25 * np.einsum("efij,bcmk,abcijk->amef", T.aa, T.ab, L3.aab, optimize=True)
    rdm2.aa.vovv += 0.5 * np.einsum("ecjk,abcijk->eiba", L.ab, R3.aab, optimize=True)
    rdm2.aa.vovv += np.einsum("fami,ebkj,cabkij->cmef", T.ab, T.ab, L3.abb, optimize=True)

    rdm2.aa.vvov = 0.25 * np.einsum("ecjk,abcijk->abie", T.aa, L3.aaa, optimize=True)
    rdm2.aa.vvov += 0.5 * np.einsum("ecjk,abcijk->abie", T.ab, L3.aab, optimize=True)

    rdm2.ab.ooov = -0.5 * np.einsum("afin,bcmk,abcijk->mnjf", T.ab, T.aa, L3.aaa, optimize=True)
    rdm2.ab.ooov -= 0.5 * np.einsum("fcnk,abim,abcijk->mnjf", T.bb, T.aa, L3.aab, optimize=True)
    rdm2.ab.ooov += np.einsum("acnm,bejk,abcijk->nmie", T.ab, T.ab, L3.aab, optimize=True)
    rdm2.ab.ooov += np.einsum("afnk,bcjm,abcijk->nmif", T.ab, T.ab, L3.aab, optimize=True)
    rdm2.ab.ooov -= np.einsum("afin,bcmk,abcijk->mnjf", T.ab, T.ab, L3.aab, optimize=True)
    rdm2.ab.ooov -= 0.5 * np.einsum("abim,abcijk->jkmc", L.aa, R3.aab, optimize=True)
    rdm2.ab.ooov += 0.5 * np.einsum("cbmn,aeij,cabkij->mnke", T.ab, T.bb, L3.abb, optimize=True)
    rdm2.ab.ooov += 0.5 * np.einsum("cfnj,abim,cabkij->nmkf", T.ab, T.bb, L3.abb, optimize=True)
    rdm2.ab.ooov -= np.einsum("afin,cbmj,cabkij->mnkf", T.bb, T.ab, L3.abb, optimize=True)
    rdm2.ab.ooov -= np.einsum("cbmj,cabkij->kima", L.ab, R3.abb, optimize=True)

    rdm2.ab.oovo = 0.5 * np.einsum("bcnm,aeij,abcijk->nmek", T.ab, T.aa, L3.aab, optimize=True)
    rdm2.ab.oovo += 0.5 * np.einsum("fcjn,abim,abcijk->mnfk", T.ab, T.aa, L3.aab, optimize=True)
    rdm2.ab.oovo -= np.einsum("afin,bcjm,abcijk->nmfk", T.aa, T.ab, L3.aab, optimize=True)
    rdm2.ab.oovo -= np.einsum("bcjm,abcijk->ikam", L.ab, R3.aab, optimize=True)
    rdm2.ab.oovo -= 0.5 * np.einsum("fcnk,abim,cabkij->nmfj", T.aa, T.bb, L3.abb, optimize=True)
    rdm2.ab.oovo += np.einsum("camn,ebkj,cabkij->mnei", T.ab, T.ab, L3.abb, optimize=True)
    rdm2.ab.oovo += np.einsum("fakn,cbmj,cabkij->mnfi", T.ab, T.ab, L3.abb, optimize=True)
    rdm2.ab.oovo -= np.einsum("fani,cbkm,cabkij->nmfj", T.ab, T.ab, L3.abb, optimize=True)
    rdm2.ab.oovo -= 0.5 * np.einsum("abim,cabkij->kjcm", L.bb, R3.abb, optimize=True)
    rdm2.ab.oovo -= 0.5 * np.einsum("fani,bcmk,abcijk->nmfj", T.ab, T.bb, L3.bbb, optimize=True)

    rdm2.ab.oovv = np.einsum("ai,abcijk->jkbc", L.a, R3.aab, optimize=True)
    rdm2.ab.oovv += np.einsum("ai,cabkij->kjcb", L.b, R3.abb, optimize=True)

    rdm2.ab.ovoo = -0.5 * np.einsum("abim,abcijk->mcjk", T.aa, L3.aab, optimize=True)
    rdm2.ab.ovoo -= np.einsum("cbmj,cabkij->maki", T.ab, L3.abb, optimize=True)

    rdm2.ab.ovvv = -0.5 * np.einsum("bfmk,aeij,abcijk->mcef", T.ab, T.aa, L3.aab, optimize=True)
    rdm2.ab.ovvv -= 0.5 * np.einsum("fejk,abim,abcijk->mcfe", T.ab, T.aa, L3.aab, optimize=True)
    rdm2.ab.ovvv += np.einsum("afim,bejk,abcijk->mcfe", T.aa, T.ab, L3.aab, optimize=True)
    rdm2.ab.ovvv += np.einsum("bejk,abcijk->ieac", L.ab, R3.aab, optimize=True)
    rdm2.ab.ovvv += 0.5 * np.einsum("cfkm,aeij,cabkij->mbfe", T.aa, T.bb, L3.abb, optimize=True)
    rdm2.ab.ovvv -= np.einsum("cfmi,ebkj,cabkij->maef", T.ab, T.ab, L3.abb, optimize=True)
    rdm2.ab.ovvv -= np.einsum("feki,cbmj,cabkij->mafe", T.ab, T.ab, L3.abb, optimize=True)
    rdm2.ab.ovvv += np.einsum("fani,cekj,cabkij->nbfe", T.ab, T.ab, L3.abb, optimize=True)
    rdm2.ab.ovvv += 0.5 * np.einsum("aeij,cabkij->kecb", L.bb, R3.abb, optimize=True)
    rdm2.ab.ovvv += 0.5 * np.einsum("fani,ecjk,abcijk->nbfe", T.ab, T.bb, L3.bbb, optimize=True)

    rdm2.ab.vooo = -np.einsum("bcjm,abcijk->amik", T.ab, L3.aab, optimize=True)
    rdm2.ab.vooo -= 0.5 * np.einsum("abim,cabkij->cmkj", T.bb, L3.abb, optimize=True)

    rdm2.ab.vovv = 0.5 * np.einsum("afin,ecjk,abcijk->bnef", T.ab, T.aa, L3.aaa, optimize=True)
    rdm2.ab.vovv += 0.5 * np.einsum("cfkm,aeij,abcijk->bmef", T.bb, T.aa, L3.aab, optimize=True)
    rdm2.ab.vovv -= np.einsum("fcim,bejk,abcijk->amfe", T.ab, T.ab, L3.aab, optimize=True)
    rdm2.ab.vovv -= np.einsum("efik,bcjm,abcijk->amef", T.ab, T.ab, L3.aab, optimize=True)
    rdm2.ab.vovv += np.einsum("afin,ecjk,abcijk->bnef", T.ab, T.ab, L3.aab, optimize=True)
    rdm2.ab.vovv += 0.5 * np.einsum("aeij,abcijk->ekbc", L.aa, R3.aab, optimize=True)
    rdm2.ab.vovv -= 0.5 * np.einsum("fbkm,aeij,cabkij->cmfe", T.ab, T.bb, L3.abb, optimize=True)
    rdm2.ab.vovv -= 0.5 * np.einsum("efkj,abim,cabkij->cmef", T.ab, T.bb, L3.abb, optimize=True)
    rdm2.ab.vovv += np.einsum("afim,ebkj,cabkij->cmef", T.bb, T.ab, L3.abb, optimize=True)
    rdm2.ab.vovv += np.einsum("ebkj,cabkij->eica", L.ab, R3.abb, optimize=True)

    rdm2.ab.vvov = np.einsum("bejk,abcijk->acie", T.ab, L3.aab, optimize=True)
    rdm2.ab.vvov += 0.5 * np.einsum("aeij,cabkij->cbke", T.bb, L3.abb, optimize=True)

    rdm2.ab.vvvo = 0.5 * np.einsum("aeij,abcijk->bcek", T.aa, L3.aab, optimize=True)
    rdm2.ab.vvvo += np.einsum("ebkj,cabkij->caei", T.ab, L3.abb, optimize=True)

    rdm2.bb.ooov = -np.einsum("afin,bcjm,abcijk->mnkf", T.ab, T.ab, L3.aab, optimize=True)
    rdm2.bb.ooov -= 0.5 * np.einsum("cfkn,abim,cabkij->mnjf", T.ab, T.bb, L3.abb, optimize=True)
    rdm2.bb.ooov += 0.25 * np.einsum("abmn,cekj,cabkij->mnie", T.bb, T.ab, L3.abb, optimize=True)
    rdm2.bb.ooov -= np.einsum("afin,cbkm,cabkij->mnjf", T.bb, T.ab, L3.abb, optimize=True)
    rdm2.bb.ooov -= 0.5 * np.einsum("cbkm,cabkij->jima", L.ab, R3.abb, optimize=True)
    rdm2.bb.ooov -= 0.5 * np.einsum("afin,bcmk,abcijk->mnjf", T.bb, T.bb, L3.bbb, optimize=True)
    rdm2.bb.ooov += 0.125 * np.einsum("abmn,ecjk,abcijk->mnie", T.bb, T.bb, L3.bbb, optimize=True)
    rdm2.bb.ooov -= 0.25 * np.einsum("bcmk,abcijk->jima", L.bb, R3.bbb, optimize=True)

    rdm2.bb.oovv = 0.25 * np.einsum("ck,cabkij->ijab", L.a, R3.abb, optimize=True)
    rdm2.bb.oovv += 0.25 * np.einsum("ck,abcijk->ijab", L.b, R3.bbb, optimize=True)

    rdm2.bb.vooo = -0.5 * np.einsum("cbkm,cabkij->amij", T.ab, L3.abb, optimize=True)
    rdm2.bb.vooo -= 0.25 * np.einsum("bcmk,abcijk->amij", T.bb, L3.bbb, optimize=True)

    rdm2.bb.vovv = np.einsum("afim,bejk,abcijk->cmef", T.ab, T.ab, L3.aab, optimize=True)
    rdm2.bb.vovv += 0.5 * np.einsum("cfkm,aeij,cabkij->bmef", T.ab, T.bb, L3.abb, optimize=True)
    rdm2.bb.vovv += np.einsum("afin,cekj,cabkij->bnef", T.bb, T.ab, L3.abb, optimize=True)
    rdm2.bb.vovv -= 0.25 * np.einsum("efij,cbkm,cabkij->amef", T.bb, T.ab, L3.abb, optimize=True)
    rdm2.bb.vovv += 0.5 * np.einsum("cekj,cabkij->eiba", L.ab, R3.abb, optimize=True)
    rdm2.bb.vovv -= 0.125 * np.einsum("efij,bcmk,abcijk->amef", T.bb, T.bb, L3.bbb, optimize=True)
    rdm2.bb.vovv += 0.5 * np.einsum("afin,ecjk,abcijk->bnef", T.bb, T.bb, L3.bbb, optimize=True)
    rdm2.bb.vovv += 0.25 * np.einsum("ecjk,abcijk->eiba", L.bb, R3.bbb, optimize=True)

    rdm2.bb.vvov = 0.5 * np.einsum("cekj,cabkij->abie", T.ab, L3.abb, optimize=True)
    rdm2.bb.vvov += 0.25 * np.einsum("ecjk,abcijk->abie", T.bb, L3.bbb, optimize=True)

    return rdm1, rdm2


def calc_crcc23_rdm1_denominator(R3, L3, system):
    """-d(sum_K l_K r_K D_K)/df_pp for the bare Fock diagonal (Eq. 9): occupied p gets
    -sum_{K contains p} l_K r_K, virtual p gets +sum_{K contains p} l_K r_K."""
    rdm1 = Integral.from_empty(system, 1, use_none=True)
    # sum over unique triples = (1/36) sum over aaa/bbb arrays, (1/4) over aab/abb arrays
    lr_aaa = (1.0 / 36.0) * R3.aaa * L3.aaa
    lr_aab = 0.25 * R3.aab * L3.aab
    lr_abb = 0.25 * R3.abb * L3.abb
    lr_bbb = (1.0 / 36.0) * R3.bbb * L3.bbb
    rdm1.a.oo = -np.diag(
        3.0 * np.einsum("abcijk->i", lr_aaa, optimize=True)
        + 2.0 * np.einsum("abcijk->i", lr_aab, optimize=True)
        + np.einsum("abcijk->i", lr_abb, optimize=True)
    )
    rdm1.a.vv = np.diag(
        3.0 * np.einsum("abcijk->a", lr_aaa, optimize=True)
        + 2.0 * np.einsum("abcijk->a", lr_aab, optimize=True)
        + np.einsum("abcijk->a", lr_abb, optimize=True)
    )
    rdm1.b.oo = -np.diag(
        np.einsum("abcijk->k", lr_aab, optimize=True)
        + 2.0 * np.einsum("abcijk->k", lr_abb, optimize=True)
        + 3.0 * np.einsum("abcijk->k", lr_bbb, optimize=True)
    )
    rdm1.b.vv = np.diag(
        np.einsum("abcijk->c", lr_aab, optimize=True)
        + 2.0 * np.einsum("abcijk->c", lr_abb, optimize=True)
        + 3.0 * np.einsum("abcijk->c", lr_bbb, optimize=True)
    )
    return rdm1
