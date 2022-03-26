import numpy as np
import time

from pyscf import gto, scf

from ccpy.models.operators import ClusterOperator

from ccpy.utilities.active_space import fill_t3aaa, fill_t3aab, fill_t3abb, fill_t3bbb, get_active_slices,\
    zero_t3aaa_outside_active_space, zero_t3bbb_outside_active_space, zero_t3aab_outside_active_space, zero_t3abb_outside_active_space

from ccpy.models.calculation import Calculation
from ccpy.interfaces.gamess_tools import load_from_gamess
from ccpy.interfaces.pyscf_tools import load_pyscf_integrals
from ccpy.drivers.driver import cc_driver, eomcc_driver

from ccpy.hbar.hbar_ccsdt import build_hbar_ccsdt, build_hbar_ccsdt1
from ccpy.hbar.eomccsdt_intermediates import get_eomccsd_intermediates, add_R3_terms
from ccpy.eomcc.eomccsdt import build_HR_3A, build_HR_3B, build_HR_3C, build_HR_3D, build_HR_1A, build_HR_1B, build_HR_2A, build_HR_2B, build_HR_2C

from ccpy.eomcc.initial_guess import get_initial_guess

from ccpy.eomcc import eomccsdt1

def check_intermediates(HR_act, HR):

    print('HR.a')
    for attr in [a for a in dir(HR.a) if not a.startswith('__')]:
        print('Error in ', attr, '= ', np.linalg.norm(getattr(HR.a, attr).flatten() - getattr(HR_act.a, attr).flatten()))
    print('HR.b')
    for attr in [a for a in dir(HR.b) if not a.startswith('__')]:
        print('Error in ', attr, '= ', np.linalg.norm(getattr(HR.b, attr).flatten() - getattr(HR_act.b, attr).flatten()))
    print('HR.aa')
    for attr in [a for a in dir(HR.aa) if not a.startswith('__')]:
        print('Error in ', attr, '= ', np.linalg.norm(getattr(HR.aa, attr).flatten() - getattr(HR_act.aa, attr).flatten()))
    print('HR.ab')
    for attr in [a for a in dir(HR.ab) if not a.startswith('__')]:
        print('Error in ', attr, '= ', np.linalg.norm(getattr(HR.ab, attr).flatten() - getattr(HR_act.ab, attr).flatten()))
    print('HR.bb')
    for attr in [a for a in dir(HR.bb) if not a.startswith('__')]:
        print('Error in ', attr, '= ', np.linalg.norm(getattr(HR.bb, attr).flatten() - getattr(HR_act.bb, attr).flatten()))

def calc_error(dT_act, dT, system):

    def _get_error(x, y):
        error = x - y
        return np.linalg.norm(error.flatten())

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    print('Error in R1.a')
    print('-------------------------')
    print("Total error = ", _get_error(dT_act.a, dT.a))

    print('Error in R1.b')
    print('-------------------------')
    print("Total error = ", _get_error(dT_act.b, dT.b))

    print('Error in R2.aa')
    print('-------------------------')
    print("Total error = ", _get_error(dT_act.aa, dT.aa))

    print('Error in R2.ab')
    print('-------------------------')
    print("Total error = ", _get_error(dT_act.ab, dT.ab))

    print('Error in R2.bb')
    print('-------------------------')
    print("Total error = ", _get_error(dT_act.bb, dT.bb))

    print('Error in R3.aaa')
    print('-------------------------')
    print("Error in VVVOOO = ", _get_error(dT_act.aaa.VVVOOO, dT.aaa[Va, Va, Va, Oa, Oa, Oa]))
    print("Error in VVvOOO = ", _get_error(dT_act.aaa.VVvOOO, dT.aaa[Va, Va, va, Oa, Oa, Oa]))
    print("Error in VVVoOO = ", _get_error(dT_act.aaa.VVVoOO, dT.aaa[Va, Va, Va, oa, Oa, Oa]))
    print("Error in VVvoOO = ", _get_error(dT_act.aaa.VVvoOO, dT.aaa[Va, Va, va, oa, Oa, Oa]))
    print("Error in VvvOOO = ", _get_error(dT_act.aaa.VvvOOO, dT.aaa[Va, va, va, Oa, Oa, Oa]))
    print("Error in VVVooO = ", _get_error(dT_act.aaa.VVVooO, dT.aaa[Va, Va, Va, oa, oa, Oa]))
    print("Error in VVvooO = ", _get_error(dT_act.aaa.VVvooO, dT.aaa[Va, Va, va, oa, oa, Oa]))
    print("Error in VvvoOO = ", _get_error(dT_act.aaa.VvvoOO, dT.aaa[Va, va, va, oa, Oa, Oa]))
    print("Error in VvvooO = ", _get_error(dT_act.aaa.VvvooO, dT.aaa[Va, va, va, oa, oa, Oa]))

    print('Error in R3.aab')
    print('-------------------------')
    print("Error in VVVOOO = ", _get_error(dT_act.aab.VVVOOO, dT.aab[Va, Va, Vb, Oa, Oa, Ob]))
    # one inactive
    print("Error in VVVOOo = ", _get_error(dT_act.aab.VVVOOo, dT.aab[Va, Va, Vb, Oa, Oa, ob]))
    print("Error in VvVOOO = ", _get_error(dT_act.aab.VvVOOO, dT.aab[Va, va, Vb, Oa, Oa, Ob]))
    print("Error in VVvOOO = ", _get_error(dT_act.aab.VVvOOO, dT.aab[Va, Va, vb, Oa, Oa, Ob]))
    print("Error in VVVoOO = ", _get_error(dT_act.aab.VVVoOO, dT.aab[Va, Va, Vb, oa, Oa, Ob]))
    # two inactive
    print("Error in VVVooO = ", _get_error(dT_act.aab.VVVooO, dT.aab[Va, Va, Vb, oa, oa, Ob]))
    print("Error in VVVoOo = ", _get_error(dT_act.aab.VVVoOo, dT.aab[Va, Va, Vb, oa, Oa, ob]))
    print("Error in vvVOOO = ", _get_error(dT_act.aab.vvVOOO, dT.aab[va, va, Vb, Oa, Oa, Ob]))
    print("Error in VvvOOO = ", _get_error(dT_act.aab.VvvOOO, dT.aab[Va, va, vb, Oa, Oa, Ob]))
    print("Error in VVvoOO = ", _get_error(dT_act.aab.VVvoOO, dT.aab[Va, Va, vb, oa, Oa, Ob]))
    print("Error in VvVOOo = ", _get_error(dT_act.aab.VvVOOo, dT.aab[Va, va, Vb, Oa, Oa, ob]))
    print("Error in VvVoOO = ", _get_error(dT_act.aab.VvVoOO, dT.aab[Va, va, Vb, oa, Oa, Ob]))
    print("Error in VVvOOo = ", _get_error(dT_act.aab.VVvOOo, dT.aab[Va, Va, vb, Oa, Oa, ob]))
    # three inactive
    print("Error in VvVooO = ", _get_error(dT_act.aab.VvVooO, dT.aab[Va, va, Vb, oa, oa, Ob]))
    print("Error in VvVoOo = ", _get_error(dT_act.aab.VvVoOo, dT.aab[Va, va, Vb, oa, Oa, ob]))
    print("Error in VVvooO = ", _get_error(dT_act.aab.VVvooO, dT.aab[Va, Va, vb, oa, oa, Ob]))
    print("Error in VVvoOo = ", _get_error(dT_act.aab.VVvoOo, dT.aab[Va, Va, vb, oa, Oa, ob]))
    print("Error in vvVoOO = ", _get_error(dT_act.aab.vvVoOO, dT.aab[va, va, Vb, oa, Oa, Ob]))
    print("Error in vvVOOo = ", _get_error(dT_act.aab.vvVOOo, dT.aab[va, va, Vb, Oa, Oa, ob]))
    print("Error in VvvoOO = ", _get_error(dT_act.aab.VvvoOO, dT.aab[Va, va, vb, oa, Oa, Ob]))
    print("Error in VvvOOo = ", _get_error(dT_act.aab.VvvOOo, dT.aab[Va, va, vb, Oa, Oa, ob]))
    # four inactive
    print("Error in vvVooO = ", _get_error(dT_act.aab.vvVooO, dT.aab[va, va, Vb, oa, oa, Ob]))
    print("Error in vvVoOo = ", _get_error(dT_act.aab.vvVoOo, dT.aab[va, va, Vb, oa, Oa, ob]))
    print("Error in VvvooO = ", _get_error(dT_act.aab.VvvooO, dT.aab[Va, va, vb, oa, oa, Ob]))
    print("Error in VvvoOo = ", _get_error(dT_act.aab.VvvoOo, dT.aab[Va, va, vb, oa, Oa, ob]))

    print('Error in R3.abb')
    print('-------------------------')
    print("Error in VVVOOO = ", _get_error(dT_act.abb.VVVOOO, dT.abb[Va, Vb, Vb, Oa, Ob, Ob]))
    print("Error in VVVOoO = ", _get_error(dT_act.abb.VVVOoO, dT.abb[Va, Vb, Vb, Oa, ob, Ob]))
    print("Error in VVVoOO = ", _get_error(dT_act.abb.VVVoOO, dT.abb[Va, Vb, Vb, oa, Ob, Ob]))
    print("Error in VVvOOO = ", _get_error(dT_act.abb.VVvOOO, dT.abb[Va, Vb, vb, Oa, Ob, Ob]))
    print("Error in vVVOOO = ", _get_error(dT_act.abb.vVVOOO, dT.abb[va, Vb, Vb, Oa, Ob, Ob]))
    print("Error in VVVOoo = ", _get_error(dT_act.abb.VVVOoo, dT.abb[Va, Vb, Vb, Oa, ob, ob]))
    print("Error in VVVooO = ", _get_error(dT_act.abb.VVVooO, dT.abb[Va, Vb, Vb, oa, ob, Ob]))
    print("Error in vVvOOO = ", _get_error(dT_act.abb.vVvOOO, dT.abb[va, Vb, vb, Oa, Ob, Ob]))
    print("Error in VvvOOO = ", _get_error(dT_act.abb.VvvOOO, dT.abb[Va, vb, vb, Oa, Ob, Ob]))
    print("Error in VVvoOO = ", _get_error(dT_act.abb.VVvoOO, dT.abb[Va, Vb, vb, oa, Ob, Ob]))
    print("Error in vVVOoO = ", _get_error(dT_act.abb.vVVOoO, dT.abb[va, Vb, Vb, Oa, ob, Ob]))
    print("Error in vVVoOO = ", _get_error(dT_act.abb.vVVoOO, dT.abb[va, Vb, Vb, oa, Ob, Ob]))
    print("Error in VVvOoO = ", _get_error(dT_act.abb.VVvOoO, dT.abb[Va, Vb, vb, Oa, ob, Ob]))
    print("Error in VVvOoo = ", _get_error(dT_act.abb.VVvOoo, dT.abb[Va, Vb, vb, Oa, ob, ob]))
    print("Error in VVvooO = ", _get_error(dT_act.abb.VVvooO, dT.abb[Va, Vb, vb, oa, ob, Ob]))
    print("Error in vVVOoo = ", _get_error(dT_act.abb.vVVOoo, dT.abb[va, Vb, Vb, Oa, ob, ob]))
    print("Error in vVVooO = ", _get_error(dT_act.abb.vVVooO, dT.abb[va, Vb, Vb, oa, ob, Ob]))
    print("Error in VvvoOO = ", _get_error(dT_act.abb.VvvoOO, dT.abb[Va, vb, vb, oa, Ob, Ob]))
    print("Error in VvvOoO = ", _get_error(dT_act.abb.VvvOoO, dT.abb[Va, vb, vb, Oa, ob, Ob]))
    print("Error in vVvoOO = ", _get_error(dT_act.abb.vVvoOO, dT.abb[va, Vb, vb, oa, Ob, Ob]))
    print("Error in vVvOoO = ", _get_error(dT_act.abb.vVvOoO, dT.abb[va, Vb, vb, Oa, ob, Ob]))
    print("Error in vVvOoo = ", _get_error(dT_act.abb.vVvOoo, dT.abb[va, Vb, vb, Oa, ob, ob]))
    print("Error in vVvooO = ", _get_error(dT_act.abb.vVvooO, dT.abb[va, Vb, vb, oa, ob, Ob]))
    print("Error in VvvOoo = ", _get_error(dT_act.abb.VvvOoo, dT.abb[Va, vb, vb, Oa, ob, ob]))
    print("Error in VvvooO = ", _get_error(dT_act.abb.VvvooO, dT.abb[Va, vb, vb, oa, ob, Ob]))

    print('Error in R3.bbb')
    print('-------------------------')
    print("Error in VVVOOO = ", _get_error(dT_act.bbb.VVVOOO, dT.bbb[Vb, Vb, Vb, Ob, Ob, Ob]))
    print("Error in VVvOOO = ", _get_error(dT_act.bbb.VVvOOO, dT.bbb[Vb, Vb, vb, Ob, Ob, Ob]))
    print("Error in VVVoOO = ", _get_error(dT_act.bbb.VVVoOO, dT.bbb[Vb, Vb, Vb, ob, Ob, Ob]))
    print("Error in VVvoOO = ", _get_error(dT_act.bbb.VVvoOO, dT.bbb[Vb, Vb, vb, ob, Ob, Ob]))
    print("Error in VvvOOO = ", _get_error(dT_act.bbb.VvvOOO, dT.bbb[Vb, vb, vb, Ob, Ob, Ob]))
    print("Error in VVVooO = ", _get_error(dT_act.bbb.VVVooO, dT.bbb[Vb, Vb, Vb, ob, ob, Ob]))
    print("Error in VVvooO = ", _get_error(dT_act.bbb.VVvooO, dT.bbb[Vb, Vb, vb, ob, ob, Ob]))
    print("Error in VvvoOO = ", _get_error(dT_act.bbb.VvvoOO, dT.bbb[Vb, vb, vb, ob, Ob, Ob]))
    print("Error in VvvooO = ", _get_error(dT_act.bbb.VvvooO, dT.bbb[Vb, vb, vb, ob, ob, Ob]))

def calc_full_HR1A(R, T, H):
    X1A = -np.einsum("mi,am->ai", H.a.oo, R.a, optimize=True)
    X1A += np.einsum("ae,ei->ai", H.a.vv, R.a, optimize=True)
    X1A += np.einsum("amie,em->ai", H.aa.voov, R.a, optimize=True)
    X1A += np.einsum("amie,em->ai", H.ab.voov, R.b, optimize=True)
    X1A -= 0.5 * np.einsum("mnif,afmn->ai", H.aa.ooov, R.aa, optimize=True)
    X1A -= np.einsum("mnif,afmn->ai", H.ab.ooov, R.ab, optimize=True)
    X1A += 0.5 * np.einsum("anef,efin->ai", H.aa.vovv, R.aa, optimize=True)
    X1A += np.einsum("anef,efin->ai", H.ab.vovv, R.ab, optimize=True)
    X1A += np.einsum("me,aeim->ai", H.a.ov, R.aa, optimize=True)
    X1A += np.einsum("me,aeim->ai", H.b.ov, R.ab, optimize=True)

    X1A += 0.25 * np.einsum("mnef,aefimn->ai", H.aa.oovv, R.aaa, optimize=True)
    X1A += np.einsum("mnef,aefimn->ai", H.ab.oovv, R.aab, optimize=True)
    X1A += 0.25 * np.einsum("mnef,aefimn->ai", H.bb.oovv, R.abb, optimize=True)

    return X1A

def calc_full_HR1B(R, T, H):
    X1B = -np.einsum("mi,am->ai", H.b.oo, R.b, optimize=True)
    X1B += np.einsum("ae,ei->ai", H.b.vv, R.b, optimize=True)
    X1B += np.einsum("maei,em->ai", H.ab.ovvo, R.a, optimize=True)
    X1B += np.einsum("amie,em->ai", H.bb.voov, R.b, optimize=True)
    X1B -= np.einsum("nmfi,fanm->ai", H.ab.oovo, R.ab, optimize=True)
    X1B -= 0.5 * np.einsum("mnif,afmn->ai", H.bb.ooov, R.bb, optimize=True)
    X1B += np.einsum("nafe,feni->ai", H.ab.ovvv, R.ab, optimize=True)
    X1B += 0.5 * np.einsum("anef,efin->ai", H.bb.vovv, R.bb, optimize=True)
    X1B += np.einsum("me,eami->ai", H.a.ov, R.ab, optimize=True)
    X1B += np.einsum("me,aeim->ai", H.b.ov, R.bb, optimize=True)

    X1B += 0.25 * np.einsum("mnef,efamni->ai", H.aa.oovv, R.aab, optimize=True)
    X1B += np.einsum("mnef,efamni->ai", H.ab.oovv, R.abb, optimize=True)
    X1B += 0.25 * np.einsum("mnef,aefimn->ai", H.bb.oovv, R.bbb, optimize=True)

    return X1B

def calc_full_HR2A(R, T, H, X):
    D1 = -np.einsum("mi,abmj->abij", H.a.oo, R.aa, optimize=True)  # A(ij)
    D2 = np.einsum("ae,ebij->abij", H.a.vv, R.aa, optimize=True)  # A(ab)
    X2A = 0.5 * np.einsum("mnij,abmn->abij", H.aa.oooo, R.aa, optimize=True)
    X2A += 0.5 * np.einsum("abef,efij->abij", H.aa.vvvv, R.aa, optimize=True)
    D3 = np.einsum("amie,ebmj->abij", H.aa.voov, R.aa, optimize=True)  # A(ij)A(ab)
    D4 = np.einsum("amie,bejm->abij", H.ab.voov, R.ab, optimize=True)  # A(ij)A(ab)
    D5 = -np.einsum("bmji,am->abij", H.aa.vooo, R.a, optimize=True)  # A(ab)
    D6 = np.einsum("baje,ei->abij", H.aa.vvov, R.a, optimize=True)  # A(ij)

    Q1 = -0.5 * np.einsum("mnef,bfmn->eb", H.aa.oovv, R.aa, optimize=True)
    D7 = np.einsum("eb,aeij->abij", Q1, T.aa, optimize=True)  # A(ab)
    Q2 = -np.einsum("mnef,bfmn->eb", H.ab.oovv, R.ab, optimize=True)
    D8 = np.einsum("eb,aeij->abij", Q2, T.aa, optimize=True)  # A(ab)

    Q1 = 0.5 * np.einsum("mnef,efjn->mj", H.aa.oovv, R.aa, optimize=True)
    D9 = -np.einsum("mj,abim->abij", Q1, T.aa, optimize=True)  # A(ij)
    Q2 = np.einsum("mnef,efjn->mj", H.ab.oovv, R.ab, optimize=True)
    D10 = -np.einsum("mj,abim->abij", Q2, T.aa, optimize=True)  # A(ij)

    Q1 = np.einsum("amfe,em->af", H.aa.vovv, R.a, optimize=True)
    D11 = np.einsum("af,fbij->abij", Q1, T.aa, optimize=True)  # A(ab)
    Q2 = np.einsum("nmie,em->ni", H.aa.ooov, R.a, optimize=True)
    D12 = -np.einsum("ni,abnj->abij", Q2, T.aa, optimize=True)  # A(ij)

    Q1 = np.einsum("amfe,em->af", H.ab.vovv, R.b, optimize=True)
    D13 = np.einsum("af,fbij->abij", Q1, T.aa, optimize=True)  # A(ab)
    Q2 = np.einsum("nmie,em->ni", H.ab.ooov, R.b, optimize=True)
    D14 = -np.einsum("ni,abnj->abij", Q2, T.aa, optimize=True)  # A(ij)

    # Parts contracted with T3
    X2A += np.einsum("me,abeijm->abij", X.a.ov, T.aaa, optimize=True)
    X2A += np.einsum("me,abeijm->abij", X.b.ov, T.aab, optimize=True)

    DR3_1 = np.einsum("me,abeijm->abij", H.a.ov, R.aaa, optimize=True)
    DR3_2 = np.einsum("me,abeijm->abij", H.b.ov, R.aab, optimize=True)
    DR3_3 = -0.5 * np.einsum("mnjf,abfimn->abij", H.aa.ooov, R.aaa, optimize=True)
    DR3_4 = -1.0 * np.einsum("mnjf,abfimn->abij", H.ab.ooov, R.aab, optimize=True)
    DR3_5 = 0.5 * np.einsum("bnef,aefijn->abij", H.aa.vovv, R.aaa, optimize=True)
    DR3_6 = np.einsum("bnef,aefijn->abij", H.ab.vovv, R.aab, optimize=True)

    D_ij = D1 + D6 + D9 + D10 + D12 + D14 + DR3_3 + DR3_4
    D_ab = D2 + D5 + D7 + D8 + D11 + D13 + DR3_5 + DR3_6
    D_abij = D3 + D4

    D_ij -= np.einsum("abij->abji", D_ij, optimize=True)
    D_ab -= np.einsum("abij->baij", D_ab, optimize=True)
    D_abij += (
            -np.einsum("abij->baij", D_abij, optimize=True)
            - np.einsum("abij->abji", D_abij, optimize=True)
            + np.einsum("abij->baji", D_abij, optimize=True)
    )

    X2A += D_ij + D_ab + D_abij + DR3_1 + DR3_2

    return X2A


def calc_full_HR2B(R, T, H, X):
    X2B = np.einsum("ae,ebij->abij", H.a.vv, R.ab, optimize=True)
    X2B += np.einsum("be,aeij->abij", H.b.vv, R.ab, optimize=True)
    X2B -= np.einsum("mi,abmj->abij", H.a.oo, R.ab, optimize=True)
    X2B -= np.einsum("mj,abim->abij", H.b.oo, R.ab, optimize=True)
    X2B += np.einsum("mnij,abmn->abij", H.ab.oooo, R.ab, optimize=True)
    X2B += np.einsum("abef,efij->abij", H.ab.vvvv, R.ab, optimize=True)
    X2B += np.einsum("amie,ebmj->abij", H.aa.voov, R.ab, optimize=True)
    X2B += np.einsum("amie,ebmj->abij", H.ab.voov, R.bb, optimize=True)
    X2B += np.einsum("mbej,aeim->abij", H.ab.ovvo, R.aa, optimize=True)
    X2B += np.einsum("bmje,aeim->abij", H.bb.voov, R.ab, optimize=True)
    X2B -= np.einsum("mbie,aemj->abij", H.ab.ovov, R.ab, optimize=True)
    X2B -= np.einsum("amej,ebim->abij", H.ab.vovo, R.ab, optimize=True)
    X2B += np.einsum("abej,ei->abij", H.ab.vvvo, R.a, optimize=True)
    X2B += np.einsum("abie,ej->abij", H.ab.vvov, R.b, optimize=True)
    X2B -= np.einsum("mbij,am->abij", H.ab.ovoo, R.a, optimize=True)
    X2B -= np.einsum("amij,bm->abij", H.ab.vooo, R.b, optimize=True)

    Q1 = -0.5 * np.einsum("mnef,afmn->ae", H.aa.oovv, R.aa, optimize=True)
    X2B += np.einsum("ae,ebij->abij", Q1, T.ab, optimize=True)
    Q2 = 0.5 * np.einsum("mnef,efin->mi", H.aa.oovv, R.aa, optimize=True)
    X2B -= np.einsum("mi,abmj->abij", Q2, T.ab, optimize=True)

    Q1 = -np.einsum("nmfe,fbnm->be", H.ab.oovv, R.ab, optimize=True)
    X2B += np.einsum("be,aeij->abij", Q1, T.ab, optimize=True)
    Q2 = -np.einsum("mnef,afmn->ae", H.ab.oovv, R.ab, optimize=True)
    X2B += np.einsum("ae,ebij->abij", Q2, T.ab, optimize=True)
    Q3 = np.einsum("nmfe,fenj->mj", H.ab.oovv, R.ab, optimize=True)
    X2B -= np.einsum("mj,abim->abij", Q3, T.ab, optimize=True)
    Q4 = np.einsum("mnef,efin->mi", H.ab.oovv, R.ab, optimize=True)
    X2B -= np.einsum("mi,abmj->abij", Q4, T.ab, optimize=True)

    Q1 = -0.5 * np.einsum("mnef,bfmn->be", H.bb.oovv, R.bb, optimize=True)
    X2B += np.einsum("be,aeij->abij", Q1, T.ab, optimize=True)
    Q2 = 0.5 * np.einsum("mnef,efjn->mj", H.bb.oovv, R.bb, optimize=True)
    X2B -= np.einsum("mj,abim->abij", Q2, T.ab, optimize=True)

    Q1 = np.einsum("mbef,em->bf", H.ab.ovvv, R.a, optimize=True)
    X2B += np.einsum("bf,afij->abij", Q1, T.ab, optimize=True)
    Q2 = np.einsum("mnej,em->nj", H.ab.oovo, R.a, optimize=True)
    X2B -= np.einsum("nj,abin->abij", Q2, T.ab, optimize=True)
    Q3 = np.einsum("amfe,em->af", H.aa.vovv, R.a, optimize=True)
    X2B += np.einsum("af,fbij->abij", Q3, T.ab, optimize=True)
    Q4 = np.einsum("nmie,em->ni", H.aa.ooov, R.a, optimize=True)
    X2B -= np.einsum("ni,abnj->abij", Q4, T.ab, optimize=True)

    Q1 = np.einsum("amfe,em->af", H.ab.vovv, R.b, optimize=True)
    X2B += np.einsum("af,fbij->abij", Q1, T.ab, optimize=True)
    Q2 = np.einsum("nmie,em->ni", H.ab.ooov, R.b, optimize=True)
    X2B -= np.einsum("ni,abnj->abij", Q2, T.ab, optimize=True)
    Q3 = np.einsum("bmfe,em->bf", H.bb.vovv, R.b, optimize=True)
    X2B += np.einsum("bf,afij->abij", Q3, T.ab, optimize=True)
    Q4 = np.einsum("nmje,em->nj", H.bb.ooov, R.b, optimize=True)
    X2B -= np.einsum("nj,abin->abij", Q4, T.ab, optimize=True)

    # Parts contracted with T3
    X2B += np.einsum("me,aebimj->abij", X.a.ov, T.aab, optimize=True)
    X2B += np.einsum("me,aebimj->abij", X.b.ov, T.abb, optimize=True)

    X2B += np.einsum("me,aebimj->abij", H.a.ov, R.aab, optimize=True)
    X2B += np.einsum("me,aebimj->abij", H.b.ov, R.abb, optimize=True)
    X2B -= np.einsum("nmfj,afbinm->abij", H.ab.oovo, R.aab, optimize=True)
    X2B -= 0.5 * np.einsum("mnjf,abfimn->abij", H.bb.ooov, R.abb, optimize=True)
    X2B -= 0.5 * np.einsum("mnif,afbmnj->abij", H.aa.ooov, R.aab, optimize=True)
    X2B -= np.einsum("mnif,abfmjn->abij", H.ab.ooov, R.abb, optimize=True)
    X2B += np.einsum("nbfe,afeinj->abij", H.ab.ovvv, R.aab, optimize=True)
    X2B += 0.5 * np.einsum("bnef,aefijn->abij", H.bb.vovv, R.abb, optimize=True)
    X2B += 0.5 * np.einsum("anef,efbinj->abij", H.aa.vovv, R.aab, optimize=True)
    X2B += np.einsum("anef,efbinj->abij", H.ab.vovv, R.abb, optimize=True)

    return X2B


def calc_full_HR2C(R, T, H, X):
    D1 = -np.einsum("mi,abmj->abij", H.b.oo, R.bb, optimize=True)  # A(ij)
    D2 = np.einsum("ae,ebij->abij", H.b.vv, R.bb, optimize=True)  # A(ab)
    X2C = 0.5 * np.einsum("mnij,abmn->abij", H.bb.oooo, R.bb, optimize=True)
    X2C += 0.5 * np.einsum("abef,efij->abij", H.bb.vvvv, R.bb, optimize=True)
    D3 = np.einsum("amie,ebmj->abij", H.bb.voov, R.bb, optimize=True)  # A(ij)A(ab)
    D4 = np.einsum("maei,ebmj->abij", H.ab.ovvo, R.ab, optimize=True)  # A(ij)A(ab)
    D5 = -np.einsum("bmji,am->abij", H.bb.vooo, R.b, optimize=True)  # A(ab)
    D6 = np.einsum("baje,ei->abij", H.bb.vvov, R.b, optimize=True)  # A(ij)

    Q1 = -0.5 * np.einsum("mnef,bfmn->eb", H.bb.oovv, R.bb, optimize=True)
    D7 = np.einsum("eb,aeij->abij", Q1, T.bb, optimize=True)  # A(ab)
    Q2 = -np.einsum("nmfe,fbnm->eb", H.ab.oovv, R.ab, optimize=True)
    D8 = np.einsum("eb,aeij->abij", Q2, T.bb, optimize=True)  # A(ab)

    Q1 = 0.5 * np.einsum("mnef,efjn->mj", H.bb.oovv, R.bb, optimize=True)
    D9 = -np.einsum("mj,abim->abij", Q1, T.bb, optimize=True)  # A(ij)
    Q2 = np.einsum("nmfe,fenj->mj", H.ab.oovv, R.ab, optimize=True)
    D10 = -np.einsum("mj,abim->abij", Q2, T.bb, optimize=True)  # A(ij)

    Q1 = np.einsum("amfe,em->af", H.bb.vovv, R.b, optimize=True)
    D11 = np.einsum("af,fbij->abij", Q1, T.bb, optimize=True)  # A(ab)
    Q2 = np.einsum("nmie,em->ni", H.bb.ooov, R.b, optimize=True)
    D12 = -np.einsum("ni,abnj->abij", Q2, T.bb, optimize=True)  # A(ij)

    Q1 = np.einsum("maef,em->af", H.ab.ovvv, R.a, optimize=True)
    D13 = np.einsum("af,fbij->abij", Q1, T.bb, optimize=True)  # A(ab)
    Q2 = np.einsum("mnei,em->ni", H.ab.oovo, R.a, optimize=True)
    D14 = -np.einsum("ni,abnj->abij", Q2, T.bb, optimize=True)  # A(ij)

    # Parts contracted with T3
    X2C += np.einsum("me,eabmij->abij", X.a.ov, T.abb, optimize=True)
    X2C += np.einsum("me,abeijm->abij", X.b.ov, T.bbb, optimize=True)

    DR3_1 = np.einsum("me,eabmij->abij", H.a.ov, R.abb, optimize=True)
    DR3_2 = np.einsum("me,abeijm->abij", H.b.ov, R.bbb, optimize=True)
    DR3_3 = -0.5 * np.einsum("mnjf,abfimn->abij", H.bb.ooov, R.bbb, optimize=True)
    DR3_4 = -1.0 * np.einsum("nmfj,fabnim->abij", H.ab.oovo, R.abb, optimize=True)
    DR3_5 = 0.5 * np.einsum("bnef,aefijn->abij", H.bb.vovv, R.bbb, optimize=True)
    DR3_6 = np.einsum("nbfe,faenij->abij", H.ab.ovvv, R.abb, optimize=True)

    D_ij = D1 + D6 + D9 + D10 + D12 + D14 + DR3_3 + DR3_4
    D_ab = D2 + D5 + D7 + D8 + D11 + D13 + DR3_5 + DR3_6
    D_abij = D3 + D4

    D_ij -= np.einsum("abij->abji", D_ij, optimize=True)
    D_ab -= np.einsum("abij->baij", D_ab, optimize=True)
    D_abij += (
            -np.einsum("abij->baij", D_abij, optimize=True)
            - np.einsum("abij->abji", D_abij, optimize=True)
            + np.einsum("abij->baji", D_abij, optimize=True)
    )

    X2C += D_ij + D_ab + D_abij + DR3_1 + DR3_2

    return X2C

def calc_full_HR3A(R, T, H, X):
    # <ijkabc| [H(R1+R2)]_C | 0 >
    X3A = 0.25 * np.einsum("baje,ecik->abcijk", X.aa.vvov, T.aa, optimize=True)
    X3A += 0.25 * np.einsum("baje,ecik->abcijk", H.aa.vvov, R.aa, optimize=True)

    X3A -= 0.25 * np.einsum("bmji,acmk->abcijk", X.aa.vooo, T.aa, optimize=True)
    X3A -= 0.25 * np.einsum("bmji,acmk->abcijk", H.aa.vooo, R.aa, optimize=True)

    # additional terms with T3 in <ijkabc|[ H(R1+R2)]_C | 0>
    X3A += (1.0 / 12.0) * np.einsum("be,aecijk->abcijk", X.a.vv, T.aaa, optimize=True)
    X3A -= (1.0 / 12.0) * np.einsum("mj,abcimk->abcijk", X.a.oo, T.aaa, optimize=True)
    X3A += (1.0 / 24.0) * np.einsum("mnij,abcmnk->abcijk", X.aa.oooo, T.aaa, optimize=True)
    X3A += (1.0 / 24.0) * np.einsum("abef,efcijk->abcijk", X.aa.vvvv, T.aaa, optimize=True)
    X3A += 0.25 * np.einsum("bmje,aecimk->abcijk", X.aa.voov, T.aaa, optimize=True)
    X3A += 0.25 * np.einsum("bmje,aceikm->abcijk", X.ab.voov, T.aab, optimize=True)

    # < ijkabc | (HR3)_C | 0 >
    X3A -= (1.0 / 12.0) * np.einsum("mj,abcimk->abcijk", H.a.oo, R.aaa, optimize=True)
    X3A += (1.0 / 12.0) * np.einsum("be,aecijk->abcijk", H.a.vv, R.aaa, optimize=True)
    X3A += (1.0 / 24.0) * np.einsum("mnij,abcmnk->abcijk", H.aa.oooo, R.aaa, optimize=True)
    X3A += (1.0 / 24.0) * np.einsum("abef,efcijk->abcijk", H.aa.vvvv, R.aaa, optimize=True)
    X3A += 0.25 * np.einsum("amie,ebcmjk->abcijk", H.aa.voov, R.aaa, optimize=True)
    X3A += 0.25 * np.einsum("amie,bcejkm->abcijk", H.ab.voov, R.aab, optimize=True)

    # antisymmetrize terms and add up: A(abc)A(ijk) = A(a/bc)A(bc)A(i/jk)A(jk)
    X3A -= np.transpose(X3A, (0, 1, 2, 3, 5, 4))
    X3A -= np.transpose(X3A, (0, 1, 2, 4, 3, 5)) + np.transpose(X3A, (0, 1, 2, 5, 4, 3))
    X3A -= np.transpose(X3A, (0, 2, 1, 3, 4, 5))
    X3A -= np.transpose(X3A, (1, 0, 2, 3, 4, 5)) + np.transpose(X3A, (2, 1, 0, 3, 4, 5))

    return X3A

def calc_full_HR3B(R, T, H, X):
    # < ijk~abc~ | [ H(R1+R2) ]_C | 0 >
    # Intermediate 1: X2B(bcek)*Y2A(aeij) -> Z3B(abcijk)
    X3B = 0.5 * np.einsum("bcek,aeij->abcijk", X.ab.vvvo, T.aa, optimize=True)
    X3B += 0.5 * np.einsum("bcek,aeij->abcijk", H.ab.vvvo, R.aa, optimize=True)
    # Intermediate 2: X2B(ncjk)*Y2A(abin) -> Z3B(abcijk)
    X3B -= 0.5 * np.einsum("ncjk,abin->abcijk", X.ab.ovoo, T.aa, optimize=True)
    X3B -= 0.5 * np.einsum("mcjk,abim->abcijk", H.ab.ovoo, R.aa, optimize=True)
    # Intermediate 3: X2A(baje)*Y2B(ecik) -> Z3B(abcijk)
    X3B += 0.5 * np.einsum("baje,ecik->abcijk", X.aa.vvov, T.ab, optimize=True)
    X3B += 0.5 * np.einsum("baje,ecik->abcijk", H.aa.vvov, R.ab, optimize=True)
    # Intermediate 4: X2A(bnji)*Y2B(acnk) -> Z3B(abcijk)
    X3B -= 0.5 * np.einsum("bnji,acnk->abcijk", X.aa.vooo, T.ab, optimize=True)
    X3B -= 0.5 * np.einsum("bnji,acnk->abcijk", H.aa.vooo, R.ab, optimize=True)
    # Intermediate 5: X2B(bcje)*Y2B(aeik) -> Z3B(abcijk)
    X3B += np.einsum("bcje,aeik->abcijk", X.ab.vvov, T.ab, optimize=True)
    X3B += np.einsum("bcje,aeik->abcijk", H.ab.vvov, R.ab, optimize=True)
    # Intermediate 6: X2B(bnjk)*Y2B(acin) -> Z3B(abcijk)
    X3B -= np.einsum("bnjk,acin->abcijk", X.ab.vooo, T.ab, optimize=True)
    X3B -= np.einsum("bnjk,acin->abcijk", H.ab.vooo, R.ab, optimize=True)

    # additional terms with T3 (these contractions mirror the form of
    # the ones with R3 later on)
    X3B += 0.5 * np.einsum("be,aecijk->abcijk", X.a.vv, T.aab, optimize=True)
    X3B += 0.25 * np.einsum("ce,abeijk->abcijk", X.b.vv, T.aab, optimize=True)
    X3B -= 0.5 * np.einsum("mj,abcimk->abcijk", X.a.oo, T.aab, optimize=True)
    X3B -= 0.25 * np.einsum("mk,abcijm->abcijk", X.b.oo, T.aab, optimize=True)
    X3B += 0.5 * np.einsum("nmjk,abcinm->abcijk", X.ab.oooo, T.aab, optimize=True)
    X3B += 0.125 * np.einsum("mnij,abcmnk->abcijk", X.aa.oooo, T.aab, optimize=True)
    X3B += 0.5 * np.einsum("bcfe,afeijk->abcijk", X.ab.vvvv, T.aab, optimize=True)
    X3B += 0.125 * np.einsum("abef,efcijk->abcijk", X.aa.vvvv, T.aab, optimize=True)
    X3B += 0.25 * np.einsum("ncfk,abfijn->abcijk", X.ab.ovvo, T.aaa, optimize=True)
    X3B += 0.25 * np.einsum("cnkf,abfijn->abcijk", X.bb.voov, T.aab, optimize=True)
    X3B -= 0.5 * np.einsum("bmfk,afcijm->abcijk", X.ab.vovo, T.aab, optimize=True)
    X3B -= 0.5 * np.einsum("ncje,abeink->abcijk", X.ab.ovov, T.aab, optimize=True)
    X3B += np.einsum("bmje,aecimk->abcijk", X.aa.voov, T.aab, optimize=True)
    X3B += np.einsum("bmje,aecimk->abcijk", X.ab.voov, T.abb, optimize=True)

    # < ijk~abc~ | (HR3)_C | 0 >
    X3B -= 0.5 * np.einsum("mj,abcimk->abcijk", H.a.oo, R.aab, optimize=True)
    X3B -= 0.25 * np.einsum("mk,abcijm->abcijk", H.b.oo, R.aab, optimize=True)
    X3B += 0.5 * np.einsum("be,aecijk->abcijk", H.a.vv, R.aab, optimize=True)
    X3B += 0.25 * np.einsum("ce,abeijk->abcijk", H.b.vv, R.aab, optimize=True)
    X3B += 0.125 * np.einsum("mnij,abcmnk->abcijk", H.aa.oooo, R.aab, optimize=True)
    X3B += 0.5 * np.einsum("mnjk,abcimn->abcijk", H.ab.oooo, R.aab, optimize=True)
    X3B += 0.125 * np.einsum("abef,efcijk->abcijk", H.aa.vvvv, R.aab, optimize=True)
    X3B += 0.5 * np.einsum("bcef,aefijk->abcijk", H.ab.vvvv, R.aab, optimize=True)
    X3B += np.einsum("amie,ebcmjk->abcijk", H.aa.voov, R.aab, optimize=True)
    X3B += np.einsum("amie,becjmk->abcijk", H.ab.voov, R.abb, optimize=True)
    X3B += 0.25 * np.einsum("mcek,abeijm->abcijk", H.ab.ovvo, R.aaa, optimize=True)
    X3B += 0.25 * np.einsum("cmke,abeijm->abcijk", H.bb.voov, R.aab, optimize=True)
    X3B -= 0.5 * np.einsum("bmek,aecijm->abcijk", H.ab.vovo, R.aab, optimize=True)
    X3B -= 0.5 * np.einsum("mcje,abeimk->abcijk", H.ab.ovov, R.aab, optimize=True)

    X3B -= (
            np.transpose(X3B, (0, 1, 2, 4, 3, 5))
            + np.transpose(X3B, (1, 0, 2, 3, 4, 5))
            - np.transpose(X3B, (1, 0, 2, 4, 3, 5))
    )
    return X3B

def calc_full_HR3C(R, T, H, X):
    # < ij~k~ab~c~ | [ H(R1+R2) ]_C | 0 >
    # Intermediate 1: X2B(cbke)*Y2C(aeij) -> Z3C(cbakji)
    X3C = 0.5 * np.einsum("cbke,aeij->cbakji", X.ab.vvov, T.bb, optimize=True)
    X3C += 0.5 * np.einsum("cbke,aeij->cbakji", H.ab.vvov, R.bb, optimize=True)
    # Intermediate 2: X2B(cnkj)*Y2C(abin) -> Z3C(cbakji)
    X3C -= 0.5 * np.einsum("cnkj,abin->cbakji", X.ab.vooo, T.bb, optimize=True)
    X3C -= 0.5 * np.einsum("cmkj,abim->cbakji", H.ab.vooo, R.bb, optimize=True)
    # Intermediate 3: X2C(baje)*Y2B(ceki) -> Z3C(cbakji)
    X3C += 0.5 * np.einsum("baje,ceki->cbakji", X.bb.vvov, T.ab, optimize=True)
    X3C += 0.5 * np.einsum("baje,ceki->cbakji", H.bb.vvov, R.ab, optimize=True)
    # Intermediate 4: X2C(bnji)*Y2B(cakn) -> Z3C(cbakji)
    X3C -= 0.5 * np.einsum("bnji,cakn->cbakji", X.bb.vooo, T.ab, optimize=True)
    X3C -= 0.5 * np.einsum("bnji,cakn->cbakji", H.bb.vooo, R.ab, optimize=True)
    # Intermediate 5: X2B(cbej)*Y2B(eaki) -> Z3C(cbakji)
    X3C += np.einsum("cbej,eaki->cbakji", X.ab.vvvo, T.ab, optimize=True)
    X3C += np.einsum("cbej,eaki->cbakji", H.ab.vvvo, R.ab, optimize=True)
    # Intermediate 6: X2B(nbkj)*Y2B(cani) -> Z3C(cbakji)
    X3C -= np.einsum("nbkj,cani->cbakji", X.ab.ovoo, T.ab, optimize=True)
    X3C -= np.einsum("nbkj,cani->cbakji", H.ab.ovoo, R.ab, optimize=True)

    # additional terms with T3
    X3C += 0.5 * np.einsum("be,ceakji->cbakji", X.b.vv, T.abb, optimize=True)
    X3C += 0.25 * np.einsum("ce,ebakji->cbakji", X.a.vv, T.abb, optimize=True)
    X3C -= 0.5 * np.einsum("mj,cbakmi->cbakji", X.b.oo, T.abb, optimize=True)
    X3C -= 0.25 * np.einsum("mk,cbamji->cbakji", X.a.oo, T.abb, optimize=True)
    X3C += 0.5 * np.einsum("mnkj,cbamni->cbakji", X.ab.oooo, T.abb, optimize=True)
    X3C += 0.125 * np.einsum("mnij,cbaknm->cbakji", X.bb.oooo, T.abb, optimize=True)
    X3C += 0.5 * np.einsum("cbef,efakji->cbakji", X.ab.vvvv, T.abb, optimize=True)
    X3C += 0.125 * np.einsum("abef,cfekji->cbakji", X.bb.vvvv, T.abb, optimize=True)
    X3C += 0.25 * np.einsum("cnkf,abfijn->cbakji", X.ab.voov, T.bbb, optimize=True)
    X3C += 0.25 * np.einsum("cnkf,fbanji->cbakji", X.aa.voov, T.abb, optimize=True)
    X3C -= 0.5 * np.einsum("mbkf,cfamji->cbakji", X.ab.ovov, T.abb, optimize=True)
    X3C -= 0.5 * np.einsum("cnej,ebakni->cbakji", X.ab.vovo, T.abb, optimize=True)
    X3C += np.einsum("bmje,ceakmi->cbakji", X.bb.voov, T.abb, optimize=True)
    X3C += np.einsum("mbej,ceakmi->cbakji", X.ab.ovvo, T.aab, optimize=True)

    # < ijk~abc~ | (HR3)_C | 0 >
    X3C -= 0.5 * np.einsum("mj,cbakmi->cbakji", H.b.oo, R.abb, optimize=True)
    X3C -= 0.25 * np.einsum("mk,cbamji->cbakji", H.a.oo, R.abb, optimize=True)
    X3C += 0.5 * np.einsum("be,ceakji->cbakji", H.b.vv, R.abb, optimize=True)
    X3C += 0.25 * np.einsum("ce,ebakji->cbakji", H.a.vv, R.abb, optimize=True)
    X3C += 0.125 * np.einsum("mnij,cbaknm->cbakji", H.bb.oooo, R.abb, optimize=True)
    X3C += 0.5 * np.einsum("nmkj,cbanmi->cbakji", H.ab.oooo, R.abb, optimize=True)
    X3C += 0.125 * np.einsum("abef,cfekji->cbakji", H.bb.vvvv, R.abb, optimize=True)
    X3C += 0.5 * np.einsum("cbfe,feakji->cbakji", H.ab.vvvv, R.abb, optimize=True)
    X3C += np.einsum("amie,cbekjm->cbakji", H.bb.voov, R.abb, optimize=True)
    X3C += np.einsum("maei,cebkmj->cbakji", H.ab.ovvo, R.aab, optimize=True)
    X3C += 0.25 * np.einsum("cmke,ebamji->cbakji", H.ab.voov, R.bbb, optimize=True)
    X3C += 0.25 * np.einsum("cmke,ebamji->cbakji", H.aa.voov, R.abb, optimize=True)
    X3C -= 0.5 * np.einsum("mbke,ceamji->cbakji", H.ab.ovov, R.abb, optimize=True)
    X3C -= 0.5 * np.einsum("cmej,ebakmi->cbakji", H.ab.vovo, R.abb, optimize=True)

    X3C -= (
            np.transpose(X3C, (0, 1, 2, 3, 5, 4))
            + np.transpose(X3C, (0, 2, 1, 3, 4, 5))
            - np.transpose(X3C, (0, 2, 1, 3, 5, 4))
    )
    return X3C

def calc_full_HR3D(R, T, H, X):
    # <i~j~k~a~b~c~| [H(R1+R2)]_C | 0 >
    X3D = 0.25 * np.einsum("baje,ecik->abcijk", X.bb.vvov, T.bb, optimize=True)
    X3D += 0.25 * np.einsum("baje,ecik->abcijk", H.bb.vvov, R.bb, optimize=True)

    X3D -= 0.25 * np.einsum("bmji,acmk->abcijk", X.bb.vooo, T.bb, optimize=True)
    X3D -= 0.25 * np.einsum("bmji,acmk->abcijk", H.bb.vooo, R.bb, optimize=True)

    # additional terms with T3 in <ijkabc|[ H(R1+R2)]_C | 0>
    X3D += (1.0 / 12.0) * np.einsum("be,aecijk->abcijk", X.b.vv, T.bbb, optimize=True)
    X3D -= (1.0 / 12.0) * np.einsum("mj,abcimk->abcijk", X.b.oo, T.bbb, optimize=True)
    X3D += (1.0 / 24.0) * np.einsum("mnij,abcmnk->abcijk", X.bb.oooo, T.bbb, optimize=True)
    X3D += (1.0 / 24.0) * np.einsum("abef,efcijk->abcijk", X.bb.vvvv, T.bbb, optimize=True)
    X3D += 0.25 * np.einsum("bmje,aecimk->abcijk", X.bb.voov, T.bbb, optimize=True)
    X3D += 0.25 * np.einsum("mbej,ecamki->abcijk", X.ab.ovvo, T.abb, optimize=True)

    # < i~j~k~a~b~c~ | (HR3)_C | 0 >
    X3D -= (1.0 / 12.0) * np.einsum("mj,abcimk->abcijk", H.b.oo, R.bbb, optimize=True)
    X3D += (1.0 / 12.0) * np.einsum("be,aecijk->abcijk", H.b.vv, R.bbb, optimize=True)
    X3D += (1.0 / 24.0) * np.einsum("mnij,abcmnk->abcijk", H.bb.oooo, R.bbb, optimize=True)
    X3D += (1.0 / 24.0) * np.einsum("abef,efcijk->abcijk", H.bb.vvvv, R.bbb, optimize=True)
    X3D += 0.25 * np.einsum("amie,ebcmjk->abcijk", H.bb.voov, R.bbb, optimize=True)
    X3D += 0.25 * np.einsum("maei,ecbmkj->abcijk", H.ab.ovvo, R.abb, optimize=True)

    # antisymmetrize terms and add up: A(abc)A(ijk) = A(a/bc)A(bc)A(i/jk)A(jk)
    X3D -= np.transpose(X3D, (0, 1, 2, 3, 5, 4))
    X3D -= np.transpose(X3D, (0, 1, 2, 4, 3, 5)) + np.transpose(X3D, (0, 1, 2, 5, 4, 3))
    X3D -= np.transpose(X3D, (0, 2, 1, 3, 4, 5))
    X3D -= np.transpose(X3D, (1, 0, 2, 3, 4, 5)) + np.transpose(X3D, (2, 1, 0, 3, 4, 5))
    return X3D

if __name__ == "__main__":

    case = 'F2'

    if case == 'CH+':
        system, H = load_from_gamess(
                "chplus_re.log",
                "onebody.inp",
                "twobody.inp",
                nfrozen=0,
        )
        system.set_active_space(nact_occupied=2, nact_unoccupied=10)

    if case == 'F2':
        mol = gto.Mole()
        mol.build(
            atom="""F 0.0 0.0 -2.66816
                    F 0.0 0.0  2.66816""",
            basis="ccpvdz",
            charge=0,
            spin=0,
            symmetry="D2H",
            cart=True,
            unit='Bohr',
        )
        mf = scf.ROHF(mol)
        mf.kernel()
        system, H = load_pyscf_integrals(mf, nfrozen=2)
        system.set_active_space(nact_occupied=5, nact_unoccupied=9)

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    calculation = Calculation(
        order=3,
        calculation_type="ccsdt",
        convergence_tolerance=1.0e-08,
        maximum_iterations=5,
        RHF_symmetry=False,
    )

    T, total_energy, is_converged = cc_driver(calculation, system, H)

    # zero amplitudes outside active space
    T = zero_t3aaa_outside_active_space(T, system, 1)
    T = zero_t3aab_outside_active_space(T, system, 1)
    T = zero_t3abb_outside_active_space(T, system, 1)
    T = zero_t3bbb_outside_active_space(T, system, 1)

    Hbar = build_hbar_ccsdt(T, H)

    calculation = Calculation(
        order=3,
        calculation_type="eomccsdt",
        maximum_iterations=5,
        convergence_tolerance=1.0e-08,
        multiplicity=1,
        RHF_symmetry=False,
        low_memory=False,
    )

    R_list, omega = get_initial_guess(calculation, system, Hbar, 1, noact=0, nuact=0, guess_order=1)
    R_list, omega, r0, is_converged = eomcc_driver(calculation, system, Hbar, T, R_list, omega)
    R = R_list[0]

    R = zero_t3aaa_outside_active_space(R, system, 1)
    R = zero_t3aab_outside_active_space(R, system, 1)
    R = zero_t3abb_outside_active_space(R, system, 1)
    R = zero_t3bbb_outside_active_space(R, system, 1)

    # Full update
    dR = ClusterOperator(system, order=3)

    HR = get_eomccsd_intermediates(Hbar, R, T, system)
    HR = add_R3_terms(HR, Hbar, R)

    dR.a = calc_full_HR1A(R, T, Hbar)
    dR.b = calc_full_HR1B(R, T, Hbar)
    dR.aa = calc_full_HR2A(R, T, Hbar, HR)
    dR.ab = calc_full_HR2B(R, T, Hbar, HR)
    dR.bb = calc_full_HR2C(R, T, Hbar, HR)
    dR.aaa = calc_full_HR3A(R, T, Hbar, HR)
    dR.aab = calc_full_HR3B(R, T, Hbar, HR)
    dR.abb = calc_full_HR3C(R, T, Hbar, HR)
    dR.bbb = calc_full_HR3D(R, T, Hbar, HR)


    # create acitve-space cluster operator and excitation operator
    T_act = ClusterOperator(system, order=3, active_orders=[3], num_active=[1])
    R_act = ClusterOperator(system, order=3, active_orders=[3], num_active=[1])
    dR_act = ClusterOperator(system, order=3, active_orders=[3], num_active=[1])

    # fill in the active-space cluster operator
    T_act.a = T.a.copy()
    T_act.b = T.b.copy()
    T_act.aa = T.aa.copy()
    T_act.ab = T.ab.copy()
    T_act.bb = T.bb.copy()
    T_act = fill_t3aaa(T_act, T, system)
    T_act = fill_t3aab(T_act, T, system)
    T_act = fill_t3abb(T_act, T, system)
    T_act = fill_t3bbb(T_act, T, system)

    R_act.a = R.a.copy()
    R_act.b = R.b.copy()
    R_act.aa = R.aa.copy()
    R_act.ab = R.ab.copy()
    R_act.bb = R.bb.copy()
    R_act = fill_t3aaa(R_act, R, system)
    R_act = fill_t3aab(R_act, R, system)
    R_act = fill_t3abb(R_act, R, system)
    R_act = fill_t3bbb(R_act, R, system)

    Hbar1 = build_hbar_ccsdt1(T_act, H, system)
    check_intermediates(Hbar1, Hbar)

    dR_act = eomccsdt1.HR(dR_act, R_act, T_act, Hbar1, False, system)

    # HR_act = get_eomccsd_intermediates(Hbar, R_act, T_act, system)
    # HR_act = add_HR3_intermediates(HR_act, Hbar, R_act, system)

    # # aaa updates
    # dR_act.aaa.VVVOOO = r3a_111111.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.aaa.VVVoOO = r3a_111011.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.aaa.VVvOOO = r3a_110111.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.aaa.VVvoOO = r3a_110011.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.aaa.VvvOOO = r3a_100111.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.aaa.VVVooO = r3a_111001.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.aaa.VvvoOO = r3a_100011.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.aaa.VVvooO = r3a_110001.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.aaa.VvvooO = r3a_100001.build(R_act, T_act, Hbar, HR_act, system)
    # # aab updates
    # dR_act.aab.VVVOOO = r3b_111111.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.aab.VVVOOo = r3b_111110.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.aab.VVvOOO = r3b_110111.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.aab.VvVOOO = r3b_101111.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.aab.VVVoOO = r3b_111011.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.aab.VVVooO = r3b_111001.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.aab.VVVoOo = r3b_111010.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.aab.vvVOOO = r3b_001111.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.aab.VvvOOO = r3b_100111.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.aab.VVvoOO = r3b_110011.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.aab.VvVOOo = r3b_101110.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.aab.VvVoOO = r3b_101011.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.aab.VVvOOo = r3b_110110.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.aab.VvVooO = r3b_101001.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.aab.VvVoOo = r3b_101010.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.aab.VVvooO = r3b_110001.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.aab.VVvoOo = r3b_110010.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.aab.vvVoOO = r3b_001011.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.aab.vvVOOo = r3b_001110.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.aab.VvvoOO = r3b_100011.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.aab.VvvOOo = r3b_100110.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.aab.vvVooO = r3b_001001.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.aab.vvVoOo = r3b_001010.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.aab.VvvooO = r3b_100001.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.aab.VvvoOo = r3b_100010.build(R_act, T_act, Hbar, HR_act, system)
    # # abb updates
    # dR_act.abb.VVVOOO = r3c_111111.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.abb.VVVOoO = r3c_111101.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.abb.VVVoOO = r3c_111011.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.abb.VVvOOO = r3c_110111.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.abb.vVVOOO = r3c_011111.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.abb.VVVOoo = r3c_111100.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.abb.VVVooO = r3c_111001.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.abb.vVvOOO = r3c_010111.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.abb.VvvOOO = r3c_100111.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.abb.VVvoOO = r3c_110011.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.abb.vVVOoO = r3c_011101.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.abb.vVVoOO = r3c_011011.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.abb.VVvOoO = r3c_110101.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.abb.VVvOoo = r3c_110100.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.abb.VVvooO = r3c_110001.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.abb.vVVOoo = r3c_011100.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.abb.vVVooO = r3c_011001.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.abb.VvvoOO = r3c_100011.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.abb.VvvOoO = r3c_100101.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.abb.vVvoOO = r3c_010011.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.abb.vVvOoO = r3c_010101.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.abb.vVvOoo = r3c_010100.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.abb.vVvooO = r3c_010001.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.abb.VvvOoo = r3c_100100.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.abb.VvvooO = r3c_100001.build(R_act, T_act, Hbar, HR_act, system)
    # # bbb updates
    # dR_act.bbb.VVVOOO = r3d_111111.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.bbb.VVVoOO = r3d_111011.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.bbb.VVvOOO = r3d_110111.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.bbb.VVvoOO = r3d_110011.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.bbb.VvvOOO = r3d_100111.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.bbb.VVVooO = r3d_111001.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.bbb.VvvoOO = r3d_100011.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.bbb.VVvooO = r3d_110001.build(R_act, T_act, Hbar, HR_act, system)
    # dR_act.bbb.VvvooO = r3d_100001.build(R_act, T_act, Hbar, HR_act, system)

    # Get the error
    calc_error(dR_act, dR, system)
