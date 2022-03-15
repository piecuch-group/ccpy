import numpy as np
import time

from pyscf import gto, scf

from ccpy.models.operators import ClusterOperator

from ccpy.models.calculation import Calculation
from ccpy.interfaces.pyscf_tools import load_pyscf_integrals
from ccpy.drivers.driver import cc_driver

from ccpy.hbar.hbar_ccsd import build_hbar_ccsd, get_ccsd_intermediates
from ccpy.hbar.hbar_ccs import get_ccs_intermediates

from ccpy.utilities.active_space import fill_t3aaa, fill_t3aab, fill_t3abb, fill_t3bbb, get_active_slices,\
    zero_t3aaa_outside_active_space, zero_t3bbb_outside_active_space, zero_t3aab_outside_active_space, zero_t3abb_outside_active_space

from ccpy.cc.ccsdt1 import update
from ccpy.cc.ccsdt1_updates import *
from ccpy.utilities.updates import cc_loops2
from testing_updates import *


def print_error(x, y):
    error = x - y
    #print(np.linalg.norm(x.flatten()))
    #print(np.linalg.norm(y.flatten()))
    print(np.linalg.norm(error.flatten()))

def calc_error(dT_act, dT, system):

    def _get_error(x, y):
        error = x - y
        return np.linalg.norm(error.flatten())

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    print('Error in T1.a')
    print('-------------------------')
    print("Total error = ", _get_error(dT_act.a, dT.a))

    print('Error in T1.b')
    print('-------------------------')
    print("Total error = ", _get_error(dT_act.b, dT.b))

    print('Error in T2.aa')
    print('-------------------------')
    print("Total error = ", _get_error(dT_act.aa, dT.aa))

    print('Error in T2.ab')
    print('-------------------------')
    print("Total error = ", _get_error(dT_act.ab, dT.ab))

    print('Error in T2.bb')
    print('-------------------------')
    print("Total error = ", _get_error(dT_act.bb, dT.bb))

    print('Error in T3.aaa')
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

    print('Error in T3.aab')
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

    print('Error in T3.abb')
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

    print('Error in T3.bbb')
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


def calc_full_update_t1a(T, dT, H, shift, system):

    chi1A_vv = H.a.vv.copy()
    chi1A_vv += np.einsum("anef,fn->ae", H.aa.vovv, T.a, optimize=True)
    chi1A_vv += np.einsum("anef,fn->ae", H.ab.vovv, T.b, optimize=True)

    chi1A_oo = H.a.oo.copy()
    chi1A_oo += np.einsum("mnif,fn->mi", H.aa.ooov, T.a, optimize=True)
    chi1A_oo += np.einsum("mnif,fn->mi", H.ab.ooov, T.b, optimize=True)

    h1A_ov = H.a.ov.copy()
    h1A_ov += np.einsum("mnef,fn->me", H.aa.oovv, T.a, optimize=True)
    h1A_ov += np.einsum("mnef,fn->me", H.ab.oovv, T.b, optimize=True)

    h1B_ov = H.b.ov.copy()
    h1B_ov += np.einsum("nmfe,fn->me", H.ab.oovv, T.a, optimize=True)
    h1B_ov += np.einsum("mnef,fn->me", H.bb.oovv, T.b, optimize=True)

    h1A_oo = chi1A_oo.copy()
    h1A_oo += np.einsum("me,ei->mi", h1A_ov, T.a, optimize=True)

    h2A_ooov = H.aa.ooov + np.einsum("mnfe,fi->mnie", H.aa.oovv, T.a, optimize=True)
    h2B_ooov = H.ab.ooov + np.einsum("mnfe,fi->mnie", H.ab.oovv, T.a, optimize=True)
    h2A_vovv = H.aa.vovv - np.einsum("mnfe,an->amef", H.aa.oovv, T.a, optimize=True)
    h2B_vovv = H.ab.vovv - np.einsum("nmef,an->amef", H.ab.oovv, T.a, optimize=True)

    dT.a = H.a.vo.copy()
    dT.a -= np.einsum("mi,am->ai", h1A_oo, T.a, optimize=True)
    dT.a += np.einsum("ae,ei->ai", chi1A_vv, T.a, optimize=True)
    dT.a += np.einsum("anif,fn->ai", H.aa.voov, T.a, optimize=True)
    dT.a += np.einsum("anif,fn->ai", H.ab.voov, T.b, optimize=True)
    dT.a += np.einsum("me,aeim->ai", h1A_ov, T.aa, optimize=True)
    dT.a += np.einsum("me,aeim->ai", h1B_ov, T.ab, optimize=True)
    dT.a -= 0.5 * np.einsum("mnif,afmn->ai", h2A_ooov, T.aa, optimize=True)
    dT.a -= np.einsum("mnif,afmn->ai", h2B_ooov, T.ab, optimize=True)
    dT.a += 0.5 * np.einsum("anef,efin->ai", h2A_vovv, T.aa, optimize=True)
    dT.a += np.einsum("anef,efin->ai", h2B_vovv, T.ab, optimize=True)
    # T3 parts
    dT.a += 0.25 * np.einsum("mnef,aefimn->ai", H.aa.oovv, T.aaa, optimize=True)
    dT.a += np.einsum("mnef,aefimn->ai", H.ab.oovv, T.aab, optimize=True)
    dT.a += 0.25 * np.einsum("mnef,aefimn->ai", H.bb.oovv, T.abb, optimize=True)

    # T.a, dT.a = cc_loops2.cc_loops2.update_t1a(
    #     T.a,
    #     dT.a + H.a.vo,
    #     H.a.oo,
    #     H.a.vv,
    #     shift,
    # )
    return T, dT

def calc_full_update_t1b(T, dT, H, shift, system):

    chi1B_vv = H.b.vv.copy()
    chi1B_vv += np.einsum("anef,fn->ae", H.bb.vovv, T.b, optimize=True)
    chi1B_vv += np.einsum("nafe,fn->ae", H.ab.ovvv, T.a, optimize=True)

    chi1B_oo = H.b.oo.copy()
    chi1B_oo += np.einsum("mnif,fn->mi", H.bb.ooov, T.b, optimize=True)
    chi1B_oo += np.einsum("nmfi,fn->mi", H.ab.oovo, T.a, optimize=True)

    h1A_ov = H.a.ov.copy()
    h1A_ov += np.einsum("mnef,fn->me", H.aa.oovv, T.a, optimize=True)
    h1A_ov += np.einsum("mnef,fn->me", H.ab.oovv, T.b, optimize=True)

    h1B_ov = H.b.ov.copy()
    h1B_ov += np.einsum("nmfe,fn->me", H.ab.oovv, T.a, optimize=True)
    h1B_ov += np.einsum("mnef,fn->me", H.bb.oovv, T.b, optimize=True)

    h1B_oo = chi1B_oo + np.einsum("me,ei->mi", h1B_ov, T.b, optimize=True)

    h2C_ooov = H.bb.ooov + np.einsum("mnfe,fi->mnie", H.bb.oovv, T.b, optimize=True)
    h2B_oovo = H.ab.oovo + np.einsum("nmef,fi->nmei", H.ab.oovv, T.b, optimize=True)
    h2C_vovv = H.bb.vovv - np.einsum("mnfe,an->amef", H.bb.oovv, T.b, optimize=True)
    h2B_ovvv = H.ab.ovvv - np.einsum("mnfe,an->mafe", H.ab.oovv, T.b, optimize=True)

    dT.b = H.b.vo.copy()
    dT.b -= np.einsum("mi,am->ai", h1B_oo, T.b, optimize=True)
    dT.b += np.einsum("ae,ei->ai", chi1B_vv, T.b, optimize=True)
    dT.b += np.einsum("anif,fn->ai", H.bb.voov, T.b, optimize=True)
    dT.b += np.einsum("nafi,fn->ai", H.ab.ovvo, T.a, optimize=True)
    dT.b += np.einsum("me,eami->ai", h1A_ov, T.ab, optimize=True)
    dT.b += np.einsum("me,aeim->ai", h1B_ov, T.bb, optimize=True)
    dT.b -= 0.5 * np.einsum("mnif,afmn->ai", h2C_ooov, T.bb, optimize=True)
    dT.b -= np.einsum("nmfi,fanm->ai", h2B_oovo, T.ab, optimize=True)
    dT.b += 0.5 * np.einsum("anef,efin->ai", h2C_vovv, T.bb, optimize=True)
    dT.b += np.einsum("nafe,feni->ai", h2B_ovvv, T.ab, optimize=True)
    # T3 parts
    dT.b += 0.25 * np.einsum("mnef,aefimn->ai", H.bb.oovv, T.bbb, optimize=True)
    dT.b += 0.25 * np.einsum("mnef,efamni->ai", H.aa.oovv, T.aab, optimize=True)
    dT.b += np.einsum("mnef,efamni->ai", H.ab.oovv, T.abb, optimize=True)

    # T.b, dT.b = cc_loops2.cc_loops2.update_t1b(
    #     T.b,
    #     dT.b + H.b.vo,
    #     H.b.oo,
    #     H.b.vv,
    #     shift,
    # )
    return T, dT

def calc_full_update_t2a(T, dT, H, H0, shift, system):
    """
    Update t2a amplitudes by calculating the projection <ijab|(H_N e^(T1+T2+T3))_C|0>.
    """
    # intermediates
    I1A_oo = (
        H.a.oo
        + 0.5 * np.einsum("mnef,efin->mi", H.aa.oovv, T.aa, optimize=True)
        + np.einsum("mnef,efin->mi", H.ab.oovv, T.ab, optimize=True)
    )

    I1A_vv = (
        H.a.vv
        - 0.5 * np.einsum("mnef,afmn->ae", H.aa.oovv, T.aa, optimize=True)
        - np.einsum("mnef,afmn->ae", H.ab.oovv, T.ab, optimize=True)
    )

    I2A_voov = (
        H.aa.voov
        + 0.5 * np.einsum("mnef,afin->amie", H.aa.oovv, T.aa, optimize=True)
        + np.einsum("mnef,afin->amie", H.ab.oovv, T.ab, optimize=True)
    )

    I2A_oooo = H.aa.oooo + 0.5 * np.einsum(
        "mnef,efij->mnij", H.aa.oovv, T.aa, optimize=True
    )

    I2B_voov = H.ab.voov + 0.5 * np.einsum(
        "mnef,afin->amie", H.bb.oovv, T.ab, optimize=True
    )

    dT.aa = 0.25 * H.aa.vvoo
    dT.aa -= 0.5 * np.einsum("amij,bm->abij", H.aa.vooo, T.a, optimize=True)
    dT.aa += 0.5 * np.einsum("abie,ej->abij", H.aa.vvov, T.a, optimize=True)
    dT.aa += 0.5 * np.einsum("ae,ebij->abij", I1A_vv, T.aa, optimize=True)
    dT.aa -= 0.5 * np.einsum("mi,abmj->abij", I1A_oo, T.aa, optimize=True)
    dT.aa += np.einsum("amie,ebmj->abij", I2A_voov, T.aa, optimize=True)
    dT.aa += np.einsum("amie,bejm->abij", I2B_voov, T.ab, optimize=True)
    dT.aa += 0.125 * np.einsum("abef,efij->abij", H.aa.vvvv, T.aa, optimize=True)
    dT.aa += 0.125 * np.einsum("mnij,abmn->abij", I2A_oooo, T.aa, optimize=True)
    # T3 parts
    dT.aa += 0.25 * np.einsum("me,abeijm->abij", H.a.ov, T.aaa, optimize=True)
    dT.aa += 0.25 * np.einsum("me,abeijm->abij", H.b.ov, T.aab, optimize=True)
    dT.aa -= 0.5 * np.einsum("mnif,abfmjn->abij", H.ab.ooov, T.aab, optimize=True)
    dT.aa -= 0.25 * np.einsum("mnif,abfmjn->abij", H.aa.ooov, T.aaa, optimize=True)
    dT.aa += 0.25 * np.einsum("anef,ebfijn->abij", H.aa.vovv, T.aaa, optimize=True)
    dT.aa += 0.5 * np.einsum("anef,ebfijn->abij", H.ab.vovv, T.aab, optimize=True)

    dT.aa -= np.transpose(dT.aa, (1, 0, 2, 3))
    dT.aa -= np.transpose(dT.aa, (0, 1, 3, 2))

    # T.aa, dT.aa = cc_loops2.cc_loops2.update_t2a(
    #     T.aa,
    #     dT.aa + 0.25 * H0.aa.vvoo,
    #     H0.a.oo,
    #     H0.a.vv,
    #     shift,
    # )
    return T, dT

def calc_full_update_t2b(T, dT, H, H0, shift, system):
    # intermediates
    I1A_vv = (
        H.a.vv
        - 0.5 * np.einsum("mnef,afmn->ae", H.aa.oovv, T.aa, optimize=True)
        - np.einsum("mnef,afmn->ae", H.ab.oovv, T.ab, optimize=True)
    )

    I1B_vv = (
        H.b.vv
        - np.einsum("nmfe,fbnm->be", H.ab.oovv, T.ab, optimize=True)
        - 0.5 * np.einsum("mnef,fbnm->be", H.bb.oovv, T.bb, optimize=True)
    )

    I1A_oo = (
        H.a.oo
        + 0.5 * np.einsum("mnef,efin->mi", H.aa.oovv, T.aa, optimize=True)
        + np.einsum("mnef,efin->mi", H.ab.oovv, T.ab, optimize=True)
    )

    I1B_oo = (
        H.b.oo
        + np.einsum("nmfe,fenj->mj", H.ab.oovv, T.ab, optimize=True)
        + 0.5 * np.einsum("mnef,efjn->mj", H.bb.oovv, T.bb, optimize=True)
    )

    I2A_voov = (
        H.aa.voov
        + np.einsum("mnef,aeim->anif", H.aa.oovv, T.aa, optimize=True)
        + np.einsum("nmfe,aeim->anif", H.ab.oovv, T.ab, optimize=True)
    )

    I2B_voov = (
        H.ab.voov
        + np.einsum("mnef,aeim->anif", H.ab.oovv, T.aa, optimize=True)
        + np.einsum("mnef,aeim->anif", H.bb.oovv, T.ab, optimize=True)
    )

    I2B_oooo = H.ab.oooo + np.einsum("mnef,efij->mnij", H.ab.oovv, T.ab, optimize=True)

    I2B_vovo = H.ab.vovo - np.einsum("mnef,afmj->anej", H.ab.oovv, T.ab, optimize=True)

    dT.ab = H.ab.vvoo.copy()
    dT.ab -= np.einsum("mbij,am->abij", H.ab.ovoo, T.a, optimize=True)
    dT.ab -= np.einsum("amij,bm->abij", H.ab.vooo, T.b, optimize=True)
    dT.ab += np.einsum("abej,ei->abij", H.ab.vvvo, T.a, optimize=True)
    dT.ab += np.einsum("abie,ej->abij", H.ab.vvov, T.b, optimize=True)
    dT.ab += np.einsum("ae,ebij->abij", I1A_vv, T.ab, optimize=True)
    dT.ab += np.einsum("be,aeij->abij", I1B_vv, T.ab, optimize=True)
    dT.ab -= np.einsum("mi,abmj->abij", I1A_oo, T.ab, optimize=True)
    dT.ab -= np.einsum("mj,abim->abij", I1B_oo, T.ab, optimize=True)
    dT.ab += np.einsum("amie,ebmj->abij", I2A_voov, T.ab, optimize=True)
    dT.ab += np.einsum("amie,ebmj->abij", I2B_voov, T.bb, optimize=True)
    dT.ab += np.einsum("mbej,aeim->abij", H.ab.ovvo, T.aa, optimize=True)
    dT.ab += np.einsum("bmje,aeim->abij", H.bb.voov, T.ab, optimize=True)
    dT.ab -= np.einsum("mbie,aemj->abij", H.ab.ovov, T.ab, optimize=True)
    dT.ab -= np.einsum("amej,ebim->abij", I2B_vovo, T.ab, optimize=True)
    dT.ab += np.einsum("mnij,abmn->abij", I2B_oooo, T.ab, optimize=True)
    dT.ab += np.einsum("abef,efij->abij", H.ab.vvvv, T.ab, optimize=True)
    # T3 parts
    dT.ab -= 0.5 * np.einsum("mnif,afbmnj->abij", H.aa.ooov, T.aab, optimize=True)
    dT.ab -= np.einsum("nmfj,afbinm->abij", H.ab.oovo, T.aab, optimize=True)
    dT.ab -= 0.5 * np.einsum("mnjf,afbinm->abij", H.bb.ooov, T.abb, optimize=True)
    dT.ab -= np.einsum("mnif,afbmnj->abij", H.ab.ooov, T.abb, optimize=True)
    dT.ab += 0.5 * np.einsum("anef,efbinj->abij", H.aa.vovv, T.aab, optimize=True)
    dT.ab += np.einsum("anef,efbinj->abij", H.ab.vovv, T.abb, optimize=True)
    dT.ab += np.einsum("nbfe,afeinj->abij", H.ab.ovvv, T.aab, optimize=True)
    dT.ab += 0.5 * np.einsum("bnef,afeinj->abij", H.bb.vovv, T.abb, optimize=True)
    dT.ab += np.einsum("me,aebimj->abij", H.a.ov, T.aab, optimize=True)
    dT.ab += np.einsum("me,aebimj->abij", H.b.ov, T.abb, optimize=True)

    # T.ab, dT.ab = cc_loops2.cc_loops2.update_t2b(
    #     T.ab,
    #     dT.ab + H0.ab.vvoo,
    #     H0.a.oo,
    #     H0.a.vv,
    #     H0.b.oo,
    #     H0.b.vv,
    #     shift,
    # )

    return T, dT

def calc_full_update_t2c(T, dT, H, H0, shift, system):
    """
    Update t2c amplitudes by calculating the projection <i~j~a~b~|(H_N e^(T1+T2+T3))_C|0>.
    """
    # intermediates
    I1B_oo = (
        H.b.oo
        + 0.5 * np.einsum("mnef,efin->mi", H.bb.oovv, T.bb, optimize=True)
        + np.einsum("nmfe,feni->mi", H.ab.oovv, T.ab, optimize=True)
    )

    I1B_vv = (
        H.b.vv
        - 0.5 * np.einsum("mnef,afmn->ae", H.bb.oovv, T.bb, optimize=True)
        - np.einsum("nmfe,fanm->ae", H.ab.oovv, T.ab, optimize=True)
    )

    I2C_oooo = H.bb.oooo + 0.5 * np.einsum(
        "mnef,efij->mnij", H.bb.oovv, T.bb, optimize=True
    )

    I2B_ovvo = (
        H.ab.ovvo
        + np.einsum("mnef,afin->maei", H.ab.oovv, T.bb, optimize=True)
        + 0.5 * np.einsum("mnef,fani->maei", H.aa.oovv, T.ab, optimize=True)
    )

    I2C_voov = H.bb.voov + 0.5 * np.einsum(
        "mnef,afin->amie", H.bb.oovv, T.bb, optimize=True
    )

    dT.bb = 0.25 * H.bb.vvoo
    dT.bb -= 0.5 * np.einsum("amij,bm->abij", H.bb.vooo, T.b, optimize=True)
    dT.bb += 0.5 * np.einsum("abie,ej->abij", H.bb.vvov, T.b, optimize=True)
    dT.bb += 0.5 * np.einsum("ae,ebij->abij", I1B_vv, T.bb, optimize=True)
    dT.bb -= 0.5 * np.einsum("mi,abmj->abij", I1B_oo, T.bb, optimize=True)
    dT.bb += np.einsum("amie,ebmj->abij", I2C_voov, T.bb, optimize=True)
    dT.bb += np.einsum("maei,ebmj->abij", I2B_ovvo, T.ab, optimize=True)
    dT.bb += 0.125 * np.einsum("abef,efij->abij", H.bb.vvvv, T.bb, optimize=True)
    dT.bb += 0.125 * np.einsum("mnij,abmn->abij", I2C_oooo, T.bb, optimize=True)
    # T3 parts
    dT.bb += 0.25 * np.einsum("me,eabmij->abij", H.a.ov, T.abb, optimize=True)
    dT.bb += 0.25 * np.einsum("me,abeijm->abij", H.b.ov, T.bbb, optimize=True)
    dT.bb += 0.25 * np.einsum("anef,ebfijn->abij", H.bb.vovv, T.bbb, optimize=True)
    dT.bb += 0.5 * np.einsum("nafe,febnij->abij", H.ab.ovvv, T.abb, optimize=True)
    dT.bb -= 0.25 * np.einsum("mnif,abfmjn->abij", H.bb.ooov, T.bbb, optimize=True)
    dT.bb -= 0.5 * np.einsum("nmfi,fabnmj->abij", H.ab.oovo, T.abb, optimize=True)

    dT.bb -= np.transpose(dT.bb, (1, 0, 2, 3))
    dT.bb -= np.transpose(dT.bb, (0, 1, 3, 2))

    # T.bb, dT.bb = cc_loops2.cc_loops2.update_t2c(
    #     T.bb,
    #     dT.bb + 0.25 * H0.bb.vvoo,
    #     H0.b.oo,
    #     H0.b.vv,
    #     shift,
    # )

    return T, dT


def calc_full_update_t3a(T, dT, H, H0, shift, system):

    I2A_vvov = -0.5 * np.einsum("mnef,abfimn->abie", H0.aa.oovv, T.aaa, optimize=True)
    I2A_vvov -= np.einsum("mnef,abfimn->abie", H0.ab.oovv, T.aab, optimize=True)
    I2A_vvov += H.aa.vvov + np.einsum("me,abim->abie", H.a.ov, T.aa, optimize=True)

    I2A_vooo = 0.5 * np.einsum("mnef,aefijn->amij", H0.aa.oovv, T.aaa, optimize=True)
    I2A_vooo += H.aa.vooo + np.einsum("mnef,aefijn->amij", H0.ab.oovv, T.aab, optimize=True)

    # MM(2,3)A
    dT.aaa = -0.25 * np.einsum("amij,bcmk->abcijk", I2A_vooo, T.aa, optimize=True)
    dT.aaa += 0.25 * np.einsum("abie,ecjk->abcijk", I2A_vvov, T.aa, optimize=True)
    # (H(2) * T3)_C
    dT.aaa -= (1.0 / 12.0) * np.einsum("mk,abcijm->abcijk", H.a.oo, T.aaa, optimize=True)
    dT.aaa += (1.0 / 12.0) * np.einsum("ce,abeijk->abcijk", H.a.vv, T.aaa, optimize=True)
    dT.aaa += (1.0 / 24.0) * np.einsum("mnij,abcmnk->abcijk", H.aa.oooo, T.aaa, optimize=True)
    dT.aaa += (1.0 / 24.0) * np.einsum("abef,efcijk->abcijk", H.aa.vvvv, T.aaa, optimize=True)
    dT.aaa += 0.25 * np.einsum("cmke,abeijm->abcijk", H.aa.voov, T.aaa, optimize=True)
    dT.aaa += 0.25 * np.einsum("cmke,abeijm->abcijk", H.ab.voov, T.aab, optimize=True)

    dT.aaa -= np.transpose(dT.aaa, (1, 0, 2, 3, 4, 5)) + np.transpose(dT.aaa, (0, 2, 1, 3, 4, 5)) \
           + np.transpose(dT.aaa, (2, 1, 0, 3, 4, 5)) - np.transpose(dT.aaa, (1, 2, 0, 3, 4, 5)) \
           - np.transpose(dT.aaa, (2, 0, 1, 3, 4, 5))

    dT.aaa -= np.transpose(dT.aaa, (0, 1, 2, 4, 3, 5)) + np.transpose(dT.aaa, (0, 1, 2, 3, 5, 4)) \
           + np.transpose(dT.aaa, (0, 1, 2, 5, 4, 3)) - np.transpose(dT.aaa, (0, 1, 2, 4, 5, 3)) \
           - np.transpose(dT.aaa, (0, 1, 2, 5, 3, 4))

    # T.aaa, dT.aaa = cc_loops2.cc_loops2.update_t3a(
    #     T.aaa,
    #     dT.aaa,
    #     H0.a.oo,
    #     H0.a.vv,
    #     shift,
    # )
    #T = zero_t3aaa_outside_active_space(T, system, 1)
    #dT = zero_t3aaa_outside_active_space(dT, system, 1)

    return T, dT

def calc_full_update_t3b(T, dT, H, H0, shift, system):

    # <ijk~abc~ | H(2) | 0 > + (VT3)_C intermediates
    I2A_vvov = -0.5 * np.einsum("mnef,abfimn->abie", H0.aa.oovv, T.aaa, optimize=True)
    I2A_vvov += -np.einsum("mnef,abfimn->abie", H0.ab.oovv, T.aab, optimize=True)
    I2A_vvov += H.aa.vvov

    I2A_vooo = 0.5 * np.einsum("mnef,aefijn->amij", H0.aa.oovv, T.aaa, optimize=True)
    I2A_vooo += np.einsum("mnef,aefijn->amij", H0.ab.oovv, T.aab, optimize=True)
    I2A_vooo += -np.einsum("me,aeij->amij", H.a.ov, T.aa, optimize=True)
    I2A_vooo += H.aa.vooo

    I2B_vvvo = -0.5 * np.einsum("mnef,afbmnj->abej", H0.aa.oovv, T.aab, optimize=True)
    I2B_vvvo += -np.einsum("mnef,afbmnj->abej", H0.ab.oovv, T.abb, optimize=True)
    I2B_vvvo += H.ab.vvvo

    I2B_ovoo = 0.5 * np.einsum("mnef,efbinj->mbij", H0.aa.oovv, T.aab, optimize=True)
    I2B_ovoo += np.einsum("mnef,efbinj->mbij", H0.ab.oovv, T.abb, optimize=True)
    I2B_ovoo += -np.einsum("me,ecjk->mcjk", H.a.ov, T.ab, optimize=True)
    I2B_ovoo += H.ab.ovoo

    I2B_vvov = -np.einsum("nmfe,afbinm->abie", H0.ab.oovv, T.aab, optimize=True)
    I2B_vvov += -0.5 * np.einsum("nmfe,afbinm->abie", H0.bb.oovv, T.abb, optimize=True)
    I2B_vvov += H.ab.vvov

    I2B_vooo = np.einsum("nmfe,afeinj->amij", H0.ab.oovv, T.aab, optimize=True)
    I2B_vooo += 0.5 * np.einsum("nmfe,afeinj->amij", H0.bb.oovv, T.abb, optimize=True)
    I2B_vooo += -np.einsum("me,aeik->amik", H.b.ov, T.ab, optimize=True)
    I2B_vooo += H.ab.vooo

    # MM(2,3)B
    dT.aab = 0.5 * np.einsum("bcek,aeij->abcijk", I2B_vvvo, T.aa, optimize=True)
    dT.aab -= 0.5 * np.einsum("mcjk,abim->abcijk", I2B_ovoo, T.aa, optimize=True)
    dT.aab += np.einsum("acie,bejk->abcijk", I2B_vvov, T.ab, optimize=True)
    dT.aab -= np.einsum("amik,bcjm->abcijk", I2B_vooo, T.ab, optimize=True)
    dT.aab += 0.5 * np.einsum("abie,ecjk->abcijk", I2A_vvov, T.ab, optimize=True)
    dT.aab -= 0.5 * np.einsum("amij,bcmk->abcijk", I2A_vooo, T.ab, optimize=True)
    # (H(2) * T3)_C
    dT.aab -= 0.5 * np.einsum("mi,abcmjk->abcijk", H.a.oo, T.aab, optimize=True)
    dT.aab -= 0.25 * np.einsum("mk,abcijm->abcijk", H.b.oo, T.aab, optimize=True)
    dT.aab += 0.5 * np.einsum("ae,ebcijk->abcijk", H.a.vv, T.aab, optimize=True)
    dT.aab += 0.25 * np.einsum("ce,abeijk->abcijk", H.b.vv, T.aab, optimize=True)
    dT.aab += 0.125 * np.einsum("mnij,abcmnk->abcijk", H.aa.oooo, T.aab, optimize=True)
    dT.aab += 0.5 * np.einsum("mnjk,abcimn->abcijk", H.ab.oooo, T.aab, optimize=True)
    dT.aab += 0.125 * np.einsum("abef,efcijk->abcijk", H.aa.vvvv, T.aab, optimize=True)
    dT.aab += 0.5 * np.einsum("bcef,aefijk->abcijk", H.ab.vvvv, T.aab, optimize=True)
    dT.aab += np.einsum("amie,ebcmjk->abcijk", H.aa.voov, T.aab, optimize=True)
    dT.aab += np.einsum("amie,becjmk->abcijk", H.ab.voov, T.abb, optimize=True)
    dT.aab += 0.25 * np.einsum("mcek,abeijm->abcijk", H.ab.ovvo, T.aaa, optimize=True)
    dT.aab += 0.25 * np.einsum("cmke,abeijm->abcijk", H.bb.voov, T.aab, optimize=True)
    dT.aab -= 0.5 * np.einsum("amek,ebcijm->abcijk", H.ab.vovo, T.aab, optimize=True)
    dT.aab -= 0.5 * np.einsum("mcie,abemjk->abcijk", H.ab.ovov, T.aab, optimize=True)

    dT.aab -= np.transpose(dT.aab, (1, 0, 2, 3, 4, 5))
    dT.aab -= np.transpose(dT.aab, (0, 1, 2, 4, 3, 5))

    return T, dT

def calc_full_update_t3c(T, dT, H, H0, shift, system):
    I2B_vvvo = -0.5 * np.einsum("mnef,afbmnj->abej", H0.aa.oovv, T.aab, optimize=True)
    I2B_vvvo += -np.einsum("mnef,afbmnj->abej", H0.ab.oovv, T.abb, optimize=True)
    I2B_vvvo += H.ab.vvvo

    I2B_ovoo = 0.5 * np.einsum("mnef,efbinj->mbij", H0.aa.oovv, T.aab, optimize=True)
    I2B_ovoo += np.einsum("mnef,efbinj->mbij", H0.ab.oovv, T.abb, optimize=True)
    I2B_ovoo += H.ab.ovoo

    I2B_ovoo -= np.einsum("me,ebij->mbij", H.a.ov, T.ab, optimize=True)
    I2B_vvov = -np.einsum("nmfe,afbinm->abie", H0.ab.oovv, T.aab, optimize=True)
    I2B_vvov += -0.5 * np.einsum("nmfe,afbinm->abie", H0.bb.oovv, T.abb, optimize=True)
    I2B_vvov += H.ab.vvov

    I2B_vooo = np.einsum("nmfe,afeinj->amij", H0.ab.oovv, T.aab, optimize=True)
    I2B_vooo += 0.5 * np.einsum("nmfe,afeinj->amij", H0.bb.oovv, T.abb, optimize=True)
    I2B_vooo += H.ab.vooo
    I2B_vooo -= np.einsum("me,aeij->amij", H.b.ov, T.ab, optimize=True)

    I2C_vvov = -0.5 * np.einsum("mnef,abfimn->abie", H0.bb.oovv, T.bbb, optimize=True)
    I2C_vvov += -np.einsum("nmfe,fabnim->abie", H0.ab.oovv, T.abb, optimize=True)
    I2C_vvov += H.bb.vvov

    I2C_vooo = np.einsum("nmfe,faenij->amij", H0.ab.oovv, T.abb, optimize=True)
    I2C_vooo += 0.5 * np.einsum("mnef,aefijn->amij", H0.bb.oovv, T.bbb, optimize=True)
    I2C_vooo -= np.einsum("me,cekj->cmkj", H.b.ov, T.bb, optimize=True)
    I2C_vooo += H.bb.vooo

    # MM(2,3)C
    dT.abb = 0.5 * np.einsum("abie,ecjk->abcijk", I2B_vvov, T.bb, optimize=True)
    dT.abb -= 0.5 * np.einsum("amij,bcmk->abcijk", I2B_vooo, T.bb, optimize=True)
    dT.abb += 0.5 * np.einsum("cbke,aeij->abcijk", I2C_vvov, T.ab, optimize=True)
    dT.abb -= 0.5 * np.einsum("cmkj,abim->abcijk", I2C_vooo, T.ab, optimize=True)
    dT.abb += np.einsum("abej,ecik->abcijk", I2B_vvvo, T.ab, optimize=True)
    dT.abb -= np.einsum("mbij,acmk->abcijk", I2B_ovoo, T.ab, optimize=True)
    # (H(2) * T3)_C
    dT.abb -= 0.25 * np.einsum("mi,abcmjk->abcijk", H.a.oo, T.abb, optimize=True)
    dT.abb -= 0.5 * np.einsum("mj,abcimk->abcijk", H.b.oo, T.abb, optimize=True)
    dT.abb += 0.25 * np.einsum("ae,ebcijk->abcijk", H.a.vv, T.abb, optimize=True)
    dT.abb += 0.5 * np.einsum("be,aecijk->abcijk", H.b.vv, T.abb, optimize=True)
    dT.abb += 0.125 * np.einsum("mnjk,abcimn->abcijk", H.bb.oooo, T.abb, optimize=True)
    dT.abb += 0.5 * np.einsum("mnij,abcmnk->abcijk", H.ab.oooo, T.abb, optimize=True)
    dT.abb += 0.125 * np.einsum("bcef,aefijk->abcijk", H.bb.vvvv, T.abb, optimize=True)
    dT.abb += 0.5 * np.einsum("abef,efcijk->abcijk", H.ab.vvvv, T.abb, optimize=True)
    dT.abb += 0.25 * np.einsum("amie,ebcmjk->abcijk", H.aa.voov, T.abb, optimize=True)
    dT.abb += 0.25 * np.einsum("amie,ebcmjk->abcijk", H.ab.voov, T.bbb, optimize=True)
    dT.abb += np.einsum("mbej,aecimk->abcijk", H.ab.ovvo, T.aab, optimize=True)
    dT.abb += np.einsum("bmje,aecimk->abcijk", H.bb.voov, T.abb, optimize=True)
    dT.abb -= 0.5 * np.einsum("mbie,aecmjk->abcijk", H.ab.ovov, T.abb, optimize=True)
    dT.abb -= 0.5 * np.einsum("amej,ebcimk->abcijk", H.ab.vovo, T.abb, optimize=True)

    dT.abb -= np.transpose(dT.abb, (0, 2, 1, 3, 4, 5))
    dT.abb -= np.transpose(dT.abb, (0, 1, 2, 3, 5, 4))

    return T, dT

def calc_full_update_t3d(T, dT, H, H0, shift, system):

    I2C_vvov = -0.5 * np.einsum("mnef,abfimn->abie", H0.bb.oovv, T.bbb, optimize=True)
    I2C_vvov -= np.einsum("nmfe,fabnim->abie", H0.ab.oovv, T.abb, optimize=True)
    I2C_vvov += np.einsum("me,abim->abie", H.b.ov, T.bb, optimize=True)
    I2C_vvov += H.bb.vvov

    I2C_vooo = 0.5 * np.einsum("mnef,aefijn->amij", H0.bb.oovv, T.bbb, optimize=True)
    I2C_vooo += np.einsum("nmfe,faenij->amij", H0.ab.oovv, T.abb, optimize=True)
    I2C_vooo += H.bb.vooo

    # MM(2,3)D
    dT.bbb = -0.25 * np.einsum("amij,bcmk->abcijk", I2C_vooo, T.bb, optimize=True)
    dT.bbb += 0.25 * np.einsum("abie,ecjk->abcijk", I2C_vvov, T.bb, optimize=True)

    # (H(2) * T3)_C
    dT.bbb -= (1.0 / 12.0) * np.einsum("mk,abcijm->abcijk", H.b.oo, T.bbb, optimize=True)
    dT.bbb += (1.0 / 12.0) * np.einsum("ce,abeijk->abcijk", H.b.vv, T.bbb, optimize=True)
    dT.bbb += (1.0 / 24.0) * np.einsum("mnij,abcmnk->abcijk", H.bb.oooo, T.bbb, optimize=True)
    dT.bbb += (1.0 / 24.0) * np.einsum("abef,efcijk->abcijk", H.bb.vvvv, T.bbb, optimize=True)
    dT.bbb += 0.25 * np.einsum("maei,ebcmjk->abcijk", H.ab.ovvo, T.abb, optimize=True)
    dT.bbb += 0.25 * np.einsum("amie,ebcmjk->abcijk", H.bb.voov, T.bbb, optimize=True)

    dT.bbb -= np.transpose(dT.bbb, (1, 0, 2, 3, 4, 5)) + np.transpose(dT.bbb, (0, 2, 1, 3, 4, 5)) \
           + np.transpose(dT.bbb, (2, 1, 0, 3, 4, 5)) - np.transpose(dT.bbb, (1, 2, 0, 3, 4, 5)) \
           - np.transpose(dT.bbb, (2, 0, 1, 3, 4, 5))

    dT.bbb -= np.transpose(dT.bbb, (0, 1, 2, 4, 3, 5)) + np.transpose(dT.bbb, (0, 1, 2, 3, 5, 4)) \
           + np.transpose(dT.bbb, (0, 1, 2, 5, 4, 3)) - np.transpose(dT.bbb, (0, 1, 2, 4, 5, 3)) \
           - np.transpose(dT.bbb, (0, 1, 2, 5, 3, 4))

    return T, dT

if __name__ == "__main__":

    mol = gto.Mole()

    case = 'F2'

    if case == 'F2':
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

    if case == 'H2O':
        mol.build(
            atom="""H 0.0 -1.515263  -1.058898
                    H 0.0 1.515263  -1.058898
                    O 0.0 0.0 -0.0090""",
            basis="ccpvdz",
            charge=0,
            spin=0,
            symmetry="C2V",
            cart=True,
            unit='Bohr',
        )
        mf = scf.RHF(mol)
        mf.kernel()



    system, H = load_pyscf_integrals(mf, nfrozen=2,
                                     num_act_holes_alpha=5, num_act_particles_alpha=9,
                                     num_act_holes_beta=5, num_act_particles_beta=9,
                                     )
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    calculation = Calculation(
        order=3,
        active_orders=[None],
        num_active=[None],
        calculation_type="ccsdt",
        convergence_tolerance=1.0e-08,
        maximum_iterations=5,
    )
    T, total_energy, is_converged = cc_driver(calculation, system, H)

    # create full residual container
    dT = ClusterOperator(system, order=3)

    # create acitve-space cluster operator and residual container
    T_act = ClusterOperator(system, order=3, active_orders=[3], num_active=[1])
    dT_act = ClusterOperator(system, order=3, active_orders=[3], num_active=[1])

    # zero amplitudes outside active space
    T = zero_t3aaa_outside_active_space(T, system, 1)
    T = zero_t3aab_outside_active_space(T, system, 1)
    T = zero_t3abb_outside_active_space(T, system, 1)
    T = zero_t3bbb_outside_active_space(T, system, 1)

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

    HBar1 = get_ccs_intermediates(T, H)
    HBar2 = get_ccsd_intermediates(T, H)
    # # get the full update
    # _, dT = calc_full_update_t1a(T, dT, H, 0.0, system)
    # _, dT = calc_full_update_t1b(T, dT, H, 0.0, system)
    # #HBar1 = get_ccs_intermediates(T1b, H)
    # _, dT = calc_full_update_t2a(T, dT, HBar1, H, 0.0, system)
    # _, dT = calc_full_update_t2b(T, dT, HBar1, H, 0.0, system)
    # _, dT = calc_full_update_t2c(T, dT, HBar1, H, 0.0, system)
    #HBar2 = build_hbar_ccsd(T2c, H)
    _, dT = calc_full_update_t3a(T, dT, HBar2, H, 0.0, system)
    # _, dT = calc_full_update_t3b(T, dT, HBar2, H, 0.0, system)
    # _, dT = calc_full_update_t3c(T, dT, HBar2, H, 0.0, system)
    # _, dT = calc_full_update_t3d(T, dT, HBar2, H, 0.0, system)

    # active-space update
    # CCSD intermediates
    hbar = get_ccsd_intermediates(T_act, H)
    # add on (V * T3)_C intermediates
    hbar = intermediates.build_VT3_intermediates(T_act, hbar, system)
    T_act, dT_act = update(T_act, dT_act, H, 0.0, False, system)

    T, dT = update_t3a_full(T, dT, H, system)


    print('VVVOOO')
    print_error(dT_act.aaa.VVVOOO, dT.aaa[Va, Va, Va, Oa, Oa, Oa])
    print_error(T_act.aaa.VVVOOO, T.aaa[Va, Va, Va, Oa, Oa, Oa])
    print('VVvOOO')
    print_error(dT_act.aaa.VVvOOO, dT.aaa[Va, Va, va, Oa, Oa, Oa])
    print_error(T_act.aaa.VVvOOO, T.aaa[Va, Va, va, Oa, Oa, Oa])
    print('VVVoOO')
    print_error(dT_act.aaa.VVVoOO, dT.aaa[Va, Va, Va, oa, Oa, Oa])
    print_error(T_act.aaa.VVVoOO, T.aaa[Va, Va, Va, oa, Oa, Oa])
    print('VVvoOO')
    print_error(dT_act.aaa.VVvoOO, dT.aaa[Va, Va, va, oa, Oa, Oa])
    print_error(T_act.aaa.VVvoOO, T.aaa[Va, Va, va, oa, Oa, Oa])
    print('VvvOOO')
    print_error(dT_act.aaa.VvvOOO, dT.aaa[Va, va, va, Oa, Oa, Oa])
    print_error(T_act.aaa.VvvOOO, T.aaa[Va, va, va, Oa, Oa, Oa])
    print('VVVooO')
    print_error(dT_act.aaa.VVVooO, dT.aaa[Va, Va, Va, oa, oa, Oa])
    print_error(T_act.aaa.VVVooO, T.aaa[Va, Va, Va, oa, oa, Oa])
    print('VVvooO')
    print_error(dT_act.aaa.VVvooO, dT.aaa[Va, Va, va, oa, oa, Oa])
    print_error(T_act.aaa.VVvooO, T.aaa[Va, Va, va, oa, oa, Oa])
    print('VvvoOO')
    print_error(dT_act.aaa.VvvoOO, dT.aaa[Va, va, va, oa, Oa, Oa])
    print_error(T_act.aaa.VvvoOO, T.aaa[Va, va, va, oa, Oa, Oa])
    print('VvvooO')
    print_error(dT_act.aaa.VvvooO, dT.aaa[Va, va, va, oa, oa, Oa])
    print_error(T_act.aaa.VvvooO, T.aaa[Va, va, va, oa, oa, Oa])


    # Get the error
    #calc_error(T_act, T, system)