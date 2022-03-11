import numpy as np

from pyscf import gto, scf

from ccpy.models.operators import ClusterOperator

from ccpy.models.calculation import Calculation
from ccpy.interfaces.pyscf_tools import load_pyscf_integrals
from ccpy.drivers.driver import cc_driver

from ccpy.hbar.hbar_ccsd import build_hbar_ccsd

from ccpy.utilities.active_space import fill_t3aaa, fill_t3aab, fill_t3abb, fill_t3bbb, get_active_slices,\
    zero_t3aaa_outside_active_space, zero_t3bbb_outside_active_space, zero_t3aab_outside_active_space, zero_t3abb_outside_active_space

from ccpy.cc.ccsdt1_updates import *

def calc_error(dT_act, dT, system):

    def _get_error(x, y):
        error = x - y
        return np.linalg.norm(error.flatten())

    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

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



def calc_full_update_t3a(T, dT, H, H0, shift, system):

    I2A_vvov = -0.5 * np.einsum("mnef,abfimn->abie", H0.aa.oovv, T.aaa, optimize=True)
    I2A_vvov -= np.einsum("mnef,abfimn->abie", H0.ab.oovv, T.aab, optimize=True)
    I2A_vvov += H.aa.vvov + np.einsum("me,abim->abie", H.a.ov, T.aa, optimize=True)

    I2A_vooo = 0.5 * np.einsum("mnef,aefijn->amij", H0.aa.oovv, T.aaa, optimize=True)
    I2A_vooo += H.aa.vooo + np.einsum("mnef,aefijn->amij", H0.ab.oovv, T.aab, optimize=True)

    # MM(2,3)A
    dT.aaa = -0.25 * np.einsum("amij,bcmk->abcijk", I2A_vooo, T.aa, optimize=True)
    dT.aaa += 0.25 * np.einsum("abie,ecjk->abcijk", I2A_vvov, T.aa, optimize=True)

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

    VT3 = {'aa': {'vooo': I2A_vooo, 'vvov': I2A_vvov},
           'ab': {'vvvo' : I2B_vvvo, 'ovoo': I2B_ovoo, 'vvov' : I2B_vvov, 'vooo' : I2B_vooo},
           'bb': {},
           }

    # MM(2,3)B
    dT.aab = 0.5 * np.einsum("bcek,aeij->abcijk", I2B_vvvo, T.aa, optimize=True)
    dT.aab -= 0.5 * np.einsum("mcjk,abim->abcijk", I2B_ovoo, T.aa, optimize=True)
    dT.aab += np.einsum("acie,bejk->abcijk", I2B_vvov, T.ab, optimize=True)
    dT.aab -= np.einsum("amik,bcjm->abcijk", I2B_vooo, T.ab, optimize=True)
    dT.aab += 0.5 * np.einsum("abie,ecjk->abcijk", I2A_vvov, T.ab, optimize=True)
    dT.aab -= 0.5 * np.einsum("amij,bcmk->abcijk", I2A_vooo, T.ab, optimize=True)

    # calculate full update
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

    return T, dT, VT3

def calc_full_update_t3c(T, dT, H, H0, shift, system):

    # calculate full update
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

    # calcualte full update
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

def update_t3a(T, dT, H, H0, shift, system):
    _, dT = update_t3a_111111.update(T, dT, H, H0, shift, system)
    _, dT = update_t3a_110111.update(T, dT, H, H0, shift, system)
    _, dT = update_t3a_111011.update(T, dT, H, H0, shift, system)
    _, dT = update_t3a_110011.update(T, dT, H, H0, shift, system)
    _, dT = update_t3a_100111.update(T, dT, H, H0, shift, system)
    _, dT = update_t3a_111001.update(T, dT, H, H0, shift, system)
    _, dT = update_t3a_100011.update(T, dT, H, H0, shift, system)
    _, dT = update_t3a_110001.update(T, dT, H, H0, shift, system)
    _, dT = update_t3a_100001.update(T, dT, H, H0, shift, system)

    return T, dT

def update_t3b(T, dT, H, H0, shift, system):

    _, dT = update_t3b_111111.update(T, dT, H, H0, shift, system)
    # one inactive
    _, dT = update_t3b_111110.update(T, dT, H, H0, shift, system)
    _, dT = update_t3b_111011.update(T, dT, H, H0, shift, system)
    _, dT = update_t3b_110111.update(T, dT, H, H0, shift, system)
    _, dT = update_t3b_101111.update(T, dT, H, H0, shift, system)
    # two inactive
    _, dT = update_t3b_111001.update(T, dT, H, H0, shift, system)
    _, dT = update_t3b_111010.update(T, dT, H, H0, shift, system)
    _, dT = update_t3b_001111.update(T, dT, H, H0, shift, system)
    _, dT = update_t3b_100111.update(T, dT, H, H0, shift, system)
    _, dT = update_t3b_110011.update(T, dT, H, H0, shift, system)
    _, dT = update_t3b_101110.update(T, dT, H, H0, shift, system)
    _, dT = update_t3b_101011.update(T, dT, H, H0, shift, system)
    _, dT = update_t3b_110110.update(T, dT, H, H0, shift, system)
    # three inactive
    _, dT = update_t3b_101001.update(T, dT, H, H0, shift, system)
    _, dT = update_t3b_101010.update(T, dT, H, H0, shift, system)
    _, dT = update_t3b_110001.update(T, dT, H, H0, shift, system)
    _, dT = update_t3b_110010.update(T, dT, H, H0, shift, system)
    _, dT = update_t3b_001011.update(T, dT, H, H0, shift, system)
    _, dT = update_t3b_001110.update(T, dT, H, H0, shift, system)
    _, dT = update_t3b_100011.update(T, dT, H, H0, shift, system)
    _, dT = update_t3b_100110.update(T, dT, H, H0, shift, system)
    # four inactive
    _, dT = update_t3b_001001.update(T, dT, H, H0, shift, system)
    _, dT = update_t3b_001010.update(T, dT, H, H0, shift, system)
    _, dT = update_t3b_100001.update(T, dT, H, H0, shift, system)
    _, dT = update_t3b_100010.update(T, dT, H, H0, shift, system)

    return T, dT

def update_t3c(T, dT, H, H0, shift, system):

    _, dT = update_t3c_111111.update(T, dT, H, H0, shift, system)
    # one inactive
    _, dT = update_t3c_111101.update(T, dT, H, H0, shift, system)
    _, dT = update_t3c_111011.update(T, dT, H, H0, shift, system)
    _, dT = update_t3c_110111.update(T, dT, H, H0, shift, system)
    _, dT = update_t3c_011111.update(T, dT, H, H0, shift, system)
    # two inactive
    _, dT = update_t3c_111100.update(T, dT, H, H0, shift, system)
    _, dT = update_t3c_111001.update(T, dT, H, H0, shift, system)
    _, dT = update_t3c_010111.update(T, dT, H, H0, shift, system)
    _, dT = update_t3c_100111.update(T, dT, H, H0, shift, system)
    _, dT = update_t3c_110011.update(T, dT, H, H0, shift, system)
    _, dT = update_t3c_011101.update(T, dT, H, H0, shift, system)
    _, dT = update_t3c_011011.update(T, dT, H, H0, shift, system)
    _, dT = update_t3c_110101.update(T, dT, H, H0, shift, system)
    # three inactive
    _, dT = update_t3c_110100.update(T, dT, H, H0, shift, system)
    _, dT = update_t3c_110001.update(T, dT, H, H0, shift, system)
    _, dT = update_t3c_011100.update(T, dT, H, H0, shift, system)
    _, dT = update_t3c_011001.update(T, dT, H, H0, shift, system)
    _, dT = update_t3c_100011.update(T, dT, H, H0, shift, system)
    _, dT = update_t3c_100101.update(T, dT, H, H0, shift, system)
    _, dT = update_t3c_010011.update(T, dT, H, H0, shift, system)
    _, dT = update_t3c_010101.update(T, dT, H, H0, shift, system)
    # four inactive
    _, dT = update_t3c_010100.update(T, dT, H, H0, shift, system)
    _, dT = update_t3c_010001.update(T, dT, H, H0, shift, system)
    _, dT = update_t3c_100100.update(T, dT, H, H0, shift, system)
    _, dT = update_t3c_100001.update(T, dT, H, H0, shift, system)

    return T, dT

def update_t3d(T, dT, H, H0, shift, system):

    _, dT = update_t3d_111111.update(T, dT, H, H0, shift, system)
    _, dT = update_t3d_110111.update(T, dT, H, H0, shift, system)
    _, dT = update_t3d_111011.update(T, dT, H, H0, shift, system)
    _, dT = update_t3d_110011.update(T, dT, H, H0, shift, system)
    _, dT = update_t3d_100111.update(T, dT, H, H0, shift, system)
    _, dT = update_t3d_111001.update(T, dT, H, H0, shift, system)
    _, dT = update_t3d_100011.update(T, dT, H, H0, shift, system)
    _, dT = update_t3d_110001.update(T, dT, H, H0, shift, system)
    _, dT = update_t3d_100001.update(T, dT, H, H0, shift, system)

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
                                     num_act_holes_alpha=4, num_act_particles_alpha=10,
                                     num_act_holes_beta=4, num_act_particles_beta=10
                                     )

    calculation = Calculation(
        order=3,
        active_orders=[None],
        num_active=[None],
        calculation_type="ccsdt",
        convergence_tolerance=1.0e-08
    )

    T, total_energy, is_converged = cc_driver(calculation, system, H)
    HBar = build_hbar_ccsd(T, H)

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

    # get the full update
    _, dT = calc_full_update_t3a(T, dT, HBar, H, 0.0, system)
    _, dT, VT3_exact = calc_full_update_t3b(T, dT, HBar, H, 0.0, system)
    _, dT = calc_full_update_t3c(T, dT, HBar, H, 0.0, system)
    _, dT = calc_full_update_t3d(T, dT, HBar, H, 0.0, system)

    # update CCSD HBar with the (V*T3)_C two-body intermediates
    HBar = intermediates.build_VT3_intermediates(T_act, HBar, system)

    #error = VT3['ab']['vvov'] - VT3_exact['ab']['vvov']
    #print("error in vvov = ", np.linalg.norm(error.flatten()))

    _, dT_act = update_t3a(T_act, dT_act, HBar, H, 0.0, system)
    _, dT_act = update_t3b(T_act, dT_act, HBar, H, 0.0, system)
    _, dT_act = update_t3c(T_act, dT_act, HBar, H, 0.0, system)
    _, dT_act = update_t3d(T_act, dT_act, HBar, H, 0.0, system)

    # Get the error
    calc_error(dT_act, dT, system)