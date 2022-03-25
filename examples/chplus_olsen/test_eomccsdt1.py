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

from ccpy.hbar.hbar_ccsdt import build_hbar_ccsdt
from ccpy.hbar.eomccsdt_intermediates import get_eomccsd_intermediates, add_R3_terms
from ccpy.eomcc.eomccsdt import build_HR_3A, build_HR_3B, build_HR_3C, build_HR_3D, build_HR_1A, build_HR_1B, build_HR_2A, build_HR_2B, build_HR_2C

from ccpy.eomcc.initial_guess import get_initial_guess

from ccpy.eomcc.eomccsdt1_updates.intermediates import add_HR3_intermediates
from ccpy.eomcc.eomccsdt1_updates import *

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
    dR.aaa = calc_full_HR3A(R, T, Hbar, HR)


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

    HR_act = get_eomccsd_intermediates(Hbar, R_act, T_act, system)
    HR_act = add_HR3_intermediates(HR_act, Hbar, R_act, system)


    dR_act.aaa.VVVOOO = r3a_111111.build(R_act, T_act, Hbar, HR_act, system)
    dR_act.aaa.VVVoOO = r3a_111011.build(R_act, T_act, Hbar, HR_act, system)
    dR_act.aaa.VVvOOO = r3a_110111.build(R_act, T_act, Hbar, HR_act, system)
    dR_act.aaa.VVvoOO = r3a_110011.build(R_act, T_act, Hbar, HR_act, system)
    dR_act.aaa.VvvOOO = r3a_100111.build(R_act, T_act, Hbar, HR_act, system)
    dR_act.aaa.VVVooO = r3a_111001.build(R_act, T_act, Hbar, HR_act, system)
    dR_act.aaa.VvvoOO = r3a_100011.build(R_act, T_act, Hbar, HR_act, system)
    dR_act.aaa.VVvooO = r3a_110001.build(R_act, T_act, Hbar, HR_act, system)
    dR_act.aaa.VvvooO = r3a_100001.build(R_act, T_act, Hbar, HR_act, system)


    # Get the error
    calc_error(dR_act, dR, system)