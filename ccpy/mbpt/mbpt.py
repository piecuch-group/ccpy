import time
from ccpy.lib.core import mbpt_loops

# [TODO]: Generalize MPn methods to MBPT(n) methods for non-HF orbitals.

def calc_mp2(system, H):

    t_start = time.perf_counter()
    t_cpu_start = time.process_time()
    corr_energy = mbpt_loops.mp2(H.a.oo, H.a.vv, H.b.oo, H.b.vv,
                                            H.aa.oovv, H.aa.vvoo, H.ab.oovv, H.ab.vvoo, H.bb.oovv, H.bb.vvoo)
    t_end = time.perf_counter()
    t_cpu_end = time.process_time()
    minutes, seconds = divmod(t_end - t_start, 60)

    print('\n   MBPT(2) Calculation Summary')
    print('   -------------------------------------')
    print("   Total wall time: {:0.2f}m  {:0.2f}s".format(minutes, seconds))
    print(f"   Total CPU time: {t_cpu_end - t_cpu_start} seconds\n")
    print("   Reference = {:>10.10f}".format(system.reference_energy))
    print("   MBPT(2) = {:>10.10f}     ΔE = {:>10.10f}".format(system.reference_energy + corr_energy, corr_energy))

    return corr_energy

def calc_mp3(system, H):

    t_start = time.perf_counter()
    t_cpu_start = time.process_time()
    corr_energy2 = mbpt_loops.mp2(H.a.oo, H.a.vv, H.b.oo, H.b.vv,
                                            H.aa.oovv, H.aa.vvoo, H.ab.oovv, H.ab.vvoo, H.bb.oovv, H.bb.vvoo)
    corr_energy3 = mbpt_loops.mp3(H.a.oo, H.a.vv, H.b.oo, H.b.vv,
                                             H.aa.oovv, H.aa.vvoo, H.aa.voov, H.aa.oooo, H.aa.vvvv,
                                             H.ab.oovv, H.ab.vvoo, H.ab.voov, H.ab.ovvo, H.ab.vovo, H.ab.ovov, H.ab.oooo, H.ab.vvvv,
                                             H.bb.oovv, H.bb.vvoo, H.bb.voov, H.bb.oooo, H.bb.vvvv)
    t_end = time.perf_counter()
    t_cpu_end = time.process_time()
    minutes, seconds = divmod(t_end - t_start, 60)

    corr_energy = corr_energy2 + corr_energy3

    print('\n   MBPT(3) Calculation Summary')
    print('   -------------------------------------')
    print("   Total wall time: {:0.2f}m  {:0.2f}s".format(minutes, seconds))
    print(f"   Total CPU time: {t_cpu_end - t_cpu_start} seconds\n")
    print("   Reference = {:>10.10f}".format(system.reference_energy))
    print("   2nd-order contribution = {:>10.10f}".format(corr_energy2))
    print("   3rd-order contribution = {:>10.10f}".format(corr_energy3))
    print("   MBPT(3) = {:>10.10f}     ΔE = {:>10.10f}".format(system.reference_energy + corr_energy, corr_energy))

    return corr_energy

# [TODO]: Fix error in MP4. Correlation energy is not correct.
def calc_mp4(system, H):

    t_start = time.perf_counter()
    t_cpu_start = time.process_time()
    corr_energy2 = mbpt_loops.mp2(H.a.oo, H.a.vv, H.b.oo, H.b.vv,
                                            H.aa.oovv, H.aa.vvoo, H.ab.oovv, H.ab.vvoo, H.bb.oovv, H.bb.vvoo)
    corr_energy3 = mbpt_loops.mp3(H.a.oo, H.a.vv, H.b.oo, H.b.vv,
                                             H.aa.oovv, H.aa.vvoo, H.aa.voov, H.aa.oooo, H.aa.vvvv,
                                             H.ab.oovv, H.ab.vvoo, H.ab.voov, H.ab.ovvo, H.ab.vovo, H.ab.ovov, H.ab.oooo, H.ab.vvvv,
                                             H.bb.oovv, H.bb.vvoo, H.bb.voov, H.bb.oooo, H.bb.vvvv)
    corr_energy4, corr_singles, corr_doubles, corr_triples, corr_quadruples = mbpt_loops.mp4(H.a.oo, H.a.vv, H.b.oo, H.b.vv,
                                             H.aa.oovv, H.aa.vvoo, H.aa.voov, H.aa.oooo, H.aa.vvvv, H.aa.vooo, H.aa.vvov, H.aa.ooov, H.aa.vovv,
                                             H.ab.oovv, H.ab.vvoo, H.ab.voov, H.ab.ovvo, H.ab.vovo, H.ab.ovov, H.ab.oooo, H.ab.vvvv,
                                             H.ab.vooo, H.ab.ovoo, H.ab.vvov, H.ab.vvvo, H.ab.ooov, H.ab.oovo, H.ab.vovv, H.ab.ovvv,
                                             H.bb.oovv, H.bb.vvoo, H.bb.voov, H.bb.oooo, H.bb.vvvv, H.bb.vooo, H.bb.vvov, H.bb.ooov, H.bb.vovv)
    t_end = time.perf_counter()
    t_cpu_end = time.process_time()
    minutes, seconds = divmod(t_end - t_start, 60)

    corr_energy = corr_energy2 + corr_energy3 + corr_energy4

    print('\n   MBPT(4) Calculation Summary')
    print('   -------------------------------------')
    print("   Total wall time: {:0.2f}m  {:0.2f}s".format(minutes, seconds))
    print(f"   Total CPU time: {t_cpu_end - t_cpu_start} seconds\n")
    print("   Reference = {:>10.10f}".format(system.reference_energy))
    print("   2nd-order contribution = {:>10.10f}".format(corr_energy2))
    print("   3rd-order contribution = {:>10.10f}".format(corr_energy3))
    print("   4th-order contribution = {:>10.10f}".format(corr_energy4))
    print("        E(T1) = {:>10.10f}".format(corr_singles))
    print("        E(T2) = {:>10.10f}".format(corr_doubles))
    print("        E(T3) = {:>10.10f}".format(corr_triples))
    print("        E(T2**2) = {:>10.10f}".format(corr_quadruples))
    print("   MBPT(4) = {:>10.10f}     ΔE = {:>10.10f}".format(system.reference_energy + corr_energy, corr_energy))

    return corr_energy
