"""Functions to calculate the ground-state CR-CC(2,4)-like Q space quadruples correction to CC(P)."""
import time

from ccpy.moments.ccp34 import calc_ccp4_with_selection


def calc_ccp4(T, L, corr_energy, H, H0, system, t4_excitations, use_RHF=False):
    """
    Calculate the ground-state CC(P;4) correction: the CR-CC(2,4) quadruples correction over the
    quadruples outside the P space ``t4_excitations``.
    """
    t_start = time.perf_counter()
    t_cpu_start = time.process_time()

    delta, _, _, _ = calc_ccp4_with_selection(T, L, t4_excitations, H0, H, system, use_RHF=use_RHF)
    deltap4 = dict(zip("ABCD", delta))
    Eccp4 = {v: system.reference_energy + corr_energy + d for v, d in deltap4.items()}

    minutes, seconds = divmod(time.perf_counter() - t_start, 60)
    print('   CC(P;4) Calculation Summary')
    print('   -------------------------------------')
    print("   Total wall time: {:0.2f}m  {:0.2f}s".format(minutes, seconds))
    print(f"   Total CPU time: {time.process_time() - t_cpu_start} seconds\n")
    print("   CC(P) = {:>10.10f}".format(system.reference_energy + corr_energy))
    for v in "ABCD":
        print("   CC(P;4)_{} = {:>10.10f}     ΔE_{} = {:>10.10f}     δ_{} = {:>10.10f}".format(
            v, Eccp4[v], v, corr_energy + deltap4[v], v, deltap4[v]))
    print()

    return Eccp4, deltap4
