import datetime
from ccpy.utilities.utilities import get_memory_usage

WHITESPACE = "  "

ITERATION_HEADER_FMT = "{:>10} {:>12} {:>14} {:>17} {:>19} {:>12}"
ITERATION_FMT = "{:>8} {:>17.10f} {:>17.10f} {:>17.10f} {:>15} {:>12}"

CC_ITERATION_HEADER = ITERATION_HEADER_FMT.format(
    "Iter.", "Residuum", "δE", "ΔE", "Wall time", "Memory"
)
EOMCC_ITERATION_HEADER = ITERATION_HEADER_FMT.format(
    "Iter.", "Residuum", "ω", "δω", "Wall time", "Memory"
)

def get_timestamp():
        return datetime.datetime.strptime(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H:%M:%S")

def cc_calculation_summary(T, reference_energy, cc_energy, system, print_thresh):
    DATA_FMT = "{:<30} {:>20.8f}"
    print("\n   CC Calculation Summary")
    print("  --------------------------------------------------")
    print(DATA_FMT.format("   Reference energy", reference_energy))
    print(DATA_FMT.format("   CC correlation energy", cc_energy))
    print(DATA_FMT.format("   Total CC energy", reference_energy + cc_energy))
    print_ee_amplitudes(T, system, T.order, print_thresh)
    print("")

def lrcc_calculation_summary(T1, lrcc_energy, system, print_thresh):
    DATA_FMT = "{:<30} {:>20.8f}"
    print("\n   LR-CC Calculation Summary")
    print("  --------------------------------------------------")
    print(DATA_FMT.format("   LR-CC correlation property", lrcc_energy))
    print_ee_amplitudes(T1, system, T1.order, print_thresh)
    print("")

def eomcc_calculation_summary(R, omega, corr_energy, r0, rel, is_converged, istate, system, print_thresh):
    DATA_FMT = "{:<30} {:>20.8f}"
    if is_converged:
        convergence_label = 'converged'
    else:
        convergence_label = 'not converged'
    print("\n   EOMCC Calculation Summary (%s) - Root %i" % (convergence_label, istate))
    print("  --------------------------------------------------")
    print(DATA_FMT.format("   Vertical excitation energy", omega))
    print(DATA_FMT.format("   Reference state weight r0", r0))
    print(DATA_FMT.format("   Relative excitation level", rel))
    print(DATA_FMT.format("   Total EOMCC energy", system.reference_energy + corr_energy + omega))
    print_ee_amplitudes(R, system, R.order, print_thresh)
    print("")

def sfeomcc_calculation_summary(R, omega, corr_energy, is_converged, system, print_thresh):
    DATA_FMT = "{:<30} {:>20.8f}"
    if is_converged:
        convergence_label = 'converged'
    else:
        convergence_label = 'not converged'
    print("\n   SF-EOMCC Calculation Summary (%s)" % convergence_label)
    print("  --------------------------------------------------")
    print(DATA_FMT.format("   Vertical excitation energy", omega))
    print(DATA_FMT.format("   Total EOMCC energy", system.reference_energy + corr_energy + omega))
    print_sf_amplitudes(R, system, R.order, print_thresh)
    print("")

def deaeomcc_calculation_summary(R, omega, corr_energy, is_converged, system, print_thresh):
    DATA_FMT = "{:<30} {:>20.8f}"
    if is_converged:
        convergence_label = 'converged'
    else:
        convergence_label = 'not converged'
    print("\n   DEA-EOMCC Calculation Summary (%s)" % convergence_label)
    print("  --------------------------------------------------")
    print(DATA_FMT.format("   Vertical excitation energy", omega))
    print(DATA_FMT.format("   Total DEA-EOMCC energy", system.reference_energy + corr_energy + omega))
    print_dea_amplitudes(R, system, R.order, print_thresh)
    print("")

def dipeomcc_calculation_summary(R, omega, corr_energy, is_converged, system, print_thresh):
    DATA_FMT = "{:<30} {:>20.8f}"
    if is_converged:
        convergence_label = 'converged'
    else:
        convergence_label = 'not converged'
    print("\n   DIP-EOMCC Calculation Summary (%s)" % convergence_label)
    print("  --------------------------------------------------")
    print(DATA_FMT.format("   Vertical excitation energy", omega))
    print(DATA_FMT.format("   Total DIP-EOMCC energy", system.reference_energy + corr_energy + omega))
    print_dip_amplitudes(R, system, R.order, print_thresh)
    print("")

def ipeomcc_calculation_summary(R, omega, corr_energy, rel, is_converged, system, print_thresh):
    DATA_FMT = "{:<30} {:>20.8f}"
    if is_converged:
        convergence_label = 'converged'
    else:
        convergence_label = 'not converged'
    print("\n   IP-EOMCC Calculation Summary (%s)" % convergence_label)
    print("  --------------------------------------------------")
    print(DATA_FMT.format("   Vertical excitation energy", omega))
    print(DATA_FMT.format("   Relative excitation level", rel))
    print(DATA_FMT.format("   Total IP-EOMCC energy", system.reference_energy + corr_energy + omega))
    print_ip_amplitudes(R, system, R.order, print_thresh)
    print("")

def eaeomcc_calculation_summary(R, omega, corr_energy, rel, is_converged, system, print_thresh):
    DATA_FMT = "{:<30} {:>20.8f}"
    if is_converged:
        convergence_label = 'converged'
    else:
        convergence_label = 'not converged'
    print("\n   EA-EOMCC Calculation Summary (%s)" % convergence_label)
    print("  --------------------------------------------------")
    print(DATA_FMT.format("   Vertical excitation energy", omega))
    print(DATA_FMT.format("   Relative excitation level", rel))
    print(DATA_FMT.format("   Total EA-EOMCC energy", system.reference_energy + corr_energy + omega))
    print_ea_amplitudes(R, system, R.order, print_thresh)
    print("")

def leftcc_calculation_summary(L, omega, LR, is_converged, system, print_thresh):
    print("\n   Left CC Calculation Summary")
    print("  --------------------------------------------------------")
    if is_converged:
        convergence_label = 'converged'
    else:
        convergence_label = 'not converged'
    print("   Root", convergence_label, "   ω = %.8f" % omega, "  LR = %.8f" % LR)
    print_ee_amplitudes(L, system, L.order, print_thresh)
    print("")

def leftipcc_calculation_summary(L, omega, LR, is_converged, system, print_thresh):
    print("\n   Left IP-EOMCC Calculation Summary")
    print("  --------------------------------------------------------")
    if is_converged:
        convergence_label = 'converged'
    else:
        convergence_label = 'not converged'
    print("   Root", convergence_label, "   ω = %.8f" % omega, "  LR = %.8f" % LR)
    print_ip_amplitudes(L, system, L.order, print_thresh)
    print("")

def lefteacc_calculation_summary(L, omega, LR, is_converged, system, print_thresh):
    print("\n   Left EA-EOMCC Calculation Summary")
    print("  --------------------------------------------------------")
    if is_converged:
        convergence_label = 'converged'
    else:
        convergence_label = 'not converged'
    print("   Root", convergence_label, "   ω = %.8f" % omega, "  LR = %.8f" % LR)
    print_ea_amplitudes(L, system, L.order, print_thresh)
    print("")

def print_cc_iteration_header():
    print("\n", CC_ITERATION_HEADER)
    print('    '+(len(CC_ITERATION_HEADER)) * "-")

def print_eomcc_iteration_header():
    print("\n", EOMCC_ITERATION_HEADER)
    print('    '+(len(EOMCC_ITERATION_HEADER)) * "-")

def print_cc_iteration(
    iteration_idx, residuum, delta_energy, correlation_energy, elapsed_time
):
    minutes, seconds = divmod(elapsed_time, 60)
    time_str = f"({minutes:.1f}m {seconds:.1f}s)"
    memory = f"{round(get_memory_usage(), 2)} MB"
    print(
        ITERATION_FMT.format(
            iteration_idx, residuum, delta_energy, correlation_energy, time_str, memory,
        )
    )

def print_eomcc_iteration(
    iteration_idx, omega, residuum, delta_energy, elapsed_time
):
    minutes, seconds = divmod(elapsed_time, 60)
    time_str = f"({minutes:.1f}m {seconds:.1f}s)"
    memory = f"{round(get_memory_usage(), 2)} MB"
    print(
        ITERATION_FMT.format(
            iteration_idx, residuum, omega, delta_energy, time_str, memory
        )
    )

def print_block_eomcc_iteration(
    iteration_idx, curr_size, omega, residuum, delta_energy, elapsed_time, state_index
):
    minutes, seconds = divmod(elapsed_time, 60)
    time_str = f"({minutes:.1f}m {seconds:.1f}s)"
    memory = f"{round(get_memory_usage(), 2)} MB"
    for j, istate in enumerate(state_index):
        if j == 0:
            print(
                ITERATION_FMT.format(
                    iteration_idx, residuum[j], omega[istate], delta_energy[j], time_str, memory
                )
            )
        else:
            print(
                ITERATION_FMT.format(
                    "", residuum[j], omega[istate], delta_energy[j], time_str, memory
                )
            )
    print("      Current subspace size = ", curr_size)
    print("      ............................................................")

def print_ee_amplitudes(R, system, order, thresh_print):

    # Zero out the non-unique R amplitudes related by permutational symmetry
    for a in range(system.nunoccupied_alpha):
        for b in range(a + 1, system.nunoccupied_alpha):
            for i in range(system.noccupied_alpha):
                for j in range(i + 1, system.noccupied_alpha):
                    R.aa[b, a, j, i] = 0.0
                    R.aa[a, b, j, i] = 0.0
                    R.aa[b, a, i, j] = 0.0
    for a in range(system.nunoccupied_beta):
        for b in range(a + 1, system.nunoccupied_beta):
            for i in range(system.noccupied_beta):
                for j in range(i + 1, system.noccupied_beta):
                    R.bb[b, a, j, i] = 0.0
                    R.bb[a, b, j, i] = 0.0
                    R.bb[b, a, i, j] = 0.0

    print("\n   Largest Singly and Doubly Excited Amplitudes:")
    n = 1
    for a in range(system.nunoccupied_alpha):
        for i in range(system.noccupied_alpha):
            if abs(R.a[a, i]) <= thresh_print: continue
            print(
                "      [{}]     {}A  ->  {}A   =   {:.6f}".format(
                    n,
                    i + system.nfrozen + 1,
                    a + system.nfrozen + system.noccupied_alpha + 1,
                    R.a[a, i],
                )
            )
            n += 1
    for a in range(system.nunoccupied_beta):
        for i in range(system.noccupied_beta):
            if abs(R.b[a, i]) <= thresh_print: continue
            print(
                "      [{}]     {}B  ->  {}B   =   {:.6f}".format(
                    n,
                    i + system.nfrozen + 1,
                    a + system.nfrozen + system.noccupied_beta + 1,
                    R.b[a, i],
                )
            )
            n += 1
    for a in range(system.nunoccupied_alpha):
        for b in range(a + 1, system.nunoccupied_alpha):
            for i in range(system.noccupied_alpha):
                for j in range(i + 1, system.noccupied_alpha):
                    if abs(R.aa[a, b, i, j]) <= thresh_print: continue
                    print(
                        "      [{}]     {}A  {}A  ->  {}A  {}A  =   {:.6f}".format(
                            n,
                            i + system.nfrozen + 1,
                            j + system.nfrozen + 1,
                            a + system.noccupied_alpha + system.nfrozen + 1,
                            b + system.noccupied_alpha + system.nfrozen + 1,
                            R.aa[a, b, i, j],
                        )
                    )
                    n += 1
    for a in range(system.nunoccupied_beta):
        for b in range(a + 1, system.nunoccupied_beta):
            for i in range(system.noccupied_beta):
                for j in range(i + 1, system.noccupied_beta):
                    if abs(R.bb[a, b, i, j]) <= thresh_print: continue
                    print(
                        "      [{}]     {}B  {}B  ->  {}B  {}B  =   {:.6f}".format(
                            n,
                            i + system.nfrozen + 1,
                            j + system.nfrozen + 1,
                            a + system.noccupied_beta + system.nfrozen + 1,
                            b + system.noccupied_beta + system.nfrozen + 1,
                            R.bb[a, b, i, j],
                        )
                    )
                    n += 1
    for a in range(system.nunoccupied_alpha):
        for b in range(system.nunoccupied_beta):
            for i in range(system.noccupied_alpha):
                for j in range(system.noccupied_beta):
                    if abs(R.ab[a, b, i, j]) <= thresh_print: continue
                    print(
                        "      [{}]     {}A  {}B  ->  {}A  {}B  =   {:.6f}".format(
                            n,
                            i + system.nfrozen + 1,
                            j + system.nfrozen + 1,
                            a + system.noccupied_alpha + system.nfrozen + 1,
                            b + system.noccupied_beta + system.nfrozen + 1,
                            R.ab[a, b, i, j],
                        )
                    )
                    n += 1
    # Restore permutationally redundant amplitudes
    for a in range(system.nunoccupied_alpha):
        for b in range(a + 1, system.nunoccupied_alpha):
            for i in range(system.noccupied_alpha):
                for j in range(i + 1, system.noccupied_alpha):
                    R.aa[b, a, j, i] = R.aa[a, b, i, j]
                    R.aa[a, b, j, i] = -1.0 * R.aa[a, b, i, j]
                    R.aa[b, a, i, j] = -1.0 * R.aa[a, b, i, j]
    for a in range(system.nunoccupied_beta):
        for b in range(a + 1, system.nunoccupied_beta):
            for i in range(system.noccupied_beta):
                for j in range(i + 1, system.noccupied_beta):
                    R.bb[b, a, j, i] = R.bb[a, b, i, j]
                    R.bb[a, b, j, i] = -1.0 * R.bb[a, b, i, j]
                    R.bb[b, a, i, j] = -1.0 * R.bb[a, b, i, j]
    return

def print_ip_amplitudes(R, system, order, thresh_print):

    # Zero out the non-unique R amplitudes related by permutational symmetry
    for i in range(system.noccupied_alpha):
        for b in range(system.nunoccupied_alpha):
            for j in range(i + 1, system.noccupied_alpha):
                R.aa[j, b, i] = 0.0

    print("\n   Largest 1h and 2h-1p Excited Amplitudes:")
    n = 1
    for i in range(system.noccupied_alpha):
        if abs(R.a[i]) <= thresh_print: continue
        print(
            "      [{}]     {}A  ->   =   {:.6f}".format(
                n,
                i + system.nfrozen + 1,
                R.a[i],
            )
        )
        n += 1
    for i in range(system.noccupied_alpha):
        for b in range(system.nunoccupied_alpha):
            for j in range(i + 1, system.noccupied_alpha):
                if abs(R.aa[i, b, j]) <= thresh_print: continue
                print(
                    "      [{}]     {}A  {}A  ->  {}A  =   {:.6f}".format(
                        n,
                        i + system.nfrozen + 1,
                        j + system.nfrozen + 1,
                        b + system.noccupied_alpha + system.nfrozen + 1,
                        R.aa[i, b, j],
                    )
                )
                n += 1
    for i in range(system.noccupied_alpha):
        for b in range(system.nunoccupied_beta):
            for j in range(system.noccupied_beta):
                if abs(R.ab[i, b, j]) <= thresh_print: continue
                print(
                    "      [{}]     {}A  {}B  ->  {}B  =   {:.6f}".format(
                        n,
                        i + system.nfrozen + 1,
                        j + system.nfrozen + 1,
                        b + system.noccupied_beta + system.nfrozen + 1,
                        R.ab[i, b, j],
                    )
                )
                n += 1
    # Restore permutationally redundant amplitudes
    for i in range(system.noccupied_alpha):
        for b in range(system.nunoccupied_alpha):
            for j in range(i + 1, system.noccupied_alpha):
                R.aa[j, b, i] = -R.aa[i, b, j]
    return

def print_ea_amplitudes(R, system, order, thresh_print):

    # Zero out the non-unique R amplitudes related by permutational symmetry
    for a in range(system.nunoccupied_alpha):
        for b in range(a + 1, system.nunoccupied_alpha):
            for j in range(system.noccupied_alpha):
                R.aa[b, a, j] = 0.0

    print("\n   Largest 1p and 2p-1h Excited Amplitudes:")
    n = 1
    for a in range(system.nunoccupied_alpha):
        if abs(R.a[a]) <= thresh_print: continue
        print(
            "      [{}]     ->  {}A  =   {:.6f}".format(
                n,
                a + system.noccupied_alpha + system.nfrozen + 1,
                R.a[a],
            )
        )
        n += 1
    for a in range(system.nunoccupied_alpha):
        for b in range(a + 1, system.nunoccupied_alpha):
            for j in range(system.noccupied_alpha):
                if abs(R.aa[a, b, j]) <= thresh_print: continue
                print(
                    "      [{}]     {}A  ->  {}A  {}A  =   {:.6f}".format(
                        n,
                        j + system.nfrozen + 1,
                        a + system.noccupied_alpha + system.nfrozen + 1,
                        b + system.noccupied_alpha + system.nfrozen + 1,
                        R.aa[a, b, j],
                    )
                )
                n += 1
    for a in range(system.nunoccupied_alpha):
        for b in range(system.nunoccupied_beta):
            for j in range(system.noccupied_beta):
                if abs(R.ab[a, b, j]) <= thresh_print: continue
                print(
                    "      [{}]     {}B  ->  {}A  {}B  =   {:.6f}".format(
                        n,
                        j + system.nfrozen + 1,
                        a + system.noccupied_alpha + system.nfrozen + 1,
                        b + system.noccupied_beta + system.nfrozen + 1,
                        R.ab[a, b, j],
                    )
                )
                n += 1
    # Restore permutationally redundant amplitudes
    for a in range(system.nunoccupied_alpha):
        for b in range(a + 1, system.nunoccupied_alpha):
            for j in range(system.noccupied_alpha):
                R.aa[b, a, j] = -R.aa[a, b, j]
    return

def print_dea_amplitudes(R, system, order, thresh_print):

    # Zero out the non-unique R amplitudes related by permutational symmetry
    if R.order > 2:
        for a in range(system.nunoccupied_alpha):
            for b in range(system.nunoccupied_beta):
                for c in range(a + 1, system.nunoccupied_alpha):
                    for k in range(system.noccupied_alpha):
                        R.aba[c, b, a, k] = 0.0
        for a in range(system.nunoccupied_alpha):
            for b in range(system.nunoccupied_beta):
                for c in range(b + 1, system.nunoccupied_beta):
                    for k in range(system.noccupied_beta):
                        R.abb[a, c, b, k] = 0.0

    print("\n   Largest 2p and 3p-1h Excited Amplitudes:")
    n = 1
    for a in range(system.nunoccupied_alpha):
        for b in range(system.nunoccupied_beta):
            if abs(R.ab[a, b]) <= thresh_print: continue
            print(
                "      [{}]     ->  {}A  {}B  =   {:.6f}".format(
                    n,
                    a + system.nfrozen + system.noccupied_alpha + 1,
                    b + system.nfrozen + system.noccupied_beta + 1,
                    R.ab[a, b],
                )
            )
            n += 1
    if R.order > 2:
        for a in range(system.nunoccupied_alpha):
            for b in range(system.nunoccupied_beta):
                for c in range(a + 1, system.nunoccupied_alpha):
                    for k in range(system.noccupied_alpha):
                        if abs(R.aba[a, b, c, k]) <= thresh_print: continue
                        print(
                            "      [{}]     {}A  ->  {}A  {}B  {}A  =   {:.6f}".format(
                                n,
                                k + system.nfrozen + 1,
                                a + system.nfrozen + system.noccupied_alpha + 1,
                                b + system.noccupied_beta + system.nfrozen + 1,
                                c + system.noccupied_alpha + system.nfrozen + 1,
                                R.aba[a, b, c, k],
                            )
                        )
                        n += 1
        for a in range(system.nunoccupied_alpha):
            for b in range(system.nunoccupied_beta):
                for c in range(b + 1, system.nunoccupied_beta):
                    for k in range(system.noccupied_beta):
                        if abs(R.abb[a, b, c, k]) <= thresh_print: continue
                        print(
                            "      [{}]     {}B  ->  {}A  {}B  {}B  =   {:.6f}".format(
                                n,
                                k + system.nfrozen + 1,
                                a + system.nfrozen + system.noccupied_alpha + 1,
                                b + system.noccupied_beta + system.nfrozen + 1,
                                c + system.noccupied_beta + system.nfrozen + 1,
                                R.abb[a, b, c, k],
                            )
                        )
                        n += 1
        # Restore permutationally redundant amplitudes
        for a in range(system.nunoccupied_alpha):
            for b in range(system.nunoccupied_beta):
                for c in range(a + 1, system.nunoccupied_alpha):
                    for k in range(system.noccupied_alpha):
                        R.aba[c, b, a, k] = -R.aba[a, b, c, k]
        for a in range(system.nunoccupied_alpha):
            for b in range(system.nunoccupied_beta):
                for c in range(b + 1, system.nunoccupied_beta):
                    for k in range(system.noccupied_beta):
                        R.abb[a, c, b, k] = -R.abb[a, b, c, k]
    return

def print_dip_amplitudes(R, system, order, thresh_print):

    # Zero out the non-unique R amplitudes related by permutational symmetry
    if R.order > 2:
        for i in range(system.noccupied_alpha):
            for j in range(system.noccupied_beta):
                for c in range(system.nunoccupied_alpha):
                    for k in range(i + 1, system.noccupied_alpha):
                        R.aba[k, j, c, i] = 0.0
        for i in range(system.noccupied_alpha):
            for j in range(system.noccupied_beta):
                for c in range(system.nunoccupied_beta):
                    for k in range(j + 1, system.noccupied_beta):
                        R.abb[i, k, c, j] = 0.0

    print("\n   Largest 2h and 3h-1p Excited Amplitudes:")
    n = 1
    for i in range(system.noccupied_alpha):
        for j in range(system.noccupied_beta):
            if abs(R.ab[i, j]) <= thresh_print: continue
            print(
                "      [{}]     {}A  {}B  ->   =   {:.6f}".format(
                    n,
                    i + system.nfrozen + 1,
                    j + system.nfrozen + 1,
                    R.ab[i, j],
                )
            )
            n += 1
    if R.order > 2:
        for i in range(system.noccupied_alpha):
            for j in range(system.noccupied_beta):
                for c in range(system.nunoccupied_alpha):
                    for k in range(i + 1, system.noccupied_alpha):
                        if abs(R.aba[i, j, c, k]) <= thresh_print: continue
                        print(
                            "      [{}]     {}A  {}B  {}A  ->  {}A  =   {:.6f}".format(
                                n,
                                i + system.nfrozen + 1,
                                j + system.nfrozen + 1,
                                k + system.nfrozen + 1,
                                c + system.noccupied_alpha + system.nfrozen + 1,
                                R.aba[i, j, c, k],
                            )
                        )
                        n += 1
        for i in range(system.noccupied_alpha):
            for j in range(system.noccupied_beta):
                for c in range(system.nunoccupied_beta):
                    for k in range(j + 1, system.noccupied_beta):
                        if abs(R.abb[i, j, c, k]) <= thresh_print: continue
                        print(
                            "      [{}]     {}A  {}B  {}B  ->  {}B  =   {:.6f}".format(
                                n,
                                i + system.nfrozen + 1,
                                j + system.nfrozen + 1,
                                k + system.nfrozen + 1,
                                c + system.noccupied_beta + system.nfrozen + 1,
                                R.abb[i, j, c, k],
                            )
                        )
                        n += 1
        # Restore permutationally redundant amplitudes
        for i in range(system.noccupied_alpha):
            for j in range(system.noccupied_beta):
                for c in range(system.nunoccupied_alpha):
                    for k in range(i + 1, system.noccupied_alpha):
                        R.aba[k, j, c, i] = -R.aba[i, j, c, k]
        for i in range(system.noccupied_alpha):
            for j in range(system.noccupied_beta):
                for c in range(system.nunoccupied_beta):
                    for k in range(j + 1, system.noccupied_beta):
                        R.abb[i, k, c, j] = -R.abb[i, j, c, k]
    return

def print_sf_amplitudes(R, system, order, thresh_print):

    # Zero out the non-unique R amplitudes related by permutational symmetry
    for a in range(system.nunoccupied_alpha):
        for b in range(system.nunoccupied_beta):
            for i in range(system.noccupied_alpha):
                for j in range(i + 1, system.noccupied_alpha):
                    R.ab[a, b, j, i] = 0.0
    for a in range(system.nunoccupied_beta):
        for b in range(a + 1, system.nunoccupied_beta):
            for i in range(system.noccupied_beta):
                for j in range(system.noccupied_alpha):
                    R.bb[b, a, i, j] = 0.0

    print("\n   Largest Singly and Doubly Excited Amplitudes:")
    n = 1
    for a in range(system.nunoccupied_beta):
        for i in range(system.noccupied_alpha):
            if abs(R.b[a, i]) <= thresh_print: continue
            print(
                "      [{}]     {}A  ->  {}B   =   {:.6f}".format(
                    n,
                    i + system.nfrozen + 1,
                    a + system.nfrozen + system.noccupied_beta + 1,
                    R.b[a, i],
                )
            )
            n += 1
    for a in range(system.nunoccupied_alpha):
        for b in range(system.nunoccupied_beta):
            for i in range(system.noccupied_alpha):
                for j in range(i + 1, system.noccupied_alpha):
                    if abs(R.ab[a, b, i, j]) <= thresh_print: continue
                    print(
                        "      [{}]     {}A  {}A  ->  {}A  {}B  =   {:.6f}".format(
                            n,
                            i + system.nfrozen + 1,
                            j + system.nfrozen + 1,
                            a + system.noccupied_alpha + system.nfrozen + 1,
                            b + system.noccupied_beta + system.nfrozen + 1,
                            R.ab[a, b, i, j],
                        )
                    )
                    n += 1
    for a in range(system.nunoccupied_beta):
        for b in range(a + 1, system.nunoccupied_beta):
            for i in range(system.noccupied_beta):
                for j in range(system.noccupied_alpha):
                    if abs(R.bb[a, b, i, j]) <= thresh_print: continue
                    print(
                        "      [{}]     {}B  {}A  ->  {}B  {}B  =   {:.6f}".format(
                            n,
                            i + system.nfrozen + 1,
                            j + system.nfrozen + 1,
                            a + system.noccupied_beta + system.nfrozen + 1,
                            b + system.noccupied_beta + system.nfrozen + 1,
                            R.bb[a, b, i, j],
                        )
                    )
                    n += 1
    # Restore permutationally redundant amplitudes
    for a in range(system.nunoccupied_alpha):
        for b in range(system.nunoccupied_beta):
            for i in range(system.noccupied_alpha):
                for j in range(i + 1, system.noccupied_alpha):
                    R.ab[a, b, j, i] = -R.ab[a, b, i, j]
    for a in range(system.nunoccupied_beta):
        for b in range(a + 1, system.nunoccupied_beta):
            for i in range(system.noccupied_beta):
                for j in range(system.noccupied_alpha):
                    R.bb[b, a, i, j] = -R.bb[a, b, i, j]
    return

class SystemPrinter:
    def __init__(self, system):
        self.system = system

    def header(self):

        print(WHITESPACE, "System Information:")
        print(WHITESPACE, "----------------------------------------------------")
        print(WHITESPACE, "  Number of correlated electrons =", self.system.nelectrons)
        print(WHITESPACE, "  Number of correlated orbitals =", self.system.norbitals)
        print(WHITESPACE, "  Number of frozen orbitals =", self.system.nfrozen)
        print(
            WHITESPACE,
            "  Number of alpha occupied orbitals =",
            self.system.noccupied_alpha,
        )
        print(
            WHITESPACE,
            "  Number of alpha unoccupied orbitals =",
            self.system.nunoccupied_alpha,
        )
        print(
            WHITESPACE,
            "  Number of beta occupied orbitals =",
            self.system.noccupied_beta,
        )
        print(
            WHITESPACE,
            "  Number of beta unoccupied orbitals =",
            self.system.nunoccupied_beta,
        )
        print(WHITESPACE, "  Charge =", self.system.charge)
        print(WHITESPACE, "  Point group =", self.system.point_group)
        print(WHITESPACE, "  Symmetry of reference =", self.system.reference_symmetry)
        print(
            WHITESPACE, "  Spin multiplicity of reference =", self.system.multiplicity
        )
        print("")

        HEADER_FMT = "{:>10} {:>20} {:>13} {:>13}"
        MO_FMT = "{:>10} {:>20.6f} {:>13} {:>13.1f}"

        header = HEADER_FMT.format("MO #", "Energy (a.u.)", "Symmetry", "Occupation")
        print(header)
        print(len(header) * "-")
        for i in range(self.system.norbitals + self.system.nfrozen):
            print(
                MO_FMT.format(
                    i + 1,
                    self.system.mo_energies[i],
                    self.system.orbital_symmetries_all[i],
                    self.system.mo_occupation[i],
                )
            )
        print("")
        print(WHITESPACE, "Memory Usage =", get_memory_usage(), "MB")
        print(WHITESPACE, "Nuclear Repulsion Energy =", self.system.nuclear_repulsion)
        print(WHITESPACE, "Reference Energy =", self.system.reference_energy)
        print("")
        return
