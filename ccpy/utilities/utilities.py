# gives a single float value
# psutil.cpu_percent()
# gives an object with many fields
# psutil.virtual_memory()
# you can convert that object to a dictionary
# dict(psutil.virtual_memory()._asdict())
# you can have the percentage of used RAM
# psutil.virtual_memory().percent
# 79.2
# you can calculate percentage of available memory
# psutil.virtual_memory().available * 100 / psutil.virtual_memory().total
# 20.8

import numpy as np
from ccpy.utilities.updates import reorder

def reorder_triples_amplitudes(L, l3_excitations, t3_excitations):
    """Reorder the P-space triples amplitudes in L corresponding to
    the excitation array l3_excitations to the order provided by
    t3_excitations."""
    L.aaa, _ = reorder.reorder.reorder_amplitudes(L.aaa, l3_excitations["aaa"].T, t3_excitations["aaa"].T)
    L.aab, _ = reorder.reorder.reorder_amplitudes(L.aab, l3_excitations["aab"].T, t3_excitations["aab"].T)
    L.abb, _ = reorder.reorder.reorder_amplitudes(L.abb, l3_excitations["abb"].T, t3_excitations["abb"].T)
    L.bbb, _ = reorder.reorder.reorder_amplitudes(L.bbb, l3_excitations["bbb"].T, t3_excitations["bbb"].T)
    return L

def zero_small_values(x, threshold):
    low_values_flags = np.abs(x) < threshold  # Where values are low
    x[low_values_flags] = 0.0  # All low values set to 0
    return x

def gramschmidt(A):
    """Orthogonalize a set of vectors stored as the columns of matrix A."""
    # Get the number of vectors.
    n = A.shape[1]
    for j in range(n):
        # To orthogonalize the vector in column j with respect to the
        # previous vectors, subtract from it its projection onto
        # each of the previous vectors.
        for k in range(j):
            A[:, j] -= np.dot(A[:, k], A[:, j]) * A[:, k]
        A[:, j] = A[:, j] / np.linalg.norm(A[:, j])
    return A

def convert_excitations_c_to_f(excitations):
    if excitations is None:
        return excitations
    for key, value in excitations.items():
        if value.flags["F_CONTIGUOUS"]:
            continue
        else:
            excitations[key] = np.asfortranarray(value)
    return excitations

def print_memory_usage():
    """Displays the percentage of used RAM and available memory. Useful for
    investigating the memory usages of various routines."""
    import os

    import psutil

    current_process = psutil.Process(os.getpid())
    memory = current_process.memory_info().rss
    print(int(memory / (1024 * 1024)), "MB")
    return

def clean_up(fid, n):
    for i in range(n):
        remove_file(fid + "-" + str(i + 1) + ".npy")
    return

def remove_file(filePath):
    import os

    try:
        os.remove(filePath)
    except OSError:
        pass
    return

def read_amplitudes_from_jun(amlitude_file, system, order, amp_type='T', iroot=0):
    from scipy.io import FortranFile
    from ccpy.models.operators import ClusterOperator

    if amp_type == "T":
        with FortranFile(amlitude_file, "r") as f:
            first_line_reals = f.read_reals(dtype=np.float64)
            amps = f.read_reals(dtype=np.float64)
    else:
        amps = np.fromfile(amlitude_file, sep="", dtype=np.float64)

    if amp_type == "R": iroot -= 1

    X = ClusterOperator(system, order)
    reclen = X.ndim

    n = 0

    for i in range(system.noccupied_alpha):
        for a in range(system.nunoccupied_alpha):
            X.a[a, i] = amps[n + iroot * reclen]
            n += 1
    for i in range(system.noccupied_beta):
        for a in range(system.nunoccupied_beta):
            X.b[a, i] = amps[n + iroot * reclen]
            n += 1

    if order == 1: return X

    for i in range(system.noccupied_alpha):
        for j in range(system.noccupied_alpha):
            for a in range(system.nunoccupied_alpha):
                for b in range(system.nunoccupied_alpha):
                    if amp_type == "T":
                        X.aa[a, b, i, j] = -1.0 * amps[n + iroot * reclen]
                    else:
                        X.aa[a, b, i, j] = amps[n + iroot * reclen]
                    n += 1
    for i in range(system.noccupied_alpha):
        for j in range(system.noccupied_beta):
            for a in range(system.nunoccupied_alpha):
                for b in range(system.nunoccupied_beta):
                    if amp_type == "T":
                        X.bb[a, b, i, j] = -1.0 * amps[n + iroot * reclen]
                    else:
                        X.ab[a, b, i, j] = amps[n + iroot * reclen]
                    n += 1
    for i in range(system.noccupied_beta):
        for j in range(system.noccupied_beta):
            for a in range(system.nunoccupied_beta):
                for b in range(system.nunoccupied_beta):
                    if amp_type == "T":
                        X.ab[a, b, i, j] = -1.0 * amps[n + iroot * reclen]
                    else:
                        X.bb[a, b, i, j] = amps[n + iroot * reclen]
                    n += 1

    if order == 2: return X

    for i in range(system.noccupied_alpha):
        for j in range(system.noccupied_alpha):
            for k in range(system.noccupied_alpha):
                for a in range(system.nunoccupied_alpha):
                    for b in range(system.nunoccupied_alpha):
                        for c in range(system.nunoccupied_alpha):
                            if amp_type == "T":
                                X.aaa[a, b, c, i, j, k] = -1.0 * amps[n + iroot * reclen]
                            else:
                                X.aaa[a, b, c, i, j, k] = amps[n + iroot * reclen]
                            n += 1
    for i in range(system.noccupied_alpha):
        for j in range(system.noccupied_alpha):
            for k in range(system.noccupied_beta):
                for a in range(system.nunoccupied_alpha):
                    for b in range(system.nunoccupied_alpha):
                        for c in range(system.nunoccupied_beta):
                            if amp_type == "T":
                                X.bbb[a, b, c, i, j, k] = -1.0 * amps[n + iroot * reclen]
                            else:
                                X.aab[a, b, c, i, j, k] = amps[n + iroot * reclen]
                            n += 1
    for i in range(system.noccupied_alpha):
        for j in range(system.noccupied_beta):
            for k in range(system.noccupied_beta):
                for a in range(system.nunoccupied_alpha):
                    for b in range(system.nunoccupied_beta):
                        for c in range(system.nunoccupied_beta):
                            if amp_type == "T":
                                X.aab[a, c, b, i, k, j] = -1.0 * amps[n + iroot * reclen]
                            else:
                                X.abb[a, b, c, i, j, k] = amps[n + iroot * reclen]
                            n += 1
    for i in range(system.noccupied_beta):
        for j in range(system.noccupied_beta):
            for k in range(system.noccupied_beta):
                for a in range(system.nunoccupied_beta):
                    for b in range(system.nunoccupied_beta):
                        for c in range(system.nunoccupied_beta):
                            if amp_type == "T":
                                X.abb[a, b, c, i, j, k] = -1.0 * amps[n + iroot * reclen]
                            else:
                                X.bbb[a, b, c, i, j, k] = amps[n + iroot * reclen]
                            n += 1

    if order == 3: return X
