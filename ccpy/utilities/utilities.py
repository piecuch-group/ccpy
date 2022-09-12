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
        remove_files(fid + "-" + str(i + 1) + ".npy")
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
