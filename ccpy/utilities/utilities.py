import os
import numpy as np
from ccpy.lib.core import reorder

def convert_t3_from_pspace(driver, t3_excitations):

    print("   Converting T3(P) operator into full T3 array")

    # Check for empty spincases in t3 list. Remember that [1., 1., 1., 1., 1., 1.]
    # is defined as the "empty" state in the Fortran modules.
    do_t3 = {"aaa" : True, "aab" : True, "abb" : True, "bbb" : True}
    if np.array_equal(t3_excitations["aaa"][0,:], np.array([1., 1., 1., 1., 1., 1.])):
        do_t3["aaa"] = False
    if np.array_equal(t3_excitations["aab"][0,:], np.array([1., 1., 1., 1., 1., 1.])):
        do_t3["aab"] = False
    if np.array_equal(t3_excitations["abb"][0,:], np.array([1., 1., 1., 1., 1., 1.])):
        do_t3["abb"] = False
    if np.array_equal(t3_excitations["bbb"][0,:], np.array([1., 1., 1., 1., 1., 1.])):
        do_t3["bbb"] = False

    T_new = unravel_triples_amplitudes(driver.T, t3_excitations, driver.system, do_t3)
    setattr(driver, "T", T_new)
    return

def unravel_triples_amplitudes(T, t3_excitations, system, do_t3):
    """Replaces the triples parts of the T operator defined as P-space vectors
    with the corresponding 6-dimensional arrays."""
    from ccpy.models.operators import ClusterOperator
    n3aaa = t3_excitations["aaa"].shape[0]
    n3aab = t3_excitations["aab"].shape[0]
    n3abb = t3_excitations["abb"].shape[0]
    n3bbb = t3_excitations["bbb"].shape[0]
    # Make a new cluster operator
    T_new = ClusterOperator(system, order=3)
    setattr(T_new, "a", T.a)
    setattr(T_new, "b", T.b)
    setattr(T_new, "aa", T.aa)
    setattr(T_new, "ab", T.ab)
    setattr(T_new, "bb", T.bb)
    # Unravel aaa
    if do_t3["aaa"]:
        for idet in range(n3aaa):
            a, b, c, i, j, k = [x - 1 for x in t3_excitations["aaa"][idet, :]]
            T_new.aaa[a, b, c, i, j, k] = T.aaa[idet]
            T_new.aaa[a, b, c, i, k, j] = -T.aaa[idet]
            T_new.aaa[a, b, c, j, i, k] = -T.aaa[idet]
            T_new.aaa[a, b, c, j, k, i] = T.aaa[idet]
            T_new.aaa[a, b, c, k, i, j] = T.aaa[idet]
            T_new.aaa[a, b, c, k, j, i] = -T.aaa[idet]
            T_new.aaa[a, c, b, i, j, k] = -T.aaa[idet]
            T_new.aaa[a, c, b, i, k, j] = T.aaa[idet]
            T_new.aaa[a, c, b, j, i, k] = T.aaa[idet]
            T_new.aaa[a, c, b, j, k, i] = -T.aaa[idet]
            T_new.aaa[a, c, b, k, i, j] = -T.aaa[idet]
            T_new.aaa[a, c, b, k, j, i] = T.aaa[idet]
            T_new.aaa[b, c, a, i, j, k] = T.aaa[idet]
            T_new.aaa[b, c, a, i, k, j] = -T.aaa[idet]
            T_new.aaa[b, c, a, j, i, k] = -T.aaa[idet]
            T_new.aaa[b, c, a, j, k, i] = T.aaa[idet]
            T_new.aaa[b, c, a, k, i, j] = T.aaa[idet]
            T_new.aaa[b, c, a, k, j, i] = -T.aaa[idet]
            T_new.aaa[b, a, c, i, j, k] = -T.aaa[idet]
            T_new.aaa[b, a, c, i, k, j] = T.aaa[idet]
            T_new.aaa[b, a, c, j, i, k] = T.aaa[idet]
            T_new.aaa[b, a, c, j, k, i] = -T.aaa[idet]
            T_new.aaa[b, a, c, k, i, j] = -T.aaa[idet]
            T_new.aaa[b, a, c, k, j, i] = T.aaa[idet]
            T_new.aaa[c, a, b, i, j, k] = T.aaa[idet]
            T_new.aaa[c, a, b, i, k, j] = -T.aaa[idet]
            T_new.aaa[c, a, b, j, i, k] = -T.aaa[idet]
            T_new.aaa[c, a, b, j, k, i] = T.aaa[idet]
            T_new.aaa[c, a, b, k, i, j] = T.aaa[idet]
            T_new.aaa[c, a, b, k, j, i] = -T.aaa[idet]
            T_new.aaa[c, b, a, i, j, k] = -T.aaa[idet]
            T_new.aaa[c, b, a, i, k, j] = T.aaa[idet]
            T_new.aaa[c, b, a, j, i, k] = T.aaa[idet]
            T_new.aaa[c, b, a, j, k, i] = -T.aaa[idet]
            T_new.aaa[c, b, a, k, i, j] = -T.aaa[idet]
            T_new.aaa[c, b, a, k, j, i] = T.aaa[idet]
    # Unravel aab
    if do_t3["aab"]:
        for idet in range(n3aab):
            a, b, c, i, j, k = [x - 1 for x in t3_excitations["aab"][idet, :]]
            T_new.aab[a, b, c, i, j, k] = T.aab[idet]
            T_new.aab[b, a, c, i, j, k] = -T.aab[idet]
            T_new.aab[a, b, c, j, i, k] = -T.aab[idet]
            T_new.aab[b, a, c, j, i, k] = T.aab[idet]
    # Unravel abb
    if do_t3["abb"]:
        for idet in range(n3abb):
            a, b, c, i, j, k = [x - 1 for x in t3_excitations["abb"][idet, :]]
            T_new.abb[a, b, c, i, j, k] = T.abb[idet]
            T_new.abb[a, c, b, i, j, k] = -T.abb[idet]
            T_new.abb[a, b, c, i, k, j] = -T.abb[idet]
            T_new.abb[a, c, b, i, k, j] = T.abb[idet]
    # Unravel bbb
    if do_t3["bbb"]:
        for idet in range(n3bbb):
            a, b, c, i, j, k = [x - 1 for x in t3_excitations["bbb"][idet, :]]
            T_new.bbb[a, b, c, i, j, k] = T.bbb[idet]
            T_new.bbb[a, b, c, i, k, j] = -T.bbb[idet]
            T_new.bbb[a, b, c, j, i, k] = -T.bbb[idet]
            T_new.bbb[a, b, c, j, k, i] = T.bbb[idet]
            T_new.bbb[a, b, c, k, i, j] = T.bbb[idet]
            T_new.bbb[a, b, c, k, j, i] = -T.bbb[idet]
            T_new.bbb[a, c, b, i, j, k] = -T.bbb[idet]
            T_new.bbb[a, c, b, i, k, j] = T.bbb[idet]
            T_new.bbb[a, c, b, j, i, k] = T.bbb[idet]
            T_new.bbb[a, c, b, j, k, i] = -T.bbb[idet]
            T_new.bbb[a, c, b, k, i, j] = -T.bbb[idet]
            T_new.bbb[a, c, b, k, j, i] = T.bbb[idet]
            T_new.bbb[b, c, a, i, j, k] = T.bbb[idet]
            T_new.bbb[b, c, a, i, k, j] = -T.bbb[idet]
            T_new.bbb[b, c, a, j, i, k] = -T.bbb[idet]
            T_new.bbb[b, c, a, j, k, i] = T.bbb[idet]
            T_new.bbb[b, c, a, k, i, j] = T.bbb[idet]
            T_new.bbb[b, c, a, k, j, i] = -T.bbb[idet]
            T_new.bbb[b, a, c, i, j, k] = -T.bbb[idet]
            T_new.bbb[b, a, c, i, k, j] = T.bbb[idet]
            T_new.bbb[b, a, c, j, i, k] = T.bbb[idet]
            T_new.bbb[b, a, c, j, k, i] = -T.bbb[idet]
            T_new.bbb[b, a, c, k, i, j] = -T.bbb[idet]
            T_new.bbb[b, a, c, k, j, i] = T.bbb[idet]
            T_new.bbb[c, a, b, i, j, k] = T.bbb[idet]
            T_new.bbb[c, a, b, i, k, j] = -T.bbb[idet]
            T_new.bbb[c, a, b, j, i, k] = -T.bbb[idet]
            T_new.bbb[c, a, b, j, k, i] = T.bbb[idet]
            T_new.bbb[c, a, b, k, i, j] = T.bbb[idet]
            T_new.bbb[c, a, b, k, j, i] = -T.bbb[idet]
            T_new.bbb[c, b, a, i, j, k] = -T.bbb[idet]
            T_new.bbb[c, b, a, i, k, j] = T.bbb[idet]
            T_new.bbb[c, b, a, j, i, k] = T.bbb[idet]
            T_new.bbb[c, b, a, j, k, i] = -T.bbb[idet]
            T_new.bbb[c, b, a, k, i, j] = -T.bbb[idet]
            T_new.bbb[c, b, a, k, j, i] = T.bbb[idet]
    return T_new

def unravel_3p2h_amplitudes(R, r3_excitations, system, do_r3):
    """Replaces the 3p2h parts of the R (or L) operator defined as P-space vectors
    with the corresponding 5-dimensional arrays."""
    from ccpy.models.operators import FockOperator
    n3aaa = r3_excitations["aaa"].shape[0]
    n3aab = r3_excitations["aab"].shape[0]
    n3abb = r3_excitations["abb"].shape[0]
    # Make a new cluster operator
    R_new = FockOperator(system, 3, 2)
    setattr(R_new, "a", R.a)
    setattr(R_new, "aa", R.aa)
    setattr(R_new, "ab", R.ab)
    # Unravel aaa
    if do_r3["aaa"]:
        for idet in range(n3aaa):
            a, b, c, j, k = [x - 1 for x in r3_excitations["aaa"][idet, :]]
            R_new.aaa[a, b, c, j, k] = R.aaa[idet]
            R_new.aaa[a, c, b, j, k] = -R.aaa[idet]
            R_new.aaa[b, c, a, j, k] = R.aaa[idet]
            R_new.aaa[b, a, c, j, k] = -R.aaa[idet]
            R_new.aaa[c, a, b, j, k] = R.aaa[idet]
            R_new.aaa[c, b, a, j, k] = -R.aaa[idet]
            R_new.aaa[a, b, c, k, j] = -R.aaa[idet]
            R_new.aaa[a, c, b, k, j] = R.aaa[idet]
            R_new.aaa[b, c, a, k, j] = -R.aaa[idet]
            R_new.aaa[b, a, c, k, j] = R.aaa[idet]
            R_new.aaa[c, a, b, k, j] = -R.aaa[idet]
            R_new.aaa[c, b, a, k, j] = R.aaa[idet]
    # Unravel aab
    if do_r3["aab"]:
        for idet in range(n3aab):
            a, b, c, j, k = [x - 1 for x in r3_excitations["aab"][idet, :]]
            R_new.aab[a, b, c, j, k] = R.aab[idet]
            R_new.aab[b, a, c, j, k] = -R.aab[idet]
    # Unravel abb
    if do_r3["abb"]:
        for idet in range(n3abb):
            a, b, c, j, k = [x - 1 for x in r3_excitations["abb"][idet, :]]
            R_new.abb[a, b, c, j, k] = R.abb[idet]
            R_new.abb[a, c, b, j, k] = -R.abb[idet]
            R_new.abb[a, b, c, k, j] = -R.abb[idet]
            R_new.abb[a, c, b, k, j] = R.abb[idet]
    return R_new

def unravel_3h2p_amplitudes(R, r3_excitations, system, do_r3):
    """Replaces the 3h2p parts of the R (or L) operator defined as P-space vectors
    with the corresponding 5-dimensional arrays."""
    from ccpy.models.operators import FockOperator
    n3aaa = r3_excitations["aaa"].shape[0]
    n3aab = r3_excitations["aab"].shape[0]
    n3abb = r3_excitations["abb"].shape[0]
    # Make a new cluster operator
    R_new = FockOperator(system, 2, 3)
    setattr(R_new, "a", R.a)
    setattr(R_new, "aa", R.aa)
    setattr(R_new, "ab", R.ab)
    # Unravel aaa
    if do_r3["aaa"]:
        for idet in range(n3aaa):
            b, c, i, j, k = [x - 1 for x in r3_excitations["aaa"][idet, :]]
            R_new.aaa[i, b, c, j, k] = R.aaa[idet]
            R_new.aaa[i, b, c, k, j] = -R.aaa[idet]
            R_new.aaa[j, b, c, k, i] = R.aaa[idet]
            R_new.aaa[j, b, c, i, k] = -R.aaa[idet]
            R_new.aaa[k, b, c, i, j] = R.aaa[idet]
            R_new.aaa[k, b, c, j, i] = -R.aaa[idet]
            R_new.aaa[i, c, b, j, k] = -R.aaa[idet]
            R_new.aaa[i, c, b, k, j] = R.aaa[idet]
            R_new.aaa[j, c, b, k, i] = -R.aaa[idet]
            R_new.aaa[j, c, b, i, k] = R.aaa[idet]
            R_new.aaa[k, c, b, i, j] = -R.aaa[idet]
            R_new.aaa[k, c, b, j, i] = R.aaa[idet]
    # Unravel aab
    if do_r3["aab"]:
        for idet in range(n3aab):
            b, c, i, j, k = [x - 1 for x in r3_excitations["aab"][idet, :]]
            R_new.aab[i, b, c, j, k] = R.aab[idet]
            R_new.aab[j, b, c, i, k] = -R.aab[idet]
    # Unravel abb
    if do_r3["abb"]:
        for idet in range(n3abb):
            b, c, i, j, k = [x - 1 for x in r3_excitations["abb"][idet, :]]
            R_new.abb[i, b, c, j, k] = R.abb[idet]
            R_new.abb[j, b, c, i, k] = -R.abb[idet]
            R_new.abb[i, c, b, j, k] = -R.abb[idet]
            R_new.abb[j, c, b, i, k] = R.abb[idet]
    return R_new

def reorder_triples_amplitudes(L, l3_excitations, t3_excitations):
    """Reorder the P-space triples amplitudes in L corresponding to
    the excitation array l3_excitations to the order provided by
    t3_excitations."""
    L.aaa, _ = reorder.reorder_amplitudes(L.aaa, l3_excitations["aaa"].T, t3_excitations["aaa"].T)
    L.aab, _ = reorder.reorder_amplitudes(L.aab, l3_excitations["aab"].T, t3_excitations["aab"].T)
    L.abb, _ = reorder.reorder_amplitudes(L.abb, l3_excitations["abb"].T, t3_excitations["abb"].T)
    L.bbb, _ = reorder.reorder_amplitudes(L.bbb, l3_excitations["bbb"].T, t3_excitations["bbb"].T)
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

def get_memory_usage():
    """Displays the percentage of used RAM and available memory. Useful for
    investigating the memory usages of various routines.
    Usage:
    >> # gives a single float value
    >> psutil.cpu_percent()
    >> # gives an object with many fields
    >> psutil.virtual_memory()
    >> # you can convert that object to a dictionary
    >> dict(psutil.virtual_memory()._asdict())
    >> # you can have the percentage of used RAM
    >> psutil.virtual_memory().percent
    >> 79.2
    >> # you can calculate percentage of available memory
    >> psutil.virtual_memory().available * 100 / psutil.virtual_memory().total
    >> 20.
    """
    import psutil
    current_process = psutil.Process(os.getpid())
    memory = current_process.memory_info().rss # RSS (e.g., RAM usage) memory in bytes
    return memory / (1024 * 1024)

def clean_up(fid, n):
    for i in range(n):
        remove_file(fid + "-" + str(i + 1) + ".npy")
    return

def remove_file(filePath):
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
