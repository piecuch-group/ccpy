"""This module contains functions to calculate EA/IP energies
using the many-body Green's function (MBGF) formalism. Contains
MBGF(2), MBGF(3), and OVGF solvers."""
import mbgf_loops
import numpy as np

# print(mbgf_loops.mbgf2_selfenergy.__doc__)


def calc_mp2_selfenergy(omega, ints, sys):
    """Calculate the self-energy matrix \Sigma_{pq} for all p,q at
    2nd-order MBPT.

    Parameters
    ----------
    omega : float
        Energy parameter of self-energy
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary

    Returns
    -------
    sigma_* : ndarray(dtype=np.float64, shape=(norb,norb))
        MBPT(2) estimates of self-energy matrix for the alpha and beta spincases
    """
    fA = ints["fA"]
    fB = ints["fB"]
    vA = ints["vA"]
    vB = ints["vB"]
    vC = ints["vC"]
    # allocate separate onebody matrices for sigma_a and sigma_b (they will be the same!)
    sigma_a = np.zeros(
        (sys["Nocc_a"] + sys["Nunocc_a"], sys["Nocc_a"] + sys["Nunocc_a"])
    )
    sigma_b = np.zeros(
        (sys["Nocc_b"] + sys["Nunocc_b"], sys["Nocc_b"] + sys["Nunocc_b"])
    )
    # ooa block
    for i in range(sys["Nocc_a"]):
        for j in range(sys["Nocc_a"]):
            # alpha loop
            for n in range(sys["Nocc_a"]):
                for f in range(sys["Nunocc_a"]):
                    e_nf = fA["oo"][n, n] - fA["vv"][f, f]
                    # hole diagram
                    for m in range(n + 1, sys["Nocc_a"]):
                        denom = fA["oo"][m, m] + e_nf - omega
                        sigma_a[i, j] -= (
                            vA["ooov"][m, n, i, f] * vA["ovoo"][j, f, m, n] / denom
                        )
                    # particle diagram
                    for e in range(f + 1, sys["Nunocc_a"]):
                        denom = e_nf - fA["vv"][e, e] + omega
                        sigma_a[i, j] += (
                            vA["oovv"][j, n, e, f] * vA["vvoo"][e, f, i, n] / denom
                        )
            # beta loop
            for n in range(sys["Nocc_b"]):
                for f in range(sys["Nunocc_b"]):
                    e_nf = fB["oo"][n, n] - fB["vv"][f, f]
                    # hole diagram
                    for m in range(sys["Nocc_a"]):
                        denom = fA["oo"][m, m] + e_nf - omega
                        sigma_a[i, j] -= (
                            vB["ooov"][m, n, i, f] * vB["ovoo"][j, f, m, n] / denom
                        )
                    # particle diagram
                    for e in range(sys["Nunocc_a"]):
                        denom = e_nf - fA["vv"][e, e] + omega
                        sigma_a[i, j] += (
                            vB["oovv"][j, n, e, f] * vB["vvoo"][e, f, i, n] / denom
                        )
    # oob block
    for i in range(sys["Nocc_b"]):
        for j in range(sys["Nocc_b"]):
            # alpha loop
            for n in range(sys["Nocc_a"]):
                for f in range(sys["Nunocc_a"]):
                    e_nf = fA["oo"][n, n] - fA["vv"][f, f]
                    # hole diagram
                    for m in range(sys["Nocc_a"]):
                        denom = fB["oo"][m, m] + e_nf - omega
                        sigma_b[i, j] -= (
                            vB["oovo"][n, m, f, i] * vB["vooo"][f, j, n, m]
                        ) / denom
                    # particle diagram
                    for e in range(sys["Nunocc_a"]):
                        denom = e_nf - fB["vv"][e, e] + omega
                        sigma_b[i, j] += (
                            vB["oovv"][n, j, f, e] * vB["vvoo"][f, e, n, i]
                        ) / denom
            # beta loop
            for n in range(sys["Nocc_b"]):
                for f in range(sys["Nunocc_b"]):
                    e_nf = fB["oo"][n, n] - fB["vv"][f, f]
                    # hole diagram
                    for m in range(n + 1, sys["Nocc_b"]):
                        denom = fB["oo"][m, m] + e_nf - omega
                        sigma_b[i, j] -= (
                            vC["ooov"][m, n, i, f] * vC["ovoo"][j, f, m, n]
                        ) / denom
                    # particle diagram
                    for e in range(f + 1, sys["Nunocc_b"]):
                        denom = e_nf - fB["vv"][e, e] + omega
                        sigma_b[i, j] += (
                            vC["oovv"][j, n, e, f] * vC["vvoo"][e, f, j, n]
                        ) / denom
    return sigma_a, sigma_b


def gf2_ip(nroot, ints, sys, maxit=50, tol=1.0e-08):
    """Perform MBGF(2) iterations to relax the Koopmans IP energy.

    Parameters
    ----------
    nroot : int
        Number of IP roots. IP solver obtains nroot IP energies starting
        with HOMO. Can only request up to max(Nocc_a,Nocc_b) roots.
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary
    excitation : str
        Specify whether EA or IP roots should be solved for
    maxit : int, optional
        Maximum number of iterations for each root. Default is 100.
    tol : float, optional
        Convergence threshold on self-consistency of Dyson equation. Default is 1.0e-08.

    Returns
    -------
    omega : ndarray(dtype=np.float64, shape=(nroot))
        MBGF(2) IP energies (hartree)
    """
    # get integrals and root/convergence containers
    fA = ints["fA"]
    fB = ints["fB"]
    vA = ints["vA"]
    vB = ints["vB"]
    vC = ints["vC"]
    omega = np.zeros(nroot)
    is_converged = [False] * nroot
    # loop over all requested roots starting from HOMO
    for p in range(sys["Nocc_a"] - 1, sys["Nocc_a"] - nroot - 1, -1):
        iroot = sys["Nocc_a"] - p
        print("\nStarting MBGF(2) iterations for root {}".format(iroot))
        print(
            "Koopmans estimate of IP energy = {:>8f} hartree".format(
                -1.0 * fA["oo"][p, p]
            )
        )
        print("Iter         IP Energy        Residuum")
        print("========================================")
        e0 = fA["oo"][p, p]
        # MBGF iterations for each root
        for it in range(maxit):
            # sigma_a, _ = calc_mp2_selfenergy(e0,ints,sys)
            sigma_a, _ = mbgf_loops.mbgf2_selfenergy(
                e0,
                fA["oo"],
                fA["vv"],
                fB["oo"],
                fB["vv"],
                vA["oovv"],
                vA["vvoo"],
                vA["ooov"],
                vA["ovoo"],
                vB["oovv"],
                vB["vvoo"],
                vB["ooov"],
                vB["ovoo"],
                vB["oovo"],
                vB["vooo"],
                vC["oovv"],
                vC["vvoo"],
                vC["ooov"],
                vC["ovoo"],
            )
            e1 = fA["oo"][p, p] + sigma_a[p, p]
            resid = e1 - e0
            print("  {}          {:.8f}      {:.8f}".format(it + 1, -1.0 * e1, resid))
            if abs(resid) < tol:
                omega[iroot - 1] = -1.0 * e1
                is_converged[iroot - 1] = True
                print("IP root #{} converged!\n".format(iroot))
                break
            e0 = e1
        else:
            print("Failed to converge IP root #{}\n".format(iroot))

    # print root summary
    print("           IP-MBGF(2) CALCULATION SUMMARY")
    print("------------------------------------------------------")
    for i, ip_energy in enumerate(omega):
        if is_converged[i]:
            conv_str = "CONVERGED"
        else:
            conv_str = "NOT CONVERGED"
        print(
            "     Root - {}      IP ENERGY = {:>8f}      [{}]".format(
                i + 1, ip_energy, conv_str
            )
        )

    return omega


# def calc_mp2_selfenergy(omega,p,q,ints,sys):
#    """Calculates the matrix element \Sigma_{pq} of the self-energy
#    matrix to 2nd-order in MBPT.
#
#    Parameters
#    ----------
#    omega : float
#        Energy parameter of self-energy
#    p, q : int
#        Index of single-particle functions (MOs) outside the frozen core.
#    ints : dict
#        Collection of MO integrals defining the bare Hamiltonian H_N
#    sys : dict
#        System information dictionary
#
#    Returns
#    -------
#    sigma_mp2 : float
#        MP2 estimate of self-energy
#    """
#    # obtain the FULL integral matrices (these include core, occ, and unocc)
#    fA = ints['Fmat']['A']; fB = ints['Fmat']['B'];
#    vA = ints['Vmat']['A']; vB = ints['Vmat']['B']; vC = ints['Vmat']['C'];
#    sigma_mp2 = 0.0
#    N0 = sys['Nfroz'] # frozen spatial orbs
#    N1 = N0 + sys['Nocc_a'] # extent of occupied alpha orbs
#    N2 = N0 + sys['Nocc_b'] # extent of occupied beta orbs
#    N3 = N1 + sys['Nunocc_a'] # extent of unoccupied alpha orbs
#    N4 = N2 + sys['Nunocc_b'] # extent of unoccupied beta orbs
#    for n in range(N0,N1):
#        for f in range(N1,N3):
#            e_nf = fA[n,n] - fA[f,f]
#            # hole diagram
#            for m in range(N0,N1):
#                denom = fA[m,m] + e_nf - omega
#                sigma_mp2 -= 0.5*(vA[m,n,p+N0,f]*vA[q+N0,f,m,n])/denom
#            # particle diagram
#            for e in range(N1,N3):
#                denom = e_nf - fA[e,e] + omega
#                sigma_mp2 += 0.5*(vA[q+N0,n,e,f]*vA[e,f,p+N0,n])/denom
#    for n in range(N0,N2):
#        for f in range(N2,N4):
#            e_nf = fB[n,n] - fB[f,f]
#            # hole diagram
