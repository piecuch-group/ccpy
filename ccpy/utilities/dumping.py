def dumpIntegralstoPGFiles(e1int, e2int, system):

    norbitals = e1int.shape[0]
    with open("onebody.inp", "w") as f:
        ct = 1
        for i in range(norbitals):
            for j in range(i + 1):
                f.write("   {:>.11f}    {}\n".format(e1int[i, j], ct))
                ct += 1
    with open("twobody.inp", "w") as f:
        for i in range(norbitals):
            for k in range(norbitals):
                for j in range(norbitals):
                    for l in range(norbitals):
                        f.write(
                            "    {}    {}    {}    {}       {:.11f}\n".format(
                                i + 1, j + 1, k + 1, l + 1, e2int[i, j, k, l]
                            )
                        )
        f.write(
            "    {}    {}    {}    {}        {:.11f}\n".format(
                0, 0, 0, 0, system.nuclear_repulsion
            )
        )
    return


def dumpPBCIntegralstoPGFiles(e1int, e2int, system):

    nkpts = e1int.shape[0]
    norbitals = e1int.shape[2]
    with open("onebody.inp", "w") as f:
        for kp in range(nkpts):
            for kq in range(nkpts):
                ct = 1
                for i in range(norbitals):
                    for j in range(i + 1):
                        f.write("     {}    {:.11f}\n".format(e1int[kp, kq, i, j], ct))
                        ct += 1

    # inefficient. we should be saving only those V(kp,kq,kr,ks) that
    # obey symmetry constraint
    with open("twobody.inp", "w") as f:
        for kp in range(nkpts):
            for kr in range(nkpts):
                for kq in range(nkpts):
                    for ks in range(nkpts):
                        for i in range(norbitals):
                            for k in range(norbitals):
                                for j in range(norbitals):
                                    for l in range(norbitals):
                                        f.write(
                                            "    {}    {}    {}    {}       {:.11f}\n".format(
                                                i + 1,
                                                j + 1,
                                                k + 1,
                                                l + 1,
                                                e2int[kp, kq, kr, ks, i, j, k, l],
                                            )
                                        )
        f.write(
            "    {}    {}    {}    {}        {:.11f}\n".format(
                0, 0, 0, 0, system.e_nuclear
            )
        )
    return


def dumpSystemToPGFiles(sys, integrals, projectName, nfrozen):
    """Dumps the molecule information into the cc.inp, *.inf, and *.gjf files used
    by the Piecuch Group codes."""
    # Dictionary of default values used for typical CC and EOMCC calculations
    defaultValues = {
        "CC Convergence": 8,
        "Active CC Occupied": 0,
        "Active CC Unoccupied": 0,
        "Active EOM Guess Occupied": 0,
        "Active EOM Guess Unoccupied": 0,
        "EOM Multiplicity": 1,
        "EOM Maximum Iterations": 200,
        "EOM Convergence": 8,
        "EOM Number of Roots": 10,
        "Memory": 2000,
    }
    # Write the first line of the cc.inp file (this is all that Jun's code reads)
    ccinp = [
        sys.nelectrons,
        2 * sys.norbitals - sys.nelectrons,
        2 * nfrozen,
        defaultValues["Memory"],
        defaultValues["CC Convergence"],
        0,
        sys.multiplicity,
    ]
    with open("cc.inp", "w") as ccinpfile:
        ccinpfile.write("  ".join(map(str, ccinp)))
    # Write the values in the *.gjf file
    entranceString = {
        "Active CC Occupied": "m1",
        "Active CC Unoccupied": "m2",
        "Active EOM Guess Occupied": "m3",
        "Active EOM Guess Unoccupied": "m4",
        "EOM Multiplicity": "mult",
        "EOM Maximum Iterations": "itEOM",
        "EOM Convergence": "conver",
        "EOM Number of Roots": "nroot",
    }
    with open(projectName + ".gjf", "w") as gjffile:
        gjffile.write("%mem=800mw\n")
        gjffile.write("UCC[uhf")
        for key, value in entranceString.items():
            gjffile.write("  " + str(value) + "=" + str(defaultValues[key]))
        gjffile.write("]")
