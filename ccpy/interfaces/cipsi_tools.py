import numpy as np
from ccpy.utilities.permutations import calculate_permutation_parity

def hex2bin(hex_string):
    bin_string = bin(int(hex_string, 16))[2:]
    return bin_string

def process_cipsi_vectors(inFile, system):

    nlines_det_print = 5           # Each determinant in the QP2 output is printed in 5 lines, starting with the flag "Determinant"
    offset = nlines_det_print - 1  # Useful offset to jump from the Determinant line to the CI coefficient

    outFile = open(inFile + "_proc", "w")
    with open(inFile) as f:
        ndet = 0
        lines = f.readlines()
        for j in range(len(lines) - offset):
            line = lines[j]

            if "Determinant" in line.split():
                
                ndet += 1

                alpha = hex2bin(lines[j + 1].split("|")[0])[::-1]
                beta = hex2bin(lines[j + 1].split("|")[1])[::-1]
                coef = float(lines[j + offset].split()[0])

                occ_alpha = [2 * i + 1 for i in range(len(alpha)) if alpha[i] == "1"]
                occ_beta = [2 * i + 2 for i in range(len(beta)) if beta[i] == "1"]
                det_occ = occ_alpha + occ_beta          # determinant in aaa...|bbb... ordering
                permutation = list(np.argsort(det_occ)) # permutation to print into ababa... ordering
                det_occ = [det_occ[i] for i in permutation]
                sign = calculate_permutation_parity(permutation) # compute sign of permutation from aabb -> abab
                coef *= sign

                det_occ_correlated = [det_occ[i + 2 * system.nfrozen] - 2 * system.nfrozen for i in range(system.nelectrons)]


                outLine = [ndet, coef] + det_occ_correlated
                outFile.write("    ")
                outFile.writelines(str(x).ljust(10, " ") for x in outLine[:1])
                outFile.writelines(str(x).ljust(20, " ") for x in outLine[1:2])
                outFile.writelines("    ")
                outFile.writelines(str(x).ljust(4, " " ) for x in outLine[2:])
                outFile.write("\n")
    outFile.close()






