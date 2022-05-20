
from ccpy.density.rdm1 import calc_rdm1
from ccpy.density.ccsd_no import convert_to_ccsd_no



def main(storage_directory):

    
    rdm1 = calc_rdm1(T, L, system)
    H, system = convert_to_ccsd_no(rdm1, H, system)
