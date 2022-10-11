
hartree_to_kcal = 627.5

def compute_st_gap(st_dict, label=""):
    print(label, "S-T gap = ",
          hartree_to_kcal * (st_dict['singlet'] - st_dict['triplet']),
          "kcal/mol")
    return

def main():

    torsion = [45]

    ref = {'singlet': -231.6536718917063, 'triplet': -231.75092611}
    ccsd = {'singlet' : -232.5719300762, 'triplet' : -232.6100107904}
    crcc23 = {'singlet' : -232.6329980043, 'triplet' : -232.6421985449}
    ccsdt1 = {'singlet' : -232.6286042436, 'triplet' : -232.6266934539}
    cct3 = {'singlet' : -232.6485743549, 'triplet' : -232.6440195785}

    compute_st_gap(ccsd, "CCSD")
    compute_st_gap(crcc23, "CR-CC(2,3)")
    compute_st_gap(ccsdt1, "CCSDt")
    compute_st_gap(cct3, "CC(t;3)")




if __name__ == "__main__":

    main()
