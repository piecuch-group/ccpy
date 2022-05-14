def get_pg_irreps(pg):
    pg = pg.upper()
    pg_irreps = {
        "C1": {"A": 0},
        "CS": {"A'": 0, "A''": 1},
        "C2V": {"A1": 0, "A2": 1, "B1": 2, "B2": 3},
        "D2H": {
            "AG": 0,
            "B1G": 1,
            "B2G": 2,
            "B3G": 3,
            "AU": 4,
            "B1U": 5,
            "B2U": 6,
            "B3U": 7,
        },
    }
    return pg_irreps[pg]


def sym_table(pg):
    # Use multiplication table of point group pg
    sym_mult = {
        "C1": [[0]],
        "CS": [[0, 1], [1, 0]],
        "C2V": [[0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]],
        "D2H": [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [1, 0, 3, 2, 5, 4, 7, 6],
            [2, 3, 0, 1, 6, 7, 4, 5],
            [3, 2, 1, 0, 7, 6, 5, 4],
            [4, 5, 6, 7, 0, 1, 2, 3],
            [5, 4, 7, 6, 1, 0, 3, 2],
            [6, 7, 4, 5, 2, 3, 0, 1],
            [7, 6, 5, 4, 3, 2, 1, 0],
        ],
    }

    return sym_mult[pg]


def count_singles(system):

    countsym = [0 for i in range(len(system.point_group_irrep_to_number))]

    n_refsym = system.point_group_irrep_to_number[system.reference_symmetry]

    for i in range(system.noccupied_alpha):
        symi = system.orbital_symmetries[i]
        ni = system.point_group_irrep_to_number[symi]
        for a in range(system.noccupied_alpha, system.norbitals):
            syma = system.orbital_symmetries[a]
            na = system.point_group_irrep_to_number[syma]
            sym = na ^ ni
            sym = sym ^ n_refsym
            countsym[sym] += 1  

    for i in range(system.noccupied_beta):
        symi = system.orbital_symmetries[i]
        ni = system.point_group_irrep_to_number[symi]
        for a in range(system.noccupied_beta, system.norbitals):
            syma = system.orbital_symmetries[a]
            na = system.point_group_irrep_to_number[syma]
            sym = na ^ ni
            sym = sym ^ n_refsym
            countsym[sym] += 1  

    total = sum(countsym)

    return countsym, total


def count_doubles(system):

    countsym = [0 for i in range(len(system.point_group_irrep_to_number))]

    n_refsym = system.point_group_irrep_to_number[system.reference_symmetry]

    for i in range(system.noccupied_alpha):
        symi = system.orbital_symmetries[i]
        ni = system.point_group_irrep_to_number[symi]
        for j in range(i + 1, system.noccupied_alpha):
            symj = system.orbital_symmetries[j]
            nj = system.point_group_irrep_to_number[symj]
            for a in range(system.noccupied_alpha, system.norbitals):
                syma = system.orbital_symmetries[a]
                na = system.point_group_irrep_to_number[syma]
                for b in range(a + 1, system.norbitals):
                    symb = system.orbital_symmetries[b]
                    nb = system.point_group_irrep_to_number[symb]

                    sym = na ^ nb ^ nj ^ ni
                    sym = sym ^ n_refsym
                    countsym[sym] += 1  

    for i in range(system.noccupied_alpha):
        symi = system.orbital_symmetries[i]
        ni = system.point_group_irrep_to_number[symi]
        for j in range(system.noccupied_beta):
            symj = system.orbital_symmetries[j]
            nj = system.point_group_irrep_to_number[symj]
            for a in range(system.noccupied_alpha, system.norbitals):
                syma = system.orbital_symmetries[a]
                na = system.point_group_irrep_to_number[syma]
                for b in range(system.noccupied_beta, system.norbitals):
                    symb = system.orbital_symmetries[b]
                    nb = system.point_group_irrep_to_number[symb]

                    sym = na ^ nb ^ nj ^ ni
                    sym = sym ^ n_refsym
                    countsym[sym] += 1  

    for i in range(system.noccupied_beta):
        symi = system.orbital_symmetries[i]
        ni = system.point_group_irrep_to_number[symi]
        for j in range(i + 1, system.noccupied_beta):
            symj = system.orbital_symmetries[j]
            nj = system.point_group_irrep_to_number[symj]
            for a in range(system.noccupied_beta, system.norbitals):
                syma = system.orbital_symmetries[a]
                na = system.point_group_irrep_to_number[syma]
                for b in range(a + 1, system.norbitals):
                    symb = system.orbital_symmetries[b]
                    nb = system.point_group_irrep_to_number[symb]

                    sym = na ^ nb ^ nj ^ ni
                    sym = sym ^ n_refsym
                    countsym[sym] += 1  

    total = sum(countsym)

    return countsym, total

def count_triples(system):

    countsym = [0 for i in range(len(system.point_group_irrep_to_number))]

    n_refsym = system.point_group_irrep_to_number[system.reference_symmetry]

    for i in range(system.noccupied_alpha):
        symi = system.orbital_symmetries[i]
        ni = system.point_group_irrep_to_number[symi]
        for j in range(i + 1, system.noccupied_alpha):
            symj = system.orbital_symmetries[j]
            nj = system.point_group_irrep_to_number[symj]
            for k in range(j + 1, system.noccupied_alpha):
                symk = system.orbital_symmetries[k]
                nk = system.point_group_irrep_to_number[symk]
                for a in range(system.noccupied_alpha, system.norbitals):
                    syma = system.orbital_symmetries[a]
                    na = system.point_group_irrep_to_number[syma]
                    for b in range(a + 1, system.norbitals):
                        symb = system.orbital_symmetries[b]
                        nb = system.point_group_irrep_to_number[symb]
                        for c in range(b + 1, system.norbitals):
                            symc = system.orbital_symmetries[c]
                            nc = system.point_group_irrep_to_number[symc]

                            sym = n_refsym
                            sym = na ^ nb ^ nc ^ nk ^ nj ^ ni
                            countsym[sym] += 1  

    for i in range(system.noccupied_alpha):
        symi = system.orbital_symmetries[i]
        ni = system.point_group_irrep_to_number[symi]
        for j in range(i + 1, system.noccupied_alpha):
            symj = system.orbital_symmetries[j]
            nj = system.point_group_irrep_to_number[symj]
            for k in range(system.noccupied_beta):
                symk = system.orbital_symmetries[k]
                nk = system.point_group_irrep_to_number[symk]
                for a in range(system.noccupied_alpha, system.norbitals):
                    syma = system.orbital_symmetries[a]
                    na = system.point_group_irrep_to_number[syma]
                    for b in range(a + 1, system.norbitals):
                        symb = system.orbital_symmetries[b]
                        nb = system.point_group_irrep_to_number[symb]
                        for c in range(system.noccupied_beta, system.norbitals):
                            symc = system.orbital_symmetries[c]
                            nc = system.point_group_irrep_to_number[symc]

                            sym = n_refsym
                            sym = na ^ nb ^ nc ^ nk ^ nj ^ ni
                            countsym[sym] += 1 

    for i in range(system.noccupied_alpha):
        symi = system.orbital_symmetries[i]
        ni = system.point_group_irrep_to_number[symi]
        for j in range(system.noccupied_beta):
            symj = system.orbital_symmetries[j]
            nj = system.point_group_irrep_to_number[symj]
            for k in range(j + 1, system.noccupied_beta):
                symk = system.orbital_symmetries[k]
                nk = system.point_group_irrep_to_number[symk]
                for a in range(system.noccupied_alpha, system.norbitals):
                    syma = system.orbital_symmetries[a]
                    na = system.point_group_irrep_to_number[syma]
                    for b in range(system.noccupied_beta, system.norbitals):
                        symb = system.orbital_symmetries[b]
                        nb = system.point_group_irrep_to_number[symb]
                        for c in range(b + 1, system.norbitals):
                            symc = system.orbital_symmetries[c]
                            nc = system.point_group_irrep_to_number[symc]

                            sym = n_refsym
                            sym = na ^ nb ^ nc ^ nk ^ nj ^ ni
                            countsym[sym] += 1 

    for i in range(system.noccupied_beta):
        symi = system.orbital_symmetries[i]
        ni = system.point_group_irrep_to_number[symi]
        for j in range(i + 1, system.noccupied_beta):
            symj = system.orbital_symmetries[j]
            nj = system.point_group_irrep_to_number[symj]
            for k in range(j + 1, system.noccupied_beta):
                symk = system.orbital_symmetries[k]
                nk = system.point_group_irrep_to_number[symk]
                for a in range(system.noccupied_beta, system.norbitals):
                    syma = system.orbital_symmetries[a]
                    na = system.point_group_irrep_to_number[syma]
                    for b in range(a + 1, system.norbitals):
                        symb = system.orbital_symmetries[b]
                        nb = system.point_group_irrep_to_number[symb]
                        for c in range(b + 1, system.norbitals):
                            symc = system.orbital_symmetries[c]
                            nc = system.point_group_irrep_to_number[symc]

                            #sym = n_refsym
                            #sym = na ^ nb ^ nc ^ nk ^ nj ^ ni
                            sym = ni ^ nj
                            sym = sym ^ nk
                            sym = sym ^ na
                            sym = sym ^ nb
                            sym = sym ^ nc
                            sym = sym ^ n_refsym
                            countsym[sym] += 1  
    
    total = sum(countsym)

    return countsym, total
