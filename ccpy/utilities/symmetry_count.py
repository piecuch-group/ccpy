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


def count_singles(ncore, noa, nob, norb, ordered_syms, sym_mult_table, ref_sym):
    # Assign symmetry and count r1
    countsym = [0 for i in range(len(sym_mult_table))]
    for i in range(ncore, noa + ncore):
        symi = ordered_syms[i]
        for a in range(noa + ncore, norb):
            syma = ordered_syms[a]
            x = sym_mult_table[symi][syma]
            x = sym_mult_table[x][ref_sym]
            countsym[x] += 1  # r1(a)

    for i in range(ncore, nob + ncore):
        symi = ordered_syms[i]
        for a in range(nob + ncore, norb):
            syma = ordered_syms[a]
            x = sym_mult_table[symi][syma]
            x = sym_mult_table[x][ref_sym]
            countsym[x] += 1  # r1(b)

    total = sum(countsym)

    return countsym, total


def count_doubles(ncore, noa, nob, norb, ordered_syms, sym_mult_table, ref_sym):
    # Assign symmetry and count r2
    countsym = [0 for i in range(len(sym_mult_table))]
    for i in range(ncore, noa - 1 + ncore):
        symi = ordered_syms[i]
        for j in range(i + 1, noa + ncore):
            symj = ordered_syms[j]
            for a in range(noa + ncore, norb - 1):
                syma = ordered_syms[a]
                for b in range(a + 1, norb):
                    symb = ordered_syms[b]
                    x = sym_mult_table[symi][symj]
                    x = sym_mult_table[x][syma]
                    x = sym_mult_table[x][symb]
                    x = sym_mult_table[x][ref_sym]
                    countsym[x] += 1  # r2(aa)

    for i in range(ncore, nob - 1 + ncore):
        symi = ordered_syms[i]
        for j in range(i + 1, nob + ncore):
            symj = ordered_syms[j]
            for a in range(nob + ncore, norb - 1):
                syma = ordered_syms[a]
                for b in range(a + 1, norb):
                    symb = ordered_syms[b]
                    x = sym_mult_table[symi][symj]
                    x = sym_mult_table[x][syma]
                    x = sym_mult_table[x][symb]
                    x = sym_mult_table[x][ref_sym]
                    countsym[x] += 1  # r2(bb)

    for i in range(ncore, noa + ncore):
        symi = ordered_syms[i]
        for j in range(ncore, nob + ncore):
            symj = ordered_syms[j]
            for a in range(noa + ncore, norb):
                syma = ordered_syms[a]
                for b in range(nob + ncore, norb):
                    symb = ordered_syms[b]
                    x = sym_mult_table[symi][symj]
                    x = sym_mult_table[x][syma]
                    x = sym_mult_table[x][symb]
                    x = sym_mult_table[x][ref_sym]
                    countsym[x] += 1  # r2(ab)

    total = sum(countsym)

    return countsym, total


def count_triples(ncore, noa, nob, norb, ordered_syms, sym_mult_table, ref_sym):
    # Assign symmetry and count r3
    countsym = [0 for i in range(len(sym_mult_table))]
    for i in range(ncore, noa - 2 + ncore):
        symi = ordered_syms[i]
        for j in range(i + 1, noa - 1 + ncore):
            symj = ordered_syms[j]
            for k in range(j + 1, noa + ncore):
                symk = ordered_syms[k]
                for a in range(noa + ncore, norb - 2):
                    syma = ordered_syms[a]
                    for b in range(a + 1, norb - 1):
                        symb = ordered_syms[b]
                        for c in range(b + 1, norb):
                            symc = ordered_syms[c]
                            x = sym_mult_table[symi][symj]
                            x = sym_mult_table[x][symk]
                            x = sym_mult_table[x][syma]
                            x = sym_mult_table[x][symb]
                            x = sym_mult_table[x][symc]
                            x = sym_mult_table[x][ref_sym]
                            countsym[x] += 1  # r3(aaa)

    for i in range(ncore, nob - 2 + ncore):
        symi = ordered_syms[i]
        for j in range(i + 1, nob - 1 + ncore):
            symj = ordered_syms[j]
            for k in range(j + 1, nob + ncore):
                symk = ordered_syms[k]
                for a in range(nob + ncore, norb - 2):
                    syma = ordered_syms[a]
                    for b in range(a + 1, norb - 1):
                        symb = ordered_syms[b]
                        for c in range(b + 1, norb):
                            symc = ordered_syms[c]
                            x = sym_mult_table[symi][symj]
                            x = sym_mult_table[x][symk]
                            x = sym_mult_table[x][syma]
                            x = sym_mult_table[x][symb]
                            x = sym_mult_table[x][symc]
                            x = sym_mult_table[x][ref_sym]
                            countsym[x] += 1  # r3(bbb)

    for i in range(ncore, noa - 1 + ncore):
        symi = ordered_syms[i]
        for j in range(i + 1, noa + ncore):
            symj = ordered_syms[j]
            for k in range(ncore, nob + ncore):
                symk = ordered_syms[k]
                for a in range(noa + ncore, norb - 1):
                    syma = ordered_syms[a]
                    for b in range(a + 1, norb):
                        symb = ordered_syms[b]
                        for c in range(nob + ncore, norb):
                            symc = ordered_syms[c]
                            x = sym_mult_table[symi][symj]
                            x = sym_mult_table[x][symk]
                            x = sym_mult_table[x][syma]
                            x = sym_mult_table[x][symb]
                            x = sym_mult_table[x][symc]
                            x = sym_mult_table[x][ref_sym]
                            countsym[x] += 1  # r3(aab)

    for i in range(ncore, noa + ncore):
        symi = ordered_syms[i]
        for j in range(ncore, nob - 1 + ncore):
            symj = ordered_syms[j]
            for k in range(j + 1, nob + ncore):
                symk = ordered_syms[k]
                for a in range(noa + ncore, norb):
                    syma = ordered_syms[a]
                    for b in range(nob + ncore, norb - 1):
                        symb = ordered_syms[b]
                        for c in range(b + 1, norb):
                            symc = ordered_syms[c]
                            x = sym_mult_table[symi][symj]
                            x = sym_mult_table[x][symk]
                            x = sym_mult_table[x][syma]
                            x = sym_mult_table[x][symb]
                            x = sym_mult_table[x][symc]
                            x = sym_mult_table[x][ref_sym]
                            countsym[x] += 1  # r3(abb)

    total = sum(countsym)

    return countsym, total


def get_symmetry_count(sys, nexc):

    pg = sys["point_group"]

    norb = sys["Norb"]
    ncore = sys["Nfroz"]
    noa = sys["Nocc_a"]
    nob = sys["Nocc_b"]

    mo_syms = sys["sym"]
    pg_irreps = get_pg_irreps(pg)

    ordered_syms = []
    for sym in mo_syms:
        ordered_syms.append(pg_irreps[sym])

    print("\nSymmetry Count")
    print("------------------------------")
    print(" Symmetry: {}".format(pg))
    sym_mult_table = sym_table(pg)

    irrep_keys = []
    for i in pg_irreps.keys():
        irrep_keys.append(i)

    if noa == nob:
        ref_sym = 0
    elif noa - nob == 1:
        ref_sym = ordered_syms[nob + ncore]
    elif noa - nob > 1:
        ref_sym = ordered_syms[nob + ncore]
        for i in range(ncore + nob + 1, noa + ncore):
            orb_sym = ordered_syms[i]
            ref_sym = sym_mult_table[ref_sym][orb_sym]
    print(" Irrep of reference state: {}".format(irrep_keys[ref_sym]))

    # Count singles
    if nexc == 1:
        countsym, total = count_singles(
            ncore, noa, nob, norb, ordered_syms, sym_mult_table, ref_sym
        )
        print(" Singles obtained from counting:")
        for irrep in range(0, len(countsym)):
            print(" {:5} {: 13d}".format(irrep_keys[irrep], countsym[irrep]))
        print(" Total = {: 11d}".format(total))
        print("")

    # Count doubles
    if nexc == 2:
        countsym, total = count_doubles(
            ncore, noa, nob, norb, ordered_syms, sym_mult_table, ref_sym
        )
        print(" Doubles obtained from counting:")
        for irrep in range(0, len(countsym)):
            print(" {:5} {: 13d}".format(irrep_keys[irrep], countsym[irrep]))
        print(" Total = {: 11d}".format(total))
        print("")

    # Count triples
    if nexc == 3:
        countsym, total = count_triples(
            ncore, noa, nob, norb, ordered_syms, sym_mult_table, ref_sym
        )
        print(" Triples obtained from counting:")
        for irrep in range(0, len(countsym)):
            print(" {:5} {: 13d}".format(irrep_keys[irrep], countsym[irrep]))
        print(" Total = {: 11d}".format(total))
        print("")

    return countsym, countsym[ref_sym]
