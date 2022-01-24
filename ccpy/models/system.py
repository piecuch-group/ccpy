from pydantic import BaseModel

from ccpy.utilities.symmetry_count import get_pg_irreps, sym_table
from ccpy.models.integrals import Integrals


class System(BaseModel):

    nfrozen: int
    nelectrons: int
    norbitals: int
    noccupied_alpha: int
    noccupied_beta: int
    nunoccupied_alpha: int
    nunoccupied_beta: int
    multiplicity: int
    is_closed_shell: bool
    charge: int
    basis: str
    integrals: Integrals


def build_system(gamess_file, Nfroz):
    import cclib

    eVtohartree = 0.036749308136649

    data = cclib.io.ccread(gamess_file)

    print_gamess_info(data, gamess_file)

    Nelec = data.nelectrons
    multiplicity = data.mult
    Nocc_a = (Nelec + (multiplicity - 1)) // 2
    Nocc_b = (Nelec - (multiplicity - 1)) // 2
    Norb = data.nmo
    methods = data.metadata["methods"]
    scftyp = methods[0]

    charge = data.charge
    basis = data.gbasis

    mo_energies = data.moenergies

    mo_occupation =  \
        [2.0] * Nocc_b \
        + [1.0] * (Nocc_a - Nocc_b) \
        + [0.0] * (Norb - Nocc_a)

    assert len(mo_occupation) == Norb, "Occupation number vector has wrong size"

    Nunocc_a = Norb - Nocc_a
    Nunocc_b = Norb - Nocc_b

    mo_energies_hartree = [data.moenergies[0][i]*eVtohartree for i in range(Norb)]

    # converting A1, A2, B1, B2, etc. irrep symbols to numbers using a consistent scheme
    point_group = get_gamess_pointgroup(gamess_file)
    pg_irrep_map = get_pg_irreps(point_group)
    mosyms_num = [pg_irrep_map[x.upper()] for x in data.mosyms[0]]

    sys_t = {'Nelec' : Nelec-2*Nfroz,
             'Nocc_a' : Nocc_a-Nfroz,
             'Nocc_b' : Nocc_b-Nfroz,
             'Nunocc_a' : Nunocc_a,
             'Nunocc_b' : Nunocc_b,
             'Norb' : Norb,
             'Nfroz' : Nfroz,
             'sym' : data.mosyms[0],
             'sym_nums' : mosyms_num,
             'mo_energy' : mo_energies_hartree,
             'mo_vector' : data.mocoeffs,
             'point_group' : point_group,
             'irrep_map' : pg_irrep_map,
             'pg_mult_table' : sym_table(point_group),
             'charge' : charge,
             'multiplicity' : multiplicity,
             'basis' : basis,
             'mo_occ' : mo_occupation}

    return sys_t


def print_gamess_info(data, gamess_file):

    basis_name = data.metadata.get("basis_set", "User-defined")
    scf_type = data.metadata["methods"][0]

    print('')
    print('GAMESS Run Information:')
    print('-----------------------------------------------')
    print('  GAMESS file location : {}'.format(gamess_file))
    print('  SCF Type : {}'.format(scf_type))
    print('  Basis : {}'.format(basis_name))
    print('')


def get_gamess_pointgroup(gamess_file):
    """Dumb way of getting the point group from GAMESS and GAMESS only"""
    point_group = 'C1'
    flag_found = False
    with open(gamess_file,'r') as f:
        for line in f.readlines():

            if flag_found:
                order = line.split()[-1]
                if len(point_group) == 3:
                    point_group = point_group[0] + order + point_group[2]
                if len(point_group) == 2:
                    point_group = point_group[0] + order
                if len(point_group) == 1:
                    point_group = point_group[0] + order
                break

            if 'THE POINT GROUP OF THE MOLECULE IS' in line:
                point_group = line.split()[-1]
                flag_found = True

    return point_group

def main(args):

    sys_t = build_system(args.gamess_file,args.frozen)
    for key, value in sys_t.items():
        print(key,'->',value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('gamess_file',type=str,help='Path to GAMESS log file containing SCF calculation')
    parser.add_argument('-f','--frozen',type=int,default=0,help='Number of frozen spatial orbitals')
    args = parser.parse_args()
    main(args)
