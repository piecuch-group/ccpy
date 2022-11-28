import numpy as np

def print_table(result, E_parent, total_num_triples):

    print("  N_det(in)   % T      CC(P)     CC(P;Q)")
    print("--------------------------------------------")
    for i in range(len(result['CCP'])):
            ndet = result['Ndet(in)'][i]
            perc_triples = result['num_triples'][i] / total_num_triples * 100
            error_ccp = (result['CCP'][i] - E_parent) * 1000
            error_ccpq = (result['CCPQ'][i] - E_parent) * 1000
            print(f'{ndet:9d}    {perc_triples:4.2f}    {error_ccp:9.6f}    {error_ccpq:9.6f}')

def swati_data():

    Eccsdt = -154.1953887948
    Eccsd = -154.170742321053
    Ecrcc23 = -154.195422049255
    total_num_triples = 1

    Ndet = [1, 100000, 250000, 500000, 1000000, 5000000, 10000000]

    num_triples = [0, 0, 0, 0, 0, 0, 0]

    Eccp = [-154.170742321053,
            -154.17494233579362,
            -154.17704819795009,
            -154.17865197643658,
            -154.17982163306846,
            -154.18291527686890,
            -154.18475211327194]

    Eccpq = [-154.195422049255,
             -154.19536091863378,
             -154.19528884902635,
             -154.19523469422802,
             -154.19521622428275,
             -154.19522135477368,
             -154.19525515412184]

    result = {'Ndet(in)': Ndet, "num_triples": num_triples, "CCP": Eccp, "CCPQ": Eccpq}
    print_table(result, Eccsdt, total_num_triples)

if __name__ == "__main__":

        Eccsd = -39.0415137828
        Ecrcc23 = -39.0436937424
        Eccsdt = -39.04367754
        total_num_triples = 45680


        Ndet = [1, 1000, 5000, 10000, 50000]

        num_triples = [0, 568, 2467, 4632, 17125]

        Eccp = [-39.0415137828,
                -39.0417362377,
                -39.0425834186,
                -39.0429922807,
                -39.0435672110,
                ]

        Eccpq = [-39.0436937424,
                 -39.0436985166,
                 -39.0436925767,
                 -39.0436879232,
                 -39.0436765194,
                 ]

        Eccpq_jun = [-39.0436937424,
                     -39.043689404034005,
                     -39.043684686744434,
                     -39.043682260082022,
                     -39.043679233294498,
                     ]

        result = {'Ndet(in)' : Ndet, "num_triples" : num_triples, "CCP" : Eccp, "CCPQ" : Eccpq_jun}
        print_table(result, Eccsdt, total_num_triples)

        swati_data()