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


if __name__ == "__main__":

        Eccsd = -38.9802624745
        Ecrcc23 = -38.9810572210
        Eccsdt = -38.98105947
        total_num_triples = 3913


        Ndet = [1, 1000, 5000, 10000, 50000]

        num_triples = [0, 386, 2043, 2963, 3759]

        Eccp = [-38.9802624745,
                -38.9806474054,
                -38.9809970624,
                -38.9810531086,
                -38.9810592029]

        Eccpq = [-38.9810572210,
                 -38.9810550552,
                 -38.9810584291,
                 -38.9810590987,
                 -38.9810594727]

        Eccpq_jun = [-38.9810572210,
                     -38.981062884641894,
                     -38.981060662919653,
                     -38.981059483263245,
                     -38.981059472828051]

        result = {'Ndet(in)' : Ndet, "num_triples" : num_triples, "CCP" : Eccp, "CCPQ" : Eccpq_jun}
        print_table(result, Eccsdt, total_num_triples)