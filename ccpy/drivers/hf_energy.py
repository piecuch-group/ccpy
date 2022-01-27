import numpy as np

def calc_hf_energy(Z, V, sys):

    e1a = np.einsum('ii->', Z.aa.oo)
    e1b = np.einsum('ii->', Z.bb.oo)
    e2a = 0.5*np.einsum('ijij->', V.aaaa.oooo)