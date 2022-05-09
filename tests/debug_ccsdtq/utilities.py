import numpy as np

def antisymmetrize_aaaa(x):
 
    # antisymmetrize the spin-integrated residuals
    x -= np.transpose(x, (0, 1, 2, 3, 4, 6, 5, 7)) # (jk)
    x -= np.transpose(x, (0, 1, 2, 3, 4, 7, 6, 5)) + np.transpose(x, (0, 1, 2, 3, 4, 5, 7, 6)) # (l/jk)
    x -= np.transpose(x, (0, 1, 2, 3, 5, 4, 6, 7)) + np.transpose(x, (0, 1, 2, 3, 6, 5, 4, 7)) + np.transpose(x, (0, 1, 2, 3, 7, 5, 6, 4)) # (i/jkl)
    x -= np.transpose(x, (0, 2, 1, 3, 4, 5, 6, 7)) # (bc)
    x -= np.transpose(x, (0, 3, 2, 1, 4, 5, 6, 7)) + np.transpose(x, (0, 1, 3, 2, 4, 5, 6, 7)) # (d/bc)
    x -= np.transpose(x, (1, 0, 2, 3, 4, 5, 6, 7)) + np.transpose(x, (2, 1, 0, 3, 4, 5, 6, 7)) + np.transpose(x, (3, 1, 2, 0, 4, 5, 6, 7)) # (a/bcd)

    return x

def antisymmetrize_aaa(x):
    x -= np.transpose(x, (0, 1, 2, 3, 5, 4)) # (jk)
    x -= np.transpose(x, (0, 1, 2, 4, 3, 5)) + np.transpose(x, (0, 1, 2, 5, 4, 3)) # (i/jk)
    x -= np.transpose(x, (0, 2, 1, 3, 4, 5)) # (bc)
    x -= np.transpose(x, (2, 1, 0, 3, 4, 5)) + np.transpose(x, (1, 0, 2, 3, 4, 5)) # (a/bc)
    return x

def antisymmetrize_aa(x):

    x -= np.transpose(x, (1, 0, 2, 3))
    x -= np.transpose(x, (0, 1, 3, 2))

    return x
