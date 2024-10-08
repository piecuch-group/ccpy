import numpy as np

def build_2index_batch_vvvv_aa(a, b, H):
    '''Builds the 2-index batch of 4-particle ERIs out of Cholesky vectors
    V_ef(a, b) = <x|ae><x|bf> - <x|af><x|be>'''
    batch_ints = np.einsum("xe,xf->ef", H.chol.a.vv[:, a, :], H.chol.a.vv[:, b, :], optimize=True)
    batch_ints -= batch_ints.T
    return batch_ints

def build_2index_batch_vvvv_bb(a, b, H):
    '''Builds the 2-index batch of 4-particle ERIs out of Cholesky vectors
    V_ef(a~, b~) = <x~|a~e~><x~|b~f~> - <x~|a~f~><x~|b~e~>'''
    batch_ints = np.einsum("xe,xf->ef", H.chol.b.vv[:, a, :], H.chol.b.vv[:, b, :], optimize=True)
    batch_ints -= batch_ints.T
    return batch_ints

def build_2index_batch_vvvv_ab(a, b, H):
    '''Builds the 2-index batch of 4-particle ERIs out of Cholesky vectors
    V_ef(a, b~) = <x|ae><x~|b~f~>'''
    batch_ints = np.einsum("xe,xf->ef", H.chol.a.vv[:, a, :], H.chol.b.vv[:, b, :], optimize=True)
    return batch_ints

def build_3index_batch_vvvv_ab(a, H):
    '''Builds the 3-index batch of 4-particle ERIs out of Cholesky vectors
    V_ef(a) = <x|ae><x~|b~f~>'''
    batch_ints = np.einsum("xe,xbf->bef", H.chol.a.vv[:, a, :], H.chol.b.vv, optimize=True)
    return batch_ints

def build_2index_batch_vvvv_aa_herm(a, b, H):
    '''Builds the 2-index batch of 4-particle ERIs out of Cholesky vectors
    V_ef(a, b) = <x|ea><x|fb> - <x|eb><x|fa>'''
    batch_ints = np.einsum("xe,xf->ef", H.chol.a.vv[:, :, a], H.chol.a.vv[:, :, b], optimize=True)
    batch_ints -= batch_ints.T
    return batch_ints

def build_2index_batch_vvvv_bb_herm(a, b, H):
    '''Builds the 2-index batch of 4-particle ERIs out of Cholesky vectors
    V_ef(a~, b~) = <x~|e~a~><x~|f~b~> - <x~|f~a~><x~|e~b~>'''
    batch_ints = np.einsum("xe,xf->ef", H.chol.b.vv[:, :, a], H.chol.b.vv[:, :, b], optimize=True)
    batch_ints -= batch_ints.T
    return batch_ints

def build_3index_batch_vvvv_ab_herm(a, H):
    '''Builds the 3-index batch of 4-particle ERIs out of Cholesky vectors
    V_ef(a) = <x|ea><x~|f~b~>'''
    batch_ints = np.einsum("xe,xfb->bef", H.chol.a.vv[:, :, a], H.chol.b.vv, optimize=True)
    return batch_ints

def build_2index_batch_vvvv_ab_herm(a, b, H):
    '''Builds the 2-index batch of 4-particle ERIs out of Cholesky vectors
    V_ef(a, b~) = <x|ae><x~|b~f~>'''
    batch_ints = np.einsum("xe,xf->ef", H.chol.a.vv[:, :, a], H.chol.b.vv[:, :, b], optimize=True)
    return batch_ints