import numpy as np

import itertools
from itertools import combinations, product

import scipy as sp
import scipy.linalg as la
from scipy.linalg import block_diag

def decompose_skew_schur(h, **args):
    """
        In this function we implement the decomposition of a skew-symmetric matrix
        with output block-diagonalized matrix well-sorted in order.
    """
    assert np.allclose(h+h.T, np.zeros_like(h))
    assert h.shape[0] % 2 == 0

    t, z = la.schur(h, output='real', **args)

    
    for i in range(t.shape[0]//2):
        """
            Check the diagonalized blocks one by one, and flip those disordered
        """
        i1 = i*2
        i2 = i1+1
        if t[i1, i2] < t[i2, i1]:
            P = np.identity(t.shape[0])
            P[i1,i1]=0
            P[i2,i2]=0
            P[i1,i2]=1
            P[i2,i1]=1
            t = P@t@P 
            z = P@z
    return t, z

def H_fermionic_skew(Jx, Jy, Jz, N, SigmaY, D_list, bc):
    """
        Construct the skew-symmetric matrix used for the 
        Majorana Hamiltonian
    """
    assert bc=='open' or bc=='periodic'
    assert len(D_list)==2*N
    if bc=='periodic':
        assert SigmaY==1 or SigmaY==-1
        # else it doesn't matter lol

    mat_size = 4*N
    L=2*N
    H = np.zeros((mat_size, mat_size))
    for i in range(L-1):
        H[2*i, 2*i+1] = -Jz
        H[2*i+1, 2*i+2] = Jx
        H[2*i, 2*i+3] = Jy * D_list[i]
    H[mat_size-2, mat_size-1] = -Jz

    if bc=='periodic':
        H[mat_size-1, 0] = -Jx*SigmaY 
        H[mat_size-2, 1] = Jy * D_list[-1] *(-SigmaY)
    

    return H - H.T
    
def get_Majorana_spectrum(
    Jx, Jy, Jz, 
    N, 
    SigmaY, 
    D_list, 
    bc, 
    method='M2',
    ):
    """
        Calculate the Majorana spectrum of a given configuration
    """
    H_skew = H_fermionic_skew(Jx, Jy, Jz, N, SigmaY, D_list, bc)

    # The matrix H is of dim 4N
    # and the eigenvalues will be given in descending order,
    # so by simply only taking the first 2N values one can successfully get the true values
    assert method=='iM' or method=='M2'
    if method=='iM': # get the eigenvalues via solving the eigen-problem of iM
        H = 1j * H_skew
        eigenvalues = np.linalg.eigvalsh(H)[0:2*N] # not necessary sort again since it's already sorted
    elif method=='M2': # get the eigenvalues via solving the M**2
        H = H_skew @ H_skew 
        eigenvalues = np.abs(np.linalg.eigvalsh(H)[::2])
        eigenvalues = -np.sqrt(eigenvalues)

    if any(eigenvalues>0):
        raise ValueError("WRONG: Positiveness in the spectra!")
    return eigenvalues

def get_parity(
    Jx, Jy, Jz, 
    N, 
    SigmaY, 
    D_list, 
    bc, 
    method='M2',
    ):
    H_skew = H_fermionic_skew(Jx, Jy, Jz, N, SigmaY, D_list, bc)
    t, Q = decompose_skew_schur(H_skew)
    return np.round(la.det(Q) * np.prod(D_list))

def get_two_E(Jx, Jy, Jz, N, bc='periodic'):
    E_list = []
    
    D_list = [1 for i in range(2*N)]
    for SigmaY in [1 , -1]:
        if get_parity(Jx, Jy, Jz, N, SigmaY, D_list, bc)==1:
            E_list.append(np.sum(get_Majorana_spectrum(Jx, Jy, Jz, N, SigmaY, D_list, bc)))
    
    D_list[-1] *= -1
    for SigmaY in [1 , -1]:
        if get_parity(Jx, Jy, Jz, N, SigmaY, D_list, bc)==1:
            E_list.append(np.sum(get_Majorana_spectrum(Jx, Jy, Jz, N, SigmaY, D_list, bc)))
            
    return E_list

def get_lowest_E(Jx, Jy, Jz, N, D_list, bc='periodic'):
    E_list = []
    for SigmaY in [1, -1]:
        E_spec = get_Majorana_spectrum(Jx, Jy, Jz, N, SigmaY, D_list, bc)
        E_sum = np.sum(E_spec)
        E_min = np.max(E_spec)
        if get_parity(Jx, Jy, Jz, N, SigmaY, D_list, bc) == 1:
            E_list.append(E_sum)
        else:
            E_list.append(E_sum - 2*E_min)
    return E_list

def test():
    Jx=1
    Jy=1
    Jz=1
    N=6
    bc = 'open'

    D_list = [1 for i in range(2*N)]
    spec = get_Majorana_spectrum(Jx,Jy,Jz,N,1,D_list,bc)
    print((spec))

    # D_list = [-1 for i in range(2*N)]
    # spec = get_Majorana_spectrum(Jx,Jy,Jz,N,1,D_list,bc)
    # print((spec))

    method = 'M2'
    print(get_Majorana_spectrum(Jx, Jy, Jz, N, 1, D_list, bc, method=method))

# test()
