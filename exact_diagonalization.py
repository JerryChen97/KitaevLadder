import scipy
from scipy import sparse
from scipy.sparse.linalg import eigs, eigsh # use eigsh instead of eigs to speed up as well as to get real eigenvalues

import numpy as np

## the elementary Pauli matrices
X = sparse.csr_matrix(np.array([[0., 1.], [1., 0.]]))
Y = sparse.csr_matrix(np.array([[0., -1.j], [1.j, 0.]]))
Z = sparse.csr_matrix(np.array([[1., 0.], [0., -1.]]))
I = sparse.eye(2)

sigma_list = [I, X, Y, Z]

# the function to generate the many-body onsite Pauli operators
def Sigma(op_num, site, L):
    assert op_num==0 or op_num==1 or op_num==2 or op_num==3
    assert site>=0 and site<L
    
    op = sigma_list[op_num]
    
    if site==0:
        return sparse.kron(op, sparse.csr_matrix(np.eye(2**(L-1))))
    elif site==L-1:
        return sparse.kron(sparse.csr_matrix(np.eye(2**(L-1))), op)
    else: 
        IA = sparse.kron(sparse.csr_matrix(np.eye(2**(site))), op)
        return sparse.kron(IA, sparse.csr_matrix(np.eye(2**(L-1-site))))

def get_H_Kitaev_Ladder1(
    Jx, Jy, Jz, 
    N, # the number of unitcells
    bc, # boundary condition
    ):
    # currently i've only implemented open and periodic boundary conditions
    assert bc=='open' or bc=='periodic'

    # parameter setting
    unit_size = 4 # every unitcell is to be composed of 4 different spins
    L = N * unit_size # N is the number of unitcell
    unit_number = 2*N # the number of snake-order units

    H = sparse.csr_matrix((2**L, 2**L)) # initialize the sparse matrix for representing the Hamiltonian
    for i in range(unit_number - 1):
        site1 = 2*i + 1
        site2 = site1 + 1

        op_num = 1 # 1 represents x

    #     print("i = ", i)
        X1 = Sigma(op_num, site1, L=L)
        X2 = Sigma(op_num, site2, L=L)
        H += Jx*X1.dot(X2)

        pass

    for i in range(unit_number - 1):
        site1 = 2*i
        site2 = site1 + 3

        op_num = 2
        Y1 = Sigma(op_num, site1, L=L)
        Y2 = Sigma(op_num, site2, L=L)
        H += Jy*Y1.dot(Y2)

        pass
    
    for i in range(unit_number):
        site1 = 2*i
        site2 = site1 + 1

        op_num = 3
        Z1 = Sigma(op_num, site1, L=L)
        Z2 = Sigma(op_num, site2, L=L)
        H += Jz*Z1.dot(Z2)

        pass

    if bc=='periodic':
        # add the Y term crossing the boundary
        site1 = L - 2
        site2 = 1
        
        op_num = 2
        Y1 = Sigma(op_num, site1, L=L)
        Y2 = Sigma(op_num, site2, L=L)
        H += Jy*Y1.dot(Y2)
        
        # add the X term crossing the boundary
        site1 = L - 1
        site2 = 0
        
        op_num = 1
        X1 = Sigma(op_num, site1, L=L)
        X2 = Sigma(op_num, site2, L=L)
        H += Jx*X1.dot(X2)
        pass
    
    return H

def get_D_Kitaev_Ladder1(
    N, # the number of unitcells
    bc, # boundary condition
    ):
    """
        To get the local symmetries of the Kitaev ladder given the system size and the boundary condition
    """

    # currently i've only implemented open and periodic boundary conditions
    assert bc=='open' or bc=='periodic'

    # parameter setting
    unit_size = 4 # every unitcell is to be composed of 4 different spins
    L = N * unit_size # N is the number of unitcell
    unit_number = 2*N # the number of snake-order units

    D_list = []
    for i in range(unit_number-1): # all the obc D's
        D = sparse.csr_matrix((2**L, 2**L)) # initialize the sparse matrix for representing the local symmetries
        # x0y1y2x3, x2y3y4x5, ..., x4N-4y4N-3y4N-2x4N-1
        site1 = 2*i   
        site2 = site1+1
        site3 = site2+1
        site4 = site3+1 

        X1 = Sigma(1, site1, L=L)
        Y2 = Sigma(2, site2, L=L)
        Y3 = Sigma(2, site3, L=L)
        X4 = Sigma(1, site4, L=L)

        D += X1.dot(Y2.dot(Y3.dot(X4)))

        D_list.append(D)
        

    if bc=='periodic':
        D = sparse.csr_matrix((2**L, 2**L)) # initialize the sparse matrix for representing the local symmetries
        # x4N-2y4N-1y0x1
        site1 = 4*N - 2
        site2 = site1+1
        site3 = 0
        site4 = 1

        X1 = Sigma(1, site1, L=L)
        Y2 = Sigma(2, site2, L=L)
        Y3 = Sigma(2, site3, L=L)
        X4 = Sigma(1, site4, L=L)

        D += X1.dot(Y2.dot(Y3.dot(X4)))

        D_list.append(D)
    
    return D_list

class KitaevLadder:
    def __init__(self, Jx, Jy, Jz, N, bc):
        assert bc=='open' or bc=='periodic'
        self.N = N
        self.bc = bc 
        self.J = (Jx, Jy, Jz)
        self.Hamiltonian = get_H_Kitaev_Ladder1(Jx, Jy, Jz, N, bc)
        self.D_list = get_D_Kitaev_Ladder1(N, bc)

    def get_eigen(self, k=10):
        """
            Return the k lowest-lying vectors.
        """
        eigenvalues, eigenvectors = eigsh(self.Hamiltonian, k=k, which='SA')
        eig_dict = {}
        for i in range(eigenvalues.shape[0]):
            eigenvalue = eigenvalues[i]
            eigenvector = eigenvectors[:,i]
            eig_dict[eigenvalue] = eigenvector
        eigenvalues_sorted = np.sort(eigenvalues)
        eigenvectors_sorted = [eig_dict[v] for v in eigenvalues_sorted]

        return eigenvalues_sorted, eigenvectors_sorted

# M=KitaevLadder(Jx=1,Jy=1,Jz=1,N=1,bc='open')
# print(M.Hamiltonian)
# for D in M.D_list:
#     print(D)
