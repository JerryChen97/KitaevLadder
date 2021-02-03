import numpy as np
import matplotlib.pyplot as plt

from tenpy.networks.site import SpinHalfSite
from tenpy.networks.mps import TransferMatrix
import tenpy.linalg.np_conserved as npc

from utility import linspace

def detect_SPT_onsite(psi, symmetry):
    assert symmetry in ['Sigmax', 'Sigmay', 'Sigmaz', 'upperSigmaz', 'lowerSigmaz']
    if symmetry=='Sigmax' or symmetry=='Sigmay' or symmetry=='Sigmaz':
        op = symmetry
        
        s = SpinHalfSite(conserve=None)
        op = npc.expm(1.j * (np.pi / 2) * s.get_op(op))
        
        # First of all, save a copy of the input wavefunction psi
        psi_copy = psi.copy()
        for i in range(psi_copy.L): # apply the op to all sites
            psi_copy.apply_local_op(i, op)
        
        TM = TransferMatrix(psi, psi_copy)
        eta, G = TM.eigenvectors(num_ev=1)
#         if not np.allclose(eta, 1):
#             print(f'eta={eta}')
#             raise ValueError('The largest eigenvalue of the transfer matrix is not 1!')
        if not np.allclose(np.abs(eta), 1, atol=0.01):
            return 0
        
        U = G[0]
        Uop = U.split_legs()
        chi=(Uop.shape[0])
        Uop *= np.sqrt(chi) # rescale it
        
    else: # if the symmetry is the upper or the lower Sigmaz
        op = 'Sigmaz'
        
        
        s = SpinHalfSite(conserve=None)
        op = npc.expm(1.j * (np.pi / 2) * s.get_op(op))
        
        # First of all, save a copy of the input wavefunction psi
        psi_copy = psi.copy()
        # Then calculate the specific properties of the wavefunction
        N = psi_copy.L // 4 # The number of unicells
        
        if symmetry == 'upperSigmaz':
            for i in range(N):
                psi_copy.apply_local_op(2*i, op)
                psi_copy.apply_local_op(2*i+3, op)
        else:
            for i in range(N):
                psi_copy.apply_local_op(2*i+1, op)
                psi_copy.apply_local_op(2*i+2, op)
            
        
        TM = TransferMatrix(psi, psi_copy)
        eta, G = TM.eigenvectors(num_ev=1)
#         if not np.allclose(eta, 1):
#             print(f'eta={eta}')
#             raise ValueError('The largest eigenvalue of the transfer matrix is not 1!')
        if not np.allclose(np.abs(eta), 1, atol=0.01):
            return 0
        
        U = G[0]
        Uop = U.split_legs()
        chi=(Uop.shape[0])
        Uop *= np.sqrt(chi) # rescale it
        
    return Uop, chi

def detect_SPT_D2(psi):
    
    op_list = ['Sigmax', 'Sigmay']
    U_list = []
    
    for op in op_list:
        
        
        s = SpinHalfSite(conserve=None)
        op = npc.expm(1.j * (np.pi/2) * s.get_op(op))
        
        # First of all, save a copy of the input wavefunction psi
        psi_copy = psi.copy()
        for i in range(psi_copy.L):
            psi_copy.apply_local_op(i, op)
        
        TM = TransferMatrix(psi, psi_copy)
        eta, G = TM.eigenvectors(num_ev=1)
#         if not np.allclose(eta, 1):
#             print(f'eta={eta}')
#             raise ValueError('The largest eigenvalue of the transfer matrix is not 1!')
        if not np.allclose(np.abs(eta), 1, atol=0.01):
            return 0
        
        U = G[0]
        Uop = U.split_legs()
        chi=(Uop.shape[0])
        Uop *= np.sqrt(chi) # rescale it
        U_list.append(Uop)

    U1 = U_list[0]
    U2 = U_list[1]
    U12 = npc.tensordot(U1, U2, axes=([1], [0]))
    U21 = npc.tensordot(U2, U1, axes=([1], [0]))
    return npc.inner(U12, U21.conj()) / chi
    
def detect_SPT_D2_upper_Z(psi):
    
#     op_list = ['Sigmax', 'Sigmay']
    U_list = []
    
    op = 'Sigmaz'

    s = SpinHalfSite(conserve=None)
    op = npc.expm(1.j * (np.pi/2) * s.get_op(op))
    
    
    ###### The upper ######
    # First of all, save a copy of the input wavefunction psi
    psi_copy = psi.copy()
    
    # Then calculate the specific properties of the wavefunction
    N = psi_copy.L // 4 # The number of unicells
    for i in range(N):
        psi_copy.apply_local_op(2*i, op)
        psi_copy.apply_local_op(2*i+3, op)

    TM = TransferMatrix(psi, psi_copy)
    eta, G = TM.eigenvectors(num_ev=1)
#         if not np.allclose(eta, 1):
#             print(f'eta={eta}')
#             raise ValueError('The largest eigenvalue of the transfer matrix is not 1!')
    if not np.allclose(np.abs(eta), 1, atol=0.01):
        return 0
    
    
    U = G[0]
    Uop = U.split_legs()
    chi=(Uop.shape[0])
    Uop *= np.sqrt(chi) # rescale it
    U_list.append(Uop)

    ###### The lower ######
    # First of all, save a copy of the input wavefunction psi
    psi_copy = psi.copy()
    
    # Then calculate the specific properties of the wavefunction
    N = psi_copy.L // 4 # The number of unicells
    for i in range(N): # apply the operators
        psi_copy.apply_local_op(2*i+1, op)
        psi_copy.apply_local_op(2*i+2, op)

    TM = TransferMatrix(psi, psi_copy)
    eta, G = TM.eigenvectors(num_ev=1)
#         if not np.allclose(eta, 1):
#             print(f'eta={eta}')
#             raise ValueError('The largest eigenvalue of the transfer matrix is not 1!')
    if not np.allclose(np.abs(eta), 1, atol=0.01):
        return 0
    
    
    U = G[0]
    Uop = U.split_legs()
    chi=(Uop.shape[0])
    Uop *= np.sqrt(chi) # rescale it
    U_list.append(Uop)
        
    
    U1 = U_list[0]
    U2 = U_list[1]
    U12 = npc.tensordot(U1, U2, axes=([1], [0]))
    U21 = npc.tensordot(U2, U1, axes=([1], [0]))
    return npc.inner(U12, U21.conj()) / chi