import numpy as np

def pollmann_turner(self, Op = 'inversion', charge='all'):
    """ Obtains the matrices of the action of a symmetry 'Op' on the right Schmidt states:
    
        Op_R |i; R> = U_{i, j} |j; R>
        
    Op is either: 
        - list of single site matrices to be applied to the sites
        - 'inversion' : inverts orbital ordering
        - 'time_reversal': complex conjugation
        - an MPS representing Op psi
    
    Returns:
        Tr(U U^*)/chi   (pm 1 for inv and time reversal)
        U
        q, the charge sector of dominant eigenvalue
    """
    self.convert_to_form('B')
    if Op=='inversion':
        psi_inv = self.copy().inversion() #Form spatial reflection of the WF
    elif Op=='time_reversal':
        psi_inv = self.copy()
        for i in range(psi_inv.L):
            psi_inv.setB(i, psi_inv.getB(i).iconj() )
    elif type(Op)==iMPS:
        psi_inv = Op
    else:
        if type(Op)!=list:
            Op = list(Op)
            
        psi_inv = self.copy()
        for i in range(psi_inv.L):
            psi_inv.setB(i, npc.tensordot(self.get(Op, i), psi_inv.getB(i), axes = [[1], [0]] ) )
            
            
    T = transfer_matrix(psi1 = psi_inv, psi2 = self) #Represents generalized T matrix
    
    
    #T does diagonalization by charge sector, so we must loop over sectors
    if charge =='all':
        q_sectors = T.nonzero_q_sectors()
    else:
        q_sectors = charge
        
    eta_max = 0	#largest eta (eigenvalue) we've found so far
    
    for charge_sector in q_sectors:
        try:
            eta, G = T(charge_sector).eigenvectors(num_ev=1, verbose=False) #Find dominant eigenvector
            print("Q, |eta|", charge_sector, np.abs(eta))
        except:
            print("Sector", charge_sector, "did not converge.")
            
            raise
            #eta = [0.]
            #G = [None]
            
            #If this is getting to be a problem, we can do dense diag as below.
            #m = matvec_to_array(T(charge_sector))
            #eta, G =  np.linalg.eig(m)
            #perm = np.argsort(-np.abs(eta))
            #eta = eta[perm]
            #print eta
            #G = [ T(charge_sector).to_npc_array(G[:, 0]) ]
        
        #eta = eta[0] #eigenvalue
        #G = G[0] #This should be the 'U_I/chi' object of Pollmann-Turner
        
        if np.abs(eta) > np.abs(eta_max): #Or we could just check if it is 1...?
            if np.abs(eta_max - 1.) < 10**(-8):
                print("WARNING: T-mat has degeneracy at eta = 1.")
            U = G[0]
            eta_max = eta
            q_max = charge_sector
            #print G.shape[0]*npc.tensordot(G, G.conj(), [[1], [1]]) #This should be identity
            #print "eta, ph, G:", eta, ph, G
            
    U = U*np.sqrt(U.shape[0])
    print("Using charge sector", q_max, "|eta| = ", np.abs(eta_max))
    print("| U.Ud - 1|", npc.norm(npc.tensordot(U, U.conj(), axes = [[1], [1]]) - npc.eye_like(U)))
    s = self.s[-1]
    print("|Us - sU|", npc.norm(U.scale_axis(s, 0) - U.scale_axis(s, 1)))
    p = npc.inner(U, U.conj(), [[0, 1], [1, 0]])/U.shape[0]
            

    return p, U, q_max