import numpy as np
from kitaev_ladder_snake1 import run_atomic, save_after_run, load_data

# the folder name for storing data
folder_prefix = 'snake1/'

# generate the function for running and storing the data in the folder we want
run_save = save_after_run(run_atomic, folder_prefix=folder_prefix)

S = 1
initial_psi = None

Jx = 2.0
Jy = 2.0
Jz = 1.0

chi = 128

L = 4

J_list = np.linspace(2.0, 0, 201)

for J in J_list:
    Jx = J
    Jy = J 
    result = run_save(Jx=Jx, Jy=Jy, Jz=Jz, L=L, chi=chi, initial_psi=initial_psi)
    if result==0: # this file already exists
        result = load_data(Jx=Jx,Jy=Jy,Jz=Jz,L=L,chi=chi,prefix=folder_prefix)
        pass 
    initial_psi=result['psi'].copy()
