import sys
sys.path += ["/home/ychen2/projects/rrg-yche/ychen2/KitaevLadder"]

import numpy as np
from kitaev_ladder_snake1 import run_atomic, save_after_run, load_data

# the folder name for storing data
folder_prefix = 'snake1/'

# generate the function for running and storing the data in the folder we want
run_save = save_after_run(run_atomic, folder_prefix=folder_prefix)

# get the number of this program to determine which J to start with
number = int(sys.argv[1])
assert number > 0 and number <= 200

S = 1

chi = 128

L = 4

J_list = np.linspace(2.0, 0, 201)
# use the number calculated above to select the correct J. Note that the number starts with 1 and ends with 200, and the first number indicates 2.0 of course.
J = J_list[number-1]

# and then this J value serves as the start point of this sweep
Jx = J 
Jy = J
Jz = 1.0


# initial_psi = load_data(Jx=Jx,Jy=Jy,Jz=Jz,L=L,chi=chi,prefix=folder_prefix)

# fixing the value of Jx, ranging Jy from J to 0
N = 201 - number
Jy_list = np.linspace(J-0.01, 0, N)

for Jy in Jy_list:
    print(f'J = ({Jx}, {Jy}, {Jz})')
    # result = run_save(Jx=Jx, Jy=Jy, Jz=Jz, L=L, chi=chi, initial_psi=initial_psi)
    # if result==0: # this file already exists
    #     result = load_data(Jx=Jx,Jy=Jy,Jz=Jz,L=L,chi=chi,prefix=folder_prefix)
    #     pass 
    # initial_psi=result['psi'].copy()

