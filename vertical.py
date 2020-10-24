import sys
# sys.path += ["/home/ychen2/projects/rrg-yche/ychen2/KitaevLadder"]

import numpy as np
from kitaev_ladder_snake1 import run_atomic, save_after_run, load_data
from utility import linspace

chi = 64

L = 2
# the folder name for storing data
folder_prefix = f'Spin_half_snake1_L_{L}_chi_{chi}_high_res/' # high resolution means the smallest step is 0.01

# generate the function for running and storing the data in the folder we want
run_save = save_after_run(run_atomic, folder_prefix=folder_prefix)

# get the number of this program to determine which J to start with
# number = int(sys.argv[1])
# assert number > 0 and number <= 200

J_list = linspace(2.0, 0, 201)
# use the number calculated above to select the correct J. Note that the number starts with 1 and ends with 200, and the first number indicates 2.0 of course.

# and then this J value serves as the start point of this sweep
Jz = 1.0

# fixing the value of Jx, ranging Jy from J to 0
for number in range(1, 200): # note that the last vertical line should be trivial since there will be only one pixel over there, i.e. Jx=Jy=0,Jz=1
    N = 201 - number # the number of pixels

    J = J_list[number-1] # the value of Jx will be fixed here and for loop body in this time
    Jy_list = linspace(J-0.01, 0, N)

    # the first initial wave function should be located on the diagonal line
    initial_result = load_data(Jx=J,Jy=J,Jz=Jz,L=L,chi=chi,prefix=folder_prefix)

    initial_psi = initial_result['psi']

    Jx = J
    for Jy in Jy_list:
        # print(f'J = ({Jx}, {Jy}, {Jz})')
        result = run_save(Jx=Jx, Jy=Jy, Jz=Jz, L=L, chi=chi, initial_psi=initial_psi, verbose=0)
        if result==0: # this file already exists
            result = load_data(Jx=Jx,Jy=Jy,Jz=Jz,L=L,chi=chi,prefix=folder_prefix)
            pass 
        initial_psi=result['psi'].copy()
