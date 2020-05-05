""" In this script we will read out the psi for each point from the .h5 fiels in /data folder
"""

# numpy
import numpy as np
from numpy import pi

# matplot
import matplotlib.pyplot as plt

# model and dmrg
from kitaev_ladder import KitaevLadderModel, save_after_run, load_data, run_atomic

# triangular coordinates
from triangular_coordinates import triangular_to_decarte, decimals

# path toolkits
from pathlib import Path

# region selection
# Here we use the step
a_step = 0.1
b_step = 0.1

point_list = []

for a in np.arange(0, 1+a_step, a_step):
    for b in np.arange(a-1, 1-a+b_step, b_step):
        point_list.append(triangular_to_decarte(1, a, b))


chi = 100
L = 2
N_sweeps_check=1
max_sweeps=20
verbose=0
# folder name
prefix = f'data_L_{L}/'
Path(prefix).mkdir(parents=True, exist_ok=True)
run_save = save_after_run(run_atomic, folder_prefix=prefix)

# def get_J(theta, phi):
#     Jx, Jy, Jz = triangular_to_decarte(r=1.0, theta=theta, phi=phi)
#     if Jx == -0.0:
#         Jx = 0.0
#     if Jy == -0.0:
#         Jy = 0.0
#     if Jz == -0.0:
#         Jz = 0.0
#     # if Jx = Jy then slightly differ them to avoid the symmetric state
#     if Jx == Jy:
#         Jx += 0.001
#         Jy -= 0.001
#     return (Jx, Jy, Jz)

# # store those points reaching the max sweeps
unclear_points = []

# run the DMRG 
for a in np.arange(0, 1+a_step, a_step):
    psi = None
    for b in np.arange(a-1, 1-a+b_step, b_step):
        if psi is not None:
            initial_psi = psi.copy()
        else:
            initial_psi = None

        Jx, Jy, Jz = triangular_to_decarte(1, a, b)
        J = (Jx, Jy, Jz)
        
        print("\n\n\n Calculating the (Jx, Jy, Jz) = (%.3f, %.3f, %.3f) ground state" % (Jx, Jy, Jz))

        result = run_save(
            chi=chi, 
            Jx=Jx, 
            Jy=Jy, 
            Jz=Jz, 
            L=L,
            initial_psi=initial_psi,
            N_sweeps_check=N_sweeps_check,
            max_sweeps=max_sweeps,
            verbose=0, 
            )

        if result != 0:
            sweeps_stat = result['sweeps_stat']
            last_sweep = len(sweeps_stat['sweep']) * N_sweeps_check
            max_sweeps = result['parameters']['max_sweeps']
            if max_sweeps == last_sweep:
                unclear_points.append(J)
            else:
                initial_psi = result['psi']

        

print("Finished.\n", "Unclear points: ", unclear_points)
        