""" In this script we will read out the psi for each point from the .h5 fiels in /data folder
"""

import numpy as np
from numpy import pi
from kitaev_ladder import KitaevLadderModel, run_save, read_psi

from spherical_coordinates import spherical_to_decarte

# region selection
# first we decide how many points we want 
N = 20
# the number of theta points should be N+1 considering the closed end point
Ntheta = N + 1
# the number of phi points should be 2N+1
Nphi = 2 * N + 1

# use linspace to conveniently create the desired number of points
# the default linspace provides intervals with both closed boundaries
theta_list = np.linspace(0, pi/2, Ntheta)
phi_list = np.linspace(0, pi, Nphi)

for theta in theta_list:
    for phi in phi_list:
        Jx, Jy, Jz = spherical_to_decarte(r=1.0, theta=theta, phi=phi)
        if Jx == -0.0:
            Jx = 0.0
        if Jy == -0.0:
            Jy = 0.0
        if Jz == -0.0:
            Jz = 0.0
        run_save(Jx=Jx, Jy=Jy, Jz=Jz, verbose=0)
        
