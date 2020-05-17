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
from rotation import get_z_plus, decimals, get_xyz, clean_minus_zero, avoid_special_points

# path toolkits
from pathlib import Path


# region selection
# Here we use the step
a_step = 0.05
b_step = 0.05

a_list_1 = np.arange(1, -a_step, -a_step)
a_list_1 = [clean_minus_zero(a) for a in a_list_1]
a_list_2 = np.arange(-1, a_step, a_step)
a_list_2 = [clean_minus_zero(a) for a in a_list_2]
b_list_1 = np.arange(1, -b_step, -b_step)
b_list_1 = [clean_minus_zero(b) for b in b_list_1]
b_list_2 = np.arange(-1, b_step, b_step)
b_list_2 = [clean_minus_zero(b) for b in b_list_2]


# DMRG Setting
chi = 100
L = 2
max_E_err=1.e-6
max_S_err=1.e-4
N_sweeps_check=5
max_sweeps=100
verbose=1
# folder name
prefix = f'data_L_{L}_comb/'
Path(prefix).mkdir(parents=True, exist_ok=True)
run_save = save_after_run(run_atomic, folder_prefix=prefix)

# # store those points reaching the max sweeps
unclear_points = []

# run the DMRG 

# First region: generated from a_list_1, as the base list
# first of all, run at the line Jx=Jy, i.e. b=0
initial_psi = None
for a in a_list_1:
    a = clean_minus_zero(a)
    b = 0
    Jx, Jy, Jz = get_xyz(a, b)
    J = (Jx, Jy, Jz)
    print("\n\n\n Calculating the (Jx, Jy, Jz) = (%.3f, %.3f, %.3f) ground state" % (Jx, Jy, Jz))
    result = run_save(
        chi=chi, 
        Jx=Jx, 
        Jy=Jy, 
        Jz=Jz, 
        L=L,
        initial_psi=initial_psi,
        max_E_err=max_E_err,
        max_S_err=max_S_err,
        N_sweeps_check=N_sweeps_check,
        max_sweeps=max_sweeps,
        verbose=verbose, 
        )
    
    if result != 0:
        sweeps_stat = result['sweeps_stat']
        last_sweep = len(sweeps_stat['sweep']) * N_sweeps_check
        max_sweeps_no = result['parameters']['max_sweeps']
        if max_sweeps_no == last_sweep:
            unclear_points.append(J)
            print("Maximum sweeps reached!")
            
        initial_psi = result['psi']
    else:
        result = load_data(chi=chi, Jx=Jx, Jy=Jy, Jz=Jz, L=L, prefix=prefix)
        initial_psi = result['psi']

# Then spawn all the other points along the same vertical line, starting from the point we just computed.
for a in a_list_1:
    Jx0, Jy0, Jz0 = get_xyz(a, 0)
    base_result = load_data(chi=chi, Jx=Jx0, Jy=Jy0, Jz=Jz0, L=L, prefix=prefix)
    initial_psi = base_result['psi']
    for b in np.arange(0, clean_minus_zero(a+b_step), b_step):

        print(f"(a, b) = ({a}, {b})")
        Jx, Jy, Jz = get_xyz(a, b)
        J = (Jx, Jy, Jz)
        
        print("\n\n\n Calculating the (Jx, Jy, Jz) = (%.3f, %.3f, %.3f) ground state" % (Jx, Jy, Jz))

        result = run_save(
            chi=chi, 
            Jx=Jx, 
            Jy=Jy, 
            Jz=Jz, 
            L=L,
            initial_psi=initial_psi,
            max_E_err=max_E_err,
            max_S_err=max_S_err,
            N_sweeps_check=N_sweeps_check,
            max_sweeps=max_sweeps,
            verbose=verbose, 
            )

        if result != 0:
            sweeps_stat = result['sweeps_stat']
            last_sweep = len(sweeps_stat['sweep']) * N_sweeps_check
            max_sweeps_no = result['parameters']['max_sweeps']
            if max_sweeps_no == last_sweep:
                unclear_points.append(J)
                print("Maximum sweeps reached!")
                
            initial_psi = result['psi']

for a in a_list_1:
    Jx0, Jy0, Jz0 = get_xyz(a, 0)
    base_result = load_data(chi=chi, Jx=Jx0, Jy=Jy0, Jz=Jz0, L=L, prefix=prefix)
    initial_psi = base_result['psi']
    for b in np.arange(0, clean_minus_zero(-a-b_step), -b_step):

        print(f"(a, b) = ({a}, {b})")

        Jx, Jy, Jz = get_xyz(a, b)
        J = (Jx, Jy, Jz)
        
        print("\n\n\n Calculating the (Jx, Jy, Jz) = (%.3f, %.3f, %.3f) ground state" % (Jx, Jy, Jz))

        result = run_save(
            chi=chi, 
            Jx=Jx, 
            Jy=Jy, 
            Jz=Jz, 
            L=L,
            initial_psi=initial_psi,
            max_E_err=max_E_err,
            max_S_err=max_S_err,
            N_sweeps_check=N_sweeps_check,
            max_sweeps=max_sweeps,
            verbose=verbose, 
            )

        if result != 0:
            sweeps_stat = result['sweeps_stat']
            last_sweep = len(sweeps_stat['sweep']) * N_sweeps_check
            max_sweeps_no = result['parameters']['max_sweeps']
            if max_sweeps_no == last_sweep:
                unclear_points.append(J)
                print("Maximum sweeps reached!")
                
            initial_psi = result['psi']

# Second region: generated from a_list_1, as the base list
# first of all, run at the line Jx=Jy, i.e. b=0
initial_psi = None
for a in a_list_2:
    a = clean_minus_zero(a)
    b = 0
    Jx, Jy, Jz = get_xyz(a, b)
    J = (Jx, Jy, Jz)
    print("\n\n\n Calculating the (Jx, Jy, Jz) = (%.3f, %.3f, %.3f) ground state" % (Jx, Jy, Jz))
    result = run_save(
        chi=chi, 
        Jx=Jx, 
        Jy=Jy, 
        Jz=Jz, 
        L=L,
        initial_psi=initial_psi,
        max_E_err=max_E_err,
        max_S_err=max_S_err,
        N_sweeps_check=N_sweeps_check,
        max_sweeps=max_sweeps,
        verbose=verbose, 
        )
    
    if result != 0:
        sweeps_stat = result['sweeps_stat']
        last_sweep = len(sweeps_stat['sweep']) * N_sweeps_check
        max_sweeps_no = result['parameters']['max_sweeps']
        if max_sweeps_no == last_sweep:
            unclear_points.append(J)
            print("Maximum sweeps reached!")
            
        initial_psi = result['psi']
    else:
        result = load_data(chi=chi, Jx=Jx, Jy=Jy, Jz=Jz, L=L, prefix=prefix)
        initial_psi = result['psi']

# Then spawn all the other points along the same vertical line, starting from the point we just computed.
for a in a_list_2:
    Jx0, Jy0, Jz0 = get_xyz(a, 0)
    base_result = load_data(chi=chi, Jx=Jx0, Jy=Jy0, Jz=Jz0, L=L, prefix=prefix)
    initial_psi = base_result['psi']
    for b in np.arange(0, clean_minus_zero(-a+b_step), b_step):

        print(f"(a, b) = ({a}, {b})")
        Jx, Jy, Jz = get_xyz(a, b)
        J = (Jx, Jy, Jz)
        
        print("\n\n\n Calculating the (Jx, Jy, Jz) = (%.3f, %.3f, %.3f) ground state" % (Jx, Jy, Jz))

        result = run_save(
            chi=chi, 
            Jx=Jx, 
            Jy=Jy, 
            Jz=Jz, 
            L=L,
            initial_psi=initial_psi,
            max_E_err=max_E_err,
            max_S_err=max_S_err,
            N_sweeps_check=N_sweeps_check,
            max_sweeps=max_sweeps,
            verbose=verbose, 
            )

        if result != 0:
            sweeps_stat = result['sweeps_stat']
            last_sweep = len(sweeps_stat['sweep']) * N_sweeps_check
            max_sweeps_no = result['parameters']['max_sweeps']
            if max_sweeps_no == last_sweep:
                unclear_points.append(J)
                print("Maximum sweeps reached!")
                
            initial_psi = result['psi']

for a in a_list_2:
    Jx0, Jy0, Jz0 = get_xyz(a, 0)
    base_result = load_data(chi=chi, Jx=Jx0, Jy=Jy0, Jz=Jz0, L=L, prefix=prefix)
    initial_psi = base_result['psi']
    for b in np.arange(0, clean_minus_zero(a-b_step), -b_step):

        print(f"(a, b) = ({a}, {b})")

        Jx, Jy, Jz = get_xyz(a, b)
        J = (Jx, Jy, Jz)
        
        print("\n\n\n Calculating the (Jx, Jy, Jz) = (%.3f, %.3f, %.3f) ground state" % (Jx, Jy, Jz))

        result = run_save(
            chi=chi, 
            Jx=Jx, 
            Jy=Jy, 
            Jz=Jz, 
            L=L,
            initial_psi=initial_psi,
            max_E_err=max_E_err,
            max_S_err=max_S_err,
            N_sweeps_check=N_sweeps_check,
            max_sweeps=max_sweeps,
            verbose=verbose, 
            )

        if result != 0:
            sweeps_stat = result['sweeps_stat']
            last_sweep = len(sweeps_stat['sweep']) * N_sweeps_check
            max_sweeps_no = result['parameters']['max_sweeps']
            if max_sweeps_no == last_sweep:
                unclear_points.append(J)
                print("Maximum sweeps reached!")
                
            initial_psi = result['psi']

# Third region: generated from a_list_1, as the base list
# first of all, run at the line Jx=Jy, i.e. b=0
initial_psi = None
for b in b_list_1:
    b = clean_minus_zero(b)
    a = 0
    Jx, Jy, Jz = get_xyz(a, b)
    J = (Jx, Jy, Jz)
    print("\n\n\n Calculating the (Jx, Jy, Jz) = (%.3f, %.3f, %.3f) ground state" % (Jx, Jy, Jz))
    result = run_save(
        chi=chi, 
        Jx=Jx, 
        Jy=Jy, 
        Jz=Jz, 
        L=L,
        initial_psi=initial_psi,
        max_E_err=max_E_err,
        max_S_err=max_S_err,
        N_sweeps_check=N_sweeps_check,
        max_sweeps=max_sweeps,
        verbose=verbose, 
        )
    
    if result != 0:
        sweeps_stat = result['sweeps_stat']
        last_sweep = len(sweeps_stat['sweep']) * N_sweeps_check
        max_sweeps_no = result['parameters']['max_sweeps']
        if max_sweeps_no == last_sweep:
            unclear_points.append(J)
            print("Maximum sweeps reached!")
            
        initial_psi = result['psi']
    else:
        result = load_data(chi=chi, Jx=Jx, Jy=Jy, Jz=Jz, L=L, prefix=prefix)
        initial_psi = result['psi']

# Then spawn all the other points along the same vertical line, starting from the point we just computed.
for b in b_list_1:
    Jx0, Jy0, Jz0 = get_xyz(0, b)
    base_result = load_data(chi=chi, Jx=Jx0, Jy=Jy0, Jz=Jz0, L=L, prefix=prefix)
    initial_psi = base_result['psi']
    for a in np.arange(0, clean_minus_zero(b+a_step), a_step):

        print(f"(a, b) = ({a}, {b})")
        Jx, Jy, Jz = get_xyz(a, b)
        J = (Jx, Jy, Jz)
        
        print("\n\n\n Calculating the (Jx, Jy, Jz) = (%.3f, %.3f, %.3f) ground state" % (Jx, Jy, Jz))

        result = run_save(
            chi=chi, 
            Jx=Jx, 
            Jy=Jy, 
            Jz=Jz, 
            L=L,
            initial_psi=initial_psi,
            max_E_err=max_E_err,
            max_S_err=max_S_err,
            N_sweeps_check=N_sweeps_check,
            max_sweeps=max_sweeps,
            verbose=verbose, 
            )

        if result != 0:
            sweeps_stat = result['sweeps_stat']
            last_sweep = len(sweeps_stat['sweep']) * N_sweeps_check
            max_sweeps_no = result['parameters']['max_sweeps']
            if max_sweeps_no == last_sweep:
                unclear_points.append(J)
                print("Maximum sweeps reached!")
                
            initial_psi = result['psi']

for b in b_list_1:
    Jx0, Jy0, Jz0 = get_xyz(0, b)
    base_result = load_data(chi=chi, Jx=Jx0, Jy=Jy0, Jz=Jz0, L=L, prefix=prefix)
    initial_psi = base_result['psi']
    for a in np.arange(0, clean_minus_zero(-b-a_step), -a_step):

        print(f"(a, b) = ({a}, {b})")

        Jx, Jy, Jz = get_xyz(a, b)
        J = (Jx, Jy, Jz)
        
        print("\n\n\n Calculating the (Jx, Jy, Jz) = (%.3f, %.3f, %.3f) ground state" % (Jx, Jy, Jz))

        result = run_save(
            chi=chi, 
            Jx=Jx, 
            Jy=Jy, 
            Jz=Jz, 
            L=L,
            initial_psi=initial_psi,
            max_E_err=max_E_err,
            max_S_err=max_S_err,
            N_sweeps_check=N_sweeps_check,
            max_sweeps=max_sweeps,
            verbose=verbose, 
            )

        if result != 0:
            sweeps_stat = result['sweeps_stat']
            last_sweep = len(sweeps_stat['sweep']) * N_sweeps_check
            max_sweeps_no = result['parameters']['max_sweeps']
            if max_sweeps_no == last_sweep:
                unclear_points.append(J)
                print("Maximum sweeps reached!")
                
            initial_psi = result['psi']

# Fourth region: generated from a_list_1, as the base list
# first of all, run at the line Jx=Jy, i.e. b=0
initial_psi = None
for b in b_list_2:
    b = clean_minus_zero(b)
    a = 0
    Jx, Jy, Jz = get_xyz(a, b)
    J = (Jx, Jy, Jz)
    print("\n\n\n Calculating the (Jx, Jy, Jz) = (%.3f, %.3f, %.3f) ground state" % (Jx, Jy, Jz))
    result = run_save(
        chi=chi, 
        Jx=Jx, 
        Jy=Jy, 
        Jz=Jz, 
        L=L,
        initial_psi=initial_psi,
        max_E_err=max_E_err,
        max_S_err=max_S_err,
        N_sweeps_check=N_sweeps_check,
        max_sweeps=max_sweeps,
        verbose=verbose, 
        )
    
    if result != 0:
        sweeps_stat = result['sweeps_stat']
        last_sweep = len(sweeps_stat['sweep']) * N_sweeps_check
        max_sweeps_no = result['parameters']['max_sweeps']
        if max_sweeps_no == last_sweep:
            unclear_points.append(J)
            print("Maximum sweeps reached!")
            
        initial_psi = result['psi']
    else:
        result = load_data(chi=chi, Jx=Jx, Jy=Jy, Jz=Jz, L=L, prefix=prefix)
        initial_psi = result['psi']

# Then spawn all the other points along the same vertical line, starting from the point we just computed.
for b in b_list_2:
    Jx0, Jy0, Jz0 = get_xyz(0, b)
    base_result = load_data(chi=chi, Jx=Jx0, Jy=Jy0, Jz=Jz0, L=L, prefix=prefix)
    initial_psi = base_result['psi']
    for a in np.arange(0, clean_minus_zero(-b+a_step), a_step):

        print(f"(a, b) = ({a}, {b})")
        Jx, Jy, Jz = get_xyz(a, b)
        J = (Jx, Jy, Jz)
        
        print("\n\n\n Calculating the (Jx, Jy, Jz) = (%.3f, %.3f, %.3f) ground state" % (Jx, Jy, Jz))

        result = run_save(
            chi=chi, 
            Jx=Jx, 
            Jy=Jy, 
            Jz=Jz, 
            L=L,
            initial_psi=initial_psi,
            max_E_err=max_E_err,
            max_S_err=max_S_err,
            N_sweeps_check=N_sweeps_check,
            max_sweeps=max_sweeps,
            verbose=verbose, 
            )

        if result != 0:
            sweeps_stat = result['sweeps_stat']
            last_sweep = len(sweeps_stat['sweep']) * N_sweeps_check
            max_sweeps_no = result['parameters']['max_sweeps']
            if max_sweeps_no == last_sweep:
                unclear_points.append(J)
                print("Maximum sweeps reached!")
                
            initial_psi = result['psi']

for b in b_list_2:
    Jx0, Jy0, Jz0 = get_xyz(0, b)
    base_result = load_data(chi=chi, Jx=Jx0, Jy=Jy0, Jz=Jz0, L=L, prefix=prefix)
    initial_psi = base_result['psi']
    for a in np.arange(0, clean_minus_zero(b-a_step), -a_step):

        print(f"(a, b) = ({a}, {b})")

        Jx, Jy, Jz = get_xyz(a, b)
        J = (Jx, Jy, Jz)
        
        print("\n\n\n Calculating the (Jx, Jy, Jz) = (%.3f, %.3f, %.3f) ground state" % (Jx, Jy, Jz))

        result = run_save(
            chi=chi, 
            Jx=Jx, 
            Jy=Jy, 
            Jz=Jz, 
            L=L,
            initial_psi=initial_psi,
            max_E_err=max_E_err,
            max_S_err=max_S_err,
            N_sweeps_check=N_sweeps_check,
            max_sweeps=max_sweeps,
            verbose=verbose, 
            )

        if result != 0:
            sweeps_stat = result['sweeps_stat']
            last_sweep = len(sweeps_stat['sweep']) * N_sweeps_check
            max_sweeps_no = result['parameters']['max_sweeps']
            if max_sweeps_no == last_sweep:
                unclear_points.append(J)
                print("Maximum sweeps reached!")
                
            initial_psi = result['psi']

# run_over(a_list_1, b_list_1[::-1], b_list_2[::-1])
# run_over(a_list_2, b_list_1[::-1], b_list_2[::-1])
# run_over(b_list_1, a_list_1[::-1], a_list_2[::-1])
# run_over(b_list_2, a_list_1[::-1], a_list_2[::-1])
        