from kitaev_ladder_snake1 import run_atomic, save_after_run
import numpy as np

# KL: Kitaev Ladder
def extract_phase_diagram_KL(
    S=1, 
    J_min=0,
    J_max=2,
    J_d=0.01,
    ):
    # check the range
    n_pts = int((J_max-J_min) / (J_d)) + 1
    # print(n_pts)
    ###
    #   We want the sweep to start from the end with bigger J
    ###
    J_list = np.linspace(J_max, J_min, n_pts)
    # print(J_list)

    ###
    #   Initial sweep: generate the Jx=Jy line
    ###
    for J in J_list:
        Jx = J
        Jy = J

        


    pass

extract_phase_diagram_KL()
