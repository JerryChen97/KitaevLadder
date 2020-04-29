import numpy as np
from kitaev_ladder import KitaevLadderModel, run_save, read_psi

# region selection
Jx_list = [1.]
Jx_list = np.round(Jx_list, decimals=3)

Jy_list = np.arange(-2, 2, .2)
Jy_list = np.round(Jy_list, decimals=3)

Jz_list = np.arange(-2, 2, .2)
Jz_list = np.round(Jz_list, decimals=3)
chi_list = [30]
J_list = [(Jx, Jy, Jz, chi) for Jx in Jx_list for Jy in Jy_list for Jz in Jz_list for chi in chi_list]

# prepare other arguments
verbose = 0

if __name__ == "__main__":
    for Jx in Jx_list:
        for Jy in Jy_list:
            for Jz in Jz_list:
                for chi in chi_list:
                    run_save(chi=chi, Jx=Jx, Jy=Jy, Jz=Jz, verbose=verbose)
