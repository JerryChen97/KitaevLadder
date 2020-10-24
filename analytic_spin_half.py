import numpy as np
from scipy import integrate
from scipy.integrate import quad

def energy(k, Jx=1, Jy=1, Jz=1, D=1):
    assert D == -1 or D == 1
    cos_term = Jz + (Jx + D * Jy) * np.cos(k)
    sin_term = (Jx - D * Jy) * np.sin(k)
    return np.sqrt(cos_term**2 + sin_term**2)

def energy_true(k, Jx=1, Jy=1, Jz=1):
    D = 1 if Jx*Jy <= 0 else -1
    cos_term = Jz + (Jx + D * Jy) * np.cos(k)
    sin_term = (Jx - D * Jy) * np.sin(k)
    return np.sqrt(cos_term**2 + sin_term**2)

def sampling_momentum(bc='periodic', N=100):
    if bc == 'periodic':
        return [np.pi * (2*m + 1) / (2 * N) for m in np.arange(-N, N)]
    elif bc == 'antiperiodic':
        return [np.pi * (2*m) / (2 * N) for m in np.arange(-N, N)]
    else:
        print("Wrong boundary condition!!!!")
        raise ValueError

def energy_sum(Jx=1, Jy=1, Jz=1, bc='periodic', N=100):
    return np.sum([energy_true(k, Jx=Jx, Jy=Jy, Jz=Jz) for k in sampling_momentum(bc=bc, N=N)])

def energy_average(Jx=1, Jy=1, Jz=1, bc='periodic', N=100):
    return energy_sum(Jx=Jx, Jy=Jy, Jz=Jz, bc=bc, N=N) / N

def energy_average_true(Jx, Jy, Jz):
    return quad(lambda x:(energy_true(Jx=Jx, Jy=Jy, Jz=Jz, k=x) / (2 * np.pi)), 0, np.pi)

def energy_difference(Jx=1, Jy=1, Jz=1, N=100):
    return energy_sum(Jx=Jx, Jy=Jy, Jz=Jz, bc='periodic', N=N)-energy_sum(Jx=Jx, Jy=Jy, Jz=Jz, bc='antiperiodic', N=N)
