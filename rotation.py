# in this file we will implement the rotation for (Jx, Jy) -> (a, b)

import numpy as np


def clean_minus_zero(x):
    x = np.round(x, decimals=decimals)
    if x == -0.0:
        x = 0.0
        pass
    return x 
# control the precision
decimals=3

def get_ab(x, y):
    NotImplementedError("This method has not been implemented!!!")
    return np.round((x, y), decimals=decimals)

def get_xy(a, b):
    x = 0.5 * a + 0.5 * b
    y = 0.5 * a - 0.5 * b
    x = np.round(x, decimals=decimals)
    if x == -0.0:
        x = 0.0
    y = np.round(y, decimals=decimals)
    if y == -0.0:
        y = 0.0
    return (x, y)

def get_z_plus(x, y):
    return clean_minus_zero(1 - np.abs(x) - np.abs(y))

def get_z_minus(x, y):
    return -get_z_plus(x, y)

def get_xyz(a, b):
    x, y = get_xy(a, b)
    z = get_z_plus(x, y)
    return (x, y, z)
