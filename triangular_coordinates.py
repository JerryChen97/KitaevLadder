# This is not a real coordinate system, but the local system on the positive triangular x+y+z=1

import numpy as np

# control the precision
decimals=3

def decarte_to_triangular(x, y, z):
    NotImplementedError("This method has not been implemented!!!")
    return np.round((r, a, b), decimals=decimals)

def triangular_to_decarte(r, a, b):
    x = 0.5 - 0.5 * a + 0.5 * b
    y = 0.5 - 0.5 * a - 0.5 * b
    z = a
    x = np.round(x*r, decimals=decimals)
    if x == -0.0:
        x = 0.0
    y = np.round(y*r, decimals=decimals)
    if y == -0.0:
        y = 0.0
    z = np.round(z*r, decimals=decimals)
    if z == -0.0:
        z = 0.0

    return (x, y, z)
