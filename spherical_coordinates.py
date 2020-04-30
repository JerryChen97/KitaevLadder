# In this script we provide a convenient toolkit for converting between Descartes coordinates and spherical coordinates
import numpy as np 

# this global variable controls the truncation digits number
decimals = 3

def decarte_to_spherical(x, y, z):
    if x == 0 and y == 0 and z == 0:
        r = 0
        theta = 0
        phi = 0
    else:
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(float(z) / r)
        phi = np.arctan2(float(y), float(x))

    return np.round((r, theta, phi), decimals=decimals)

def spherical_to_decarte(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.round((x, y, z), decimals=decimals)

def test():
    x = np.random.rand()
    y = np.random.rand()
    z = np.random.rand()
    r, theta, phi = decarte_to_spherical(x, y, z)
    x2, y2, z2 = spherical_to_decarte(r, theta, phi)
    print(x, y, z)
    print(x2, y2, z2)

# test()