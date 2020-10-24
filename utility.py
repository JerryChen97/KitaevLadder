import numpy as np

def linspace(start, end, number, prec=5):
    raw = np.linspace(start, end, number)
    return [np.round(v, prec) for v in raw]

def arange(start, end, step, prec=5):
    raw = np.arange(start, end, step)
    return [np.round(v, prec) for v in raw]

def test_print():
    for i in linspace(0, 1, 101):
        print(i)

# test_print()
