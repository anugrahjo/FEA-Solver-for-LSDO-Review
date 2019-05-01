
import numpy as np

def Shape_Function(nn, s):

    N = np.zeros(nn)

    if nn == 2:
        N[0] = .5*(1-s)
        N[1] = .5*(1+s)

    if nn == 3:
        N[0] = .5 * (s**2 - 1)
        N[1] = 1 - s**2
        N[2] = .5 * (s**2 + 1)

    return N

def Diff_Shape_Function(nn, s):
    dN = np.zeros(nn)

    if nn == 2:
        dN[0] = -.5
        dN[1] = .5

    if nn == 3:
        dN[0] = s
        dN[1] = -2 * s
        dN[2] = s

    return dN
