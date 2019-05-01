
import numpy as np


def Gauss_Points(ng):
    G = np.zeros((ng,1))
    W = np.zeros((ng,1))
    if ng == 2:
        G[0] = -.5773502691896257
        G[1] = -1 * G[0]
        W[0] = 1.0
        W[1] = 1.0

    if ng == 3:
        G[0] = -0.7745966692414834
        G[1] = 0.0
        G[2] = -1 * G[0]
        W[0] = 0.5555555555555556
        W[1] = 0.8888888888888888
        W[0] = 0.5555555555555556

    return G, W
