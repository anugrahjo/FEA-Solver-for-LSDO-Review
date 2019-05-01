
from element import Element
from shape_functions import Shape_Function, Diff_Shape_Function
from gauss_points import Gauss_Points

import numpy as np


class RectangularElement(Element):

    def setup(self):
        self.nn = 4
        self.nn_per_dim = int(self.nn/2)
        self.ndof = 2
        self.edof = 8
        self.ng = 2
        self.element_type = 2
        self.coord_position = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

    def shape_function_value(self):
        nn = self.nn
        ng = self.ng
        G, W = Gauss_Points(ng)
        N_value = np.zeros((ng ** 2, 1, nn))

        xi = np.array([G[0], G[1], G[1], G[0]])
        eta = np.array([G[0], G[0], G[1], G[1]])
        # w_xi = np.array([W[0], W[1], W[0], W[1]])
        # w_eta = np.array([W[0], W[0], W[1], W[1]])

        for i in range(ng ** 2):
            for j in range(nn):
                N_xi_j = Shape_Function(self.nn_per_dim, xi[i])[self.coord_position[j][0]]
                N_eta_j = Shape_Function(self.nn_per_dim, eta[i])[self.coord_position[j][1]]
                N_value[i][0][j] = N_xi_j * N_eta_j

        return N_value

    def shape_function_partial(self):
        nn = self.nn
        ng = self.ng
        G, W = Gauss_Points(ng)
        pN_value = np.zeros((ng ** 2, 2, nn))
        xi = np.array([G[0], G[1], G[1], G[0]])
        eta = np.array([G[0], G[0], G[1], G[1]])
        # w_xi = np.array([W[0], W[1], W[0], W[1]])
        # w_eta = np.array([W[0], W[0], W[1], W[1]])

        for i in range(ng ** 2):
            for j in range(nn):
                N_xi_j = Shape_Function(self.nn_per_dim, xi[i])[self.coord_position[j][0]]
                N_eta_j = Shape_Function(self.nn_per_dim, eta[i])[self.coord_position[j][1]]
                pN_xi_j = Diff_Shape_Function(self.nn_per_dim, xi[i])[self.coord_position[j][0]]
                pN_eta_j = Diff_Shape_Function(self.nn_per_dim, eta[i])[self.coord_position[j][1]]
                pN_value[i][0][j] = pN_xi_j * N_eta_j
                pN_value[i][1][j] = pN_eta_j * N_xi_j

        return pN_value

    def gaussian_weights(self):
        ng = self.ng
        G, W = Gauss_Points(ng)
        w_xi = np.array([W[0], W[1], W[0], W[1]])
        w_eta = np.array([W[0], W[0], W[1], W[1]])
        W = np.multiply(w_xi, w_eta)
        W = W.reshape(ng**2)

        return W



# rec = Rectangular_Element()
# print(rec.shape_function_partial())

# ------------------
# for value check

# xi = -0.57735
# eta = -0.57735
# N_xi_1 = 1/2*(1-xi)
# N_xi_2 = 1/2*(1+xi)
# N_eta_1 = 1/2*(1-eta)
# N_eta_2 = 1/2*(1+eta)
# N_i = [N_xi_1 * N_eta_1, N_xi_2 * N_eta_1, N_xi_2*N_eta_2, N_xi_1 * N_eta_2]
# pN_value_i = [[-1/4 * (1 - eta), 1/4 * (1 - eta), 1/4 * (1 + eta), -1/4 * (1 + eta)], [-1/4 * (1 - xi), -1/4 * (1 + xi), 1/4 * (1 + xi), 1/4 * (1 - xi)]]
#
# print(pN_value_i)
