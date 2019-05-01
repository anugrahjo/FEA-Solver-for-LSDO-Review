
from element import Element
from shape_functions import Shape_Function, Diff_Shape_Function
from gauss_points import Gauss_Points

import numpy as np

class TrussElement(Element):
    
    def setup(self):
        self.nn = 2
        self.ndof = 2
        self.edof = 4
        self.ng = 2
        self.element_type = 1
    
    def shape_function_value(self):
        nn = self.nn
        ng = self.ng
        G, W = Gauss_Points(ng)
        N_value = np.zeros((ng, 1, nn))
        for i in range(ng):
            N_value[i] = Shape_Function(2, G[i])

        return N_value

    def shape_function_partial(self):
        nn = self.nn
        ng = self.ng
        G, W = Gauss_Points(ng)
        pN_value = np.zeros((ng, 1, nn))
        for i in range(ng):
            pN_value[i] = Diff_Shape_Function(nn, G[i])
        return pN_value



