
from openmdao.api import ExplicitComponent

import numpy as np

class DComp(ExplicitComponent):
    def initialize(self):
        self.options.declare('C', np.ndarray)
        self.options.declare('problem_type', str)
    def setup(self):
        self.add_output('D', shape =(3,3))
        
    def compute(self, inputs, outputs):
        C = self.options['C']
        problem_type = self.options['problem_type']
        if problem_type == 'plane_stress':
            D = np.zeros((3,3))
            D[0][0] = C[0][0] - C[0][2] ** 2 / C[2][2]
            D[1][1] = C[1][1] - C[1][2] ** 2 / C[2][2]
            D[0][1] = C[0][1] - C[0][2] * C[1][2] / C[2][2]
            D[1][0] = C[0][1] - C[0][2] * C[1][2] / C[2][2]
            D[2][2] = C[3][3]

        # no sigma_3
        if problem_type == 'plane_strain':
            D = np.zeros((3,3))
            D[0][0] = C[0][0] 
            D[1][1] = C[1][1]
            D[0][1] = C[0][1] 
            D[1][0] = C[0][1] 
            D[2][2] = C[3][3]

        if problem_type == 'truss':
            D = C[0][0]

        outputs['D'] = D
