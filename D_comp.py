
from openmdao.api import ExplicitComponent

import numpy as np

class DComp(ExplicitComponent):

    def setup(self):
        self.add_input('C')
        self.add_input('problem_type')
        self.add_output('D')
        self.declare_partials('D', '*')
        
    def compute(self, inputs, outputs):
        C = inputs['C']
        problem_type = inputs['problem_type']
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

    def compute_partials(self, inputs, partials):
        C = inputs['C']

        partials['D', 'C'] = 0