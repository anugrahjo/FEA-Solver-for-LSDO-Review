
from openmdao.api import ExplicitComponent
from mesh import Mesh
from element import Element

import numpy as np
from truss_element import TrussElement
from rectangular_plate import RectangularElement

ele = RectangularElement()


class BComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('ng', types=int)
        self.options.declare('NDIM', types=int)         # defined in the problem, constant for all elements.
        self.options.declare('max_nn', types=int)
        self.options.declare('NEL', types=int)
        self.options.declare('max_edof', types=int)         
        self.options.declare('problem_type', types=str)
        self.options.declare('pN', types=np.ndarray)

    def setup(self):
        ng = self.options['ng']
        NDIM = self.options['NDIM']
        NEL = self.options['NEL']
        max_edof = self.options['max_edof']
        problem_type = self.options['problem_type']
        if problem_type == 'plane_stress' or 'plane_strain':
            n_D = 3
        if problem_type == 'truss':
            n_D = 1
        self.add_input('J', shape=(NEL, ng**2, NDIM, NDIM))
        self.add_output('B', shape=(NEL, ng**2, n_D, max_edof))
        self.declare_partials('B', '*', method = 'cs')


    def compute(self, inputs, outputs):
        ng = self.options['ng']
        NDIM = self.options['NDIM']
        max_nn = self.options['max_nn']
        NEL = self.options['NEL']
        max_edof = self.options['max_edof']
        problem_type = self.options['problem_type']
        pN = self.options['pN']

        J = inputs['J']

        if problem_type == 'plane_stress' or 'plane_strain':
            B = np.zeros((NEL, ng**2, 3, max_edof))
            for i in range(NEL):
                pN[i] = ele.shape_function_partial()
                for j in range(ng**2):
                    J[i][j] = np.identity(NDIM)
                    pN_ele_global = np.dot(np.linalg.inv(J[i][j]), pN[i][j])
                    for k in range(max_nn):
                        B[i][j][0][2*k] = pN[i][j][0][k]
                        B[i][j][1][2*k+1] = pN[i][j][1][k]
                        B[i][j][2][2*k] = pN[i][j][1][k]
                        B[i][j][2][2*k+1] = pN[i][j][0][k]



        if problem_type == 'truss':
            B = np.zeros((NEL, ng, 1, max_edof))
            for i in range(NEL):
                for j in range(ng):
                    B[i][j][0][0:2] = [-1/2, 1/2]

        outputs['B'] = B


    # def compute_partials(self, inputs, partials):
    #     pass


if __name__ == '__main__':
    from openmdao.api import Problem


    prob = Problem()

    comp = BComp(ng=4, NDIM=2, max_nn=4, NEL=1, max_edof=8, n_D=3, problem_type='plane_stress')

    prob.model = comp
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    B = prob['B']
    print(prob['B'])
    prob.check_partials(compact_print=True)


# D = np.identity(3)
# DB = np.einsum('km, ijmn -> ijkn', D, B)
# BTDB = np.einsum('ijlk, ijlm -> ikm', B, DB)

# print(DB)
# print(BTDB)
