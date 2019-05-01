from openmdao.api import ExplicitComponent
from mesh import Mesh
from element import Element


import numpy as np

class JacobianComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('ng', types=int)
        self.options.declare('NDIM', types=int)         # defined in the problem, constant for all elements.
        self.options.declare('NEL', types=int)
        self.options.declare('pN', types=np.ndarray)
        self.options.declare('ENT', types=np.ndarray )
        self.options.declare('Node_Coords', types=np.ndarray)


    def setup(self):
        ng = self.options['ng']
        NDIM = self.options['NDIM']
        NEL = self.options['NEL']

        self.add_output('J', shape=(NEL, ng**2, NDIM, NDIM))
        # self.declare_partials('J', '*', method = 'cs')

    def compute(self, inputs, outputs):
        ng = self.options['ng']
        NDIM = self.options['NDIM']
        NEL = self.options['NEL']

        pN = self.options['pN']
        ENT = self.options['ENT']
        Node_Coords = self.options['Node_Coords']
        J = np.zeros((NEL, ng**2, NDIM, NDIM))          ## ng ** 2 for rectangular elements

        for i in range(NEL):
            ent_position = np.where(ENT[i]>-1)
            ent_position = ent_position[0]
            nn = ent_position.shape[0]
            ent = ENT[i][0:nn]
            pN_ele = pN[i][:][:][0:nn]
            coords_ele = np.zeros((nn, NDIM))
            for j in range(nn):
                position = int(ent[j])
                coords_ele[j] = Node_Coords[position - 1]
            np.einsum('ijk, km -> ijm', pN_ele, coords_ele)
            J[i] = np.einsum('ijk, km -> ijm', pN_ele, coords_ele)

        outputs['J'] = J

    def compute_partials(self, inputs, partials):
        pass


if __name__ == '__main__':
    from openmdao.api import Problem

    prob = Problem()

    comp = JacobianComp(ng=2, NDIM=2, max_nn=4, NN=10, NEL=3)
    prob.model = comp
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    print(prob['J'])
    prob.check_partials(compact_print=True)
