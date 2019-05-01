
from openmdao.api import ExplicitComponent
from mesh import Mesh
from jacobian_comp import JacobianComp


import numpy as np

class TComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('ng', types=int)
        self.options.declare('NDIM', types=int)         # defined in the problem, constant for all elements.
        self.options.declare('max_nn', types=int)
        self.options.declare('NN', types=int)
        self.options.declare('NEL', types=int)
        self.options.declare('edof', types=int)

    def setup(self):
        ng = self.options['ng']
        NDIM = self.options['NDIM']
        max_nn = self.options['max_nn']
        NN = self.options['NN']
        NEL = self.options['NEL']
        edof = self.options['edof']

        self.add_input('J', val=np.tile(np.identity(2),[3,2,1,1]), shape=(NEL, ng, NDIM, NDIM))
        self.add_output('T', shape=(NEL, ng, edof, edof))
        self.declare_partials('T', '*')
        
    def compute(self, inputs, outputs):
        J = inputs['J']
        ng = self.options['ng']
        max_nn = self.options['max_nn']
        NN = self.options['NN']
        NEL = self.options['NEL']
        NDIM = self.options['NDIM']
        edof = self.options['edof']
        
        T = np.zeros((NEL, ng, edof, edof))
        for i in range(NEL):
            for j in range(ng):
                J_ele = J[i][j]
                J_ele_inv = np.linalg.inv(J_ele)
                J_ele_norm = np.linalg.norm(J_ele)
                Tr = np.zeros((NDIM, NDIM))
                for m in range(NDIM):
                    J_ele_norm = np.linalg.norm(J_ele[m])
                    Tr[m] = J_ele_norm * J_ele_inv[m]
                zeros = np.zeros((NDIM, NDIM))
                print(T[i][j])
                print(Tr)
                T[i][j][0:2][0:2] = Tr  # for truss case
                # T[i][j][0:NDIM][NDIM:] = zeros
                # T[i][j][NDIM:][0:NDIM] = zeros
                # T[i][j][NDIM:][NDIM:] = Tr
                
                

        # if NDIM == 1:
        #     Tr = np.identity(max_edof)
        
        # elif NDIM == 2:
        #     theta_lx_gx = 
        #     theta_lx_gy =
        #     theta_ly_gx =
        #     theta_ly_gy =

        #     Tr =

        # elif NDIM == 2:
        #     theta_lx_gx = 
        #     theta_lx_gy =
        #     theta_lx_gz =
        #     theta_ly_gx =
        #     theta_ly_gy =
        #     theta_ly_gz =
        #     theta_lz_gx =
        #     theta_lz_gy =
        #     theta_lz_gz =

        #     Tr = 

        outputs['T'] = T

    def compute_partials(self, inputs, partials):
        # E1 = inputs['E1']
        # v23 = inputs['v23']

        partials['T', '*'] = 0


if __name__ == '__main__':
    from openmdao.api import Problem

    prob = Problem()

    comp = TComp(ng=2, NDIM=2, max_nn=2, NN=4, NEL=3, edof=4)
    prob.model = comp
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    print(prob['T'])
    prob.check_partials(compact_print=True)