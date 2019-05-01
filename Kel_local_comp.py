from openmdao.api import ExplicitComponent
from rectangular_plate import RectangularElement
import numpy as np

from sparse_algebra.core.dense_to_sparse import dense_to_sparse
from sparse_algebra.core.sparse_to_dense import sparse_to_dense
from sparse_algebra.core.sparse_tensor import SparseTensor
from sparse_algebra.sparse_einsum.sparse_einsum import sparse_einsum



class Kel_localComp(ExplicitComponent):
    def initialize(self):
        self.options.declare('ng', types=int)
        self.options.declare('max_edof', types=int)
        self.options.declare('NEL', types=int)
 

    def setup(self):
        ng = self.options['ng']
        max_edof = self.options['max_edof']
        NEL = self.options['NEL']

        self.add_input('B', shape=(NEL, ng**2, 3, max_edof))            #3 only for a rectangular element
        self.add_input('D', shape=(3, 3))                               #(3,3) only for a rectangular element
        # self.add_input('t', shape = (NEL))
        self.add_output('Kel_local', shape=(NEL, max_edof, max_edof))
        self.declare_partials('Kel_local', '*', method ='cs')
        
    def compute(self, inputs, outputs):
        B = inputs['B']
        D = inputs['D']
        # t = inputs['t']
        R = RectangularElement()
        W = R.gaussian_weights()                         #only for a rectangular element

        Kel_local_pre1 = np.einsum('ijkl, ijno, kn -> ijlo', B, B, D)
        Kel_local = np.einsum('ijlo, j ->ilo',Kel_local_pre1, W)
        # Kel_local = np.einsum('ilo, i ->ilo',Kel_local_pre2, t)
        
        # B_sp = dense_to_sparse(B)
        # D_sp = dense_to_sparse(D)
        # W_sp = dense_to_sparse(W)
        # t_sp = dense_to_sparse(t)
        
        # Kel_local_pre1_sp = dense_to_sparse(Kel_local_pre1)
        # Kel_local_pre2_sp = dense_to_sparse(Kel_local_pre2)
        # Kel_local_sp = dense_to_sparse(Kel_local)

        # Kel_local_pre1_sp = sparse_einsum([[0,1,2,3],[0,1,4,5],[2,4],[0,1,3,5]], B_sp, B_sp, D_sp)
        # Kel_local_pre2_sp = sparse_einsum([[0,1,3,5],[1],[0,3,5]],Kel_local_pre1_sp, W_sp)
        # Kel_local_sp = sparse_einsum([[0,3,5],[0],[0,3,5]],Kel_local_pre2_sp, t_sp)


        # Kel_local = sparse_to_dense(Kel_local_sp)


        outputs['Kel_local'] = Kel_local

    # def compute_partials(self, inputs, partials):
        # partials['Kel_local', 'B'] = inputs['Kel_local'] * 2
        # partials['Kel_local', 'D'] = inputs['Kel_local'] * 2
        # pass