from openmdao.api import ExplicitComponent
import numpy as np

from sparse_algebra.core.dense_to_sparse import dense_to_sparse
from sparse_algebra.core.sparse_to_dense import sparse_to_dense
from sparse_algebra.core.sparse_tensor import SparseTensor
from sparse_algebra.sparse_einsum.sparse_einsum import sparse_einsum

class KglobalComp(ExplicitComponent):
    def initialize(self):
        self.options.declare('max_edof', types=int)
        self.options.declare('NDOF', types=int)
        self.options.declare('NEL', types=int)
        self.options.declare('S', types = np.ndarray)           #shape = (NEL, max_edof, NDOF)

    def setup(self):
        max_edof = self.options['max_edof']
        NDOF = self.options['NDOF']
        NEL = self.options['NEL']

        self.add_input('Kel_local', shape=(NEL, max_edof, max_edof))
        self.add_output('Kglobal', shape=( NEL, NDOF, NDOF))
        self.declare_partials( 'Kglobal', 'Kel_local', method = 'cs')
        
    def compute(self, inputs, outputs):
        S = self.options['S']
        Kel_local = inputs['Kel_local']

        # Kel_local_sp = dense_to_sparse(Kel_local)
        # S_sp = dense_to_sparse(S)

        Kglobal = np.einsum('ijk, imn, ijm  -> ikn', S, S, Kel_local)

        # Kglobal_sp = sparse_einsum([[0,1,2],[0,3,4],[0,1,3],[2,4]], S_sp, S_sp, Kel_local_sp)

        # Kglobal = sparse_to_dense(Kglobal_sp)


        outputs['Kglobal'] = Kglobal

    # def compute_partials(self, inputs, partials):
    #     # partials['Kglobal', 'Kel_local'] = inputs['Kel_local'] * 2
    #     pass