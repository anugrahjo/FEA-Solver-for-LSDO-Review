import numpy as np

from sparse_algebra.core.dense_to_sparse import dense_to_sparse
from sparse_algebra.core.sparse_to_dense import sparse_to_dense
from sparse_algebra.core.sparse_tensor import SparseTensor
from sparse_algebra.sparse_einsum.sparse_einsum import sparse_einsum

from openmdao.api import ExplicitComponent


class StressComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('max_nn', types=int)
        self.options.declare('NEL', types=int)
        self.options.declare('max_edof', types=int)         
        self.options.declare('problem_type', types=str)

    def setup(self):
        NDOF = self.options['NDOF']
        ng = self.options['ng']
        NEL = self.options['NEL']
        max_edof = self.options['max_edof']
        problem_type = self.options['problem_type']
        
        if problem_type == 'plane_stress' or 'plane_strain':
            n_D = 3
        if problem_type == 'truss':
            n_D = 1

        self.add_input('D', shape = (n_D, n_D))
        self.add_input('strain', shape=(NEL, n_D))
        self.add_output('stress', shape=(NEL, n_D))

        self.declare_partials('strain', '*', method ='cs')

    def compute(self, inputs, outputs):
        D = inputs['D']
        strain = inputs['strain']

        # D_sp = dense_to_sparse(D)
        # strain_sp = dense_to_sparse(strain)

        stress = np.einsum('ij, kj -> ki', D, strain)

        # stress_sp = sparse_einsum([[0,1],[2,1],[2,0]], D_sp, strain_sp)
        # stress = sparse_to_dense(stress_sp)

        outputs['stress'] = stress