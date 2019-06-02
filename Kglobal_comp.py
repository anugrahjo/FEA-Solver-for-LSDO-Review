from openmdao.api import ExplicitComponent
import numpy as np

from sparse_algebra import dense
from sparse_algebra import sparse
from sparse_algebra import SparseTensor
from sparse_algebra import sparse_einsum
from sparse_algebra import einsum_partial

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
        self.add_input('t', shape = NEL)
        self.add_output('Kglobal', shape=( NDOF, NDOF))
        self.declare_partials( 'Kglobal', 't')
        
    def compute(self, inputs, outputs):
        max_edof = self.options['max_edof']
        NDOF = self.options['NDOF']
        NEL = self.options['NEL']
        S_ind = self.options['S']
        # print(S_ind)
        S_shape = np.array([NEL, max_edof, NDOF])
        S_val = np.ones(S_ind.shape[0])
        t = inputs['t']
        Kel_local = inputs['Kel_local']

        Kel_local_sp = sparse(Kel_local)
        t_sp = sparse(t)
        S = SparseTensor()
        S.initialize(S_shape, S_val, S_ind)
        # print(S.ind.shape)

        # Kglobal = np.einsum('ijk, imn, ijm  -> ikn', S, S, Kel_local)
        # Kglobal = np.einsum('ijk, imn, ijm  -> kn', S, S, Kel_local)
        print('itrn')

        Kglobal_sp = sparse_einsum([[0,1,2],[0,3,4],[0,1,3], [0], [2, 4]], S, S, Kel_local_sp, t_sp)
        # Kglobal_sp = sparse_einsum([[0,1,2],[0,3,4],[0,1,3],[2, 4]], S_sp, S_sp, Kel_local_sp)

        Kglobal = dense(Kglobal_sp)

        outputs['Kglobal'] = Kglobal

    def compute_partials(self, inputs, partials):
        # partials['Kglobal', 'Kel_local'] = inputs['Kel_local'] * 2
        max_edof = self.options['max_edof']
        NDOF = self.options['NDOF']
        NEL = self.options['NEL']
        S_ind = self.options['S']
        S_shape = np.array([NEL, max_edof, NDOF])
        S_val = np.ones(S_ind.shape[0])
        t = inputs['t']
        Kel_local = inputs['Kel_local']
        Kel_local_sp = sparse(Kel_local)
        t_sp = sparse(t)
        S = SparseTensor()
        S.initialize(S_shape, S_val, S_ind)

        partial_sp = einsum_partial([[0,1,2],[0,3,4],[0,1,3], [0], [2, 4], [3]], S, S, Kel_local_sp, t_sp)
        partial = dense(partial_sp)
        partials['Kglobal', 't'] = partial

if __name__ == '__main__':
    from openmdao.api import Problem, IndepVarComp

    prob = Problem()
    ivc = IndepVarComp()

    comp = KglobalComp(max_edof=5, NDOF=15, NEL=3, S=np.arange(15))
    ivc.add_output('Kel_local', val=np.random.rand(3,5,5))
    ivc.add_output('t', val=np.random.rand(3))

    prob.model.add_subsystem('ivc', ivc, promotes=['*'])
    prob.model.add_subsystem('comp', comp, promotes=['*'])
    prob.setup()
    prob.run_model()
    prob.check_partials(compact_print=True)