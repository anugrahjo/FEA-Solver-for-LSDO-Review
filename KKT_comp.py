import numpy as np


from openmdao.api import ExplicitComponent


class KKTComp(ExplicitComponent):
    
    def initialize(self):
        self.options.declare('NDOF', types=int)
        # self.options.declare('NEL', types=int)
        self.options.declare('A', types = np.ndarray)
        self.options.declare('f', types = np.ndarray)
        self.options.declare('constraints', types = np.ndarray)
        
 

    def setup(self):
        NDOF = self.options['NDOF']
        # NEL = self.options['NEL']
        constraints = self.options['constraints']
        self.add_input('Kglobal', shape=(NDOF, NDOF))
        self.add_output('K_temp', shape = (NDOF + len(constraints),NDOF + len(constraints)))
        self.add_output('f_temp', shape = (NDOF + len(constraints)))
        # self.declare_partials('d', 'Kglobal', method ='fd')
        col_ind = np.arange(NDOF*NDOF)
        # for rows
        arange = np.arange(NDOF)
        rows = np.tile(arange, NDOF)
        cols = np.repeat(arange, NDOF)
        row_ind = np.block([[rows], [cols]])
        row_ind = np.ravel_multi_index(row_ind, (NDOF + len(constraints),NDOF + len(constraints)))
        self.declare_partials('K_temp', 'Kglobal', val=1., rows=row_ind, cols=col_ind)


    def compute(self, inputs, outputs):
        A = self.options['A']
        f = self.options['f']
        constraints = self.options['constraints']
        nc = len(constraints)
        Kglobal = inputs['Kglobal']

        outputs['K_temp'] = np.block([[Kglobal, A.T],[A, np.zeros((nc,nc))]])
        outputs['f_temp'] = np.append(f, constraints)