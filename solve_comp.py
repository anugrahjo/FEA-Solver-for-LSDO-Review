import numpy as np

from openmdao.api import Group, ExplicitComponent

class SolveComp(ExplicitComponent):
    
    def initialize(self):
        self.options.declare('NDOF', types=int)
        self.options.declare('NEL', types=int)
        self.options.declare('A', types = np.ndarray)
        self.options.declare('f', types = np.ndarray)
        self.options.declare('constraints', types = np.ndarray)
        
 

    def setup(self):
        NDOF = self.options['NDOF']
        NEL = self.options['NEL']
        self.add_input('t', shape = (NEL))
        self.add_input('Kglobal', shape=(NEL, NDOF, NDOF))
        self.add_output('d', shape = NDOF)
        self.declare_partials('d', 'Kglobal', method ='cs')
        
    def compute(self, inputs, outputs):
        NDOF = self.options['NDOF']
        NEL = self.options['NEL']
        A = self.options['A']
        f = self.options['f']
        t = inputs['t']
        constraints = self.options['constraints']
        nc = len(constraints)                                           #num of constraints

        Kglobal = inputs['Kglobal']
        Kglobal1 = np.einsum('ijk, i -> jk', Kglobal , t)
        K_temp = np.block([[Kglobal1, A.T],[A, np.zeros((nc,nc))]])
        f_temp = np.append(f, constraints)


        d_temp = np.linalg.solve(K_temp, f_temp)

        d = d_temp[0 : NDOF]

        outputs['d'] = d