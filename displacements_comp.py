

import numpy as np

from openmdao.api import ExplicitComponent


class DisplacementsComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('NDOF', types=int)
        self.options.declare('constraints', types = np.ndarray)

    def setup(self):
        NDOF = self.options['NDOF']
        constraints = self.options['constraints']

        self.add_input('d', shape= NDOF + constraints.size)
        self.add_output('displacements', shape=NDOF)

        arange = np.arange(NDOF)
        self.declare_partials('displacements', 'd', val=1., rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        NDOF = self.options['NDOF']

        outputs['displacements'] = inputs['d'][:NDOF]