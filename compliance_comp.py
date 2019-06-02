import numpy as np

from openmdao.api import ExplicitComponent


class ComplianceComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('NDOF', types=int)
        self.options.declare('f', types=np.ndarray)

    def setup(self):
        NDOF = self.options['NDOF']
        f = self.options['f']

        self.add_input('displacements', shape = NDOF)
        self.add_output('compliance')

        self.declare_partials('compliance', 'displacements', val = f)

    def compute(self, inputs, outputs):
        f = self.options['f']
        displacements = self._inputs['displacements']

        outputs['compliance'] = np.dot(f, displacements)