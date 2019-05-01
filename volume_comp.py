
from openmdao.api import ExplicitComponent
import numpy as np


class VolumeComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('NEL', types=int)
        self.options.declare('be', types = float)
        self.options.declare('le', types = float)

    def setup(self):
        NEL = self.options['NEL']
        be = self.options['be']
        le = self.options['le']

        self.add_input('t', shape=NEL)
        self.add_output('volume')

        self.declare_partials('volume', 't', method = 'cs')

    def compute(self, inputs, outputs):
        be = self.options['be']
        le = self.options['le']

        outputs['volume'] = np.sum(inputs['t'] * be * le)