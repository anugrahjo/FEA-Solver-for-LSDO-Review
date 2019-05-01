
from openmdao.api import ExplicitComponent

import numpy as np

class MaterialComp(ExplicitComponent):
    
    def setup(self):
        self.add_input('E1')
        self.add_input('E2')
        self.add_input('E3')
        self.add_input('G12')
        self.add_input('G13')
        self.add_input('G23')
        self.add_input('v12')
        self.add_input('v13')
        self.add_input('v23')
        self.add_output('C')
        self.declare_partials('C', '*')
        
    def compute(self, inputs, outputs):
        E1 = inputs['E1']
        E2 = inputs['E2']
        E3 = inputs['E3']
        G12 = inputs['G12']
        G13 = inputs['G13']
        G23 = inputs['G23']
        v12 = inputs['v12']
        v13 = inputs['v13']
        v23 = inputs['v23']

        stiffness = np.zeros((6, 6))
        stiffness[0][0] = 1/E1
        stiffness[0][1] = -v12/E1
        stiffness[0][2] = -v13/E1
        stiffness[1][0] = -v12/E1
        stiffness[1][1] = 1/E2
        stiffness[1][2] = -v23/E2
        stiffness[2][0] = -v13/E1
        stiffness[2][1] = -v23/E2
        stiffness[2][2] = 1/E3
        stiffness[3][3] = .5/G23
        stiffness[4][4] = .5/G13
        stiffness[5][5] = .5/G12

        outputs['C'] = np.linalg.inv(stiffness)

    def compute_partials(self, inputs, partials):
        E1 = inputs['E1']
        E2 = inputs['E2']
        E3 = inputs['E3']
        G12 = inputs['G12']
        G13 = inputs['G13']
        G23 = inputs['G23']
        v12 = inputs['v12']
        v13 = inputs['v13']
        v23 = inputs['v23']

        partials['C', '*'] = 0


if __name__ == '__main__':
    from openmdao.api import Problem, Group, IndepVarComp
    pass