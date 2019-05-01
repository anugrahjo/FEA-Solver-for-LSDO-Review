import numpy as np

from mesh import Mesh
from openmdao.api import Group, ExplicitComponent, IndepVarComp
from mesh import Mesh
from jacobian_comp import JacobianComp
from B_comp import BComp
from D_comp import DComp
from Kel_local_comp import Kel_localComp
from Kglobal_comp import KglobalComp
from solve_comp import SolveComp
from compliance_comp import ComplianceComp
from volume_comp import VolumeComp


class FEAGroup(Group):
    def initialize(self):
        self.options.declare('mesh', types = Mesh)
        self.options.declare('C', types = np.ndarray)
        self.options.declare('problem_type', types = str )
        self.options.declare('ng', types = int)
        self.options.declare('A', types = np.ndarray)
        self.options.declare('f', types = np.ndarray)
        self.options.declare('constraints', types = np.ndarray)
        self.options.declare('be', types = float)
        self.options.declare('le', types = float)

    def setup(self):
        mesh = self.options['mesh']
        C = self.options['C']
        problem_type = self.options['problem_type']
        ng = self.options['ng']
        be = self.options['be']
        le = self.options['le']

        pN = mesh.pN
        ENT = mesh.ENT
        Node_Coords = mesh.Node_Coords
        NDOF = mesh.NDOF
        NEL = mesh.NEL
        NDIM = mesh.NDIM
        max_nn = mesh.max_nn
        max_edof = mesh.max_edof
        NN = mesh.NN
        S = mesh.S
        A = self.options['A']
        f = self.options['f']
        constraints = self.options['constraints']




        # comp = ExplicitComponent()
        # comp.add_output('d', shape = (NDOF))
        # self.add_subsystem('d_comp', comp, promotes=['*'])

        comp = IndepVarComp()
        comp.add_output('t', shape = (NEL))
        self.add_subsystem('t_comp', comp, promotes=['*'])

        comp = DComp(C = C, problem_type = problem_type)
        self.add_subsystem('D_comp', comp, promotes=['*'])

        comp = JacobianComp(ng= ng, NDIM =NDIM, NEL = NEL, pN =pN, ENT = ENT, Node_Coords = Node_Coords)
        self.add_subsystem('J_comp', comp, promotes=['*'])

        comp = BComp(ng= ng, NDIM =NDIM, max_nn = max_nn, NEL = NEL, max_edof = max_edof, pN = pN, problem_type = problem_type)
        self.add_subsystem('B_comp', comp, promotes=['*'])

        comp = Kel_localComp(ng = ng, max_edof = max_edof, NEL = NEL)
        self.add_subsystem('Kl_comp', comp, promotes=['*'])

        comp = KglobalComp(S = S, max_edof = max_edof, NEL =NEL, NDOF = NDOF)
        self.add_subsystem('Kg_comp', comp, promotes=['*'])

        comp = SolveComp(NDOF = NDOF, A = A , f = f , NEL = NEL, constraints = constraints)
        self.add_subsystem('Solve_comp', comp, promotes=['*'])
        
        comp = ComplianceComp(NDOF = NDOF, f = f)
        self.add_subsystem('Compliance_comp', comp, promotes=['*'])

        comp = VolumeComp(NEL = NEL, be = be, le = le )
        self.add_subsystem('Volume_comp', comp, promotes=['*'])


if __name__ == '__main__':
    from openmdao.api import Problem, ScipyOptimizeDriver

    prob = Problem()
    prob.model = FEAGroup()

    