import numpy as np
from mesh import Mesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from openmdao.api import Problem, ScipyOptimizeDriver, ExecComp, view_model
from openmdao.api import pyOptSparseDriver
from sparse_algebra import SparseTensor
from sparse_algebra import compute_indices
from sparse_algebra import sparse, dense

from FEA_group import FEAGroup

nu = 0.3
m = (1 - nu)/2
E = 2
q = E / (1 - nu**2)

C = q * np.array([[1, nu, nu, 0, 0, 0],
                  [nu, 1, nu, 0, 0, 0],
                  [nu, nu, 1, 0, 0, 0],
                  [0, 0, 0, m, 0, 0],
                  [0, 0, 0, 0, m, 0],
                  [0, 0, 0, 0, 0, m]])

prob_type = 'plane_strain'

ng = 2

# meshing using parameters: nodal coordinates
xstart = 0
ystart = 0
xend = 1
yend = 1
l = abs(xend - xstart)
b = abs(yend - ystart)
NELx = 3
NELy = 3
le = l/NELx
be = b/NELy
NEL = NELx * NELy
NNx = NELx + 1
NNy = NELy + 1
NN = NNx * NNy

x = np.linspace(xstart, xend, NNx)
y = np.linspace(ystart, yend, NNy)
xm = np.linspace(xstart + le/2, xend - le/2, NELx)
ym = np.linspace(ystart + be/2, yend - be/2, NELy)

x_coords = np.tile(x, NNy)
y_coords = np.repeat(y, NNx)
xm_coords = np.tile(xm, NELy)
ym_coords = np.repeat(ym, NELx)

node_coords1 = np.zeros((NN, 2)) 
node_coords1[:, 0] = x_coords
node_coords1[:, 1] = y_coords

#meshing using parameters: element nodes table
node1 = np.arange(1, NNx * (NNy-1) + 1)
node2 = node1 + 1
node4 = np.arange(NNx+1, NNx * NNy + 1)
node3 = node4 + 1
ent1 = np.zeros((NEL + NELy, 4))
ent1[:, 0] = node1
ent1[:, 1] = node2
ent1[:, 2] = node3
ent1[:, 3] = node4

ent1 = np.delete(ent1, np.s_[NNx-1 :: NNx], 0)       # np.s_[] :slicing or can use list like: list(range(NNx-1, ent1.shape[0], NNx))

#costraints with parameters
ndof1 = 2
constraints = np.zeros(((NNx+NELy)*ndof1))
A = np.zeros(((NNx+NELy)*ndof1, NN * ndof1))
t = 0
for i in range(NNx*2):
    A[i,i] = 1
    t += 1
for i in range(NELy):
    A[t, NNx * ndof1 * (i + 1)] = 1
    t += 1
    A[t, NNx * ndof1 * (i + 1) + 1] = 1
    t += 1


#force distribution with parameters
f_dbn = 10               #(kN/m = N/mm)
f_tot = f_dbn * (yend - ystart)
f_ele = f_tot/(NELy)
f = np.zeros(NN * ndof1)
for i in range(NNy):
    if i == 0 or i == (NNy-1):
        f[(NNx * ndof1)*(i+1) - 2] = f_ele/2
    else:
        f[(NNx * ndof1)*(i+1) - 2] = f_ele


# constraints = np.zeros(10)
# A = np.zeros((10, 18))
# A[0,0] = A[1][1] = A[2][2] = A[3][3] = A[4][4] = A[5][5] = A[6][6] = A[7][7] = A[8][12] = A[9][13] = 1
# f = np.zeros(18)
# f[4] = f[16] = 2.5
# f[10] = 5


mesh = Mesh()
mesh.set_nodes(node_coords1, ndof1)
mesh.add_elem_group(ent1, 2)                            # 2 is element type for rectangular elements
mesh.add_elem_group_partials()

# S = mesh.S
# NEL = mesh.NEL
# max_edof = mesh.max_edof



# prob = Problem(model = FEAGroup(C = C, mesh = mesh))

prob = Problem()
prob.model = FEAGroup(C=C, mesh=mesh, problem_type = prob_type, ng = ng, A =A, f = f, constraints = constraints, be = be, le = le)
prob.model.connect('K_temp', 'Solve_comp.A')
prob.model.connect('f_temp', 'Solve_comp.b')
prob.model.connect('Solve_comp.x','d')
prob.model.add_design_var('t')
prob.model.add_constraint('volume', upper = l*b*3)
prob.model.add_constraint('t', lower = 1)




prob.model.add_objective('compliance')

# prob.driver = ScipyOptimizeDriver()
# prob.driver.options['optimizer'] = 'SLSQP'
prob.driver = pyOptSparseDriver()
prob.driver.options['optimizer'] = 'SNOPT'


# prob.driver.options['tol'] = 1e-9
# prob.driver.options['obj'] = True

prob.setup()
prob.run_driver()
# prob.run_model()
print(prob['compliance'])
print(prob['displacements'])
print(prob['t'])
print(prob['volume'])

view_model(prob)

##Contour plot
# thickness_dbn = np.reshape(prob['t'], (NELx,NELy))
# x = np.arange(NELx)
# y = np.arange(NELy)
# X, Y = np.meshgrid(x, y)
# plt.contour(X, Y, thickness_dbn)

#3D Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xm_coords, ym_coords, prob['t'])
plt.show()

# prob.run_model()
# prob.model.list_outputs()
# prob['']