import numpy as np
from mesh import Mesh
from truss_element import TrussElement

from openmdao.api import Problem, ScipyOptimizeDriver, ExecComp

# from FEA_group import FEAGroup



# mesh = Mesh()
# node_coords1 = np.array([[0,0],[0.5,0],[1,0],[0,0.5],[0.5,0.5],[1,0.5],[0,1],[0.5,1],[1,1]])
# node_coords1 = 1e-3 * node_coords1
# ndof1 = 2
# mesh.set_nodes(node_coords1, ndof1)
# ent1 = np.array([[1,2,5,4],[2,3,6,5],[4,5,8,7],[5,6,9,8]])
# mesh.add_elem_group(ent1, 2)
# mesh.add_elem_group_partials()

# t = TrussElement()
# k = t.shape_function_partial()
# print(k)

# meshing using parameters: nodal coordinates
xstart = 0
ystart = 0
xend = 1
yend = 1
NELx = 2
NELy = 2
NEL = NELx * NELy
NNx = NELx + 1
NNy = NELy + 1
NN = NNx * NNy

x = np.linspace(xstart, xend, NNx)
y = np.linspace(ystart, yend, NNy)

x_coords = np.tile(x, NNy)
y_coords = np.repeat(y, NNx)

node_coords1 = np.zeros((NN, 2)) 
node_coords1[:, 0] = x_coords
node_coords1[:, 1] = y_coords

#meshing using parametes: element nodes table
node1 = np.arange(1, NNx * (NNy-1) + 1)
node2 = node1 + 1
node4 = np.arange(NNx+1, NNx * NNy + 1)
node3 = node4 + 1
ent1 = np.zeros((NEL + NELy, 4))
ent1[:, 0] = node1
ent1[:, 1] = node2
ent1[:, 2] = node3
ent1[:, 3] = node4

ent1 = np.delete( ent1, list(range(NNx-1, ent1.shape[0], NNx)), axis = 0)       # or np.s_[] :slicing

ndof1 = 2
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

