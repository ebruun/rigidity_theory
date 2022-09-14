import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
import random
from sympy import Matrix

from helper_funcs import calc_edge_lengths

from config_c import d,V,E


# Build rigidity matrix
def build_R_mat(E, V, d):
    R = np.zeros((E.shape[0],d*V.shape[0]))

    for row,e in enumerate(E):

        i = e[0]
        j = e[1]
        
        if d == 2:
            R[row,d*(i-1)] = V[i-1][0] - V[j-1][0]
            R[row,d*(i-1)+1] = V[i-1][1] - V[j-1][1]

            R[row,d*(j-1)] = V[j-1][0] - V[i-1][0]
            R[row,d*(j-1)+1] = V[j-1][1] - V[i-1][1]
        elif d == 3:
            R[row,d*(i-1)] = V[i-1][0] - V[j-1][0]
            R[row,d*(i-1)+1] = V[i-1][1] - V[j-1][1]
            R[row,d*(i-1)+2] = V[i-1][2] - V[j-1][2]

            R[row,d*(j-1)] = V[j-1][0] - V[i-1][0]
            R[row,d*(j-1)+1] = V[j-1][1] - V[i-1][1]
            R[row,d*(j-1)+2] = V[j-1][2] - V[i-1][2]

    return R

# Find kernel (nullspace) of rigidity matrix
def rigidity_kernel(R,method="sympy"):
    if method == "sympy":
        R_ker = Matrix(R).nullspace()

        a = np.array(R_ker[0]).astype(np.float64)
        for col in R_ker[1:]:
            a = np.column_stack((a,np.array(col).astype(np.float64)))
            
        R_ker = a

    elif method=="scipy":
        R_ker = scipy.linalg.null_space(R)

    print("Rigidity Kernel using {}".format(method))

    return R_ker


##############################
# Rigidity Matrix
R = build_R_mat(E,V,d)
R_ker = rigidity_kernel(R,method="sympy")


#########################
# flex test of R kernel columns
def unconnected_node_pairs():
    print("find unconnected nodes\n")
    bins = np.arange(0,len(V)*d,d)
    adj_matrix = np.add.reduceat(np.abs(R),bins,axis=1)
    adj_matrix[adj_matrix>1] = 1

    unconnected_nodes = np.empty((0,2), int)

    # for each node(column), see what connected to
    for i in range(len(V)):
        not_adj_nodes = []
        rows=np.where(adj_matrix[:,i]==1)

        # columns w/ 0 are nodes not connected to
        adj_cnt = sum(adj_matrix[rows])
        not_adj_nodes = np.argwhere(adj_cnt == 0).flatten()

        # if sum is V-1, it is connected to all nodes
        if adj_cnt[i] < len(V)-1:
            for n in not_adj_nodes:

                if n > i:
                    non_member = np.array([i+1,n+1]).reshape(1,2)
                    unconnected_nodes = np.append(unconnected_nodes, non_member ,axis=0)
                else:
                    print("node {} is already listed for node {}".format(n+1,i+1))
        else:
            print("node {} is fully connected".format(i+1))

    print("\nunconnected node pairs\n", unconnected_nodes)
    return unconnected_nodes


E_not = unconnected_node_pairs()
R_not = build_R_mat(E_not,V,d)

rigid_motions = np.sum(abs(R_not.dot(R_ker)),axis=0)

eps = 1e-12
rigid_motions[rigid_motions < eps] = 0

##############################
print("\n---CHECK OUTPUTS---")

# Checking if infinitessimally rigid (ranks are the same)
def binom(n, k):
    return math.factorial(n) // math.factorial(k) // math.factorial(n - k)

lhs = np.linalg.matrix_rank(R)
rhs = d*len(V) - binom(d+1,2)

if lhs == rhs:
    print("\ninfinitessimally rigid!, {} = {}".format(lhs, rhs))
else:
    print("\nnot infinitessimally rigid!, {} not equal {}".format(lhs, rhs))


print("R\n",R)
print("R unconnected\n", R_not)
print("R kernel\n", R_ker)

print("R x R_ker\n", R.dot(R_ker))
print("R_not x R_ker\n", R_not.dot(R_ker))

# print("null space check\n", R.dot(test_motion))
calc_edge_lengths(V,E)


##############################
print("\nplot motion")

if d == 2:
    fig, axs = plt.subplots(3, 3, figsize=(10,10))
elif d == 3:
    fig, axs = plt.subplots(3, 3, figsize=(10,10), subplot_kw=dict(projection='3d'))

random.seed(1)

for t,ax in enumerate(axs.flatten()):

    if t >= R_ker.shape[1]:
        break

    print("\nmotion #{}".format(t))
    move_vec = np.zeros((len(V)*d))

    if t < len(R_ker[0]):
        move_vec = R_ker[:,t]
    else:
        move_vec = [0]* len(R_ker)

    move_vec = move_vec/np.linalg.norm(move_vec)
    V_move = V + move_vec.reshape(len(V),d)

    if d == 2:
        for i in E:
            s = i[0] - 1
            e = i[1] - 1
            ax.plot(
                [V[s][0],V[e][0]], [V[s][1],V[e][1]],
                'k-', 
                linewidth = 1,
                marker='.',
                markersize=5,
                )

        for i in E:
            s = i[0] - 1
            e = i[1] - 1
            ax.plot(
                [V_move[s][0],V_move[e][0]], [V_move[s][1],V_move[e][1]],
                'r-', 
                linewidth = 0.75,
                marker='.',
                markersize=3,
                )

        ax.axis('equal')
        ax.axhline(0, color='k', linewidth=0.2)
        ax.axvline(0, color='k', linewidth=0.2)

        for i in range(len(V)):
            ax.text(V[i,0], V[i,1], i+1)
            ax.text(V_move[i,0], V_move[i,1], i+1, color='red')

    elif d == 3:
        for i in E:
            s = i[0] - 1
            e = i[1] - 1
            ax.plot3D(
                [V[s][0],V[e][0]], [V[s][1],V[e][1]],[V[s][2],V[e][2]], 
                'k-', 
                linewidth = 1,
                marker='.',
                markersize=5,
                )

        for i in E:
            s = i[0] - 1
            e = i[1] - 1
            ax.plot3D(
                [V_move[s][0],V_move[e][0]], [V_move[s][1],V_move[e][1]],[V_move[s][2],V_move[e][2]],
                'r-', 
                linewidth = 0.75,
                marker='.',
                markersize=3,
                )   

        ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

        for i in range(len(V)):
            ax.text(V[i,0], V[i,1], V[i,2], i+1, 'x')
            ax.text(V_move[i,0], V_move[i,1], V_move[i,2], i+1, 'x', color='red')

        

    if rigid_motions[t] > 0:    
        ax.set_title("M#{}: FLEX".format(t+1))
    else:
         ax.set_title("M#{}: RIGID".format(t+1))
    
    calc_edge_lengths(V_move,E)


fig.tight_layout()
plt.show()

