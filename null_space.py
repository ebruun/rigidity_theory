import numpy as np
import matplotlib.pyplot as plt
import math
import scipy

H = np.array(
    [[2,-1],
    [-4,2]]
)

H = np.array([
    [2,2,1],
    [2,-3,-4],
    [4,-1,-3]
])

H = np.array([
    [3,3,15,11],
    [1,-3,1,1],
    [2,3,11,8]
])

a = scipy.linalg.null_space(H,rcond=0.01)
print(a)

def nullspace(A, atol=1e-13, rtol=0):
    A = np.atleast_2d(A)
    u, s, vh = np.linalg.svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns

a = nullspace(H)

print(a)

R_flex = np.array([
    [-2.  ,0.,  2.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
    [-1., -1.,  0.,  0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
    [ 0.,  0.,  1., -1., -1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,],
    [ 0., -4.,  0.,  0.,  0.,  0.,  0.,  4.,  0.,  0.,  0.,  0.],
    [ 0.,  0.,  0., -4.,  0.,  0.,  0.,  0.,  0.,  4.,  0.,  0.],
    [ 0.,  0.,  0.,  0.,  0.,  0., -2.,  0.,  2.,  0.,  0.,  0.],
    [ 0.,  0.,  0.,  0., 0.,  0., -1., -1.,  0.,  0.,  1.,  1.],
    [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1., -1., -1.,  1.],
    [ 0.,  0.,  0.,  0.,  0., -4.,  0.,  0.,  0.,  0.,  0.,  4.],
])

R_rigid = np.array([
    [-2.  ,0.,  2.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
    [-1., -1.,  0.,  0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
    [ 0.,  0.,  1., -1., -1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,],
    [ 0., -4.,  0.,  0.,  0.,  0.,  0.,  4.,  0.,  0.,  0.,  0.],
    [ 0.,  0.,  0., -4.,  0.,  0.,  0.,  0.,  0.,  4.,  0.,  0.],
    [ 0.,  0.,  0.,  0.,  0.,  0., -2.,  0.,  2.,  0.,  0.,  0.],
    [ 0.,  0.,  0.,  0., 0.,  0., -1.1, -1.,  0.,  0.,  1.1,  1.],
    [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.9, -1., -0.9,  1.],
    [ 0.,  0.,  0.,  0.,  -0.1, -4.,  0.,  0.,  0.,  0.,  0.1,  4.],
])

print(R_flex)
print(R_rigid)

K_R_flex = scipy.linalg.null_space(R_flex)
K_R_rigid = scipy.linalg.null_space(R_rigid)

print(K_R_flex)
print(K_R_rigid)

angle = scipy.linalg.subspace_angles(K_R_flex, K_R_flex[:,0].reshape(12,1))

print(angle)