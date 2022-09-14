import numpy as np


def calc_edge_lengths(V,E):
    lens = []
    for e in E:
        lens.append(np.linalg.norm(V[e[0]-1,:] - V[e[1]-1,:]))
    print("lengths:",lens)