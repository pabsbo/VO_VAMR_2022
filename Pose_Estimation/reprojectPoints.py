import numpy as np


def reprojectPoints(P, M, K):
    coords = []
    ones = np.ones((12, 1))
    X_Y_Z_1 = np.hstack((P, ones))

    for i in X_Y_Z_1:
        sol = (K @ M) @ i
        u = float(sol[0] / sol[2])
        v = float(sol[1] / sol[2])
        coords.append([u, v])

    return np.array(coords)
