import torch
import numpy as np

def linearLSTriangulation(kpts0, kpts1, P0, P1):
    """
    线性三角化

    Args:
        kpts0: (N, 2)
        kpts1: (N, 2)
        P0: (3, 4)
        P1: (3, 4)
    """

    N = kpts0.shape[0]
    kpts0 = torch.cat([kpts0, torch.ones(N, 1)], dim=1)
    kpts1 = torch.cat([kpts1, torch.ones(N, 1)], dim=1)

    A = torch.stack([
        kpts0[:, 0] * P0[2] - P0[0],
        kpts0[:, 1] * P0[2] - P0[1],
        kpts1[:, 0] * P1[2] - P1[0],
        kpts1[:, 1] * P1[2] - P1[1],
    ], dim=1)

    _, _, V = torch.svd(A)
    X = V[:, -1]
    X = X / X[-1]

    return X[:3]

def iterativeLinearLSTriangulation(kpts0, kpts1, P0, P1):
    EPSILON = 0.0001

    X = np.zeros((4, 1))
    X_ = linearLSTriangulation(kpts0, P0, kpts1, P1)
    X[0] = X_[0]
    X[1] = X_[1]
    X[2] = X_[2]
    X[3] = 1.0

    for i in range(10):  # Hartley suggests 10 iterations at most
        # recalculate weights
        p2x = np.dot(P0[2, :], X)[0]
        p2x1 = np.dot(P1[2, :], X)[0]

        # breaking point
        if abs(wi - p2x) <= EPSILON and abs(wi1 - p2x1) <= EPSILON:
            break

        wi = p2x
        wi1 = p2x1

        # reweight equations and solve
        A = np.array([
            [(u[0]*P[2,0]-P[0,0])/wi,       (u[0]*P[2,1]-P[0,1])/wi,         (u[0]*P[2,2]-P[0,2])/wi],
            [(u[1]*P[2,0]-P[1,0])/wi,       (u[1]*P[2,1]-P[1,1])/wi,         (u[1]*P[2,2]-P[1,2])/wi],
            [(u1[0]*P1[2,0]-P1[0,0])/wi1,   (u1[0]*P1[2,1]-P1[0,1])/wi1,     (u1[0]*P1[2,2]-P1[0,2])/wi1],
            [(u1[1]*P1[2,0]-P1[1,0])/wi1,   (u1[1]*P1[2,1]-P1[1,1])/wi1,     (u1[1]*P1[2,2]-P1[1,2])/wi1]
        ])
        B = np.array([
            [-(u[0]*P[2,3]    -P[0,3])/wi],
            [-(u[1]*P[2,3]  -P[1,3])/wi],
            [-(u1[0]*P1[2,3]    -P1[0,3])/wi1],
            [-(u1[1]*P1[2,3]    -P1[1,3])/wi1]
        ])

        X_ = np.linalg.lstsq(A, B, rcond=None)[0]
        X[0] = X_[0]
        X[1] = X_[1]
        X[2] = X_[2]
        X[3] = 1.0
        