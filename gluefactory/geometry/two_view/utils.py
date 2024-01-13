import numpy as np
import cv2
import torch

# --- GEOMETRY ---


def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    if kpts0.shape[0] < 5:
        print(f"kpts_shape[0]:{kpts0.shape[0]}")
        return None

    f_mean = np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
    norm_thresh = thresh / f_mean

    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=norm_thresh, prob=conf,
        method=cv2.RANSAC)

    assert E is not None

    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(
            _E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            best_num_inliers = n
            ret = (R, t[:, 0], mask.ravel() > 0)
    return ret

# 计算在同一个相机下两个变换矩阵的相对变换矩阵
def compute_relative_pose(pose0: torch.Tensor, pose1: torch.Tensor) -> torch.Tensor:
    """计算两个变换矩阵的相对变换矩阵 Math: T = T_01 * T_10^-1"""
    # 将返回值用小数表示而不是科学计数
    return torch.matmul(torch.inverse(pose0), pose1)