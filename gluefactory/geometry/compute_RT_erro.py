"""计算两张图象经过superpoint提取特征和lightglue匹配后的相对位姿误差"""

import argparse

from pathlib import Path
import time
import torch
import numpy as np
from kornia.geometry.epipolar import triangulate_points
from two_view.estimate_relative_pose import estimate_relative_pose_w8pt, run_bundle_adjust_2_view, normalize
from two_view.compute_pose_error import compute_rotation_error, compute_translation_error_as_angle
from two_view.utils import estimate_pose, compute_relative_pose
from ..models.matchers.lightglue import LightGlue
from ..models.extractors.superpoint_open import SuperPoint
from ..utils.image import load_image
from ..utils.tensor import rbd


def couculate_reprojection_error(intr, pred_T, gt_T0, gt_T1, gt_T, kpts0, kpts1, confidence):
    """计算重投影误差"""
    intr = intr.squeeze(0).type(torch.float64)
    gt_T = gt_T.squeeze(0).type(torch.float64)
    pred_T = pred_T.squeeze(0).type(torch.float64)
    kpts0 = kpts0.squeeze(0).type(torch.float64)
    kpts1 = kpts1.squeeze(0).type(torch.float64)

    pred_T = gt_T1 @ pred_T

    # 相机投影矩阵
    P_w0 = intr @ gt_T0[:3, :]
    P_w1 = intr @ gt_T1[:3, :]
    pred_P_w1 = intr @ pred_T[:3, :]

    # 三角化
    pts_3d = triangulate_points(P_w0, P_w1, kpts0, kpts1)
    # print(f"pts_3d:{pts_3d.shape}")
    pts_3d_0 = torch.ones((pts_3d.shape[0], 4, 1))
    pts_3d_0[:,:3] = pts_3d[:, :].unsqueeze(-1)
    pred_P_w1 = pred_P_w1.type(torch.float64)
    pts_3d_0 = pts_3d_0.type(torch.float64)

    # 重投影, 3*4矩阵乘以N*4*1矩阵
    pts_2d_0 = torch.einsum('mn, bnl -> bml', P_w1, pts_3d_0).squeeze(-1)
    pts_2d_1 = torch.einsum('mn, bnl -> bml', pred_P_w1, pts_3d_0).squeeze(-1)

    # 归一化
    pts_2d_0 = pts_2d_0[:, :2] / pts_2d_0[:, 2:]
    pts_2d_1 = pts_2d_1[:, :2] / pts_2d_1[:, 2:]
    # print(f"pts_2d_1:{pts_2d_1}")
    # print(f"kpts1:{kpts1}")
    # 计算重投影误差
    reproj_err_0 = torch.linalg.norm(pts_2d_0 - kpts0, dim=-1)
    reproj_err_1 = torch.linalg.norm(pts_2d_1 - kpts0, dim=-1)


    return reproj_err_1


def pairs_compute_RT_erro(data, recover, datasets_dir, match_pairs=100):
    """计算两张图象经过superpoint提取特征和lightglue匹配后的相对位姿误差

    Args:
        datasets_dir (str): 数据集路径
        bundle_adjust (bool): 是否进行优化
        match_pairs (int, optional): 匹配对数. Defaults to 100.
    """

    if 'replica' in data:
        images_dir = Path(datasets_dir) / 'col'
        pose_dir = Path(datasets_dir)
        intr = torch.tensor([[600.0, 0.0, 599.5], [0.0, 600, 339.5], [0.0, 0.0, 1.0]]).unsqueeze(0)
        # intr = torch.tensor([[600.0, 0.0, -10.0], [0.0, 600.0, 200.0], [0.0, 0.0, 1.0]]).unsqueeze(0)
        with open(pose_dir / 'traj.txt', 'r') as f:
            poses = f.readlines()
        gt_pose0 = torch.from_numpy(np.array([float(x) for x in poses[match_pairs].split()])).reshape( 4, 4)
        gt_pose1 = torch.from_numpy(np.array([float(x) for x in poses[match_pairs + 19].split()])).reshape(4, 4)

        # 加载图像, replica数据集的图像是以frame+六位数字命名
        image0 = load_image(images_dir / f'frame{match_pairs:06d}.jpg')
        image1 = load_image(images_dir / f'frame{match_pairs + 19:06d}.jpg')

    else:
        # 路径 
        images_dir = Path(datasets_dir) / 'color'
        pose_dir = Path(datasets_dir) / 'pose'
        intr_dir = Path(datasets_dir) / 'intrinsic'

        # 加载位姿
        gt_pose0 = torch.from_numpy(np.genfromtxt(pose_dir / f'{match_pairs}.txt'))
        gt_pose1 = torch.from_numpy(np.genfromtxt(pose_dir / f'{match_pairs + 20}.txt'))

        # 加载相机内参
        intr = np.genfromtxt(intr_dir / 'intrinsic_color.txt')[:3, :3]
        intr = torch.from_numpy(intr).unsqueeze(0)
        # intr[0, 1:2, 2:] = -intr[0, 1:2, 2:]
        
        # 加载图像
        image0 = load_image(images_dir / f'{match_pairs}.jpg')
        image1 = load_image(images_dir / f'{match_pairs + 106}.jpg')
        print(images_dir / f'{match_pairs}.jpg')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
    matcher = LightGlue(features='superpoint').eval().to(device)

    # 初始化误差
    r_erro = 0
    t_erro = 0

    time0 = time.time()

    gt_T = compute_relative_pose(gt_pose1, gt_pose0)

    feats0 = extractor.extract(image0.to(device))
    feats1 = extractor.extract(image1.to(device))
    matches01 = matcher({'image0': feats0, 'image1': feats1})

    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

    kpts0, kpts1, matches = feats0['keypoints'].cpu(), feats1['keypoints'].cpu(), matches01['matches'].cpu()
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    confidence = matches01['scores'].cpu()

    m_kpts0 = m_kpts0.unsqueeze(0)
    m_kpts1 = m_kpts1.unsqueeze(0)
    confidence = confidence.unsqueeze(0)
    
    if "w8pt" in recover:
        pred_T, info = estimate_relative_pose_w8pt(m_kpts0, m_kpts1, intr, intr, confidence)
        print("---------optimize by bundle_adjustment in w8pt---------")
        # 优化pred_T
        bundle_T, _ = run_bundle_adjust_2_view(info["kpts0_norm"], info["kpts1_norm"], confidence, pred_T, \
                    n_iterations=10)
        pred_T = bundle_T.clone().detach()
    else:
        print("---------optimize by bundle_adjustment in opencv---------")
        ret = estimate_pose(m_kpts0.cpu().numpy().squeeze(0), m_kpts1.cpu().numpy().squeeze(0), intr.cpu().numpy().squeeze(0), intr.cpu().numpy().squeeze(0), thresh=1)
        pred_T = torch.eye(4).unsqueeze(0)
        pred_T[0, :3, :3] = torch.from_numpy(ret[0]).unsqueeze(0)
        pred_T[0, :3, 3] = torch.from_numpy(ret[1]).unsqueeze(0)

        assert pred_T is not None, 'Failed to estimate pose'
        print(f"ret_shape:{pred_T.shape}")

        # normalize
        norm_kpts0 = normalize(m_kpts0, intr).cpu()
        norm_kpts1 = normalize(m_kpts1, intr).cpu()

        bundle_T = run_bundle_adjust_2_view(norm_kpts0, norm_kpts1, confidence, pred_T, n_iterations=10)[0]
        pred_T = bundle_T.type(torch.float64).cpu()

    gt_T = gt_T.type(torch.float64).unsqueeze(0)
    pred_T = pred_T.type(torch.float64)

    r_erro = compute_rotation_error(pred_T, gt_T)
    t_erro = compute_translation_error_as_angle(pred_T, gt_T)

    # 计算重投影误差
    reproj_err_0 = couculate_reprojection_error(intr, pred_T, gt_pose0, gt_pose1, gt_T, m_kpts0, m_kpts1, confidence)

    return r_erro, t_erro, reproj_err_0, time.time() - time0


if __name__ == '__main__':
    # 初始化误差
    r_erro_totol = 0
    t_erro_totol = 0
    parser = argparse.ArgumentParser(description='Compute relative pose error')
    parser.add_argument('--datasets', type=str, default='scanet', help='datasets name')
    parser.add_argument('--datasets_dir', type=str, default='E:\Projects\datasets\scans_export\scans_export\scene0000_00', help='datasets directory include color, pose, intrinsic')
    parser.add_argument('--recover', type=str, default='', help='recover method')
    parser.add_argument('--match_pairs_frame', type=int, default=20, help='match pairs frame')
    parser.add_argument('--match_pairs', type=int, default=100, help='match pairs')
    args = parser.parse_args()
    totol = 0

    for i in range(args.match_pairs):
        # try:
        r_erro, t_erro, reproj_err_0, times = pairs_compute_RT_erro(args.datasets, args.recover, args.datasets_dir, (i+1) * args.match_pairs_frame)
        # except Exception as e:
        #     print(e)
        #     continue
        totol += 1
        print(f"进度:{i+1}/{args.match_pairs}")
        print(f"r_erro:{r_erro}, t_erro:{t_erro}, total_erro:{r_erro + t_erro}")
        print(f"reproj_err_0:{reproj_err_0.mean()}")
        t_erro_totol += t_erro
        r_erro_totol += r_erro
        print('time: ', times)
        print()
    print(f"final: r_erro:{r_erro_totol} + t_erro:{t_erro_totol} = total_erro:{r_erro_totol + t_erro_totol}")
    print(f"mean: r_erro:{r_erro_totol / totol} + t_erro:{t_erro_totol / totol} = total_erro:{(r_erro_totol + t_erro_totol) / totol}")
