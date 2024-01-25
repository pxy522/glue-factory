"""计算两张图象经过superpoint提取特征和lightglue匹配后的相对位姿误差"""

import torch

from kornia.geometry.epipolar import triangulate_points

from .two_view.estimate_relative_pose import estimate_relative_pose_w8pt, run_bundle_adjust_2_view, normalize
from .two_view.compute_pose_error import compute_rotation_error, compute_translation_error_as_angle
from .two_view.utils import estimate_pose


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


def compute_RT_erro(m_kpts0, m_kpts1, intr0, intr1, confidence, gt_T, recover):
    """计算旋转误差和平移误差
    Args:
        m_kpts0 (tensor): 匹配点
        m_kpts1 (tensor): 匹配点
        intr (tensor): 相机内参
        confidence (tensor): 置信度
        gt_T (tensor): 真实位姿
        recover (str): 优化方法

    Returns:
        r_erro (float): 旋转误差
        t_erro (float): 平移误差

    """
    # 初始化batch_size的pred_T, info存放字典
    pred_T = torch.zeros((intr0.shape[0],1, 4, 4))
    info = [{} for i in range(intr0.shape[0])]
    # 初始化误差矩阵
    r_erro = torch.zeros((intr0.shape[0]))
    t_erro = torch.zeros((intr0.shape[0]))

    if "w8pt" in recover:
        for i in range(intr0.shape[0]):
            try:
                pred_T[i], info[i] = estimate_relative_pose_w8pt(torch.tensor(m_kpts0[i]).cuda(), torch.tensor(m_kpts1[i]).cuda(), intr0[i], intr1[i], confidence[i])
            except:
                print("w8pt error")
                pred_T[i] = torch.eye(4).unsqueeze(0)
                continue

            # 优化pred_T
            bundle_T, _ = run_bundle_adjust_2_view(info[i]["kpts0_norm"], info[i]["kpts1_norm"], confidence[i].unsqueeze(0), pred_T[i], \
                        n_iterations=10)
            pred_T[i] = bundle_T.clone().detach()
    # TODO: 优化方法-opencv
    # else:
    #     print("---------optimize by bundle_adjustment in opencv---------")
    #     for i in range(intr0.shape[0]):
    #         ret = estimate_pose(m_kpts0[i].cpu().numpy().squeeze(0), m_kpts1[i].cpu().numpy().squeeze(0), intr0[i].cpu().numpy().squeeze(0), intr1.cpu().numpy().squeeze(0), thresh=1)
    #         pred_T = torch.eye(4).unsqueeze(0)
    #         pred_T[0, :3, :3] = torch.from_numpy(ret[0]).unsqueeze(0)
    #         pred_T[0, :3, 3] = torch.from_numpy(ret[1]).unsqueeze(0)

    #         assert pred_T is not None, 'Failed to estimate pose'
    #         print(f"ret_shape:{pred_T.shape}")

    #         # normalize
    #         norm_kpts0 = normalize(m_kpts0, intr).cpu()
    #         norm_kpts1 = normalize(m_kpts1, intr).cpu()

    #         bundle_T = run_bundle_adjust_2_view(norm_kpts0, norm_kpts1, confidence, pred_T, n_iterations=10)[0]
    #         pred_T = bundle_T.type(torch.float64).cpu()
    
    gt_T = gt_T.type(torch.float64).unsqueeze(0) # gt_T : B x 1 x 4 x 4
    # 将gt_T: B x 12 转换为 B x 1 x 4 x 4 
    gt_T = gt_T.view(-1, 1, 3, 4)
    gt_T = torch.cat((gt_T, torch.tensor([0, 0, 0, 1], dtype=torch.float64).cuda().unsqueeze(0).unsqueeze(0).repeat(gt_T.shape[0], 1, 1, 1)), dim=2)
    gt_T = gt_T.type(torch.float64).cuda()
    pred_T = pred_T.type(torch.float64).cuda()

    for i in range(intr0.shape[0]):
        # 旋转误差
        r_erro[i] = compute_rotation_error(pred_T[i], gt_T[i])
        # 平移误差
        t_erro[i] = compute_translation_error_as_angle(pred_T[i], gt_T[i])

    return r_erro , t_erro

