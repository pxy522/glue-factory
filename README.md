# Glue Factory
## 第一次训练
命令行语句
`python -m gluefactory.train sp+lg_megadepth  --conf gluefactory/configs/superpoint+lightglue_megadepth.yaml train.load_experiment=sp+lg_homography`
### 使用的损失函数为：均方差损失（MSELoss）
### 计算旋转矩阵误差：
`def compute_rotation_error(T0, T1, reduce=True):
    # 矩阵的迹不会因为矩阵的旋转而改变，所以计算迹的误差就是计算旋转矩阵的误差
    # use diagonal and sum to compute trace of a batch of matrices
    cos_a = ((T0[..., :3, :3].transpose(-1, -2) @ T1[..., :3, :3]).diagonal(offset=0, dim1=-1, dim2=-2).sum(-1) \
        - 1.) / 2.
    print(f"min_cos_a:{cos_a.min()}")
    print(f"max_cos_a:{cos_a.max()}")
    cos_a = torch.clamp(cos_a, -1., 1.) # avoid nan
    abs_acos_a = torch.abs(torch.arccos(cos_a))
    if reduce:
        return abs_acos_a.mean()
    else:
        return abs_acos_a`
### 计算平移矩阵误差：
`
def compute_translation_error_as_angle(T0, T1, reduce=True):
    """
    计算平移矩阵误差
    Args:
        T0: (..., 4, 4)
        T1: (..., 4, 4)
    math:
        err = arccos( (t0.dot(t1)) / (|t0| * |t1|) )
    """
    # 计算平移向量的模长
    n = torch.linalg.norm(T0[..., :3, 3], dim=-1) * torch.linalg.norm(T1[..., :3, 3], dim=-1)
    # 找出模长大于1e-6的,避免除0
    valid_n = n > 1e-6
    # 计算两个平移向量的点积
    T0_dot_T1 = (T0[..., :3, 3][valid_n] * T1[..., :3, 3][valid_n]).sum(-1)
    err = torch.abs(torch.arccos((T0_dot_T1 / n[valid_n]).clamp(-1., 1.)))
    if reduce:
        return err.mean()
    else:
        return err
`
### 处理得到的旋转误差(R_loss)与平移误差(T_loss): 将R_loss、T_loss分别取sin(), 再取log
`
    R_loss = 1 - torch.sin(R_loss / 2)
    T_loss = 1 - torch.sin(T_loss / 2)

    R_loss = R_loss.unsqueeze(-1).cuda()
    T_loss = T_loss.unsqueeze(-1).cuda()
    # B x 2048 和 B x 1 进行数乘，对计算结果除2
    RT_logit0 = (torch.sum(logit0[:] * R_loss[:], -1)/2048  + torch.sum(logit0[:] * T_loss[:], -1) / 2048) / 2
`
