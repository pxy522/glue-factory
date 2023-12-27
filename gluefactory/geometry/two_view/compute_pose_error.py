import torch

"""传入变换矩阵,计算误差"""

# 计算旋转矩阵误差
def compute_rotation_error(T0, T1, reduce=True):
    # use diagonal and sum to compute trace of a batch of matrices
    cos_a = ((T0[..., :3, :3].transpose(-1, -2) @ T1[..., :3, :3]).diagonal(offset=0, dim1=-1, dim2=-2).sum(-1) \
        - 1.) / 2.
    cos_a = torch.clamp(cos_a, -1., 1.) # avoid nan
    abs_acos_a = torch.abs(torch.arccos(cos_a))
    if reduce:
        return abs_acos_a.mean()
    else:
        return abs_acos_a

# 计算平移矩阵误差
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
    # 找出模长大于1e-6的, 避免除0
    valid_n = n > 1e-6
    # 计算两个平移向量的点积
    T0_dot_T1 = (T0[..., :3, 3][valid_n] * T1[..., :3, 3][valid_n]).sum(-1)
    err = torch.abs(torch.arccos((T0_dot_T1 / n[valid_n]).clamp(-1., 1.)))
    if reduce:
        return err.mean()
    else:
        return err
