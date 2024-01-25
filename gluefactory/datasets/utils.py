import cv2
import numpy as np
import torch
import os


def read_image(path, grayscale=False):
    """Read an image from path as RGB or grayscale"""
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise IOError(f"Could not read image at {path}.")
    if not grayscale:
        image = image[..., ::-1]
    return image


def numpy_image_to_torch(image):
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return torch.tensor(image / 255.0, dtype=torch.float)


def rotate_intrinsics(K, image_shape, rot):
    """image_shape is the shape of the image after rotation"""
    assert rot <= 3
    h, w = image_shape[:2][:: -1 if (rot % 2) else 1]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    rot = rot % 4
    if rot == 1:
        return np.array(
            [[fy, 0.0, cy], [0.0, fx, w - cx], [0.0, 0.0, 1.0]], dtype=K.dtype
        )
    elif rot == 2:
        return np.array(
            [[fx, 0.0, w - cx], [0.0, fy, h - cy], [0.0, 0.0, 1.0]],
            dtype=K.dtype,
        )
    else:  # if rot == 3:
        return np.array(
            [[fy, 0.0, h - cy], [0.0, fx, cx], [0.0, 0.0, 1.0]], dtype=K.dtype
        )


def rotate_pose_inplane(i_T_w, rot):
    rotation_matrices = [
        np.array(
            [
                [np.cos(r), -np.sin(r), 0.0, 0.0],
                [np.sin(r), np.cos(r), 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        for r in [np.deg2rad(d) for d in (0, 270, 180, 90)]
    ]
    return np.dot(rotation_matrices[rot], i_T_w)


def scale_intrinsics(K, scales):
    """Scale intrinsics after resizing the corresponding image."""
    scales = np.diag(np.concatenate([scales, [1.0]]))
    return np.dot(scales.astype(K.dtype, copy=False), K)


def get_divisible_wh(w, h, df=None):
    if df is not None:
        w_new, h_new = map(lambda x: int(x // df * df), [w, h])
    else:
        w_new, h_new = w, h
    return w_new, h_new


def resize(image, size, fn=None, interp="linear", df=None):
    """Resize an image to a fixed size, or according to max or min edge."""
    h, w = image.shape[:2]
    if isinstance(size, int):
        scale = size / fn(h, w)
        h_new, w_new = int(round(h * scale)), int(round(w * scale))
        w_new, h_new = get_divisible_wh(w_new, h_new, df)
        scale = (w_new / w, h_new / h)
    elif isinstance(size, (tuple, list)):
        h_new, w_new = size
        scale = (w_new / w, h_new / h)
    else:
        raise ValueError(f"Incorrect new size: {size}")
    mode = {
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "nearest": cv2.INTER_NEAREST,
        "area": cv2.INTER_AREA,
    }[interp]
    return cv2.resize(image, (w_new, h_new), interpolation=mode), scale


def crop(image, size, random=True, other=None, K=None, return_bbox=False):
    """Random or deterministic crop of an image, adjust depth and intrinsics."""
    h, w = image.shape[:2]
    h_new, w_new = (size, size) if isinstance(size, int) else size
    top = np.random.randint(0, h - h_new + 1) if random else 0
    left = np.random.randint(0, w - w_new + 1) if random else 0
    image = image[top : top + h_new, left : left + w_new]
    ret = [image]
    if other is not None:
        ret += [other[top : top + h_new, left : left + w_new]]
    if K is not None:
        K[0, 2] -= left
        K[1, 2] -= top
        ret += [K]
    if return_bbox:
        ret += [(top, top + h_new, left, left + w_new)]
    return ret


def zero_pad(size, *images):
    """zero pad images to size x size"""
    ret = []
    for image in images:
        if image is None:
            ret.append(None)
            continue
        h, w = image.shape[:2]
        padded = np.zeros((size, size) + image.shape[2:], dtype=image.dtype)
        padded[:h, :w] = image
        ret.append(padded)
    return ret


# 创建txt文件, 写入图片对路径
def create_txt(dir):
    """
    创建txt文件, 写入图片对路径
    :return: None
    """
    # 创建txt文件
    txt_dir = dir
    with open(txt_dir, "w") as f:
        for i in range(40):
            for j in range(25):
                f.write(f"frame{49*i+j+i:0>6d} frame{49*(i+1)+i-j:0>6d} depth{49*i+j+i:0>6d} depth{49*(i+1)+i-j:0>6d}\n")
                f.write(f"frame{49*(i+1)-j+i:0>6d} frame{49*i+i+j:0>6d} depth{49*(i+1)+i-j:0>6d} depth{49*i+j+i:0>6d}\n")

def split_jpg_png(file_path):
    """
    使用os.walk获取文件夹下的所有文件名
    :param file_path: 文件路径

    """
    for root, dirs, files in os.walk(file_path):
        for file in files:
            if file.endswith(".jpg"):
                #移动到col文件夹下, col文件夹不存在则创建,col文件夹在file_path上一级目录下
                col_dir = os.path.join(file_path, "..", "col")
                if not os.path.exists(col_dir):
                    os.mkdir(col_dir)
                    print("col文件夹不存在, 已创建")
                # 移动文件
                old_name = os.path.join(file_path, file)
                new_name = os.path.join(col_dir, file)
                os.rename(old_name, new_name)

            if file.endswith(".png"):
                # 移动到dep文件夹下, dep文件夹不存在则创建,dep文件夹在file_path上一级目录下
                dep_dir = os.path.join(file_path, "..", "dep")
                if not os.path.exists(dep_dir):
                    os.mkdir(dep_dir)
                    print("dep文件夹不存在, 已创建")
                # 移动文件
                old_name = os.path.join(file_path, file)
                new_name = os.path.join(dep_dir, file)
                os.rename(old_name, new_name)

