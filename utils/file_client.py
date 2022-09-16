# 读取图像数据的函数，先写了从disk读的函数，后面再补充从nori读的版本
import mmcv
import torch
import numpy as np

def load_img(img_path):
    """
    Load RGB image via file path and convert it to tensor.

    Args:
        img_path (str)
    Return:
        img_tensor (Tensor): (c, h, w)
    """
    with open(img_path, 'rb') as f:
        value_buf = f.read()
    
    img_ndarray = mmcv.imfrombytes(
        value_buf,
    )   # 这样读出来是 uint8 类型

    img_ndarray = img_ndarray / 255.
    img_ndarray = img_ndarray.astype(np.float32).transpose(2, 0, 1)
    img_tensor = torch.from_numpy(img_ndarray)

    return img_tensor