import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

def apply_tps(image_tensor, grid_size=10, strength=0.02):
    _, H, W = image_tensor.shape

    # 生成标准网格点
    grid_x, grid_y = torch.meshgrid(
        torch.linspace(-1, 1, W), torch.linspace(-1, 1, H), indexing="xy"
    )

    # 生成扭曲的控制点
    delta_x = (torch.rand(grid_size, grid_size) * 2 - 1) * strength
    delta_y = (torch.rand(grid_size, grid_size) * 2 - 1) * strength

    # 生成控制点网格
    control_points_x = torch.linspace(-1, 1, grid_size)
    control_points_y = torch.linspace(-1, 1, grid_size)
    control_x, control_y = torch.meshgrid(control_points_x, control_points_y, indexing="xy")

    # 将控制点坐标和扭曲量转换为插值数据
    control_x = control_x.flatten()
    control_y = control_y.flatten()
    delta_x = delta_x.flatten()
    delta_y = delta_y.flatten()

    # 计算扭曲后的新坐标
    grid_x_flat = grid_x.flatten()
    grid_y_flat = grid_y.flatten()
    new_grid_x = grid_x_flat.clone()
    new_grid_y = grid_y_flat.clone()

    # 进行 Thin-Plate Splines 插值
    for i in range(len(control_x)):
        r2 = (grid_x_flat - control_x[i])**2 + (grid_y_flat - control_y[i])**2
        r2[r2 == 0] = 1e-6  # 避免除零错误
        U = r2 * torch.log(r2)
        new_grid_x += U * delta_x[i]
        new_grid_y += U * delta_y[i]

    # 归一化并限制坐标范围
    new_grid_x = torch.clamp(new_grid_x, -1, 1).reshape(H, W)
    new_grid_y = torch.clamp(new_grid_y, -1, 1).reshape(H, W)

    # 生成新的采样网格
    grid = torch.stack((new_grid_x, new_grid_y), dim=-1).unsqueeze(0)  # (1, H, W, 2)

    # 进行网格采样
    transformed_image = torch.nn.functional.grid_sample(
        image_tensor.unsqueeze(0), grid, mode="bilinear", align_corners=True
    )

    return transformed_image.squeeze(0)  # 去掉 batch 维度
