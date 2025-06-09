import torch
import torch.nn.functional as F
import numpy as np
import math

def generate_tps_grid(H, W, grid_shape, device):
    W_grid, H_grid = grid_shape
    x = torch.linspace(0, W_grid-1, W_grid, device=device)#作用是生成一个从0到W_grid-1的等差数列
    y = torch.linspace(0, H_grid-1, H_grid, device=device)#作用是生成一个从0到H_grid-1的等差数列
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')#生成两个尺寸为(H_grid, W_grid)的网格
    grid = torch.stack([grid_x, grid_y], dim=-1)#grid的形状为(H_grid, W_grid, 2)
    return grid

def tps_transform(target_grid, weights, source_points):
    # 确保 target_grid 和 weights 是 3D 张量
    if target_grid.dim() == 2:
        target_grid = target_grid.unsqueeze(0)
    if weights.dim() == 2:
        weights = weights.unsqueeze(0)
    # 调整 weights 的形状
    weights = weights.view(1, -1, 2)
    transformed_grid = torch.bmm(target_grid, weights)
    return transformed_grid

def tps_augmentation(img, num_points, std):
    # 获取图像尺寸
    C, H, W = img.shape#【3，256，256】
    
    # 生成TPS网格
    grid = generate_tps_grid(H, W, (W, H), img.device)
    
    # TPS 变换的实现
    target_grid = grid.view(-1, 2)#【65536，2】
    source_points = torch.rand(num_points, 2, device=img.device) * torch.tensor([W, H], device=img.device)
    weights = torch.randn(num_points, 2, device=img.device)
    
    # 确保 weights 的形状正确
    if weights.size(0) != target_grid.size(1):
        weights = weights[:target_grid.size(1), :]
    
    transformed_grid = tps_transform(target_grid, weights, source_points)
    transformed_grid = transformed_grid.view(H, W, 2)
    
    # 使用 grid_sample 进行图像变换
    transformed_img = torch.nn.functional.grid_sample(img.unsqueeze(0), transformed_grid.unsqueeze(0), align_corners=True)
    return transformed_img.squeeze(0)
