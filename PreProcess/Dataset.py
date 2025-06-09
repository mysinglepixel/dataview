import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
from torchvision.transforms import ToTensor, ToPILImage, transforms
import random
import math
from PIL import Image

def generate_tps_grid(height, width, tps_params, device='cpu'):
    """
    生成TPS变形网格（适配PyTorch的grid_sample）
    
    参数：
        height: 输出图像高度
        width: 输出图像宽度
        tps_params: 元组 (W, A)
        device: 计算设备
    
    返回：
        grid: [H, W, 2] 归一化坐标网格
    """
    W, A = tps_params
    N = W.size(0)
    
    # 生成归一化网格坐标（PyTorch的grid_sample要求[-1,1]范围）
    x = torch.linspace(-1, 1, width, device=device)  # 宽度方向
    y = torch.linspace(-1, 1, height, device=device) # 高度方向
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')  # [H, W]
    flat_grid = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)  # [H*W, 2]

    # 计算径向基函数项
    source_points = A[:, 1:]  # 源控制点 [3,2] 中的后两列
    diff = flat_grid.unsqueeze(1) - source_points.unsqueeze(0)  # [H*W, N, 2]
    r = torch.norm(diff, dim=2)  # [H*W, N]
    U = (r ** 2) * torch.log(r + 1e-9)  # [H*W, N]

    # 计算仿射变换项
    A_term = torch.mm(
        torch.cat([
            torch.ones((flat_grid.size(0), 1), device=device),  # 齐次项
            flat_grid
        ], dim=1),  # [H*W, 3]
        A  # [3, 2]
    )  # [H*W, 2]

    # 合成最终变换坐标
    transformed = torch.mm(U, W) + A_term  # [H*W, 2]
    
    # 重构为网格格式 [H, W, 2]
    return transformed.view(height, width, 2)

def compute_tps_transform(source, target, reg=1e-6):
    """
    完整TPS参数计算函数（支持批量处理）
    
    参数：
        source: [N, 2] 源控制点坐标 (x,y)
        target: [N, 2] 目标控制点坐标
        reg: 正则化系数
    
    返回：
        W: [N, 2] 径向基权重
        A: [3, 2] 仿射变换矩阵
    """
    N = source.size(0)
    device = source.device

    # 构造K矩阵（TPS核函数）
    diff = source.unsqueeze(1) - source.unsqueeze(0)  # [N, N, 2]
    r = torch.norm(diff, dim=2)  # [N, N]
    K = (r ** 2) * torch.log(r + 1e-9)  # [N, N]

    # 构造P矩阵 [N, 3]
    P = torch.cat([
        torch.ones(N, 1, device=device),  # 齐次项
        source
    ], dim=1)  # [N, 3]

    # 构造L矩阵 [[K, P], [P^T, 0]] + 正则化
    L = torch.zeros(N+3, N+3, device=device)
    L[:N, :N] = K
    L[:N, N:N+3] = P
    L[N:N+3, :N] = P.t()
    L += torch.eye(N+3, device=device) * reg  # 正则化项

    # 构造Y矩阵 [N+3, 2]
    Y = torch.cat([
        target,  # [N, 2]
        torch.zeros(3, 2, device=device)
    ], dim=0)  # [N+3, 2]

    # 求解线性方程组（稳定实现）
    theta = torch.linalg.lstsq(L, Y).solution

    return theta[:N], theta[N:N+3]  # W [N,2], A [3,2]

class DynamicTPSTransform:
    """动态TPS变换类，集成到Dataset中"""
    def __init__(self, 
                 control_points_range=(9, 25), 
                 std_range=(0.02, 0.15)):
        """
        参数：
            control_points_range: 控制点数量范围（随机选择平方数）
            std_range: 变形强度范围
        """
        self.cp_range = control_points_range
        self.std_range = std_range

    def __call__(self, img):
        # 随机选择参数
        num_points = random.choice([
            i**2 for i in range(
                int(math.sqrt(self.cp_range[0])),
                int(math.sqrt(self.cp_range[1]))+1
            )
        ])
        std = random.uniform(*self.std_range)
        
        return tps_augmentation(
            img,
            num_points=num_points,
            std=std,
            deterministic=False
        )

def tps_augmentation(img, num_points=9, std=0.05, deterministic=False):
    """改进后的TPS函数"""
    # 参数设置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 输入处理
    if isinstance(img, Image.Image):
        img_tensor = ToTensor()(img).unsqueeze(0).to(device)
    else:
        img_tensor = img.unsqueeze(0).to(device) if img.dim()==3 else img.to(device)
    
    B, C, H, W = img_tensor.shape
    
    # 生成随机控制点（确保包含边界）
    grid_size = int(math.sqrt(num_points))
    x = torch.linspace(0, W-1, grid_size, device=device)
    y = torch.linspace(0, H-1, grid_size, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
    source = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)
    
    # 生成随机位移
    if not deterministic:
        displacement = torch.randn_like(source) * std * min(H,W)
        target = source + displacement
    else:
        target = source  # 验证模式时不形变

    # 计算TPS参数
    W, A = compute_tps_transform(source, target)
    
    # 生成采样网格
    grid = generate_tps_grid(H, W, (W, A), device)
    
    # 执行采样
    warped = F.grid_sample(
        img_tensor, 
        grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )
    
    # 输出处理
    if isinstance(img, Image.Image):
        return ToPILImage()(warped.squeeze().cpu())
    return warped.squeeze(0) if img.dim()==3 else warped

class AnimeDiffusionDataset(Dataset):
    def __init__(self, real_dir, line_art_dir, transform=None):
        """
        新版Dataset结构：
        - real_dir: 原图目录
        - line_art_dir: 线稿目录
        - transform: 动态TPS变换
        """
        self.filenames = sorted(os.listdir(real_dir))
        self.real_dir = real_dir
        self.line_art_dir = line_art_dir
        self.tps_transform = DynamicTPSTransform()
        
        # 基础转换
        self.base_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
        ])
        self.normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5], 
            std=[0.5, 0.5, 0.5]
        )

    def __getitem__(self, idx):
        # 加载原始数据
        real_img = Image.open(
            os.path.join(self.real_dir, self.filenames[idx])
        ).convert('RGB')
        line_art = Image.open(
            os.path.join(self.line_art_dir, self.filenames[idx])
        ).convert('L')

        # 基础处理
        real_img = self.base_transform(real_img)
        line_art = self.base_transform(line_art)
        
        # 动态生成扭曲参考图
        ref_img = self.tps_transform(real_img)  # 应用随机TPS
        
        # 转换为Tensor
        return (
            transforms.ToTensor()(line_art),   # 线稿 [1,256,256]
            transforms.ToTensor()(ref_img),    # 参考图 [3,256,256]
            transforms.ToTensor()(real_img)    # 原图 [3,256,256]
        )

    def __len__(self):
        return len(self.filenames)