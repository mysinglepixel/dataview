import math
import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage
import torch
def to_grayscale(img):
    """
    将 (3, H, W) 的彩色图像转换为 (1, H, W) 的灰度图像。
    
    Args:
        img (torch.Tensor): 输入图像，形状 (3, H, W)
        
    Returns:
        torch.Tensor: 灰度图像，形状 (1, H, W)
    """
    if img.shape[0] != 3:
        raise ValueError("输入图像必须为3通道彩色图")
    r, g, b = img[0], img[1], img[2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray.unsqueeze(0)

def gaussian_kernel(sigma, kernel_size=None, device='cpu'):
    """
    生成二维高斯核。
    
    Args:
        sigma (float): 高斯核的标准差。
        kernel_size (int, optional): 核尺寸。如果为 None，则自动计算为 2*ceil(3*sigma)+1。
        device (str): 使用的设备，如 'cpu' 或 'cuda'。
    
    Returns:
        torch.Tensor: 高斯核，形状为 (1, 1, kernel_size, kernel_size)
    """
    if kernel_size is None:
        kernel_size = int(2 * math.ceil(3 * sigma) + 1)
    x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, 
                      dtype=torch.float32, device=device)
    gauss = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    kernel2d = torch.outer(gauss, gauss)
    return kernel2d.unsqueeze(0).unsqueeze(0)

def gaussian_blur(img, sigma, device='cpu'):
    """
    对灰度图像应用高斯模糊。
    
    Args:
        img (torch.Tensor): 灰度图，形状 (1, H, W)
        sigma (float): 高斯核标准差。
        device (str): 设备类型。
    
    Returns:
        torch.Tensor: 模糊后的图像，形状 (1, H, W)
    """
    img = img.unsqueeze(0)  # 增加 batch 维度，变为 (1, 1, H, W)
    kernel = gaussian_kernel(sigma, device=device)
    padding = kernel.shape[-1] // 2
    blurred = F.conv2d(img, kernel, padding=padding)
    return blurred.squeeze(0)

def xdog(input_img, sigma=0.5, k=1.6, p=10.0, threshold=0.3, device='cpu'):
    input_img = input_img.to(device)
    gray = to_grayscale(input_img)
    blur1 = gaussian_blur(gray, sigma, device=device)
    blur2 = gaussian_blur(gray, sigma * k, device=device)

    xdog_result = (1 + p) * blur1 - p * blur2
    xdog_norm = (xdog_result - xdog_result.min()) / (xdog_result.max() - xdog_result.min() + 1e-9)
    line_drawing = (xdog_norm > threshold).float()

    return line_drawing

# 示例：直接调用 xdog 方法处理单张 (3,256,256) 的图像
def XDog(raw_img):

    tensor_img = ToTensor()(raw_img)  # 自动归一化到[0,1]
    
    # 获取设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 执行XDoG处理
    edge_tensor = xdog(tensor_img.to(device), 
                      sigma=0.3, 
                      k=4.5,
                      p=9.0,
                      threshold=0.45,
                      device=device)
    
    # 转换回PIL图像 (注意：输出是[1, H, W]的0/1二值图)
    edge_pil = ToPILImage()(edge_tensor.cpu().squeeze(0))  # 移除通道维度
    
    # 确保输出是二值图像（可选步骤）
    return edge_pil.convert('1')  # 转换为二值模式