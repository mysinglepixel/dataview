import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import math
from XDog import XDog
#from pre.StrongerTPS import tps_augmentation
from TPS import apply_tps
from Pretrain import train
from config import Config
# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AnimeDiffusionDataset(Dataset):
    def __init__(self, img_dir, img_size=256):
        """
        动态生成训练数据的Dataset类
        参数：
            img_dir: 原始图像目录
            img_size: 统一图像尺寸
        """
        self.img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.img_dir = img_dir
        self.img_size = img_size
        
        # 基础图像变换
        self.base_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
        ])
        
        # TPS参数范围
        self.tps_params = {
            'num_points_range': (9, 25),  # 控制点数量范围（平方数）
            'std_range': (0.02, 0.15)     # 变形强度范围
        }

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # 加载原始图像
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        real_img = Image.open(img_path).convert('RGB')
        
        # 基础处理
        real_img = self.base_transform(real_img)
        
        # 实时生成线稿
        line_art = self.xdog_process(real_img)
        
        # 将 PIL 图像转换为 Tensor
        real_img_tensor = transforms.ToTensor()(real_img)
        
        # 实时生成TPS扭曲图像
        #ref_img = self.tps_process(real_img_tensor)
        ref_img = apply_tps(real_img_tensor)
        # 转换为Tensor
        return (
            self.img_to_tensor(line_art, channels=1),  # 线稿 [1,H,W]
            ref_img,   # 参考图 [3,H,W] 已经是 Tensor
            real_img_tensor   # 原图 [3,H,W] 已经是 Tensor
        )

    def xdog_process(self, img):
        """实时XDoG线稿生成"""
        # XDoG处理
        edge_tensor = XDog(img)  # 传递 PIL 图像
        
        # 直接返回 PIL 图像
        return edge_tensor

    #def tps_process(self, img):
        """实时TPS几何扭曲"""
        # 随机参数
        num_points = np.random.randint(self.tps_params['num_points_range'][0], self.tps_params['num_points_range'][1])
        num_points = int(math.sqrt(num_points))**2  # 确保是平方数
        std = np.random.uniform(self.tps_params['std_range'][0], self.tps_params['std_range'][1])
        
        # 执行TPS变换
        return tps_augmentation(img, num_points,std)

    def img_to_tensor(self, img, channels=3):
        """统一图像到Tensor的转换"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5]*channels,
                std=[0.5]*channels
            )
        ])
        return transform(img)

# 使用示例
if __name__ == "__main__":
    dataset = AnimeDiffusionDataset("C:/Users/SinglePixel/Desktop/香港理工/AnimeDiffusion Dataset/train_data/reference")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

    # 验证数据格式
    save_dir = "./output_images"
    os.makedirs(save_dir, exist_ok=True)
    count = 0
    train(Config.model, dataloader, device, 1)
    for batch in dataloader:
        line_art, ref_img, real_img = batch
        print(f"线稿尺寸: {line_art.shape}")   # 应输出 torch.Size([8, 1, 256, 256])
        print(f"参考图尺寸: {ref_img.shape}")   # 应输出 torch.Size([8, 3, 256, 256])
        print(f"原图尺寸: {real_img.shape}")    # 应输出 torch.Size([8, 3, 256, 256])

        # 保存每种图片的两张
        for i in range(2):
            transforms.ToPILImage()(line_art[i]).save(os.path.join(save_dir, f"line_art_{count}.png"))
            transforms.ToPILImage()(ref_img[i]).save(os.path.join(save_dir, f"ref_img_{count}.png"))
            transforms.ToPILImage()(real_img[i]).save(os.path.join(save_dir, f"real_img_{count}.png"))
            count += 1

        break