import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from pre.XDog import XDog
from pre.StrongerTPS import tps_augmentation

def preprocess_images(source_dir, 
                     line_art_dir, 
                     ref_img_dir, 
                     real_img_dir,
                     img_size=256):
    """
    预处理并存储训练所需的三种图像
    参数：
        source_dir: 原始图像目录
        line_art_dir: 线稿存储目录
        ref_img_dir: TPS扭曲图存储目录
        real_img_dir: 原图存储目录
        img_size: 图像尺寸
    """
    # 创建输出目录
    os.makedirs(line_art_dir, exist_ok=True)
    os.makedirs(ref_img_dir, exist_ok=True)
    os.makedirs(real_img_dir, exist_ok=True)

    # 获取所有图像文件
    img_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for filename in tqdm(img_files, desc="Processing images"):
        # 原始图像路径
        src_path = os.path.join(source_dir, filename)
        
        try:
            # 读取并统一图像尺寸
            img = Image.open(src_path).convert('RGB')
            img = img.resize((img_size, img_size))
            
            # 保存原图副本
            real_path = os.path.join(real_img_dir, filename)
            img.save(real_path)

            # 生成并保存线稿（单通道）
            line_art = XDog(img) 
            line_art_path = os.path.join(line_art_dir, os.path.splitext(filename)[0] + '.png')
            line_art.convert('L').save(line_art_path)  # 保存为单通道PNG

            # 生成并保存TPS扭曲图（三通道）
            ref_img = tps_augmentation(img)
            ref_path = os.path.join(ref_img_dir, filename)
            ref_img.convert('RGB').save(ref_path)

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue

if __name__ == "__main__":
    # 使用示例
    preprocess_images(
        source_dir="./data/train/airplane",
        line_art_dir="./preprocessed/line_art",
        ref_img_dir="./preprocessed/ref_img",
        real_img_dir="./preprocessed/real_img",
        img_size=256
    )