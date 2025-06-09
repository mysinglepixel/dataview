import os
import pandas as pd
from PIL import Image

def preprocess_pipeline(raw_dir, output_dir):
    # 创建目录
    os.makedirs(os.path.join(output_dir, 'processed/line_art'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'processed/tps'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'processed/xdog'), exist_ok=True)
    
    # 获取原始图像列表
    raw_images = sorted([f for f in os.listdir(raw_dir) if f.endswith(('.jpg', '.png'))])
    
    index_data = []
    
    for img_name in raw_images:
        # 原始图像路径
        raw_path = os.path.join(raw_dir, img_name)
        base_name = os.path.splitext(img_name)[0]
        
        # 加载原始图像
        raw_img = Image.open(raw_path).convert('RGB')
        
        # 处理步骤（假设已实现）
        line_art = raw_img#process_dataloader(raw_img)   # -> 1通道
        tps_img = raw_img#TPS(raw_img)                   # -> 3通道
        xdog_img = raw_img#XDog(raw_img)                 # -> 3通道
        
        # 保存处理结果
        line_path = os.path.join(output_dir, f'processed/line_art/{base_name}.png')
        line_art.save(line_path)
        
        tps_path = os.path.join(output_dir, f'processed/tps/{base_name}.jpg')
        tps_img.save(tps_path)
        
        xdog_path = os.path.join(output_dir, f'processed/xdog/{base_name}.jpg')
        xdog_img.save(xdog_path)
        
        # 记录索引
        index_data.append({
            'original_path': raw_path,
            'line_art_path': line_path,
            'tps_path': tps_path,
            'xdog_path': xdog_path
        })
    
    # 保存索引文件
    pd.DataFrame(index_data).to_csv(os.path.join(output_dir, 'index.csv'), index=False)

if __name__ == "__main__":
    preprocess_pipeline('./data/train/airplane', './')