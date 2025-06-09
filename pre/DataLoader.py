import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from PIL import Image

def load_custom_data(data_root, batch_size):
    """
    加载自定义数据集，要求目录结构为：
    data_root/
        class_0/
            img1.jpg
            img2.jpg
            ...
        class_1/
            img1.jpg
            ...
        ...
    """
    
    # 数据预处理流程
    transform = transforms.Compose([
        transforms.Resize(256),          # 调整图像大小
        transforms.CenterCrop(256),      # 中心裁剪
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
 # 加载数据集
    full_dataset = datasets.ImageFolder(
        root=data_root,
        transform=transform
    )

    # 输出数据结构信息
    print("\n=== 数据集结构分析 ===")
    print(f"数据集根目录: {data_root}")
    print(f"总类别数: {len(full_dataset.classes)}")
    print(f"类别到索引的映射: {full_dataset.class_to_idx}")
    
    # 统计每个类别的样本数量
    class_counts = {class_name: 0 for class_name in full_dataset.classes}
    for root, dirs, files in os.walk(data_root):
        if os.path.basename(root) in full_dataset.class_to_idx:
            class_name = os.path.basename(root)
            class_counts[class_name] = len([
                f for f in files 
                if f.lower().endswith(('png', 'jpg', 'jpeg'))
            ])

    print("\n各类别样本数量:")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count} 张")

    # 显示样本图像信息
    sample_img_path = os.path.join(
        data_root,
        full_dataset.classes[0],  # 取第一个类别
        os.listdir(os.path.join(data_root, full_dataset.classes[0]))[0]
    )
    with Image.open(sample_img_path) as img:
        print(f"\n样本图像尺寸 (原始): {img.size}")
        print(f"处理后 Tensor 形状: {transform(img).shape}")

    # 划分训练集和测试集（按98:2划分）
    train_size = int(0.98 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size]
    )

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, test_loader