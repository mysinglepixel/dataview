import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
#from torchvision.datasets import Dataset
import math
from DDPM import DDPM
import torch.optim as optim
import os
import torch
from config import Config
from torch.utils.data import DataLoader
from torchvision import transforms
from UNet import UNet
import torch.optim as optim
import torch.nn as nn
from DDPM import DDPM
import glob
import os
import csv
import time

log_file = "./training_log.csv"
device = Config.device
# 训练循环
def train_unet(model, dataloader, optimizer, criterion, num_epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 查找最新的 checkpoint
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "unet_checkpoint_*.pth"))
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)  # 按创建时间排序
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"Loaded latest checkpoint: {latest_checkpoint}")
    else:
        print("No checkpoint found, training from scratch.")

    # 确保日志文件存在
    if not os.path.exists(log_file):
        with open(log_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Epoch", "Loss"])  # 写入表头
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0

        for line_art, ref_img, real_img in dataloader:
            line_art, ref_img, real_img = line_art.to(device), ref_img.to(device), real_img.to(device)

            # 采样 t（时间步）
            batch_size = real_img.shape[0]
            t = torch.randint(0, 1000, (batch_size,), device=device).long()

            # 生成带噪声图像
            ddpm = DDPM(T=1000, device=device)
            xt, noise = ddpm.forward_process(real_img, t)

            # 组合 U-Net 输入
            input_tensor = torch.cat([line_art, ref_img, xt], dim=1)  # (B, 7, H, W)
            #print("input shape = ",input_tensor.shape)
            # 预测噪声
            pred_noise = model(input_tensor)

            # 计算 L2 损失
            loss = criterion(pred_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            # 清理 CUDA 缓存
            torch.cuda.empty_cache()
        avg_loss = total_loss / len(dataloader)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")  # 记录当前时间

        # 记录 loss 到 CSV
        with open(log_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, epoch + 1, avg_loss])

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Time: {timestamp}")

        # 训练完成后保存 checkpoint
        timestamp_filename = time.strftime("%Y%m%d-%H%M%S")
        checkpoint_path = os.path.join(checkpoint_dir, f"unet_checkpoint_{timestamp_filename}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved: {checkpoint_path}")

def train(model, dataloader, device, epochs=10):
    ddpm = DDPM()
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    #optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

   
    model.train()

    for epoch in range(epochs):
        for batch in dataloader:
            # 加载数据（假设数据格式为[线稿, 参考图, 真实彩色图]）
            #line_art = batch[0].to(device)       # [B,1,256,256]
            #ref_img = batch[1].to(device)        # [B,3,256,256]
            #real_img = batch[2].to(device)       # [B,3,256,256]
            line_art, ref_img, real_img = batch
            # 生成随机时间步
            t = torch.randint(0, ddpm.T, (line_art.size(0),), device=device)#生成batch_size个随机时间步,每个时间步是一个0到1000之间的整数，均匀分布
            
            # 前向扩散过程生成噪声图像
            noise_img, noise = ddpm.forward_process(real_img, t)#返回第t时刻的噪声图像和一个高斯噪音
            noise_img = noise_img.to(device)
            
            # 拼接输入（线稿1ch + 参考3ch + 噪声图3ch）
            model_input = torch.cat([line_art, ref_img, noise_img], dim=1)
            
            # 预测噪声
            pred_noise = model(model_input, t)
            
            # 计算损失
            loss = criterion(pred_noise, noise)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch+1}/{epochs} Loss: {loss.item():.4f}")

