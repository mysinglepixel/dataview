from Preprocess import AnimeDiffusionDataset
from Pretrain import train
from config import Config
import os
from PIL import Image
import torch
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
from Pretrain import train_unet
import sys
sys.path.append("C:\\Users\\SinglePixel\\Desktop\\香港理工\\元宇宙项目\\My code")
import Fineturn

def train_fine_turn(model, dataloader, optimizer, criterion, num_epochs=100):
    latest_checkpoint = "./checkpoints/unet_checkpoint_20250310-022332.pth"
    if os.path.exists(latest_checkpoint):
        model.load_state_dict(torch.load(latest_checkpoint))
    else:
        print("No checkpoint found, training from scratch.")
    Fineturn.fine_tune_unet(model, dataloader, optimizer, num_epochs)

if __name__ == "__main__":
    dataset = AnimeDiffusionDataset("C:/Users/SinglePixel/Desktop/香港理工/AnimeDiffusion Dataset/train_data/reference")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)  # 将 batch_size 从 8 减少到 4

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化 U-Net
    unet = UNet().to(device)
    optimizer = optim.AdamW(unet.parameters(), lr=1e-4)

    # 损失函数（MSE Loss）
    criterion = nn.MSELoss()
    
    # 训练模型
    train_unet(unet, dataloader, optimizer, criterion, num_epochs=100)
    #train_fine_turn(unet, dataloader, optimizer, criterion, num_epochs=100)