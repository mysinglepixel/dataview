import torch

class Config:
    # 数据参数
    input_size = 256
    batch_size = 32
    num_workers = 8
    
    # 训练参数
    epochs = 1
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 路径配置
    data_root = "../data/train"
    model_save_path = "./highresnet.pth"

config = Config()