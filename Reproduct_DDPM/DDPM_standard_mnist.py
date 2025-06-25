import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.nn import Module, ModuleList
import math
from einops import rearrange, reduce,repeat
from einops.layers.torch import Rearrange
from functools import partial
import torch.nn.functional as F
from torch import nn, einsum, Tensor
from attend import Attend, AttentionConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
T = 1000          
beta_start = 0.0001
beta_end = 0.02
beta = torch.linspace(beta_start, beta_end, T).to(device)
alpha = 1. - beta
alpha_bar = torch.cumprod(alpha, dim=0)
sqrt_alpha_bar = torch.sqrt(alpha_bar)
sqrt_one_minus_alpha_bar = torch.sqrt(1. - alpha_bar)

class CustomMNIST(torch.utils.data.Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32) / 255.0
        if self.transform:
            img = self.transform(img)
        # 返回元组 (image, dummy_label) 以保持接口一致
        return img, 0  
def load_mnist_images(path):
    with open(path, 'rb') as f:
        f.read(16)
        # 读取图像数据（60000张28x28图像）
        data = np.frombuffer(f.read(), dtype=np.uint8)#以uint8二进制格式读取后规范形状
        return data.reshape(-1, 28, 28)

def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    device = t.device
    vals = vals.to(device)
    tt=t.long()
    out = vals.gather(-1, tt)
    return out.reshape(batch_size, *((1,)*(len(x_shape)-1)))

def forward_diffusion_sample(x_0, t, device):
    x_0 = x_0.to(device)
    t = t.to(device)
    noise = torch.randn_like(x_0)
    sqrt_alpha_bar_t = get_index_from_list(sqrt_alpha_bar, t, x_0.shape)
    sqrt_one_minus_alpha_bar_t = get_index_from_list(sqrt_one_minus_alpha_bar, t, x_0.shape)
    #输入x_0和t，返回噪声图像和当前噪声
    return sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise, noise

@torch.no_grad()
def sample_timestep(x, t,model):
    """
    从x_t生成x_{t-1}
    """
    # 预测噪声
    predicted_noise = model(x, t.float())
    
    # 计算系数
    beta_t = get_index_from_list(beta, t, x.shape)
    sqrt_recip_alpha = 1.0 / torch.sqrt(get_index_from_list(alpha, t, x.shape))
    sqrt_one_minus_alpha_bar_t = get_index_from_list(sqrt_one_minus_alpha_bar, t, x.shape)
    
    # 计算x_{t-1}的核心部分
    x = sqrt_recip_alpha * (x - beta_t / sqrt_one_minus_alpha_bar_t * predicted_noise)
    
    # 添加随机噪声（除了最后一步）
    # 正确方法：使用掩码只对非零时间步添加噪声
    mask = t > 0
    if mask.any():
        # 计算当前时间步的噪声标准差
        sigma_t = torch.sqrt(beta_t)
        
        # 为需要添加噪声的图像生成噪声
        z = torch.zeros_like(x)
        z[mask] = torch.randn_like(x[mask])
        
        # 添加噪声
        x = x + sigma_t * z
    
    return x

@torch.no_grad()
def sample(model,n_images=4, T=1000,image_size=(32,32)):
    model.eval()
    
    # 从随机噪声开始
    x = torch.randn((n_images, 1, *image_size), device=device)
    
    # 从T到0逐步采样
    for i in range(T-1, -1, -1):
        # 创建相同的时间步张量
        t = torch.full((n_images,), i, device=device, dtype=torch.float)
        
        # 采样一步
        x = sample_timestep(x, t ,model)
    
    # 返回最终结果
    return x.clamp(-1, 1)
def visualize_tensor(tensor, title="", ncols=4, denormalize=True):
    """
    可视化一批张量图像
    """
    # 确保在CPU上
    tensor = tensor.cpu()
    
    # 反归一化 [-1,1] -> [0,1]
    if denormalize:
        tensor = (tensor * 0.5) + 0.5
    
    # 转换为NumPy并调整维度顺序
    images = tensor.detach().numpy()
    images = np.transpose(images, (0, 2, 3, 1))  # (N,C,H,W) -> (N,H,W,C)
    
    # 创建网格显示
    nrows = (len(images) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*2, nrows*2))
    
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            if images[i].shape[-1] == 1:  # 灰度图
                ax.imshow(images[i].squeeze(), cmap='gray')
            else:  # 彩色图
                ax.imshow(images[i])
            ax.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig
def visualize_samples(loader, num_samples=4):
    """验证数据加载是否正确"""
    images, _ = next(iter(loader))
    print(images.shape)  # 输出形状 (batch_size, channels, height, width)
    plt.figure(figsize=(10, 3))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i+1)
        # 反归一化：[-1,1] -> [0,1]
        img = (images[i].permute(1,2,0).numpy() * 0.5) + 0.5
        plt.imshow(img.squeeze(), cmap='gray')
        plt.axis('off')
    plt.suptitle('Loaded MNIST Samples')
    plt.show()
def train_ddpm(model, dataloader, device, num_epochs=1, lr=2e-4):
    
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    mse_loss = nn.MSELoss()
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=lr,
        total_steps=num_epochs * len(dataloader),
        pct_start=0.1  # 前10%作为warmup
    )

    epochs = num_epochs
    save_interval = 5
    for epoch in range(num_epochs):
        model.train()
        for step, (images, _) in enumerate(dataloader):
            images = images.to(device)
            
            # 添加mixup增强
            lam = np.random.beta(0.2, 0.2)
            mixed_images = lam * images + (1-lam) * images.flip(0)
            t = torch.rand(images.shape[0], device=device) * T
            
            # 使用混合图像进行扩散
            noisy_images, noise = forward_diffusion_sample(mixed_images, t, device)
            predicted_noise = model(noisy_images, t)
            
            loss = mse_loss(noise, predicted_noise)
            
            optimizer.zero_grad()
            loss.backward()
            # 添加梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # 打印训练信息
            if step % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Step [{step}/{len(train_loader)}] | Loss: {loss.item():.4f}")
                if (epoch) % 5 == 0:
                    generated_images = sample(model,n_images=8)
                    fig = visualize_tensor(generated_images, title=f"Epoch {epoch}")
                    fig.savefig(f'samples_epoch_{epoch}.png')
                    plt.close(fig)
       
    # 模型保存
        if (epoch+1) % save_interval == 0:
            torch.save(model.state_dict(), f"ddpm_mnist_epoch_{epoch+1}.pth")
#####################################################################################################################
def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class LinearAttention(Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = nn.GroupNorm(8,dim)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            nn.GroupNorm(8,dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h c n -> b h c n', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -1), ((mk, k), (mv, v)))

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

def Block(dim_in, dim_out):
            return nn.Sequential(
                nn.Conv2d(dim_in, dim_out, 3, padding=1),
                nn.GroupNorm(8, dim_out),
                nn.SiLU(),
                nn.Dropout(0.1)
            )
class ResnetBlock(Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, dropout=0.):
        super().__init__()
        self.mlp = (
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, dim_out * 2)
            ) 
            if exists(time_emb_dim) 
            else None
        )
        
        # 使用GroupNorm代替RMSNorm (符合DDPM原始设计)
        def Block(dim_in, dim_out):
            return nn.Sequential(
                nn.Conv2d(dim_in, dim_out, 3, padding=1),
                nn.GroupNorm(8, dim_out),  # 固定8组符合原始DDPM方案
                nn.SiLU(),
                nn.Dropout(dropout)
            )
        
        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
    
    # 添加完整的forward方法
    def forward(self, x, time_emb=None):
        # 原始输入用于残差连接
        identity = self.res_conv(x)
        
        # 第一个卷积块
        x = self.block1(x)
        
        # 处理时间嵌入（如果有）
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            # 扩展维度: [B, C] -> [B, C, 1, 1]
            time_emb = time_emb.view(time_emb.shape[0], -1, 1, 1)
            # 分割为缩放因子和偏置
            scale, shift = time_emb.chunk(2, dim=1)
            # 应用条件调制
            x = x * (scale + 1) + shift
        
        # 第二个卷积块
        x = self.block2(x)
        
        # 残差连接
        return x + identity

class RMSNorm(Module):
    #def __init__(self, dim):
    #    super().__init__()
    #    self.scale = dim ** 0.5
    #    self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    #def forward(self, x):
    #    return F.normalize(x, dim = 1) * self.g * self.scale
    def __init__(self, dim, dim_out, *, time_emb_dim=None, dropout=0.):
                super().__init__()
                self.block1 = Block(dim, dim_out)
                self.block2 = Block(dim_out, dim_out)
                self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

class Attention(Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4,
        flash = False
    ):
        super().__init__()
        self.heads = int(heads)
        dim_head = int(dim_head)
        hidden_dim = dim_head * heads

        self.norm = nn.GroupNorm(8,dim)
        self.attend = Attend(flash = flash)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -2), ((mk, k), (mv, v)))

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)
class SinusoidalPosEmb(Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
def divisible_by(numer, denom):
    return (numer % denom) == 0
class RandomOrLearnedSinusoidalPosEmb(Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered
def exists(x):
    return x is not None
def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d
def cosine_beta_schedule(T, s=0.008):
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)
class Unet(Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults = (1, 2, 4, 8),
        channels = 1,
        self_condition = False,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        dropout = 0.,
        attn_dim_head = 32,
        attn_heads = 4,
        full_attn = (False, False, True, False),    # defaults to full attention only for inner most layer
        flash_attn = False
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)
        #print(f"Input channels: {input_channels}")
        init_dim = default(init_dim, dim)
        #print(f"Initial dimension: {init_dim}")
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)
        #print("111")
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        #print(f"Dimensions: {dims}")
        in_out = list(zip(dims[:-1], dims[1:]))
        #print(f"In-Out pairs: {in_out}")
        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = dim
        sinu_pos_emb = SinusoidalPosEmb(dim, theta=10000)  # 固定正弦编码
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # attention

        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)

        num_stages = len(dim_mults)
        full_attn  = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        # prepare blocks
        resnet_block = partial(
            ResnetBlock, 
            time_emb_dim=time_dim, 
            dropout=dropout
        )
        FullAttention = partial(Attention, flash = flash_attn)
        resnet_block = partial(ResnetBlock, time_emb_dim = time_dim, dropout = dropout)

        # layers

        self.downs = ModuleList([])
        self.ups = ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.downs.append(ModuleList([
                resnet_block(dim_in, dim_in),
                resnet_block(dim_in, dim_in),
                attn_klass(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))
        if not isinstance(attn_heads, tuple):
            attn_heads = cast_tuple(attn_heads, num_stages)
        if not isinstance(attn_dim_head, tuple):
            attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim)
        mid_heads = attn_heads[-1]
        mid_dim_head = attn_dim_head[-1]
        self.mid_attn = Attention(
            mid_dim, 
            heads=mid_heads,
            dim_head=mid_dim_head,
            num_mem_kv=4,  # 添加默认值
            flash=False
        )
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.ups.append(ModuleList([
                resnet_block(dim_out + dim_in, dim_out),
                resnet_block(dim_out + dim_in, dim_out),
                attn_klass(dim_out, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = resnet_block(init_dim * 2, init_dim)
        self.final_conv = nn.Conv2d(init_dim, self.out_dim, 1)
        #print("successfully initialized Unet")
    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, time, x_self_cond = None):
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)
        #print(f"Input shape: {x.shape}, Time shape: {time.shape}")
        x = self.init_conv(x)
        #print(f"After init_conv shape: {x.shape}")
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x) + x
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x) + x

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_images = load_mnist_images('./data/MNIST/raw/train-images.idx3-ubyte')
    train_dataset = CustomMNIST(
        train_images,
        transform=transforms.Compose([
            transforms.ToTensor(),  # 转为Tensor (C,H,W)
            transforms.Pad(padding=2, fill=0, padding_mode='constant'),
            transforms.Normalize((0.5,), (0.5,))  # 归一化到[-1,1]
        ])
        )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=100, 
        shuffle=True, 
        num_workers=2
    )
    visualize_samples(train_loader)
    model = Unet(
        dim = 64,
        init_dim=64,
        dim_mults = (1, 2, 4,8)
    ).to(device)
    train_ddpm(model, train_loader, device, num_epochs=100, lr=1e-4)