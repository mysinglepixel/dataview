import torch

class DDPM:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.T = T
        self.device = device

        # 计算 beta 调度
        self.beta = torch.linspace(beta_start, beta_end, T, device=device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        # 预计算 sqrt(ᾱ_t) 和 sqrt(1 - ᾱ_t)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1. - self.alpha_bar)

    def forward_process(self, x0, t):
        """前向扩散过程：根据时间步 t 添加噪声"""
        t = t.to(self.device)  # 确保 t 在同一设备上
        noise = torch.randn_like(x0, device=self.device)  # 生成高斯噪声

        # 取出 alpha_bar 的 t 时刻值，保持 batch 维度
        sqrt_alpha_bar_t = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)

        # 计算带噪声图像
        xt = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise
        return xt, noise  # 返回带噪声图像和噪声

    def reverse_process(self, xt, pred_noise, t):

        # 确保 t 在正确的设备上，并转换为 batch 形状
        t = t.to(self.device)
        
        # 获取 alpha_t, beta_t, alpha_bar_t
        alpha_t = self.alpha[t].view(-1, 1, 1, 1)  # (B, 1, 1, 1)
        beta_t = self.beta[t].view(-1, 1, 1, 1)  # (B, 1, 1, 1)
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)  # (B, 1, 1, 1)

        # 计算 x0（去噪后最有可能的原始图像）
        pred_x0 = (xt - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)

        # 计算去噪图像 x_{t-1}
        mean = torch.sqrt(1.0 / alpha_t) * (xt - beta_t / torch.sqrt(1 - alpha_bar_t) * pred_noise)

        # 采样随机噪声（只有 t > 0 时才加噪）
        if t[0] > 0:
            noise = torch.randn_like(xt).to(self.device)
            sigma_t = torch.sqrt(beta_t)  # 采样方差
            x_prev = mean + sigma_t * noise
            #log_snr = torch.log(alpha_t / (1 - alpha_t))
            #sigma_t = torch.sqrt(beta_t) * torch.exp(0.5 * log_snr)
            #x_prev = mean + sigma_t * noise
        else:
            x_prev = mean  # 最后一轮不添加噪声

        return x_prev
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建 DDPM 实例
ddpm = DDPM(T=1000, device=device)

# 生成测试数据
batch_size = 4
image_size = (batch_size, 3, 256, 256)  # (B, C, H, W)
x0 = torch.randn(image_size, device=device)  # 伪造的彩色图像
t = torch.randint(0, ddpm.T, (batch_size,), device=device)  # 随机时间步

# 执行前向扩散
xt, noise = ddpm.forward_process(x0, t)

# 检查形状是否正确
#print("xt shape:", xt.shape)  # 应该是 (B, 3, 256, 256)
#print("noise shape:", noise.shape)  # 应该是 (B, 3, 256, 256)