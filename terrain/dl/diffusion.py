"""DDPM 调度器：前向加噪 + 反向去噪采样。

阶段 1 用线性 β 调度（与原论文一致）。阶段 2 可换 cosine 调度。
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


class DDPM:
    """DDPM 调度器。

    Parameters
    ----------
    n_steps : int
        扩散步数 T。
    beta_start, beta_end : float
        线性 β 调度上下界。
    device : torch.device | str
        预计算 ᾱ 等张量所在设备。
    """

    def __init__(self, n_steps: int = 1000, beta_start: float = 1e-4,
                 beta_end: float = 2e-2, device: torch.device | str = "cpu"):
        self.n = n_steps
        betas = torch.linspace(beta_start, beta_end, n_steps, device=device)
        alphas = 1.0 - betas
        self.alpha_bar = torch.cumprod(alphas, dim=0)        # (n,)
        self.betas = betas
        self.alphas = alphas

    # ---------- 前向 ----------
    def q_sample(self, x0, t, noise=None):
        """前向加噪：x_t = sqrt(ᾱ_t)·x0 + sqrt(1-ᾱ_t)·ε"""
        if noise is None:
            noise = torch.randn_like(x0)
        ab = self.alpha_bar.to(x0.device)[t].view(-1, 1, 1, 1)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise, noise

    # ---------- 训练损失 ----------
    def p_loss(self, model, x0, t):
        """训练目标：预测噪声 ε。"""
        x_t, noise = self.q_sample(x0, t)
        pred = model(x_t, t)
        return F.mse_loss(pred, noise)

    # ---------- 反向采样 ----------
    @torch.no_grad()
    def sample(self, model, shape, device, progress: bool = False):
        """反向采样（DDPM 完整 n 步）。

        Parameters
        ----------
        model : nn.Module
            去噪 U-Net。
        shape : tuple
            输出形状 (B, C, H, W)。
        device : torch.device
        progress : bool
            是否显示进度条。
        """
        try:
            from tqdm import trange
        except ImportError:
            trange = range  # type: ignore

        x = torch.randn(shape, device=device)
        iters = range(self.n - 1, -1, -1)
        if progress:
            try:
                from tqdm import tqdm as _tqdm
                iters = _tqdm(iters, desc="sampling")
            except ImportError:
                pass

        for i in iters:
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            eps = model(x, t)
            a = self.alphas[i]
            ab = self.alpha_bar[i]
            x = (1 / a.sqrt()) * (x - (1 - a) / (1 - ab).sqrt() * eps)
            if i > 0:
                x += self.betas[i].sqrt() * torch.randn_like(x)
        return x
