"""阶段 1 训练入口：无监督训练 VAE 与像素空间 DDPM。

完整训练循环特性
----------------
- **EMA**：维护模型权重的指数滑动平均，采样用 EMA 模型（更稳定）
- **余弦学习率调度 + warmup**：前 500 步线性预热，之后余弦衰减
- **混合精度**：CUDA/MPS 启用 autocast + GradScaler；CPU 自动退化为 fp32
- **梯度裁剪**：max_norm=1.0，防止扩散训练后期梯度爆炸
- **断点续训**：每 epoch 保存 ``last.pt``（含 optimizer/调度器/EMA/step 状态）
- **周期评估**：每 ``EVAL_INTERVAL_EPOCHS`` 用 EMA 模型采样并计算全套指标
- **最佳模型**：基于 Evaluator 的综合质量分自动保存 ``best.pt``

用法
----
    python -m terrain.dl.train --stage vae --epochs 50
    python -m terrain.dl.train --stage diffusion --epochs 200 --resume
"""
from __future__ import annotations
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import (
    DEVICE, CKPT_DIR, LOG_DIR, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    WARMUP_STEPS, GRAD_CLIP, EMA_DECAY, USE_AMP, DIFFUSION_STEPS,
    BETA_START, BETA_END, IN_CHANNELS, EVAL_INTERVAL_EPOCHS, SAMPLE_NUM,
    DEM_TILES_DIR,
)
from .dataset import DEMTileDataset, make_loaders
from .vae import TerrainVAE, vae_loss
from .unet import DiffusionUNet
from .diffusion import DDPM
from .logger import TrainLogger
from .evaluator import Evaluator
from . import metrics as M


# ==================== 工具：EMA / 调度 / AMP ====================

class EMA:
    """模型参数指数滑动平均。

    维护一份与 model 同形状的"影子"参数；``apply()`` 临时把 model 替换为 EMA
    权重，``restore()`` 恢复原始权重。
    """

    def __init__(self, model, decay: float = EMA_DECAY):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters()
                       if p.requires_grad}
        self.backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    def apply(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])

    def restore(self, model):
        for n, p in model.named_parameters():
            if n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup.clear()


def cosine_warmup_lr(step: int, max_steps: int, warmup: int,
                     base_lr: float) -> float:
    """warmup + 余弦衰减学习率。"""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def amp_enabled() -> bool:
    """AMP 仅在 CUDA/MPS 启用；CPU 无效。"""
    return USE_AMP and DEVICE.type in ("cuda", "mps")


# ==================== 采样回调 ====================

def make_sample_fn_vae(vae: TerrainVAE, ema: EMA, n_steps: int = 1):
    """为 VAE 构造采样回调：从先验 z 解码。"""
    @torch.no_grad()
    def sample(n: int) -> np.ndarray:
        ema.apply(vae)
        try:
            z = torch.randn(n, 4, 32, 32, device=DEVICE)
            x = vae.decode(z)
        finally:
            ema.restore(vae)
        x = x.squeeze(1).cpu().numpy()
        from .config import norm_to_elev
        return norm_to_elev(x)
    return sample


def make_sample_fn_ddpm(model: DiffusionUNet, ema: EMA, ddpm: DDPM,
                        size: int = 256):
    """为 DDPM 构造采样回调。"""
    @torch.no_grad()
    def sample(n: int) -> np.ndarray:
        ema.apply(model)
        try:
            x = ddpm.sample(model, (n, IN_CHANNELS, size, size), DEVICE,
                            progress=False)
        finally:
            ema.restore(model)
        x = x.squeeze(1).cpu().numpy()
        from .config import norm_to_elev
        return norm_to_elev(x)
    return sample


# ==================== 公共训练循环 ====================

def _train_one_stage(model, optimizer, ema, scheduler_kind: str,
                     loader, val_loader, loss_fn, sample_fn,
                     run_name: str, epochs: int, n_steps_total: int,
                     ckpt_path: Path, resume: bool = False):
    """单阶段训练通用循环。

    Parameters
    ----------
    loss_fn : callable
        ``loss_fn(model, batch) -> (loss_tensor, info_dict)``。
    sample_fn : callable
        ``sample_fn(n) -> ndarray (n, H, W)``，供 Evaluator 周期采样。
    scheduler_kind : str
        "vae" 或 "diffusion"，仅用于日志区分。
    """
    logger = TrainLogger(run_name, LOG_DIR)
    evaluator = Evaluator(run_name, sample_dir=LOG_DIR / f"samples_{run_name}",
                          logger=logger)

    # ---------- 断点续训 ----------
    start_epoch = 0
    global_step = 0
    if resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        ema.shadow = ckpt["ema"]
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["step"]
        print(f"[resume] {run_name} 从 epoch {start_epoch} 续训 (step={global_step})")

    scaler = torch.amp.GradScaler(DEVICE.type) if amp_enabled() else None
    print(f"[{scheduler_kind}] device={DEVICE}, amp={amp_enabled()}, "
          f"epochs={epochs}, batch={loader.batch_size}")

    best_score = -float("inf")

    for epoch in range(start_epoch, epochs):
        model.train()
        t0 = time.time()
        epoch_loss = 0.0
        n_batches = 0

        for batch in loader:
            batch = batch.to(DEVICE)
            # ---- 学习率 ----
            lr = cosine_warmup_lr(global_step, n_steps_total,
                                  WARMUP_STEPS, LEARNING_RATE)
            for g in optimizer.param_groups:
                g["lr"] = lr

            optimizer.zero_grad()
            if amp_enabled():
                with torch.amp.autocast(DEVICE.type):
                    loss, info = loss_fn(model, batch)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, info = loss_fn(model, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            ema.update(model)
            epoch_loss += info.get("loss", float(loss.item()))
            n_batches += 1
            global_step += 1

            if global_step % 50 == 0:
                logger.log_scalars({
                    "loss": float(loss.item()),
                    "lr": lr,
                    **{k: v for k, v in info.items() if k != "loss"},
                }, step=global_step)

        # ---- epoch 结束 ----
        avg_loss = epoch_loss / max(1, n_batches)
        dt = time.time() - t0
        print(f"[{scheduler_kind}] ep{epoch:03d} avg_loss={avg_loss:.4f} "
              f"lr={lr:.2e} {dt:.1f}s")

        # ---- 保存 last ----
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "ema": ema.shadow,
            "epoch": epoch,
            "step": global_step,
        }, ckpt_path)

        # ---- 周期评估 ----
        if (epoch + 1) % EVAL_INTERVAL_EPOCHS == 0 or epoch == epochs - 1:
            model.eval()
            result = evaluator.step(epoch, global_step, sample_fn, n=SAMPLE_NUM)
            print(f"  [eval] score={result['quality_score']:.3f} "
                  f"stripe={result['stripe_index']:.3f} "
                  f"psd={result['psd_slope']:.3f} "
                  f"land={result['land_ratio']:.3f}")

            # 保存最佳
            if result["quality_score"] > best_score:
                best_score = result["quality_score"]
                best_path = ckpt_path.parent / f"best_{ckpt_path.name}"
                torch.save({
                    "model": model.state_dict(),
                    "ema": ema.shadow,
                    "epoch": epoch,
                    "step": global_step,
                    "score": best_score,
                }, best_path)
                print(f"  [best] 新最佳 score={best_score:.3f} → {best_path}")

    logger.close()
    print(f"[{scheduler_kind}] 训练完成. best_score={best_score:.3f}")


# ==================== VAE 训练 ====================

def train_vae(epochs: int = 50, batch_size: int = BATCH_SIZE,
              resume: bool = False):
    """训练地形 VAE。"""
    from .config import LATENT_CHANNELS, TILE_SIZE
    train_loader, _ = make_loaders(batch_size=batch_size)
    n_steps_total = epochs * len(train_loader)

    model = TerrainVAE(in_ch=IN_CHANNELS, latent_ch=LATENT_CHANNELS).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    ema = EMA(model)
    ckpt_path = CKPT_DIR / "vae_phase1.pt"
    sample_fn = make_sample_fn_vae(model, ema)

    def loss_fn(m, x):
        recon, mu, logvar = m(x)
        loss, info = vae_loss(recon, x, mu, logvar)
        info["loss"] = float(loss.item())
        return loss, info

    _train_one_stage(model, optimizer, ema, "vae",
                     train_loader, None, loss_fn, sample_fn,
                     "vae_phase1", epochs, n_steps_total,
                     ckpt_path, resume)


# ==================== DDPM 训练 ====================

def train_diffusion(epochs: int = 200, batch_size: int = BATCH_SIZE,
                    n_steps: int = DIFFUSION_STEPS, resume: bool = False):
    """训练像素空间 DDPM。"""
    train_loader, _ = make_loaders(batch_size=batch_size)
    n_steps_total = epochs * len(train_loader)

    model = DiffusionUNet(in_ch=IN_CHANNELS).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    ema = EMA(model)
    ddpm = DDPM(n_steps=n_steps, beta_start=BETA_START, beta_end=BETA_END,
                device=DEVICE)
    ckpt_path = CKPT_DIR / "ddpm_phase1.pt"
    sample_fn = make_sample_fn_ddpm(model, ema, ddpm)

    def loss_fn(m, x):
        t = torch.randint(0, n_steps, (x.size(0),), device=DEVICE)
        loss = ddpm.p_loss(m, x, t)
        return loss, {"loss": float(loss.item())}

    _train_one_stage(model, optimizer, ema, "diffusion",
                     train_loader, None, loss_fn, sample_fn,
                     "ddpm_phase1", epochs, n_steps_total,
                     ckpt_path, resume)


# ==================== CLI ====================

def main():
    p = argparse.ArgumentParser(description="阶段 1 训练入口")
    p.add_argument("--stage", choices=["vae", "diffusion"], required=True,
                   help="训练阶段：vae 或 diffusion")
    p.add_argument("--epochs", type=int, default=None,
                   help="训练轮数，默认 vae=50 / diffusion=200")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--resume", action="store_true", help="从 last.pt 断点续训")
    p.add_argument("--n-steps", type=int, default=DIFFUSION_STEPS,
                   help="DDPM 扩散步数（仅 diffusion 阶段）")
    args = p.parse_args()

    if args.stage == "vae":
        train_vae(epochs=args.epochs or 50, batch_size=args.batch_size,
                  resume=args.resume)
    else:
        train_diffusion(epochs=args.epochs or 200, batch_size=args.batch_size,
                        n_steps=args.n_steps, resume=args.resume)


if __name__ == "__main__":
    main()
