# -*- coding: utf-8 -*-
"""biome_before_after.py — 生成优化前后群系对比图 (并排, 直观展示边界变化).

用法: python3 biome_before_after.py [seed ...]  默认 2 5 8
输出: terrain/output/biome_before_after.png  (3种子×2列 并排)
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from terrain.core.terrain_engine import (continentality_index, ocean_current_effects,
    temperature_field, precipitation_field, whittaker_biome, BIOME_COLORS, BIOME_NAMES,
    smooth_biome_labels)

_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT = os.path.join(_dir, "terrain", "output")


def render_biome_rgb(bi, land, dem):
    """群系标签 → RGB 图像 (纯群系色, 海洋按深度渐变)."""
    size = bi.shape[0]
    rgb = np.zeros((size, size, 3), np.float32)
    rgb[land] = BIOME_COLORS[bi[land]]
    sea = ~land
    depth = np.clip(-dem / 6000.0, 0, 1)
    rgb[sea, 0] = 0.20 + 0.15 * (1 - depth[sea])
    rgb[sea, 1] = 0.35 + 0.20 * (1 - depth[sea])
    rgb[sea, 2] = 0.55 + 0.20 * (1 - depth[sea])
    return rgb


def compute_biomes(seed, size=2048):
    """计算优化前(raw)和优化后(smooth)的群系标签."""
    dem = np.load(os.path.join(OUTPUT, f"seed{seed}_dem.npy")).astype(np.float32)
    lats = (-90.0 + (np.arange(size) + 0.5) * (180.0 / size)).astype(np.float32)
    land = dem >= 1.0
    ci = continentality_index(dem, lats)
    dT, dP = ocean_current_effects(dem, lats)
    T = temperature_field(lats, dem, dT, ci=ci)
    rng = np.random.RandomState(seed + 17)
    P = precipitation_field(lats, dem, rng, dP, ci=ci)
    # 优化前: 原始 Whittaker 硬阈值 (无 T 平滑, 无标签平滑)
    bi_raw = whittaker_biome(T, P, lats=lats, ci=ci)
    # 优化后: T 高斯平滑 + smooth_biome_labels
    T_s = gaussian_filter(T.astype(np.float32), 2.0)
    bi_smooth = whittaker_biome(T_s, P, lats=lats, ci=ci)
    bi_smooth = smooth_biome_labels(bi_smooth, land, median_size=7, min_patch_px=50, iters=3)
    return bi_raw, bi_smooth, land, dem


def main():
    seeds = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [2, 5, 8]
    n = len(seeds)

    fig, axes = plt.subplots(n, 2, figsize=(16, 8 * n), squeeze=False)
    if n == 1:
        axes = axes.reshape(1, 2)

    for row, seed in enumerate(seeds):
        bi_raw, bi_smooth, land, dem = compute_biomes(seed)
        rgb_raw = render_biome_rgb(bi_raw, land, dem)
        rgb_smooth = render_biome_rgb(bi_smooth, land, dem)

        axes[row, 0].imshow(rgb_raw)
        axes[row, 0].set_title(f"seed{seed}  优化前 (Whittaker 硬阈值, 无平滑)",
                               fontsize=13, fontweight="bold")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(rgb_smooth)
        axes[row, 1].set_title(f"seed{seed}  优化后 (v7.6: T平滑 + 5×5中值 + 碎片清理)",
                               fontsize=13, fontweight="bold")
        axes[row, 1].axis("off")

    # 图例
    legend_elems = [Patch(facecolor=BIOME_COLORS[i], edgecolor="0.4",
                          label=BIOME_NAMES[i]) for i in range(len(BIOME_NAMES))]
    fig.legend(handles=legend_elems, loc="lower center", ncol=9, fontsize=10,
               frameon=True, edgecolor="0.5", bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    out = os.path.join(OUTPUT, "biome_before_after.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"→ {out}")

    # 同时生成各种子的单独大图 (方便看细节)
    for seed in seeds:
        bi_raw, bi_smooth, land, dem = compute_biomes(seed)
        for tag, bi in [("before", bi_raw), ("after", bi_smooth)]:
            rgb = render_biome_rgb(bi, land, dem)
            p = os.path.join(OUTPUT, f"biome_{tag}_seed{seed}.png")
            plt.imsave(p, rgb)
            print(f"→ {p}")


if __name__ == "__main__":
    main()
