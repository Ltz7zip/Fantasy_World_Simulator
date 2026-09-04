# -*- coding: utf-8 -*-
"""render_biome_only.py — 渲染纯群系图(无地形阴影/河流/雪线), 直观看群系形状.

用法: python3 render_biome_only.py [seed ...]  默认 2 5 8
输出: terrain/output/biome_only_seed{seed}.png
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from terrain.core.terrain_engine import (continentality_index, ocean_current_effects,
    temperature_field, precipitation_field, whittaker_biome, BIOME_COLORS,
    smooth_biome_labels)
from scipy.ndimage import gaussian_filter

_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT = os.path.join(_dir, "terrain", "output")


def render(seed, size=2048):
    dem = np.load(os.path.join(OUTPUT, f"seed{seed}_dem.npy")).astype(np.float32)
    lats = (-90.0 + (np.arange(size) + 0.5) * (180.0 / size)).astype(np.float32)
    land = dem >= 1.0
    ci = continentality_index(dem, lats)
    dT, dP = ocean_current_effects(dem, lats)
    T = temperature_field(lats, dem, dT, ci=ci)
    rng = np.random.RandomState(seed + 17)
    P = precipitation_field(lats, dem, rng, dP, ci=ci)
    T_s = gaussian_filter(T.astype(np.float32), 2.5)
    P_s = gaussian_filter(P.astype(np.float32), 1.5)
    bi = whittaker_biome(T_s, P_s, lats=lats, ci=ci)
    bi = smooth_biome_labels(bi, land, median_size=7, min_patch_px=50, iters=3)
    # 纯群系色, 海洋用灰白
    rgb = np.zeros((size, size, 3), np.float32)
    rgb[land] = BIOME_COLORS[bi[land]]
    # 海洋浅蓝
    sea = ~land
    depth = np.clip(-dem / 6000.0, 0, 1)
    rgb[sea, 0] = 0.20 + 0.15 * (1 - depth[sea])
    rgb[sea, 1] = 0.35 + 0.20 * (1 - depth[sea])
    rgb[sea, 2] = 0.55 + 0.20 * (1 - depth[sea])
    out = os.path.join(OUTPUT, f"biome_only_seed{seed}.png")
    plt.imsave(out, rgb)
    print(f"seed{seed} → {out}")
    return out


def main():
    seeds = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [2, 5, 8]
    for s in seeds:
        render(s)


if __name__ == "__main__":
    main()
