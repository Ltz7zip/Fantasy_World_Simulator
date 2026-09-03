# -*- coding: utf-8 -*-
"""biome_zoom_compare.py — 局部放大对比: 展示群系边界细节变化.

对每个种子选取一个有代表性的区域 (沙漠/草原过渡带), 放大展示优化前后的
边界锯齿、碎片飞地变化.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from terrain.core.terrain_engine import (continentality_index, ocean_current_effects,
    temperature_field, precipitation_field, whittaker_biome, BIOME_COLORS, BIOME_NAMES,
    smooth_biome_labels)

_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT = os.path.join(_dir, "terrain", "output")


def render_biome_rgb(bi, land, dem):
    rgb = np.zeros(bi.shape + (3,), np.float32)
    rgb[land] = BIOME_COLORS[bi[land]]
    sea = ~land
    depth = np.clip(-dem / 6000.0, 0, 1)
    rgb[sea, 0] = 0.20 + 0.15 * (1 - depth[sea])
    rgb[sea, 1] = 0.35 + 0.20 * (1 - depth[sea])
    rgb[sea, 2] = 0.55 + 0.20 * (1 - depth[sea])
    return rgb


def find_zoom_region(bi, land, target_biome=5, size=2048, zoom=350):
    """找目标群系最大的连通块中心, 返回放大窗口."""
    from scipy.ndimage import label as ndi_label
    m = land & (bi == target_biome)
    if not m.any():
        return None
    lab, n = ndi_label(m)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    biggest = sizes.argmax()
    ys, xs = np.where(lab == biggest)
    cy, cx = int(ys.mean()), int(xs.mean())
    y0 = max(0, cy - zoom // 2)
    x0 = max(0, cx - zoom // 2)
    y1 = min(size, y0 + zoom)
    x1 = min(size, x0 + zoom)
    return y0, y1, x0, x1


def main():
    seeds = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [2, 5, 8]

    fig, axes = plt.subplots(len(seeds), 3, figsize=(18, 6 * len(seeds)), squeeze=False)

    for row, seed in enumerate(seeds):
        dem = np.load(os.path.join(OUTPUT, f"seed{seed}_dem.npy")).astype(np.float32)
        size = dem.shape[0]
        lats = (-90.0 + (np.arange(size) + 0.5) * (180.0 / size)).astype(np.float32)
        land = dem >= 1.0
        ci = continentality_index(dem, lats)
        dT, dP = ocean_current_effects(dem, lats)
        T = temperature_field(lats, dem, dT, ci=ci)
        rng = np.random.RandomState(seed + 17)
        P = precipitation_field(lats, dem, rng, dP, ci=ci)
        bi_raw = whittaker_biome(T, P, lats=lats, ci=ci)
        T_s = gaussian_filter(T.astype(np.float32), 2.0)
        bi_smooth = whittaker_biome(T_s, P, lats=lats, ci=ci)
        bi_smooth = smooth_biome_labels(bi_smooth, land, median_size=7, min_patch_px=50, iters=3)

        # 全局图 (左列): 标注放大框
        rgb_full = render_biome_rgb(bi_smooth, land, dem)
        zoom_box = find_zoom_region(bi_smooth, land, target_biome=5, size=size)
        axes[row, 0].imshow(rgb_full)
        axes[row, 0].set_title(f"seed{seed} 全局 (优化后)", fontsize=12)
        axes[row, 0].axis("off")
        if zoom_box:
            y0, y1, x0, x1 = zoom_box
            rect = Rectangle((x0, y0), x1 - x0, y1 - y0,
                              linewidth=2, edgecolor="red", facecolor="none")
            axes[row, 0].add_patch(rect)

        # 放大: 优化前 (中列)
        if zoom_box:
            y0, y1, x0, x1 = zoom_box
            rgb_raw_z = render_biome_rgb(bi_raw[y0:y1, x0:x1], land[y0:y1, x0:x1], dem[y0:y1, x0:x1])
            rgb_smooth_z = render_biome_rgb(bi_smooth[y0:y1, x0:x1], land[y0:y1, x0:x1], dem[y0:y1, x0:x1])
            axes[row, 1].imshow(rgb_raw_z)
            axes[row, 1].set_title(f"优化前 (放大 {x1-x0}px≈{(x1-x0)*19.5:.0f}km)", fontsize=12)
            axes[row, 2].imshow(rgb_smooth_z)
            axes[row, 2].set_title(f"优化后 (放大)", fontsize=12)
        axes[row, 1].axis("off")
        axes[row, 2].axis("off")

    legend_elems = [Patch(facecolor=BIOME_COLORS[i], edgecolor="0.4",
                          label=BIOME_NAMES[i]) for i in range(len(BIOME_NAMES))]
    fig.legend(handles=legend_elems, loc="lower center", ncol=9, fontsize=9,
               frameon=True, edgecolor="0.5", bbox_to_anchor=(0.5, -0.005))

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    out = os.path.join(OUTPUT, "biome_zoom_compare.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"→ {out}")


if __name__ == "__main__":
    main()
