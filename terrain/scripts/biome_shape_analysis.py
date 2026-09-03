# -*- coding: utf-8 -*-
"""biome_shape_analysis.py — 定量分析群系形状问题.

指标:
  1. 连通域数量 (fragmentation): 每个群系被切成的独立块数. >30=过度破碎.
  2. 边界锯齿度 (jaggedness): 周长²/面积 (圆=4π≈12.57, 越大越锯齿).
  3. 平均块面积 (mean patch area): 破碎程度.
  4. 边界平滑度: 相邻像素群系变化次数 (transition count).

用法: python3 biome_shape_analysis.py [seed ...]  默认 2 5 8
"""
import os, sys
import numpy as np
from scipy.ndimage import label as ndi_label, binary_dilation, distance_transform_edt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from terrain.core.terrain_engine import (continentality_index, ocean_current_effects,
    temperature_field, precipitation_field, whittaker_biome, BIOME_NAMES,
    smooth_biome_labels)
from scipy.ndimage import gaussian_filter

_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT = os.path.join(_dir, "terrain", "output")

# 经度周期环绕的连通域标注
def label_periodic(mask):
    lab, n = ndi_label(mask)
    if n == 0:
        return lab, 0
    parent = list(range(n + 1))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    right = lab[:, -1]
    left = lab[:, 0]
    for r in np.unique(right):
        if r == 0:
            continue
        rows = np.nonzero(right == r)[0]
        ls = left[rows]
        ls = np.unique(ls[ls > 0])
        for l in ls:
            parent[find(int(r))] = find(int(l))
    lut = np.array([find(i) for i in range(n + 1)])
    lab2 = lut[lab]
    vals = np.unique(lab2)
    vals = vals[vals > 0]
    remap = np.zeros(lab2.max() + 1, np.int32)
    remap[vals] = np.arange(1, len(vals) + 1)
    return remap[lab2], len(vals)


def analyze_biome_shapes(bi, land, name=""):
    """分析单个群系的形状指标."""
    size = bi.shape[0]
    results = {}
    total_transitions = 0
    # 相邻像素群系变化次数 (水平+垂直, 陆地内)
    bi_land = np.where(land, bi, -1)
    h_trans = np.sum(bi_land[:, :-1] != bi_land[:, 1:])
    v_trans = np.sum(bi_land[:-1, :] != bi_land[1:, :])
    total_transitions = h_trans + v_trans
    results["total_boundary_transitions"] = int(total_transitions)

    for idx, bn in enumerate(BIOME_NAMES):
        m = land & (bi == idx)
        npx = int(m.sum())
        if npx == 0:
            results[bn] = {"pixels": 0, "patches": 0}
            continue
        # 周期连通域
        lab, n_patch = label_periodic(m)
        # 去除极小块 (<5px) 后统计
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        n_real = int((sizes >= 5).sum())
        # 边界像素数 (群系内部边界, 即与不同群系/海洋相邻的像素)
        edge = m & (binary_dilation(m ^ land) | ~land)  # 与海洋或不同群系相邻
        # 更准确: 与不同群系相邻的像素数
        edge_biome = np.zeros_like(m)
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            shifted = np.roll(np.roll(bi_land, dy, 0), dx, 1)
            edge_biome |= m & (shifted != idx) & (shifted != -1)
        edge_count = int(edge_biome.sum())
        # 锯齿度: 周长²/面积 (圆=4π≈12.57, 正方形=16, 越大越锯齿)
        jaggedness = (edge_count ** 2) / max(npx, 1)
        results[bn] = {
            "pixels": npx,
            "patches": n_patch,
            "real_patches_ge5px": n_real,
            "edge_pixels": edge_count,
            "jaggedness": round(float(jaggedness), 2),
            "mean_patch_px": round(float(npx / max(n_patch, 1)), 1),
        }
    return results


def run(seed, size=2048):
    dem = np.load(os.path.join(OUTPUT, f"seed{seed}_dem.npy")).astype(np.float32)
    lats = (-90.0 + (np.arange(size) + 0.5) * (180.0 / size)).astype(np.float32)
    land = dem >= 1.0
    ci = continentality_index(dem, lats)
    dT, dP = ocean_current_effects(dem, lats)
    T = temperature_field(lats, dem, dT, ci=ci)
    rng = np.random.RandomState(seed + 17)
    P = precipitation_field(lats, dem, rng, dP, ci=ci)
    bi_raw = whittaker_biome(T, P, lats=lats, ci=ci)
    # v7.6 平滑后
    T_s = gaussian_filter(T.astype(np.float32), 2.0)
    bi_smooth = whittaker_biome(T_s, P, lats=lats, ci=ci)
    bi_smooth = smooth_biome_labels(bi_smooth, land, median_size=7, min_patch_px=50, iters=3)
    res_raw = analyze_biome_shapes(bi_raw, land, name=f"seed{seed}_raw")
    res_smooth = analyze_biome_shapes(bi_smooth, land, name=f"seed{seed}_smooth")
    res = {"raw": res_raw, "smooth": res_smooth, "seed": seed,
           "land_frac": round(float(land.sum() / land.size), 4)}
    return res


def _print_table(r, label):
    print(f"\n  [{label}]  transitions={r['total_boundary_transitions']}")
    print(f"  {'群系':<12} {'像素':>8} {'块数':>6} {'≥5px块':>7} {'边界px':>8} {'锯齿度':>8} {'平均块px':>9}")
    for bn in BIOME_NAMES:
        d = r[bn]
        if d["pixels"] == 0:
            print(f"  {bn:<12} {0:>8} {0:>6} {0:>7} {0:>8} {0:>8} {0:>9}")
        else:
            print(f"  {bn:<12} {d['pixels']:>8} {d['patches']:>6} {d['real_patches_ge5px']:>7} "
                  f"{d['edge_pixels']:>8} {d['jaggedness']:>8} {d['mean_patch_px']:>9}")


def main():
    import json
    seeds = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [2, 5, 8]
    all_res = {}
    for s in seeds:
        r = run(s)
        all_res[f"seed{s}"] = r
        print(f"\n{'='*70}")
        print(f"seed{s}  land_frac={r['land_frac']}")
        _print_table(r["raw"], "RAW (Whittaker 硬阈值, 无平滑)")
        _print_table(r["smooth"], "SMOOTH (v7.6 T平滑 + 5×5中值 + 碎片清理)")
        # 改善摘要
        print(f"\n  --- 改善摘要 (raw → smooth) ---")
        print(f"  {'群系':<12} {'锯齿度':>16} {'块数':>14} {'平均块px':>16}")
        for bn in BIOME_NAMES:
            dr, ds = r["raw"][bn], r["smooth"][bn]
            if dr["pixels"] == 0 and ds["pixels"] == 0:
                continue
            print(f"  {bn:<12} {dr['jaggedness']:>7}→{ds['jaggedness']:<7} "
                  f"{dr['patches']:>6}→{ds['patches']:<6} "
                  f"{dr['mean_patch_px']:>7}→{ds['mean_patch_px']:<7}")
    out = os.path.join(OUTPUT, "biome_shape_analysis.json")
    with open(out, "w") as f:
        json.dump(all_res, f, ensure_ascii=False, indent=2)
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
