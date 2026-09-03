# -*- coding: utf-8 -*-
"""biome_compare_new.py —— 新版气候模型(v7.4, 传 ci/lats) 群系对比分析.

加载地球 ETOPO1 + 3 个种子 DEM, 用当前 continentality_index/temperature_field/
precipitation_field/whittaker_biome (传 ci) 重算群系, 输出对比表 + JSON.

用法: python3 biome_compare_new.py
"""
import os, sys, json
import numpy as np
from scipy.io.netcdf import netcdf_file
from scipy.ndimage import zoom as ndi_zoom

# 绝对导入避免 shim 循环
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from terrain.core.terrain_engine import (continentality_index, ocean_current_effects,
    temperature_field, precipitation_field, whittaker_biome, BIOME_NAMES)

_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TERRAIN = os.path.join(_dir, "terrain")
OUTPUT = os.path.join(TERRAIN, "output")
DATA = os.path.join(TERRAIN, "data")
S = 2048
LATS = (-90.0 + (np.arange(S) + 0.5) * (180.0 / S)).astype(np.float32)
BN = BIOME_NAMES  # 9 类名


def block_mean_to(a, ny, nx):
    a = a.astype(np.float64)
    def red(x, nbins, axis):
        b = np.linspace(0, x.shape[axis], nbins + 1).round().astype(int)
        s = np.add.reduceat(x, b[:-1], axis=axis)
        c = np.diff(b).astype(np.float64)
        sh = [1] * x.ndim; sh[axis] = nbins
        return s / c.reshape(sh)
    return red(red(a, ny, 0), nx, 1).astype(np.float32)


def load_earth():
    """ETOPO1(Ice) → 2048×2048 米制 DEM (南→北, 经度周期)"""
    path = os.path.join(DATA, "ETOPO1_Ice_g_gmt4.grd")
    f = netcdf_file(path, "r", mmap=False)
    zv = f.variables["z"]; z = np.array(zv.data)
    scale = float(np.atleast_1d(getattr(zv, "scale_factor", 1.0))[0])
    off = float(np.atleast_1d(getattr(zv, "add_offset", 0.0))[0])
    if z.shape[0] > z.shape[1]: z = z.T
    if "y" in f.variables:
        y = np.array(f.variables["y"][:], np.float64).ravel()
        if y[0] > y[-1]: z = z[::-1]
    f.close()
    if z.shape[1] % 21600 == 1: z = z[:, :-1]
    return block_mean_to(z * scale + off, S, S)


def analyze(name, dem, seed=0):
    """新 climate 管线 (传 ci/lats) 群系统计 + 地形分类 + 面积加权."""
    lats = LATS
    coslat = np.clip(np.cos(np.deg2rad(lats)), 0.0, 1.0)
    Wm = coslat[:, None].astype(np.float64)
    land = dem >= 1.0
    Aw = Wm / Wm.sum()
    r = {"name": name}
    r["land_frac"] = float((land * Wm).sum() / (Wm.sum() * S))
    r["peak"] = round(float(dem.max()), 0)
    r["deepest"] = round(float(dem.min()), 0)
    # 大陆度 + 洋流 + 温度 + 降水 + 群系 (v7.4 新口径)
    ci = continentality_index(dem, lats)
    dT, dP = ocean_current_effects(dem, lats)
    T = temperature_field(lats, dem, dT, ci=ci)
    rng = np.random.RandomState(seed + 17)
    P = precipitation_field(lats, dem, rng, dP, ci=ci)
    bi = whittaker_biome(T, P, lats=lats, ci=ci)
    for i, bn in enumerate(BN):
        m = land & (bi == i)
        r[f"biome_{bn}"] = round(float((m * Wm).sum() / max((land * Wm).sum(), 1e-9) * 100), 2)
    # 地形高程带 (陆地加权 %)
    for lo, hi in [(0, 500), (500, 1500), (1500, 3500), (3500, 9000)]:
        m = land & (dem >= lo) & (dem < hi)
        r[f"elev_{lo}_{hi}"] = round(float((m * Wm).sum() / max((land * Wm).sum(), 1e-9) * 100), 2)
    # 沙漠/雨林/雪线 专项
    a2 = np.abs(lats)[:, None].repeat(S, 1)
    belt = land & (a2 >= 15) & (a2 < 35)
    r["desert_in_subtropics"] = round(float(((P < 250) & belt).sum() / max(belt.sum(), 1) * 100), 2)
    trop = land & (a2 < 10)
    r["rainforest_in_tropics"] = round(float(((bi == 8) & trop).sum() / max(trop.sum(), 1) * 100), 2)
    r["mean_T_land"] = round(float(T[land].mean()), 2)
    r["mean_P_land"] = round(float(P[land].mean()), 1)
    r["mean_CI_land"] = round(float(ci[land].mean()), 3)
    return r


def main():
    seeds = [2, 5, 8]
    results = {}
    print("加载真实地球 ETOPO1 (933MB) ...", flush=True)
    earth = load_earth()
    results["earth"] = analyze("earth", earth)
    print("earth 完成", flush=True)
    for s in seeds:
        p = os.path.join(OUTPUT, f"seed{s}_dem.npy")
        if not os.path.exists(p):
            print(f"seed{s} 缺失, 跳过", file=sys.stderr); continue
        dem = np.load(p).astype(np.float32)
        results[f"seed{s}"] = analyze(f"seed{s}", dem, seed=s)
        print(f"seed{s} 完成", flush=True)

    # ---- 汇总表 ----
    print("\n" + "=" * 92)
    print("群系占比 (陆地加权 %) — 新版气候模型 v7.4 (传 ci/lats)")
    print("=" * 92)
    cols = ["earth"] + [f"seed{s}" for s in seeds]
    print(f"{'群系':<12}" + "".join(f"{c:>10}" for c in cols))
    for bn in BN:
        row = f"{bn:<12}"
        for c in cols:
            row += f"{results[c][f'biome_{bn}']:>10.2f}"
        print(row)
    print("-" * 92)
    print("地形高程带 (陆地加权 %)")
    print(f"{'类型':<16}" + "".join(f"{c:>10}" for c in cols))
    for lo, hi in [(0, 500), (500, 1500), (1500, 3500), (3500, 9000)]:
        k = f"elev_{lo}_{hi}"
        row = f"{k:<16}"
        for c in cols:
            row += f"{results[c][k]:>10.2f}"
        print(row)
    print("-" * 92)
    print("专项指标 (%)")
    print(f"{'指标':<22}" + "".join(f"{c:>10}" for c in cols))
    for k in ["desert_in_subtropics", "rainforest_in_tropics"]:
        row = f"{k:<22}"
        for c in cols:
            row += f"{results[c][k]:>10.2f}"
        print(row)
    print("-" * 92)
    print("气候总览")
    print(f"{'指标':<16}" + "".join(f"{c:>10}" for c in cols))
    for k in ["land_frac", "peak", "deepest", "mean_T_land", "mean_P_land", "mean_CI_land"]:
        row = f"{k:<16}"
        for c in cols:
            row += f"{results[c][k]:>10}"
        print(row)

    out = os.path.join(OUTPUT, "biome_compare_new.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nJSON 已保存: {out}")


if __name__ == "__main__":
    main()
