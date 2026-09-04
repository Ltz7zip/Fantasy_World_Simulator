# -*- coding: utf-8 -*-
"""diagnose_issues.py — 诊断海洋方块化 + 陆地群系碎片 + 地貌缺失.

输出:
  1. 海洋方块化指标: 256px 块内方差比 (块内/块间, >1.5=方块化)
  2. 海底地貌分层占比: 陆架/陆坡/陆隆/深海平原/洋中脊/海沟
  3. 陆地地貌多样性: 高程分带占比 (平原/丘陵/山地/高原)
  4. 群系形状指标: 碎片块数 + 边界锯齿度
"""
import os, sys
import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from terrain.core.terrain_engine import (continentality_index, ocean_current_effects,
    temperature_field, precipitation_field, whittaker_biome, BIOME_NAMES,
    smooth_biome_labels, classify_landforms)

_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT = os.path.join(_dir, "terrain", "output")

# v7.8 地貌粗类名称 (低4位)
LF_COARSE_NAMES = {0: "山地", 1: "平原", 2: "高原", 3: "河谷",
                   4: "沙丘", 5: "丘陵", 6: "盆地"}

# v7.8 地貌子类名称 (粗类 → 高4位值 → 名称)
LF_SUB_NAMES = {
    0: {0x00: "高山(侵蚀)", 0x10: "中山(侵蚀)", 0x20: "低山(侵蚀)",
        0x40: "褶皱山", 0x80: "断层山", 0xC0: "火山"},
    2: {0x00: "高原(默认)", 0x10: "堆积型高原", 0x20: "切割型高原", 0x30: "侵蚀型高原"},
    5: {0x00: "丘陵(默认)", 0x10: "高丘陵", 0x20: "低丘陵"},
    6: {0x00: "盆地(默认)", 0x10: "沉积盆地", 0x20: "克拉通盆地",
        0x30: "构造-前陆盆地", 0x40: "内流盆地", 0x50: "外流盆地"},
}


def diagnose_ocean_blocks(dem):
    """海洋方块化: 将海洋分成 256px 网格, 测块内方差/块间方差比."""
    sea = dem < 0.0
    if not sea.any():
        return {"blockiness_ratio": 0, "n_blocks": 0}
    size = dem.shape[0]
    bs = 256  # GAN tile size
    nb = size // bs
    block_means = []
    block_vars = []
    for i in range(nb):
        for j in range(nb):
            blk = dem[i*bs:(i+1)*bs, j*bs:(j+1)*bs]
            blk_sea = blk[blk < 0]
            if len(blk_sea) > 100:
                block_means.append(blk_sea.mean())
                block_vars.append(blk_sea.var())
    block_means = np.array(block_means)
    block_vars = np.array(block_vars)
    # 方块化指标: 块间方差 / 平均块内方差 (>>1 = 方块化)
    inter_var = block_means.var() if len(block_means) > 1 else 0
    intra_var = block_vars.mean() if len(block_vars) > 0 else 1
    ratio = inter_var / max(intra_var, 1e-6)
    return {
        "blockiness_ratio": round(float(ratio), 3),
        "n_blocks": len(block_means),
        "inter_block_var": round(float(inter_var), 1),
        "intra_block_mean_var": round(float(intra_var), 1),
        "block_mean_range": f"{block_means.min():.0f}~{block_means.max():.0f}m" if len(block_means) else "N/A",
    }


def diagnose_ocean_landforms(dem):
    """海底地貌分层占比 (地球参考)."""
    sea = dem < 0.0
    total_sea = sea.sum()
    if total_sea == 0:
        return {}
    d = -dem[sea]  # 正深度
    # 地球参考 (ETOPO1):
    #  陆架 0-200m (~5%), 陆坡 200-2000m (~7%), 陆隆 2000-4000m (~6%)
    #  深海平原 4000-5500m (~30%), 洋中脊 2000-3500m? 实际脊顶~2500m
    #  海沟 >6000m (~1%)
    layers = {
        "陆架(0-200m)": (d <= 200),
        "陆坡(200-2000m)": (d > 200) & (d <= 2000),
        "陆隆(2000-4000m)": (d > 2000) & (d <= 4000),
        "深海平原(4000-5500m)": (d > 4000) & (d <= 5500),
        "深洋盆(5500-6500m)": (d > 5500) & (d <= 6500),
        "海沟(>6500m)": (d > 6500),
    }
    res = {}
    for name, m in layers.items():
        res[name] = round(float(m.sum() / total_sea * 100), 2)
    return res


def diagnose_land_landforms(dem, land):
    """陆地地貌分带占比 (高程+坡度)."""
    elev = dem[land]
    # 地球陆地高程分带:
    #  平原 0-200m, 丘陵 200-500m, 台地 500-1000m, 山地 1000-3500m, 高山>3500m
    bands = {
        "平原(0-200m)": (elev >= 0) & (elev <= 200),
        "丘陵(200-500m)": (elev > 200) & (elev <= 500),
        "台地(500-1000m)": (elev > 500) & (elev <= 1000),
        "山地(1000-2000m)": (elev > 1000) & (elev <= 2000),
        "高山(2000-3500m)": (elev > 2000) & (elev <= 3500),
        "极高山(>3500m)": (elev > 3500),
    }
    res = {}
    total = len(elev)
    for name, m in bands.items():
        res[name] = round(float(m.sum() / total * 100), 2)
    return res


def _load_tect_ctx(seed, size_px, tect_res=512):
    """从板块缓存加载并构建构造场上下文 (复用 generate_world 的扭曲+上采样).

    缓存命中时返回 dict(envS, beltS, cf, tect_T, margin_up, riftS);
    缓存缺失或构建失败时返回 None (诊断退化为粗类, 盆地仍出现但无子类).
    """
    import glob
    from terrain.core.terrain_engine import periodic_fbm, warp_lonlat
    from scipy.ndimage import zoom as ndi_zoom
    cache_dir = os.path.join(_dir, "terrain", "cache")
    ms = sorted(glob.glob(os.path.join(cache_dir,
                               f"plate_sim_*_T{tect_res}_seed{seed}.npz")))
    if not ms:
        return None
    # 优先选键最全的缓存 (v3 含 margin/belt/rift)
    ms.sort(key=lambda p: -len(np.load(p, mmap_mode='r').files))
    try:
        d = np.load(ms[0])
        tect_T = d["tect"]; crust_T = d["crust"]
        margin_T = d["margin"] if "margin" in d.files else None
        belt_T = d["belt"] if "belt" in d.files else None
        rift_T = d["rift"] if "rift" in d.files else None
        up = size_px // tect_res
        rng = np.random.RandomState(seed + 5)
        wxT = periodic_fbm(tect_res, 6.0, 3, rng) * 11.0 + \
              periodic_fbm(tect_res, 20.0, 2, rng) * 3.0
        wyT = periodic_fbm(tect_res, 6.0, 3, rng) * 11.0 + \
              periodic_fbm(tect_res, 20.0, 2, rng) * 3.0
        tect_T = np.stack([warp_lonlat(tect_T[i], wxT, wyT, 1.0)
                           for i in range(4)])
        if margin_T is not None:
            margin_T = warp_lonlat(margin_T.astype(np.float32), wxT, wyT, 1.0)
            margin_up = np.clip(ndi_zoom(margin_T, up, order=0)
                                [:size_px, :size_px], 0, 2).astype(np.int8)
        else:
            margin_up = None
        if belt_T is not None:
            belt_T = warp_lonlat(belt_T.astype(np.float32), wxT, wyT, 1.0)
            beltS = np.clip(ndi_zoom(belt_T, up, order=1)
                            [:size_px, :size_px], 0, 1).astype(np.float32)
        else:
            beltS = np.zeros((size_px, size_px), np.float32)
        if rift_T is not None:
            rift_T = warp_lonlat(rift_T.astype(np.float32), wxT, wyT, 1.0)
            riftS = np.clip(ndi_zoom(rift_T, up, order=1)
                            [:size_px, :size_px], 0, 1).astype(np.float32)
        else:
            riftS = np.zeros((size_px, size_px), np.float32)
        envS = np.clip(ndi_zoom(tect_T[1], up, order=1),
                       0, 1)[:size_px, :size_px]
        cf = gaussian_filter(crust_T.astype(np.float32), 2.0)
        cf = ndi_zoom(cf, up, order=1)[:size_px, :size_px]
        return dict(envS=envS, beltS=beltS, cf=cf, tect_T=tect_T,
                    margin_up=margin_up, riftS=riftS)
    except Exception as e:
        print(f"  [warn] 构造场加载失败, 退化为粗类: {e}")
        return None


def diagnose_landforms_v78(dem, land, lats, tect_ctx=None):
    """v7.8 陆地地貌分项统计 (粗类 + 子类).

    基于 :func:`classify_landforms` 的位打包标签输出粗类占比与子类占比.
    无 ``tect_ctx`` 时仅输出粗类 (盆地仍出现, 但无子类).
    """
    tc = tect_ctx or {}
    lf = classify_landforms(dem, lats, land=land,
                            envS=tc.get('envS'), beltS=tc.get('beltS'),
                            cf=tc.get('cf'), tect_T=tc.get('tect_T'),
                            margin_up=tc.get('margin_up'), riftS=tc.get('riftS'))
    total = int(land.sum())
    res = {"coarse": {}, "sub": {}}
    # 粗类 (低4位)
    coarse = np.where(land, lf & 0x0F, -1)
    for c, name in LF_COARSE_NAMES.items():
        res["coarse"][name] = round(float((coarse == c).sum() /
                                         max(total, 1) * 100), 2)
    # 子类 (高4位, 仅对有子类定义的粗类统计)
    for c, sub_map in LF_SUB_NAMES.items():
        m = (coarse == c)
        if not m.any():
            continue
        sub = np.where(m, lf & 0xF0, -1)
        for sv, sname in sub_map.items():
            res["sub"][sname] = round(float((sub == sv).sum() /
                                            max(total, 1) * 100), 2)
    return res


def diagnose_biome_fragments(bi, land):
    """群系碎片统计."""
    from scipy.ndimage import label as ndi_label
    res = {}
    total_transitions = 0
    bi_land = np.where(land, bi, -1)
    h_trans = np.sum(bi_land[:, :-1] != bi_land[:, 1:])
    v_trans = np.sum(bi_land[:-1, :] != bi_land[1:, :])
    total_transitions = int(h_trans + v_trans)
    res["total_boundary_transitions"] = total_transitions
    for idx, bn in enumerate(BIOME_NAMES):
        m = land & (bi == idx)
        npx = int(m.sum())
        if npx == 0:
            res[bn] = {"pixels": 0, "patches": 0}
            continue
        lab, n_patch = ndi_label(m)
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        n_real = int((sizes >= 5).sum())
        # 平均块面积 (px)
        mean_px = round(float(npx / max(n_real, 1)), 1)
        res[bn] = {
            "pixels": npx,
            "patches": n_patch,
            "real_patches": n_real,
            "mean_patch_px": mean_px,
            "land_pct": round(float(npx / land.sum() * 100), 2),
        }
    return res


def run(seed, size=2048):
    dem = np.load(os.path.join(OUTPUT, f"seed{seed}_dem.npy")).astype(np.float32)
    land = dem >= 1.0
    lats = (-90.0 + (np.arange(size) + 0.5) * (180.0 / size)).astype(np.float32)
    ci = continentality_index(dem, lats)
    dT, dP = ocean_current_effects(dem, lats)
    T = temperature_field(lats, dem, dT, ci=ci)
    rng = np.random.RandomState(seed + 17)
    P = precipitation_field(lats, dem, rng, dP, ci=ci)
    T_s = gaussian_filter(T.astype(np.float32), 2.5)
    P_s = gaussian_filter(P.astype(np.float32), 1.5)
    bi = whittaker_biome(T_s, P_s, lats=lats, ci=ci)
    bi = smooth_biome_labels(bi, land, median_size=7, min_patch_px=50, iters=3)
    tect_ctx = _load_tect_ctx(seed, size)
    return {
        "seed": seed,
        "land_frac": round(float(land.sum() / land.size), 4),
        "ocean_blocks": diagnose_ocean_blocks(dem),
        "ocean_landforms": diagnose_ocean_landforms(dem),
        "land_landforms": diagnose_land_landforms(dem, land),
        "landforms_v78": diagnose_landforms_v78(dem, land, lats, tect_ctx),
        "tect_ctx_loaded": tect_ctx is not None,
        "biome_fragments": diagnose_biome_fragments(bi, land),
    }


def main():
    import json
    seeds = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [2, 5, 8]
    all_res = {}
    for s in seeds:
        r = run(s)
        all_res[f"seed{s}"] = r
        print(f"\n{'='*65}")
        print(f"seed{s}  land_frac={r['land_frac']}")
        print(f"\n--- 海洋方块化 ---")
        for k, v in r["ocean_blocks"].items():
            print(f"  {k}: {v}")
        print(f"\n--- 海底地貌分层占比 ---")
        for k, v in r["ocean_landforms"].items():
            print(f"  {k}: {v}%")
        print(f"\n--- 陆地地貌分带占比 (高程分带) ---")
        for k, v in r["land_landforms"].items():
            print(f"  {k}: {v}%")
        v78 = r["landforms_v78"]
        tag = "含子类" if r.get("tect_ctx_loaded") else "仅粗类, 无构造场"
        print(f"\n--- v7.8 陆地地貌粗类占比 ({tag}) ---")
        for k, v in v78["coarse"].items():
            print(f"  {k}: {v}%")
        if v78["sub"]:
            print(f"  -- 子类 --")
            for k, v in v78["sub"].items():
                print(f"  {k}: {v}%")
        bf = r["biome_fragments"]
        print(f"\n--- 群系碎片 (boundary_transitions={bf['total_boundary_transitions']}) ---")
        print(f"  {'群系':<12} {'像素':>8} {'块数':>6} {'≥5px块':>7} {'平均px':>8} {'占陆%':>7}")
        for bn in BIOME_NAMES:
            d = bf[bn]
            if d["pixels"] == 0:
                continue
            print(f"  {bn:<12} {d['pixels']:>8} {d['patches']:>6} {d['real_patches']:>7} "
                  f"{d['mean_patch_px']:>8} {d['land_pct']:>7}")
    out = os.path.join(OUTPUT, "diagnose_issues.json")
    with open(out, "w") as f:
        json.dump(all_res, f, ensure_ascii=False, indent=2)
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
