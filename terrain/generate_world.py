# -*- coding: utf-8 -*-
# ======================================================================
# generate_world.py v6.3 —— 板块演化模拟世界生成器
# 一级: run_plate_sim (450 Myr, 有缓存) → 粗DEM + 4构造场
# 二级: 12通道条件 → 逐瓦片重训版GAN → 低频对齐 → 全局CDF匹配
#       → 全球D8河网 → 雪线渲染     运行: python generate_world.py --seed 7 --size 2048
# ======================================================================
import os
import time
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import (gaussian_filter, distance_transform_edt,
                           zoom as ndi_zoom, label as ndi_label,
                           minimum_filter, maximum_filter, binary_dilation)
from terrain_engine import (device, m2v, v2m, block_reduce, d8_flow_accumulation,
                            snowline_m, value_fbm, ridged_fbm, sharpen_coastline,
                            lowfreq_consistency, suppress_hallucinated_land,
                            polar_ice_mask, make_terrain_cmap, UNet,
                            temperature_field, precipitation_field, whittaker_biome,
                            BIOME_COLORS, fill_depressions, droplet_erosion,
                            periodic_fbm, warp_lonlat)
from plate_sim import run_plate_sim


def fbm1d(n, sigma, rng):
    x = rng.randn(n)
    X = np.fft.rfft(x)
    X *= np.exp(-(np.fft.rfftfreq(n) * sigma) ** 2)
    return np.fft.irfft(X, n)


def despeckle(v01, min_area=48):
    v = v01.copy()
    land = v >= 0.5
    lab, n = ndi_label(land)
    if n:
        sz = np.bincount(lab.ravel()); sz[0] = 0
        drop = np.isin(lab, np.nonzero(sz < min_area)[0])
        v[drop] = minimum_filter(v, 9)[drop]
    lab, n = ndi_label(~land)
    if n:
        sz = np.bincount(lab.ravel()); sz[0] = 0
        fill = np.isin(lab, np.nonzero(sz < min_area)[0])
        v[fill] = maximum_filter(v, 9)[fill]
    return v.astype(np.float32)


def global_envelope_match(v01, bm, bstd, bmin, bmax, glac, block=8,
                          blend=0.60, seed=0, cap_m=8200.0):
    """全局一次 CDF 匹配 + 逐像素块包络裁剪 + 行星峰值截断 + 冰盖低blend保纹理"""
    rng = np.random.default_rng(seed)
    h, w = v01.shape
    n = h * w
    idx = rng.integers(0, bm.size, n)
    s = rng.normal(bm.ravel()[idx], bstd.ravel()[idx] + 1e-4)
    s = np.clip(s, bmin.ravel()[idx], bmax.ravel()[idx])
    ts = np.sort(s)
    order = np.argsort(v01, axis=None)
    ranked = np.empty(n, np.int64); ranked[order] = np.arange(n)
    matched = ts[ranked].reshape(h, w)
    bmin_up = np.clip(ndi_zoom(bmin, block, order=1)[:h, :w], 0, 1)
    bmax_up = np.clip(ndi_zoom(bmax, block, order=1)[:h, :w], 0, 1)
    matched = np.minimum(np.clip(matched, bmin_up, bmax_up), m2v(cap_m))
    glac_up = ndi_zoom(glac, block, order=1)[:h, :w]
    bl = np.clip(blend - 0.30 * glac_up, 0.30, blend)
    return np.clip(bl * matched + (1 - bl) * v01, 0, 1).astype(np.float32)


def render_world(v01, lats, landS, cmap, rng):
    """气候因果链渲染: 温度/降水 → 生物群系 + 山体阴影 + 可见河流/湖泊
    + 温度雪线(纬度+海拔双因素连续过渡, 无硬截断) + 海冰"""
    S = v01.shape[0]
    land = v01 >= 0.5
    elev_m = v2m(v01)
    T = temperature_field(lats, elev_m)
    P = precipitation_field(lats, elev_m, rng)
    bi = whittaker_biome(T, P)
    # 山体阴影 (NW 光照, 仅陆地) → 地形层次感
    g = gaussian_filter(v01, 1.5)
    gy, gx = np.gradient(g)
    hs = np.clip(0.88 - 22.0 * (gx + gy), 0.60, 1.15)
    shade = (0.78 + 0.55 * np.clip((v01 - 0.5) / 0.5, 0, 1)) * (0.72 + 0.28 * hs)
    land_rgb = BIOME_COLORS[bi] * shade[..., None]
    rgb = cmap(np.clip(v01, 0, 1))[..., :3]
    rgb = np.where(land[..., None], land_rgb, rgb).astype(np.float32)
    # 湖泊: 填洼面与实际地形之差 → 湖面 (盆地积水)
    filled = fill_depressions(v01)
    lake = land & ((filled - v01) > 0.012)
    rgb[lake] = rgb[lake] * 0.25 + np.array([0.12, 0.30, 0.46], np.float32) * 0.75
    # 河流: 填洼 D8 流累积 → 可见河道 (宽度随流量)
    acc = d8_flow_accumulation(filled, rng)
    aL = acc[land]
    thr = max(float(np.quantile(aL, 0.995)) if aL.size else 300.0, 250.0)
    riv = land & (acc > thr)
    big = land & (acc > 5.0 * thr)
    rgb[riv] = rgb[riv] * 0.55 + np.array([0.10, 0.26, 0.42], np.float32) * 0.45
    big_d = binary_dilation(big, iterations=1) & land
    rgb[big_d] = rgb[big_d] * 0.35 + np.array([0.08, 0.22, 0.38], np.float32) * 0.65
    # 雪线: 纬度雪线(干旱抬升) ∪ 温度场雪线(年均温≤-12°C 常年积雪, 连续过渡)
    dry = np.clip(1.0 - P / 700.0, 0, 1)
    sl01 = m2v(snowline_m(lats)[:, None] + 800.0 * dry)
    a_sl = 0.12 + 0.83 * np.clip((v01 - sl01) / 0.06, 0, 1)
    a_t = np.clip((-12.0 - T) / 6.0, 0, 1)
    alpha = np.maximum(a_sl, a_t) * land
    rs = np.clip((v01 - gaussian_filter(v01, 6.0)) / 0.05 + 0.6, 0, 1)  # 山脊雪骨架
    alpha = alpha * (0.35 + 0.65 * rs)
    a3 = alpha[..., None]
    rgb = rgb * (1 - a3) + np.array([0.96, 0.97, 0.99])[None, None, :] * a3
    # 海冰: 纬度 + 噪声, 平滑边缘
    edge_sea = 64.5 + 2.5 * fbm1d(S, 320, rng)
    sig = 1.0 / (1.0 + np.exp(-(np.abs(lats)[:, None] - edge_sea[None, :]) / 1.6))
    sea_ice = (sig > 0.5) & (v01 < 0.5)
    floe = gaussian_filter(value_fbm(S, 24, 3, rng=rng), 0.8)
    sea_ice &= (floe > -1.1) | (np.abs(lats)[:, None] > edge_sea[None, :] + 4)
    rgb[sea_ice] = rgb[sea_ice] * 0.30 + np.array([0.88, 0.93, 0.97]) * 0.70
    return np.clip(rgb, 0, 1)


def world_to_channels(elev_m, lats, tect_T, block=8, seed=1):
    rng = np.random.RandomState(seed)
    v01 = m2v(elev_m)
    H = v01.shape[0]
    B = H // block
    bm = block_reduce(v01, block, "mean"); bstd = block_reduce(v01, block, "std")
    bmin = block_reduce(v01, block, "min"); bmax = block_reduce(v01, block, "max")
    lf = block_reduce((elev_m > 0).astype(np.float32), block, "mean")
    acc = d8_flow_accumulation(bm, rng)
    lmask = lf > 0.05
    thr = max(float(np.quantile(acc[lmask], 0.99)), 2.0) if lmask.any() else 2.0
    river = np.clip(np.log(np.maximum(acc, 1.0001)) / np.log(thr), 0, 1) * lmask
    glac = block_reduce((elev_m > snowline_m(lats)[:, None]).astype(np.float32),
                        block, "mean")
    lat_rows = (np.abs(lats[block // 2::block]) / 90.0)[:B]
    latn = np.broadcast_to(lat_rows[:, None], (B, B)).astype(np.float32)
    up = H // tect_T.shape[1]
    tect_full = np.stack([ndi_zoom(tect_T[i], up, order=1) for i in range(4)],
                         0)[:, :H, :H]
    tch = np.stack([block_reduce(tect_full[i], block, "mean") for i in range(4)])
    ch = np.concatenate([np.stack([bm, bstd, bmin, bmax, lf, river, glac, latn]),
                         tch], 0).astype(np.float32)
    return ch, bm, bstd, bmin, bmax, lf


@torch.no_grad()
def generate_world(seed=0, size_px=2048, ckpt_path=None,
                   tect_res=512, out_png=None, out_npy=None):
    _dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(_dir, "cache"), exist_ok=True)
    os.makedirs(os.path.join(_dir, "output"), exist_ok=True)
    ckpt_path = ckpt_path or os.path.join(_dir, "checkpoints", "best_terrain_gan_v5.pth")
    out_png = out_png or os.path.join(_dir, "output", "world_map.png")
    out_npy = out_npy or os.path.join(_dir, "output", "world_dem.npy")
    ck = torch.load(ckpt_path, map_location=device)
    assert int(ck.get("cfg", {}).get("in_ch", 0)) == 12, "请先用 v6.3 重训 (12通道)"
    gen = UNet(in_ch=12, base=32).to(device)
    gen.load_state_dict(ck["gen_ema"]); gen.eval()
    ch_stats = {"mean": np.array(ck["ch_stats_mean"], np.float32),
                "std": np.array(ck["ch_stats_std"], np.float32)}
    block = int(ck.get("cfg", {}).get("block", 8))
    tb, ovb = 256 // block, 6
    step, ovp = tb - ovb, ovb * block

    print(f"[1/4] 板块演化模拟 (T={tect_res}, 450 Myr, 有缓存)...")
    cache = os.path.join(_dir, "cache", f"plate_sim_v2_T{tect_res}_seed{seed}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        elev_T, tect_T, crust_T = d["elev"], d["tect"], d["crust"]
        print(f"      载入缓存 {os.path.basename(cache)}")
    else:
        elev_T, tect_T, crust_T = run_plate_sim(T=tect_res, seed=seed)
        np.savez_compressed(cache, elev=elev_T, tect=tect_T, crust=crust_T)

    up = size_px // tect_res
    rng = np.random.RandomState(seed + 5)
    lats = (-90.0 + (np.arange(size_px) + 0.5) * (180.0 / size_px)).astype(np.float32)
    # v6.5: 板块骨架域扭曲 (周期fBm, 数据线无缝) → 边界/海岸弧形化, 消"细胞壁"直线
    wxT = periodic_fbm(tect_res, 6.0, 3, rng) * 16.0 + periodic_fbm(tect_res, 20.0, 2, rng) * 4.0
    wyT = periodic_fbm(tect_res, 6.0, 3, rng) * 16.0 + periodic_fbm(tect_res, 20.0, 2, rng) * 4.0
    elev_T = warp_lonlat(elev_T.astype(np.float32), wxT, wyT, 1.0)
    tect_T = np.stack([warp_lonlat(tect_T[i], wxT, wyT, 1.0) for i in range(4)])
    crust_T = warp_lonlat(crust_T.astype(np.float32), wxT, wyT, 1.0)
    elev_m = ndi_zoom(elev_T, up, order=1)[:size_px, :size_px].astype(np.float32)
    # 像素级细节: 山体纹理跟随构造场 + 海岸分形噪声(基于连续crust概率场, 非二值)
    envS = np.clip(ndi_zoom(tect_T[1], up, order=1), 0, 1)[:size_px, :size_px]
    cf = gaussian_filter(crust_T, 2.0)
    cf = ndi_zoom(cf, up, order=1)[:size_px, :size_px]
    # 海岸带(420m 半高宽) 分形级联 D≈1.3: 宏观湾/半岛 → 岬角 → 碎屑岸逐级细化
    band = np.exp(-((np.abs(elev_m)) / 420.0) ** 2)
    elev_m = elev_m + band * 480.0 * (cf - 0.5)          # 宏观: 大湾/半岛 (扭曲后不规则)
    elev_m += 190.0 * value_fbm(size_px, 24.0, 4, rng=rng)
    # 山脉: 束状分支(脊状fBm) + 宽度空间变化 → 消"贴边条"直线感
    wvar = np.clip(1.0 + 0.9 * value_fbm(size_px, 5.0, 3, rng=rng), 0.25, 2.0)
    elev_m += 700.0 * envS * wvar * (ridged_fbm(size_px, base_freq=14.0, octaves=4, rng=rng) ** 2)
    elev_m += band * 300.0 * (value_fbm(size_px, 34.0, 3, rng=rng) - 0.35)   # 中观: 岬角/海湾
    elev_m += band * 150.0 * (value_fbm(size_px, 90.0, 2, rng=rng) - 0.3)
    elev_m += band * 85.0 * (value_fbm(size_px, 200.0, 2, rng=rng) - 0.28)   # 细观: 碎屑岸
    elev_m += band * 45.0 * (value_fbm(size_px, 450.0, 1, rng=rng) - 0.25)
    # 山间盆地 (费尔干纳型): 造山带内部低频凹坑
    imb = np.clip(value_fbm(size_px, 10.0, 3, rng=rng) - 1.1, 0, None)
    elev_m = elev_m - 2000.0 * imb * np.clip(envS * 1.6 - 0.2, 0, 1) * (elev_m > 150)
    # 大陆腹地盆地 (刚果/塔里木型): 距海岸远的内部低地
    d_in = distance_transform_edt(elev_m > 0)
    deep_int = np.clip((d_in - 130.0) / 240.0, 0, 1)
    basins = np.clip(value_fbm(size_px, 6.0, 3, rng=rng) - 1.0, 0, None)
    elev_m = elev_m - 1600.0 * deep_int * basins
    elev_m = np.clip(elev_m, -10500, 8900).astype(np.float32)
    landS = elev_m > 0
    plt.imsave(os.path.join(_dir, "output", "world_skeleton.png"),
               make_terrain_cmap()(m2v(elev_m))[..., :3])
    print(f"      陆地 {(elev_m > 0).mean() * 100:.1f}% | 峰 {elev_m.max():.0f}m | 沟 {elev_m.min():.0f}m")

    print("[2/4] 构建 12 通道全球粗底图 (含构造场)...")
    ch_raw, bm, bstd, bmin, bmax, lf = world_to_channels(elev_m, lats, tect_T, block, seed)
    glac = ch_raw[6]
    W = bm.shape[0]
    mu, sd = ch_stats["mean"][:, None, None], ch_stats["std"][:, None, None]
    grng = np.random.default_rng(seed + 999)
    rows = list(range(0, W, step))
    total = len(rows) ** 2
    print(f"[3/4] 逐瓦片 GAN + 低频对齐 ({len(rows)}x{len(rows)})...")
    canvas = np.zeros((size_px, size_px), np.float32)
    wgt = np.zeros((size_px, size_px), np.float32)
    ramp = np.ones(256, np.float32)
    ramp[:ovp] = 0.5 - 0.5 * np.cos(np.pi * np.arange(ovp) / ovp)
    ramp[-ovp:] = ramp[:ovp][::-1]
    done, t0 = 0, time.time()
    pad2 = lambda a, p: np.pad(a, ((0, p[0]), (0, p[1])), mode="edge")
    sm_ = lambda a: gaussian_filter(a.astype(np.float32), 0.6)
    for by in rows:
        for bx in rows:
            by1, bx1 = min(by + tb, W), min(bx + tb, W)
            hb, wb = by1 - by, bx1 - bx
            py, px_ = tb - hb, tb - wb
            sl = (slice(by, by1), slice(bx, bx1))
            ch = ch_raw[:, sl[0], sl[1]] + \
                 grng.normal(0, 0.015, (12, hb, wb)).astype(np.float32)
            ch = (ch - mu) / sd
            if py or px_:
                ch = np.pad(ch, ((0, 0), (0, py), (0, px_)), mode="edge")
            inp = np.ascontiguousarray(np.repeat(np.repeat(ch, block, 1), block, 2))
            raw = gen(torch.from_numpy(inp[None]).float().to(device)).cpu().numpy()[0, 0]
            s = lowfreq_consistency(raw.astype(np.float32),
                                    pad2(sm_(bm[sl]), (py, px_)), block)

            # 深海块的硬上界 = 块均值+0.06 (比训练版cap=0.46强得多:
            # 0.46仍在海平面附近, 会把深海钳成浅海; 这版直接钳回深海)
            lf_up = ndi_zoom(pad2(lf[sl], (py, px_)), block, order=1)[:256, :256]
            bm_up = ndi_zoom(pad2(sm_(bm[sl]), (py, px_)), block, order=1)[:256, :256]
            sea_m = lf_up < 0.02
            cap_m = np.where(sea_m, np.maximum(bm_up + 0.06, 0.12), 1.0)
            s = np.minimum(s, cap_m).astype(np.float32)

            h, w = hb * block, wb * block
            win = np.outer(ramp, ramp)
            if by == 0: win[:ovp, :] = 1.0
            if bx == 0: win[:, :ovp] = 1.0
            if by1 == W and hb >= ovb: win[max(0, h - ovp):, :] = 1.0
            if bx1 == W and wb >= ovb: win[:, max(0, w - ovp):] = 1.0
            y0p, x0p = by * block, bx * block
            canvas[y0p:y0p + h, x0p:x0p + w] += s[:h, :w] * win[:h, :w]
            wgt[y0p:y0p + h, x0p:x0p + w] += win[:h, :w]
            done += 1
            if done % 50 == 0 or done == total:
                print(f"      瓦片 {done}/{total} | {time.time() - t0:.0f}s", flush=True)

    out = np.clip(canvas / np.maximum(wgt, 1e-8), 0, 1).astype(np.float32)
    print("[4/4] 全局阶段: CDF匹配 → 低频复核 → 去斑 → 水力侵蚀 → 填洼D8河网...")
    out = global_envelope_match(out, bm, bstd, bmin, bmax, glac, block,
                                blend=0.60, seed=seed, cap_m=8200.0)
    out = lowfreq_consistency(out, bm, block, iters=2)
    out = despeckle(out, min_area=64)
    out = droplet_erosion(out, rng=np.random.default_rng(seed + 77))   # V形谷/冲积扇
    out = sharpen_coastline(out)
    ice = polar_ice_mask(lats, out)
    filled = fill_depressions(out)                                     # 河网连通
    acc = d8_flow_accumulation(filled, np.random.RandomState(seed + 31))
    thr = max(float(np.quantile(acc, 0.995)), 2.0)
    t = np.clip(np.log(np.maximum(acc, 1.0001)) / np.log(thr), 0, 1)
    landness = np.clip((out - 0.5) / 0.02, 0, 1)
    out = np.clip(out - 0.035 * (t ** 0.8) * landness * (~ice), 0, 1).astype(np.float32)

    dem_m = v2m(out)
    np.save(out_npy, dem_m.astype(np.float32))
    plt.imsave(out_png, render_world(out, lats, landS, make_terrain_cmap(),
                                     np.random.RandomState(seed)))
    print(f"\n完成: {out_png} + {out_npy} + world_skeleton.png")
    print(f"  陆地 {(dem_m > 0).mean() * 100:.1f}% | 峰 {dem_m.max():.0f}m | 沟 {dem_m.min():.0f}m")
    return dem_m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=2048, choices=[2048, 4096])
    ap.add_argument("--ckpt", default=None)   # 默认: 脚本目录内 best_terrain_gan_v5.pth
    ap.add_argument("--tect-res", type=int, default=512)
    a = ap.parse_args()
    generate_world(seed=a.seed, size_px=a.size, ckpt_path=a.ckpt, tect_res=a.tect_res)