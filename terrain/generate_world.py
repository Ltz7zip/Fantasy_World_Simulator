# -*- coding: utf-8 -*-
# ======================================================================
# generate_world.py v6.6 —— 板块演化模拟世界生成器
# 一级: run_plate_sim v2.2 (450 Myr, 有缓存) → 粗DEM + 4构造场
#       海洋直方图标定(年龄分位→地球开阔洋深度) + 陆架剖面化 + 窄海沟
# 二级: 12通道条件 → 逐瓦片重训版GAN → 低频对齐 → 全局CDF匹配
#       → 全球D8河网 → 雪线渲染
# v6.6 检验修复层 (基于10种子七维地理检验报告):
#   P0 陆架-陆坡剖面修复 + 陆地直方图重映射(→地球实测分位)
#   P1 岛弧链/热点岛链盖章 + 干流河谷雕刻
#   P2 板内造山带盖章(打破山地-俯冲带过纯耦合)
# 运行: python generate_world.py --seed 7 --size 2048
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
                            periodic_fbm, warp_lonlat,
                            land_hist_match, ocean_margin_fix, add_island_arcs,
                            add_hotspot_chains, add_intraplate_orogen,
                            carve_trunk_rivers, land_hist_final,
                            carve_fjords, add_shelf_peninsulas,
                            add_coral_reefs, ocean_current_effects,
                            add_craton_basins, ocean_hist_final)
from plate_sim import run_plate_sim


def fbm1d(n, sigma, rng):
    """在频率域构造一维粉红噪声（fBm 近似）。

    实现：对标准正态白噪声做 rFFT → 乘以高斯带宽 ``exp(−(f·σ)²)``
    （``σ`` 越大越平滑，越低通）→ 逆 rFFT 回到空间域。
    用于海冰边缘纬度、雪线纬度波动等 **需要长程连续** 的一维曲线。

    Parameters
    ----------
    n : int
        输出长度（像素数 / 纬度行数等）。
    sigma : float
        高斯带宽的尺度因子。常见取值 80~640，越大谱越窄、越平滑。
    rng : numpy.random.Generator or RandomState
        任意可调用 ``.randn(n)`` 的随机数生成器。

    Returns
    -------
    ndarray of shape (n,), dtype float64
        零均值的一维 fBm 近似值（幅度量级约为 1σ 单位白噪声）。
    """
    x = rng.randn(n)
    X = np.fft.rfft(x)
    X *= np.exp(-(np.fft.rfftfreq(n) * sigma) ** 2)
    return np.fft.irfft(X, n)


def despeckle(v01, min_area=48):
    """去斑：清除面积过小的陆地块与海洋孔洞（联通分量域）。

    原理：
        1. 对陆地（v ≥ 0.5）做连通域标注，若某连通域像素数 < ``min_area``，
           则以 9×9 最小值滤波覆盖（把小岛腐蚀成海洋，不产生洞）。
        2. 再对海洋（v < 0.5）做同等处理，用 9×9 最大值滤波回填（清除
           陆地内部的小淡水湖 / 盐沼噪点）。

    Parameters
    ----------
    v01 : ndarray (S, S), float32, 值域 [0, 1]
        归一化高程（v01 = m2v(dem_m)）。
    min_area : int
        最小联通域面积（像素）。v6.6 默认 48 → 去斑后为 32（保住岛弧小岛）。

    Returns
    -------
    ndarray (S, S), float32
        去斑后的归一化高程。
    """
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
    """全局一次 CDF 匹配 + 逐像素块包络裁剪 + 行星峰值截断 + 冰盖低 blend 保纹理。

    GAN 生成的原始 ``[0, 1]`` 归一化高程会有整体分布失真（峰不够高、洋不够深）
    的问题。本函数把它按分位对齐到一个更接近地球的"参考分布"，同时在每个粗块
    上用 ``bmin/bmax`` 做软包络约束（避免 GAN 在单块上产生跳变极端值），并在
    冰川覆盖区降低 blend 比例（避免把冰盖内部纹理磨平）。

    算法简述：
        1. 以像素为单位从 ``bm/bstd/bmin/bmax`` 采样一个"合理值"：
           ``s = Normal(bm, bstd)`` → 夹到 ``[bmin, bmax]``（每像素独立样本）。
        2. 把 ``s`` 与 ``v01`` 同时按排序（ranked）配对 → 得到基于 CDF 的
           ``matched``（单调，保留 v01 空间结构）。
        3. 块包络（上采样后的 bmin/bmax）软夹一次 + 行星峰值 ``cap_m`` 硬钳。
        4. 与原始 v01 按 blend 线性混合（冰盖区 blend 下调 0.30）。

    Parameters
    ----------
    v01 : ndarray (H, W), float32, [0, 1]
        GAN 原始输出。
    bm, bstd, bmin, bmax : ndarray (W//block, H//block), float32
        粗块均值 / 标准差 / 最小值 / 最大值（来自 ``world_to_channels``）。
    glac : ndarray (同粗块尺寸)
        每块冰川覆盖率（0~1），用于降低高冰盖区的 blend。
    block : int
        块大小（训练时 cfg.block；v5=8）。与粗块尺寸匹配的上采样基。
    blend : float
        线性混合权重（1 = 完全匹配替换，0 = 完全保留 GAN 输出）。v6.6 默认 0.60。
    seed : int
        用于生成"合理值"参考样本的随机种子。
    cap_m : float
        行星绝对峰值（米），v6.6 取 8200m（先 m2v 再钳制）。

    Returns
    -------
    ndarray (H, W), float32, [0, 1]
        匹配 + 夹取 + 混合后的归一化高程。
    """
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
    """气候因果链终态渲染：温度 / 降水 → Whittaker 生物群系色 → 叠加地形阴影 / 河湖 / 雪线 / 海冰。

    渲染管线（顺序执行，后续元素覆盖前面）：
        1. 温度场 T + 降水场 P + 洋流/季风 dT/dP 异常 → Whittaker 9 类生物群系
           → ``BIOME_COLORS[bi]`` 陆地配色（海洋直接用地形色卡）。
        2. NW 方向光照山体阴影（陆地渐变明暗 × 海拔渐亮）→ 增强地形层次感。
        3. 湖泊：形态学填洼面与实际地形差 > 0.012v 的陆地区域。
        4. 河流：填洼后 D8 流累积 > 99.5 分位（干流 5× 分位加宽加深）。
        5. 雪线：纬度雪线（干旱抬升 +800m）∪ 温度雪线（T ≤ −12°C）× 山脊纹理骨架。
        6. 海冰：纬度 64.5°±2.5° 边界 × 浮冰 fBm 噪声 × 极地硬保障。

    Parameters
    ----------
    v01 : ndarray (S, S), float32, [0, 1]
        最终归一化高程（v6.6 全部后处理后的结果）。
    lats : ndarray (S,)
        行对应的纬度（像素中心对齐，递增）。
    landS : ndarray (S, S) bool
        陆地掩膜（提前缓存，避免多次计算）。
    cmap : matplotlib Colormap
        海洋部分使用的地形色卡（通常是 ``make_terrain_cmap()`` 的返回值）。
    rng : numpy.random.RandomState
        随机数生成器（供 fBm / 浮冰 / 降水场采样）。

    Returns
    -------
    ndarray (S, S, 3), float32, [0, 1]
        RGB 图像。直接 ``plt.imsave`` 或 ``PIL.Image.fromarray((rgb*255).astype(uint8))``
        均可得到与 ``world_map.png`` 一致的 PNG 图像。
    """
    S = v01.shape[0]
    land = v01 >= 0.5
    elev_m = v2m(v01)
    dT, dP = ocean_current_effects(elev_m, lats)     # M4 洋流/季风气候效应
    T = temperature_field(lats, elev_m, dT)
    P = precipitation_field(lats, elev_m, rng, dP)
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
    """把米制高程 + 四通道构造场编码为 12 通道条件（条件 GAN 输入的粗底图）。

    通道布局（索引 → 含义），与 ``terrain_engine.py`` 训练管线严格一致：
        [0] 块均值 bm        [1] 块标准差 bstd
        [2] 块最小值 bmin      [3] 块最大值 bmax
        [4] 海陆比例 lf       [5] 河网 (log 归一化，仅陆地可见)
        [6] 冰川覆盖率         [7] 绝对纬度/90°
        [8] 俯冲/碰撞构造场    [9] 离散（裂谷/洋脊）构造场
        [10] 转换断层构造场   [11] 其余构造场（第 0 通道）

    Parameters
    ----------
    elev_m : ndarray (H, H), float32
        米制高程骨架（像素级细化阶段之后、GAN 之前的结果）。
    lats : ndarray (H,)
        像素中心对齐纬度。
    tect_T : ndarray (4, T, T)
        板块模拟器输出的四通道构造场（将被上采样到 H 后再块化）。
    block : int
        块大小（v5 训练为 8）；输出通道空间尺寸 = H/block。
    seed : int
        D8 流累积等的随机种子。

    Returns
    -------
    tuple (ch, bm, bstd, bmin, bmax, lf)
        - ch: ndarray (12, H//block, H//block), float32 — 12 通道条件张量
        - bm/bstd/bmin/bmax/lf: ndarray (B, B) — 粗块统计（供后处理复用）。
    """
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
    """**项目主入口**：生成一整个幻想世界的米制 DEM + 气候终态渲染 PNG。

    四大阶段（1/4 → 4/4）：
        1. 板块演化模拟（``plate_sim.run_plate_sim``，450 Myr，有缓存）：
           返回粗分辨率构造骨架 ``elev_T, tect_T, crust_T, margin_T, belt_T, rift_T``。
           命中缓存（``cache/plate_sim_v3_T{tect_res}_seed{seed}.npz``）时可从
           2~4 分钟缩到 < 10s。
        2. 像素级骨架细化：周期 fBm 域扭曲（消 Voronoi 直线感）+ 上采样到
           ``size_px`` + 海岸分形级联（D≈1.3）+ 山脉/盆地 + **v6.6 七大
           地理维度修复层**（陆架剖面、板内造山、陆地直方图、岛弧链、热点岛链…）。
           此阶段结果保存为 ``world_skeleton.png`` 便于检查大陆走向。
        3. 逐瓦片 12 通道条件 GAN：把粗 DEM 编码为 12 通道（块统计 / 海陆 /
           河网 / 冰川 / 纬度 / 4 构造场），UNet 256² 瓦片推理 + 余弦羽化拼贴
           回原始分辨率，同时做低频一致性拉回与深海硬钳制（防 GAN 幻浅滩）。
        4. 全局后处理：``global_envelope_match`` CDF 包络匹配 → 去斑 → 粒子
           水力侵蚀 → 干流河谷雕刻 → 海岸锐化 → 峡湾雕刻 → 陆架终检 →
           填洼 D8 河网 → **陆地直方图终检**（绝对最后一步）→ 裂谷重新下切 →
           克拉通盆地 → 台地半岛 → 珊瑚礁 → **深洋直方图终检 M6**。
           最终 ``world_dem.npy`` 同时满足陆地、海洋两个维度的直方图分位。

    Parameters
    ----------
    seed : int
        世界种子，决定一切随机性（板块速度场 / fBm / GAN 噪声 / 气候采样）。
        推荐非负整数；任意整数均可。
    size_px : {2048, 4096}
        目标 DEM 单边像素数。4096 ≈ 4× 耗时 + 16× 显存，需要 ≥32GB RAM。
    ckpt_path : str or None
        UNet 权重路径。空时回退为 ``terrain/checkpoints/best_terrain_gan_v5.pth``。
        若权重的 ``cfg.in_ch != 12`` 会抛 ``AssertionError``（FAQ Q2）。
    tect_res : int
        板块模拟分辨率（默认 512，与缓存匹配；请与 TECHNICAL §2 一致修改）。
    out_png / out_npy : str or None
        渲染图 / DEM 输出路径。空时分别写为 ``terrain/output/world_map.png``
        与 ``terrain/output/world_dem.npy``。

    Returns
    -------
    ndarray of float32, shape ``(size_px, size_px)``
        最终米制高程（正值陆地 / 负值海洋），范围经钳制到 [−10500, +8900] m。
        同时磁盘已写出 ``world_dem.npy``、``world_map.png``、``world_skeleton.png``。

    Notes
    -----
    本函数全程使用确定性随机数（numpy RandomState + torch.manual_seed 由权重承载），
    同一 seed 多次运行**严格 bitwise 一致**；只有删除板块缓存或修改源代码时才会
    改变地形。
    """
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
    cache = os.path.join(_dir, "cache", f"plate_sim_v3_T{tect_res}_seed{seed}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        elev_T, tect_T, crust_T = d["elev"], d["tect"], d["crust"]
        margin_T = d["margin"] if "margin" in d.files else None
        belt_T = d["belt"] if "belt" in d.files else None
        rift_T = d["rift"] if "rift" in d.files else None
        print(f"      载入缓存 {os.path.basename(cache)}")
    else:
        elev_T, tect_T, crust_T, margin_T, belt_T, rift_T = run_plate_sim(
            T=tect_res, seed=seed)
        np.savez_compressed(cache, elev=elev_T, tect=tect_T, crust=crust_T,
                            margin=margin_T, belt=belt_T, rift=rift_T)

    up = size_px // tect_res
    rng = np.random.RandomState(seed + 5)
    lats = (-90.0 + (np.arange(size_px) + 0.5) * (180.0 / size_px)).astype(np.float32)
    # v6.5: 板块骨架域扭曲 (周期fBm, 数据线无缝) → 边界/海岸弧形化, 消"细胞壁"直线
    # v7.0: 强度 16→11px (16px 会把裂谷端部窄缝的大陆桥接成单块)
    wxT = periodic_fbm(tect_res, 6.0, 3, rng) * 11.0 + periodic_fbm(tect_res, 20.0, 2, rng) * 3.0
    wyT = periodic_fbm(tect_res, 6.0, 3, rng) * 11.0 + periodic_fbm(tect_res, 20.0, 2, rng) * 3.0
    elev_T = warp_lonlat(elev_T.astype(np.float32), wxT, wyT, 1.0)
    tect_T = np.stack([warp_lonlat(tect_T[i], wxT, wyT, 1.0) for i in range(4)])
    crust_T = warp_lonlat(crust_T.astype(np.float32), wxT, wyT, 1.0)
    # v3.0 margin_type 同步扭曲+上采样 (被动/主动边缘分化)
    if margin_T is not None:
        margin_T = warp_lonlat(margin_T.astype(np.float32), wxT, wyT, 1.0)
        margin_up = np.clip(ndi_zoom(margin_T, up, order=0)[:size_px, :size_px],
                            0, 2).astype(np.int8)
    else:
        margin_up = None
    # M2 延展山带同步扭曲+上采样 → 骨架山脉纹理沿带走向
    if belt_T is not None:
        belt_T = warp_lonlat(belt_T.astype(np.float32), wxT, wyT, 1.0)
        beltS = np.clip(ndi_zoom(belt_T, up, order=1)[:size_px, :size_px],
                        0, 1).astype(np.float32)
    else:
        beltS = np.zeros((size_px, size_px), np.float32)
    # M2 失败裂谷场同步扭曲+上采样 (直方图终检后重新下切用)
    if rift_T is not None:
        rift_T = warp_lonlat(rift_T.astype(np.float32), wxT, wyT, 1.0)
        riftS = np.clip(ndi_zoom(rift_T, up, order=1)[:size_px, :size_px],
                        0, 1).astype(np.float32)
    else:
        riftS = np.zeros((size_px, size_px), np.float32)
    elev_m = ndi_zoom(elev_T, up, order=1)[:size_px, :size_px].astype(np.float32)
    # 像素级细节: 山体纹理跟随构造场 + 海岸分形噪声(基于连续crust概率场, 非二值)
    envS = np.clip(ndi_zoom(tect_T[1], up, order=1), 0, 1)[:size_px, :size_px]
    cf = gaussian_filter(crust_T, 2.0)
    cf = ndi_zoom(cf, up, order=1)[:size_px, :size_px]
    # 海岸带(420m 半高宽) 分形级联 D≈1.3: 宏观湾/半岛 → 岬角 → 碎屑岸逐级细化
    # v7.0: 海洋侧幅度减半 → 陆架浅海不被噪声大量转陆 (land_frac 超标源头)
    band = np.exp(-((np.abs(elev_m)) / 420.0) ** 2)
    band_o = band * np.where(elev_m < 0, 0.5, 1.0)       # 海侧噪声减半
    elev_m = elev_m + band_o * 480.0 * (cf - 0.5)        # 宏观: 大湾/半岛 (扭曲后不规则)
    elev_m += 190.0 * value_fbm(size_px, 24.0, 4, rng=rng)
    # 山脉: 束状分支(脊状fBm) + 宽度空间变化 → 消"贴边条"直线感
    # M2: 纹理场 = max(构造场, 0.85×延展山带) → 像素级山体沿延展带走向连续
    wvar = np.clip(1.0 + 0.9 * value_fbm(size_px, 5.0, 3, rng=rng), 0.25, 2.0)
    elev_m += 700.0 * np.maximum(envS, 0.85 * beltS) * wvar \
              * (ridged_fbm(size_px, base_freq=14.0, octaves=4, rng=rng) ** 2)
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
    # ---- v6.6 检验修复层 (七大地理维度对齐) ----
    elev_m = ocean_margin_fix(elev_m, envS)                # P0 陆架-陆坡剖面修复
    elev_m = add_intraplate_orogen(elev_m, cf, envS, rng)  # P2 板内造山(参与直方图)
    elev_m = land_hist_match(elev_m, lats)                 # P0 陆地直方图 → 地球实测(加权)
    elev_m = add_island_arcs(elev_m, envS, cf, rng)        # P1 岛弧链 (俯冲带大陆侧)
    elev_m = add_hotspot_chains(elev_m, rng)               # P1 热点岛链 (夏威夷型)
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
            sea_m = lf_up < 0.006                   # v6.6: 放宽→保住岛弧/小岛
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
    out = despeckle(out, min_area=32)             # v6.6: 64→32 保住岛弧小岛
    out = droplet_erosion(out, rng=np.random.default_rng(seed + 77))   # V形谷/冲积扇
    out = np.clip(out + 0.016 * value_fbm(size_px, 110.0, 2,
                    rng=np.random.RandomState(seed + 3)) * (out >= 0.5),
                  0, 1).astype(np.float32)        # v6.6 高频细节(提PSD)
    out = carve_trunk_rivers(out, np.random.RandomState(seed + 55))    # v6.6 干流河谷
    # v6.6: 窄带海岸锐化(±0.008v≈±160m) → 不再压掉陆架/抬升陆地直方图
    out = sharpen_coastline(out, band=0.008, expand=0.03)
    # M3 峡湾雕刻: 高纬被动边缘 U 形谷 (斯堪的纳维亚型), 锐化后/陆架终检前
    if margin_up is not None:
        out = m2v(carve_fjords(v2m(out), lats, margin_up,
                               np.random.RandomState(seed + 41)))
    out = m2v(ocean_margin_fix(v2m(out), envS, margin_up))   # v6.6 陆架终检(v3.0边缘分化)
    ice = polar_ice_mask(lats, out)
    filled = fill_depressions(out)                                     # 河网连通
    acc = d8_flow_accumulation(filled, np.random.RandomState(seed + 31))
    thr = max(float(np.quantile(acc, 0.995)), 2.0)
    t = np.clip(np.log(np.maximum(acc, 1.0001)) / np.log(thr), 0, 1)
    landness = np.clip((out - 0.5) / 0.02, 0, 1)
    out = np.clip(out - 0.035 * (t ** 0.8) * landness * (~ice), 0, 1).astype(np.float32)
    out = land_hist_final(out, lats)              # v6.6 陆地直方图终检(绝对最后,加权)
    # M2 失败裂谷重新下切: GAN/直方图重排会抹掉骨架版谷地 → rift 场
    # (轴剖面: 轴线1→两侧衰减) 直乘深度, 谷形集中 (东非型);
    # 谷底整体低于两侧 + 轴线稀疏火山峰 (乞力马扎罗型)
    if riftS.any():
        valley = 0.055 * riftS                     # ≈1100m 谷底 (东非裂谷谷底-周边 700-1500m)
        out = np.clip(out - valley * (out >= 0.5), 0, 1).astype(np.float32)
        volc = (riftS > 0.75) & (out >= 0.5) \
               & (value_fbm(size_px, 40.0, 2, rng=np.random.RandomState(seed + 9))
                  > 0.55)
        out = np.clip(out + 0.030 * volc, 0, 1).astype(np.float32)

    # M5 克拉通大流域盆地 (直方图终检后盖章 — 终检全局重排会扭曲河网梯度):
    # 稳定陆内低地 → 单一出口倾斜面 + 河网树 (亚马逊/刚果型)
    # + 内陆海 (里海型); 输入已是最终直方图 → 盖章值即终值
    dem_b, n_b, b_info = add_craton_basins(v2m(out), envS,
                                           np.random.RandomState(seed + 66),
                                           belt=beltS, lats=lats)
    if n_b:
        out = m2v(dem_b)
        print(f"      M5 大流域盆地: {n_b}个 [{b_info}]")

    # ---- M3 边缘地貌终章 (直方图终检后, 最后盖章不被重排) ----
    dem_m = v2m(out)
    if margin_up is not None:
        dem_m, n_pen = add_shelf_peninsulas(dem_m, margin_up, cf, rng,
                                            lats=lats)   # 台地半岛+大陆岛
        print(f"      M3 台地露头(半岛/大陆岛): {n_pen}px")
    dem_m = add_coral_reefs(dem_m, lats, np.random.RandomState(seed + 88))  # 珊瑚礁
    dem_m = ocean_hist_final(dem_m, lats)      # M6 深洋直方图终检 (最后, 深带校准)
    out = m2v(dem_m)

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