# -*- coding: utf-8 -*-
"""
generate_world.py v7.0 —— 板块演化模拟世界生成器

=============================================================
一级: run_plate_sim v3.0 (450 Myr, 有缓存) → 粗DEM + 4构造场
      海洋直方图标定(年龄分位→地球开阔洋深度) + 陆架剖面化 + 窄海沟
二级: 12通道条件 → 逐瓦片重训版GAN → 低频对齐 → 全局CDF匹配
      → 全球D8河网 → 雪线渲染
v7.0 七大地理维度修复层（M1-M6） (基于10种子七维地理检验报告):
  P0 陆架-陆坡剖面修复 + 陆地直方图重映射(→地球实测分位)
  P1 岛弧链/热点岛链盖章 + 干流河谷雕刻
  P2 板内造山带盖章(打破山地-俯冲带过纯耦合)
运行: python generate_world.py --seed 7 --size 2048
=============================================================
"""
import os
import time
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import (gaussian_filter, distance_transform_edt,
                           zoom as ndi_zoom, label as ndi_label,
                           minimum_filter, maximum_filter, binary_dilation,
                           median_filter)
from terrain_engine import (device, m2v, v2m, block_reduce, d8_flow_accumulation,
                            snowline_m, value_fbm, ridged_fbm, sharpen_coastline,
                            directional_band_noise,
                            lowfreq_consistency, suppress_hallucinated_land,
                            polar_ice_mask, make_terrain_cmap, UNet,
                            temperature_field, precipitation_field, whittaker_biome,
                            BIOME_COLORS, fill_depressions, droplet_erosion,
                            periodic_fbm, warp_lonlat,
                            land_hist_match, ocean_margin_fix, add_island_arcs,
                            add_hotspot_chains, add_intraplate_orogen,
                            carve_trunk_rivers, land_hist_final,
                            climate_flow_accumulation, paint_rivers_rgb,
                            carve_fjords, add_shelf_peninsulas,
                            add_coral_reefs, ocean_current_effects,
                            add_craton_basins, ocean_hist_final,
                            continentality_index, smooth_biome_labels,
                            enhance_ocean_floor, classify_landforms)
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


def render_world(v01, lats, landS, cmap, rng, tect_ctx=None):
    """气候因果链终态渲染：温度 / 降水 → Whittaker 生物群系色 → 叠加地形阴影 / 河湖 / 雪线 / 海冰。

    渲染管线（顺序执行，后续元素覆盖前面）：
        1. 温度场 T + 降水场 P + 洋流/季风 dT/dP 异常 → Whittaker 9 类生物群系
           → ``BIOME_COLORS[bi]`` 陆地配色（海洋直接用地形色卡）。
        2. NW 方向光照山体阴影（陆地渐变明暗 × 海拔渐亮）→ 增强地形层次感。
        3. 湖泊：形态学填洼面与实际地形差 > 0.012v 的陆地区域。
        4. 河流：填洼后 D8 流累积 > 99.5 分位（干流 5× 分位加宽加深）。
        5. 雪线：纬度雪线（干旱抬升 +800m）∪ 温度雪线（T ≤ −12°C）× 山脊纹理骨架。
        6. 海冰：纬度 64.5°±2.5° 边界 × 浮冰 fBm 噪声 × 极地硬保障。
        7. v7.8 地貌纹理：盆地/山地高度+成因/高原切割+侵蚀/丘陵高低的差异化纹理.

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
    tect_ctx : dict, optional
        构造场上下文 (envS/beltS/cf/tect_T/margin_up/riftS). None 时地貌子类退化为旧 v7.7.

    Returns
    -------
    ndarray (S, S, 3), float32, [0, 1]
        RGB 图像。直接 ``plt.imsave`` 或 ``PIL.Image.fromarray((rgb*255).astype(uint8))``
        均可得到与 ``world_map.png`` 一致的 PNG 图像。
    """
    S = v01.shape[0]
    land = v01 >= 0.5
    elev_m = v2m(v01)
    ci = continentality_index(elev_m, lats)              # v7.1 大陆度指数 (一次计算, 多函数复用)
    dT, dP = ocean_current_effects(elev_m, lats)         # M4 洋流/季风气候效应
    T = temperature_field(lats, elev_m, dT, ci=ci)       # v7.1: 大陆度温度修正 + 极地缓和
    P = precipitation_field(lats, elev_m, rng, dP, ci=ci)# v7.1: 大陆度降水 + 副高 + 屏障 + 长波扰动
    # ---- v7.7: T 场高斯平滑 (σ=2.5) + P 预平滑 (σ=1.5), 消除等温/等雨线像素级抖动 ----
    #   P 场在 precipitation_field 末尾已做 [2.2,1.2] 高斯, 再加 σ=1.5 总平滑≈3.0;
    #   T 场 σ=2.5≈50km 对应真实 ecotone 宽度下限, 过大会模糊气候带边界.
    T_s = gaussian_filter(T.astype(np.float32), 2.5)
    P_s = gaussian_filter(P.astype(np.float32), 1.5)
    bi = whittaker_biome(T_s, P_s, lats=lats, ci=ci)   # v7.1: 苔原/泰加林/草原门限大陆度自适应
    # ---- v7.7: 群系边界平滑 + 碎片清理 (7×7 中值 + <50px 碎片清理 + 3轮迭代) ----
    #   7px≈137km 对应 ecotone 尺度; 50px≈980km²(塞浦路斯级)以下飞地无气候意义;
    #   3轮收敛消除中值产生的 1-2px 新碎片. 目标: 温带落叶林<300块, 热带季节林<250块.
    bi = smooth_biome_labels(bi, land, median_size=7, min_patch_px=50, iters=3)
    # 山体阴影 (NW 光照, 仅陆地) → 地形层次感
    g = gaussian_filter(v01, 1.5)
    gy, gx = np.gradient(g)
    hs = np.clip(0.88 - 22.0 * (gx + gy), 0.60, 1.15)
    shade = (0.78 + 0.55 * np.clip((v01 - 0.5) / 0.5, 0, 1)) * (0.72 + 0.28 * hs)
    land_rgb = BIOME_COLORS[bi] * shade[..., None]
    # ---- v7.9: 群系边界颜色软过渡 (修复整数索引导致的方块状硬切换) ----
    #   整数标签索引 BIOME_COLORS[bi] 让相邻群系颜色像素级硬切换, 配合中值滤波的
    #   网格对齐效应 → 斜边界呈"阶梯/方块"状. 这里在渲染端做边界带颜色软混合:
    #     1. 找群系边界像素 = 陆地内 4 邻域存在不同群系标签的像素
    #     2. 距离场 d = 陆地像素到最近群系边界的欧氏距离 (mask=land 限制陆地)
    #     3. 边界带 d≤band 内, 用 σ=band/2 各向同性高斯混合 land_rgb → 颜色渐变
    #        群系核心 (d>band) 保持纯色, 边界做加权过渡 → 对应真实生态交错带 ecotone
    #   物理: 真实群系过渡带宽 30-100km, 2048px 下 band=3≈60km 对应中值尺度.
    if land.any():
        bi_p = np.pad(bi, 1, mode='edge')
        bnd = land & (
            (bi_p[1:-1, 1:-1] != bi_p[:-2, 1:-1]) |   # 上邻不同
            (bi_p[1:-1, 1:-1] != bi_p[2:, 1:-1]) |    # 下邻不同
            (bi_p[1:-1, 1:-1] != bi_p[1:-1, :-2]) |   # 左邻不同
            (bi_p[1:-1, 1:-1] != bi_p[1:-1, 2:]))     # 右邻不同
        if bnd.any():
            # 全图距离场 → 仅取陆地部分 (海洋 d 置 0, 后续被 rgb.where 覆盖)
            d = np.where(land, distance_transform_edt(~bnd), 0.0).astype(np.float32)
            band = 3.0
            w = np.clip(1.0 - d / band, 0.0, 1.0).astype(np.float32)  # 边界=1, 核心=0
            if w.any():
                # 各向同性高斯混合 (σ=band/2, 圆形核 → 不会引入方向性方块)
                land_rgb_b = np.empty_like(land_rgb)
                for c in range(land_rgb.shape[2]):
                    land_rgb_b[..., c] = gaussian_filter(land_rgb[..., c],
                                                         sigma=band * 0.5)
                land_rgb = ((1.0 - w[..., None]) * land_rgb
                            + w[..., None] * land_rgb_b).astype(np.float32)
    rgb = cmap(np.clip(v01, 0, 1))[..., :3]
    # ---- v7.7: 海底分层着色 (大陆边缘/大洋盆地/大洋中脊 三大单元 + 次级纹理) ----
    # ---- v8.0: 软深度权重替代硬阈值 mask, 消除 -2000/-5500m 等深线颜色阶跃
    #           (旧版在岛屿周边深度跨阈值时产生方块色环; 现 200m 过渡带平滑混合) ----
    sea = ~land
    if sea.any():
        depth = elev_m
        # 软深度权重 (200m tanh 过渡): w_margin 在 0~-2000m=1, w_basin 在 -2000~-5500m=1, w_abyss 在 <-5500m=1
        _dt = 200.0  # 颜色过渡带宽 (米)
        w_margin = 0.5 * (1.0 - np.tanh((depth + 2000.0) / _dt)) * sea
        w_basin = 0.5 * (np.tanh((depth + 2000.0) / _dt)
                         - np.tanh((depth + 5500.0) / _dt)) * sea
        w_abyss = 0.5 * (1.0 + np.tanh((depth + 5500.0) / _dt)) * sea
        # 各区目标颜色 (rgb 是 cmap 原始色)
        margin_color = rgb * 0.6 + np.array([0.18, 0.42, 0.55], np.float32) * 0.4
        basin_color = rgb * 0.5 + np.array([0.05, 0.18, 0.38], np.float32) * 0.5
        abyss_color = rgb * 0.4 + np.array([0.02, 0.08, 0.20], np.float32) * 0.6
        # 软混合三区颜色 (权重和=1 in sea, 0 in land)
        sea_rgb = (w_margin[..., None] * margin_color
                   + w_basin[..., None] * basin_color
                   + w_abyss[..., None] * abyss_color).astype(np.float32)
        # 陆地保持原色 (rgb), 仅海洋部分替换
        rgb = np.where(sea[..., None], sea_rgb, rgb).astype(np.float32)
        # 洋中脊相对隆起 (深海中相对浅的条带) 提亮 — 保留旧版硬阈值 (局部细节, 不影响块状)
        ridge_relieve = sea & (depth > -6000.0) & (depth < -4000.0)
        # 用局部起伏找脊: 比邻域均值浅 → 提亮
        loc_mean = gaussian_filter(depth.astype(np.float32), 8.0)
        ridge_band = ridge_relieve & (depth > loc_mean + 150.0)
        sea_rgb2 = rgb.copy()
        sea_rgb2[ridge_band] = (sea_rgb2[ridge_band] * 0.7
                               + np.array([0.10, 0.22, 0.40], np.float32) * 0.3)
        rgb = sea_rgb2
    # ---- v7.8: 陆地地貌纹理 (盆地/山地高度+成因/高原切割+侵蚀/丘陵高低) ----
    #   先记录地貌掩膜, land_rgb 覆盖后再叠加纹理, 避免被覆盖.
    tc = tect_ctx or {}
    lf = classify_landforms(elev_m, lats, land=land,
                            envS=tc.get('envS'), beltS=tc.get('beltS'),
                            cf=tc.get('cf'), tect_T=tc.get('tect_T'),
                            margin_up=tc.get('margin_up'), riftS=tc.get('riftS'))
    # 粗类掩膜 (位打包: 用 &0x0F 取粗类, 向后兼容)
    dune_m = ((lf & 0x0F) == 4)
    plateau_m = ((lf & 0x0F) == 2)
    mtn_m = ((lf & 0x0F) == 0)
    basin_m = ((lf & 0x0F) == 6)
    hill_m = ((lf & 0x0F) == 5)
    rgb = np.where(land[..., None], land_rgb, rgb).astype(np.float32)
    # 沙丘 (4): 叠加纵向条纹纹理
    if dune_m.any():
        stripes = directional_band_noise(S, direction_deg=15.0,
                                         freq_center=20.0, anisotropy=0.6, rng=rng)
        stripes = np.clip((stripes + 0.3), 0, 1)
        rgb[dune_m] = rgb[dune_m] * 0.7 + \
            np.array([0.76, 0.62, 0.40], np.float32) * stripes[dune_m, None] * 0.3
    # 高原 (2): 叠加等高线纹理
    if plateau_m.any():
        contour = np.sin(elev_m * 0.02) * 0.5 + 0.5
        rgb[plateau_m] = rgb[plateau_m] * 0.75 + \
            np.array([0.55, 0.50, 0.42], np.float32) * contour[plateau_m, None] * 0.25
    # 山地 (0): 叠加岩石纹理 (高频噪声)
    if mtn_m.any():
        rock = value_fbm(S, base_freq=40.0, octaves=3, gain=0.5, rng=rng)
        rock = np.clip(rock + 0.5, 0, 1)
        rgb[mtn_m] = rgb[mtn_m] * 0.65 + \
            np.array([0.40, 0.36, 0.32], np.float32) * rock[mtn_m, None] * 0.35
    # ---- v7.8 新增子类纹理 ----
    # 盆地 (6): 沉积层理 / 内流盐渍 / 前陆碎屑
    if basin_m.any():
        # 沉积盆地: 横向层理 (低频正弦)
        sed = basin_m & ((lf & 0xF0) == 0x10)
        if sed.any():
            layer_1d = np.sin(np.linspace(0, 6 * np.pi, S)) * 0.5 + 0.5
            layer = np.broadcast_to(layer_1d[:, None], (S, S))
            rgb[sed] = rgb[sed] * 0.82 + \
                np.array([0.62, 0.55, 0.40], np.float32) * layer[sed, None] * 0.18
        # 内流盆地: 盐渍斑 (亮白)
        endo = basin_m & ((lf & 0xF0) == 0x40)
        if endo.any():
            salt = np.clip(value_fbm(S, 30, 3, rng=rng) + 0.4, 0, 1)
            rgb[endo] = rgb[endo] * 0.55 + \
                np.array([0.85, 0.80, 0.72], np.float32) * salt[endo, None] * 0.45
        # 构造-前陆盆地: 山前碎屑扇 (偏暗)
        foreland = basin_m & ((lf & 0xF0) == 0x30)
        if foreland.any():
            rgb[foreland] = rgb[foreland] * 0.78 + \
                np.array([0.45, 0.38, 0.30], np.float32) * 0.22
    # 山地高度细分: 高山偏冷 / 低山偏暖
    if mtn_m.any():
        high_m = mtn_m & ((lf & 0x30) == 0x00)
        low_m = mtn_m & ((lf & 0x30) == 0x20)
        if high_m.any():
            rgb[high_m] = rgb[high_m] * 0.85 + \
                np.array([0.50, 0.45, 0.40], np.float32) * 0.15
        if low_m.any():
            soil = np.clip(value_fbm(S, 18, 2, rng=rng) + 0.3, 0, 1)
            rgb[low_m] = rgb[low_m] * 0.7 + \
                np.array([0.50, 0.42, 0.30], np.float32) * soil[low_m, None] * 0.30
    # 山地成因: 褶皱平行脊 / 火山玄武岩
    if mtn_m.any():
        fold_m = mtn_m & ((lf & 0xC0) == 0x40)
        if fold_m.any():
            ridge_1d = np.sin(np.arange(S) * 0.08) * 0.5 + 0.5
            ridge_pat = np.broadcast_to(ridge_1d[:, None], (S, S))
            rgb[fold_m] = rgb[fold_m] * 0.75 + \
                np.array([0.42, 0.36, 0.30], np.float32) * ridge_pat[fold_m, None] * 0.25
        volc_m = mtn_m & ((lf & 0xC0) == 0xC0)
        if volc_m.any():
            rgb[volc_m] = rgb[volc_m] * 0.6 + \
                np.array([0.25, 0.20, 0.18], np.float32) * 0.4
    # 高原切割沟壑 / 侵蚀风化
    if plateau_m.any():
        dissect = plateau_m & ((lf & 0xF0) == 0x20)
        if dissect.any():
            g = gaussian_filter(v01, 1.0); gy2, gx2 = np.gradient(g)
            gmag = np.hypot(gx2, gy2)
            rgb[dissect] = rgb[dissect] * 0.7 + \
                np.array([0.40, 0.34, 0.26], np.float32) * \
                np.clip(gmag[dissect, None] * 3, 0, 1) * 0.3
        eros = plateau_m & ((lf & 0xF0) == 0x30)
        if eros.any():
            rgb[eros] = rgb[eros] * 0.85 + \
                np.array([0.55, 0.48, 0.32], np.float32) * 0.15
    # 丘陵高低差异化
    if hill_m.any():
        hi_h = hill_m & ((lf & 0xF0) == 0x10)
        lo_h = hill_m & ((lf & 0xF0) == 0x20)
        if hi_h.any():
            rgb[hi_h] = rgb[hi_h] * 0.8 + \
                np.array([0.45, 0.40, 0.32], np.float32) * 0.2
        if lo_h.any():
            rgb[lo_h] = rgb[lo_h] * 0.85 + \
                np.array([0.50, 0.55, 0.30], np.float32) * 0.15
    # 湖泊: 填洼面与实际地形之差 → 湖面 (盆地积水)
    filled = fill_depressions(v01)
    lake = land & ((filled - v01) > 0.012)
    rgb[lake] = rgb[lake] * 0.25 + np.array([0.12, 0.30, 0.46], np.float32) * 0.75
    # 河流: 填洼气候加权流累积 → 可见河道 (水量=降水权重×融雪门控)
    # v8.1: 水源按降水权重+融雪门控 (气候加权 acc), 干流宽度直接由水量决定
    # v8.2: paint_rivers_rgb 做分级染色 — 源头细淡(s小, 窄, 半透浅蓝)
    #       → 中段适中 → 河口粗深蓝(s大, 宽, 全覆深蓝); 软边消除像素锯齿
    acc = climate_flow_accumulation(filled, P, T, rng)
    rgb = paint_rivers_rgb(rgb, acc, land, rng, base_q=0.995)
    # 雪线: 纬度雪线(干旱抬升) ∪ 温度场雪线(年均温≤-12°C 常年积雪, 连续过渡)
    # v7.2: dry 先 σ=2.0 高斯再进入雪线抬升公式, 切断 "P条带 → dry条带 → 雪线条带" 的三级拍频链
    dry_raw = np.clip(1.0 - P / 700.0, 0, 1)
    dry = gaussian_filter(dry_raw.astype(np.float32), 2.0)
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
                   tect_res=512, out_png=None, out_npy=None,
                   enable_supervisor: bool = True,
                   apply_suggestions: bool = True):
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
    enable_supervisor : bool
        v8.0 新增：是否在生成末尾调用 ``terrain.supervisor`` 审计五维指标
        （地形/温度/降水/群系/风带雪盖）并写学习日志。默认 True。
    apply_suggestions : bool
        v8.0 新增：是否把低风险反思建议自动写入 ``overrides.json``（默认 True）。
        群系阈值等高风险项永不自动注入。设 False 则仅打印建议。

    Returns
    -------
    ndarray of float32, shape ``(size_px, size_px)``
        最终米制高程（正值陆地 / 负值海洋），范围经钳制到 [−10500, +8900] m。
        同时磁盘已写出 ``world_dem.npy``、``world_map.png``、``world_skeleton.png``。
        若 ``enable_supervisor=True``，还会写 ``terrain/supervisor/log.jsonl`` 与
        ``summary.md``，并在终端打印审计简报。

    Notes
    -----
    本函数全程使用确定性随机数（numpy RandomState + torch.manual_seed 由权重承载），
    同一 seed 多次运行**严格 bitwise 一致**；只有删除板块缓存或修改源代码时才会
    改变地形。
    """
    _file_dir = os.path.dirname(os.path.abspath(__file__))
    _dir = (os.path.dirname(_file_dir)
            if os.path.basename(_file_dir) in ("core", "scripts")
            else _file_dir)
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
            # v7.2: 移除 L498-501 的 "图像边界瓦片强制 win=1.0" —— canvas/wgt 全局加权合成
            #      即使图像外无邻居瓦片，ramp 边缘 taper=0.5 也能通过除以单边权重 wgt 正常重建。
            #      保留汉宁羽化 taper 使得边界瓦片与内部瓦片在 overlap 区的权重和=1.0,
            #      彻底消除 tile 208px 网格线的竖/横硬接缝.
            win = np.outer(ramp, ramp)
            y0p, x0p = by * block, bx * block
            canvas[y0p:y0p + h, x0p:x0p + w] += s[:h, :w] * win[:h, :w]
            wgt[y0p:y0p + h, x0p:x0p + w] += win[:h, :w]
            done += 1
            if done % 50 == 0 or done == total:
                print(f"      瓦片 {done}/{total} | {time.time() - t0:.0f}s", flush=True)

    out = np.clip(canvas / np.maximum(wgt, 1e-8), 0, 1).astype(np.float32)
    # v7.2: 瓦片接缝二次羽化 —— 在 step*block (208px) 整数倍的行/列上做 3 像素 σ=0.6 的横向/纵向条带平滑
    #      (仅对 seam 附近 3px 宽 做高斯混合, 不破坏全局高频细节; 经纬度双重)
    seam_rows = sorted(set([r * step * block for r in range(1, len(rows)) if r * step * block < size_px]))
    seam_cols = sorted(set([c * step * block for c in range(1, len(rows)) if c * step * block < size_px]))
    seam_mask = np.zeros_like(out, dtype=bool)
    for r in seam_rows:
        seam_mask[max(0, r-2):min(size_px, r+3), :] = True
    for c in seam_cols:
        seam_mask[:, max(0, c-2):min(size_px, c+3)] = True
    if seam_mask.any():
        out_s = gaussian_filter(out, 0.6)
        # 仅 seam 区混合 (blend 0.5 平滑 + 0.5 原版)
        out = np.where(seam_mask, 0.5 * out + 0.5 * out_s, out).astype(np.float32)
    print("[4/4] 全局阶段: CDF匹配 → 低频复核 → 去斑 → 水力侵蚀 → 填洼D8河网...")
    out = global_envelope_match(out, bm, bstd, bmin, bmax, glac, block,
                                blend=0.60, seed=seed, cap_m=8200.0)
    out = lowfreq_consistency(out, bm, block, iters=2)
    out = despeckle(out, min_area=32)             # v6.6: 64→32 保住岛弧小岛
    out = droplet_erosion(out, rng=np.random.default_rng(seed + 77))   # V形谷/冲积扇
    out = np.clip(out + 0.016 * value_fbm(size_px, 110.0, 2,
                    rng=np.random.RandomState(seed + 3)) * (out >= 0.5),
                  0, 1).astype(np.float32)        # v6.6 高频细节(提PSD)
    # v8.1: 为 carve_trunk_rivers 计算气候场 (P/T) — 干流走廊按水量加权 + 曲率扰动
    # 注: render_world 末段还会再算一次 (供雪线/生物群系用); 此处仅用于干流走廊雕刻
    elev_mid = v2m(out)
    ci_mid = continentality_index(elev_mid, lats)
    dT_mid, dP_mid = ocean_current_effects(elev_mid, lats)
    T_mid = temperature_field(lats, elev_mid, dT_mid, ci=ci_mid)
    P_mid = precipitation_field(lats, elev_mid, np.random.RandomState(seed + 88),
                                 dP_mid, ci=ci_mid)
    out = carve_trunk_rivers(out, np.random.RandomState(seed + 55),
                             precip=P_mid, temp=T_mid,
                             warp_amp=3.0, warp_freq=10.0)    # v8.1 气候加权 + 曲率扰动干流河谷
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
    # ---- v7.2: 海底 GAN 伪影 (同心等高线波纹 / Tile DC 方块色) 专属平滑
    #       海陆严格分离 → 只对 dem_m < 0 的海洋像素应用, 陆地/海岸线 100% 保留.
    # ---- v7.7: 强化 DC 偏移修正 (blockiness_ratio 1.123→目标<1.03) ----
    # ---- v8.0: 软权重过渡替代 norm_long>0.02 硬阈值, 消除岛屿周边 ~330m DC 阶跃 ----
    sea = dem_m < 0.0
    if sea.any():
        f = dem_m.astype(np.float32)
        # 1) 两级去噪: σ=1.4 高斯 (跨瓦片边界轻度混合, 保边) + 5×5 中值 (清同心等高线+1-2px抖动)
        dem_g = gaussian_filter(f, 1.4)
        dem_mm = median_filter(f, size=5)
        # 深度分层加权: 浅海(0~-500m) 30% 混合 → 深海(<-4000m) 75% 混合 (深海本应平坦)
        w = 0.30 + 0.45 * np.clip((-f - 500.0) / 3500.0, 0.0, 1.0)
        dem_mix = (1.0 - w) * f + w * (0.60 * dem_g + 0.40 * dem_mm)
        # 2) Tile DC 偏移均衡化: σ=32 覆盖~8瓦片 (旧σ=12仅5瓦片不足), 减78%偏移 (旧60%)
        #    v8.0: 软权重 w_dc 在 norm_long∈[0.02, 0.40] 内线性过渡 (旧版硬阈值
        #          在 0.02 处产生 ~329m DC 阶跃, 在岛屿周边形成方块光晕)
        smooth_long = gaussian_filter(np.where(sea, f, 0.0), 32.0)
        norm_long = gaussian_filter(sea.astype(np.float32), 32.0)
        ref_sea = smooth_long / np.maximum(norm_long, 1e-4)
        w_dc = np.clip((norm_long - 0.02) / 0.38, 0.0, 1.0).astype(np.float32)
        ref = w_dc * ref_sea + (1.0 - w_dc) * f
        dem_dc = f - 0.78 * (f - ref)              # 减掉 78% DC 偏移, 留 22% 真实地形
        # 合并: 50% 去波纹 + 50% 去方块偏移 → 最终海底 (均衡两类修复)
        sea_final = 0.50 * dem_mix + 0.50 * dem_dc
        dem_m = np.where(sea, sea_final.astype(dem_m.dtype), dem_m)
    # ---- v7.7: 海底地貌增强 (深海丘陵/洋中脊/海山链/陆隆收窄) ----
    dem_m = enhance_ocean_floor(dem_m, lats, envS=envS,
                                rng=np.random.RandomState(seed + 22))
    out = m2v(dem_m)

    np.save(out_npy, dem_m.astype(np.float32))
    # v7.8: 构造 tect_ctx 供 render_world 地貌子类分类
    tect_T_full = np.stack([ndi_zoom(tect_T[i], up, order=1)[:size_px, :size_px]
                            for i in range(4)])
    tect_ctx = dict(envS=envS, beltS=beltS, cf=cf,
                    tect_T=tect_T_full, margin_up=margin_up, riftS=riftS)
    plt.imsave(out_png, render_world(out, lats, landS, make_terrain_cmap(),
                                     np.random.RandomState(seed), tect_ctx=tect_ctx))
    print(f"\n完成: {out_png} + {out_npy} + world_skeleton.png")
    print(f"  陆地 {(dem_m > 0).mean() * 100:.1f}% | 峰 {dem_m.max():.0f}m | 沟 {dem_m.min():.0f}m")

    # ===== v8.0: 地形监督模块审计 + 反思 (生成→审计→反思→学习闭环) =====
    # 审计五维指标 (地形/温度/降水/群系/风带雪盖) → 对比地球参考 → 写学习日志
    # 低风险建议自动写入 overrides.json 供下次生成注入; 群系阈值等高风险仅打印
    if enable_supervisor:
        try:
            from terrain.supervisor import audit_world, reflect_and_learn
            report = audit_world(dem_m, lats, seed, size_px, tect_ctx=tect_ctx)
            reflect_and_learn(report, apply_suggestions=apply_suggestions)
        except Exception as e:
            print(f"[Supervisor] 审计跳过 (不阻断主流程): {e}")
    return dem_m


def main():
    """``generate_world.py`` 命令行入口。

    解析 ``--seed / --size / --ckpt / --tect-res``，调用
    :func:`generate_world` 生成一幅完整世界地图。
    作为薄封装的 ``terrain/generate_world.py`` 以脚本方式直接运行时，
    会经由此入口触发实际流程。
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=2048, choices=[2048, 4096])
    ap.add_argument("--ckpt", default=None)   # 默认: 脚本目录内 best_terrain_gan_v5.pth
    ap.add_argument("--tect-res", type=int, default=512)
    a = ap.parse_args()
    generate_world(seed=a.seed, size_px=a.size, ckpt_path=a.ckpt, tect_res=a.tect_res)


if __name__ == "__main__":
    main()