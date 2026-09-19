# -*- coding: utf-8 -*-
"""
generate_world.py v9.1 —— 板块演化模拟世界生成器

=============================================================
一级: run_plate_sim v3.0 (450 Myr, 有缓存) → 粗DEM + 4构造场
      海洋直方图标定(年龄分位→地球开阔洋深度) + 陆架剖面化 + 窄海沟
二级: 12通道条件 → 逐瓦片重训版GAN → 低频对齐 → 全局CDF匹配
      → 全球D8河网 → 雪线渲染
v7.0 七大地理维度修复层（M1-M6） (基于10种子七维地理检验报告):
  P0 陆架-陆坡剖面修复 + 陆地直方图重映射(→地球实测分位)
  P1 岛弧链/热点岛链盖章 + 干流河谷雕刻
  P2 板内造山带盖章(打破山地-俯冲带过纯耦合)
v8.3 河流水系系统: 骨架提取+单调修复 / Hack 定律蜿蜒 / 牛轭湖 / 连续蓝度渲染
v9.0 监督反思: generate_world 末尾接入 supervisor 审计 → 反思 → overrides 注入
运行: python -m terrain.core.generate_world --seed 7 --size 2048
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
                           minimum_filter, maximum_filter,
                           median_filter)
from .terrain_engine import (device, m2v, v2m, block_reduce, d8_flow_accumulation,
                            snowline_m, value_fbm, ridged_fbm, sharpen_coastline,
                            directional_band_noise, _lowres_fbm, _lowres_directional_noise,
                            lowfreq_consistency,
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
from .plate_sim import run_plate_sim


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


def render_world(v01, lats, landS, cmap, rng, tect_ctx=None, *, _profile=False):
    """渲染入口；_profile=True 时打开逐段计时打印（开发用，不影响结果）。"""
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
    import time as _time
    _t = [0.0]
    def _mark(label):
        """性能打点：profile 开启时打印标签耗时并记录新时间戳。"""
        if _profile:
            print(f"    [render] {label}: {_time.time()-_t[-1]:.2f}s")
        _t.append(_time.time())

    S = v01.shape[0]
    land = v01 >= 0.5
    elev_m = v2m(v01)
    _t.append(_time.time())
    # v9.x: 预计算 elev_climate_smooth / hf_norm / gate (temperature_field 与
    # precipitation_field 都需要, 原本各自重复 3 个 gaussian_filter, 共 ~2s×2=4s 浪费)
    elev_pos = np.maximum(elev_m, 0.0)
    elev_climate_smooth = gaussian_filter(elev_pos, 2.2)
    hf = elev_pos - gaussian_filter(elev_pos, 0.9)
    hf = gaussian_filter(hf, 0.5)
    hf_norm = np.clip(hf / np.maximum(np.max(np.abs(hf)), 1.0), -1.0, 1.0)
    gate = np.clip((elev_climate_smooth - 800.0) / 3200.0, 0.0, 1.0)
    _mark("elev_climate_precompute")
    # v9.x: 预计算 d_sea_km (continentality_index 与 precipitation_field 共用, 省 ~1s)
    kmpp = 360.0 / max(S, 1) * 111.32
    d_sea_km = distance_transform_edt(~(elev_m < 0.0)) * kmpp
    _mark("d_sea_km_precompute")
    ci = continentality_index(elev_m, lats, _d_sea_km=d_sea_km)  # v9.x: 复用预计算
    _mark("continentality_index")
    dT, dP = ocean_current_effects(elev_m, lats)         # M4 洋流/季风气候效应
    _mark("ocean_current_effects")
    T = temperature_field(lats, elev_m, dT, ci=ci,
                           _elev_climate_smooth=elev_climate_smooth,
                           _hf_norm=hf_norm, _gate=gate)  # v9.x: 复用预计算
    _mark("temperature_field")
    P = precipitation_field(lats, elev_m, rng, dP, ci=ci,
                            _elev_climate_smooth=elev_climate_smooth,
                            _hf_norm=hf_norm, _gate=gate,
                            _d_sea_km=d_sea_km)  # v9.x: 复用预计算
    _mark("precipitation_field")
    # ---- v7.7: T 场高斯平滑 (σ=2.5) + P 预平滑 (σ=1.5), 消除等温/等雨线像素级抖动 ----
    #   P 场在 precipitation_field 末尾已做 [2.2,1.2] 高斯, 再加 σ=1.5 总平滑≈3.0;
    #   T 场 σ=2.5≈50km 对应真实 ecotone 宽度下限, 过大会模糊气候带边界.
    T_s = gaussian_filter(T.astype(np.float32), 2.5)
    P_s = gaussian_filter(P.astype(np.float32), 1.5)
    bi = whittaker_biome(T_s, P_s, lats=lats, ci=ci)   # v7.1: 苔原/泰加林/草原门限大陆度自适应
    _mark("whittaker_biome")
    # ---- v7.7: 群系边界平滑 + 碎片清理 (7×7 中值 + <50px 碎片清理 + 3轮迭代) ----
    #   7px≈137km 对应 ecotone 尺度; 50px≈980km²(塞浦路斯级)以下飞地无气候意义;
    #   3轮收敛消除中值产生的 1-2px 新碎片. 目标: 温带落叶林<300块, 热带季节林<250块.
    bi = smooth_biome_labels(bi, land, median_size=7, min_patch_px=50, iters=3)
    _mark("smooth_biome_labels")
    # 山体阴影 (NW 光照, 仅陆地) → 地形层次感
    g = gaussian_filter(v01, 1.5)
    gy, gx = np.gradient(g)
    hs = np.clip(0.88 - 22.0 * (gx + gy), 0.60, 1.15)
    shade = (0.78 + 0.55 * np.clip((v01 - 0.5) / 0.5, 0, 1)) * (0.72 + 0.28 * hs)
    # v35b: 低地 shade 保底亮度 —— 原公式对低地(v01→0.5)压到 0.76 倍, 叠加暗
    #      色森林群系后大陆中央呈大片阴影区; 卫星图陆地不暗, 低地保底 0.86
    shade = np.where(v01 < 0.64, np.maximum(shade, np.float32(0.86)), shade)
    land_rgb = BIOME_COLORS[bi] * shade[..., None]
    _mark("hillshade+biome_color")
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
                # v9.x: sigma=[s,s,0] 仅空间模糊 (不混色 RGB), 比标量 sigma 快 ~40%
                land_rgb_b = gaussian_filter(land_rgb, sigma=[band * 0.5, band * 0.5, 0])
                land_rgb = ((1.0 - w[..., None]) * land_rgb
                            + w[..., None] * land_rgb_b).astype(np.float32)
    _mark("biome_boundary_blend")
    rgb = cmap(np.clip(v01, 0, 1))[..., :3]
    _mark("cmap_init")
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
    _mark("sea_layer")
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
    _mark("classify_landforms+land_overlay")
    # 沙丘 (4): 叠加纵向条纹纹理
    if dune_m.any():
        stripes = _lowres_directional_noise(S, direction_deg=15.0,
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
    # v33: 斑块 fBm 混入脊状噪声 (ridged) → 山脉呈连续脊线而非"石头疮斑"
    if mtn_m.any():
        rock = 0.55 * _lowres_fbm(S, 40.0, 3, rng=rng) + \
            0.45 * ridged_fbm(S, 18.0, 3, rng=rng)
        rock = np.clip(rock + 0.35, 0, 1)
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
            salt = np.clip(_lowres_fbm(S, 30, 3, rng=rng) + 0.4, 0, 1)
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
            soil = np.clip(_lowres_fbm(S, 18, 2, rng=rng) + 0.3, 0, 1)
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
    _mark("landform_textures")
    # v35b: 陆地区域暗部补光 + 最低明度钳制 (消除"大陆中央阴影/污渍斑块")
    # 诊断: 陆地亮度下尾过长 (p10 0.28-0.36, p25≈0.40, 地球卫星陆地暗部 >0.45);
    # 沙丘条纹/高原等高线/山地岩石等暗色纹理叠加在低地底图上产生大片暗褐斑块.
    # 解法: ① 对明度<0.55 陆地像素补光 (暗部最多+45%); ② 非水陆地明度钳制 ≥0.42
    #      (保持色调比例伸缩), 压缩明暗断裂 → 连续明亮的陆地 (湖/冰/雪线后续覆盖).
    _m_dark_ = rgb.max(-1).astype(np.float32)
    _darkw_ = np.clip((0.55 - _m_dark_) / 0.55, 0.0, 1.0) * land
    if _darkw_.any():
        rgb = np.clip(rgb * (1.0 + 0.45 * _darkw_[..., None]), 0.0, 1.0).astype(np.float32)
    _m_lo_ = rgb.max(-1)
    _kk_ = np.where(land & (_m_lo_ < 0.42), 0.42 / np.maximum(_m_lo_, 1e-3), 1.0)
    if (_kk_ != 1.0).any():
        rgb = np.clip(rgb * _kk_[..., None], 0.0, 1.0).astype(np.float32)
    _mark("land_dark_lift")
    # 湖泊: 填洼面与实际地形之差 → 湖面 (盆地积水)
    filled = fill_depressions(v01)
    lake = land & ((filled - v01) > 0.018)   # v35c: 阈值 0.012→0.018 (≈28m) 减少碎湖
    # v35: 巨型 fill 水面 → 内流盐沼/干湖床 (不涂蓝)
    # 诊断 (seeds 32-36/50-52 实测): fill_depressions 会把大陆中心大片闭合洼地
    # 整体填入水面 (最大单体占陆地 15-52% vs 地球里海级 6%, inland_share 54-74%
    # vs 地球 24%)。真实地球上不存在占陆地 >6% 的内陆湖 —— 巨型闭合低地底部
    # 是干盐沼/冲积平原 (乌尤尼盐沼/塔里木盆地), 而非整片水域。因此:
    #   单体 >6% 陆地 → 渲染为盐沼白(内流干湖床), 不涂蓝;
    #   ≤6% 的 (里海/维多利亚级) 保留真实湖泊渲染.
    if lake.any():
        from scipy.ndimage import label as _ndl_v35, binary_dilation as _bdl_v35
        _lab_v, _n_v = _ndl_v35(lake)
        _sz_v = np.bincount(_lab_v.ravel())
        _sz_v[0] = 0
        _land_n_v = int(land.sum())
        _cut_v = _land_n_v * 0.06
        big_dep = np.zeros_like(lake)
        for _bi in range(1, _n_v + 1):
            if _sz_v[_bi] > _cut_v:
                big_dep |= (_lab_v == _bi)
        if big_dep.any():
            # 盐沼色: 以干盐壳白为主导 (乌尤尼式亮白盐壳), 只叠少量原底色纹理
            # v35b: 原实现 rgb*0.38 把大片区域压暗 → 大陆中央"阴影"; 盐沼应亮
            # v35d: salt = fbm*1.1+0.62, clip 下界 0.2 → fbm 低处出现 (0.16,0.15,0.13)
            #       暗灰蓝"污渍"(诊断色 0.29,0.34,0.33 吻合). 钳制 0.60-0.96,
            #       纹理保持温和起伏但永不发暗 (盐壳总近乎白).
            salt = np.clip(_lowres_fbm(S, 28, 3, rng=rng) * 0.35 + 0.80, 0.60, 0.96)
            salcol = np.array([0.93, 0.88, 0.76], np.float32)
            rgb[big_dep] = (salcol * (0.88 * salt[big_dep, None])
                            + np.maximum(rgb[big_dep], 0.35) * 0.12)
            # 边缘过渡带: 盐沼周缘弱化, 避免生硬分界
            _ring = _bdl_v35(big_dep, iterations=3) & land & ~big_dep
            if _ring.any():
                rgb[_ring] = (rgb[_ring] * 0.80
                              + salcol * (0.45 * salt[_ring, None]) * 0.20)
            # 湖内孤立高地保持原色 (盐沼中的"岛山")
            print(f"      [v35] 巨型内流水域盐沼化: {int(big_dep.sum())}px "
                  f"({big_dep.sum()/max(_land_n_v,1)*100:.1f}% 陆地)")
        lake = lake & ~big_dep
        # v35c: 湖面积预算 — 只渲染合计 ≤5% 陆地 的最大的湖
        # 诊断: 盐沼化后仍剩 11-23% 陆地是涂蓝碎湖 (3000+ 个分量, 20-30m 浅洼),
        # 在渲染图上呈大片深蓝"污渍"(地球湖泊+内海 ~5%). 超过预算的碎湖不涂蓝
        # (显示为低地陆地).
        if lake.any():
            _lab2, _n2 = _ndl_v35(lake)
            _sz2 = np.bincount(_lab2.ravel()); _sz2[0] = 0
            _budget = int(_land_n_v * 0.05)
            _keep2 = np.zeros_like(lake)
            _used = 0
            for _ci in (np.argsort(_sz2[1:])[::-1] + 1):
                _sd = int(_sz2[_ci])
                if _used + _sd <= _budget:
                    _keep2 |= (_lab2 == _ci)
                    _used += _sd
                if _used >= _budget:
                    break
            if int((lake & ~_keep2).sum()) > 0:
                print(f"      [v35c] 碎湖预算截断: 保留 {_used}px "
                      f"({_used/max(_land_n_v,1)*100:.1f}% 陆地), 丢弃 "
                      f"{int((lake & ~_keep2).sum())}px")
            lake = _keep2
    rgb[lake] = rgb[lake] * 0.25 + np.array([0.12, 0.30, 0.46], np.float32) * 0.75
    # 河流: 填洼气候加权流累积 → 可见河道 (水量=降水权重×融雪门控)
    # v8.1: 水源按降水权重+融雪门控 (气候加权 acc), 干流宽度直接由水量决定
    # v8.2: paint_rivers_rgb 做分级染色 — 源头细淡(s小, 窄, 半透浅蓝)
    #       → 中段适中 → 河口粗深蓝(s大, 宽, 全覆深蓝); 软边消除像素锯齿
    acc = climate_flow_accumulation(filled, P, T, rng)
    _mark("fill+climate_flow_accumulation")
    # v33: 0.995 → 0.985 让支流也可见, 消除"无源之水"感 (0.5% 分位稀疏不足以呈现水系骨架)
    rgb = paint_rivers_rgb(rgb, acc, land, rng, base_q=0.985)
    _mark("paint_rivers_rgb")
    # 雪线: 纬度雪线(干旱抬升) ∪ 温度场雪线(年均温≤-12°C 常年积雪, 连续过渡)
    # v7.2: dry 先 σ=2.0 高斯再进入雪线抬升公式, 切断 "P条带 → dry条带 → 雪线条带" 的三级拍频链
    dry_raw = np.clip(1.0 - P / 700.0, 0, 1)
    dry = gaussian_filter(dry_raw.astype(np.float32), 2.0)
    sl01 = m2v(snowline_m(lats)[:, None] + 800.0 * dry)
    a_sl = 0.12 + 0.83 * np.clip((v01 - sl01) / 0.06, 0, 1)
    a_t = np.clip((-6.0 - T) / 8.0, 0, 1)
    alpha = np.maximum(a_sl, a_t) * land
    rs = np.clip((v01 - gaussian_filter(v01, 6.0)) / 0.05 + 0.6, 0, 1)  # 山脊雪骨架
    alpha = alpha * (0.35 + 0.65 * rs)
    # 极地冰盖：>65° 陆地强制全覆盖（模拟南极/格陵兰冰盖，在 rs 之后不被削弱）
    # 65°≈南极圈，地球南极大陆几乎全部在此纬度以南被冰盖覆盖
    # v33: 硬阈值改 64°→66.5° 渐变斜升 (原来 >65° 一刀切 → 与下方植被带呈突兀直边)
    _ice_ramp = np.clip((np.abs(lats)[:, None] - 64.0) / 2.5, 0.0, 1.0)
    polar_ice = (_ice_ramp * land).astype(np.float32)
    alpha = np.maximum(alpha, polar_ice)
    a3 = alpha[..., None]
    rgb = rgb * (1 - a3) + np.array([0.96, 0.97, 0.99])[None, None, :] * a3
    # v34: 冰盖融水水系 (南极/格陵兰型) — 冰盖下暗河 + 边缘融水辫状河.
    # 在冰盖着色之后重绘融水河道 → 白色冰盖上出现深蓝融水线条,
    # 冰盖边缘 (58-78°) 过渡带也可见, 消除"冰盖与河网割裂"感.
    _lat_abs_ = np.abs(lats)[:, None]
    _melt_belt = (_lat_abs_ >= 58.0) & (_lat_abs_ <= 78.0) & land
    if _melt_belt.any():
        _thr_mt = float(np.quantile(acc[land], 0.985))
        melt_riv = _melt_belt & (acc >= _thr_mt)
        if melt_riv.any():
            _w_melt = np.clip(0.30 + 0.70 * alpha[melt_riv], 0.0, 1.0)
            rgb[melt_riv] = (rgb[melt_riv] * (1.0 - 0.55 * _w_melt[:, None])
                             + np.array([0.14, 0.30, 0.52], np.float32)
                             * (0.55 * _w_melt[:, None]))
    # 海冰: 纬度 + 噪声, 平滑边缘 (地球海冰边界约 70-75°)
    edge_sea = 78.0 + 2.5 * fbm1d(S, 320, rng)
    sig = 1.0 / (1.0 + np.exp(-(np.abs(lats)[:, None] - edge_sea[None, :]) / 1.6))
    sea_ice = (sig > 0.5) & (v01 < 0.5)
    floe = gaussian_filter(_lowres_fbm(S, 24, 3, rng=rng), 0.8)
    sea_ice &= (floe > -1.1) | (np.abs(lats)[:, None] > edge_sea[None, :] + 4)
    rgb[sea_ice] = rgb[sea_ice] * 0.30 + np.array([0.88, 0.93, 0.97]) * 0.70
    # v33: 极点末端平滑 → 纯净白冰盘 (88°→90° 渐变过渡), 消除等距圆柱投影
    #      在南北极的硬裁剪边 (底部陆地/海冰直切, 呈"绘图工具裁切"假象)
    _pole_ramp = np.clip((np.abs(lats)[:, None] - 86.5) * 1.5, 0.0, 1.0)
    if (_pole_ramp > 0).any():
        rgb = rgb * (1 - _pole_ramp[..., None]) + \
            np.array([0.97, 0.98, 0.995])[None, None, :] * _pole_ramp[..., None]
    _mark("snowline+sea_ice")
    if _profile:
        print(f"    [render] TOTAL render_world: {_time.time()-_t[1]:.2f}s")
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


def balance_land_latitude(dem_m, lats, target_land_frac=0.29,
                          polar_thresh=60.0, max_polar_frac=0.22,
                          shelf_depth=-400.0):
    """纬度带陆地平衡：限制极地陆地比例，保持自然地形过渡。

    原理：
        1. 统计 |lat| > polar_thresh 的南北极陆地占总陆地比例。
        2. 若超过 max_polar_frac，将最低的极地陆地淹没为浅海 (-150~-50m 大陆架)，
           避免产生海岸到深海的陡崖。
        3. 为补偿陆地面积损失，抬升温带/热带浅海为陆地。新陆地高程 = 原海底高程的
           镜像 (负值取反)，保留地形起伏，避免平坦台地。
        4. 高斯平滑海陆边界过渡。

    适用于任意生成的世界，避免单一极区陆地过多导致气候失真。
    """
    size = dem_m.shape[0]
    lat_grid = np.broadcast_to(lats[:, None], dem_m.shape)
    abs_lat = np.abs(lat_grid)

    # 1) 淹没极地低地 → 浅海大陆架 (自然过渡，无陡崖)
    n_flooded = 0
    for sign in (-1.0, 1.0):
        polar_mask = (sign * lat_grid) > polar_thresh
        polar_land = polar_mask & (dem_m > 0)
        if not polar_land.any():
            continue
        polar_land_count = polar_land.sum()
        current_total_land = (dem_m > 0).sum()
        if polar_land_count / max(current_total_land, 1) <= max_polar_frac:
            continue
        keep_count = int(current_total_land * max_polar_frac)
        if keep_count >= polar_land_count:
            continue
        n_remove = polar_land_count - keep_count
        polar_indices = np.argwhere(polar_land)
        polar_vals = dem_m[polar_indices[:, 0], polar_indices[:, 1]]
        order = np.argsort(polar_vals)
        flood_idx = polar_indices[order[:n_remove]]
        # 淹没为浅海：原高程越高，淹没越浅（保留大陆架坡度）
        orig_elev = dem_m[flood_idx[:, 0], flood_idx[:, 1]]
        dem_m[flood_idx[:, 0], flood_idx[:, 1]] = -50.0 - orig_elev * 0.3
        n_flooded += n_remove

    # 2) 抬升温带/热带浅海造陆 → 镜像高程 (保留地形起伏)
    if n_flooded > 0:
        weight = np.where((abs_lat >= 20) & (abs_lat <= 60), 2.0,
                 np.where(abs_lat < 20, 1.0, 0.3))
        shallow = (dem_m < 0) & (dem_m > shelf_depth)
        shallow_idx = np.argwhere(shallow)
        if len(shallow_idx) > 0:
            w = weight[shallow_idx[:, 0], shallow_idx[:, 1]]
            d = dem_m[shallow_idx[:, 0], shallow_idx[:, 1]]
            score = w * 1000 + d
            order = np.argsort(-score)
            n_raise = min(n_flooded, len(shallow_idx))
            raise_idx = shallow_idx[order[:n_raise]]
            # 镜像+基础抬升: -50m → 225m, -200m → 300m (避免 0-200m 陆地过多)
            # v9.x: 原版 -dem_m 产生 -50→+50 低海拔陆地, 使 0-200m 段超标 19%;
            #       改为 0.5*(-dem_m) + 200 给予 200m 基础高程, 落入 200-1000m 丘陵段
            orig = dem_m[raise_idx[:, 0], raise_idx[:, 1]]
            dem_m[raise_idx[:, 0], raise_idx[:, 1]] = (-orig) * 0.5 + 200.0

    # 3) 平滑海陆边界
    land = dem_m > 0
    dem_smooth = gaussian_filter(np.where(land, dem_m, 0.0), 1.5)
    boundary = gaussian_filter(land.astype(float), 1.5)
    boundary = (boundary > 0.05) & (boundary < 0.95)
    dem_m[boundary] = dem_smooth[boundary] * 0.5 + dem_m[boundary] * 0.5

    final_land = (dem_m > 0).sum()
    north_polar = ((lat_grid > polar_thresh) & (dem_m > 0)).sum()
    south_polar = ((lat_grid < -polar_thresh) & (dem_m > 0)).sum()
    print(f"  [纬度平衡] 陆地 {final_land / dem_m.size * 100:.1f}% | "
          f"均高 {dem_m[land].mean():.0f}m | "
          f"南极 {south_polar / max(final_land, 1) * 100:.1f}% | "
          f"北极 {north_polar / max(final_land, 1) * 100:.1f}%")
    return dem_m


def create_continental_shelf(dem_m, lats, shelf_width_px=8,
                             inner_depth=-50.0, outer_depth=-200.0):
    """生成大陆架：在海岸线周围创造平缓的浅海过渡。

    原理：
        1. 计算每个海洋像素到最近陆地的距离 (像素)。
        2. 距离 < shelf_width_px 的海洋像素，高程按距离线性插值：
           紧贴海岸 → inner_depth (-50m)
           陆架边缘 → outer_depth (-200m)
        3. 仅修改比目标深度更深的像素（不抬升已有的浅海岛屿）。
        4. 保留深海地貌不被修改。

    地球大陆架：平均宽 80km, 深 -130m, 占海洋 8%。
    """
    from scipy.ndimage import distance_transform_edt

    land = dem_m > 0
    sea = ~land
    if not sea.any():
        return dem_m

    # 海洋像素到陆地的距离 (像素)
    dist = distance_transform_edt(sea)

    # 大陆架区域：距离 < shelf_width_px
    shelf_mask = sea & (dist < shelf_width_px)
    if not shelf_mask.any():
        return dem_m

    # 按距离插值深度 (v33b: 线性→二次缓坡: 近岸保持缓深度更久, 外缘渐深,
    # 消除海岸外 2-3 条明显色带"阶梯/等高线"感, 更贴近地球陆架缓坡形态)
    t = np.clip(dist[shelf_mask] / shelf_width_px, 0.0, 1.0)
    target_depth = inner_depth + (outer_depth - inner_depth) * (t * t)

    # 只修改比目标深度更深的像素（保留原有浅海）
    current = dem_m[shelf_mask]
    new_vals = np.where(current < target_depth, target_depth, current)
    dem_m[shelf_mask] = new_vals

    # 平滑陆架边缘过渡
    shelf_edge = sea & (dist >= shelf_width_px - 1) & (dist <= shelf_width_px + 2)
    if shelf_edge.any():
        smooth = gaussian_filter(dem_m, 1.0)
        dem_m[shelf_edge] = 0.6 * dem_m[shelf_edge] + 0.4 * smooth[shelf_edge]

    new_land = (dem_m > 0).sum()
    shallow = ((dem_m > outer_depth) & (dem_m < 0)).sum()
    print(f"  [大陆架] 浅海(-200~0m) {shallow / (~(dem_m>0)).sum() * 100:.1f}% "
          f"(地球~8%) | 陆地 {new_land / dem_m.size * 100:.1f}%")
    return dem_m


def enhance_terrain_texture(dem_m, lats, rng=None):
    """添加多尺度高频地形纹理，使功率谱斜率接近地球的 β≈2.0-2.5。

    地球地形在所有尺度上都有丰富的分形细节（f^-β, β≈2.0-2.5）。
    GAN + 多次后处理平滑会削弱中高频细节，导致 β>2.9（过于平滑）。

    方案：在陆地添加多倍频 fBm 噪声（频率 30/80/200/500），
    幅度随频率递减，模拟真实地形的分形特征。
    海洋仅在浅海添加少量纹理，深海保持平滑。
    """
    if rng is None:
        rng = np.random.RandomState(42)

    from terrain.core.terrain_engine import value_fbm, ridged_fbm
    size = dem_m.shape[0]
    land = dem_m > 0
    sea = ~land

    # 多尺度 fBm 噪声（频率递增，幅度递减）
    # v31: 高频幅度整体降 ~40% (200/120/60/30 → 120/72/36/18) —
    #      修复平均陆地坡度过陡 (seed11 实测 1.14° vs 地球 0.41°),
    #      微尺度粗糙度主要拉高 P99 坡度, 对 PSD β 影响小
    detail = np.zeros_like(dem_m)
    scales = [
        (30.0, 4, 120.0),    # 中尺度: 山脉/盆地细节
        (80.0, 3, 72.0),     # 中小尺度: 丘陵/谷地
        (200.0, 2, 36.0),    # 小尺度: 坡面纹理
        (500.0, 2, 18.0),    # 微尺度: 地表粗糙度
    ]
    for freq, octs, amp in scales:
        detail += amp * (value_fbm(size, freq, octs, rng=rng) - 0.5)

    # 山脊噪声增强山脉纹理
    ridge = ridged_fbm(size, base_freq=60.0, octaves=3, rng=rng)
    mountain_mask = land & (dem_m > 1000)
    detail += 80.0 * (ridge - 0.5) * mountain_mask.astype(float)

    # 陆地添加纹理，浅海少量，深海不加
    shallow_sea = sea & (dem_m > -200)
    deep_sea = sea & ~shallow_sea
    weight = np.where(land, 1.0,
                      np.where(shallow_sea, 0.15, 0.0))
    dem_m = dem_m + detail * weight

    # 保持海陆边界不变：纹理噪声不能让陆地变海洋或海洋变陆地
    new_land = dem_m > 0
    flipped_land = (~new_land) & land       # 原本是陆地但变成海洋
    flipped_sea = new_land & sea            # 原本是海洋但变成陆地
    if flipped_land.any():
        dem_m[flipped_land] = np.maximum(dem_m[flipped_land], 1.0)
    if flipped_sea.any():
        dem_m[flipped_sea] = np.minimum(dem_m[flipped_sea], -1.0)

    # 功率谱检查
    from scipy.fft import fft2, fftfreq
    d = dem_m.copy()
    d[~land] = 0
    f = np.abs(fft2(d - d.mean()))**2
    freqs = np.sqrt(fftfreq(size)[:, None]**2 + fftfreq(size)[None, :]**2)
    bins_f = np.linspace(0.001, 0.5, 40)
    radial, centers = [], []
    for i in range(len(bins_f) - 1):
        mask = (freqs >= bins_f[i]) & (freqs < bins_f[i+1])
        if mask.any() and f[mask].mean() > 0:
            radial.append(f[mask].mean())
            centers.append((bins_f[i] + bins_f[i+1]) / 2)
    x = np.log10(np.array(centers))
    y = np.log10(np.array(radial))
    beta = -np.polyfit(x, y, 1)[0]
    print(f"  [纹理增强] 功率谱 β = {beta:.2f} (地球 2.0-2.5, 过平滑>2.8)")
    return dem_m


@torch.no_grad()
def generate_world(seed=0, size_px=2048, ckpt_path=None,
                   tect_res=512, out_png=None, out_npy=None,
                   enable_supervisor: bool = True,
                   apply_suggestions: bool = True,
                   export_3d: bool = False,
                   export_3d_mode: str = "both",
                   export_3d_decimate: int = 4):
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
    ck = torch.load(ckpt_path, map_location=device, weights_only=True)
    assert int(ck.get("cfg", {}).get("in_ch", 0)) == 12, "请先用 v6.3 重训 (12通道)"
    gen = UNet(in_ch=12, base=32).to(device)
    gen.load_state_dict(ck["gen_ema"]); gen.eval()
    ch_stats = {"mean": np.array(ck["ch_stats_mean"], np.float32),
                "std": np.array(ck["ch_stats_std"], np.float32)}
    block = int(ck.get("cfg", {}).get("block", 8))
    tb, ovb = 256 // block, 6
    step, ovp = tb - ovb, ovb * block

    print(f"[1/4] 板块演化模拟 (T={tect_res}, 450 Myr, 有缓存)...")
    _stage_t0 = time.time()
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
    # v27: 增强扭曲强度 11→22px — 板块 Voronoi 分配产生块状矩形大陆,
    # 需更强扭曲打破直线/长条形边界, 使海岸线更有机自然.
    # 同时增加低频成分 (freq 3.0) 产生大尺度弯曲, 避免仅小尺度锯齿.
    wxT = (periodic_fbm(tect_res, 6.0, 3, rng) * 22.0
           + periodic_fbm(tect_res, 3.0, 2, rng) * 12.0
           + periodic_fbm(tect_res, 20.0, 2, rng) * 5.0)
    wyT = (periodic_fbm(tect_res, 6.0, 3, rng) * 22.0
           + periodic_fbm(tect_res, 3.0, 2, rng) * 12.0
           + periodic_fbm(tect_res, 20.0, 2, rng) * 5.0)
    elev_T = warp_lonlat(elev_T.astype(np.float32), wxT, wyT, 1.0)
    # v38: 构造场弱扭曲 — 全幅 warp (±39px) 把细长山带切碎 (belt>0.35 AR 0.19,
    # envS>0.2 205 碎片 vs 未扭曲 6.9/8 块), 后续 >2000m 山块全成 blob
    # (belt_share 0.02-0.13 vs 地球 0.84). 构造带用 0.25 幅扭曲: 保持弧状弯曲
    # 同时不撕裂长链; 海岸线 (elev_T/coast) 仍全幅扭曲, 差异 ≤~30px 视觉不可见.
    _wk = float(os.environ.get("FWS_TECT_WARP", "0.25"))
    tect_T = np.stack([warp_lonlat(tect_T[i], wxT * _wk, wyT * _wk, 1.0)
                       for i in range(4)])
    crust_T = warp_lonlat(crust_T.astype(np.float32), wxT, wyT, 1.0)
    # v27: 海岸线形态学平滑 — 用圆形结构元对陆掩膜做开运算+闭运算,
    # 消除板块 Voronoi 产生的矩形尖角和长条狭缝, 使大陆轮廓更圆润自然.
    # 半径 4px on 512 grid ≈ 32px on 4096.
    from scipy.ndimage import binary_closing, binary_opening
    _land_T = elev_T > 0
    _cy, _cx = np.ogrid[-4:5, -4:5]
    _selem = (_cy**2 + _cx**2) <= 16
    _land_smooth = binary_closing(binary_opening(_land_T, structure=_selem),
                                  structure=_selem)
    # 将平滑后的陆地/海洋边界反映回 elev_T: 平滑区外的陆地降为浅海,
    # 平滑区内的海洋抬升为低地. 仅改海陆边界, 不碰内部高程.
    _to_ocean = _land_T & ~_land_smooth   # 被平滑掉的陆地 → 浅海
    _to_land = ~_land_T & _land_smooth   # 被平滑填入的海洋 → 低地
    elev_T = np.where(_to_ocean, -30.0, elev_T)
    elev_T = np.where(_to_land, 30.0, elev_T)
    crust_T = np.where(_land_smooth, np.maximum(crust_T, 0.3), crust_T)
    # v3.0 margin_type 同步扭曲+上采样 (被动/主动边缘分化)
    if margin_T is not None:
        margin_T = warp_lonlat(margin_T.astype(np.float32), wxT, wyT, 1.0)
        margin_up = np.clip(ndi_zoom(margin_T, up, order=0)[:size_px, :size_px],
                            0, 2).astype(np.int8)
    else:
        margin_up = None
    # M2 延展山带同步扭曲+上采样 → 骨架山脉纹理沿带走向 (v38: 弱扭曲保链)
    if belt_T is not None:
        belt_T = warp_lonlat(belt_T.astype(np.float32), wxT * _wk, wyT * _wk, 1.0)
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
    # ===== v38: 山链连续化重构 =====
    # 诊断 (diag_belt.py, seed0): plate_sim 输出的 belt 场本身 belt_share 0.94 /
    # mean_AR 6.9 (地球 0.84/6.4), 但最终 DEM >2000m 山块只有 5% 落在带内
    # (belt_share 0.02) —— 原 700·wvar·(ridged_fbm²) 的高频噪声把整条带切成
    # 碎斑 (<0.3 的位置几乎不抬升), 散山 (板内造山/噪声丘) 反而主导山地.
    # 修复: 带内**保底抬升** — 噪声只做相对起伏 (0.45+0.55·ridge), 整条构造带
    # 走廊整体抬升 → 连续长链 >2000m; 带外山丘幅度降低 (反比 m_field) 减少碎山.
    m_field = np.maximum(envS, 0.85 * beltS)
    wvar = np.clip(1.0 + 0.9 * value_fbm(size_px, 5.0, 3, rng=rng), 0.25, 2.0)
    _ridge = (ridged_fbm(size_px, base_freq=14.0, octaves=4, rng=rng) ** 2)
    # v38: 脊状对比度 0.45+0.55·ridge → 0.80+0.20·ridge, 幅度 1600→1100.
    # 地球高山带是平顶高原 (Tibet/Altiplano): 2000-3500m 带平均坡度仅 0.32°,
    # 3500+ 0.50°; 本生成器 1.17°/1.70° (3-4x) —— 高频脊状纹理把山链刻成尖脊.
    # 弱化带内起伏 → 高地带成"宽缓高原+坡缘", 坡度回地球量级 (山链连续性由
    # 末次迁移保证, 此处只需给地形性格).
    elev_m += 1100.0 * m_field * wvar * (0.80 + 0.20 * _ridge)
    elev_m += band * 300.0 * (value_fbm(size_px, 34.0, 3, rng=rng) - 0.35)   # 中观: 岬角/海湾
    elev_m += band * 150.0 * (value_fbm(size_px, 90.0, 2, rng=rng) - 0.3)
    elev_m += band * 85.0 * (value_fbm(size_px, 200.0, 2, rng=rng) - 0.28)   # 细观: 碎屑岸
    elev_m += band * 45.0 * (value_fbm(size_px, 450.0, 1, rng=rng) - 0.25)
    # 山间盆地 (费尔干纳型): 造山带内部低频凹坑
    # v38: -2000→-1200 · 更弱掩码 — 窄山带 (4-6px@512) 被深切削整条切碎 →
    # >2000m 山链断裂; 降到 1200m 让盆底仍高于山链底, 链保持连通
    imb = np.clip(value_fbm(size_px, 10.0, 3, rng=rng) - 1.1, 0, None)
    elev_m = elev_m - 1200.0 * imb * np.clip(envS * 1.6 - 0.2, 0, 1) * (elev_m > 150)
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
    print(f"    [阶段计时] [1/4] 板块演化: {time.time()-_stage_t0:.2f}s"); _stage_t0 = time.time()

    print("[2/4] 构建 12 通道全球粗底图 (含构造场)...")
    ch_raw, bm, bstd, bmin, bmax, lf = world_to_channels(elev_m, lats, tect_T, block, seed)
    print(f"    [阶段计时] [2/4] 粗底图构建: {time.time()-_stage_t0:.2f}s"); _stage_t0 = time.time()
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
    print(f"    [阶段计时] [3/4] GAN 瓦片+接缝: {time.time()-_stage_t0:.2f}s"); _stage_t0 = time.time()
    print("[4/4] 全局阶段: CDF匹配 → 低频复核 → 去斑 → 水力侵蚀 → 填洼D8河网...")
    _g_t0 = time.time()
    out = global_envelope_match(out, bm, bstd, bmin, bmax, glac, block,
                                blend=0.60, seed=seed, cap_m=8200.0)
    print(f"    [4/4 计时] global_envelope_match: {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
    out = lowfreq_consistency(out, bm, block, iters=2)
    print(f"    [4/4 计时] lowfreq_consistency: {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
    out = despeckle(out, min_area=150)            # v33: 32→90; v37: 90→150 (新海陆几何下 n_islands 220 vs 地球 113)
    print(f"    [4/4 计时] despeckle: {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
    out = droplet_erosion(out, rng=np.random.default_rng(seed + 77))   # V形谷/冲积扇
    print(f"    [4/4 计时] droplet_erosion: {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
    out = np.clip(out + 0.016 * value_fbm(size_px, 110.0, 2,
                    rng=np.random.RandomState(seed + 3)) * (out >= 0.5),
                  0, 1).astype(np.float32)        # v6.6 高频细节(提PSD)
    print(f"    [4/4 计时] value_fbm(110.0,2) 高频: {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
    # v8.1: 为 carve_trunk_rivers 计算气候场 (P/T) — 干流走廊按水量加权 + 曲率扰动
    # 注: render_world 末段还会再算一次 (供雪线/生物群系用); 此处仅用于干流走廊雕刻
    elev_mid = v2m(out)
    ci_mid = continentality_index(elev_mid, lats)
    dT_mid, dP_mid = ocean_current_effects(elev_mid, lats)
    T_mid = temperature_field(lats, elev_mid, dT_mid, ci=ci_mid)
    print(f"    [4/4 计时] continentality+temp(ocean_currents): {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
    P_mid = precipitation_field(lats, elev_mid, np.random.RandomState(seed + 88),
                                 dP_mid, ci=ci_mid)
    print(f"    [4/4 计时] precipitation_field(mid): {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
    out = carve_trunk_rivers(out, np.random.RandomState(seed + 55),
                             precip=P_mid, temp=T_mid,
                             warp_amp=3.0, warp_freq=0.015)    # v8.1 气候加权 + 曲率扰动干流河谷
                                                               # v34: warp_freq 10.0→0.015 修复 —
                                                               # meander base_freq 是"每单位路径长度
                                                               # 振荡数", 10.0 会让 500px 河道产生上千
                                                               # 次像素级震荡, 平滑后河道退化为直线
    print(f"    [4/4 计时] carve_trunk_rivers: {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
    # v6.6: 窄带海岸锐化(±0.008v≈±160m) → 不再压掉陆架/抬升陆地直方图
    out = sharpen_coastline(out, band=0.008, expand=0.03)
    # M3 峡湾雕刻: 高纬被动边缘 U 形谷 (斯堪的纳维亚型), 锐化后/陆架终检前
    if margin_up is not None:
        out = m2v(carve_fjords(v2m(out), lats, margin_up,
                               np.random.RandomState(seed + 41)))
    out = m2v(ocean_margin_fix(v2m(out), envS, margin_up))   # v6.6 陆架终检(v3.0边缘分化)
    ice = polar_ice_mask(lats, out)
    print(f"    [4/4 计时] sharpen+carve_fjords+ocean_margin_fix+polar_ice: {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
    filled = fill_depressions(out)                                     # 河网连通
    print(f"    [4/4 计时] fill_depressions: {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
    acc = d8_flow_accumulation(filled, np.random.RandomState(seed + 31))
    thr = max(float(np.quantile(acc, 0.995)), 2.0)
    t = np.clip(np.log(np.maximum(acc, 1.0001)) / np.log(thr), 0, 1)
    landness = np.clip((out - 0.5) / 0.02, 0, 1)
    out = np.clip(out - 0.035 * (t ** 0.8) * landness * (~ice), 0, 1).astype(np.float32)
    print(f"    [4/4 计时] d8_flow_accumulation+apply: {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
    out = land_hist_final(out, lats)              # v6.6 陆地直方图终检(绝对最后,加权)
    print(f"    [4/4 计时] land_hist_final: {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
    # M2 失败裂谷重新下切: GAN/直方图重排会抹掉骨架版谷地 → rift 场
    # (轴剖面: 轴线1→两侧衰减) 直乘深度, 谷形集中 (东非型);
    # 谷底整体低于两侧 + 轴线稀疏火山峰 (乞力马扎罗型)
    if riftS.any():
        valley = 0.055 * riftS                     # ≈1100m 谷底 (东非裂谷谷底-周边 700-1500m)
        out = np.clip(out - valley * (out >= 0.5), 0, 1).astype(np.float32)
        _vf = value_fbm(size_px, 40.0, 2, rng=np.random.RandomState(seed + 9))
        volc = (riftS > 0.75) & (out >= 0.5) & (_vf > 0.55)
        out = np.clip(out + 0.030 * volc, 0, 1).astype(np.float32)
        print(f"    [4/4 计时] rift valley+volc(value_fbm 40.0,2): {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()

    # M5 克拉通大流域盆地已移至 high_elev_compress 之后执行
    # (原位置在 enhance_terrain_texture 之前, ±200m 纹理噪声会摧毁
    #  M5 创建的 ~4m/px 线性倾斜面, 导致 largest_basin_share 仅 0.5%)
    # ---- M3 边缘地貌终章 (直方图终检后, 最后盖章不被重排) ----
    dem_m = v2m(out)
    if margin_up is not None:
        dem_m, n_pen = add_shelf_peninsulas(dem_m, margin_up, cf, rng,
                                            lats=lats)   # 台地半岛+大陆岛
        print(f"      M3 台地露头(半岛/大陆岛): {n_pen}px")
    dem_m = add_coral_reefs(dem_m, lats, np.random.RandomState(seed + 88))  # 珊瑚礁
    dem_m = ocean_hist_final(dem_m, lats)      # M6 深洋直方图终检 (最后, 深带校准)
    print(f"    [4/4 计时] shelf_peninsulas+coral+ocean_hist_final: {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
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
    print(f"    [4/4 计时] sea 伪影平滑 (gauss σ=1.4/32 + median): {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
    # ---- v7.7: 海底地貌增强 (深海丘陵/洋中脊/海山链/陆隆收窄) ----
    dem_m = enhance_ocean_floor(dem_m, lats, envS=envS,
                                rng=np.random.RandomState(seed + 22))
    print(f"    [4/4 计时] enhance_ocean_floor: {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
    # ---- v9.x: 俯冲带海沟加深 → 恢复 hadal 带占比 (地球 0.96%, v11 仅 0.13%) ----
    # sea 伪影平滑把窄海沟填平 (σ=1.4 + median 5×5), 在俯冲核带重新下切
    sea_deep = dem_m < -3000.0
    if sea_deep.any() and envS is not None:
        # 俯冲强度 > 0.4 的深海 → 海沟加深, 幅度随深度和俯冲强度递增
        trench_mask = sea_deep & (envS > 0.4)
        if trench_mask.any():
            env_w = np.clip((envS - 0.4) / 0.5, 0.0, 1.0)
            depth_w = np.clip((-dem_m - 3000.0) / 3000.0, 0.0, 1.0)
            deepening = -2500.0 * env_w * depth_w  # 最深 -2500m (→ -5500~-8500m)
            dem_m = np.where(trench_mask, dem_m + deepening, dem_m)
            dem_m = np.clip(dem_m, -11000.0, 8900.0).astype(np.float32)
            n_trench = int(trench_mask.sum())
            print(f"      [海沟加深] {n_trench}px | 最深 {dem_m.min():.0f}m")
    print(f"    [4/4 计时] trench_deepening: {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
    # ---- 纬度带陆地平衡：限制极地陆地比例，维持全球陆海比 ----
    dem_m = balance_land_latitude(dem_m, lats)
    print(f"    [4/4 计时] balance_land_latitude: {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
    # ---- 大陆架生成：在海岸线周围创造平缓的浅海过渡 ----
    # v32: 收窄大陆架 8→5px, 外缘 -200→-180m. 让大陆边缘架宽更接近地球 80km 平均。
    # 注意: 实测 ocean_-200_0 指标(0.174 vs 地球0.071)对该参数几乎不变 —— 该浅海大环
    # 主导来源是上游过量造陆(岛弧/热点/半岛/珊瑚)被 land_fraction_calibration 淹没成
    # 浅海所致, 属全局造陆量问题, 需单独重构, 非本参数可调。
    dem_m = create_continental_shelf(dem_m, lats, shelf_width_px=5, outer_depth=-180.0)
    print(f"    [4/4 计时] create_continental_shelf: {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
    # ---- v9.x: 海岸带平滑 → 合并 <2000km² 近岸碎岛, 减少流域出口碎片化 ----
    # 地球大流域占比 4.4%, v11 仅 0.5%; 主因是碎岛过多导致出口分散 (51165 vs 37178)
    land_b = dem_m > 0
    if land_b.any():
        from scipy.ndimage import label as _ndi_label
        lab_b, n_b = _ndi_label(land_b)
        if n_b > 1:
            sizes_b = np.bincount(lab_b.ravel())
            sizes_b[0] = 0
            total_land = land_b.sum()
            small_isle_mask = np.zeros_like(land_b)
            for cid in range(1, n_b + 1):
                if 0 < sizes_b[cid] < total_land * 0.001:  # v33: <0.1% 陆地 = 碎岛合并 (原 0.03%)
                    small_isle_mask |= (lab_b == cid)
            if small_isle_mask.any():
                # 碎岛降为浅海 (-30m), 与大陆架融合 → 减少出口
                dem_m[small_isle_mask] = -30.0
                # 海岸带高斯平滑 (仅碎岛周边 3px)
                coast_band = np.zeros_like(land_b)
                coast_band[small_isle_mask] = True
                from scipy.ndimage import binary_dilation
                coast_band = binary_dilation(coast_band, iterations=3)
                if coast_band.any():
                    smooth_dem = gaussian_filter(dem_m.astype(np.float32), 1.2)
                    dem_m = np.where(coast_band, 0.5 * dem_m + 0.5 * smooth_dem, dem_m)
                n_merged = int(small_isle_mask.sum())
                print(f"      [海岸平滑] 合并碎岛 {n_merged}px → 浅海")
    print(f"    [4/4 计时] coast_smooth: {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
    # ---- 地形纹理增强：多尺度高频噪声 → 功率谱 β 接近地球 2.0-2.5 ----
    dem_m = enhance_terrain_texture(dem_m, lats, rng=np.random.RandomState(seed + 101))
    print(f"    [4/4 计时] enhance_terrain_texture: {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
    # M5 克拉通大流域盆地 (移至纹理增强之后, 确保线性倾斜面不被 ±200m 噪声摧毁):
    # 稳定陆内低地 → 单一出口倾斜面 + 河网树 (亚马逊/刚果型) + 内陆海 (里海型)
    # 注: M5 河床 z_riv=z_out+1.8·d_o 对大盆地可达数千米, 须由后续 high_elev_compress 压缩
    dem_b, n_b, b_info = add_craton_basins(dem_m, envS,
                                           np.random.RandomState(seed + 66),
                                           belt=beltS, lats=lats, max_exo=3,
                                           max_endo=1, precip=P_mid,
                                           endo_p_max=800.0,
                                           target_lo=0.03, target_hi=0.08,
                                           dyke_coast_excl=float(os.environ.get("FWS_DYKE_COAST", "40.0")),
                                           dyke_height=float(os.environ.get("FWS_DYKE_H", "250.0")),
                                           basin_max_share=float(os.environ.get("FWS_BASIN_MAX", "0.10")))
    if n_b:
        dem_m = dem_b
        print(f"      M5 大流域盆地: {n_b}个 [{b_info}]")
    print(f"    [4/4 计时] add_craton_basins: {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
    # ---- v33c: 造山带聚合已撤销 ----
    # 实测 (seeds 32-36, 三种方案): 沿 envS/beltS 的 DEM 抬升 (脊状或连续) 均把山地
    #   切成更多孤立块 (n_mtn_ranges 38→66↑), 降低 belt_share_of_mtn 并横切 M5 大流域
    #   (largest_basin_share 0.105→0.006↓), 湖泊反升. 根因: 本生成器的板块边界场
    #   (envS/beltS) 本身就是零散短弧, 抬升只是放大碎片化, 无法造出地球式的少数
    #   巨长山链. 山系连带的真正修复需强化 plate_sim 的俯冲带连续性, 属下一阶段.
    #   保留的改进: 山地脊状渲染纹理 (v33) 让已有山地视觉上更线性.
    print(f"    [4/4 计时] v33 belt_aggregate(removed): 0.00s"); _g_t0 = time.time()
    # ---- v27: 全球陆地比例双向校准到 ~30% (地球 29.16%), 在 M5 之后 ----
    # 陆地过多 → 淹没最低低地; 陆地不足 → 抬升最浅海床. 双向保证目标比例.
    _target_land = 0.30
    _cur_land = float((dem_m > 0).sum()) / dem_m.size
    if _cur_land > _target_land + 0.01:
        _land_px = dem_m[dem_m > 0]
        _flood_frac = 1.0 - (_target_land * dem_m.size / max(_land_px.size, 1))
        if 0 < _flood_frac < 1:
            _thresh = float(np.quantile(_land_px, _flood_frac))
            _flood = (dem_m > 0) & (dem_m <= _thresh)
            dem_m = np.where(_flood, -20.0 - (_thresh - dem_m) * 0.3, dem_m).astype(np.float32)
            print(f"      [陆地校准] {_cur_land*100:.1f}% → {(dem_m>0).mean()*100:.1f}% "
                  f"(淹没阈值 {_thresh:.0f}m, {_flood.sum()}px)")
    elif _cur_land < _target_land - 0.01:
        _ocean_px = dem_m[(dem_m <= 0) & (dem_m > -300)]  # 浅海抬升到 -300m
        if _ocean_px.size > 100:
            _need = (_target_land - _cur_land) * dem_m.size
            _raise_frac = min(1.0, _need / _ocean_px.size)
            _thresh = float(np.quantile(_ocean_px, 1.0 - _raise_frac))
            _raise = (dem_m <= 0) & (dem_m > _thresh) & (dem_m > -300)
            dem_m = np.where(_raise, 10.0 + (dem_m - _thresh) * 0.3, dem_m).astype(np.float32)
            print(f"      [陆地校准] {_cur_land*100:.1f}% → {(dem_m>0).mean()*100:.1f}% "
                  f"(抬升阈值 {_thresh:.0f}m, {_raise.sum()}px)")
    print(f"    [4/4 计时] land_fraction_calibration: {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
    # ---- v9.x: 高海拔线性压缩 (M5 之后, 压缩 M5 河床/dyke 超高值) ----
    land_hi = (dem_m > 1500.0)
    if land_hi.any():
        # 凸函数压缩 (exp 2.0), 起点 1500m, norm 截断到 1.0
        # (M5 河床 z_riv=z_out+1.8·d_o 对大盆地可达 9000m+, norm>1 会使输出>输入)
        hi_vals = dem_m[land_hi]
        norm = np.clip((hi_vals - 1500.0) / 7500.0, 0.0, 1.0)
        out = 1500.0 + (norm ** 2.0) * 4850.0
        dem_m[land_hi] = out.astype(np.float32)
    n_high = int(((dem_m > 3500) & (dem_m > 0)).sum())
    print(f"      [高海拔压缩] >3500m 陆地 {n_high}px | 峰 {dem_m.max():.0f}m | 均高 {dem_m[dem_m>0].mean():.0f}m")
    print(f"    [4/4 计时] high_elev_compress: {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
    # ---- v31: 陆地归一化平滑 (修复坡度过陡 + 平原破碎) ----
    # seed11 实测平均陆地坡度 1.14° (地球 0.41°), 主因是低频直方图拉伸后
    # 平原-高地边界的高频起伏被放大; 0-200m 平原也被切成上千碎块 (最大块仅
    # 占陆地 12%, 地球为连片大平原)。用海洋填 0 + 归一化卷积对陆地做 σ=1.5
    # 平滑: 线性斜坡 (M5 克拉通盆地 4m/px) 对高斯模糊近似不变 → 大流域指标
    # 不受影响; 只消掉亚像素/2px 级陡峭微起伏。平滑后再末次直方图对齐,
    # 海拔分布与峰值被 land_hist_final 重新精确恢复。
    _land_s = dem_m > 0
    if _land_s.any():
        _smooth_v = gaussian_filter(np.where(_land_s, dem_m, 0.0), 1.5)
        _norm_w = gaussian_filter(_land_s.astype(np.float32), 1.5)
        _sm = np.where(_norm_w > 1e-4, _smooth_v / np.maximum(_norm_w, 1e-4), 0.0)
        dem_m = np.where(_land_s, _sm, dem_m).astype(np.float32)
    print(f"    [4/4 计时] land_smooth(v31 σ=1.5): {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
    # ---- v30: 绝对最后的陆地直方图对齐 (修复"台地星球") ----
    # 前序诸步 (balance_land_latitude / create_continental_shelf / coast_smooth /
    # enhance_terrain_texture / add_craton_basins / land_fraction_calibration /
    # high_elev_compress) 会持续雕塑陆地高程: 其中 land_fraction_calibration
    # 在把陆地比例砍回 30% 时会优先淹没最低低地, high_elev_compress 又把 >1500m
    # 压缩到峰值 ~3530m —— 两者叠加导致陆地几乎全部堆在 1000-2000m 台地带,
    # 0-200m 平原~0%、>3500m 高山~0%、峰值仅 ~3100m (地球 28%平原/64%<1000m/高峰~6350m)。
    # 在最末尾按地球实测 cos 纬度加权分位做单调重排: 不改变陆地份额(仍 30%)，
    # 只按排名恢复 平原/台地/山地 的正确比例与真实高峰, 河流与流域的相对顺序保持。
    # 注: 必须先 m2v(dem_m) 重建 out —— high_elev_compress 内的局部 out 是 1D 掩码数组。
    out = land_hist_final(m2v(dem_m), lats)
    dem_m = v2m(out)
    print(f"    [4/4 计时] land_hist_final(v30 末次): {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
    # ---- v38: 终稿山链迁移 (修复"山系不成带") ----
    # 诊断 (diag_belt, seed0): plate_sim 的 belt/envS 场本身 AR 6.9 (≈地球 6.4),
    # 但 >2000m 高山被 GAN 质量块/板内造山/岛弧等"散山"占满 (最大单体 31k px
    # AR 2.4, 平均 4.3km), 带内山块占比仅 16% → belt_share 0.02-0.12 vs 地球 0.84.
    # 历次前置抬升 (700-1600m) 都会被后续 GAN/直方图重排稀释, 无法把带顶到
    # 最高的 10-11% 排名区 (那里的山才是 >2000m 山块统计源).
    # 解法 (v30 同款收尾模式): 在**末次直方图之后**给构造带走廊二次抬升, 再跑
    # 一次 land_hist_final 恢复地球海拔分布 —— 单调重排不改变全球陆地/海陆比例,
    # 只让"最高排名"净迁移到连续山链上 → 高山成带, 散山降为低山丘陵(坡度同步降).
    _beltX = np.maximum(envS, 0.85 * beltS)
    if _beltX.max() > 0.05:
        _ld = float(os.environ.get("FWS_LIFT", "2000.0"))
        _pd = float(os.environ.get("FWS_PEN", "1800.0"))
        _sd = float(os.environ.get("FWS_SIG", "3.0"))
        # 平滑加宽带走廊 (σ≈50px), 弥合带内 <2000m 空隙, 链不断裂
        _beltB = gaussian_filter(_beltX.astype(np.float32), _sd)
        # v38: 先重度平滑山核 (σ=4≈80px) 再整块抬升 — 末次 CDF 重排会把带内
        # 残余纹理放大成坡度; 高原核心须在抬升**前**就是平的 (地球青藏/安第斯
        # 高原核 ~0.3°), 纹理只留在 <2000m 边缘带.
        _sm_belt = gaussian_filter(dem_m.astype(np.float32), 8.0)
        _beltC = (_beltB > 0.30).astype(np.float32)
        dem_m = np.where((dem_m > 0), dem_m * (1.0 - 0.92 * _beltC)
                         + _sm_belt * (0.92 * _beltC), dem_m).astype(np.float32)
        # 阶跃抬升: 山核 rank 均匀 → 平顶高原, 陡峭只在带缘过渡带
        _beltM = np.clip((_beltB - 0.20) / 0.30, 0.0, 1.0).astype(np.float32)
        _lift = (_ld * _beltM
                 + 0.1 * _ld * _beltM * (value_fbm(size_px, 40.0, 3,
                                                   rng=np.random.RandomState(seed + 9999)) - 0.5))
        dem_m = np.where((dem_m > 0), dem_m + _lift, dem_m).astype(np.float32)
        # 非带高地压降: 把占住排名高位但不成带的 GAN 质量块/散山按高度递减压回,
        # 让"最高排名"净迁移到连续山链 (散山 → 低山丘陵, 顶部带与后者平滑交界)
        _pen = _pd * (1.0 - _beltM) * np.clip((dem_m - 1100.0) / 1800.0, 0.0, 1.0)
        dem_m = np.where((dem_m > 0), dem_m - _pen, dem_m).astype(np.float32)
        out = land_hist_final(m2v(dem_m), lats)
        dem_m = v2m(out)
        print(f"      [v38] 山链迁移: lift={_ld:.0f} pen={_pd:.0f} sig={_sd:.1f} "
              f"带 {int((_beltB > 0.2).sum())}px, 压降 {int((_pen > 50).sum())}px",
              flush=True)
    print(f"    [4/4 计时] belt_migrate(v38 末次): {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
    # ---- v38: 山地坡度过陡软化 ----
    # 山链迁移把高山净搬到持续山带, 但带缘/岩壁仍比地球陡 (0.60 vs 0.41).
    # 对陆地局部坡度 >1.2° 的像素按 (sl-1.2)/2.2 ∈[0,1] 权重混合 σ=2 平滑:
    # 只软化亚像素-数像素级陡坎, 不削高原/山链 (>1 度以上的真实山地保留).
    _sm_ = gaussian_filter(dem_m.astype(np.float32), 2.0)
    _gy_, _gx_ = np.gradient(dem_m.astype(np.float64))
    _dy_m_ = 111320.0 * 180.0 / size_px
    _dx_m_ = (111320.0 * 360.0 / size_px) * \
        np.cos(np.deg2rad(lats.astype(np.float64)))
    _sl_ = np.rad2deg(np.arctan(np.hypot(_gx_ / _dx_m_[:, None], _gy_ / _dy_m_)))
    _soft_ = (dem_m > 0) & (_sl_ > 1.0)
    if _soft_.any():
        _w_ = np.clip((_sl_ - 1.0) / 2.0, 0.0, 1.0) * 0.65
        dem_m = np.where(_soft_, dem_m * (1.0 - _w_) + _sm_ * _w_, dem_m) \
            .astype(np.float32)
        # 注: 不再补跑 land_hist_final — 平滑的对称性使海拔分布漂移极小 (<1%),
        # 避免 CDF 重排把坡度重新拉高 (已验证 0.60 不动的元凶)
        print(f"      [v38] 坡度软化: {int(_soft_.sum())}px (sl>1.0°)", flush=True)
    print(f"    [4/4 计时] slope_relax(v38): {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
    # ---- v37: 末次大陆架重建 (修复 land_fraction_calibration 吃掉陆架) ----
    # 早期 create_continental_shelf(1213) 建的 -200~0m 陆架带会被后续 M5 重塑与
    # land_fraction_calibration(抬升浅海补齐陆地) 覆盖/消耗 —— v37 实测 ocean_-200_0
    # 降到 0.004 (地球 0.071), 3D 球体上海岸直落入深海、无陆架. 在末次高程后重建:
    # 仅作用于 dem<0 且 <5px 的海, 与陆地/河网/直方图无冲突.
    dem_m = create_continental_shelf(dem_m, lats, shelf_width_px=6, outer_depth=-180.0)
    print(f"    [4/4 计时] shelf_rebuild(v37 末次): {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
    # ---- v34: M5 倾斜面后重雕刻蜿蜒河道 (打破"平行直线河"纹理) ----
    # 诊断 (seeds 32-36 实测): M5 线性倾斜面使流域内 D8 流线全部平行笔直,
    # 地图上呈现 45° 等间距斜线纹理 (河流蜿蜒度 1.00-1.04, 地球 ~2+);
    # 且早期 carve_trunk_rivers 的蜿蜒河道被 M5 倾斜面覆盖抹除.
    # 解决: 在绝对地形终稿上重跑轻量蜿蜒雕刻 —— 从最终 DEM 提取河网骨架
    # → meander 蜿蜒化 → 沿中心线雕刻, 使最终渲染的河流 (含 M5 盆地内的
    # 干流与支流) 恢复自然蜿蜒形态, 同时保留源头/分水岭结构.
    _v34_t0 = time.time()
    try:
        v01_end = m2v(dem_m)
        v01_end = carve_trunk_rivers(
            v01_end, np.random.RandomState(seed + 777),
            precip=P_mid, temp=T_mid,
            sigma=2.5, q=0.998, max_depth=0.022, width=2.8,
            warp_amp=4.5, warp_freq=0.015,
            q_levels=(0.992, 0.976, 0.955),
            hack_law=True, oxbow=True)
        dem_m = v2m(v01_end)
        print(f"    [4/4 计时] v34_redecorate_rivers(末次蜿蜒雕刻): {time.time()-_v34_t0:.2f}s")
    except Exception as _e:
        print(f"      [v34_redecorate_rivers 跳过] {_e}")
    _g_t0 = time.time()

    np.save(out_npy, dem_m.astype(np.float32))
    # v7.8: 构造 tect_ctx 供 render_world 地貌子类分类
    tect_T_full = np.stack([ndi_zoom(tect_T[i], up, order=1)[:size_px, :size_px]
                            for i in range(4)])
    tect_ctx = dict(envS=envS, beltS=beltS, cf=cf,
                    tect_T=tect_T_full, margin_up=margin_up, riftS=riftS)
    plt.imsave(out_png, render_world(out, lats, landS, make_terrain_cmap(),
                                     np.random.RandomState(seed), tect_ctx=tect_ctx))
    # ---- 3D 导出（可选，默认关闭；产物写在 DEM 同目录）----
    # v36+: 放在 render_world 之后，使 --texture auto 能复用刚生成的 world_map.png
    if export_3d:
        from .export_3d import export_world
        export_world(dem_m, os.path.dirname(os.path.abspath(out_npy)),
                     mode=export_3d_mode, decimate=export_3d_decimate)
    print(f"    [4/4 计时] save npy + render_world: {time.time()-_g_t0:.2f}s")
    print(f"    [阶段计时] [4/4] 全局阶段总计: {time.time()-_stage_t0:.2f}s")
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
    ``python -m terrain.core.generate_world`` 以模块方式直接运行时，
    会经由此入口触发实际流程。
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=4096, choices=[2048, 4096])
    ap.add_argument("--ckpt", default=None)   # 默认: 脚本目录内 best_terrain_gan_v5.pth
    ap.add_argument("--tect-res", type=int, default=512)
    ap.add_argument("--export-3d", action="store_true",
                    help="生成后导出 3D 模型 (OBJ+glTF, 平面+球体)")
    ap.add_argument("--export-3d-mode", choices=["plane", "globe", "both"], default="both")
    ap.add_argument("--export-3d-decimate", type=int, default=4)
    a = ap.parse_args()
    generate_world(seed=a.seed, size_px=a.size, ckpt_path=a.ckpt, tect_res=a.tect_res,
                   export_3d=a.export_3d, export_3d_mode=a.export_3d_mode,
                   export_3d_decimate=a.export_3d_decimate)


if __name__ == "__main__":
    main()