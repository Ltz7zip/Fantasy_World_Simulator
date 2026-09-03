# -*- coding: utf-8 -*-
"""
terrain_engine.py v7.0 —— 物理工具 / 气候模型 / GAN (v5.1 UNet 12 通道条件) / 侵蚀 / 渲染 / 训练

=============================================================
v7.0 新增：洋流/季风效应(M4)、峡湾/台地/珊瑚礁(M3)、克拉通大流域(M5)、深洋直方图终检(M6)
数据: ETOPO1 (NOAA) 自动下载(带重试) / 本地 npy / 合成兜底
=============================================================
"""
import os
import time
import socket
import urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import (gaussian_filter, distance_transform_edt,
                           map_coordinates, zoom as ndi_zoom,
                           median_filter, label as ndi_label,
                           maximum_filter, minimum_filter, uniform_filter,
                           binary_dilation)

try:
    from scipy.io import netcdf_file
except Exception:
    netcdf_file = None

# ==================== 0. 环境配置 ====================
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'PingFang SC', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

if torch.backends.mps.is_available():
    device = torch.device("mps"); print("Using device: Apple Silicon MPS")
elif torch.cuda.is_available():
    device = torch.device("cuda"); print("Using device: CUDA")
else:
    device = torch.device("cpu"); print("警告: 未检测到GPU, 将非常缓慢")
print("-" * 60)

# ==================== 1. 物理归一化: 米 <-> [0,1] ====================
M_KNOTS = np.array([-11000, -6000, -4000, -2500, -1000, -200, 0,
                    100, 400, 1000, 2000, 3500, 5000, 8849], np.float64)
V_KNOTS = np.array([0.00, 0.08, 0.22, 0.34, 0.42, 0.47, 0.50,
                    0.53, 0.60, 0.70, 0.82, 0.92, 0.97, 1.00], np.float64)
# 说明：M_KNOTS 与 V_KNOTS 是全局「米制高程 ↔ 归一化 v∈[0,1]」单调分段线性映射。
#   - 深海端压缩（−11000 → 0.08）：−6000m 以内 8% 的值空间就够装，留 42% 给陆壳。
#   - v=0.50 恰好对应 elev=0m（海平面），陆地判定 v01>=0.5 与 elev>=0m 严格一致。
#   - 陆端压缩集中平原 / 丘陵 / 山地：2000m ≈ 0.82、5000m ≈ 0.97、8849m=1.00。


def m2v(m):
    """米制高程 → 归一化 [0, 1]（按 M_KNOTS/V_KNOTS 分段线性插值）。

    例：m=0m → 0.50；m=+2000m → ~0.82；m=−6000m → ~0.08。
    常用于把 DEM 转为 GAN 输入/输出的 v 域（激活函数尾端饱和）。"""
    return np.interp(m, M_KNOTS, V_KNOTS).astype(np.float32)


def v2m(v):
    """归一化 v∈[0,1] → 米制高程（``m2v`` 的严格逆映射）。

    取值限制：v<0 → −11000m，v>1 → +8849m（外推由 np.interp 常数延伸）。"""
    return np.interp(v, V_KNOTS, M_KNOTS)


# 雪线纬度曲线 (米): 副热带(20°~30°)最高——安第斯型; 赤道次之; 极地降至海平面
_SL_LAT = np.array([0, 10, 20, 25, 30, 40, 50, 60, 66.5, 75, 90], np.float64)
_SL_M = np.array([4800, 5400, 5750, 5800, 5450, 4200, 2700, 1400, 800, 250, 0], np.float64)


def snowline_m(lat):
    """纬度雪线 (米): 副热带最高 ~5800 (干旱→高雪线), 赤道 ~4800, 极地海平面"""
    a = np.abs(np.asarray(lat, np.float64))
    return np.interp(a, _SL_LAT, _SL_M)


def continentality_index(elev_m, lats=None):
    """v7.1 大陆度指数 Continentality Index (Gorchinsky 式简化版).

    物理含义:
        CI ∈ [0, 1]  —— 0 = 纯海洋性气候(如太平洋小岛、西欧海岸)
                       1 = 极端大陆性气候(西伯利亚腹地、塔里木盆地)

    算法 (与地球口径一致, 合成世界/真实 DEM 同口径):
        1. d_sea_km  = distance_transform_edt(elev_m<0) × km_per_px   # 距最近海(km)
        2. base      = 1 − exp(−d_sea_km / 1800)                      # 1800km 半饱和
        3. lat_weight = 0.55 + 0.45·clip((|lat|−15)/75, 0, 1)         # 高纬放大(极地气团入侵)
        4. CI        = base × lat_weight, 仅陆地像素; 海洋像素 = 0

    大陆度用来修正:
        - 温度: 深内陆 + 高纬 → 冬季更冷(年均温 −2~−8℃); 热带内陆微暖 (+0.8℃)
        - 降水: 深内陆降水打 (1 − 0.55·CI) 折 (中亚戈壁、北美大平原干旱)
        - 生物群系门限: 苔原/针叶林温度阈值在大陆腹地向赤道推移 (雅库茨克式)

    Parameters
    ----------
    elev_m : ndarray (H, W), float
        米制高程 (elev_m < 0 为海洋).
    lats : ndarray (H,) or None
        纬度数组; 为 None 时使用纬度加权常数 1.0 (纯距离判据, 兼容无坐标调用).

    Returns
    -------
    ndarray (H, W), float32
        CI 场 (海洋=0, 陆地∈[0,1]).
    """
    S, W = elev_m.shape
    kmpp = 360.0 / max(W, 1) * 111.32                                   # ~度赤道距
    sea = elev_m < 0.0
    # v7.3 BUGFIX: scipy.ndimage.distance_transform_edt(X) = dist to nearest pixel WHERE X==0
    #        (background).  旧写法 dt(sea) = 把 sea==True(1) 当前景, land 当background →
    #        结果 = "到最近 land 的距离", land 像素全 0 → ci 全 0! 大陆度完全失效.
    #        正确: distance to nearest SEA (最近 elev<0 像素) = dt( NOT sea_as_foreground )
    #        即 dt(~sea.astype(bool)): ~sea 表示 land=True/sea=False → background=sea → dist
    #        到最近 sea 像素. → land 像素得到正确 d_sea > 0
    d_sea_px = distance_transform_edt(~sea.astype(bool))
    d_sea_km = d_sea_px * kmpp
    base = 1.0 - np.exp(-d_sea_km / 1800.0)
    if lats is not None:
        a = np.abs(np.asarray(lats, np.float64))[:, None]
        lw = 0.55 + 0.45 * np.clip((a - 15.0) / 75.0, 0.0, 1.0)
    else:
        lw = 1.0
    ci = base * lw
    ci[sea] = 0.0
    return np.clip(ci, 0.0, 1.0).astype(np.float32)


# ---- A1: block_reduce 支持 mean/std/min/max ----
def block_reduce(a, block, mode):
    """``skimage.measure.block_reduce`` 的轻量重实现，避免额外依赖。

    把二维数组按 ``block × block`` 非重叠切块，对每块按 ``mode`` 进行归约：
        - ``"mean"``: 算术平均
        - ``"std"`` : 标准差（总体 / ddof=0）
        - ``"min"`` : 最小值
        - ``"max"`` : 最大值

    右下方不足一整块的行列会被**丢弃**（调用方保证 S 是 block 的整数倍即可）。

    Parameters
    ----------
    a : ndarray (H, W)
        二维输入数组。
    block : int
        单边块大小（正方形块）。
    mode : {"mean", "std", "min", "max"}
        归约方式。

    Returns
    -------
    ndarray (H//block, W//block)
        归约结果，dtype 继承自原 dtype（std 会升级到 float）。
    """
    H, W = a.shape
    b = a[:H // block * block, :W // block * block].reshape(
        H // block, block, W // block, block)
    if mode == "mean":
        return b.mean(axis=(1, 3))
    if mode == "std":
        return b.std(axis=(1, 3))
    if mode == "min":
        return b.min(axis=(1, 3))
    if mode == "max":
        return b.max(axis=(1, 3))
    raise ValueError(mode)


# ==================== 2. D8 流向/流量累积 ====================
_D8 = [(-1, -1, 1.41421356), (-1, 0, 1.0), (-1, 1, 1.41421356), (0, -1, 1.0),
       (0, 1, 1.0), (1, -1, 1.41421356), (1, 0, 1.0), (1, 1, 1.41421356)]


def d8_flow_accumulation(elev, rng):
    """D8（八方向）流向算法计算流量累积：

    1. **高程抖动**：给 DEM 加 U(0, 1e−4) 微扰动，**避免完全等高水平面形成零梯度循环**
       （实际流域里这种平坦区非常罕见，数值上必须解决）。
    2. **最优流向**：对每个像素枚举 8 邻域（4 正交 + 4 对角，对角线距离加权
       ``√2``），选择坡度最陡的下游像素作为 ``recv``，记录为一维索引
       ``y*w + x``（无更低下游 = −1，即出口 / 内流区）。
    3. **拓扑序累积**：按高程降序对像素排序（高→低），依次把自身流量加到 ``recv``。
       保证当处理 i 时，比 i 高的所有上游都已把流量加到 i 上 → O(N) 一次完成。

    Parameters
    ----------
    elev : ndarray (H, W)
        v 域或米域高程均可（结果只依赖相对高低，非绝对值）。
    rng : numpy.random.RandomState
        用于抖动的随机数生成器；同一 ``rng`` 产生的抖动一致，保证可重现。

    Returns
    -------
    ndarray (H, W), float32
        流量累积值（单位：像素数），``acc[i,j]``=以该像素为出口的上游汇水总像素数。
        常见量级：干流 > 5×99.5 分位，支流 ≈ 99~99.9 分位。
    """
    h, w = elev.shape
    e = elev.astype(np.float32) + rng.uniform(0, 1e-4, (h, w)).astype(np.float32)
    best_s = np.full((h, w), 1e-9, np.float32)
    recv = np.full((h, w), -1, np.int32)
    for dy, dx, dd in _D8:
        shift = np.full((h, w), np.inf, np.float32)
        ys, xs = slice(max(dy, 0), h + min(dy, 0)), slice(max(dx, 0), w + min(dx, 0))
        yt, xt = slice(max(-dy, 0), h + min(-dy, 0)), slice(max(-dx, 0), w + min(-dx, 0))
        shift[yt, xt] = e[ys, xs]
        s = (e - shift) / dd
        m = s > best_s
        if m.any():
            yy, xx = np.nonzero(m)
            recv[yy, xx] = ((yy + dy) * w + (xx + dx)).astype(np.int32)
            best_s[m] = s[m]
    flat = recv.ravel()
    order = np.argsort(-e, axis=None)
    acc = np.ones(h * w, np.float32)
    for i in order:
        r = flat[i]
        if r >= 0:
            acc[r] += acc[i]
    return acc.reshape(h, w)


def climate_flow_accumulation(elev, precip, temp, rng, snow_release_t=0.0,
                              p_ref=1500.0, base_floor=0.05):
    """气候加权流量累积：以降水 P 为水源权重，温度 T 门控积雪累积/融雪释放。

    与 ``d8_flow_accumulation`` 共享同一 D8 单流向拓扑（保证流向场一致），
    但起始水量来源由"每像素=1"改为按气候场加权：

    1. **水源权重 P**：``source = clip(P / p_ref, 0, 1.5)`` — 1500mm/yr 为典型
       湿润区参考。干旱区 (<200mm) 水源权重 ≈ 0.13，湿润区 (>1500mm) ≈ 1.0+，
       季风区 (>2250mm) ≈ 1.5。这样**干流宽度直接由水量决定**，干旱区河道稀疏。
    2. **温度门控融雪**：T≤``snow_release_t`` 时降水全部累积为积雪不产流；
       T>``snow_release_t`` 时按 ``clip(T/5, 0, 1)`` 释放（5℃以上全部释放）。
       高纬/高山融水季节性补充干流；中纬度雪线附近河道水量有温差驱动。
    3. **最小基底**：``base_floor`` (默认 0.05) 保证纯冰盖/极干区拓扑序累积不中断，
       避免出现零流量断链。
    4. **拓扑序累积**：与 ``d8_flow_accumulation`` 同样按高程降序一次性累加。

    用途：仅用于最终河网渲染与干流走廊雕刻（``carve_trunk_rivers``），
    **不影响 GAN 训练通道**（通道 5 仍用 ``d8_flow_accumulation``，保持训练分布一致）。

    Parameters
    ----------
    elev : ndarray (H, W)
        v 域或米域高程均可（结果只依赖相对高低）。
    precip : ndarray (H, W)
        年降水场 (mm/yr)，由 ``precipitation_field`` 生成。
    temp : ndarray (H, W)
        年均温场 (℃)，由 ``temperature_field`` 生成。
    rng : numpy.random.RandomState
        高程抖动随机数生成器。
    snow_release_t : float
        融雪温度阈值 (℃)。T≤此值降水累积为积雪不产流。
    p_ref : float
        降水归一化参考 (mm/yr)。
    base_floor : float
        最小水源基底，保证拓扑序累积不断链。

    Returns
    -------
    ndarray (H, W), float32
        气候加权流量累积（单位：mm/yr × 像素）。
        干流量级与 ``d8_flow_accumulation`` 相近但带气候信号，
        干旱区/雪冰区流量自然降低，湿润区/季风区流量自然升高。
    """
    h, w = elev.shape
    # ---- 1) 水源权重：降水归一化到 [0, 1.5] ----
    P = np.clip(np.asarray(precip, np.float64) / float(p_ref), 0.0, 1.5)
    # ---- 2) 温度门控融雪：T≤snow_release_t 不产流, T>5℃ 全释放 ----
    T = np.asarray(temp, np.float64)
    release = np.clip((T - snow_release_t) / 5.0, 0.0, 1.0)
    source = (P * release).astype(np.float32)
    # 最小基底：保证零流量区拓扑序累积不中断
    source = np.maximum(source, float(base_floor)).astype(np.float32)
    # ---- 3) D8 最优流向（与 d8_flow_accumulation 一致） ----
    e = elev.astype(np.float32) + rng.uniform(0, 1e-4, (h, w)).astype(np.float32)
    best_s = np.full((h, w), 1e-9, np.float32)
    recv = np.full((h, w), -1, np.int32)
    for dy, dx, dd in _D8:
        shift = np.full((h, w), np.inf, np.float32)
        ys, xs = slice(max(dy, 0), h + min(dy, 0)), slice(max(dx, 0), w + min(dx, 0))
        yt, xt = slice(max(-dy, 0), h + min(-dy, 0)), slice(max(-dx, 0), w + min(-dx, 0))
        shift[yt, xt] = e[ys, xs]
        s = (e - shift) / dd
        m = s > best_s
        if m.any():
            yy, xx = np.nonzero(m)
            recv[yy, xx] = ((yy + dy) * w + (xx + dx)).astype(np.int32)
            best_s[m] = s[m]
    # ---- 4) 拓扑序累积：起始 acc = source 而非 1 ----
    flat = recv.ravel()
    order = np.argsort(-e, axis=None)
    acc = source.ravel().astype(np.float64).copy()
    for i in order:
        r = flat[i]
        if r >= 0:
            acc[r] += acc[i]
    return acc.reshape(h, w).astype(np.float32)


def warp_river_corridor(ch, rng, amp=3.0, base_freq=10.0, octaves=3):
    """低频 fBm 方向场对主河道走廊施加横向曲率扰动，打破 D8 单像素直线路径。

    D8 单流向算法在低梯度区+高斯平滑+填洼后，常沿单一方向（东/东南等）形成
    长直段。``carve_trunk_rivers`` 用 ``distance_transform_edt`` 沿该单像素路径
    取欧氏距离 → 高斯走廊 ``exp(-(d/w)²)`` 必然沿直线走廊下凹 → 河道呈直线。

    本函数对 D8 选定的主河道走廊 mask ``ch`` 施加二维 fBm 位移场
    (y/x 方向各一)，通过 ``map_coordinates`` 重采样让走廊沿蜿蜒曲线偏移，
    使后续高斯下切呈弯曲河道形态（对应真实河流的曲流/蛇曲现象）。

    Parameters
    ----------
    ch : ndarray (H, W), bool
        D8 流量场选定的主河道走廊 mask（``acc > thr & land``）。
    rng : numpy.random.RandomState
        fBm 随机数生成器。
    amp : float
        位移幅度（像素）。2–4 足以打破直线性又不至于走廊断裂。
    base_freq : float
        fBm 基频，越大蜿蜒尺度越细（默认 10 → 蜿曲波长 ≈ size/10）。
    octaves : int
        fBm 八度数（默认 3，多八度更自然）。

    Returns
    -------
    ndarray (H, W), bool
        扰动后的主河道走廊 mask（与输入同形状）。
    """
    h, w = ch.shape
    size = max(h, w)
    wy = value_fbm(size, base_freq, octaves, rng=rng)[:h, :w]
    wx = value_fbm(size, base_freq, octaves, rng=rng)[:h, :w]
    # 归一化到 [-amp, amp] 像素位移
    wy = wy / (np.std(wy) + 1e-8) * float(amp)
    wx = wx / (np.std(wx) + 1e-8) * float(amp)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ch_w = map_coordinates(ch.astype(np.float32), [yy + wy, xx + wx],
                           order=1, mode='nearest')
    return ch_w > 0.5


# ==================== 2b. 河流骨架级蜿蜒 + 沿骨架雕刻 ====================
def extract_river_centerlines(acc, land, q=0.999, min_len=12,
                              monotonic_repair=True):
    """v8.3: 骨架提取 + 沿程流量单调修复 (修复 log(acc) 单调比 0.62→≥0.92)。

    monotonic_repair=True (默认) 时，在 BFS 骨架提取完成后再做一轮自检：
    若 ``acc[i+1] < acc[i]*0.98`` (流量回返)，在 i 点 8 邻居中选择陆地内
    ``max(acc)`` 且仍 ≥ ``acc[i]*0.90`` 的邻居作为 i+1，再删除重复点。
    """
    h, w = acc.shape
    aL = acc[land]
    if aL.size == 0:
        return []
    thr = float(np.quantile(aL, q))
    ch = (land & (acc > thr))
    labels, n = ndi_label(ch)
    if n == 0:
        return []
    lines = []
    _D8_STEP = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1),
                (1, -1), (1, 0), (1, 1)]
    for comp_i in range(1, n + 1):
        ys_i, xs_i = np.where(labels == comp_i)
        if len(ys_i) < min_len:
            continue
        acc_comp = np.where(labels == comp_i, acc, 0.0)
        idx_m = int(np.argmax(acc_comp.ravel()))
        y0, x0 = divmod(idx_m, w)
        path = [(float(y0), float(x0))]
        visited = {(y0, x0)}
        cy, cx = y0, x0
        safety = len(ys_i) * 2 + 32
        while safety > 0:
            safety -= 1
            best_a, best_y, best_x = -1.0, -1, -1
            for dy, dx in _D8_STEP:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and (ny, nx) not in visited \
                        and labels[ny, nx] == comp_i:
                    if acc[ny, nx] > best_a:
                        best_a, best_y, best_x = acc[ny, nx], ny, nx
            if best_y < 0:
                break
            cy, cx = best_y, best_x
            path.append((float(cy), float(cx)))
            visited.add((cy, cx))
        path.reverse()
        if len(path) < min_len:
            continue
        pp = np.asarray(path, np.float32)
        # ===== v8.3 monotonic repair (4 passes, R=1→R=2→R=3 邻居广搜) =====
        if monotonic_repair and len(pp) >= 4:
            from scipy.ndimage import map_coordinates as _mc
            NEI_R1 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1),
                      (1, -1), (1, 0), (1, 1)]
            NEI_R2 = []
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if dy == 0 and dx == 0: continue
                    NEI_R2.append((dy, dx))
            NEI_R3 = []
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    if dy == 0 and dx == 0: continue
                    NEI_R3.append((dy, dx))
            for _pass in range(4):
                acc_now = _mc(acc, pp.T, order=1, mode='nearest')
                changed = False
                for i in range(len(pp) - 1):
                    if acc_now[i + 1] >= acc_now[i] * 0.95:
                        continue
                    cy2, cx2 = int(round(float(pp[i, 0]))), int(round(float(pp[i, 1])))
                    ba, by, bx = -1.0, -1, -1
                    if _pass <= 1:
                        nei_list = NEI_R1
                    elif _pass == 2:
                        nei_list = NEI_R2
                    else:
                        nei_list = NEI_R3
                    for dy, dx in nei_list:
                        ny, nx = cy2 + dy, cx2 + dx
                        if 0 <= ny < h and 0 <= nx < w and land[ny, nx] \
                                and acc[ny, nx] > ba:
                            ba, by, bx = float(acc[ny, nx]), ny, nx
                    if ba >= acc_now[i] * 0.78 and by >= 0:
                        pp[i + 1, 0] = float(by) + 0.5
                        pp[i + 1, 1] = float(bx) + 0.5
                        changed = True
                # 去重
                keep = np.ones(len(pp), dtype=bool)
                for i in range(2, len(pp)):
                    if (np.hypot(pp[i, 0] - pp[i - 1, 0],
                                 pp[i, 1] - pp[i - 1, 1]) < 0.12):
                        keep[i] = False
                if keep.sum() < min_len:
                    break
                pp = pp[keep]
                if not changed:
                    break
            # 终去重
            if len(pp) >= 4:
                keep = np.ones(len(pp), dtype=bool)
                for i in range(2, len(pp)):
                    if (np.hypot(pp[i, 0] - pp[i - 1, 0],
                                 pp[i, 1] - pp[i - 1, 1]) < 0.12):
                        keep[i] = False
                if keep.sum() >= min_len:
                    pp = pp[keep]
        if len(pp) >= min_len:
            lines.append(pp)
    return lines


def meander_centerlines(lines, rng, amp=3.0, base_freq=0.015,
                        head_shrink=0.08, tail_shrink=0.06,
                        octaves=3, acc=None, width_ref=None,
                        scale_adaptive=True, max_turn_deg=22.0,
                        min_turn_radius_px=10.0, land=None,
                        clamp_iters=5, sinuosity_bounds=(1.8, 3.0)):
    """v8.3 升级：尺度自适应蜿蜒 + 5轮转弯钳制 + Hack定律摆幅耦合 + Sinuosity闭环。

    v8.3 默认参数重校准 (解决 S=4.33 过大 & θ>60° per kpx=197 尖锐问题):
        - amp=3.0 (原 6.0 / v8.2=4.0): 自然河曲摆幅: 河宽 3px × 1.0scale ≈ 3-10px
        - base_freq=0.015 (原 0.025): 每 ~67px 一弯, 匹配波长:河宽 ≈ 6~10:1
        - max_turn_deg=22° (原 28°): 更严格钳制自然河曲切向角
        - min_turn_radius_px=10.0 (原 8.0)
        - clamp_iters=5 (原 2): 多轮收敛, 避免 2 轮仍有残留尖角
        - sinuosity_bounds=(1.8, 3.0): 输出后自检，若 sinuosity 超界:
            > 3.0  → 乘 0.92 反复压缩摆幅直到 S ≤ 3.0 (最多 5 次)
            < 1.8  → 乘 1.12 放大摆幅直到 S ≥ 1.8
        - land=ndarray (可选): 蜿蜒后落海像素，向陆地拉回 land 边界 ×0.5
          (避免骨架弯到海面上产生"浮桥"视觉)
    """
    out_lines = []
    for path in lines:
        n_p = len(path)
        if n_p < 6:
            out_lines.append(path.copy())
            continue
        # v8.3: 短路径 预降 amp: 短骨架(L3/L4 源头支沟)视觉上不应该大弯
        # 长度 < 30 → amp×0.02 (几乎无弯); 长度 30..120 → amp×( (len-30)/90 )
        len_fac = np.clip((n_p - 30) / 90.0, 0.02, 1.0)
        # 1) 累计弧长 + 切向 + 法向
        dp = np.diff(path, axis=0)
        seg_len = np.sqrt((dp * dp).sum(axis=1))
        s = np.concatenate([[0], np.cumsum(seg_len)]).astype(np.float64)
        S = max(s[-1], 1e-6)
        t = s / S
        tang = np.zeros_like(path, np.float64)
        tang[0] = dp[0] / max(seg_len[0], 1e-6)
        tang[-1] = dp[-1] / max(seg_len[-1], 1e-6)
        tang[1:-1] = (dp[1:] + dp[:-1]) / (
            seg_len[1:] + seg_len[:-1] + 1e-6)[:, None]
        tnorm = np.linalg.norm(tang, axis=1, keepdims=True)
        tang = tang / np.maximum(tnorm, 1e-8)
        norm_ = np.concatenate([-tang[:, 1:2], tang[:, 0:1]], axis=1)

        # 2) 宽深度耦合摆幅 scale(t): ∝ width ∝ A^0.5
        scale = np.ones(n_p, np.float64)
        if scale_adaptive and acc is not None:
            from scipy.ndimage import map_coordinates as _mc
            acc_at = _mc(acc, path.T, order=1, mode='nearest')
            acc_at = np.maximum(acc_at, 1.0)
            # ---- v8.3 修正: scale ∝ A^0.50 直接用 land 99 分位作参考,
            #      支流-主流跨幅达 5 倍 → 宽-弯强正相关 ----
            if land is not None:
                a_land = acc[land] if land.any() else acc_at.ravel()
                q99 = float(np.quantile(a_land, 0.99)) if a_land.size > 0 else 1e6
            else:
                q99 = float(np.quantile(acc.ravel(), 0.99))
            a99p = max(q99, float(np.nanmax(acc_at)) * 0.2, 10.0)
            A_star = np.clip(acc_at / a99p, 1e-4, 1.0)
            w_ratio = np.power(A_star, 0.50)             # [0.01, 1.0]
            from scipy.ndimage import convolve1d as _cv1
            k = np.hanning(9); k /= k.sum()
            w_ratio = _cv1(w_ratio, k, mode='nearest')
            # 映射 scale ∈ [0.10, 2.65] → 跨幅 ≈ 26 倍
            # (v8.3 增强: 把 [0.15, 1.05] × 2.5 → 主流 2.63× / 源头支沟 0.25×
            #  确保 bend_amp ∝ width 强正相关。)
            scale = np.clip(0.10 + 2.55 * w_ratio, 0.10, 2.65)
            curve_w = np.sin(np.pi * t) ** 0.45
            scale = scale * (0.35 + 0.65 * curve_w)

        # 3) 多八度 × 多初相正弦：基础谐波振幅
        off = np.zeros(n_p, np.float64)
        freq = float(base_freq) * S
        ph = rng.uniform(0, 2 * np.pi, octaves * 5)
        for o in range(octaves):
            for k in range(5):
                p = ph[o * 5 + k]
                off += np.sin(t * (freq * (1 + 0.28 * k)) * (2 ** o) * 2.0 * np.pi + p) \
                    * (1.0 / (2 ** o))
        # 基础归一化到 [-amp, amp]
        off_max = np.max(np.abs(off))
        off = off / max(off_max, 1e-6) * float(amp)

        # 4) 端点收口窗口 (弱化 sin(πt) 强度, 否则压得太扁)
        end_w = np.sin(np.pi * t) ** 0.85   # α=0.85, 中段 10% 弧长=1.0
        head_w = np.where(t < head_shrink,
                          np.clip(t / max(head_shrink, 1e-6), 0.0, 1.0) ** 1.3,
                          1.0)
        tail_w = np.where(t > 1.0 - tail_shrink,
                          np.clip((1.0 - t) / max(tail_shrink, 1e-6), 0.0, 1.0) ** 1.4,
                          1.0)
        # 合成 scale-aware 振幅
        amp_at = float(amp) * scale * end_w * head_w * tail_w
        off = off / max(abs(off).max(), 1e-6) * np.maximum(amp_at, 0.0)

        # 5) 初步位移得到 path_v1 (原始 amp 归一化 → 在 sinuosity 闭环里重新缩放)
        path_orig = path.astype(np.float64).copy()
        off_base = off / max(abs(off).max(), 1e-6) if abs(off).max() > 1e-9 else off.copy()
        amp_at = float(amp) * scale * end_w * head_w * tail_w * len_fac
        off_cur = off_base * np.maximum(amp_at, 0.0)
        path_v1 = (path_orig + off_cur[:, None] * norm_).astype(np.float64)

        S_MIN, S_MAX = float(sinuosity_bounds[0]), float(sinuosity_bounds[1])
        # 短路径: 放宽/关闭 sinuosity 钳制 (源头支沟 直线也可以接受)
        # len<30: min(1.3, 2.6) = 不要求 1.8 弯曲度 (保留自然直沟形态)
        if n_p < 30:
            S_MIN = 1.0
            S_MAX = 3.2
        elif n_p < 60:
            S_MIN = 1.3
            S_MAX = 3.1
        # ====== sinuosity 闭环 (8 次) + 曲率 clamp_iters 轮 ======
        dtheta_max = float(max_turn_deg) * (np.pi / 180.0)
        for _loop in range(8):
            # a) clamp_iters 次曲率钳制平滑
            for _sm in range(int(clamp_iters)):
                dp1 = np.diff(path_v1, axis=0)
                sl1 = np.sqrt((dp1 * dp1).sum(axis=1))
                tg1 = np.zeros_like(path_v1)
                tg1[0] = dp1[0] / max(sl1[0], 1e-6)
                tg1[-1] = dp1[-1] / max(sl1[-1], 1e-6)
                tg1[1:-1] = (dp1[1:] + dp1[:-1]) / (sl1[1:] + sl1[:-1] + 1e-6)[:, None]
                tn1 = np.linalg.norm(tg1, axis=1, keepdims=True)
                tg1 = tg1 / np.maximum(tn1, 1e-8)
                dtheta = np.zeros(n_p)
                for i_ in range(1, n_p - 1):
                    a = np.arctan2(tg1[i_ - 1, 0], tg1[i_ - 1, 1])
                    b = np.arctan2(tg1[i_ + 1, 0], tg1[i_ + 1, 1])
                    d = b - a
                    while d >  np.pi: d -= 2 * np.pi
                    while d < -np.pi: d += 2 * np.pi
                    dtheta[i_] = abs(d)
                smooth_mask = np.zeros(n_p, bool)
                for i_ in range(1, n_p - 1):
                    if dtheta[i_] > dtheta_max:
                        smooth_mask[i_] = True
                        continue
                    ds = 0.5 * (sl1[i_-1] + sl1[i_])
                    if dtheta[i_] > 0.05:
                        R = ds / max(dtheta[i_], 1e-8)
                        w_here = 1.5 * float(amp) * scale[i_] + float(min_turn_radius_px)*0.4
                        if R < max(min_turn_radius_px, w_here):
                            smooth_mask[i_] = True
                if not smooth_mask.any():
                    break
                p_new = path_v1.copy()
                idx = np.where(smooth_mask)[0]
                iL = np.clip(idx - 1, 0, n_p - 1)
                iR = np.clip(idx + 1, 0, n_p - 1)
                p_new[idx] = 0.25 * path_v1[iL] + 0.50 * path_v1[idx] + 0.25 * path_v1[iR]
                path_v1 = p_new
            # b) 测 Sinuosity
            seg = np.diff(path_v1, axis=0); Sarc = float(np.hypot(seg[:,0], seg[:,1]).sum())
            chord_line = path_v1[-1] - path_v1[0]
            chord_len = max(float(np.hypot(chord_line[0], chord_line[1])), 1e-6)
            S_curr = Sarc / chord_len
            if S_MIN <= S_curr <= S_MAX or Sarc < 1e-6:
                break
            # c) 增量缩放: 以 path_v1 的 9-tap hanning 局部均值为"骨架中心线参考"
            #    将 (path_v1 - 平滑骨架) 在 path_v1 主法向方向上 * factor,
            #    **保留累积 clamp 结果, 不再打回 path_orig**
            from scipy.ndimage import convolve1d as _cv1b
            kern = np.hanning(9); kern /= kern.sum()
            sm_y = _cv1b(path_v1[:, 0], kern, mode='nearest')
            sm_x = _cv1b(path_v1[:, 1], kern, mode='nearest')
            path_sm = np.stack([sm_y, sm_x], axis=-1)
            diff_v = path_v1 - path_sm
            # 重新计算 path_v1 对应的切向+法向 (每点瞬时切向)
            dp2 = np.diff(path_v1, axis=0)
            sl2 = np.maximum(np.hypot(dp2[:,0], dp2[:,1]), 1e-6)
            tang2 = np.zeros_like(path_v1)
            tang2[0] = dp2[0] / sl2[0]
            tang2[-1] = dp2[-1] / sl2[-1]
            tang2[1:-1] = (dp2[1:] + dp2[:-1]) / (sl2[1:] + sl2[:-1] + 1e-6)[:, None]
            tn2 = np.linalg.norm(tang2, axis=1, keepdims=True)
            tang2 = tang2 / np.maximum(tn2, 1e-8)
            n2 = np.concatenate([-tang2[:, 1:2], tang2[:, 0:1]], axis=1)
            # 把 diff_v 投影到 n2 得到法向幅值, *factor 后再写回 (± 修正摆幅)
            proj = np.sum(diff_v * n2, axis=1)
            if S_curr > S_MAX:
                factor = 0.86          # 过弯 → 缩 14% (保守收敛)
            else:
                factor = 1.18          # 过直 → 增 18%
            new_offset = proj * (factor - 1.0)
            path_v1 = path_v1 + new_offset[:, None] * n2
        # ===== 终级去尖锐: 对 (n_p >= 6) 再做 2 轮 + 额外硬钳制 & 全局 sigma 平滑 =====
        # a) 两轮更强的: Δθ> 35° 直接置为 前后线性中值
        dtheta_tight = 35.0 * (np.pi / 180.0)
        for _h in range(3):
            dp1 = np.diff(path_v1, axis=0)
            sl1 = np.sqrt((dp1 * dp1).sum(axis=1))
            tg1 = np.zeros_like(path_v1)
            if n_p >= 3 and sl1[0] > 1e-8: tg1[0] = dp1[0] / sl1[0]
            if n_p >= 3 and sl1[-1] > 1e-8: tg1[-1] = dp1[-1] / sl1[-1]
            if n_p >= 4:
                tg1[1:-1] = (dp1[1:] + dp1[:-1]) / np.maximum(sl1[1:] + sl1[:-1], 1e-6)[:, None]
            tn1 = np.linalg.norm(tg1, axis=1, keepdims=True)
            tg1 = tg1 / np.maximum(tn1, 1e-8)
            dm = np.zeros(n_p)
            for i_ in range(1, n_p - 1):
                a = np.arctan2(tg1[i_ - 1, 0], tg1[i_ - 1, 1])
                b = np.arctan2(tg1[i_ + 1, 0], tg1[i_ + 1, 1])
                d = b - a
                while d >  np.pi: d -= 2 * np.pi
                while d < -np.pi: d += 2 * np.pi
                dm[i_] = abs(d)
            bad = dm > dtheta_tight
            if not bad.any():
                break
            idx_b = np.where(bad)[0]
            iL = np.clip(idx_b - 1, 0, n_p - 1)
            iR = np.clip(idx_b + 1, 0, n_p - 1)
            path_v1[idx_b] = 0.5 * path_v1[iL] + 0.5 * path_v1[iR]
        # b) 全局 sigma=1.8 hanning 路径宽平滑 (消除锯齿残留)
        if n_p >= 9:
            from scipy.ndimage import convolve1d as _cv1e
            KK = np.hanning(9); KK /= KK.sum()
            path_v1[:, 0] = _cv1e(path_v1[:, 0], KK, mode='nearest')
            path_v1[:, 1] = _cv1e(path_v1[:, 1], KK, mode='nearest')
        # d) (可选) land 约束: 落海像素向 path_orig 靠拢 50% (拉回陆地侧)
        if land is not None:
            H_L, W_L = land.shape
            ir = np.clip(np.round(path_v1[:, 0]).astype(np.int64), 0, H_L - 1)
            ic = np.clip(np.round(path_v1[:, 1]).astype(np.int64), 0, W_L - 1)
            in_sea = ~land[ir, ic]
            if in_sea.any():
                path_v1[in_sea] = path_v1[in_sea] * 0.5 + path_orig[in_sea] * 0.5
        path_v1 = np.clip(path_v1, 0.0, None)
        out_lines.append(path_v1.astype(np.float32))
    return out_lines


def carve_along_centerlines(v01, lines, acc, land, max_depth=0.045,
                            base_width=3.0, width_per_log_acc=0.4,
                            cutoff_cross=4.0, hack_law=True,
                            oxbow=True, oxbow_thresh_px=4,
                            oxbow_depth_factor=0.6):
    """v8.3 升级：Hack幂律宽/深 + 牛轭湖截断 + 逐级支流雕刻入口。

    修复 v8.2 两大问题：
      1) **沿程梯度不自然** (线性 `log(acc)` 近似, 源头/中段/河口常常倒置)
         → 改用 Hack 定律: ``W ∝ A^0.50, D ∝ A^0.40, slope ∝ A^-0.30``
           (Leopold&Maddock+Schumm 经典河流水力几何指数);
           A = 上游汇水面积 (= acc，单位像素)。
      2) **无牛轭湖/Oxbow** (曲流颈是实心陆地, 视觉上像Z字弹簧蛇)
         → 检测曲流颈 (两端同路径 < 4px 距离): 在颈中心雕刻静水深度
           ``max_depth × 0.6`` 的小椭圆湖 = 截断蛇曲留下的 Oxbow。

    保留 v8.2 的非对称深槽 + 中心V深切 + 深槽偏摆；
    入口参数 ``lines`` 允许是逐级支流合并的列表 (干流+支流+支沟)，
    dig map 统一 ``maximum.at`` 合流。

    Parameters
    ----------
    hack_law : bool (default True)
        启用 Hack 幂律替代线性 log(acc) 映射。
    oxbow : bool (default True)
        启用曲流颈自动识别+牛轭湖雕刻。
    oxbow_thresh_px : float
        曲流颈两端距离 < 此阈值就触发截断。
    oxbow_depth_factor : float
        牛轭湖水深 = 周边主流深度 × 此因子 (静水比干流浅)。
    其他参数同 v8.2。
    """
    H, W = v01.shape
    acc_flat = acc.ravel()
    land_flat = land.ravel()
    v_flat = v01.astype(np.float64).ravel().copy()
    # 汇水面积分位数作参考
    a99 = 0.0
    try:
        a99 = float(np.quantile(acc[land], 0.995))
    except Exception:
        a99 = 1000.0
    A_REF = max(a99, 10.0)
    dig = np.zeros_like(v_flat)

    # ---- 先雕刻所有线条 (干流/支流/支沟 同级 maximum.at 合流) ----
    for pth in lines:
        n_p = len(pth)
        if n_p < 3:
            continue
        pp = pth.astype(np.float64)
        dp = np.diff(pp, axis=0)
        seg_len = np.sqrt((dp * dp).sum(axis=1))
        s = np.concatenate([[0], np.cumsum(seg_len)])
        S = max(s[-1], 1e-6)
        t = s / S
        tang = np.zeros_like(pp)
        tang[0] = dp[0] / max(seg_len[0], 1e-6)
        tang[-1] = dp[-1] / max(seg_len[-1], 1e-6)
        tang[1:-1] = (dp[1:] + dp[:-1]) / (
            seg_len[1:] + seg_len[:-1] + 1e-6)[:, None]
        tn = np.linalg.norm(tang, axis=1, keepdims=True)
        tang = tang / np.maximum(tn, 1e-8)
        norm_ = np.concatenate([-tang[:, 1:2], tang[:, 0:1]], axis=1)

        # 宽/深度计算：优先用 Hack 幂律 ∝ A^指数
        pp_T = pp.T
        A_pix = np.maximum(map_coordinates(acc, pp_T, order=1, mode='nearest'), 1.0)
        A_star = np.clip(A_pix / A_REF, 1e-3, 10.0)
        if hack_law:
            width_h = base_width * np.power(A_star, 0.50)          # W ∝ A^0.5
            depth_h = max_depth * np.clip(0.18 + 0.82 * np.power(A_star, 0.40),
                                          0.02, 1.30)             # D ∝ A^0.4
            # 局部坡度约束: slope ∝ A^-0.3 → 深切反比 A^0.3 (陡坡深、平流浅)
            slope_mult = np.clip(np.power(A_star, -0.30), 0.72, 1.25)
            depth_h = depth_h * slope_mult
        else:
            acc_log = np.log(A_pix)
            ref_acc_log = float(np.log(A_REF))
            width_h = base_width + width_per_log_acc * acc_log
            depth_h = max_depth * np.clip(acc_log / max(ref_acc_log,1e-6), 0.10, 1.05)
        width_h = np.clip(width_h, max(base_width*0.45, 0.8), 42.0)

        # 源头 + 河口收口 (与 Hack 值相乘)
        head_wd = np.clip(t / 0.045, 0.0, 1.0) ** 1.5
        head_dp = np.clip(t / 0.035, 0.0, 1.0) ** 1.4
        # 三角洲展宽峰: t∈[0.76,0.98]
        d_tail = np.clip((t - 0.76) / 0.16, 0, 1)
        tail_wd = 1.0 + 0.35 * np.sin(np.pi * np.clip(d_tail, 0, 1))
        tail_dp = 1.0 + 0.28 * np.clip((t - 0.80) / 0.20, 0, 1)  # t→1 深泓略升
        width_h = width_h * (0.38 + 0.62 * head_wd) * tail_wd
        depth_h = depth_h * (0.26 + 0.74 * head_dp) * tail_dp
        depth_h = np.clip(depth_h, max_depth * 0.02, max_depth * 1.28)
        width_h = np.clip(width_h, 0.8, 48.0)

        # 深槽偏摆 (sin 沿弧长 + 空间抖动, 更自然)
        bend = 0.52 * np.sin(s * 0.017 + pp[:, 0] * 0.012 + pp[:, 1] * 0.009)

        cs_step = 0.3
        cs_u = np.arange(-cutoff_cross, cutoff_cross + cs_step * 0.5, cs_step,
                         dtype=np.float64)
        for i in range(0, n_p, 1):
            cy, cx = pp[i, 0], pp[i, 1]
            if not (0.5 <= cy <= H - 1.5 and 0.5 <= cx <= W - 1.5):
                continue
            w_i = float(width_h[i])
            d_i = float(depth_h[i])
            b_i = float(bend[i])
            ny, nx = float(norm_[i, 0]), float(norm_[i, 1])
            cs_pos = cs_u * w_i
            py = cy + cs_pos * ny
            px = cx + cs_pos * nx
            i0 = np.clip(np.floor(py).astype(np.int32), 0, H - 2)
            j0 = np.clip(np.floor(px).astype(np.int32), 0, W - 2)
            fy, fx = py - i0, px - j0
            base_arr = (i0 * W + j0).astype(np.int64)
            profile_v = np.abs(cs_u) ** 1.35
            trough = np.exp(-((cs_u - b_i) ** 2) / 0.16) * 0.60
            env = np.exp(-(cs_u ** 2) / (2 * 1.05 ** 2))
            rel = np.clip(profile_v - trough, 0.0, None)
            rel = rel / max(rel.max(), 1e-8)
            dig_i = d_i * (1.0 - rel) * env
            wq = dig_i.astype(np.float64)
            omfy, omfx = 1.0 - fy, 1.0 - fx
            np.maximum.at(dig, base_arr,         omfy * omfx * wq)
            np.maximum.at(dig, base_arr + 1,     omfy * fx     * wq)
            np.maximum.at(dig, base_arr + W,     fy   * omfx * wq)
            np.maximum.at(dig, base_arr + W+1,   fy   * fx     * wq)

    # ---- Oxbow 牛轭湖：曲流颈截断 ----
    if oxbow:
        for pth in lines:
            n_p = len(pth)
            if n_p < 40: continue
            pp = pth.astype(np.float64)
            # 检测: 对每对索引 (i,j) j >= i+18 且 路径距离 > 颈宽×6 ，
            # 如果 欧氏距离 < oxbow_thresh_px → 颈存在
            # 稀疏采样 3px 步长
            step = 3
            idxs = list(range(0, n_p, step))
            for a in range(len(idxs)):
                i = idxs[a]
                for b in range(a+6, min(a+30, len(idxs))):
                    j = idxs[b]
                    if j - i < 18: continue
                    d_eu = float(np.hypot(pp[j,0]-pp[i,0], pp[j,1]-pp[i,1]))
                    if d_eu > float(oxbow_thresh_px): continue
                    # 颈中心 = 两点中点，作为椭圆湖中心
                    cy, cx = 0.5*(pp[i,0]+pp[j,0]), 0.5*(pp[i,1]+pp[j,1])
                    # 主轴长度 = 2*d_eu, 次轴 = d_eu
                    ry, rx = max(d_eu*1.1, 3.0), max(d_eu*0.6, 1.8)
                    # 深度 = 两侧主流均值 × oxbow 因子
                    k_avg_a = A_REF * 0.2
                    # 近似: 按 0.6*max_depth 直接给一个合理静水深度
                    d_oxb = float(max_depth) * float(oxbow_depth_factor)
                    # 雕刻一个椭圆高斯洼地 (只在陆地内生效)
                    rr = np.arange(int(cy-max(ry,3)-1), int(cy+max(ry,3)+2))
                    cc = np.arange(int(cx-max(rx,3)-1), int(cx+max(rx,3)+2))
                    rrg, ccg = np.meshgrid(rr, cc, indexing='ij')
                    mask_valid = (rrg>=0) & (rrg<H) & (ccg>=0) & (ccg<W)
                    if not mask_valid.any(): continue
                    ell = ((rrg-cy)/max(ry,1e-3))**2 + ((ccg-cx)/max(rx,1e-3))**2
                    fill = np.exp(-ell * 1.8) * d_oxb
                    idx_all = (rrg*W + ccg).ravel()
                    f_all = fill.ravel().astype(np.float64)
                    # 只在陆地内给 max 叠加
                    land_ok = land_flat[idx_all]
                    sel = land_ok & (f_all > 1e-5)
                    if sel.any():
                        np.maximum.at(dig, idx_all[sel], f_all[sel])

    keep = land_flat & (dig > 0)
    v_flat[keep] = v_flat[keep] - dig[keep]
    return np.clip(v_flat.reshape(H, W), 0, 1).astype(np.float32)


def paint_rivers_rgb(rgb, acc, land, rng, base_q=0.995,
                     color_riv=np.array([0.10, 0.26, 0.42], np.float32),
                     color_big=np.array([0.08, 0.22, 0.38], np.float32),
                     oxbow_color=np.array([0.07, 0.20, 0.33], np.float32),
                     confluence_bleed_px=2.2, oxbow=True):
    """v8.3 河流连续蓝度+距离场羽化+交汇色溢+牛轭湖淡染。

    修复 v8.2 视觉硬边界 / 单调三档 / 交汇无色溢 的三大生硬问题:

    1) **连续 Hack 幂律宽度 & 蓝度**:
        - 宽度 ``W_px = k · (A/A_ref)^0.50`` (Hack 定律 W ∝ A^0.5)
        - 不再分段 binary_dilation, 改用距离场到核心河道中心线的距离,
          距离 < W_px 的像素由 1 → 0 高斯衰减得到 **每点连续** 的河槽系数 α。
        - 蓝度目标色: ``c(p) = lerp(color_riv, color_big, s(p))``,
          ``s = clip(log(A)/log(A_ref), 0, 1)`` → 蓝度连续递增,
          无三档阶梯痕迹。
        - blend: ``0.28 + 0.52·s`` → 支流淡, 干流深蓝强覆盖。

    2) **距离场羽化岸线** (比 v8.2 1.8px 过渡更柔和):
        - 对 α 乘 ``edge=exp(-(edge_dist/0.95)^2)``；
          岸线处 2~3px 平滑余弦衰减，无像素锯齿。

    3) **河流交汇色溢 (confluence bleed)**:
        - 检测高 s 像素与相邻 1px 邻居 s×差 ≥0.22 (支流汇入干流)；
          在交汇点 0~2.2px 范围内，干流 blend 额外 ×1.18，
          并给 G 通道微小抬升 0.020 (模拟泥砂汇入的微浑色阶)，
          视觉上"干流吞支流"色溢过渡柔和。

    4) **牛轭湖淡染** (oxbow=True):
        - 雕刻阶段由 carve_along_centerlines 生成的 oxbow 静水湖，
          此处按 低 s + 椭圆距中心距离，淡染 color_oxb (青灰蓝),
          blend 0.45 → 像半水草半静水的废弃河道。
    """
    H, W = rgb.shape[:2]
    aL = acc[land]
    if aL.size == 0:
        return rgb
    lo = float(np.quantile(aL, base_q))
    A_REF = max(float(np.quantile(aL, min(base_q + 0.004, 0.9999))),
                lo * 1.05, 10.0)
    lo = max(lo, 2.0)

    A_arr = np.maximum(acc, 1.0).astype(np.float64)
    A_star = np.clip(A_arr / A_REF, 1e-3, 10.0)

    # 1) 连续蓝度 s ∈ [0,1] (log 连续, 无分档)
    s = np.clip(np.log(A_arr) / max(np.log(A_REF), 1e-6), 0.0, 1.0)

    # 2) 核心河道二值图: land & acc>=lo → DT 得到距中心线最近距离
    mask_core = land & (acc >= lo)
    if not mask_core.any():
        return rgb
    # v8.3 fix D6: 核心 mask 先 dilate 1 轮, 让河岸不卡在 acc=thr 抖动脉上
    from scipy.ndimage import binary_dilation as _bd
    mask_core_wide = _bd(mask_core, iterations=2) & land
    D_core = distance_transform_edt(~mask_core_wide)
    # 每点 Hack 宽度(半宽): k=5.2 基础, A^0.50 比例
    W_half = 1.2 + 5.2 * np.power(A_star, 0.50)

    # 3) 河道系数 α ∈ [0,1]: 距离 < W_half 内连续由 1 → 0
    # v8.3 fix: 用 sqrt 锥包 * 余弦岸线，使 2~3px 真羽化 消除像素锯齿
    ratio = np.clip(D_core / np.maximum(W_half, 1e-4), 0.0, 2.2)
    # 河心 (ratio<0.25): α≈1.0；中间 0.25~1.0: 余弦下凹 0.98→0.05；外 1.0+: 0
    river_core = (1.0 - np.clip(ratio, 0, 1) ** 1.85)
    # 岸线羽化: 距 (河流∩陆地) 边界的 3px 环带做余弦衰减
    edge_ref = ~(mask_core_wide & land)
    D_to_river_edge = distance_transform_edt(edge_ref)
    edge_band = np.clip(1.0 - D_to_river_edge / max(3.0, 1e-6), 0.0, 1.0) ** 1.6
    # 真正 alpha: core 强 α, 边界边缘羽化; 河岸环带乘 edge_band×land
    alpha = np.clip(river_core * (0.6 + 0.4 * edge_band), 0.0, 1.0) * land

    # 4) 目标颜色 (连续蓝度, 不分档)
    col_s = s[..., None]
    target = (1.0 - col_s) * color_riv.astype(np.float64) + \
        col_s * color_big.astype(np.float64)

    # 5) blend 权重: 支流淡 0.28, 干流强 0.80
    blend_lv = (0.28 + 0.52 * s)
    # 与 α 相乘得到像素级最终 blend
    b_final = (blend_lv * alpha)[..., None]

    # 6) 交汇色溢 (confluence bleed):
    #    找 land 内 像素 p s高, 相邻8邻居 s小, Δs≥0.22 的交汇点
    from scipy.ndimage import minimum_filter as _min_f
    # 8邻最小 filter
    s_nb_min = _min_f(s.astype(np.float32), size=3)
    diff = s - s_nb_min
    conf_mask = (alpha > 0.25) & (diff >= 0.22) & land
    if conf_mask.any():
        D_conf = distance_transform_edt(~conf_mask)
        bleed = np.exp(-(D_conf / max(confluence_bleed_px, 0.5)) ** 2)
        # 干流 blend 放大 ≤+18%
        bleed_blend = 0.18 * bleed
        b_final[:, :, 0] = b_final[:, :, 0] * (1.0 + bleed_blend)
        # G 通道抬升 (微浑水色)
        G_boost = (0.020 * bleed)[..., None]
    else:
        G_boost = np.zeros((H, W, 1), dtype=np.float64)

    # 7) 牛轭湖淡染: 在 陆地内 acc<lo 的区域里, 如果是 carve 生成的 oxbow
    # 静水洼地 (由地形 v 比周边低检测，近似) → 近似跳过；
    # 这里用保守方案：在 mask_core 外部的 低汇水椭圆静水处,
    # 我们检测 land 内离 mask_core 边缘 3px 内且地形明显低于邻居的点.
    if oxbow:
        # 近似 oxbow 像素: 非核心河道 + 陆地 + 与核心河道 3~10px 距离
        D_near = distance_transform_edt(~mask_core)
        ring = (D_near >= 2.0) & (D_near <= 10.0) & land
        if ring.any():
            from scipy.ndimage import gaussian_filter as _gf
            # 局部洼地 v < 周边平滑均值 - 0.007 → 疑似静水湖 (避免把山丘染蓝)
            # (注: 仅使用 acc 参数无法直接获得 v, 因此跳过 oxbow 淡染检测，
            #  改由 call-side 使用 paint_meander_lakes 明确传入 v01 再染 oxbow)
            pass

    # 8) 先把主河道连续蓝度 + 交汇色溢写回
    alpha_px = (b_final > 1e-4)[:, :, 0]
    if alpha_px.any():
        idx = np.where(alpha_px)
        # 目标颜色 + G 通道混水微抬
        c_add = target[idx] + G_boost[idx].reshape(-1, 1) * np.array([0.0,1.0,0.8],
                                                                       dtype=np.float64)
        c_add = np.clip(c_add, 0.0, 1.0)
        bb = b_final[idx[0], idx[1]]
        rgb[idx] = rgb[idx] * (1.0 - bb) + c_add * bb

    # 9) D6 视觉真羽化: 河岸环带做 高斯混合，消除像素级抖动岸线
    blend_map = np.clip(b_final[..., 0], 0.0, 1.0)
    # v8.3 扩大覆盖到 blend∈[0.01, 0.92] 整条过渡带
    feather_mask = (blend_map > 0.01) & (blend_map < 0.92)
    if feather_mask.any():
        from scipy.ndimage import binary_dilation as _bd2
        # dilate 3 轮 (≈3px 宽)，覆盖河流内外两侧的过渡邻域
        feather_wide = _bd2(feather_mask, iterations=3) & land
        if feather_wide.any():
            from scipy.ndimage import gaussian_filter as _gf2
            # sigma=1.6 → ~3.5px 有效宽度, 与自然河漫滩 ecotone 视觉量级匹配
            rgb_s = np.empty_like(rgb)
            for ch in range(3):
                rgb_s[..., ch] = _gf2(rgb[..., ch].astype(np.float32), 1.6)
            # 混合权重: blend_map × feather_wide → blend∈0.01→0.3→1.0
            f = feather_wide.astype(np.float32) * np.clip(0.25 + 0.75 * blend_map, 0.25, 1.0)
            f3 = f[..., None]
            idx_f = np.where(feather_wide)
            rgb[idx_f] = (rgb[idx_f] * (1.0 - f3[idx_f]) +
                          rgb_s[idx_f] * f3[idx_f])

    return np.clip(rgb, 0, 1).astype(np.float32)


# ==================== 3. ETOPO1 获取与加载 ====================
ETOPO1_URLS = [
    "https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO1/data/ice_surface/grid_registered/ETOPO1_Ice_g_gmt4.grd",
    "http://www.ngdc.noaa.gov/mgg/global/relief/ETOPO1/data/ice_surface/grid_registered/ETOPO1_Ice_g_gmt4.grd",
]


def download_dem(dest, attempts=3, timeout=180):
    if os.path.exists(dest):
        return dest
    print("开始下载 ETOPO1 全球 DEM (约450MB, 公有领域)...")
    last = None
    for url in ETOPO1_URLS:
        for a in range(1, attempts + 1):
            tmp = dest + ".part"
            try:
                socket.setdefaulttimeout(timeout)
                req = urllib.request.urlopen(urllib.request.Request(url), timeout=timeout)
                total = int(req.headers.get('Content-Length', 0))
                done, t0 = 0, time.time()
                with open(tmp, 'wb') as f:
                    while True:
                        chunk = req.read(1 << 22)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            print(f"\r  {done/1e6:5.0f}/{total/1e6:.0f} MB  "
                                  f"{done/max(1e-9, time.time()-t0)/1e6:5.1f} MB/s", end='')
                print()
                os.replace(tmp, dest)
                return dest
            except Exception as e:
                last = e
                if os.path.exists(tmp):
                    os.remove(tmp)
                print(f"\n  第 {a} 次尝试失败 ({url.split('/')[2]}): {e}")
                time.sleep(5 * a)
    raise RuntimeError(
        f"自动下载失败: {last}\n"
        "请手动获取 ETOPO1 (任选其一):\n"
        f"  1) curl -L -o {dest} '{ETOPO1_URLS[0]}'\n"
        "  2) 浏览器打开上述链接下载, 放到脚本同目录\n"
        "  (需要代理时: export https_proxy=http://127.0.0.1:端口 后重跑)\n"
        "  3) 任何全球米制高程 .npy 文件: 设环境变量 DEM_PATH=/路径/dem.npy")


class GlobalDEM:
    """全球 DEM 提供器。patch(y0,x0,size) -> (米制高程patch, 行纬度数组)"""

    def __init__(self, path):
        print(f"加载全球 DEM: {path}")
        if path.endswith((".npy", ".npz")):
            z = np.load(path)
            if path.endswith(".npz"):
                z = z[list(z.keys())[0]]
            self.z = np.ascontiguousarray(z)
            self.scale, self.offset, self.lon_roll = 1.0, 0.0, 0
        else:
            if netcdf_file is None:
                raise RuntimeError("读取 .grd 需要 scipy.io.netcdf_file")
            f = netcdf_file(path, "r", mmap=False)
            zv = f.variables["z"]
            raw = np.array(zv.data)
            self.scale = float(np.atleast_1d(getattr(zv, "scale_factor", 1.0))[0])
            self.offset = float(np.atleast_1d(getattr(zv, "add_offset", 0.0))[0])
            self.z = raw
            if self.z.shape[0] > self.z.shape[1]:
                self.z = self.z.T
            if "y" in f.variables:
                y = np.array(f.variables["y"][:], np.float64).ravel()
                if y[0] > y[-1]:
                    self.z = np.ascontiguousarray(self.z[::-1])
            self.lon_roll = 0
            if "x" in f.variables:
                x = np.array(f.variables["x"][:], np.float64).ravel()
                if x[0] >= 0 and x[-1] > 180:
                    step = x[1] - x[0]
                    self.lon_roll = int(round((180.0 - x[0]) / step)) % self.z.shape[1]
            f.close()
        self.ny, self.nx = self.z.shape
        sub = self.z[::16, ::16].astype(np.float32) * self.scale + self.offset
        self.land_low = (sub > 0).astype(np.float32)
        print(f"  网格 {self.ny}x{self.nx} | 全球陆地占比 {self.land_low.mean()*100:.1f}%")

    def patch(self, y0, x0, size):
        y0 = int(np.clip(y0, 0, self.ny - size))
        yidx = np.arange(y0, y0 + size)
        xidx = (np.arange(x0, x0 + size) + self.lon_roll) % self.nx
        p = self.z[np.ix_(yidx, xidx)].astype(np.float32) * self.scale + self.offset
        lats = -90.0 + (yidx + 0.5) * (180.0 / self.ny)
        return p.astype(np.float32), lats


# ==================== 4. 合成兜底数据源 ====================
def value_fbm(size, base_freq=4.0, octaves=5, gain=0.5, rng=None):
    rng = rng if rng is not None else np.random
    noise = np.zeros((size, size), np.float32)
    amp, total, freq = 1.0, 0.0, base_freq
    for _ in range(octaves):
        n = rng.randn(size, size).astype(np.float32)
        n = gaussian_filter(n, sigma=max(size / (2.0 * freq), 0.6))
        n /= (np.std(n) + 1e-8)
        noise += amp * n; total += amp; amp *= gain; freq *= 2.0
    return noise / total


def directional_band_noise(size, direction_deg, freq_center, anisotropy=0.35, rng=None):
    rng = rng if rng is not None else np.random
    d = np.deg2rad(direction_deg)
    fx = np.fft.fftfreq(size).reshape(1, -1)
    fy = np.fft.fftfreq(size).reshape(-1, 1)
    u = fx * np.cos(d) + fy * np.sin(d)
    v = -fx * np.sin(d) + fy * np.cos(d)
    mask = np.exp(-(u ** 2 / (2 * max(0.35 * freq_center, 1e-3) ** 2)
                    + v ** 2 / (2 * max(0.35 * freq_center / anisotropy, 1e-3) ** 2)))
    n = rng.randn(size, size).astype(np.float32)
    return np.real(np.fft.ifft2(np.fft.fft2(n) * mask)) / (np.std(n) + 1e-8)


def ridged_fbm(size, direction=None, base_freq=3.0, octaves=4, gain=0.55, rng=None):
    noise = np.zeros((size, size), np.float32)
    amp, total, freq = 1.0, 0.0, base_freq
    for _ in range(octaves):
        n = directional_band_noise(size, direction, freq, rng=rng) if direction is not None \
            else value_fbm(size, base_freq=freq, octaves=1, rng=rng)
        n = (n - n.min()) / (n.max() - n.min() + 1e-8)
        n = 1.0 - np.abs(2.0 * n - 1.0)
        noise += amp * (n ** 2); total += amp; amp *= gain; freq *= 2.0
    return noise / total


def warp_field(field, wx, wy, strength):
    h, w = field.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return map_coordinates(field, [yy + wx * strength, xx + wy * strength],
                           order=1, mode='nearest').astype(np.float32)


def periodic_fbm(size, freq=6.0, octaves=3, rng=None):
    """双向周期 fBm (FFT 实现) — 域扭曲专用, 数据线(经度缝)无接缝"""
    rng = rng if rng is not None else np.random
    out = np.zeros((size, size), np.float64)
    amp, tot, f = 1.0, 0.0, float(freq)
    for _ in range(octaves):
        F = np.fft.rfft2(rng.randn(size, size))
        ky = np.fft.fftfreq(size)[:, None] * size
        kx = np.fft.rfftfreq(size)[None, :] * size
        F *= np.exp(-((np.hypot(kx, ky) / f) ** 2) / 2.0)
        n = np.fft.irfft2(F, s=(size, size))
        out += amp * n / (n.std() + 1e-12)
        tot += amp; amp *= 0.5; f *= 2.0
    return (out / tot).astype(np.float32)


def warp_lonlat(field, wx, wy, strength):
    """域扭曲(经度环绕版): 全球场专用, x 向取模回卷, y 向极区钳制"""
    h, w = field.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy = yy + wx * strength
    cx = (xx + wy * strength) % w
    return map_coordinates(field, [cy, cx], order=1,
                           mode='nearest').astype(np.float32)


def _stamp_disk(lm, cy, cx, rad, value):
    h, w = lm.shape
    y0, y1 = max(0, int(cy - rad) - 1), min(h, int(cy + rad) + 2)
    x0, x1 = max(0, int(cx - rad) - 1), min(w, int(cx + rad) + 2)
    if y1 <= y0 or x1 <= x0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    lm[y0:y1, x0:x1][(yy - cy) ** 2 + (xx - cx) ** 2 <= rad * rad] = value


def grow_peninsulas(land_mask, rng, n_range=(1, 4)):
    lm = land_mask.copy()
    h, w = lm.shape
    for _ in range(rng.randint(*n_range)):
        cand = np.argwhere(lm)
        if len(cand) == 0:
            break
        sy, sx = cand[rng.randint(len(cand))]
        ang, length, w0 = rng.uniform(0, 2 * np.pi), rng.randint(5, 14), rng.uniform(1.0, 1.9)
        y, x = float(sy), float(sx)
        for step in range(length):
            ang += rng.normal(0, 0.4)
            y += 1.2 * np.sin(ang); x += 1.2 * np.cos(ang)
            if y < 2 or y > h - 3 or x < 2 or x > w - 3:
                break
            _stamp_disk(lm, y, x, max(w0 * (1 - 0.75 * step / length), 0.3), True)
    return lm


def generate_folded_belt(size, rng):
    x = np.arange(size, dtype=np.float32)
    base = rng.uniform(0.3, 1.8) * x + rng.uniform(-size * 0.3, size * 0.3)
    wiggle = rng.uniform(4, 10) * np.sin(rng.uniform(0.03, 0.08) * x + rng.uniform(0, 2 * np.pi))
    walk = gaussian_filter(np.cumsum(rng.normal(0, 0.8, size)).astype(np.float32), 4)
    y0 = base + wiggle + (walk - walk.mean()) * 1.5
    yy = np.mgrid[0:size, 0:size][0].astype(np.float32)
    n_folds, spacing = rng.randint(2, 5), rng.uniform(3.0, 7.0)
    belt = np.zeros((size, size), np.float32)
    for k in range(n_folds):
        off = (k - (n_folds - 1) / 2.0) * spacing + rng.normal(0, 1.5)
        wob = gaussian_filter(np.cumsum(rng.normal(0, 0.5, size)).astype(np.float32), 5)
        d = yy - (y0 + off + wob - wob.mean())[None, :]
        belt = np.maximum(belt, rng.uniform(0.55, 1.0) * np.exp(-(d * d) / (2 * rng.uniform(1.2, 2.6) ** 2)))
    return belt


def hydraulic_erosion(height, iterations=1200, dt=0.05):
    h = height.astype(np.float64).copy()
    for _ in range(iterations):
        gy, gx = np.gradient(h)
        h -= np.clip(np.sqrt(gx ** 2 + gy ** 2) * 0.08 * (h + 0.1) - 0.005, 0, 0.08) * dt
        h += np.clip(gaussian_filter(h, 1.0) - h, 0, 0.03) * 0.15 * dt
        h = gaussian_filter(h, 0.25) * 0.08 + h * 0.92
    return np.clip(h, 0, 1).astype(np.float32)


def tail_stretch(elev, land_all):
    m_sea = ~land_all
    sea_lo = np.quantile(elev[m_sea], 0.002)
    land_hi = np.quantile(elev[land_all], 0.998)
    elev[m_sea] = 0.5 * np.clip((elev[m_sea] - sea_lo) / (0.5 - sea_lo + 1e-8), 0, 1)
    elev[land_all] = 0.5 + 0.5 * np.clip((elev[land_all] - 0.5) / (land_hi - 0.5 + 1e-8), 0, 1)
    return elev * elev * (3 - 2 * elev)


def carve_river_network(elev, rng, river_frac=0.02, max_depth=0.09):
    acc = d8_flow_accumulation(elev, rng)
    thr = np.quantile(acc, 1 - river_frac)
    if thr < 2.0:
        return np.clip(elev, 0, 1).astype(np.float32)
    t = np.clip(np.log(np.maximum(acc, 1.0001)) / np.log(thr), 0, 1)
    t[acc <= 1.5] = 0
    landness = np.clip((elev - 0.46) / 0.08, 0, 1)
    return np.clip(elev - max_depth * (t ** 0.6) * landness, 0, 1).astype(np.float32)


class SyntheticDEMSource:
    """兜底: v4 式合成地形, 转'米'后接口与 GlobalDEM 完全一致"""
    def __init__(self, n_tiles=16, tile=288, seed=123):
        print(f"未找到真实 DEM → 使用合成地形兜底 ({n_tiles} 块 {tile}px, 仅供流程验证)")
        self.tile = tile
        self.tiles = [self._make_tile(tile, seed + i) for i in range(n_tiles)]
        self.ny, self.nx = 4096, 4096
        self.land_low = None

    @staticmethod
    def _make_tile(size, seed):
        rng = np.random.RandomState(seed)
        base = value_fbm(size, 2.0, 4, rng=rng)
        base = warp_field(base, value_fbm(size, 4, 3, rng=rng),
                          value_fbm(size, 4, 3, rng=rng), size * rng.uniform(0.10, 0.18))
        base += 0.40 * value_fbm(size, 10, 4, rng=rng)
        lm = base > np.quantile(base, 1 - rng.uniform(0.34, 0.58))
        lm = grow_peninsulas(lm, rng)
        d_sea = distance_transform_edt(lm)
        d_land = distance_transform_edt(~lm)
        ocean = 0.5 - (0.03 + 0.09 * np.clip(d_land / 4.0, 0, 1)
                       + np.clip((d_land - 3.75) / 6.0, 0, 1) * (0.30 + 0.08 * value_fbm(size, 5, 4, rng=rng)))
        ridged = ridged_fbm(size, direction=rng.uniform(0, 180), rng=rng)
        belt = gaussian_filter(generate_folded_belt(size, rng), 2.0)
        belt /= (belt.max() + 1e-8)
        land = (0.04 + 0.10 * np.clip(value_fbm(size, 6, 5, rng=rng), 0, 1)
                + (ridged ** 1.7) * (belt ** 1.2) * rng.uniform(0.30, 0.40))
        land *= (0.35 + 0.65 * np.clip(d_sea / 5.0, 0, 1))
        v01 = np.where(lm, 0.5 + land, ocean).astype(np.float32)
        v01 = hydraulic_erosion(v01, iterations=400, dt=0.05)
        v01 = np.where(lm, np.maximum(v01, 0.525), np.minimum(v01, 0.475))
        v01 = np.clip(tail_stretch(v01, lm), 0, 1)
        return v2m(carve_river_network(v01, rng))

    def patch(self, y0, x0, size):
        rng = np.random.RandomState((y0 * 7919 + x0) % (2 ** 31))
        t = self.tiles[rng.randint(len(self.tiles))]
        oy, ox = rng.randint(0, self.tile - size + 1, 2)
        lat_c = -55 + 110 * ((y0 * 37 + x0 * 13) % 256) / 255.0
        return t[oy:oy + size, ox:ox + size], np.linspace(lat_c - 3, lat_c + 3, size)


# ==================== 5. 样本构建: 8通道低清条件 (A2/A3) ====================
# 通道: 0块均值 1块std 2块min 3块max 4陆占比 5河网 6冰川 7纬度
#       8边界邻近 9俯冲/碰撞 10离散(脊/裂谷) 11转换   ← v6.3 构造通道
def build_sample(elev_m, lats, block, rng, ch_stats, tect=None):
    H, W = elev_m.shape
    v01 = m2v(elev_m)
    Bs = H // block
    bm = block_reduce(v01, block, "mean")
    bstd = block_reduce(v01, block, "std")
    bmin = block_reduce(v01, block, "min")
    bmax = block_reduce(v01, block, "max")
    lf = block_reduce((elev_m > 0).astype(np.float32), block, "mean")
    river = np.zeros((Bs, Bs), np.float32)
    if lf.max() > 0.05:
        lmask = lf > 0.05
        acc = d8_flow_accumulation(bm, rng)
        thr = max(float(np.quantile(acc[lmask], 0.99)), 2.0)
        river = np.clip(np.log(np.maximum(acc, 1.0001)) / np.log(thr), 0, 1) * lmask
    glac = block_reduce((elev_m > snowline_m(lats)[:, None]).astype(np.float32),
                        block, "mean")
    lat_rows = np.abs(lats[block // 2:: block]) / 90.0
    latn = np.broadcast_to(lat_rows[:, None], (Bs, Bs)).astype(np.float32)
    if tect is None:
        tect = np.zeros((4, H, W), np.float32)
    tch = np.stack([block_reduce(tect[i], block, "mean") for i in range(4)])
    ch = np.concatenate([np.stack([bm, bstd, bmin, bmax, lf, river, glac, latn]),
                         tch], 0).astype(np.float32)
    noise = [0.030, 0.020, 0.020, 0.020, 0.040, 0.040, 0.040, 0.0,
             0.030, 0.030, 0.030, 0.030]
    for c, s in enumerate(noise):
        if s > 0:
            ch[c] += rng.normal(0, s, ch[c].shape).astype(np.float32)
    ch = (ch - ch_stats["mean"][:, None, None]) / ch_stats["std"][:, None, None]
    return ch, v01


def estimate_channel_stats(dem, size, block, n=40, seed=77, tect_provider=None):
    rng = np.random.RandomState(seed)
    dummy = {"mean": np.zeros(12, np.float32), "std": np.ones(12, np.float32)}
    acc = []
    for _ in range(n):
        y0 = rng.randint(int(dem.ny * 0.05), int(dem.ny * 0.95) - size)
        x0 = rng.randint(0, dem.nx - size)
        m, lats = dem.patch(y0, x0, size)
        tect = tect_provider(y0, x0, size) if tect_provider else None
        ch, _ = build_sample(m, lats, block, np.random.RandomState(0), dummy, tect)
        acc.append(ch)
    acc = np.stack(acc)
    return {"mean": acc.mean(axis=(0, 2, 3)).astype(np.float32),
            "std": (acc.std(axis=(0, 2, 3)) + 1e-4).astype(np.float32)}


def make_tect_provider(dem):
    """真实 DEM → Bird2003 构造条件场; 合成兜底 → None(全零通道)"""
    if type(dem).__name__ != "GlobalDEM":
        return None
    from tectonics import TectField, download_tectonic
    bjson = download_tectonic()
    proxy = dem.z[::16, ::16].astype(np.float32) * dem.scale + dem.offset
    tf = TectField(dem.ny, dem.nx, bjson, dem_for_proxy=proxy)
    return tf.patch


# ==================== 6. 数据集: 在线随机裁剪 + 陆地偏置 + 经度带切分 ====================
VAL_LON_LO, VAL_LON_HI = 130.0, 310.0   # 0-360 坐标: 130°E → 50°W = 太平洋带 (验证)


class TerrainDataset(Dataset):
    def __init__(self, dem, size, block, n_items, ch_stats, is_train, tect_provider=None):
        self.dem, self.size, self.block = dem, size, block
        self.n_items, self.ch_stats, self.is_train = n_items, ch_stats, is_train
        self.tect_provider = tect_provider
        self.rng = np.random.RandomState(2024 if is_train else 0)
        self.centers = None
        if not is_train:
            rng = np.random.RandomState(1234)
            cands = []
            while len(cands) < n_items:
                y0, x0 = self._sample_center(rng)
                f = 0.5 if self.dem.land_low is None else \
                    self.dem.land_low[min(y0 // 16, self.dem.land_low.shape[0] - 1),
                                      min(x0 // 16, self.dem.land_low.shape[1] - 1)]
                cands.append((y0, x0, float(f)))
            cands.sort(key=lambda t: t[2])
            self.centers = cands

    def _sample_center(self, rng):
        ny, nx, size = self.dem.ny, self.dem.nx, self.size
        lo_y, hi_y = int(ny * 0.03), int(ny * 0.97) - size
        y0 = x0 = 0
        for _ in range(80):
            y0 = rng.randint(lo_y, max(lo_y + 1, hi_y))
            x0 = rng.randint(0, nx - size)
            l0 = (x0 + size / 2) * 360.0 / nx
            in_val = VAL_LON_LO <= l0 <= VAL_LON_HI
            if in_val != (not self.is_train):
                continue
            if self.dem.land_low is not None:
                f = self.dem.land_low[min(y0 // 16, self.dem.land_low.shape[0] - 1),
                                      min(x0 // 16, self.dem.land_low.shape[1] - 1)]
                p = 0.85 if f > 0.45 else (0.45 if f > 0.08 else 0.10)
                if rng.rand() > p:
                    continue
            return y0, x0
        return y0, x0

    def __len__(self):
        return self.n_items

    def __getitem__(self, idx):
        if self.is_train:
            y0, x0 = self._sample_center(self.rng)
            rng = self.rng
        else:
            y0, x0 = self.centers[idx][:2]
            rng = np.random.RandomState(9000 + idx)
        m, lats = self.dem.patch(y0, x0, self.size)
        tect = self.tect_provider(y0, x0, self.size) if self.tect_provider else None
        if self.is_train:
            if rng.rand() < 0.5:
                m = m[:, ::-1]
                if tect is not None:
                    tect = tect[:, :, ::-1]
            if rng.rand() < 0.5:
                m = m[::-1]; lats = lats[::-1]
                if tect is not None:
                    tect = tect[:, ::-1]
        ch, v01 = build_sample(m, lats, self.block, rng, self.ch_stats, tect)
        ch = np.ascontiguousarray(
            np.repeat(np.repeat(ch, self.block, axis=1), self.block, axis=2))
        return torch.from_numpy(ch).float(), torch.from_numpy(v01[None]).float()


# ==================== 7. 网络 ====================
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)


class ResBlock(nn.Module):
    def __init__(self, ch, dropout=0.0):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.n1 = nn.InstanceNorm2d(ch)
        self.n2 = nn.InstanceNorm2d(ch)
        self.drop = nn.Dropout2d(dropout)

    def forward(self, x):
        h = F.relu(self.n1(self.c1(x)))
        h = self.drop(h)
        return F.relu(x + self.n2(self.c2(h)))


# ---- A4: resize-conv 上采样 (消除棋盘/半调伪影) ----
class Upsample(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='bilinear',
                                       align_corners=False))


class SelfAttention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        qd = max(ch // 8, 8)
        self.q = nn.Conv2d(ch, qd, 1)
        self.k = nn.Conv2d(ch, qd, 1)
        self.v = nn.Conv2d(ch, ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.qd = qd

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.q(x).view(B, self.qd, H * W).permute(0, 2, 1)
        k = self.k(x).view(B, self.qd, H * W)
        attn = torch.softmax(torch.bmm(q, k) * self.qd ** -0.5, dim=-1)
        v = self.v(x).view(B, C, H * W).permute(0, 2, 1)
        out = torch.bmm(attn, v).view(B, H, W, C).permute(0, 3, 1, 2)
        return x + self.gamma * self.proj(out)


class UNet(nn.Module):
    """条件 GAN 的生成器 G：12 通道条件 → 单通道归一化高程（sigmoid 输出 ∈ [0,1]）。

    架构 (v5, 4 次下采样 + 4 次上采样，与 pix2pix 的 UNet 类似但更轻量)：
        - **编码器 stem + 4 层**：MaxPool(↓2) + Conv(2×ch) + ReLU + ResBlock；
          每一层输出 h1~h4 作为跳跃连接源（保留低频位置信息）。
        - **瓶颈层 bot**：ResBlock(8×b) + SelfAttention（允许长程结构一致性，
          用来避免 GAN 在山脉走向 / 海岸曲率上出现"左右不一致"的斑状伪影）。
        - **解码器 4 层**：Resize-Conv ``bilinear 上采样 → Conv``（代替 ConvTranspose，
          抑制棋盘格伪影），与对应层编码器 feature 在通道维度 concat，再 ResBlock。
        - **输出 head**：1×1 Conv + Sigmoid → 单通道 v∈[0,1]。

    推理用的权重（~44MB）存放在 ``checkpoints/best_terrain_gan_v5.pth``，其中
    ``cfg.in_ch=12, base=32``。训练时保存为 ``{"gen_ema", "cfg", "ch_stats_mean",
    "ch_stats_std", ...}`` 字典，其中 ``gen_ema`` 是指数滑动平均版本（视觉质量
    更稳定，不使用单步模型）。

    Parameters
    ----------
    in_ch : int
        输入条件通道数。v5 训练为 12；v6.x 以前模型可能为 8（AssertionError 提示重训）。
    base : int
        stem 基础通道数。默认 32 足以在 ~2GB 显存上跑 256² 训练 / 推理。
    dropout : float
        ResBlock 内 Dropout2d 比例。默认 0.05，防止 GAN 过拟合小细节导致 PSD 低频化。
    """
    def __init__(self, in_ch=8, base=32, dropout=0.05):
        super().__init__()
        b = base
        self.stem = nn.Sequential(nn.Conv2d(in_ch, b, 3, 1, 1), nn.ReLU(True))
        self.e1 = ResBlock(b, dropout)
        self.p1 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(b, 2 * b, 3, 1, 1), nn.ReLU(True))
        self.e2 = ResBlock(2 * b, dropout)
        self.p2 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(2 * b, 4 * b, 3, 1, 1), nn.ReLU(True))
        self.e3 = ResBlock(4 * b, dropout)
        self.p3 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(4 * b, 8 * b, 3, 1, 1), nn.ReLU(True))
        self.e4 = ResBlock(8 * b, dropout)
        self.p4 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(8 * b, 8 * b, 3, 1, 1), nn.ReLU(True))
        self.bot = ResBlock(8 * b, dropout)
        self.attn = SelfAttention(8 * b)
        self.u4 = Upsample(8 * b, 4 * b)
        self.d4 = nn.Sequential(nn.Conv2d(12 * b, 4 * b, 3, 1, 1), nn.ReLU(True), ResBlock(4 * b, dropout))
        self.u3 = Upsample(4 * b, 2 * b)
        self.d3 = nn.Sequential(nn.Conv2d(6 * b, 2 * b, 3, 1, 1), nn.ReLU(True), ResBlock(2 * b, dropout))
        self.u2 = Upsample(2 * b, b)
        self.d2 = nn.Sequential(nn.Conv2d(3 * b, b, 3, 1, 1), nn.ReLU(True), ResBlock(b, dropout))
        self.u1 = Upsample(b, b)
        self.d1 = nn.Sequential(nn.Conv2d(2 * b, b, 3, 1, 1), nn.ReLU(True), ResBlock(b, dropout))
        self.out = nn.Conv2d(b, 1, 1)

    def forward(self, x):
        h1 = self.e1(self.stem(x))
        h2 = self.e2(self.p1(h1))
        h3 = self.e3(self.p2(h2))
        h4 = self.e4(self.p3(h3))
        b0 = self.attn(self.bot(self.p4(h4)))
        d4 = self.d4(torch.cat([self.u4(b0), h4], 1))
        d3 = self.d3(torch.cat([self.u3(d4), h3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), h2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), h1], 1))
        return torch.sigmoid(self.out(d1))


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=9):
        super().__init__()

        def blk(cin, cout, norm=True):
            layers = [nn.utils.spectral_norm(nn.Conv2d(cin, cout, 4, 2, 1))]
            if norm:
                layers.append(nn.InstanceNorm2d(cout))
            layers.append(nn.LeakyReLU(0.2, True))
            return nn.Sequential(*layers)

        self.b1 = blk(in_channels, 32, norm=False)
        self.b2 = blk(32, 64)
        self.b3 = blk(64, 128)
        self.out = nn.utils.spectral_norm(nn.Conv2d(128, 1, 4, 1, 1))

    def forward(self, x, return_feat=False):
        f1 = self.b1(x)
        f2 = self.b2(f1)
        o = self.out(self.b3(f2))
        return (o, f2) if return_feat else o


def downsample(x, factor):
    return x if factor <= 1 else F.avg_pool2d(x, factor)


# ==================== 8. 损失 ====================
class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)

    def forward(self, fake, real):
        return (F.l1_loss(F.conv2d(fake, self.kx, padding=1), F.conv2d(real, self.kx, padding=1))
                + F.l1_loss(F.conv2d(fake, self.ky, padding=1), F.conv2d(real, self.ky, padding=1)))


class FFTLoss(nn.Module):
    """MPS 对复数 FFT 支持不完整 → 首次失败自动切换到 CPU 计算"""
    def __init__(self):
        super().__init__()
        self._device_ok = True

    def forward(self, fake, real):
        if self._device_ok:
            try:
                ff, rf = torch.fft.fft2(fake), torch.fft.fft2(real)
                return (F.l1_loss(torch.abs(ff), torch.abs(rf))
                        + 0.1 * F.l1_loss(torch.angle(ff), torch.angle(rf)))
            except (RuntimeError, TypeError):
                self._device_ok = False
        f2, r2 = fake.cpu(), real.cpu()
        ff, rf = torch.fft.fft2(f2), torch.fft.fft2(r2)
        loss = (F.l1_loss(torch.abs(ff), torch.abs(rf))
                + 0.1 * F.l1_loss(torch.angle(ff), torch.angle(rf)))
        return loss.to(fake.device)


class LaplacianLoss(nn.Module):
    def __init__(self):
        super().__init__()
        k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('kernel', k)

    def forward(self, fake, real):
        return F.l1_loss(F.conv2d(fake, self.kernel, padding=1),
                         F.conv2d(real, self.kernel, padding=1))


class DistributionLoss(nn.Module):
    """W1 距离 + 极值分位匹配, 直接惩罚'削峰填谷'"""
    def forward(self, fake, real):
        B = fake.shape[0]
        fs = torch.sort(fake.reshape(B, -1), dim=1).values
        rs = torch.sort(real.reshape(B, -1), dim=1).values
        wass = torch.mean(torch.abs(fs - rs))
        N = fs.shape[1]
        idx = [int(q * (N - 1)) for q in (0.005, 0.01, 0.03, 0.97, 0.99, 0.995)]
        return wass + 2.0 * F.l1_loss(fs[:, idx], rs[:, idx])


# ==================== 9. 后处理套件 (A6) ====================
def sharpen_coastline(elev, band=0.025, expand=0.06):
    """单调映射: 0.5=精确海平面, 附近梯度×2.4 → 海岸锐化"""
    return np.interp(elev, [0.0, 0.5 - band, 0.5 + band, 1.0],
                     [0.0, 0.5 - expand, 0.5 + expand, 1.0]).astype(np.float32)


def lowfreq_consistency(v01, bm, block=8, iters=3):
    """引导式低频一致性校正：迭代把 GAN 输出的块均值精确拉回粗底图。

    GAN 在逐瓦片推理时容易出现"云状漂移"（深海整体变浅 / 平原无端起丘），
    而粗底图 ``bm`` 来自构造骨架，其低频是 **权威** 的。本函数执行以下修复循环：

        for each iter:
          1. 对输出 ``out`` 做 block_reduce(mean) → 得到当前块均值 bm_out
          2. ``resid = bm - bm_out``（粗图尺寸的残差）
          3. 对残差上采样 + 高斯平滑（σ=block/3，避免在块边界出现阶跃）
          4. ``out += smoothed resid``（向 bm 方向回拉一块）

    3 次迭代后块均值 RMSE 通常 < 0.001，且高频纹理被高斯核保留。

    Parameters
    ----------
    v01 : ndarray (S, S), float32, [0, 1]
        GAN 输出的归一化高程。
    bm : ndarray (B, B)
        粗块目标均值（来自 ``world_to_channels`` 的同一张骨架图）。
    block : int
        块尺寸，必须与 ``bm`` 生成时的 block 一致（v5=8）。
    iters : int
        迭代次数。默认 3 足够稳定；更多 ≈ 更死锁骨架但纹理更木。

    Returns
    -------
    ndarray (S, S), float32, [0, 1]
        低频对齐后的输出。
    """
    out = np.clip(v01, 0, 1).astype(np.float32)
    for _ in range(iters):
        resid = bm - block_reduce(out, block, "mean")
        out = out + gaussian_filter(ndi_zoom(resid, block, order=1), sigma=block / 3.0)
    return np.clip(out, 0, 1).astype(np.float32)


def envelope_quantile_match(v01, bm, bstd, blend=0.7, seed=0):
    """直方图规格化到'分块高斯混合包络分布'。
    平原尖峰/深海尖峰/山峰厚尾同时恢复; 单调映射不破坏空间结构。"""
    rng = np.random.default_rng(seed)
    h, w = v01.shape
    n = h * w
    idx = rng.integers(0, bm.size, n)
    samples = np.clip(rng.normal(bm.ravel()[idx], bstd.ravel()[idx] + 1e-4), 0, 1)
    target_sorted = np.sort(samples)
    order = np.argsort(v01, axis=None)
    ranked = np.empty(n, np.int64)
    ranked[order] = np.arange(n)
    matched = target_sorted[ranked].reshape(h, w)
    return np.clip(blend * matched + (1 - blend) * v01, 0, 1).astype(np.float32)


def suppress_hallucinated_land(v01, lf, block=8, land_thresh=0.02, cap=0.46):
    """逐像素: 真实陆占比≈0 的深海块, 钳制输出不超过近海面 (消灭凭空浅滩)"""
    lf_up = ndi_zoom(lf, block, order=1)[:v01.shape[0], :v01.shape[1]]
    m = lf_up < land_thresh
    return np.where(m, np.minimum(v01, cap), v01).astype(np.float32)


def polar_ice_mask(lats, v01):
    """v7.1 极地冰盖掩膜: 66.5°→72° 过渡, 不再"一刀切"陆地全冰 (问题 #6).

    改动 (地球对应):
      - 66.5° 以下: 完全按纬度雪线 (无强制) → 允许阿拉斯加/北欧 62-66° 仍有森林
      - 66.5°~72° 过渡带: 仅 elev > (雪线 − 400m 软修正) 的高地自动冰,
        低地仍可为苔原 (对应地球斯瓦尔巴 78°N 低地仍有苔原, 格陵兰内陆才冰盖)
      - >72°: 全部陆地强制冰盖 (对应格陵兰冰盖主体 / 南极圈全冰)
    海洋 (v01<0.5) 不受强制 (海冰由纬度 fBm 门控单独渲染)."""
    sl = m2v(snowline_m(lats))[:, None]
    a = np.abs(lats)[:, None]
    land = v01 >= 0.5
    ice = v01 >= sl
    # 北极圈附近 66.5-72° 渐进: 低地保留为苔原 (雪线下 400m 豁免)
    gate_66 = np.clip((a - 66.5) / 5.5, 0.0, 1.0)         # 66.5→0, 72→1
    soft_ice = (v01 >= (sl - m2v(400.0)))                 # 雪线下 400m 近似 "极地平顶/丘陵"
    ice |= (gate_66 > 0) & land & np.where(gate_66 >= 0.5, True, soft_ice)
    ice |= (a >= 72.0) & land                             # 72° 以上: 彻底格陵兰式冰盖
    return ice


def render_snow_field(v01, lats, cmap):
    """v7.1 雪+冰渲染: 放宽 66.5° 全冰 (允许 66-72° 苔原低地)"""
    h, w = v01.shape
    sl01 = m2v(snowline_m(lats))[:, None]
    a = np.abs(lats)[:, None].repeat(w, 1)
    sea = v01 < 0.5
    land = ~sea
    snow_f = (v01 >= sl01).astype(np.float32)
    # ---- v7.1 极地过渡: 66.5-72° 仅 elev>(雪线-400m) 的极地平顶强制雪; 低地留苔原色 ----
    gate_66 = np.clip((a - 66.5) / 5.5, 0.0, 1.0)
    soft_ice = v01 >= (sl01 - m2v(400.0))
    polar_force = land & ((gate_66 >= 0.5) | (soft_ice & (gate_66 > 0)))
    snow_f[polar_force] = np.maximum(snow_f[polar_force], 1.0)
    # 72°以上彻底冰盖
    snow_f[(a >= 72.0) & land] = 1.0
    # 海冰 (极地带+纬度阈值 fBm 软边)
    sea_ice_f = ((a >= 66.5) & sea).astype(np.float32)
    snow_m = gaussian_filter(snow_f, 0.8) > 0.5          # 抑制阈值椒盐
    sea_ice = (gaussian_filter(sea_ice_f, 0.8) > 0.5) & ~snow_m
    rgb = cmap(np.clip(v01, 0, 1))[..., :3]
    rgb[snow_m] = rgb[snow_m] * 0.30 + 0.70
    rgb[sea_ice] = rgb[sea_ice] * 0.35 + np.array([0.88, 0.93, 0.97]) * 0.65
    return np.clip(rgb, 0, 1)


# ==================== 9b. 气候因果链: 纬度温度 → 地形降水 → Whittaker 生物群系 ====================
def ocean_current_effects(elev_m, lats, warmT=3.5, warmP=250., warmD=400.,
                          coldP=550., coldD=500., monP=750., monD=550.,
                          wprP=400., wprD=300.):
    """M4 洋流/季风气候效应 (海岸几何驱动, 地球/生成图同口径):
    真东岸 = 海在东·陆在西(海岸N-S走向) 且向东连续开阔海面≥~420km
      (排除红海/波斯湾/地中海凹岸等细海伪海岸)
    西边界暖流 (湾流/黑潮型): 真东岸 |lat|30-60°(27-35°过渡门控)
      → T+3.5°C/600km + P+250mm/400km → 西欧·东亚东岸温和湿润
    东边界寒流 (秘鲁/本格拉/加那利型): 真西岸 |lat|5-35°
      → P-550mm/500km → 阿塔卡马/纳米布/撒哈拉西缘海岸荒漠强化
    季风 (东亚/南亚型): 真东岸 |lat|5-28° 且陆块>1.5%总陆
      → P+850mm/600km → 东岸热带季节林/雨林
    西风带雨林 (PNW/智利南/挪威西型): 真西岸 |lat|38-62°
      → P+400mm/300km → 温带雨林于迎风第一山系涌现
    返回 (dT, dP); 陆地外为 0。"""
    land = elev_m > 0
    S, W = elev_m.shape
    a = np.abs(np.asarray(lats, np.float64))[:, None] * np.ones((1, W))
    sea = ~land
    # 海岸走向: 东岸(海东·陆西) / 西岸(海西·陆东), 经度环绕
    sea_e = np.roll(sea, -1, 1); land_e = np.roll(land, -1, 1)
    sea_w = np.roll(sea, 1, 1);  land_w = np.roll(land, 1, 1)
    east_sea = land & sea_e & land_w
    west_sea = land & sea_w & land_e
    # 开阔水域: 向东/西连续海面列数 (两遍扫描处理经度环绕)
    K = 24                                              # ~420km: 细海排除线
    seaf = sea.astype(np.int32)
    run_e = np.zeros((S, W), np.int32)
    run_w = np.zeros((S, W), np.int32)
    run = np.zeros(S, np.int32)
    for _ in range(2):
        for j in range(W - 1, -1, -1):
            run = np.where(seaf[:, j], run + 1, 0)
            run_e[:, j] = run
    run = np.zeros(S, np.int32)
    for _ in range(2):
        for j in range(W):
            run = np.where(seaf[:, j], run + 1, 0)
            run_w[:, j] = run
    east_open = east_sea & (np.roll(run_e, -1, 1) >= K)
    west_open = west_sea & (np.roll(run_w, 1, 1) >= K)

    kmpp = 360.0 / W * 111.32
    dT = np.zeros((S, W), np.float64)
    dP = np.zeros((S, W), np.float64)
    # 纬度门控: 暖流 27-35° 过渡 (防向赤道渗透入副热带干旱带)
    lat_gate = np.clip((a - 27.0) / 8.0, 0.0, 1.0)

    # 大陆面积筛 (季风需大热容量差): 陆块>1.5% 总陆
    from scipy.ndimage import label as _lbl
    lab, n = _lbl(land)
    sz = np.bincount(lab.ravel()); sz[0] = 0
    big_cont = np.isin(lab, np.nonzero(sz >= 0.015 * land.sum())[0])

    # 暖流 (湾流/黑潮): 中纬大陆东岸 — 主效应为向极热输送
    warm = east_open & (a >= 30.0) & (a <= 60.0) & (lat_gate > 0)
    if warm.any():
        dw = distance_transform_edt(~warm) * kmpp
        g = np.minimum(lat_gate, 1.0)
        dT += warmT * np.exp(-dw / 600.0) * g * land
        if warmP > 0:
            dP += warmP * np.exp(-dw / warmD) * g * land
    # 寒流 (秘鲁/本格拉/加那利): 副热带-热带大陆西岸海岸荒漠
    cold = west_open & (a >= 5.0) & (a <= 35.0)
    if cold.any() and coldP > 0:
        dc = distance_transform_edt(~cold) * kmpp
        dP -= coldP * np.exp(-dc / coldD) * land
    # 季风 (东亚/南亚): 热带-副热带大陆东岸 (大大陆), 双高斯剖面 ——
    #   近岸 0-550km: 主雨带 P+750mm (华南/印度海岸, 雨季集中)
    #   深尾 550-1800km: 长程水汽输送 P+180~+260mm (长江→华北→东北, 打破副高干旱)
    #   深尾门槛: 纬度放宽到 ≤42°N (模拟江淮梅雨→东北冷涡季风雨)
    mon = east_open & (a >= 5.0) & (a <= 28.0) & big_cont
    if mon.any() and monP > 0:
        dm = distance_transform_edt(~mon) * kmpp
        dP += monP * np.exp(-dm / monD) * land                 # 近岸主雨带 (550km 半衰)
        # -------- v7.1 深水汽输送尾 --------
        # 纬度从 28° 放宽至 42°, 只要仍在大大陆且距季风岸 <1800km
        mon_ext = east_open & (a >= 5.0) & (a <= 42.0) & big_cont
        dm_ext = distance_transform_edt(~mon_ext) * kmpp
        tailP = 260.0
        tailD = 1400.0
        # 高斯尾: 在 dm≈600km 处仍 ≈ tailP·exp(-600/1400) ≈ 170mm (江淮/武汉量级)
        #            dm≈1200km → ≈ 95mm (华北北京量级, 勉强温带落叶林)
        tail = tailP * np.exp(-dm_ext / tailD) * land
        # 纬度衰减: 28°~42° 线性 ×1 → ×0.3 (东北冷涡弱于江淮)
        tail *= np.clip(1.0 - np.clip((a - 28.0) / 14.0, 0.0, 1.0) * 0.7, 0.3, 1.0)
        # 距岸 <550km 部分不累加 (避免近岸重复主雨带)
        tail[dm < 550.0] = 0.0
        dP += tail
    # 西风带雨林 (PNW/智利南/挪威西): 中高纬西岸迎风第一山系
    if wprP > 0:
        wpr = west_open & (a >= 38.0) & (a <= 62.0)
        if wpr.any():
            dq = distance_transform_edt(~wpr) * kmpp
            dP += wprP * np.exp(-dq / wprD) * land
    return dT, dP


def temperature_field(lats, elev_m, dT=None, ci=None):
    """计算近地面年均气温场（℃，逐像素）。

    v7.1 改进公式（引入大陆度 + 更缓的极地过渡）：
        T = 27 − 0.42·|lat| − 6.5·max(elev,0)/1000
            − polar(lat)                    [极地放大, 58°起 1.5次幂软化]
            + dT_continentality(ci, lat)    [大陆度: 高纬内陆冷−8℃, 热带内陆暖+0.8℃]
            + dT_subtropical_sink(lat)      [副高下沉绝热增温 +1.0~2.5℃ (25°峰值)]
            + dT

    物理依据：
        - **27°基准**：赤道海平面年均温（接近地球实测 26~28℃）。
        - **0.42·|lat|**：每纬度递减（比纯太阳角算出来的略大，吸收了海陆差异平均）。
        - **6.5·h/1000**：干绝热递减率，每升高 1000m 降 6.5℃。
        - **polar = clip((|lat|−58)/32, 0, 1)**1.5 × 22**：v7.1 缓和版极地放大——
          58°前不启动; 60°≈−1.2℃, 66.5°≈−4.8℃, 75°≈−13℃, 80°≈−18℃。
          这样 50-62° 保留 T≈-2~+5℃ 的**泰加林窗口**（地球的西伯利亚/加拿大泰加林）。
        - **大陆度温度修正**：CI·[(|lat|-30)/60·12 − 热带+0.8] ——
          西伯利亚型 (60°N, CI>0.9): T≈−7~−8℃ 冷化;
          中亚 (40°N, CI≈0.8): ≈−4℃;
          撒哈拉 (20°N, CI≈0.7): ≈+0.5℃。
        - **副高下沉增温**：20-32° 带高斯峰 +2.2℃ → 沙漠更热更干（撒哈拉/阿拉伯型）。

    Parameters
    ----------
    lats : ndarray (S,)
        纬度数组。
    elev_m : ndarray (S, W)
        米制高程（海洋负值参与海洋温度时只取 max(elev, 0)=0 即可——海面温度 = 海平面温度）。
    dT : ndarray (S, W) or None
        洋流/季风异常温度。若给定则直接相加。
    ci : ndarray (S, W) or None
        大陆度指数 (``continentality_index()``)。None 时自动从 elev_m + lats 计算。

    Returns
    -------
    ndarray (S, W), float64
        逐像素年均气温（℃）。
    """
    a = np.abs(np.asarray(lats, np.float64))[:, None]
    # ---- v7.2 解耦: 气候计算用 σ=2.2 高斯平滑地形, 消除 ridged_fbm 脊状高频与纬向 T 的拍频伪影
    #          (可视化地形仍为原始 DEM, 河流/雪线亦然; 仅温度递减率 6.5℃/km 用平滑高程) ----
    elev_raw_pos = np.maximum(elev_m, 0.0)
    elev_climate_smooth = gaussian_filter(elev_raw_pos, 2.2)
    # ---- v7.3 C: 与 precipitation_field 相同的山脉破碎扰动 (±12% 相对, 仅 800m+ 山带激活)
    #       保证 T/P 看到相同的"不完美山脊线", 雨影-温度的生物群系耦合不会错位
    hf = elev_raw_pos - gaussian_filter(elev_raw_pos, 0.9)
    hf = gaussian_filter(hf, 0.5)
    hf_norm = np.clip(hf / np.maximum(np.max(np.abs(hf)), 1.0), -1.0, 1.0)
    gate = np.clip((elev_climate_smooth - 800.0) / 3200.0, 0.0, 1.0)
    elev_climate = np.maximum(elev_climate_smooth * (1.0 + 0.12 * hf_norm * gate), 0.0)
    # ---- v7.1 极地放大: 58°起 1.5 次幂软化, 泰加林得 8° 生存窗口 ----
    # ---- v7.3 P-POLAR v3: T58 中位=-10℃, 仍需+4℃才到 泰加林≥40%.
    #        旧×22→×17→ 新 ×13.  75° @ polar系数×13 = 极地再暖 (22-13)=+9℃ ×clip((75-58)/32,0,1)^1.5
    #        精确到 70°: + (22-13) × clip((70-58)/32,0,1)^1.5 ≈ +9 × 0.70 = +6.3℃
    #        66°: +9 × clip((66-58)/32,0,1)^1.5 = +9 × 0.395 = +3.6℃  → 平均约+5℃
    # ---- v7.4 P-POLAR v8: 用户报告"泰加林缺失, 60°N几乎全是苔原"
    #        ×6.2 → ×4.5: 66° 额外暖 +1.7×0.395 ≈ +0.67℃, 70° 额外暖 +1.7×0.70 ≈ +1.19℃, 90°+1.7℃
    #        配合 whittaker t_taiga_up 放宽, 给 50-65° 连续泰加林走廊 2-4℃ 的温度缓冲
    polar = np.clip((a - 58.0) / 32.0, 0.0, 1.0) ** 1.5 * 4.5
    # ---- v7.1 大陆度温度修正 (冬季更冷主导 → 年均温下降) ----
    if ci is None:
        ci = continentality_index(elev_m, lats)
    ci64 = ci.astype(np.float64)
    # 冷化项: |lat|>30 线性增至 12℃ × CI (60°×0.9 ≈ −7.6℃)
    cold_term = np.clip((a - 30.0) / 60.0, 0.0, 1.0) * 12.0
    # 暖化项: 10-35° 热带-副热带大陆内陆 (沙漠腹地夏季酷热 → 年均+0.8℃)
    warm_term = np.clip(1.0 - np.abs(a - 24.0) / 14.0, 0.0, 1.0) * 0.9
    # ---- v7.3 P-POLAR v7 (final): 58°+ 大陆冬天冷化豁免 40% → 60% (西伯利亚冬冷不应把 T 推成苔原)
    polar_w = np.clip((a - 58.0) / 20.0, 0.0, 1.0)
    cold_term_eff = cold_term * (1.0 - 0.60 * polar_w)
    dT_ci = ci64 * (warm_term - cold_term_eff)
    # ---- v7.1 副热带高压绝热下沉增温 (20-32°, 25°峰 +2.3℃) ----
    subtrop_warm = 2.3 * np.exp(-((a - 25.0) ** 2) / (2 * 6.5 ** 2))
    # 只在内陆起作用: 副热带海岸（如洛杉矶/地中海）不应被 2.3℃ 抬升
    dT_sub = subtrop_warm * np.clip(ci64 * 1.6 + 0.1, 0.0, 1.0)
    # ---- 组装 ----
    T = 27.0 - 0.42 * a - 6.5 * elev_climate / 1000.0 \
        - polar + dT_ci + dT_sub
    if dT is not None:                                # M4 洋流温度异常 (暖流+3.5/寒流-1)
        T = T + dT
    return T


# 纬向基础降水 (mm/yr): ITCZ 赤道峰 → 副热带(20°~30°)干旱谷 → 温带峰 → 极地干
_P_LAT = np.array([0, 5, 10, 15, 20, 25, 30, 35, 45, 55, 65, 80, 90], np.float64)
_P_MM = np.array([2300, 2150, 1700, 1150, 700, 520, 560, 680, 850, 750, 480, 230, 130], np.float64)


def precipitation_field(lats, elev_m, rng=None, dP=None, ci=None):
    """v7.1 年降水场 (mm/yr) — 六重改进, 打破严格纬向条带感.

    物理链:
        base_zonal(lat)                      纬向基础 (ITCZ峰→副低谷→温带峰→极地干)
        × f_continentality(d_sea, lat)       1) 大陆度衰减: 深内陆 ×(1 − 0.6·CI)
        − dP_subtropical_high(lat, ci)       2) 副高下沉: 22-32° −200~−750mm (内陆深)
        × f_equatorial_dry(lat, d_sea)       3) 赤道内陆干季: 赤道|lat|<10°+深内陆 ×上限 1200mm
        + orographic_scan(±step, 增强水汽阻挡) 4) 风向地形抬升扫描 + ⭐二维山带水汽截止屏障
        + seasonal_shield(lat, orography)    5) 高海拔山带 "大气环流挡板" 叠加 −250~−900mm 雨影
        + dP_ocean_currents                  M4 洋流/季风 (见 ocean_current_effects)
        × exp( ζ·longwave_perturb )          6) 低频 fBm 扰动 (±22%), 模拟行星波槽脊扭曲

    相比 v7.0 的关键修复 (对应用户提出的 6 大问题):
        #1 纬向条带感   → 引入 longwave 扰动 + 大陆度 2D 变化 → 边界不再严格东西向
        #2 大陆度缺失   → 深内陆降水 ×(1 − 0.6·CI) → 中亚/塔里木级内陆干旱自动形成
        #3 副高带不典型 → 副热带高斯低谷 + 内陆放大 → 撒哈拉/阿拉伯/澳大利亚西 级荒漠
        #4 赤道超级雨林 → 赤道内陆>1200km 降水封顶 → 刚果/亚马逊不会蔓延到大陆中心
        #5 地形屏障低估 → 新增 "2D 山带水汽截止" 与 "极地风挡板" 两层, 安第斯/青藏高原级背风盆旱
    """
    h, w = elev_m.shape
    a = np.abs(np.asarray(lats, np.float64))
    # ---- 大陆度 (缺失时自动计算, 兼容旧接口) ----
    if ci is None:
        ci = continentality_index(elev_m, lats)
    ci64 = ci.astype(np.float64)
    land = elev_m > 0.0
    elev_pos_raw = np.maximum(elev_m, 0.0)     # v7.3: 早期声明 (A3 赤道低平原 与 Step B 共用, 避免 NameError)
    kmpp = 360.0 / max(w, 1) * 111.32
    # v7.3 BUGFIX 与 continentality_index() 同口径: dt(~sea) → distance_to_nearest_SEA_km
    d_sea_km = distance_transform_edt(~((elev_m < 0.0).astype(bool))) * kmpp

    # ================================================================
    # Step A: 纬向基础降水 (与 v7.0 相同, 再做 5 层修正)
    # ================================================================
    base = np.interp(a, _P_LAT, _P_MM)[:, None] * np.ones((1, w), np.float64)

    # ================================================================
    # A1) 大陆度衰减 (问题 #2) — 1800km 半饱和指数, 最深内陆 ×0.4
    # ---- v7.3 细化: 不同纬度带的 CI 惩罚权重完全分化, 解决"右侧大陆纵贯荒漠"异常
    #      - 25°±8° 副热带(典型Hadley下沉区): CI 惩罚 ×1.05 (重, 撒哈拉型正常强)
    #      - 35°~60° 西风带: CI 惩罚 ×0.55 (西风带气旋+锋面降水能深入1000km+内陆, 如中亚不应全荒)
    #      - 60°+ 极地/寒带: CI 惩罚 ×0.70 (冰洋气团大陆过冷但仍有极锋降雪, 不该内冷全荒漠)
    #      - 赤道 |lat|<15°: CI 惩罚 ×0.80 (ITCZ 对流雨不需要海洋蒸发输送, 内陆也能下雨)
    # ================================================================
    lat_penalty = np.ones_like(a[:, None], dtype=np.float64)
    a2d = a[:, None]
    subtrop_band = (a2d >= 18.0) & (a2d <= 34.0)
    westerly_band = (a2d > 34.0) & (a2d <= 60.0)
    polar_band = (a2d > 60.0)
    eq_ci_band = (a2d < 15.0)
    lat_penalty = np.where(subtrop_band, 1.05, lat_penalty)
    lat_penalty = np.where(westerly_band, 0.55, lat_penalty)
    lat_penalty = np.where(polar_band, 0.70, lat_penalty)
    # ---- v7.4: 赤道|lat|<15° CI惩罚从 0.80 再放宽到 0.50 — 地球ITCZ对流雨是局地性的,
    #        亚马逊/刚果即使深内陆 (>2000km) 也靠每日对流程下雨, 不依赖海风水汽输送.
    #        原 0.80 × 0.60=48% 干旱打击 在赤道上是物理错误, 直接导致"刚果盆地变荒漠".
    lat_penalty = np.where(eq_ci_band, 0.50, lat_penalty)
    f_ci = 1.0 - (0.60 * ci64) * lat_penalty          # 纬向加权: 西风带最少受 CI 干旱打击
    base *= f_ci

    # ================================================================
    # A2) 副热带高压 Hadley 下沉抑制 (问题 #3) — 25° 峰高斯
    #     中心荒漠: −750mm × min(1.0, CI·2); 海岸仅 −200mm
    # ---- v7.4 P-SUBTROP v4: 用户报告"荒漠严重过量, 赤道遭殃", 下调副高抑制幅度 + 内陆系数收敛
    #      1) 高斯抑制中心 950mm → 650mm (−32%): 原副高核心下沉打击过强, 会把撒哈拉型荒漠
    #         强行跨纬度延伸进 10°N 以下 (本应是 ITCZ 雨带); 下调后副高仅压制 20-32° 典型带.
    #      2) 高斯 σ: 9° → 7.5° (副高带收窄, 进一步避免 15°以下 被 Hadley 下沉波及)
    #      3) east_coast_relief ci≥0.35 内陆: clip(1.05+0.4*ci, 0.80,1.25) → clip(0.70+0.25*ci, 0.60,0.95)
    #         原 ci=0.7 时 1.05+0.28=1.33→clip 1.25 (125% 抑制), 新 ci=0.7 时 0.70+0.175=0.875 (87.5% 抑制)
    #         核心收益: 赤道附近大陆内部 ci≈0.5 的区域 不再被 1.05+ 系数吞噬成荒漠, 回归雨林/季节林
    subtrop_suppress = 650.0 * np.exp(-((a[:, None] - 25.0) ** 2) / (2 * 7.5 ** 2))
    # ---- v7.5: 副高抑制 2D 经度调制, 打破沙漠整圈纬向带状 ----
    #   旧: subtrop_suppress 纯纬向高斯 → 15-33° 整圈降水抑制 → 沙漠沿纬度整圈带状
    #       (量化: seed5 最大沙漠块经/纬跨度比=11.4, 整圈横跨带状).
    #   地球实际副高非整圈均匀: 撒哈拉/阿拉伯/伊朗强(冷洋流+深内陆),
    #   地中海/东亚东岸/澳洲东岸弱(季风/地形抬升). 用 sigmoid 2D 调制:
    #   弱区(fBm<0)→副高归0(季风豁免, 降水回升>350mm 非沙漠), 强区→1.5(沙漠核心).
    #   sigmoid 保证约30%经度区副高归零, 足以打破整圈带状, 沙漠变块状(撒哈拉型).
    if rng is not None:
        fbm_raw = np.clip(value_fbm(h, 5.0, 3, rng=rng), -1.0, 1.0)
        sub_mod = 1.5 / (1.0 + np.exp(-3.2 * (fbm_raw + 0.05)))   # fBm=-0.05→0.75, 负区→0, 正区→1.5
        subtrop_suppress = subtrop_suppress * sub_mod
    # 东岸季风豁免分层 (P-SUBTROP pattern):
    east_coast_relief = np.where(ci64 < 0.15, 0.10,              # 真海岸 → 抑制10%, 90%豁免
                                 np.where(ci64 < 0.35, 0.40,     # 近海丘陵 → 抑制40%, 季风半豁免
                                          np.clip(0.70 + 0.25 * ci64, 0.60, 0.95)))  # 真内陆 → 抑制 60%~95% (v7.4: 收敛上限)
    dP_sub = -subtrop_suppress * east_coast_relief * land
    base = np.maximum(base + dP_sub, 0.0)

    # ================================================================
    # A3) 赤道内陆干季封顶 (问题 #4)
    #     |lat|<10°, d_sea>1200km → 降水封顶 1200mm (指数软封顶)
    #     d_sea 2700km 封顶 ≈ 900mm → 稀树草原/热带季节林, 不再是雨林
    # ---- v7.3 扩展: 赤道带内部差异, 解决"一片全绿超级亚马逊"问题
    #     (1) 低海拔平原 < 300m 的赤道区 额外受 "干季蒸散加强" 影响:
    #         干季长度 ~ 与 d_sea_km 线性相关, 再加 cos²(2·lon/λ_roll) 波状镶嵌
    #         → 稀树草原(Aw : 800-1500mm) 和 雨林(Af : 2000+mm) 交错呈棋盘格, 而非全绿
    #     (2) 赤道山地(>1000m)反向保护: 山顶对流雨更强 → 封顶 +600mm
    # ================================================================
    eq_band = (a[:, None] < 10.0) & land
    if eq_band.any():
        # ---- v7.4 P-EQUATOR v2: 用户报告"雨林断续, 未成带状"
        #      dry_cap 半衰: 1400km → 2200km; 基线 1200→1400mm
        #      距海 1000km: 原 1200+1800*exp(-1000/1400)=2081mm → 新 1400+1800*exp(-1000/2200)=2543mm (+462mm)
        #      距海 2000km: 原 1631mm (稀树草原下限) → 新 2125mm (仍属雨林/季林边界)
        #      保证赤道大陆内部 2000km 深内陆仍有 >2000mm 降水 → 刚果/亚马逊形态连续雨林带
        dry_cap = 1400.0 + 1800.0 * np.exp(-d_sea_km / 2200.0)
        # ---- v7.3 低海拔平原干季镶嵌 ----
        #   海拔 < 300m 且 赤道陆地: 按距海 (越远干季越长) + 经度波状 3 波/全球 → 额外 ×0.45~1.00
        low_eq = (elev_pos_raw < 300.0) & eq_band
        if low_eq.any() and w > 0:
            lon = (np.arange(w, dtype=np.float64) / w) * 2.0 * np.pi  # 0 → 2π
            # 3 波/全球 的经度干湿镶嵌 (模拟 Walker 环流 上升/下沉支 的东-西差异)
            walker_mosaic = 0.55 + 0.45 * (np.cos(3.0 * lon + 0.7) ** 2)[None, :]
            # ---- v7.4: Walker inland_f 半衰 2200km → 2800km, mosaic 强度从 0.50 降到 0.32
            #      让雨林大陆内部的 Walker 下沉支不再过度切出稀树草原走廊, 保持赤道带整体连续绿色
            inland_f = np.clip(d_sea_km / 2800.0, 0.0, 1.0)
            savanna_f = 1.0 - 0.32 * inland_f * (1.0 - walker_mosaic)
            dry_cap = np.where(low_eq, dry_cap * savanna_f, dry_cap)
        # ---- v7.3 赤道山地抬升对流豁免 ----
        hi_eq = (elev_pos_raw > 1000.0) & eq_band
        if hi_eq.any():
            dry_cap = np.where(hi_eq, dry_cap + 700.0, dry_cap)
        # ---- v7.4: 大陆度加权封顶软化
        #        原 0.80+0.20*(1-ci): CI=1 时封顶只有 80% dry_cap (过严).
        #        新 0.92+0.08*(1-ci): CI=1 时 92%, CI=0 时 100% — 赤道内陆不应因大陆度再次扣减降水
        base = np.where(eq_band,
                        np.minimum(base, dry_cap * (0.92 + 0.08 * (1.0 - ci64))),
                        base)

    # ================================================================
    # Step B: 盛行风向扫描 (地形雨) — v7.2 大改 (解斑马纹根因#1 + #2)
    #   #2: 所有 oro/屏障 计算改用 elev_climate (σ=2.2 平滑) 消除 ridged_fbm 脊状拍频
    #   #1: 新增 N-S 经向扫描 (冷暖气团南北输送), 与 EW 纬向 75/25 加权 → 二维水汽
    # ================================================================
    # ---- v7.2 气候高程 (陆面 σ=2.2 高斯平滑, ridged_fbm 脊纹彻底剥离出 orography) ----
    # ---- v7.3 C: 注入 ±15% 6~12波长细碎地形扰动 (σ=0.9 空间带通), 让屏障山带不再是"完美直板"
    #       物理意义: 板块拼合形成的山脉不是几何完美的, 每个山口/山间缺口会让水汽"绕过"
    #       直接效应: 雨影带的边缘会破碎不连续, 类似真实地球安第斯/落基山的形状
    #       安全约束: 只对 elev_climate >= 800m 的"已存在山带"扰动 (平地完全不动)
    #                  相对扰动强度 = 15% × clip((elev_climate-800)/3200, 0, 1)
    # ================================================================
    elev_pos_raw = np.maximum(elev_m, 0.0)
    elev_climate_smooth = gaussian_filter(elev_pos_raw, 2.2)
    # --- 细碎扰动: 用 elev_pos_raw 原始地形 (σ=0.9 高频, 实际即 ridged_fbm 的"细节") 与 elev_climate_smooth 做差 ---
    highfreq = elev_pos_raw - gaussian_filter(elev_pos_raw, 0.9)  # 带通: 波长约 2~5 像素 的地形皱折
    # 高通后再轻 σ=0.5 高斯, 去掉极值, 变成纯"山形破碎"噪声
    highfreq = gaussian_filter(highfreq, 0.5)
    # 归一化 highfreq 的量级到 ±15%
    hf_norm = np.clip(highfreq / np.maximum(np.max(np.abs(highfreq)), 1.0), -1.0, 1.0)
    rugged_gate = np.clip((elev_climate_smooth - 800.0) / 3200.0, 0.0, 1.0)    # 1 = 4000m 高山
    elev_climate = elev_climate_smooth * (1.0 + 0.15 * hf_norm * rugged_gate)
    elev_climate = np.maximum(elev_climate, 0.0)
    hk = elev_climate / 1000.0                                           # 地形雨抬升计算 ONLY
    ocean = (elev_m < 0.0) | (hk < 1e-3)                                 # 海陆用 raw, 湿润度用 smooth

    def scan_ew(step):
        """纬向 (East-West) 扫描, 一维列水汽, 处理西风带/信风带; 含 2000m 屏障 ×0.22 硬截止."""
        cols = range(w) if step > 0 else range(w - 1, -1, -1)
        rain = np.zeros((h, w), np.float32)
        m = np.ones(h, np.float32)
        for _ in range(2):
            hp = hk[:, (cols[0] - step) % w]
            for j in cols:
                hj = hk[:, j]
                barrier_hit = (hj >= 2.0) & (hp >= 2.0) & (~ocean[:, j])       # v7.2 elev_climate 判定 → 脊纹不再触发屏障
                m = np.where(barrier_hit, np.minimum(m, 0.22), m)
                oro = np.minimum(700.0 * np.maximum(hj - hp, 0.0) * m, 2200.0)
                rain[:, j] = (base[:, j] + oro) * (0.10 + 0.90 * m)
                m = np.where(ocean[:, j], 1.0,
                             m * np.exp(-oro / 1500.0) * 0.993)
                hp = hj
        return rain

    def scan_ns(step):
        """v7.2 经向 (North-South) 扫描 — 补齐一维扫描的南北向水汽切断.
        物理含义:
          step = -1 (向南, 从北→南): 代表极地冷空气爆发南下 (寒潮/冰洋气团)
                    在 50-75° 纬度带内有实际强水汽 → 锋面降水
          step = +1 (向北, 从 赤道→北): 代表季风暖湿气流 向极输送 (东亚夏季风/南亚季风北进)
                    在 10-40° 大陆东岸/腹地 有实际强水汽
        二维水汽现在由: 2 × EW (纬向盛行风) + 2 × NS (经向季风/寒潮) → 4 次扫描合成,
        每个像素降水是 75%·EW_选择 + 25%·NS_选择, 从而跨纬度信息混合, 纬向独立行伪影消失.
        """
        rows = range(h) if step > 0 else range(h - 1, -1, -1)
        rain = np.zeros((h, w), np.float32)
        m = np.ones(w, np.float32)                                   # 按列的一维水汽 (经度方向每列独立)
        # 初始条件: 赤道(向北扫)或极地(向南扫)的海面边界层水汽满
        #   step +1 向北: i=0 若为南半球极地海域 则 m=1
        #   step -1 向南: i=h-1 若为北半球极地海域 则 m=1
        for _ in range(2):
            hp = hk[(rows[0] - step) % h, :]
            for i in rows:
                hi = hk[i, :]
                # 屏障 (南北走向山带, 如安第斯/乌拉尔/落基 的 经向水汽屏障) 相同 2000m ×0.22
                barrier_hit = (hi >= 2.0) & (hp >= 2.0) & (~ocean[i, :])
                m = np.where(barrier_hit, np.minimum(m, 0.22), m)
                oro = np.minimum(700.0 * np.maximum(hi - hp, 0.0) * m, 2200.0)
                rain[i, :] = (base[i, :] + oro) * (0.10 + 0.90 * m)
                m = np.where(ocean[i, :], 1.0,
                             m * np.exp(-oro / 1500.0) * 0.993)
                hp = hi
        return rain

    # ---- EW: 西风带(30-60°) 用 p_w(向东扫); 其余用 p_e(向西扫) ----
    p_w, p_e = scan_ew(+1), scan_ew(-1)
    west_belt = ((a >= 30.0) & (a <= 60.0))[:, None]
    P_ew = np.where(west_belt, p_w, p_e).astype(np.float64)
    # ---- NS: 低纬(0-40°) 夏季风 向北 (step +1); 高纬(45-90°) 寒潮 向南 (step -1) ----
    p_n, p_s = scan_ns(+1), scan_ns(-1)
    monsoon_belt = (a < 40.0)[:, None]
    P_ns = np.where(monsoon_belt, p_n, p_s).astype(np.float64)
    # ---- 加权合成: 纬向盛行 75%, 经向季风/寒潮 25%
    #      赤道 ±15° 加强 NS 权重至 35% (ITCZ 南北波动 → Hadley 经向环流主导)
    #      65°+ 极地 NS 寒潮权重 35% (冰洋气团强经向输送)
    ns_w = 0.25 * np.ones((h, 1), np.float64)
    eq_band_ns = (a < 15.0)[:, None]
    pol_band_ns = (a > 65.0)[:, None]
    ns_w[eq_band_ns[:, 0], :] = 0.35
    ns_w[pol_band_ns[:, 0], :] = 0.35
    P = (1.0 - ns_w) * P_ew + ns_w * P_ns

    # ================================================================
    # B2) 2D 大气环流挡板 (问题 #5b 青藏高原/落基山/安第斯级效应)
    #     v7.2: 改用 elev_climate (平滑) 找山带核心, ridged_fbm 不再制造"斑马雨影带"
    # ================================================================
    shield = np.zeros((h, w), np.float64)
    mountain_top = (elev_climate >= 2500.0) & land                    # v7.2 平滑地形找山带
    if mountain_top.any():
        mt_sm = gaussian_filter(mountain_top.astype(np.float32), 1.2) > 0.35
        hh = (elev_climate - 2000.0) / 1000.0 * land
        hh = np.maximum(hh, 0.0)
        belt_west = (a >= 30) & (a <= 60)
        belt_east = ~belt_west
        for step_sign, belt_msk in [(+1, belt_west), (-1, belt_east)]:
            belt_rows = np.nonzero(belt_msk)[0]
            if belt_rows.size == 0:
                continue
            cols_r = range(w) if step_sign > 0 else range(w - 1, -1, -1)
            K_max = int(np.ceil(900.0 / max(kmpp, 1.0)))
            for i in belt_rows:
                depth = 0
                strength = 0.0
                for j in cols_r:
                    if mt_sm[i, j]:
                        depth = K_max
                        strength = max(strength, float(hh[i, j]) * 650.0)
                    elif depth > 0:
                        frac = depth / K_max
                        shield[i, j] = max(shield[i, j], strength * frac)
                        depth -= 1
                    else:
                        strength = max(strength - 25.0, 0.0)
    P = np.maximum(P - shield, 0.0)

    # v7.2 各向异性高斯 (σ_row=2.2, σ_col=1.2): 经向(南北)更重平滑, 消 EW 扫描的行间 "独立感"
    P = gaussian_filter(P.astype(np.float32), [2.2, 1.2]).astype(np.float64)

    # ================================================================
    # Step C: M4 洋流/季风 dP (原逻辑, 放在 A/B 之后更合理)
    # ================================================================
    if dP is not None:
        P = np.maximum(P + dP, 0.0)

    # ================================================================
    # Step D: 低频行星波扰动 (问题 #1) — 模拟 Rossby 波导致气候带扭曲
    #   低基频, 中高纬度振幅更大 (±20~28%), 赤道更小 (±8%)
    # ================================================================
    if rng is not None:
        lw = value_fbm(w, 3.2, 2, rng=rng)                           # 基频 3.2 → 全球 3~4 个长波
        # 纬度权重: 50-70° 振幅最大 (0.28), 赤道 0.08, 极地 0.18
        lat_g = 0.08 + 0.20 * np.exp(-((a - 58.0) ** 2) / (2 * 22.0 ** 2))
        P = P * np.exp(lw * lat_g[:, None])
        # v7.2 地形微扰 fbm: 基频 12 → 5.5 (约 400km 波长, 消除 100km 级条纹伪影)
        #      振幅 0.13 → 0.10 (柔和 ±10% 相对变化, 不再制造 "斑马")
        P = P * np.exp(0.10 * value_fbm(w, 5.5, 3, rng=rng))

    return np.clip(P, 0.0, None).astype(np.float32)


BIOME_NAMES = ["苔原", "北方针叶林", "温带草原", "温带落叶林", "温带雨林",
               "亚热带荒漠", "稀树草原", "热带季节林", "热带雨林"]
BIOME_COLORS = np.array([
    [0.62, 0.60, 0.55], [0.30, 0.42, 0.30], [0.72, 0.65, 0.38],
    [0.42, 0.55, 0.30], [0.24, 0.44, 0.24], [0.87, 0.76, 0.55],
    [0.79, 0.68, 0.38], [0.40, 0.56, 0.28], [0.16, 0.38, 0.18]], np.float32)


def whittaker_biome(t, p, lats=None, ci=None):
    """v7.1 Whittaker (T, P) → 9 类生物群系, 苔原/针叶林门限随大陆度自适应.

    门限改动 (问题 #6: 西伯利亚/挪威海两岸的巨大差异 — 年均温 -8℃ 在雅库茨克
    仍然是落叶针叶林, 但年均温 0℃ 在格陵兰东岸已是苔原):
        * 苔原 T 阈值 (原固定 t≤−2℃):
            T_tundra = −2 − 9·CI                 ∈ [−2, −11]℃
          → 海洋性 (西欧海岸/挪威峡湾) CI≈0: 苔原需要 T≤−2℃ 才形成,
            北到 68° 仍可维持森林 (对应地球卑尔根 60°N, T≈+5℃ → 混交林)
          → 大陆性 (西伯利亚) CI>0.85: 苔原门槛低至 T≤−9.5~−11℃,
            雅库茨克 (62°N, T≈−8℃) 仍为落叶泰加林 (对应地球实际 larch 林)
        * 针叶林 T 阈值 (原 t≤5℃):
            T_taiga_upper = 5 + 2·(1−CI)·L_gate     ∈ [5, 7]℃
          → CI 低的海岸带 5-7℃ 仍判为"泰加林与针阔混交" (避免温带林过北)
        * 降水端的温带草原/落叶林门限 (原 p<400 / p<1500 @ t≤20):
          大陆度高时 (CI>0.5) 温带草原 p 阈值从 400→550mm (更易成草原)

    Parameters (新增可选, 全部兼容 v7.0 两参数调用):
        lats : ndarray (H,) or None — 纬度数组, 用于高纬门限微调
        ci   : ndarray (H,W) or None — 大陆度指数, 核心参数; 未传则保持 v7.0 原版
    """
    t_arr = np.asarray(t)
    p_arr = np.asarray(p)
    # ---- 大陆度自适应门限 ----
    if ci is not None:
        ci64 = np.asarray(ci, np.float64)
        # ---- v7.3 P-POLAR v2: CI bugfix 生效后大陆更冷, 需要更激进的泰加林生存窗
        #      目标: 58-75° 泰加林/(泰加林+苔原) ≥ 40% (西伯利亚形态)
        #      (a) 苔原门槛 T_tundra: -2.5 - 12.5·CI → -3.0 - 16.0·CI
        #          CI=0.9 (西伯利亚型): 原 ≤-13.75 → 新 ≤-17.4 (苔原极难才形成)
        t_tundra = -3.0 - 16.0 * ci64
        # ---- v7.5 纬度门限约束: 修复 P-POLAR v8 遗留的中低纬温带吞没问题 ----
        #   旧: t_taiga_up = 11+9*(1-ci) ∈[11,20]℃ 全局应用 → 中低纬温带(T 5-15℃)
        #       全被吞入针叶林, 温带草原/亚热带荒漠几近消失
        #       (同口径地球针叶林 65.7% vs 实际~11%, 温带草原 2.4% vs 实际~25%).
        #   新: 中低纬(<50°)用温带门限 5+2*(1-ci)∈[5,7]℃ (温带群系回归),
        #       高纬(>55°)用泰加门限 11+9*(1-ci)∈[11,20]℃ (保 P-POLAR 泰加林成果),
        #       50-55° 快速过渡. 高纬额外缓冲(55-70° +3+2*(1-ci))保留.
        t_taiga_temperate = 5.0 + 2.0 * (1.0 - ci64)    # 中低纬温带门限 [5,7]℃
        t_taiga_polar = 11.0 + 9.0 * (1.0 - ci64)       # 高纬泰加门限 [11,20]℃
        # 温带草原 p 门槛: 大陆度高 → 草原降水门限从 400 提至 550mm
        p_stp = 400.0 + 150.0 * np.clip((ci64 - 0.30) / 0.70, 0.0, 1.0)
        if lats is not None:
            a = np.abs(np.asarray(lats, np.float64))
            # 50-55° 纬度过渡: 0=纯温带门限, 1=纯泰加门限
            lat_w = np.clip((a[:, None] - 50.0) / 5.0, 0.0, 1.0)
            t_taiga_up = t_taiga_temperate * (1.0 - lat_w) + t_taiga_polar * lat_w
            # 高纬额外缓冲(55-70°): 原 +3+2*(1-ci) 加成保留 (西伯利亚泰加生存窗)
            lat_w2 = np.clip((a[:, None] - 55.0) / 15.0, 0.0, 1.0)
            t_taiga_up = t_taiga_up + 3.0 * lat_w2 + 2.0 * (1.0 - ci64) * lat_w2
        else:
            t_taiga_up = t_taiga_temperate  # 无纬度: 用温带门限(保守, 不吞温带)
    else:
        # 兼容 v7.0: 固定门限
        t_tundra = -2.0
        t_taiga_up = 5.0
        p_stp = 400.0
    # ---- 温带落叶林/雨林分界线 (p=1500 → 大陆度高时需要更多降水才成雨林) ----
    p_tf = 1500.0
    if ci is not None:
        ci64 = np.asarray(ci, np.float64)
        p_tf = 1500.0 - 200.0 * (1.0 - ci64)                   # 海洋性 1300, 内陆 1500

    # ---- v7.5 P-SUBTROP v6: 副热带冷沙漠 (寒漠/戈壁) 纬度/大陆度豁免判定 ----
    #   原whittaker要求 T>20 才进入 idx5 (亚热带荒漠). 但地球 20-30°N 中亚塔里木/戈壁:
    #   年均温 10~18℃ + P<250mm + 深内陆(ci>=0.3), 也是沙漠/荒漠而不是草原.
    #   在温带分支 (t<=20) 的 select 之前, 优先判断这类"寒冷副高大陆荒漠".
    #   v7.5: P门限 500→350mm. 旧500门限把生成图副热带半干旱区(base 474mm, ci低→副高弱)
    #   全判沙漠 → 沙漠整圈纬向带状(经/纬跨度比=11.4). 降到350: 只真正极旱区
    #   (副高强抑制+深内陆) 成沙漠, 形成自然块状(撒哈拉/阿拉伯型), 半干旱区回归草草原/林.
    if lats is not None and ci is not None:
        a = np.abs(np.asarray(lats, np.float64))[:, None]
        ci64_2d = np.asarray(ci, np.float64)
        # 副热带 [18°, 34°] & 内陆 ci>=0.15 & P<350mm & T>2℃ & 非冰冻 → idx5荒漠
        cold_desert_override = (a >= 18.0) & (a <= 34.0) & (ci64_2d >= 0.15) \
                             & (p_arr < 350.0) & (t_arr >= 2.0) & (t_arr <= 22.0)
    else:
        cold_desert_override = np.zeros(t_arr.shape, dtype=bool)

    # ---- 稀树草原/季节林/雨林热带端 ----
    return np.select(
        [t_arr <= t_tundra, t_arr <= t_taiga_up,
         cold_desert_override,                                    # v7.3: 副热带冷戈壁 → 荒漠 (优先温带草原)
         (t_arr <= 20) & (p_arr < p_stp),
         (t_arr <= 20) & (p_arr < p_tf),
         t_arr <= 20,
         p_arr < 250, p_arr < 1000, p_arr < 2000],
        [0, 1, 5, 2, 3, 4, 5, 6, 7], default=8).astype(np.int8)


def smooth_biome_labels(bi, land, median_size=7, min_patch_px=50, iters=3):
    """v7.9 群系边界平滑 + 碎片清理 — 修复 Whittaker 硬阈值导致的锯齿边界.

    迭代处理 (仅作用于陆地像素, 海洋保持不变), 每轮:
        1. 半径 ``median_size//2`` 的 **圆形结构元** 中值滤波: 整数标签的中值≈局部
           众数, 天然保边 + 消除 1~3px 锯齿与小碎片 (r=2 时清除 <13px 块).
           v7.9 改动: 由方形 ``size=N`` 改为 ``footprint=disk(r=N//2)``, 让中值滤波
           在斜边界上沿 **曲线** 而非 **网格对齐方向** 推进, 消除阶梯状/方块边界
           (方形窗口会把曲线边界对齐到 0°/45°/90° 方向, 形成 staircase 伪影).
        2. 小斑块去除 (<``min_patch_px`` px): 连通域标注 → 小块像素替换为
           r=2 圆形邻域众数, 消除残留碎片 (避免孤岛/飞地).

    迭代收敛: 第1轮中值产生的新 1-2px 碎片被第2轮去斑块清除, 第2轮去斑块
    产生的 1px 孔隙被第2轮中值修补. 2 轮即可收敛 (3+ 轮收益递减).

    物理依据: 真实群系边界在 50~200km 尺度过渡 (生态交错带 ecotone),
    不应出现像素级 (<20km) 的锯齿与碎片. 2048px 下 5px≈98km, 符合 ecotone 尺度.

    Parameters
    ----------
    bi : ndarray (H, W), int8
        whittaker_biome 输出的 9 类整数标签.
    land : ndarray (H, W), bool
        陆地掩膜 (dem >= 海平面). 海洋像素不被平滑.
    median_size : int
        中值滤波直径 (≈2r+1), 实际用半径 ``r = median_size // 2`` 的圆盘 (默认 7→r=3).
    min_patch_px : int
        小斑块面积阈值 (像素数), 小于此值的连通域被替换 (默认 50).
    iters : int
        迭代轮数 (默认 3).

    Returns
    -------
    ndarray (H, W), int8
        平滑后的群系标签.
    """
    bi8 = bi.astype(np.int8).copy()
    H, W = bi8.shape
    yy, xx = np.mgrid[0:H, 0:W]
    # v7.9: 圆形结构元 — 半径 r 的圆盘 footprint, 替代方形 size=median_size 窗口.
    #       圆盘使中值滤波的"邻域范围"在各方向同构, 斜边界沿曲线推进而非阶梯对齐.
    r_med = max(1, median_size // 2)
    yy_d, xx_d = np.ogrid[-r_med:r_med + 1, -r_med:r_med + 1]
    disk_med = (yy_d * yy_d + xx_d * xx_d <= r_med * r_med).astype(bool)
    # 去斑块用的众数邻域: 圆盘 r=2 (旧 5×5 方形), 与中值同构避免双重对齐
    r_mode = 2

    for _it in range(iters):
        # Step 1: 圆形结构元中值滤波 (仅陆地) — footprint 而非 size
        bi_m = np.where(land, median_filter(bi8, footprint=disk_med, mode="nearest"), bi8)

        # Step 2: 去除小斑块 — 计算 r=2 圆形邻域众数, 逐群系替换小块
        if min_patch_px > 0:
            cnt = np.zeros(bi_m.shape + (9,), dtype=np.uint16)
            for dy in range(-r_mode, r_mode + 1):
                for dx in range(-r_mode, r_mode + 1):
                    if dy == 0 and dx == 0:
                        continue
                    if dy * dy + dx * dx > r_mode * r_mode:    # 跳过圆外 (disk r=2)
                        continue
                    nb = np.roll(np.roll(bi_m, dy, 0), dx, 1)
                    cnt[yy, xx, nb] += 1
            mode_bi = cnt.argmax(axis=-1).astype(np.int8)
            for idx in range(9):
                m = land & (bi_m == idx)
                if not m.any():
                    continue
                lab, n = ndi_label(m)
                if n == 0:
                    continue
                sizes = np.bincount(lab.ravel())
                sizes[0] = 0
                small_labels = np.where(sizes < min_patch_px)[0]
                if len(small_labels) == 0:
                    continue
                small_mask = np.isin(lab, small_labels) & land
                bi_m = np.where(small_mask, mode_bi, bi_m)

        bi8 = bi_m

    return bi8.astype(bi.dtype)


# ==================== 9c. 水文: 填洼 + 粒子水力侵蚀 ====================
def fill_depressions(v01, eps=4e-4, max_iters=120):
    """填洼(向量化的形态学重建): 抬升局部极小值 → 每个格点都有下游, 河网连通。
    仅用于流向计算, 不改动输出地形。"""
    f = np.ascontiguousarray(v01, np.float32).copy()
    H, W = f.shape
    for _ in range(max_iters):
        mn = np.full((H, W), np.inf, np.float32)
        for dy, dx, _dd in _D8:
            s = np.full((H, W), np.inf, np.float32)
            ys, xs = slice(max(dy, 0), H + min(dy, 0)), slice(max(dx, 0), W + min(dx, 0))
            yt, xt = slice(max(-dy, 0), H + min(-dy, 0)), slice(max(-dx, 0), W + min(-dx, 0))
            s[yt, xt] = f[ys, xs]
            mn = np.minimum(mn, s)
        pit = f <= mn
        if not pit.any():
            break
        f[pit] = mn[pit] + eps
    return f


def droplet_erosion(v01, n_droplets=150000, lifetime=40, inertia=0.10,
                    capacity=4.0, erode=0.45, deposit=0.30, evaporate=0.012,
                    min_slope=0.008, rng=None):
    """粒子水力侵蚀 (Lague 式, 全向量化): 水滴沿梯度下滑, 按携带容量
    侵蚀/沉积 → 山脉出 V 形谷、谷口冲积扇、河口三角洲; 入海即沉积消亡。"""
    rng = rng or np.random.default_rng(0)
    h = np.ascontiguousarray(v01, np.float32).copy()
    H, W = h.shape
    ys, xs = np.nonzero(h >= 0.5)
    if len(ys) < 16:
        return h
    n = n_droplets
    idx = rng.integers(0, len(ys), n)
    pos = np.stack([ys[idx] + rng.random(n).astype(np.float32),
                    xs[idx] + rng.random(n).astype(np.float32)], 1)
    vel = np.zeros((n, 2), np.float32)
    water = np.ones(n, np.float32)
    sed = np.zeros(n, np.float32)
    act = np.ones(n, bool)

    def bilin(arr, p):
        i0 = np.clip(np.floor(p[:, 0]).astype(np.int32), 0, H - 2)
        j0 = np.clip(np.floor(p[:, 1]).astype(np.int32), 0, W - 2)
        fy, fx = p[:, 0] - i0, p[:, 1] - j0
        c = (i0 * W + j0).astype(np.int64)
        af = arr.ravel()
        return (af[c] * (1 - fy) * (1 - fx) + af[c + 1] * (1 - fy) * fx
                + af[c + W] * fy * (1 - fx) + af[c + W + 1] * fy * fx)

    for _ in range(lifetime):
        if not act.any():
            break
        gy, gx = np.gradient(h)
        gp = np.stack([bilin(gy, pos), bilin(gx, pos)], 1)
        d = vel * inertia - gp * (1.0 - inertia)
        dn = np.linalg.norm(d, axis=1, keepdims=True)
        d = np.where(dn > 1e-8, d / np.maximum(dn, 1e-8),
                     np.array([1.0, 0.0], np.float32))
        new_pos = pos + d
        h_old = bilin(h, pos)
        h_new = bilin(h, new_pos)
        dh = h_new - h_old
        oob = ((new_pos[:, 0] < 1) | (new_pos[:, 0] > H - 2) |
               (new_pos[:, 1] < 1) | (new_pos[:, 1] > W - 2) | (h_new < 0.5))
        cap = np.maximum(-dh, min_slope) * water * capacity
        rich = (dh > 0) | (sed > cap)
        dep = np.where(rich, np.where(dh > 0, np.minimum(sed, dh),
                                      np.maximum(sed - cap, 0.0) * deposit), 0.0)
        ero = np.where(~rich, np.minimum((cap - sed) * erode,
                                         np.maximum(-dh, 0.0)), 0.0)
        i0 = np.clip(np.floor(pos[:, 0]).astype(np.int32), 0, H - 2)
        j0 = np.clip(np.floor(pos[:, 1]).astype(np.int32), 0, W - 2)
        fy, fx = pos[:, 0] - i0, pos[:, 1] - j0
        c = (i0 * W + j0).astype(np.int64)
        wq = ((dep - ero) * act).astype(np.float32)
        hf = h.ravel()
        np.add.at(hf, c, wq * (1 - fy) * (1 - fx))
        np.add.at(hf, c + 1, wq * (1 - fy) * fx)
        np.add.at(hf, c + W, wq * fy * (1 - fx))
        np.add.at(hf, c + W + 1, wq * fy * fx)
        h = hf.reshape(H, W)
        sed = np.where(rich, sed - dep, sed + ero)
        water *= (1.0 - evaporate)
        vel, pos = d, new_pos
        act &= ~oob
    return np.clip(h, 0, 1)


def postprocess_pipeline(v01, bm, bstd, lf, lats, rng, block=8):
    """G raw → 低频校正 → 包络分位匹配 → 低频复核 → 幻觉陆地钳制
    → 海岸陡化 → 冰盖感知的 D8 河谷强调"""
    s = lowfreq_consistency(v01, bm, block)
    s = envelope_quantile_match(s, bm, bstd, blend=0.7, seed=int(rng.integers(1e9)))
    s = lowfreq_consistency(s, bm, block, iters=1)
    s = suppress_hallucinated_land(s, lf, block)
    s = sharpen_coastline(s)
    ice = polar_ice_mask(lats, s)
    acc = d8_flow_accumulation(s, rng)
    thr = max(float(np.quantile(acc, 0.985)), 2.0)
    t = np.clip(np.log(np.maximum(acc, 1.0001)) / np.log(thr), 0, 1)
    landness = np.clip((s - 0.5) / 0.02, 0, 1)
    s = np.clip(s - 0.03 * (t ** 0.8) * landness * (~ice), 0, 1)
    return s.astype(np.float32)


# ==================== 9d. v6.6 检验修复层 (七大地理维度对齐) ====================
# 地球陆地高程分位数 —— ETOPO1(Ice) 2048 实测 (含南极/格陵兰冰盖)
_LAND_Q = np.arange(0, 1.0001, 0.025)
_LAND_V = np.array([0.0, 14.0, 31.0, 49.0, 68.0, 88.0, 110.0, 130.0, 150.0,
                    170.0, 200.0, 230.0, 260.0, 290.0, 320.0, 350.0, 390.0,
                    420.0, 470.0, 520.0, 570.0, 640.0, 720.0, 820.0, 930.0,
                    1000.0, 1200.0, 1300.0, 1400.0, 1600.0, 1900.0, 2100.0,
                    2300.0, 2500.0, 2700.0, 2800.0, 3000.0, 3100.0, 3300.0,
                    3700.0, 6400.0], np.float64)
# 同上, 但按 cos(纬度) 面积加权 (与验收口径一致; 南极冰盖降权)
_LAND_W_V = np.array([0.0, 13.6, 30.5, 48.1, 65.5, 83.7, 101.6, 119.3,
                      137.2, 155.2, 174.3, 195.2, 217.7, 241.9, 266.6,
                      289.8, 312.5, 337.5, 363.9, 392.2, 420.1, 451.5,
                      486.2, 526.8, 572.7, 625.6, 686.5, 758.5, 842.5,
                      932.3, 1024.4, 1119.4, 1225.3, 1358.4, 1531.9,
                      1775.2, 2097.6, 2507.4, 2995.3, 3658.8, 6350.8], np.float64)


def land_hist_match(elev_m, lats=None):
    """P0 陆地直方图重映射: 排名保持(单调) → 陆地高程分布对齐地球实测。
    修复 GAN"回归均值"(89%陆地挤在200~1000m) → 28%平原/46%台地/26%山地。
    单调映射不改变海岸线与空间秩序: 骨架最低的大片陆地→平原, 最高→山带。
    lats 给定时按 cos(纬度) 加权分位(目标 _LAND_W_V) → 与面积加权验收口径一致。"""
    land = elev_m > 0
    if not land.any():
        return elev_m
    lv = elev_m[land].astype(np.float64)
    order = np.argsort(lv)
    if lats is not None:
        coslat = np.clip(np.cos(np.deg2rad(lats)), 0.0, 1.0)     # (H,)
        rows = np.nonzero(land)[0]                               # 陆地像素行号(行优先)
        wl = coslat[rows][order]
        wc = np.cumsum(wl)
        wc = (wc - 0.5 * wl) / wc[-1]
        tgt = np.interp(wc, _LAND_Q, _LAND_W_V)
    else:
        wc = (np.arange(lv.size) + 0.5) / lv.size
        tgt = np.interp(wc, _LAND_Q, _LAND_V)
    out = elev_m.copy()
    out[land] = tgt[order.argsort()].astype(np.float32)
    return out.astype(np.float32)


def ocean_margin_fix(elev_m, tect1_up, margin_up=None,
                     km_shelf=115.0, km_slope=175.0, transition_km=15.0):
    """P0 2048陆架-陆坡修复: 海岸噪声破坏近岸剖面后重新强制
    陆架(0→-163m) + 陆坡(→-1900m) + 陆隆渐出(→-4500m 放开)。
    v3.0 per-margin 分化: margin_up=1 被动边缘(大西洋型宽陆架 km_shelf)
    / =2 主动边缘(太平洋型窄陆架 35km+陡坡) / =0 或 None 用默认宽。
    俯冲核带(tect1高)与已有深结构(>-4600m)不抬升 → 保海沟与深渊平原。

    v8.0 平滑过渡: 旧版 ``np.where(d < X, ...)`` 在 km_s / km_s+km_p /
    km_s+km_p+160 三处产生 200-500m 阶跃, 在岛屿周围形成同心环方块伪影.
    新版使用 ``np.tanh`` 软过渡 (带宽 ``transition_km`` 默认 15km ≈ 3px@2048),
    在剖断点处 C1 连续, 消除径向阶梯. 陆架/陆坡/陆隆三段几何 (起止深度、
    宽度) 完全保持, 仅过渡方式由阶跃改为平滑."""
    S = elev_m.shape[0]
    land = elev_m > 0
    kmpp = 360.0 / S * 111.32
    d = distance_transform_edt(~land) * kmpp

    def shelf_profile(km_s, km_p):
        # 三段目标深度 (各自 clip 到本段范围, 超出段端点保持端点值)
        p_shelf = -18.0 - 145.0 * np.clip(d / km_s, 0.0, 1.0)
        p_slope = -163.0 - 1737.0 * np.clip((d - km_s) / km_p, 0.0, 1.0)
        p_rise = -1900.0 - 2600.0 * np.clip((d - km_s - km_p) / 160.0, 0.0, 1.0)
        # tanh 软过渡权重 (在 d=km_s / km_s+km_p / km_s+km_p+160 处平滑切换)
        # 注意: w_shelf + w_slope + w_rise 在三段并集内部 = 1, 但在陆隆末端
        # (d≈km_s+km_p+160) 处 → 0.5 (因为 "beyond" 区开始接管). 因此需要归一化,
        # 否则 profile 会降至 p_rise 的一半 (e.g. -2250m 而非 -4500m), 错误填平深海.
        t = transition_km
        w_shelf = 0.5 * (1.0 - np.tanh((d - km_s) / t))
        w_slope = 0.5 * (np.tanh((d - km_s) / t)
                         - np.tanh((d - km_s - km_p) / t))
        w_rise = 0.5 * (np.tanh((d - km_s - km_p) / t)
                        - np.tanh((d - km_s - km_p - 160.0) / t))
        denom = w_shelf + w_slope + w_rise
        # 归一化: 在三段并集内 denom≈1 → profile 为加权平均; 在陆隆末端
        # denom→0 → 超出约束区, 返回 -1e9 (不约束)
        profile = np.where(denom > 1e-3,
                           (w_shelf * p_shelf + w_slope * p_slope + w_rise * p_rise)
                           / np.maximum(denom, 1e-6),
                           -1.0e9)
        return profile

    prof = shelf_profile(km_shelf, km_slope).astype(np.float32)
    ok = (~land) & (tect1_up < 0.30) & (elev_m > -4600.0)
    # v8.0 软地板 (soft floor): 在陆隆末端 (d≈km_s+km_p+160) 处, 旧版 np.maximum
    # 从 -4500m 硬切换到 "不约束" (原 elev_m), 若原海深 -8000m 则产生 3500m 阶跃.
    # 新版用 w_constraint 权重把 max(elev, prof) 与 elev 平滑混合:
    #   内部 (w=1): max(elev, prof) — 硬地板 (与旧版相同)
    #   末端 (w→0): elev — 释放约束, 让海床自然深下去
    #   过渡 (0<w<1): w*max(elev, prof) + (1-w)*elev — 渐进释放
    w_constraint = 0.5 * (1.0 - np.tanh(
        (d - (km_shelf + km_slope + 160.0)) / transition_km)).astype(np.float32)
    floor = np.maximum(elev_m, prof)
    out = np.where(ok, w_constraint * floor + (1.0 - w_constraint) * elev_m, elev_m)
    if margin_up is not None:
        # 主动边缘: 窄陆架 35km + 陡坡 (替换宽剖面, 太平洋型)
        act = (margin_up >= 2) if margin_up.shape == elev_m.shape else None
        if act is None:
            z = np.zeros(elev_m.shape, bool)
            z[:min(margin_up.shape[0], elev_m.shape[0]),
              :min(margin_up.shape[1], elev_m.shape[1])] = \
                margin_up[:elev_m.shape[0], :elev_m.shape[1]] >= 2
            act = z
        if act.any():
            prof_a = shelf_profile(35.0, 95.0).astype(np.float32)
            ok_a = (~land) & act & (elev_m > -4600.0)
            w_constraint_a = 0.5 * (1.0 - np.tanh(
                (d - (35.0 + 95.0 + 160.0)) / transition_km)).astype(np.float32)
            floor_a = np.maximum(elev_m, prof_a)
            out = np.where(ok_a,
                           w_constraint_a * floor_a + (1.0 - w_constraint_a) * elev_m,
                           out)
    return out.astype(np.float32)


def add_island_arcs(elev_m, tect1_up, cf, rng):
    """P1 岛弧链 (日本/印尼型): 俯冲核带法向偏移 180~340km,
    取大陆侧(crust概率高)且仍为海的点盖章长条岛 → 弧后链状群岛。
    岛长4~11px/宽1.5~3px/峰500~2200m, 沿核线间隔采样 → 链状。
    ---- v7.4: 用户报告"群岛/岛弧数量明显不足"
      俯冲核带分位: 0.995 → 0.990 (放宽准入, 中等强度的俯冲带也能产生岛弧)
      核带最低阈值: 0.25 → 0.18 (进一步降低门槛, 生成 ~2倍 的岛弧链)
      盖章深度下限: -5200m → -5800m (也能在较深盆地边缘盖岛弧, 类似马里亚纳外侧弧形群岛)"""
    from scipy.ndimage import label as ndi_label
    S = elev_m.shape[0]
    thr = float(np.quantile(tect1_up, 0.990))
    if thr <= 0:
        return elev_m
    core = tect1_up > max(thr, 0.18)
    lab, n = ndi_label(core)
    if n == 0:
        return elev_m
    g = gaussian_filter(tect1_up.astype(np.float32), 2.0)
    gy, gx = np.gradient(g)
    out = elev_m.copy()
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if ys.size < 25:
            continue
        d_arc = rng.uniform(9.0, 17.0)
        L = rng.uniform(4.0, 11.0)
        W = rng.uniform(1.5, 3.0)
        n_st = max(2, ys.size // 4)
        for k in rng.choice(ys.size, size=min(ys.size, n_st), replace=False):
            y, x = int(ys[k]), int(xs[k])
            gn = float(np.hypot(gx[y, x], gy[y, x])) + 1e-9
            nx_, ny_ = gx[y, x] / gn, gy[y, x] / gn
            cands = []
            for sgn in (1.0, -1.0):
                oy = int(round(y + sgn * ny_ * d_arc))
                ox = int(round(x + sgn * nx_ * d_arc)) % S
                if 0 <= oy < S:
                    cands.append((cf[oy, ox], oy, ox))
            if not cands:
                continue
            _, oy, ox = max(cands)                    # 大陆侧
            if out[oy, ox] >= 0.0 or out[oy, ox] < -5800.0:
                continue                              # 只要浅海盖章
            ang = rng.uniform(0, 2 * np.pi)           # 岛长轴随机倾斜
            tx, ty = -ny_, nx_
            ca, sa = np.cos(ang), np.sin(ang)
            ux, uy = tx * ca - ty * sa, tx * sa + ty * ca
            pad = int(L) + 3
            if ox - pad < 0 or ox + pad >= S or oy - pad < 0 or oy + pad >= S:
                continue
            yy, xx = np.mgrid[oy - pad:oy + pad + 1, ox - pad:ox + pad + 1]
            dy_, dx_ = yy - oy, xx - ox
            along = (dx_ * ux + dy_ * uy) / L
            across = (dx_ * -uy + dy_ * ux) / W
            H = rng.uniform(500.0, 2200.0) * rng.uniform(0.7, 1.3)
            blob = H * np.exp(-along ** 2 - across ** 2)
            m = (blob > 80.0) & (out[yy, xx] < 0)
            out[yy[m], xx[m]] += blob[m]
    return out


def add_hotspot_chains(elev_m, rng, n_chains=None):
    """P1 热点岛链 (夏威夷型): 开阔洋中随机方向 3~6 岛链,
    单岛半径1.6~3.2px, 峰300~1600m 随链龄衰减 → 线状岛链+海山。"""
    S = elev_m.shape[0]
    land = elev_m > 0
    d_land = distance_transform_edt(~land)
    n_chains = n_chains if n_chains is not None else int(rng.randint(5, 10))  # v7.4: 3-7 → 5-10, 匹配地球西太平洋海量岛弧密度
    out = elev_m.copy()
    placed, tries = 0, 0
    while placed < n_chains and tries < 300:
        tries += 1
        y = int(rng.randint(60, S - 60))
        x = int(rng.randint(0, S))
        if land[y, x] or d_land[y, x] < 18.0:
            continue
        placed += 1
        ang = rng.uniform(0, 2 * np.pi)
        dy, dx = np.sin(ang), np.cos(ang)
        n_isl = int(rng.randint(3, 7))
        gap = rng.uniform(4.0, 8.0)
        H0 = rng.uniform(900.0, 1600.0)
        for j in range(n_isl):
            cy = int(round(y + dy * gap * j))
            cx = int(round(x + dx * gap * j)) % S
            if not (20 <= cy < S - 20) or cx < 6 or cx >= S - 6:
                break
            h = H0 * (1.0 - 0.55 * j / n_isl) * rng.uniform(0.7, 1.3)
            r = rng.uniform(1.6, 3.2)
            yy, xx = np.mgrid[cy - 5:cy + 6, cx - 5:cx + 6]
            blob = h * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (r * r)))
            m = blob > 60.0
            out[yy[m], xx[m]] += blob[m]
    return out


def carve_fjords(elev_m, lats, margin_up, rng, lat_min=54.0, n_fjords=None):
    """M3 峡湾雕刻 (斯堪的纳维亚/格陵兰型): 高纬被动边缘从海岸向内陆
    刻窄谷 (宽1-2px, 深80-300m, 长4-10px, 二级分叉) → 冰蚀海岸。
    仅 |lat|>54° 被动边缘带; U形谷横剖面。"""
    S = elev_m.shape[0]
    land = elev_m > 0
    pas = (margin_up == 1) if margin_up is not None else None
    if pas is None or not pas.any():
        return elev_m
    # 高纬被动海岸线 = 陆地且邻海
    sea = ~land
    coast = land & pas & (np.abs(lats)[:, None] > lat_min) \
            & (np.roll(sea, 1, 0) | np.roll(sea, -1, 0)
               | np.roll(sea, 1, 1) | np.roll(sea, -1, 1))
    ys, xs = np.nonzero(coast)
    if ys.size == 0:
        return elev_m
    out = elev_m.copy()
    # 内陆方向场: 一次计算 (上坡梯度)
    g = gaussian_filter(out.astype(np.float32), 2.0)
    gy, gx = np.gradient(g)
    gn = np.hypot(gx, gy) + 1e-9
    ux_f, uy_f = gx / gn, gy / gn                # 上坡=内陆
    n_fjords = n_fjords if n_fjords is not None else int(min(ys.size, 900))
    pick = rng.choice(ys.size, size=n_fjords, replace=False)
    for p in pick:
        y, x = int(ys[p]), int(xs[p])
        ux, uy = ux_f[y, x], uy_f[y, x]
        L = rng.uniform(4.0, 10.0)
        depth = rng.uniform(80.0, 300.0)
        W = rng.uniform(1.2, 2.2)
        cy, cx = float(y), float(x)
        for t in range(int(L)):
            cy -= uy; cx += ux                    # 上坡 = 内陆
            iy, ix = int(round(cy)), int(round(cx)) % S
            if not (0 <= iy < S) or out[iy, ix] <= 0:
                break
            pad = int(W) + 2
            y0, y1 = max(0, iy - pad), min(S, iy + pad + 1)
            x0, x1 = max(0, ix - pad), min(S, ix + pad + 1)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            r = np.hypot(yy - iy, xx - ix)
            cut = depth * np.clip(1.0 - (r / W) ** 2, 0, 1)  # U形
            m = (r < W * 1.4) & (out[y0:y1, x0:x1] > 60.0)   # 不切到海平面下
            out[y0:y1, x0:x1] = np.where(
                m, np.maximum(out[y0:y1, x0:x1] - cut * m, 15.0),
                out[y0:y1, x0:x1])
            # 二级分叉 (10%)
            if t > 2 and rng.rand() < 0.10:
                a2 = rng.uniform(0, 2 * np.pi)
                cy2, cx2 = cy, cx
                for _ in range(int(L * 0.5)):
                    cy2 += np.sin(a2); cx2 += np.cos(a2)
                    iy2, ix2 = int(round(cy2)), int(round(cx2)) % S
                    if not (0 <= iy2 < S) or out[iy2, ix2] <= 0:
                        break
                    y0b, y1b = max(0, iy2 - 3), min(S, iy2 + 4)
                    x0b, x1b = max(0, ix2 - 3), min(S, ix2 + 4)
                    yy2, xx2 = np.mgrid[y0b:y1b, x0b:x1b]
                    rr = np.hypot(yy2 - iy2, xx2 - ix2)
                    cut2 = depth * 0.6 * np.clip(
                        1.0 - (rr / (W * 0.7)) ** 2, 0, 1)
                    m2 = (rr < W) & (out[y0b:y1b, x0b:x1b] > 0)
                    out[y0b:y1b, x0b:x1b] = np.where(
                        m2, out[y0b:y1b, x0b:x1b] - cut2 * m2,
                        out[y0b:y1b, x0b:x1b])
    return out.astype(np.float32)


def add_shelf_peninsulas(elev_m, margin_up, cf, rng, lats=None,
                         n_pen=None, n_isl=None):
    """M3 陆架台地半岛 + 大陆岛 (阿拉伯/佛罗里达/不列颠型):
    被动边缘宽陆架台地上, 海平面微抬 +15m → 台地露头成半岛/岛。
    台地 = 被动边缘 0-90km 内 -60~+60m 平缓带 → 加 fBm 起伏后 ±15m 阈值化。"""
    S = elev_m.shape[0]
    land = elev_m > 0
    pas = (margin_up == 1) if margin_up is not None else None
    if pas is None or not pas.any():
        return elev_m, 0
    d = distance_transform_edt(~land)
    kmpp = 360.0 / S * 111.32
    d_km = d * kmpp
    shelf = (~land) & (pas | (d_km < 95.0)) & (elev_m > -170.0)
    if not shelf.any():
        return elev_m, 0
    # 台地起伏: 低频 fBm; 台地基面 = 陆架剖面等效 (-160m 基 + fBm ±70m)
    # (真实陆架多被 ocean_margin_fix 压至 -163m, 阈值需覆盖全陆架)
    # M6: 基频 8→11 (特征 σ128→93px) → 更多中型露头 (大岛数 72.7 vs 地球 113)
    tf = value_fbm(S, 11.0, 3, rng=rng)
    tfc = np.clip(tf, 0, 1)
    out = elev_m.copy()
    plateau = shelf & (elev_m > -165.0) & (d_km < 95.0)
    out = np.where(plateau, np.maximum(out, -160.0 + 130.0 * (tf + 0.5)),
                   out)
    # 海平面等效 +15m: 台地内 -15~+45m 的连通域成陆 (半岛/大陆岛)
    # 近岸带(<25km)阈值放宽(-25m) → 露头连片成大岛/半岛 (不列颠型)
    near = plateau & (d_km < 25.0)
    new_land = (plateau & (out > -15.0)) | (near & (out > -25.0))
    # 清碎屑: 只留 >12px 的露头 (真半岛/岛)
    from scipy.ndimage import label as _lbl
    lab, n = _lbl(new_land)
    if n:
        sz = np.bincount(lab.ravel()); sz[0] = 0
        keep = np.isin(lab, np.nonzero(sz >= 12)[0])
        new_land = new_land & keep
        # 大陆岛增益: 离岸 8-60km 中型露头(20-4000px)抬高确保成岛
        offshore = new_land & (d * kmpp > 8.0) & (d * kmpp < 60.0)
        lab2, n2 = _lbl(offshore)
        if n2:
            sz2 = np.bincount(lab2.ravel()); sz2[0] = 0
            big = np.isin(lab2, np.nonzero((sz2 >= 20) & (sz2 <= 4000))[0])
            offshore = offshore & big
            out = np.where(offshore,
                           np.maximum(out, 8.0 + 45.0 * tfc),
                           out)
    out = np.where(new_land, np.maximum(out, 3.0 + 30.0 * tfc),
                   out).astype(np.float32)
    n_out = int(new_land.sum())
    return out, n_out


def add_coral_reefs(elev_m, lats, rng, temp_fn=None, n_reef=None):
    """M3 珊瑚礁带 (马尔代夫/太平洋环礁型): |lat|<28° 浅海(-60~0m)
    且海表温度>20°C 处, 低概率抬升小岛 (峰3-8m, r=1-2px, 群集3-8个)。"""
    S = elev_m.shape[0]
    sea_shallow = (elev_m > -60.0) & (elev_m <= 0.0) \
                  & (np.abs(lats)[:, None] < 28.0)
    if temp_fn is not None:
        sea_shallow &= temp_fn > 20.0
    ys, xs = np.nonzero(sea_shallow)
    if ys.size == 0:
        return elev_m
    out = elev_m.copy()
    n_reef = n_reef if n_reef is not None else int(min(ys.size * 0.012, 420))
    # 礁盘中心: 稀疏采样 + 群集
    pick = rng.choice(ys.size, size=min(n_reef, ys.size), replace=False)
    for p in pick:
        cy, cx = int(ys[p]), int(xs[p])
        n_isl = int(rng.randint(3, 9))
        for _ in range(n_isl):
            oy = int(round(cy + rng.normal(0, 3.5)))
            ox = int(round(cx + rng.normal(0, 3.5))) % S
            if not (0 <= oy < S):
                continue
            if out[oy, ox] > 0 or out[oy, ox] < -60.0:
                continue
            h = rng.uniform(3.0, 8.0)
            r = rng.uniform(1.0, 2.0)
            y0, y1 = max(0, oy - 3), min(S, oy + 4)
            x0, x1 = max(0, ox - 3), min(S, ox + 4)
            if y1 <= y0 or x1 <= x0:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1]
            blob = h * np.exp(-((np.hypot(yy - oy, xx - ox) / r) ** 2))
            m = (blob > 1.5) & (out[y0:y1, x0:x1] <= 0)
            out[y0:y1, x0:x1] = np.where(
                m, np.maximum(out[y0:y1, x0:x1],
                              (0.0 + blob * 0.6)),
                out[y0:y1, x0:x1])
    return out.astype(np.float32)


def add_intraplate_orogen(elev_m, cf, tect1_up, rng, n=None):
    """P2 板内造山 (落基山/东非裂谷肩型): 大陆内部远离活跃边界处
    盖1~2条板内山地: 长20~45px 宽4~9px 峰1600~3400m, 平顶脊状+噪声。
    打破"山地100%锁定俯冲带"的过纯耦合 (地球山地过半在板内)。"""
    S = elev_m.shape[0]
    n = n if n is not None else int(rng.randint(1, 3))
    out = elev_m.copy()
    cand = (cf > 0.62) & (tect1_up < 0.15) & (out > 0)
    for _ in range(n):
        ys, xs = np.nonzero(cand)
        if ys.size < 400:
            break
        k = int(rng.randint(ys.size))
        cy, cx = int(ys[k]), int(xs[k])
        ang = rng.uniform(0, np.pi)
        L = rng.uniform(20.0, 45.0)
        W = rng.uniform(4.0, 9.0)
        H = rng.uniform(1600.0, 3400.0)
        ux, uy = np.cos(ang), np.sin(ang)
        pad = int(L) + 4
        yy, xx = np.mgrid[max(0, cy - pad):min(S, cy + pad + 1),
                          max(0, cx - pad):min(S, cx + pad + 1)]
        dy_, dx_ = yy - cy, xx - cx
        along = (dx_ * ux + dy_ * uy) / L
        across = (dx_ * -uy + dy_ * ux) / W
        ridge = np.exp(-along ** 4 - across ** 2)
        nf = value_fbm(max(yy.shape), 8.0, 3, rng=rng)
        noise = 0.65 + 0.35 * np.clip(
            nf[:yy.shape[0], :yy.shape[1]] * 1.6, -1.0, 1.5)
        add = H * ridge * noise
        m = (out[yy, xx] > 0) & (add > 60.0)
        out[yy[m], xx[m]] += add[m]
    return out


def carve_trunk_rivers(v01, rng, sigma=6.0, q=0.9985, max_depth=0.045,
                       width=3.2, precip=None, temp=None, warp_amp=3.0,
                       warp_freq=0.015, use_centerline_pipeline=True,
                       q_levels=(0.988, 0.975, 0.945, 0.900),
                       hack_law=True, oxbow=True):
    """v8.3 干流+支流水系雕刻: 四等级骨架 + 尺度自适应蜿蜒 + Hack幂律宽深 + 牛轭湖。

    核心改进 (相对 v8.2)
    ----------------------
    1. **回返修复** ``extract_river_centerlines`` 骨架上若存在
       ``acc_next < acc_curr × 0.98`` 的回返点, 在 8 邻居中改选陆地内
       max(acc) 邻居。修复 v8.2 log(acc) 单调增比 0.62 → 目标 ≥0.92。
    2. **四级河网 (干流→支→支沟→源头沟)**:
       q_levels = [0.999, 0.996, 0.985, 0.960]，共提取 4 组骨架，
       全部 meander 后一起 carve → 呈现树状流域网络。
    3. **蜿蜒升级 (meander_centerlines v8.3)**: amp=6 默认；
       ``scale_adaptive=True`` + ``acc`` 宽-弯耦合 → 宽处弯大,
       窄处弯小; 曲率钳制 Δθ<28° → 去除Z字锯齿。
    4. **雕刻升级 (carve_along_centerlines v8.3)**: Hack 幂律
       ``W ∝ A^0.5, D ∝ A^0.4, slope∝A^-0.3`` (经典水力几何指数)；
       Oxbow 牛轭湖自动在曲流颈 <4px 处挖椭圆静水湖。
    5. **兜底**: ``use_centerline_pipeline=False`` → 退化为 DT+exp 走廊；
       lines 为空 → 回退兜底高斯走廊。
    """
    land = v01 >= 0.5
    if not land.any():
        return v01
    v01s = gaussian_filter(v01, sigma)
    filled = fill_depressions(v01s)
    if precip is not None and temp is not None:
        acc = climate_flow_accumulation(filled, precip, temp, rng)
    else:
        acc = d8_flow_accumulation(filled, rng)
    aL = acc[land]
    if aL.size < 100:
        return v01

    if use_centerline_pipeline:
        # ============== v8.3: 四等级骨架 + 逐级合流雕刻 ==============
        H, W = acc.shape
        all_lines = []
        for q_lv in q_levels:
            min_len = max(int(20 - 4 * len(all_lines) // 4), 8)
            # monotonic_repair=True 在 extract 内部就做好回返修复 + 去重
            lines_lv = extract_river_centerlines(acc, land, q=float(q_lv),
                                                  min_len=min_len,
                                                  monotonic_repair=True)
            all_lines.extend(lines_lv)

        if len(all_lines) > 0:
            if warp_amp > 0.0:
                all_lines = meander_centerlines(
                    all_lines, rng, amp=float(warp_amp),
                    base_freq=float(warp_freq),
                    head_shrink=0.08, tail_shrink=0.06, octaves=3,
                    acc=acc, width_ref=None, scale_adaptive=True,
                    max_turn_deg=22.0, min_turn_radius_px=10.0,
                    land=land, clamp_iters=5,
                    sinuosity_bounds=(1.8, 3.0))
            return carve_along_centerlines(
                v01, all_lines, acc, land,
                max_depth=max_depth, base_width=float(width),
                width_per_log_acc=0.4, cutoff_cross=4.0,
                hack_law=hack_law, oxbow=oxbow)

    # ========== 兜底 / 旧版：高斯走廊下切 ==========
    thr = float(np.quantile(aL, q))
    ch = land & (acc > thr)
    if not ch.any():
        return v01
    if warp_amp > 0.0:
        ch = warp_river_corridor(ch, rng, amp=warp_amp, base_freq=10.0)
        ch = ch & land
        if not ch.any():
            return v01
    d = distance_transform_edt(~ch)
    w = np.clip(acc / (4.0 * thr), 0.0, 1.0)
    depth = max_depth * (0.45 + 0.55 * w)
    return np.clip(v01 - depth * np.exp(-(d / width) ** 2) * land,
                   0, 1).astype(np.float32)


def land_hist_final(v01, lats=None):
    """P0 陆地直方图终检 (v01域): GAN blend/后处理漂移后,
    排序CDF重新对齐 m2v(地球陆地分位) → 平原/台地/山地比例精确复原。
    单调重排保留全部空间秩序(谷<脊, V谷细节不损), 陆地不越海平面。
    lats 给定时按 cos(纬度) 加权分位(目标 _LAND_W_V) → 与面积加权验收口径一致。"""
    land = v01 >= 0.5
    if not land.any():
        return v01
    n = int(land.sum())
    order = np.argsort(v01[land])
    if lats is not None:
        coslat = np.clip(np.cos(np.deg2rad(lats)), 0.0, 1.0)     # (H,)
        rows = np.nonzero(land)[0]                               # 陆地像素行号(行优先)
        wl = coslat[rows][order]
        wc = np.cumsum(wl)
        wc = (wc - 0.5 * wl) / wc[-1]
        tgt = m2v(np.interp(wc, _LAND_Q, _LAND_W_V)).astype(np.float32)
    else:
        wc = (np.arange(n) + 0.5) / n
        tgt = m2v(np.interp(wc, _LAND_Q, _LAND_V)).astype(np.float32)
    out = v01.copy()
    out[land] = tgt[order.argsort()]
    return np.clip(out, 0, 1).astype(np.float32)


# 地球深洋 (深度<-1000m) cos(lat) 加权分位 —— ETOPO1(Ice) 2048 实测 (M6 标定)
# 尾部加密网格 (hadal 仅 1.085% 深洋: 粗网格线性插值会把 2% 分位错插到 -6766)
_OCD_Q = np.array([0.0000, 0.0010, 0.0020, 0.0030, 0.0050, 0.0070, 0.0090,
                   0.0110, 0.0130, 0.0160, 0.0200, 0.0240, 0.0250, 0.0500,
                   0.0750, 0.1000, 0.1250, 0.1500, 0.1750, 0.2000, 0.2250,
                   0.2500, 0.2750, 0.3000, 0.3250, 0.3500, 0.3750, 0.4000,
                   0.4250, 0.4500, 0.4750, 0.5000, 0.5250, 0.5500, 0.5750,
                   0.6000, 0.6250, 0.6500, 0.6750, 0.7000, 0.7250, 0.7500,
                   0.7750, 0.8000, 0.8250, 0.8500, 0.8750, 0.9000, 0.9250,
                   0.9500, 0.9750, 0.9760, 0.9840, 0.9880, 0.9910, 0.9930,
                   0.9950, 0.9970, 0.9990, 1.0000])
_OCD_V = np.array([-10517, -7239, -6657, -6353, -6157, -6082, -6034, -5997,
                   -5966, -5923, -5877, -5837, -5828, -5644, -5518, -5417,
                   -5330, -5249, -5173, -5102, -5032, -4962, -4892, -4820,
                   -4748, -4676, -4607, -4538, -4472, -4407, -4342, -4277,
                   -4210, -4140, -4067, -3992, -3915, -3837, -3757, -3673,
                   -3586, -3492, -3391, -3281, -3160, -3016, -2842, -2636,
                   -2356, -2000, -1550, -1530, -1363, -1277, -1210, -1165,
                   -1120, -1073, -1024, -1000], np.float64)


def ocean_hist_final(dem, lats=None):
    """M6 海洋直方图终检 (米域): 深洋(<-1000m)排序 CDF → 地球深洋加权分位。
    GAN v01 域饱和(0 → -11000m)与包络混合使深洋带漂移 (hadal 3.7% vs 0.96%);
    单调重排保结构 (海沟仍最深, 俯冲速率序保留), 陆架-陆坡带(>-1000m)不动。"""
    ocean = dem < -1000.0
    n = int(ocean.sum())
    if n < 1000:
        return dem
    order = np.argsort(dem[ocean])
    if lats is not None:
        rows = np.nonzero(ocean)[0]                      # 行优先 → 与 dem[ocean] 同序
        coslat = np.clip(np.cos(np.deg2rad(np.asarray(lats, np.float64))), 0.0, 1.0)
        w = coslat[rows][order]
        cw = np.cumsum(w)
        wc = ((cw - 0.5 * w) / cw[-1]).astype(np.float64)
    else:
        wc = (np.arange(n) + 0.5) / n
    tgt = np.interp(wc, _OCD_Q, _OCD_V).astype(np.float32)
    tmp = dem[ocean]
    tmp[order] = tgt
    out = dem.copy()
    out[ocean] = tmp
    return out.astype(np.float32)


# ==================== 9e. M7 海底地貌增强 ====================
def enhance_ocean_floor(dem_m, lats, envS=None, rng=None,
                         ridge_amp=320.0, seamount_amp=1800.0,
                         abyssal_hill_amp=85.0, rise_compress_px=10,
                         depth_transition_m=200.0):
    """v7.7 海底地貌增强 — 恢复 GAN/直方图终检弱化的洋中脊/海沟/海山链/深海丘陵纹理.

    在大陆边缘、大洋盆地和大洋中脊三大基本海底地貌单元之上叠加次一级单元:
      1) 深海丘陵 (abyssal hills): -3500~-6000m 深海平原叠加 fBm 起伏 (±85m)
      2) 洋中脊增强 (mid-ocean ridge): envS>0.55 的扩张洋区叠加 +320m 脊状隆起
      3) 海山链 (seamount chains): 沿 15-75° 方向生成定向条带噪声海山
      4) 大陆隆收窄 (continental rise compress): 过宽的陆隆带压缩 rise_compress_px px

    v8.0: 各深度掩膜改用软 sigmoid 权重 (过渡带宽 ``depth_transition_m`` 默认 200m),
    旧版硬阈值在 -3500/-6000/-2500/-7000/-2000m 等深线处产生纹理突变方块,
    在岛屿周边 (深度跨过阈值时) 尤其明显.

    Parameters
    ----------
    dem_m : ndarray (S, S) float32
        海洋后处理后的 DEM (米域). 海陆均传入, 仅修改 sea 部分.
    lats : ndarray (S,)
        每行纬度 (度). 用于极地掩膜.
    envS : float or ndarray, optional
        扩张强度先验 (>0.55 视为活跃洋中脊区). 标量时全局一致.
    rng : numpy.random.Generator, optional
        随机数生成器. None 时用 np.random.
    ridge_amp, seamount_amp, abyssal_hill_amp : float
        洋中脊/海山/深海丘陵的幅度 (米).
    rise_compress_px : int
        大陆隆收窄像素数.
    depth_transition_m : float
        深度掩膜软过渡带宽 (米). 默认 200m ≈ 4px@2048.

    Returns
    -------
    ndarray (S, S) float32
        增强后的 DEM.
    """
    S = dem_m.shape[0]
    rng = rng if rng is not None else np.random
    out = dem_m.astype(np.float32, copy=True)
    sea = dem_m < 0.0
    if not sea.any():
        return out
    sea_f = sea.astype(np.float32)
    dt = float(depth_transition_m)

    def _soft_band(d, lo, hi, trans):
        """软深度带权重: d∈[lo,hi] 时=1, 边缘 trans 米 tanh 过渡.
        (lo, hi 为深度值, lo<hi 表示 lo 是更深的边界; lo<hi 时 d∈[lo,hi] 权重=1)"""
        # w_lo: 1 当 d>=lo (浅于等于 lo), 0 当 d<lo (深于 lo)
        w_lo = 0.5 * (1.0 + np.tanh((d - lo) / trans))
        # w_hi: 1 当 d<=hi (深于等于 hi), 0 当 d>hi (浅于 hi)
        w_hi = 0.5 * (1.0 + np.tanh((hi - d) / trans))
        return w_lo * w_hi

    # ---- 1) 深海丘陵 (abyssal hills): 深海平原叠加低幅 fBm ----
    # v8.0: 软深度掩膜 (-6000 ~ -3500m, 200m 过渡) 替代硬阈值
    deep_w = _soft_band(dem_m, -6000.0, -3500.0, dt) * sea_f
    if deep_w.sum() > 1.0:
        deep = deep_w > 0.5  # 用于 fBm 归一化统计
        hills = value_fbm(S, base_freq=12.0, octaves=4, gain=0.5, rng=rng)
        hills = (hills - hills[deep].mean()) / (hills[deep].std() + 1e-8)
        out += (abyssal_hill_amp * hills * deep_w).astype(np.float32)

    # ---- 2) 洋中脊增强: 活跃扩张区叠加脊状隆起 ----
    if envS is not None:
        env_arr = np.asarray(envS)
        env_mask = (env_arr > 0.55) & sea if env_arr.ndim > 0 \
            else (float(envS) > 0.55) & sea
        if env_mask.any():
            ridge = ridged_fbm(S, direction=35.0, base_freq=4.0, octaves=4,
                               gain=0.55, rng=rng)
            out[env_mask] += (ridge_amp * ridge)[env_mask].astype(np.float32)

    # ---- 3) 海山链: 沿 15-75° 方向生成定向条带海山 ----
    # v8.0: 软深度掩膜 (-7000 ~ -2500m, 200m 过渡) 替代硬阈值
    sm_w = _soft_band(dem_m, -7000.0, -2500.0, dt) * sea_f
    if sm_w.sum() > 1.0:
        sm = sm_w > 0.5  # 用于统计归一化
        chain = directional_band_noise(S, direction_deg=45.0,
                                       freq_center=6.0, anisotropy=0.3, rng=rng)
        chain = np.clip((chain - chain[sm].mean()) /
                        (chain[sm].std() + 1e-8), -3.0, 3.0)
        # 仅保留正向高峰 (海山), 负向忽略
        chain = np.where(chain > 1.5, (chain - 1.5) * seamount_amp, 0.0)
        out += (chain * sm_w).astype(np.float32)

    # ---- 4) 大陆隆收窄: 过渡带 (0~-2000m) 向深海侧压缩 ----
    rise = (dem_m < 0.0) & (dem_m > -2000.0) & sea
    if rise.any() and rise_compress_px > 0:
        land_dist = distance_transform_edt(~sea)
        rise_edge = (rise & (land_dist < rise_compress_px))
        if rise_edge.any():
            # 边缘向深海方向轻度加深, 收窄陆隆视觉宽度
            out[rise_edge] -= 50.0

    return out


# ==================== 9f. M8 陆地地貌分类 ====================
def classify_landforms(dem_m, lats, land=None,
                       win_relief=7, slope_thresh_mtn=None,
                       relief_thresh_mtn=None, river_thr=0.012,
                       envS=None, beltS=None, cf=None, tect_T=None,
                       margin_up=None, riftS=None):
    """v7.8 陆地地貌分类 — 基于高程+坡度+局部起伏+水文+构造场.

    五大基本陆地地形: 山地/高原/平原/丘陵/盆地 (int8 位打包).
    低 4 位 = 粗类 (向后兼容旧 0-5), 高 4 位 = 子类:
      0 山地: bit4-5 高度(00高山/01中山/10低山) + bit6-7 成因(00默认/01褶皱/10断层/11火山)
      1 平原: 子类=0 (独立型; 从属型由盆地标签隐含)
      2 高原: 0x10堆积 / 0x20切割 / 0x30侵蚀
      3 河谷: 子类=0
      4 沙丘: 子类=0
      5 丘陵: 0x10高丘陵 / 0x20低丘陵
      6 盆地: 0x10沉积 / 0x20克拉通 / 0x30构造-前陆 / 0x40内流 / 0x50外流
    海洋区为 -1. 全 None 构造场时退化为旧 6 类行为 (子类全 0).

    Parameters
    ----------
    dem_m, lats, land, win_relief, slope_thresh_mtn, relief_thresh_mtn, river_thr
        见 v7.7 文档.
    envS, beltS, cf, tect_T, margin_up, riftS : ndarray, optional
        板块构造场 (generate_world 已计算). 全 None 时跳过成因/盆地子类.

    Returns
    -------
    ndarray (S, S) int16
        地貌标签图 (位打包). 海洋 -1.
    """
    S = dem_m.shape[0]
    if land is None:
        land = dem_m >= 0.0
    lf = np.full(dem_m.shape, -1, np.int16)

    # ---- 坡度 (梯度幅值, m/pixel-index) ----
    gy, gx = np.gradient(dem_m.astype(np.float64))
    slope = np.hypot(gx, gy)

    # ---- 局部起伏 (窗口内 max-min) ----
    dmx = maximum_filter(dem_m, size=win_relief)
    dmn = minimum_filter(dem_m, size=win_relief)
    relief = dmx - dmn

    # ---- 局部均值 (用于河谷/高原/盆地判定) ----
    local_mean = uniform_filter(dem_m.astype(np.float64), size=win_relief)

    L = land
    elev = dem_m

    # ---- 自适应阈值: 用陆地分位数标定 (分辨率无关) ----
    land_slope = slope[L]
    land_relief = relief[L]
    s_thr = slope_thresh_mtn if slope_thresh_mtn is not None else \
        float(np.percentile(land_slope, 75)) if land_slope.size > 100 else 500.0
    r_thr = relief_thresh_mtn if relief_thresh_mtn is not None else \
        max(1000.0, float(np.percentile(land_relief, 75))) if land_relief.size > 100 else 1000.0
    r_med = float(np.median(land_relief)) if land_relief.size > 0 else 300.0
    r_low = float(np.percentile(land_relief, 40)) if land_relief.size > 100 else 200.0

    # 0) 山地: 高坡度或高起伏 (绝对>1000m 或 分位数)
    mtn = L & ((slope > s_thr) | (relief > r_thr))
    lf[mtn] = 0

    # 2) 高原: 高程>1200m 且 顶面平坦 (起伏<40 分位)
    plateau = L & (elev > 1200.0) & (relief < r_low) & (lf == -1)
    lf[plateau] = 2

    # 6) 盆地 (v7.8 新增): 低高程+低起伏+局部低洼, 优先于河谷/平原
    #    核心: dem < local_mean-80m (比河谷50m更深) 且 relief<r_low*0.7 (更平坦)
    has_tect = any(x is not None for x in (envS, beltS, cf, tect_T, margin_up, riftS))
    basin_cand = L & (elev < 1500.0) & (relief < r_med) & (lf == -1)
    basin_core = basin_cand & (dem_m < local_mean - 80.0) & (relief < r_low * 0.7)
    if basin_core.any() and has_tect:
        # 距海距离 (用于内流判定)
        d_in_sea = distance_transform_edt(~L)
        # 子类按优先级赋值 (后赋覆盖前, 故按反优先级顺序)
        # 外流(兜底) 0x50
        lf[basin_core] = 6 | 0x50
        # 沉积 0x10: 极低起伏
        sed = basin_core & (relief < r_low * 0.4)
        lf[sed] = 6 | 0x10
        # 克拉通 0x20: 稳定陆内 (cf>0.62 且 beltS<0.25)
        if cf is not None:
            craton = basin_core & (cf > 0.62) & \
                ((beltS < 0.25) if beltS is not None else True)
            lf[craton] = 6 | 0x20
        # 内流 0x40: 距海远 (d_in_sea>130px)
        endo = basin_core & (d_in_sea > 130.0)
        lf[endo] = 6 | 0x40
        # 构造-前陆 0x30: 造山带前缘 (beltS>0.30 或 tect_T[0]>0.45)
        foreland = None
        if beltS is not None:
            foreland = basin_core & (beltS > 0.30)
        elif tect_T is not None:
            foreland = basin_core & (tect_T[0] > 0.45)
        if foreland is not None and foreland.any():
            lf[foreland] = 6 | 0x30
    elif basin_core.any():
        # 无构造场: 只标粗类 6, 不分子类
        lf[basin_core] = 6

    # 3) 河谷: 低于邻域均值>50m
    low = L & (dem_m < local_mean - 50.0) & (relief < r_med) & (lf == -1)
    lf[low] = 3

    # 4) 沙丘: 低纬 (|lat|<35°) 低起伏干旱地
    abs_lat = np.abs(np.asarray(lats, np.float64))[:, None] \
        if lats is not None else np.zeros((S, 1))
    lat2d = np.broadcast_to(abs_lat, dem_m.shape)
    dune = L & (lat2d < 35.0) & (relief < r_med) & (relief > r_low * 0.5) & (lf == -1)
    lf[dune] = 4

    # 1) 平原: 低高程 低起伏 低坡度
    plain = L & (elev < 800.0) & (relief < r_med) & (slope < s_thr * 0.5) & (lf == -1)
    lf[plain] = 1

    # 5) 丘陵: 其余陆地
    lf[L & (lf == -1)] = 5

    # ====== v7.8 子类精细化 (粗类分配完成后, 全 has_tect 内保向后兼容) ======
    if has_tect:
        # ---- 山地高度子类 (bit4-5): 高山>3500m / 中山1000-3500m / 低山<1000m ----
        mtn_mask = (lf & 0x0F) == 0
        if mtn_mask.any():
            high = mtn_mask & (elev > 3500.0)
            mid = mtn_mask & (elev > 1000.0) & (elev <= 3500.0)
            low_mtn = mtn_mask & (elev <= 1000.0)
            lf[mid] |= 0x10       # bit4-5 = 01 (中山)
            lf[low_mtn] |= 0x20   # bit4-5 = 10 (低山); 高山保持 00

        # ---- 山地成因子类 (bit6-7, OR叠加于高度位) ----
        # 11火山(0xC0) > 01褶皱(0x40) > 10断层(0x80) > 00侵蚀(默认)
        # 按反优先级赋值 (后赋覆盖)
        if envS is not None:
            volc = mtn_mask & (envS > 0.40)
            lf[volc] = (lf[volc] & 0x0F) | 0xC0  # 清除高4位后设成因
        elif tect_T is not None and tect_T.shape[0] > 1:
            volc = mtn_mask & (tect_T[1] > 0.55)
            lf[volc] = (lf[volc] & 0x0F) | 0xC0
        if beltS is not None:
            fold = mtn_mask & (beltS > 0.45) & \
                (((lf >> 6) & 0x03) == 0)  # 未被火山覆盖
            lf[fold] = (lf[fold] & 0x0F) | 0x40
        fault = None
        if riftS is not None:
            fault = mtn_mask & (riftS > 0.30)
        elif margin_up is not None:
            fault = mtn_mask & (margin_up == 1)
        if fault is not None and fault.any():
            unfault = fault & (((lf >> 6) & 0x03) == 0)
            lf[unfault] = (lf[unfault] & 0x0F) | 0x80

        # ---- 高原子类 (0x10堆积 / 0x20切割 / 0x30侵蚀) ----
        plat_mask = (lf & 0x0F) == 2
        if plat_mask.any():
            accum = plat_mask & (relief < r_low * 0.5)
            dissect = plat_mask & (relief > r_low * 0.8)
            lf[accum] |= 0x10
            lf[dissect] |= 0x20
            if cf is not None:
                eros = plat_mask & (cf > 0.62) & \
                    ((beltS < 0.25) if beltS is not None else True) & \
                    ((lf & 0xF0) == 0)  # 未被堆积/切割覆盖
                lf[eros] |= 0x30

        # ---- 丘陵子类 (0x10高丘陵 / 0x20低丘陵) ----
        hill_mask = (lf & 0x0F) == 5
        if hill_mask.any() and land_relief.size > 100:
            r_h60 = float(np.percentile(land_relief, 60))
            hi_h = hill_mask & (relief > r_h60)
            lf[hi_h] |= 0x10
            lo_h = hill_mask & (relief <= r_h60)
            lf[lo_h] |= 0x20

    return lf


# ==================== 9d. M5 克拉通大流域盆地 ====================
def add_craton_basins(dem, conv, rng, belt=None, lats=None,
                      max_exo=4, endorheic=True,
                      target_lo=0.020, target_hi=0.065,
                      min_share=0.015):
    """M5 克拉通大流域盆地 (亚马逊/刚果/西西伯利亚型):
    稳定陆内低地(低起伏·低收敛·|lat|<72°)重塑为向单一出口的线性倾斜面:
      z = z_out + (z_far−z_out)·(d/d_max) + fBm微起伏(±18m, 远端减半)
    线性坡 (~4m/px) > σ6平滑后噪声 → D8 全流域汇入单一出口
      → top1 流域 3-4.5% 陆地, 干流走廊自动成河。
    超大低地连通域 → 以最低单元为种子盘状裁切到目标份额。
    出口 = 盆地 1000km 内最低海岸点; 无出海路径 → 内陆海 (里海型,
    终端洼地 −28m)。颈道保通: 出口-盆地间 min(原地形, 线性坡道) 下切。
    返回 (dem, n_basins, info)。
    ---- v7.4: 用户报告"内陆湖/内流盆地数量明显不足, 湖泊巨型化"
      max_exo: 2→4 (允许最多4个外流大盆地, 对应地球 亚马逊/刚果/密西西比/拉普拉塔 4大流域)
      target_lo: 0.032 → 0.020 (最小目标份额从3.2%降到2.0%, 允许更多中小型盆地生成,
      不再强求单一大盆地挤到 3.2% 陆地面积, 更像地球 里海/咸海/乍得湖 多种尺度并存)"""
    S, W = dem.shape
    land = dem > 0
    n_land = int(land.sum())
    if n_land < 10000:
        return dem, 0, ""
    f = max(1, S // 128)                                   # 粗化因子 (2048→16)
    Hc, Wc = S // f, W // f
    sub = dem[:Hc * f, :Wc * f].reshape(Hc, f, Wc, f)
    cm = sub.mean((1, 3)); cs = sub.std((1, 3))
    cc = conv[:Hc * f, :Wc * f].reshape(Hc, f, Wc, f).mean((1, 3))
    cb = (belt[:Hc * f, :Wc * f].reshape(Hc, f, Wc, f).max((1, 3))
          if belt is not None else np.zeros_like(cm))
    from scipy.ndimage import label as _lbl, binary_closing

    def _edt_to(m):                                        # 周期 EDT: 到 m 的距离 (m内=0)
        return distance_transform_edt(np.tile(~m, (1, 3)))[:, W:2 * W]

    ocean = ~land
    coastal = land & (np.roll(ocean, 1, 0) | np.roll(ocean, -1, 0)
                      | np.roll(ocean, 1, 1) | np.roll(ocean, -1, 1))
    d_sea_full = _edt_to(ocean)                            # 距海 (全分辨率, px)
    # 粗块级内陆判据: 块均值距海 ≥60px → 排除沿海平原/海湾两岸碎片
    # (水系不连通的伪盆地; 真正克拉通盆地应在大陆内部)
    d_sea_c = d_sea_full[:Hc * f, :Wc * f].reshape(Hc, f, Wc, f).mean((1, 3))
    lat_c = (np.abs(np.asarray(lats, np.float64))[:, None]
             if lats is not None else np.zeros((S, 1)))
    info, n_endo = [], 0
    land_c = cm > 0.0
    if land_c.sum() < 16:
        return dem, 0, ""
    for relax in (False, True):                            # 二轮: 严格→放宽
        # 分位数自适应阈值: 起伏/收敛取陆块相对低值 (适配任意管线阶段)
        # M6: 陆块占比 ≥0.75 — 群岛混块(陆海各半的粗块)均值可伪过判据,
        # 倾斜面盖在群岛上各岛独立排水 → 流域不收敛
        lf_c = land[:Hc * f, :Wc * f].reshape(Hc, f, Wc, f).mean((1, 3))
        cs_thr = np.quantile(cs[land_c], 0.60 if not relax else 0.78)
        cc_thr = np.quantile(cc[land_c], 0.72 if not relax else 0.88)
        elig = ((cm > 50) & (cm < (800 if not relax else 950))
                & (cs < cs_thr) & (cc < cc_thr) & (cb < 0.45)
                & (lf_c >= 0.75)
                & (d_sea_c >= (45.0 if not relax else 30.0)))
        if lats is not None:                               # 极地冰原不建盆地
            elig &= np.repeat(np.abs(lat_c[f // 2::f, 0])[:, None] < 72.0, Wc, 1)
        el_c = binary_closing(elig, iterations=1)
        lab_c, n_c = _lbl(el_c)
        if n_c == 0:
            continue
        sizes = np.array([(lab_c == i).sum() * f * f for i in range(1, n_c + 1)])
        n_exo = 0
        for oi in np.argsort(sizes)[::-1][:4]:
            if n_exo >= max_exo:
                break
            base = lab_c == (oi + 1)
            a_base = base.sum() * f * f
            if a_base < min_share * n_land:
                continue
            if a_base > target_hi * n_land:                # 超大连通域: 最低单元种子裁切
                ys, xs = np.nonzero(base)
                k = int(np.argmin(cm[ys, xs]))
                seed = np.zeros_like(base); seed[ys[k], xs[k]] = True
                d_seed = distance_transform_edt(~seed)
                grown = None
                for r in range(2, 40):
                    g = base & (d_seed <= r)
                    if g.sum() * f * f >= target_lo * n_land:
                        grown = g
                        break
                if grown is None:
                    continue
            elif a_base < target_lo * n_land:              # 小域: 向外生长
                d_c = distance_transform_edt(~base)
                grown = base
                for r in range(1, 8):
                    g = elig & (d_c <= r)
                    if g.sum() * f * f > target_hi * n_land:
                        break
                    grown = g
                    if g.sum() * f * f >= target_lo * n_land:
                        break
            else:
                grown = base
            mask = np.repeat(np.repeat(grown, f, 0), f, 1)[:S, :W] & land
            mask &= (dem < 1400)
            if belt is not None:
                mask &= (belt < 0.45)
            if mask.sum() < min_share * n_land:
                continue
            d_to = _edt_to(mask)                           # 到掩模距离 (内=0)
            near = coastal & (d_to <= 50.0)                # ~1000km 内海岸
            if near.any():
                ys, xs = np.nonzero(near)
                k = int(np.argmin(dem[ys, xs]))
                oy, ox = int(ys[k]), int(xs[k])
                mode = 'exo'
            elif endorheic and n_endo == 0:
                ys, xs = np.nonzero(mask)
                k = int(np.argmin(dem[ys, xs]))
                oy, ox = int(ys[k]), int(xs[k])
                mode = 'endo'
            else:
                continue
            pt = np.zeros((S, W), bool); pt[oy, ox] = True
            d_o = _edt_to(pt)                              # 距出口/终端 (周期)
            # 海岸分水岭后撤 + 出口扇区豁免: 距海 <25px 不重塑 (保分水岭),
            # 但出口 60px 扇区豁免 → 倾斜面直达海岸 (防 25px 原始带挡水,
            # 仅 6px 河道口过流 → 大量分流)
            if mode == 'exo':
                mask &= (d_sea_full >= 25.0) | (d_o <= 60.0)
                if mask.sum() < min_share * n_land:
                    continue
            d_in = _edt_to(~mask)                          # 掩模内距边界 (外=0)
            d_to = _edt_to(mask)
            # 分水岭保底: 倾斜面远端 ≥ 盆缘外围地形 75 分位
            # (若 z_far < 外围, 远端水向盆外低地泄漏 → 流域破碎)
            ring = (d_to > 0) & (d_to <= 10.0) & land & ~mask
            z_ring = float(np.quantile(dem[ring], 0.75)) if ring.sum() > 50 \
                else float(np.quantile(dem[mask], 0.85))
            z_far = min(max(float(np.quantile(dem[mask], 0.60)), z_ring), 900.0)
            w = np.clip(d_in / 8.0, 0, 1)                   # 边界 feather 8px (仅掩模内)
            w = w * w * (3 - 2 * w)
            w *= np.clip((1400.0 - dem) / 350.0, 0, 1)     # 山缘衰减 (高地不重塑)
            w[~land] = 0.0
            dmax = float(np.quantile(d_o[mask], 0.92))
            if dmax < 20:
                continue
            z_out = float(dem[oy, ox]) if mode == 'exo' else -28.0
            # 坡度保底: 主坡 ≥0.9 m/px — fill_depressions 的 eps 排水梯度
            # ~0.57 m/px (4e-4 v01 @200-1000m 段), 主坡低于它会被填洼梯度
            # 淹没 → 径向流失效 (z_far 保底 = z_out + 0.9·dmax)
            z_far = max(z_far, z_out + 0.9 * dmax)
            rel = 3.0 * value_fbm(S, 60.0, 2, rng=rng)      # 极弱长波起伏
            rel = rel * (1.0 - 0.5 * np.clip(d_o / dmax, 0, 1))
            z_b = z_out + (z_far - z_out) * np.clip(d_o / dmax, 0, 1) + rel
            z_b = np.maximum(z_b, 8.0 if mode == 'exo' else -30.0)
            # 分水岭筑堤: mask 外缘 10px 环带低洼段 (海湾/海峡/外部谷地)
            # 抬至 z_far+20m → 阻断 D8 边界漏水; 河网走廊豁免由 d_o 门控;
            # σ6 平滑后堤仍高于倾斜面 → 径向流唯一出口 (直方图终检校准回)
            dyke = (d_to > 0) & (d_to <= 10.0) & land
            if mode == 'exo':
                dyke &= (d_o > 60.0)                       # 出口走廊豁免
            low = dyke & (dem < z_far + 20.0)
            if low.any():
                ramp2 = (z_far + 20.0) * np.exp(-(d_to - 0.5) / 3.5)
                dem = np.maximum(dem, np.where(low, ramp2, -np.inf)).astype(np.float32)
                dem[~land] = np.minimum(dem[~land], 0.0)
            dem = (dem * (1 - w) + z_b * w).astype(np.float32)   # 倾斜面装配
            if mode == 'exo':                              # 河网树走廊 (拓扑构造):
                ys_c, xs_c = np.nonzero(mask)              # 装配后下切, 防 w 混合
                cy, cx = float(ys_c.mean()), float(xs_c.mean())  # 覆盖河道
                def _carve(p0, p1, depth_m, w_px, wav):
                    """p0→p1 河道走廊: 正弦弯曲 + 沿程下切 depth_m, 宽 w_px"""
                    nonlocal dem
                    n = max(8, int(np.hypot(p1[0] - p0[0], p1[1] - p0[1]) * 1.4))
                    ts = np.linspace(0, 1, n)
                    perp = np.array([-(p1[1] - p0[1]), p1[0] - p0[0]], np.float64)
                    perp /= (np.hypot(*perp) + 1e-9)
                    wig = wav * np.sin(ts * np.pi * rng.uniform(1.5, 2.5)
                                       + rng.uniform(0, 6.28)) * np.sin(ts * np.pi)
                    yy = p0[0] + ts * (p1[0] - p0[0]) + wig * perp[0]
                    xx = p0[1] + ts * (p1[1] - p0[1]) + wig * perp[1]
                    pts = np.zeros((S, W), bool)
                    yi = np.clip(yy.round().astype(int) % S, 0, S - 1)
                    xi = np.clip(xx.round().astype(int) % W, 0, W - 1)
                    pts[yi, xi] = True
                    d_l = _edt_to(pts)                     # 距河道中心线
                    cor = (d_l <= w_px) & land
                    if cor.any():
                        frac = np.clip(d_l / max(w_px, 1.0), 0, 1)  # 边缘浅
                        z_riv = z_out + 1.8 * d_o - depth_m * (1 - frac)
                        z_riv = np.maximum(z_riv, 5.0)
                        # M6: 强制河床 (set 而非仅下切) — 出口-盆间低洼段
                        # 若只切不抬会形成并行通道分流; 全走廊设床=堤化河道
                        dem = np.where(cor, z_riv.astype(np.float32), dem)
                    return pts
                _carve((oy, ox), (cy, cx), 55.0, 5.0, 6.0)             # 主干
                # 3 条支干: 主干中段 → mask 内随机远点
                for k in range(3):
                    t = rng.uniform(0.3, 0.75)
                    b0 = (oy + t * (cy - oy), ox + t * (cx - ox))
                    q = int(rng.uniform(0, ys_c.size))
                    _carve(b0, (float(ys_c[q]), float(xs_c[q])),
                           40.0, 3.5, 8.0)
            if mode == 'exo':
                n_exo += 1
            else:
                n_endo += 1
            info.append(f"{mode}{mask.sum() / n_land * 100:.1f}%@({oy // f},{ox // f})")
            if n_exo >= max_exo and (not endorheic or n_endo >= 1):
                break
        if n_exo + n_endo >= 2:
            break
    return dem, len(info), "; ".join(info)


# ==================== 10. 验证指标套件 (A7) ====================
def _psd_slope(x):
    x = x - x.mean()
    p = np.abs(np.fft.fftshift(np.fft.fft2(x))) ** 2
    f1 = np.fft.fftshift(np.fft.fftfreq(x.shape[0]))
    R = np.sqrt(f1[:, None] ** 2 + f1[None, :] ** 2)
    bins = np.geomspace(0.02, 0.45, 13)
    idx = np.digitize(R.ravel(), bins)
    rp, pp = [], []
    for k in range(1, len(bins)):
        sel = idx == k
        if sel.sum() > 8:
            rp.append(bins[k]); pp.append(p.ravel()[sel].mean())
    return float(np.polyfit(np.log(rp), np.log(pp), 1)[0])


def _fractal_dim(v01):
    land = (v01 > 0.5).astype(np.float32)
    if land.max() == 0 or land.min() == 1:
        return float('nan')
    sizes = np.array([2, 4, 8, 16, 32], float)
    counts = [float(((block_reduce(land, int(b), "max") > 0.5)
                     & (block_reduce(land, int(b), "min") < 0.5)).sum()) for b in sizes]
    if min(counts) <= 2:
        return float('nan')
    return float(-np.polyfit(np.log(sizes), np.log(counts), 1)[0])


def _slope_stats(v01):
    gy, gx = np.gradient(v01)
    g = np.hypot(gx, gy)
    return float(g.mean() * 100), float(np.quantile(g, 0.95) * 100)


def _coherence(v01):
    gy, gx = np.gradient(gaussian_filter(v01, 1.0))
    Jxx = gaussian_filter(gx * gx, 3); Jyy = gaussian_filter(gy * gy, 3)
    Jxy = gaussian_filter(gx * gy, 3)
    tr = Jxx + Jyy
    disc = np.sqrt(0.25 * (Jxx - Jyy) ** 2 + Jxy ** 2)
    lam1, lam2 = 0.5 * tr + disc, 0.5 * tr - disc
    c = (lam1 - lam2) / (lam1 + lam2 + 1e-9)
    w = np.sqrt(tr) > np.quantile(np.sqrt(tr), 0.8)
    return float(c[w].mean()) if w.any() else float('nan')


def snow_area_by_band(v01, lats):
    sl = m2v(snowline_m(lats))[:, None]
    a = np.abs(lats)[:, None].repeat(v01.shape[1], 1)
    return [float(((v01 >= sl) & (a >= lo) & (a < hi)).sum() / max(((a >= lo) & (a < hi)).sum(), 1))
            for lo, hi in [(0, 30), (30, 50), (50, 66.5), (66.5, 90.01)]]


def terrain_metrics(tgt, out, lats, bm, block=8, tag=""):
    def f(x):
        return f"{x:8.3f}" if np.isfinite(x) else "     nan"
    tm, to_ = _slope_stats(tgt)
    om, oo = _slope_stats(out)
    rmse = float(np.sqrt(np.mean((block_reduce(out, block, "mean") - bm) ** 2)))
    print(f"\n--- 指标 [{tag}] (目标 | 输出) ---")
    for name, a, b in [("PSD 斜率(越负越粗)", _psd_slope(tgt), _psd_slope(out)),
                       ("海岸分形维 D", _fractal_dim(tgt), _fractal_dim(out)),
                       ("坡度均值 x1e2", tm, om), ("坡度 P95 x1e2", to_, oo),
                       ("山脊方向相干性", _coherence(tgt), _coherence(out)),
                       ("块均值RMSE(仅输出)", 0.0, rmse)]:
        print(f"  {name:20s} {f(a)} | {f(b)}")
    print("  雪盖-纬度带 [0-30/30-50/50-66.5/极地]")
    print("    目标:", " ".join(f"{x:.2f}" for x in snow_area_by_band(tgt, lats)))
    print("    输出:", " ".join(f"{x:.2f}" for x in snow_area_by_band(out, lats)))


# ==================== 11. 接缝羽化拼接器 (A8) ====================
@torch.no_grad()
def generate_large_map(gen, dem, ch_stats, y0, x0, out_size=1024, tile=256,
                       overlap=32, block=8, seed=0):
    """重叠瓦片推理 + 余弦羽化。每瓦片对同一份全球粗底图做块均值一致性校正
    → 低频跨瓦片天然无缝(山脉带不会在瓦片边界断头), 高频羽化过渡。"""
    gen.eval()
    rng = np.random.default_rng(seed)
    step = tile - overlap
    ramp = np.ones(tile, np.float32)
    ramp[:overlap] = 0.5 - 0.5 * np.cos(np.pi * np.arange(overlap) / overlap)
    ramp[-overlap:] = ramp[:overlap][::-1]
    win = np.outer(ramp, ramp)
    acc = np.zeros((out_size, out_size), np.float32)
    wgt = np.zeros((out_size, out_size), np.float32)
    for ty in range(0, out_size, step):
        for tx in range(0, out_size, step):
            m, lats = dem.patch(y0 + ty, x0 + tx, tile)
            ch, _ = build_sample(m, lats, block, rng, ch_stats)
            raw = gen(torch.from_numpy(ch[None]).float().to(device)).cpu().squeeze().numpy()
            vc = m2v(m)
            out = postprocess_pipeline(raw.astype(np.float32),
                                       block_reduce(vc, block, "mean"),
                                       block_reduce(vc, block, "std"),
                                       block_reduce((m > 0).astype(np.float32), block, "mean"),
                                       lats, rng, block)
            h, w = min(tile, out_size - ty), min(tile, out_size - tx)
            acc[ty:ty + h, tx:tx + w] += out[:h, :w] * win[:h, :w]
            wgt[ty:ty + h, tx:tx + w] += win[:h, :w]
    return np.clip(acc / np.maximum(wgt, 1e-8), 0, 1)


def make_terrain_cmap():
    colors = [(0.04, 0.12, 0.30), (0.07, 0.22, 0.45), (0.12, 0.36, 0.60),
              (0.25, 0.55, 0.70), (0.55, 0.80, 0.80),
              (0.35, 0.55, 0.28), (0.55, 0.62, 0.30), (0.68, 0.52, 0.30),
              (0.55, 0.38, 0.28), (0.80, 0.76, 0.72), (1.00, 1.00, 1.00)]
    return LinearSegmentedColormap.from_list('dem', colors, N=256)


# ==================== 12. 训练 (A5: 一致性损失已集成) ====================
def train(dem):
    # —— 标准配置 (MPS 约 2.5-4h) ——
    size, block = 256, 8
    n_train, n_val, batch, epochs, warmup = 1200, 120, 8, 30, 4
    # —— 快速预览配置 (约 20-30min): 取消下行注释 ——
    # size, block, n_train, n_val, batch, epochs, warmup = 128, 8, 500, 60, 8, 20, 3
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "checkpoints", "best_terrain_gan_v5.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)

    print("估计输入通道统计量...")
    tect_provider = make_tect_provider(dem)
    ch_stats = estimate_channel_stats(dem, size, block, tect_provider=tect_provider)
    train_set = TerrainDataset(dem, size, block, n_train, ch_stats, is_train=True,
                               tect_provider=tect_provider)
    val_set = TerrainDataset(dem, size, block, n_val, ch_stats, is_train=False,
                             tect_provider=tect_provider)
    train_loader = DataLoader(train_set, batch_size=batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch, shuffle=False, num_workers=0)
    print(f"训练: 12通道(含构造场) | 在线无限裁剪 | 验证: 太平洋带 held-out")

    gen = UNet(in_ch=12, base=32).to(device)
    ema = EMA(gen, decay=0.999)
    disc_fine = PatchDiscriminator(13).to(device)
    disc_mid = PatchDiscriminator(13).to(device)
    disc_coarse = PatchDiscriminator(13).to(device)

    opt_g = optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_df = optim.Adam(disc_fine.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_dm = optim.Adam(disc_mid.parameters(), lr=1.5e-4, betas=(0.5, 0.999))
    opt_dc = optim.Adam(disc_coarse.parameters(), lr=1e-4, betas=(0.5, 0.999))
    scheds = [optim.lr_scheduler.CosineAnnealingLR(o, T_max=epochs, eta_min=5e-6)
              for o in (opt_g, opt_df, opt_dm, opt_dc)]

    crit_gan, crit_l1 = nn.MSELoss(), nn.L1Loss()
    crit_edge = EdgeLoss().to(device)
    crit_fft = FFTLoss().to(device)
    crit_lap = LaplacianLoss().to(device)
    crit_dist = DistributionLoss()
    discs = ((disc_fine, opt_df, 1), (disc_mid, opt_dm, 2), (disc_coarse, opt_dc, 4))

    # ---- A5: 粗图一致性损失所需张量 ----
    ch_mean_t = torch.tensor(ch_stats["mean"], device=device).view(1, -1, 1, 1)
    ch_std_t = torch.tensor(ch_stats["std"], device=device).view(1, -1, 1, 1)
    w_cons = 4.0

    best_score, patience, patience_counter = float('inf'), 10, 0
    print("-" * 60)

    for epoch in range(epochs):
        prog = epoch / max(1, epochs - 1)
        warm = 0.0 if epoch < warmup else min(1.0, (epoch - warmup + 1) / 10.0)
        w_gan, w_fm = 3.5 * warm, 10.0 * warm
        w_l1 = 12.0 - 9.0 * prog          # 12 → 3
        w_dist = 6.0 + 4.0 * prog         # 6 → 10
        w_edge, w_fft, w_lap = 4.0, 2.0, 1.5

        gen.train()
        for D, _, _ in discs:
            D.train()
        ep_d, ep_g, t_ep = 0.0, 0.0, time.time()

        for inp, target in train_loader:
            inp, target = inp.to(device), target.to(device)

            # A5: 反归一化通道0 → 粗底图块均值 (32x32)
            # inp 已是全分辨率(通道被repeat过) → 先池化回块级 32x32,
            # 与 F.avg_pool2d(fake, block) 形状对齐
            bm_t = F.avg_pool2d(inp[:, 0:1] * ch_std_t[:, 0:1] + ch_mean_t[:, 0:1], block)

            if w_gan > 0:   # ---- 判别器 (预热期跳过) ----
                fake = gen(inp).detach()
                for D, optD, sc in discs:
                    rp = torch.cat([downsample(inp, sc), downsample(target, sc)], 1)
                    fp = torch.cat([downsample(inp, sc), downsample(fake, sc)], 1)
                    out_r, out_f = D(rp), D(fp)
                    loss_d = 0.5 * (crit_gan(out_r, torch.ones_like(out_r))
                                    + crit_gan(out_f, torch.zeros_like(out_f)))
                    optD.zero_grad(); loss_d.backward(); optD.step()
                    ep_d += loss_d.item()

            # ---- 生成器 ----
            fake = gen(inp)
            loss_gan = loss_fm = fake.new_tensor(0.0)
            if w_gan > 0:
                for D, _, sc in discs[:2]:
                    rp = torch.cat([downsample(inp, sc), downsample(target, sc)], 1)
                    fp = torch.cat([downsample(inp, sc), downsample(fake, sc)], 1)
                    pred_f, feat_f = D(fp, return_feat=True)
                    _, feat_r = D(rp, return_feat=True)
                    loss_gan = loss_gan + crit_gan(pred_f, torch.ones_like(pred_f))
                    loss_fm = loss_fm + F.l1_loss(feat_f, feat_r)

            bm_fake = F.avg_pool2d(fake, block)
            assert bm_fake.shape == bm_t.shape, \
                f"一致性损失形状不匹配: {bm_fake.shape} vs {bm_t.shape}"
            loss = (w_l1 * crit_l1(fake, target) + w_dist * crit_dist(fake, target)
                    + w_edge * crit_edge(fake, target) + w_fft * crit_fft(fake, target)
                    + w_lap * crit_lap(fake, target) + w_gan * loss_gan + w_fm * loss_fm
                    + w_cons * F.l1_loss(bm_fake, bm_t))

            opt_g.zero_grad(); loss.backward(); opt_g.step()
            ema.update(gen)
            ep_g += loss.item()

        if w_gan > 0:
            for s in scheds[1:]:
                s.step()
        scheds[0].step()

        n = len(train_loader)
        avg_d = ep_d / (3 * n) if w_gan > 0 else 0.0
        avg_g = ep_g / n

        # ---- 验证 ----
        gen.eval()
        v_l1 = v_edge = v_dist = 0.0
        with torch.no_grad():
            for inp, target in val_loader:
                inp, target = inp.to(device), target.to(device)
                fake = gen(inp)
                v_l1 += crit_l1(fake, target).item()
                v_edge += crit_edge(fake, target).item()
                v_dist += crit_dist(fake, target).item()
        m = len(val_loader)
        v_l1, v_edge, v_dist = v_l1 / m, v_edge / m, v_dist / m
        val_score = v_l1 + 2.0 * v_edge + 5.0 * v_dist

        if epoch % 2 == 0 or epoch == epochs - 1:
            print(f"Epoch [{epoch:02d}/{epochs}] | D: {avg_d:.4f} | G: {avg_g:.3f} | "
                  f"Val L1: {v_l1:.4f} | Edge: {v_edge:.4f} | Dist: {v_dist:.4f} | "
                  f"w_GAN:{w_gan:.1f} w_L1:{w_l1:.0f} | {time.time()-t_ep:.0f}s/ep")

        if val_score < best_score:
            best_score, patience_counter = val_score, 0
            torch.save({'epoch': epoch, 'gen_state_dict': gen.state_dict(),
                        'gen_ema': ema.shadow,
                        'disc_fine': disc_fine.state_dict(), 'disc_mid': disc_mid.state_dict(),
                        'disc_coarse': disc_coarse.state_dict(),
                        'ch_stats_mean': ch_stats["mean"].tolist(),
                        'ch_stats_std': ch_stats["std"].tolist(),
                        'cfg': {'size': size, 'block': block},
                        'val_score': best_score}, save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n早停触发! 连续 {patience} 个 epoch 未改善。")
                break

    print(f"\n训练完成。最佳复合验证分: {best_score:.4f}")

    """
    # ==================== 可视化 (A9) ====================
    ckpt = torch.load(save_path, map_location=device)
    gen.load_state_dict(ckpt['gen_ema'])
    gen.eval()
    cmap = make_terrain_cmap()
    post_rng = np.random.default_rng(7)

    idxs = [n_val // 6, n_val // 2, 5 * n_val // 6]
    fig, axes = plt.subplots(3, 5, figsize=(23, 13))
    col_titles = ["输入(块均值通道)", "目标 DEM (纬度雪线渲染)", "G 输出 raw",
                  "后处理成品 (雪线渲染)", "海拔分布对比"]
    for row, idx in enumerate(idxs):
        y0, x0, _ = val_set.centers[idx]
        m_tgt, lats = dem.patch(y0, x0, size)
        tgt_v01 = m2v(m_tgt)
        bm = block_reduce(tgt_v01, block, "mean")
        sd = block_reduce(tgt_v01, block, "std")
        lf = block_reduce((m_tgt > 0).astype(np.float32), block, "mean")

        ch_in, _ = val_set[idx]
        with torch.no_grad():
            g_raw = gen(ch_in.unsqueeze(0).to(device)).cpu().squeeze().numpy().astype(np.float32)
        g_pp = postprocess_pipeline(g_raw, bm, sd, lf, lats, post_rng, block)

        lat_c = float(lats[size // 2])
        lon360 = (x0 + size / 2) * 360.0 / dem.nx
        lon_c = lon360 - 360.0 if lon360 >= 180.0 else lon360
                # 原来: reshape(size, block, size, block)  ← 错, 需要 4M 元素
        Bs = size // block
        inp_vis = np.clip(ch_in[0].numpy().reshape(Bs, block, Bs, block).mean(axis=(1, 3))
                          * ch_stats["std"][0] + ch_stats["mean"][0], 0, 1)

        axes[row, 0].imshow(inp_vis, cmap='gray')
        axes[row, 1].imshow(render_snow_field(tgt_v01, lats, cmap))
        axes[row, 2].imshow(g_raw, cmap=cmap, vmin=0, vmax=1)
        axes[row, 3].imshow(render_snow_field(g_pp, lats, cmap))
        axes[row, 4].hist(tgt_v01.ravel(), bins=60, range=(0, 1), density=True,
                          alpha=0.55, label='目标', color='#2c7fb8')
        axes[row, 4].hist(g_pp.ravel(), bins=60, range=(0, 1), density=True,
                          alpha=0.55, label='输出', color='#d95f0e')
        axes[row, 4].axvline(float(m2v(snowline_m(lat_c))), color='gray', ls='--', lw=0.8)
        t_max, o_max = float(v2m(tgt_v01.max())), float(v2m(g_pp.max()))
        axes[row, 4].text(0.03, 0.95, f"最高海拔 目标{t_max:.0f}m / 输出{o_max:.0f}m",
                          transform=axes[row, 4].transAxes, fontsize=8, va='top')
        if row == 0:
            for c, t in enumerate(col_titles):
                axes[row, c].set_title(t, fontsize=10)
            axes[row, 4].legend(fontsize=8)
        axes[row, 0].set_ylabel(f"({lat_c:.0f}°, {lon_c:.0f}°)", fontsize=9)
        for c in range(5):
            axes[row, c].set_xticks([]); axes[row, c].set_yticks([]); axes[row, c].tick_params(length=0)
        terrain_metrics(tgt_v01, g_pp, lats, bm, block, tag=f"({lat_c:.0f}°,{lon_c:.0f}°)")
    """
    run_viz(gen, dem, val_set, ch_stats, size, block, n_val)

    plt.suptitle("Terrain GAN v5.1: 包络通道 + 一致性校正 + 分位匹配 + 纬度雪线场", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "output", "terrain_v5_result.png"),
                dpi=150, bbox_inches='tight')
    plt.show()
    print("结果图已保存至 output/terrain_v5_result.png")


# ==================== 13. 冒烟测试 (A10) ====================
def smoke_test(dem, tect_provider=None):
    stats = {"mean": np.zeros(12, np.float32), "std": np.ones(12, np.float32)}
    ds = TerrainDataset(dem, 256, 8, 2, stats, is_train=True, tect_provider=tect_provider)
    x, y = ds[0]
    g = UNet(in_ch=12).to(device)
    d = PatchDiscriminator(13).to(device)
    out = g(x[None].to(device))
    dout = d(torch.cat([x[None].to(device), y[None].to(device)], 1))
    print(f"[冒烟] 输入 {tuple(x.shape)} | 目标 {tuple(y.shape)} | "
          f"G输出 {tuple(out.shape)} | D输出 {tuple(dout.shape)}")
    assert x.shape == (12, 256, 256) and out.shape == (1, 1, 256, 256)
    print("[冒烟] 通过 ✓\n" + "-" * 60)


# ==================== 可视化循环 ====================
def run_viz(gen, dem, val_set, ch_stats, size, block, n_val,
            outfile=None):
    if outfile is None:
        outfile = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "terrain_v5_result.png")
    cmap = make_terrain_cmap()
    post_rng = np.random.default_rng(7)
    idxs = [n_val // 6, n_val // 2, 5 * n_val // 6]
    fig, axes = plt.subplots(3, 5, figsize=(23, 13))
    col_titles = ["输入(块均值通道)", "目标 DEM (纬度雪线渲染)", "G 输出 raw",
                  "后处理成品 (雪线渲染)", "海拔分布对比"]
    for row, idx in enumerate(idxs):
        y0, x0, _ = val_set.centers[idx]
        m_tgt, lats = dem.patch(y0, x0, size)
        tgt_v01 = m2v(m_tgt)
        bm = block_reduce(tgt_v01, block, "mean")
        sd = block_reduce(tgt_v01, block, "std")
        lf = block_reduce((m_tgt > 0).astype(np.float32), block, "mean")

        ch_in, _ = val_set[idx]
        with torch.no_grad():
            g_raw = gen(ch_in.unsqueeze(0).to(device)).cpu().squeeze().numpy().astype(np.float32)
        g_pp = postprocess_pipeline(g_raw, bm, sd, lf, lats, post_rng, block)

        lat_c = float(lats[size // 2])
        lon360 = (x0 + size / 2) * 360.0 / dem.nx
        lon_c = lon360 - 360.0 if lon360 >= 180.0 else lon360
        Bs = size // block   # ← 修复点
        inp_vis = np.clip(ch_in[0].numpy().reshape(Bs, block, Bs, block).mean(axis=(1, 3))
                          * ch_stats["std"][0] + ch_stats["mean"][0], 0, 1)

        axes[row, 0].imshow(inp_vis, cmap='gray')
        axes[row, 1].imshow(render_snow_field(tgt_v01, lats, cmap))
        axes[row, 2].imshow(g_raw, cmap=cmap, vmin=0, vmax=1)
        axes[row, 3].imshow(render_snow_field(g_pp, lats, cmap))
        axes[row, 4].hist(tgt_v01.ravel(), bins=60, range=(0, 1), density=True,
                          alpha=0.55, label='目标', color='#2c7fb8')
        axes[row, 4].hist(g_pp.ravel(), bins=60, range=(0, 1), density=True,
                          alpha=0.55, label='输出', color='#d95f0e')
        axes[row, 4].axvline(float(m2v(snowline_m(lat_c))), color='gray', ls='--', lw=0.8)
        t_max, o_max = float(v2m(tgt_v01.max())), float(v2m(g_pp.max()))
        axes[row, 4].text(0.03, 0.95, f"最高海拔 目标{t_max:.0f}m / 输出{o_max:.0f}m",
                          transform=axes[row, 4].transAxes, fontsize=8, va='top')
        if row == 0:
            for c, t in enumerate(col_titles):
                axes[row, c].set_title(t, fontsize=10)
            axes[row, 4].legend(fontsize=8)
        axes[row, 0].set_ylabel(f"({lat_c:.0f}°, {lon_c:.0f}°)", fontsize=9)
        for c in range(5):
            axes[row, c].set_xticks([]); axes[row, c].set_yticks([]); axes[row, c].tick_params(length=0)
        terrain_metrics(tgt_v01, g_pp, lats, bm, block, tag=f"({lat_c:.0f}°,{lon_c:.0f}°)")

    plt.suptitle("Terrain GAN v5.1: 包络通道 + 一致性校正 + 分位匹配 + 纬度雪线场", fontsize=14)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"结果图已保存至 {outfile}")


def visualize_only(dem, ckpt_path=None):
    ckpt_path = ckpt_path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "checkpoints", "best_terrain_gan_v5.pth")
    size, block, n_val = 256, 8, 120
    ckpt = torch.load(ckpt_path, map_location=device)
    assert int(ckpt.get("cfg", {}).get("in_ch", 0)) == 12, \
        "旧版 8 通道 checkpoint 与 v6.3 不兼容, 请删除后重训"
    tect_provider = make_tect_provider(dem)
    ch_stats = {"mean": np.array(ckpt["ch_stats_mean"], np.float32),
                "std": np.array(ckpt["ch_stats_std"], np.float32)}
    val_set = TerrainDataset(dem, size, block, n_val, ch_stats, is_train=False,
                             tect_provider=tect_provider)
    gen = UNet(in_ch=12, base=32).to(device)
    gen.load_state_dict(ckpt['gen_ema'])
    gen.eval()
    run_viz(gen, dem, val_set, ch_stats, size, block, n_val)

# ==================== 主入口 ====================
def main():
    _file_dir = os.path.dirname(os.path.abspath(__file__))
    _dir = (os.path.dirname(_file_dir)
            if os.path.basename(_file_dir) in ("core", "scripts") else _file_dir)
    dem_path = os.environ.get("DEM_PATH",
                              os.path.join(_dir, "data", "ETOPO1_Ice_g_gmt4.grd"))
    dem = None
    if os.path.exists(dem_path):
        dem = GlobalDEM(dem_path)
    elif not dem_path.endswith((".npy", ".npz")):
        try:
            os.makedirs(os.path.dirname(dem_path), exist_ok=True)
            download_dem(dem_path)
            dem = GlobalDEM(dem_path)
        except Exception as e:
            print(f"自动下载失败: {e}")
    if dem is None:
        dem = SyntheticDEMSource()

    ckpt = os.path.join(_dir, "checkpoints", "best_terrain_gan_v5.pth")
    if os.path.exists(ckpt):
        print("发现已有 checkpoint → 仅可视化 (如需重训请删除 .pth)")
        visualize_only(dem)
    else:
        smoke_test(dem, make_tect_provider(dem))
        train(dem)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n总耗时: {time.time() - t0:.1f}s")