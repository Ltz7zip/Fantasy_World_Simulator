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
                           map_coordinates, zoom as ndi_zoom)

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
    sl = m2v(snowline_m(lats))[:, None]
    polar = (np.abs(lats)[:, None] > 66.5)
    ice = v01 >= sl
    ice |= polar & (v01 >= 0.5)   # 极地陆地强制冰盖
    return ice


def render_snow_field(v01, lats, cmap):
    """每像素纬度雪线 + 极地冰盖 + 海冰 (修复 78°S 渲染成棕地)"""
    h, w = v01.shape
    sl01 = m2v(snowline_m(lats))[:, None]
    polar = (np.abs(lats)[:, None].repeat(w, 1) > 66.5)
    sea = v01 < 0.5
    snow_f = (v01 >= sl01).astype(np.float32)
    snow_f[polar & ~sea] = 1.0
    sea_ice_f = (polar & sea).astype(np.float32)
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
    # 季风 (东亚/南亚): 热带-副热带大陆东岸 (大大陆), 海岸带集中
    mon = east_open & (a >= 5.0) & (a <= 28.0) & big_cont
    if mon.any() and monP > 0:
        dm = distance_transform_edt(~mon) * kmpp
        dP += monP * np.exp(-dm / monD) * land
    # 西风带雨林 (PNW/智利南/挪威西): 中高纬西岸迎风第一山系
    if wprP > 0:
        wpr = west_open & (a >= 38.0) & (a <= 62.0)
        if wpr.any():
            dq = distance_transform_edt(~wpr) * kmpp
            dP += wprP * np.exp(-dq / wprD) * land
    return dT, dP


def temperature_field(lats, elev_m, dT=None):
    """计算近地面年均气温场（℃，逐像素）。

    公式：
        T = 27 − 0.42·|lat| − 6.5·max(elev,0)/1000 − polar(lat)

    其中各项物理依据：
        - **27°基准**：赤道海平面年均温（接近地球实测 26~28℃）。
        - **0.42·|lat|**：每纬度递减（比纯太阳角算出来的略大，吸收了海陆差异平均）。
        - **6.5·h/1000**：干绝热递减率，每升高 1000m 降 6.5℃。
        - **polar = clip((|lat|−55)/35, 0, 1)² × 22**：极地放大项——80° 海平面
          约下降 17℃ → 80°海平面 T≈−15℃，自动形成极冠积雪。
    最后叠加洋流/季风 ``dT``（海湾流+3.5℃等，为 0 或陆地外为 0）。

    Parameters
    ----------
    lats : ndarray (S,)
        纬度数组。
    elev_m : ndarray (S, W)
        米制高程（海洋负值参与海洋温度时只取 max(elev, 0)=0 即可——海面温度 = 海平面温度）。
    dT : ndarray (S, W) or None
        洋流/季风异常温度。若给定则直接相加。

    Returns
    -------
    ndarray (S, W), float64
        逐像素年均气温（℃）。
    """
    a = np.abs(np.asarray(lats, np.float64))[:, None]
    polar = np.clip((a - 55.0) / 35.0, 0.0, 1.0) ** 2 * 22.0
    T = 27.0 - 0.42 * a - 6.5 * np.maximum(elev_m, 0.0) / 1000.0 - polar
    if dT is not None:                                # M4 洋流温度异常 (暖流+3.5/寒流-1)
        T = T + dT
    return T


# 纬向基础降水 (mm/yr): ITCZ 赤道峰 → 副热带(20°~30°)干旱谷 → 温带峰 → 极地干
_P_LAT = np.array([0, 5, 10, 15, 20, 25, 30, 35, 45, 55, 65, 80, 90], np.float64)
_P_MM = np.array([2300, 2150, 1700, 1150, 700, 520, 560, 680, 850, 750, 480, 230, 130], np.float64)


def precipitation_field(lats, elev_m, rng=None, dP=None):
    """年降水 (mm/yr): 纬向基础 × 盛行风向地形抬升。
    信风/极地东风(东→西)与西风(西→东)按纬度分带扫描:
    迎风坡地形雨 → 背风雨影; 陆上水汽随路径衰减 → 大陆腹地自动成旱区。"""
    h, w = elev_m.shape
    a = np.abs(np.asarray(lats, np.float64))
    base = np.interp(a, _P_LAT, _P_MM)[:, None] * np.ones((1, w), np.float64)
    hk = np.maximum(elev_m, 0.0) / 1000.0
    ocean = (elev_m < 0.0) | (hk < 1e-3)

    def scan(step):
        cols = range(w) if step > 0 else range(w - 1, -1, -1)
        rain = np.zeros((h, w), np.float32)
        m = np.ones(h, np.float32)                    # 逐行水汽储量
        for _ in range(2):                            # 两圈扫描: 处理经度环绕
            hp = hk[:, (cols[0] - step) % w]
            for j in cols:
                hj = hk[:, j]
                oro = np.minimum(800.0 * np.maximum(hj - hp, 0.0) * m, 2600.0)
                rain[:, j] = (base[:, j] + oro) * (0.10 + 0.90 * m)
                m = np.where(ocean[:, j], 1.0,
                             m * np.exp(-oro / 1600.0) * 0.996)
                hp = hj
        return rain

    p_w, p_e = scan(+1), scan(-1)                     # 西风带 / 信风·极地东风带
    west = ((a >= 30.0) & (a <= 60.0))[:, None]
    P = np.where(west, p_w, p_e).astype(np.float32)
    P = gaussian_filter(P, 2.0)
    if dP is not None:                                # M4 洋流/季风异常 (过滤后)
        P = np.maximum(P + dP, 0.0).astype(np.float32)
    if rng is not None:
        P = P * np.exp(0.30 * value_fbm(w, 10.0, 3, rng=rng))
    return P


BIOME_NAMES = ["苔原", "北方针叶林", "温带草原", "温带落叶林", "温带雨林",
               "亚热带荒漠", "稀树草原", "热带季节林", "热带雨林"]
BIOME_COLORS = np.array([
    [0.62, 0.60, 0.55], [0.30, 0.42, 0.30], [0.72, 0.65, 0.38],
    [0.42, 0.55, 0.30], [0.24, 0.44, 0.24], [0.87, 0.76, 0.55],
    [0.79, 0.68, 0.38], [0.40, 0.56, 0.28], [0.16, 0.38, 0.18]], np.float32)


def whittaker_biome(t, p):
    """Whittaker (T, P) 查表 → 9 类生物群系索引 (顺序匹配 BIOME_COLORS)"""
    t = np.asarray(t)
    p = np.asarray(p)
    return np.select(
        [t <= -2, t <= 5,
         (t <= 20) & (p < 400), (t <= 20) & (p < 1500), t <= 20,
         p < 250, p < 1000, p < 2000],
        [0, 1, 2, 3, 4, 5, 6, 7], default=8).astype(np.int8)


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
                     km_shelf=115.0, km_slope=175.0):
    """P0 2048陆架-陆坡修复: 海岸噪声破坏近岸剖面后重新强制
    陆架(0→-163m) + 陆坡(→-1900m) + 陆隆渐出(→-4500m 放开)。
    v3.0 per-margin 分化: margin_up=1 被动边缘(大西洋型宽陆架 km_shelf)
    / =2 主动边缘(太平洋型窄陆架 35km+陡坡) / =0 或 None 用默认宽。
    俯冲核带(tect1高)与已有深结构(>-4600m)不抬升 → 保海沟与深渊平原。"""
    S = elev_m.shape[0]
    land = elev_m > 0
    kmpp = 360.0 / S * 111.32
    d = distance_transform_edt(~land) * kmpp

    def shelf_profile(km_s, km_p):
        return np.where(d < km_s, -18.0 - 145.0 * d / km_s,
                        np.where(d < km_s + km_p,
                                 -163.0 - 1737.0 * (d - km_s) / km_p,
                                 np.where(d < km_s + km_p + 160.0,
                                          -1900.0 - 2600.0 * (d - km_s - km_p) / 160.0,
                                          -1.0e9)))

    prof = shelf_profile(km_shelf, km_slope).astype(np.float32)
    ok = (~land) & (tect1_up < 0.30) & (elev_m > -4600.0)
    out = np.where(ok, np.maximum(elev_m, prof), elev_m)
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
            out = np.where(ok_a, np.maximum(elev_m, prof_a), out)
    return out.astype(np.float32)


def add_island_arcs(elev_m, tect1_up, cf, rng):
    """P1 岛弧链 (日本/印尼型): 俯冲核带法向偏移 180~340km,
    取大陆侧(crust概率高)且仍为海的点盖章长条岛 → 弧后链状群岛。
    岛长4~11px/宽1.5~3px/峰500~2200m, 沿核线间隔采样 → 链状。"""
    from scipy.ndimage import label as ndi_label
    S = elev_m.shape[0]
    thr = float(np.quantile(tect1_up, 0.995))
    if thr <= 0:
        return elev_m
    core = tect1_up > max(thr, 0.25)
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
            if out[oy, ox] >= 0.0 or out[oy, ox] < -5200.0:
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
    n_chains = n_chains if n_chains is not None else int(rng.randint(3, 7))
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
                       width=3.2):
    """P1 干流河谷: 大陆尺度平滑面 D8 流累积 → 主河道走廊高斯下切。
    平滑面消除块均值撕裂 → 干流贯通大陆内部低地与海岸,
    全分辨率河网沿走廊汇集 → 亚马逊/密西西比级大河谷。"""
    land = v01 >= 0.5
    if not land.any():
        return v01
    v01s = gaussian_filter(v01, sigma)
    filled = fill_depressions(v01s)
    acc = d8_flow_accumulation(filled, rng)
    aL = acc[land]
    if aL.size < 100:
        return v01
    thr = float(np.quantile(aL, q))
    ch = land & (acc > thr)
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


# ==================== 9d. M5 克拉通大流域盆地 ====================
def add_craton_basins(dem, conv, rng, belt=None, lats=None,
                      max_exo=2, endorheic=True,
                      target_lo=0.032, target_hi=0.065,
                      min_share=0.015):
    """M5 克拉通大流域盆地 (亚马逊/刚果/西西伯利亚型):
    稳定陆内低地(低起伏·低收敛·|lat|<72°)重塑为向单一出口的线性倾斜面:
      z = z_out + (z_far−z_out)·(d/d_max) + fBm微起伏(±18m, 远端减半)
    线性坡 (~4m/px) > σ6平滑后噪声 → D8 全流域汇入单一出口
      → top1 流域 3-4.5% 陆地, 干流走廊自动成河。
    超大低地连通域 → 以最低单元为种子盘状裁切到目标份额。
    出口 = 盆地 1000km 内最低海岸点; 无出海路径 → 内陆海 (里海型,
    终端洼地 −28m)。颈道保通: 出口-盆地间 min(原地形, 线性坡道) 下切。
    返回 (dem, n_basins, info)。"""
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