# -*- coding: utf-8 -*-
"""
plate_sim.py v3.0 —— 前向板块演化模拟器(超大陆旋回版, 450 Myr)

=============================================================
 0) 半球不对称先验: 大陆向随机"陆半球"集中 (北陆南水型)
 1) v3.0 超大陆旋回 (M1): 三阶段速度场
      聚合期 (0-150Myr):  板块朝聚合中心大圆漂移 → 超大陆
      超大陆期 (150-260): 速度衰减, 微大陆晚到撞入(印度型半岛), orog 高峰
      裂解期 (260-450):  主裂谷(260Myr)+次裂谷(340Myr)放射分离
                         → 新生大洋对称张开(拼图边缘) + 被动边缘
 2) 微大陆/地体: 4-6 块小陆块延迟出发 → 撞入半岛 / 搁浅大陆岛
 3) 裂谷: 衰减型记忆 + 陆地占比保护 → 大陆不被蚕食
 4) 沟/弧/脊/转换: 终局"现势边界"形态学(强度∝收敛速率)
 5) 造山带: 历史累积 orog 归一化 tanh
 6) v2.2 海洋直方图标定: 洋壳年龄排名 → 地球开阔洋深度分位 (OC_Q/OC_V)
    + 陆架-陆坡剖面化 + 海沟窄带剖面
 7) v3.0 margin_type 输出: 被动边缘(近脊无沟,大西洋型宽陆架)
    vs 主动边缘(海沟,太平洋型窄陆架) → 下游 per-margin 陆架分化
=============================================================
"""
import numpy as np
from numba import njit
from scipy.ndimage import (gaussian_filter, distance_transform_edt,
                           label as ndi_label, binary_dilation,
                           binary_erosion as ndi_binary_erosion,
                           binary_closing, uniform_filter)
from .tectonics import proximity_fields

# 地球开阔洋 (深度≤-1000m, 占海洋84.6%) 高程分位数 —— ETOPO1(Ice) 2048 实测
_OC_Q = np.arange(0, 1.0001, 0.025)
_OC_V = np.array([-9000.0, -5600.0, -5480.0, -5380.0, -5300.0, -5230.0, -5150.0,
                  -5080.0, -5000.0, -4930.0, -4850.0, -4775.0, -4695.0, -4620.0,
                  -4545.0, -4470.0, -4400.0, -4334.0, -4279.0, -4221.0, -4160.0,
                  -4090.0, -4014.0, -3938.0, -3867.0, -3793.0, -3717.0, -3635.0,
                  -3550.0, -3455.0, -3353.0, -3243.0, -3128.0, -2990.0, -2828.0,
                  -2654.0, -2431.0, -2163.0, -1839.0, -1450.0, -1000.0], np.float64)


def _rot(axis, ang):
    """绕单位向量 axis 旋转 ang 弧度的 3x3 旋转矩阵。"""
    a = np.asarray(axis, np.float64); a = a / np.linalg.norm(a)
    x, y, z = a
    c, s = np.cos(ang), np.sin(ang); C = 1 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def _cross(a, b):
    """叉积, 向量沿第0轴。"""
    return np.stack([a[1] * b[2] - a[2] * b[1],
                     a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]], axis=0)


def _fbm(T, rng, freq=5.0, octaves=4):
    """在 T×T 网格上生成多倍频高斯平滑叠加的分形噪声场并归一化。"""
    noise = np.zeros((T, T), np.float32); amp, tot, f = 1.0, 0.0, freq
    for _ in range(octaves):
        n = rng.randn(T, T).astype(np.float32)
        n = gaussian_filter(n, max(T / (2.0 * f), 0.6)); n /= (n.std() + 1e-8)
        noise += amp * n; tot += amp; amp *= 0.5; f *= 2.0
    return noise / tot


def _dist(m):
    """返回掩膜 m 之外最近真值像素的欧氏距离场（全假时返回大数）。"""
    return distance_transform_edt(~m).astype(np.float32) if m.any() else np.float32(1e4)


@njit(cache=True, fastmath=True)
def _stamp_belt(canvas, cy, cx, ux, uy, L, W, val=1.0):
    """M2: 沿方向(ux,uy)盖章细长山带 (along^4 平顶脊 + across 高斯剖面)

    Numba JIT 加速版 (替代 mgrid+exp+maximum 临时数组, 直接逐像素更新 canvas)。
    原地写入 canvas (float32), 仅当新值 > 旧值时覆盖 (等价于 np.maximum)。
    """
    H, Wd = canvas.shape
    cy_i = int(round(cy))
    cx_i = int(round(cx))
    n = np.sqrt(ux * ux + uy * uy)
    if n < 1e-9:
        return
    ux_n = ux / n
    uy_n = uy / n
    pad = int(L) + 3
    y0 = max(0, cy_i - pad)
    y1 = min(H, cy_i + pad + 1)
    x0 = max(0, cx_i - pad)
    x1 = min(Wd, cx_i + pad + 1)
    if y0 >= y1 or x0 >= x1:
        return
    L_safe = L if L >= 1.0 else 1.0
    W_safe = W if W >= 1.0 else 1.0
    inv_L = 1.0 / L_safe
    inv_W = 1.0 / W_safe
    # 沿局部 pad×pad 区域逐像素更新 canvas (替代 mgrid+exp+maximum 向量化)
    for j in range(y0, y1):
        dy = float(j) - float(cy_i)
        for i in range(x0, x1):
            dx = float(i) - float(cx_i)
            along = (dx * ux_n + dy * uy_n) * inv_L
            across = (dx * -uy_n + dy * ux_n) * inv_W
            a4 = along * along
            a4 = a4 * a4
            blob = np.exp(-a4 - across * across)
            v = val * blob
            if v > canvas[j, i]:
                canvas[j, i] = v


def _plate_centroid(plate, crust, k, r_flat):
    """板块 k 的大陆质心(优先)/几何质心(无陆) → 单位球面向量"""
    m = (plate.ravel() == k)
    if not m.any():
        return None
    lm = m & (crust.ravel() == 1)
    src = np.nonzero(lm)[0] if lm.any() else np.nonzero(m)[0]
    p = r_flat[:, src].mean(axis=1)
    n = np.linalg.norm(p)
    return p / n if n > 1e-9 else None


def _vel_toward(p, c, omg_clip, rng, horizon=200.0, dt=0.625, jitter=0.15):
    """绕轴使 p 正向旋转向 c: axis = p×u, u=切向朝 c 分量。
    omg = 剩余角距/(horizon·dt) (差速收敛: 近慢远快, horizon 步后同时到 c)。
    主循环每步旋转 omg·dt rad, 故 horizon 步总转角 = ang。"""
    u = c - np.dot(c, p) * p                    # p 处切向、指向 c
    nu = np.linalg.norm(u)
    if nu < 1e-6:                               # 已在 c 或对跖
        a = rng.randn(3); a /= np.linalg.norm(a)
        return a, 0.0
    u /= nu
    axis = np.cross(p, u)
    axis /= np.linalg.norm(axis)
    ang = float(np.arctan2(nu, np.dot(c, p)))   # p→c 角距
    omg = float(np.clip(ang / (max(horizon, 1.0) * dt), *omg_clip))
    omg *= (1.0 + jitter * rng.randn())
    return axis, max(omg, 0.0)


def _vel_separate(p, a, omg_abs, rng, jitter=0.2):
    """裂解分离: 质心 p 远离裂谷大圆(法向 a) → 两侧背向漂移。
    u = sign(p·a)·a 的切向分量; axis = p×u。"""
    s = np.dot(p, a)
    u = a - np.dot(a, p) * p
    nu = np.linalg.norm(u)
    if nu < 1e-6:
        return rng.randn(3) * 0 + None, 0.0
    u = (u / nu) * np.sign(s if abs(s) > 1e-4 else 1.0)
    axis = np.cross(p, u)
    na = np.linalg.norm(axis)
    if na < 1e-9:
        return None, 0.0
    axis /= na
    omg = float(omg_abs * (1.0 + jitter * rng.randn()))
    return axis, max(omg, 1e-4)


def _classify(pl, cr, ag2d, V, r, T):
    """现势板块边界分类 → 掩膜字典(沟/弧/碰撞/脊/陆内离散/转换) + 速率场"""
    if r.ndim == 2:                     # 兼容展平输入 (3, N) → (3, T, T)
        r = r.reshape(3, T, T)
    Vg = V.reshape(3, T, T)
    out = {k: np.zeros((T, T), bool) for k in
           ("trench", "arc", "coll", "ridge", "riftland", "trans")}
    spd = np.zeros((T, T), np.float32)

    def put(mk, sel, cell):
        """把选中的像素写回对应掩膜。"""
        if sel.any():
            out[mk][cell[0][sel], cell[1][sel]] = True

    for ax_ in (0, 1):
        if ax_ == 0:
            A, B = pl[:, :-1], pl[:, 1:]
            Vd = Vg[:, :, 1:] - Vg[:, :, :-1]
        else:
            A, B = pl[:-1, :], pl[1:, :]
            Vd = Vg[:, 1:, :] - Vg[:, :-1, :]
        iB, jB = np.nonzero(A != B)
        if len(iB) == 0:
            continue
        dv = Vd[:, iB, jB].T
        if ax_ == 0:
            ra, rb = r[:, iB, jB], r[:, iB, jB + 1]
            c1, c2 = (iB, jB), (iB, jB + 1)
        else:
            ra, rb = r[:, iB, jB], r[:, iB + 1, jB]
            c1, c2 = (iB, jB), (iB + 1, jB)
        nv = (rb - ra).T
        nv /= (np.linalg.norm(nv, axis=1, keepdims=True) + 1e-12)
        cn = -(dv * nv).sum(1)
        dvp = (dv * nv).sum(1)[:, None] * nv
        am = np.linalg.norm(dv - dvp, axis=1)
        s = np.hypot(cn, am) + 1e-9
        conv, div = cn > 0.35 * s, cn < -0.35 * s
        trans = ~conv & ~div
        ca, cb = cr[c1], cr[c2]
        aa_, ab = ag2d[c1], ag2d[c2]
        w_ol = conv & (ca != cb)                       # 洋-陆: 沟在洋侧, 弧在陆侧
        put("trench", w_ol & (ca == 0), c1); put("arc", w_ol & (ca == 1), c1)
        put("trench", w_ol & (cb == 0), c2); put("arc", w_ol & (cb == 1), c2)
        w_oo = conv & (ca == 0) & (cb == 0)            # 洋-洋: 老者俯冲, 年轻侧成弧
        old1, old2 = w_oo & (aa_ <= ab), w_oo & (aa_ > ab)
        put("trench", old1, c1); put("arc", w_oo & ~old1, c1)
        put("trench", old2, c2); put("arc", w_oo & ~old2, c2)
        w_cc = conv & (ca == 1) & (cb == 1)            # 陆陆碰撞
        put("coll", w_cc, c1); put("coll", w_cc, c2)
        w_r = div & (ca == 0) & (cb == 0)              # 洋中脊
        put("ridge", w_r, c1); put("ridge", w_r, c2)
        w_l = div & ((ca == 1) | (cb == 1))            # 大陆离散(裂谷源)
        put("riftland", w_l, c1); put("riftland", w_l, c2)
        put("trans", trans, c1); put("trans", trans, c2)
        s32 = s.astype(np.float32)
        np.maximum.at(spd, (c1[0], c1[1]), s32)
        np.maximum.at(spd, (c2[0], c2[1]), s32)
    out["spd"] = spd
    return out


def _fill_claimed(P2, C2, A2, M2, T):
    """所有 plate==-1 的格 → 最近有主格(EDT最近邻), 属性继承;
    源格处于扩张边缘的窄带 → 新生洋壳(age=0), 其余继承成熟洋壳。"""
    occ = P2 != -1
    if occ.all():
        return
    # 最近有主格坐标: 距离场 + 分块爬山等价实现(与EDT索引语义一致, 版本无关)
    d_ = distance_transform_edt(~occ)
    # 迭代标记: 从每个空格沿梯度走向最近的已占格
    gy, gx = np.nonzero(~occ)
    # 用 marching 方式: 多轮把邻居的 (d, id) 取最小传播
    best_d = np.where(occ, 0.0, d_).astype(np.float32)
    best_id = np.where(occ, np.arange(T * T).reshape(T, T), -1).astype(np.int64)
    for _ in range(T + T):                # 传播轮数上限 = 2T, 必然收敛
        changed = False
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            s = np.roll(np.roll(best_id, dy, 0), dx, 1)
            sd = np.roll(np.roll(best_d, dy, 0), dx, 1) + 1.0
            upd = (~occ) & (s >= 0) & (sd < best_d)
            if upd.any():
                best_id[upd] = s[upd]; best_d[upd] = sd[upd]; changed = True
        if not changed:
            break
    sy = best_id // T; sx = best_id % T
    src_ok = best_id >= 0
    gy, gx = np.nonzero(~occ & src_ok)
    sy, sx = sy[gy, gx], sx[gy, gx]

    edge = np.zeros((T, T), bool)         # 扩张边缘格(有主格的边缘)
    er = np.ones((T, T), bool)
    er[1:-1, 1:-1] = (occ[1:-1, 1:-1] & occ[:-2, 1:-1] & occ[2:, 1:-1]
                      & occ[1:-1, :-2] & occ[1:-1, 2:])
    edge = occ & ~er
    is_new = edge[sy, sx] & (C2[sy, sx] == 0)
    P2[gy, gx] = P2[sy, sx]
    C2[gy, gx] = C2[sy, sx]
    A2[gy, gx] = np.where(is_new, 0.0, A2[sy, sx])
    M2[gy, gx] = np.where(is_new, 0.0, M2[sy, sx])


def run_plate_sim(T=512, seed=0, K=12, n_hs=7,
                  dt=0.625, steps=720, verbose=True, land_target=0.31,
                  step_agg_end=None, step_rift=None, step_rift2=None,
                  re_v=60, n_micro=None):
    """**板块演化主入口 v3.0**：球面三阶段超大陆旋回 → 粗 DEM + 4 构造场 + 陆壳 + 边缘类型。

    三阶段时间轴（默认 dt=0.625 Myr × 720 步 = 450 Myr 总演化）：
        0. 聚合期 (step < step_agg_end，默认 0~240)：板块共享一个欧拉聚合中心 c 做
           大圆漂移，产生初始碰撞 → 焊接合并（相邻 < 2px 的板块并入，滞留板块豁免）。
        1. 超大陆稳定期 (step_agg_end ~ step_rift)：持续向心挤压 → orog 场累积，
           碰撞带峰值（为后续 4900m 碰撞造山做历史累积）。
        2. 主裂解 (step_rift ~ step_rift2，默认 416~544)：沿随机大圆把最大陆板块
           切成两半，两块共享同一欧拉轴反向 ±om_s 对旋 → 大西洋型对称张开。
        3. 次裂解 (step_rift2 ~ steps)：再对另一大陆进行第二刀，产生印度洋 /
           南大西洋型二次洋盆 + 裂谷谷地带。

    核心机制（详见 TECHNICAL §2）：
        - 2 块滞留极地板块（南极 / 格陵兰型，保证极地陆量 ≥ 12%）。
        - 4~6 块微大陆 / 地体 延迟 t>100Myr 撞入（印度型半岛 / 马达加斯加型岛）。
        - 大陆陆地全局守恒：复制/湮灭对称增删 + 海岸剥蚀到 land_target ≈ 0.31。
        - 输出 margin_type（0 普通 / 1 被动边缘 / 2 主动边缘）供下游陆架分化。

    Parameters
    ----------
    T : int
        球面网格分辨率（单边像素数），与 generate_world 的 tect_res 一致；默认 512。
    seed : int
        随机种子。同一 seed 两次调用**严格一致**（板块配置 / 速度场 / 陆壳分布）。
    K : int
        主板块数（默认 12）。加上 4~6 块微大陆，总板块池 = MAXP = K + 16。
    n_hs : int
        初始热点（火山链）数量。生成年龄记忆场用于后续热点岛链盖章。
    dt : float
        每步模拟的 Myr（默认 0.625 = 720 步 × dt ≈ 450 Myr）。
    steps : int
        总模拟步数（默认 720）。
    verbose : bool
        是否打印阶段进度日志（默认 True）。
    land_target : float
        目标陆地面积占比（默认 0.31，靠近地球 0.29）。陆地守恒 + 硬剥皮双保险。
    step_agg_end / step_rift / step_rift2 : int or None
        三阶段切换步；空取默认值 150/260/340 Myr 对应步长换算结果。
    re_v : int
        俯冲带 / 海沟 / 造山带的"地形强度"尺度参数（v2.2 = 60，越大造山越高）。
    n_micro : int or None
        微大陆 / 地体数量；空时在 4~6 中随机。

    Returns
    -------
    tuple (elev, tect, crustm, margin, belt, rift)
        - elev    : (T, T) float32, 米制粗 DEM（钳制在 ±10500/8900 之前的原始骨架）
        - tect    : (4, T, T) float32, 4 构造场 [俯冲碰撞/离散/转换/余裕]
        - crustm  : (T, T) float32, 陆壳概率 [0, 1]（连续场，非二值）
        - margin  : (T, T) int8, 边缘类型 0=普通 1=被动 2=主动
        - belt    : (T, T) float32, 延展山带（板内造山 / 碰撞带走向）
        - rift    : (T, T) float32, 失败裂谷场（供裂谷重新下切）
    """
    rng = np.random.RandomState(seed)
    ii, jj = np.mgrid[0:T, 0:T].astype(np.float64)
    lat_g = (0.5 - (ii + 0.5) / T) * np.pi
    lon_g = ((jj + 0.5) / T) * 2 * np.pi
    r = np.stack([np.cos(lat_g) * np.cos(lon_g),
                  np.cos(lat_g) * np.sin(lon_g), np.sin(lat_g)])
    r_flat = r.reshape(3, -1); N = T * T

    # ---- v3.0 阶段步数 (默认: 150Myr 聚合完 / 260Myr 裂解 / 340Myr 次裂解) ----
    step_agg_end = int(step_agg_end if step_agg_end is not None else 150.0 / dt)
    step_rift = int(step_rift if step_rift is not None else 260.0 / dt)
    step_rift2 = int(step_rift2 if step_rift2 is not None else 340.0 / dt)
    step_micro_go = int(100.0 / dt)              # 微大陆 t>100Myr 才出发

    ctr = rng.randn(K, 3); ctr /= np.linalg.norm(ctr, axis=1, keepdims=True)
    plate = np.tensordot(ctr, r_flat, axes=(1, 0)).argmax(0).reshape(T, T).astype(np.int32)

    # ---- v3.0 聚合中心: 热带边缘~27°纬(盘古大陆跨赤道布局, z=0.45) ----
    # 中心太高纬→大陆集中中纬(加权陆地超标); 热带中心→裂解后各纬度均有大陆
    # M6: 超大陆入"分析北半球"概率 72% (地球北陆 67:33)。
    # 注意坐标镜像: 本函数内部"北"(row 0 = +z) = 输出数组上半 = 分析口径的
    # 南半球 → pref_north=True 实际把超大陆放进分析南半球, 故阈值取 0.28
    pref_north = rng.rand() < 0.28
    c = np.array([rng.uniform(-0.35, 0.35),
                  rng.uniform(-0.35, 0.35),
                  0.45 if pref_north else -0.45])
    c += 0.2 * rng.randn(3)
    c /= np.linalg.norm(c)

    # ---- v3.0 微大陆/地体播种: K..K+n_micro-1, 远离聚合中心的半球 ----
    n_micro = int(n_micro if n_micro is not None else rng.randint(4, 7))
    K_total = K + n_micro
    micro_info = []                              # (mi, oy, ox, rad) 陆壳后补
    for mi in range(n_micro):
        for _try in range(30):
            oy = rng.uniform(0.08, 0.92) * T
            ox = rng.uniform(0.0, 1.0) * T
            pr = np.array([np.cos((0.5 - (oy + 0.5) / T) * np.pi) *
                           np.cos((ox + 0.5) / T * 2 * np.pi),
                           np.cos((0.5 - (oy + 0.5) / T) * np.pi) *
                           np.sin((ox + 0.5) / T * 2 * np.pi),
                           np.sin((0.5 - (oy + 0.5) / T) * np.pi)])
            if np.dot(pr, c) < 0.05:             # 距聚合中心 >~87°
                break
        rad = rng.uniform(4.0, 9.0)
        ddx = (jj - ox + T / 2) % T - T / 2
        disk = ((ii - oy) ** 2 + ddx * ddx) < rad * rad
        plate[disk] = K + mi
        micro_info.append((mi, oy, ox, rad))

    axes = rng.randn(K_total, 3); axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    omg = rng.uniform(0.0004, 0.0012, K_total) * rng.choice([-1.0, 1.0], K_total)
    MAXP = K + 16                                  # 板块号池 (焊接合并/裂解切割)
    axes = np.vstack([axes, rng.randn(MAXP - K_total, 3)])
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    omg = np.concatenate([omg, np.zeros(MAXP - K_total)]).astype(np.float64)
    alive = np.zeros(MAXP, bool)
    alive[:K_total] = True                         # 焊接死号 → 裂解切割复用

    # ---- v3.0 滞留板块 (南极/格陵兰型): 2块不参与聚合, 驻陆半球极 ----
    # M6: 固定 2 块均驻主极 (对极版使 SH 陆占比超标 56% vs 地球 33%;
    # 极地陆量不变 → perm_snow/苔原收益保留, 半球分布回归)
    n_stray = 2
    strayers = sorted(int(x) for x in rng.choice(K, size=n_stray, replace=False))
    stray_pole = {sk: pref_north for sk in strayers}
    if verbose:
        print(f"  v3.0 滞留板块: {strayers} (双驻{'N' if pref_north else 'S'}极, 不参与聚合)")

    # ---- 大陆播种: 每大陆 = 2~5 个重叠圆盘的不规则并集, 迭代至 land_target ----
    # 半球先验沿用聚合中心所在半球 (pref_north 已按 c 定义) → 播种即偏向未来超大陆
    crust = np.zeros((T, T), np.int8)
    guard = 0
    while (crust == 1).sum() / (T * T) < land_target and guard < 60:
        guard += 1
        pk = int(rng.randint(K))
        cells = np.argwhere(plate == pk)
        if len(cells) == 0:
            continue
        # 主中心; 85% 概率落在陆半球 (板块漂移会稀释, 播种时加强)
        pick = cells
        if rng.rand() < 0.85:
            hc = cells[cells[:, 0] < T // 2] if pref_north else cells[cells[:, 0] >= T // 2]
            if len(hc):
                pick = hc
        cy, cx = pick[rng.randint(len(pick))]
        # 滞留板块: 强制极地播种 (南极型, 聚合后留在极地)
        # 其余: 50% 概率高纬(格陵兰型), 极地带偏向陆半球
        stray_force = pk in strayers
        if stray_force or rng.rand() < 0.5:
            band = (0.05, 0.15) if pref_north else (0.85, 0.95)
            row = int(rng.uniform(*band) * T)
            cand = cells[np.abs(cells[:, 0] - row) < 10]
            if len(cand):
                cy, cx = cand[rng.randint(len(cand))]
        # 一个大陆 = 2~5 个子圆盘的不规则并集
        for _d in range(int(rng.randint(2, 6))):
            oy = cy + rng.normal(0.0, T * 0.045)
            ox = cx + rng.normal(0.0, T * 0.045)
            rad = rng.uniform(0.05, 0.12)
            dx = (jj - ox + T / 2) % T - T / 2
            crust[(plate == pk) & (((ii - oy) ** 2 + dx * dx) / T ** 2 < rad ** 2)] = 1
    land0 = int((crust == 1).sum())

    # ---- v3.0 微大陆陆壳补盖 (主大陆播种后, 不计入 land_target) ----
    for (mi, oy, ox, rad) in micro_info:
        ddx = (jj - ox + T / 2) % T - T / 2
        crust[(plate == K + mi) &
              (np.hypot(ii - oy, ddx) < rad * 0.8)] = 1
    land_micro = int((crust == 1).sum()) - land0

    # ---- v3.0 滞留板块极地陆量保底 (≥土地13% → 南极/格陵兰型冰盖) ----
    # M6: 12→13% + 集中 75-85° 极带 + 板块域未达极带时取最近极行格子
    # (16% 超调 → 陆地碰 33% 上限被海岸剥皮 → 大陆碎片化)
    stray_need = int(0.13 * max((crust == 1).sum(), 1))
    for _ in range(80):
        stray_land = int(sum((crust == 1)[plate == sk].sum()
                             for sk in strayers))
        if stray_land >= stray_need:
            break
        sk = int(rng.choice(sorted(strayers)))
        cells = np.argwhere(plate == sk)
        if len(cells) == 0:
            break
        pol = stray_pole[sk]
        band = (0.03, 0.14) if pol else (0.86, 0.97)
        row = int(rng.uniform(*band) * T)
        cand = cells[np.abs(cells[:, 0] - row) < 14]
        if len(cand) == 0:
            pr0 = 0 if pol else T - 1
            cand = cells[np.abs(cells[:, 0] - pr0).argmin()][None]
        cy, cx = cand[rng.randint(len(cand))]
        for _d in range(int(rng.randint(2, 5))):
            oy = cy + rng.normal(0.0, T * 0.035)
            ox = cx + rng.normal(0.0, T * 0.035)
            rad = rng.uniform(0.04, 0.10)
            dx = (jj - ox + T / 2) % T - T / 2
            crust[(plate == sk) &
                  (((ii - oy) ** 2 + dx * dx) / T ** 2 < rad ** 2)] = 1
    age = np.where(crust == 1, 0.0, 60.0).astype(np.float32)

    hs_rc = [(int(rng.uniform(0.06, 0.94) * T), int(rng.rand() * T)) for _ in range(n_hs)]
    vmem = np.zeros(N, np.float32)
    orog = np.zeros((T, T), np.float32)
    div_mem = np.zeros((T, T), np.float32)
    div_tot = np.zeros((T, T), np.float32)      # M2: 无衰减累计(失败裂谷地貌)
    V = np.zeros((3, N), np.float32)
    plate_f, crust_f, age_f = plate.ravel(), crust.ravel(), age.ravel()

    # ---- v3.0 裂解轴 (裂解启动时计算) ----
    rift_axis = None                              # 主裂谷大圆法向
    rift2_axis = None                             # 次裂谷大圆法向
    pangea_frac = [0.0]                           # 超大陆连通性监控
    locked = set()                                # 已锁定对旋的裂解板块(固定欧拉轴)

    def refresh_agg(horizon):
        """聚合期差速收敛: horizon 步后全部板块同时到达 c 附近 (滞留板块除外)"""
        for k in np.nonzero(alive)[0]:
            if k in strayers:
                continue                          # 滞留板块保持原位漂移
            if k >= K and step < step_micro_go:
                continue                          # 微大陆尚未出发
            p = _plate_centroid(plate_f.reshape(T, T), crust_f.reshape(T, T),
                                k, r_flat)
            if p is None:
                continue
            hz = horizon if k < K else max(step_rift2 - step, 60)
            ax_, om = _vel_toward(p, c, (0.0, 0.014), rng, horizon=hz, dt=dt)
            axes[k], omg[k] = ax_, om

    def refresh_supercontinent():
        """超大陆期: 持续朝心挤压(缝合边界洋盆), 微大陆继续撞入 (滞留板块除外)"""
        for k in np.nonzero(alive)[0]:
            if k in strayers:
                continue
            p = _plate_centroid(plate_f.reshape(T, T), crust_f.reshape(T, T),
                                k, r_flat)
            if p is None:
                continue
            hz = (step_rift2 - step) if k >= K else (step_rift - step)
            lo, hi = (0.0015, 0.009) if k >= K else (0.0004, 0.0022)
            ax_, om = _vel_toward(p, c, (lo, hi), rng, horizon=max(hz, 60),
                                  dt=dt)
            axes[k], omg[k] = ax_, om

    def refresh_rift(a1, a2, w2):
        """裂解期: 未锁定的板块(微大陆等)远离裂谷大圆漂移。
        已锁定的裂解板块(共享欧拉轴对旋)保持固定速度 → 大西洋型对称张开。"""
        for k in range(MAXP):
            if not alive[k] or k in locked:
                continue
            p = _plate_centroid(plate_f.reshape(T, T), crust_f.reshape(T, T),
                                k, r_flat)
            if p is None:
                continue
            ax_, om = _vel_separate(p, a1, 0.011, rng)
            if ax_ is None:
                continue
            if a2 is not None:
                ax2, om2 = _vel_separate(p, a2, 0.011 * w2, rng)
                if ax2 is not None:
                    ax_, om = ax_ + 0.6 * ax2, om + 0.6 * om
            nn = np.linalg.norm(ax_)
            if nn < 1e-9:
                continue
            axes[k], omg[k] = ax_ / nn, min(om, 0.014)

    def weld_continents():
        """超大陆期焊接: 与最大陆板块大陆直接接触(≤2px)的板块全部并入 →
        消除板块边界海峡, 大陆缝合为单一超大陆板块 (真实: 碰撞缝合带)。
        滞留板块豁免 → 极地陆不被吞入超大陆拖向热带。"""
        lm = crust_f.reshape(T, T) == 1
        lab_pl = plate_f.reshape(T, T)
        counts = np.bincount(lab_pl[lm], minlength=MAXP)
        if counts.sum() == 0:
            return
        kstar = int(counts.argmax())
        m_k = lm & (lab_pl == kstar)
        if not m_k.any():
            return
        dil = binary_dilation(m_k, iterations=2)
        touching = dil & lm & (lab_pl != kstar)
        if touching.any():
            for km in np.unique(lab_pl[touching]):
                if int(km) in strayers:
                    continue                  # 滞留板块不缝合 (南极独立)
                plate_f[lab_pl.ravel() == km] = kstar
                alive[km] = False
        # 焊接后朝 c 微速维持挤压
        p = _plate_centroid(plate_f.reshape(T, T), crust_f.reshape(T, T),
                            kstar, r_flat)
        if p is not None:
            ax_, om = _vel_toward(p, c, (0.0002, 0.0012), rng, dt=dt,
                                  horizon=120)
            axes[kstar], omg[kstar] = ax_, om

    def split_supercontinent(a):
        """裂解启动: 把大陆最多的板块沿裂谷大圆切成两块,
        新板块号取空闲位 → 裂解速度使其背向漂移 (新板块边界=裂谷)。
        轴向搜索: 给定轴与超大陆相切时, 绕 pc 旋转搜索有效切口 (两侧≥25%陆地)。"""
        lm = crust_f.reshape(T, T) == 1
        counts = np.bincount(plate_f[lm.ravel()], minlength=MAXP)
        if counts.sum() == 0:
            return
        kstar = int(counts.argmax())
        m = (plate_f.reshape(T, T) == kstar)
        mland = m & lm
        n_land = int(mland.sum())
        if n_land < 200:
            return
        # 搜索有效裂谷轴: 原轴 → 绕 pc 每 30° 旋转, 共 12 个候选
        if lm.any():
            pc = r_flat[:, np.nonzero(lm.ravel())[0]].mean(axis=1)
            pc /= np.linalg.norm(pc)
        else:
            pc = np.array([0.0, 0.0, 1.0])
        ax_ok = None
        cand = a
        for i in range(12):
            side = ((r[0] * cand[0] + r[1] * cand[1] + r[2] * cand[2]) > 0)
            s1 = int((mland & side).sum()); s2 = n_land - s1
            if s1 >= 0.25 * n_land and s2 >= 0.25 * n_land:
                ax_ok = cand
                break
            # 绕 pc 旋转 30°
            cand = _rot(pc, np.pi / 6.0) @ cand
            cand /= np.linalg.norm(cand)
        if ax_ok is None:
            return
        a = ax_ok
        side = ((r[0] * a[0] + r[1] * a[1] + r[2] * a[2]) > 0)
        # 沿切线预挖 ~2px 海峡(全部陆地, 含滞留板块——否则跨切线的滞留陆
        # 会在对跖经度线把两块大陆桥接回一块)
        cut = np.abs(r[0] * a[0] + r[1] * a[1] + r[2] * a[2]) < (2.0 * np.pi / T)
        crust_f[(lm & cut).ravel()] = 0            # 就地修改(闭包), 守恒补丁会补
        mland = mland & ~cut
        free = int(np.nonzero(~alive)[0][0]) if (~alive).any() else -1
        if free < 0:
            return
        plate_f[(m & side).ravel()] = free
        alive[free] = True
        # 共享欧拉轴对旋: e = a×pc (欧拉极远离大陆) → 大西洋型对称张开,
        # 两块绕同一轴反向旋转, 不会在球面回摆重触 (对跖线桥接根因)
        e = np.cross(a, pc)
        ne = np.linalg.norm(e)
        if ne < 1e-9:
            e = np.cross(a, np.array([0.0, 0.0, 1.0]))
            ne = np.linalg.norm(e)
        e /= ne
        om_s = 0.006                    # 190步×0.006×0.625≈0.71rad=41°/侧: 分离够且纬度漂移小
        # 方向标定: kstar 侧质心速度 v=ω·(e×n) 应远离裂谷面(sign(v·a)=sign(n·a))
        n_k = _plate_centroid(plate_f.reshape(T, T), crust_f.reshape(T, T),
                              kstar, r_flat)
        n_f = _plate_centroid(plate_f.reshape(T, T), crust_f.reshape(T, T),
                              free, r_flat)
        s_k = 1.0
        if n_k is not None and np.dot(np.cross(e, n_k), a) * np.dot(n_k, a) < 0:
            s_k = -1.0
        axes[kstar], omg[kstar] = e, s_k * om_s
        if n_f is not None:
            axes[free], omg[free] = e, -s_k * om_s
        else:
            axes[free], omg[free] = e, -s_k * om_s
        locked.add(kstar); locked.add(free)
        return a                                   # 返回有效轴供速度场使用

    for step in range(steps):
        # ---- v3.0 阶段速度调度 (每 re_v 步重算朝向) ----
        if step % re_v == 0 or step == step_rift or step == step_rift2:
            if step < step_agg_end:
                refresh_agg(step_agg_end - step)
            elif step < step_rift:
                refresh_supercontinent()
            else:
                if step == step_rift or rift_axis is None:
                    # 主裂谷轴: ⊥ 超大陆质心 → 大圆穿过超大陆中心
                    lm = crust_f.reshape(T, T) == 1
                    if lm.any():
                        pc = r_flat[:, np.nonzero(lm.ravel())[0]].mean(axis=1)
                        pc /= np.linalg.norm(pc)
                    else:
                        pc = np.array([0.0, 0.0, 1.0])
                    v1 = rng.randn(3); v1 -= np.dot(v1, pc) * pc
                    v1 /= np.linalg.norm(v1)
                    a_ok = split_supercontinent(v1)      # 超大陆裂开(含轴向搜索)
                    rift_axis = a_ok if a_ok is not None else v1
                if step == step_rift2 and rng.rand() < 0.75:
                    v2 = rng.randn(3); v2 -= np.dot(v2, rift_axis) * rift_axis
                    nv2 = np.linalg.norm(v2)
                    if nv2 > 1e-6:
                        a2_ok = split_supercontinent(v2 / nv2)  # 次级再裂
                        if a2_ok is not None:
                            rift2_axis = a2_ok
                refresh_rift(rift_axis, rift2_axis, 0.8)
        elif step >= step_agg_end // 2 and step < step_rift and step % 30 == 0:
            weld_continents()          # 聚合中期起周期焊接 (接触即缝合→无重叠竞争)
        # ---- 后向映射搬运: 每目的地取逆旋转源点, 面积守恒, 大陆不撕裂 ----
        new_plate = np.full(N, -1, np.int32)
        new_crust = np.zeros(N, np.int8)
        new_age = np.zeros(N, np.float32)
        new_vmem = np.zeros(N, np.float32)
        claim = np.full(N, -1.0, np.float32)
        for k in np.nonzero(alive)[0]:
            q = _rot(axes[k], -omg[k] * dt) @ r_flat          # 目的地 → 源
            la = np.arcsin(np.clip(q[2], -1, 1))
            lo = np.arctan2(q[1], q[0])
            ri = np.clip(((0.5 - la / np.pi) * T).astype(np.int64), 0, T - 1)
            ci = ((lo / (2 * np.pi)) % 1.0 * T).astype(np.int64) % T
            src = ri * T + ci
            d = np.nonzero(plate_f[src] == k)[0]              # 源确属板块k才一致
            if len(d) == 0:
                continue
            s = src[d]
            pr = np.where(crust_f[s] == 1, 1e6,
                          np.maximum(0.01, 500.0 - age_f[s])).astype(np.float32)
            take = pr > claim[d]                              # 陆>年轻洋>老洋
            dd, ss = d[take], s[take]
            new_plate[dd] = k; new_crust[dd] = crust_f[ss]
            new_age[dd] = age_f[ss]; new_vmem[dd] = vmem[ss]
            claim[dd] = pr[take]
        # ---- 间隙归属: 继承上一步同位置板块号(必为邻接板之一), 零残留 ----
        gapm = new_plate == -1
        if gapm.any():
            new_plate[gapm] = plate_f[gapm]   # P2 是 new_plate 的视图, 同步更新
            # crust/age/vmem 刻意保持 0 → 扩张中心新生洋壳(物理: 海底扩张)
        # ---- v3.0 陆地全局总量守恒: 后向映射取整会造成复制/湮灭双计 ----
        # 全局对称增删 + 海岸平滑偏好: 湮灭→补在凹湾(邻陆密度高), 复制→删在
        # 岬角/碎屑(邻陆密度低) → 守恒同时海岸去噪 (消碎屑岛/锯齿岸)
        old_total = int((crust_f == 1).sum())
        new_total = int((new_crust == 1).sum())
        diff = old_total - new_total
        if diff != 0:
            C2v = new_crust.reshape(T, T)
            P2v = new_plate.reshape(T, T)
            landm = C2v == 1
            dens = uniform_filter(landm.astype(np.float32), size=5)
            if diff > 0:                              # 湮灭 → 凹湾补格
                ring = binary_dilation(landm, iterations=1) & ~landm
                if step >= step_rift:
                    # 裂解期排除裂谷缝隙: 邻域含≥2板块陆地 = 正在张开的洋盆,
                    # 填它会永远封死裂谷 (凹湾填补的副作用)
                    acc = np.zeros((T, T), np.int16)
                    for k in np.nonzero(alive)[0]:
                        lk = landm & (P2v == k)
                        if lk.any():
                            acc += binary_dilation(lk, iterations=3)
                    ring = ring & (acc < 2)
                idx = np.nonzero((ring & (dens >= 0.32)).ravel())[0]
                if idx.size < diff:
                    idx = np.nonzero(ring.ravel())[0]
                take = min(diff, idx.size)
                if take > 0:
                    C2v.ravel()[rng.choice(idx, size=take, replace=False)] = True
            else:                                     # 复制 → 岬角/碎屑删格
                er = ndi_binary_erosion(landm, iterations=1)
                ring = landm & ~er
                idx = np.nonzero((ring & (dens <= 0.34)).ravel())[0]
                if idx.size < -diff:
                    idx = np.nonzero(ring.ravel())[0]
                take = min(-diff, idx.size)
                if take > 0:
                    C2v.ravel()[rng.choice(idx, size=take, replace=False)] = False
        P2 = new_plate.reshape(T, T); C2 = new_crust.reshape(T, T)

        for k in np.nonzero(alive)[0]:
            m = P2 == k
            if m.any():
                V[:, m.ravel()] = _cross(axes[k], r_flat[:, m.ravel()]) * omg[k]

        ev = _classify(P2, C2, new_age.reshape(T, T), V, r_flat, T)
        if ev["coll"].any():
            np.add.at(orog, np.nonzero(ev["coll"]), ev["spd"][ev["coll"]] * dt)
        div_mem *= 0.985
        if ev["riftland"].any():
            div_mem[ev["riftland"]] += dt
            div_tot[ev["riftland"]] += dt        # M2: 累计离散时长(无衰减)

        if (C2 == 1).sum() > 0.78 * land0:                    # 裂谷+陆地保护
            rift = (C2 == 1) & (div_mem > 30)
            if rift.any():
                C2[rift] = 0
                new_age.reshape(T, T)[rift] = 0.0

        for (hi, hj) in hs_rc:                                # 热点固定柱
            new_vmem.reshape(T, T)[hi, hj] += 3.0

        new_age[new_crust == 0] += dt                         # 洋壳老化(现在能真正累积)
        plate_f, crust_f, age_f, vmem = new_plate, new_crust, new_age, new_vmem
        if verbose and (step % 90 == 0 or step == steps - 1):
            lm2 = crust_f.reshape(T, T) == 1
            _, nlab = ndi_label(lm2)
            if nlab:
                sizes = np.bincount(
                    ndi_label(lm2)[0].ravel()); sizes[0] = 0
                pangea_frac[0] = sizes.max() / max(sizes.sum(), 1)
            print(f"  t = {(step + 1) * dt:.0f} Myr | 陆地 {crust_f.mean() * 100:.1f}%"
                  f" | 超大陆 {pangea_frac[0] * 100:.0f}%")

    # ==================== 终局装配 ====================
    crustm = (crust_f.reshape(T, T) == 1)
    # v3.0 陆地硬约束: >33% 时从海岸腐蚀剥皮收缩到 ~30% (环式收缩首环恒空)
    if crustm.mean() > 0.33:
        excess = int((crustm.mean() - 0.30) * T * T)
        rem = crustm.copy()
        for _ in range(80):
            er = ndi_binary_erosion(rem, iterations=1)
            ring = rem & ~er                       # 当前陆地最外圈
            idx = np.nonzero(ring.ravel())[0]
            if idx.size == 0 or excess <= 0:
                break
            take = min(excess, idx.size)
            rem.ravel()[rng.choice(idx, size=take, replace=False)] = False
            excess -= take
        crustm = rem
        crust_f = crustm.ravel().astype(np.int8)
    f1, f2, f3 = _fbm(T, rng, 6.0, 4), _fbm(T, rng, 9.0, 4), _fbm(T, rng, 13.0, 3)

    evF = _classify(P2, new_crust.reshape(T, T), new_age.reshape(T, T), V, r_flat, T)
    d_tr, d_arc = _dist(evF["trench"]), _dist(evF["arc"])
    d_rg, d_tf = _dist(evF["ridge"]), _dist(evF["trans"])
    d_land = distance_transform_edt(~crustm).astype(np.float32)

    # ---- v3.0 margin_type: 被动边缘(近脊无沟=大西洋型) vs 主动边缘(海沟=太平洋型) ----
    # 陆缘带: 陆壳中距海 ≤2px; passive=近新生洋脊(裂解遗产)且无俯冲
    land_band = crustm & (distance_transform_edt(~crustm) <= 2.0)
    margin = np.zeros((T, T), np.int8)
    margin[land_band & (d_rg < 16.0) & (d_tr > 5.0)] = 1   # 被动 (宽陆架)
    margin[land_band & (d_tr <= 5.0)] = 2                  # 主动 (窄陆架)

    # ---- 现势边界速率场 (海沟深度随俯冲速率) ----
    if (evF["spd"] > 0).any():
        q90 = max(float(np.quantile(evF["spd"][evF["spd"] > 0], 0.9)), 1e-6)
        spn = np.clip(evF["spd"] / q90, 0, 1.5)
    else:
        spn = np.zeros((T, T), np.float32)

    # ---- 海洋 v2.2: 洋壳年龄排名 → 地球开阔洋深度分位 (直方图标定) ----
    # 年龄排名单调映射 → 空间结构保留 (活跃脊=最年轻=最浅, 俯冲前缘=最老=最深)
    oceanm = ~crustm
    base_oc = np.zeros((T, T), np.float32)
    if oceanm.any():
        ao = age_f.reshape(T, T)[oceanm]
        rk = ao.argsort().argsort() / max(ao.size - 1, 1)
        base_oc[oceanm] = np.interp(rk, _OC_Q, _OC_V).astype(np.float32)
    e_oc = base_oc + 170.0 * f2                          # 沉积盖层/小尺度起伏
    # 陆架(0~1px≈80km: 0→-140m) + 陆坡(1~1.8px: →-1800m) + 陆隆渐入开阔洋(→3.2px放开)
    prof = np.where(d_land < 1.0, -140.0 * d_land,
                    np.where(d_land < 1.8,
                             -140.0 - 1660.0 * (d_land - 1.0) / 0.8,
                             np.where(d_land < 3.2,
                                      -1800.0 - 2600.0 * (d_land - 1.8) / 1.4,
                                      -1.0e9)))
    e_oc = np.where(oceanm, np.maximum(e_oc, prof.astype(np.float32)), e_oc)
    # 火山弧: 双层结构 → 弧段状岛链 (窄基座 + 低频门控窄岛峰, 消"圆点"感)
    e_oc += 1500.0 * np.exp(-(d_arc / 1.7) ** 2)
    g_arc = _fbm(T, rng, 7.0, 2)
    e_oc += 4600.0 * np.exp(-(d_arc / 1.2) ** 2) * (g_arc > 0.05)
    # 海沟: 俯冲带窄带剖面 (深度随速率), 带宽≈100km, 面积占比≈1%
    # ---- v7.4: 用户报告"海沟深度不足(马里亚纳海沟-11000m vs 现-6400m), 且宽度偏窄(2-3px)"
    #   (1) tr_prof 基础: -(5700+700*spn) → -(6800+900*spn), 基础底深 +1100m; 叠加高斯 -2800 (原-2600)
    #       最快速率 spn≈1.0 → 理论最深 = -(6800+900) - 2800 = -10500m, 接近马里亚纳-11000m
    #   (2) 高斯 σ: 0.8 → 1.0 (更深沟的水平宽度展宽 ~25%, 不再是2-3px细线, 视觉上更明显)
    #   (3) 应用门限 d_tr < 1.3 → 1.6 (海沟覆盖的范围再宽一圈)
    tr_prof = -(6800.0 + 900.0 * spn) - 2800.0 * np.exp(-(d_tr / 1.0) ** 2)
    e_oc = np.where(oceanm & (d_tr < 1.6), np.minimum(e_oc, tr_prof), e_oc)
    # 转换断层脊/谷: 强化洋中脊的双侧扩张纹理
    # ---- v7.4: 用户报告"洋中脊两侧对称扩张纹理不明显, 缺乏转换断层切错"
    #   原正峰 +260m → 双极: 脊轴 +420m (断块抬升) + 转换沟谷 −280m (相邻段切错下陷)
    #   视觉效果: 洋中脊不再是单条平滑蜿蜒浅带, 而呈"脊-谷-脊"交替的 锯齿状扩张中心
    #   (匹配地球洋中脊被转换断层 每50-200km 横切错断的真实形态)
    e_oc += 420.0 * f1 * np.exp(-(d_tf / 2.8) ** 2) \
           - 280.0 * (1.0 - f1) * np.exp(-(d_tf / 1.9) ** 2)
    # 热点岛链: 高门槛只抬热点核心 (尾迹弱区不动, 避免大范围浅台地)
    vb = gaussian_filter(vmem.reshape(T, T), 1.3)
    if vb.max() > 0:
        vbn = vb / vb.max()
        e_oc += 7200.0 * np.where(vbn > 0.35, np.tanh(2.8 * vbn), 0.0)

    # ---- 陆地: 基底 + 造山带(M2碎片主轴延展+对齐互连) + 失败裂谷(M2) + 前陆盆地 ----
    o_s = max(float(np.quantile(orog[orog > 0], 0.85)), 1e-3) if (orog > 0).any() else 1.0
    # M2 造山带延展: orog 强核碎片 → 沿主轴 stamp 细长带(×1.6延展) +
    # 对齐碎片(轴向<35°/连线<50°/距20-90px)互连 → 安第斯型连续带状造山带
    # (orog 本身是滑动涂抹场, 直接阈值只有 AR≈2 的 blob; 碎片主轴方案保证 AR>3)
    belt = np.zeros((T, T), np.float32)
    orog_ext = orog.copy()
    if (orog > 0).any():
        lab_o, _ = ndi_label(orog > 0.3 * o_s)
        szo = np.bincount(lab_o.ravel()); szo[0] = 0
        frags = []
        for i in np.nonzero(szo >= 100)[0]:
            ys, xs = np.nonzero(lab_o == i)
            if crustm[ys, xs].mean() < 0.3:
                continue                          # 碎片需基本在陆内
            pts = np.stack([ys - ys.mean(), xs - xs.mean()])
            lam, vec = np.linalg.eigh(np.cov(pts))
            if lam[0] < 1e-9 or lam[1] < 1e-9:
                continue
            if lam[1] < 1.8 ** 2 * max(lam[0], 1e-9):
                continue                          # blob 碎片(AR<1.8)走向不可靠
            v = vec[:, 1]                         # 主方向
            fl = 2.5 * np.sqrt(lam[1])
            frags.append(dict(y=float(ys.mean()), x=float(xs.mean()),
                              ux=float(v[1]), uy=float(v[0]),
                              L=fl,
                              W=2.0 * np.sqrt(max(lam[0], 1.0)),
                              ey1=float(ys.mean() - v[0] * fl * 0.45),
                              ex1=float(xs.mean() - v[1] * fl * 0.45),
                              ey2=float(ys.mean() + v[0] * fl * 0.45),
                              ex2=float(xs.mean() + v[1] * fl * 0.45)))
        # M2 stamp 策略: 碎片按大小降序盖章, 近邻(<35px)跳过 → 聚类只留主轴
        # (多条 stamp 交叉重叠会把带局部叠成 AR≈1.5 的 blob)
        frags.sort(key=lambda f: -(f["L"] * f["W"]))
        stamped = []
        for f in frags:
            if any(np.hypot(f["y"] - cy, f["x"] - cx) < 35.0
                   for cy, cx in stamped):
                continue
            _stamp_belt(belt, f["y"], f["x"], f["ux"], f["uy"],
                        max(1.8 * f["L"], 30.0),
                        float(np.clip(f["W"], 3.5, 5.5)))
            stamped.append((f["y"], f["x"]))
        # M2 端点对端点链式连接 (每碎片最多 2 条):
        # 连最近端点对 → 共线链自然延伸, 平行侧邻碎片连线⊥轴自动失效
        # (中心连线会把平行碎片焊成宽带 → AR 崩塌); 连接段只填带间空隙
        cos35 = float(np.cos(np.radians(35.0)))
        cos50 = float(np.cos(np.radians(50.0)))
        pairs = []
        for ai in range(len(frags)):
            for bi in range(ai + 1, len(frags)):
                fa, fb = frags[ai], frags[bi]
                if abs(fa["ux"] * fb["ux"] + fa["uy"] * fb["uy"]) < cos35:
                    continue                      # 轴向不平行
                best = None
                for (ay, ax) in ((fa["ey1"], fa["ex1"]), (fa["ey2"], fa["ex2"])):
                    for (by, bx) in ((fb["ey1"], fb["ex1"]), (fb["ey2"], fb["ex2"])):
                        d = float(np.hypot(bx - ax, by - ay))
                        if best is None or d < best[0]:
                            best = (d, ay, ax, by, bx)
                dist, ay, ax, by, bx = best
                if not (6.0 < dist < 90.0):
                    continue
                cux, cuy = (bx - ax) / dist, (by - ay) / dist
                if (abs(cux * fa["ux"] + cuy * fa["uy"]) < cos50 or
                        abs(cux * fb["ux"] + cuy * fb["uy"]) < cos50):
                    continue                      # 端点连线与轴夹角过大
                score = (1.0 - dist / 100.0) * \
                    abs(cux * fa["ux"] + cuy * fa["uy"])
                pairs.append((score, ai, bi, cux, cuy, dist,
                              (ay + by) / 2.0, (ax + bx) / 2.0))
        pairs.sort(key=lambda p: -p[0])
        deg = [0] * len(frags)
        for score, ai, bi, cux, cuy, dist, my, mx in pairs:
            if deg[ai] >= 2 or deg[bi] >= 2:
                continue
            tmp = np.zeros_like(belt)
            _stamp_belt(tmp, my, mx, cux, cuy,
                        dist / 2.0 + 4.0, 3.6, val=0.8)
            belt = np.maximum(belt, tmp * (belt < 0.30))
            deg[ai] += 1; deg[bi] += 1
        belt *= crustm                            # 山带仅限陆内
        # M2 山带/散山二分: 带内 = max(碰撞峰值, 1.15·o_s 基座) → 连续带状山
        # (基座需≥1.15: σ3平滑+峰值因子低谷(0.72)会 otherwise 把带切成碎段);
        # 带外 = min(orog×0.28, 0.38·o_s) 完全不成山 (orog 最强 1.2=6·o_s,
        # ×0.28 仍 1.7·o_s 会成 blob; 散山由 generate_world 板内造山负责)
        inside = belt > 0.35
        orog_ext = np.where(
            inside,
            np.maximum(orog, np.float32(1.15 * o_s)),
            np.minimum(orog * 0.28, np.float32(0.38 * o_s))).astype(np.float32)
    # ---- M6 克拉通地块 (西伯利亚/加拿大地盾/刚果型): 古稳定核 → 内陆平原 ----
    # (地球内陆低地占陆 15.4% vs 生成图 3-7% → M5 大流域盆地材料匮乏的结构根因)
    # 稳定判据: 陆内(距海>5px≈400km) ∧ 无山带 ∧ 无裂谷带 ∧ |lat|<75°;
    # 造山阈值自适应 (严格 0.30·o_s 不足时取 quiet 区低半) 保底材料量
    lat_T = 90.0 - (ii + 0.5) * (180.0 / T)
    div_sm = gaussian_filter((div_tot > 12.0).astype(np.float32), 2.0)
    d_inland = distance_transform_edt(crustm).astype(np.float32)  # 陆地→海距
    quiet = crustm & (belt < 0.25) & (d_inland > 5.0) \
            & (div_sm < 0.35) & (np.abs(lat_T) < 75.0)
    stable = quiet & (orog_ext < 0.30 * o_s)
    n_land_T = max(int(crustm.sum()), 1)
    if (stable.sum() < 0.02 * n_land_T) and quiet.any():
        thr = float(np.quantile(orog_ext[quiet], 0.50))
        stable = quiet & (orog_ext < thr)
    crat_m = np.zeros((T, T), bool)
    lab_s, n_s = ndi_label(stable)
    if n_s:
        szs = np.bincount(lab_s.ravel()); szs[0] = 0
        for li in np.argsort(szs)[::-1][:3]:            # top3 大连通域
            if szs[li] < 0.012 * n_land_T:
                continue
            crat_m |= (lab_s == li)
    w_c = np.zeros((T, T), np.float32)
    n_craton = 0
    if crat_m.any():
        crat_m = binary_closing(crat_m, iterations=2) & crustm
        if crat_m.sum() >= 0.012 * n_land_T:
            w_c = gaussian_filter(crat_m.astype(np.float32), 2.5)
            w_c = np.clip(w_c / 0.60, 0.0, 1.0)          # 边缘 feather
            orog_ext = (orog_ext * (1.0 - 0.85 * w_c)).astype(np.float32)
            n_craton = int((w_c > 0.5).sum())
    mt = np.tanh(gaussian_filter(orog_ext, 3.0) / o_s)
    e_ld = 220.0 + 430.0 * np.clip(f1, 0, 1) \
           + 4900.0 * mt * (0.72 + 0.56 * np.clip(f3, 0, 1))
    # M2 前陆盆地: 紧贴延展带外缘 (恒河平原型, 幅度 620→800)
    wide = gaussian_filter(orog_ext, 8.0); core = gaussian_filter(orog_ext, 2.5)
    fore = np.clip(wide - 2.2 * core, 0, None)
    if fore.max() > 0:
        e_ld -= 800.0 * np.tanh(2.0 * fore / fore.max()) \
                * (core < 0.35 * max(core.max(), 1e-6)) * crustm
    # M6 克拉通平原: 低起伏台地 (150-380m, 亚马逊/西西伯利亚型基底)
    if n_craton:
        crat_fbm = gaussian_filter(_fbm(T, rng, 4.0, 3), 1.5)
        z_crat = 150.0 + 230.0 * np.clip(crat_fbm + 0.5, 0.0, 1.0)
        e_ld = (e_ld * (1.0 - 0.80 * w_c) + z_crat * (0.80 * w_c)) \
            .astype(np.float32)
    # M2 失败裂谷 (东非大裂谷型/aulacogen): 曾强烈离散但未裂开的陆内带
    # → 线性谷底 + 双侧肩部抬升 + 轴部火山链 (乞力马扎罗型)
    # rift 场输出 → generate_world 在直方图终检后重新下切 (GAN/重排会抹掉骨架版)
    rift = np.zeros((T, T), np.float32)
    rift_core = gaussian_filter(
        ((div_tot > 15.0) & crustm).astype(np.float32), 2.0) > 0.42
    n_rift = 0
    if rift_core.any():
        d_ax = distance_transform_edt(~rift_core).astype(np.float32)
        inside = d_ax < 6.0
        e_ld = e_ld - 900.0 * np.exp(-(d_ax / 1.4) ** 2) * inside * crustm
        e_ld = e_ld + 380.0 * np.exp(-((d_ax - 2.5) / 1.6) ** 2) * inside * crustm
        e_ld = e_ld + 2500.0 * (d_ax < 1.6) * (f1 > 0.55) * crustm \
                * (0.6 + 0.4 * np.clip(f3, 0, 1))   # 稀疏火山峰(乞力马扎罗型)
        # rift 输出 = 轴剖面场 (谷形直接编码: 轴线1→两侧高斯衰减),
        # 下游直接乘深度即可, 不再依赖宽域阈值
        rift = (np.exp(-(d_ax / 1.6) ** 2)
                * rift_core.astype(np.float32))
        _, n_rift = ndi_label(rift_core)
        n_rift = int(sum(1 for s in np.unique(rift_core[rift_core]) if
                         (rift_core == s).sum() > 30))

    # ---- 海岸混合: 连续概率场 (消二值矩形) ----
    cont = gaussian_filter(crustm.astype(np.float32), 2.2)
    tc = np.clip(cont * 1.7 - 0.35, 0, 1)
    e = e_oc * (1 - tc) + e_ld * tc
    elev = np.clip(e, -10500, 8900).astype(np.float32)

    # ---- 构造条件场 (格距单位与真实地球训练一致) ----
    b = np.zeros((T, T), bool)
    b[:, :-1] |= P2[:, 1:] != P2[:, :-1]
    b[:-1, :] |= P2[1:, :] != P2[:-1, :]
    tect = proximity_fields([b, evF["trench"] | evF["coll"] | evF["arc"],
                             evF["ridge"] | evF["riftland"], evF["trans"]],
                            cell_arcmin=10800.0 / T)
    if verbose:
        om = float(np.median(age_f.reshape(T, T)[~crustm])) if (~crustm).any() else 0.0
        np_pass = int((margin == 1).sum()); np_act = int((margin == 2).sum())
        # M2: 延展山带统计 (最大山带连通域长宽比)
        belt_ar = 0.0
        if belt.any():
            lab_b, nb = ndi_label(belt > 0.5)
            if nb:
                szb = np.bincount(lab_b.ravel()); szb[0] = 0
                lid = int(szb.argmax())
                ys, xs = np.nonzero(lab_b == lid)
                if ys.size > 30:
                    pts = np.stack([ys - ys.mean(), xs - xs.mean()])
                    lam = np.linalg.eigvalsh(np.cov(pts))
                    belt_ar = float(np.sqrt(max(lam[1], 1e-9) / max(lam[0], 1e-9)))
        print(f"  终局: 陆地 {crustm.mean() * 100:.1f}% | 峰 {elev.max():.0f}m | "
              f"沟 {elev.min():.0f}m | 洋龄中位 {om:.0f} Myr")
        print(f"  v3.0 旋回: 超大陆峰值连通 {pangea_frac[0] * 100:.0f}% | "
              f"微大陆 {n_micro} 块(陆壳 {land_micro}px) | "
              f"被动边缘 {np_pass}px | 主动边缘 {np_act}px")
        print(f"  M2 山带: 延展带 {int((belt > 0.5).sum())}px | 主带AR {belt_ar:.1f} | "
              f"失败裂谷 {n_rift} 条")
        if n_craton:
            print(f"  M6 克拉通: {n_craton}px "
                  f"({n_craton / n_land_T * 100:.1f}% 陆) → 内陆平原基底")
    global _LAST_OROG
    _LAST_OROG = orog.copy()
    return elev, tect, crustm, margin, belt, rift


def main():
    """``plate_sim.py`` 命令行入口 —— 450 Myr 板块演化预览。

    解析 ``--seed / --T`` 执行 :func:`run_plate_sim`，将粗 DEM 与 4 张构造场
    渲染为 5 联预览图保存到 ``terrain/output/plate_sim_preview.png``。
    ``python -m terrain.core.plate_sim`` 以模块方式运行时经由此入口进入。
    """
    import os
    import argparse
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--T", type=int, default=512)
    a = ap.parse_args()
    _file_dir = os.path.dirname(os.path.abspath(__file__))
    _dir = (os.path.dirname(_file_dir)
            if os.path.basename(_file_dir) in ("core", "scripts") else _file_dir)
    os.makedirs(os.path.join(_dir, "cache"), exist_ok=True)
    os.makedirs(os.path.join(_dir, "output"), exist_ok=True)
    elev, tect, landm, margin, belt, rift = run_plate_sim(T=a.T, seed=a.seed)
    np.savez_compressed(os.path.join(_dir, "cache", f"plate_sim_v3_T{a.T}_seed{a.seed}.npz"),
                        elev=elev, tect=tect, crust=landm, margin=margin,
                        belt=belt, rift=rift)
    from .terrain_engine import m2v, make_terrain_cmap, render_snow_field
    lats = -90.0 + (np.arange(a.T) + 0.5) * (180.0 / a.T)
    fig, ax = plt.subplots(1, 5, figsize=(26, 5.2))
    ax[0].imshow(render_snow_field(m2v(elev), lats, make_terrain_cmap()))
    ax[0].set_title(f"450 Myr 演化粗DEM (陆地 {100 * landm.mean():.0f}%)")
    for i, t in enumerate(["t0 边界邻近", "t1 俯冲/碰撞", "t2 离散(脊/裂谷)", "t3 转换"]):
        ax[i + 1].imshow(tect[i], cmap="inferno", vmin=0, vmax=1)
        ax[i + 1].set_title(t)
    for a_ in ax:
        a_.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(_dir, "output", "plate_sim_preview.png"), dpi=130)
    plt.show()


if __name__ == "__main__":
    main()