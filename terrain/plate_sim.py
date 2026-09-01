# -*- coding: utf-8 -*-
# ======================================================================
# plate_sim.py v2.1 —— 前向板块演化模拟器(修复版, 450 Myr)
#  0) 半球不对称先验: 大陆向随机"陆半球"集中 (北陆南水型)
#  1) 裂谷: 衰减型记忆(仅持续扩张~37Myr处裂开) + 陆地占比保护 → 大陆不被蚕食
#  2) 沟/弧/脊/转换: 终局"现势边界"形态学(强度∝收敛速率) → 无扫掠环/浅海化
#  3) 造山带: 保留历史累积 orog, 归一化后 tanh → 少数高峰的层级
#  4) 连续大陆概率场混合海岸 → 矩形消失
#  5) 构造条件场宽度与训练一致(cell_arcmin=8 → type半径=2格)
# ======================================================================
import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt
from tectonics import proximity_fields


def _rot(axis, ang):
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
    noise = np.zeros((T, T), np.float32); amp, tot, f = 1.0, 0.0, freq
    for _ in range(octaves):
        n = rng.randn(T, T).astype(np.float32)
        n = gaussian_filter(n, max(T / (2.0 * f), 0.6)); n /= (n.std() + 1e-8)
        noise += amp * n; tot += amp; amp *= 0.5; f *= 2.0
    return noise / tot


def _dist(m):
    return distance_transform_edt(~m).astype(np.float32) if m.any() else np.float32(1e4)


def _classify(pl, cr, ag2d, V, r, T):
    """现势板块边界分类 → 掩膜字典(沟/弧/碰撞/脊/陆内离散/转换) + 速率场"""
    if r.ndim == 2:                     # 兼容展平输入 (3, N) → (3, T, T)
        r = r.reshape(3, T, T)
    Vg = V.reshape(3, T, T)
    out = {k: np.zeros((T, T), bool) for k in
           ("trench", "arc", "coll", "ridge", "riftland", "trans")}
    spd = np.zeros((T, T), np.float32)

    def put(mk, sel, cell):
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
                  dt=0.625, steps=720, verbose=True, land_target=0.30):
    rng = np.random.RandomState(seed)
    ii, jj = np.mgrid[0:T, 0:T].astype(np.float64)
    lat_g = (0.5 - (ii + 0.5) / T) * np.pi
    lon_g = ((jj + 0.5) / T) * 2 * np.pi
    r = np.stack([np.cos(lat_g) * np.cos(lon_g),
                  np.cos(lat_g) * np.sin(lon_g), np.sin(lat_g)])
    r_flat = r.reshape(3, -1); N = T * T

    ctr = rng.randn(K, 3); ctr /= np.linalg.norm(ctr, axis=1, keepdims=True)
    plate = np.tensordot(ctr, r_flat, axes=(1, 0)).argmax(0).reshape(T, T).astype(np.int32)
    axes = rng.randn(K, 3); axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    omg = rng.uniform(0.0012, 0.005, K) * rng.choice([-1.0, 1.0], K)

    # ---- 大陆播种: 每大陆 = 2~5 个重叠圆盘的不规则并集, 迭代至 land_target ----
    # 半球不对称先验: 随机选一个"陆半球"(地球为北), 72% 大陆向其集中
    pref_north = rng.rand() < 0.5
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
        # 40% 概率强制高纬(极地大陆), 极地带也偏向陆半球
        if rng.rand() < 0.4:
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
    age = np.where(crust == 1, 0.0, 60.0).astype(np.float32)

    hs_rc = [(int(rng.uniform(0.06, 0.94) * T), int(rng.rand() * T)) for _ in range(n_hs)]
    vmem = np.zeros(N, np.float32)
    orog = np.zeros((T, T), np.float32)
    div_mem = np.zeros((T, T), np.float32)
    V = np.zeros((3, N), np.float32)
    plate_f, crust_f, age_f = plate.ravel(), crust.ravel(), age.ravel()

    for step in range(steps):
        # ---- 后向映射搬运: 每目的地取逆旋转源点, 面积守恒, 大陆不撕裂 ----
        new_plate = np.full(N, -1, np.int32)
        new_crust = np.zeros(N, np.int8)
        new_age = np.zeros(N, np.float32)
        new_vmem = np.zeros(N, np.float32)
        claim = np.full(N, -1.0, np.float32)
        for k in range(K):
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
        P2 = new_plate.reshape(T, T); C2 = new_crust.reshape(T, T)

        for k in range(K):
            m = P2 == k
            if m.any():
                V[:, m.ravel()] = _cross(axes[k], r_flat[:, m.ravel()]) * omg[k]

        ev = _classify(P2, C2, new_age.reshape(T, T), V, r_flat, T)
        if ev["coll"].any():
            np.add.at(orog, np.nonzero(ev["coll"]), ev["spd"][ev["coll"]] * dt)
        div_mem *= 0.985
        if ev["riftland"].any():
            div_mem[ev["riftland"]] += dt

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
            print(f"  t = {(step + 1) * dt:.0f} Myr | 陆地 {crust_f.mean() * 100:.1f}%")

    # ==================== 终局装配 ====================
    crustm = (crust_f.reshape(T, T) == 1)
    f1, f2, f3 = _fbm(T, rng, 6.0, 4), _fbm(T, rng, 9.0, 4), _fbm(T, rng, 13.0, 3)

    evF = _classify(P2, new_crust.reshape(T, T), new_age.reshape(T, T), V, r_flat, T)
    d_tr, d_arc = _dist(evF["trench"]), _dist(evF["arc"])
    d_rg, d_tf = _dist(evF["ridge"]), _dist(evF["trans"])
    d_land = distance_transform_edt(~crustm).astype(np.float32)

    # ---- 现势边界速率场 (海沟深度随俯冲速率) ----
    if (evF["spd"] > 0).any():
        q90 = max(float(np.quantile(evF["spd"][evF["spd"] > 0], 0.9)), 1e-6)
        spn = np.clip(evF["spd"] / q90, 0, 1.5)
    else:
        spn = np.zeros((T, T), np.float32)

    # ---- 海洋: 等效俯冲再循环年龄 → 洋盆回到训练分布(GAN不再外推) ----
    age_eff = np.clip(age_f.reshape(T, T), 0.0, rng.uniform(55.0, 85.0))
    base_oc = -(2600.0 + 360.0 * np.sqrt(age_eff))          # 洋盆中位 ≈ -5400m
    e_oc = base_oc + 260.0 * f2
    # 大陆架/陆坡: 海岸处抬至 ≈-130m (真实大陆架基准)
    shelf_amp = -130.0 - float(np.median(base_oc))
    e_oc += shelf_amp * np.exp(-(d_land / 6.0) ** 2) * (~crustm)
    # 海沟 (贴陆缘, 深度随俯冲速率)
    e_oc -= (2400.0 + 2000.0 * spn) * np.exp(-(d_tr / 1.9) ** 2) * (~crustm)
    # 弧后盆地
    e_oc -= 420.0 * np.exp(-((d_tr - 9.0) / 3.5) ** 2) * (~crustm)
    # 火山弧: 双层结构 → 弧段状岛链 (宽基座 + 低频门控窄岛峰, 消"圆点"感)
    e_oc += 2200.0 * np.exp(-(d_arc / 2.6) ** 2)
    g_arc = _fbm(T, rng, 7.0, 2)
    e_oc += 4600.0 * np.exp(-(d_arc / 1.2) ** 2) * (g_arc > 0.05)
    # 洋中脊: 宽缓隆起 + 轴谷
    e_oc += (2300.0 + 320.0 * f2) * np.exp(-(d_rg / 4.5) ** 2) \
            - 380.0 * np.exp(-(d_rg / 1.5) ** 2)
    # 转换断层脊
    e_oc += 260.0 * f1 * np.exp(-(d_tf / 2.6) ** 2)
    # 热点岛链: 只在有记忆处抬升, 峰值可露头
    vb = gaussian_filter(vmem.reshape(T, T), 1.3)
    if vb.max() > 0:
        vbn = vb / vb.max()
        e_oc += 9300.0 * np.where(vbn > 0.03, np.tanh(2.2 * vbn), 0.0)

    # ---- 陆地: 基底 + 碰撞造山带(历史累积) + 前陆盆地(与山成对) ----
    o_s = max(float(np.quantile(orog[orog > 0], 0.85)), 1e-3) if (orog > 0).any() else 1.0
    mt = np.tanh(gaussian_filter(orog, 3.0) / o_s)
    e_ld = 220.0 + 430.0 * np.clip(f1, 0, 1) \
           + 4900.0 * mt * (0.72 + 0.56 * np.clip(f3, 0, 1))
    wide = gaussian_filter(orog, 8.0); core = gaussian_filter(orog, 2.5)
    fore = np.clip(wide - 2.2 * core, 0, None)
    if fore.max() > 0:
        e_ld -= 620.0 * np.tanh(2.0 * fore / fore.max()) \
                * (core < 0.35 * max(core.max(), 1e-6))

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
        om = float(np.median(age_eff[~crustm])) if (~crustm).any() else 0.0
        print(f"  终局: 陆地 {crustm.mean() * 100:.1f}% | 峰 {elev.max():.0f}m | "
              f"沟 {elev.min():.0f}m | 洋龄中位 {om:.0f} Myr")
    return elev, tect, crustm


if __name__ == "__main__":
    import os
    import argparse
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--T", type=int, default=512)
    a = ap.parse_args()
    _dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(_dir, "cache"), exist_ok=True)
    os.makedirs(os.path.join(_dir, "output"), exist_ok=True)
    elev, tect, landm = run_plate_sim(T=a.T, seed=a.seed)
    np.savez_compressed(os.path.join(_dir, "cache", f"plate_sim_v2_T{a.T}_seed{a.seed}.npz"),
                        elev=elev, tect=tect, crust=landm)
    from terrain_engine import m2v, make_terrain_cmap, render_snow_field
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