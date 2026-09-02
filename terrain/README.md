# 幻想世界生成器 (terrain/) —— 使用文档

板块演化 + 神经网络混合驱动的随机世界地图生成器。
物理规则层负责地质/气候/水文的因果骨架，GAN 负责像素级地形纹理。

## 目录结构

```
terrain/
├── README.md              ← 本文档 (使用说明)
├── TECHNICAL.md           ← 技术文档 (算法原理与关键参数)
│
├── generate_world.py      ← 主入口: 生成完整世界
├── rerender.py            ← 仅重渲染 (改渲染参数后用, 不重算地形)
├── plate_sim.py           ← 板块演化模拟器 (可单独调试)
├── tectonics.py           ← 真实板块边界 → 构造条件场 (GAN 训练数据)
├── terrain_engine.py      ← 核心: 物理工具/气候/侵蚀/GAN 训练
│
├── data/                  ← 原始数据 (重训 GAN 需要)
│   ├── ETOPO1_Ice_g_gmt4.grd        # 全球 DEM (~890MB, 公有领域)
│   └── PB2002_boundaries.json       # Bird 2003 板块边界
├── checkpoints/           ← 模型权重
│   └── best_terrain_gan_v5.pth       # 12 通道条件 GAN (~44MB)
├── cache/                 ← 可再生缓存 (删除后会自动重建, 只是变慢)
│   ├── plate_sim_v22_T512_seed*.npz # 板块演化结果 (同参数复用)
│   └── tect_cache_*.npz              # 构造条件场
└── output/                ← 生成结果
    ├── world_map.png                 # 最终渲染图 (生物群系/河湖/雪线)
    ├── world_dem.npy                 # 全球高程 (米, 2048² float32)
    └── world_skeleton.png            # 构造骨架预览 (板块+噪声, GAN 之前)
```

## 快速开始

```bash
# 项目根目录运行 (脚本内部全部使用基于 __file__ 的绝对路径, 任意目录均可)
.venv/bin/python terrain/generate_world.py --seed 0 --size 2048
```

首次运行约 2~4 分钟（板块演化 450 Myr 模拟占大头，之后同 seed 走缓存秒级完成）。
结果写入 `terrain/output/`。

## 常用任务

| 任务 | 命令 |
|------|------|
| 换一个世界 | `--seed 任意整数`（每删一次 cache 中的对应 npz 才会重算） |
| 更大分辨率 | `--size 4096`（4 倍耗时） |
| 只调渲染不改地形 | `.venv/bin/python terrain/rerender.py` |
| 单独调试板块模拟 | `.venv/bin/python terrain/plate_sim.py --seed 7 --T 512` |
| 批量生成多种子 | `.venv/bin/python terrain/batch_gen.py --seeds 1 2 3 4 5` |
| 地理指标检验 | `.venv/bin/python terrain/analyze_worlds.py`（与 ETOPO1 真实地球同口径对比，输出 analysis_results.json） |
| 生成对比拼图 | `.venv/bin/python terrain/make_montage.py` |
| 重训 GAN | 删除 `checkpoints/best_terrain_gan_v5.pth` 后运行 `terrain_engine.py`（需要 data/ 中的 ETOPO1，约 2.5-4h on MPS） |

## 缓存管理

- **板块缓存** `cache/plate_sim_v22_T{分辨率}_seed{种子}.npz`：同一 seed+tect-res 直接复用。
  想让某 seed 的板块重新演化（如修改了 plate_sim.py），删除对应文件即可。
- **构造场缓存** `tect_cache_*.npz`：分辨率不匹配时自动重算。
- **全部缓存均可删**：删除后只是变慢，不影响正确性。

## 输出说明

- `world_dem.npy`：米制高程，正值陆地/负值海洋，形状 (S, S)，行=纬度(北→南)、列=经度。
  后续任何用途（如 3D 可视化、游戏地图）直接读取此文件。
- `world_map.png`：v6.5 气候渲染——Whittaker 生物群系着色、可见河流/湖泊、
  纬度+温度双因素雪线、极地海冰、NW 光照山体阴影。
- `world_skeleton.png`：域扭曲后的构造骨架（GAN 细化前），用于检查大陆轮廓/山脉走向。

## 常见问题

**Q: 代码疑似被 IDE 旧标签页覆盖（功能异常退化）**
A: PyCharm 等编辑器会把文件内容缓存在标签页中，外部修改磁盘后旧缓冲区保存一次就会覆盖新代码。
检查/恢复方法（项目已建 git 快照）：
```bash
git status                    # 若 terrain/*.py 显示红色 modified 即被覆盖
git checkout -- terrain/      # 一键恢复到最近提交的验证版本
git log --oneline             # 查看历史提交
```
预防：外部工具修改代码后，在 IDE 中执行 File → Reload All from Disk。

**Q: 提示 "请先用 v6.3 重训 (12通道)"**
A: 检查点版本不匹配。用 `checkpoints/` 中现成的权重，或按 TECHNICAL.md 重训。

**Q: 没有 ETOPO1 数据能跑吗？**
A: 生成世界不需要（只需 GAN 权重 + 板块模拟）。仅重训 GAN 时才需要；
缺失时会退化为合成地形训练（质量下降，仅供流程验证）。

**Q: 训练数据下载失败？**
A: ETOPO1 支持 `DEM_PATH=/路径/dem.npy` 环境变量指定任意全球米制高程 npy；
板块边界支持 `TECT_JSON` 环境变量。

**Q: 设备要求？**
A: Apple Silicon (MPS) 或 CUDA 均可，纯 CPU 也能跑（GAN 瓦片推理较慢）。

## 依赖

torch / numpy / scipy / matplotlib / PIL（均已在项目 .venv 中）。
