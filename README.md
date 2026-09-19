# 幻想世界生成器 (terrain/) —— 使用文档 · v12

板块演化 + 神经网络混合驱动的随机世界地图生成器。物理规则层负责地质 / 气候 / 水文的因果骨架，GAN 负责像素级地形纹理；同时提供 **4 面板可视化**、**单文件离线 HTML 交互报告**与 **C++ 原生加速**（Numba 自动回退）。

> **v12 升级要点**：删除 `terrain/` 根目录 10 个兼容薄封装与 60+ 个一次性诊断脚本，统一使用 `terrain.core.*` / `terrain.scripts.*` 导入；文档合并精简（唯一使用文档 = 本 README，技术文档 = `terrain/docs/TECHNICAL.md`）。
>
> **v10–v11 升级要点**：**C++ 原生加速模块 `terrain/native/`**（pybind11，15 个内核：降水扫描 / 山体遮蔽 / 中值-众数滤波 / 流量累积 / 洼地填充 / 河流蜿蜒等，全部与 Numba 逐位一致）；v11 引入 OpenMP 并行（`smooth_small_patches` 替代 9×ndi_label 碎片清理，`render_world` 28.52s → 24.99s）。
>
> **v9.1 升级要点**：SRNet（ESRGAN）超分网络正式训练完成（300 epoch / 60 mosaics / 梯度保真优于 Bicubic）、`DiffusionPipeline` 集成 SRNet 并修复权重加载逻辑（`use_sr=True` 时自动加载 `sr_net.pt`）、新增训练脚本 `terrain/scripts/_run_srnet_train.py` 与运行脚本 `terrain/scripts/_run_pipeline_sr.py`。
>
> **v9.0 升级要点**：气候系统 v7.1–v7.4（大陆度指数 / 极地冰盖 / 泰加林修复）、河流水系 v8.3（骨架提取 + 沿程单调修复 / Hack 定律蜿蜒 / 牛轭湖 / 连续蓝度渲染）、新增 **深度学习模块** **`terrain/dl/`**（DDPM → LDM → 物理监督 → 大规模推理 → 端到端管线 + ESRGAN 超分）、新增 **监督反思模块** **`terrain/supervisor/`**（审计 → 反思 → 学习日志 → 自动注入 overrides）、安全加固（`torch.load(weights_only=True)` / URL scheme 白名单）、测试套件 33 → **512 项**。

## 目录结构

> 【核心源码位于 terrain/core/ 与 terrain/modules/；v12 起已删除 terrain/ 根目录的兼容薄封装，请勿再引用旧路径】

```
Fantasy_World_Simulator/
├── README.md              ← 本文档 (安装 + 使用, GitHub 首页展示)
├── main.py                ← 项目入口: 双模式 (GUI / CLI 菜单 / 子命令直跑)
│
├── terrain/
│   ├── docs/              ← 文档目录
│   │   └── TECHNICAL.md       (架构 / 算法 / 接口 / 参数, 面向调参; 另有历史 IMPROVEMENT_PLAN.md)
│   │
│   ├── core/              ← 核心源码 (8 个模块, 请在这里编辑)
│   │   ├── __init__.py
│   │   ├── plate_sim.py           板块演化模拟器 v3.0 (可单独调试)
│   │   ├── tectonics.py           真实板块边界 → 构造条件场 (GAN 训练数据)
│   │   ├── terrain_engine.py      核心: 物理工具 / 气候 / 侵蚀 / GAN 训练 / v8.3 河流系统
│   │   ├── generate_world.py      主入口: 生成完整世界 (DEM + 渲染图)
│   │   ├── rerender.py            仅重渲染 (改渲染参数后用, 不重算地形)
│   │   ├── geo_context.py         WorldGeoContext 单例: 坐标查询 / KPI 统计 / 多边形裁切
│   │   └── visualize_context.py   4 面板 PNG + 交互 HTML 报告 (探针/多边形/剖面联动)
│   │
│   ├── modules/           ← 模块化引擎 (可独立优化, 见 modules/ARCHITECTURE.md)
│   │   ├── base.py / engine.py        WorldContext 共享上下文 + WorldEngine 编排器
│   │   ├── terrain_module.py          地形 (高程 / 坡度 / 地貌 / 地形图)
│   │   ├── hydro_module.py            水系 (流量累积 / 河流 / 湖泊 / 水系图)
│   │   ├── temperature_module.py      气温 (大陆度 / 洋流 / 温度场 / 气温图)
│   │   └── climate_module.py          气候 (降水 / 生物群系 / 气候图)
│   │
│   ├── native/            ← C++ 原生加速 (v10 起, pybind11, 15 内核)
│   │   ├── setup.py / src/            (common/climate/hydro/hydro_adv/temperature/bindings.cpp)
│   │   ├── fws_native.*.so            已内置免构建; 缺失时自动回退 Numba
│   │   └── tests/                     cmp_native.py / cmp_adv.py 对拍脚本
│   │
│   ├── dl/                ← 深度学习模块 (v9.0 新增, 5 阶段; v9.1 SRNet 训练完成)
│   │   ├── config.py / logger.py / metrics.py / evaluator.py   配置 / 日志 / 指标
│   │   ├── vae.py / unet.py / diffusion.py / ddim.py           VAE + DDPM/DDIM 扩散
│   │   ├── dataset.py / paired_dataset.py / train.py / train_ldm.py  训练
│   │   ├── infer.py / infer_ldm.py / large_infer.py            推理 (含大规模瓦片调度)
│   │   ├── physics_nets.py / physics_dataset.py / physics_metrics.py / train_physics.py / infer_physics.py  物理监督 (河网/雪线/海岸)
│   │   ├── sr_net.py        ESRGAN 超分辨率网络 (v9.1 训练完成: base=32, n_rrdb=3, CCA 注意力)
│   │   ├── multicond.py     多模态条件编码器 (草图/高度/群落)
│   │   └── pipeline.py      端到端集成管线 (v9.1 修复 SRNet 权重加载)
│   │
│   ├── supervisor/        ← 监督反思模块 (v9.0 新增)
│   │   ├── __main__.py / cli.py   CLI 入口 (审计 / 历史查询 / 重置 overrides)
│   │   ├── metrics.py             七大维度指标计算
│   │   ├── reflector.py           反思生成 + overrides 注入
│   │   ├── references.py          地球参考真值
│   │   └── learning_log.py        JSONL 学习日志 + 聚合统计
│   │
│   ├── scripts/           ← 批处理脚本 (v12 精简为 5 个, 请在这里编辑)
│   │   ├── __init__.py
│   │   ├── batch_gen.py         批量生成多种子
│   │   ├── analyze_worlds.py    地理指标检验 (vs 地球)
│   │   ├── make_montage.py      生成对比拼图
│   │   ├── _run_srnet_train.py  (v9.1) SRNet 超分网络训练 (300 epoch)
│   │   └── _run_pipeline_sr.py  (v9.1) DiffusionPipeline 运行 (启用 SRNet 超分)
│   │
│   ├── tests/                 ← 单元测试 (512 项, pytest)
│   │   ├── test_visualize_context.py   可视化 / HTML 报告 / geo_context (33 项)
│   │   ├── test_supervisor.py          监督反思模块 (32 项)
│   │   └── test_dl_phase1..5.py        深度学习 5 阶段 (447 项)
│   │
│   ├── data/                  ← 原始数据 (重训 GAN 需要; 生成世界不需要)
│   │   ├── ETOPO1_Ice_g_gmt4.grd        # 全球 DEM (~890MB, 公有领域)
│   │   └── PB2002_boundaries.json       # Bird 2003 板块边界
│   ├── checkpoints/           ← 模型权重
│   │   └── best_terrain_gan_v5.pth       # 12 通道条件 GAN (~44MB, 已内置可直接用)
│   ├── cache/                 ← 可再生缓存 (删除后会自动重建, 只是变慢)
│   │   ├── plate_sim_v3_T512_seed*.npz   # 板块演化结果 (同参数秒级复用)
│   │   └── tect_cache_*.npz              # 构造条件场
│   └── output/                ← 生成结果
│       ├── world_map.png                 # 最终气候渲染图 (生物群系 / 河湖 / 雪线)
│       ├── world_dem.npy                 # 全球高程 (米, S×S float32, 行=纬北→南 / 列=经)
│       ├── world_skeleton.png            # GAN 前的构造骨架
│       ├── geoctx_visual_seed{}.png      # 4 面板可视化 (高程/坡度/地理带/剖面)
│       └── geoctx_report_seed{}.html      # 🆕 单文件离线交互报告 (≈4 MB)
```

## 安装步骤

> 要求 **Python 3.11 \~ 3.14**（项目实测于 Python 3.14.6；Apple Silicon / x86-64 均可；torch / numpy / numba 需要匹配解释器版本）。

```bash
# 1) 克隆并进入项目根目录
git clone <your-repo-url> Fantasy_World_Simulator
cd Fantasy_World_Simulator

# 2) 创建虚拟环境 (推荐 venv, 不污染系统)
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# 或 Windows (PowerShell):  .venv\Scripts\Activate.ps1

# 3) 安装运行时依赖 (≈ 2.2 GB, 主要是 torch)
pip install --upgrade pip
pip install torch numpy scipy matplotlib Pillow  # 生成 + 可视化都需要
pip install pytest                                 # 跑单元测试 (可选, 开发用)

# 4) 快速自检 (生成 seed=1 的 2048² 世界)
# ⚠ 核心脚本含包内相对导入，必须用 `python -m` 方式调用（不能直接 `python terrain/core/xxx.py`）
python -m terrain.core.generate_world --seed 1 --size 2048

# （首次运行推荐）检查依赖并跑菜单 → python main.py
```

- **`data/ETOPO1`** **和板块边界 JSON**：生成世界**不需要**（仅重训 GAN 才需要）。缺失时重训会退化为合成地形训练。

- **`checkpoints/best_terrain_gan_v5.pth`**：项目已内置，首次运行直接可用；若被删除会触发自动重训（见 FAQ）。

## 快速开始

```bash
# 项目根目录运行 (脚本内部全部使用基于 __file__ 的绝对路径, 任意目录均可)
python -m terrain.core.generate_world --seed 0 --size 2048
```

首次运行约 **2\~4 分钟**（板块演化 450 Myr 模拟占大头；同 seed 再次运行走缓存 **< 10 秒**完成）。
结果写入 `terrain/output/`。

## 常用任务

⬇ 推荐使用 terrain/core/ 与 terrain/scripts/ 下的真实脚本（v12 已删除 terrain/ 根目录兼容薄封装）。

> **运行方式**：`terrain/core/*.py` 使用包内相对导入，**必须**以 `python -m terrain.core.xxx` 调用；
> `terrain/scripts/*.py` 自带 sys.path 注入，可直接 `python terrain/scripts/xxx.py`。

| 任务                                   | 命令                                                                                                                                                         |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 换一个世界                                | `python -m terrain.core.generate_world --seed 任意整数 --size 2048`                                                                                            |
| 更大分辨率 (4096²)                        | `--size 4096`（约 4 倍耗时, 16 倍显存）                                                                                                                             |
| **4 面板可视化 PNG**                      | `python -m terrain.core.visualize_context --seed 1`（高程 / 坡度 / 地理带 / 剖面）                                                                                    |
| **交互 HTML 报告（仅 HTML）**               | `python -m terrain.core.visualize_context --seed 1 --no-png --html`                                                                                        |
| **PNG + HTML 双产出 + 叠加元素**            | `python -m terrain.core.visualize_context --seed 1 --html --points "39,116,CityA,35,139,CityB" --poly "30,-5,30,35,45,35,45,-5" --profile "60,-10,25,125"` |
| 只调渲染不改地形                             | `python -m terrain.core.rerender`                                                                                                                          |
| **3D 模型导出 + 桌面预览**                   | `python -m terrain.core.export_3d --seed 1 --preview`（默认导出 OBJ+glTF、平面+球体；预览自动「海平面裁剪 + 自适应夸张 + 加密网格」，`--help` 查看全部选项）                                                     |
| **3D 平海导出（海岸锋利贴合）**               | `python -m terrain.core.export_3d --coast-cut --exagg-globe 80 --mode globe`（导出默认保留海底起伏，`--coast-cut` 压平海洋到海平面）                                                            |
| 生成并同步导出 3D                            | `python -m terrain.core.generate_world --seed 1 --size 2048 --export-3d`                                                                                  |
| 单独调试板块模拟                             | `python -m terrain.core.plate_sim --seed 7 --T 512`                                                                                                        |
| 批量生成多种子                              | `python terrain/scripts/batch_gen.py --seeds 1 2 3 4 5`                                                                                                    |
| 地理指标检验 (vs 地球)                       | `python terrain/scripts/analyze_worlds.py` → 输出 `analysis_results.json`                                                                                    |
| 生成对比拼图                               | `python terrain/scripts/make_montage.py`                                                                                                                   |
| 单元测试 (512 项)                         | `pytest terrain/tests/ -v`                                                                                                                                 |
| 监督审计 + 反思学习                          | `python -m terrain.supervisor --seed 42 --size-px 2048`                                                                                                    |
| 查询学习日志                               | `python -m terrain.supervisor --history 10`                                                                                                                |
| DL 模块训练 (扩散模型)                       | `python -m terrain.dl.train` (详见 TECHNICAL §21)                                                                                                            |
| **(v9.1) SRNet 超分训练**                | `python terrain/scripts/_run_srnet_train.py`（300 epoch / 约 67min / MPS）                                                                                    |
| **(v9.1) DiffusionPipeline + SRNet** | `python terrain/scripts/_run_pipeline_sr.py`（LDM 生成 → SRNet 4× 超分）                                                                                         |
| 重训 GAN                               | 删除 `checkpoints/best_terrain_gan_v5.pth` 后 `python -m terrain.core.terrain_engine`（需 `data/ETOPO1`, 2.5\~4h / MPS）                                         |

### 叠加元素的正确格式

> ⚠ 坐标必须是**纯数字**，禁止把中文地名写进坐标串。地名是**第 3 / 第 6 / 第 9 … 列**的可选 label。

```bash
# --points: (lat, lon[, name]) 三元组任意重复
--points "39.9,116.4,北京,35.7,139.7,东京,40.7,-74.0,NewYork"

# --poly: (lat, lon) 二元组重复 × N (必须是偶数个, ≥ 6, 自动闭合回起点)
--poly "30,-5, 30,35, 45,35, 45,-5"

# --profile: (latA, lonA, latB, lonB) 4 个数字
--profile "60,-10,25,125"
```

## 缓存管理

- **板块缓存** `cache/plate_sim_v3_T{分辨率}_seed{种子}.npz`：同一 seed + 分辨率直接复用（v3.0 超大陆旋回版缓存）。
  修改了 `plate_sim.py` 想让某 seed 重跑？删除对应 npz 文件即可。

- **构造场缓存** `tect_cache_*.npz`：分辨率或板块边界 JSON 版本不匹配时自动重算。

- **全部缓存均可删**：删除后只是变慢，不影响正确性。

## 输出说明

| 产物                          | 说明                                                                                                                                  |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `world_dem.npy`             | 米制高程 (正值=陆地 / 负值=海洋), 形状 `(S, S)` float32。行=纬度 北→南, 列=经度 −180→+180。**任何下游用途的标准输入**（3D 渲染 / 游戏地图 / 可视化等）。                            |
| `world_map.png`             | 气候终态渲染：9 类 Whittaker 生物群系着色、可见河流湖泊、纬度+温度双因素雪线、极地海冰、NW 光照山体阴影。                                                                       |
| `world_skeleton.png`        | GAN 细化前的构造骨架（域扭曲后的板块 + 造山带 + 裂谷 + 海岸分形）。用来检查大陆走向、山带连续性。                                                                             |
| `geoctx_visual_seed{}.png`  | 4 面板可视化：**高程 / 坡度 / 地理带 / 剖面**，附带探针点定位、多边形统计。                                                                                       |
| `geoctx_report_seed{}.html` | **单文件离线交互 HTML 报告**（≈4 MB，零外链）：高程 / 坡度 / 地理带三图层切换、鼠标悬停经纬度 + 海拔 + 坡度 + 海陆查询、探针点表 ↔ 地图跳转、多边形统计 Pill、剖面 Canvas 十字线联动地图。**双击即可在浏览器打开**。 |
| `world_3d_{plane,globe}.obj` 等 | 3D 模型导出：`.obj`（+`.mtl`+`*_tex.png` 高程着色纹理）与 `.gltf`（自包含单文件，纹理内嵌）。平面浮雕与球体星球双形态，`--preview` 打开 PyVista GPU 桌面窗口（缺失时回退 matplotlib）。默认导出保留海底起伏；`--coast-cut` 压平海洋使海岸锋利贴合。 |

## 常见问题 FAQ

### Q1: 代码疑似被 IDE 旧标签页覆盖（功能异常退化）

PyCharm / VSCode 会把文件内容缓存在标签页中；外部修改磁盘后，旧的未保存缓冲区被保存一次就会覆盖新代码。
**检查 / 恢复**（项目已建 git 快照）：

```bash
git status                    # 若 terrain/*.py 显示 modified 即被覆盖
git checkout -- terrain/      # 一键恢复到最近提交的验证版本
git log --oneline             # 查看历史提交
```

**预防**：外部工具修改代码后，在 IDE 中执行 `File → Reload All from Disk`。

### Q2: 提示 "请先用 v6.3 重训 (12 通道)"

检查点版本不匹配。用 `checkpoints/best_terrain_gan_v5.pth`（项目已内置 v5 权重，12 通道条件 GAN），或按 TECHNICAL.md §4 重训。

### Q3: `ValueError: could not convert string to float: '北海'`

把中文地名**写进了坐标字段**。正确写法：把地名放在**第三个可选列**，例如：

```
✅ 正确: --points "39,116,北京,35,139,东京"       ← lat/lon 是纯数字，地名第3列
❌ 错误: --points "39,北海,116,35,东京,139"     ← 把地名写成了 lat 或 lon
```

剖面同理：必须是 `latA,lonA,latB,lonB` 四个纯数字。

### Q4: 生成 HTML 报告时 `TypeError: not enough arguments for format string`

有人直接编辑了 `terrain/core/visualize_context.py` 里的 `_HTML_TEMPLATE` 三引号块，并且在 CSS/JS 中加入了**字面量** **`%`**（比如写 `width:100%` 而不是 `width:100%%`），与 Python 的 `%(KEY)s` 占位符冲突。
**修复**：把模板里所有 `%(KEY)format` 保留为原样，其余单独的**字面量百分号一律翻倍为** **`%%`**（重新 git checkout terrain/core/visualize_context.py 也可恢复到经过验证的版本）。

### Q5: 没有 ETOPO1 数据能跑吗？

生成世界**不需要**（只需 GAN 权重 + 板块模拟）。
仅在**重训 GAN**时才需要；缺失时会退化为合成地形训练（质量下降，仅供流程验证）。

### Q6: 训练数据下载失败？

- ETOPO1 支持 `DEM_PATH=/路径/global_dem.npy` 环境变量，格式要求：`(任意尺寸, 任意尺寸)` float32 全球米制高程（行=纬度北→南，列=经度−180→+180）。

- 板块边界支持 `TECT_JSON=/路径/PB2002.json` 环境变量。

### Q7: 想从别的 Python 脚本里直接查询某经纬度的海拔？

用 `geo_context.py` 的公开单例（见 TECHNICAL §16）：

```python
from terrain.core.geo_context import get_ctx
ctx = get_ctx(seed=1, size=2048)
print(ctx.get_elevation(39.9, 116.4))   # 北京: 海拔 米
print(ctx.region_stats(polygon=[(30,-5),(30,35),(45,35),(45,-5)]))  # 地中海方块统计
```

> v9.0 起所有核心模块使用相对/绝对导入，`terrain` 包可直接 `import terrain.core.xxx`，无需手动 `sys.path` 注入。

### Q8: 设备要求？

- **推荐**：Apple Silicon (MPS) 或 NVIDIA (CUDA)。

- **纯 CPU** 也能跑（GAN 瓦片推理会慢 5\~10 倍，但板块模拟纯 numpy 速度差不多）。

### Q9: 之前 terrain/ 下有两份同名 .py，现在只剩一份，会不会破坏导入？

**不会。** v12 已删除 terrain/ 根目录的 10 个兼容薄封装与 60+ 个一次性诊断脚本，统一使用：

- `terrain/core/*`：核心实现（plate_sim / tectonics / terrain_engine / generate_world / rerender / geo_context / visualize_context）
- `terrain/scripts/*`：批处理脚本（batch_gen / analyze_worlds / make_montage / _run_srnet_train / _run_pipeline_sr）

所有正式代码、测试（`terrain/tests/`）与 `main.py` 均已直接引用 `terrain.core.*`，删除薄封装后导入不受影响（已由 512 项测试验证）。若你的旧脚本还写着 `from terrain_engine import ...` 或 `from terrain.geo_context import ...`，请改为 `from terrain.core.terrain_engine import ...` / `from terrain.core.geo_context import ...`。

### Q10: 什么是 C++ 原生加速？需要额外安装吗？

v10 起新增 `terrain/native/`（pybind11 扩展 `fws_native`），把最耗时的 15 个内核（降水扫描 / 山体遮蔽 / 中值-众数滤波 / 流量累积 / 洼地填充 / 河流蜿蜒 / 碎片清理等）编译为 C++，与 Numba 实现**逐位一致**。`fws_native.*.so` 已随仓库内置，**开箱即用、无需安装**。

- 缺失或导入失败时，`climate_module.py` / `hydro_module.py` / `temperature_module.py` 会自动回退 Numba 实现（仅变慢，不影响结果）。
- 手动重新构建（改 C++ 后）：`cd terrain/native && python3 setup.py build_ext --inplace`，需要 `pybind11`；Apple clang 并行版本还需 `brew install libomp`（见 TECHNICAL §22）。

### Q11: 文档里有哪些版本？各版本对应什么？

| 文档          | 内容                                             | 读者      |
| ----------- | ---------------------------------------------- | ------- |
| `README.md` | 项目概述 / 安装 / 快速开始 / 常用任务 / 缓存 / 输出 / FAQ / 依赖 | 所有人    |
| `terrain/docs/TECHNICAL.md` | 管线总览 / 各算法原理 / 接口定义 / 配置参数 / native 加速 | 调参与开发 |
| `terrain/modules/ARCHITECTURE.md` | 模块化引擎抽象 / 数据契约 / 扩展指南 | 二次开发  |

## 依赖列表

| 包            | 最小版本   | 用途                               |
| ------------ | ------ | -------------------------------- |
| `torch`      | ≥ 1.13 | 12 通道条件 GAN 推理 / 训练              |
| `numpy`      | ≥ 1.22 | DEM 数组 / 板块模拟 / 统计               |
| `scipy`      | ≥ 1.9  | 高斯滤波 / 距离变换 / 双线性重采样 (HTML 报告图层) |
| `matplotlib` | ≥ 3.6  | 彩色渲染 / 4 面板 PNG / 色图             |
| `pyvista`    | ≥ 0.49 | 3D 预览 GPU 渲染（可选；缺失时回退 matplotlib，预览交互会明显变卡） |
| `Pillow`     | ≥ 9.0  | PNG 保存 / HTML 图层 base64 编码       |
| `pytest`     | ≥ 7.0  | 单元测试（开发可选）                       |
| `pybind11`   | ≥ 2.10 | 构建 C++ 原生扩展（仅改 native/ 后需要）      |
| `libomp`     | ≥ 15   | OpenMP 运行时（仅 macOS 手动重建 native 需要，`brew install libomp`） |

> **运行时不需要** pybind11 / libomp：`terrain/native/fws_native.*.so` 已内置，缺失时自动回退 Numba。

