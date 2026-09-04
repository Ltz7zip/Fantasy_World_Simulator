# 幻想世界生成器 (terrain/) —— 使用文档 · v9.0

板块演化 + 神经网络混合驱动的随机世界地图生成器。物理规则层负责地质 / 气候 / 水文的因果骨架，GAN 负责像素级地形纹理；同时提供 **4 面板可视化**与**单文件离线 HTML 交互报告**。

> **v9.0 升级要点**：气候系统 v7.1–v7.4（大陆度指数 / 极地冰盖 / 泰加林修复）、河流水系 v8.3（骨架提取 + 沿程单调修复 / Hack 定律蜿蜒 / 牛轭湖 / 连续蓝度渲染）、新增 **深度学习模块 `terrain/dl/`**（DDPM → LDM → 物理监督 → 大规模推理 → 端到端管线 + ESRGAN 超分）、新增 **监督反思模块 `terrain/supervisor/`**（审计 → 反思 → 学习日志 → 自动注入 overrides）、安全加固（`torch.load(weights_only=True)` / URL scheme 白名单）、测试套件 33 → **512 项**。

## 目录结构

```
terrain/
├── README.md              ← 本文档 (安装 + 使用)
├── TECHNICAL.md           ← 技术文档 (架构 / 算法 / 接口 / 参数)
│
├── core/                  ← 核心源码 (8 个模块, 真实实现)
│   ├── plate_sim.py           板块演化模拟器 v3.0 (可单独调试)
│   ├── tectonics.py           真实板块边界 → 构造条件场 (GAN 训练数据)
│   ├── terrain_engine.py      核心: 物理工具 / 气候 / 侵蚀 / GAN 训练 / v8.3 河流系统
│   ├── generate_world.py      主入口: 生成完整世界 (DEM + 渲染图)
│   ├── rerender.py            仅重渲染 (改渲染参数后用, 不重算地形)
│   ├── geo_context.py         WorldGeoContext 单例: 坐标查询 / KPI 统计 / 多边形裁切
│   └── visualize_context.py   4 面板 PNG + 交互 HTML 报告 (探针/多边形/剖面联动)
│
├── dl/                    ← 深度学习模块 (v9.0 新增, 5 阶段)
│   └── (VAE/DDPM/LDM/物理监督/超分/端到端管线, 详见 TECHNICAL §21)
│
├── supervisor/            ← 监督反思模块 (v9.0 新增)
│   └── (审计 metrics / reflector / learning_log / CLI)
│
├── tests/                 ← 单元测试 (512 项, pytest)
│   ├── test_visualize_context.py   可视化 / HTML / geo_context (33 项)
│   ├── test_supervisor.py          监督反思 (32 项)
│   └── test_dl_phase1..5.py        深度学习 5 阶段 (447 项)
│
├── data/                  ← 原始数据 (重训 GAN 需要; 生成世界不需要)
│   ├── ETOPO1_Ice_g_gmt4.grd        # 全球 DEM (~890MB, 公有领域)
│   └── PB2002_boundaries.json       # Bird 2003 板块边界
├── checkpoints/           ← 模型权重
│   └── best_terrain_gan_v5.pth       # 12 通道条件 GAN (~44MB, 已内置可直接用)
├── cache/                 ← 可再生缓存 (删除后会自动重建, 只是变慢)
│   ├── plate_sim_v3_T512_seed*.npz   # 板块演化结果 (同参数秒级复用)
│   └── tect_cache_*.npz              # 构造条件场
└── output/                ← 生成结果
    ├── world_map.png                 # 最终气候渲染图 (生物群系 / 河湖 / 雪线)
    ├── world_dem.npy                 # 全球高程 (米, S×S float32, 行=纬北→南 / 列=经)
    ├── world_skeleton.png            # GAN 前的构造骨架
    ├── geoctx_visual_seed{}.png      # 4 面板可视化 (高程/坡度/地理带/剖面)
    └── geoctx_report_seed{}.html      # 单文件离线交互报告 (≈4 MB)
```

## 安装步骤

> 要求 **Python 3.9 \~ 3.12**（Apple Silicon / x86-64 均可；torch / numpy 需要匹配）。

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
python terrain/generate_world.py --seed 1 --size 2048
```

- **`data/ETOPO1`** **和板块边界 JSON**：生成世界**不需要**（仅重训 GAN 才需要）。缺失时重训会退化为合成地形训练。

- **`checkpoints/best_terrain_gan_v5.pth`**：项目已内置，首次运行直接可用；若被删除会触发自动重训（见 FAQ）。

## 快速开始

```bash
# 项目根目录运行 (脚本内部全部使用基于 __file__ 的绝对路径, 任意目录均可)
python terrain/generate_world.py --seed 0 --size 2048
```

首次运行约 **2\~4 分钟**（板块演化 450 Myr 模拟占大头；同 seed 再次运行走缓存 **< 10 秒**完成）。
结果写入 `terrain/output/`。

## 常用任务

| 任务                        | 命令                                                                                                                                                    |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| 换一个世界                     | `python terrain/generate_world.py --seed 任意整数 --size 2048`                                                                                            |
| 更大分辨率 (4096²)             | `--size 4096`（约 4 倍耗时, 16 倍显存）                                                                                                                        |
| **4 面板可视化 PNG**           | `python terrain/visualize_context.py --seed 1`（高程 / 坡度 / 地理带 / 剖面）                                                                                    |
| **交互 HTML 报告（仅 HTML）**    | `python terrain/visualize_context.py --seed 1 --no-png --html`                                                                                        |
| **PNG + HTML 双产出 + 叠加元素** | `python terrain/visualize_context.py --seed 1 --html --points "39,116,CityA,35,139,CityB" --poly "30,-5,30,35,45,35,45,-5" --profile "60,-10,25,125"` |
| 只调渲染不改地形                  | `python terrain/rerender.py`                                                                                                                          |
| 单独调试板块模拟                  | `python terrain/plate_sim.py --seed 7 --T 512`                                                                                                        |
| 批量生成多种子                   | `python terrain/batch_gen.py --seeds 1 2 3 4 5`                                                                                                       |
| 地理指标检验 (vs 地球)            | `python terrain/analyze_worlds.py` → 输出 `analysis_results.json`                                                                                       |
| 生成对比拼图                    | `python terrain/make_montage.py`                                                                                                                      |
| 单元测试 (512 项)               | `pytest terrain/tests/ -v`                                                                                                   |
| 监督审计 + 反思学习              | `python -m terrain.supervisor --seed 42 --size-px 2048`                                                                       |
| 重训 GAN                    | 删除 `checkpoints/best_terrain_gan_v5.pth` 后 `python terrain/terrain_engine.py`（需 `data/ETOPO1`, 2.5\~4h / MPS）                                         |

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

有人直接编辑了 `terrain/visualize_context.py` 里的 `_HTML_TEMPLATE` 三引号块，并且在 CSS/JS 中加入了**字面量** **`%`**（比如写 `width:100%` 而不是 `width:100%%`），与 Python 的 `%(KEY)s` 占位符冲突。
**修复**：把模板里所有 `%(KEY)format` 保留为原样，其余单独的**字面量百分号一律翻倍为** **`%%`**（重新 git checkout visualize\_context.py 也可恢复到经过验证的版本）。

### Q5: 没有 ETOPO1 数据能跑吗？

生成世界**不需要**（只需 GAN 权重 + 板块模拟）。
仅在**重训 GAN**时才需要；缺失时会退化为合成地形训练（质量下降，仅供流程验证）。

### Q6: 训练数据下载失败？

- ETOPO1 支持 `DEM_PATH=/路径/global_dem.npy` 环境变量，格式要求：`(任意尺寸, 任意尺寸)` float32 全球米制高程（行=纬度北→南，列=经度−180→+180）。

- 板块边界支持 `TECT_JSON=/路径/PB2002.json` 环境变量。

### Q7: 想从别的 Python 脚本里直接查询某经纬度的海拔？

用 `geo_context.py` 的公开单例（见 TECHNICAL §16）：

```python
from terrain.geo_context import get_ctx
ctx = get_ctx(seed=1, size=2048)
print(ctx.get_elevation(39.9, 116.4))   # 北京: 海拔 米
print(ctx.region_stats(polygon=[(30,-5),(30,35),(45,35),(45,-5)]))  # 地中海方块统计
```

> v9.0 起所有核心模块使用相对/绝对导入，`terrain` 包可直接 `import terrain.xxx`，无需手动 `sys.path` 注入。

### Q8: 设备要求？

- **推荐**：Apple Silicon (MPS) 或 NVIDIA (CUDA)。

- **纯 CPU** 也能跑（GAN 瓦片推理会慢 5\~10 倍，但板块模拟纯 numpy 速度差不多）。

## 依赖列表

| 包            | 最小版本   | 用途                               |
| ------------ | ------ | -------------------------------- |
| `torch`      | ≥ 1.13 | 12 通道条件 GAN 推理 / 训练              |
| `numpy`      | ≥ 1.22 | DEM 数组 / 板块模拟 / 统计               |
| `scipy`      | ≥ 1.9  | 高斯滤波 / 距离变换 / 双线性重采样 (HTML 报告图层) |
| `matplotlib` | ≥ 3.6  | 彩色渲染 / 4 面板 PNG / 色图             |
| `Pillow`     | ≥ 9.0  | PNG 保存 / HTML 图层 base64 编码       |
| `pytest`     | ≥ 7.0  | 单元测试（开发可选）                       |

