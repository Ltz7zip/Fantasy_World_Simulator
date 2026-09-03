# -*- coding: utf-8 -*-
"""EARTH_REFERENCES — 地球地理学参考值集合 (供 SupervisorMetrics 对比).

本模块集中了原本散落在 terrain_engine.py 中的地球地理学常量（温度递减率、
纬度降水曲线、Whittaker 群系阈值、雪线纬度等），并补全地球真实值（陆地比、
最高峰、最深海沟、群系面积占比等），形成统一的"地球参考值集合"。

设计原则：
    - 纯数据模块，无副作用，可被 metrics/reflector/test 任意复用。
    - 纬度降水曲线直接复用 terrain_engine._P_LAT/_P_MM（同对象引用），
      避免重复定义导致脱节。
    - 群系索引严格对齐 terrain_engine.BIOME_NAMES 顺序。

物理依据：
    - 全球陆地/海洋比例、海拔极值来自 ETOPO1 / NOAA 实测。
    - 纬度温度带参考曲线对应地球年均海平面温度（赤道 27°C → 极地 -45°C）。
    - 群系面积占比为粗略地球占比，与 Whittaker 9 类口径不完全对齐
      （如苔原 10% 含极地荒漠、沙漠 25% 含极地荒漠），故容差较宽（±10%）。
"""
from __future__ import annotations

import numpy as np

# 复用 terrain_engine 已有的纬度降水参考曲线 (地球 ITCZ 纬向基础降水)
# terrain/__init__.py 已把 core/ 注入 sys.path, 可直接扁平导入
from terrain_engine import _P_LAT, _P_MM, BIOME_NAMES  # noqa: F401  (复用, 避免重复定义)

# ─── 全球地形参考 (来源: ETOPO1 / NOAA 实测) ───
GLOBAL = dict(
    land_ratio=0.29,               # 陆地占比 ~29%
    mean_land_elevation=840.0,     # 平均陆地海拔 ~840m
    peak_elevation=8848.0,         # 最高峰 珠峰 8848m
    ocean_trench=-10994.0,         # 最深海沟 马里亚纳 -10994m
    mean_ocean_depth=-3682.0,      # 海洋平均深度 -3682m
    mountain_ratio=0.08,           # 山地(>=1500m)占比 ~8% (与 geo_context._MOUNTAIN_THRESHOLD_M 对齐)
)

# ─── 纬度温度参考曲线 (°C, 年均海平面温度) ───
# 赤道 26-28°C / 副热带 20-22°C / 温带 5-15°C / 极地 -40°C 以下
TEMP_LAT = np.array([0, 10, 20, 25, 30, 40, 50, 60, 66.5, 75, 90], np.float64)
TEMP_C   = np.array([27, 25, 22, 21, 20, 14, 8, 2, -8, -25, -45], np.float64)

# ─── 纬度降水参考曲线 (复用 _P_LAT/_P_MM, 补注释) ───
# 这是地球 ITCZ 纬向基础降水: 赤道峰 2300mm → 副热带低谷 520mm → 温带峰 850mm → 极地干 130mm
# 注: precipitation_field 输出含 6 重修正（大陆度/副高/地形/季风/长波/屏障），
# 与本参考曲线口径不同，故容差较宽（±30%）。
PRECIP_LAT = _P_LAT  # [0,5,10,15,20,25,30,35,45,55,65,80,90]
PRECIP_MM  = _P_MM   # [2300,2150,1700,1150,700,520,560,680,850,750,480,230,130]

# ─── 群系面积占比地球参考 (Whittaker 9 类, 对齐 BIOME_NAMES 索引) ───
#   0 苔原 / 1 北方针叶林 / 2 温带草原 / 3 温带落叶林 / 4 温带雨林
#   5 亚热带荒漠 / 6 稀树草原 / 7 热带季节林 / 8 热带雨林
BIOME_RATIO = dict(
    tundra=0.10,                # 苔原 ~10% (含极地荒漠)
    taiga=0.05,                 # 北方针叶林 ~5%
    temperate_grassland=0.08,   # 温带草原 ~8%
    temperate_deciduous=0.08,   # 温带落叶林 ~8%
    temperate_rainforest=0.005, # 温带雨林 ~0.5%
    subtropical_desert=0.25,    # 亚热带荒漠 ~25% (含极地荒漠/戈壁)
    savanna=0.12,               # 稀树草原 ~12%
    tropical_seasonal=0.05,     # 热带季节林 ~5%
    tropical_rainforest=0.13,   # 热带雨林 ~13%
)
# 索引对齐数组 (供 metrics 按 BIOME_NAMES 顺序取值)
BIOME_RATIO_BY_IDX = np.array([
    BIOME_RATIO["tundra"],
    BIOME_RATIO["taiga"],
    BIOME_RATIO["temperate_grassland"],
    BIOME_RATIO["temperate_deciduous"],
    BIOME_RATIO["temperate_rainforest"],
    BIOME_RATIO["subtropical_desert"],
    BIOME_RATIO["savanna"],
    BIOME_RATIO["tropical_seasonal"],
    BIOME_RATIO["tropical_rainforest"],
], np.float64)

# ─── 风带纬度边界 (°) ───
WIND_BANDS = dict(
    trade=    (0, 30),    # 信风带 (赤道辐合带两侧)
    westerly= (30, 60),   # 西风带
    polar=    (60, 90),   # 极地东风带
)

# ─── 雪盖/冰盖纬度带 (与 snow_area_by_band 一致) ───
SNOW_BANDS = [(0, 30), (30, 50), (50, 66.5), (66.5, 90.01)]
# 雪盖占比地球参考: 热带 0 / 温带 ~2% / 亚北极 ~45% / 极地 ~90%
SNOW_RATIO_REF = np.array([0.0, 0.02, 0.45, 0.90], np.float64)

# ─── 容差 (tolerance) ───
#   地形类用相对容差 (除 mountain_ratio/land_ratio 用绝对值)
#   温度用绝对值 (°C), 降水用相对值 (波动大), 群系用绝对值
TOLERANCE = dict(
    land_ratio=0.03,           # ±3% (绝对)
    peak_elevation=0.15,       # ±15% (相对)
    ocean_trench=0.15,         # ±15% (相对)
    mean_land_elevation=0.20,  # ±20% (相对, 陆地平均海拔波动大)
    mountain_ratio=0.05,       # ±5% (绝对)
    temp_band=3.0,             # 纬度温度带 ±3°C (绝对)
    precip_band=0.30,         # 纬度降水带 ±30% (相对, ITCZ 波动大)
    biome_ratio=0.10,          # 群系面积 ±10% (绝对)
    snow_band=0.15,            # 雪盖带 ±15% (绝对)
)

# ─── 严重度阈值 (相对偏差比例) ───
SEVERITY = dict(info=0.05, warn=0.15, critical=0.30)


def get_reference(metric: str) -> float:
    """按指标名返回地球参考值 (供 reflector 查表).

    Parameters
    ----------
    metric : str
        指标名，如 "land_ratio" / "peak_elevation" / "ocean_trench" /
        "mean_land_elevation" / "mountain_ratio"。

    Returns
    -------
    float
        地球参考值。

    Raises
    ------
    KeyError
        指标名不在 GLOBAL 中。
    """
    return float(GLOBAL[metric])
