# -*- coding: utf-8 -*-
"""terrain — Fantasy World Simulator 地形子包。

公开入口（`from terrain import X` 即可使用）：
    WorldGeoContext / get_ctx               — 地形单例（查询 / 坐标 / 多边形 KPI）
    generate_world                          — 主生成器（板块→GAN→渲染）
    build_geo_figure                        — 4 面板 PNG（matplotlib Figure）
    build_html_report / export_html_report  — 单文件离线交互 HTML
    TerrainDataProtocol                     — 地形查询 Protocol（供上游替换实现）
    run_plate_sim                           — 仅板块演化（调试骨架用）

路径自动修复：首次 import 时会把 ``terrain/core`` 与 ``terrain/scripts``
追加到 ``sys.path``，保证所有旧脚本（「from terrain_engine import …」风格）
继续可用，无需手动调整。
"""
from __future__ import annotations

import os as _os
import sys as _sys

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_CORE = _os.path.join(_HERE, "core")
_SCRIPTS = _os.path.join(_HERE, "scripts")
for _p in (_HERE, _CORE, _SCRIPTS):
    if _p and _p not in _sys.path:
        _sys.path.insert(0, _p)

# ————— 公开符号 —————
from terrain.core.geo_context import (
    WorldGeoContext as WorldGeoContext,
    TerrainDataProtocol as TerrainDataProtocol,
    get_ctx as get_ctx,
)
from terrain.core.generate_world import generate_world as generate_world
from terrain.core.visualize_context import (
    build_geo_figure as build_geo_figure,
    build_html_report as build_html_report,
    export_html_report as export_html_report,
)
from terrain.core.plate_sim import run_plate_sim as run_plate_sim

__all__ = [
    "WorldGeoContext", "TerrainDataProtocol", "get_ctx",
    "generate_world", "run_plate_sim",
    "build_geo_figure", "build_html_report", "export_html_report",
]
__version__ = "7.0.0"
