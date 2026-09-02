# -*- coding: utf-8 -*-
"""兼容薄封装 · ⚠ 勿直接编辑 · 真实实现见 terrain/core/ 或 terrain/scripts/

模块用途：让「from terrain_engine import …」旧无包前缀导入 & 直接 `python terrain/terrain_engine.py`
继续可用。本模块仅做两件事：(1) import terrain 触发路径注入 terrain/core；
(2) 从 terrain.core.terrain_engine 做 ``*`` re-export。
维护者注意：请只编辑 terrain/core/ 与 terrain/scripts/ 下的真实文件。
"""
import terrain  # noqa: F401
from terrain.core.terrain_engine import *  # noqa: F401,F403
