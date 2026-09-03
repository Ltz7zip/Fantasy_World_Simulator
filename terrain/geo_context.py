# -*- coding: utf-8 -*-
"""兼容薄封装 · ⚠ 勿直接编辑 · 真实实现见 terrain/core/ 或 terrain/scripts/

模块用途：让「from geo_context import …」旧无包前缀导入 & 直接 `python terrain/geo_context.py`
继续可用。本模块仅做两件事：(1) import terrain 触发路径注入 terrain/core；
(2) 从 terrain.core.geo_context 做 ``*`` re-export。
维护者注意：请只编辑 terrain/core/ 与 terrain/scripts/ 下的真实文件。
"""
import os as _os
import sys as _sys
_PKG_DIR = _os.path.dirname(_os.path.abspath(__file__))
_PARENT_DIR = _os.path.dirname(_PKG_DIR)
if _PARENT_DIR not in _sys.path:
    _sys.path.insert(0, _PARENT_DIR)
import terrain  # noqa: F401
from terrain.core.geo_context import *  # noqa: F401,F403

if __name__ == "__main__":
    from terrain.core.geo_context import main as _main
    _main()
