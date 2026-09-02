# -*- coding: utf-8 -*-
"""兼容薄封装：``from visualize_context import …``（旧无包前缀导入）继续可用。

真实实现已搬迁到 terrain/core/visualize_context.py。本模块先 ``import terrain``
触发 sys.path 注入 terrain/core，再执行转发 import 让旧名字映射到新文件。
"""
import terrain  # noqa: F401
from terrain.core.visualize_context import *  # noqa: F401,F403
