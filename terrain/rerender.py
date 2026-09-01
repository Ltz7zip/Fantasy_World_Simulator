# rerender.py —— 只重画, 不动数据 (python terrain/rerender.py, 任意工作目录可运行)
import os
import numpy as np
import matplotlib.pyplot as plt
from terrain_engine import m2v, make_terrain_cmap
from generate_world import render_world

_dir = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(_dir, "output", "world_dem.npy")); S = d.shape[0]
lats = -90.0 + (np.arange(S) + 0.5) * (180.0 / S)
plt.imsave(os.path.join(_dir, "output", "world_map.png"),
           render_world(m2v(d), lats, d > 0,
                        make_terrain_cmap(), np.random.RandomState(7)))
print("已重绘 output/world_map.png")
