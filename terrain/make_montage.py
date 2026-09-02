# -*- coding: utf-8 -*-
# make_montage.py —— 真实地球(同风格渲染) + 10种子生成图 对比拼图
import os
import numpy as np
import matplotlib.pyplot as plt
from analyze_worlds import load_earth
from terrain_engine import m2v, make_terrain_cmap
from generate_world import render_world

_dir = os.path.dirname(os.path.abspath(__file__))
S = 2048
LATS = (-90.0 + (np.arange(S) + 0.5) * (180.0 / S)).astype(np.float32)

earth = load_earth()
rgb_e = render_world(m2v(earth), LATS, earth > 0, make_terrain_cmap(),
                     np.random.RandomState(0))
plt.imsave(os.path.join(_dir, "output", "earth_render.png"), rgb_e)

fig, axes = plt.subplots(4, 3, figsize=(21, 22))
axes[0, 0].imshow(rgb_e)
axes[0, 0].set_title("真实地球 ETOPO1 (同风格渲染)", fontsize=16)
axes[0, 0].axis("off")
cells = [axes[0, 1], axes[0, 2]] + list(axes[1]) + list(axes[2]) + list(axes[3])
for i, ax in enumerate(cells, start=1):
    p = os.path.join(_dir, "output", f"seed{i}_map.png")
    if os.path.exists(p):
        img = plt.imread(p)[..., :3]
        ax.imshow(img)
        ax.set_title(f"seed {i}", fontsize=14)
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(_dir, "output", "montage_earth_vs_10seeds.png"),
            dpi=90, bbox_inches="tight")
print("已保存 output/montage_earth_vs_10seeds.png 与 earth_render.png")
