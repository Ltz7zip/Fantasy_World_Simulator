# -*- coding: utf-8 -*-
# batch_gen.py —— 批量生成多种子世界地图 (检验用)
# 运行: python batch_gen.py --seeds 1 2 3 4 5 6 7 8 9 10 --size 2048
import os
import argparse
from generate_world import generate_world

_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--size", type=int, default=2048)
    a = ap.parse_args()
    for s in a.seeds:
        png = os.path.join(_dir, "output", f"seed{s}_map.png")
        npy = os.path.join(_dir, "output", f"seed{s}_dem.npy")
        if os.path.exists(npy):
            print(f"[skip] seed{s} 已存在")
            continue
        print(f"\n===== seed {s} =====")
        generate_world(seed=s, size_px=a.size, out_png=png, out_npy=npy)
    print("\n全部完成")
