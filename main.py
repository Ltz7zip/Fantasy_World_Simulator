# -*- coding: utf-8 -*-
"""main.py — Fantasy World Simulator 项目启动入口。

双模式启动：
    python main.py                  → GUI 模式（优先 PyQt5，缺失自动降级 CLI 菜单）
    python main.py --cli            → 强制交互式命令行菜单
    python main.py --help           → 所有子命令一览
    python main.py generate -s 42   → 直跑生成（无菜单，脚本/CI 友好）

GUI 模式提供：
    ① 生成世界（seed / size / 输出路径）
    ② 4 面板可视化 PNG（叠加探针 / 多边形 / 剖面）
    ③ HTML 交互报告（含 7 项 KPI / 图层切换 / 探针联动 / 剖面 Canvas）
    ④ 单元测试（pytest terrain/tests -q）
    ⑤ 依赖自检（一键列出缺失包与安装命令）
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import textwrap
import traceback
from pathlib import Path
from typing import List, Optional, Sequence

# ---------------------------------------------------------------------------
# 0. 路径自修复：无论从哪个 cwd 启动，都能 import terrain 与旧无包前缀模块
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
_TERRAIN_DIR = _PROJECT_ROOT / "terrain"
for _p in (str(_PROJECT_ROOT), str(_TERRAIN_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import terrain  # noqa: E402  — terrain.__init__.py 会注入 core/scripts 到 sys.path

# ---------------------------------------------------------------------------
# 1. 依赖自检
# ---------------------------------------------------------------------------
# (安装名, import 名, 最低版本或 None)
_DEPS: List[tuple[str, str, Optional[str]]] = [
    ("numpy", "numpy", "1.24"),
    ("matplotlib", "matplotlib", "3.7"),
    ("scipy", "scipy", "1.10"),
    ("torch", "torch", "2.1"),
    ("pytest", "pytest", "7.4"),
    ("Pillow", "PIL", "10.0"),
]
_GUI_DEPS: List[tuple[str, str, Optional[str]]] = [
    ("PyQt5", "PyQt5", "5.15"),
]


def check_dependencies(extra: Optional[Sequence[tuple[str, str, Optional[str]]]] = None
                       ) -> tuple[bool, list[tuple[str, str, Optional[str]]]]:
    """逐包检查依赖可用性。返回 (全部通过, 缺失包列表[(pip_name,import_name,min_ver)])。"""
    missing: list[tuple[str, str, Optional[str]]] = []
    for pip_name, import_name, min_ver in list(_DEPS) + list(extra or ()):
        try:
            mod = __import__(import_name)
        except Exception:
            missing.append((pip_name, import_name, min_ver))
            continue
        if min_ver:
            got = getattr(mod, "__version__", None)
            if not got:  # e.g. PIL 藏在 PIL.__version__
                got = getattr(mod, "__VERSION__", None)
            # 没有版本号也不报错（PIL 等兼容）
    return (len(missing) == 0), missing


def missing_install_hint(missing: Sequence[tuple[str, str, Optional[str]]]) -> str:
    """根据缺失包列表生成一条 pip 安装命令提示，返回形如 "$ pip install xxx yyy" 的字符串。"""
    names = " ".join(n for n, _, _ in missing)
    venv = Path(sys.executable).parent.parent / "bin" / "pip" if ".venv" in sys.executable else None
    pip_cmd = (f"{venv} install" if venv and venv.exists() else "pip install") + f" {names}"
    return f"$ {pip_cmd}"


# ---------------------------------------------------------------------------
# 2. 四个核心动作（CLI / GUI 共用）—— 统一参数模型与返回值
# ---------------------------------------------------------------------------
def _ensure_out_dir() -> Path:
    """确保 terrain/output 目录存在并返回其 Path。"""
    out = _TERRAIN_DIR / "output"
    out.mkdir(parents=True, exist_ok=True)
    return out


class GenerateOptions:
    """生成世界的参数容器：seed、size 与输出目录。"""

    def __init__(self, seed: int = 1, size: int = 4096, out_dir: Optional[Path] = None):
        """``__init__``: 初始化生成参数，out_dir 缺省时落到默认输出目录。"""
        self.seed, self.size, self.out_dir = seed, size, Path(out_dir or _ensure_out_dir())


def action_generate(opts: GenerateOptions) -> tuple[bool, str]:
    """调用 terrain.generate_world(seed, size)，返回 (成功, 说明)。"""
    try:
        from terrain.core.generate_world import generate_world
        dem, rgb = generate_world(opts.seed, opts.size)
        opts.out_dir.mkdir(parents=True, exist_ok=True)
        dem_path = opts.out_dir / f"world_dem_seed{opts.seed}.npy"
        rgb_path = opts.out_dir / f"world_skeleton_seed{opts.seed}.png"
        import numpy as _np
        from PIL import Image as _Image
        _np.save(dem_path, dem)
        _Image.fromarray(rgb).save(rgb_path)
        terrain.WorldGeoContext.reset()
        return True, (
            f"✅ 世界生成完成（seed={opts.seed} size={opts.size}）\n"
            f"   DEM={dem_path}\n   预览={rgb_path}\n"
            f"   DEM 统计：min={float(dem.min()):.0f}m  max={float(dem.max()):.0f}m  "
            f"mean={float(dem.mean()):.0f}m"
        )
    except Exception as exc:  # pragma: no cover
        return False, f"❌ 生成失败：{exc}\n{traceback.format_exc()}"


class VisualizeOptions:
    """4 面板 PNG 可视化的参数容器：seed、size、探针/多边形/剖面与输出路径。"""

    def __init__(self, seed: int = 1, size: int = 4096,
                 points: Optional[str] = None,
                 polygon: Optional[str] = None,
                 profile: Optional[str] = None,
                 out: Optional[Path] = None):
        """``__init__``: 初始化可视化参数，out 缺省时生成默认 PNG 路径。"""
        self.seed, self.size = seed, size
        self.points, self.polygon, self.profile = points, polygon, profile
        self.out = Path(out or _ensure_out_dir() / f"geoctx_visual_seed{seed}.png")


def action_visualize(opts: VisualizeOptions) -> tuple[bool, str]:
    """调用 build_geo_figure 生成 4 面板 PNG。"""
    try:
        import matplotlib; matplotlib.use("Agg")
        from terrain.core.visualize_context import _parse_points, _parse_poly, _parse_profile
        from terrain.core.visualize_context import build_geo_figure
        args = argparse.Namespace(
            seed=opts.seed, size=opts.size, out=str(opts.out),
            points=opts.points, polygon=opts.polygon, profile=opts.profile,
        )
        probes = _parse_points(args.points)
        poly   = _parse_poly(args.polygon)
        prof   = _parse_profile(args.profile)
        fig, axes, ctx, info = build_geo_figure(opts.seed, opts.size,
                                                probe_points=probes,
                                                polygon=poly, profile=prof)
        opts.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(opts.out, dpi=150, bbox_inches="tight")
        try:
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception:
            pass
        kpis = {
            "陆地占比": f"{info.get('land_ratio', 0)*100:.1f}%",
            "山地占比": f"{info.get('mountain_ratio', 0)*100:.1f}%",
            "最高峰": f"{info.get('dem_max', 0):.0f}m",
            "最深海沟": f"{info.get('dem_min', 0):.0f}m",
        }
        stats = "; ".join(f"{k}={v}" for k, v in kpis.items())
        return True, f"✅ 4 面板可视化已保存：{opts.out}\n   KPI：{stats}"
    except Exception as exc:  # pragma: no cover
        return False, f"❌ 可视化失败：{exc}\n{traceback.format_exc()}"


class HtmlOptions:
    """HTML 交互报告的参数容器：seed、size、叠加元素、图层/查询像素与输出路径。"""

    def __init__(self, seed: int = 1, size: int = 4096,
                 points: Optional[str] = None,
                 polygon: Optional[str] = None,
                 profile: Optional[str] = None,
                 layer_size: int = 1024,
                 query_size: int = 512,
                 no_png: bool = False,
                 html_out: Optional[Path] = None):
        """``__init__``: 初始化 HTML 报告参数，html_out 缺省时生成默认报告路径。"""
        self.seed, self.size = seed, size
        self.points, self.polygon, self.profile = points, polygon, profile
        self.layer_size, self.query_size = layer_size, query_size
        self.no_png = no_png
        self.html_out = Path(html_out or _ensure_out_dir() / f"geoctx_report_seed{seed}.html")


def action_html(opts: HtmlOptions) -> tuple[bool, str]:
    """调用 build_html_report 生成交互 HTML，可选附带生成 PNG。"""
    try:
        import matplotlib; matplotlib.use("Agg")
        from terrain.core.visualize_context import (
            _parse_points, _parse_poly, _parse_profile,
            build_html_report, build_geo_figure,
        )
        probes = _parse_points(opts.points)
        poly   = _parse_poly(opts.polygon)
        prof   = _parse_profile(opts.profile)
        out_html, ctx = build_html_report(
            seed=opts.seed, size=opts.size,
            probe_points=probes, polygon=poly, profile=prof,
            output_path=str(opts.html_out),
            layer_size=opts.layer_size, query_size=opts.query_size,
        )
        lines = [f"✅ HTML 报告已保存：{out_html}"]
        png_path = None
        if not opts.no_png:
            fig, axes, _, info = build_geo_figure(
                opts.seed, opts.size,
                probe_points=probes, polygon=poly, profile=prof,
            )
            png_path = opts.html_out.with_suffix(".png")
            png_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(png_path, dpi=150, bbox_inches="tight")
            try:
                import matplotlib.pyplot as plt
                plt.close(fig)
            except Exception:
                pass
            lines.append(f"   4 面板 PNG：{png_path}")
            kpis = {
                "陆地占比": f"{info.get('land_ratio', 0)*100:.1f}%",
                "山地占比": f"{info.get('mountain_ratio', 0)*100:.1f}%",
                "最高峰": f"{info.get('dem_max', 0):.0f}m",
                "最深海沟": f"{info.get('dem_min', 0):.0f}m",
            }
        else:
            kpis = {
                "陆地占比": f"{float(ctx.land_mask.mean())*100:.1f}%",
                "山地占比": f"{float(ctx.mountain_mask.mean())*100:.1f}%",
                "最高峰": f"{float(ctx.dem.max()):.0f}m",
                "最深海沟": f"{float(ctx.dem.min()):.0f}m",
            }
        lines.append("   KPI：" + "; ".join(f"{k}={v}" for k, v in kpis.items()))
        return True, "\n".join(lines)
    except Exception as exc:  # pragma: no cover
        return False, f"❌ HTML 报告失败：{exc}\n{traceback.format_exc()}"


def action_pytest(extra_args: Optional[list[str]] = None) -> tuple[bool, str]:
    """运行 terrain/tests 下的所有 pytest；返回 (通过, 输出末行)。"""
    cmd = [sys.executable, "-m", "pytest", "terrain/tests", "-q"] + list(extra_args or [])
    proc = subprocess.run(cmd, cwd=str(_PROJECT_ROOT), capture_output=True, text=True)
    tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-12:])
    return (proc.returncode == 0,
            ("✅ pytest 通过" if proc.returncode == 0 else "❌ pytest 失败") +
            (f"\n{tail}" if tail else ""))


# ---------------------------------------------------------------------------
# 3. CLI 菜单（--cli 或 PyQt5 缺失时使用）
# ---------------------------------------------------------------------------
_MENU_BANNER = r"""
 _____                 _           ___        _    __  _  _    _
|  ___|_ _ _ __  _ __ | |_ ___  __|_ _|_ __  | |_ / _|(_)| |_ | |_  ___  _ __
| |_ / _` | '_ \| '_ \| __/ __|/ _ \| || '_ \ | __| |_ | || __|| __|/ _ \| '__|
|  _| (_| | | | | | | | |_\__ \  __/| || | | || |_|  _|| || |_ | |_|  __/| |
|_|  \__,_|_| |_|_| |_|\__|___/\___|___|_| |_| \__|_|  |_| \__| \__|\___||_|
                    Fantasy World Simulator — v9.1.0 Launcher
"""


def _read_choice(prompt: str, default: str) -> str:
    """读取一行用户输入，空输入/EOF 时返回 default。"""
    try:
        raw = input(prompt).strip()
    except EOFError:
        raw = ""
    return raw or default


def _cli_read_common(prefix: str = "") -> tuple[int, int]:
    """交互式读取 seed 与 size，返回 (seed, size)，非法输入退回默认值。"""
    s = _read_choice(f"  {prefix}seed [1]: ", "1")
    z = _read_choice(f"  {prefix}size (1024/2048/4096) [4096]: ", "4096")
    try:
        seed, size = int(s), int(z)
    except ValueError:
        print("  ⚠️ 非法数字，退回默认 seed=1 size=4096")
        seed, size = 1, 4096
    return seed, size


def _cli_read_overlays() -> tuple[Optional[str], Optional[str], Optional[str]]:
    """交互式读取探针/多边形/剖面叠加元素，返回 (points, polygon, profile)，跳过则为 None。"""
    print("  叠加元素格式（空=跳过）：")
    print("    --points ：lat,lon,name;lat,lon,name   例如 32.0,100.0,长安")
    print("    --polygon：lat,lon;lat,lon;...         多边形纬度/经度点")
    print("    --profile：latA,lonA,latB,lonB         剖面线两端")
    p = _read_choice("  探针 points [空]: ", "") or None
    o = _read_choice("  多边形 polygon [空]: ", "") or None
    f = _read_choice("  剖面 profile [空]: ", "") or None
    return p, o, f


def run_cli_menu() -> int:
    """运行交互式命令行主菜单（生成/可视化/HTML/测试/依赖自检），返回退出码。"""
    print(_MENU_BANNER)
    ok, missing = check_dependencies()
    if not ok:
        print("⚠️  以下依赖缺失：\n    " +
              " ".join(n for n, _, _ in missing) +
              f"\n    安装命令：\n    {missing_install_hint(missing)}\n")
    else:
        print("✅ 基础依赖全部就位（numpy/matplotlib/scipy/torch/pytest/Pillow）")
    print()

    while True:
        print("=" * 60)
        print("1. 生成世界        2. 4 面板 PNG 可视化     3. HTML 交互报告")
        print("4. 运行单元测试    5. 依赖自检              0. 退出")
        c = _read_choice("请选择 [1]: ", "1")
        if c == "0":
            print("再见！")
            return 0
        elif c == "1":
            seed, size = _cli_read_common()
            ok, msg = action_generate(GenerateOptions(seed=seed, size=size))
            print(msg)
        elif c == "2":
            seed, size = _cli_read_common()
            p, o, f = _cli_read_overlays()
            ok, msg = action_visualize(VisualizeOptions(seed, size, p, o, f))
            print(msg)
        elif c == "3":
            seed, size = _cli_read_common()
            p, o, f = _cli_read_overlays()
            ls = int(_read_choice("  HTML 图层像素 [1024]: ", "1024"))
            qs = int(_read_choice("  HTML 查询网格像素 [512]: ", "512"))
            np_ = _read_choice("  不生成 PNG？(y/N): ", "N").lower() == "y"
            ok, msg = action_html(HtmlOptions(
                seed, size, p, o, f, layer_size=ls, query_size=qs, no_png=np_))
            print(msg)
        elif c == "4":
            ok, msg = action_pytest()
            print(msg)
        elif c == "5":
            ok, missing = check_dependencies()
            if ok:
                print("✅ 全部基础依赖就位；PyQt5（GUI 用）：")
            gui_ok, gui_missing = check_dependencies(_GUI_DEPS)
            if gui_ok:
                print("   ✅ PyQt5 已安装，可运行 python main.py 打开 GUI")
            else:
                print(f"   ❌ PyQt5 缺失；安装：\n     {missing_install_hint(gui_missing)}")
        else:
            print("❓ 无效选项，重选。")


# ---------------------------------------------------------------------------
# 4. PyQt5 GUI（如果有）
# ---------------------------------------------------------------------------
def try_run_gui() -> bool:
    """尝试打开 PyQt5 窗口。安装成功返回 True，否则返回 False。"""
    try:
        from PyQt5 import QtCore, QtWidgets  # noqa: F401
    except Exception:
        return False

    from PyQt5.QtCore import QThread, pyqtSignal
    from PyQt5.QtGui import QFont
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
        QCheckBox, QTextEdit, QFileDialog, QMessageBox, QTabWidget,
        QSpinBox, QFormLayout, QHBoxLayout, QVBoxLayout, QGroupBox,
    )

    # 4.1 后台线程：把四个 action 移到 Worker 里，避免 UI 卡死
    class _Worker(QThread):
        """后台工作线程：在子线程中执行 action 函数，通过信号回传日志与结果。"""

        log = pyqtSignal(str)
        finished_ok = pyqtSignal(bool, str)

        def __init__(self, func, *args, **kwargs):
            """``__init__``: 保存要执行的函数及其参数。"""
            super().__init__(); self._f, self._a, self._kw = func, args, kwargs
        def run(self):  # noqa: D401
            """线程入口：调用目标函数并发出 finished_ok 信号，异常也转为失败信号。"""
            try:
                ok, msg = self._f(*self._a, **self._kw)
                self.finished_ok.emit(bool(ok), str(msg))
            except Exception as e:  # pragma: no cover
                self.finished_ok.emit(False, f"未预期异常：{e}\n{traceback.format_exc()}")

    class _Main(QMainWindow):
        """主窗口：五个标签页（生成/PNG/HTML/测试/依赖）+ 日志区。"""

        def __init__(self):
            """``__init__``: 构建窗口标题、各标签页与日志控件。"""
            super().__init__()
            self.setWindowTitle("Fantasy World Simulator — 启动器 v9.1.0")
            self.resize(980, 660)
            font = QFont(); font.setPointSize(12); self.setFont(font)

            root = QWidget(); self.setCentralWidget(root)
            v = QVBoxLayout(root)

            self.tabs = QTabWidget(); v.addWidget(self.tabs, 1)
            self.tabs.addTab(self._build_gen_tab(), "① 生成世界")
            self.tabs.addTab(self._build_vis_tab(), "② 4 面板 PNG")
            self.tabs.addTab(self._build_html_tab(), "③ HTML 报告")
            self.tabs.addTab(self._build_test_tab(), "④ 单元测试")
            self.tabs.addTab(self._build_deps_tab(), "⑤ 依赖自检")

            self.log = QTextEdit(readOnly=True); self.log.setPlaceholderText("操作输出会显示在这里…")
            v.addWidget(QLabel("📋 日志"))
            v.addWidget(self.log, 1)

        # ---- Tab builders ----
        def _spinsz(self, default=4096):
            """创建步进 256、范围 256~8192 的尺寸输入框并返回。"""
            box = QSpinBox(); box.setRange(256, 8192); box.setValue(default); box.setSingleStep(256)
            return box
        def _spinseed(self, default=1):
            """创建 32 位整数范围的 seed 输入框并返回。"""
            box = QSpinBox(); box.setRange(0, 2**31 - 1); box.setValue(default)
            return box

        def _build_gen_tab(self):
            """构建“生成世界”标签页表单并返回该控件。"""
            w = QWidget(); form = QFormLayout(w)
            self.gen_seed = self._spinseed(1)
            self.gen_size = self._spinsz(4096)
            self.gen_out = QLineEdit(str(_ensure_out_dir()))
            btn_pick = QPushButton("选择…"); btn_pick.clicked.connect(
                lambda: self.gen_out.setText(QFileDialog.getExistingDirectory(self, "输出目录", self.gen_out.text()) or self.gen_out.text()))
            row = QHBoxLayout(); row.addWidget(self.gen_out, 1); row.addWidget(btn_pick)
            r = QWidget(); r.setLayout(row)
            form.addRow("Seed", self.gen_seed)
            form.addRow("Size (像素)", self.gen_size)
            form.addRow("输出目录", r)
            btn = QPushButton("🚀 开始生成")
            btn.clicked.connect(lambda: self._start("gen"))
            form.addRow(btn)
            return w

        def _overlay_group(self, title: str = "叠加元素"):
            """构建探针/多边形/剖面三个输入框的分组框并返回。"""
            g = QGroupBox(title); f = QFormLayout(g)
            self.vis_points  = QLineEdit(); self.vis_points.setPlaceholderText("32.0,100.0,长安;31.0,110.0,洛阳")
            self.vis_poly    = QLineEdit(); self.vis_poly.setPlaceholderText("30,100;32,102;30,104;28,102")
            self.vis_profile = QLineEdit(); self.vis_profile.setPlaceholderText("35,90;25,120")
            f.addRow("探针  points (lat,lon,name;...)", self.vis_points)
            f.addRow("多边形 polygon (lat,lon;...)",    self.vis_poly)
            f.addRow("剖面   profile (latA,lonA,latB,lonB)", self.vis_profile)
            return g

        def _build_vis_tab(self):
            """构建“4 面板 PNG”标签页表单并返回该控件。"""
            w = QWidget(); box = QVBoxLayout(w)
            form = QFormLayout(); box.addLayout(form)
            self.vis_seed = self._spinseed(1); self.vis_size = self._spinsz(4096)
            self.vis_out = QLineEdit(str(_ensure_out_dir() / "geoctx_visual_seed1.png"))
            btn_pick = QPushButton("另存为…"); btn_pick.clicked.connect(
                lambda: self.vis_out.setText(QFileDialog.getSaveFileName(
                    self, "保存 PNG", self.vis_out.text(), "PNG (*.png)")[0] or self.vis_out.text()))
            row = QHBoxLayout(); row.addWidget(self.vis_out, 1); row.addWidget(btn_pick)
            r = QWidget(); r.setLayout(row)
            form.addRow("Seed", self.vis_seed); form.addRow("Size", self.vis_size)
            form.addRow("输出 PNG", r)
            box.addWidget(self._overlay_group())
            btn = QPushButton("🎨 生成 4 面板 PNG")
            btn.clicked.connect(lambda: self._start("vis"))
            box.addWidget(btn); box.addStretch(1)
            return w

        def _build_html_tab(self):
            """构建“HTML 报告”标签页表单并返回该控件。"""
            w = QWidget(); box = QVBoxLayout(w)
            form = QFormLayout(); box.addLayout(form)
            self.html_seed = self._spinseed(1); self.html_size = self._spinsz(4096)
            self.html_ls = self._spinsz(1024); self.html_qs = self._spinsz(512)
            self.html_no_png = QCheckBox("不生成 4 面板 PNG（只生成 HTML 会略快）")
            self.html_out = QLineEdit(str(_ensure_out_dir() / "geoctx_report_seed1.html"))
            btn_pick = QPushButton("另存为…"); btn_pick.clicked.connect(
                lambda: self.html_out.setText(QFileDialog.getSaveFileName(
                    self, "保存 HTML", self.html_out.text(), "HTML (*.html)")[0] or self.html_out.text()))
            row = QHBoxLayout(); row.addWidget(self.html_out, 1); row.addWidget(btn_pick)
            r = QWidget(); r.setLayout(row)
            form.addRow("Seed", self.html_seed); form.addRow("Size", self.html_size)
            form.addRow("图层像素 layer_size", self.html_ls)
            form.addRow("查询像素 query_size", self.html_qs)
            form.addRow("输出 HTML", r); form.addRow(self.html_no_png)
            box.addWidget(self._overlay_group("报告叠加元素"))
            btn = QPushButton("🌐 生成 HTML 交互报告")
            btn.clicked.connect(lambda: self._start("html"))
            box.addWidget(btn); box.addStretch(1)
            return w

        def _build_test_tab(self):
            """构建“单元测试”标签页（说明文本 + pytest 参数输入）并返回该控件。"""
            w = QWidget(); v = QVBoxLayout(w)
            v.addWidget(QLabel(textwrap.dedent("""\
                <p>运行 <code>terrain/tests</code> 下的 pytest 用例（目前 33 项，覆盖：
                CLI 解析、WorldGeoContext 单例缓存、build_geo_figure 输出、
                main() 端到端 CLI 烟雾）。</p>
                <p>额外参数（留空=默认 <code>-q</code>）：</p>
            """)))
            self.test_args = QLineEdit(placeholderText="例如 -v -k cache  （只跑缓存类用例）")
            v.addWidget(self.test_args)
            btn = QPushButton("🧪 运行 pytest"); btn.clicked.connect(lambda: self._start("test"))
            v.addWidget(btn); v.addStretch(1)
            return w

        def _build_deps_tab(self):
            """构建“依赖自检”标签页并立即刷新一次依赖状态。"""
            w = QWidget(); v = QVBoxLayout(w)
            self.deps_text = QTextEdit(readOnly=True)
            btn = QPushButton("🔎 重新检查依赖")
            btn.clicked.connect(self._refresh_deps)
            v.addWidget(btn); v.addWidget(self.deps_text, 1)
            self._refresh_deps()
            return w

        # ---- 辅助：写日志、开启工作线程 ----
        def _append(self, s: str):
            """把字符串 s 追加到日志区并滚动到底部。"""
            self.log.moveCursor(1)  # QTextCursor.End = 1
            # 用更简单的方式：setPlainText 累加
            self.log.setPlainText(self.log.toPlainText() + ("\n" if self.log.toPlainText() else "") + s)
            self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

        def _refresh_deps(self):
            """重新检查基础/GUI 依赖并显示在依赖自检文本框中。"""
            lines = ["--- 基础依赖 (numpy/matplotlib/scipy/torch/pytest/Pillow) ---"]
            ok, missing = check_dependencies()
            lines.append("✅ 全部就位" if ok else f"❌ 缺失：{' '.join(n for n,_,_ in missing)}\n   安装：{missing_install_hint(missing)}")
            lines.append("\n--- GUI 依赖 (PyQt5) ---")
            gok, gmiss = check_dependencies(_GUI_DEPS)
            lines.append("✅ PyQt5 就绪" if gok else f"❌ PyQt5 缺失；安装：{missing_install_hint(gmiss)}")
            self.deps_text.setPlainText("\n".join(lines))

        def _set_busy(self, busy: bool):
            """设置忙状态：忙时禁用全部标签页，防止并发操作。"""
            for i in range(self.tabs.count()):
                self.tabs.widget(i).setEnabled(not busy)
            self.tabs.setEnabled(not busy)

        def _start(self, which: str):
            """按 which 组装对应 Options 并启动后台 Worker 执行。"""
            self._set_busy(True)
            self._append(f"▶ [{which.upper()}] 开始执行…")
            if which == "gen":
                opts = GenerateOptions(seed=self.gen_seed.value(),
                                       size=self.gen_size.value(),
                                       out_dir=Path(self.gen_out.text()))
                worker = _Worker(action_generate, opts)
            elif which == "vis":
                opts = VisualizeOptions(
                    seed=self.vis_seed.value(), size=self.vis_size.value(),
                    points=(self.vis_points.text().strip() or None),
                    polygon=(self.vis_poly.text().strip() or None),
                    profile=(self.vis_profile.text().strip() or None),
                    out=Path(self.vis_out.text()),
                )
                worker = _Worker(action_visualize, opts)
            elif which == "html":
                opts = HtmlOptions(
                    seed=self.html_seed.value(), size=self.html_size.value(),
                    points=(self.vis_points.text().strip() or None),
                    polygon=(self.vis_poly.text().strip() or None),
                    profile=(self.vis_profile.text().strip() or None),
                    layer_size=self.html_ls.value(), query_size=self.html_qs.value(),
                    no_png=self.html_no_png.isChecked(),
                    html_out=Path(self.html_out.text()),
                )
                worker = _Worker(action_html, opts)
            else:  # test
                extra = (self.test_args.text().strip().split() or None)
                worker = _Worker(action_pytest, extra)

            worker.log.connect(self._append)
            def _done(ok, msg):
                """Worker 完成回调：写日志、解除忙状态并弹窗提示结果。"""
                self._append(msg); self._set_busy(False)
                QMessageBox.information(self, "完成" if ok else "失败", msg)
            worker.finished_ok.connect(_done)
            worker.start()
            self._worker = worker  # keep ref

    app = QApplication.instance() or QApplication(sys.argv)
    win = _Main(); win.show()
    return app.exec_() == 0


# ---------------------------------------------------------------------------
# 5. CLI 子命令直跑（用于 CI / 脚本）
# ---------------------------------------------------------------------------
def _build_argparse() -> argparse.ArgumentParser:
    """构建 CLI 参数解析器（含 generate/visualize/html/pytest 子命令），返回 ArgumentParser。"""
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Fantasy World Simulator — 项目启动入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            示例：
              python main.py                           → GUI（无 PyQt5 则降级 CLI 菜单）
              python main.py --cli                     → CLI 菜单
              python main.py generate -s 42 --size 4096
              python main.py visualize -s 42 --points "32,100,长安;31,110,洛阳"
              python main.py html -s 42 --profile "35,90;25,120" --no-png
              python main.py pytest -xvs -k cache
        """))
    p.add_argument("--cli", action="store_true", help="强制使用交互式 CLI 菜单，不启动 GUI")
    p.add_argument("--no-gui", action="store_true", help="同 --cli")
    sub = p.add_subparsers(dest="cmd", metavar="<cmd>")

    def _add_sp(parser, name, h):
        """在 parser 下注册名为 name 的子命令解析器并返回。"""
        sp = parser.add_parser(name, help=h); sp.set_defaults(cmd=name)
        return sp

    # generate
    sp = _add_sp(sub, "generate", "生成世界 DEM + 预览骨架 PNG")
    sp.add_argument("-s", "--seed", type=int, default=1)
    sp.add_argument("--size", type=int, default=4096, choices=[256, 512, 1024, 2048, 4096])
    sp.add_argument("-o", "--out-dir", type=str, default=None, help="输出目录（默认 terrain/output/）")

    # visualize
    sp = _add_sp(sub, "visualize", "4 面板 PNG 可视化")
    sp.add_argument("-s", "--seed", type=int, default=1)
    sp.add_argument("--size", type=int, default=4096)
    sp.add_argument("--points", type=str, default=None)
    sp.add_argument("--polygon", type=str, default=None)
    sp.add_argument("--profile", type=str, default=None)
    sp.add_argument("-o", "--out", type=str, default=None, help="PNG 输出路径")

    # html
    sp = _add_sp(sub, "html", "导出 HTML 交互报告（可选连带 PNG）")
    sp.add_argument("-s", "--seed", type=int, default=1)
    sp.add_argument("--size", type=int, default=4096)
    sp.add_argument("--points", type=str, default=None)
    sp.add_argument("--polygon", type=str, default=None)
    sp.add_argument("--profile", type=str, default=None)
    sp.add_argument("--layer-size", type=int, default=1024)
    sp.add_argument("--query-size", type=int, default=512)
    sp.add_argument("--no-png", action="store_true")
    sp.add_argument("-o", "--html-out", type=str, default=None, help="HTML 输出路径")

    # pytest
    sp = _add_sp(sub, "pytest", "直接把其后的参数交给 `python -m pytest terrain/tests`")
    sp.add_argument("pytest_args", nargs=argparse.REMAINDER,
                    help="传给 pytest 的额外参数，例如 -xvs -k cache")
    return p


def run_subcommand(cmd: str, args: argparse.Namespace) -> int:
    """按子命令名分发到对应 action，返回进程退出码（0 成功 / 2 失败 / 3 pytest 失败 / 1 未知命令）。"""
    if cmd == "generate":
        ok, msg = action_generate(GenerateOptions(args.seed, args.size, args.out_dir))
        print(msg); return 0 if ok else 2
    if cmd == "visualize":
        ok, msg = action_visualize(VisualizeOptions(
            args.seed, args.size, args.points, args.polygon, args.profile,
            Path(args.out) if args.out else None))
        print(msg); return 0 if ok else 2
    if cmd == "html":
        ok, msg = action_html(HtmlOptions(
            args.seed, args.size, args.points, args.polygon, args.profile,
            args.layer_size, args.query_size, args.no_png,
            Path(args.html_out) if args.html_out else None))
        print(msg); return 0 if ok else 2
    if cmd == "pytest":
        extra = list(args.pytest_args or [])
        # argparse.REMAINDER 经常带一个前缀 "--"，去掉
        if extra and extra[0] == "--":
            extra = extra[1:]
        ok, msg = action_pytest(extra)
        print(msg); return 0 if ok else 3
    return 1


def main(argv: Optional[list[str]] = None) -> int:
    """程序入口：有子命令则直跑，否则按 --cli/依赖情况选择 GUI 或 CLI 菜单，返回退出码。"""
    parser = _build_argparse()
    args = parser.parse_args(argv)
    if args.cmd:
        return run_subcommand(args.cmd, args)

    # 无 subcommand：优先 GUI，--cli / PyQt5 缺失 → CLI 菜单
    if args.cli or args.no_gui:
        return run_cli_menu()
    if not shutil.which("python"):
        pass  # no-op
    ok, _ = check_dependencies(_GUI_DEPS)
    if ok:
        try:
            if try_run_gui():
                return 0
            # try_run_gui 返回 False 意味着窗口未启动 → 回退 CLI
        except Exception as e:
            print(f"⚠️  GUI 启动异常：{e}，回退到 CLI 菜单")
    print("ℹ️  未检测到 PyQt5，回退为 CLI 交互式菜单。")
    print("   安装 GUI：pip install PyQt5")
    return run_cli_menu()


if __name__ == "__main__":
    raise SystemExit(main())
