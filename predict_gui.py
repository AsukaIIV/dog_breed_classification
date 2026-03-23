"""
predict_gui.py — 犬类品种分类系统图形化启动器
用法: python predict_gui.py  （与 predict.py 放同一目录）
"""

import os
import sys
import subprocess
import platform
import threading
from pathlib import Path

# ── 切换工作目录到脚本所在文件夹（双击也能用相对路径）
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── 自动安装 Pillow（GUI 自己的 Python 环境）
try:
    from PIL import Image, ImageTk
    PIL_OK = True
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "Pillow", "-q"], check=False)
    try:
        from PIL import Image, ImageTk
        PIL_OK = True
    except ImportError:
        PIL_OK = False

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ──────────────────────────────────────────────────────────────
# 配色 & 字体
# ──────────────────────────────────────────────────────────────
BG       = "#0F1117"
SURFACE  = "#1A1D27"
CARD     = "#22263A"
ACCENT   = "#2E86C1"
ACCENT2  = "#5DADE2"
TEXT     = "#E8EAF0"
TEXT_DIM = "#7B82A0"
SUCCESS  = "#27AE60"
ERROR    = "#E74C3C"

_sys = platform.system()
if _sys == "Darwin":
    F_TITLE = ("PingFang SC", 17, "bold")
    F_BOLD  = ("PingFang SC", 10, "bold")
    F_BODY  = ("PingFang SC", 10)
    F_SMALL = ("PingFang SC", 9)
    F_MONO  = ("Menlo", 9)
elif _sys == "Linux":
    F_TITLE = ("WenQuanYi Micro Hei", 15, "bold")
    F_BOLD  = ("WenQuanYi Micro Hei", 10, "bold")
    F_BODY  = ("WenQuanYi Micro Hei", 10)
    F_SMALL = ("WenQuanYi Micro Hei", 9)
    F_MONO  = ("Monospace", 9)
else:
    F_TITLE = ("Microsoft YaHei UI", 15, "bold")
    F_BOLD  = ("Microsoft YaHei UI", 10, "bold")
    F_BODY  = ("Microsoft YaHei UI", 10)
    F_SMALL = ("Microsoft YaHei UI", 9)
    F_MONO  = ("Consolas", 9)


# ──────────────────────────────────────────────────────────────
# 工具
# ──────────────────────────────────────────────────────────────
def _darken(hex_color, amount=30):
    r = max(int(hex_color[1:3], 16) - amount, 0)
    g = max(int(hex_color[3:5], 16) - amount, 0)
    b = max(int(hex_color[5:7], 16) - amount, 0)
    return f"#{r:02x}{g:02x}{b:02x}"


def flat_btn(parent, text, command, bg=ACCENT, fg=TEXT, padx=14, pady=5):
    hover = _darken(bg)
    btn = tk.Button(parent, text=text, command=command,
                    bg=bg, fg=fg, activebackground=hover, activeforeground=fg,
                    relief="flat", bd=0, cursor="hand2",
                    font=F_BOLD, padx=padx, pady=pady)
    btn.bind("<Enter>", lambda e: btn.config(bg=hover))
    btn.bind("<Leave>", lambda e: btn.config(bg=bg))
    return btn


def path_row(parent, label_text, var, picker_fn, label_width=10):
    frame = tk.Frame(parent, bg=SURFACE)
    frame.pack(fill="x", padx=14, pady=3)
    tk.Label(frame, text=label_text, font=F_SMALL, bg=SURFACE, fg=TEXT_DIM,
             width=label_width, anchor="w").pack(side="left")
    tk.Entry(frame, textvariable=var, bg=CARD, fg=TEXT, insertbackground=TEXT,
             relief="flat", bd=1, font=F_SMALL,
             highlightthickness=1, highlightcolor=ACCENT,
             highlightbackground=SURFACE).pack(
                 side="left", fill="x", expand=True, ipady=4, padx=(0, 6))
    flat_btn(frame, "选择…", picker_fn, bg=CARD, fg=TEXT_DIM,
             padx=8, pady=3).pack(side="left")
    return frame


# ──────────────────────────────────────────────────────────────
# 主窗口
# ──────────────────────────────────────────────────────────────
class PredictGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("犬类品种分类系统")
        self.configure(bg=BG)
        self.resizable(False, False)
        self._running = False
        self._preview_photo = None
        self._build_ui()
        self._center()

    def _center(self):
        self.update_idletasks()
        w, h = self.winfo_width(), self.winfo_height()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"+{(sw-w)//2}+{(sh-h)//2}")

    # ── 整体布局 ──────────────────────────────────────────────
    def _build_ui(self):
        pad = tk.Frame(self, bg=BG)
        pad.pack(padx=22, pady=18, fill="both", expand=True)

        hdr = tk.Frame(pad, bg=BG)
        hdr.pack(fill="x", pady=(0, 16))
        tk.Label(hdr, text="犬类品种分类推理器",
                 font=F_TITLE, bg=BG, fg=TEXT).pack(side="left")
        tk.Label(hdr, text="EfficientNet-B3 + CBAM",
                 font=F_SMALL, bg=BG, fg=TEXT_DIM).pack(side="right", anchor="s")

        cols = tk.Frame(pad, bg=BG)
        cols.pack(fill="both", expand=True)
        left  = tk.Frame(cols, bg=BG)
        right = tk.Frame(cols, bg=BG)
        left.pack(side="left", fill="y")
        right.pack(side="left", fill="both", expand=True, padx=(16, 0))

        self._build_params(left)
        self._build_preview(right)
        self._build_log(pad)
        self._build_status(pad)

    # ── 参数面板 ──────────────────────────────────────────────
    def _build_params(self, parent):
        card = tk.Frame(parent, bg=SURFACE)
        card.pack(fill="both", expand=True)

        def sec(text):
            tk.Label(card, text=text, font=F_BOLD, bg=SURFACE, fg=ACCENT2
                     ).pack(anchor="w", padx=14, pady=(14, 2))

        # 模式
        sec("运行模式")
        self._mode = tk.StringVar(value="single")
        mrow = tk.Frame(card, bg=SURFACE)
        mrow.pack(fill="x", padx=14, pady=(0, 4))
        for val, lbl in [("single", "单张图片"), ("batch", "批量目录")]:
            tk.Radiobutton(mrow, text=lbl, variable=self._mode, value=val,
                           bg=SURFACE, fg=TEXT, selectcolor=ACCENT,
                           activebackground=SURFACE, activeforeground=TEXT,
                           font=F_SMALL, command=self._toggle_mode,
                           ).pack(side="left", padx=(0, 14))

        # 输入
        sec("输入")
        self._img_var = tk.StringVar()
        self._dir_var = tk.StringVar()
        self._img_row = path_row(card, "图片文件", self._img_var, self._pick_img)
        self._dir_row = path_row(card, "图片目录", self._dir_var, self._pick_dir)

        # 模型 & 数据集
        sec("模型 & 数据集")
        self._model_var = tk.StringVar(value="checkpoints/best_stage3.pth")
        self._data_var  = tk.StringVar(value="./data/stanford_dogs")
        self._save_var  = tk.StringVar(value="prediction_result.png")
        path_row(card, "权重文件",   self._model_var, self._pick_model)
        path_row(card, "数据集目录", self._data_var,  self._pick_data)
        path_row(card, "保存路径",   self._save_var,  self._pick_save)

        # 高级参数
        sec("高级参数")
        adv = tk.Frame(card, bg=SURFACE)
        adv.pack(fill="x", padx=14, pady=(0, 8))
        tk.Label(adv, text="Top-K", font=F_SMALL, bg=SURFACE, fg=TEXT_DIM
                 ).pack(side="left")
        self._topk_var = tk.IntVar(value=5)
        tk.Spinbox(adv, from_=1, to=10, textvariable=self._topk_var,
                   width=4, bg=CARD, fg=TEXT, insertbackground=TEXT,
                   relief="flat", font=F_SMALL).pack(side="left", padx=(6, 16))
        tk.Label(adv, text="批量最多展示", font=F_SMALL, bg=SURFACE, fg=TEXT_DIM
                 ).pack(side="left")
        self._maximg_var = tk.IntVar(value=6)
        tk.Spinbox(adv, from_=1, to=24, textvariable=self._maximg_var,
                   width=4, bg=CARD, fg=TEXT, insertbackground=TEXT,
                   relief="flat", font=F_SMALL).pack(side="left", padx=(6, 0))

        # 按钮
        brow = tk.Frame(card, bg=SURFACE)
        brow.pack(fill="x", padx=14, pady=(6, 16))
        self._run_btn = flat_btn(brow, "▶  开始推理", self._run,
                                 bg=ACCENT, padx=18, pady=7)
        self._run_btn.pack(side="left")
        flat_btn(brow, "清除日志", self._clear_log,
                 bg=CARD, fg=TEXT_DIM, padx=12, pady=7).pack(
                     side="left", padx=(10, 0))

        self._toggle_mode()

    # ── 预览面板 ──────────────────────────────────────────────
    def _build_preview(self, parent):
        card = tk.Frame(parent, bg=SURFACE)
        card.pack(fill="both", expand=True)
        tk.Label(card, text="预测结果预览", font=F_BOLD,
                 bg=SURFACE, fg=ACCENT2).pack(anchor="w", padx=14, pady=(14, 6))
        self._canvas = tk.Canvas(card, width=430, height=270,
                                 bg=CARD, bd=0, highlightthickness=0)
        self._canvas.pack(padx=14)
        self._canvas.create_text(215, 135,
                                 text="推理完成后，\n结果图将显示在这里",
                                 fill=TEXT_DIM, font=F_BODY, justify="center")
        flat_btn(card, "↗  在外部打开图片", self._open_external,
                 bg=CARD, fg=TEXT_DIM, padx=12, pady=5).pack(pady=(8, 14))

    # ── 日志 ──────────────────────────────────────────────────
    def _build_log(self, parent):
        lf = tk.Frame(parent, bg=BG)
        lf.pack(fill="x", pady=(14, 0))
        tk.Label(lf, text="运行日志", font=F_BOLD,
                 bg=BG, fg=TEXT_DIM).pack(anchor="w")
        box = tk.Frame(lf, bg=SURFACE)
        box.pack(fill="x")
        self._log = tk.Text(box, height=7, bg=SURFACE, fg="#A8B4C8",
                            insertbackground=TEXT, relief="flat",
                            font=F_MONO, bd=0, padx=8, pady=6, wrap="word")
        sb = tk.Scrollbar(box, command=self._log.yview, bg=BG)
        self._log.config(yscrollcommand=sb.set, state="disabled")
        self._log.pack(side="left", fill="x", expand=True)
        sb.pack(side="right", fill="y")
        self._log.tag_configure("dim",     foreground=TEXT_DIM)
        self._log.tag_configure("success", foreground=SUCCESS)
        self._log.tag_configure("error",   foreground=ERROR)

    # ── 状态栏 ────────────────────────────────────────────────
    def _build_status(self, parent):
        sf = tk.Frame(parent, bg=BG)
        sf.pack(fill="x", pady=(6, 0))
        self._status_var = tk.StringVar(value="就绪")
        tk.Label(sf, textvariable=self._status_var,
                 font=F_SMALL, bg=BG, fg=TEXT_DIM).pack(side="left")
        self._progress = ttk.Progressbar(sf, mode="indeterminate", length=160)
        self._progress.pack(side="right")

    # ── 文件选择器 ────────────────────────────────────────────
    def _pick_img(self):
        p = filedialog.askopenfilename(
            title="选择一张狗图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp *.webp"),
                       ("所有文件", "*.*")])
        if p:
            self._img_var.set(p)
            self._show_thumbnail(p)

    def _pick_dir(self):
        p = filedialog.askdirectory(title="选择批量图片目录")
        if p:
            self._dir_var.set(p)

    def _pick_model(self):
        init = str(Path(self._model_var.get()).parent) if self._model_var.get() else "."
        p = filedialog.askopenfilename(
            title="选择模型权重文件", initialdir=init,
            filetypes=[("PyTorch 权重", "*.pth *.pt"), ("所有文件", "*.*")])
        if p:
            self._model_var.set(p)

    def _pick_data(self):
        p = filedialog.askdirectory(title="选择 Stanford Dogs 数据集根目录")
        if p:
            self._data_var.set(p)

    def _pick_save(self):
        p = filedialog.asksaveasfilename(
            title="选择保存路径", defaultextension=".png",
            filetypes=[("PNG 图片", "*.png"), ("所有文件", "*.*")])
        if p:
            self._save_var.set(p)

    def _open_external(self):
        save = self._save_var.get()
        abs_save = Path(save) if Path(save).is_absolute() else Path(os.getcwd()) / save
        if abs_save.exists():
            if _sys == "Windows":
                os.startfile(str(abs_save))
            elif _sys == "Darwin":
                subprocess.Popen(["open", str(abs_save)])
            else:
                subprocess.Popen(["xdg-open", str(abs_save)])
        else:
            messagebox.showinfo("提示", "还没有可预览的结果图")

    # ── 模式切换 ──────────────────────────────────────────────
    def _toggle_mode(self):
        if self._mode.get() == "single":
            self._img_row.pack(fill="x", padx=14, pady=3)
            self._dir_row.pack_forget()
        else:
            self._dir_row.pack(fill="x", padx=14, pady=3)
            self._img_row.pack_forget()

    # ── 缩略图（选图时立即预览）─────────────────────────────
    def _show_thumbnail(self, path):
        if not PIL_OK:
            return
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((430, 270))
            photo = ImageTk.PhotoImage(img)
            self._preview_photo = photo
            self._canvas.delete("all")
            self._canvas.create_image(215, 135, image=photo, anchor="center")
        except Exception:
            pass

    # ── 日志工具 ──────────────────────────────────────────────
    def _log_write(self, text, tag=None):
        self._log.config(state="normal")
        self._log.insert("end", text + "\n", tag or "")
        self._log.see("end")
        self._log.config(state="disabled")

    def _clear_log(self):
        self._log.config(state="normal")
        self._log.delete("1.0", "end")
        self._log.config(state="disabled")

    # ── 推理执行 ──────────────────────────────────────────────
    def _run(self):
        if self._running:
            return

        mode     = self._mode.get()
        model    = self._model_var.get().strip()
        data     = self._data_var.get().strip()
        topk     = self._topk_var.get()
        max_imgs = self._maximg_var.get()

        # 保存路径转绝对路径，保证 GUI 和子进程指向同一文件
        save_raw = self._save_var.get().strip()
        save_abs = str(Path(save_raw).resolve())

        if not model or not Path(model).exists():
            messagebox.showerror("错误", f"找不到模型权重文件:\n{model}")
            return
        if not data or not Path(data).exists():
            messagebox.showerror("错误", f"找不到数据集目录:\n{data}")
            return

        if mode == "single":
            img_path = self._img_var.get().strip()
            if not img_path or not Path(img_path).exists():
                messagebox.showerror("错误", f"找不到图片文件:\n{img_path}")
                return
            cmd = ["python", "predict.py",
                   "--img", img_path, "--model", model,
                   "--data", data, "--topk", str(topk), "--save", save_abs]
        else:
            img_dir = self._dir_var.get().strip()
            if not img_dir or not Path(img_dir).exists():
                messagebox.showerror("错误", f"找不到图片目录:\n{img_dir}")
                return
            cmd = ["python", "predict.py",
                   "--img_dir", img_dir, "--model", model,
                   "--data", data, "--topk", str(topk),
                   "--save", save_abs, "--max_imgs", str(max_imgs)]

        self._running = True
        self._run_btn.config(text="⏳ 推理中…", state="disabled")
        self._progress.start(12)
        self._status_var.set("正在运行 predict.py …")
        self._log_write("$ " + " ".join(cmd), tag="dim")

        def worker():
            try:
                enc = "gbk" if _sys == "Windows" else "utf-8"
                proc = subprocess.Popen(
                    " ".join(f'"{c}"' if " " in c else c for c in cmd),
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding=enc, errors="replace",
                    shell=True)
                for line in proc.stdout:
                    line = line.rstrip()
                    if line:
                        self.after(0, self._log_write, line)
                proc.wait()
                if proc.returncode == 0:
                    self.after(0, self._on_success, save_abs)
                else:
                    self.after(0, self._on_fail, proc.returncode)
            except Exception as ex:
                self.after(0, self._log_write, f"[异常] {ex}", "error")
                self.after(0, self._on_fail, -1)

        threading.Thread(target=worker, daemon=True).start()

    def _on_success(self, save_path):
        self._running = False
        self._progress.stop()
        self._run_btn.config(text="▶  开始推理", state="normal")
        self._status_var.set("✅ 推理完成！")
        self._log_write("✅ 推理完成！", tag="success")
        if not PIL_OK:
            self._log_write("提示: 当前 Python 无 Pillow，预览不可用", tag="dim")
            return
        p = Path(save_path)
        if p.exists():
            try:
                img = Image.open(p)
                img.thumbnail((430, 270))
                photo = ImageTk.PhotoImage(img)
                self._preview_photo = photo
                self._canvas.delete("all")
                self._canvas.create_image(215, 135, image=photo, anchor="center")
            except Exception as e:
                self._log_write(f"[预览失败] {e}", tag="error")
        else:
            self._log_write(f"[预览] 找不到文件: {p}", tag="error")

    def _on_fail(self, code):
        self._running = False
        self._progress.stop()
        self._run_btn.config(text="▶  开始推理", state="normal")
        self._status_var.set(f"❌ 推理失败（退出码 {code}）")
        self._log_write(f"❌ 进程异常退出，代码 {code}", tag="error")


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = PredictGUI()
    style = ttk.Style(app)
    try:
        style.theme_use("clam")
    except Exception:
        pass
    style.configure("TProgressbar", troughcolor=CARD, background=ACCENT, thickness=4)
    app.mainloop()