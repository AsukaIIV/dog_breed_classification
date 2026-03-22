"""
compare_models.py
加载基线 + 改进模型，在测试集上完整推理，生成对比图与报告

生成四个子图：
  1. 分组柱状图（四项指标 + 提升标注）
  2. 雷达图
  3. 提升幅度横向条形图
  4. 每类准确率散点图（120 类，直观显示哪些类提升/下降）

用法：
    python compare_models.py \
        --data      ./data/stanford_dogs \
        --ckpt_base checkpoints/best_baseline_stage3.pth \
        --ckpt_imp  checkpoints/best_stage3.pth \
        --save_dir  ./figures
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
import timm
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

from model   import EfficientNetB3WithCBAM
from dataset import build_dataloaders

import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
# 字体
# ──────────────────────────────────────────────

def _setup_font():
    candidates = ["Microsoft YaHei", "SimHei", "SimSun",
                  "PingFang SC", "Noto Sans CJK SC"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams.update({
                "font.sans-serif":    [name, "DejaVu Sans"],
                "font.family":        "sans-serif",
                "axes.unicode_minus": False,
            })
            return
    for f in fm.fontManager.ttflist:
        if any(k in f.name for k in ["CJK", "Gothic", "Hei", "Song"]):
            plt.rcParams.update({
                "font.sans-serif":    [f.name, "DejaVu Sans"],
                "axes.unicode_minus": False,
            })
            return

_setup_font()


# ──────────────────────────────────────────────
# 基线模型（直接定义，避免导入副作用）
# ──────────────────────────────────────────────

class EfficientNetB3Baseline(nn.Module):
    def __init__(self, num_classes=120, dropout=0.3, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b3", pretrained=pretrained,
            num_classes=0, global_pool="",
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier  = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1536, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.global_pool(x).flatten(1)
        return self.classifier(x)


# ──────────────────────────────────────────────
# 自动检测并加载模型
# ──────────────────────────────────────────────

def load_model(ckpt_path: str, num_classes: int, device):
    ckpt     = torch.load(ckpt_path, map_location=device)
    keys     = set(ckpt["model"].keys())
    is_cbam  = any(k.startswith("cbam_modules") for k in keys)

    if is_cbam:
        model = EfficientNetB3WithCBAM(num_classes=num_classes, pretrained=False)
        tag   = "改进模型（+CBAM）"
    else:
        model = EfficientNetB3Baseline(num_classes=num_classes, pretrained=False)
        tag   = "基线模型"

    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    print(f"  [{tag}] {ckpt_path}  val_acc={ckpt.get('val_acc', 'N/A')}")
    return model, tag


# ──────────────────────────────────────────────
# 推理
# ──────────────────────────────────────────────

@torch.no_grad()
def predict(model, loader, device, desc="推理"):
    preds_l, labels_l = [], []
    for imgs, labels in tqdm(loader, desc=desc, leave=False):
        logits = model(imgs.to(device))
        preds_l.append(logits.argmax(1).cpu())
        labels_l.append(labels)
    return (
        torch.cat(preds_l).numpy(),
        torch.cat(labels_l).numpy(),
    )


# ──────────────────────────────────────────────
# 指标
# ──────────────────────────────────────────────

def calc_metrics(preds, labels):
    return {
        "accuracy" : (preds == labels).mean(),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall"   : recall_score   (labels, preds, average="macro", zero_division=0),
        "f1"       : f1_score       (labels, preds, average="macro", zero_division=0),
    }


# ──────────────────────────────────────────────
# 配色
# ──────────────────────────────────────────────

C_BASE = "#5B8DB8"
C_IMP  = "#E8734A"
C_DIFF = "#4CAF82"
C_BG   = "#F8F9FA"
C_GRID = "#E0E4E8"
C_TEXT = "#2C3E50"
C_SUB  = "#7F8C9A"

METRIC_KEYS   = ["accuracy", "precision", "recall", "f1"]
METRIC_LABELS = ["准确率", "宏平均精确率", "宏平均召回率", "宏平均F1分数"]


# ──────────────────────────────────────────────
# 子图 1：分组柱状图
# ──────────────────────────────────────────────

def plot_grouped_bar(ax, m_base, m_imp):
    vals_b = np.array([m_base[k] for k in METRIC_KEYS])
    vals_i = np.array([m_imp [k] for k in METRIC_KEYS])
    delta  = (vals_i - vals_b) * 100
    x      = np.arange(len(METRIC_LABELS))
    w      = 0.32

    bars_b = ax.bar(x - w / 2, vals_b, w, color=C_BASE,
                    alpha=0.92, label="基线模型", zorder=3)
    bars_i = ax.bar(x + w / 2, vals_i, w, color=C_IMP,
                    alpha=0.92, label="改进模型（+CBAM）", zorder=3)

    for bar, color in [(bars_b, C_BASE), (bars_i, C_IMP)]:
        for b in bar:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + 0.003,
                    f"{h:.4f}", ha="center", va="bottom",
                    fontsize=7.5, color=color, fontweight="bold")

    for i, (iv, d) in enumerate(zip(vals_i, delta)):
        ax.annotate(
            f"+{d:.2f}pp",
            xy=(x[i], iv + 0.003), xytext=(x[i], iv + 0.023),
            ha="center", va="bottom", fontsize=8,
            color=C_DIFF, fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color=C_DIFF,
                            lw=1.4, mutation_scale=10),
        )

    lo = max(0, vals_b.min() - 0.03)
    hi = vals_i.max() + 0.05
    ax.set_ylim(lo, hi)
    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_LABELS, fontsize=10)
    ax.set_ylabel("指标值", fontsize=10)
    ax.set_title("四项核心指标对比", fontsize=12, fontweight="bold", pad=10)
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(C_GRID)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9, edgecolor=C_GRID)


# ──────────────────────────────────────────────
# 子图 2：雷达图
# ──────────────────────────────────────────────

def plot_radar(ax, m_base, m_imp):
    vals_b = np.array([m_base[k] for k in METRIC_KEYS])
    vals_i = np.array([m_imp [k] for k in METRIC_KEYS])
    n      = len(METRIC_LABELS)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    lo = min(vals_b.min(), vals_i.min()) - 0.02
    hi = max(vals_b.max(), vals_i.max()) + 0.015

    for r in np.linspace(lo, hi, 5):
        ax.plot(angles, [r] * (n + 1), color=C_GRID,
                linewidth=0.6, linestyle="--", zorder=1)

    for vals, color in [(vals_b, C_BASE), (vals_i, C_IMP)]:
        v = vals.tolist() + [vals[0]]
        ax.plot(angles, v, color=color, linewidth=2, zorder=3)
        ax.fill(angles, v, color=color, alpha=0.18, zorder=2)
        ax.scatter(angles[:-1], vals, color=color, s=45, zorder=4)

    ax.set_thetagrids(np.degrees(angles[:-1]), METRIC_LABELS, fontsize=9)
    ax.set_ylim(lo, hi)
    ax.set_yticks(np.linspace(lo, hi, 5))
    ax.set_yticklabels(
        [f"{v:.3f}" for v in np.linspace(lo, hi, 5)],
        fontsize=6.5, color=C_SUB,
    )
    ax.grid(color=C_GRID, linewidth=0.6)
    ax.set_title("雷达图综合对比", fontsize=12, fontweight="bold", pad=20)
    ax.legend(
        handles=[
            mpatches.Patch(facecolor=C_BASE, alpha=0.8, label="基线模型"),
            mpatches.Patch(facecolor=C_IMP,  alpha=0.8, label="改进模型（+CBAM）"),
        ],
        loc="lower left", fontsize=9, framealpha=0.9, edgecolor=C_GRID,
        bbox_to_anchor=(-0.15, -0.08),
    )


# ──────────────────────────────────────────────
# 子图 3：提升幅度条形图
# ──────────────────────────────────────────────

def plot_delta(ax, m_base, m_imp):
    vals_b = np.array([m_base[k] for k in METRIC_KEYS])
    vals_i = np.array([m_imp [k] for k in METRIC_KEYS])
    delta  = (vals_i - vals_b) * 100
    y      = np.arange(len(METRIC_LABELS))

    ax.barh(y, delta, color=C_DIFF, alpha=0.85, height=0.45, zorder=3)
    for yi, d in zip(y, delta):
        ax.text(d + 0.02, yi, f"+{d:.2f} pp",
                va="center", ha="left", fontsize=9,
                color=C_DIFF, fontweight="bold")

    mean_d = delta.mean()
    ax.axvline(mean_d, color=C_IMP, linestyle="--", linewidth=1.4,
               label=f"均值 {mean_d:.2f} pp", zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels(METRIC_LABELS, fontsize=10)
    ax.set_xlabel("绝对提升（百分点）", fontsize=10)
    ax.set_xlim(0, delta.max() * 1.6)
    ax.set_title("各指标绝对提升幅度", fontsize=12, fontweight="bold", pad=10)
    ax.xaxis.grid(True, color=C_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(C_GRID)
    ax.legend(fontsize=9, framealpha=0.9, edgecolor=C_GRID)


# ──────────────────────────────────────────────
# 子图 4：每类准确率散点图
# ──────────────────────────────────────────────

def plot_per_class_scatter(ax, preds_b, preds_i, labels, n_classes):
    per_b = np.zeros(n_classes)
    per_i = np.zeros(n_classes)
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() > 0:
            per_b[c] = (preds_b[mask] == c).mean()
            per_i[c] = (preds_i[mask] == c).mean()

    improved = per_i > per_b
    worse    = per_i < per_b
    equal    = ~improved & ~worse

    ax.scatter(per_b[improved], per_i[improved], alpha=0.6, s=22,
               color=C_IMP,  label=f"提升 ({improved.sum()} 类)", zorder=3)
    ax.scatter(per_b[worse],    per_i[worse],    alpha=0.6, s=22,
               color=C_BASE, label=f"下降 ({worse.sum()} 类)",    zorder=3)
    ax.scatter(per_b[equal],    per_i[equal],    alpha=0.6, s=22,
               color=C_SUB,  label=f"持平 ({equal.sum()} 类)",    zorder=3)

    ax.plot([0, 1], [0, 1], "--", color=C_GRID, linewidth=1.2, zorder=1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("基线模型 每类准确率", fontsize=10)
    ax.set_ylabel("改进模型 每类准确率", fontsize=10)
    ax.set_title(f"每类准确率散点图（共 {n_classes} 类）",
                 fontsize=12, fontweight="bold", pad=10)
    ax.xaxis.grid(True, color=C_GRID, linewidth=0.6, zorder=0)
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(C_GRID)
    ax.legend(fontsize=9, framealpha=0.9, edgecolor=C_GRID, loc="upper left")


# ──────────────────────────────────────────────
# 汇总打印
# ──────────────────────────────────────────────

def print_summary(m_base, m_imp):
    print("\n" + "=" * 62)
    print(f"  {'指标':<14} {'基线模型':>10} {'改进模型':>10} {'提升(pp)':>10}")
    print("-" * 62)
    for key, label in zip(METRIC_KEYS, METRIC_LABELS):
        b, i = m_base[key], m_imp[key]
        print(f"  {label:<14} {b:>10.4f} {i:>10.4f} {(i-b)*100:>+10.2f}")
    print("=" * 62 + "\n")


# ──────────────────────────────────────────────
# 主绘图
# ──────────────────────────────────────────────

def make_figure(m_base, m_imp, preds_b, preds_i,
                labels, n_classes, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 13), facecolor=C_BG)
    fig.patch.set_facecolor(C_BG)

    fig.text(0.5, 0.975,
             "EfficientNet-B3  vs  EfficientNet-B3 + CBAM  ·  模型性能对比",
             ha="center", va="top", fontsize=15,
             fontweight="bold", color=C_TEXT)
    fig.text(0.5, 0.943,
             "Stanford Dogs 120 类细粒度犬种识别  ·  测试集评估结果",
             ha="center", va="top", fontsize=10, color=C_SUB)

    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        top=0.91, bottom=0.06,
        left=0.07, right=0.97,
        hspace=0.40, wspace=0.32,
    )
    ax_bar     = fig.add_subplot(gs[0, 0])
    ax_radar   = fig.add_subplot(gs[0, 1], polar=True)
    ax_delta   = fig.add_subplot(gs[1, 0])
    ax_scatter = fig.add_subplot(gs[1, 1])

    for ax in [ax_bar, ax_delta, ax_scatter]:
        ax.set_facecolor(C_BG)

    plot_grouped_bar      (ax_bar,     m_base, m_imp)
    plot_radar            (ax_radar,   m_base, m_imp)
    plot_delta            (ax_delta,   m_base, m_imp)
    plot_per_class_scatter(ax_scatter, preds_b, preds_i, labels, n_classes)

    fig.text(0.5, 0.01,
             "注：pp = 百分点；训练设置完全一致，三阶段渐进微调；"
             f"测试集共 {len(labels):,} 张图像",
             ha="center", va="bottom", fontsize=8, color=C_SUB)

    out = save_dir / "model_comparison.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"[✓] 对比图已保存至: {out}")
    return out


# ──────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="基线 vs 改进模型完整对比")
    p.add_argument("--data",        default="./data/stanford_dogs")
    p.add_argument("--ckpt_base",   required=True,
                   help="基线模型权重，例如 checkpoints/best_baseline_stage3.pth")
    p.add_argument("--ckpt_imp",    required=True,
                   help="改进模型权重，例如 checkpoints/best_stage3.pth")
    p.add_argument("--num_classes", type=int, default=120)
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--no_bbox",     action="store_true")
    p.add_argument("--save_dir",    default="./figures")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}\n")

    # ── 数据 ──
    loaders   = build_dataloaders(
        args.data, batch_size=args.batch_size,
        num_workers=args.num_workers, use_bbox=not args.no_bbox,
    )
    n_classes = args.num_classes

    # ── 加载两个模型 ──
    print("加载模型...")
    model_base, _ = load_model(args.ckpt_base, n_classes, device)
    model_imp,  _ = load_model(args.ckpt_imp,  n_classes, device)

    # ── 推理 ──
    print("\n推理中（测试集）...")
    preds_b, labels = predict(model_base, loaders["test"], device, "基线推理")
    preds_i, _      = predict(model_imp,  loaders["test"], device, "改进推理")

    # ── 计算指标 ──
    m_base = calc_metrics(preds_b, labels)
    m_imp  = calc_metrics(preds_i, labels)
    print_summary(m_base, m_imp)

    # ── 生成图 ──
    make_figure(m_base, m_imp, preds_b, preds_i,
                labels, n_classes, Path(args.save_dir))


if __name__ == "__main__":
    main()
