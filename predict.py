"""
predict.py
犬类品种分类系统 —— 推理与可视化脚本
用法：
    python predict.py --img 你的狗图片.jpg --model checkpoints/best_stage3.pth --data ./data/stanford_dogs
输出：
    prediction_result.png  （含输入图 + Top-5 预测概率条形图，可直接放入论文）
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")               # 无显示器环境也能保存图片
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager
from PIL import Image

# ── 中文字体配置 ──
def _setup_chinese_font():
    """自动查找系统中文字体，按优先级尝试"""
    candidates = [
        # Windows
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simsun.ttc",
        # macOS
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        # Linux
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            font_manager.fontManager.addfont(path)
            prop = font_manager.FontProperties(fname=path)
            matplotlib.rcParams["font.family"] = prop.get_name()
            return
    print("[警告] 未找到中文字体，中文将显示为方块，建议安装 SimHei")

_setup_chinese_font()
matplotlib.rcParams["axes.unicode_minus"] = False
import numpy as np

from model import EfficientNetB3WithCBAM
from dataset import StanfordDogsDataset   # 仅用于读取类别名

# ─────────────────────────────────────────────
# 常量（与训练保持一致）
# ─────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_SIZE      = 300


# ─────────────────────────────────────────────
# 工具：读取数据集类别列表
# ─────────────────────────────────────────────
def get_class_names(data_root: str):
    """从数据集 Images 目录读取 120 个品种名，去掉 nXXXXXXXX- 前缀"""
    image_dir = Path(data_root) / "Images"
    breed_dirs = sorted([d.name for d in image_dir.iterdir() if d.is_dir()])
    # 去掉 WordNet ID 前缀，如 "n02085620-Chihuahua" -> "Chihuahua"
    clean = []
    for b in breed_dirs:
        parts = b.split("-", 1)
        name = parts[1] if len(parts) == 2 else b
        name = name.replace("_", " ").title()
        clean.append(name)
    return clean


# ─────────────────────────────────────────────
# 推理
# ─────────────────────────────────────────────
def predict(img_path: str, model_path: str, data_root: str,
            topk: int = 5, save_path: str = "prediction_result.png"):

    # ── 设备 ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # ── 类别名 ──
    class_names = get_class_names(data_root)
    num_classes  = len(class_names)
    print(f"共 {num_classes} 个品种类别")

    # ── 加载模型 ──
    model = EfficientNetB3WithCBAM(num_classes=num_classes, pretrained=False)
    ckpt  = torch.load(model_path, map_location=device)
    # 兼容直接保存 state_dict 或包含 "model" 键的 checkpoint
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"模型已加载: {model_path}")

    # ── 图像预处理（与测试集一致）──
    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    img_pil = Image.open(img_path).convert("RGB")
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    # ── 推理 ──
    with torch.no_grad():
        logits = model(img_tensor)
        probs  = F.softmax(logits, dim=1)[0].cpu()

    top_probs, top_idxs = probs.topk(topk)
    top_probs = top_probs.numpy()
    top_names = [class_names[i] for i in top_idxs.numpy()]

    print("\n===== 预测结果 =====")
    for rank, (name, prob) in enumerate(zip(top_names, top_probs), 1):
        bar = "█" * int(prob * 40)
        print(f"  Top-{rank}: {name:<30s}  {prob*100:6.2f}%  {bar}")

    # ─────────────────────────────────────────────
    # 可视化：左侧输入图 + 右侧 Top-5 横向条形图
    # ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             gridspec_kw={"width_ratios": [1, 1.6]})
    fig.patch.set_facecolor("#F8F9FA")

    # —— 左：输入图像 ——
    ax_img = axes[0]
    ax_img.imshow(img_pil)
    ax_img.axis("off")
    ax_img.set_title("输入图像", fontsize=14, fontweight="bold",
                     pad=10, color="#2C3E50")

    # 预测标签框
    pred_label = f"预测: {top_names[0]}\n置信度: {top_probs[0]*100:.1f}%"
    ax_img.text(0.5, -0.04, pred_label, transform=ax_img.transAxes,
                ha="center", va="top", fontsize=12, color="white",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#2E86C1", alpha=0.85))

    # —— 右：Top-K 概率条形图 ——
    ax_bar = axes[1]
    ax_bar.set_facecolor("#F8F9FA")

    colors = ["#2E86C1"] + ["#85C1E9"] * (topk - 1)   # 第1名深蓝，其余浅蓝
    y_pos  = list(range(topk - 1, -1, -1))              # 从上到下排列

    bars = ax_bar.barh(y_pos, top_probs * 100, color=colors,
                       height=0.55, edgecolor="white", linewidth=0.8)

    # 每条 bar 右侧显示百分比
    for bar, prob in zip(bars, top_probs):
        ax_bar.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{prob*100:.1f}%", va="center", ha="left",
                    fontsize=11, color="#2C3E50", fontweight="bold")

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(top_names, fontsize=12, color="#2C3E50")
    ax_bar.set_xlabel("预测置信度 (%)", fontsize=12, color="#2C3E50")
    ax_bar.set_title(f"Top-{topk} 预测结果", fontsize=14,
                     fontweight="bold", color="#2C3E50", pad=10)
    ax_bar.set_xlim(0, min(top_probs[0] * 100 * 1.25, 100))
    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_bar.tick_params(axis="x", colors="#7F8C8D")
    ax_bar.xaxis.label.set_color("#7F8C8D")
    ax_bar.axvline(x=0, color="#BDC3C7", linewidth=1)

    # 标注模型信息
    model_info = "模型: EfficientNet-B3 + CBAM | 数据集: Stanford Dogs (120类)"
    fig.text(0.5, 0.01, model_info, ha="center", fontsize=9,
             color="#95A5A6", style="italic")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(save_path, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n结果图已保存至: {save_path}")


# ─────────────────────────────────────────────
# 批量推理（可选）：对多张图输出拼图
# ─────────────────────────────────────────────
def predict_batch(img_dir: str, model_path: str, data_root: str,
                  topk: int = 3, save_path: str = "batch_result.png",
                  max_imgs: int = 6):
    """
    对一个目录下的多张图批量推理，输出拼图（适合论文展示多样本效果）
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = get_class_names(data_root)

    model = EfficientNetB3WithCBAM(num_classes=len(class_names), pretrained=False)
    ckpt  = torch.load(model_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device).eval()

    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    img_paths = [p for p in Path(img_dir).iterdir()
                 if p.suffix.lower() in exts][:max_imgs]

    if not img_paths:
        print(f"目录 {img_dir} 中未找到图片")
        return

    ncols = 3
    nrows = (len(img_paths) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.5, nrows * 5))
    fig.patch.set_facecolor("#F0F4F8")
    axes = np.array(axes).flatten()

    for ax, img_path in zip(axes, img_paths):
        img_pil = Image.open(img_path).convert("RGB")
        tensor  = transform(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = F.softmax(model(tensor), dim=1)[0].cpu()
        top_probs, top_idxs = probs.topk(topk)
        top_names = [class_names[i] for i in top_idxs.numpy()]
        top_probs = top_probs.numpy()

        ax.imshow(img_pil)
        ax.axis("off")
        label_lines = "\n".join(
            [f"#{r+1} {n}  {p*100:.1f}%"
             for r, (n, p) in enumerate(zip(top_names, top_probs))]
        )
        ax.set_title(f"{img_path.stem}", fontsize=10,
                     fontweight="bold", color="#2C3E50")
        ax.text(0.5, -0.02, label_lines, transform=ax.transAxes,
                ha="center", va="top", fontsize=8.5, color="#1A252F",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white", alpha=0.85))

    # 隐藏多余的格子
    for ax in axes[len(img_paths):]:
        ax.set_visible(False)

    fig.suptitle("EfficientNet-B3 + CBAM  犬类品种分类系统 —— 推理结果展示",
                 fontsize=13, fontweight="bold", color="#2C3E50", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"批量结果图已保存至: {save_path}")


# ─────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="犬类品种分类推理脚本")
    parser.add_argument("--img",   type=str, default=None,
                        help="单张图片路径（与 --img_dir 二选一）")
    parser.add_argument("--img_dir", type=str, default=None,
                        help="批量推理的图片目录（与 --img 二选一）")
    parser.add_argument("--model", type=str,
                        default="checkpoints/best_stage3.pth",
                        help="模型权重路径")
    parser.add_argument("--data",  type=str,
                        default="./data/stanford_dogs",
                        help="Stanford Dogs 数据集根目录（用于读取类别名）")
    parser.add_argument("--topk",  type=int, default=5,
                        help="显示 Top-K 预测（默认 5）")
    parser.add_argument("--save",  type=str,
                        default="prediction_result.png",
                        help="输出图片保存路径")
    parser.add_argument("--max_imgs", type=int, default=6,
                        help="批量模式最多显示图片数（默认 6）")
    args = parser.parse_args()

    if args.img:
        predict(
            img_path   = args.img,
            model_path = args.model,
            data_root  = args.data,
            topk       = args.topk,
            save_path  = args.save,
        )
    elif args.img_dir:
        predict_batch(
            img_dir    = args.img_dir,
            model_path = args.model,
            data_root  = args.data,
            topk       = min(args.topk, 3),
            save_path  = args.save,
            max_imgs   = args.max_imgs,
        )
    else:
        parser.print_help()
        print("\n示例:")
        print("  单张: python predict.py --img dog.jpg --model checkpoints/best_stage3.pth --data ./data/stanford_dogs")
        print("  批量: python predict.py --img_dir ./test_images --model checkpoints/best_stage3.pth --data ./data/stanford_dogs")