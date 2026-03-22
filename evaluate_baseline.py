"""
evaluate_baseline.py
模型评估脚本：
- 计算准确率、精确率、召回率、F1 分数
- 绘制混淆矩阵（全量 + Top-K 易混淆子集）
- Grad-CAM 热力图可视化（对比基线 vs 改进模型）

修改说明：
- 新增 --baseline_only 参数，支持单独评估基线模型
- 修复 Grad-CAM 对比时基线模型错误使用 EfficientNetB3WithCBAM 的 bug
- 基线评估结果独立存放于 save_dir/baseline/ 子目录
- EfficientNetB3Baseline 直接在本文件定义，避免导入 train_baseline 触发副作用
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ── Windows 中文字体自动检测 ──────────────────────────────
def _setup_chinese_font():
    candidates = [
        "Microsoft YaHei", "SimHei", "SimSun", "KaiTi", "FangSong",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["font.family"]     = "sans-serif"
            plt.rcParams["axes.unicode_minus"] = False
            print(f"[字体] 使用: {name}")
            return name
    for f in fm.fontManager.ttflist:
        if any(k in f.name for k in ["CJK", "Gothic", "Hei", "Song"]):
            plt.rcParams["font.sans-serif"] = [f.name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            print(f"[字体] 使用备选: {f.name}")
            return f.name
    print("[字体] 未找到中文字体，图表标题可能显示为方块")
    return None

_FONT = _setup_chinese_font()

import matplotlib.patches as mpatches
import seaborn as sns
sns.set_theme(style="white", font=plt.rcParams["font.sans-serif"][0])
from tqdm import tqdm
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
import timm

from model   import EfficientNetB3WithCBAM
from dataset import build_dataloaders, StanfordDogsDataset, build_transforms


# ─────────────────────────────────────────────
# 基线模型（直接定义，不从 train_baseline 导入）
# 与 train_baseline.EfficientNetB3Baseline 结构完全一致
# ─────────────────────────────────────────────

class EfficientNetB3Baseline(nn.Module):
    """标准 EfficientNet-B3 + 自定义分类头，不含 CBAM"""

    def __init__(self, num_classes: int = 120, dropout: float = 0.3,
                 pretrained: bool = False):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b3",
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier  = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1536, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.global_pool(x).flatten(1)
        return self.classifier(x)


# ─────────────────────────────────────────────
# 推理
# ─────────────────────────────────────────────

@torch.no_grad()
def predict(model, loader, device) -> tuple:
    """返回 (all_preds, all_labels, all_probs)"""
    model.eval()
    preds_list, labels_list, probs_list = [], [], []

    for imgs, labels in tqdm(loader, desc="推理中"):
        imgs = imgs.to(device)
        logits = model(imgs)
        probs  = F.softmax(logits, dim=1).cpu()
        preds  = logits.argmax(1).cpu()
        preds_list.append(preds)
        labels_list.append(labels)
        probs_list.append(probs)

    return (
        torch.cat(preds_list).numpy(),
        torch.cat(labels_list).numpy(),
        torch.cat(probs_list).numpy(),
    )


# ─────────────────────────────────────────────
# 指标计算
# ─────────────────────────────────────────────

def compute_metrics(preds, labels, class_names, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)

    acc = (preds == labels).mean()
    p   = precision_score(labels, preds, average="macro", zero_division=0)
    r   = recall_score   (labels, preds, average="macro", zero_division=0)
    f1  = f1_score       (labels, preds, average="macro", zero_division=0)

    print(f"\n{'='*50}")
    print(f"  准确率  (Accuracy) : {acc:.4f}")
    print(f"  精确率  (Precision): {p:.4f}")
    print(f"  召回率  (Recall)   : {r:.4f}")
    print(f"  F1 分数 (F1-Score) : {f1:.4f}")
    print(f"{'='*50}")

    # 保存详细报告
    report = classification_report(
        labels, preds, target_names=class_names, zero_division=0
    )
    (save_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    print("详细分类报告已保存至", save_dir / "classification_report.txt")

    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}


# ─────────────────────────────────────────────
# 混淆矩阵
# ─────────────────────────────────────────────

def plot_confusion_matrix(preds, labels, class_names,
                           save_dir: Path, top_k: int = 20):
    save_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(labels, preds)

    # ── 全量混淆矩阵（归一化）──
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    fig, ax = plt.subplots(figsize=(40, 38))
    sns.heatmap(cm_norm, ax=ax, cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                vmin=0, vmax=1, linewidths=0.1)
    ax.set_xlabel("预测类别", fontsize=12)
    ax.set_ylabel("真实类别", fontsize=12)
    ax.set_title(f"混淆矩阵（归一化，共 {len(class_names)} 类）", fontsize=14)
    plt.tight_layout()
    fig.savefig(save_dir / "confusion_matrix_full.png", dpi=120)
    plt.close(fig)
    print("全量混淆矩阵保存至", save_dir / "confusion_matrix_full.png")

    # ── Top-K 易混淆子集 ──
    np.fill_diagonal(cm, 0)
    mistake_counts = cm.sum(axis=1) + cm.sum(axis=0)
    top_idx = np.argsort(mistake_counts)[-top_k:]
    cm_sub  = confusion_matrix(labels, preds)[np.ix_(top_idx, top_idx)]
    sub_names = [class_names[i].replace("_", " ") for i in top_idx]

    fig2, ax2 = plt.subplots(figsize=(16, 14))
    sns.heatmap(cm_sub, ax=ax2, cmap="OrRd", annot=True, fmt="d",
                xticklabels=sub_names, yticklabels=sub_names)
    ax2.set_xlabel("预测类别", fontsize=10)
    ax2.set_ylabel("真实类别", fontsize=10)
    ax2.set_title(f"Top-{top_k} 易混淆犬种混淆矩阵", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0,  fontsize=7)
    plt.tight_layout()
    fig2.savefig(save_dir / f"confusion_matrix_top{top_k}.png", dpi=150)
    plt.close(fig2)
    print(f"Top-{top_k} 混淆矩阵保存至", save_dir / f"confusion_matrix_top{top_k}.png")

    return cm


# ─────────────────────────────────────────────
# Grad-CAM
# ─────────────────────────────────────────────

class GradCAM:
    """
    针对 EfficientNetB3WithCBAM / EfficientNetB3Baseline 通用的 Grad-CAM 实现
    target_layer: 默认取最后一个卷积 head（conv_head）
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._hooks       = []
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(module, inp, out):
            self.activations = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self._hooks.append(
            self.target_layer.register_forward_hook(fwd_hook))
        self._hooks.append(
            self.target_layer.register_full_backward_hook(bwd_hook))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

    def __call__(self, img_tensor: torch.Tensor,
                 class_idx: int = None) -> np.ndarray:
        """
        img_tensor: (1, 3, H, W) — 已归一化
        返回 (H, W) 归一化热力图
        """
        self.model.eval()
        img_tensor.requires_grad_(True)

        logits = self.model(img_tensor)           # (1, C)
        if class_idx is None:
            class_idx = logits.argmax(1).item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        # 梯度加权激活
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1,C,1,1)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = F.relu(cam)
        cam     = F.interpolate(cam,
                                size=img_tensor.shape[-2:],
                                mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """反归一化，返回 HWC uint8"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img  = (tensor.cpu() * std + mean).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def visualize_gradcam(model_baseline, model_improved,
                       loader, device,
                       class_names, save_dir: Path,
                       num_samples: int = 6):
    """
    对比基线 vs 改进模型的 Grad-CAM 热力图
    两个模型均需是各自正确的类型（Baseline / WithCBAM）
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # 取足够数量的样本
    imgs_all, labels_all = [], []
    for imgs, labels in loader:
        imgs_all.append(imgs)
        labels_all.append(labels)
        if sum(len(x) for x in imgs_all) >= num_samples * 4:
            break
    imgs_all   = torch.cat(imgs_all)[:num_samples * 4]
    labels_all = torch.cat(labels_all)[:num_samples * 4]

    # 找改进模型正确分类的样本
    model_improved.eval()
    with torch.no_grad():
        logits = model_improved(imgs_all.to(device))
    correct = (logits.argmax(1).cpu() == labels_all).nonzero().squeeze()
    correct_idx = correct[:num_samples].tolist()
    if not isinstance(correct_idx, list):
        correct_idx = [correct_idx]

    # ── Grad-CAM hooks（两个模型均指向各自的 conv_head）──
    cam_base = GradCAM(model_baseline, model_baseline.backbone.conv_head)
    cam_imp  = GradCAM(model_improved,  model_improved.backbone.conv_head)

    fig, axes = plt.subplots(len(correct_idx), 3,
                              figsize=(12, 4 * len(correct_idx)))
    if len(correct_idx) == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(correct_idx):
        img_t  = imgs_all[idx:idx+1].to(device)
        label  = labels_all[idx].item()
        img_np = denormalize(imgs_all[idx])

        heatmap_base = cam_base(img_t.clone(), label)
        heatmap_imp  = cam_imp (img_t.clone(), label)

        axes[row, 0].imshow(img_np)
        axes[row, 0].set_title(f"原图\n真实: {class_names[label]}", fontsize=8)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(img_np)
        axes[row, 1].imshow(heatmap_base, alpha=0.5, cmap="jet")
        axes[row, 1].set_title("基线模型 Grad-CAM", fontsize=8)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(img_np)
        axes[row, 2].imshow(heatmap_imp, alpha=0.5, cmap="jet")
        axes[row, 2].set_title("改进模型 Grad-CAM (+CBAM)", fontsize=8)
        axes[row, 2].axis("off")

    plt.suptitle("Grad-CAM 对比：基线 vs EfficientNet-B3+CBAM",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    out_path = save_dir / "gradcam_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    cam_base.remove_hooks()
    cam_imp.remove_hooks()
    print("Grad-CAM 对比图保存至", out_path)


# ─────────────────────────────────────────────
# 每类准确率柱状图
# ─────────────────────────────────────────────

def plot_per_class_accuracy(preds, labels, class_names, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    n = len(class_names)
    per_class_acc = np.zeros(n)
    for c in range(n):
        mask = labels == c
        if mask.sum() > 0:
            per_class_acc[c] = (preds[mask] == c).mean()

    sort_idx = np.argsort(per_class_acc)
    fig, ax = plt.subplots(figsize=(24, 6))
    colors = ["#d9534f" if v < 0.5 else "#5cb85c" for v in per_class_acc[sort_idx]]
    ax.bar(range(n), per_class_acc[sort_idx], color=colors, width=0.8)
    ax.axhline(per_class_acc.mean(), color="orange", linestyle="--",
               label=f"均值={per_class_acc.mean():.3f}")
    ax.set_xticks(range(n))
    ax.set_xticklabels(
        [class_names[i].replace("_", "\n") for i in sort_idx],
        fontsize=4, rotation=90
    )
    ax.set_ylabel("准确率")
    ax.set_title("各类别准确率（升序排列）")
    ax.legend(fontsize=9)
    patch_low = mpatches.Patch(color="#d9534f", label="< 0.5")
    patch_hi  = mpatches.Patch(color="#5cb85c", label="≥ 0.5")
    ax.legend(handles=[patch_low, patch_hi,
                        plt.Line2D([0], [0], color="orange",
                                   linestyle="--", label=f"均值={per_class_acc.mean():.3f}")],
              fontsize=8)
    plt.tight_layout()
    fig.savefig(save_dir / "per_class_accuracy.png", dpi=150)
    plt.close(fig)
    print("每类准确率图保存至", save_dir / "per_class_accuracy.png")


# ─────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="评估 EfficientNet-B3（基线 / +CBAM）")
    p.add_argument("--data",          type=str, default="./data/stanford_dogs")
    p.add_argument("--ckpt",          type=str, required=True,
                   help="主评估模型权重路径（基线或改进模型均可）")
    p.add_argument("--ckpt_baseline", type=str, default=None,
                   help="基线模型权重路径（用于与改进模型做 Grad-CAM 对比）")
    p.add_argument("--baseline_only", action="store_true",
                   help="只评估基线模型；此时 --ckpt 指向基线权重，结果存入 save_dir/baseline/")
    p.add_argument("--num_classes",   type=int, default=120)
    p.add_argument("--batch_size",    type=int, default=32)
    p.add_argument("--num_workers",   type=int, default=4)
    p.add_argument("--no_bbox",       action="store_true")
    p.add_argument("--save_dir",      type=str, default="./eval_results")
    p.add_argument("--gradcam_n",     type=int, default=6,
                   help="Grad-CAM 展示样本数")
    return p.parse_args()


def main():
    args     = parse_args()
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)

    # ── 数据 ──────────────────────────────────────────────
    loaders = build_dataloaders(
        args.data, batch_size=args.batch_size,
        num_workers=args.num_workers, use_bbox=not args.no_bbox
    )
    class_names = loaders["test"].dataset.classes

    # ── 主评估模型加载（自动检测模型类型）────────────────
    # 先读取 checkpoint，通过 key 自动判断是基线还是改进模型
    # 无论是否传 --baseline_only，都不会因模型类型不匹配而报错
    ckpt        = torch.load(args.ckpt, map_location=device)
    state_keys  = set(ckpt["model"].keys())
    is_baseline = not any(k.startswith("cbam_modules") for k in state_keys)

    if args.baseline_only or is_baseline:
        if is_baseline and not args.baseline_only:
            print("[自动检测] checkpoint 不含 CBAM 权重，自动切换为基线模式")
        model = EfficientNetB3Baseline(
            num_classes=args.num_classes, pretrained=False).to(device)
        model.load_state_dict(ckpt["model"])
        print(f"[基线模式] 加载基线模型: {args.ckpt}  "
              f"(val_acc={ckpt.get('val_acc', 'N/A')})")
        eval_save_dir = save_dir / "baseline"
    else:
        model = EfficientNetB3WithCBAM(
            num_classes=args.num_classes, pretrained=False).to(device)
        model.load_state_dict(ckpt["model"])
        print(f"[改进模式] 加载改进模型: {args.ckpt}  "
              f"(val_acc={ckpt.get('val_acc', 'N/A')})")
        eval_save_dir = save_dir

    # ── 推理 ──────────────────────────────────────────────
    preds, labels, probs = predict(model, loaders["test"], device)

    # ── 指标 ──────────────────────────────────────────────
    metrics = compute_metrics(preds, labels, class_names, eval_save_dir)

    # ── 混淆矩阵 ──────────────────────────────────────────
    plot_confusion_matrix(preds, labels, class_names, eval_save_dir)

    # ── 每类准确率 ────────────────────────────────────────
    plot_per_class_accuracy(preds, labels, class_names, eval_save_dir)

    # ── Grad-CAM 对比（仅在评估改进模型且提供基线权重时触发）──
    if args.ckpt_baseline and not args.baseline_only:
        # ✅ 修复原始 bug：此处正确使用 EfficientNetB3Baseline，而非 EfficientNetB3WithCBAM
        baseline = EfficientNetB3Baseline(
            num_classes=args.num_classes, pretrained=False).to(device)
        b_ckpt   = torch.load(args.ckpt_baseline, map_location=device)
        baseline.load_state_dict(b_ckpt["model"])
        print(f"加载基线模型（Grad-CAM 对比用）: {args.ckpt_baseline}  "
              f"(val_acc={b_ckpt.get('val_acc', 'N/A')})")
        visualize_gradcam(
            baseline, model,
            loaders["test"], device,
            class_names, eval_save_dir,
            num_samples=args.gradcam_n,
        )

    # ── 保存数值结果 ──────────────────────────────────────
    result_txt = (
        f"Accuracy : {metrics['accuracy']:.4f}\n"
        f"Precision: {metrics['precision']:.4f}\n"
        f"Recall   : {metrics['recall']:.4f}\n"
        f"F1       : {metrics['f1']:.4f}\n"
    )
    (eval_save_dir / "metrics.txt").write_text(result_txt, encoding="utf-8")
    print("\n所有评估结果保存至:", eval_save_dir)


if __name__ == "__main__":
    main()