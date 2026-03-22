"""
train.py
EfficientNet-B3 + CBAM 犬类品种分类 — 完整训练脚本
支持 GPU（RTX 5060）/ CPU 自动切换
"""

import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model   import EfficientNetB3WithCBAM
from dataset import build_dataloaders


# ─────────────────────────────────────────────
# 诊断函数（首次运行必看）
# ─────────────────────────────────────────────

def diagnose(model, loader, device):
    print("\n" + "="*55)
    print("  训练前诊断")
    print("="*55)

    # 1. 设备
    print(f"  运行设备:  {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        vram  = props.total_memory / 1024**3
        print(f"  GPU 型号:  {props.name}")
        print(f"  显存大小:  {vram:.1f} GB")

    # 2. 参数量
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数:    {total/1e6:.2f}M")
    print(f"  可训练:    {trainable/1e6:.2f}M  "
          f"({'正常 ✓' if trainable/1e6 < 5 else '⚠ 骨干可能未冻结'})")

    # 3. 数据标签
    imgs, labels = next(iter(loader))
    print(f"  标签范围:  {labels.min().item()} ~ {labels.max().item()}  "
          f"({'正常 ✓' if labels.max().item() < 120 else '⚠ 标签越界'})")
    print(f"  Batch尺寸: {list(imgs.shape)}")

    # 4. 前向传播
    model.eval()
    with torch.no_grad():
        out = model(imgs[:2].to(device))
    probs = out.softmax(dim=1)
    print(f"  输出形状:  {list(out.shape)}  "
          f"({'正常 ✓' if out.shape[1] == 120 else '⚠ 输出维度错误'})")
    print(f"  初始最大概率均值: {probs.max(1).values.mean():.4f}  "
          f"(随机基线约 {1/120:.4f})")
    print("="*55 + "\n")


# ─────────────────────────────────────────────
# 单 epoch
# ─────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, is_train):
    model.train() if is_train else model.eval()
    total_loss = total_correct = total_n = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for imgs, labels in tqdm(loader, leave=False,
                                  desc="train" if is_train else "val  "):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            total_loss    += loss.item() * imgs.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_n       += imgs.size(0)

    return total_loss / total_n, total_correct / total_n


# ─────────────────────────────────────────────
# 阶段训练
# ─────────────────────────────────────────────

def train_stage(model, loaders, criterion, optimizer, scheduler,
                writer, device, epochs, stage_name,
                start_epoch=0, patience=10):
    best_val_acc = 0.0
    no_improve   = 0
    ckpt_dir     = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    # ── CSV 日志（追加模式，断点续训也不会丢数据）──
    log_path = Path("training_log.csv")
    if not log_path.exists():
        log_path.write_text("epoch,stage,train_acc,train_loss,val_acc,val_loss,lr\n",
                            encoding="utf-8")

    for ep in range(epochs):
        g_ep = start_epoch + ep
        t0   = time.time()

        tr_loss, tr_acc = run_epoch(model, loaders["train"], criterion,
                                     optimizer, device, True)
        vl_loss, vl_acc = run_epoch(model, loaders["val"],   criterion,
                                     optimizer, device, False)
        if scheduler:
            scheduler.step()

        lr  = optimizer.param_groups[0]["lr"]
        ela = time.time() - t0
        print(f"[{stage_name}] ep{g_ep+1:03d}  "
              f"train={tr_acc:.4f}({tr_loss:.4f})  "
              f"val={vl_acc:.4f}({vl_loss:.4f})  "
              f"lr={lr:.1e}  {ela:.0f}s")

        # ── 写入 CSV ──
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{g_ep+1},{stage_name},"
                    f"{tr_acc:.6f},{tr_loss:.6f},"
                    f"{vl_acc:.6f},{vl_loss:.6f},"
                    f"{lr:.2e}\n")

        writer.add_scalars("Loss",     {"train": tr_loss, "val": vl_loss}, g_ep)
        writer.add_scalars("Accuracy", {"train": tr_acc,  "val": vl_acc},  g_ep)
        writer.add_scalar ("LR", lr, g_ep)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            no_improve   = 0
            torch.save({"epoch": g_ep, "model": model.state_dict(),
                        "val_acc": best_val_acc},
                       ckpt_dir / f"best_{stage_name}.pth")
            print(f"  ✓ 最优模型已保存  val_acc={best_val_acc:.4f}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  早停触发（{patience} 轮无改善）")
                break

    return start_epoch + ep + 1


# ─────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="EfficientNet-B3+CBAM 犬类分类训练")
    p.add_argument("--data",        type=str,   default="./data/stanford_dogs",
                   help="Stanford Dogs 数据集根目录")
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--img_size",    type=int,   default=300)
    p.add_argument("--dropout",     type=float, default=0.3)
    p.add_argument("--label_smooth",type=float, default=0.1)
    p.add_argument("--epochs_s1",   type=int,   default=15,
                   help="Stage1：冻结骨干，只训练分类头+CBAM")
    p.add_argument("--epochs_s2",   type=int,   default=20,
                   help="Stage2：解冻后3个Stage")
    p.add_argument("--epochs_s3",   type=int,   default=25,
                   help="Stage3：全网络端到端微调")
    p.add_argument("--patience",    type=int,   default=10)
    p.add_argument("--no_bbox",     action="store_true",
                   help="不使用边界框裁剪")
    p.add_argument("--resume",      type=str,   default=None,
                   help="从 checkpoint 恢复，例如 checkpoints/best_stage2.pth")
    return p.parse_args()


def main():
    args = parse_args()

    # ── 设备自动选择 ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # ── 输入尺寸注入 ──
    import dataset as ds_module
    ds_module.IMG_SIZE = args.img_size

    # ── 数据 ──
    loaders = build_dataloaders(
        args.data,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        use_bbox    = not args.no_bbox,
    )

    # ── 模型 ──
    model = EfficientNetB3WithCBAM(
        num_classes = 120,
        dropout     = args.dropout,
        pretrained  = True,
    ).to(device)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"已从 {args.resume} 恢复模型  (val_acc={ckpt.get('val_acc','?')})")

    # ── 损失函数 ──
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)

    # ── TensorBoard ──
    writer = SummaryWriter("runs/efficientnet_cbam")

    # ══════════════════════════════════════════
    # Stage 1：冻结骨干，只训练分类头 + CBAM
    # ══════════════════════════════════════════
    print("\n" + "="*60)
    print("Stage 1：冻结骨干，训练分类头 + CBAM")
    print("="*60)
    model.freeze_backbone()

    # 诊断（只在 Stage1 开始前运行一次）
    diagnose(model, loaders["train"], device)

    opt1 = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=5e-4, weight_decay=1e-4,
    )
    sch1 = CosineAnnealingLR(opt1, T_max=args.epochs_s1, eta_min=1e-6)

    ep1 = train_stage(
        model, loaders, criterion, opt1, sch1,
        writer, device, args.epochs_s1,
        stage_name="stage1", patience=args.patience,
    )

    ckpt = torch.load("checkpoints/best_stage1.pth", map_location=device)
    model.load_state_dict(ckpt["model"])

    # ══════════════════════════════════════════
    # Stage 2：解冻后 3 个 Stage，差分学习率
    # ══════════════════════════════════════════
    print("\n" + "="*60)
    print("Stage 2：解冻后 3 个 Stage，差分学习率微调")
    print("="*60)
    model.unfreeze_last_stages(num_stages=3)

    opt2 = optim.AdamW([
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and "classifier" in n], "lr": 1e-4},
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and "cbam"       in n], "lr": 1e-4},
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad
                    and "classifier" not in n
                    and "cbam"       not in n],               "lr": 1e-5},
    ], weight_decay=1e-4)
    sch2 = CosineAnnealingLR(opt2, T_max=args.epochs_s2, eta_min=1e-7)

    ep2 = train_stage(
        model, loaders, criterion, opt2, sch2,
        writer, device, args.epochs_s2,
        stage_name="stage2", start_epoch=ep1,
        patience=args.patience,
    )

    ckpt = torch.load("checkpoints/best_stage2.pth", map_location=device)
    model.load_state_dict(ckpt["model"])

    # ══════════════════════════════════════════
    # Stage 3：全网络端到端微调
    # ══════════════════════════════════════════
    print("\n" + "="*60)
    print("Stage 3：全网络端到端微调")
    print("="*60)
    model.unfreeze_all()

    opt3 = optim.AdamW([
        {"params": [p for n, p in model.named_parameters()
                    if "classifier" in n or "cbam" in n], "lr": 5e-5},
        {"params": [p for n, p in model.named_parameters()
                    if "classifier" not in n
                    and "cbam"       not in n],           "lr": 5e-6},
    ], weight_decay=1e-4)
    sch3 = CosineAnnealingLR(opt3, T_max=args.epochs_s3, eta_min=1e-8)

    train_stage(
        model, loaders, criterion, opt3, sch3,
        writer, device, args.epochs_s3,
        stage_name="stage3", start_epoch=ep2,
        patience=args.patience,
    )

    writer.close()
    print("\n训练完成！")
    print("最优模型: checkpoints/best_stage3.pth")
    print("查看曲线: tensorboard --logdir runs/")


if __name__ == "__main__":
    main()