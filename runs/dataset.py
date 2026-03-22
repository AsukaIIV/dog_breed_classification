"""
dataset.py
Stanford Dogs Dataset 数据加载与预处理
- 解析 XML 标注获取边界框，裁剪主体区域并扩展边界
- 数据增强（训练集）/ 标准化（测试集）
- 自动划分 train / val / test
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import scipy.io


# ─────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_SIZE      = 300   # EfficientNet-B3 推荐输入


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def parse_annotation(xml_path: str) -> Optional[Tuple[int, int, int, int]]:
    """
    解析 Stanford Dogs XML 标注文件，返回边界框 (xmin, ymin, xmax, ymax)
    若解析失败返回 None
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        obj = root.find("object")
        if obj is None:
            return None
        bndbox = obj.find("bndbox")
        xmin = int(float(bndbox.find("xmin").text))
        ymin = int(float(bndbox.find("ymin").text))
        xmax = int(float(bndbox.find("xmax").text))
        ymax = int(float(bndbox.find("ymax").text))
        return xmin, ymin, xmax, ymax
    except Exception:
        return None


def crop_with_margin(img: Image.Image,
                     bbox: Tuple[int, int, int, int],
                     margin: float = 0.15) -> Image.Image:
    """
    按边界框裁剪，向四周扩展 margin 比例的边距
    """
    w, h = img.size
    xmin, ymin, xmax, ymax = bbox
    bw = xmax - xmin
    bh = ymax - ymin
    dx = int(bw * margin)
    dy = int(bh * margin)
    xmin = max(0, xmin - dx)
    ymin = max(0, ymin - dy)
    xmax = min(w, xmax + dx)
    ymax = min(h, ymax + dy)
    return img.crop((xmin, ymin, xmax, ymax))


# ─────────────────────────────────────────────
# Transform 工厂
# ─────────────────────────────────────────────

def build_transforms(is_train: bool) -> T.Compose:
    if is_train:
        return T.Compose([
            T.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.3, contrast=0.3,
                          saturation=0.3, hue=0.05),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            T.RandomErasing(p=0.3, scale=(0.02, 0.15)),   # 模拟遮挡（必须在ToTensor之后）
        ])
    else:
        return T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class StanfordDogsDataset(Dataset):
    """
    Stanford Dogs Dataset
    目录结构（官方原始包解压后）：
        root/
          Images/
            n02085620-Chihuahua/
              n02085620_10074.jpg
              ...
          Annotation/
            n02085620-Chihuahua/
              n02085620_10074      # XML 文件（无后缀）
          lists/
            train_list.mat
            test_list.mat
    """

    def __init__(self, root: str, split: str = "train",
                 use_bbox: bool = True,
                 bbox_margin: float = 0.15,
                 transform=None):
        """
        Args:
            root       : 数据集根目录
            split      : "train" | "val" | "test"
            use_bbox   : 是否用边界框裁剪主体
            bbox_margin: 边界框扩展比例
            transform  : torchvision transform
        """
        super().__init__()
        self.root        = Path(root)
        self.use_bbox    = use_bbox
        self.bbox_margin = bbox_margin
        self.transform   = transform

        # ── 解析类别 ──
        image_dir = self.root / "Images"
        breed_dirs = sorted([d for d in image_dir.iterdir() if d.is_dir()])
        self.classes    = [d.name for d in breed_dirs]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.num_classes  = len(self.classes)

        # ── 解析 .mat 文件得到官方分割 ──
        train_mat = scipy.io.loadmat(str(self.root / "lists" / "train_list.mat"))
        test_mat  = scipy.io.loadmat(str(self.root / "lists" / "test_list.mat"))

        train_files = [str(f[0][0]) for f in train_mat["file_list"]]
        test_files  = [str(f[0][0]) for f in test_mat["file_list"]]

        # ── 从训练集划分 10% 作为验证集 ──
        np.random.seed(42)
        idx      = np.random.permutation(len(train_files))
        n_val    = int(len(train_files) * 0.10)
        val_idx  = set(idx[:n_val].tolist())
        train_idx= set(idx[n_val:].tolist())

        if split == "train":
            file_list = [train_files[i] for i in sorted(train_idx)]
        elif split == "val":
            file_list = [train_files[i] for i in sorted(val_idx)]
        else:  # test
            file_list = test_files

        # ── 构建样本列表 (img_path, ann_path, label) ──
        self.samples = []
        for rel_path in file_list:
            img_path = self.root / "Images" / rel_path
            # Annotation 文件路径（无扩展名）
            ann_rel  = Path(rel_path).with_suffix("")
            ann_path = self.root / "Annotation" / ann_rel
            # 类别名 = 父目录名
            breed    = Path(rel_path).parent.name
            label    = self.class_to_idx[breed]
            if img_path.exists():
                self.samples.append((img_path, ann_path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, ann_path, label = self.samples[idx]

        # ── 读取图像 ──
        img = Image.open(img_path).convert("RGB")

        # ── 边界框裁剪 ──
        if self.use_bbox and ann_path.exists():
            bbox = parse_annotation(str(ann_path))
            if bbox is not None:
                img = crop_with_margin(img, bbox, self.bbox_margin)

        # ── Transform ──
        if self.transform:
            img = self.transform(img)

        return img, label


# ─────────────────────────────────────────────
# DataLoader 工厂
# ─────────────────────────────────────────────

def build_dataloaders(root: str,
                      batch_size: int = 32,
                      num_workers: int = 4,
                      use_bbox: bool = True) -> dict:
    """
    返回 {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    """
    splits = {
        "train": build_transforms(is_train=True),
        "val":   build_transforms(is_train=False),
        "test":  build_transforms(is_train=False),
    }
    loaders = {}
    for split, tfm in splits.items():
        ds = StanfordDogsDataset(root, split=split,
                                 use_bbox=use_bbox, transform=tfm)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
        )
        print(f"[{split:5s}] {len(ds):>6d} 张图像  {len(loaders[split]):>4d} 个 batch")

    return loaders


# ─────────────────────────────────────────────
# 快速验证
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "./data/stanford_dogs"
    loaders = build_dataloaders(root, batch_size=8, num_workers=0)
    imgs, labels = next(iter(loaders["train"]))
    print(f"Batch shape: {imgs.shape}  Labels: {labels[:4]}")
