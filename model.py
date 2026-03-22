"""
model.py
EfficientNet-B3 + CBAM 注意力模块
在 Stage3、4、5 末尾插入 CBAM，其余保留原生 SE 模块
使用 hook 方式注入 CBAM，兼容所有 timm 版本，不依赖内部属性名
"""

import torch
import torch.nn as nn
import timm


# ─────────────────────────────────────────────
# CBAM：通道注意力 + 空间注意力
# ─────────────────────────────────────────────

class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        mid = max(in_channels // reduction, 8)
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=[2, 3])
        mx  = x.amax(dim=[2, 3])
        w   = torch.sigmoid(self.shared_mlp(avg) + self.shared_mlp(mx))
        return x * w[:, :, None, None]


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size,
                              padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.amax(dim=1, keepdim=True)
        w   = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * w


class CBAM(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16,
                 kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(self.ca(x))


# ─────────────────────────────────────────────
# EfficientNet-B3 + CBAM（hook 注入方式）
# ─────────────────────────────────────────────

class EfficientNetB3WithCBAM(nn.Module):
    """
    EfficientNet-B3（预训练）
    - 保留原生 SE 通道注意力
    - 在 blocks[2][3][4] 末尾通过 forward hook 注入 CBAM
    - 分类头：Dropout + Linear(1536, num_classes)
    """

    CBAM_STAGE_INDICES = [2, 3, 4]

    def __init__(self, num_classes: int = 120, dropout: float = 0.3,
                 pretrained: bool = True):
        super().__init__()

        # ── 加载骨干
        self.backbone = timm.create_model(
            "efficientnet_b3",
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )

        # ── 自动探测各 stage 输出通道数
        stage_channels = self._probe_channels()
        print("各 Stage 输出通道数:", stage_channels)

        # ── 创建 CBAM 模块
        self.cbam_modules = nn.ModuleDict()
        for idx in self.CBAM_STAGE_INDICES:
            ch = stage_channels[idx]
            self.cbam_modules[f"cbam_s{idx}"] = CBAM(ch)
            print(f"  注入 CBAM @ blocks[{idx}]，通道数 = {ch}")

        # ── 分类头
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier  = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1536, num_classes),
        )

        # ── 注册 forward hooks
        self._hook_handles = []
        self._register_cbam_hooks()

    def _probe_channels(self) -> dict:
        """用 dummy forward 自动探测每个 stage 的输出通道数"""
        channels = {}
        hooks = []

        def make_hook(idx):
            def hook(module, inp, out):
                channels[idx] = out.shape[1]
            return hook

        for i, stage in enumerate(self.backbone.blocks):
            h = stage[-1].register_forward_hook(make_hook(i))
            hooks.append(h)

        with torch.no_grad():
            self.backbone(torch.zeros(1, 3, 300, 300))

        for h in hooks:
            h.remove()
        return channels

    def _register_cbam_hooks(self):
        """在每个目标 stage 的最后一个 block 输出后插入 CBAM"""
        for idx in self.CBAM_STAGE_INDICES:
            last_blk = self.backbone.blocks[idx][-1]
            cbam     = self.cbam_modules[f"cbam_s{idx}"]

            def make_hook(c):
                def hook(module, inp, out):
                    return c(out)
                return hook

            h = last_blk.register_forward_hook(make_hook(cbam))
            self._hook_handles.append(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)               # CBAM 通过 hook 自动生效
        x = self.global_pool(x).flatten(1)
        return self.classifier(x)

    # ── 冻结 / 解冻 API ──────────────────────

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_last_stages(self, num_stages: int = 3):
        for p in self.backbone.parameters():
            p.requires_grad = False
        # 解冻 head conv（兼容不同命名）
        for name, p in self.backbone.named_parameters():
            if any(k in name for k in ["conv_head", "bn2", "norm_head"]):
                p.requires_grad = True
        # 解冻最后 num_stages 个 stage
        total = len(self.backbone.blocks)
        for stage in self.backbone.blocks[total - num_stages:]:
            for p in stage.parameters():
                p.requires_grad = True

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()


# ─────────────────────────────────────────────
# 快速测试
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("测试 EfficientNetB3WithCBAM...")
    model = EfficientNetB3WithCBAM(num_classes=120, pretrained=False)
    out   = model(torch.randn(2, 3, 300, 300))
    print(f"输出形状: {out.shape}")    # 期望 (2, 120)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total/1e6:.2f}M  可训练: {trainable/1e6:.2f}M")
    print("测试通过 ✓")
