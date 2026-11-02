"""
训练脚本: 5层LSTM + CTC损失函数
"""

import os
import sys
import time
import json
import math
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from model import DeepSpeech2
from dataset import (
    SpeechDataset, Vocabulary, create_dataloader,
    normalize_features,
)


# ============================================================
# 配置
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="5层LSTM + CTC 语音识别训练")

    # 数据
    parser.add_argument("--train_dir", type=str, required=True, help="训练数据目录")
    parser.add_argument("--val_dir", type=str, default=None, help="验证数据目录")
    parser.add_argument("--vocab_path", type=str, default="vocab.json", help="词表保存路径")

    # 模型
    parser.add_argument("--input_dim", type=int, default=80)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--bidirectional", action="store_true", default=True)

    # 训练
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_decay", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--min_lr_ratio", type=float, default=0.01,
                        help="余弦退火最低 lr 比例（相对 base_lr），默认退火到 1%%")

    # 设备
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader 进程数；Windows 下 spawn+IPC 开销大于收益，建议 0")
    parser.add_argument("--no_cache", action="store_true", help="禁用 mel 特征缓存")
    parser.add_argument("--rebuild_cache", action="store_true", help="强制重建特征缓存")
    parser.add_argument("--use_amp", action="store_true", default=True, help="使用混合精度训练")

    # 保存
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=5)

    return parser.parse_args()


# ============================================================
# CTC损失包装器
# ============================================================
class CTCLossWrapper(nn.Module):
    """
    CTC损失函数包装器
    loss = CTCLoss(log_probs, targets, input_lengths, target_lengths)
    """

    def __init__(self, blank: int = 0, reduction: str = "mean", zero_infinity: bool = True):
        super().__init__()
        self.ctc_loss = nn.CTCLoss(
            blank=blank,
            reduction=reduction,
            zero_infinity=zero_infinity,
        )

    def forward(
        self,
        log_probs: torch.Tensor,   # [B, T, C]
        targets: torch.Tensor,      # [B, S]
        input_lengths: torch.Tensor,  # [B]
        target_lengths: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        """
        CTC损失计算

        CTC要求:
        - log_probs: [T, B, C] - 时间优先
        - targets: [sum(target_lengths)] - 一维拼接
        - input_lengths: [B] - 每个样本输入长度
        - target_lengths: [B] - 每个样本标签长度
        """
        # 转换为CTC需要的格式: [T, B, C]
        log_probs_ctc = log_probs.permute(1, 0, 2)  # [B, T, C] -> [T, B, C]

        # 确保长度在合法范围
        input_lengths = input_lengths.to(torch.long)
        target_lengths = target_lengths.to(torch.long)

        # 确保log_probs长度 >= target_lengths + 1 (CTC约束)
        T = log_probs_ctc.size(0)
        targets = targets[:, :target_lengths.max()]  # 截断填充

        loss = self.ctc_loss(log_probs_ctc, targets, input_lengths, target_lengths)
        return loss


# ============================================================
# 学习率调度器 (warmup + 余弦退火)
# ============================================================
class WarmupScheduler:
    """线性 warmup 后接余弦退火。

    原版 WarmupScheduler 在 warmup 结束后恒定返回 base_lr，与 StepLR 同时使用时
    会每个 batch 覆盖 StepLR 的衰减，导致 lr 全程不变、loss 横盘。
    本版将 warmup 与余弦退火合并为单一调度器，warmup 后 lr 从 base_lr 平滑
    退火到 base_lr * min_lr_ratio。
    """

    def __init__(self, optimizer, warmup_steps: int, base_lr: float,
                 total_steps: int, min_lr_ratio: float = 0.01):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.min_lr = base_lr * min_lr_ratio
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            progress = (self.current_step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


# ============================================================
# 训练函数
# ============================================================
def train_epoch(
    model: nn.Module,
    dataloader,
    ctc_loss_fn: CTCLossWrapper,
    optimizer: optim.Optimizer,
    scheduler: WarmupScheduler,
    scaler: GradScaler,
    device: torch.device,
    grad_clip: float,
    use_amp: bool,
    log_interval: int,
):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_batches = len(dataloader)

    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)
        input_lengths = batch["input_lengths"].to(device)
        label_lengths = batch["label_lengths"].to(device)

        optimizer.zero_grad()

        if use_amp:
            with autocast():
                log_probs, output_lengths = model(features, input_lengths)
                loss = ctc_loss_fn(log_probs, labels, output_lengths, label_lengths)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            log_probs, output_lengths = model(features, input_lengths)
            loss = ctc_loss_fn(log_probs, labels, output_lengths, label_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        lr = scheduler.step()
        total_loss += loss.item()

        if (batch_idx + 1) % log_interval == 0:
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{total_loss / (batch_idx + 1):.4f}",
                "lr": f"{lr:.2e}",
            })

    return total_loss / total_batches


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader,
    ctc_loss_fn: CTCLossWrapper,
    device: torch.device,
) -> float:
    """验证"""
    model.eval()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Validating"):
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)
        input_lengths = batch["input_lengths"].to(device)
        label_lengths = batch["label_lengths"].to(device)

        log_probs, output_lengths = model(features, input_lengths)
        loss = ctc_loss_fn(log_probs, labels, output_lengths, label_lengths)
        total_loss += loss.item()

    return total_loss / len(dataloader)


# ============================================================
# 主训练流程
# ============================================================
def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(args.device)
    print(f"使用设备: {device}")

    # ========== 1. 词表 ==========
    # 优先加载现成词表（本项目使用 prepare_timit.py 从 TIMIT 语料统计的 29 类英文词表）
    if os.path.exists(args.vocab_path):
        print(f"\n加载词表: {args.vocab_path}")
        vocab = Vocabulary.load(args.vocab_path)
        print(f"词表大小: {len(vocab)}")
    else:
        # 无词表文件时从训练文本构建（数据字符必须 add_text 收录，原代码缺失此步）
        print("\n从训练数据构建词表...")
        vocab = Vocabulary()
        for _, text in SpeechDataset(args.train_dir, vocab).samples:
            vocab.add_text(text)
        vocab.save(args.vocab_path)
        print(f"词表大小: {len(vocab)}")

    num_classes = len(vocab)

    # ========== 2. 加载数据 ==========
    print("\n加载数据...")
    train_dataset = SpeechDataset(args.train_dir, vocab,
                                  use_cache=not args.no_cache)
    val_loader = None
    if args.val_dir:
        val_dataset = SpeechDataset(args.val_dir, vocab,
                                    use_cache=not args.no_cache)

    # 特征预计算：一次性落盘，训练阶段只读缓存
    train_dataset.precompute_features(force=args.rebuild_cache)
    if args.val_dir:
        val_dataset.precompute_features(force=args.rebuild_cache)

    train_loader = create_dataloader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    if args.val_dir:
        val_loader = create_dataloader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)

    # ========== 2.5 全局 CMVN 统计量（训练/验证/推理三方共用，保证输入分布一致） ==========
    print("\n计算全局 CMVN 统计量（遍历训练集）...")
    feature_stats = train_dataset.compute_global_stats()
    if val_loader is not None:
        val_dataset.stats = feature_stats

    # ========== 3. 构建模型 ==========
    print("\n构建模型...")
    model = DeepSpeech2(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=num_classes,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # ========== 4. 定义损失函数和优化器 ==========
    ctc_loss_fn = CTCLossWrapper(blank=0, reduction="mean", zero_infinity=True)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    # 统一调度器：warmup + 余弦退火（替代原版 WarmupScheduler + StepLR 的冲突组合）
    total_steps = args.epochs * len(train_loader)
    scheduler = WarmupScheduler(
        optimizer, args.warmup_steps, args.lr,
        total_steps=total_steps, min_lr_ratio=args.min_lr_ratio,
    )

    scaler = GradScaler() if args.use_amp else None

    # ========== 5. 训练循环 ==========
    print("\n开始训练...")
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")

        # 训练（warmup + 余弦退火由统一调度器管理，无需按 epoch 重置）
        train_loss = train_epoch(
            model, train_loader, ctc_loss_fn,
            optimizer, scheduler, scaler,
            device, args.grad_clip, args.use_amp, args.log_interval,
        )
        history["train_loss"].append(train_loss)

        # 验证
        if val_loader:
            val_loss = validate(model, val_loader, ctc_loss_fn, device)
            history["val_loss"].append(val_loss)
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(args.save_dir, "best_model.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "vocab": vocab,
                    "feature_stats": feature_stats,
                    "args": vars(args),
                }, save_path)
                print(f"保存最佳模型: {save_path}")
        else:
            print(f"Train Loss: {train_loss:.4f}")

        # 定期保存
        if epoch % args.save_interval == 0:
            save_path = os.path.join(args.save_dir, f"checkpoint_epoch{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "vocab": vocab,
                "feature_stats": feature_stats,
                "args": vars(args),
            }, save_path)
            print(f"保存checkpoint: {save_path}")

    # ========== 6. 保存最终模型 ==========
    final_path = os.path.join(args.save_dir, "final_model.pt")
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "vocab": vocab,
        "feature_stats": feature_stats,
        "args": vars(args),
        "history": history,
    }, final_path)
    print(f"\n训练完成! 模型保存在: {final_path}")

    # 保存训练历史
    with open(os.path.join(args.save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
