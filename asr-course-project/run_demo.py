"""
端到端演示: TIMIT 50 条真实数据 -> 训练 -> 推理 -> CER 评估

与原合成数据版的区别:
- 数据使用 data/proc 下的真实 TIMIT 语音（50 条英文句子），不再是合成正弦波；
- 词表加载项目根目录 vocab.json（29 类，blank=0，由 prepare_timit.py 从语料统计），
  修复了原版"构建词表但从未 add_text、所有字符落入 UNK 导致 CER=100%"的 bug；
- 特征归一化采用训练集全局 CMVN，统计量随 checkpoint 保存，
  推理端 (SpeechRecognizer) 读取同一组统计量，训练/推理分布一致。

运行:
  python run_demo.py              # 全量 50 条, 60 epoch
  python run_demo.py --limit 8    # 快速冒烟（8 条）
"""

import os
import argparse
import math

import numpy as np
import soundfile as sf
import torch
import torch.optim as optim

from model import DeepSpeech2
from dataset import Vocabulary, SpeechDataset, create_dataloader
from train import CTCLossWrapper
from inference import SpeechRecognizer
from cer import compute_cer_corpus


# ============================================================
# 参数
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="TIMIT 真实数据端到端演示")
    parser.add_argument("--data", default="./data/proc", help="audio/ + transcription.txt 目录")
    parser.add_argument("--vocab", default="./vocab.json", help="词表文件（29 类）")
    parser.add_argument("--save_dir", default="./demo_checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="lr 过大且无 warmup 时 CTC 易塌缩到全 blank")
    parser.add_argument("--min_lr_ratio", type=float, default=0.01,
                        help="余弦退火的最低 lr 比例（相对 base_lr），默认退火到 1%%")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="线性 warmup 步数（防止早期大步长塌缩到全 blank）")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader 进程数；Windows 下 spawn+IPC 开销大于收益，建议 0")
    parser.add_argument("--no_cache", action="store_true",
                        help="禁用 mel 特征缓存（默认启用）")
    parser.add_argument("--rebuild_cache", action="store_true",
                        help="强制重建特征缓存")
    parser.add_argument("--limit", type=int, default=0, help="只用前 N 条数据（0=全部），快速冒烟用")
    parser.add_argument("--show", type=int, default=5, help="推理演示打印条数")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


# ============================================================
# 训练期监控（greedy 快评，无需落盘 checkpoint）
# ============================================================
def _greedy_texts(log_probs, lengths, vocab):
    """log_probs [B,T,C] -> greedy 合并去 blank 后的文本列表。"""
    blank_id = vocab.char2idx.get(vocab.BLANK, 0)
    ids = log_probs.argmax(dim=-1)                      # [B, T]
    out = []
    for b in range(ids.shape[0]):
        seq = ids[b, : int(lengths[b])].tolist()
        collapsed = [k for i, k in enumerate(seq) if i == 0 or k != seq[i - 1]]
        out.append("".join(vocab.idx2char[k] for k in collapsed if k != blank_id))
    return out


def quick_eval(model, dataset, vocab, device):
    """训练集快评：语料级 CER + 首条样例解码，用于监控 blank 塌缩。"""
    model.eval()
    refs = [text for _, text in dataset.samples]
    hyps = []
    with torch.no_grad():
        for batch in create_dataloader(dataset, 8, shuffle=False, num_workers=0):
            log_probs, output_lengths = model(
                batch["features"].to(device), batch["input_lengths"])
            hyps.extend(_greedy_texts(log_probs.cpu(), output_lengths, vocab))
    model.train()
    result = compute_cer_corpus(refs, hyps)
    return result, (refs[0], hyps[0])


# ============================================================
# 训练
# ============================================================
def train_demo(args, vocab, device):
    """在真实 TIMIT 数据上训练演示模型，返回 checkpoint 路径。"""
    print("\n" + "=" * 60)
    print("1. 加载数据与全局 CMVN 统计")
    print("=" * 60)

    dataset = SpeechDataset(args.data, vocab, use_cache=not args.no_cache,
                            limit=args.limit)
    print(f"训练条数: {len(dataset)}")

    # 特征预计算：一次性落盘，后续 epoch 只读缓存（避免每轮重跑 librosa）
    dataset.precompute_features(force=args.rebuild_cache)

    # 全局 CMVN：训练/推理共用同一组统计量（随 checkpoint 保存）
    feature_stats = dataset.compute_global_stats()
    train_loader = create_dataloader(dataset, args.batch_size, shuffle=True,
                                     num_workers=args.num_workers)

    print("\n" + "=" * 60)
    print("2. 构建模型 (5层双向LSTM)")
    print("=" * 60)

    model = DeepSpeech2(
        input_dim=80,
        hidden_dim=args.hidden_dim,   # 演示用小模型
        num_layers=args.num_layers,
        num_classes=len(vocab),
        dropout=args.dropout,
        bidirectional=True,
    ).to(device)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    print("\n" + "=" * 60)
    print("3. 开始训练 (CTC 损失, fp32)")
    print("=" * 60)

    ctc_loss = CTCLossWrapper(blank=0, reduction="mean", zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    # 学习率调度：线性 warmup -> 余弦退火到 base_lr * min_lr_ratio
    # （原版 warmup 后恒定 lr，模型逃逸 blank 塌缩后无法精修，loss 横盘）
    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_steps
    min_lr = args.lr * args.min_lr_ratio

    def get_lr(step: int) -> float:
        if step < warmup_steps:
            return args.lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr + 0.5 * (args.lr - min_lr) * (1 + math.cos(math.pi * progress))

    history = []
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            lr = get_lr(global_step)
            for g in optimizer.param_groups:
                g["lr"] = lr
            global_step += 1

            features = batch["features"].to(device)
            labels = batch["labels"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            label_lengths = batch["label_lengths"].to(device)

            optimizer.zero_grad()
            log_probs, output_lengths = model(features, input_lengths)
            loss = ctc_loss(log_probs, labels, output_lengths, label_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        history.append(avg_loss)
        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
            result, (ref, hyp) = quick_eval(model, dataset, vocab, device)
            print(f"Epoch {epoch:3d}/{args.epochs} | Loss: {avg_loss:.4f} "
                  f"| 训练CER: {result['cer']:.2%} "
                  f"({result['total_distance']}/{result['total_ref_len']})")
            print(f"          样例 参考: {ref}")
            print(f"          样例 识别: {hyp if hyp else '(空)'}")

    # 保存模型（feature_stats 一并保存，推理端从 checkpoint 读取同一组统计量）
    os.makedirs(args.save_dir, exist_ok=True)
    model_path = os.path.join(args.save_dir, "demo_model.pt")
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "vocab": vocab,
        "feature_stats": feature_stats,
        "args": {
            "input_dim": 80,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "bidirectional": True,
        },
        "history": history,
    }, model_path)
    print(f"\n模型已保存: {model_path}")
    return model_path, dataset


# ============================================================
# 推理演示
# ============================================================
def inference_demo(model_path: str, dataset: SpeechDataset, device: str, num_show: int = 5):
    """对真实 wav 逐条 greedy 解码，返回 (refs, hyps)。"""
    print("\n" + "=" * 60)
    print("4. 推理测试 (greedy, 全局 CMVN 同源)")
    print("=" * 60)

    recognizer = SpeechRecognizer(model_path=model_path, device=device, beam_width=1)

    refs, hyps = [], []
    for i, (audio_path, text) in enumerate(dataset.samples):
        audio, sr = sf.read(audio_path, dtype="float32")
        if sr != 16000 or audio.ndim != 1:
            print(f"跳过异常音频 {audio_path}: sr={sr}")
            continue
        hyp, _ = recognizer.recognize(audio, beam_width=1)
        refs.append(text)
        hyps.append(hyp)
        if i < num_show:
            print(f"[{i:02d}] 参考: {text}")
            print(f"     识别: {hyp}")

    return refs, hyps


# ============================================================
# CER 计算
# ============================================================
def cer_demo(refs: list, hyps: list):
    print("\n" + "=" * 60)
    print("5. CER (字错率) 评估")
    print("=" * 60)

    result = compute_cer_corpus(refs, hyps)

    print(f"\n总样本数:     {result['num_samples']}")
    print(f"总参考字数:   {result['total_ref_len']}")
    print(f"总错误数:     {result['total_distance']}")
    print(f"  - 替换:     {result['total_substitutions']}")
    print(f"  - 删除:     {result['total_deletions']}")
    print(f"  - 插入:     {result['total_insertions']}")
    print(f"\n★ 字错率 CER: {result['cer']:.4f} ({result['cer']:.2%})")

    return result


# ============================================================
# 主流程
# ============================================================
def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    print("=" * 60)
    print("5层LSTM + CTC 语音识别 - TIMIT 真实数据端到端演示")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    # 前置检查
    if not os.path.exists(args.vocab):
        raise FileNotFoundError(f"词表不存在: {args.vocab}")
    if not os.path.exists(os.path.join(args.data, "transcription.txt")):
        raise FileNotFoundError(
            f"数据目录不完整: {args.data}（先运行 prepare_timit.py 生成）")

    # 1. 词表（项目根 vocab.json，29 类；blank=0 与 CTCLossWrapper 对齐）
    vocab = Vocabulary.load(args.vocab)
    print(f"词表: {args.vocab} ({len(vocab)} 类)")

    # 2-3. 训练（真实 TIMIT 数据 + 全局 CMVN）
    model_path, dataset = train_demo(args, vocab, device)

    # 4. 推理（同一批数据，greedy 解码）
    refs, hyps = inference_demo(model_path, dataset, device, num_show=args.show)

    # 5. CER 评估
    result = cer_demo(refs, hyps)

    print("\n" + "=" * 60)
    print("演示完成!")

    return result


if __name__ == "__main__":
    main()
