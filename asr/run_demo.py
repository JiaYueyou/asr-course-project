"""
端到端演示: 使用合成数据训练 -> 推理 -> CER评估

运行: python run_demo.py
"""

import os
import sys
import json
import time
import random
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from model import DeepSpeech2
from dataset import Vocabulary, SpeechDataset, create_dataloader, normalize_features
from train import CTCLossWrapper
from inference import SpeechRecognizer
from cer import compute_cer_corpus

os.makedirs("./demo_data/audio", exist_ok=True)
os.makedirs("./demo_checkpoints", exist_ok=True)


# ============================================================
# 生成合成训练数据
# ============================================================
def generate_synthetic_audio(text: str, sample_rate: int = 16000, duration: float = 2.0) -> np.ndarray:
    """生成模拟音频 (正弦波组合)"""
    total_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, total_samples, endpoint=False)

    # 用不同频率的正弦波组合模拟语音
    audio = np.zeros(total_samples, dtype=np.float32)
    for i, char in enumerate(text):
        freq = 200 + (ord(char) % 50) * 20  # 不同字符对应不同频率
        audio += 0.1 * np.sin(2 * np.pi * freq * t * (1 + 0.1 * np.sin(2 * np.pi * 3 * t)))

    # 加噪声
    audio += 0.02 * np.random.randn(total_samples).astype(np.float32)
    audio = audio / (np.abs(audio).max() + 1e-8)
    return audio.astype(np.float32)


def create_demo_dataset(
    texts: list,
    data_dir: str = "./demo_data",
    sample_rate: int = 16000,
):
    """创建演示数据集"""
    import struct
    import wave

    audio_dir = os.path.join(data_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    with open(os.path.join(data_dir, "transcription.txt"), "w", encoding="utf-8") as f:
        for i, text in enumerate(texts):
            audio_name = f"sample_{i:04d}.wav"
            audio_path = os.path.join(audio_dir, audio_name)

            # 生成合成音频
            duration = max(1.0, len(text) * 0.3)  # 每个字约0.3秒
            audio = generate_synthetic_audio(text, sample_rate, duration)

            # 保存为WAV
            audio_int16 = (audio * 32767).astype(np.int16)
            with wave.open(audio_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_int16.tobytes())

            f.write(f"{audio_name} {text}\n")

    print(f"生成 {len(texts)} 条合成数据到 {data_dir}")


# ============================================================
# 训练
# ============================================================
def train_demo(
    train_dir: str,
    vocab_path: str,
    save_dir: str,
    epochs: int = 20,
    device: str = "cpu",
):
    """演示训练流程"""
    print("\n" + "="*60)
    print("1. 构建词表")
    print("="*60)

    vocab = Vocabulary()
    train_dataset = SpeechDataset(train_dir, vocab)
    vocab.save(vocab_path)
    num_classes = len(vocab)
    print(f"词表大小: {num_classes}")

    print("\n" + "="*60)
    print("2. 构建模型 (5层双向LSTM)")
    print("="*60)

    model = DeepSpeech2(
        input_dim=80,
        hidden_dim=128,       # 演示用小模型
        num_layers=5,
        num_classes=num_classes,
        dropout=0.2,
        bidirectional=True,
    ).to(device)

    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    print("\n" + "="*60)
    print("3. 开始训练 (CTC损失)")
    print("="*60)

    train_loader = create_dataloader(train_dataset, batch_size=2, shuffle=True, num_workers=0)

    ctc_loss = CTCLossWrapper(blank=0, reduction="mean", zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            label_lengths = batch["label_lengths"].to(device)

            optimizer.zero_grad()
            log_probs, output_lengths = model(features, input_lengths)
            loss = ctc_loss(log_probs, labels, output_lengths, label_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        history.append(avg_loss)
        print(f"Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.2e}")

    # 保存模型
    model_path = os.path.join(save_dir, "demo_model.pt")
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "vocab": vocab,
        "args": {
            "input_dim": 80,
            "hidden_dim": 128,
            "num_layers": 5,
            "bidirectional": True,
        },
        "history": history,
    }, model_path)
    print(f"\n模型已保存: {model_path}")

    return model_path, model, vocab


# ============================================================
# 推理演示
# ============================================================
def inference_demo(model_path: str, test_texts: list, device: str = "cpu"):
    """演示推理流程"""
    print("\n" + "="*60)
    print("4. 推理测试")
    print("="*60)

    recognizer = SpeechRecognizer(
        model_path=model_path,
        device=device,
        beam_width=1,
    )

    refs = []
    hyps = []

    for text in test_texts:
        # 生成合成音频
        duration = max(1.0, len(text) * 0.3)
        audio = generate_synthetic_audio(text, duration=duration)

        # 识别
        hyp, _ = recognizer.recognize(audio, beam_width=1)

        refs.append(text)
        hyps.append(hyp)
        print(f"参考: {text:<10s} | 识别: {hyp:<10s}")

    return refs, hyps


# ============================================================
# CER计算
# ============================================================
def cer_demo(refs: list, hyps: list):
    """演示CER计算"""
    print("\n" + "="*60)
    print("5. CER (字错率) 评估")
    print("="*60)

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
    print("="*60)
    print("5层LSTM + CTC 语音识别 - 端到端演示")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    # 训练文本
    train_texts = [
        "今天天气真好啊",
        "我想去公园散步",
        "明天会下雨吗",
        "今天吃什么好呢",
        "我喜歡學習中文",
        "语音识别很有趣",
        "深度学习真强大",
        "人工智能改变世界",
        "今天天气真好啊",
        "我想去公园散步",
        "你好世界大家好",
        "今天星期一上班",
        "苹果很好吃很甜",
        "我喜欢听音乐",
        "机器学习很重要",
        "自然语言处理",
        "计算机视觉领域",
        "数据分析与挖掘",
    ]

    # 测试文本 (含训练集未见过的)
    test_texts = [
        "今天天气真好啊",
        "语音识别技术",
        "深度学习模型",
    ]

    # 1. 生成数据
    print("\n生成合成训练数据...")
    create_demo_dataset(train_texts, "./demo_data")

    # 2. 训练
    model_path, model, vocab = train_demo(
        train_dir="./demo_data",
        vocab_path="./demo_data/vocab.json",
        save_dir="./demo_checkpoints",
        epochs=20,
        device=device,
    )

    # 3. 推理
    refs, hyps = inference_demo(model_path, test_texts, device)

    # 4. CER
    result = cer_demo(refs, hyps)

    print("\n" + "="*60)
    print("演示完成!")
    print("="*60)


if __name__ == "__main__":
    main()
