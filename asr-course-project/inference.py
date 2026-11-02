"""
推理代码: 使用训练好的5层LSTM模型进行语音识别
支持贪心解码和束搜索解码
"""

import os
import json
import argparse
import numpy as np
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import DeepSpeech2
from dataset import extract_mel_spectrogram, normalize_features, Vocabulary


class SpeechRecognizer:
    """语音识别推理器"""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        beam_width: int = 10,
        n_mels: int = 80,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        sample_rate: int = 16000,
    ):
        """
        Args:
            model_path: 模型检查点路径
            device: 推理设备
            beam_width: 束搜索宽度 (beam_width=1为贪心解码)
            n_mels: 梅尔滤波器组数
            n_fft: FFT点数
            hop_length: 帧移
            win_length: 窗长
            sample_rate: 采样率
        """
        self.device = torch.device(device)
        self.beam_width = beam_width
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sample_rate = sample_rate

        # 加载模型
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.vocab = checkpoint["vocab"]
        args_dict = checkpoint.get("args", {})
        # 全局 CMVN 统计量（训练时随 checkpoint 保存；旧 checkpoint 缺失则退化为逐条归一化）
        self.feature_stats = checkpoint.get("feature_stats")

        self.model = DeepSpeech2(
            input_dim=args_dict.get("input_dim", n_mels),
            hidden_dim=args_dict.get("hidden_dim", 512),
            num_layers=args_dict.get("num_layers", 5),
            num_classes=len(self.vocab),
            dropout=0.0,  # 推理时关闭dropout
            bidirectional=args_dict.get("bidirectional", True),
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.blank_id = 0  # CTC blank token ID

    def extract_features(self, audio: np.ndarray) -> torch.Tensor:
        """提取音频特征"""
        features = extract_mel_spectrogram(
            audio, self.sample_rate, self.n_mels,
            self.n_fft, self.hop_length, self.win_length
        )

        # 归一化：优先使用训练集全局统计量（与训练一致）；缺失时退化为逐条归一化
        features, _ = normalize_features(features, self.feature_stats)

        # 转为tensor [1, T, F]
        features = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        return features

    @torch.no_grad()
    def recognize(
        self,
        audio: np.ndarray,
        beam_width: int = None,
        use_lm: bool = False,
    ) -> Tuple[str, List[Tuple[str, float]]]:
        """
        识别音频

        Args:
            audio: 音频信号 (1D numpy array, float32)
            beam_width: 束宽度，None则使用默认值
            use_lm: 是否使用语言模型(暂未实现)

        Returns:
            text: 识别文本
            candidates: 候选结果列表 [(text, score), ...]
        """
        if beam_width is None:
            beam_width = self.beam_width

        # 提取特征
        features = self.extract_features(audio)
        input_length = torch.tensor([features.shape[1]], dtype=torch.long).to(self.device)

        # 前向传播
        log_probs, output_lengths = self.model(features, input_length)
        log_probs = log_probs[0, :output_lengths[0], :]  # [T, C]

        if beam_width == 1:
            # 贪心解码
            text, score = self._greedy_decode(log_probs)
            return text, [(text, score)]
        else:
            # 束搜索解码
            return self._beam_search_decode(log_probs, beam_width)

    # ================================================================
    # 贪心解码 (Greedy Decoding)
    # ================================================================
    def _greedy_decode(self, log_probs: torch.Tensor) -> Tuple[str, float]:
        """
        贪心解码: 每帧取概率最大的字符，合并连续重复字符，去除blank

        Args:
            log_probs: [T, C] log概率

        Returns:
            text: 解码文本
            score: 累计对数概率
        """
        # 取每帧最大概率的索引
        _, max_indices = log_probs.max(dim=-1)  # [T]
        indices = max_indices.cpu().numpy()

        # CTC解码: 去除连续重复和blank
        decoded = []
        prev_idx = -1
        total_score = 0.0

        for t, idx in enumerate(indices):
            total_score += log_probs[t, idx].item()
            if idx != self.blank_id and idx != prev_idx:
                decoded.append(idx)
            prev_idx = idx

        text = self.vocab.decode(decoded)
        return text, total_score

    # ================================================================
    # 束搜索解码 (Beam Search Decoding)
    # ================================================================
    def _beam_search_decode(
        self,
        log_probs: torch.Tensor,
        beam_width: int = 10,
    ) -> Tuple[str, List[Tuple[str, float]]]:
        """
        CTC束搜索解码

        每个beam维护:
        - prefix: 解码序列 (不含blank)
        - score: 累积对数概率
        - blank_score: 以blank结尾的分数
        - non_blank_score: 以非blank结尾的分数

        Args:
            log_probs: [T, C] log概率
            beam_width: 束宽度

        Returns:
            best_text: 最优识别文本
            candidates: 所有候选 [(text, score), ...]
        """
        T, C = log_probs.shape
        probs = log_probs.exp().cpu().numpy()  # 转为概率

        # 初始化beam
        # prefix -> (blank_score, non_blank_score)
        beam = {(): (0.0, float("-inf"))}

        for t in range(T):
            new_beam = {}

            for prefix, (b_score, nb_score) in beam.items():
                total_score = np.logaddexp(b_score, nb_score)

                for c in range(C):
                    p = probs[t, c]

                    if c == self.blank_id:
                        # 扩展blank
                        new_b_score = np.logaddexp(b_score + np.log(p), nb_score + np.log(p))
                        new_nb_score = float("-inf")
                    else:
                        if len(prefix) > 0 and prefix[-1] == c:
                            # 相同字符: 非blank扩展
                            new_b_score = b_score
                            new_nb_score = np.logaddexp(nb_score + np.log(p), b_score + np.log(p))
                        else:
                            # 新字符
                            new_prefix = prefix + (c,)
                            new_b_score = b_score
                            new_nb_score = total_score + np.log(p)

                    new_prefix = prefix if c == self.blank_id else (
                        prefix if (len(prefix) > 0 and prefix[-1] == c) else prefix + (c,)
                    )
                    old_b_score, old_nb_score = new_beam.get(new_prefix, (float("-inf"), float("-inf")))
                    new_beam[new_prefix] = (
                        np.logaddexp(old_b_score, new_b_score),
                        np.logaddexp(old_nb_score, new_nb_score),
                    )

            # 裁剪beam
            scored_beam = []
            for prefix, (b_score, nb_score) in new_beam.items():
                score = np.logaddexp(b_score, nb_score)
                scored_beam.append((prefix, b_score, nb_score, score))
            scored_beam.sort(key=lambda x: x[3], reverse=True)
            beam = {item[0]: (item[1], item[2]) for item in scored_beam[:beam_width]}

        # 收集结果
        results = []
        for prefix, (b_score, nb_score) in beam.items():
            score = np.logaddexp(b_score, nb_score)
            text = self.vocab.decode(list(prefix))
            results.append((text, float(score)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[0][0], results

    # ================================================================
    # 前缀束搜索 (Prefix Beam Search) - 更准确的CTC解码
    # ================================================================
    def _prefix_beam_search(
        self,
        log_probs: torch.Tensor,
        beam_width: int = 10,
    ) -> Tuple[str, List[Tuple[str, float]]]:
        """
        前缀束搜索: 在beam的每一步合并相同前缀的不同路径

        参考: "First-Pass Large Vocabulary Continuous Speech Recognition using Bi-Directional Recurrent DNNs"
        """
        T, C = log_probs.shape
        probs = log_probs.exp().cpu().numpy()

        # 初始化: 空前缀
        # 每个beam: prefix -> (p_b, p_nb)
        # p_b: 以blank结尾的概率
        # p_nb: 以非blank结尾的概率
        beam = {tuple(): (1.0, 0.0)}

        for t in range(T):
            new_beam = {}

            for prefix, (p_b, p_nb) in beam.items():
                # 跳过概率为0的beam
                if p_b == 0.0 and p_nb == 0.0:
                    continue

                for c in range(C):
                    prob_c = probs[t, c]

                    if c == self.blank_id:
                        # blank: 不改变prefix
                        new_p_b, new_p_nb = new_beam.get(prefix, (0.0, 0.0))
                        new_p_b += (p_b + p_nb) * prob_c
                        new_beam[prefix] = (new_p_b, new_p_nb)
                    else:
                        if len(prefix) > 0 and prefix[-1] == c:
                            # 相同字符扩展
                            new_prefix = prefix
                            new_p_b, new_p_nb = new_beam.get(new_prefix, (0.0, 0.0))
                            new_p_nb += p_b * prob_c  # 从blank来
                            new_p_nb += p_nb * prob_c  # 延续非blank
                            new_beam[new_prefix] = (new_p_b, new_p_nb)
                        else:
                            # 新字符
                            new_prefix = prefix + (c,)
                            new_p_b, new_p_nb = new_beam.get(new_prefix, (0.0, 0.0))
                            new_p_nb += (p_b + p_nb) * prob_c
                            new_beam[new_prefix] = (new_p_b, new_p_nb)

            # 裁剪beam
            scored = [(pref, b + nb) for pref, (b, nb) in new_beam.items()]
            scored.sort(key=lambda x: x[1], reverse=True)
            beam = {pref: new_beam[pref] for pref, _ in scored[:beam_width]}

        # 收集最终结果
        results = []
        for prefix, (p_b, p_nb) in beam.items():
            score = np.log(p_b + p_nb + 1e-30)
            text = self.vocab.decode(list(prefix))
            results.append((text, float(score)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[0][0], results


# ============================================================
# 批处理推理
# ============================================================
@torch.no_grad()
def batch_recognize(
    recognizer: SpeechRecognizer,
    audio_list: List[np.ndarray],
) -> List[str]:
    """
    批量识别多个音频

    Args:
        recognizer: 识别器
        audio_list: 音频信号列表

    Returns:
        texts: 识别结果列表
    """
    results = []
    for audio in audio_list:
        text, _ = recognizer.recognize(audio)
        results.append(text)
    return results


# ============================================================
# 命令行入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="语音识别推理")
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--audio", type=str, required=True, help="音频文件路径")
    parser.add_argument("--beam_width", type=int, default=10, help="束搜索宽度 (1=贪心)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # 加载识别器
    recognizer = SpeechRecognizer(
        model_path=args.model,
        device=args.device,
        beam_width=args.beam_width,
    )

    # 加载音频
    try:
        import soundfile as sf
        audio, sr = sf.read(args.audio, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
    except ImportError:
        print("请安装 soundfile: pip install soundfile")
        return

    # 重采样
    if sr != recognizer.sample_rate:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=recognizer.sample_rate)

    # 识别
    text, candidates = recognizer.recognize(audio, beam_width=args.beam_width)

    print(f"\n识别结果: {text}")
    print(f"\n候选结果 (Top {min(5, len(candidates))}):")
    for i, (cand_text, score) in enumerate(candidates[:5], 1):
        print(f"  {i}. {cand_text} (score: {score:.2f})")


if __name__ == "__main__":
    main()
