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
        lm=None,
        lm_alpha: float = 0.0,
        lm_beta: float = 0.0,
    ) -> Tuple[str, List[Tuple[str, float]]]:
        """
        CTC束搜索解码（可选语言模型浅融合）

        每个beam维护:
        - prefix: 解码序列 (不含blank)
        - (blank_score, non_blank_score): CTC 对数概率（blank/非blank结尾分账）
        - lm_score / n_words / words / partial: 语言模型状态（无 LM 时不更新）

        浅融合打分: score = log P_ctc + lm_alpha * log P_lm + lm_beta * 词数
        LM 打分在空格定出词边界时增量触发（4-gram 上文取最近 3 个词）。

        Args:
            log_probs: [T, C] log概率
            beam_width: 束宽度
            lm: WordLM 实例（lm.py），None 则纯声学解码
            lm_alpha: 语言模型融合权重
            lm_beta: 词插入奖励（补偿声学模型倾向删词）

        Returns:
            best_text: 最优识别文本
            candidates: 所有候选 [(text, score), ...]
        """
        T, C = log_probs.shape
        # 直接在对数域运算，避免 exp/log 往返造成的下溢（原实现的 log(0) 警告来源）
        log_p = log_probs.cpu().numpy()

        NEG_INF = float("-inf")
        use_lm = lm is not None and lm_alpha > 0.0
        space_id = self.vocab.char2idx.get(" ")
        # 词表里没有空格符号时无法做词级融合，退回纯声学
        if use_lm and space_id is None:
            use_lm = False

        def _add(table, prefix, b, nb, lm_st=None):
            """合并 (blank 分数, 非 blank 分数)；lm_st 为该 prefix 的语言模型状态"""
            old = table.get(prefix)
            if old is None:
                table[prefix] = (np.logaddexp(NEG_INF, b), np.logaddexp(NEG_INF, nb),
                                 *lm_st)
            else:
                table[prefix] = (np.logaddexp(old[0], b), np.logaddexp(old[1], nb),
                                 *lm_st)  # lm 状态只依赖 prefix 字符串，各路径一致

        # 初始 LM 状态: (lm_score, n_words, words, partial)
        init_lm = (0.0, 0, (), "")
        # 初始化 beam: 空前缀 -> (blank 分数, 非 blank 分数, LM 状态)
        beam = {(): (0.0, NEG_INF, *init_lm)}

        for t in range(T):
            new_beam = {}

            for prefix, (b_score, nb_score, lm_score, n_words, words, partial) in beam.items():
                total_score = np.logaddexp(b_score, nb_score)

                for c in range(C):
                    p = log_p[t, c]
                    # 跳过概率可忽略的字符（p < 3e-7），大幅减少无效扩展
                    if p < -15.0:
                        continue

                    if c == self.blank_id:
                        # 发射 blank：prefix 不变、以 blank 结尾。
                        # 注意 blank/非 blank 两条路径都必须消耗本帧发射概率，
                        # 不允许分数"免费携带"（原实现的 bug 会导致概率质量
                        # 随帧数虚增，重复字符前缀胜出）
                        _add(new_beam, prefix, total_score + p, NEG_INF,
                             (lm_score, n_words, words, partial))
                    elif len(prefix) > 0 and prefix[-1] == c:
                        # 与前缀末字符相同，分两种转移：
                        # (a) 紧接重复（中间无 blank）-> CTC 折叠规则，prefix
                        #     不变，只能来自"非 blank 结尾"的路径
                        _add(new_beam, prefix, NEG_INF, nb_score + p,
                             (lm_score, n_words, words, partial))
                        # (b) blank 之后再出现 c -> 是新字符，应生成 prefix+c，
                        #     只能来自"blank 结尾"的路径
                        # （若 c 是连续空格，不拼进残词）
                        _add(new_beam, prefix + (c,), NEG_INF, b_score + p,
                             (lm_score, n_words, words,
                              partial if c == space_id else partial + self.vocab.idx2char[c]))
                    elif use_lm and c == space_id:
                        # 空格：词边界。partial 非空时把这个词写入 LM 分数
                        if partial:
                            new_lm = (lm_score + lm.log_prob(partial, words[-3:]),
                                      n_words + 1, words + (partial,), "")
                        else:
                            new_lm = (lm_score, n_words, words, partial)
                        _add(new_beam, prefix + (c,), NEG_INF, total_score + p, new_lm)
                    else:
                        # 新字符：blank / 非 blank 结尾的路径均可转移
                        _add(new_beam, prefix + (c,), NEG_INF, total_score + p,
                             (lm_score, n_words, words, partial + self.vocab.idx2char[c]))

            # 裁剪 beam（排序分数含 LM 项）
            scored_beam = []
            for prefix, (b_score, nb_score, lm_score, n_words, _, _) in new_beam.items():
                score = (np.logaddexp(b_score, nb_score)
                         + (lm_alpha * lm_score + lm_beta * n_words if use_lm else 0.0))
                scored_beam.append((prefix, b_score, nb_score, score, new_beam[prefix]))
            scored_beam.sort(key=lambda x: x[3], reverse=True)
            beam = {item[0]: item[4] for item in scored_beam[:beam_width]}

        # 收集结果（句尾残词也计入 LM 分数）
        results = []
        for prefix, (b_score, nb_score, lm_score, n_words, words, partial) in beam.items():
            score = np.logaddexp(b_score, nb_score)
            if use_lm:
                lm_s, nw = lm_score, n_words
                if partial:
                    lm_s += lm.log_prob(partial, words[-3:])
                    nw += 1
                score += lm_alpha * lm_s + lm_beta * nw
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
