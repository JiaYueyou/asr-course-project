"""
数据集和数据预处理模块
支持从音频文件提取特征，构建字符级词表
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional


# ============================================================
# 音频特征提取 (Mel频谱)
# ============================================================
def extract_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int = 16000,
    n_mels: int = 80,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
    f_min: float = 80.0,
    f_max: float = 7600.0,
) -> np.ndarray:
    """
    提取梅尔频谱特征

    Args:
        audio: 音频信号 (1D numpy array)
        sample_rate: 采样率
        n_mels: 梅尔滤波器组数量
        n_fft: FFT点数
        hop_length: 帧移
        win_length: 窗长
        f_min: 最低频率
        f_max: 最高频率

    Returns:
        mel_spec: 梅尔频谱 [time, n_mels]
    """
    try:
        import librosa
        mel_spec = librosa.feature.melspectrogram(
            y=audio.astype(np.float32),
            sr=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            fmin=f_min,
            fmax=f_max,
        )
        # 转换为对数刻度
        log_mel = np.log10(np.maximum(mel_spec, 1e-10))
        # 转置为 [time, n_mels]
        return log_mel.T.astype(np.float32)
    except ImportError:
        # 简易版: 使用STFT + 梅尔滤波器组 (不依赖librosa)
        return _simple_mel(audio, sample_rate, n_mels, n_fft, hop_length, win_length)


def _simple_mel(
    audio: np.ndarray,
    sample_rate: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
) -> np.ndarray:
    """不依赖librosa的简化梅尔频谱提取"""
    # Hann窗
    window = np.hanning(win_length)
    num_frames = 1 + (len(audio) - win_length) // hop_length

    spec = np.zeros((num_frames, n_fft // 2 + 1), dtype=np.float32)
    for i in range(num_frames):
        start = i * hop_length
        frame = audio[start:start + win_length] * window
        fft = np.fft.rfft(frame, n=n_fft)
        spec[i] = np.abs(fft)

    # 简化的梅尔滤波器组
    mel_fb = _create_mel_filterbank(sample_rate, n_fft, n_mels)
    mel_spec = np.dot(spec, mel_fb.T)
    log_mel = np.log10(np.maximum(mel_spec, 1e-10))
    return log_mel.astype(np.float32)


def _create_mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    """创建梅尔滤波器组矩阵"""
    low_mel = 2595 * np.log10(1 + 80 / 700)
    high_mel = 2595 * np.log10(1 + sr / 2 / 700)
    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bin_indices = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    filters = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        left = bin_indices[m - 1]
        center = bin_indices[m]
        right = bin_indices[m + 1]

        for k in range(left, center):
            filters[m - 1, k] = (k - left) / (center - left)
        for k in range(center, right):
            filters[m - 1, k] = (right - k) / (right - center)

    return filters


# ============================================================
# 特征归一化
# ============================================================
def normalize_features(features: np.ndarray, stats: dict = None) -> Tuple[np.ndarray, dict]:
    """
    全局均值/方差归一化

    Args:
        features: [time, n_mels]
        stats: 预计算的统计量 {"mean": ..., "std": ...}

    Returns:
        normalized: 归一化后的特征
        stats: 统计量
    """
    if stats is None:
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True) + 1e-8
        stats = {"mean": mean, "std": std}
    normalized = (features - stats["mean"]) / stats["std"]
    return normalized, stats


# ============================================================
# 字符词表构建
# ============================================================
class Vocabulary:
    """字符级词表"""

    BLANK = "<blank>"
    SPACE = "<space>"
    UNK = "<unk>"
    PAD = "<pad>"

    def __init__(self):
        self.idx2char = [self.BLANK, self.SPACE, self.UNK, self.PAD]
        self.char2idx = {c: i for i, c in enumerate(self.idx2char)}

    def add_text(self, text: str):
        """从文本中提取字符"""
        for char in text:
            if char not in self.char2idx:
                self.char2idx[char] = len(self.idx2char)
                self.idx2char.append(char)

    def encode(self, text: str) -> List[int]:
        """将文本编码为索引序列"""
        return [self.char2idx.get(c, self.char2idx[self.UNK]) for c in text]

    def decode(self, indices: List[int], remove_blank: bool = True) -> str:
        """将索引序列解码为文本"""
        chars = []
        for idx in indices:
            if remove_blank and idx == 0:  # skip blank
                continue
            char = self.idx2char[idx]
            if char == self.SPACE:
                chars.append(" ")
            elif char not in (self.BLANK, self.UNK, self.PAD):
                chars.append(char)
        return "".join(chars)

    def __len__(self):
        return len(self.idx2char)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"idx2char": self.idx2char}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls()
        vocab.idx2char = data["idx2char"]
        vocab.char2idx = {c: i for i, c in enumerate(vocab.idx2char)}
        return vocab


# ============================================================
# 数据集
# ============================================================
class SpeechDataset(Dataset):
    """
    语音识别数据集

    期望的数据格式:
    - data_dir/audio/*.wav (音频文件)
    - data_dir/transcription.txt (每行: "音频文件名 文本标注")
    """

    def __init__(
        self,
        data_dir: str,
        vocab: Vocabulary,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        max_duration: float = 20.0,  # 最大音频时长(秒)
    ):
        self.data_dir = data_dir
        self.vocab = vocab
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.max_frames = int(max_duration * sample_rate / hop_length)

        # 加载数据清单
        self.samples = self._load_manifest()

        # 特征统计量 (用于在线归一化)
        self.stats = None

    def _load_manifest(self) -> List[Tuple[str, str]]:
        """加载音频文件路径和文本标注"""
        samples = []
        trans_path = os.path.join(self.data_dir, "transcription.txt")

        if os.path.exists(trans_path):
            with open(trans_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(maxsplit=1)
                    if len(parts) == 2:
                        audio_name, text = parts
                        audio_path = os.path.join(self.data_dir, "audio", audio_name)
                        if os.path.exists(audio_path):
                            samples.append((audio_path, text))
        else:
            # 遍历audio目录
            audio_dir = os.path.join(data_dir, "audio")
            if os.path.exists(audio_dir):
                for fname in sorted(os.listdir(audio_dir)):
                    if fname.endswith((".wav", ".flac", ".mp3")):
                        audio_path = os.path.join(audio_dir, fname)
                        # 默认文本为空，需自行标注
                        samples.append((audio_path, ""))

        print(f"加载 {len(samples)} 条语音数据")
        return samples

    def _load_audio(self, path: str) -> np.ndarray:
        """加载音频文件"""
        try:
            import soundfile as sf
            audio, sr = sf.read(path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)  # 转单声道
            if sr != self.sample_rate:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            return audio.astype(np.float32)
        except ImportError:
            import wave
            import struct
            with wave.open(path, "rb") as wf:
                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                n_frames = wf.getnframes()
                sr = wf.getframerate()
                raw = wf.readframes(n_frames)

            if sample_width == 2:
                fmt = f"<{n_frames * n_channels}h"
                audio = np.array(struct.unpack(fmt, raw), dtype=np.float32)
            else:
                raise ValueError("不支持的音频格式")

            if n_channels > 1:
                audio = audio.reshape(-1, n_channels).mean(axis=1)
            audio = audio / 32768.0

            if sr != self.sample_rate:
                audio = np.interp(
                    np.linspace(0, len(audio), int(len(audio) * self.sample_rate / sr)),
                    np.arange(len(audio)),
                    audio,
                )
            return audio.astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        audio_path, text = self.samples[idx]

        # 加载音频并提取特征
        audio = self._load_audio(audio_path)
        features = extract_mel_spectrogram(
            audio, self.sample_rate, self.n_mels,
            self.n_fft, self.hop_length, self.win_length
        )

        # 截断过长的音频
        if features.shape[0] > self.max_frames:
            features = features[:self.max_frames]

        # 编码文本
        if text:
            label = self.vocab.encode(text)
        else:
            label = []

        return {
            "features": torch.FloatTensor(features),
            "label": torch.LongTensor(label),
            "input_length": features.shape[0],
            "label_length": len(label),
            "audio_path": audio_path,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    DataLoader的collate函数，处理变长序列
    """
    batch_size = len(batch)

    # 找到最大长度
    max_input_len = max(item["features"].shape[0] for item in batch)
    max_label_len = max(item["label"].shape[0] for item in batch)
    feat_dim = batch[0]["features"].shape[1]

    # 初始化填充后的张量
    features = torch.zeros(batch_size, max_input_len, feat_dim)
    labels = torch.zeros(batch_size, max_label_len, dtype=torch.long)
    input_lengths = torch.zeros(batch_size, dtype=torch.long)
    label_lengths = torch.zeros(batch_size, dtype=torch.long)

    for i, item in enumerate(batch):
        feat_len = item["features"].shape[0]
        label_len = item["label"].shape[0]

        features[i, :feat_len, :] = item["features"]
        labels[i, :label_len] = item["label"]
        input_lengths[i] = feat_len
        label_lengths[i] = label_len

    return {
        "features": features,
        "labels": labels,
        "input_lengths": input_lengths,
        "label_lengths": label_lengths,
    }


def create_dataloader(
    dataset: SpeechDataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """创建DataLoader"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )
