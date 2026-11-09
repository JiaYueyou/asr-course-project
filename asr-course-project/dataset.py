"""
数据集和数据预处理模块
支持从音频文件提取特征，构建字符级词表
"""

import os
import json
import time
import hashlib
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor


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

    直接使用向量化 NumPy 实现（而非 librosa）。原因：本环境 librosa 0.11 的
    melspectrogram 首次调用存在数分钟级的 numba JIT 开销，多进程预计算时
    每个 worker 都要重复编译，导致取数极慢。NumPy 路径 ~2ms/条，且训练、
    缓存、推理三处共用同一实现，避免特征定义漂移。

    Args:
        audio: 音频信号 (1D numpy array)
        sample_rate: 采样率
        n_mels: 梅尔滤波器组数量
        n_fft: FFT点数
        hop_length: 帧移
        win_length: 窗长
        f_min: 最低频率 (保留参数以兼容原调用)
        f_max: 最高频率 (保留参数以兼容原调用)

    Returns:
        mel_spec: 梅尔频谱 [time, n_mels]
    """
    return _simple_mel(audio, sample_rate, n_mels, n_fft, hop_length, win_length)


def _simple_mel(
    audio: np.ndarray,
    sample_rate: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
) -> np.ndarray:
    """不依赖librosa的简化梅尔频谱提取（向量化，避免逐帧 Python 循环）"""
    window = np.hanning(win_length).astype(np.float32)
    if len(audio) < win_length:
        audio = np.pad(audio, (0, win_length - len(audio)))
    num_frames = 1 + (len(audio) - win_length) // hop_length

    # 分帧: 用滑窗视图取帧并做 hop 抽步，避免显式循环
    frames = np.lib.stride_tricks.sliding_window_view(audio, win_length)[::hop_length]
    frames = frames[:num_frames] * window
    spec = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)).astype(np.float32)

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
# 音频读取 (模块级，供 DataLoader worker 与多进程预计算复用)
# ============================================================
def load_audio_file(path: str, sample_rate: int = 16000) -> np.ndarray:
    """读取音频为单声道 float32；soundfile 不可用时回退到 wave 模块。"""
    try:
        import soundfile as sf
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # 转单声道
        if sr != sample_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
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

        if sr != sample_rate:
            audio = np.interp(
                np.linspace(0, len(audio), int(len(audio) * sample_rate / sr)),
                np.arange(len(audio)),
                audio,
            )
        return audio.astype(np.float32)


# ============================================================
# 特征缓存
# ============================================================
CACHE_VERSION = 1


def _mel_backend() -> str:
    """当前 mel 提取后端标识，写入缓存签名以便实现变化时自动重建。"""
    return "numpy"


def _save_npy_atomic(path: str, array: np.ndarray):
    """先写临时文件再原子替换，避免 DataLoader 多进程写坏文件。"""
    tmp = f"{path}.{os.getpid()}.tmp.npy"
    np.save(tmp, array)
    os.replace(tmp, path)


def _scan_dir_meta(dir_path: str) -> Dict[str, Tuple[int, int]]:
    """一次 os.scandir 收集目录内文件名 -> (size, mtime)。

    Windows 上单次 stat 约 0.8ms，逐条样本比对 mtime 会产生上万次系统调用；
    这里改为一趟扫描，之后全部是内存字典查找。目录不存在时返回空字典。
    """
    meta = {}
    try:
        with os.scandir(dir_path) as it:
            for entry in it:
                try:
                    st = entry.stat()
                except OSError:
                    continue
                meta[entry.name] = (st.st_size, int(st.st_mtime))
    except OSError:
        pass
    return meta


def _precompute_worker(task) -> str:
    """多进程任务：读音频 -> 提特征 -> 落盘。"""
    audio_path, out_path, params = task
    audio = load_audio_file(audio_path, params["sample_rate"])
    feats = extract_mel_spectrogram(audio, **params)
    _save_npy_atomic(out_path, feats)
    return out_path


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
# SpecAugment 数据增强（仅训练集，在 CMVN 归一化后施加）
# ============================================================
def spec_augment(
    features: np.ndarray,
    n_freq_masks: int = 2,
    freq_mask_width: int = 15,
    n_time_masks: int = 2,
    time_mask_width: int = 40,
    max_time_ratio: float = 0.2,
) -> np.ndarray:
    """对归一化后的特征做频域/时域遮罩（SpecAugment 简化版）。

    遮罩填 0（CMVN 归一化后 0 即全局均值）。TIMIT 句子较短（约
    100-500 帧），时间遮罩总量限制在时长的 max_time_ratio 内，避免
    把短句遮成全 0。每次调用随机采样，仅用于训练集。
    """
    feats = features.copy()
    T, F = feats.shape

    for _ in range(n_freq_masks):
        f = random.randint(1, min(freq_mask_width, F))
        f0 = random.randint(0, F - f)
        feats[:, f0:f0 + f] = 0.0

    max_total = int(T * max_time_ratio)
    masked = 0
    for _ in range(n_time_masks):
        if masked >= max_total:
            break
        t = random.randint(1, min(time_mask_width, max_total - masked, T))
        t0 = random.randint(0, T - t)
        feats[t0:t0 + t, :] = 0.0
        masked += t

    return feats


def split_speakers(data_dir: str, dev_ratio: float, seed: int = 42):
    """按说话人切分 data_dir 下的数据（说话人 = 文件名首个下划线前的前缀）。

    返回 (train_speakers, dev_speakers)，均为 set[str]。同一份数据 +
    同一 seed 结果确定，训练可重复。按说话人而非按句子切分，避免同一
    说话人的句子同时出现在训练/验证两侧造成泄漏。
    """
    trans_path = os.path.join(data_dir, "transcription.txt")
    speakers = set()
    if os.path.exists(trans_path):
        with open(trans_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if parts:
                    speakers.add(parts[0].split("_", 1)[0])
    spk_list = sorted(speakers)
    random.Random(seed).shuffle(spk_list)
    n_dev = int(round(len(spk_list) * dev_ratio))
    return set(spk_list[n_dev:]), set(spk_list[:n_dev])


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
        """将文本编码为索引序列

        词表含 <unk> 时集外字符映射到 UNK；不含 <unk>（封闭词表，如本项目
        29 类英文词表）时集外字符直接报错。注意不能写
        char2idx.get(c, char2idx[UNK])——默认值会被无条件求值，
        词表没有 <unk> 时即使字符全部在表内也会 KeyError。
        """
        unk_idx = self.char2idx.get(self.UNK)
        ids = []
        for c in text:
            idx = self.char2idx.get(c)
            if idx is None:
                if unk_idx is None:
                    raise KeyError(f"字符 {c!r} 不在词表中，且词表未定义 <unk>")
                idx = unk_idx
            ids.append(idx)
        return ids

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
        use_cache: bool = True,      # 是否启用 mel 特征磁盘缓存
        cache_dir: Optional[str] = None,  # 默认 <data_dir>/cache
        limit: int = 0,              # >0 时只用前 N 条（冒烟测试）
        speakers: Optional[set] = None,  # 仅保留这些说话人的样本（说话人不相交切分）
        apply_specaug: bool = False,     # 训练集 SpecAugment 在线增强
    ):
        self.data_dir = data_dir
        self.vocab = vocab
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.max_frames = int(max_duration * sample_rate / hop_length)

        # 加载数据清单（先扫一次 audio 目录，清单校验与缓存对比都复用它）
        self._audio_meta = _scan_dir_meta(os.path.join(data_dir, "audio"))
        self.samples = self._load_manifest()
        if speakers is not None:
            before = len(self.samples)
            self.samples = [
                (p, t) for p, t in self.samples
                if os.path.basename(p).split("_", 1)[0] in speakers
            ]
            print(f"说话人过滤: {before} -> {len(self.samples)} 条")
        if limit and limit > 0:
            self.samples = self.samples[:limit]
            print(f"仅使用前 {len(self.samples)} 条数据（limit={limit}）")
        self.apply_specaug = apply_specaug

        # 特征统计量 (用于在线归一化)
        self.stats = None

        # ---------- 特征缓存布局 ----------
        # 目录按"特征参数 + 提取后端"哈希，参数一变自动换目录，不会读到旧特征
        self.use_cache = use_cache
        self._mel_params = {
            "sample_rate": sample_rate, "n_mels": n_mels, "n_fft": n_fft,
            "hop_length": hop_length, "win_length": win_length,
        }
        self._sig = {**self._mel_params, "backend": _mel_backend(),
                     "version": CACHE_VERSION}
        self._sig_hash = hashlib.md5(
            json.dumps(self._sig, sort_keys=True).encode()).hexdigest()[:10]
        cache_root = cache_dir or os.path.join(data_dir, "cache")
        self.cache_root = os.path.join(cache_root, f"mel_{self._sig_hash}")
        # 缓存目录同样只扫一次，之后逐条比对 mtime 全在内存里做，
        # 避免每条样本两次 os.stat（4618 条即 9000+ 次系统调用，Windows 上约 8s）
        self._cache_meta = _scan_dir_meta(self.cache_root)
        self._manifest_fp = self._manifest_fingerprint()
        # CMVN 缓存文件名含清单指纹与样本数，子集/数据变化时自动失效
        self.stats_cache_path = os.path.join(
            cache_root,
            f"cmvn_{self._sig_hash}_{self._manifest_fp}_{len(self.samples)}.npz")
        self._cache_ready = self._cache_meta_valid()

    def _load_manifest(self) -> List[Tuple[str, str]]:
        """加载音频文件路径和文本标注。

        文件存在性复用 self._audio_meta（目录扫描结果），避免逐行 os.path.exists。
        """
        samples = []
        audio_dir = os.path.join(self.data_dir, "audio")
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
                        if audio_name in self._audio_meta:
                            samples.append(
                                (os.path.join(audio_dir, audio_name), text))
        else:
            # 无标注文件时遍历 audio 目录，文本留空待自行标注
            for fname in sorted(self._audio_meta):
                if fname.endswith((".wav", ".flac", ".mp3")):
                    samples.append((os.path.join(audio_dir, fname), ""))

        print(f"加载 {len(samples)} 条语音数据")
        return samples

    def _load_audio(self, path: str) -> np.ndarray:
        """加载音频文件（实现见模块级 load_audio_file，便于多进程复用）"""
        return load_audio_file(path, self.sample_rate)

    # ---------- 特征缓存 ----------
    def _manifest_fingerprint(self) -> str:
        """音频清单指纹：复用 __init__ 中一次 scandir 的结果，不额外 stat。

        音频被重新生成时指纹变化，CMVN 统计缓存随之失效。
        """
        h = hashlib.md5()
        for path, _ in self.samples:
            name = os.path.basename(path)
            size, mtime = self._audio_meta.get(name, (0, 0))
            h.update(f"{name}:{size}:{mtime}".encode())
        return h.hexdigest()[:10]

    def _feature_cache_path(self, idx: int) -> str:
        stem = os.path.splitext(os.path.basename(self.samples[idx][0]))[0]
        return os.path.join(self.cache_root, f"{stem}.npy")

    def _cache_stale(self, idx: int, cache_path: str) -> bool:
        """缓存是否缺失或比源音频旧（源被重新生成则需要重算）。

        只查内存中的扫描结果，无系统调用。
        """
        audio = self._audio_meta.get(os.path.basename(self.samples[idx][0]))
        cache = self._cache_meta.get(os.path.basename(cache_path))
        if audio is None or cache is None:
            return True
        return cache[1] < audio[1]

    def _refresh_cache_meta(self):
        """预计算落盘后重新扫描缓存目录，让内存视图与磁盘一致。"""
        self._cache_meta = _scan_dir_meta(self.cache_root)

    def _cache_meta_valid(self) -> bool:
        """缓存目录是否对应当前特征参数（逐条失效由 mtime 负责）。"""
        if not self.use_cache:
            return False
        meta_path = os.path.join(self.cache_root, "meta.json")
        if not os.path.exists(meta_path):
            return False
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, ValueError):
            return False
        return (meta.get("signature") == self._sig
                and meta.get("version") == CACHE_VERSION)

    def precompute_features(self, force: bool = False,
                            num_workers: Optional[int] = None) -> int:
        """预计算全部 mel 特征并落盘，返回本次实际计算的条数。

        命中缓存时直接返回 0；DataLoader 取数时只做 np.load，避免每 epoch
        重复跑 librosa（原实现的约 0.3s/条 CPU 开销）。
        """
        if not self.use_cache:
            print("特征缓存已禁用，跳过预计算")
            return 0

        need_rebuild = force or not self._cache_meta_valid()
        if need_rebuild:
            targets = list(range(len(self.samples)))
        else:
            # 只补缺失或比源音频旧的条目（数据重新生成后自动刷新）
            targets = [i for i in range(len(self.samples))
                       if self._cache_stale(i, self._feature_cache_path(i))]

        if not targets:
            self._cache_ready = True
            print(f"特征缓存已就绪（{len(self.samples)} 条全部命中）: {self.cache_root}")
            return 0

        os.makedirs(self.cache_root, exist_ok=True)
        print(f"特征缓存预计算: {len(targets)}/{len(self.samples)} 条 -> {self.cache_root}")
        tasks = [(self.samples[i][0], self._feature_cache_path(i), self._mel_params)
                 for i in targets]
        workers = num_workers if num_workers is not None else min(8, os.cpu_count() or 1)
        t0 = time.time()
        done = 0
        if len(tasks) >= 64 and workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                for _ in ex.map(_precompute_worker, tasks, chunksize=16):
                    done += 1
                    if done % 200 == 0 or done == len(tasks):
                        print(f"  {done}/{len(tasks)}  ({time.time() - t0:.1f}s)")
        else:
            for task in tasks:
                _precompute_worker(task)
                done += 1
                if done % 200 == 0 or done == len(tasks):
                    print(f"  {done}/{len(tasks)}  ({time.time() - t0:.1f}s)")

        meta = {
            "version": CACHE_VERSION,
            "signature": self._sig,
            "manifest_fingerprint": self._manifest_fp,
            "num_files": len(self.samples),
            "elapsed_sec": round(time.time() - t0, 2),
        }
        with open(os.path.join(self.cache_root, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        self._refresh_cache_meta()
        self._cache_ready = True
        return len(targets)

    def _get_features(self, idx: int) -> np.ndarray:
        """取一条样本的 mel 特征：缓存命中直接读盘，缺失则现算并补写缓存。"""
        if self._cache_ready:
            path = self._feature_cache_path(idx)
            if not self._cache_stale(idx, path):
                try:
                    return np.load(path)
                except (OSError, ValueError):
                    pass  # 文件损坏则重算

        audio = self._load_audio(self.samples[idx][0])
        feats = extract_mel_spectrogram(
            audio, self.sample_rate, self.n_mels,
            self.n_fft, self.hop_length, self.win_length,
        )
        if self.use_cache:
            try:
                os.makedirs(self.cache_root, exist_ok=True)
                path = self._feature_cache_path(idx)
                _save_npy_atomic(path, feats)
                try:
                    st = os.stat(path)
                    self._cache_meta[os.path.basename(path)] = (st.st_size, int(st.st_mtime))
                except OSError:
                    pass
            except OSError:
                pass
        return feats

    def compute_global_stats(self) -> dict:
        """遍历全部样本统计全局 CMVN；结果优先从缓存读取。

        全局方案（而非逐条归一化）：训练/验证/推理共用同一组 mean/std，
        统计量随 checkpoint 保存，保证三方输入分布一致。
        """
        if self.use_cache and os.path.exists(self.stats_cache_path):
            try:
                data = np.load(self.stats_cache_path)
                self.stats = {
                    "mean": data["mean"].astype(np.float32),
                    "std": data["std"].astype(np.float32),
                }
                print(f"从缓存加载全局 CMVN 统计: {int(data['num_frames'][0])} 帧 "
                      f"x {self.n_mels} 维")
                return self.stats
            except (OSError, ValueError, KeyError):
                pass  # 缓存损坏则重算

        all_feats = [self._get_features(i) for i in range(len(self.samples))]
        concat = np.concatenate(all_feats, axis=0)
        self.stats = {
            "mean": concat.mean(axis=0).astype(np.float32),
            "std": concat.std(axis=0).astype(np.float32) + 1e-8,
        }
        if self.use_cache:
            try:
                os.makedirs(os.path.dirname(self.stats_cache_path), exist_ok=True)
                np.savez_compressed(
                    self.stats_cache_path,
                    mean=self.stats["mean"], std=self.stats["std"],
                    num_frames=np.array([concat.shape[0]], dtype=np.int64),
                )
            except OSError:
                pass
        print(f"全局 CMVN 统计完成: {concat.shape[0]} 帧 x {concat.shape[1]} 维")
        return self.stats

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        audio_path, text = self.samples[idx]

        # 取特征（缓存命中时等价于一次 np.load，不再重跑 librosa）
        features = self._get_features(idx)

        # 截断过长的音频
        if features.shape[0] > self.max_frames:
            features = features[:self.max_frames]

        # 全局 CMVN：stats 由 train.py 从训练集统计（或推理端从 checkpoint 恢复）；
        # stats 为 None 时退化为逐条归一化（保持原行为）
        features, _ = normalize_features(features, self.stats)

        # SpecAugment：仅训练集，在归一化后施加（遮罩 0 = 全局均值）
        if self.apply_specaug:
            features = spec_augment(features)

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
    num_workers: int = 0,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
) -> DataLoader:
    """创建 DataLoader。

    实测（Windows + 缓存命中）：nw=0 约 65ms/batch，nw=4 约 244ms/batch ——
    spawn 启动与 IPC 序列化开销超过收益，故默认 0；Linux 下可适当调大。
    """
    kwargs = {}
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = persistent_workers
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
        **kwargs,
    )
