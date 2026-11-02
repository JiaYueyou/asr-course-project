"""从 TIMIT（ModelScope Apache-2.0 镜像）选取 50 条英文语音，生成与 asr 兼容的数据目录。

用法（在 asr-my-project 下）:
  python -E src/prepare_timit.py --src data/raw/TIMIT/raw/data/TRAIN --dst data/proc --num 50

产出:
  data/proc/audio/<utt>.wav       # RIFF PCM16, 16 kHz 单声道
  data/proc/transcription.txt     # 每行 "<utt>.wav <清洗后文本>"（文本内含空格）
  data/proc/vocab.json            # 字符级词表，blank=0

选取规则（对应复现规划第三节）:
  1. 仅用 TRAIN 划分；排除 SA1/SA2（固定句，630 人重复，不利于词汇多样性）；
  2. 说话人集中：按方言区/说话人字典序遍历，每人最多 5 条，凑满 50 条为止；
  3. 时长 2-8 s、清洗后文本 20-150 字符（含空格）；
  4. 硬性校验 CTC 约束 T >= 2L+1（帧率 100 fps，帧数 T = 1 + (N-400)/160）。
"""
import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
WIN, HOP = 400, 160                      # 与规划对齐：25 ms 窗 / 10 ms 步
MIN_DUR, MAX_DUR = 1.0, 8.0              # 全量数据放宽到 1s 起（TIMIT 句长 1–5s）
MIN_CHARS, MAX_CHARS = 10, 200
PER_SPEAKER = 5                          # 仅 --num > 0（子集模式）生效；全量时不限制
WORD_RE = re.compile(r"[a-z']+")         # 仅保留小写字母与词内撇号


def clean_text(txt_line: str) -> str:
    """TXT 行 '<start> <end> 文本' -> 小写、去标点、仅留 a-z/撇号，空格分词。"""
    parts = txt_line.strip().split(None, 2)
    text = parts[2] if len(parts) == 3 else ""
    text = unicodedata.normalize("NFKC", text).lower()
    return " ".join(WORD_RE.findall(text))


def frames_of(num_samples: int) -> int:
    return max(1, 1 + (num_samples - WIN) // HOP)


def select_items(src: Path, num: int, include_sa: bool = True):
    """按规则选取 (utt_id, wav_path, text, dur)。

    - num == 0: 全量模式，不限制每人条数；
    - num >  0: 子集模式，每人最多 PER_SPEAKER 条，凑满 num 条为止；
    - include_sa: 是否包含 SA1/SA2（全说话人重复的校准句）。
    """
    picked, speakers = [], []
    for dr in sorted(src.iterdir()):
        if not dr.is_dir():
            continue
        for spk in sorted(dr.iterdir()):
            if not spk.is_dir():
                continue
            count = 0
            # SI/SX 文本唯一；SA 按 include_sa 决定是否纳入
            sents = sorted(p.stem for p in spk.glob("*.TXT")
                           if include_sa or not p.stem.upper().startswith("SA"))
            for sent in sents:
                # 子集模式达到每人上限或总数上限则停；全量模式不限
                if num > 0 and (count >= PER_SPEAKER or len(picked) >= num):
                    break
                txt_path = spk / f"{sent}.TXT"
                # 本数据集里音频文件形如 SA1.WAV / SA1.WAV.wav（均为 NIST SPHERE）
                wav_path = next((spk / f"{sent}.WAV.wav" for v in (spk / f"{sent}.WAV",)
                                 if v.exists()), spk / f"{sent}.WAV")
                wav_path = wav_path if wav_path.exists() else spk / f"{sent}.WAV"
                if not wav_path.exists():
                    continue
                text = clean_text(txt_path.read_text(encoding="utf-8", errors="ignore"))
                if not (MIN_CHARS <= len(text) <= MAX_CHARS):
                    continue
                info = sf.info(str(wav_path))
                dur = info.frames / info.samplerate
                if not (MIN_DUR <= dur <= MAX_DUR):
                    continue
                # CTC 约束: 帧数 T >= 2L+1（L 含空格）
                T, L = frames_of(info.frames), len(text)
                if T < 2 * L + 1:
                    continue
                picked.append((f"{spk.name}_{sent}", wav_path, text, dur))
                count += 1
            if count > 0:
                speakers.append(spk.name)
            if num > 0 and len(picked) >= num:
                return picked, speakers
    return picked, speakers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="TIMIT 的 TRAIN 目录")
    ap.add_argument("--dst", required=True, help="输出目录 (data/proc)")
    ap.add_argument("--num", type=int, default=0,
                    help="选取条数；0 表示全量（默认）")
    ap.add_argument("--include_sa", action="store_true", default=True,
                    help="包含 SA1/SA2 校准句（默认包含，全量训练推荐开启）")
    ap.add_argument("--exclude_sa", dest="include_sa", action="store_false",
                    help="排除 SA1/SA2（630 人重复朗读，词汇多样性低）")
    args = ap.parse_args()

    dst = Path(args.dst)
    audio_dir = dst / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    picked, speakers = select_items(Path(args.src), args.num, args.include_sa)
    if args.num > 0 and len(picked) < args.num:
        print(f"警告: 仅选出 {len(picked)}/{args.num} 条", file=sys.stderr)

    # 写出音频与 manifest
    with open(dst / "transcription.txt", "w", encoding="utf-8") as f:
        for utt, wav_path, text, dur in picked:
            data, sr = sf.read(str(wav_path))
            assert sr == SR and data.ndim == 1, f"{utt}: sr={sr}, shape={data.shape}"
            sf.write(str(audio_dir / f"{utt}.wav"), data.astype(np.float32), SR,
                     subtype="PCM_16")
            f.write(f"{utt}.wav {text}\n")

    # 词表：blank=0，其余按字符排序（含空格）
    chars = sorted({c for _, _, t, _ in picked for c in t})
    idx2char = ["<blank>"] + chars
    vocab = {
        "blank_id": 0,
        "blank": "<blank>",
        "idx2char": idx2char,
        "num_classes": len(idx2char),
        "meta": {
            "source": "TIMIT (ModelScope Apache-2.0 mirror), TRAIN split",
            "num_utterances": len(picked),
            "num_speakers": len(speakers),
            "text_normalization": "NFKC -> lower -> 仅保留 a-z 与撇号，空格分词",
            "includes_space": " " in chars,
            "includes_sa": args.include_sa,
        },
    }
    with open(dst / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    # 同时复制一份到项目根目录，供 train.py / run_demo.py 直接加载
    root_vocab = Path(__file__).resolve().parent / "vocab.json"
    with open(root_vocab, "w", encoding="utf-8") as f:
        json.dump({"idx2char": idx2char}, f, ensure_ascii=False, indent=2)

    # 汇总报告
    durs = np.array([d for _, _, _, d in picked])
    print(f"选取 {len(picked)} 条 / {len(speakers)} 个说话人")
    print(f"时长: min {durs.min():.2f}s, max {durs.max():.2f}s, "
          f"total {durs.sum()/60:.1f}min, mean {durs.mean():.2f}s")
    print(f"词表: {vocab['num_classes']} 类 (blank + {len(chars)} 字符): {chars}")
    space = sum(1 for _, _, t, _ in picked if " " in t)
    print(f"含空格文本: {space}/{len(picked)}")
    print(f"SA 句包含: {'是' if args.include_sa else '否'}")


if __name__ == "__main__":
    main()
