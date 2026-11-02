"""
CER (Character Error Rate) 字错率计算代码

字错率 = (替换数 + 删除数 + 插入数) / 参考文本总字符数

支持:
- 标准编辑距离 (Levenshtein)
- 批量评估: 在整个测试集上计算平均CER
- 与推理代码集成: 端到端评估
"""
import torch
import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional


# ============================================================
# 编辑距离 (Levenshtein Distance)
# ============================================================
def levenshtein_distance(ref: str, hyp: str) -> Tuple[int, int, int, int]:
    """
    计算编辑距离，返回 (距离, 替换, 删除, 插入)

    Args:
        ref: 参考文本 (ground truth)
        hyp: 识别文本 (hypothesis)

    Returns:
        distance: 编辑距离
        substitutions: 替换数
        deletions: 删除数
        insertions: 插入数
    """
    ref_chars = list(ref)
    hyp_chars = list(hyp)

    len_ref = len(ref_chars)
    len_hyp = len(hyp_chars)

    # dp[i][j] = ref前i个字符与hyp前j个字符的编辑距离
    dp = np.zeros((len_ref + 1, len_hyp + 1), dtype=np.int32)

    for i in range(len_ref + 1):
        dp[i, 0] = i
    for j in range(len_hyp + 1):
        dp[0, j] = j

    for i in range(1, len_ref + 1):
        for j in range(1, len_hyp + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                dp[i, j] = dp[i - 1, j - 1]
            else:
                dp[i, j] = min(
                    dp[i - 1, j] + 1,      # 删除
                    dp[i, j - 1] + 1,      # 插入
                    dp[i - 1, j - 1] + 1,  # 替换
                )

    # 回溯统计各类错误
    i, j = len_ref, len_hyp
    substitutions = 0
    deletions = 0
    insertions = 0

    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_chars[i - 1] == hyp_chars[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i, j] == dp[i - 1, j - 1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i, j] == dp[i - 1, j] + 1:
            deletions += 1
            i -= 1
        elif j > 0 and dp[i, j] == dp[i, j - 1] + 1:
            insertions += 1
            j -= 1

    distance = int(dp[len_ref, len_hyp])
    return distance, substitutions, deletions, insertions


def compute_cer(
    ref: str,
    hyp: str,
    remove_space: bool = False,
    normalize: bool = True,
) -> float:
    """
    计算单条样本的字错率

    CER = (S + D + I) / N
    S = 替换数, D = 删除数, I = 插入数, N = 参考文本字符数

    Args:
        ref: 参考文本
        hyp: 识别文本
        remove_space: 是否去除空格
        normalize: 是否对中文文本做规范化

    Returns:
        cer: 字错率 (0.0 ~ 无穷大, 越小越好)
    """
    if remove_space:
        ref = ref.replace(" ", "").replace("\t", "")
        hyp = hyp.replace(" ", "").replace("\t", "")

    if normalize:
        ref = _normalize_text(ref)
        hyp = _normalize_text(hyp)

    if len(ref) == 0:
        return float(len(hyp)) if len(hyp) > 0 else 0.0

    distance, s, d, i = levenshtein_distance(ref, hyp)
    cer = distance / len(ref)
    return cer


def compute_cer_detailed(
    ref: str,
    hyp: str,
    remove_space: bool = False,
) -> Dict:
    """
    计算单条样本的详细错误信息

    Returns:
        dict with: cer, distance, substitutions, deletions, insertions, ref_len
    """
    if remove_space:
        ref = ref.replace(" ", "").replace("\t", "")
        hyp = hyp.replace(" ", "").replace("\t", "")

    ref = _normalize_text(ref)
    hyp = _normalize_text(hyp)

    distance, s, d, i = levenshtein_distance(ref, hyp)
    ref_len = len(ref)

    return {
        "cer": distance / ref_len if ref_len > 0 else (len(hyp) if len(hyp) > 0 else 0.0),
        "distance": distance,
        "substitutions": s,
        "deletions": d,
        "insertions": i,
        "ref_len": ref_len,
        "hyp_len": len(hyp),
        "ref": ref,
        "hyp": hyp,
    }


def compute_cer_corpus(
    refs: List[str],
    hyps: List[str],
    remove_space: bool = False,
) -> Dict:
    """
    在整个语料库(测试集)上计算CER

    使用累计方式: 累加所有错误后统一计算
    CER = Σ(S + D + I) / Σ N

    Args:
        refs: 参考文本列表
        hyps: 识别文本列表
        remove_space: 是否去除空格

    Returns:
        dict with: cer, total_distance, total_substitutions,
                   total_deletions, total_insertions, total_ref_len, per_sample_cers
    """
    assert len(refs) == len(hyps), "参考文本和识别文本数量必须相同"

    total_distance = 0
    total_substitutions = 0
    total_deletions = 0
    total_insertions = 0
    total_ref_len = 0
    per_sample_cers = []

    for ref, hyp in zip(refs, hyps):
        detail = compute_cer_detailed(ref, hyp, remove_space)

        total_distance += detail["distance"]
        total_substitutions += detail["substitutions"]
        total_deletions += detail["deletions"]
        total_insertions += detail["insertions"]
        total_ref_len += detail["ref_len"]
        per_sample_cers.append(detail["cer"])

    return {
        "cer": total_distance / total_ref_len if total_ref_len > 0 else float("inf"),
        "total_distance": total_distance,
        "total_substitutions": total_substitutions,
        "total_deletions": total_deletions,
        "total_insertions": total_insertions,
        "total_ref_len": total_ref_len,
        "substitution_rate": total_substitutions / total_ref_len if total_ref_len > 0 else 0,
        "deletion_rate": total_deletions / total_ref_len if total_ref_len > 0 else 0,
        "insertion_rate": total_insertions / total_ref_len if total_ref_len > 0 else 0,
        "per_sample_cers": per_sample_cers,
        "num_samples": len(refs),
    }


# ============================================================
# 文本规范化
# ============================================================
def _normalize_text(text: str) -> str:
    """
    文本规范化: 去除标点符号、统一全角半角等
    """
    import unicodedata
    import re

    # 统一全角数字和字母为半角
    text = unicodedata.normalize("NFKC", text)

    # 去除常用标点 (保留中文字符、字母、数字)
    # 可根据需要自定义保留/去除的字符
    text = re.sub(r"[，。！？、；：""''（）《》【】\-,.!?:;\'\"()\[\]{}]", "", text)
    text = text.strip()

    return text


# ============================================================
# 端到端评估脚本
# ============================================================
def evaluate_model(
    model_path: str,
    test_dir: str,
    output_path: str = None,
    beam_width: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict:
    """
    端到端评估模型CER

    使用训练好的模型对测试集进行推理并计算CER

    Args:
        model_path: 模型路径
        test_dir: 测试数据目录 (含audio/和transcription.txt)
        output_path: 评估结果保存路径
        beam_width: 束搜索宽度

    Returns:
        评估结果字典
    """
    import torch
    from inference import SpeechRecognizer
    from dataset import SpeechDataset, Vocabulary

    print(f"加载模型: {model_path}")
    recognizer = SpeechRecognizer(
        model_path=model_path,
        device=device,
        beam_width=beam_width,
    )

    # 加载测试数据
    vocab = recognizer.vocab
    test_dataset = SpeechDataset(test_dir, vocab)

    refs = []
    hyps = []

    print(f"测试样本数: {len(test_dataset)}")
    for idx in tqdm(range(len(test_dataset)), desc="推理中"):
        sample = test_dataset[idx]
        audio_path = sample["audio_path"]
        ref_text = test_dataset.samples[idx][1]

        try:
            import soundfile as sf
            audio, sr = sf.read(audio_path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != recognizer.sample_rate:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=recognizer.sample_rate)
        except Exception as e:
            print(f"加载音频失败 {audio_path}: {e}")
            continue

        hyp_text, _ = recognizer.recognize(audio, beam_width=beam_width)

        refs.append(ref_text)
        hyps.append(hyp_text)

    # 计算CER
    result = compute_cer_corpus(refs, hyps)

    print(f"\n{'='*50}")
    print(f"评估结果")
    print(f"{'='*50}")
    print(f"总样本数:     {result['num_samples']}")
    print(f"总参考字数:   {result['total_ref_len']}")
    print(f"总错误数:     {result['total_distance']}")
    print(f"  - 替换:     {result['total_substitutions']} ({result['substitution_rate']:.2%})")
    print(f"  - 删除:     {result['total_deletions']} ({result['deletion_rate']:.2%})")
    print(f"  - 插入:     {result['total_insertions']} ({result['insertion_rate']:.2%})")
    print(f"\n字错率 (CER): {result['cer']:.4f} ({result['cer']:.2%})")

    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        result_save = {k: v for k, v in result.items() if k != "per_sample_cers"}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_save, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存至: {output_path}")

    return result


# ============================================================
# 错误分析
# ============================================================
def analyze_errors(ref: str, hyp: str) -> str:
    """
    分析单个case的错误, 返回带标记的对比文本

    Args:
        ref: 参考文本
        hyp: 识别文本

    Returns:
        带有错误标记的对比报告
    """
    distance, subs, dels, ins = levenshtein_distance(ref, hyp)

    report = []
    report.append(f"参考: {ref}")
    report.append(f"识别: {hyp}")
    report.append(f"编辑距离: {distance} (替换:{subs}, 删除:{dels}, 插入:{ins})")

    # 使用difflib进行可视化对比
    try:
        import difflib
        differ = difflib.Differ()
        diff = list(differ.compare(list(ref), list(hyp)))
        report.append("逐字对比:")
        report.append("  " + " ".join(diff))
    except Exception:
        pass

    return "\n".join(report)


# ============================================================
# 命令行入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="CER字错率计算")

    subparsers = parser.add_subparsers(dest="mode", help="模式")

    # 单条计算
    single_parser = subparsers.add_parser("single", help="计算单条样本CER")
    single_parser.add_argument("--ref", type=str, required=True, help="参考文本")
    single_parser.add_argument("--hyp", type=str, required=True, help="识别文本")
    single_parser.add_argument("--remove_space", action="store_true", help="去除空格")
    single_parser.add_argument("--detailed", action="store_true", help="输出详细信息")

    # 批量计算 (从JSON文件)
    batch_parser = subparsers.add_parser("batch", help="批量计算CER")
    batch_parser.add_argument("--input", type=str, required=True, help="输入JSON文件 [{ref, hyp}, ...]")
    batch_parser.add_argument("--output", type=str, default=None, help="输出JSON文件")
    batch_parser.add_argument("--remove_space", action="store_true", help="去除空格")

    # 端到端评估
    eval_parser = subparsers.add_parser("eval", help="端到端评估模型")
    eval_parser.add_argument("--model", type=str, required=True, help="模型路径")
    eval_parser.add_argument("--test_dir", type=str, required=True, help="测试数据目录")
    eval_parser.add_argument("--output", type=str, default="eval_result.json", help="结果保存路径")
    eval_parser.add_argument("--beam_width", type=int, default=10)
    eval_parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    if args.mode == "single":
        if args.detailed:
            result = compute_cer_detailed(args.ref, args.hyp, args.remove_space)
            for k, v in result.items():
                print(f"{k}: {v}")
        else:
            cer = compute_cer(args.ref, args.hyp, args.remove_space)
            print(f"CER: {cer:.4f} ({cer:.2%})")
            print(analyze_errors(args.ref, args.hyp))

    elif args.mode == "batch":
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)

        refs = [item["ref"] for item in data]
        hyps = [item["hyp"] for item in data]
        result = compute_cer_corpus(refs, hyps, args.remove_space)

        print(f"总样本: {result['num_samples']}")
        print(f"CER: {result['cer']:.4f} ({result['cer']:.2%})")

        if args.output:
            result_save = {k: v for k, v in result.items() if k != "per_sample_cers"}
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result_save, f, ensure_ascii=False, indent=2)

    elif args.mode == "eval":
        evaluate_model(
            model_path=args.model,
            test_dir=args.test_dir,
            output_path=args.output,
            beam_width=args.beam_width,
            device=args.device,
        )

    else:
        # 交互示例
        print("CER 计算示例:\n")
        ref = "今天天气真好"
        hyp = "今天天气很好"

        cer = compute_cer(ref, hyp)
        print(analyze_errors(ref, hyp))
        print(f"\nCER: {cer:.4f} ({cer:.2%})")

        print("\n" + "="*50)
        print("使用方法:")
        print("  python cer.py single --ref '参考文本' --hyp '识别文本' --detailed")
        print("  python cer.py batch --input data.json --output result.json")
        print("  python cer.py eval --model model.pt --test_dir ./test_data")


if __name__ == "__main__":
    main()
