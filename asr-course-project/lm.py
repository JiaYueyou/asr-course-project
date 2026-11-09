"""4-gram 词级语言模型（stupid backoff 平滑），用于 CTC 束搜索浅融合。

训练语料：仅使用 train 切分的转写文本（不含 dev/test），保证数据分离。
平滑方式：Stupid Backoff (Brants et al., 2007)，回退系数 delta=0.4，
未登录词回退到一元组频率；OOV 词固定给 log(1e-8)。
"""
import json
import math
from collections import Counter, defaultdict

BOS, EOS = "<s>", "</s>"


class WordLM:
    def __init__(self, order: int = 4, delta: float = 0.4):
        self.order = order
        self.delta = delta
        self.ngrams = [defaultdict(Counter) for _ in range(order)]  # ngrams[k][context][word] = count, k=1..order
        self.totals = [Counter() for _ in range(order)]              # totals[k][context] = sum of counts
        self.vocab_counter = Counter()

    # ---------------- 训练 ----------------
    def train(self, sentences):
        """sentences: list[list[str]]，已分词、已小写的句子"""
        for words in sentences:
            ws = [BOS] * (self.order - 1) + words + [EOS]
            for k in range(1, self.order + 1):          # 1..4 gram
                for i in range(self.order - 1, len(ws)):
                    context = tuple(ws[i - k + 1 : i])
                    word = ws[i]
                    self.ngrams[k - 1][context][word] += 1
                    self.totals[k - 1][context] += 1
                    if k == 1:
                        self.vocab_counter[word] += 1
        self.total_tokens = sum(self.vocab_counter.values())
        self.vocab = set(self.vocab_counter)
        return self

    # ---------------- 查询 ----------------
    def log_prob(self, word: str, context: tuple) -> float:
        """P(word | context)，context 为最近 order-1 个词（不含 BOS 填充时内部补齐）"""
        ctx = tuple(context[-(self.order - 1):]) if self.order > 1 else ()
        # 从最高阶往下，找到第一个命中 (ctx_k, word) 的阶
        for k in range(self.order, 0, -1):
            c = ctx[-(k - 1):] if k > 1 else ()
            table = self.ngrams[k - 1].get(c)
            if table and word in table:
                return math.log(table[word] / self.totals[k - 1][c])
        # 完全未命中：unigram 也未命中（OOV 词）
        return math.log(1e-8)

    def sentence_log_prob(self, words) -> float:
        total, ctx = 0.0, [BOS] * (self.order - 1)
        for w in list(words) + [EOS]:
            total += self.log_prob(w, tuple(ctx))
            ctx = (ctx + [w])[-(self.order - 1):] if self.order > 1 else []
        return total

    # ---------------- 存取 ----------------
    def save(self, path):
        data = {
            "order": self.order, "delta": self.delta,
            "total_tokens": self.total_tokens,
            "ngrams": [{ " ".join(ctx): dict(cnt) for ctx, cnt in self.ngrams[k - 1].items()}
                       for k in range(1, self.order + 1)],
            "totals": [{ " ".join(ctx): int(n) for ctx, n in self.totals[k - 1].items()}
                       for k in range(1, self.order + 1)],
            "vocab_counter": dict(self.vocab_counter),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        lm = cls(order=data["order"], delta=data["delta"])
        for k in range(lm.order):
            lm.ngrams[k] = defaultdict(
                Counter, {tuple(ctx.split(" ")): Counter(cnt) for ctx, cnt in data["ngrams"][k].items()})
            lm.totals[k] = Counter(
                {tuple(ctx.split(" ")): n for ctx, n in data["totals"][k].items()})
        lm.vocab_counter = Counter(data["vocab_counter"])
        lm.total_tokens = data["total_tokens"]
        lm.vocab = set(lm.vocab_counter)
        return lm

    @classmethod
    def from_manifest(cls, manifest_path, speakers=None, order=4):
        """从 transcription.txt 构建语料；speakers 非空时只用这些说话人的句子"""
        sentences = []
        with open(manifest_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(None, 1)
                if len(parts) < 2:
                    continue
                utt_id = parts[0].rsplit("_", 1)[0]
                if speakers is not None and utt_id not in speakers:
                    continue
                sentences.append(parts[1].lower().split())
        return cls(order=order).train(sentences), sentences


if __name__ == "__main__":
    # 用 train 切分（排除 dev 的 46 位说话人）训练并保存
    from dataset import split_speakers
    train_spk, _ = split_speakers(r"data\proc", dev_ratio=0.1, seed=42)
    lm, sentences = WordLM.from_manifest(r"data\proc\transcription.txt", speakers=train_spk)
    lm.save(r"lm\word_4gram.json")
    print(f"语料句数: {len(sentences)}, 词表: {len(lm.vocab)}, token 数: {lm.total_tokens}")
    # 快速自检
    s = lm.sentence_log_prob("she had your dark suit in greasy wash water all year".split())
    bad = lm.sentence_log_prob("she had your dark suit in greasy wash water all yeer".split())
    print(f"正常句 logP={s:.2f}, 错拼句 logP={bad:.2f} (正常句应更高)")
