# 5层LSTM + CTC 语音识别模型

基于 PyTorch 实现的端到端语音识别系统，使用 **5层双向LSTM** 作为声学模型，**CTC (Connectionist Temporal Classification)** 作为损失函数。

## 项目结构

```
├── model.py          # 模型定义 (DeepSpeech2 / DeepSpeech2Small)
├── dataset.py        # 数据加载、梅尔频谱提取、词表构建、说话人切分、SpecAugment
├── train.py          # 训练脚本 (CTC损失 + warmup + 混合精度 + 验证CER选模型/早停)
├── inference.py      # 推理脚本 (贪心解码 / 束搜索，可选LM浅融合)
├── lm.py             # 4-gram 词级语言模型 (stupid backoff，仅train切分语料)
├── cer.py            # CER字错率评估
├── run_demo.py       # 端到端演示 (合成数据 → 训练 → 推理 → 评估)
├── lm/               # 训练好的语言模型 (word_4gram.json)
├── evals/            # 各阶段评测结果归档 (JSON)
├── docs/             # 汇报材料 + 优化实验总结.md
└── requirements.txt  # 依赖
```

## 模型架构

```
输入音频 → 梅尔频谱 [B, T, 80]
    ↓
Linear + ReLU + Dropout
    ↓
┌─────────────────────────┐
│  LSTM Layer 1 (BiLSTM)  │
│  LSTM Layer 2 (BiLSTM)  │
│  LSTM Layer 3 (BiLSTM)  │
│  LSTM Layer 4 (BiLSTM)  │
│  LSTM Layer 5 (BiLSTM)  │
└─────────────────────────┘
    ↓
Linear + ReLU + Dropout + Linear
    ↓
Log Softmax [B, T, num_classes]
    ↓
CTC Loss (训练) / CTC Decode (推理)
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行演示

```bash
python run_demo.py
```

该脚本会：生成合成数据 → 训练模型 → 推理测试 → 计算CER。

### 3. 数据格式

在 `data_dir/` 下准备：
```
data_dir/
├── audio/
│   ├── 001.wav
│   ├── 002.wav
│   └── ...
└── transcription.txt    # 每行: "音频文件名 文本标注"
```

`transcription.txt` 示例：
```
001.wav 今天天气真好
002.wav 我想去公园散步
```

## 训练

```bash
python train.py \
    --train_dir ./train_data \
    --batch_size 16 \
    --epochs 100 \
    --hidden_dim 512 \
    --num_layers 5 \
    --lr 1e-3 \
    --dropout 0.3 \
    --save_dir ./checkpoints
```

未指定 `--val_dir` 时，自动从训练集按**说话人**切出 10% 作验证集（同一说话人的句子不会同时出现在两侧）。训练集默认启用 SpecAugment 数据增强，最佳模型按验证 CER 保存。

### 关键参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--train_dir` | 训练数据目录 | 必填 |
| `--val_dir` | 验证数据目录 | None（自动按说话人切分） |
| `--dev_ratio` | 从训练集切出的验证说话人比例 | 0.1 |
| `--dev_seed` | 说话人切分随机种子 | 42 |
| `--no_specaug` | 禁用训练集 SpecAugment 增强 | 关（默认启用） |
| `--patience` | 验证 CER 连续 N 个 epoch 无改善则早停 | 10 |
| `--batch_size` | 批次大小 | 16 |
| `--epochs` | 训练轮数 | 100 |
| `--hidden_dim` | LSTM隐藏维度 | 512 |
| `--num_layers` | LSTM层数 | 5 |
| `--lr` | 初始学习率 | 1e-3 |
| `--weight_decay` | AdamW 解耦权重衰减 | 1e-4 |
| `--dropout` | Dropout比率 | 0.3 |
| `--grad_clip` | 梯度裁剪阈值 | 5.0 |
| `--use_amp` | 混合精度训练 | True |

## 推理

```bash
# 贪心解码 (最快)
python inference.py --model checkpoints/best_model.pt --audio test.wav --beam_width 1

# 束搜索解码 (更准确)
python inference.py --model checkpoints/best_model.pt --audio test.wav --beam_width 10
```

### Python API

```python
import soundfile as sf
from inference import SpeechRecognizer

recognizer = SpeechRecognizer("checkpoints/best_model.pt", beam_width=10)
audio, sr = sf.read("test.wav", dtype="float32")

text, candidates = recognizer.recognize(audio)
print(f"识别结果: {text}")
```

### 解码策略

- **贪心解码** (`beam_width=1`): 每帧取最大概率字符，合并重复并去blank
- **束搜索** (`beam_width>1`): CTC 前缀束搜索（Hannun et al., 2014），维护 top-K 条路径并正确合并相同前缀的不同对齐路径；无语言模型时与贪心基本持平

## CER 字错率

```
CER = (替换数 + 删除数 + 插入数) / 参考文本总字符数
```

### 单条计算

```bash
python cer.py single --ref "今天天气真好" --hyp "今天天气很好" --detailed
```

输出：
```
距离: 1
参考: 今天天气真好
识别: 今天天气很好
CER: 0.1667 (16.67%)
```

### 批量评估

```bash
# 从JSON批量计算
python cer.py batch --input data.json --output result.json
```

输入JSON格式：
```json
[
  {"ref": "今天天气好", "hyp": "今天天气不错"},
  {"ref": "你好世界", "hyp": "你好世界"}
]
```

### 端到端评估

```bash
python cer.py eval --model checkpoints/best_model.pt --test_dir ./test_data --beam_width 10
```

## 配置文件说明

| 文件 | 说明 |
|------|------|
| `model.py` | `DeepSpeech2`: 5层双向LSTM (推荐) / `DeepSpeech2Small`: 5层单向LSTM (轻量) |
| `dataset.py` | `SpeechDataset`: 数据加载 / `Vocabulary`: 词表管理 / `extract_mel_spectrogram`: 特征提取 |
| `train.py` | `CTCLossWrapper`: CTC损失包装 / `WarmupScheduler`: 学习率预热 |
| `inference.py` | `SpeechRecognizer`: 统一推理接口 / 贪心/束搜索 |
| `cer.py` | `compute_cer`: 单条CER / `compute_cer_corpus`: 批量CER / `evaluate_model`: 端到端评估 |
