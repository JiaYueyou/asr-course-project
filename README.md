# ASR Course Project：5 层 BiLSTM + CTC 语音识别

基于 PyTorch 的端到端英文语音识别实验。输入一段 wav，输出对应文本，并用字错率 CER 评估。数据集为 TIMIT，声学模型为 5 层双向 LSTM，训练使用 CTC 损失，支持贪婪解码与束搜索。

仓库包含两份代码：

| 目录 | 说明 |
|---|---|
| `asr/` | 教师提供的原版代码，保留用于对照 |
| `asr-course-project/` | 本项目实际实现、训练、推理与评测代码，`docs/` 下是四人分工文档 |

原版与当前实现的差异见 [`asr-course-project/docs/5_原版代码对比_改进说明.md`](asr-course-project/docs/5_原版代码对比_改进说明.md)。

---

## 一、环境准备

实验在 Windows + conda 环境 `DL-Env` 下完成，GPU 为 RTX 3060 Laptop 6GB。

```powershell
conda create -n DL-Env python=3.10 -y
conda activate DL-Env
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
cd asr-course-project
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

所有训练、推理、评测命令都在 `asr-course-project/` 目录下执行。

---

## 二、数据集处理

### 2.1 数据来源与放置位置

使用 TIMIT 英文语料。仓库**不包含**原始音频和大体积特征缓存（超出 Gitee 容量限制），需要自行获取后解压到：

```
asr-course-project/data/raw/TIMIT/raw/data/
├── TRAIN/     # 训练集，462 位说话人
└── TEST/      # 测试集，168 位说话人
```

### 2.2 转换为项目格式

`prepare_timit.py` 负责挑选句子、统一为 16 kHz 单声道、写出标准 wav 与文本清单，并生成词表。

```powershell
# 训练集：全量 4618 条
python prepare_timit.py --src data\raw\TIMIT\raw\data\TRAIN --dst data\proc

# 测试集：1680 条
python prepare_timit.py --src data\raw\TIMIT\raw\data\TEST --dst data\proc_test

# 可选：小样本冒烟，每人最多若干条，凑满 50 条
python prepare_timit.py --src data\raw\TIMIT\raw\data\TRAIN --dst data\proc --num 50
```

筛选规则：音频时长 1.0–8.0 s；清洗后文本 10–200 字符；校验 CTC 约束 `T >= 2L+1`（100 帧/秒）。文本处理为 NFKC → 小写 → 仅保留 a–z 与撇号，空格分词。

### 2.3 产物说明

```
data/proc/
├── audio/                # 4618 个 wav
├── transcription.txt     # 每行：文件名 + 文本
├── vocab.json            # 29 类：<blank> + 空格 + 撇号 + a-z
└── cache/                # 特征缓存（可自动重建）
```

实测数据规模：训练集 4618 条 / 462 人 / 236.5 分钟；测试集 1680 条 / 168 人。词表固定 29 类，`blank_id = 0`，无 `<unk>`。

### 2.4 特征缓存

`dataset.py` 会把每条音频的 80 维 log-mel 特征缓存为 `.npy`，目录名由特征参数签名哈希得到（如 `cache/mel_25da37945f/`）。全量缓存约 452 MB，命中后训练只做 `np.load`，不再重复计算频谱。

```powershell
# 默认：缺失才计算
python run_demo.py --limit 8

# 关闭缓存（排错用）
python run_demo.py --no_cache

# 强制重建
python run_demo.py --rebuild_cache
```

删除 `data/proc/cache` 后下次运行会自动重建。

---

## 三、复现实验

### 3.1 快速验证链路（几分钟）

```powershell
python run_demo.py --limit 8
```

该命令会依次完成：加载数据 → 训练小模型 → 推理 → 计算 CER，输出 `demo_checkpoints/demo_model.pt`（128 维、1,892,253 参数）。

### 3.2 正式训练（全量）

```powershell
python train.py `
  --train_dir data\proc `
  --vocab_path vocab.json `
  --batch_size 32 `
  --epochs 100 `
  --hidden_dim 512 `
  --num_layers 5 `
  --num_workers 0
```

未指定 `--val_dir` 时，自动从训练集按**说话人**切出 10% 作验证集（`--dev_ratio` 调整比例，`--dev_seed=42` 保证可复现）。同一说话人的句子不会同时出现在训练/验证两侧，避免说话人泄漏导致验证结果虚高。测试集 `data\proc_test` 只在最终评测时使用。

训练配置：AdamW，学习率 1e-3，weight decay 1e-4，warmup 1000 步后余弦退火到 1%，梯度裁剪 5.0，混合精度开启；训练集启用 SpecAugment 数据增强（2 个频率遮罩 ≤15 通道 + 2 个时间遮罩 ≤40 帧、总量 ≤时长 20%，`--no_specaug` 关闭）。输出到 `checkpoints/`：

- `best_model.pt`：**验证 CER 最低**的一份（贪心解码，与最终评测指标一致）；
- `checkpoint_epochN.pt`：每 5 个 epoch 存档；
- `final_model.pt`：最后一个 epoch；
- `history.json`：逐 epoch 训练/验证损失与验证 CER。

验证 CER 连续 10 个 epoch 无改善自动早停（`--patience` 调整，0 禁用）。

checkpoint 内除权重外还保存词表、`feature_stats`（全局 CMVN 统计量）和训练参数，推理端直接恢复，保证训练/推理输入一致。

### 3.3 单条推理

```powershell
# 贪婪解码
python inference.py --model checkpoints\best_model.pt --audio data\proc\audio\FCJF0_SI1027.wav --beam_width 1

# 束搜索
python inference.py --model checkpoints\best_model.pt --audio data\proc\audio\FCJF0_SI1027.wav --beam_width 5
```

### 3.4 测试集评测

```powershell
python cer.py eval `
  --model checkpoints\best_model.pt `
  --test_dir data\proc_test `
  --beam_width 1 `
  --output evals\eval_test.json
```

---

## 四、实验结果

| 项目 | 结果 |
|---|---|
| 正式模型参数量 | 29,982,237（hidden=512，5 层 BiLSTM，29 类输出） |
| 验证集 CER（说话人不相交，贪心） | **12.87%**（460 条，e94 最优） |
| 测试集 CER（贪心） | 23.35%（1680 条，参考字数 78,758） |
| 测试集 CER（束搜索 beam=25） | **23.16%**（当前最优提交配置） |
| 测试集 CER（LM 浅融合 beam=25） | 23.64%（负收益，见下方说明） |
| 单条样例 | `FCJF0_SI1027.wav` → `even then if she took one step forward he could catch her`，与标注逐字一致 |

验证集（dev）从训练集按说话人切出、与测试集均为未见过说话人，dev→test 的约 10.5 点差距来自 TIMIT 测试集的录音域偏移（错误均匀分布于全部 168 位说话人，最差 10 人仅贡献 8.2% 错误，见 `evals/error_analysis.json`）。

优化历程：基线（无增强、验证集误设）测试 CER 19.96% → 说话人切分验证集 + SpecAugment + wd 1e-4 后 dev CER 由 ~20% 降至 12.87%（同域大幅改善）→ 修复 CTC 前缀束搜索实现缺陷（两处：blank 分数免费携带、同字符经 blank 扩展丢失），beam10 恢复正常（测试集 23.17%，修复前为 60%+ 的垃圾输出）。

**语言模型浅融合实验（负结果，如实记录）**：自实现 4-gram 词级 LM（仅 train 切分 4158 句训练，dev 调 α/β），dev 上收益 -0.61 点（12.80%→12.19%），但测试集 +0.48 点（23.16%→23.64%）。原因：TIMIT 测试集与训练集**句子文本不相交**，测试集词型 OOV 率达 50.8%，LM 对含未登录词的正确句子施加过重惩罚；dev 与 train 同源文本（OOV≈0）掩盖了这一点。详见 `evals/lm_tuning*.json`、`evals/eval_test_lm.json`。全量重训对照实验（train+dev 合并 4618 条）测试 CER 24.38%，亦无收益（训练方差大于 10% 数据的边际价值），证据见 `checkpoints_full/history.json`、`evals/eval_test_full_data.json`。

各 epoch 测试结果已归档：

```
evals/
├── eval_result.json   # 训练集
├── eval_test.json     # 测试集
├── eval_e50.json      # 21.16%
├── eval_e75.json      # 20.40%
├── eval_e85.json      # 20.02%
└── eval_e95.json      # 19.99%
```

---

## 五、仓库内容与忽略规则

受 Gitee 单文件 100 MB、仓库容量限制，以下内容通过 `.gitignore` 排除，不入库：

- 根目录实验报告 `241411040167方耀正.docx`；
- `data/raw/`（原始 TIMIT 压缩包与解压文件）；
- `data/proc/audio/`、`data/proc_test/audio/`（全部 wav）；
- `data/**/cache/`（约 452 MB 特征缓存）；
- `checkpoints/*.pt`（23 个权重文件共约 7.3 GB，单个约 343 MB）；
- 压缩包、`__pycache__`、编辑器配置等。

仓库中保留的轻量证据：`transcription.txt`、`vocab.json`、`evals/*.json`、`history.json`、`docs/`、全部源码，以及 7.2 MB 的 `demo_checkpoints/demo_model.pt`。按第二、三节的步骤重新准备数据并训练，即可得到被忽略的音频、缓存与权重。

---

## 六、分工文档

`asr-course-project/docs/` 下按四人分工组织，每人都有一份「工作内容」和一份「演讲说明内容」：

| 分工 | 文档 |
|---|---|
| 数据处理 | `1_数据处理_工作内容.md` / `1_数据处理_演讲说明内容.md` / `1_数据处理_两分钟速讲.md` |
| 模型搭建 | `2_模型搭建_工作内容.md` / `2_模型搭建_演讲说明内容.md` / `2_模型搭建_两分钟速讲.md` |
| 模型推理 | `3_模型推理_工作内容.md` / `3_模型推理_演讲说明内容.md` / `3_模型推理_两分钟速讲.md` |
| 准确率测试 | `4_数据准确率测试_工作内容.md` / `4_数据准确率测试_演讲说明内容.md` / `4_数据准确率测试_两分钟速讲.md` |

另有原版代码对比文档 `5_原版代码对比_改进说明.md`。

---

## 七、常见问题

1. **报 `ModuleNotFoundError: torch`**：当前 Python 不在 `DL-Env` 环境，执行 `conda activate DL-Env`。
2. **`torch.load` 报找不到 `dataset`**：在 `asr-course-project/` 目录下运行，checkpoint 内含 `Vocabulary` 对象需要该模块可导入。
3. **Windows 下训练很慢**：`--num_workers` 保持 0。实测缓存命中时 `nw=0` 约 65 ms/batch，`nw=4` 约 244 ms/batch，多进程启动与序列化开销超过收益。
4. **束搜索与贪心基本持平**：`_beam_search_decode` 原实现有两处缺陷（blank 分数免费携带导致概率虚增、同字符经 blank 的扩展丢失），已按标准 CTC 前缀束搜索（Hannun et al., 2014）修复。修复后 beam=10 与贪心持平（全量测试集见 `evals/eval_test_beam.json`）。无语言模型时束搜索收益≈0（CTC 条件独立假设下贪心已近似最优），其价值在于后续接入词级语言模型（浅融合）时承载 n-best 候选。
