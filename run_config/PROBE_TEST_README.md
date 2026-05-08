# Probe Test 使用指南

## 📋 概述

本目录包含了两种 probe test 配置：
1. **CTC Probe Test** - 使用简单的线性分类器
2. **Seq2Seq Probe Test** - 使用 Transformer Decoder

两种方法都**完全冻结编码器**，只训练解码器部分，用于评估预训练模型的表示质量。

---

## 🎯 训练配置

### 1. CTC Probe Test

**配置文件**: `avhubert/conf/finetune/base_cnvsrc_ctc.yaml`
**运行脚本**: `run_config/finetune/cnvsrc_ctc.sh`

**关键参数**:
- 训练步数: 5,000 步
- 学习率: 0.001
- 编码器冻结: `freeze_finetune_updates: 999999`
- 特征梯度: `feature_grad_mult: 0.0`

**训练命令**:
```bash
cd /home/lixiaolou/code/CN-AV-HuBERT
bash run_config/finetune/cnvsrc_ctc.sh
```

**输出目录**: `/hdd2/avhubert_exp/finetune/clean_pt_cncvs3_probe_ft_cnvsrc_vsr_ctc`

---

### 2. Seq2Seq Probe Test

**配置文件**: `avhubert/conf/finetune/base_cnvsrc_s2s.yaml`
**运行脚本**: `run_config/finetune/cnvsrc_s2s.sh`

**关键参数**:
- 训练步数: 8,000 步
- 学习率: 0.001
- 编码器冻结: `freeze_finetune_updates: 999999`
- 特征梯度: `feature_grad_mult: 0.0`
- Decoder 层数: 6 层 Transformer

**训练命令**:
```bash
cd /home/lixiaolou/code/CN-AV-HuBERT
bash run_config/finetune/cnvsrc_s2s.sh
```

**输出目录**: `/hdd2/avhubert_exp/finetune/clean_pt_cncvs3_probe_ft_cnvsrc_vsr_s2s`

---

## 🔍 推理评估

### CTC 推理

**推理脚本**: `avhubert/infer_ctc.py`
**运行脚本**: `run_config/infer/infer_ctc_probe.sh`

**推理命令**:
```bash
cd /home/lixiaolou/code/CN-AV-HuBERT
bash run_config/infer/infer_ctc_probe.sh
```

**修改测试集**:
编辑 `infer_ctc_probe.sh` 中的 `subset` 变量：
```bash
subset=test  # 或 valid
```

---

### Seq2Seq 推理

**推理脚本**: `avhubert/infer_s2s_probe.py`
**运行脚本**: `run_config/infer/infer_s2s_probe.sh`

**推理命令**:
```bash
cd /home/lixiaolou/code/CN-AV-HuBERT
bash run_config/infer/infer_s2s_probe.sh
```

**修改解码参数**:
编辑 `infer_s2s_probe.sh`：
```bash
generation.beam=50      # Beam size
generation.lenpen=1.0   # Length penalty
```

---

## 📊 结果文件

推理完成后，会在结果目录生成以下文件：

1. **decode.log** - 详细的解码日志
2. **hypo-{id}.json** - 假设和参考文本（JSON 格式）
3. **wer.{id}** - WER 评分结果

### 示例结果位置：

**CTC**:
```
/hdd2/avhubert_exp/finetune/clean_pt_cncvs3_probe_ft_cnvsrc_vsr_ctc/decode_test_ctc/
```

**Seq2Seq**:
```
/hdd2/avhubert_exp/finetune/clean_pt_cncvs3_probe_ft_cnvsrc_vsr_s2s/decode_test_s2s/
```

---

## 🔧 关键差异对比

| 特性 | CTC Probe | Seq2Seq Probe |
|------|-----------|---------------|
| **解码器** | 线性层 | 6 层 Transformer |
| **训练速度** | 快 | 中等 |
| **训练步数** | 5,000 | 8,000 |
| **解码方式** | Greedy CTC | Beam Search |
| **预期性能** | 基线 | 高 20-30% |
| **适用场景** | 快速验证表示 | 全面评估 |

---

## 💡 使用建议

1. **首先运行 CTC Probe Test**:
   - 快速验证预训练模型的基础表示质量
   - 训练时间短，适合快速迭代

2. **然后运行 Seq2Seq Probe Test**:
   - 更全面地评估实际应用性能
   - 接近真实下游任务的设置

3. **对比两者的性能差异**:
   - 如果 Seq2Seq 显著优于 CTC，说明预训练表示需要更强的解码器来利用
   - 如果两者性能接近，说明预训练表示已经非常好

---

## 📝 注意事项

1. **编码器冻结**: 两种配置都完全冻结编码器，这是 probe test 的核心
2. **数据路径**: 确保 `avhubert/bpe/cnvsrc/` 目录下有正确的数据
3. **显存要求**: Seq2Seq 需要更多显存（有 6 层 decoder）
4. **预训练模型**: 确保预训练模型路径正确：
   ```
   /hdd2/avhubert_exp/pretrain/clean_pt_cncvs3_speech_iter2/checkpoints/checkpoint_best.pt
   ```

---

## 🚀 完整流程示例

```bash
# 1. 训练 CTC Probe
bash run_config/finetune/cnvsrc_ctc.sh

# 2. CTC 推理
bash run_config/infer/infer_ctc_probe.sh

# 3. 训练 Seq2Seq Probe
bash run_config/finetune/cnvsrc_s2s.sh

# 4. Seq2Seq 推理
bash run_config/infer/infer_s2s_probe.sh

# 5. 对比结果
cat /hdd2/avhubert_exp/finetune/clean_pt_cncvs3_probe_ft_cnvsrc_vsr_ctc/decode_test_ctc/wer.*
cat /hdd2/avhubert_exp/finetune/clean_pt_cncvs3_probe_ft_cnvsrc_vsr_s2s/decode_test_s2s/wer.*
```

---

## 📧 问题排查

如果遇到问题，请检查：
1. CUDA 是否可用
2. 数据路径是否正确
3. 预训练模型是否存在
4. BPE 模型文件是否存在（Seq2Seq 需要 sentencepiece.model）

