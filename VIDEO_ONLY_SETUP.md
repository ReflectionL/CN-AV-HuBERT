# AV-HuBERT Video-Only 训练配置

## 概述
对于Video-only训练，你不需要准备实际的音频文件，但仍需要在manifest文件中提供音频路径字段（使用dummy值）。

## 主要区别

### 1. 配置文件
确保设置 `modalities: ["video"]`：
- 在yaml配置文件中已默认设置
- 或在运行脚本中覆盖：`task.modalities=["video"]`

### 2. Manifest文件格式
虽然不使用音频，但manifest格式保持不变（为了兼容性）：
```
/path/to/video/root
sample_001	videos/sample_001.mp4	dummy/sample_001.wav	150	48000
sample_002	videos/sample_002.mp4	dummy/sample_002.wav	200	64000
```

注意：
- `audio_path` 字段使用dummy路径（不会实际读取）
- `num_samples` 基于视频时长估算（duration * 16000）

### 3. 数据准备

#### 准备JSON数据文件
创建 `video_data.json`：
```json
[
    {
        "id": "sample_001",
        "video": "videos/sample_001.mp4",
        "text": "你好世界"
    },
    {
        "id": "sample_002",
        "video": "videos/sample_002.mp4",
        "text": "这是一个测试"
    }
]
```

#### 运行数据准备脚本
```bash
# 准备训练数据
python prepare_video_only_data.py \
    --data-json video_train.json \
    --output-dir /home/lixiaolou/code/CN-AV-HuBERT/avhubert/bpe/ \
    --data-root /path/to/your/video/root/ \
    --split train

# 准备验证数据
python prepare_video_only_data.py \
    --data-json video_valid.json \
    --output-dir /home/lixiaolou/code/CN-AV-HuBERT/avhubert/bpe/ \
    --data-root /path/to/your/video/root/ \
    --split valid
```

## 运行训练

使用已配置好的脚本：
```bash
cd /home/lixiaolou/code/CN-AV-HuBERT
bash run_config/finetune/cncvs3.sh
```

脚本已包含关键配置：
- `task.modalities=["video"]` - 只使用视频模态
- `task.tokenizer_bpe_name=char_tokenizer` - 使用字符级tokenizer
- `task.char_dict_path` - 指向字符词典

## 视频要求

1. **格式**：支持mp4, avi等常见格式
2. **内容**：需要包含人脸（系统会自动检测和裁剪ROI）
3. **帧率**：建议25fps或以上
4. **分辨率**：至少包含清晰的人脸区域（88x88像素以上）

## 性能考虑

Video-only训练的优势：
- 不需要音频数据和处理
- 训练速度更快
- 适用于只有视频标注的数据集

潜在限制：
- 准确率可能低于音视频联合训练
- 对光照、角度等视觉因素更敏感

## 验证设置

运行以下Python代码验证配置：
```python
import os

# 检查文件
files_to_check = [
    "/home/lixiaolou/code/CN-AV-HuBERT/avhubert/bpe/train.tsv",
    "/home/lixiaolou/code/CN-AV-HuBERT/avhubert/bpe/train.wrd",
    "/home/lixiaolou/code/CN-AV-HuBERT/avhubert/bpe/valid.tsv",
    "/home/lixiaolou/code/CN-AV-HuBERT/avhubert/bpe/valid.wrd",
    "/home/lixiaolou/code/CN-AV-HuBERT/avhubert/bpe/dict.wrd.txt",
    "/home/lixiaolou/code/CN-AV-HuBERT/avhubert/bpe/char_units.txt"
]

for f in files_to_check:
    if os.path.exists(f):
        print(f"✓ {f}")
    else:
        print(f"✗ {f} - 需要创建")

# 验证manifest和label对齐
def check_alignment(tsv_file, wrd_file):
    with open(tsv_file) as f:
        tsv_lines = len(f.readlines()) - 1  # 减去header
    with open(wrd_file) as f:
        wrd_lines = len(f.readlines())
    
    if tsv_lines == wrd_lines:
        print(f"✓ {tsv_file} 和 {wrd_file} 对齐 ({tsv_lines} samples)")
    else:
        print(f"✗ 不对齐: {tsv_file}({tsv_lines}) vs {wrd_file}({wrd_lines})")

if os.path.exists("/home/lixiaolou/code/CN-AV-HuBERT/avhubert/bpe/train.tsv"):
    check_alignment(
        "/home/lixiaolou/code/CN-AV-HuBERT/avhubert/bpe/train.tsv",
        "/home/lixiaolou/code/CN-AV-HuBERT/avhubert/bpe/train.wrd"
    )
```

## 常见问题

**Q: 为什么manifest文件还需要音频路径？**
A: 为了保持代码兼容性。Video-only模式下不会实际读取音频文件。

**Q: num_samples字段如何设置？**
A: 基于视频时长估算：`duration_in_seconds * 16000`

**Q: 可以混合使用有音频和无音频的数据吗？**
A: 不建议。应该统一使用video-only或audiovisual模式。

**Q: 如何确认是否在video-only模式？**
A: 训练日志会显示 `modalities: ['video']`
