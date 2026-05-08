# AV-HuBERT 数据格式说明

## 概述
AV-HuBERT需要准备以下文件：
1. **Manifest文件** (`.tsv`格式) - 描述音视频文件路径
2. **Label文件** (`.wrd`格式) - 包含对应的文本标注
3. **字典文件** (`dict.wrd.txt`) - 定义词汇表

## 文件格式详解

### 1. Manifest文件格式 (`train.tsv`, `valid.tsv`, `test.tsv`)

manifest文件是一个制表符分隔的文件，格式如下：

**第一行**：数据根目录路径
```
/path/to/data/root
```

**后续行**：每行代表一个样本，包含以下字段（用制表符`\t`分隔）：
```
audio_id	video_path	audio_path	num_frames	num_samples
```

字段说明：
- `audio_id`: 样本的唯一标识符
- `video_path`: 视频文件的相对路径（相对于根目录）
- `audio_path`: 音频文件的相对路径（相对于根目录）
- `num_frames`: 视频帧数
- `num_samples`: 音频采样点数

**示例**：
```
/home/user/data
sample_001	videos/sample_001.mp4	audios/sample_001.wav	150	48000
sample_002	videos/sample_002.mp4	audios/sample_002.wav	200	64000
sample_003	videos/sample_003.mp4	audios/sample_003.wav	180	57600
```

### 2. Label文件格式 (`train.wrd`, `valid.wrd`, `test.wrd`)

Label文件包含每个样本对应的文本标注，每行对应manifest文件中的一个样本。

对于使用字符级tokenizer的情况：
- 每行是一个字符序列，字符之间用**空格**分隔
- 行数必须与manifest文件中的样本数完全一致
- 顺序必须与manifest文件中的样本顺序一一对应

**示例**（对应上面的manifest）：
```
你 好 世 界
这 是 一 个 测 试
语 音 识 别 系 统
```

### 3. 字典文件格式 (`dict.wrd.txt`)

字典文件定义了所有可能的token及其频率，格式为：
```
token frequency
```

每行包含一个token和它的频率（用空格分隔）。频率可以是任意正整数。

**示例**：
```
<unk> 1000
你 1000
好 1000
世 1000
界 1000
这 1000
是 1000
...
```

## 目录结构示例

建议的目录结构：
```
/home/user/CN-AV-HuBERT/
├── avhubert/
│   └── bpe/
│       ├── char_units.txt      # tokenizer词表
│       ├── dict.wrd.txt        # fairseq字典文件
│       ├── train.tsv           # 训练集manifest
│       ├── train.wrd           # 训练集标注
│       ├── valid.tsv           # 验证集manifest
│       ├── valid.wrd           # 验证集标注
│       ├── test.tsv            # 测试集manifest（可选）
│       └── test.wrd            # 测试集标注（可选）
└── data/
    ├── videos/                  # 视频文件
    │   ├── sample_001.mp4
    │   ├── sample_002.mp4
    │   └── ...
    └── audios/                  # 音频文件
        ├── sample_001.wav
        ├── sample_002.wav
        └── ...
```

## 数据准备脚本示例

创建一个Python脚本来准备数据：

```python
import os
import pandas as pd

def prepare_manifest_and_labels(data_list, output_prefix, data_root):
    """
    data_list: 列表，每个元素是 (audio_id, video_path, audio_path, transcript)
    output_prefix: 输出文件前缀，如 'train', 'valid'
    data_root: 数据根目录
    """
    
    # 准备manifest文件
    manifest_file = f"{output_prefix}.tsv"
    with open(manifest_file, 'w') as f:
        # 写入根目录
        f.write(data_root + '\n')
        
        # 写入每个样本
        for audio_id, video_path, audio_path, transcript in data_list:
            # 获取音频信息（需要实际读取文件）
            import wave
            with wave.open(os.path.join(data_root, audio_path), 'rb') as wav:
                num_samples = wav.getnframes()
                sample_rate = wav.getframerate()
                
            # 假设视频是25fps，计算帧数
            duration = num_samples / sample_rate
            num_frames = int(duration * 25)
            
            f.write(f"{audio_id}\t{video_path}\t{audio_path}\t{num_frames}\t{num_samples}\n")
    
    # 准备label文件
    label_file = f"{output_prefix}.wrd"
    with open(label_file, 'w') as f:
        for _, _, _, transcript in data_list:
            # 将文本转换为字符序列（空格分隔）
            chars = ' '.join(list(transcript))
            f.write(chars + '\n')
    
    print(f"Created {manifest_file} and {label_file}")

# 使用示例
train_data = [
    ("sample_001", "videos/sample_001.mp4", "audios/sample_001.wav", "你好世界"),
    ("sample_002", "videos/sample_002.mp4", "audios/sample_002.wav", "这是测试"),
    # ...
]

prepare_manifest_and_labels(train_data, "train", "/home/user/data")
```

## 注意事项

1. **数据对齐**：音频时长和标注长度需要合理对齐，否则训练时会跳过不对齐的样本
2. **采样率**：音频文件建议使用16kHz采样率
3. **视频格式**：支持mp4等常见格式，会自动提取人脸ROI区域
4. **字符编码**：所有文本文件使用UTF-8编码
5. **标注格式**：
   - 对于字符级tokenizer：每个字符用空格分隔
   - 对于BPE tokenizer：需要先进行分词处理

## 验证数据

创建验证脚本检查数据格式：

```python
def validate_data(manifest_path, label_path):
    # 读取manifest
    with open(manifest_path) as f:
        root = f.readline().strip()
        manifest_lines = f.readlines()
    
    # 读取labels
    with open(label_path) as f:
        label_lines = f.readlines()
    
    # 检查行数是否一致
    assert len(manifest_lines) == len(label_lines), \
        f"Manifest has {len(manifest_lines)} samples but label has {len(label_lines)}"
    
    print(f"✓ Data validation passed: {len(manifest_lines)} samples")
    
    # 检查几个样本
    for i in range(min(3, len(manifest_lines))):
        items = manifest_lines[i].strip().split('\t')
        label = label_lines[i].strip()
        print(f"Sample {i}: {items[0]} -> {label}")

# 使用
validate_data("train.tsv", "train.wrd")
```
