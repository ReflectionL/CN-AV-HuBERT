import os
import logging
import soundfile as sf
from pathlib import Path
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level="INFO",
)
logger = logging.getLogger("generate_tsv")

def get_audio_info(file_path):
    """获取音频文件信息

    Args:
        file_path: 音频文件路径
    Returns:
        sample_rate: 采样率
        n_samples: 样本数
    """
    try:
        info = sf.info(file_path)
        return info.samplerate, info.frames
    except Exception as e:
        logger.error(f"Error processing audio {file_path}: {str(e)}")
        return None, None

def convert_video_path_to_audio(video_path):
    """将视频路径转换为对应的音频路径

    Args:
        video_path: 视频文件路径
    Returns:
        audio_path: 对应的音频文件路径
    """
    path = Path(video_path)
    # 替换目录名 video -> audio
    parts = list(path.parts)
    video_idx = parts.index('video')
    parts[video_idx] = 'audio'
    # 替换扩展名 .mp4 -> .wav
    new_path = Path(*parts).with_suffix('.wav')
    return str(new_path)

def generate_tsv(csv_path, output_path, root_dir):
    """根据CSV文件生成TSV文件

    Args:
        csv_path: 输入的CSV文件路径
        output_path: 输出的TSV文件路径
        root_dir: 数据根目录
    """
    logger.info(f"Processing CSV file: {csv_path}")

    # 读取CSV文件
    df = pd.read_csv(csv_path, header=None)
    total_rows = len(df)
    logger.info(f"Found {total_rows} entries in CSV file")

    # 写入TSV文件
    with open(output_path, 'w', encoding='utf-8') as f:
        # 写入根目录
        f.write(f"{root_dir}\n")

        # 处理每一行
        for idx, row in tqdm(df.iterrows(), total=total_rows):
            # 获取视频路径和帧数
            root_path, video_path = row[0:2]
            video_path = os.path.join(root_path, video_path)  # 拼接完整路径
            frames = row[2]  # 假设第3列是帧数

            # 获取对应的音频路径
            audio_path = convert_video_path_to_audio(video_path)
            full_audio_path = os.path.join(root_dir, audio_path)

            # 检查音频文件是否存在
            if not os.path.exists(full_audio_path):
                logger.warning(f"Audio file not found: {full_audio_path}")
                continue

            # 获取音频信息
            sr, n_samples = get_audio_info(full_audio_path)
            if sr is None or n_samples is None:
                continue

            # 生成文件ID
            file_id = Path(video_path).stem

            # 写入TSV文件
            # id path video_path n_frames n_samples
            f.write(f"{file_id}\t{video_path}\t{audio_path}\t{frames}\t{n_samples}\n")

    logger.info(f"Generated TSV file at {output_path}")

def validate_tsv(tsv_path):
    """验证生成的TSV文件

    Args:
        tsv_path: TSV文件路径
    """
    logger.info(f"Validating TSV file: {tsv_path}")

    with open(tsv_path, 'r', encoding='utf-8') as f:
        root = f.readline().strip()

        for i, line in enumerate(f, 1):
            parts = line.strip().split('\t')
            if len(parts) != 5:  # 确保每行都有5个字段
                logger.error(f"Invalid format at line {i}: expected 5 fields, got {len(parts)}")
                continue

            id_, video_path, audio_path, n_frames, n_samples = parts

            # 验证数值字段
            try:
                n_frames = int(n_frames)
                n_samples = int(n_samples)
            except ValueError as e:
                logger.error(f"Invalid numeric value at line {i}: {e}")

    logger.info("TSV validation completed")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", help="Input CSV file path", default="/home/lixiaolou/code/CNVSRC2024Baseline/data/cnvsrc-single/train.csv")
    parser.add_argument("--output_path", help="Output TSV file path", default="/home/lixiaolou/code/CN-AV-HuBERT/data/cnvsrc-single/train.tsv")
    parser.add_argument("--root_dir", help="Root directory of dataset", default="/home/lixiaolou/data/processed")

    args = parser.parse_args()

    generate_tsv(args.csv_path, args.output_path, args.root_dir)
    validate_tsv(args.output_path)
