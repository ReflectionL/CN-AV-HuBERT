#!/usr/bin/env python3
"""
将CNVSRC CSV格式数据转换为AV-HuBERT格式
"""

import os
import csv
import json
import argparse
from pathlib import Path
import cv2

def load_char_dict(dict_path):
    """加载字符词典，返回ID到字符的映射"""
    id_to_char = {}
    with open(dict_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                char = parts[0]
                char_id = parts[1]
                id_to_char[char_id] = char
    return id_to_char

def ids_to_text(token_ids_str, id_to_char):
    """将token ID序列转换为文本"""
    token_ids = token_ids_str.strip().split()
    chars = []
    for tid in token_ids:
        if tid in id_to_char:
            char = id_to_char[tid]
            if char not in ['<unk>', '<blank>', '<eos>']:
                chars.append(char)
    return ''.join(chars)

def text_to_char_sequence(text):
    """将文本转换为空格分隔的字符序列（用于label文件）"""
    return ' '.join(list(text))

def get_video_info(video_path):
    """获取视频信息"""
    if not os.path.exists(video_path):
        return None, None, None
    
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        return frame_count, fps, duration
    except Exception as e:
        print(f"Error reading video {video_path}: {e}")
        return None, None, None

def convert_cnvsrc_to_avhubert(
    csv_file,
    output_dir,
    data_root,
    char_dict_path,
    split_name='train',
    max_samples=None
):
    """
    转换CNVSRC CSV格式到AV-HuBERT格式
    
    CSV格式: dataset,video_path,frame_count,token_ids
    """
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载字符词典
    id_to_char = load_char_dict(char_dict_path)
    print(f"Loaded {len(id_to_char)} characters from dictionary")
    
    # 读取CSV文件
    samples = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if max_samples and i >= max_samples:
                break
            
            if len(row) < 4:
                print(f"Warning: Skipping invalid row {i}: {row}")
                continue
            
            dataset_name = row[0]
            video_rel_path = row[1]  # e.g., "dev/video/00023065.mp4"
            frame_count_csv = int(row[2])
            token_ids = row[3]
            
            # 构建完整视频路径
            video_full_path = os.path.join(data_root, dataset_name, video_rel_path)
            
            # 获取视频信息
            num_frames, fps, duration = get_video_info(video_full_path)
            
            if num_frames is None:
                print(f"Warning: Cannot read video {video_full_path}")
                continue
            
            # 转换token IDs为文本
            text = ids_to_text(token_ids, id_to_char)
            if not text:
                print(f"Warning: Empty text for {video_rel_path}")
                continue
            
            # 生成样本ID（从文件名提取）
            sample_id = Path(video_rel_path).stem
            
            samples.append({
                'id': sample_id,
                'video_path': os.path.join(dataset_name, video_rel_path),
                'num_frames': num_frames,
                'duration': duration,
                'text': text,
                'char_sequence': text_to_char_sequence(text)
            })
    
    print(f"Loaded {len(samples)} valid samples from CSV")
    
    # 写入manifest文件
    manifest_path = output_dir / f"{split_name}.tsv"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        # 写入根目录
        f.write(data_root + '\n')
        
        # 写入每个样本
        for sample in samples:
            # 对于video-only，使用dummy音频路径
            dummy_audio = f"dummy/{sample['id']}.wav"
            # 估算音频采样数（16kHz）
            num_audio_samples = int(sample['duration'] * 16000) if sample['duration'] > 0 else sample['num_frames'] * 640
            
            f.write(f"{sample['id']}\t{sample['video_path']}\t{dummy_audio}\t"
                   f"{sample['num_frames']}\t{num_audio_samples}\n")
    
    # 写入label文件
    label_path = output_dir / f"{split_name}.wrd"
    with open(label_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(sample['char_sequence'] + '\n')
    
    # 打印统计
    print(f"\n✓ Created {manifest_path}")
    print(f"✓ Created {label_path}")
    print(f"✓ Converted {len(samples)} samples")
    
    if samples:
        # 显示一些样本
        print("\nSample examples:")
        for i, sample in enumerate(samples[:3]):
            text_preview = sample['text'][:30] + "..." if len(sample['text']) > 30 else sample['text']
            print(f"  {i+1}. {sample['id']}: {text_preview}")
    
    return samples

def main():
    parser = argparse.ArgumentParser(description='Convert CNVSRC CSV to AV-HuBERT format')
    
    # 必需参数
    parser.add_argument('--train-csv', required=True, help='Path to training CSV file')
    parser.add_argument('--valid-csv', required=True, help='Path to validation CSV file')
    parser.add_argument('--data-root', required=True, 
                       help='Root directory containing video files (e.g., /path/to/CNVSRC2024Baseline/data)')
    
    # 可选参数
    parser.add_argument('--output-dir', default='/home/lixiaolou/code/CN-AV-HuBERT/avhubert/bpe/',
                       help='Output directory for manifest and label files')
    parser.add_argument('--char-dict', default='/home/lixiaolou/code/CN-AV-HuBERT/avhubert/bpe/char_units.txt',
                       help='Path to character dictionary')
    parser.add_argument('--max-train-samples', type=int, default=None,
                       help='Maximum number of training samples (for testing)')
    parser.add_argument('--max-valid-samples', type=int, default=None,
                       help='Maximum number of validation samples (for testing)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Converting CNVSRC data to AV-HuBERT format")
    print("=" * 60)
    
    # 转换训练集
    print("\n[1/2] Converting training set...")
    train_samples = convert_cnvsrc_to_avhubert(
        args.train_csv,
        args.output_dir,
        args.data_root,
        args.char_dict,
        split_name='train',
        max_samples=args.max_train_samples
    )
    
    # 转换验证集
    print("\n[2/2] Converting validation set...")
    valid_samples = convert_cnvsrc_to_avhubert(
        args.valid_csv,
        args.output_dir,
        args.data_root,
        args.char_dict,
        split_name='valid',
        max_samples=args.max_valid_samples
    )
    
    # 总结
    print("\n" + "=" * 60)
    print("Conversion completed!")
    print(f"Training samples: {len(train_samples)}")
    print(f"Validation samples: {len(valid_samples)}")
    print(f"Output directory: {args.output_dir}")
    print("\nNext steps:")
    print("1. Verify the generated files:")
    print(f"   - {args.output_dir}/train.tsv")
    print(f"   - {args.output_dir}/train.wrd")
    print(f"   - {args.output_dir}/valid.tsv")
    print(f"   - {args.output_dir}/valid.wrd")
    print("2. Run training with: bash run_config/finetune/cncvs3.sh")

if __name__ == '__main__':
    main()
