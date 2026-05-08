#!/usr/bin/env python3
"""
从 train.tsv 和 train.wrd 中随机挑选总时长约为1小时和10小时的子集。
"""

import random
import os
import argparse

AUDIO_SAMPLE_RATE = 16000  # Hz
TARGET_HOURS = [1, 10]


def parse_args():
    parser = argparse.ArgumentParser(description="按时长采样 TSV/WRD 子集")
    parser.add_argument(
        "--tsv", default="avhubert/bpe/cnvsrc/train.tsv", help="输入 TSV 文件路径"
    )
    parser.add_argument(
        "--wrd", default="avhubert/bpe/cnvsrc/train.wrd", help="输入 WRD 文件路径"
    )
    parser.add_argument(
        "--output_dir",
        default="avhubert/bpe/cnvsrc",
        help="输出目录（子集文件将存放在此）",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def load_data(tsv_path, wrd_path):
    with open(tsv_path, "r") as f:
        lines = f.readlines()

    root = lines[0].strip()
    data_lines = [l.rstrip("\n") for l in lines[1:]]

    with open(wrd_path, "r") as f:
        wrd_lines = [l.rstrip("\n") for l in f.readlines()]

    assert len(data_lines) == len(wrd_lines), (
        f"TSV 数据行数 ({len(data_lines)}) 与 WRD 行数 ({len(wrd_lines)}) 不一致"
    )

    samples = []
    for tsv_line, wrd_line in zip(data_lines, wrd_lines):
        parts = tsv_line.split("\t")
        audio_samples = int(parts[4])
        duration_sec = audio_samples / AUDIO_SAMPLE_RATE
        samples.append({
            "tsv": tsv_line,
            "wrd": wrd_line,
            "duration": duration_sec,
        })

    return root, samples


def sample_subset(samples, target_seconds, seed):
    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)

    selected = []
    total = 0.0
    for s in shuffled:
        if total >= target_seconds:
            break
        selected.append(s)
        total += s["duration"]

    return selected, total


def write_subset(root, selected, tsv_out, wrd_out):
    os.makedirs(os.path.dirname(tsv_out), exist_ok=True)
    with open(tsv_out, "w") as ft, open(wrd_out, "w") as fw:
        ft.write(root + "\n")
        for s in selected:
            ft.write(s["tsv"] + "\n")
            fw.write(s["wrd"] + "\n")


def main():
    args = parse_args()

    print(f"读取数据：{args.tsv}")
    root, samples = load_data(args.tsv, args.wrd)

    total_duration = sum(s["duration"] for s in samples)
    print(f"共 {len(samples)} 条样本，总时长 {total_duration/3600:.2f} 小时")

    for hours in TARGET_HOURS:
        target_sec = hours * 3600
        selected, actual_sec = sample_subset(samples, target_sec, seed=args.seed)

        tag = f"{hours}h"
        tsv_out = os.path.join(args.output_dir, f"train_{tag}.tsv")
        wrd_out = os.path.join(args.output_dir, f"train_{tag}.wrd")

        write_subset(root, selected, tsv_out, wrd_out)

        print(
            f"[{tag}] 选取 {len(selected)} 条样本，"
            f"实际时长 {actual_sec/3600:.3f} 小时 → {tsv_out}"
        )


if __name__ == "__main__":
    main()
