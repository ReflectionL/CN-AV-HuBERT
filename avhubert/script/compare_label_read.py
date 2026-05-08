"""
对比新旧两种 label 读取方式的差异。

旧方案：文本模式打开文件，read(n) 读 n 个字符（而 offset 是字节数）→ 错位
新方案：二进制模式打开文件，read(n) 读 n 个字节，再 decode → 正确

用法：
    python avhubert/script/compare_label_read.py \
        --label_path avhubert/bpe/cnvsrc/train.wrd \
        --num_samples 5
"""

import argparse
import itertools


def load_label_offset(label_path):
    """与 hubert_dataset.py 中相同：按 UTF-8 字节数计算偏移量。"""
    with open(label_path) as f:
        lines = f.readlines()
    code_lengths = [len(line.encode("utf-8")) for line in lines]
    offsets = list(itertools.accumulate([0] + code_lengths))
    offsets = [(offsets[i], offsets[i + 1]) for i in range(len(lines))]
    return offsets, len(lines)


def read_label_old(label_path, offset_s, offset_e):
    """
    旧方案：文本模式（bug）
    - Python 3 文本模式下 seek(字节偏移) 是 undefined behavior
    - read(n) 读 n 个字符，不是字节
    """
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            f.seek(offset_s)
            raw = f.read(offset_e - offset_s)
            # 原始代码还会 .decode('utf-8')，但文本模式下 read() 返回 str，
            # 调用 .decode() 会 AttributeError，所以这里只做 .strip()
            label = raw.strip()
        return label
    except Exception as e:
        return f"<ERROR: {e}>"


def read_label_new(label_path, offset_s, offset_e):
    """
    新方案：二进制模式（正确）
    - seek(字节偏移) 精确
    - read(n) 读 n 个字节
    - decode('utf-8') 得到字符串
    """
    with open(label_path, "rb") as f:
        f.seek(offset_s)
        label = f.read(offset_e - offset_s).decode("utf-8").strip()
    return label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label_path",
        default="avhubert/bpe/cnvsrc/train.wrd",
        help="label 文件路径（.wrd）",
    )
    parser.add_argument(
        "--num_samples", type=int, default=5, help="对比前 N 条样本"
    )
    args = parser.parse_args()

    offsets, total = load_label_offset(args.label_path)
    n = min(args.num_samples, total)

    print(f"文件：{args.label_path}  |  总行数：{total}  |  对比前 {n} 条\n")
    print("=" * 80)

    all_match = True
    for i in range(n):
        offset_s, offset_e = offsets[i]
        byte_len = offset_e - offset_s

        label_new = read_label_new(args.label_path, offset_s, offset_e)
        label_old = read_label_old(args.label_path, offset_s, offset_e)

        char_len_new = len(label_new)
        char_len_old = len(label_old)
        match = label_old == label_new

        if not match:
            all_match = False

        print(f"[样本 {i}]  字节区间: [{offset_s}, {offset_e})  字节数={byte_len}")
        print(f"  新方案(rb) len={char_len_new:4d}  →  {label_new[:60]!r}")
        print(f"  旧方案(r)  len={char_len_old:4d}  →  {label_old[:60]!r}")
        if not match:
            # 找出第一个不同的字符位置
            for pos, (c_new, c_old) in enumerate(
                zip(label_new, label_old)
            ):
                if c_new != c_old:
                    print(
                        f"  *** 首个不同位置: pos={pos}  "
                        f"新='{c_new}'(U+{ord(c_new):04X})  "
                        f"旧='{c_old}'(U+{ord(c_old):04X})"
                    )
                    break
            if char_len_old > char_len_new:
                extra = label_old[char_len_new:]
                print(
                    f"  *** 旧方案多读了 {char_len_old - char_len_new} 个字符: "
                    f"{extra[:40]!r}  ← 这些来自下一行！"
                )
        else:
            print(f"  ✓ 两者一致")
        print()

    print("=" * 80)
    if all_match:
        print("结论：所有样本两种方案结果一致（当前环境下文本模式 seek 碰巧没有出错）")
        print("      但这不代表 bug 不存在，在不同平台/Python 版本上行为可能不同。")
    else:
        print("结论：两种方案结果不一致，旧方案存在明显 bug！")
        print("      旧方案读取了错误数量的字符，导致 label 混入相邻样本内容。")


if __name__ == "__main__":
    main()
