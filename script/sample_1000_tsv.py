import random
import argparse

def split_tsv(input_tsv, output_sample_tsv, output_remain_tsv, n_samples=1000, random_seed=42):
    """
    从TSV文件中随机采样数据，并将数据分成两部分保存

    Args:
        input_tsv (str): 输入TSV文件路径
        output_sample_tsv (str): 采样数据输出路径
        output_remain_tsv (str): 剩余数据输出路径
        n_samples (int): 需要采样的数据条数
        random_seed (int): 随机种子
    """
    random.seed(random_seed)

    try:
        # 读取所有行
        with open(input_tsv, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 获取header和内容
        header = lines[0]
        content_lines = lines[1:]

        # 检查数据条数
        total_rows = len(content_lines)
        if total_rows < n_samples:
            print(f"警告: 输入文件只有 {total_rows} 条数据，少于请求的 {n_samples} 条")
            n_samples = total_rows

        # 生成所有索引并打乱
        all_indices = list(range(total_rows))
        random.shuffle(all_indices)

        # 分割索引
        sampled_indices = all_indices[:n_samples]
        remaining_indices = all_indices[n_samples:]

        # 获取采样和剩余的行
        sampled_lines = [content_lines[i] for i in sampled_indices]
        remaining_lines = [content_lines[i] for i in remaining_indices]

        # 保存采样数据
        with open(output_sample_tsv, 'w', encoding='utf-8') as f:
            f.write(header)
            for line in sampled_lines:
                f.write(line)

        # 保存剩余数据
        with open(output_remain_tsv, 'w', encoding='utf-8') as f:
            f.write(header)
            for line in remaining_lines:
                f.write(line)

        print(f"成功采样 {len(sampled_lines)} 条数据并保存到 {output_sample_tsv}")
        print(f"剩余 {len(remaining_lines)} 条数据保存到 {output_remain_tsv}")

    except Exception as e:
        print(f"处理文件时出错: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='从TSV文件中随机采样数据并分割')
    parser.add_argument('--input_tsv', type=str, required=True,
                        help='输入TSV文件路径')
    parser.add_argument('--output_sample_tsv', type=str, required=True,
                        help='采样数据输出TSV文件路径')
    parser.add_argument('--output_remain_tsv', type=str, required=True,
                        help='剩余数据输出TSV文件路径')
    parser.add_argument('--n_samples', type=int, default=1000,
                        help='需要采样的数据条数（默认：1000）')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='随机种子（默认：42）')

    args = parser.parse_args()

    split_tsv(args.input_tsv,
             args.output_sample_tsv,
             args.output_remain_tsv,
             args.n_samples,
             args.random_seed)

if __name__ == '__main__':
    main()
