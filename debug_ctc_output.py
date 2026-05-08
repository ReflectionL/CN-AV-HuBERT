#!/usr/bin/env python3
"""
调试 CTC 输出，查看模型实际预测了什么
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'fairseq'))

import torch
import numpy as np
from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf

def debug_ctc_predictions(ckpt_path, n_samples=5):
    """查看模型的 CTC 预测"""
    
    # 设置 user_dir
    avhubert_dir = os.path.join(os.path.dirname(__file__), 'avhubert')
    utils.import_user_module(type('Args', (), {'user_dir': avhubert_dir})())
    
    print(f"加载模型: {ckpt_path}")
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
    model = models[0].eval().cuda()
    
    print(f"字典大小: {len(task.target_dictionary)}")
    print(f"字典前10个: {task.target_dictionary.symbols[:10]}")
    
    # 加载验证集
    task.load_dataset('valid')
    dataset = task.dataset('valid')
    
    print(f"\n验证集大小: {len(dataset)}")
    print(f"=" * 80)
    
    # 使用 dataloader 批量加载
    from torch.utils.data import DataLoader
    from fairseq.data import data_utils
    
    # 创建一个小的 batch iterator
    batch_sampler = [[idx] for idx in range(min(n_samples, len(dataset)))]
    
    for batch_idx, indices in enumerate(batch_sampler):
        # 获取样本并整理成 batch
        samples = [dataset[idx] for idx in indices]
        sample = dataset.collater(samples)
        
        print(f"\n样本 {batch_idx + 1}:")
        
        # 检查样本结构
        if 'net_input' in sample:
            net_input = sample['net_input']
            if 'source' in net_input:
                source = net_input['source']
                if isinstance(source, dict) and 'video' in source:
                    print(f"  视频形状: {source['video'].shape}")
                elif hasattr(source, 'shape'):
                    print(f"  源数据形状: {source.shape}")
        
        # 解码目标
        if 'target' in sample:
            target_tokens = sample['target'][0]  # 取第一个样本
            target_text = task.target_dictionary.string(target_tokens)
            print(f"  参考文本: {target_text}")
            print(f"  目标长度: {len(target_tokens)}")
        
        # 模型前向传播
        with torch.no_grad():
            # 将样本移到 GPU
            sample = utils.move_to_cuda(sample)
            
            # 前向传播
            net_output = model(**sample['net_input'])
            
            # 获取 logits
            if hasattr(model, "get_logits"):
                logits = model.get_logits(net_output)
            else:
                logits = model.get_normalized_probs(net_output, log_probs=True)
            
            # 转换形状
            if logits.dim() == 3:
                if logits.size(0) == 1:  # (B, T, V)
                    logits = logits[0]  # (T, V)
                elif logits.size(1) == 1:  # (T, B, V)
                    logits = logits[:, 0, :]  # (T, V)
            
            # CTC Greedy 解码
            pred_tokens = logits.argmax(dim=-1).cpu()
            
            # 打印原始预测序列信息
            print(f"  原始预测序列长度: {len(pred_tokens)}")
            unique_tokens = torch.unique(pred_tokens)
            print(f"  预测中的唯一 token 数量: {len(unique_tokens)}")
            print(f"  唯一 tokens: {unique_tokens.tolist()[:20]}")  # 打印前20个
            
            # 检查 blank token 的索引
            blank_idx = task.target_dictionary.bos()  # 尝试用 bos 作为 blank
            print(f"  BOS (可能是 blank): {blank_idx}")
            print(f"  PAD: {task.target_dictionary.pad()}")
            print(f"  EOS: {task.target_dictionary.eos()}")
            
            # 统计预测中各个 token 的频率
            token_counts = {}
            for token in pred_tokens.tolist():
                token_counts[token] = token_counts.get(token, 0) + 1
            sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
            print(f"  最常见的 3 个 tokens: {sorted_tokens[:3]}")
            
            # 去除重复和 blank - 尝试不同的 blank_idx
            pred_text = ""
            for possible_blank in [0, blank_idx, task.target_dictionary.pad()]:
                pred_tokens_unique = []
                prev = None
                for token in pred_tokens.tolist():
                    if token != possible_blank and token != prev:
                        pred_tokens_unique.append(token)
                    prev = token
                
                if len(pred_tokens_unique) > 0:
                    pred_text = task.target_dictionary.string(pred_tokens_unique)
                    print(f"\n  [blank={possible_blank}] 预测文本: {pred_text}")
                    print(f"  [blank={possible_blank}] 预测长度: {len(pred_tokens_unique)}")
                    break
            else:
                print(f"  警告: 所有解码尝试都为空！")
                print(f"  前10个预测 tokens: {pred_tokens.tolist()[:10]}")
            
            # 计算字符级错误
            if len(target_text) > 0 and len(pred_text) > 0:
                import editdistance
                cer = editdistance.eval(list(pred_text), list(target_text)) / len(target_text) * 100
                print(f"  CER: {cer:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python debug_ctc_output.py <checkpoint路径>")
        print("\n示例:")
        print("python debug_ctc_output.py /hdd2/avhubert_exp/finetune/probe_pt_cncvs3_iter1_cnvsrc_ctc/checkpoints/checkpoint_best.pt")
        sys.exit(1)
    
    ckpt_path = sys.argv[1]
    debug_ctc_predictions(ckpt_path, n_samples=5)

