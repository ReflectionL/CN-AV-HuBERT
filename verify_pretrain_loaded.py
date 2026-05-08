#!/usr/bin/env python3
"""
验证 CTC Probe Test 是否正确加载了预训练模型权重
"""

import os
import sys

# 设置正确的路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'fairseq'))

import torch

def compare_checkpoints(pretrain_ckpt_path, finetune_ckpt_path):
    """对比预训练模型和微调模型的权重"""
    
    print(f"加载预训练模型: {pretrain_ckpt_path}")
    pretrain_state = torch.load(pretrain_ckpt_path, map_location='cpu')
    pretrain_model = pretrain_state['model']
    
    print(f"加载微调模型: {finetune_ckpt_path}")
    finetune_state = torch.load(finetune_ckpt_path, map_location='cpu')
    finetune_model = finetune_state['model']
    
    print("\n" + "="*80)
    print("检查 Encoder 权重是否来自预训练模型")
    print("="*80)
    
    # 检查几个关键层的权重
    encoder_keys_to_check = [
        'encoder.pos_conv.0.weight',  # 位置编码
        'encoder.layers.0.self_attn.k_proj.weight',  # 第一层注意力
        'encoder.layers.11.self_attn.k_proj.weight',  # 最后一层注意力
        'feature_extractor_video.resnet.frontend3D.0.weight',  # 视频特征提取器
        'post_extract_proj.weight',  # 投影层
    ]
    
    match_count = 0
    total_count = 0
    
    for key in encoder_keys_to_check:
        pretrain_key = key
        finetune_key = f'w2v_encoder.w2v_model.{key}'
        
        if pretrain_key in pretrain_model and finetune_key in finetune_model:
            pretrain_weight = pretrain_model[pretrain_key]
            finetune_weight = finetune_model[finetune_key]
            
            is_same = torch.allclose(pretrain_weight, finetune_weight, rtol=1e-5, atol=1e-7)
            total_count += 1
            
            if is_same:
                match_count += 1
                print(f"✅ {key[:60]:<60} MATCH")
            else:
                print(f"❌ {key[:60]:<60} DIFFERENT")
                print(f"   预训练权重范围: [{pretrain_weight.min():.6f}, {pretrain_weight.max():.6f}]")
                print(f"   微调权重范围:   [{finetune_weight.min():.6f}, {finetune_weight.max():.6f}]")
        else:
            if pretrain_key not in pretrain_model:
                print(f"⚠️  预训练模型中没有找到: {pretrain_key}")
            if finetune_key not in finetune_model:
                print(f"⚠️  微调模型中没有找到: {finetune_key}")
    
    print("\n" + "="*80)
    print(f"结果: {match_count}/{total_count} 个检查的权重匹配")
    print("="*80)
    
    if match_count == total_count and total_count > 0:
        print("✅ 预训练权重已成功加载！")
        return True
    elif match_count == 0:
        print("❌ 预训练权重未加载！这是从随机初始化开始训练的。")
        return False
    else:
        print("⚠️  部分权重匹配，请检查模型配置。")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python verify_pretrain_loaded.py <预训练模型路径> <微调模型路径>")
        print("\n示例:")
        print("python verify_pretrain_loaded.py \\")
        print("    /hdd2/avhubert_exp/pretrain/clean_pt_cncvs3_speech_iter2/checkpoints/checkpoint_best.pt \\")
        print("    /hdd2/avhubert_exp/finetune/probe_clean_pt_cncvs3_iter2_ft_cnvsrc_vsr_ctc/checkpoints/checkpoint_best.pt")
        sys.exit(1)
    
    pretrain_path = sys.argv[1]
    finetune_path = sys.argv[2]
    
    compare_checkpoints(pretrain_path, finetune_path)

