#!/usr/bin/env python3
"""Test script to verify custom tokenizer setup."""

import sys
import os


# 导入必要的模块
from argparse import Namespace
from fairseq.data import encoders

# 导入自定义tokenizer模块，确保它被注册
import char_tokenizer

def test_char_tokenizer():
    """测试字符tokenizer是否正常工作"""
    
    print("=" * 50)
    print("测试自定义Character Tokenizer")
    print("=" * 50)
    
    # 1. 测试tokenizer是否已注册
    print("\n1. 检查tokenizer是否已注册...")
    if 'char_tokenizer' in encoders.BPE_REGISTRY:
        print("✓ char_tokenizer 已成功注册到fairseq")
    else:
        print("✗ char_tokenizer 未注册")
        print("已注册的tokenizers:", list(encoders.BPE_REGISTRY.keys()))
        return False
    
    # 2. 测试创建tokenizer实例
    print("\n2. 创建tokenizer实例...")
    try:
        bpe_args = Namespace(**{
            'bpe': 'char_tokenizer',
            'char_dict_path': '/home/lixiaolou/code/CNVSRC2024Baseline/datamodule/char_units.txt'
        })
        tokenizer = encoders.build_bpe(bpe_args)
        print("✓ Tokenizer实例创建成功")
    except Exception as e:
        print(f"✗ 创建tokenizer失败: {e}")
        return False
    
    # 3. 测试编码功能
    print("\n3. 测试编码功能...")
    test_texts = [
        "你好世界",
        "测试文本",
        "这是一个测试"
    ]
    
    for text in test_texts:
        try:
            encoded = tokenizer.encode(text)
            print(f"  原文: '{text}'")
            print(f"  编码: '{encoded}'")
            
            # 测试解码
            decoded = tokenizer.decode(encoded)
            print(f"  解码: '{decoded}'")
            print()
        except Exception as e:
            print(f"✗ 编码/解码失败 '{text}': {e}")
            return False
    
    print("✓ 所有测试通过！")
    return True

if __name__ == "__main__":
    success = test_char_tokenizer()
    sys.exit(0 if success else 1)
