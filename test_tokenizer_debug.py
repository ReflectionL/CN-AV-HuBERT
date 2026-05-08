#!/usr/bin/env python3
"""测试char_tokenizer的编码行为"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from avhubert.char_tokenizer import CharTokenizerBPE
from argparse import Namespace

def test_char_tokenizer():
    # 创建配置
    cfg = Namespace()
    cfg.char_dict_path = "/home/lixiaolou/code/CN-AV-HuBERT/avhubert/bpe/char_units.txt"
    
    # 初始化tokenizer
    tokenizer = CharTokenizerBPE(cfg)
    
    print("=== Char Tokenizer 调试 ===")
    print(f"字典大小: {len(tokenizer.hashmap)}")
    print(f"Token列表长度: {len(tokenizer.token_list)}")
    
    # 测试一些字符
    test_chars = ["但", "是", "老", "百", "姓"]
    print("\n字符映射测试:")
    for char in test_chars:
        if char in tokenizer.hashmap:
            print(f"  '{char}' -> ID: {tokenizer.hashmap[char]}")
        else:
            print(f"  '{char}' -> 不在字典中!")
    
    # 测试编码
    test_text = "但 是 老 百 姓"
    print(f"\n测试文本: '{test_text}'")
    encoded = tokenizer.encode(test_text)
    print(f"编码结果: '{encoded}'")
    
    # 测试解码
    decoded = tokenizer.decode(encoded)
    print(f"解码结果: '{decoded}'")
    
    # 检查是否所有结果都是3
    token_ids = encoded.split()
    unique_ids = set(token_ids)
    print(f"\n编码后的唯一ID: {unique_ids}")
    if len(unique_ids) == 1 and "3" in unique_ids:
        print("⚠️ 发现问题：所有字符都被编码为ID 3!")
        print("这解释了为什么训练时所有target都是3")
        
        # 检查ID 3对应的字符
        id_to_char = {v: k for k, v in tokenizer.hashmap.items()}
        if "3" in id_to_char:
            print(f"ID 3 对应的字符: '{id_to_char['3']}'")

if __name__ == "__main__":
    test_char_tokenizer()
