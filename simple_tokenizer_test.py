#!/usr/bin/env python3
"""简单的tokenizer测试，不依赖fairseq"""

def load_char_dict(dict_path):
    """加载字符词典"""
    hashmap = {}
    with open(dict_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                char = parts[0]
                char_id = parts[1]
                hashmap[char] = char_id
    return hashmap

def encode_text(text, hashmap):
    """编码文本"""
    tokens = []
    for char in text:
        if char in hashmap:
            tokens.append(hashmap[char])
        else:
            tokens.append(hashmap.get("<unk>", "1"))  # 使用<unk>的ID
    return " ".join(tokens)

def main():
    print("=== 简单Tokenizer测试 ===")
    
    # 加载字典
    dict_path = "/home/lixiaolou/code/CN-AV-HuBERT/avhubert/bpe/char_units.txt"
    hashmap = load_char_dict(dict_path)
    
    print(f"字典大小: {len(hashmap)}")
    
    # 检查一些关键字符
    test_chars = ["但", "是", "老", "百", "姓", "<unk>", "惶"]
    print("\n字符映射测试:")
    for char in test_chars:
        if char in hashmap:
            print(f"  '{char}' -> ID: {hashmap[char]}")
        else:
            print(f"  '{char}' -> 不在字典中!")
    
    # 测试编码
    test_text = "但 是 老 百 姓"
    print(f"\n测试文本: '{test_text}'")
    encoded = encode_text(test_text, hashmap)
    print(f"编码结果: '{encoded}'")
    
    # 分析编码结果
    token_ids = encoded.split()
    unique_ids = set(token_ids)
    print(f"编码后的唯一ID: {unique_ids}")
    print(f"所有token ID: {token_ids}")
    
    if len(unique_ids) == 1 and "3" in unique_ids:
        print("⚠️ 发现问题：所有字符都被编码为ID 3!")
        print("这解释了为什么训练时所有target都是3")
        
        # 检查ID 3对应的字符
        id_to_char = {v: k for k, v in hashmap.items()}
        if "3" in id_to_char:
            print(f"ID 3 对应的字符: '{id_to_char['3']}'")
    
    # 测试空格字符的处理
    print(f"\n空格字符测试:")
    print(f"  空格 ' ' 在字典中: {' ' in hashmap}")
    if ' ' in hashmap:
        print(f"  空格的ID: {hashmap[' ']}")
    
    # 测试单个字符（不包含空格）
    print(f"\n单个字符测试:")
    single_text = "但是老百姓"  # 没有空格
    single_encoded = encode_text(single_text, hashmap)
    print(f"文本: '{single_text}'")
    print(f"编码: '{single_encoded}'")
    single_token_ids = single_encoded.split()
    single_unique_ids = set(single_token_ids)
    print(f"唯一ID: {single_unique_ids}")

if __name__ == "__main__":
    main()
