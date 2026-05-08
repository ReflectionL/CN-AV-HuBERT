# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import torch
from fairseq import file_utils
from fairseq.data.encoders import register_bpe
from fairseq.dataclass import FairseqDataclass


@dataclass
class CharTokenizerConfig(FairseqDataclass):
    char_dict_path: str = field(
        default="bpe/char_units.txt", metadata={"help": "path to character dictionary file"}
    )


@register_bpe("char_tokenizer", dataclass=CharTokenizerConfig)
class CharTokenizerBPE(object):
    """Character-based tokenizer wrapper for fairseq."""
    
    def __init__(self, cfg):
        dict_path = file_utils.cached_path(cfg.char_dict_path)
        # Load units and create dictionary
        units = open(dict_path, encoding='utf8').read().splitlines()
        self.hashmap = {unit.split()[0]: unit.split()[-1] for unit in units}
        # 0 will be used for "blank" in CTC
        self.token_list = ["<blank>"] + list(self.hashmap.keys()) + ["<eos>"]
        self.ignore_id = -1
    
    def encode(self, text: str) -> str:
        """Encode text to space-separated characters for fairseq dictionary."""
        # 将字符转换为字符序列（不是ID），让fairseq dictionary来处理ID映射
        tokens = []
        for char in text:
            # 跳过空格字符，因为空格不在字典中，且在中文处理中通常用于分词
            if char == ' ':
                continue
            elif char in self.hashmap:
                tokens.append(char)  # 直接使用字符，不是ID
            else:
                tokens.append("<unk>")  # 使用<unk>符号，不是ID
        return " ".join(tokens)
    
    def decode(self, x: str) -> str:
        """Decode space-separated tokens back to text."""
        if not x:
            return ""
        
        tokens = x.strip().split()
        chars = []
        
        for token in tokens:
            # 如果是数字ID，需要转换为字符
            if token.isdigit():
                # 创建反向映射：ID -> 字符
                id_to_char = {v: k for k, v in self.hashmap.items()}
                char = id_to_char.get(token, "")
                if char and char not in ["<unk>", "<blank>", "<eos>"]:
                    chars.append(char)
            else:
                # 如果已经是字符，直接使用（过滤特殊符号）
                if token not in ["<unk>", "<blank>", "<eos>", "<s>", "</s>", "<pad>"]:
                    chars.append(token)
        
        text = "".join(chars)
        return text.replace("▁", " ").strip()  # 处理空格符号
    
    def is_beginning_of_word(self, x: str) -> bool:
        """Check if token is beginning of word (for whole word masking)."""
        if x in ["<unk>", "<s>", "</s>", "<pad>", "<blank>", "<eos>"]:
            return True
        # 对于中文字符，每个字符都可以视为词的开始
        return True
