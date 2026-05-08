import torch
from torch.utils.data import Dataset
import torchaudio
import json, os

class SimpleAVCtcDataset(Dataset):
    """
    假设你准备一个 manifest/tsv，列:
      path \t label
    label 为字符序列（已无空格；或 BPE 需先分词成 id）
    这里仅示例：真正实现请对齐你已有 AV-HuBERT 预训练的输入提取（帧率、ROI 等）。
    """
    def __init__(self, tsv_path, dict_path, tokenizer="char"):
        self.items = []
        with open(tsv_path, "r") as f:
            for line in f:
                sp = line.strip().split("\t")
                if len(sp) < 2: continue
                self.items.append((sp[0], sp[1]))
        # 字典
        self.symbols = ["<pad>", "<blank>"]
        with open(dict_path, "r") as f:
            for line in f:
                tok = line.strip().split()[0]
                if tok not in self.symbols:
                    self.symbols.append(tok)
        self.sym2idx = {s:i for i,s in enumerate(self.symbols)}
        self.blank_idx = self.sym2idx["<blank>"]

    def __len__(self):
        return len(self.items)

    def text_to_ids(self, txt):
        return [self.sym2idx[c] for c in txt if c in self.sym2idx]

    def __getitem__(self, idx):
        wav_path, text = self.items[idx]
        wav, sr = torchaudio.load(wav_path)
        # TODO: 视频特征读取 / 对齐（留给你填充）
        # 这里只返回音频示例
        ids = self.text_to_ids(text)
        return {
            "id": idx,
            "audio": wav,
            "sr": sr,
            "target": torch.tensor(ids, dtype=torch.long)
        }

def collate(batch, pad_idx=0):
    # 极简示例；真实实现需与 avhubert 预训练对齐
    max_tgt = max(len(x["target"]) for x in batch)
    targets = []
    target_lengths = []
    for x in batch:
        t = x["target"]
        target_lengths.append(len(t))
        if len(t) < max_tgt:
            t = torch.cat([t, torch.full((max_tgt-len(t),), pad_idx, dtype=torch.long)])
        targets.append(t)
    targets = torch.stack(targets)  # B x L
    sample = {
        "id": torch.tensor([x["id"] for x in batch]),
        "target": targets,
        "target_lengths": torch.tensor(target_lengths, dtype=torch.long),
        "net_input": {
            # 这里应该放与预训练相同的特征 (如 log-mel+视频)
            # 简化：用音频波形占位
            "source": torch.cat([x["audio"] for x in batch], dim=0).unsqueeze(0),  # 占位
        }
    }
    return sample