import torch
import torch.nn as nn
import os
from fairseq import checkpoint_utils
from avhubert.hubert import AVHubertModel

class AVHubertLinearProbe(nn.Module):
    def __init__(self, ckpt_path, vocab_size, device="cuda", input_modality="av", encoder_output_layer=None):
        super().__init__()
        assert input_modality in ["av", "v"], "input_modality 只能是 av(音频+视频) 或 v(仅视频)"
        state = checkpoint_utils.load_checkpoint_to_cpu(ckpt_path)
        args = state["args"] if "args" in state else state["cfg"]
        # 加载底座
        models, _, _ = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.encoder: AVHubertModel = models[0]
        self.encoder.eval()
        self.freeze_encoder()

        self.input_modality = input_modality
        self.encoder_output_layer = encoder_output_layer
        hid = self.encoder.cfg.encoder_embed_dim
        self.proj = nn.Linear(hid, vocab_size, bias=True)

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def extract_feat(self, sample):
        """
        sample 需包含:
          "net_input": {
             "source": Tensor  # B x C x T (或已是预处理后的特征), 兼容 AV 结构
             "video":  Tensor? # 若你的视频在单独键
             "padding_mask": BoolTensor (B x T)
          }
        这里根据你现有数据管线调整。我们假设你已有一个与预训练一致的 collate。
        """
        net_input = sample["net_input"]
        # 关闭 mask, features_only 获取隐藏层
        # avhubert forward 输入需要是 dict?
        # 这里直接调用底层 encoder 前的 wrapper:
        out = self.encoder(
            source=net_input["source"],
            padding_mask=net_input.get("padding_mask", None),
            mask=False,
            features_only=True,
            output_layer=self.encoder_output_layer
        )
        # 返回 x: T x B x C
        feats = out["x"]  # T x B x C
        padding_mask = out["padding_mask"]  # B x T
        return feats, padding_mask

    def forward(self, sample):
        with torch.no_grad():
            feats, padding_mask = self.extract_feat(sample)
        # feats: T x B x C -> 直接线性
        logits = self.proj(feats)  # T x B x V
        return logits, padding_mask