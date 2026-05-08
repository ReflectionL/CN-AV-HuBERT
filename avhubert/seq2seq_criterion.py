# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
    LabelSmoothedCrossEntropyCriterionConfig,
    label_smoothed_nll_loss,
)


@dataclass
class LabelSmoothedCEWithCTCCriterionConfig(LabelSmoothedCrossEntropyCriterionConfig):
    reverse_weight: float = field(
        default=0.0,
        metadata={"help": "weight for reverse decoder loss (0.0-1.0), 0 = disabled"},
    )
    ctc_weight: float = field(
        default=0.1,
        metadata={"help": "weight for CTC auxiliary loss (0.0-1.0), 0 = disabled"},
    )
    zero_infinity: bool = field(
        default=True,
        metadata={"help": "zero inf CTC loss when source length <= target length"},
    )


@register_criterion(
    "label_smoothed_cross_entropy_with_ctc",
    dataclass=LabelSmoothedCEWithCTCCriterionConfig,
)
class LabelSmoothedCEWithCTCCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
        reverse_weight=0.0,
        ctc_weight=0.1,
        zero_infinity=True,
    ):
        super().__init__(
            task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy
        )
        self.reverse_weight = reverse_weight
        self.ctc_weight = ctc_weight
        self.zero_infinity = zero_infinity
        # CTC blank: use pad index (index 1 in fairseq dicts), avoids conflict
        # with real tokens since pad never appears in targets
        self.blank_idx = task.target_dictionary.pad()
        self.eos_idx = task.target_dictionary.eos()

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])

        # ── Attention (seq2seq) loss ──────────────────────────────────────────
        att_loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)

        # ── Reverse decoder loss ──────────────────────────────────────────────
        r_loss = torch.tensor(0.0, device=att_loss.device)
        r_nll_loss = torch.tensor(0.0, device=att_loss.device)
        has_reverse = "r_decoder_logits" in net_output[1]
        if has_reverse and self.reverse_weight > 0:
            r_loss, r_nll_loss = self.compute_reverse_loss(
                net_output, sample, reduce=reduce
            )

        # ── CTC loss ──────────────────────────────────────────────────────────
        ctc_loss = torch.tensor(0.0, device=att_loss.device)
        has_ctc = "ctc_logits" in net_output[1]
        if has_ctc and self.ctc_weight > 0:
            ctc_loss = self.compute_ctc_loss(net_output, sample)

        # ── Combined loss ─────────────────────────────────────────────────────
        # att_weight + ctc_weight + reverse_weight should sum to 1
        att_w = 1.0 - self.ctc_weight - (self.reverse_weight if has_reverse else 0.0)
        att_w = max(att_w, 0.0)
        total_loss = att_w * att_loss
        if has_ctc and self.ctc_weight > 0:
            total_loss = total_loss + self.ctc_weight * ctc_loss
        if has_reverse and self.reverse_weight > 0:
            total_loss = total_loss + self.reverse_weight * r_loss

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": total_loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "att_loss": utils.item(att_loss.data),
        }

        if has_ctc and self.ctc_weight > 0:
            logging_output["ctc_loss"] = utils.item(ctc_loss.data)

        if has_reverse and self.reverse_weight > 0:
            logging_output["r_loss"] = utils.item(r_loss.data)
            logging_output["r_nll_loss"] = utils.item(r_nll_loss.data)

        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)

        return total_loss, sample_size, logging_output

    def compute_ctc_loss(self, net_output, sample):
        ctc_logits = net_output[1]["ctc_logits"]  # T x B x vocab
        lprobs = F.log_softmax(ctc_logits.float(), dim=-1)  # T x B x vocab

        # Input lengths from encoder padding mask
        padding_mask = net_output[1].get("encoder_padding_mask", None)
        if padding_mask is not None:
            input_lengths = (~padding_mask).long().sum(-1)  # B
        else:
            input_lengths = lprobs.new_full(
                (lprobs.size(1),), lprobs.size(0), dtype=torch.long
            )

        # Target: strip EOS and PAD, flatten
        target = sample["target"]
        pad_mask = (target != self.padding_idx) & (target != self.eos_idx)
        targets_flat = target.masked_select(pad_mask)
        target_lengths = pad_mask.sum(-1)  # B

        with torch.backends.cudnn.flags(enabled=False):
            loss = F.ctc_loss(
                lprobs,
                targets_flat,
                input_lengths,
                target_lengths,
                blank=self.blank_idx,
                reduction="sum",
                zero_infinity=self.zero_infinity,
            )
        return loss

    def compute_reverse_loss(self, net_output, sample, reduce=True):
        r_logits = net_output[1]["r_decoder_logits"]
        lprobs = F.log_softmax(r_logits.float(), dim=-1)
        lprobs = lprobs.view(-1, lprobs.size(-1))

        reversed_target = self._reverse_target(sample["target"])
        reversed_target = reversed_target.view(-1)

        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            reversed_target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def _reverse_target(self, target):
        """Reverse content tokens, keep EOS and PAD positions.

        target: (B, T) = [t1, t2, ..., tn, EOS, PAD, ...]
        returns:         [tn, ..., t2, t1, EOS, PAD, ...]
        """
        reversed_target = target.clone()
        non_pad = target.ne(self.padding_idx)
        lengths = non_pad.sum(dim=1)
        for i in range(target.size(0)):
            l = lengths[i].item()
            if l > 1:
                reversed_target[i, : l - 1] = target[i, : l - 1].flip(0)
        return reversed_target

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        super().reduce_metrics(logging_outputs)

        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)

        # CTC loss
        ctc_loss_sum = sum(log.get("ctc_loss", 0) for log in logging_outputs)
        if ctc_loss_sum > 0 and ntokens > 0:
            metrics.log_scalar(
                "ctc_loss",
                ctc_loss_sum / ntokens / math.log(2),
                ntokens,
                round=3,
            )

        # Reverse decoder loss
        r_loss_sum = sum(log.get("r_loss", 0) for log in logging_outputs)
        if r_loss_sum > 0 and sample_size > 0:
            metrics.log_scalar(
                "r_loss",
                r_loss_sum / sample_size / math.log(2),
                sample_size,
                round=3,
            )
            r_nll_loss_sum = sum(log.get("r_nll_loss", 0) for log in logging_outputs)
            if ntokens > 0:
                metrics.log_scalar(
                    "r_nll_loss",
                    r_nll_loss_sum / ntokens / math.log(2),
                    ntokens,
                    round=3,
                )


# ── Legacy criterion kept for backward compatibility ──────────────────────────

@dataclass
class LabelSmoothedCEWithReverseCriterionConfig(LabelSmoothedCrossEntropyCriterionConfig):
    reverse_weight: float = field(
        default=0.3,
        metadata={"help": "weight for reverse decoder loss (0.0-1.0)"},
    )


@register_criterion(
    "label_smoothed_cross_entropy_with_reverse",
    dataclass=LabelSmoothedCEWithReverseCriterionConfig,
)
class LabelSmoothedCEWithReverseCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
        reverse_weight=0.3,
    ):
        super().__init__(
            task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy
        )
        self.reverse_weight = reverse_weight

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])

        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)

        r_loss = torch.tensor(0.0, device=loss.device)
        r_nll_loss = torch.tensor(0.0, device=loss.device)
        has_reverse = "r_decoder_logits" in net_output[1]

        if has_reverse:
            r_loss, r_nll_loss = self.compute_reverse_loss(
                net_output, sample, reduce=reduce
            )
            total_loss = (1 - self.reverse_weight) * loss + self.reverse_weight * r_loss
        else:
            total_loss = loss

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": total_loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

        if has_reverse:
            logging_output["r_loss"] = utils.item(r_loss.data)
            logging_output["r_nll_loss"] = utils.item(r_nll_loss.data)

        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)

        return total_loss, sample_size, logging_output

    def compute_reverse_loss(self, net_output, sample, reduce=True):
        r_logits = net_output[1]["r_decoder_logits"]
        lprobs = F.log_softmax(r_logits.float(), dim=-1)
        lprobs = lprobs.view(-1, lprobs.size(-1))

        reversed_target = self._reverse_target(sample["target"])
        reversed_target = reversed_target.view(-1)

        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            reversed_target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def _reverse_target(self, target):
        """Reverse content tokens, keep EOS and PAD positions.

        target: (B, T) = [t1, t2, ..., tn, EOS, PAD, ...]
        returns:         [tn, ..., t2, t1, EOS, PAD, ...]
        """
        pad_idx = self.padding_idx
        reversed_target = target.clone()
        non_pad = target.ne(pad_idx)
        lengths = non_pad.sum(dim=1)
        for i in range(target.size(0)):
            l = lengths[i].item()
            if l > 1:
                reversed_target[i, : l - 1] = target[i, : l - 1].flip(0)
        return reversed_target

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        super().reduce_metrics(logging_outputs)

        r_loss_sum = sum(log.get("r_loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        if r_loss_sum > 0 and sample_size > 0:
            metrics.log_scalar(
                "r_loss",
                r_loss_sum / sample_size / math.log(2),
                sample_size,
                round=3,
            )
            r_nll_loss_sum = sum(
                log.get("r_nll_loss", 0) for log in logging_outputs
            )
            ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
            if ntokens > 0:
                metrics.log_scalar(
                    "r_nll_loss",
                    r_nll_loss_sum / ntokens / math.log(2),
                    ntokens,
                    round=3,
                )
