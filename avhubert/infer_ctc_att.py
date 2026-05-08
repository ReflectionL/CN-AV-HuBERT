# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# CTC + Attention Joint Beam Search Inference Script for AVHubertSeq2Seq.
# Uses ESPnet's BatchBeamSearch (from CNVSRC2024Baseline) with custom Fairseq adapters.

import sys
import os

# ESPnet lives inside CNVSRC2024Baseline; add it to path before any espnet imports.
_CNVSRC_PATH = os.environ.get(
    "CNVSRC_PATH", "/home/lixiaolou/code/CNVSRC2024Baseline"
)
if _CNVSRC_PATH not in sys.path:
    sys.path.insert(0, _CNVSRC_PATH)

import hashlib
import json
import logging
from argparse import Namespace
from dataclasses import dataclass, field, is_dataclass
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import editdistance
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

import hydra
from hydra.core.config_store import ConfigStore

from fairseq import checkpoint_utils, distributed_utils, tasks, utils
from fairseq.dataclass.configs import (
    CheckpointConfig,
    CommonConfig,
    CommonEvalConfig,
    DatasetConfig,
    DistributedTrainingConfig,
    FairseqDataclass,
    GenerationConfig,
)
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter

# ESPnet beam search components (from CNVSRC2024Baseline).
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.scorer_interface import BatchScorerInterface
from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet.nets.scorers.length_bonus import LengthBonus

logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config_path = Path(__file__).resolve().parent / "conf"


# ---------------------------------------------------------------------------
# Hydra config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class OverrideConfig(FairseqDataclass):
    noise_wav: Optional[str] = field(default=None, metadata={"help": "noise wav file"})
    noise_prob: float = field(default=0, metadata={"help": "noise probability"})
    noise_snr: float = field(default=0, metadata={"help": "noise SNR in audio"})
    modalities: List[str] = field(
        default_factory=lambda: [""], metadata={"help": "which modality to use"}
    )
    data: Optional[str] = field(default=None, metadata={"help": "path to test data directory"})
    label_dir: Optional[str] = field(
        default=None, metadata={"help": "path to test label directory"}
    )
    ctc_weight: float = field(
        default=0.3, metadata={"help": "CTC weight for joint decoding (0=att-only, 1=ctc-only)"}
    )
    beam_size: int = field(default=10, metadata={"help": "beam search size"})
    lm_weight: float = field(default=0.0, metadata={"help": "LM weight (unused unless LM provided)"})
    length_bonus: float = field(default=0.0, metadata={"help": "length bonus penalty weight"})
    maxlenratio: float = field(
        default=0.1,
        metadata={"help": "max output len = maxlenratio * encoder_frames; 0 means = encoder_frames (no limit)"},
    )
    minlenratio: float = field(
        default=0.0, metadata={"help": "min output len = minlenratio * encoder_frames"}
    )


@dataclass
class InferCTCAttConfig(FairseqDataclass):
    task: Any = None
    generation: GenerationConfig = GenerationConfig()
    common: CommonConfig = CommonConfig()
    common_eval: CommonEvalConfig = CommonEvalConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    distributed_training: DistributedTrainingConfig = DistributedTrainingConfig()
    dataset: DatasetConfig = DatasetConfig()
    override: OverrideConfig = OverrideConfig()
    is_ax: bool = field(
        default=False,
        metadata={"help": "if true, returns a tuple for ax tuning"},
    )


# ---------------------------------------------------------------------------
# ESPnet scorer adapters for Fairseq model components
# ---------------------------------------------------------------------------

class FairseqCTCModule:
    """Wraps Fairseq's CTC linear projection to satisfy ESPnet CTCPrefixScorer's
    expectation of a module with ``log_softmax(x)`` where x is (1, T, D)."""

    def __init__(self, ctc_proj: torch.nn.Module):
        self.ctc_proj = ctc_proj

    def log_softmax(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (1, T, D) encoder output (batch-of-one added by CTCPrefixScorer).
        Returns:
            (1, T, V) CTC log-probabilities.
        """
        logits = self.ctc_proj(x)  # (1, T, V)
        return F.log_softmax(logits.float(), dim=-1)


class FairseqDecoderScorer(BatchScorerInterface):
    """Adapts Fairseq's TransformerDecoder to ESPnet's BatchScorerInterface.

    The decoder is called without incremental state (full recompute each step).
    This is simple and correct; efficiency can be improved with KV-cache later.
    """

    def __init__(self, decoder: torch.nn.Module, encoder_out_single: dict):
        """
        Args:
            decoder: Fairseq TransformerDecoder.
            encoder_out_single: encoder_out dict with batch_size == 1.
        """
        self.decoder = decoder
        self.encoder_out_single = encoder_out_single

    def _expand_encoder_out(self, B: int) -> dict:
        """Broadcast single-utterance encoder_out to batch size B (no copy)."""
        enc = self.encoder_out_single
        enc_tensor = enc["encoder_out"]  # (T, 1, D)
        result: dict = {
            "encoder_out": enc_tensor.expand(-1, B, -1).contiguous(),  # (T, B, D)
            "encoder_padding_mask": None,
            "padding_mask": None,
        }
        if enc.get("encoder_padding_mask") is not None:
            result["encoder_padding_mask"] = enc["encoder_padding_mask"].expand(B, -1).contiguous()
        if enc.get("padding_mask") is not None:
            result["padding_mask"] = enc["padding_mask"].expand(B, -1).contiguous()
        return result

    # ------------------------------------------------------------------
    # ScorerInterface / BatchScorerInterface implementation
    # ------------------------------------------------------------------

    def init_state(self, x: torch.Tensor) -> Any:
        return None

    def score(
        self, y: torch.Tensor, state: Any, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Any]:
        """Single-hypothesis scoring (falls back from batch_score when B==1).

        Args:
            y: (L,) int64 prefix tokens starting with eos-as-bos.
            state: unused (no incremental state).
            x: (T, D) encoder output – ignored; we use stored encoder_out.
        Returns:
            (V,) log-prob over vocab, None state.
        """
        prev_tokens = y.unsqueeze(0)  # (1, L)
        with torch.no_grad():
            decoder_out, _ = self.decoder(
                prev_output_tokens=prev_tokens,
                encoder_out=self.encoder_out_single,
            )
        logprob = F.log_softmax(decoder_out[0, -1].float(), dim=-1)  # (V,)
        return logprob, None

    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Batched scoring across all active beam hypotheses.

        Args:
            ys: (B, L) int64 prefix tokens (padded to same length with eos).
            states: list of B None values (no incremental state used).
            xs: (B, T, D) encoder output – ignored; we use stored encoder_out.
        Returns:
            (B, V) log-probs, list of B None states.
        """
        B = ys.size(0)
        enc_out = self._expand_encoder_out(B)
        with torch.no_grad():
            decoder_out, _ = self.decoder(
                prev_output_tokens=ys,
                encoder_out=enc_out,
            )
        # decoder_out: (B, L, V) — take last position for each hypothesis.
        logprobs = F.log_softmax(decoder_out[:, -1, :].float(), dim=-1)  # (B, V)
        return logprobs, [None] * B


# ---------------------------------------------------------------------------
# Build beam search for one utterance
# ---------------------------------------------------------------------------

def build_beam_search(
    model,
    encoder_out: dict,
    dictionary,
    ctc_weight: float = 0.3,
    beam_size: int = 40,
    lm_weight: float = 0.0,
    length_bonus: float = 0.0,
) -> BatchBeamSearch:
    """Construct an ESPnet BatchBeamSearch for a single utterance.

    In Fairseq, EOS (index 2) is used as BOS for the decoder, matching the
    training ``prev_output_tokens`` convention. We therefore set sos = eos =
    ``dictionary.eos()``.

    Args:
        model: AVHubertSeq2Seq model (must have .ctc_proj and .decoder).
        encoder_out: Fairseq encoder output dict with batch_size == 1.
        dictionary: Fairseq Dictionary for the target language.
        ctc_weight: weight for CTC prefix scorer (0 → att-only, 1 → ctc-only).
        beam_size: number of beams.
        lm_weight: external LM weight (0 to disable).
        length_bonus: length bonus weight.
    """
    eos = dictionary.eos()
    vocab_size = len(dictionary)
    token_list = list(dictionary.symbols)

    ctc_module = FairseqCTCModule(model.ctc_proj)
    ctc_scorer = CTCPrefixScorer(ctc_module, eos)
    dec_scorer = FairseqDecoderScorer(model.decoder, encoder_out)

    scorers: dict = {
        "decoder": dec_scorer,
        "ctc": ctc_scorer,
        "length_bonus": LengthBonus(vocab_size),
    }
    weights: dict = {
        "decoder": 1.0 - ctc_weight,
        "ctc": ctc_weight,
        "length_bonus": length_bonus,
    }

    return BatchBeamSearch(
        beam_size=beam_size,
        vocab_size=vocab_size,
        weights=weights,
        scorers=scorers,
        sos=eos,
        eos=eos,
        token_list=token_list,
        pre_beam_score_key=None if ctc_weight == 1.0 else "decoder",
    )


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _decode_hypo(hypo_yseq: torch.Tensor, eos: int, dictionary, label_processor) -> str:
    """Convert beam search output token IDs to a text string.

    Strips leading sos and trailing eos tokens, then decodes.
    """
    tokens = hypo_yseq.long().cpu()
    # Remove leading sos (== eos in Fairseq) added by beam search initialisation.
    if len(tokens) > 0 and tokens[0].item() == eos:
        tokens = tokens[1:]
    # Remove trailing eos if present.
    if len(tokens) > 0 and tokens[-1].item() == eos:
        tokens = tokens[:-1]

    symbols_ignore = {dictionary.pad(), dictionary.eos()}
    if hasattr(label_processor, "decode"):
        return label_processor.decode(tokens, symbols_ignore)
    chars = dictionary.string(tokens, extra_symbols_to_ignore=symbols_ignore)
    words = " ".join("".join(chars.split()).replace("|", " ").split())
    return words


# ---------------------------------------------------------------------------
# Main inference logic
# ---------------------------------------------------------------------------

def _main(cfg: DictConfig, output_file):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=output_file,
    )
    _logger = logging.getLogger("ctc_att_beam_search")
    _logger.propagate = False   # prevent double-print via root logger
    _logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    # always write to the result file
    fh = logging.StreamHandler(output_file)
    fh.setFormatter(fmt)
    _logger.addHandler(fh)
    # also mirror to stdout when writing to a file
    if output_file is not sys.stdout:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        _logger.addHandler(sh)

    utils.import_user_module(cfg.common)

    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [cfg.common_eval.path]
    )
    model = models[0].eval()
    if torch.cuda.is_available():
        model = model.cuda()

    saved_cfg.task.modalities = cfg.override.modalities
    task = tasks.setup_task(saved_cfg.task)
    task.build_tokenizer(saved_cfg.tokenizer)
    task.build_bpe(saved_cfg.bpe)

    _logger.info(cfg)

    if cfg.common.seed is not None:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_cuda = torch.cuda.is_available()
    dictionary = task.target_dictionary

    # ---- Model architecture verification ----
    model_cfg = saved_cfg.model
    _logger.info(
        f"Loaded model type      : {type(model).__name__}\n"
        f"  use_ctc              : {getattr(model, 'use_ctc', '?')}\n"
        f"  ctc_proj             : {model.ctc_proj}\n"
        f"  use_reverse_decoder  : {getattr(model, 'use_reverse_decoder', '?')}\n"
        f"  decoder layers       : {getattr(model_cfg, 'decoder_layers', '?')}\n"
        f"  decoder embed_dim    : {getattr(model_cfg, 'decoder_embed_dim', '?')}\n"
        f"  decoder attn_heads   : {getattr(model_cfg, 'decoder_attention_heads', '?')}\n"
        f"  reverse_decoder_layers: {getattr(model_cfg, 'reverse_decoder_layers', '?')}\n"
        f"  vocab size           : {len(task.target_dictionary)}"
    )
    if not (hasattr(model, "ctc_proj") and model.ctc_proj is not None):
        raise RuntimeError(
            "Model does not have a CTC projection (ctc_proj). "
            "Make sure the checkpoint was trained with use_ctc=true."
        )

    task.cfg.noise_prob = cfg.override.noise_prob
    task.cfg.noise_snr = cfg.override.noise_snr
    task.cfg.noise_wav = cfg.override.noise_wav
    if cfg.override.data is not None:
        task.cfg.data = cfg.override.data
    if cfg.override.label_dir is not None:
        task.cfg.label_dir = cfg.override.label_dir
    task.load_dataset(cfg.dataset.gen_subset, task_cfg=saved_cfg.task)

    itr = task.get_batch_iterator(
        dataset=task.dataset(cfg.dataset.gen_subset),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=utils.resolve_max_positions(
            task.max_positions(), *[m.max_positions() for m in models]
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_shards=cfg.distributed_training.distributed_world_size,
        shard_id=cfg.distributed_training.distributed_rank,
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size,
    ).next_epoch_itr(shuffle=False)

    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format="tqdm" if not cfg.common.no_progress_bar else "simple",
    )

    ctc_weight = cfg.override.ctc_weight
    beam_size = cfg.override.beam_size
    lm_weight = cfg.override.lm_weight
    length_bonus = cfg.override.length_bonus
    maxlenratio = cfg.override.maxlenratio
    minlenratio = cfg.override.minlenratio
    eos = dictionary.eos()
    _logger.info(
        f"Beam search config — ctc_weight={ctc_weight}, beam_size={beam_size}, "
        f"maxlenratio={maxlenratio}, lm_weight={lm_weight}, length_bonus={length_bonus}"
    )

    device = next(model.parameters()).device
    label_processor = task.datasets[cfg.dataset.gen_subset].label_processors[0]

    # Build beam search once; dec_scorer.encoder_out_single is updated per utterance.
    _dummy_enc_out = {
        "encoder_out": torch.zeros(1, 1, 1, device=device),
        "encoder_padding_mask": None,
        "padding_mask": None,
    }
    dec_scorer = FairseqDecoderScorer(model.decoder, _dummy_enc_out)
    ctc_module = FairseqCTCModule(model.ctc_proj)
    ctc_scorer = CTCPrefixScorer(ctc_module, eos)
    scorers = {
        "decoder": dec_scorer,
        "ctc": ctc_scorer,
        "length_bonus": LengthBonus(len(dictionary)),
    }
    weights = {
        "decoder": 1.0 - ctc_weight,
        "ctc": ctc_weight,
        "length_bonus": length_bonus,
    }
    beam_search = BatchBeamSearch(
        beam_size=beam_size,
        vocab_size=len(dictionary),
        weights=weights,
        scorers=scorers,
        sos=eos,
        eos=eos,
        token_list=list(dictionary.symbols),
        pre_beam_score_key=None if ctc_weight == 1.0 else "decoder",
    ).to(device)

    gen_timer = StopwatchMeter()
    num_sentences = 0
    result_dict = {"utt_id": [], "ref": [], "hypo": []}
    total_err = 0   # cumulative character edit distance
    total_len = 0   # cumulative reference character count

    with torch.no_grad():
        for sample in progress:
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            if "net_input" not in sample:
                continue

            net_input = sample["net_input"]
            batch_size = sample["id"].size(0)

            for i in range(batch_size):
                # net_input["source"] is {"audio": tensor|None, "video": tensor|None}.
                src = net_input["source"]
                source_i = {
                    "audio": src["audio"][i: i + 1] if src.get("audio") is not None else None,
                    "video": src["video"][i: i + 1] if src.get("video") is not None else None,
                }
                padding_mask_i = (
                    net_input["padding_mask"][i: i + 1]
                    if net_input.get("padding_mask") is not None
                    else None
                )

                gen_timer.start()

                # --- Encoder ---
                encoder_out = model.encoder(
                    source=source_i, padding_mask=padding_mask_i
                )

                # enc_feat: (T, D) — the ESPnet beam search entry point.
                enc_feat = encoder_out["encoder_out"][:, 0, :]  # (T, D)

                # Update decoder scorer with this utterance's encoder output
                # (pointer swap only, no GPU allocation).
                dec_scorer.encoder_out_single = encoder_out

                # --- Run beam search ---
                nbest_hyps = beam_search(enc_feat, maxlenratio=maxlenratio, minlenratio=minlenratio)

                gen_timer.stop(len(nbest_hyps[0].yseq) if nbest_hyps else 0)

                # --- Decode best hypothesis ---
                hypo_str = _decode_hypo(
                    nbest_hyps[0].yseq, eos, dictionary, label_processor
                )

                ref_str = _decode_hypo(
                    sample["target"][i].int(), eos, dictionary, label_processor
                )

                utt_id = (
                    sample["utt_id"][i]
                    if "utt_id" in sample
                    else str(sample["id"][i].item())
                )
                result_dict["utt_id"].append(utt_id)
                result_dict["ref"].append(ref_str)
                result_dict["hypo"].append(hypo_str)

                # Per-sample CER (character-level edit distance).
                hypo_chars = list(hypo_str.replace(" ", ""))
                ref_chars = list(ref_str.replace(" ", ""))
                sample_err = editdistance.eval(hypo_chars, ref_chars)
                sample_len = len(ref_chars)
                sample_cer = 100.0 * sample_err / sample_len if sample_len > 0 else 0.0

                # Cumulative CER.
                total_err += sample_err
                total_len += sample_len
                running_cer = 100.0 * total_err / total_len if total_len > 0 else 0.0

                _logger.info(
                    f"UTT: {utt_id}\n"
                    f"REF: {ref_str}\n"
                    f"HYP: {hypo_str}\n"
                    f"CER: {sample_cer:.2f}%  |  running CER: {running_cer:.2f}% "
                    f"({num_sentences + i + 1} utts)\n"
                )

            num_sentences += (
                sample["nsentences"] if "nsentences" in sample else sample["id"].numel()
            )

    _logger.info(
        "Recognised {:,} utterances ({} tokens) in {:.1f}s "
        "({:.2f} sentences/s, {:.2f} tokens/s)".format(
            num_sentences,
            gen_timer.n,
            gen_timer.sum,
            num_sentences / gen_timer.sum if gen_timer.sum > 0 else 0,
            1.0 / gen_timer.avg if gen_timer.avg > 0 else 0,
        )
    )

    # --- Save results ---
    os.makedirs(cfg.common_eval.results_path, exist_ok=True)
    yaml_str = OmegaConf.to_yaml(cfg)
    fid = int(hashlib.md5(yaml_str.encode("utf-8")).hexdigest(), 16) % 1_000_000
    result_fn = f"{cfg.common_eval.results_path}/hypo-{fid}.json"
    json.dump(result_dict, open(result_fn, "w"), indent=4, ensure_ascii=False)

    n_err, n_total = 0, 0
    for hypo, ref in zip(result_dict["hypo"], result_dict["ref"]):
        hypo_toks = hypo.strip().split()
        ref_toks = ref.strip().split()
        n_err += editdistance.eval(hypo_toks, ref_toks)
        n_total += len(ref_toks)

    wer = 100 * n_err / n_total if n_total > 0 else 0.0
    wer_fn = f"{cfg.common_eval.results_path}/wer.{fid}"
    with open(wer_fn, "w") as fo:
        fo.write(f"WER: {wer:.2f}%\n")
        fo.write(f"err / num_ref_words = {n_err} / {n_total}\n\n")
        fo.write(f"ctc_weight={ctc_weight}  beam_size={beam_size}\n\n")
        fo.write(yaml_str)

    _logger.info(f"WER: {wer:.2f}%")
    _logger.info(f"Results → {result_fn}")
    _logger.info(f"WER    → {wer_fn}")


def main(cfg: DictConfig):
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)
    assert cfg.common_eval.path is not None, "--path required for recognition!"
    if cfg.common_eval.results_path is not None:
        os.makedirs(cfg.common_eval.results_path, exist_ok=True)
        output_path = os.path.join(cfg.common_eval.results_path, "decode.log")
        with open(output_path, "w", buffering=1, encoding="utf-8") as h:
            return _main(cfg, h)
    return _main(cfg, sys.stdout)


@hydra.main(config_path=str(config_path), config_name="infer_ctc_att")
def hydra_main(cfg: InferCTCAttConfig) -> Union[float, Tuple[float, Optional[float]]]:
    container = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    cfg = OmegaConf.create(container)
    OmegaConf.set_struct(cfg, True)

    try:
        if cfg.common.profile:
            with torch.cuda.profiler.profile():
                with torch.autograd.profiler.emit_nvtx():
                    distributed_utils.call_main(cfg, main)
        else:
            distributed_utils.call_main(cfg, main)
    except BaseException as e:
        if not cfg.common.suppress_crashes:
            raise
        else:
            logger.error("Crashed! %s", str(e))


def cli_main() -> None:
    try:
        from hydra._internal.utils import get_args
        cfg_name = get_args().config_name or "infer_ctc_att"
    except ImportError:
        cfg_name = "infer_ctc_att"

    cs = ConfigStore.instance()
    cs.store(name=cfg_name, node=InferCTCAttConfig)
    for k in InferCTCAttConfig.__dataclass_fields__:
        if is_dataclass(InferCTCAttConfig.__dataclass_fields__[k].type):
            v = InferCTCAttConfig.__dataclass_fields__[k].default
            cs.store(name=k, node=v)
    hydra_main()


if __name__ == "__main__":
    cli_main()
