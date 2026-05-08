# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
# CTC Probe Test Inference Script

import ast
from itertools import chain
import logging
import math
import os
import sys
import json
import hashlib
import editdistance
from argparse import Namespace
import itertools as it

import numpy as np
import torch
from fairseq import checkpoint_utils, options, tasks, utils, distributed_utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from fairseq.models import FairseqLanguageModel
from omegaconf import DictConfig

from pathlib import Path
import hydra
from hydra.core.config_store import ConfigStore
from fairseq.dataclass.configs import (
    CheckpointConfig,
    CommonConfig,
    CommonEvalConfig,
    DatasetConfig,
    DistributedTrainingConfig,
    GenerationConfig,
    FairseqDataclass,
)
from dataclasses import dataclass, field, is_dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from omegaconf import OmegaConf

logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config_path = Path(__file__).resolve().parent / "conf"

@dataclass
class OverrideConfig(FairseqDataclass):
    noise_wav: Optional[str] = field(default=None, metadata={'help': 'noise wav file'})
    noise_prob: float = field(default=0, metadata={'help': 'noise probability'})
    noise_snr: float = field(default=0, metadata={'help': 'noise SNR in audio'})
    modalities: List[str] = field(default_factory=lambda: [""], metadata={'help': 'which modality to use'})
    data: Optional[str] = field(default=None, metadata={'help': 'path to test data directory'})
    label_dir: Optional[str] = field(default=None, metadata={'help': 'path to test label directory'})
    lm_path: Optional[str] = field(default=None, metadata={'help': 'path to KenLM arpa or binary LM file'})
    beam_size: int = field(default=1, metadata={'help': 'CTC beam size; 1 = greedy decoding'})
    lm_weight: float = field(default=2.0, metadata={'help': 'LM interpolation weight for beam search'})
    word_score: float = field(default=0.0, metadata={'help': 'word insertion score for beam search'})

@dataclass
class InferConfig(FairseqDataclass):
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
        metadata={
            "help": "if true, assumes we are using ax for tuning and returns a tuple for ax to consume"
        },
    )


def greedy_ctc_decode(log_probs, blank_idx, pad_idx):
    """
    CTC Greedy Decoding
    Args:
        log_probs: (T, B, V) log probabilities
        blank_idx: blank token index
        pad_idx: padding token index
    Returns:
        List of decoded sequences (one per sample in batch)
    """
    # Get best prediction at each timestep
    max_indices = log_probs.argmax(dim=-1)  # (T, B)
    
    batch_size = max_indices.size(1)
    decoded_sequences = []
    
    for b in range(batch_size):
        sequence = max_indices[:, b].tolist()
        # Remove consecutive duplicates and blanks
        decoded = []
        prev = None
        for idx in sequence:
            if idx != blank_idx and idx != prev and idx != pad_idx:
                decoded.append(idx)
            prev = idx
        decoded_sequences.append(torch.LongTensor(decoded))
    
    return decoded_sequences


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


def _main(cfg, output_file):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=output_file,
    )
    logger = logging.getLogger("hybrid.speech_recognize")
    if output_file is not sys.stdout:  # also print to stdout
        logger.addHandler(logging.StreamHandler(sys.stdout))

    utils.import_user_module(cfg.common)
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([cfg.common_eval.path])
    models = [model.eval().cuda() for model in models]
    saved_cfg.task.modalities = cfg.override.modalities
    task = tasks.setup_task(saved_cfg.task)

    task.build_tokenizer(saved_cfg.tokenizer)
    task.build_bpe(saved_cfg.bpe)

    logger.info(cfg)

    # Fix seed for reproducibility
    if cfg.common.seed is not None:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_cuda = torch.cuda.is_available()

    # Set dictionary
    dictionary = task.target_dictionary
    
    # Get blank index for CTC
    blank_idx = dictionary.index("<ctc_blank>") if "<ctc_blank>" in dictionary.indices else dictionary.bos()
    logger.info(f"CTC blank index: {blank_idx}")

    # Build CTC decoder (beam search or greedy)
    use_beam = cfg.override.beam_size > 1
    beam_decoder = None
    if use_beam:
        from torchaudio.models.decoder import ctc_decoder as build_ctc_decoder
        tokens = list(dictionary.symbols)
        blank_token = dictionary.symbols[blank_idx]
        # For Chinese char-level, use <pad> as sil_token (won't appear in real output)
        sil_token = "|" if "|" in dictionary.indices else dictionary.symbols[dictionary.pad()]
        lm_path = cfg.override.lm_path or None  # treat empty string as None
        beam_decoder = build_ctc_decoder(
            lexicon=None,
            tokens=tokens,
            lm=lm_path,
            beam_size=cfg.override.beam_size,
            lm_weight=cfg.override.lm_weight if lm_path else 0.0,
            word_score=cfg.override.word_score,
            blank_token=blank_token,
            sil_token=sil_token,
            nbest=1,
        )
        logger.info(
            f"CTC beam search: beam={cfg.override.beam_size}, "
            f"lm={cfg.override.lm_path}, lm_weight={cfg.override.lm_weight}"
        )
    else:
        logger.info("CTC greedy decoding")

    # loading the dataset should happen after the checkpoint has been loaded
    task.cfg.noise_prob = cfg.override.noise_prob
    task.cfg.noise_snr = cfg.override.noise_snr
    task.cfg.noise_wav = cfg.override.noise_wav
    if cfg.override.data is not None:
        task.cfg.data = cfg.override.data
    if cfg.override.label_dir is not None:
        task.cfg.label_dir = cfg.override.label_dir
    task.load_dataset(cfg.dataset.gen_subset, task_cfg=saved_cfg.task)

    # Optimize model for generation
    model = models[0]
    if cfg.common.fp16:
        model.half()
    if use_cuda:
        model.cuda()

    # Load dataset (possibly sharded)
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
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
    )

    def decode_fn(x):
        """Decode token IDs to text"""
        symbols_ignore = {dictionary.pad(), dictionary.eos()}
        if hasattr(task.datasets[cfg.dataset.gen_subset].label_processors[0], 'decode'):
            return task.datasets[cfg.dataset.gen_subset].label_processors[0].decode(x, symbols_ignore)
        chars = dictionary.string(x, extra_symbols_to_ignore=symbols_ignore)
        words = " ".join("".join(chars.split()).replace('|', ' ').split())
        return words

    num_sentences = 0
    gen_timer = StopwatchMeter()
    result_dict = {'utt_id': [], 'ref': [], 'hypo': []}
    
    with torch.no_grad():
        for sample in progress:
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            if "net_input" not in sample:
                continue

            gen_timer.start()
            
            # Forward pass through CTC model
            net_output = model(**sample["net_input"])
            
            # Get log probabilities
            if hasattr(model, "get_logits"):
                logits = model.get_logits(net_output)
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            else:
                log_probs = model.get_normalized_probs(net_output, log_probs=True)
            
            # log_probs shape: (T, B, V) from HubertEncoder, or (B, T, V) from some other models
            # If first dim equals batch size, it's (B, T, V) and needs transposing to (T, B, V)
            if log_probs.dim() == 3 and log_probs.size(0) == sample["target"].size(0):
                # (B, T, V), transpose to (T, B, V)
                log_probs = log_probs.transpose(0, 1)
            
            # CTC Decoding (greedy or beam search)
            if use_beam:
                # torchaudio decoder expects (B, T, V) on CPU
                emissions = log_probs.transpose(0, 1).cpu().float().contiguous()  # (T,B,V) -> (B,T,V)
                beam_results = beam_decoder(emissions)
                decoded_sequences = [r[0].tokens for r in beam_results]  # list of LongTensor
            else:
                decoded_sequences = greedy_ctc_decode(
                    log_probs.cpu(),
                    blank_idx=blank_idx,
                    pad_idx=dictionary.pad()
                )

            gen_timer.stop(sum(len(seq) for seq in decoded_sequences))

            # Process results
            symbols_ignore = {dictionary.pad(), dictionary.eos(), dictionary.bos()}
            for i in range(len(sample["id"])):
                utt_id = sample['utt_id'][i] if 'utt_id' in sample else str(sample['id'][i].item())
                result_dict['utt_id'].append(utt_id)

                # Reference
                ref_tokens = sample['target'][i].int().cpu()
                ref_sent = decode_fn(ref_tokens)
                result_dict['ref'].append(ref_sent)

                # Hypothesis
                hypo_tokens = decoded_sequences[i]
                hypo_str = decode_fn(hypo_tokens)
                result_dict['hypo'].append(hypo_str)

                logger.info(f"UTT: {utt_id}")
                logger.info(f"REF: {ref_sent}")
                logger.info(f"HYP: {hypo_str}\n")
                
            num_sentences += sample["nsentences"] if "nsentences" in sample else sample["id"].numel()

    logger.info("Recognized {:,} utterances in {:.1f}s ({:.2f} sentences/s)".format(
        num_sentences, gen_timer.sum, num_sentences / gen_timer.sum if gen_timer.sum > 0 else 0))

    # Save results
    yaml_str = OmegaConf.to_yaml(cfg)
    fid = int(hashlib.md5(yaml_str.encode("utf-8")).hexdigest(), 16)
    fid = fid % 1000000
    result_fn = f"{cfg.common_eval.results_path}/hypo-{fid}.json"
    json.dump(result_dict, open(result_fn, 'w'), indent=4, ensure_ascii=False)
    
    # Calculate CER (character error rate)
    n_err, n_total = 0, 0
    assert len(result_dict['hypo']) == len(result_dict['ref'])
    for hypo, ref in zip(result_dict['hypo'], result_dict['ref']):
        hypo = list(hypo.strip().replace(' ', ''))
        ref = list(ref.strip().replace(' ', ''))
        n_err += editdistance.eval(hypo, ref)
        n_total += len(ref)

    cer = 100 * n_err / n_total if n_total > 0 else 0
    wer_fn = f"{cfg.common_eval.results_path}/cer.{fid}"
    with open(wer_fn, "w") as fo:
        fo.write(f"CER: {cer:.2f}%\n")
        fo.write(f"err / num_ref_chars = {n_err} / {n_total}\n\n")
        fo.write(f"{yaml_str}")

    logger.info(f"CER: {cer:.2f}%")
    logger.info(f"Results saved to {result_fn}")
    logger.info(f"CER saved to {wer_fn}")
    return


@hydra.main(config_path=config_path, config_name="infer")
def hydra_main(cfg: InferConfig) -> Union[float, Tuple[float, Optional[float]]]:
    container = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    cfg = OmegaConf.create(container)
    OmegaConf.set_struct(cfg, True)

    if cfg.common.reset_logging:
        reset_logging()

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
    return


def cli_main() -> None:
    try:
        from hydra._internal.utils import get_args
        cfg_name = get_args().config_name or "infer"
    except ImportError:
        logger.warning("Failed to get config name from hydra args")
        cfg_name = "infer"

    cs = ConfigStore.instance()
    cs.store(name=cfg_name, node=InferConfig)

    for k in InferConfig.__dataclass_fields__:
        if is_dataclass(InferConfig.__dataclass_fields__[k].type):
            v = InferConfig.__dataclass_fields__[k].default
            cs.store(name=k, node=v)

    hydra_main()


if __name__ == "__main__":
    cli_main()

