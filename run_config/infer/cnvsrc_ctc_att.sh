#!/bin/bash
# CTC + Attention Joint Beam Search Inference
# 支持单卡和多卡数据分片两种模式。

root=/home/lixiaolou/code/CN-AV-HuBERT
exp_path=/hdd2/avhubert_exp

# ---- 修改这里 ----
checkpoint_path=${exp_path}/finetune/clean_pt_cncvs3_iter5_ft_cnvsrc_vsr_btd_v5/checkpoints/checkpoint_best.pt
results_path=${exp_path}/infer/clean_pt_cncvs3_iter5_ft_cnvsrc_vsr_btd_v5_ctc_att

# ---- 解码超参 ----
CTC_WEIGHT=0.3
BEAM_SIZE=20
LM_WEIGHT=0.0
LENGTH_BONUS=0.0
MAXLENRATIO=-300   # <0 = absolute cap: max 200 output tokens per utterance
MINLENRATIO=0.0

# ---- 显卡设置 ----
# 单卡：GPUS=1，CUDA_VISIBLE_DEVICES 指定哪张卡（如 0、1、2…）
# 多卡：GPUS=N，CUDA_VISIBLE_DEVICES 列出所有要用的卡（如 "0,1,2,3"）
GPUS=1
CUDA_VISIBLE_DEVICES=0

# ------------------------------------------------------------------
mkdir -p ${results_path}

export PYTHONPATH="/home/lixiaolou/code/CN-AV-HuBERT/fairseq:${PYTHONPATH:-}"
export CNVSRC_PATH="/home/lixiaolou/code/CNVSRC2024Baseline"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}

echo "==== CTC+Att Beam Search Inference ===="
echo "Checkpoint  : ${checkpoint_path}"
echo "Results     : ${results_path}"
echo "ctc_weight  : ${CTC_WEIGHT}  beam_size: ${BEAM_SIZE}"
echo "GPUs        : ${GPUS}  (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
echo "======================================="

cd ${root}

if [ ${GPUS} -eq 1 ]; then
    # ---- 单卡 ----
    python avhubert/infer_ctc_att.py \
        --config-dir ${root}/avhubert/conf \
        --config-name infer_ctc_att \
        common.user_dir=${root}/avhubert \
        common.log_format=simple \
        common.log_interval=1 \
        common_eval.path=${checkpoint_path} \
        common_eval.results_path=${results_path} \
        common_eval.quiet=false \
        override.data=${root}/avhubert/bpe/cnvsrc/ \
        override.label_dir=${root}/avhubert/bpe/cnvsrc/ \
        override.modalities='["video"]' \
        override.ctc_weight=${CTC_WEIGHT} \
        override.beam_size=${BEAM_SIZE} \
        override.lm_weight=${LM_WEIGHT} \
        override.length_bonus=${LENGTH_BONUS} \
        override.maxlenratio=${MAXLENRATIO} \
        override.minlenratio=${MINLENRATIO} \
        dataset.gen_subset=valid
else
    # ---- 多卡（数据分片，每张卡处理 1/GPUS 的测试集，结果分别写入 results_path） ----
    # 每张卡的结果保存在独立目录，最后自动汇总 CER。
    torchrun \
        --nproc_per_node=${GPUS} \
        --master_port=29600 \
        avhubert/infer_ctc_att.py \
        --config-dir ${root}/avhubert/conf \
        --config-name infer_ctc_att \
        common.user_dir=${root}/avhubert \
        common.log_format=simple \
        common.log_interval=1 \
        common_eval.path=${checkpoint_path} \
        common_eval.results_path=${results_path} \
        common_eval.quiet=false \
        override.data=${root}/avhubert/bpe/cnvsrc/ \
        override.label_dir=${root}/avhubert/bpe/cnvsrc/ \
        override.modalities='["video"]' \
        override.ctc_weight=${CTC_WEIGHT} \
        override.beam_size=${BEAM_SIZE} \
        override.lm_weight=${LM_WEIGHT} \
        override.length_bonus=${LENGTH_BONUS} \
        override.maxlenratio=${MAXLENRATIO} \
        override.minlenratio=${MINLENRATIO} \
        dataset.gen_subset=valid \
        distributed_training.distributed_world_size=${GPUS} \
        distributed_training.distributed_port=29600

    # ---- 汇总各分片的结果 ----
    echo ""
    echo "==== 汇总多卡结果 ===="
    python - <<'EOF'
import json, glob, editdistance, os, sys

results_path = sys.argv[1] if len(sys.argv) > 1 else "."
files = sorted(glob.glob(f"{results_path}/hypo-*.json"))
if not files:
    print("没有找到 hypo-*.json 文件")
    sys.exit(1)

merged = {"utt_id": [], "ref": [], "hypo": []}
for f in files:
    d = json.load(open(f))
    merged["utt_id"] += d["utt_id"]
    merged["ref"]    += d["ref"]
    merged["hypo"]   += d["hypo"]

n_err = sum(editdistance.eval(list(h.replace(" ","")), list(r.replace(" ","")))
            for h, r in zip(merged["hypo"], merged["ref"]))
n_total = sum(len(list(r.replace(" ",""))) for r in merged["ref"])
cer = 100 * n_err / n_total if n_total else 0

merged_fn = f"{results_path}/merged.json"
json.dump(merged, open(merged_fn, "w"), indent=4, ensure_ascii=False)
print(f"总句数: {len(merged['utt_id'])}")
print(f"CER:    {cer:.2f}%  ({n_err}/{n_total})")
with open(f"{results_path}/cer_merged.txt", "w") as fo:
    fo.write(f"CER: {cer:.2f}%\nerr/total = {n_err}/{n_total}\n")
EOF
    python - ${results_path}
fi

echo ""
echo "Done."
[ ${GPUS} -eq 1 ] && cat ${results_path}/wer.* 2>/dev/null || cat ${results_path}/cer_merged.txt 2>/dev/null
