#!/bin/bash
# Seq2Seq Probe Test Inference Script

root=/home/lixiaolou/code/CN-AV-HuBERT
exp_path=/hdd2/avhubert_exp

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/home/lixiaolou/code/CN-AV-HuBERT/fairseq:${PYTHONPATH:-}"

# 配置参数
ckpt_dir=${exp_path}/finetune/probe_pt_cncvs3_iter4_ft_cnvsrc_vsr_s2s_10h/checkpoints/
ckpt_name=checkpoint_best.pt
subset=valid  # test / valid

# 数据路径
data_dir=${root}/avhubert/bpe/cnvsrc
label_dir=${root}/avhubert/bpe/cnvsrc

# 输出路径
results_path=${ckpt_dir}/decode_${subset}_s2s

mkdir -p ${results_path}

echo "开始 Seq2Seq Probe Test 推理..."
echo "检查点: ${ckpt_dir}/${ckpt_name}"
echo "数据集: ${subset}"
echo "结果保存到: ${results_path}"

python -B ${root}/avhubert/infer_s2s_probe.py \
    --config-dir ${root}/avhubert/conf \
    --config-name infer \
    dataset.gen_subset=${subset} \
    common_eval.path=${ckpt_dir}/${ckpt_name} \
    common_eval.results_path=${results_path} \
    override.modalities=['video'] \
    override.data=${data_dir} \
    override.label_dir=${label_dir} \
    common.user_dir=${root}/avhubert \
    dataset.max_tokens=1000 \
    dataset.num_workers=0 \
    generation.beam=20 \
    generation.lenpen=1.0

echo "推理完成！"
echo "结果保存在: ${results_path}"

