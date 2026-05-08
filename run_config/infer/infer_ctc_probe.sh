#!/bin/bash
# CTC Probe Test Inference Script

root=/home/lixiaolou/code/CN-AV-HuBERT
exp_path=/hdd2/avhubert_exp

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/home/lixiaolou/code/CN-AV-HuBERT/fairseq:${PYTHONPATH:-}"

# 配置参数
ckpt_dir=${exp_path}/probe_test/pt_cncvs3_iter4_ft_cnvsrc_vsr_ctc/checkpoints/
ckpt_name=checkpoint_best.pt
subset=valid  # test / valid

# 数据路径
data_dir=${root}/avhubert/bpe/cnvsrc
label_dir=${root}/avhubert/bpe/cnvsrc

# LM 路径（注释掉则不使用 LM）
# lm_path=/hdd2/avhubert_exp/lm/lm_4gram.bin  # 或 .arpa
lm_path=  # 注释掉上面一行并启用这行 = 不用 LM，纯 beam search
beam_size=1
lm_weight=2.0
word_score=0.0

# 输出路径
results_path=${ckpt_dir}/decode_${subset}_beam${beam_size}${lm_path:+_lm${lm_weight}}

mkdir -p ${results_path}

echo "开始 CTC 推理..."
echo "检查点: ${ckpt_dir}/${ckpt_name}"
echo "数据集: ${subset}, beam=${beam_size}, lm_weight=${lm_weight}"
echo "结果保存到: ${results_path}"

# 只在 lm_path 非空时才传入 lm 相关参数
extra_args=""
if [ -n "${lm_path}" ]; then
    extra_args="override.lm_path=${lm_path} override.lm_weight=${lm_weight}"
fi

python -B ${root}/avhubert/infer_ctc.py \
    --config-dir ${root}/avhubert/conf \
    --config-name infer \
    dataset.gen_subset=${subset} \
    common_eval.path=${ckpt_dir}/${ckpt_name} \
    common_eval.results_path=${results_path} \
    override.modalities=['video'] \
    override.data=${data_dir} \
    override.label_dir=${label_dir} \
    override.beam_size=${beam_size} \
    ${extra_args} \
    common.user_dir=${root}/avhubert \
    dataset.max_tokens=3000 \
    dataset.num_workers=0

echo "推理完成！"
echo "结果保存在: ${results_path}"

