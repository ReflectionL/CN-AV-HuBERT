#!/bin/bash

root=/home/lixiaolou/code/CN-AV-HuBERT
exp_path=/hdd2/avhubert_exp
checkpoint_path=${exp_path}/finetune/clean_pt_cncvs3_iter2_ft_cnvsrc_vsr/checkpoints/checkpoint_best.pt
results_path=${exp_path}/infer/clean_pt_cncvs3_iter2_ft_cnvsrc_vsr

# 创建结果目录
mkdir -p ${results_path}

# 设置环境变量
export PYTHONPATH="/home/lixiaolou/code/CN-AV-HuBERT/fairseq:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0

echo "开始推理测试..."
echo "使用checkpoint: ${checkpoint_path}"
echo "结果保存到: ${results_path}"

cd ${root}

python avhubert/infer_s2s.py \
    --config-dir ${root}/avhubert/conf \
    --config-name infer \
    common.user_dir=${root}/avhubert \
    common.log_format=simple \
    common.log_interval=1 \
    common_eval.path=${checkpoint_path} \
    common_eval.results_path=${results_path} \
    common_eval.quiet=false \
    override.data=${root}/avhubert/bpe/cnvsrc/ \
    override.label_dir=${root}/avhubert/bpe/cnvsrc/ \
    override.modalities='["video"]' \
    dataset.gen_subset=valid \
  

echo "推理完成！结果保存在: ${results_path}"
echo "查看WER结果: cat ${results_path}/wer.*"
