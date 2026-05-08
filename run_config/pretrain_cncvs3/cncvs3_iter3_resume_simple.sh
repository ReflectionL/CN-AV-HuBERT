 #!/bin/bash

root=/home/lixiaolou/code/CN-AV-HuBERT
label_path=/hdd1/avhubert_processed_data/cncvs3
exp_path=/hdd2/avhubert_exp/pretrain/clean_pt_cncvs3_speech_iter3_resume
checkpoint_path=/hdd2/avhubert_exp/pretrain/clean_pt_cncvs3_speech_iter3/checkpoints/checkpoint_last.pt

echo 'root path' ${root}
echo 'label path' ${label_path}
echo 'exp path' ${exp_path}
echo 'checkpoint path' ${checkpoint_path}

export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export NCCL_P2P_DISABLE=1       # 禁用 P2P 通信
export NCCL_IB_DISABLE=1        # 禁用 InfiniBand

export PYTHONPATH="/home/lixiaolou/code/CN-AV-HuBERT/fairseq:${PYTHONPATH:-}"

python -m fairseq_cli.hydra_train \
    --config-dir ${root}/avhubert/conf/pretrain_cncvs3 \
    --config-name base_cncvs3_iter3.yaml \
    task.data=${root}/data/cncvs3/ \
    task.label_dir=${label_path}/avhubert_label/iter2_L12 \
    hydra.run.dir=${exp_path} \
    common.user_dir=${root}/avhubert \
    distributed_training.distributed_port=29673 \
    distributed_training.distributed_init_method="tcp://localhost:29673" \
    checkpoint.restore_file=${checkpoint_path} \
    optimization.lr=[0.0005] \
    lr_scheduler.warmup_updates=5000