root=/home/lixiaolou/code/CN-AV-HuBERT
label_path=/home/lixiaolou/data/avhubert_pretrain_data

echo 'root path' ${root}
echo 'label path' ${label_path}

export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=4,5

# export NCCL_P2P_DISABLE=1       # 禁用 P2P 通信
# export NCCL_IB_DISABLE=1        # 禁用 InfiniBand

export PYTHONPATH="/home/lixiaolou/code/CN-AV-HuBERT/fairseq:${PYTHONPATH:-}"

fairseq-hydra-train \
    --config-dir ${root}/avhubert/conf/pretrain \
    --config-name base_cncvs_iter1.yaml \
    task.data=${root}/data/cnvsrc-multi/ \
    task.label_dir=${label_path}/avhubert_label/mfcc \
    hydra.run.dir=${root}/exp/pretrain/clean_pt_cncvs_speech_iter1/ \
    common.user_dir=${root}/avhubert \
    distributed_training.distributed_port=29671 \
    distributed_training.distributed_init_method="tcp://localhost:29671"
