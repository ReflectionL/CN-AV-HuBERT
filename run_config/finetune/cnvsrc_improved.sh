root=/home/lixiaolou/code/CN-AV-HuBERT
exp_path=/hdd2/avhubert_exp

export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export NCCL_P2P_DISABLE=1       # 禁用 P2P 通信
export NCCL_IB_DISABLE=1        # 禁用 InfiniBand
export MASTER_ADDR=localhost    # 设置主节点地址
export MASTER_PORT=29674        # 设置主节点端口

export PYTHONPATH="/home/lixiaolou/code/CN-AV-HuBERT/fairseq:${PYTHONPATH:-}"

# 使用改进的配置文件
python -m fairseq_cli.hydra_train \
    --config-dir ${root}/avhubert/conf/finetune \
    --config-name large_cnvsrc_v3.yaml \
    task.data=${root}/avhubert/bpe/cnvsrc/ \
    task.label_dir=${root}/avhubert/bpe/cnvsrc/ \
    task.tokenizer_bpe_name=char_tokenizer \
    +task.char_dict_path=/home/lixiaolou/code/CN-AV-HuBERT/avhubert/bpe/char_units.txt \
    model.w2v_path=${exp_path}/pretrain/clean_pt_cncvs3_speech_iter5/checkpoints/checkpoint_best.pt \
    hydra.run.dir=${exp_path}/finetune/clean_pt_cncvs3_iter5_ft_cnvsrc_vsr_v3 \
    common.user_dir=${root}/avhubert \
    distributed_training.distributed_port=29671 \
    distributed_training.distributed_init_method="tcp://localhost:29671"
