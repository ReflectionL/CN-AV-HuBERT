root=/home/lixiaolou/code/CN-AV-HuBERT
exp_path=/hdd2/avhubert_exp

export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export NCCL_P2P_DISABLE=1       # 禁用 P2P 通信
export NCCL_IB_DISABLE=1        # 禁用 InfiniBand
export MASTER_ADDR=localhost    # 设置主节点地址
export MASTER_PORT=29671      # 设置主节点端口

export PYTHONPATH="/home/lixiaolou/code/CN-AV-HuBERT/fairseq:${PYTHONPATH:-}"

# 使用改进的配置文件
python -m fairseq_cli.hydra_train \
    --config-dir ${root}/avhubert/conf/finetune \
    --config-name base_cnvsrc_ctc.yaml \
    task.data=${root}/avhubert/bpe/cnvsrc/ \
    task.label_dir=${root}/avhubert/bpe/cnvsrc/ \
    task.tokenizer_bpe_name=char_tokenizer \
    +task.char_dict_path=/home/lixiaolou/code/CN-AV-HuBERT/avhubert/bpe/char_units.txt \
    common.user_dir=${root}/avhubert \
    distributed_training.distributed_port=29671 \
    distributed_training.distributed_init_method="tcp://localhost:29671" \
    model.w2v_path=${exp_path}/pretrain/clean_pt_cncvs3_speech_iter4/checkpoints/checkpoint_last.pt \
    model.freeze_finetune_updates=0 \
    hydra.run.dir=${exp_path}/probe_test/pt_cncvs3_iter4_ft_cnvsrc_vsr_ctc_1h \
    dataset.train_subset=train_1h \
    optimization.max_update=400 \
    lr_scheduler.warmup_steps=10 \
    lr_scheduler.decay_steps=300 
    # +model.no_pretrained_weights=true
