root=/home/lixiaolou/code/CN-AV-HuBERT
exp_path=/hdd2/avhubert_exp

export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export NCCL_P2P_DISABLE=1       # 禁用 P2P 通信
export NCCL_IB_DISABLE=1        # 禁用 InfiniBand
export MASTER_ADDR=localhost    # 设置主节点地址
export MASTER_PORT=29674        # 设置主节点端口

export PYTHONPATH="/home/lixiaolou/code/CN-AV-HuBERT/fairseq:${PYTHONPATH:-}"

# Seq2Seq Probe Test 配置（冻结 encoder）
python -m fairseq_cli.hydra_train \
    --config-dir ${root}/avhubert/conf/finetune \
    --config-name base_cnvsrc_s2s.yaml \
    task.data=${root}/avhubert/bpe/cnvsrc/ \
    task.label_dir=${root}/avhubert/bpe/cnvsrc/ \
    task.tokenizer_bpe_name=char_tokenizer \
    +task.char_dict_path=/home/lixiaolou/code/CN-AV-HuBERT/avhubert/bpe/char_units.txt \
    hydra.run.dir=${exp_path}/finetune/probe_pt_cncvs3_iter4_ft_cnvsrc_vsr_s2s_10h \
    common.user_dir=${root}/avhubert \
    distributed_training.distributed_port=29674 \
    distributed_training.distributed_init_method="tcp://localhost:29674" \
    model.w2v_path=${exp_path}/pretrain/clean_pt_cncvs3_speech_iter4/checkpoints/checkpoint_best.pt \
    model.freeze_finetune_updates=1000 \
    optimization.max_update=2200 \
    lr_scheduler.warmup_steps=500 \
    lr_scheduler.decay_steps=2500 \
    dataset.train_subset=train_10h
    # +model.no_pretrained_weights=true

