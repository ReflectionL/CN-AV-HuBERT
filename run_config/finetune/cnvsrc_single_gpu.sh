root=/home/lixiaolou/code/CN-AV-HuBERT
exp_path=/hdd2/avhubert_exp

# 单GPU训练版本，用于测试配置
CUDA_VISIBLE_DEVICES=0 \
fairseq-hydra-train \
    --config-dir ${root}/avhubert/conf/finetune \
    --config-name base_cnvsrc.yaml \
    task.data=${root}/avhubert/bpe/cnvsrc/ \
    task.label_dir=${root}/avhubert/bpe/cnvsrc/ \
    task.tokenizer_bpe_name=char_tokenizer \
    +task.char_dict_path=/home/lixiaolou/code/CN-AV-HuBERT/avhubert/bpe/char_units.txt \
    model.w2v_path=${exp_path}/pretrain/clean_pt_cncvs3_speech_iter2/checkpoints/checkpoint_best.pt \
    hydra.run.dir=${exp_path}/finetune/clean_pt_cncvs3_iter2_ft_cnvsrc_vsr_single \
    common.user_dir=${root}/avhubert \
    distributed_training.distributed_world_size=1 \
    distributed_training.nprocs_per_node=1