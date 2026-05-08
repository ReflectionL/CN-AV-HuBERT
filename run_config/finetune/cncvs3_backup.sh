root=/home/lixiaolou/code/CN-AV-HuBERT
exp_path=/hdd2/avhubert_exp

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
fairseq-hydra-train \
    --config-dir ${root}/avhubert/conf/finetune \
    --config-name base_cnvsrc.yaml \
    task.data=${root}/data/cnvsrc/ \
    task.label_dir=${root}/data/cnvsrc/ \
    task.tokenizer_bpe_name=char_tokenizer \
    model.w2v_path=${exp_path}/pretrain/clean_pt_cncvs3_speech_iter2/checkpoints/checkpoint_best.pt \
    hydra.run.dir=${exp_path}/finetune/clean_pt_cnvsrc_speech_iter2_ft \
    common.user_dir=${root}/avhubert