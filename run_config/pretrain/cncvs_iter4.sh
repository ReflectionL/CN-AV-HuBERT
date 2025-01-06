CUDA_VISIBLE_DEVICES=2,3,4,5 \
fairseq-hydra-train \
    --config-dir /work104/cchen/VTS/ReVISE/pavsr/av_hubert/avhubert/conf/pretrain \
    --config-name base_cncvs_iter4.yaml \
    task.data=/work104/cchen/data/audio-visual/avhubert_pretrain_data/filelist/pavsr/ \
    task.label_dir=/work104/cchen/data/audio-visual/avhubert_pretrain_data/avhubert_label/iter3_L12 \
    hydra.run.dir=/work104/cchen/VTS/ReVISE/pavsr/av_hubert/avhubert/experiment/pretrain/clean_pt_cncvs_speech_iter4/ \
    common.user_dir=`pwd`
