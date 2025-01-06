## paths
tsv_dir=/work104/cchen/data/audio-visual/avhubert_pretrain_data/filelist/pavsr/
model_root=/work104/cchen/VTS/ReVISE/pavsr/av_hubert/avhubert/experiment/pretrain/
feature_dir=/work104/cchen/data/audio-visual/avhubert_pretrain_data/avhubert_feature/
km_dir=/work104/cchen/data/audio-visual/avhubert_pretrain_data/avhubert_km/
label_dir=/work104/cchen/data/audio-visual/avhubert_pretrain_data/avhubert_label/

## extract feature
mkdir ${feature_dir}/iter3_L12/
for split in "train" "valid" "test";
do echo $split;
    # for i in `seq 0 3`;
    # do
    #     CUDA_VISIBLE_DEVICES=5 python dump_hubert_feature.py \
    #     ${tsv_dir} \
    #     $split \
    #     ${model_root}/clean_pt_cncvs_speech_iter2/checkpoints/checkpoint_best.pt \
    #     12 \
    #     4 \
    #     ${i} \
    #     ${feature_dir}/iter3_L12/ \
    #     --user_dir `pwd`/../ &
    # done
    CUDA_VISIBLE_DEVICES=2 python dump_hubert_feature.py \
        ${tsv_dir} \
        $split \
        ${model_root}/clean_pt_cncvs_speech_iter3/checkpoints/checkpoint_best.pt \
        12 \
        4 \
        0 \
        ${feature_dir}/iter3_L12/ \
        --user_dir `pwd`/../ &
    CUDA_VISIBLE_DEVICES=3 python dump_hubert_feature.py \
        ${tsv_dir} \
        $split \
        ${model_root}/clean_pt_cncvs_speech_iter2/checkpoints/checkpoint_best.pt \
        12 \
        4 \
        1 \
        ${feature_dir}/iter3_L12/ \
        --user_dir `pwd`/../ &
    CUDA_VISIBLE_DEVICES=4 python dump_hubert_feature.py \
        ${tsv_dir} \
        $split \
        ${model_root}/clean_pt_cncvs_speech_iter2/checkpoints/checkpoint_best.pt \
        12 \
        4 \
        2 \
        ${feature_dir}/iter3_L12/ \
        --user_dir `pwd`/../ &
    CUDA_VISIBLE_DEVICES=5 python dump_hubert_feature.py \
        ${tsv_dir} \
        $split \
        ${model_root}/clean_pt_cncvs_speech_iter2/checkpoints/checkpoint_best.pt \
        12 \
        4 \
        3 \
        ${feature_dir}/iter3_L12/ \
        --user_dir `pwd`/../ &
    wait
done

## Learn K-means clustering model
mkdir ${km_dir}/iter3_L12/
CUDA_VISIBLE_DEVICES=5 python learn_kmeans.py \
${feature_dir}/iter3_L12/ \
train \
4 \
${km_dir}/iter3_L12/train_1000.bin \
1000 \
--percent -1

## Apply K-means clustering to get label
mkdir ${label_dir}/iter3_L12/;
for split in "train" "test" "valid";
do echo $split;
    for i in `seq 0 3`;
    do
        CUDA_VISIBLE_DEVICES=5 python dump_km_label.py \
        ${feature_dir}/iter3_L12/ \
        $split \
        ${km_dir}/iter3_L12/train_1000.bin \
        4 \
        $i \
        ${label_dir}/iter3_L12/;
    done
done

## merge km labels
for rank in $(seq 0 3); do
    cat ${label_dir}iter3_L12/train_${rank}_4.km
done > ${label_dir}iter3_L12/train.km
for rank in $(seq 0 3); do
    cat ${label_dir}iter3_L12/valid_${rank}_4.km
done > ${label_dir}iter3_L12/valid.km
for rank in $(seq 0 3); do
    cat ${label_dir}iter3_L12/test_${rank}_4.km
done > ${label_dir}iter3_L12/test.km

## generate dict.km.txt
for i in $(seq 0 999);do
    echo $i 10000
done > ${label_dir}/iter3_L12/dict.km.txt
