## paths
root=/home/lixiaolou/code/CN-AV-HuBERT/
tsv_dir=/home/lixiaolou/code/CN-AV-HuBERT/data/cnvsrc-multi/
model_root=/home/lixiaolou/code/CN-AV-HuBERT/exp/pretrain/
feature_dir=/home/lixiaolou/data/avhubert_pretrain_data/avhubert_feature/
km_dir=/home/lixiaolou/data/avhubert_pretrain_data/avhubert_km/
label_dir=/home/lixiaolou/data/avhubert_pretrain_data/avhubert_label/

## extract feature
# for split in "train" "valid" "test";
# do echo $split;
#     for i in `seq 0 3`;
#     do
#         CUDA_VISIBLE_DEVICES=4 python dump_hubert_feature.py \
#         ${tsv_dir} \
#         $split \
#         ${model_root}/clean_pt_cncvs_speech_iter1/checkpoints/checkpoint_best.pt \
#         9 \
#         4 \
#         ${i} \
#         ${feature_dir}/iter1_L9/ \
#         --user_dir ${root}avhubert/ &
#     done
#     wait
# done

## Learn K-means clustering model
# mkdir ${feature_dir}/iter1_L9/
# CUDA_VISIBLE_DEVICES=4 python learn_kmeans.py \
# ${feature_dir}/iter1_L9/ \
# train \
# 4 \
# ${km_dir}/iter1_L9/train_100.bin \
# 100 \
# --percent -1

# echo 'Apply K-means clustering to get label'
# mkdir ${label_dir}/iter1_L9/;
# for split in "test" "train" "valid";
# do echo $split;
#     for i in `seq 0 3`;
#     do
#         python dump_km_label.py \
#         ${feature_dir}/iter1_L9/ \
#         $split \
#         ${km_dir}/iter1_L9/train_100.bin \
#         4 \
#         $i \
#         ${label_dir}/iter1_L9/;
#     done
# done

echo 'merge km labels'
for rank in $(seq 0 3); do
    cat ${label_dir}iter1_L9/train_${rank}_4.km
done > ${label_dir}iter1_L9/train.km
for rank in $(seq 0 3); do
    cat ${label_dir}iter1_L9/valid_${rank}_4.km
done > ${label_dir}iter1_L9/valid.km
for rank in $(seq 0 3); do
    cat ${label_dir}iter1_L9/test_${rank}_4.km
done > ${label_dir}iter1_L9/test.km

echo 'generate dict.km.txt'
for i in $(seq 0 99);do
    echo $i 10000
done > ${label_dir}/iter1_L9/dict.km.txt
