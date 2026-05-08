label_path_root=/hdd1/avhubert_processed_data
code_root=/home/lixiaolou/code/CN-AV-HuBERT
data_root=/ssd1/processed

dataset=cncvs3
# split=test
num_shard=0
total_shards=1


label_path=${label_path_root}/${dataset}

# echo 'Step 1: Extract MFCC feature'

# echo 'extract MFCC feature for' ${dataset} ${split} ${num_shard} ${total_shards}
# CUDA_VISIBLE_DEVICES=0 python dump_mfcc_feature.py \
#     ${code_root}/data/${dataset}/ \
#     test \
#     ${total_shards} \
#     ${num_shard} \
#     ${label_path}/avhubert_feature/mfcc/

# echo 'extract MFCC feature for '${dataset}' valid' ${num_shard} ${total_shards}
# CUDA_VISIBLE_DEVICES=0 python dump_mfcc_feature.py \
#     ${code_root}/data/${dataset}/ \
#     valid \
#     ${total_shards} \
#     ${num_shard} \
#     ${label_path}/avhubert_feature/mfcc/

# echo 'extract MFCC feature for' ${dataset} 'train' ${num_shard} ${total_shards}
# CUDA_VISIBLE_DEVICES=0 python dump_mfcc_feature.py \
#     ${code_root}/data/${dataset}/ \
#     train \
#     ${total_shards} \
#     ${num_shard} \
#     ${label_path}/avhubert_feature/mfcc/

# for i in {30..35}
# do
#     echo 'extract MFCC feature for' ${dataset} 'train' ${i} 36
#     CUDA_VISIBLE_DEVICES=5 python dump_mfcc_feature.py \
#         ${code_root}/data/${dataset}/ \
#         train \
#         36 \
#         ${i} \
#         ${label_path}/avhubert_feature/mfcc/ \
#         > ${code_root}/data/${dataset}/train_${i}_36.log 2>&1 &
# done


# echo 'Step 2: Learn K-means clustering model'
# python learn_kmeans.py \
#     ${label_path}/avhubert_feature/mfcc/ \
#     train \
#     36 \
#     ${label_path}/avhubert_km/mfcc/train_100.bin \
#     100 \
#     --percent 1

# echo 'Step 3: Apply K-means clustering to get label'
# for i in {0..35}
# do
#     echo 'apply km label for train' ${i} 36
#     CUDA_VISIBLE_DEVICES=0 python dump_km_label.py \
#         ${label_path}/avhubert_feature/mfcc/ \
#         train \
#         ${label_path}/avhubert_km/mfcc/train_100.bin \
#         36 \
#         ${i} \
#         ${label_path}/avhubert_label/mfcc/ \
#         > ${code_root}/data/${dataset}/train_${i}_36.log 2>&1 &
# done

# python dump_km_label.py \
#     ${label_path}/avhubert_feature/mfcc/ \
#     train \
#     ${label_path}/avhubert_km/mfcc/train_100.bin \
#     1 \
#     0 \
#     ${label_path}/avhubert_label/mfcc/

# echo 'apply km label for valid'
# python dump_km_label.py \
#     ${label_path}/avhubert_feature/mfcc/ \
#     valid \
#     ${label_path}/avhubert_km/mfcc/train_100.bin \
#     1 \
#     0 \
#     ${label_path}/avhubert_label/mfcc/

# echo 'apply km label for test'
# python dump_km_label.py \
#     ${label_path}/avhubert_feature/mfcc/ \
#     test \
#     ${label_path}/avhubert_km/mfcc/train_100.bin \
#     1 \
#     0 \
#     ${label_path}/avhubert_label/mfcc/

# echo 'Step 4: merge km labels'
# for rank in $(seq 0 35); do
#   cat ${label_path}/avhubert_label/mfcc/train_${rank}_36.km
# done > ${label_path}/avhubert_label/mfcc/train.km

# for rank in $(seq 0 0); do
#   cat ${label_path}/avhubert_label/mfcc/valid_${rank}_1.km
# done > ${label_path}/avhubert_label/mfcc/valid.km

# for rank in $(seq 0 0); do
#   cat ${label_path}/avhubert_label/mfcc/test_${rank}_1.km
# done > ${label_path}/avhubert_label/mfcc/test.km

echo 'Step 5: generate dict.km.txt'
for i in $(seq 0 99);do
    echo $i 10000
done > ${label_path}/avhubert_label/mfcc/dict.mfcc.txt
for i in $(seq 0 99);do
    echo $i 10000
done > ${label_path}/avhubert_label/mfcc/dict.km.txt