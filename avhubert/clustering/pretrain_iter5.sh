label_path_root=/hdd1/avhubert_processed_data
code_root=/home/lixiaolou/code/CN-AV-HuBERT
data_root=/ssd1/processed
training_root=/hdd2/avhubert_exp

dataset=cncvs3
# split=test
num_shard=0
total_shards=36


label_path=${label_path_root}/${dataset}
tsv_dir=${code_root}/data/${dataset}
model_root=${training_root}/pretrain/clean_pt_cncvs3_speech_iter4
feature_dir=${label_path}/avhubert_feature/iter4_L12
km_dir=${label_path}/avhubert_km/iter4_L12
label_dir=${label_path}/avhubert_label/iter4_L12

# echo 'Step 1: extract avhubertfeature'
# # extract feature

# echo 'extract MFCC feature for ' ${dataset} 'test' '0' '1'
# CUDA_VISIBLE_DEVICES=0 python dump_hubert_feature.py \
#     ${tsv_dir}/ \
#     test \
#     ${model_root}/checkpoints/checkpoint_best.pt \
#     12 \
#     1 \
#     0 \
#     ${feature_dir}/ \
#     --user_dir ${code_root}/avhubert/

# echo 'extract MFCC feature for ' ${dataset} 'valid' '0' '1'
# CUDA_VISIBLE_DEVICES=0 python dump_hubert_feature.py \
#     ${tsv_dir}/ \
#     valid \
#     ${model_root}/checkpoints/checkpoint_best.pt \
#     12 \
#     1 \
#     0 \
#     ${feature_dir}/ \
#     --user_dir ${code_root}/avhubert/

# echo 'extract MFCC feature for ' ${dataset} 'train'  ${total_shards}
# for i in {0..35}
# do
#     cuda_device=$((i%6))
#     echo 'extract MFCC feature for ' ${dataset} 'train' ${i} ${total_shards} 'on cuda_device' ${cuda_device}
#     CUDA_VISIBLE_DEVICES=${cuda_device} python dump_hubert_feature.py \
#         ${tsv_dir}/ \
#         train \
#         ${model_root}/checkpoints/checkpoint_best.pt \
#         12 \
#         36 \
#         ${i} \
#         ${feature_dir}/ \
#         --user_dir ${code_root}/avhubert/ \
#         > ${code_root}/data/${dataset}/train_${i}_36.log 2>&1 &
# done


# export OMP_NUM_THREADS=8
# export OPENBLAS_NUM_THREADS=8
# export MKL_NUM_THREADS=8
# export NUMEXPR_NUM_THREADS=8
# export VECLIB_MAXIMUM_THREADS=8
# export BLIS_NUM_THREADS=8

# echo 'Step 2: Learn K-means clustering model'
# mkdir ${feature_dir}/
# CUDA_VISIBLE_DEVICES=4 python learn_kmeans.py \
#     ${feature_dir}/ \
#     train \
#     36 \
#     ${km_dir}/train_2000.bin \
#     2000 \
#     --percent 0.1


# echo 'Step 3: Apply K-means clustering to get label'
# echo 'apply km label for valid'
# python dump_km_label.py \
#     ${feature_dir}/ \
#     valid \
#     ${km_dir}/train_2000.bin \
#     1 \
#     0 \
#     ${label_dir}/;

# echo 'apply km label for test'
# python dump_km_label.py \
#     ${feature_dir}/ \
#     test \
#     ${km_dir}/train_2000.bin \
#     1 \
#     0 \
#     ${label_dir}/;

# echo 'apply km label for train'
# for i in {0..35}
# do
#     echo 'apply km label for train' ${i} 36
#     CUDA_VISIBLE_DEVICES=0 python dump_km_label.py \
#     ${feature_dir}/ \
#     train \
#     ${km_dir}/train_2000.bin \
#     36 \
#     ${i} \
#     ${label_dir}/
# done

echo 'Step 4: merge km labels'
for rank in $(seq 0 35); do
    cat ${label_dir}/train_${rank}_36.km
done > ${label_dir}/train.km

for rank in $(seq 0 0); do
cat ${label_dir}/valid_0_1.km
done > ${label_dir}/valid.km

for rank in $(seq 0 0); do
    cat ${label_dir}/test_0_1.km
done > ${label_dir}/test.km

echo 'Step 5: generate dict.km.txt'
for i in $(seq 0 1999);do
    echo $i 10000
done > ${label_dir}/dict.km.txt
