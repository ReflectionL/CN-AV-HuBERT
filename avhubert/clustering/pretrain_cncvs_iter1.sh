code_path = '/home/lixiaolou/code/CN-AV-HuBERT'
label_path_root = '/hdd1/avhubert_processed_data'
dataset = 'cncvs3'
label_path = f'{label_path_root}/{dataset}'

echo 'extract MFCC feature'
CUDA_VISIBLE_DEVICES=4 python dump_mfcc_feature.py \
    ${code_path}/data/${dataset}/ \
    test \
    1 \
    0 \
    ${label_path}/avhubert_feature/mfcc/

CUDA_VISIBLE_DEVICES=4 python dump_mfcc_feature.py \
    ${code_path}/data/${dataset}/ \
    valid \
    1 \
    0 \
    ${label_path}/avhubert_feature/mfcc/

# CUDA_VISIBLE_DEVICES=4 python dump_mfcc_feature.py \
#     ${code_path}/data/${dataset}/ \
#     train \
#     1 \
#     0 \
#     ${label_path}/avhubert_feature/mfcc/


# echo 'Learn K-means clustering model'
# python learn_kmeans.py \
#     ${label_path}/avhubert_feature/mfcc/ \
#     train \
#     1 \
#     ${label_path}/avhubert_km/mfcc/train_100.bin \
#     100 \
#     --percent 1

# echo 'Apply K-means clustering to get label'
# python dump_km_label.py \
#     ${label_path}/avhubert_feature/mfcc/ \
#     train \
#     ${label_path}/avhubert_km/mfcc/train_100.bin \
#     1 \
#     0 \
#     ${label_path}/avhubert_label/mfcc/
# python dump_km_label.py \
#     ${label_path}/avhubert_feature/mfcc/ \
#     valid \
#     ${label_path}/avhubert_km/mfcc/train_100.bin \
#     1 \
#     0 \
#     ${label_path}/avhubert_label/mfcc/
# python dump_km_label.py \
#     ${label_path}/avhubert_feature/mfcc/ \
#     test \
#     ${label_path}/avhubert_km/mfcc/train_100.bin \
#     1 \
#     0 \
#     ${label_path}/avhubert_label/mfcc/

# echo 'merge km labels'
# for rank in $(seq 0 0); do
#   cat ${label_path}/avhubert_label/mfcc/train_${rank}_1.km
# done > ${label_path}/avhubert_label/mfcc/train.km
# for rank in $(seq 0 0); do
#   cat ${label_path}/avhubert_label/mfcc/valid_${rank}_1.km
# done > ${label_path}/avhubert_label/mfcc/valid.km
# for rank in $(seq 0 0); do
#   cat ${label_path}/avhubert_label/mfcc/test_${rank}_1.km
# done > ${label_path}/avhubert_label/mfcc/test.km

echo 'generate dict.km.txt'
for i in $(seq 1 99);do
    echo $i 10000
done > ${label_path}/avhubert_label/mfcc/dict.mfcc.txt
for i in $(seq 1 99);do
    echo $i 10000
done > ${label_path}/avhubert_label/mfcc/dict.km.txt
