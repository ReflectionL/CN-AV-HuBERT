

model_root=/work104/cchen/VTS/ReVISE/pavsr/av_hubert/pretrained_model/
feature_dir=/work104/cchen/data/audio-visual/avhubert_pretrain_data/avhubert_feature/
mkdir ${feature_dir}/clean_pt_large_vox_iter5_L24/;

for split in "train" "valid" "test";
do echo $split;
    CUDA_VISIBLE_DEVICES=4 python dump_hubert_feature.py \
        /work104/cchen/data/audio-visual/LRS3/30h_data/ \
        $split \
        ${model_root}/clean_pt_large_vox_iter5.pt \
        24 \
        1 \
        0 \
        ${feature_dir}/clean_pt_large_vox_iter5_L24/ \
        --user_dir `pwd`/../
done
