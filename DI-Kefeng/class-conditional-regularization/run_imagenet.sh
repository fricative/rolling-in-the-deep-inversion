#!/bin/bash
DATE_PREFIX=`date +"%Y%m%d%H%M"`
#bs=16
bs=64
#lr=0.25
lr=0.25
r_feature=0.01
#labeled_reg_feature=0.01
#target_label=130       # flamingo 火烈鸟
#target_label=933       # cheeseburger 芝士汉堡

# target label range:
START=350           # label start from, if not set, start from 0
#END=0              # label end at, if not set, end at 999

#setting_id=0
#setting_id=1
setting_id=2

#di_var_scale=0.01
#di_l2_scale=0.1

#exp_name="$DATE_PREFIX-rn50-bs$bs-lr$lr-label$target_label"
exp_name="$DATE_PREFIX-rn50-labeled-reg-batch-gen-bs$bs-lr$lr-label$target_label-setting_id$setting_id"

if [ $r_feature ]; then
    if [ $r_feature != 0 ]; then
        exp_name="$exp_name-r_feature$r_feature"
    fi
fi

if [ $labeled_reg_feature ]; then
    if [ $labeled_reg_feature != 0 ]; then
        exp_name="$exp_name-labeled_feature$labeled_reg_feature"
    fi
fi

python -u imagenet_inversion_extension.py \
--bs=$bs \
--lr=${lr:-"0.25"} \
--do_flip \
--exp_name=$exp_name \
--arch_name="resnet50" \
--adi_scale=0.0 \
--setting_id=$setting_id \
--save_dir="/workspace/results" \
--labeled_stats="/workspace/stats/imagenet/resnet50_mean_var_all_classes_val.npy" \
--r_feature=${r_feature:-"0.0"} \
--labeled_reg_feature=${labeled_reg_feature:-"0.0"} \
--store_best_images \
--gen_all_labels \
--fp16 \
--start=${START:-"0"} \
--end=${END:-"999"} \

#--verifier \
#--verifier_arch="resnet50" \
#--labeled_stats="/data/ImageNet2012/get_mean_var/resnet50_mean_var_all_classes.npy" \
#--target_label=$target_label \
#--fp16 \
