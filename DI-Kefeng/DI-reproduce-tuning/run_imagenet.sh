 
python -u imagenet_inversion.py \
--bs=64\
--do_flip \
--exp_name="rn50_inversion" \
--r_feature=0.01 \
--arch_name="resnet50" \
--verifier \
--verifier_arch="resnet50" \
--adi_scale=0.0 \
--setting_id=0 \
--lr 0.25 \
--fp16

#--bs=152 \
