#!/bin/bash
#===============================================================================
# Filename: run_di_cifar10_reproduce_kefeng.sh 
# Created on Thu Jun 24 07:29:36 UTC 2021
#
# @author: Kefeng Zhu 
#   run deepinversion on cifar10, reproduced version, with parameters
#===============================================================================

#--------------------------------------------------------------------------------
# TODO: 
#   must update the "save_dir" to a proper path
#   may update teh "exp_descr", which define the current experiment name. it's a part of the saving subfolder's name
#
#   the pretrained model's weight is in the current folder: "teacher-resnet34-9567.weights"
#
#   The "dataset" is only used for testing acc of the pretrained model, if you comment out the test() in the "__main__",
#   you don't need to set the "dataset" parameter. 
#   If you're going to use it, update the "dataset" parameter to the path where your cifar10 dataset locates:
#       it should be the path to the folder CONTAINING "cifar-10-batches-py" folder!
#--------------------------------------------------------------------------------

python di_cifar10_reproduce_kefeng.py \
--save_dir="./results" \
--exp_descr="best_Hps_for_IS_non_grid_V1" \
--bs=500 \
--di_lr=0.05 \
--alpha_f=10 \
--teacher_weights="./teacher-resnet34-9567.weights" \
--dataset="/data/cifar10_dataset" \
--device="gpu" \

#--teacher_weights="/data/pretrained_zoo/pytorch/teacher-resnet34-9567.weights" \
