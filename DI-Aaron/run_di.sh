#!/bin/bash
#===============================================================================
# Filename: run_di_resnext101.sh 
# Created on Sun Jul 11 14:31:57 2021
# 
# @author: Aaron Reding
#   run deepinversion on ResNext101
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

# shell script is not working well on colab - things get done out of order for some reason.
i="1"
while [ $i -lt 53 ]
do
python imagenet_inversion.py \
--exp_name="testing_allt_si0_layer_1" \
--bs=42 \
--arch_name="resnet50" \
--store_best_images \
--setting_id=0 \
--all_targets \
--specified_layer=1
	i=$[$i+1]
done


#--targets="1,25,63,92,94,107,151,154,207,250,270,277,283,292" \

#--stop_layer=10 \


#--------------------------------------------------------------------------------
# NOTES:
#   - tried resnet152 with 20000 epochs and batch size 126 but hit memory limit after 2000 epochs
#   - next run with resnet152, 2000 epochs, bs 126.
#       MEMORY ERROR AGAIN!!! loss_r_feature was under 20
#   - resnet152, 2000 epochs, bs 42.
#	  	why is it going to 2100+ epochs? check the epochs parameter
#		at 2100 loss jumps from .91 to 4.3? then starts reducing again
#	    why? finished! interesting images in ar-ResNet152.
#   - okay the epochs param does nothing...
#   - resnext101 32x8d. batch size 42. finished no problems.
#	- setting_id seems to control the epochs
#	- tried resnet50_SIN with setting_id 0. weird images
#	- tried resnet50_combined_SIN_IN 
#	- try random_label
#	- should i be saving all images?
#
#
#	- New experiment. testing the new layers code. both functions working.
#	- 
#
#
#
#

#--------------------------------------------------------------------------------
# Possible arguments

#--targets: choose the classes of image you want saved. --random_label must be not be on for this to work. see imagenet_class_dict for class names/indices.
#        	must enter in this format (quotes with a comma between) "1,2,3,4,5,6"
#          
#          original defaults were:
#		   [1, 25, 63, 92, 94, 107, 151, 154, 207, 250, 270, 277, 283, 292, 294, 309, 311,
#           325, 340, 360, 386, 402, 403, 409, 417, 440, 468, 487, 530, 574,  590, 670, 762, 
#           817, 920, 933, 946, 949, 963, 967, 980, 985]
#--all_targets: if this tag is inserted, will override targets variable with all 1000 classes
# NOTE for stop_layer and specified_layer: there are 53 batchnorm layers in resnet50
# so valid values are 0-52.
#--stop_layer: choose at which layer to stop pulling batchnorm data. cumulative up til this point
#--specified_layer: choose a single batchnorm layer to pull data from. mutually exclusive with stop_layer
#
#

#'-s', '--worldsize', type=int, default=1, help='Number of processes participating in the job.'
#'--local_rank', '--rank', type=int, default=0, help='Rank of the current process.'
#'--adi_scale', type=float, default=0.0, help='Coefficient for Adaptive Deep Inversion'
#'--no-cuda', action='store_true'

#'--epochs', default=20000, type=int
#'--setting_id', default=0, type=int, help='settings for optimization: 0 - multi resolution, 1 - 2k iterations, 2 - 20k iterations'
#'--bs', default=64, type=int
#'--jitter', default=30, type=int
#'--comment', default='', type=str

#'--arch_name', default='resnet50', type=str, help='model name from torchvision (see below)'
#   see all architectures in torchvision below

# fp16 is not available without getting apex and AMP working
#'--fp16', action='store_true', help='use FP16 for optimization'

#'--exp_name', type=str, default='test', help='where to store experimental data'

#'--verifier', action='store_true', help='evaluate batch with another model'
#'--verifier_arch', type=str, default='mobilenet_v2', help = "arch name from torchvision models to act as a verifier"

#'--do_flip', action='store_true', help='apply flip during model inversion'
#'--random_label', action='store_true', help='generate random label for optimization'
#'--r_feature', type=float, default=0.05, help='coefficient for feature distribution regularization'
#'--first_bn_multiplier', type=float, default=10., help='additional multiplier on first bn layer of R_feature'
#'--tv_l1', type=float, default=0.0, help='coefficient for total variation L1 loss'
#'--tv_l2', type=float, default=0.0001, help='coefficient for total variation L2 loss'
#'--lr', type=float, default=0.2, help='learning rate for optimization'
#'--l2', type=float, default=0.00001, help='l2 loss on the image'
#'--main_loss_multiplier', type=float, default=1.0, help='coefficient for the main loss in optimization'
#'--store_best_images', action='store_true', help='save best images as separate files'
#--------------------------------------------------------------------------------
#	from https://github.com/rgeirhos/texture-vs-shape
#	ResNet50 trained on Stylized ImageNet or a mix of SIN and IN
#	
#	use filenames below with arch_name parameter
#
#	filenames
#	resnet50_combined_with_decay.pth.tar
#	resnet50_combined_SIN_IN.pth.tar
#	resnet50_SIN.pth.tar 
#--------------------------------------------------------------------------------
# some architectures in torchvision to try
# data from https://pytorch.org/vision/stable/models.html
# Model                             arch_name        Acc@1       Acc@5      Tried?
#--------------------------------------------------------------------------------

# AlexNet                           alexnet          56.522      79.066

# VGG-11 with batch normalization   vgg11_bn         70.370      89.810

# VGG-13 with batch normalization   vgg13_bn         71.586      90.374

# VGG-16 with batch normalization   vgg16_bn         73.360      91.516

# VGG-19 with batch normalization   vgg19_bn         74.218      91.842		Tried, failed to run, haven't dug in yet

# ResNet-18                         resnet18         69.758      89.078

# ResNet-34                         resnet34         73.314      91.420

# ResNet-50                         resnet50         76.130      92.862		Ran successfully

# ResNet-101                        resnet101        77.374      93.546

# ResNet-152                        resnet152        78.312      94.046		Ran successfully

# SqueezeNet 1.0                    squeezenet1_0    58.092      80.420

# SqueezeNet 1.1                    squeezenet1_1    58.178      80.624

# Densenet-121                      densenet121      74.434      91.972

# Densenet-169                      densenet169      75.600      92.806

# Densenet-201                      densenet201      76.896      93.370

# Densenet-161                      densenet161      77.138      93.560

# GoogleNet                         googlenet        69.778      89.530

# ResNeXt-50-32x4d                  resnext50_32x4d  77.618      93.698		Ran successfully

# ResNeXt-101-32x8d                 resnext101_32x8d 79.312      94.526		Tried, memory issue with minimum batch size of 42

# Wide ResNet-50-2                  wide_resnet50_2  78.468      94.086      

# Wide ResNet-101-2                 wide_resnet101_2 78.848      94.284      
#--------------------------------------------------------------------------------
