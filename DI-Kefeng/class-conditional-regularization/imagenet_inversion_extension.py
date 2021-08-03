# --------------------------------------------------------
# Modified specifically for Class-Conditional Regularization
# by Kefeng Zhu
# On 2021.07.10 Mon 18:01:32 CST
# --------------------------------------------------------

from __future__ import division, print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import argparse
import torch
from torch import distributed, nn
import random
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torchvision import datasets, transforms

import numpy as np
from apex import amp
import os
import torchvision.models as models
from utils.utils import load_model_pytorch, distributed_is_initialized, create_folder
import time

random.seed(0)

def validate_one(input, target, model):
    """Perform validation on the validation set"""

    def accuracy(output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    with torch.no_grad():
        output = model(input)
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

    print("Verifier accuracy: ", prec1.item())


def run(args):
    torch.manual_seed(args.local_rank)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    if args.arch_name == "resnet50v15":
        from models.resnetv15 import build_resnet
        net = build_resnet("resnet50", "classic")
    else:
        print("loading torchvision model for inversion with the name: {}".format(args.arch_name))
        #net = models.__dict__[args.arch_name](pretrained=True)
        net = models.__dict__[args.arch_name]()

    net = net.to(device)

    ### load models
    if args.arch_name=="resnet50v15":
        print('==> Resuming ResNet50v15 from checkpoint..')
        path_to_model = "./models/resnet50v15/model_best.pth.tar"
        #path_to_model = "/data/pretrained_zoo/pytorch/resnet50-19c8e357.pth"
        load_model_pytorch(net, path_to_model, gpu_n=torch.cuda.current_device())
    else:
        print('==> Resuming ResNet50 from checkpoint..')
        path_to_model = "/data/pretrained_zoo/pytorch/resnet50-19c8e357.pth"
        checkpoint = torch.load(path_to_model)
        net.load_state_dict(checkpoint)

    net.to(device)
    net.eval()

    use_fp16 = args.fp16
    if use_fp16:
        net, _ = amp.initialize(net, [], opt_level="O2")

    # reserved to compute test accuracy on generated images by different networks
    net_verifier = None
    if args.verifier and args.adi_scale == 0:
        # if multiple GPUs are used then we can change code to load different verifiers to different GPUs
        if args.local_rank == 0:
            print("loading verifier: ", args.verifier_arch)
            net_verifier = models.__dict__[args.verifier_arch](pretrained=True).to(device)
            net_verifier.eval()

            if use_fp16:
                net_verifier = net_verifier.half()

    if args.adi_scale != 0.0:
        student_arch = "resnet18"
        net_verifier = models.__dict__[student_arch](pretrained=True).to(device)
        net_verifier.eval()

        if use_fp16:
            net_verifier, _ = amp.initialize(net_verifier, [], opt_level="O2")

        net_verifier = net_verifier.to(device)
        net_verifier.train()

        if use_fp16:
            for module in net_verifier.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval().half()

    #from deepinversion import DeepInversionClass
    from di_extension import DeepInversionClass

    exp_name = args.exp_name
    # final images will be stored here:
    #adi_data_path = "./final_images/%s"%exp_name
    adi_data_path = os.path.join(args.save_dir, "imagenet_generation", "final_images", exp_name)
    # temporal data and generations will be stored here
    #exp_name = "generations/%s"%exp_name
    #current_time = time.strftime("%Y%m%d%H%M", time.localtime())
    #exp_name = "generations_%s/%s"%(current_time,exp_name)
    exp_name = os.path.join(args.save_dir, "imagenet_generation", exp_name)
    #create_folder(exp_name)

    #--------------------------------------------------------------------------------
    # NEW METHOD
    #--------------------------------------------------------------------------------
    # load labeled original data stats 
    #2020.9.23 changed bu tong
    labeled_stats = None
    if os.path.exists(args.labeled_stats):
        labeled_stats = np.load(args.labeled_stats, allow_pickle=True).item()
        print("--- labeled_stats loaded !!! ---")
    # specify the target label
    target_label = args.target_label
    #================================================================================

    args.iterations = 2000
    args.start_noise = True
    # args.detach_student = False

    args.resolution = 224
    bs = args.bs
    jitter = 30

    parameters = dict()
    parameters["resolution"] = 224
    parameters["random_label"] = False
    parameters["start_noise"] = True
    parameters["detach_student"] = False
    parameters["do_flip"] = True

    parameters["do_flip"] = args.do_flip
    parameters["random_label"] = args.random_label
    parameters["store_best_images"] = args.store_best_images

    criterion = nn.CrossEntropyLoss()

    coefficients = dict()
    coefficients["r_feature"] = args.r_feature
    coefficients["first_bn_multiplier"] = args.first_bn_multiplier
    coefficients["tv_l1"] = args.tv_l1
    coefficients["tv_l2"] = args.tv_l2
    coefficients["l2"] = args.l2
    coefficients["lr"] = args.lr
    coefficients["main_loss_multiplier"] = args.main_loss_multiplier
    coefficients["adi_scale"] = args.adi_scale

    coefficients["labeled_reg_feature"] = args.labeled_reg_feature

    network_output_function = lambda x: x

    # check accuracy of verifier
    if args.verifier:
        hook_for_display = lambda x,y: validate_one(x, y, net_verifier)
    else:
        hook_for_display = None

    # changed by tong 2020.9.23
    #if not args.gen_all_labels:
    #    DeepInversionEngine = DeepInversionClass(net_teacher=net,
    #                                         final_data_path=adi_data_path,
    #                                         path=exp_name,
    #                                         parameters=parameters,
    #                                         setting_id=args.setting_id,
    #                                         bs = bs,
    #                                         use_fp16 = args.fp16,
    #                                         jitter = jitter,
    #                                         criterion=criterion,
    #                                         coefficients = coefficients,
    #                                         network_output_function = network_output_function,
    #                                         hook_for_display = hook_for_display,
    #                                         target_label=target_label, labeled_stats=labeled_stats)
    #    net_student=None
    #    if args.adi_scale != 0:
    #        net_student = net_verifier
    #    DeepInversionEngine.generate_batch(net_student=net_student)

    #else:
    #    # zkf - on Wed Sep 16 11:25:29 CST 2020
    #    # if gen_all_labels is true, loop in the range (0 - 999) and generate images for each label
    #    for target_label in range(1000):
    #        DeepInversionEngine = DeepInversionClass(net_teacher=net,
    #                                         final_data_path=adi_data_path,
    #                                         path=exp_name,
    #                                         parameters=parameters,
    #                                         setting_id=args.setting_id,
    #                                         bs = bs,
    #                                         use_fp16 = args.fp16,
    #                                         jitter = jitter,
    #                                         criterion=criterion,
    #                                         coefficients = coefficients,
    #                                         network_output_function = network_output_function,
    #                                         hook_for_display = hook_for_display,
    #                                         target_label=target_label, labeled_stats=labeled_stats)

    #        net_student=None
    #        if args.adi_scale != 0:
    #            net_student = net_verifier
    #        DeepInversionEngine.generate_batch(net_student=net_student)


    # zkf - on Wed Sep 16 11:25:29 CST 2020
    # if gen_all_labels is true, loop in the range (0 - 999) and generate images for each label
    DeepInversionEngine = DeepInversionClass(net_teacher=net,
                                     final_data_path=adi_data_path,
                                     path=exp_name,
                                     parameters=parameters,
                                     setting_id=args.setting_id,
                                     bs = bs,
                                     use_fp16 = args.fp16,
                                     jitter = jitter,
                                     criterion=criterion,
                                     coefficients = coefficients,
                                     network_output_function = network_output_function,
                                     hook_for_display = hook_for_display,
                                     target_label=target_label, labeled_stats=labeled_stats)

    net_student=None
    if args.adi_scale != 0:
        net_student = net_verifier
    DeepInversionEngine.generate_batch(net_student=net_student)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--worldsize', type=int, default=1, help='Number of processes participating in the job.')
    parser.add_argument('--local_rank', '--rank', type=int, default=0, help='Rank of the current process.')
    parser.add_argument('--adi_scale', type=float, default=0.0, help='Coefficient for Adaptive Deep Inversion')
    parser.add_argument('--no-cuda', action='store_true')

    #parser.add_argument('--epochs', default=20000, type=int, help='batch size')
    parser.add_argument('--setting_id', default=0, type=int, help='settings for optimization: 0 - multi resolution, 1 - 2k iterations, 2 - 20k iterations')
    parser.add_argument('--bs', default=64, type=int, help='batch size')
    parser.add_argument('--jitter', default=30, type=int, help='batch size')
    parser.add_argument('--comment', default='', type=str, help='batch size')
    parser.add_argument('--arch_name', default='resnet50', type=str, help='model name from torchvision or resnet50v15')

    parser.add_argument('--fp16', action='store_true', help='use FP16 for optimization')
    parser.add_argument('--exp_name', type=str, default='test', help='where to store experimental data')

    parser.add_argument('--verifier', action='store_true', help='evaluate batch with another model')
    parser.add_argument('--verifier_arch', type=str, default='mobilenet_v2', help = "arch name from torchvision models to act as a verifier")

    parser.add_argument('--do_flip', action='store_true', help='apply flip during model inversion')
    parser.add_argument('--random_label', action='store_true', help='generate random label for optimization')
    parser.add_argument('--r_feature', type=float, default=0.05, help='coefficient for feature distribution regularization')
    parser.add_argument('--first_bn_multiplier', type=float, default=10., help='additional multiplier on first bn layer of R_feature')
    parser.add_argument('--tv_l1', type=float, default=0.0, help='coefficient for total variation L1 loss')
    parser.add_argument('--tv_l2', type=float, default=0.0001, help='coefficient for total variation L2 loss')
    parser.add_argument('--lr', type=float, default=0.2, help='learning rate for optimization')
    parser.add_argument('--l2', type=float, default=0.00001, help='l2 loss on the image')
    parser.add_argument('--main_loss_multiplier', type=float, default=1.0, help='coefficient for the main loss in optimization')
    parser.add_argument('--store_best_images', action='store_true', help='save best images as separate files')

    parser.add_argument('--save_dir', default="./", type=str, help='base directory to save generated results')
    parser.add_argument('--labeled_reg_feature', default=1e2, type=float, help='weight for new labeled regularization statistics')
    parser.add_argument('--labeled_stats', type=str, help='npy file of the labeled statistics of the orginal data')
    parser.add_argument('--target_label', default=0, type=int, help='the target label of image to be generated')
    parser.add_argument('--gen_all_labels', action='store_true', help='batch generate images for ALL labels (0 - 999) for imagenet')
    parser.add_argument('--start', default=0, type=int, help='start label')
    parser.add_argument('--end', default=999, type=int, help='end label')

    args = parser.parse_args()
    print(args)

    torch.backends.cudnn.benchmark = True

    # changed by tong 2020/9/23 change cycle from run()
    if not args.gen_all_labels:
        run(args)
    else:
        for target_label in range(args.start, args.end + 1):
            args.target_label = target_label
            print("")
            print("="*80)
            print("[kf info] Generating label: %d" % target_label)
            print("="*80)
            run(args)
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
