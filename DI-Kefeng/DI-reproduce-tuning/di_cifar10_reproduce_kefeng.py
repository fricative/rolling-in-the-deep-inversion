"""
Filename: di_cifar10_reproduce_kefeng.py 

@author: Kefeng Zhu 

Description:
    This work aims to reproduce the deep inversion algorithm on CIFAR10 dataset.
    The method is propose by NVIDIA lab's with their CVPR20 paper. 
"""
import argparse
import torch
import torch.nn as nn
import random
import torch.backends.cudnn as cudnn

import numpy as np
import os
import glob
import collections
import time

import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.optim as optim


from resnet import ResNet34, ResNet18

#================================================================================
# >>> Global parameters >>>
#================================================================================
#--------------------------------------------------------------------------------
# The use of APEX
#--------------------------------------------------------------------------------
try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex import amp, optimizers
    USE_APEX = True
except ImportError:
    print("Please install apex from " \
            "https://www.github.com/nvidia/apex to run this example.")
    print("will attempt to run without it")
    USE_APEX = False

#--------------------------------------------------------------------------------
# Provide intermeiate information
#--------------------------------------------------------------------------------
debug_output = False
debug_output = True

#================================================================================
# Test the pretrained model
#================================================================================
def test(model, testloader, criterion, device):
    print("I[KF INFO] >>> Test the input model with the testloader")
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print('[KF INFO] >>> Loss: %.3f | Acc: %.3f%% (%d/%d)'
          % (test_loss / (batch_idx + 1), 100.*correct / total, correct, total))


#================================================================================
# Define the Feature Hook Class
#================================================================================
# Note: In pytorch, the so-called "hook" aims to intercept intermediate 
#       information in the model either during the forward or backward process. 
#       We just need some information from the BN layer during the forward 
#       process.
# For more information about hooks in pytorch, just google it. There's a
# bunch of articles on it, e.g., these're good ones:
#
#   https://blog.paperspace.com/pytorch-hooks-gradient-clipping-debugging/
#   https://zhuanlan.zhihu.com/p/87853615
#--------------------------------------------------------------------------------
# -- KF Continues on Thu Jul  1 06:56:00 UTC 2021
class DeepInversionFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a
    loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''
    def __init__(self, module):
        '''
        In this init function, call the register_forward_hook to register
        customized hook function defined below to initialize a hook object.
        '''
        self.hook = module.register_forward_hook(self.hook_get_feature)

    def hook_get_feature(self, module, input, output):
        '''
        Define such a hook function to be registered as a hook, this function 
        have fixed arguments: the module, the input and the output.
        In this function, apply the extraction of whatever feature or stats
        you need and save it/them as an member of the current object.
        This function must NOT have any output.

        In the deep inversion method, according to Equation 4 in the DI paper,
        the major part of the algorithm is implemented HERE, in which we

          (1) Calculate the means and vars (channel-wise) of the current module 
        (acutally will definitely be a BN layer when it's used);
          (2) Get the means and vars originally stored in the pretrained model;
          (3) Calculate "R_Feature" regularization term which defined in the paper
          as the L2 sum of mean and var.
          (4) Save it as a member of the current object: self.r_feature
        '''
        # (0) Get the channel number of the current module/layer:
        # Note that the order of the input is: (N,C,H,W)
        nch = input[0].shape[1]

        # (1) Calculate the means and vars of the current (BN) module:
        # Mean (CHANNEL-WISE!):
        mean = input[0].mean([0, 2, 3])
        # Variance (CHANNEL-WISE!):
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, 
                unbiased=False)

        # (2) Get the means and vars stored in BN:
        bn_stored_mean = module.running_mean.data.type(var.type()) 
        bn_stored_var = module.running_var.data.type(var.type()) 

        # (3) Calculate "R_Feature" regularization term:
        r_feature = torch.norm(mean - bn_stored_mean, 2) + torch.norm(var - bn_stored_var, 2)
        # (4) Save it as a member of the current object: self.r_feature
        self.r_feature = r_feature

        # Must have no output!


    def close(self):
        '''
        This is also a required function in which we need to remove the hook 
        object.
        '''
        self.hook.remove()


#================================================================================
# Run the Deep Inversion iteration to generate images
#================================================================================
def run_deep_inversion(
        model, bs=256, epochs=1000,
        alpha_tv=0.00005,
        alpha_l2=0.0,
        alpha_f=0.0,
        optimizer=None,
        inputs=None,
        device='cuda',
        use_amp=False,
        random_label=False,
        prefix=None,
        ):
    '''
    Function invert images from the pretrained model. 
    No implementation of ADI (Adaptive Deep Inversion) is included in this 
    reproduction version.
    Parameters:
        bs: batch size
        epochs: total number of iterations to generate inverted images, training 
            longer helps a lot!
        alpha_tv: the scaling factor for total variance loss regularization R_TV.
            this may vary depending on bs
            larger - more blurred but less noise
        alpha_l2: scaling factor for L2 regularization term on input R_l2.
        alpha_f: scaling factor for r_feature regularization term R_feature.
        optimizer: potimizer to be used for model inversion
        inputs: data place holder for optimization, will be reinitialized to 
            noise.
        device: CPU or CUDA
        random_labels: sample labels from random distribution or use columns of 
            the same class
        use_amp: boolean to indicate usage of APEX AMP for FP16 calculations - 
            twice faster and less memory on TensorCores
        prefix: defines the path to store images

    Return:
    '''
    best_cost = 1e6

    # Initialize gaussian inputs (noise)
    # Notice: setting requires_grad to True is important. It allows the inputs 
    #   to be updated towards natural images
    inputs.data = torch.randn((bs, 3, 32, 32), requires_grad=True, device=device)

    # Set up criterion
    criterion = nn.CrossEntropyLoss()

    # Reset state of optimizer: why do this? 
    optimizer.state = collections.defaultdict(dict)

    # Target output to generate
    if random_label:
        targets = torch.LongTensor(
                [random.randint(0,9) for _ in range(bs)]).to(device)
    else:
        if bs == 256:
            targets = torch.LongTensor([i for i in range(10)] * 25 + 
                    [0, 1, 2, 3, 4, 5]).to(device)
        elif bs == 128:
            targets = torch.LongTensor([i for i in range(10)] * 12 + 
                    [0, 1, 2, 3, 4, 5, 6, 7]).to(device)
        elif bs % 10 == 0:
            targets = torch.LongTensor(
                    [i for i in range(10)] * (bs // 10)).to(device)
    print("Targets shape (batch size):", targets.shape[0])

    #----------------------------------------------------------------------------
    # Initialize hook objects.
    # Note: the hook function aims to intercept intermediate information in the 
    #       model either during the forward or backward process. 
    #       We just need some information from the BN layer during the forward 
    #       process.
    # For more information about hooks in pytorch, just google it. There's a
    # bunch of articles on it, e.g., these're good ones:
    #
    #   https://blog.paperspace.com/pytorch-hooks-gradient-clipping-debugging/
    #   https://zhuanlan.zhihu.com/p/87853615
    #
    # -- KF on Fri Jun 25 07:43:27 UTC 2021
    #----------------------------------------------------------------------------
    ## Create hooks for feature statistics catching
    loss_r_feature_layers = []
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):  # filter the BN layers only!
            loss_r_feature_layers.append(DeepInversionFeatureHook(module))

    # Setting up the range for jitter
    lim_0, lim_1 = 2, 2
    #############################################################################
    #                   Start of the deep inversion loop                        #
    #############################################################################
    # Initialize the best cost for saving best results
    best_cost = 1e6
    # Set a timer
    time_start = time.time()
    print("[KF INFO] Starting Model Inversion ...")
    for epoch in range(epochs):
        #------------------------------------------------------------------------
        # Apply random jitter offsets  KF: this is not that important I think.
        #------------------------------------------------------------------------
        off1 = random.randint(-lim_0, lim_0)
        off2 = random.randint(-lim_1, lim_1)
        # torch.roll: https://pytorch.org/docs/stable/generated/torch.roll.html
        inputs_jit = torch.roll(inputs, shifts=(off1,off2), dims=(2,3))

        #------------------------------------------------------------------------
        # Foward with jit images
        #------------------------------------------------------------------------
        optimizer.zero_grad()
        model.zero_grad()
        outputs = model(inputs_jit)
        loss = criterion(outputs, targets)  
        # NOTE: the "loss" here is a vanilla CE loss, to which we will then add a 
        #       bunch of customized regularization terms.
        loss_ce = loss.item()   # Save this amount for print and compare

        #------------------------------------------------------------------------
        # Ignore the Adaptive Deep Inversion part!
        #------------------------------------------------------------------------

        #------------------------------------------------------------------------
        # Apply total variation regularization: R_TV, 
        # [no need to dig into this, not the core of the DI method]
        #------------------------------------------------------------------------
        diff1 = inputs_jit[:,:,:,:-1] - inputs_jit[:,:,:,1:]
        diff2 = inputs_jit[:,:,:-1,:] - inputs_jit[:,:,1:,:]
        diff3 = inputs_jit[:,:,1:,:-1] - inputs_jit[:,:,:-1,1:]
        diff4 = inputs_jit[:,:,:-1,:-1] - inputs_jit[:,:,1:,1:]
        r_tv = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)

        #------------------------------------------------------------------------
        # Apply L2 reqularization: R_l2
        # [no need to dig into this, not the core of the DI method]
        #------------------------------------------------------------------------
        r_l2 = torch.norm(inputs_jit, 2)

        #------------------------------------------------------------------------
        # Add the R_TV and R_l2 to the loss, 
        # according to Equation 3 in the DI paper
        # [no need to dig into this, not the core of the DI method]
        #------------------------------------------------------------------------
        loss = loss + alpha_tv * r_tv + alpha_l2 * r_l2

        #------------------------------------------------------------------------
        # Apply the feature distribution regularization: R_feature
        # See Equation 4-7 in the DI paper
        # This part is IMPORTANT!
        #------------------------------------------------------------------------
        r_feature = sum([mod.r_feature for mod in loss_r_feature_layers])
        loss = loss + alpha_f * r_feature

        #------------------------------------------------------------------------
        # NOTE: Up to this point, the formulation of the loss has been done.
        #       The next step is to backward the loss to update the input 
        #       iteratively and achieve the effect of image generation.
        #------------------------------------------------------------------------

        #------------------------------------------------------------------------
        # Save the best inputs according to "best_cost" (in memory)
        #------------------------------------------------------------------------
        if best_cost > loss.item():
            best_cost = loss.item()
            best_inputs = inputs.data

        #------------------------------------------------------------------------
        # Backward & Update
        #------------------------------------------------------------------------
        if use_amp: 
            # The AMP is an accelerating method provided by NVIDIA, may just 
            # ignore this
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            # Back propagation
            loss.backward()

        # Update the variables assigned to the optimizer, it's the INPUT!
        # (Check the optimizer initialization again!)
        optimizer.step()

        #------------------------------------------------------------------------
        # Print the progress of the iteration
        #------------------------------------------------------------------------
        # Set the step size:
        step = 100
        # Time elapsed till current iteration:
        time_elapsed = time.time() - time_start
        if debug_output and (epoch+1) % step == 0:
            print(f"[Iteration] {epoch+1:4d}    [Losses] Total:{loss.item():8.3f},  CE_loss:{loss_ce:8.5f},  R_feature_loss unscaled:{r_feature.item():7.3f}    [Time elapsed]: {time_elapsed:7.2f} seconds")
            vutils.save_image(inputs.data.clone(),
                              '{}/output_{}.png'.format(prefix, (epoch+1)//step),
                              normalize=True, scale_each=True, nrow=10)


    #############################################################################
    #                   End of the deep inversion loop!                         #
    #############################################################################
    print("[KF INFO] Model Inversion DONE!")
    print(f"Total time spent: {time.time() - time_start:7.2f} seconds")

    #----------------------------------------------------------------------------
    # Save the best synthesized images to file
    #----------------------------------------------------------------------------
    name_use = "best_images"
    if prefix is not None:
        name_use = os.path.join(prefix, name_use)
    next_batch = len(glob.glob("%s/*.png" % name_use)) // 1

    vutils.save_image(best_inputs[:20].clone(),
                      '{}/output_{}.png'.format(name_use, next_batch),
                      normalize=True, scale_each = True, nrow=10)


#================================================================================
# >>> Main (The program starts here!) >>>
#================================================================================
if __name__ == "__main__":

    #----------------------------------------------------------------------------
    # Parsing the parameters from the prompt or shell scirpt
    #----------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
            description="PyTorch CIFAR10 DeepInversion reproduced by Kefeng")
    parser.add_argument('--bs', default=256, type=int, help="batch size")
    parser.add_argument('--di_lr', default=0.1, type=float, help='lr for deep inversion')
    parser.add_argument('--epochs', default=2000, type=int, help='number of iterations for model inversion')
    parser.add_argument('--alpha_tv', default=2.5e-5, type=float, help='TV regularization scaling factor')
    parser.add_argument('--alpha_l2', default=0.0, type=float, help='L2 regularization scaling factor')
    parser.add_argument('--alpha_f', default=1e2, type=float, help='scaling factor for BN regularization statistic')

    parser.add_argument('--amp', action='store_true', help='use APEX AMP O1 acceleration')
    parser.add_argument('--teacher_weights', default="'./checkpoint/teacher_resnet34_only.weights'", type=str, help="path to load weights of the model")
    parser.add_argument('--save_dir', default="./", type=str, help="base directory to save generated results")
    parser.add_argument('--dataset', default="./", type=str, help="path to the cifar10 dataset, only for test")
    parser.add_argument('--exp_descr', default="try1", type=str, help="name to be added to experiment name")
    parser.add_argument('--device', default="cpu", type=str, help="force to set the device to be used, either 'cpu' or 'gpu'")

    args = parser.parse_args()

    #----------------------------------------------------------------------------
    # Here I only use single model instead the teacher/student pair used in the 
    # original code provided by NIDIA lab
    # ---  by Kefeng on Thu Jun 24 07:14:00 UTC 2021
    #----------------------------------------------------------------------------
    print("[KF INFO] Loading resnet34 ...")
    pretrained_model = ResNet34()

    #----------------------------------------------------------------------------
    # Determine the device to be used, eithr CPU or NVIDIA CUDA GPU
    #----------------------------------------------------------------------------
    if args.device == 'cpu':
        device = 'cpu'
        print("[DEVICE] Using CPU ...")
    if args.device == 'gpu':
        #device = 'cuda'
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            print("[DEVICE] Using GPU ...")
        if device == 'cpu':
            print("[DEVICE] You don't have any GPU devices, using CPU instead ...")
    pretrained_model = pretrained_model.to(device)

    #----------------------------------------------------------------------------
    # Determine the criterion/loss-function to be used
    #----------------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()

    #----------------------------------------------------------------------------
    # Initialize the input placeholder
    #   the init here is merely for setting up the optimizer, it'll be 
    #   re-initialized in function run_deep_inversion
    #----------------------------------------------------------------------------
    data_type = torch.half if args.amp else torch.float
    inputs = torch.randn((args.bs, 3, 32, 32), requires_grad=True, 
            device=device, dtype=data_type)

    #----------------------------------------------------------------------------
    # Determine optimizer
    # IMPORTANT: we set the first parameter as "inputs" to let the optimizer 
    #   update the input instead of the model weights to achieve the effect of 
    #   image generation.
    #----------------------------------------------------------------------------
    optimizer_di = optim.Adam([inputs], lr=args.di_lr)

    #----------------------------------------------------------------------------
    # Load the pretrained model with it's given path
    #----------------------------------------------------------------------------
    checkpoint = torch.load(args.teacher_weights)
    pretrained_model.load_state_dict(checkpoint)

    #----------------------------------------------------------------------------
    # IMPORTANT: otherwise generated image will be non natural
    # Why use net.eval():
    # https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
    #----------------------------------------------------------------------------
    pretrained_model.eval()

    #----------------------------------------------------------------------------
    # AMP support is optional and will be added later
    #----------------------------------------------------------------------------
    if args.amp:
        opt_level = "O1"
        loss_scale = 'dynamic'
        pretrained_mdoel, optimizer_di = amp.initialize(
            pretrained_model, optimizer_di,
            opt_level=opt_level,
            loss_scale=loss_scale)
    
    #----------------------------------------------------------------------------
    # Enable this mode (cudnn.benchmark) to accelerate the compuation on GPU
    #----------------------------------------------------------------------------
    cudnn.benchmark = True


    #----------------------------------------------------------------------------
    # Create folders and paths for saving if needed
    # prefix: the saving path for current try
    #----------------------------------------------------------------------------
    prefix = os.path.join(args.save_dir, "data_generation", args.exp_descr)
    for create_folder in [prefix, prefix+"/best_images/"]:
        if not os.path.exists(create_folder):
            os.makedirs(create_folder)

    #----------------------------------------------------------------------------
    # [Optional] Test/Validate the pretrained model on the test dataset
    # The accuracy of the pretrained model on CIFAR10 I got should be 95.67%
    # Note: This part can be comment out. The 'dataset' parameter is used only 
    # when running the following test.
    #----------------------------------------------------------------------------
    # Loading the dataset, search "torchvision.dataset" and "DataLoader"
    #if 1:
    if 0:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = torchvision.datasets.CIFAR10(root=args.dataset, train=False, 
                download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, 
                shuffle=True, num_workers=6, drop_last=True)
        # Checking the accuracy of the pretrained model
        print("[KF INFO] Testing the accuracy of the pretrained model")
        test(pretrained_model, testloader, criterion, device)

    
    #----------------------------------------------------------------------------
    # Run Deep Inversion to generate images
    #----------------------------------------------------------------------------
    # Some global variables
    global_iteration = 0

    run_deep_inversion(model=pretrained_model,
            prefix=prefix,
            bs=args.bs,
            epochs=args.epochs,
            alpha_tv=args.alpha_tv,
            alpha_l2=args.alpha_l2,
            alpha_f=args.alpha_f,
            inputs=inputs,
            optimizer=optimizer_di,
            device=device,
            use_amp=args.amp,
            )

