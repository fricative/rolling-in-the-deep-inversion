import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

import torchvision.transforms as transforms
from torchvision.models.inception import inception_v3

import os
from torchvision.io import read_image

import numpy as np
from scipy.stats import entropy


normalize_mean = [0.485, 0.456, 0.406]
normalize_std = [0.229, 0.224, 0.225]


def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):

    N = len(imgs)
    assert batch_size > 0
    assert N > batch_size

    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]
        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)



class CustomImageDataset(torch.utils.data.Dataset):

    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.transform = transforms.Compose([
            transforms.Resize(32),
            transforms.Normalize(normalize_mean, normalize_std)
        ])
        self.img_file_list = os.listdir(img_dir)
        print('\nfound {} imgs in directory\n'.format(len(self.img_file_list)))

    def __len__(self):
        return len(self.img_file_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_file_list[idx])
        image = read_image(img_path).double()
        return self.transform(image)


img_data_folder = 'best_Hps_for_IS_non_grid_V1/'
IS_dataset = CustomImageDataset(img_data_folder)
print(inception_score(IS_dataset, cuda=True, batch_size=10, resize=True, splits=10))