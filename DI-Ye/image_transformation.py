import os

import torch
import torch.utils.data
from torchvision.io import read_image
import torchvision.transforms as transforms
import torchvision.utils as vutils
 

class CustomImageDataset(torch.utils.data.Dataset):

    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.transform = transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.img_file_list = os.listdir(img_dir)
        print('\nfound {} imgs in directory\n'.format(len(self.img_file_list)))

    def __len__(self):
        return len(self.img_file_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_file_list[idx])
        image = read_image(img_path).double()
        return self.transform(image)


IS_dataset = CustomImageDataset('imagenet_generated_raw/')

i = 0
for img in IS_dataset:
    vutils.save_image(img, 'imagenet_generated_processed/{}.png'.format(i), normalize=True)
    i += 1